# KV Cache 系统级优化深度解读 —— 大规模分布式推理集群视角（SGLang vs vLLM/vllm-ascend）

> 生成时间：2026-06-24
> 走读版本：vllm `0d2961229` / vllm-ascend `8afdf356` / sglang `b5e0965b07`
> 聚焦维度：**KV Cache 系统级优化**（块/前缀管理 → 分层存储 → 跨节点传输 → 集群级复用）
> 视角：大规模分布式推理集群（多机多卡、PD 分离、多租户、长上下文、MoE）
> 原则：基于代码走读，行号已 grep 实测；函数体逻辑实读；说不清处标注疑问。
> 关联：[[20260624-005610-sglang-vs-vllm-大规模分布式推理集群优化-对比分析]]

---

## 〇、为什么 KV Cache 是大规模集群的系统级矛盾

在单机视角，KV Cache 优化 = 显存够不够、命中率高不高。但到**大规模分布式集群**视角，KV Cache 变成贯穿全栈的系统级问题：

1. **它是 PD 分离的"货物"** —— Prefill 算出的 KV 必须跨节点搬到 Decode，传输带宽/延迟直接决定 PD 分离能否成立。
2. **它是显存墙的主要来源** —— 长上下文 + 高并发，KV 占用远超权重，必须分层外溢（HBM→DRAM→SSD→远端池）。
3. **它是集群复用的载体** —— 跨请求、跨节点、跨会话的前缀复用，决定集群整体有效吞吐（Goodput）。
4. **它牵涉正确性与隔离** —— 多租户/多 LoRA/不同 sampling 下，KV 复用不能串味。

下面按"**块/前缀管理 → 分层存储 → 跨节点/远端池 → 集群级复用与正确性**"四层，对比两家的系统级实现。

---

## 一、层次一：块/前缀管理（HBM 内的复用引擎）

### 1.1 vLLM：哈希块表（hash-indexed block）

vLLM 的复用引擎是**满块哈希表**。核心链路：

- 请求级入口 `KVCacheManager.get_computed_blocks()`（[kv_cache_manager.py:202](vllm/vllm/v1/core/kv_cache_manager.py)）查命中、`allocate_slots()`（kv_cache_manager.py:244）分配。
- 复用核心在 `BlockPool`：满块计算 `block_hash` 后，`cache_full_blocks()`（[block_pool.py:211](vllm/vllm/v1/core/block_pool.py)）把块按 `make_block_hash_with_group_id(block_hash, group_id)` 插入 `cached_block_hash_to_block`（block_pool.py:281）；`get_cached_block()`（block_pool.py:184）按哈希查命中。
- 驱逐：`_maybe_evict_cached_block()`（block_pool.py:365）+ `free_blocks()`（block_pool.py:419），LRU 风格 + `touch()`（block_pool.py:402）刷新访问。

**系统级亮点 —— Hybrid 多类型 KV 协调**：`HybridKVCacheCoordinator`（[kv_cache_coordinator.py:466](vllm/vllm/v1/core/kv_cache_coordinator.py)）+ `find_longest_cache_hit_per_group()`（kv_cache_coordinator.py:694），对 **SWA（滑窗）/ Mamba / 全注意力混合模型** 分组做最长命中。`cache_full_blocks` 的 `block_mask`（block_pool.py:219）让 SWA 尾窗外的块不进哈希表——这是处理新一代混合架构（如 Mamba-Transformer、SWA 模型）的系统级设计。

### 1.2 SGLang：Radix 前缀树（树形最长前缀）

SGLang 的复用引擎是 **Radix 树**，语义比哈希块更强：

- `match_prefix()`（[radix_cache.py:360](sglang/python/sglang/srt/mem_cache/radix_cache.py)）做**最长前缀匹配**，命中到 token 边界（不止块对齐），`_match_prefix_helper` 走树 + `page_aligned(page_size)`（radix_cache.py:403）对接 paged allocator。匹配结束在段中间会**分裂节点**精确暴露边界（radix_cache.py:393-395）。
- 插入/驱逐：`insert()`（radix_cache.py:420）、`cache_finished_req()`（radix_cache.py:442）、`evict()`（radix_cache.py:568）+ `TreeNode` 引用计数（radix_cache.py:223）。
- **正确性隔离（集群多租户关键）**：`RadixKey` 带 `extra_key`，不同 LoRA / sampling salt / cache 版本 / RAG 上下文的请求**强制不共享前缀节点**（radix_cache.py:363-372）——大集群多租户场景下防止 KV 串味，这是哈希块方案需要额外 group_id/extra_keys 才能等价的能力。

> **判断（层次一）**：vLLM = 哈希块，O(1) 查、实现简单、对 hybrid/SWA 模型支持系统化；SGLang = Radix 树，前缀复用语义更细（token 级 + 命名空间隔离），天然 cache-aware 调度。**集群多轮对话/共享 system prompt 场景 SGLang 复用率结构性更优；混合架构模型 vLLM 的 coordinator 更成体系。**

---

## 二、层次二：分层存储（突破单卡显存墙）

这是 KV 系统级优化在**长上下文 + 高并发集群**下的核心，把 KV 从 HBM 外溢到更廉价的存储层。

### 2.1 vLLM：kv_offload 三级 tiering

vLLM 把分层做成独立子系统 `v1/kv_offload/`：

- 抽象 `OffloadingManager`（[base.py:149](vllm/vllm/v1/kv_offload/base.py)）：`lookup` / `prepare_load`（base.py:168）/ `touch`。
- **三级层次**：`TieringOffloadingManager`（[tiering/manager.py:111](vllm/vllm/v1/kv_offload/tiering/manager.py)）+ `CPUPrimaryTierOffloadingManager`（tiering/manager.py:62）+ `SecondaryTierManager`（tiering/base.py:42），构成 HBM → CPU（一级）→ fs/obj（二级）三层，带 `PendingPromotion`（tiering/manager.py:54）做层间晋升。
- **异步 lookup**：`AsyncLookupManager`（[tiering/async_lookup.py:52](vllm/vllm/v1/kv_offload/tiering/async_lookup.py)）异步查二级层，避免阻塞调度。
- 调度侧接入：`OffloadingConnector`（[offloading_connector.py:46](vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py)）的 `get_num_new_matched_tokens()`（offloading_connector.py:131）告诉调度器"外存里还有多少能命中"，把外存命中纳入调度决策。

### 2.2 SGLang：HiRadixCache + HiCacheController

SGLang 把分层**直接长在 Radix 树上**（树级分层），由独立控制器 `HiCacheController`（[cache_controller.py:209](sglang/python/sglang/srt/managers/cache_controller.py)）驱动：

- **回写**：`write_backup`（hiradix_cache.py:758）把冷节点 KV 下沉到 host；`HiCacheController.write()`（cache_controller.py:656）。
- **回载（工程细节扎实）**：`load_back()`（[hiradix_cache.py:1140](sglang/python/sglang/srt/mem_cache/hiradix_cache.py)）逻辑实读——
  - **全有或全无**：低于 `load_back_threshold` 或超 `mem_quota` 则跳过（hiradix_cache.py:1162）；
  - **锁祖先防驱逐**：`inc_lock_ref(ancester_node)`（hiradix_cache.py:1157）回载期间锁住祖先；
  - **失败重试**：GPU 不足先 `evict()` 再重试一次 `cache_controller.load()`（hiradix_cache.py:1174-1180）；
  - 仍失败则告警返回 None（hiradix_cache.py:1182），不崩溃。
- **异步线程隐藏 IO**：`prefetch_thread`（cache_controller.py:352）+ write 线程 + `LayerLoadingEvent`/`LayerDoneCounter`（cache_controller.py:56/74）做 **layer-wise 加载**，逐层就绪逐层算。
- `prefetch_from_storage`（hiradix_cache.py:1471）从远端存储预取。

> **判断（层次二）**：两家都做到 HBM↔CPU↔远端三级。差异在**组织方式**——vLLM 是"独立 offload 子系统 + connector 把外存命中喂给调度"，模块边界清晰、可插拔；SGLang 是"分层内嵌 Radix 树 + 中央 cache_controller + 异步线程 + layer-wise"，**树语义与分层统一**、回载工程细节（锁祖先/配额/重试/逐层）更完整。**SGLang 的 layer-wise 加载对降低长前缀回载的首 token 延迟更有针对性。**

---

## 三、层次三：跨节点传输与远端 KV 池（PD 分离的命脉）

集群里 KV 要跨机搬运（PD 分离）或存进共享池（跨实例复用）。

### 3.1 传输/池化后端矩阵

| 后端 | vLLM | vllm-ascend | SGLang |
|------|------|-------------|--------|
| RDMA 通用（NIXL） | ✅ pull/push 两态 | 〜（NPU 走自有） | ✅ nixl |
| Mooncake（KV 池/传输） | ✅ + layerwise❓ | ✅ mooncake/layerwise/hybrid | ✅ mooncake_store |
| DeepSeek 3FS 直存 | ✅ hf3fs | 〜 | ✅ hf3fs |
| LMCache 远端缓存 | ✅ | ✅ lmcache_ascend | ✅ lmcache |
| 对象/文件存储 | ✅ tiering/obj,fs | ✅ ascend_store | ✅ file / simm / eic / aibrix_kvcache |
| 厂商池化（LLMDataDist/UCM） | — | ✅ kv_pool（ucm_connector/cpu_offload） | — |

- **vLLM NIXL 拉/推两态**：`NixlPullConnector`/`NixlPushConnector`（connector.py:321/349），`register_kv_caches`（base_worker.py:843）把本地 KV 注册给 NIXL agent 做 RDMA 单边读写（零拷贝/绕 CPU）。
- **SGLang 远端池接口**：`HiCacheStorage`（[hicache_storage.py:140](sglang/python/sglang/srt/mem_cache/hicache_storage.py)）统一 `batch_get_v2`/`batch_set_v2`（hicache_storage.py:188/199）批量接口，后端 file/mooncake_store/nixl/hf3fs/eic/aibrix_kvcache/simm（storage/ 目录）。
- **vllm-ascend 池化差异**：`kv_pool/` 下 `ucm_connector.py` + `ascend_store` + `cpu_offload`（基于昇腾 LLMDataDist），KV 池化走昇腾原生路径；`AscendMultiConnector`（ascend_multi_connector.py:19）叠加多后端。

> **判断（层次三）**：传输后端高度趋同（都接 Mooncake/NIXL/hf3fs/LMCache）。vLLM 的 connector 抽象更标准化，SGLang 的 HiCacheStorage 批量接口 + 后端最多（含 aibrix/eic/simm 等更多生产后端）。**昇腾线靠 LLMDataDist/UCM 自建池化，与 NV 生态平行。**

---

## 四、层次四：集群级复用、压缩与正确性

### 4.1 跨请求公共前缀 → Cascade Attention（vLLM）

vLLM 不止复用 KV 块，还在**算子层**复用：`get_num_common_prefix_blocks()`（kv_cache_coordinator.py:274）算同批运行请求的公共前缀块，调度器在 schedule 时计算（[scheduler.py:956](vllm/vllm/v1/core/sched/scheduler.py)），喂给 `gpu_model_runner` 触发 **cascade attention**（[gpu_model_runner.py:2508](vllm/vllm/v1/worker/gpu_model_runner.py)，`use_cascade_attn`）——**多请求共享的前缀只算一次 attention**，省的不只是显存，还有算力。

### 4.2 MLA 低秩 KV + 稀疏 KV（SGLang，面向 DeepSeek/长上下文）

SGLang 在 KV 布局层做了两类压缩：

- **MLA 低秩 latent KV**：`MLATokenToKVPool`（[memory_pool.py:2101](sglang/python/sglang/srt/mem_cache/memory_pool.py)）+ `HybridLinearKVPool`（memory_pool.py:1875），DeepSeek MLA 只存压缩后的 latent，KV 显存数量级下降，是大集群跑 DeepSeek-V3/R1 的关键。FP4 进一步压缩 `MLATokenToKVPoolFP4`（memory_pool.py:2352）。
- **稀疏 KV**：`mem_cache/sparsity/`（algorithms/backend/core）+ `HiSparseTokenToKVPoolAllocator`（[allocator/hisparse.py:15](sglang/python/sglang/srt/mem_cache/allocator/hisparse.py)）+ `DeepSeekV4HiSparseTokenToKVPoolAllocator`（hisparse.py:276），对长上下文做选择性 KV 保留（NSA/DSA 类稀疏注意力），`alloc_logical_only`（hisparse.py:107）逻辑/物理分离分配。

### 4.3 正确性与隔离

- SGLang `extra_key` 命名空间（radix_cache.py:363）——多租户/多 LoRA 防串味。
- vLLM `cache_salt` + extra_keys（cache_full_blocks 的 extra_keys 逻辑，block_pool.py:301）——同等隔离能力。

---

## 五、达到了什么提升（指标归因）

| 优化层 | 关键机制 | 直接提升的指标 |
|--------|----------|----------------|
| 块/前缀管理 | hash块 / Radix树 + cache-aware 调度 | **TTFT↓**（跳过重复 prefill）、**前缀复用率↑** |
| 跨请求复用 | cascade attention（vLLM）、共享前缀树（SGLang） | **prefill 算力↓**、共享 system prompt 场景吞吐↑ |
| 分层存储 | kv_offload 三级 / HiRadix + layer-wise | **有效显存↑（等效扩容）**、**最大并发/上下文长度↑**、回载 TTFT↓ |
| 跨节点传输 | NIXL RDMA 单边 / Mooncake / 远端池 | **PD 分离可行**、KV 传输延迟↓、跨实例缓存命中↑ |
| KV 压缩 | MLA latent / FP4 / 稀疏 KV | **KV 显存数量级↓**、长上下文承载↑、带宽占用↓ |
| 正确性隔离 | extra_key / cache_salt | 多租户 Goodput（不牺牲正确性） |

**一句话**：KV 系统级优化把集群的瓶颈从"单卡显存装不下"和"重复 prefill 浪费算力"两条线同时打开，核心收益是 **TTFT↓ + 有效显存/并发↑ + 集群 Goodput↑**。

---

## 六、后续可进一步优化的切入点（分析判断）

按"价值 × 可行性"排序，结合本次走读看到的现状与空白：

### 6.1 高价值切入点

1. **分层存储的命中预测与预取智能化**
   现状：SGLang `prefetch_from_storage`（hiradix_cache.py:1471）+ vLLM `AsyncLookupManager` 已有预取/异步查，但预取**触发仍偏被动**（命中冷层才回载）。
   切入点：基于请求模式（多轮对话、RAG 固定文档）做**前缀预测性预取**，在请求到达前把热前缀从远端拉回 HBM，进一步压 TTFT。可结合调度器的 waiting 队列预判。

2. **KV 压缩与分层/传输的协同**
   现状：MLA/FP4/稀疏（SGLang）和分层、跨节点传输是**相对独立的两条线**。
   切入点：**压缩后再传输/分层**——PD 分离时传压缩 KV（latent/FP4）而非原始 KV，跨节点带宽可数量级下降；分层下沉时存压缩态。需打通 KV 压缩态在 connector/storage 的序列化。这是大集群带宽墙的高杠杆点（对齐 CLAUDE.md「框架引擎层优先」——这是引擎层而非 kernel）。

3. **跨实例/全局前缀缓存的集群级共享**
   现状：远端 KV 池（Mooncake store / LMCache / aibrix）已能跨实例共享，但**全局前缀索引的一致性与路由**仍弱（哪个实例有哪段前缀，请求该路由到谁）。
   切入点：把 cache-aware 路由从单实例（SGLang LPM/DFS 调度）**上升到集群级**——router 感知全局 KV 池命中分布，做"亲和性路由"（把带相同前缀的请求路由到已有 KV 的实例），最大化全局复用率。这是 KV 复用从"实例内"到"集群内"的关键一跳。

### 6.2 中价值切入点

4. **cascade attention 的跨请求复用扩展到 SGLang / 分层场景**
   vLLM 的 cascade attention（共享前缀只算一次 attention）目前是同批同实例。可探索：① SGLang 侧是否有等价算子复用（待核实）；② 公共前缀在分层/远端命中时能否同样触发 cascade。

5. **混合架构（SWA/Mamba/MLA）下分层与压缩的统一抽象**
   vLLM 的 `HybridKVCacheCoordinator` 已处理混合 KV 的命中，但**分层/压缩对混合架构的支持成熟度待核实**。SGLang 有 mamba_radix_cache/swa_radix_cache，但与 HiRadix 分层的耦合度需进一步走读。统一抽象能降低新架构接入成本。

6. **回载/驱逐策略的负载自适应**
   SGLang `load_back` 用固定 `load_back_threshold` + `mem_quota`（hiradix_cache.py:1162）。切入点：根据集群实时负载（队列深度、命中率、带宽占用）**动态调阈值**，高负载时更激进下沉、低负载时更激进预取。

### 6.3 昇腾线专项

7. **vllm-ascend 的 KV 压缩 attention（kvcomp_attn）成熟度与 MLA 对齐**
   `kvcomp_attn`（attention_utils.py）已现雏形，但相比 SGLang 的 MLA/稀疏体系，昇腾侧 KV 压缩的覆盖与算子支持需进一步走读确认。切入点：在 NPU 上补齐 MLA latent KV / FP4 的 pool 与 allocator，缩小与 GPU 线的 KV 显存效率差距。

---

## 七、待核实/存疑（不推测）

1. SGLang 是否有等价 vLLM **cascade attention** 的"共享前缀只算一次"算子层复用？本次未在 attention backend 走读确认。
2. vLLM 各 connector 的 **layer-wise 流式传输** 覆盖度（只确认 ascend mooncake_layerwise 明确，vLLM 主线 NIXL 是否 layer-wise 待核实）。
3. vLLM 分层存储（kv_offload）对 **MLA/混合架构** 的支持成熟度，未深入到 spec 层确认。
4. vllm-ascend `kvcomp_attn` 的压缩算法（是否 MLA、是否稀疏）与 SGLang 路线的对应关系，需读 `kvcomp_utils` 确认。
5. SGLang `HiCacheController` 的 **layer-wise 加载** 是否已默认开启、对哪些后端生效，需读 server_args 确认。
