# DeepSeek V4 Flash + 昇腾 A2(910B3) 长上下文/Agentic 场景：KV Cache 系统优化 与 PD/MoE 弹性 深度分析

> 生成时间：2026-06-24
> 走读版本：vllm `0d2961229` / vllm-ascend `8afdf356` / sglang `b5e0965b07` / sgl-kernel-npu(同期)
> 场景设定：**DeepSeek V4 Flash（KV = HCS/CSA/SWA 三套稀疏压缩机制 + MoE；非 MLA latent）** + **昇腾 A2 节点（910B3）** + **512k~1M 长上下文 + agentic 超多轮交互**
> 框架：vllm/vllm-ascend 与 sglang/sgl-kernel-npu 两套
> 原则：结论基于代码走读，行号 grep 实测；说不清处标注存疑，不推测。
> 关联：[[20260624-011205-sglang-vs-vllm-kvcache系统级优化-大规模集群视角-深度分析]] [[20260624-005610-sglang-vs-vllm-大规模分布式推理集群优化-对比分析]]

---

## 〇、场景关键事实先对齐（这套组合特有的，决定后续所有判断）

在动结论前，先把"DeepSeek V4 Flash + 910B3"这个组合**特有的、影响优化方向**的事实从代码里坐实：

### 0.1 DeepSeek V4 Flash 的 KV 本质 = HCS + CSA + SWA 三套稀疏/压缩机制（无 MLA latent）

> **重要校正**：V4 Flash 的 KV **不是 MLA 低秩 latent**。`MLATokenToKVPool` / `get_flashmla_metadata` / `mla_preprocess` 这些名字里的 "MLA" 只是 **vllm/vllm-ascend 实现层沿用的算子/后端别名（FlashMLA 内核家族）**，承载的实际是 V4 的稀疏压缩 KV，而非 latent KV。V4 的 KV 本质是三套机制并存，按 `compress_ratio ∈ {0, 4, 128}` 三档分发：`get_flashmla_metadata` 分发 `c1 / c4 / c128_flashmla_metadata`（[deepseek_v4_backend.py:177-185](sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py)）。

V4 不是普通 dense KV，也不是 MLA latent，KV 系统优化的所有判断要建立在 **HCS / CSA / SWA** 三套机制上：

- **HCS（Hi-Sparse，分级稀疏 KV 存储）**：`HiSparseC4DevicePool`（[deepseek_v4_memory_pool.py:165](sglang/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py)）+ `full_to_hisparse_device_index_mapping`（memory_pool.py:200）+ `translate_loc_from_full_to_hisparse_device`（memory_pool.py:218）——把"全量 token 空间"映射到"压缩稀疏设备空间"，只为被选中的 token 在设备上落位。配套分配器 `HiSparseTokenToKVPoolAllocator` / `DeepSeekV4HiSparseTokenToKVPoolAllocator`（allocator/hisparse.py:15/276）。
- **CSA（Compress + Sparse，压缩稀疏 + topk 选择）**：双档压缩 `compress_ratio ∈ [4, 128]`（memory_pool.py:33），decode 时按压缩态选 topk —— `c4_sparse_topk` / `c4_topk_lengths_clamp1` / `c128_topk_lengths_clamp1`（deepseek_v4_backend.py:154/160/167）。压缩态 KV 带重要性 `KVAndScore`（[deepseek_v4_compress_state.py:23](sglang/python/sglang/srt/mem_cache/deepseek_v4_compress_state.py)，kv 与 score 共存一个 buffer）。c128 还有 online 模式 `ONLINE_C128`（memory_pool.py:27，单 (max,sum,kv) 状态）。
- **SWA（Sliding Window Attention，滑窗）**：近窗 token 走滑窗注意力 —— `swa_topk_lengths` / `swa_page_indices` / `BaseSWAKVPool`（deepseek_v4_backend.py:152，pool 基类 `DeepSeekV4TokenToKVPool(BaseSWAKVPool)`，memory_pool.py:438）。

**三者协同**：SWA 管近窗（精确全量近期上下文）、CSA 管中段（4 倍压缩 + topk）、c128/HCS 管远段冷历史（128 倍压缩，只留极少代表 token）。一条 1M 序列在三套机制下被分层压缩稀疏化。

- **两框架实现入口**：SGLang `deepseek_v4_backend.py`（c1/c4/c128 三档 + swa）；vllm-ascend `AscendDSABackend`（[dsa_v1.py:180](vllm-ascend/vllm_ascend/attention/dsa_v1.py)，`cu_c4_cmp_seqlen` / `cu_c128_cmp_seqlen` / `compressor_ratio`，dsa_v1.py:245/246/416）+ `sfa_v1.py`（sparse flash attention）+ `mla_v1.py`（仅算子别名，非 latent）。

> **第一性原理（校正后）**：V4 Flash 在长上下文下，**显存占用是次线性（SWA 近窗 + CSA/HCS 分级压缩稀疏），decode 算力也是次线性（按压缩态选 topk 而非全量）**。这从根上缓解 512k-1M 的显存墙和 decode 算力墙——但带来新的系统级开销：**topk 选择的算力、full↔稀疏索引映射的管理（HCS mapping）、三档压缩态 KV 的搬运与分层对齐**。这是本场景所有优化的出发点。注意：MLA 那套 latent 低秩压缩在 V4 上**不适用**，凡是基于"latent KV"的优化设想都要换成"三档压缩稀疏态 KV"。

### 0.2 昇腾 A2(910B3) 的硬件约束

- 单卡 HBM 容量通过 `npu-smi` 实时查（[platform.py:84 `_get_npu_smi_hbm_capacity_mb`](vllm-ascend/vllm_ascend/platform.py)），框架按实测 HBM 算 KV 块数——**910B3 单卡 HBM 约 64GB 量级**（具体型号配置以 npu-smi 为准）。
- 组网：HCCL（`pyhccl.py`），A2 节点内卡间互联 + 跨节点 RoCE。**A2 不是 A3 超节点**，跨卡带宽弱于 A3 的 UB 总线 —— 这点直接影响 CP（上下文并行）和 PD 传输的代价。
- 长上下文靠**上下文并行 CP**：vllm-ascend 有 `context_parallel/dsa_cp.py`（[dsa_cp.py:145 AscendDSACPMetadataBuilder](vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py)）+ `mla_cp.py` / `sfa_cp.py`，把单条长序列的 KV 拆到多卡。

### 0.3 sgl-kernel-npu 的 PD/MoE 关键算子

- `mla_preprocess`（FlashMLA 预处理算子——名字含 MLA 是内核家族别名，V4 上承载稀疏压缩 KV 的预处理，非 latent）
- `transfer_kv_dim_exchange`（PD KV 布局转换：`[layers,pages,...]`↔`[pages,layers,...]`，见 test:29/37）—— PD 分离传输时 KV 维度重排
- `cache_location_assign` / `assign_cache_op`（KV 槽位分配）
- `deepep`（昇腾版 DeepEP，MoE 专家全互联通信）

---

## 一、问题一：KV Cache 系统优化——已有能力 / 解决了什么 / 待增强 / 待构建

### 1.1 已具备能力（按本场景梳理）

| 能力 | sglang/sgl-kernel-npu | vllm/vllm-ascend | 在本场景解决的问题 |
|------|----------------------|------------------|---------------------|
| SWA 滑窗 KV | ✅ BaseSWAKVPool / swa_topk_lengths | ✅ dsa_v1 swa 路径 | 近窗精确上下文，显存按窗口而非全长 |
| HCS 分级稀疏 KV pool（KV pool 实现名沿用 MLA 别名，非 latent） | ✅ HiSparseC4DevicePool + full↔稀疏索引映射 / DeepSeekV4SingleKVPool | ✅ mla_v1.py（别名）+ mla_preprocess 算子 | 全量空间映射到压缩稀疏设备空间，512k 可装下 |
| CSA 压缩稀疏 + topk | ✅ c4/c128 + dsa indexer top-k | ✅ dsa_v1.py / sfa_v1.py（compressor_ratio） | decode 算力次线性，长上下文 TPOT 可控 |
| 三档压缩(0/4/128) | ✅ HiSparseC4DevicePool + 索引映射 + KVAndScore | ✅ compress_ratio in metadata builder | 显存进一步压缩，冷历史走 128 倍 |
| 前缀复用 | ✅ Radix/HiRadix + extra_key 隔离 | ✅ hash块 + cache_salt | agentic 多轮共享前缀复用 |
| 分层 KV(HBM↔CPU↔远端) | ✅ HiRadixCache + HiCacheController(预取/回写线程) | ✅ kv_offload 三级 tiering | 突破单卡 HBM，承载更多并发 |
| 远端 KV 池 | ✅ mooncake_store/3fs/eic/aibrix | ✅ Mooncake + ascend kv_pool(LLMDataDist/UCM) | 跨实例/跨会话 KV 共享 |
| PD 分离 KV 传输 | ✅ disaggregation + transfer_kv_dim_exchange 算子 | ✅ connector + ascend mooncake_layerwise | Prefill KV 跨节点搬到 Decode |
| 上下文并行 CP | 〜（需核实昇腾 CP） | ✅ dsa_cp/mla_cp/sfa_cp | 单条 1M 序列 KV 拆多卡 |

### 1.2 解决了什么问题 / 提升了什么

1. **显存墙（最关键）**：SWA 近窗 + CSA 压缩稀疏 + HCS 分级（c4/c128），三套机制叠加把 512k-1M 上下文的 KV 从"装不下"变成"装得下且有余量做并发"。
2. **decode 算力墙**：DSA top-k 让 decode attention 从 O(L) 降到 O(top-k)，长上下文 TPOT 不再随序列长度线性恶化。
3. **多轮复用**：Radix/HiRadix 前缀复用让 agentic 多轮的共享前缀（system prompt + 历史）跳过重复 prefill，TTFT↓。
4. **跨节点可行性**：PD 分离 + KV 维度交换算子 + 远端池，让 Prefill 算力与 Decode 显存解耦部署。

### 1.3 还可增强/优化的功能（已有基础上）

1. **压缩态 KV 的端到端打通**（高杠杆）：现在 c4/c128 压缩稀疏态主要在**计算与本地存储**生效，但 **PD 跨节点传输、分层下沉时是否全程保持压缩态**待核实（见存疑）。若传输/分层仍按解压态走，A2 较弱的跨卡带宽会成瓶颈。增强点：connector/storage 全链路传压缩稀疏 KV（c4/c128 态）。
2. **DSA 稀疏索引的分层化**：top-k 选出的"热历史"应常驻 HBM，未选中的冷历史下沉到 CPU/远端。当前分层（HiRadix/kv_offload）是**前缀粒度**，与 DSA 的 **token 重要性粒度**未对齐。增强点：让分层驱逐策略感知 DSA 的 `KVAndScore`，按重要性而非 LRU 下沉。
3. **昇腾侧 KV 压缩成熟度**：`kvcomp_attn`（attention_utils.py）尚是雏形，相比 SGLang 的 dsv4 三档（c1/c4/c128）+ swa 体系，vllm-ascend 的压缩档位/算子覆盖待补齐。

### 1.4 可补充/构建的功能（当前缺口）

1. **跨请求 cascade 复用 + DSA 的结合**：vLLM 的 cascade attention（公共前缀只算一次）目前与 DSA 稀疏是两条线，长 system prompt 的公共前缀在 DSA 下能否只算一次 indexer + 共享 top-k，待构建。
2. **agentic KV 的语义化生命周期管理**：多轮 agent 的 KV 有明确"轮次"语义（上一轮工具结果可能不再需要），当前框架按通用 LRU/前缀树管理，缺**按 agent 轮次/会话的主动 KV 回收与 pin**。
3. **全局 KV 池的前缀索引 + 亲和路由**：跨实例共享要靠 router 知道"哪个实例/远端池有哪段前缀"，当前 cache-aware 路由仍偏单实例，集群级全局索引待构建。

---

## 二、问题二：512k~1M + agentic 超多轮 —— 降 TTFT、提并发、扩展点

### 2.1 长上下文 + 多轮下 TTFT 的真实构成

TTFT = 前缀命中判定 + （未命中部分）prefill 计算 + （PD 分离时）KV 传输 + 首 token 解码。1M 上下文下，**未命中部分的 prefill 是 TTFT 大头**，agentic 多轮的关键是**让命中部分尽可能大**。

### 2.2 降 TTFT 的抓手（已有能力 + 增强）

| 抓手 | 已有能力 | 进一步优化点 |
|------|----------|--------------|
| **最大化前缀命中** | Radix/HiRadix 前缀树 + cache-aware 调度(LPM) | agentic 多轮的 KV **跨实例持久化**：上一轮在实例A，下一轮路由回A或从远端池秒拉，避免重算。需全局亲和路由 |
| **命中部分免重算** | 前缀复用直接拿 KV | DSA 下连 indexer 的 top-k 结果都可缓存复用（同前缀同 top-k），省 indexer 算力 |
| **未命中部分快速 prefill** | chunked prefill + CP 多卡分摊 | A2 带宽弱，CP 通信是瓶颈——增强 CP 的通信/计算重叠（参考 dsa_cp 的 ring 模式核实） |
| **预取隐藏延迟** | HiCacheController prefetch_thread | **预测性预取**：agent 多轮有强模式，请求到达前把上一轮 KV 从远端预拉回 HBM |
| **PD 分离传输提速** | transfer_kv_dim_exchange + layerwise | 传**压缩稀疏态 KV**（c4/c128 两档），A2 跨节点带宽占用数量级↓ |

### 2.3 提并发的抓手（显存是约束）

1M 上下文下，单请求 KV 占用大，**并发数 ≈ 总 KV 容量 / 单请求 KV 占用**。提并发就是两条：分母变小、分子变大。

- **分母（单请求 KV）变小**：CSA topk 只保留被选中的 token + SWA 近窗按窗口大小 + HCS 冷历史走 128 倍。这是 V4 的天然优势，要**把压缩档位用足**——近窗走 SWA、中段 c4、远段 c128，历史越久压得越狠。
- **分子（总容量）变大**：分层把冷 KV 外溢到 CPU/远端（HiRadix/kv_offload），HBM 只放热的。增强点：分层驱逐**对齐 DSA 重要性 score**，把 DSA 永远选不到的冷 token 优先下沉。
- **CP 横向扩容**：单条超长序列用 CP 拆多卡，单卡 KV 压力分摊。

### 2.4 待增强/补充的功能（问题二专项）

1. **DSA-aware 分层驱逐**（核心缺口）：当前分层 LRU 不感知 DSA 的 token 重要性。构建"重要性分层"——`KVAndScore` 的 score 驱动下沉决策，让 HBM 只留高分 token。
2. **压缩态 KV 全链路**：PD 传输**已确认传压缩态**（见 5.1，`_compute_transfer_block_ids` 按压缩比缩短传输块）；待补的是**分层下沉/远端池是否也保持压缩态**（5.1/5.7 未坐实）。
3. **agentic 轮次级 KV pin/evict API**：让上层 agent 框架显式声明"这段历史还要用/可丢弃"，框架据此 pin 或主动回收，比通用 LRU 精准。
4. **indexer top-k 结果缓存**：DSA 的 top-k **每步重算**（见 5.2，框架只缓存 indexer-K 不缓存 topk 结果）——同前缀复用时连 top-k 一起缓存仍是真实缺口。
5. **CP 通信优化**（A2 专项）：A2 跨卡带宽弱于 A3，CP 的 all-gather/ring 通信是长上下文瓶颈，需计算/通信重叠 + 拓扑感知。

---

## 三、问题三：P/D 负载变化 + MoE 专家负载变化 —— 系统性扩缩容与弹性

### 3.1 P 和 D 的资源配比矛盾（本质）

PD 分离后，Prefill 是**算力密集、突发**（长上下文 prefill 吃满算力），Decode 是**显存密集、持续**（KV 常驻、长尾）。固定 P:D 配比必然有一方先成瓶颈。本场景下 1M 上下文让 Prefill 更重、agentic 多轮让 Decode 的 KV 更多——配比矛盾被放大。

**已有能力**：
- vllm-ascend `epd_disaggregated`（E/P/D 三段分离）+ `epd_load_balance_proxy_layerwise_server`，E/P/D 可各自独立部署、独立扩缩。
- vllm-ascend `disaggregated_prefill_v1` 示例 + Qwen3-235B-disagg-pd 配置（实证多机 PD 部署存在）。

**待构建的弹性机制**：
1. **P/D 动态配比调度器**：基于实时指标（Prefill 队列深度/算力利用率 vs Decode KV 占用/TPOT）动态调整 P:D 实例数。当前是静态配比 + 手动，缺自动闭环。
2. **角色可切换实例**：空闲 Prefill 实例临时承接 Decode（或反之），而非固定角色。需要 KV 状态的快速迁移能力（已有 transfer_kv 算子作基础）。
3. **goodput 驱动的全局调度**：以"满足 SLO 的有效吞吐"为目标函数，统一调度 P/D 资源，而非各自打满。

### 3.2 MoE 专家负载均衡（已有能力较强）

DeepSeek V4 是 MoE，专家热度不均会让个别 NPU 过载。两框架都有 EPLB：

- **vllm-ascend EPLB 多策略**（成熟度高）：
  - `SwiftBalanceEplb`（[policy_swift_balancer.py:29](vllm-ascend/vllm_ascend/eplb/core/policy/policy_swift_balancer.py)）：动态热度统计 + 不均衡阈值 `imbalance_threshold=1.01`（policy_swift_balancer.py:38），超阈值才重排，且 `constraint_expert_local_exchange` 约束本地交换（减少搬运）。
  - `policy_flashlb`：`lpt_deployment`（最长处理时间优先部署，flashlb.py:212）+ `compute_score`（负载均衡分 = max_device_load×num_devices/total_load，flashlb.py:294）—— 这是把专家部署当**多机调度 LPT 问题**求解。
  - default / random 兜底。
  - `EplbUpdator.warm_up_eplb`（eplb_updator.py:150）预热。
- **vLLM 主线**：EPLB 在线 rearrange + 异步 + 弹性 EP `standby` 组（[standby_state.py:21-33](vllm/vllm/distributed/elastic_ep/standby_state.py)，预备 world/dp/ep/eplb group 热扩容）。
- **昇腾 DeepEP 算子**（sgl-kernel-npu/csrc/deepep）：专家全互联通信的底层支撑。

### 3.3 系统性弹性策略机制（综合判断）

把 P/D 配比 + MoE 均衡 + 扩缩容拉通成一个闭环，建议三层：

```
1. 监控层（颗粒度对齐）→ verify: 实时采集 P队列/D KV占用/专家热度/SLO
   - P侧：prefill 队列深度、算力利用率、TTFT
   - D侧：KV 显存水位、running 数、TPOT
   - MoE：per-layer 专家 imbalance（已有 calculate_imbalance）

2. 决策层（统一目标 goodput）→ verify: 三类弹性动作不打架
   - P/D 配比：队列深→加P；KV满/TPOT涨→加D
   - MoE 均衡：imbalance>阈值→EPLB rearrange（已有 SwiftBalance）
   - 整体扩缩：goodput 不达 SLO→整体 scale up（已有 elastic_ep standby）

3. 执行层（已有算子/机制作基础）→ verify: 扩缩不中断服务
   - 专家重排：constraint_local_exchange 减搬运（已有）
   - 实例增减：standby group 热接入（已有 vLLM）
   - KV 迁移：transfer_kv_dim_exchange + 远端池（已有）
```

**待增强/构建**：
1. **P/D 与 MoE 弹性的协同决策**（当前缺口）：现在 EPLB（专家）和 PD 扩缩是**独立两条线**，可能互相打架（加 D 实例又触发 EPLB 全局重排）。需统一决策器，对齐扩缩节奏。
2. **专家均衡感知 PD 角色**：Prefill 和 Decode 的专家热度分布不同（prefill batch 大、decode batch 小），EPLB 应**分 P/D 角色独立均衡**，而非全局一张表。
3. **弹性的状态迁移成本建模**：A2 带宽弱，scale 时的 KV/专家权重搬运代价高，决策需把搬运成本纳入（避免抖动式频繁重排）。SwiftBalance 的阈值 1.01 偏激进，A2 上可能需调高滞后。
4. **goodput 闭环自动化**：目前 P/D 配比偏静态/手动，缺以 SLO 为目标的自动配比闭环。

---

## 四、综合判断（一页纸结论）

1. **这个场景的底层逻辑**：V4 Flash 的 HCS+CSA+SWA 三套稀疏压缩机制（非 MLA latent），已从根上把"长上下文显存墙 + decode 算力墙"变成次线性问题——**框架已具备承载 512k-1M 的基础能力**。瓶颈从"装不下/算不动"转移到了"**怎么把压缩/稀疏的优势在传输、分层、调度、弹性各环节用足**"。

2. **三个最高杠杆的增强点**（按价值排序，且都在引擎/框架层，契合"框架层优先于算子"）：
   - **① 压缩态 KV 全链路**：PD 传输已确认走压缩态（5.1），杠杆点收窄为**把压缩态延伸到分层下沉/远端池**（5.7 待坐实），直击 A2 带宽弱的痛点。
   - **② DSA-aware 分层驱逐**：分层下沉对齐 token 重要性 score，而非通用 LRU——长上下文提并发的关键。
   - **③ P/D + MoE 统一弹性闭环**：把现有的 EPLB（成熟）、elastic_ep standby（成熟）、PD 三段分离（已有）拉通成 goodput 驱动的统一决策器。

3. **昇腾线的差距**：vllm-ascend 的稀疏注意力/EPLB/PD 已成体系（dsa_v1/sfa_v1/mla_cp[别名]/SwiftBalance/epd），但 **KV 压缩档位（kvcomp 雏形）、CP 通信效率（A2 带宽）** 是相对 SGLang dsv4 三档+swa 体系的待补点。

---

## 五、待核实问题的代码核实（基于 vllm/vllm-ascend，model_runner_v1 入口）

> 本节把原 6 条存疑逐条用本地代码坐实。每条给出**结论 + 代码依据 + 状态标识**（✅已确认 / ⚠️部分确认 / ❓仍不确定）。核实入口：`vllm-ascend/vllm_ascend/worker/model_runner_v1.py`。

### 5.1 压缩态 KV 是否全链路保持（PD 传输） — ✅ 已确认：传压缩态

**结论**：PD 跨节点传输传的是**压缩态 KV**，按 group 的压缩比缩短传输块数，不是解压全量态。

**代码依据**（`mooncake_hybrid_connector.py`）：
- `use_compress = hasattr(hf_config, "compress_ratios")`（:424/1113）——按模型 config 识别 V4 压缩。
- 逐 group 取压缩比：`group_compress_ratio[i] = g.kv_cache_spec.compress_ratio`（:1144）。
- **决定性证据** `_compute_transfer_block_ids`（:1206）：
  ```python
  if self.use_compress and self.num_swa_blocks[i] == 0:
      group_token_len = prompt_len // self.group_compress_ratio[i]   # :1210 压缩比缩短长度
  ...
  group_block_len = math.ceil(group_token_len / self.group_block_size[i])  # :1213
  transfer_block_ids.append(blocks[:group_block_len])                # :1215 只传压缩后的块
  ```
  压缩比 4 只传 1/4 的块，压缩比 128 只传 1/128 —— **传输量随压缩比数量级下降，确凿是压缩态传输**。
- KV buffer 维度也是压缩维：`k_head_dim = kv_lora_rank`（:436，压缩后的 latent 维度名，注意这里是 V4 压缩 KV 复用了 kv_lora_rank 这个字段名）。
- 配套：`need_truncate` + `_truncate_request_for_prefill`（:1116/1180）做 prefill 末 token 截断重算（压缩/SWA/Mamba 的累积态依赖最后一个 token 在 decode 侧重算，:1176 `num_prompt_tokens - 1`）——这是压缩态 PD 分离的正确性处理。

> **修正原增强点①**：原文担心"传输可能走解压态、A2 带宽吃紧"——代码证明 vllm-ascend **已经在传压缩态**。增强点①应改为：确认**分层下沉/远端池**是否也保持压缩态（本次只坐实了 PD 传输，分层侧未读到，见 5.7）。

### 5.2 DSA top-k 结果是否可缓存复用 — ⚠️ 部分确认：缓存的是 indexer-K，topk 每步重算

**结论**：框架缓存的是 **DSA indexer 的 K cache**（用于 topk 计算的输入），**不是 topk 选择结果本身**；topk 每个 forward 重新算。

**代码依据**（`dsa/dsa_indexer.py`）：
- 缓存 indexer-K：`_store_index_k_cache`（:1259）"Store DSA indexer K cache for current step"，走 `fused_store_index_k_cache`（:1293）。这复用的是**历史 K**（避免重算 K），跨 step 累积。
- topk 每步算：`_get_topk_ragged`（:794）在每次 forward 调用（:166），产出 `topk_result`；没有"按前缀缓存 topk 结果并复用"的逻辑。
- 有 CP 变体 `_get_topk_ragged_with_cp`（:1021），是为上下文并行分布式算 topk，仍是每步算。

> **判断**：indexer-K 的缓存已省掉"重算 K"的开销，但 **topk 选择本身每步重算**——5.x 章提的"topk 结果缓存复用"优化点**仍成立、是真实缺口**。状态：⚠️（K 缓存确认，topk 复用确认缺失）。

### 5.3 SGLang 昇腾 CP（上下文并行）成熟度 — ⚠️ 部分确认：通用后端有，DSV4 专用路径未确认

**结论**：SGLang 昇腾**通用 attention 后端已支持 CP**；但 **DSV4 专用昇腾后端的 CP 覆盖未坐实**。

**代码依据**：
- server_args 有开关：`attn_cp_size`（:888）、`enable_dsa_prefill_context_parallel` + `dsa_prefill_cp_mode="round-robin-split"`（:913/914）、`enable_prefill_context_parallel`（:915）。`attn_cp_size = tp_size // dp_size`（:3594）。
- 通用昇腾后端**实际用 CP**：`ascend_backend.py` 里 `cp_size`（:230/263）、`self.attn_cp_size = model_runner.attn_cp_size`（:365）、`attn_cp_metadata`（:915）——✅ 落地。
- **但** `ascend_dsv4_backend.py` 里 grep `cp_size/context_parallel/attn_cp` **为空**——DSV4 专用路径是否走 CP，或复用通用后端的 CP，本次未读出明确证据。

> 状态：⚠️。通用 CP 成熟，DSV4×CP×昇腾 的组合需进一步走读 `ascend_dsv4_backend.py` 的 forward 是否调用通用 CP 路径。

### 5.4 EPLB 是否分 P/D 角色 — ❓ 仍不确定（偏向"不区分"，但无直接否定证据）

**结论**：未找到 EPLB **按 prefill/decode 角色分别均衡**的代码证据；adaptor 层 grep `prefill/decode/disagg/role` **为空**，倾向"全局一张专家热度表"，但不能据"grep 空"下死结论。

**代码依据**：
- `eplb/adaptor/vllm_adaptor.py` 中 grep `prefill|decode|is_prefill|disagg|role|kv_producer|kv_consumer` **无命中**。
- `EplbUpdator.compute_and_set_moe_load`（eplb_updator.py:138）计算 MoE 负载——未见按 P/D 角色分表的参数。

> 状态：❓**仍不确定**。原因：grep 关键字为空只能说明"没有显式按角色命名的逻辑"，但 PD 分离下 P/D 是**独立部署的不同实例/进程**，每个实例的 EPLB 天然只统计本实例（本角色）的负载——也就是说"分 P/D"可能是**部署拓扑自带的**，而非 EPLB 代码里显式区分。要确认需读 PD 分离部署下 EPLB 实例的初始化范围（是否每个 P/D 实例各跑一份 EplbUpdator），本次未追到该层，**没有明确结论**。

### 5.5 910B3 实际 HBM/带宽数字 — ✅ 已确认：纯运行时读取，代码无硬编码

**结论**：框架**不硬编码** 910B3 的 HBM/带宽，全部运行时从 `npu-smi` 实测。

**代码依据**（`platform.py`）：`_get_npu_smi_hbm_capacity_mb`（:84）解析 `npu-smi` 的 "HBM Capacity(MB)" 字段（:94），`get_*` 用 `hbm_capacity_mb * 1024 * 1024`（:287）算可用显存。全仓未见 `910B3=64GB` 之类常量。

> 状态：✅。**本文不臆测具体 GB/带宽数字**——需以实机 `npu-smi info` 输出为准。这条本就是"按实测"，结论：代码侧无硬编码、依赖实机。

### 5.6 elastic_ep 在昇腾的可用性 — ✅ 已确认：有动态 EPLB（专家重排），无 elastic_ep（整机弹性扩缩容）

**结论**：要区分两个不同的东西——
- **动态 EPLB（专家重排）**：✅ 昇腾**有**。`model_runner_v1.py` import `EplbUpdator`/`D2DExpertWeightLoader`/`EplbProcess`（:128-131），`self.dynamic_eplb = eplb_config.dynamic_eplb`（:493）。运行中按热度重排专家、D2D 搬运专家权重。
- **elastic_ep（运行中增减 GPU/改 world_size 的整机弹性扩缩容）**：✅ 确认昇腾**没有**。vllm-ascend 全仓 grep `elastic_ep / enable_elastic_ep / ElasticEP / standby_state / scale_up / scale_down` **零命中**。`EplbUpdator` 的 `world_size = dist.get_world_size()`（:45）是**固定值**，只有 `update_iteration`/`wakeup_eplb_worker`/`update_expert_weight`（专家重排），**无 add_node/remove/reconfigure world_size** 的弹性逻辑。

> 状态：✅。**这是 vLLM 主线（有 `distributed/elastic_ep/` + standby group）与 vllm-ascend 的实质差距**：昇腾能在固定规模内做专家负载均衡，但不能像主线那样运行时弹性增减实例。第三章"P/D + MoE 统一弹性闭环"在昇腾上，**MoE 重排有基础（dynamic_eplb），整机弹性扩缩容需从零构建或等主线 elastic_ep 适配 NPU**。

### 5.7 核实后仍开放的问题（明确标注未结）

1. **分层下沉/远端池是否保持压缩态**（5.1 只坐实了 PD 传输是压缩态，HiCache/kv_offload 侧未读）— ❓ 未确认。
2. **DSV4 专用昇腾后端是否走 CP**（5.3）— ⚠️ 通用后端有，DSV4 专用路径未坐实。
3. **EPLB 是否按 P/D 角色均衡**（5.4）— ❓ 无明确结论，需读 PD 分离下 EplbUpdator 的实例化范围。
4. **elastic_ep 适配 NPU 的计划/分支**（5.6）— 确认当前主干无，是否有 WIP 分支不在本地代码范围，无法判断。
