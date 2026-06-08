# 大模型推理集群：KV Cache 与调度优化分析

> 范围：当前目录全部开源代码仓（vllm / vllm-ascend / sglang / sgl-kernel-npu / dynamo / Mooncake / tokenspeed / speculators / TileRT）
> 维度：围绕「推理集群」性能优化，聚焦 **KV Cache** 与 **调度** 两条主线
> 证据来源：各仓 `docs/design/` 设计文档 + 近 3 周（2026-05-18 → 2026-06-07）提交记录
> 生成日期：2026-06-08

---

## 第一部分：关键优化特性、算法与框架（跨仓体系）

按「解决什么问题 → 提出什么方案 → 达到什么效果」组织。

### 1. KV Cache 分页与前缀复用（单实例基础层）

| 框架 | 算法 | 问题 | 方案 | 效果 |
|------|------|------|------|------|
| **vllm** | **PagedAttention** | KV cache 连续分配造成显存碎片、利用率低 | 借鉴 OS 分页，把 KV 切成固定大小 block，按需分配、逻辑/物理地址解耦 | 显存碎片趋零，吞吐显著提升，成为行业事实标准 |
| **vllm** | **Automatic Prefix Caching (APC)** | 多请求共享前缀（system prompt/few-shot）被重复 prefill | block 内容哈希 → 相同前缀 block 命中复用，免重算 | 共享前缀场景 TTFT 大幅下降 |
| **sglang** | **RadixAttention** | APC 的 block 哈希匹配粒度粗、前缀树管理弱 | 用 **基数树（Radix Tree）** 组织 KV：根→叶路径=请求前缀，节点可分裂，支持高效 match/insert/evict（LRU） | 前缀复用率更高，multi-QA/agent 场景命中率提升 |
| **dynamo** | **KVPublisher + KVIndexer** | 集群里每个 worker 各自有前缀树，路由层看不到全局缓存分布 | worker 发出 KV stored/removed 事件，Indexer 用全局前缀树聚合 → 供路由决策 | 把单机前缀复用提升为**集群级缓存感知** |
| **tokenspeed** | C++ `radix_tree` + `kv_prefix_cache` | Python 调度开销大、前缀缓存逻辑分散 | 纯 C++ 调度器内置 radix tree / paged cache / eviction（LRU），page_hasher 做块哈希 | 调度路径低开销，前缀缓存与调度紧耦合 |

### 2. 分层 KV Cache —— 多级缓存卸载（突破单卡显存墙）

核心问题：GPU 显存有限，KV cache 容量是长上下文/高并发的瓶颈；但 host DRAM、SSD、分布式存储是「闲置」的廉价容量。

| 框架 | 体系 | 方案 | 效果 |
|------|------|------|------|
| **sglang** | **HiCache**（L1/L2/L3 三级） | 仿 CPU cache：L1=GPU 显存（私有）、L2=host DRAM（私有）、L3=分布式存储（集群共享）。`HiRadixTree` 记录每段 KV 所在层级；三大操作 local match / prefetch / write-back | KV 容量大幅扩展，长上下文/多轮命中率提升；详见 lmsys hicache blog |
| **sglang** | HiCache 数据传输优化 | ① 计算-传输 overlap（算第 N 层时载入 N+1 层）② **GPU-assisted I/O kernel**（比 cudaMemcpyAsync 快 **最高 3×**）③ page_first / page_first_direct 内存布局做 zero-copy ④ MLA 只让一个 rank 回写去冗余 | CPU↔GPU 传输延迟被隐藏，I/O 带宽利用率提升 |
| **vllm** | 多级 KV offload + `OffloadConnector` | CPU 作为二级 tier，新增**对象存储**作为三级 tier；token-offset 选择性卸载；Triton 小块 CPU→GPU swap 快路径 | 显存外溢有处可去，HMA 混合模型也支持 tiering |
| **dynamo** | **KVBM**（G1-G4 四级） | G1 GPU / G2 host pinned / G3 本地 NVMe SSD / G4 远程对象存储；TransferManager 异步编排 4 条传输路径（D→H/H→D/H→Disk/Disk→D）；NIXL + GDS zero-copy；按 sequence_hash 去重 | 统一的 KV 分层块管理，跨框架（vLLM V1 connector）可插拔 |
| **Mooncake** | **Mooncake Store**（分布式 KV pool） | 定位为**分布式 KV cache**（非通用缓存）：对象级 Put/Get/Remove、多副本、强一致、RAM→SSD 多层；Master 管元数据，Client 双角色（既发请求又贡献内存段） | 把整个集群 DRAM/SSD 聚成一个全局 KV 池，跨实例共享 |

### 3. Hybrid / 混合注意力架构的 KV 管理（新模型适配）

问题：Mamba/SWA/线性注意力等混合模型，不同层 KV 形态不同（全量 vs 滑窗 vs 状态），传统统一 block 管理失效。

- **vllm — Hybrid KV Cache Manager**：核心挑战是「不同注意力类型层数不同 → page size 不一致」。方案：单内存池，按注意力类型分 **KV Cache Group**（组内同类型、组间统一 page size），用层数比例分组（Gemma-3 取 min 层数分组）减少分配调用；滑窗层只为最近 `sliding_window_size` token 留槽。效果：混合模型也能高效分页 + 支持层级化前缀缓存规则。
- **vllm — SWA 选择性前缀保留**：DSv4 滑窗 KV 命中只需保留最后 window 内 token，无需全量。
- **tokenspeed — `hybrid_prefix_cache` + `mamba_slot` + `mamba_eviction_manager`**：为 mamba 状态族设计独立的 slot/eviction 管理，device-side 实现 v4 hybrid cache 前缀缓存。
- **vllm-ascend — Hybrid & Mamba Align Prefix Cache**：昇腾侧对齐 mamba 前缀缓存、310P 支持 Prefix Mamba Cache。

### 4. PD 分离（Prefill / Decode Disaggregation，集群级架构）

问题：prefill（计算密集、决定 TTFT）与 decode（访存密集、决定 ITL）混在一个实例里互相干扰，且无法独立调参/扩缩。

- **vllm — Disaggregated Prefilling**：prefill / decode 分到不同实例，可分别配 TP/PP 独立调 TTFT 与 ITL；解决 chunked prefill 难调 chunk size 导致的尾延迟问题。**明确说明：不提升吞吐，目的是控制尾 ITL**。支持 **9 类 connector**：Nixl / LMCache / P2pNccl / **Mooncake** / MoRIIO / Multi / Offloading / FlexKV / Example。
- **Mooncake 架构**：KVCache-centric 分离式架构的奠基者——独立 prefill/decode 池 + 全局 KV cache 池，靠 Transfer Engine 跨池传 KV。
- **sglang — PD Disaggregation**：经 Mooncake TransferEngine 实现；HiCache 可在 P/D 两端开启进一步优化。
- **dynamo — Disagg Serving + 拓扑感知**：topology-aware KV transfer / scheduling，把 KV 路由约束传播到 decode 端。

### 5. KV-aware 调度与路由（集群级负载均衡）

问题：多 worker 集群中，把请求随机/轮询分发会浪费前缀缓存命中，且 prefill/decode 负载不均。

- **dynamo — KV Router（核心）**：跟踪每 worker 的 **Potential Active Blocks（decode 负载）** 与 **Potential New Prefill Blocks（需新算的 token）**。代价函数：
  ```
  adjusted_prefill = max(prefill - overlap_credit·device_overlap
                         - host_weight·host_overlap - disk_weight·disk_overlap
                         - shared_mult·shared_beyond, 0)
  cost = prefill_load_scale · adjusted_prefill + decode_blocks
  ```
  选 cost 最低的 worker；`router_temperature` 做 softmax 采样平衡负载。**效果：高 overlap credit 偏向缓存复用（改善 TTFT），低 credit 偏向均衡（改善 ITL），可按 SLO 调。**
- **sglang — cache-aware router / LoadBasedPolicy**：agentic router 引入基于负载的策略；`num_waiting_uncached_tokens` 作为新负载指标。
- **Mooncake — Conductor**：全局调度器 + indexer，负责 KV 池的放置与调度。

### 6. 传输层（KV 搬运的底座）

- **Mooncake — Transfer Engine**：GPUDirect RDMA **zero-copy**，DRAM/VRAM↔DRAM/VRAM 直传；**多 NIC 聚合带宽**；大对象 striping + 并行 I/O 打满线速；CPU 开销极低。新增 TEnT（transport selector / slice-spraying / QoS / failover）。
- **NIXL**：统一存储插件 API（3FS / GDS / S3 兼容对象存储），被 vllm、dynamo、sglang 共用作传输后端。

### 7. 调度器机制（迭代级吞吐/延迟）

- **连续批处理（continuous batching）** + **chunked prefill**：贯穿 vllm / sglang / tokenspeed，prefill 切块插入 decode 流，平衡 TTFT 与 ITL。
- **overlap / 零开销调度**：sglang 用 CudaGraphBufferRegistry 等减少调度气泡；vllm Model Runner V2 避免 PP 气泡。
- **mixed prefill-decode batching**：tokenspeed 把 prefill 与 decode 混批，且兼容 MLA 的投机解码。
- **tokenspeed C++ 调度器**：scheduler / execution_plan / FSM 事件机（pd_events / cache_events / forward_events）全 C++，调度开销低。

### 8. 自动扩缩容（集群弹性）

- **dynamo — Planner**：自动伸缩控制器，throughput-based（profiling+流量预测）与 load-based（实时指标+在线回归）两种模式；用 `prefill_correction = actual_ttft/expected_ttft` 等校正因子，**显式建模 prefix cache 命中对 TTFT 的影响**，按 SLO 调 P/D 实例数。

---

## 第二部分：近 3 周更新中 KV Cache / 调度相关内容

按主题归类，标注代表性 PR、解决的问题与效果。

### A. 分层 KV Cache / 卸载

| 仓 | 更新 | 解决问题 / 效果 |
|----|------|----------------|
| vllm | 对象存储作为多级 offload 第三 tier（#41968）；HMA 混合模型支持 tiering（#44287）；SharedOffloadRegion 对齐 page size（#43689）；`on_schedule_end()` 钩子分离 step 生命周期与事件 drain（#44206） | KV 卸载层级更深、混合模型可用、卸载与调度解耦更干净 |
| vllm | OffloadConnector 小块 CPU→GPU swap 的 Triton 快路径（#42212）；解决多 async KV load 死锁（#44560） | 小块搬运提速、异步加载稳定性 |
| vllm | XPU 平台支持 CPU KV offload + tiering（#36423） | 卸载能力扩展到 Intel XPU |
| vllm-ascend | **Simple yet General CPU KV Cache Offloading**（#8743）；Mooncake **SSD offload** embedded client（#9731）；KV pool 加载失败块免 hybrid 重算（#9701）；layerwise KV cache events（#9468） | 昇腾侧补齐 CPU/SSD 多级卸载与事件机制 |
| sglang | HiCache 支持 mooncake store **layer-first 布局**（#27454）；修 SWA admission 预算漏算 HiCache load-back（#27391）；修 HiMamba HiCache L3 prefetch 卡死（#27366）；mamba prefetch 长度截断到可用 host KV（#26945）；修 PD 下 L3 cache 命中统计（#27046） | HiCache 在 SWA/Mamba/PD/PP 各组合下的正确性与布局优化 |
| Mooncake | 标准 store 服务支持 **SSD offload 配置**（#2261）；Master 重启后 SSD offload **自动恢复**（#2077）；RemoveAll 真正删 SSD 文件（#2283）；空 offload 心跳避免 INVALID_REPLICA（#2151） | SSD 卸载的配置化、容错与一致性 |
| dynamo | KVBM **G4 对象存储 offload** 在 replay 中模拟（#9939）；router 暴露 host/disk cache 命中权重 CLI/env（#10157） | 四级卸载可测、路由可调缓存权重 |
| tokenspeed | kvstore 支持 **mamba L2 cache 传输**（#162）；写/执行流解耦（#279）；修 KV cache pool 过小 bug（#194） | mamba 状态卸载、流水正确性 |

### B. 前缀缓存 / Radix 匹配

| 仓 | 更新 | 解决问题 / 效果 |
|----|------|----------------|
| sglang | **改匹配算法降低 radix cache match 开销**（#27364）；`UnifiedTree` 统一树体系（多 PR）；MambaTokenToKVPool / SWATokenToKVPool allocator 重构（#27256 #26676） | 前缀匹配热路径提速、分配器模块化 |
| vllm | DSv4 滑窗 KV **选择性前缀保留**（#43447）；修 EAGLE/MTP lookahead block 在 SWA 前缀掩码缓存（#44082）；修 DFlash lookahead block 缺失致前缀缓存损坏（#42971） | 混合/投机场景前缀缓存正确性 |
| vllm-ascend | **PCP/DCP 下启用前缀缓存**（#9638）；Hybrid & Mamba Align Prefix Cache（#9533）；310P 支持 Prefix Mamba Cache（#9514） | 昇腾并行/混合模型前缀缓存落地 |
| dynamo | tokenizer 层 **多轮 L1 前缀缓存扩展**（moka W-TinyLFU 后端，#10201）；positional indexer find_matches 可选二分查找（#10181） | 多轮对话前缀复用、索引匹配提速 |
| tokenspeed | DSv4 **device-side hybrid cache 前缀缓存**（#146）；split prefill 复用前缀缓存（mha backend，#178）；prefix state snapshot replay 复用减少快照（#329） | v4 混合缓存设备侧前缀缓存、prefill 复用 |

### C. PD 分离 / KV Connector / 传输

| 仓 | 更新 | 解决问题 / 效果 |
|----|------|----------------|
| vllm | KVConnector PP-aware handshake 聚合 + 中间 PP 输出（#43720）；KVCacheSpec 可插拔（#37505）；NixlConnector 启动 `kv_both` 弃用周期（#43874）；EPLB Nixl zero-copy（#41633） | connector 适配 PP、规范化、零拷贝 |
| vllm | **PD + Nixl Mamba 前缀缓存模式**（#42554）；`scheduler_block_size` 贯穿 KVCacheManager/Coordinator（#44165） | mamba 模型 PD 前缀缓存、调度块大小可配 |
| vllm-ascend | Mooncake Connector 支持 **Hybrid PCP/DCP**（Qwen3.5，#9809）；hybrid attention for mooncake connector（#8850）；HMA in AscendMultiConnector（#9782）；compress ratio + block_ids cutting（#9808） | 昇腾 Mooncake 连接器支持混合并行/注意力 |
| dynamo | **拓扑就绪的分离式服务 Phase 3**（#9815）；拓扑约束传播到 decode（#9893）；MM-aware KV routing via pad_value（#9561）；per-request Mooncake trace 抓取（#10381） | 拓扑感知 PD、多模态 KV 路由 |
| Mooncake | 启用 **Ubtransport / 优化 UrmaEndpoint**（#2196）；TEnT transport-selector / slice-spraying / QoS / progress_worker（多文件） | 传输后端扩展与 QoS/故障切换 |
| tokenspeed | 修 PD 投机 bootstrap 输入种子（#286） | PD + 投机解码正确性 |

### D. 调度器 / 路由 / 弹性

| 仓 | 更新 | 解决问题 / 效果 |
|----|------|----------------|
| dynamo | KV-router **独立 slot tracker 服务**（#10291）；从独立 KV-router 向 frontend 透传 per-request timing（#10182）；暴露建模的 prefill time（#10190）；DP-rank 粘性会话亲和（#9920）；硬化取消与副本同步（#10331）；移除 scheduler hop 改善 BSI（#10200）；实验性 **thunderagent_router 程序级调度**（#9448） | KV 路由服务化、时序可观测、亲和性、链路缩短 |
| dynamo | mocker 加 **TRT-LLM 调度模拟**（#10193）、终止拒绝不可调度 TRT-LLM 请求（#10287）；VRAM-aware GPU 测试调度（#10126） | 调度策略可仿真验证 |
| sglang | `num_waiting_uncached_tokens` 负载指标（#27174）；健康失败触发调度诊断（#26757）；agentic router LoadBasedPolicy（#26480）；router 改用 CLI flags 配置（#27073） | 负载感知路由、可观测、易用 |
| tokenspeed | **pause/resume 调度控制 API**（#346）；修 scheduler req pool move 赋值泄漏（#182）；修状态族 paged-cache 组 admission 过度授信（#249） | 调度可控性与正确性 |
| vllm | 校验 `max_num_scheduled_tokens >= 0`（#44207）；PD 测试默认对齐 HMA（#44174） | 调度参数健壮性 |
| vllm-ascend | 纯 prefill 批的**计算-通信 overlap**（#9504）；A5 server PD 端点配置（#9690） | prefill 阶段通信隐藏 |

### E. 混合模型 KV 状态管理（新架构驱动）

- **vllm**：KDA conv 状态统一进一个 cache 匹配 2-state SSM 布局（#44539）；从 EngineCore 向 frontend 同步 hybrid Mamba 的 block_size（#42967）。
- **tokenspeed**：mamba `l2 cache` 传输（#162）、DSv4 state-family paged-cache 组的 admission/sizing 修正（#249 #213）。
- **sglang**：mamba prefill 分配开销降低（#25000）、mamba_extra_buffer ping-pong 槽泄漏修复（#26941）。

### F. 可靠性 / 正确性框架（值得单列）

- **sglang — scripted-runtime + KV-canary**：新增脚本化运行时测试框架与 **KV-canary 故障注入自检体系**（#27410-27413、#26816-26821 等成体系）——real-data KV 校验、token-id 校验、SWA 偏离报告、PD 扰动 e2e。**解决：KV cache 在 PD/PP/SWA 等复杂组合下的静默损坏难以发现的问题**，提供 kernel-run-counter 健康检查。

---

## 小结：集群级优化的「四横一纵」

- **横向四层**：① 单卡分页复用（PagedAttention/RadixAttention）② 多级卸载（HiCache/KVBM/Mooncake Store）③ PD 分离（Transfer Engine/NIXL connector）④ 集群路由（KV Router/cache-aware）。
- **纵向一线**：传输底座（Mooncake Transfer Engine RDMA zero-copy + NIXL）贯穿②③④。
- **近期趋势**：(1) 卸载向**对象存储/SSD 第 3-4 级**延伸且强调容错；(2) 前缀缓存与**混合模型（Mamba/SWA）**深度适配；(3) 路由**服务化 + 缓存权重可调 + 时序可观测**；(4) 正确性验证体系化（KV-canary）。

> 可延伸专题：① DSv4 跨框架 KV/前缀缓存实现对比（vllm vs sglang vs tokenspeed vs vllm-ascend）；② HiCache vs KVBM vs Mooncake Store 三套分层方案的架构对比；③ dynamo KV Router 代价函数与 SGLang cache-aware router 的调度策略对比。
