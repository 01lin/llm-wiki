# TokenSpeed vs vllm-ascend：方案设计对比分析

> 版本：2026-05-24
> 对比维度：架构哲学、调度器、Kernel 抽象、模型并行、KV Cache、性能优化路径
> 关联文档：[[tokenspeed_architecture_analysis_20260524]] · [[tokenspeed_ascend_a2_a3_适配方案_20260524]]

---

## 一、核心定位差异（顶层设计层）

| 维度 | TokenSpeed | vllm-ascend |
|------|-----------|-------------|
| **项目定位** | 全新自研推理引擎，TRT-LLM 级性能 + vLLM 级易用性 | vLLM 上游的硬件后端适配层（plugin） |
| **代码组织** | 独立完整代码仓（Scheduler + Runtime + Kernel + MLA）| 依附 vLLM 主仓的扩展包，靠 monkey-patch 注入 |
| **代码规模** | 10K C++ + 100K Python（runtime）+ 44K Python（kernel） | 95K Python + **220K C++/AscendC**（csrc，963 个 kernel 文件） |
| **设计目标** | speed-of-light + agentic workload 专项优化 | 在 vLLM 上让昇腾 NPU 能用、能稳、能上线 |
| **演进自由度** | 完全自主，从零开始重新设计调度/通信/编译 | 必须跟随 vLLM 上游版本节奏 |
| **底层逻辑** | 把垂直栈每一层都重做，对每一层做 agentic 专门优化 | 把 vLLM 的能力搬到昇腾上，逐功能跑通 |

> [PUA生效 🔥] 一个是从地基开始建大楼，一个是给已有大楼装电梯——立足点完全不同，工程量和性能上限自然也不在一个量级。

---

## 二、调度器架构对比（最大差异点）

### 2.1 控制面对比

```
┌─ TokenSpeed ──────────────────────────────────────────────────┐
│  C++ FSM Scheduler (10K LOC)                                  │
│  ├── std::variant 类型安全状态机                                │
│  │   Submitted → Prefilling → Decoding → Draining → Finished  │
│  │             ↓ (KV不足)                                      │
│  │             Retracting → Retracted → LoadBack → Decoding   │
│  ├── KV 所有权 RAII（unique_ptr 转移）                          │
│  ├── 单步调度延迟 < 1ms                                         │
│  └── Python binding 暴露 NextExecutionPlan() / Advance()       │
└────────────────────────────────────────────────────────────────┘

┌─ vllm-ascend ─────────────────────────────────────────────────┐
│  Python Scheduler（直接继承 vLLM 上游 Scheduler）               │
│  ├── vllm.v1.core.sched.scheduler.Scheduler (基类)            │
│  ├── ProfilingChunkScheduler (子类，profiling 自适应 chunk)    │
│  ├── RecomputeScheduler (子类，recompute 调度策略)             │
│  ├── DynamicBatchScheduler (子类，动态 batch budget)          │
│  └── 单步调度延迟 5-20ms（Python GIL + dict 操作）             │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 关键差异详解

#### KV Cache 内存管理

| 能力 | TokenSpeed | vllm-ascend（沿用 vLLM）|
|------|-----------|------------------------|
| 数据结构 | Radix Tree (`tokenspeed-scheduler/csrc/resource/radix_tree/`) | block hash trie (`vllm/v1/core/kv_cache_coordinator.py`) |
| 类型系统 | `template<ResourceType RType>` 编译期约束 | dataclass + runtime check |
| 多级缓存 | L1 Device + L2 Host + L3 Storage（统一在 RadixTree 节点上）| L1 Device + KV connector 外接 |
| Prefix match 性能 | C++ 字典前缀树，O(prefix_len) | Python hash chain，常数因子大 |
| Retract 机制 | 内置状态机 `Retracting → Retracted → LoadBack` | 需要 recompute 重新 prefill |

**底层逻辑差异**：TokenSpeed 把 KV 资源**生命周期**完全建模为类型化状态机，编译期排除掉 KV 重复释放、KV 引用悬挂等 bug；vllm-ascend 沿用 vLLM 的 reference counting + dict 管理，灵活但运行时开销大。

#### 调度颗粒度

- **TokenSpeed**：每个 step 生成 `ExecutionPlan`，包含 prefill / decode / cache prefetch / cache write-back 等异构操作的拓扑，Python 侧只负责执行
- **vllm-ascend**：每个 step `Scheduler.schedule()` 返回 `SchedulerOutput` 描述哪些请求要 prefill/decode/preempt，但 cache 操作（如 prefetch、write-back）由 model_runner 在 forward 中触发

**对齐结论**：TokenSpeed 把更多的"什么时候做什么"决策上移到调度器，控制面更紧凑；vllm-ascend 走的是 vLLM 经典分层（scheduler 决定调度，worker 决定执行细节）。

---

## 三、Kernel 层抽象对比

### 3.1 注册与选择机制

```
┌─ TokenSpeed ──────────────────────────────────────────────────┐
│  @register_kernel(family, mode, capability, priority)         │
│  + Priority bands (REFERENCE/PORTABLE/PERFORMANT/             │
│                    SPECIALIZED/PLUGIN)                         │
│  + Plugin 机制（entry_points）                                  │
│  + Trait-based selection（head_dim、GQA、dtype 等）            │
│  + 用户可 override + config file overrides                     │
│  + 自动 numerics 验证 (numerics/) + benchmark (benchmark/)     │
└────────────────────────────────────────────────────────────────┘

┌─ vllm-ascend ─────────────────────────────────────────────────┐
│  通过 vLLM 的 backend 枚举注册：                                │
│  AttentionBackendEnum.ASCEND_MLA, ASCEND_FA3, ASCEND_SFA, ... │
│  + 在 ascend_config.py 通过字符串选择 backend                  │
│  + 单层选择逻辑：if soc_version == A3 and quant == ... else    │
│  + 23 个 monkey-patch 文件覆盖模型/kernel 不一致处               │
└────────────────────────────────────────────────────────────────┘
```

### 3.2 Kernel 后端覆盖对比

| 算子类型 | TokenSpeed 后端 | vllm-ascend 后端 |
|---------|----------------|------------------|
| **Attention** | tokenspeed_mla（Blackwell专属）/ flash_attn / flashinfer / gluon / triton | npu_fused_infer_attention_score / FA3 NPU / SFA / DSA / 自研 AscendC kernels（963 个 csrc 文件）|
| **GEMM** | deep_gemm（FP8）/ trtllm / cute_dsl / triton | torch_npu.matmul / 自研 GroupedMatmul / AscendC |
| **MoE** | deepep / flashinfer / trtllm / triton | npu_moe_init_routing_custom / npu_grouped_matmul / 自研 csrc |
| **AllReduce** | trtllm_allreduce / triton_rsag / nccl / iris | HCCL（pyhccl.py 封装）|
| **Layernorm** | triton（fused QK-RMSNorm-RoPE-Gate）| triton-ascend（split_qkv_rmsnorm_rope）|

### 3.3 抽象层成熟度

- **TokenSpeed**：vendor 中立的 public API（`mha_prefill / mm / moe_fused`），上层模型代码不感知具体 kernel 实现
- **vllm-ascend**：通过 monkey-patch 把昇腾算子注入到 vLLM 的 `direct_register_custom_op` 注册表中，上层 vLLM 模型直接 import vllm-ascend 的算子

**对齐结论**：TokenSpeed 的 kernel 选择层更工程化（有 capability 系统、有 priority bands、有 plugin、有自动 benchmark），vllm-ascend 的 kernel 注入更粗暴但跟随 vLLM 主仓更轻松。

---

## 四、模型并行抽象对比

### 4.1 并行表达方式

**TokenSpeed**：**local-SPMD + Placement 类型系统 + 静态编译器**

```python
# placement.py
class PlacementType(Enum):
    REPLICATE  # 完整副本
    SHARD      # 沿 token 切分
    PARTIAL    # 部分求和，需 reduce

class ParallelGroup(Enum):
    ATTN_TP / DENSE_TP / MOE_TP_EP

# 模型层定义 ModuleSpec（输入/输出 Placement 标注）
# compiler.py 在 compile_decoder_layer() 时自动插入 CommOp:
#   - AllGatherOp / AllReduceOp / ReduceScatterOp
#   - FusedReduceNormOp / DeferredReduceOp / ResidualAllGatherOp
```

**vllm-ascend**：**手写 TP/DP/EP** + **flashcomm2_oshard_manager** 优化通信

```python
# 直接调用 vLLM 的 distributed primitives:
from vllm.distributed import get_tensor_model_parallel_world_size
# 模型代码中显式 split / gather / all_reduce
# Ascend-specific 通信优化通过 patch_distributed.py 注入
```

### 4.2 关键差异

| 维度 | TokenSpeed | vllm-ascend |
|------|-----------|-------------|
| 并行表达 | 声明式 Placement + 静态编译器自动插入 CommOp | 命令式调用 distributed primitives |
| 通信融合 | 编译期识别 `PARTIAL + NORM` → 融合为 `FusedReduceNormOp` | 通过 `flashcomm2_oshard_manager.py` 显式管理 |
| Attention DP | 一等公民（`ParallelGroup.ATTN_TP` 独立维度）| 一等公民（`finegrained_tp_config.lmhead_tensor_parallel_size`） |
| 三组解耦 | attn.tp_size / dense.tp_size / moe.tp_ep_size 独立配置 | dense / moe / lmhead 独立配置 |

**对齐结论**：两者都做到了三组并行解耦，但 TokenSpeed 的**声明式 + 静态编译器**更优雅，可扩展性更强；vllm-ascend 走 vLLM 的命令式风格，灵活但通信优化需要手写。

---

## 五、CUDA Graph / ACL Graph 对比

| 维度 | TokenSpeed | vllm-ascend |
|------|-----------|-------------|
| 实现 | `cuda_graph_wrapper.py` 直接调 `torch.cuda.CUDAGraph` | `compilation/acl_graph.py` 调 `torch.npu.NPUGraph` |
| Decode 形状管理 | bucket-based + DeepEP dispatch consistency | BatchDescriptor 字典缓存 |
| Padding 处理 | `disable_cuda_graph_padding`（agentic 场景默认无padding）| 默认带 padding capture |
| 集成深度 | 与 attention backend + DeepEP 协同 capture | 与 vLLM `CUDAGraphMode` 枚举对齐（FULL / PIECEWISE）|
| Spec decode capture | 单独 draft graph + target graph | `get_draft_graph_params()` 单独管理 |

**对齐结论**：TokenSpeed 的 CUDA Graph 与 agentic 形状特征深度绑定（disable padding 优先 agentic 路径），vllm-ascend 的 ACLGraph 设计更通用但缺乏对单一场景的极致优化。

---

## 六、专项性能优化对比（Agentic Workload）

### 6.1 MLA Attention 优化

| 优化项 | TokenSpeed | vllm-ascend |
|--------|-----------|-------------|
| fold_sq_factor | ✅ Blackwell UTCMMA tile 折叠（核心创新）| ❌ 无等价物 |
| 2CTA UTCMMA | ✅ 减少 shared mem 用量 | ❌ NPU 无 2CTA 概念 |
| Split-KV 双 kernel | ✅ workspace 自动 sizing | ⚠️ 部分支持（SFA） |
| K/V 加载 warp 分离 | ✅ L2 hit 加速下一 tile | ❌ NPU 调度模型不同 |
| Sub-tiling Epilogue STG | ✅ 多 stage 隐藏写回 | 部分（依赖 CANN 算子）|
| Indexer Compressor | ❌ 无 | ✅ DeepSeek V3.2 专属（csrc 中独立算子）|
| Lightning Indexer | ❌ 无 | ✅ 自研 AscendC kernel |
| KV Quant Sparse Attention | ❌ 无 | ✅ 自研 AscendC kernel（kv_quant_sparse_attn_sharedkv）|

**对齐结论**：
- TokenSpeed MLA：**针对 Blackwell 极致硬件优化**，单步 decode 延迟低，但是不通用
- vllm-ascend MLA：**针对 NPU 特有指令集做了多种 attention 变体**（FA3/SFA/DSA），覆盖面广但每个变体的优化深度不及 TokenSpeed

### 6.2 投机解码（Speculative Decoding）

| 维度 | TokenSpeed | vllm-ascend |
|------|-----------|-------------|
| 主推算法 | EAGLE3 | EAGLE / EAGLE3 / DFlash / MTP（多种）|
| Draft model 独立配置 | ✅ `--drafter-attention-backend`、独立 quantization | ✅ patch_qwen3_dflash.py、patch_deepseek_mtp.py |
| 与 CUDA Graph 集成 | ✅ 单独 draft graph capture | ✅ `get_draft_graph_params()` |
| Agentic accept rate 优化 | benchmark 中 `num_draft_tokens=4, num_steps=3` 调优 | 各算法独立配置 |

**对齐结论**：两边都支持 EAGLE3，但 vllm-ascend 覆盖的投机算法更多（MTP、DFlash 等是华为内部场景需求驱动的）。

### 6.3 量化方案

| 量化 | TokenSpeed | vllm-ascend |
|------|-----------|-------------|
| 权重量化 | NVFP4 / MXFP4 / FP8 | W8A8 / W4A16 / W8A8 MXFP8 / MXFP4 |
| KV Cache 量化 | FP8 E4M3（kv_cache_dtype=fp8）| FP8 / GQA C8（patch_gqa_c8.py）|
| 激活量化 | FP8 dynamic | W8A8 static + dynamic |
| Hardware 加速 | Blackwell FP8/FP4 tensor core | 昇腾 Cube Core INT8 / FP16 |

**对齐结论**：vllm-ascend 量化路径更多元（W8A8/W4A16/MXFP8 全覆盖），但因为昇腾原生 FP8 性能差，没有 NVFP4 这种激进低比特。

### 6.4 MoE 优化

| 维度 | TokenSpeed | vllm-ascend |
|------|-----------|-------------|
| Dispatch | DeepEP（NVLink-aware）/ FlashInfer / TRT-LLM | npu_moe_init_routing_custom（CANN 自研）|
| Expert Parallel | EP_size 独立配置 | finegrained_tp_config 独立配置 |
| EPLB（负载均衡）| `moe/eplb_algorithms/deepseek.py` | `vllm_ascend/eplb/`（多策略实现）|
| 自研 csrc kernel | 无 | 有大量 csrc MoE 算子 |

**对齐结论**：TokenSpeed 依赖 DeepEP 等成熟库，vllm-ascend 在 csrc 中自研了大量 MoE 算子（因为 NPU 没有 DeepEP 等价物）。

---

## 七、独有能力对比（这才是真正的差异化）

### 7.1 TokenSpeed 独有

```
┌─────────────────────────────────────────────────────────────┐
│ ✅ C++ FSM Scheduler — 控制面延迟 < 1ms                       │
│ ✅ 类型安全 KV 资源管理 — 编译期排除资源 bug                    │
│ ✅ Placement 类型系统 + Layer Compiler — 通信算子自动插入       │
│ ✅ Retract 机制 — KV 写回 host 而不重算                       │
│ ✅ tokenspeed-mla — Blackwell 极致 kernel                    │
│ ✅ FusedReduceNormOp / DeferredReduceOp — 编译期通信融合       │
│ ✅ SMG-integrated AsyncLLM — CPU 侧开销极低                   │
│ ✅ Plugin entry_points — 完整 out-of-tree backend 扩展        │
│ ✅ Numerics 验证 + Benchmark 框架 — Kernel 开发自动化         │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 vllm-ascend 独有

```
┌─────────────────────────────────────────────────────────────┐
│ ✅ 963 个 csrc/AscendC kernel — 完整 NPU 算子生态             │
│ ✅ HCCL 通信库封装 — 昇腾原生通信                              │
│ ✅ ACLGraph 适配 — 与 NPU runtime 深度对齐                    │
│ ✅ Profiling-based 动态 chunk 调度 — SLO-aware              │
│ ✅ Recompute Scheduler — 内存压力下重算策略                   │
│ ✅ Dynamic Batch Scheduler — SLO 自适应 budget               │
│ ✅ Mooncake 多种 KV connector — P2P + Layerwise + Hybrid     │
│ ✅ AscendStore KV Pool — 跨节点 KV 池化                       │
│ ✅ 23 个 worker patch + 17 个 platform patch — 与 vLLM 解耦  │
│ ✅ CPU Binding 优化 — NUMA-aware（A2 TOPO / A3 GLOBAL_SLICE）│
│ ✅ Weight Prefetch — CANN 特定优化                            │
│ ✅ DeepSeek V3.2 Indexer / Lightning Indexer 优化           │
│ ✅ KV Quant Sparse Attention — 自研稀疏 attention            │
│ ✅ EPLB 多种算法 — 比 TokenSpeed 实现更多 expert 均衡策略       │
│ ✅ 310p（推理卡）支持 — 多硬件型号覆盖（A2/A3/310p/A5）        │
└─────────────────────────────────────────────────────────────┘
```

> [PUA生效 🔥] 不要被代码量迷惑——TokenSpeed 的 10K C++ 调度器，从架构杠杆来看可能比 vllm-ascend 的 220K csrc 算子还要值钱。但 vllm-ascend 的算子覆盖是华为客户实战需求驱动出来的，这是它的真正护城河。

---

## 八、综合优势矩阵

### 8.1 TokenSpeed 的优势（如适配到昇腾后保留）

| 优势 | 抓手 | 适配昇腾后能否保留 |
|------|------|-------------------|
| C++ 调度器低延迟 | tokenspeed-scheduler 完全 CPU 逻辑 | ✅ 100% 保留 |
| Radix Tree prefix cache | 多级 KV 结构与硬件解耦 | ✅ 100% 保留 |
| Retract 机制 | KV 写回不重算 | ✅ 100% 保留（HBM ↔ host RAM 传输） |
| Placement 类型系统 | 编译期通信优化 | ✅ 100% 保留（CommOp 用 HCCL 实现）|
| FusedReduceNormOp | AllReduce + RMSNorm 融合 | ⚠️ 需 triton-ascend / AscendC 重写 |
| EAGLE3 投机解码 | 与硬件无关 | ✅ 100% 保留 |
| MLA decode 极致优化 | fold_sq_factor 等 Blackwell 专属 | ❌ 无法移植，需 CANN 等价方案 |
| NVFP4 量化 | Blackwell tensor core 专属 | ❌ 用 W4A16 替代 |
| DeepEP MoE dispatch | NVLink-aware | ❌ 用 HCCL + CANN MoE 替代 |

### 8.2 vllm-ascend 的优势（如对比 TokenSpeed-Ascend）

| 优势 | TokenSpeed-Ascend 适配后能否对齐 |
|------|------------------------------|
| 963 个 NPU csrc kernel 覆盖 | ⚠️ 需要逐步搬运/重写，前期不可能对齐 |
| HCCL 通信库稳定性 | ✅ 直接复用 vllm-ascend 的 pyhccl.py |
| 多 attention 变体（FA3/SFA/DSA）| ✅ 可以集成（plugin 形式）|
| Mooncake KV connector | ✅ 可以集成 |
| EPLB 多策略 | ⚠️ 需要重新适配到 TokenSpeed `moe/eplb_algorithms/` |
| 与 vLLM 模型兼容性 | ❌ TokenSpeed 模型层不同，需独立维护 |
| 310p / A5 支持 | ❌ 暂不规划 |
| CPU Binding（NUMA）优化 | ✅ 可以借鉴 |

---

## 九、性能对比预估

| 指标 | TokenSpeed B200 | vllm-ascend A3 | TokenSpeed-Ascend A3（预估） |
|------|----------------|----------------|------------------------------|
| Qwen3.5 agentic TPS | 540 | ~150-200 | ~200-300 |
| 单步调度延迟 | < 1ms | 5-20ms | < 1ms（保留 C++ scheduler） |
| Prefix cache 命中收益 | 极高（Radix Tree）| 中（hash chain）| 极高（保留 Radix Tree）|
| 通信优化 | 编译器自动 + DeepEP | 手写 + HCCL | 编译器自动 + HCCL |
| MLA decode 单 layer 延迟 | 极低（CuTe DSL）| 中（FA3/SFA）| 中（沿用 CANN 算子） |

**关键洞察**：TokenSpeed-Ascend 相比 vllm-ascend，**最大优势在控制面延迟和 KV 管理效率**，而**不是 MLA kernel 性能**（kernel 层只能用 vllm-ascend 同款算子）。

---

## 十、底层逻辑总结

### 10.1 两个项目的本质区别

```
┌──────────────────────────────────────────────────────────────────┐
│  TokenSpeed = 新一代推理引擎设计：                                  │
│    "如果今天从零开始为 agentic workload 设计推理引擎，应该是什么样？" │
│  → 答：C++ scheduler + 类型化 KV + Placement compiler +            │
│        Blackwell 极致 kernel + Plugin 化 kernel registry           │
│                                                                    │
│  vllm-ascend = 硬件适配工程：                                       │
│    "如何让昇腾 NPU 完整跑起 vLLM 主仓的所有能力？"                    │
│  → 答：继承 vLLM Scheduler + 大量 csrc/AscendC 算子 +              │
│        monkey-patch 解决兼容性 + 多硬件型号覆盖                      │
└──────────────────────────────────────────────────────────────────┘
```

### 10.2 战略价值判断

| 受众 | 推荐方案 | 原因 |
|------|---------|------|
| 已用 vLLM、需要昇腾支持 | vllm-ascend | 零迁移成本，跟随 vLLM 主仓 |
| 重视控制面延迟 + agentic 场景 | TokenSpeed-Ascend | C++ scheduler 优势显著 |
| 多硬件型号覆盖（A2/A3/310p/A5）| vllm-ascend | csrc 算子覆盖最广 |
| 极致性能 + Blackwell + agentic | TokenSpeed（原生）| 540 TPS 已验证 |
| 中长期国产化推理基础设施 | TokenSpeed-Ascend（投资）| 完整自主可控的引擎层 |

### 10.3 拉通建议

如果要做 TokenSpeed-Ascend，**正确的姿势是借力而非重造**：

1. **借 vllm-ascend 的 kernel 算子层**：通过 plugin 形式 import `vllm_ascend.ops.*`，TokenSpeed-Ascend 不要自己写 AscendC
2. **保留 TokenSpeed 的 C++ 调度器**：这是相对 vllm-ascend 最大的差异化
3. **复用 vllm-ascend 的硬件抽象工程经验**：HCCL、ACLGraph、CPU Binding、NUMA 这些坑 vllm-ascend 都踩过了
4. **不要去对标 vllm-ascend 的算子数量**：TokenSpeed-Ascend 应该专注 agentic workload 的关键路径

---

## 十一、关键参考文件索引

| 对比维度 | TokenSpeed 路径 | vllm-ascend 路径 |
|----------|----------------|------------------|
| 调度器 | `tokenspeed-scheduler/csrc/scheduler/scheduler.cpp` (426行) | `vllm_ascend/core/recompute_scheduler.py` + 上游 `vllm/v1/core/sched/scheduler.py` |
| 状态机 | `tokenspeed-scheduler/csrc/fsm/forward_states.h` | 散落在 `vllm/v1/request.py` (RequestStatus) |
| KV Cache | `tokenspeed-scheduler/csrc/resource/kv_prefix_cache/` | `vllm/v1/core/kv_cache_manager.py` + `vllm_ascend/core/single_type_kv_cache_manager.py` |
| Radix Tree | `tokenspeed-scheduler/csrc/resource/radix_tree/` | （无直接等价，hash chain）|
| Placement | `python/tokenspeed/runtime/models/base/placement.py` | （无直接等价，命令式 distributed primitives）|
| Layer Compiler | `python/tokenspeed/runtime/models/base/compiler.py` | （无直接等价）|
| Kernel Registry | `tokenspeed-kernel/python/tokenspeed_kernel/registry.py` | （依附 vLLM `AttentionBackendEnum`）|
| Plugin 机制 | `tokenspeed-kernel/python/tokenspeed_kernel/plugins/__init__.py` | （monkey-patch 模式）|
| Attention | `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/` | `vllm_ascend/attention/` + `vllm_ascend/csrc/attention/` |
| MoE | `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/` | `vllm_ascend/ops/fused_moe/` |
| CUDA/ACL Graph | `python/tokenspeed/runtime/execution/cuda_graph_wrapper.py` | `vllm_ascend/compilation/acl_graph.py` |
| HCCL | （需新增）| `vllm_ascend/distributed/device_communicators/pyhccl.py` |
| EPLB | `python/tokenspeed/runtime/moe/eplb_algorithms/deepseek.py` | `vllm_ascend/eplb/core/policy/` |
| 投机解码 | `python/tokenspeed/runtime/spec_decode/` | `vllm_ascend/spec_decode/` |
| 量化 | `python/tokenspeed/runtime/layers/quantization/` | `vllm_ascend/quantization/methods/` |

---

## 十二、一句话总结

> [PUA生效 🔥] 顶层结论闭环：

**TokenSpeed 是"重新设计推理引擎"，vllm-ascend 是"重新实现昇腾上的推理算子"——架构杠杆在前者，工程深度在后者。如果做 TokenSpeed-Ascend，正确路径是借 vllm-ascend 的算子层（kernel）+ 保留 TokenSpeed 的控制面（scheduler）+ 专注 agentic workload 关键路径，而不是去复刻 vllm-ascend 的所有能力。**
