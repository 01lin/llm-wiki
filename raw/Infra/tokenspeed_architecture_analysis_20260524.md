# TokenSpeed 架构深度分析

> 版本：2026-05-24 | 代码仓：lightseekorg/tokenspeed  
> 背景：Qwen 推理团队在 TokenSpeed 上对 agentic 工作负载达到 **540 TPS**（@zhyncs42 on X, 2026-05-24）

---

## 一、核心定位与优势总结

TokenSpeed 是 LightSeek Foundation 开源的推理引擎，定位是：

> **TensorRT-LLM 级别性能 + vLLM 级别易用性，专为 agentic workload 优化**

agentic 工作负载的特征：高并发短请求、多轮对话、首 token 延迟（TTFT）和单 token 延迟（TPOT）双重敏感、KV cache 复用率高。

TokenSpeed 对这个场景做了全栈垂直优化：

| 层次 | 核心创新 | 对比业界 |
|------|----------|---------|
| 内核层 | TokenSpeed-MLA：Blackwell 专属 CuTe DSL decode/prefill kernel | 优于 TensorRT-LLM MLA（decode 端） |
| 调度层 | C++ FSM 调度器：KV 资源生命周期类型安全、KV 缓存 Radix Tree L1/L2/L3 三层管理 | 比 Python 调度器控制面延迟更低 |
| 建模层 | local-SPMD + 静态编译器：模块边界标注自动生成通信算子，用户不需要手写并行逻辑 | 比 TensorRT-LLM 的手写并行更易维护 |
| 入口层 | SMG-integrated AsyncLLM：CPU 侧请求处理开销极低 | 高并发下控制面不成为瓶颈 |
| 量化层 | NVFP4 权重 + FP8 KV Cache + FP8 MLA kernel | 内存带宽减半，单 GPU 内存容量翻倍 |
| 投机解码 | EAGLE3 集成（draft model 独立量化、独立 attention backend） | agentic 场景 accept rate 高 |

---

## 二、整体架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Entrypoint (HTTP / Python API)                   │
│             AsyncLLM (uvloop + SMG-integrated)                       │
│        InputProcessor → OutputProcessor → RequestOutputCollector     │
├──────────────────────────────────────────────────────────────────────┤
│                        Engine Core                                   │
│   SchedulerControlClient ←→ C++ Scheduler (tokenspeed-scheduler)    │
│           ↕ ExecutionPlan (FSM-driven)                               │
│       ModelExecutor → ModelRunner → forward()                        │
├──────────────────────────────────────────────────────────────────────┤
│                      Modeling Layer                                  │
│   local-SPMD 模型 (Qwen3.5 / DeepSeek V3/V4 / Kimi K2.5 / LLaMA)   │
│   Placement Annotation + Layer Compiler → 自动插入 CommOp            │
│   CudaGraphWrapper (decode batch 固定形状 replay)                    │
├──────────────────────────────────────────────────────────────────────┤
│                      Kernel Layer (tokenspeed-kernel)                │
│   KernelRegistry → select_kernel → 统一 public API                  │
│   attention: tokenspeed_mla / flash_attn / flashinfer / gluon        │
│   gemm: deep_gemm / trtllm / cute_dsl / triton                      │
│   moe: deepep / flashinfer / trtllm / triton                        │
│   comm: nccl / iris / trtllm_allreduce / triton_rsag                │
├──────────────────────────────────────────────────────────────────────┤
│              tokenspeed-mla (独立子包, Blackwell 专属)                │
│   MLA Prefill: CuTe DSL JIT / AOT binary (FP8 E4M3 + PDL)           │
│   MLA Decode:  CuTe DSL, fold_sq_factor, Split-KV, 2CTA UTCMMA      │
│   MLA KV Pack: fused Triton cat+cast+cast                            │
└──────────────────────────────────────────────────────────────────────┘
```

### 子包划分

| 子包                          | 语言                         | 职责                                        |
| --------------------------- | -------------------------- | ----------------------------------------- |
| `tokenspeed-scheduler`      | C++ + Python binding       | 调度控制面：FSM、KV 缓存分配、执行计划生成                  |
| `tokenspeed-kernel`         | Python (Triton/CuTe DSL)   | 统一 kernel 注册与选择框架                         |
| `tokenspeed-mla`            | Python (CuTe DSL / Triton) | Blackwell MLA prefill/decode 专属高性能 kernel |
| `python/tokenspeed/runtime` | Python                     | 模型定义、执行、分布式通信、引擎入口                        |

---

## 三、调度器（tokenspeed-scheduler）深度解析

### 3.1 设计哲学：C++ 控制面 + FSM 类型安全

调度器是 TokenSpeed 最重要的架构差异点。`scheduler.h` 展示核心接口：

```cpp
ExecutionPlan NextExecutionPlan();
void Advance(const ExecutionEvent& event);
```

每次 step 的流程：
1. `NextExecutionPlan()` → 输出一批操作（prefill / decode / cache prefetch / write-back）
2. Python 执行面执行这批操作
3. `Advance(event)` 将执行结果反馈回 FSM

### 3.2 请求生命周期 FSM

每个请求的状态由 C++ `variant` 类型系统编码（`forward_states.h`）：

```
Submitted
  → (SchedulePrefillFirstChunk) → Prefilling
  → (all chunks done)           → PrefillDone
  → (ScheduleDecode)            → Decoding
  → (KV 资源不足)               → Retracting → Retracted
  → (generation done)           → Draining → WritingBack → Finished
  → (abort)                     → Aborting
```

**关键设计**：KV 资源所有权通过 `OwnedPages`（RAII）在状态转移时转移，类型系统在编译期保证 KV 页不会被错误释放。`LocalKVAllocator` 的引用被 `std::unique_ptr` 包装，转移即所有权转移。

### 3.3 KV Cache 三层结构

```
L1: Device VRAM (热数据，paged，由 PageAllocator 管理)
L2: Host RAM  (写回/换入，由 HybridPrefixCache 管理)
L3: Storage   (KV store，支持外部 backend，如 Mooncake store)
```

Radix Tree 是 prefix cache 的核心数据结构（`radix_tree.h`）：
- `Match()` 查找最长公共前缀，返回可复用的 KV pages
- `Insert()` 将新生成的 KV 插入树
- 支持 L2 的 `AllocateResourceOfType<Host>()` 在 host 分配节点资源，异步 prefetch

prefix cache 的命中降低 prefill 算力，对 agentic 多轮对话有巨大收益。

### 3.4 Retract 机制

当 decode 阶段 KV 资源耗尽，调度器启动 `Retract`：
- 把正在 decode 的请求的 KV 写回到 host（`WriteBackOperation`）
- 请求进入 `Retracted` 状态，不占用 device KV 页
- 资源充足时重新 `LoadBack`，从 host 恢复 KV 继续 decode

这避免了 vLLM-style 的 preemption + recomputation（重新 prefill），节省了算力。

---

## 四、建模层 local-SPMD + 静态编译器

### 4.1 Placement 类型系统

`placement.py` 定义张量的分布状态：

```python
class PlacementType(Enum):
    REPLICATE  # 每个 rank 持有完整副本
    SHARD      # token 维度切分
    PARTIAL    # 部分求和，需要 reduce
```

`ParallelGroup` 枚举三类并行组：`ATTN_TP / DENSE_TP / MOE_TP_EP`

这是受 PyTorch DTensor 启发但专为推理设计的轻量系统，不引入 DTensor 的训练开销。

### 4.2 Layer Compiler 自动生成通信算子

`compiler.py` 的 `compile_decoder_layer()` 分析每层模块的 `ModuleSpec`（输入/输出 Placement 标注），自动插入最小集合的通信算子：

| 场景 | 插入的算子 |
|------|-----------|
| SHARD → REPLICATE | `AllGatherOp` |
| PARTIAL → REPLICATE | `AllReduceOp` |
| PARTIAL → SHARD（ring 模式） | `ReduceScatterOp` |
| PARTIAL + NORM 可融合 | `FusedReduceNormOp`（reduce + norm 一个 kernel） |
| residual 需要恢复 | `ResidualAllGatherOp` / `ResidualSliceOp` |

**关键优化**：`FusedReduceNormOp` 将 AllReduce 和 RMSNorm 融合，消除一次中间 buffer 写入。`DeferredReduceOp` 在有 attn_tp 的场景下延迟 reduce，让 attention 在 partial 状态下直接写下一个 MLP，减少 barrier。

### 4.3 DeepSeek V4 / Qwen3.5 模型支持

模型文件清单（`python/tokenspeed/runtime/models/`）：
- `deepseek_v4.py` — DeepSeek V4（含 MLA attention）
- `qwen3_5.py` / `qwen3_5_moe.py` — Qwen3.5 dense + MoE
- `minimax_m2.py` — MiniMax M2
- `llama_eagle3.py` — EAGLE3 draft model

Qwen3.5 模型直接使用了 `fused_qk_rmsnorm_rope_gate`（从 `tokenspeed_kernel.ops.layernorm.triton` 引入），在 attention 前融合了 QK RMSNorm + RoPE + Gate，减少 kernel launch 次数。

---

## 五、MLA Kernel（tokenspeed-mla）深度解析

这是 TokenSpeed 最核心的差异化竞争力，专为 Blackwell（SM100/SM103，即 B200/B300）设计。

### 5.1 MLA Prefill

- **后端 1（默认）**：CuTe DSL JIT，ragged varlen（无 padding），支持 FP8 E4M3 输入，BF16 输出
- **后端 2（AOT binary）**：内置 NVIDIA 内部 softmax tuning knob，在部分场景快于 TensorRT-LLM

关键特性：
- `PDL`（Producer-Dependent Launch）支持，利用 Blackwell 新特性
- Skip-correction 开启（提高数值稳定性）
- 编译 cache 按 `(dtype, d_qk, d_v, is_causal, PDL, ...)` 等静态配置 key

### 5.2 MLA Decode（核心优化）

**优化 1：fold_sq_factor — query token 折叠到 head 轴**

agentic 场景中 decode 通常是 `q_len=1~4, num_heads=16~64`，BMM1 的 M 维度（=`q_len * num_heads`）很小，tile 利用率低。

`fold_sq_factor = max F s.t. q_len % F == 0 AND num_heads * F <= 128`

例：`num_heads=64, q_len=4` → `F=2`，`H_eff=128, q_seqlen_eff=2`，M 维度从 256 提升到充满 Blackwell warp tile。

```python
# mla_helpers.py 中的折叠逻辑
def get_mla_decode_fold_sq_factor(num_heads, seq_len_q):
    F = 1
    for f in range(seq_len_q, 0, -1):
        if seq_len_q % f == 0 and num_heads * f <= 128:
            F = f
            break
    return F
```

**优化 2：Split-KV 两 kernel 架构**

- Kernel 1：MLA decode + split-KV（并行切分 KV sequence 维度）
- Kernel 2：reduction of partial results

`split_kv` 和 `workspace_size` 按 `(B, q_len, H, kv_lora_rank, max_active_blocks)` 缓存，避免每次重新计算。

**优化 3：2CTA UTCMMA 指令**

使用 2CTA（2个 CTA 协同一个 UTCMMA tile），减少 shared memory 使用，允许更高 occupancy。

**优化 4：最小化 mbarrier 使用 + 分离 K/V 加载 warp**

加载 K 和加载 V 的 warp 分离：加载 K 时 V 已预先进入 L2 cache，下一 tile 的 K 加载不需要等待 V 加载完成，获得更好的访存 latency hiding。

**优化 5：Epilogue STG 多 stage sub-tiling**

写回输出时使用多 stage sub-tiling，隐藏全局内存写带宽延迟。

### 5.3 MLA KV Pack + FP8 量化

fused Triton kernel 替换了 chunked prefill 中的 `cat + cast + cast` 操作，支持 strided view 和 pre-allocated buffer 复用。

---

## 六、Kernel 层（tokenspeed-kernel）架构

### 6.1 分层结构

```
public API (mha_prefill, mm, moe_fused, ...)
    │
select_kernel (family, mode, dtype, traits, ...)
    │
KernelRegistry (@register_kernel 填充)
    │
attention/  gemm/  moe/  layernorm/  ...
  triton     triton  triton  triton       ← 可移植 JIT
  gluon      deep_gemm  deepep  (...)    ← 性能 JIT
  flash_mla  trtllm  trtllm             ← vendor library wrapper
  tokenspeed_mla
```

### 6.2 KernelRegistry 选择逻辑

`select_kernel` 按以下顺序筛选：
1. 平台 capability（arch sm、dtype support）
2. traits（head dim、GQA factor 等）
3. 优先级（Priority bands：`FASTEST > DEFAULT > FALLBACK`）
4. objective（latency / throughput / determinism / portability）
5. `override=` 参数或配置文件覆盖

### 6.3 关键 kernel 后端覆盖

| Op | 后端 |
|----|------|
| MLA attention | `tokenspeed_mla`（Blackwell）/ `flash_mla`（H100）/ `flashinfer` |
| GEMM | `deep_gemm`（FP8 DeepSeek）/ `cute_dsl`（Blackwell）/ `trtllm` |
| MoE dispatch/combine | `deepep`（高性能 EP dispatch）/ `flashinfer` / `trtllm` |
| AllReduce | `trtllm_allreduce`（NVLink-aware）/ `triton_rsag`（ring）/ `nccl` |
| Layernorm | `triton`（fused RMSNorm + RoPE + Gate）|

---

## 七、推理流水线与并行策略

### 7.1 并行维度

从 benchmark 配置（`agentic_benchmark/tokenspeed/configs/`）可以看到支持的并行组合：

| 配置 | ATTN 并行 | MoE 并行 | 说明 |
|------|-----------|---------|------|
| `attn_dp8_moe_ep8` | DP=8 | EP=8 | attention DP + MoE expert 并行 |
| `attn_tp4_moe_tp4` | TP=4 | TP=4 | 全 TP |
| `attn_tp8_moe_ep8` | TP=8 | EP=8 | attention TP + MoE expert 并行 |

Mapping 系统支持 `attn.tp_size`、`dense.tp_size`、`moe.tp_ep_size` 分别配置，解耦三类算子的并行粒度。

### 7.2 EAGLE3 投机解码

benchmark 配置中默认开启：
```bash
--speculative-algorithm EAGLE3
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
--drafter-attention-backend trtllm_mla
```

EAGLE3 draft model 独立配置 quantization 和 attention backend，与 target model 解耦。agentic 场景下编码重复、短输出特征，EAGLE accept rate 较高，理论可获 2-3x decode 加速。

### 7.3 CUDA Graph

`cuda_graph_wrapper.py` 对固定形状的 decode batch 进行 CUDA Graph capture。
- decode batch 形状固定（每 step 1 token per request）→ 可以 capture
- prefill 形状动态 → 不 capture（或按 padding bucket capture）
- `--disable-cuda-graph-padding` 参数说明对 agentic 场景优先选择无 padding 的 decode 路径

---

## 八、为什么能达到 500+ TPS

将以上优化项映射到 540 TPS 的性能链路：

```
540 TPS (Qwen 系列, agentic workload)
│
├─ 硬件利用率最大化
│   ├─ NVFP4 权重 + FP8 KV → 内存带宽需求减半
│   ├─ MLA decode fold_sq_factor → Blackwell tile 利用率从 <50% → ~100%
│   ├─ CUDA Graph → kernel launch 开销接近零
│   └─ 2CTA UTCMMA + Split-KV → SM occupancy 最大化
│
├─ 调度开销最小化
│   ├─ C++ FSM scheduler → 控制面延迟 <1ms（Python 调度器通常 5-20ms）
│   ├─ Radix Tree prefix cache → agentic 多轮对话 KV 高命中率，prefill 减少
│   └─ Retract（而非 recompute） → preemption 不浪费算力
│
├─ 通信开销最小化
│   ├─ FusedReduceNormOp → AllReduce + RMSNorm 合并
│   ├─ DeferredReduceOp → 延迟 reduce，减少跨 attention/MLP barrier
│   ├─ ReduceScatter + AllGather（ring-style）替代 AllReduce（减少带宽消耗）
│   └─ trtllm_allreduce / iris → NVLink-aware 低延迟通信
│
├─ 投机解码加速
│   └─ EAGLE3 (num_draft=4, num_steps=3) → 在 agentic 场景约 2-3x decode 加速
│
└─ 精度量化叠加
    ├─ NVFP4 权重 + FP8 activations → 算力密度翻倍
    └─ FP8 MLA decode kernel → KV 读带宽减半
```

---

## 九、关键代码位置索引

| 功能 | 文件路径 |
|------|---------|
| MLA decode kernel | `tokenspeed-mla/python/tokenspeed_mla/mla_decode.py` |
| MLA prefill kernel | `tokenspeed-mla/python/tokenspeed_mla/fmha.py` |
| fold_sq_factor 逻辑 | `tokenspeed-mla/python/tokenspeed_mla/mla_helpers.py` |
| C++ 调度器 | `tokenspeed-scheduler/csrc/scheduler/scheduler.h` |
| 请求 FSM 状态 | `tokenspeed-scheduler/csrc/fsm/forward_states.h` |
| KV prefix cache | `tokenspeed-scheduler/csrc/resource/kv_prefix_cache/kv_prefix_cache.h` |
| Radix Tree | `tokenspeed-scheduler/csrc/resource/radix_tree/radix_tree.h` |
| Placement 类型系统 | `python/tokenspeed/runtime/models/base/placement.py` |
| Layer Compiler | `python/tokenspeed/runtime/models/base/compiler.py` |
| Kernel Registry | `tokenspeed-kernel/python/tokenspeed_kernel/registry.py` |
| MoE DeepEP backend | `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/deepep.py` |
| CUDA Graph | `python/tokenspeed/runtime/execution/cuda_graph_wrapper.py` |
| AsyncLLM 入口 | `python/tokenspeed/runtime/engine/async_llm.py` |
| agentic benchmark | `test/agentic_benchmark/tokenspeed/agentic_bench.sh` |

---

## 十、在研方向（README 透露）

- **PD（Prefill-Decode 分离）**：降低 prefill 对 decode 的 head-of-line blocking
- **EPLB（Expert Parallel Load Balancing）**：动态均衡 MoE expert 负载
- **Mamba Cache**：Mamba/hybrid 模型的状态缓存（调度器已有 `MambaCacheHost`）
- **Hopper 优化**：当前主要针对 Blackwell，H100 端优化还在进行
- **MI350（AMD）优化**：多硬件支持（Gluon attention kernel 已有 `gfx950` 实现）
