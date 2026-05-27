# TokenSpeed 架构深度分析

> 分析时间：2026-05-25  
> 代码路径：`/Users/linyi/code/Documents/code/tokenspeed/`  
> 版本状态：Preview Release（LightSeek Foundation，面向 B200/Blackwell 的 agentic workload 优化）

---

## 一、代码目录结构

```
tokenspeed/
├── python/tokenspeed/          # Python 运行时主包（轻量 CLI + 环境）
│   ├── __init__.py             # 入口：仅抑制三方日志噪声
│   ├── bench.py                # benchmark 工具
│   └── env.py                  # 环境变量与配置
│
├── tokenspeed-kernel/          # 算子库（独立包 tokenspeed_kernel）
│   └── python/tokenspeed_kernel/
│       ├── registry.py         # KernelRegistry：集中注册表（单例）
│       ├── selection.py        # 算子选择逻辑（优先级打分 + 缓存）
│       ├── platform.py         # 硬件能力检测（arch、vendor、features）
│       ├── thirdparty/         # 第三方算子封装（CUDA/TRT-LLM/FlashInfer/CuTe DSL）
│       └── ops/                # 算子族（attention/gemm/moe/kvcache/…）
│           ├── attention/
│           ├── gemm/           # deep_gemm、trtllm、triton、cute_dsl
│           ├── moe/            # deepep、triton、trtllm、flashinfer
│           ├── sampling/
│           └── …
│
├── tokenspeed-scheduler/       # C++ 控制面调度器（独立包 tokenspeed_scheduler）
│   ├── csrc/
│   │   ├── scheduler/          # 调度核心
│   │   │   ├── scheduler.h/.cpp    # Scheduler 主类
│   │   │   ├── execution_plan.h    # ExecutionPlan（操作容器）
│   │   │   ├── execution_event.h   # ExecutionEvent（反馈容器）
│   │   │   ├── request.h/.cpp      # Request（FSM 持有者）
│   │   │   └── operations/         # Forward/Cache 操作结构体
│   │   ├── fsm/                # 有限状态机（请求生命周期）
│   │   │   ├── forward_states.h    # 所有 forward 状态
│   │   │   ├── forward_events.h    # forward 事件（Schedule*/Abort/Finish）
│   │   │   ├── cache_states.h      # cache 状态
│   │   │   └── cache_events.h      # cache 事件
│   │   └── resource/           # 资源管理
│   │       ├── allocator/          # PageAllocator、KVAllocator、MambaAllocator
│   │       ├── kv_prefix_cache/    # Radix Tree 前缀缓存
│   │       ├── hybrid_prefix_cache/    # Hybrid：KV + Mamba + PagedGroup
│   │       └── radix_tree/         # RadixTree + TreeNode
│   └── bindings/python_module.cpp  # nanobind Python 绑定
│
├── tokenspeed-mla/             # MLA 专项算子（Blackwell 优化）
│   └── python/tokenspeed_mla/
│       ├── fmha.py             # MLA prefill（CuTe DSL JIT / AOT binary）
│       └── mla_decode*.py      # MLA decode（fold_sq_factor 优化）
│
├── test/                       # 集成测试 + CI 流水线
│   ├── cli/                    # CLI / serve / SMG 集成测试
│   └── runtime/                # attention、prefix cache、prefix_cache_e2e 等
└── docs/                       # VitePress 文档
```

**关键依赖关系**：

```
tokenspeed (Python runtime)
    └── tokenspeed-scheduler  (C++ via nanobind)
    └── tokenspeed-kernel     (Python kernel 抽象层)
            └── tokenspeed-mla   (Blackwell MLA 专项算子)
            └── thirdparty:  TRT-LLM / FlashInfer / DeepGEMM / CuTe DSL / DeepEP
```

---

## 二、核心架构设计

### 2.1 四层架构总览

```
┌─────────────────────────────────────────────────────────┐
│ Layer 4 - Entrypoint (Python)                           │
│  AsyncLLM + SMG（SMG = 网关进程管理器）                  │
│  tokenspeed serve → tokenspeed.cli → smg launch        │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP/gRPC
┌────────────────────────▼────────────────────────────────┐
│ Layer 3 - Scheduler (C++ Control Plane)                 │
│  Scheduler::NextExecutionPlan() → ExecutionPlan         │
│  请求生命周期：Submitted → Prefilling → PrefillDone      │
│               → Decoding → Draining → Finished          │
│  资源：RadixTree KVCache + PageAllocator (Dev+Host)     │
│        HybridPrefixCache（KV + Mamba + PagedGroup）     │
└────────────────────────┬────────────────────────────────┘
                         │ Python bindings (nanobind)
┌────────────────────────▼────────────────────────────────┐
│ Layer 2 - Modeling Layer (local-SPMD)                   │
│  静态编译器从模块边界标注自动生成集合通信（不需手写并行）  │
│  执行 FlatForwardOperation（Prefill + Decode 混合批）    │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│ Layer 1 - Kernels (tokenspeed-kernel)                   │
│  KernelRegistry → select_kernel() → SelectedKernel      │
│  支持：attention/gemm/moe/kvcache/sampling/comm/...      │
│  backends: CUDA/TRT-LLM/FlashInfer/DeepGEMM/Triton      │
└─────────────────────────────────────────────────────────┘
```

### 2.2 调度器 FSM（有限状态机）

请求在 C++ 侧的完整状态机：

```
Submitted
  │ SchedulePrefillFirstChunkEvent  →  Prefilling (分 chunk prefill)
  │ SchedulePrefillEvent            →  Prefilling (整段 prefill)
  │
Prefilling
  │ ExtendResult (GPU 返回 token)   →  PrefillDone / Decoding
  │
PrefillDone
  │ ScheduleDecodeEvent             →  Decoding
  │
Decoding
  │ ExtendResult                    →  Decoding（继续 decode）
  │ Finish                          →  Draining（写回 L2 host）
  │ Abort                           →  Aborting
  │ ScheduleRetractEvent            →  Retracting（内存压力回退）
  │
Retracting
  │ WriteBackDone                   →  Retracted（host 已持有 KV）
  │
Retracted
  │ ScheduleDecodeFromRetractedEvent→  Decoding（从 host 加载回来）
  │
Draining
  │ CommitDrainingEvent             →  WritingBack（异步写回 host）
  │ AbortEvent（无页可写）          →  ～
  │
WritingBack
  │ WriteBackDone                   →  Finished
  │
Finished → 从 requests_ map 删除
```

**Retract（回退）** 是 TokenSpeed 的独特设计：当 GPU 显存不足时，将正在 Decode 的请求 KV cache 异步写回到 host 内存，释放 GPU 显存，后续再 LoadBack 回来继续服务，避免 OOM 导致请求丢失。这是 agentic 长上下文场景的关键保障。

### 2.3 Kernel 选择系统

算子选择是 **三维词典序打分**，高者优先：

```
Score = (oracle_score, objective_score, priority_score)
  - oracle_score:    per-family domain oracle 调整 [0,20)，默认 10
  - objective_score: 1 if kernel tags 匹配请求 objective，否则 0
  - priority_score:  kernel 声明的优先级 [0,20)

Priority bands:
  REFERENCE  = 0   (仅用于数值参考测试)
  PORTABLE   = 4   (通用 Triton / PyTorch fallback)
  PERFORMANT = 8   (经过优化的 Triton，Hopper+ 默认)
  SPECIALIZED= 12  (极致优化，窄 arch+shape 门控，如 Blackwell FP8)
  PLUGIN     = 16  (留给第三方插件，in-tree 不用)
```

选择结果全局缓存（key = family+mode+dtype+arch+objective+features+traits），热路径 = 单次 dict lookup。

### 2.4 KV Cache 三级存储

```
L1：GPU Device  (PageAllocator) — 在线服务的热 KV
L2：CPU Host    (PageAllocator) — Retract 写回 / Prefetch 加载
L3：持久化存储  (enable_l3_storage) — 跨请求 KV 存储（进行中）
```

RadixTree 作为前缀缓存索引：token 序列按 page 粒度 rolling hash，相同前缀跨请求复用 KV cache。HybridPrefixCache 同时管理 KV cache 和 Mamba cache（SSM 模型的状态），通过 eviction callback 联动驱逐。

---

## 三、执行时序图（关键路径）

### 3.1 请求启动到 Prefill 完成

```
Client HTTP                tokenspeed.cli / SMG           Scheduler (C++)          GPU (Modeling+Kernel)
    │                              │                             │                        │
    │─── POST /v1/chat ──────────>│                             │                        │
    │                              │─── AsyncLLM.generate() ──>│                        │
    │                              │                             │ SubmitRequests()       │
    │                              │                             │ Request{Submitted}     │
    │                              │                             │                        │
    │                  ┌───────────┴──────────────────┐         │                        │
    │                  │      Scheduler Loop          │         │                        │
    │                  │  (Python execution plane)    │         │                        │
    │                  └───────────┬──────────────────┘         │                        │
    │                              │─── NextExecutionPlan() ───>│                        │
    │                              │                             │ Match RadixTree        │
    │                              │                             │ schedulePrefillFirstChunk()│
    │                              │                             │ Request→Prefilling     │
    │                              │<── ExecutionPlan{          │                        │
    │                              │    FlatForwardOp(prefill)} │                        │
    │                              │                             │                        │
    │                              │─────── model.forward() ─────────────────────────────>│
    │                              │                             │     attention + MLA    │
    │                              │                             │     KV写入 page table  │
    │                              │<─── output tokens ──────────────────────────────────│
    │                              │                             │                        │
    │                              │─── Advance(ExtendResult) ─>│                        │
    │                              │                             │ Request→PrefillDone    │
    │                              │                             │   or →Decoding         │
```

### 3.2 Decode 循环（稳态）

```
Scheduler Loop               Scheduler (C++)              GPU
    │                              │                        │
    │─── NextExecutionPlan() ─────>│                        │
    │                              │ scheduleDecode()       │
    │                              │ 分配 tail page         │
    │                              │ Request remains Decoding│
    │<── ExecutionPlan{            │                        │
    │    FlatForwardOp(decode)}    │                        │
    │                              │                        │
    │──── model.forward(decode) ────────────────────────────>│
    │                              │        decode step     │
    │<── next_token ─────────────────────────────────────────│
    │                              │                        │
    │─── Advance(ExtendResult) ───>│                        │
    │                              │ token_container.Extend()│
    │                              │                        │
    │        [重复直到 EOS]         │                        │
    │                              │                        │
    │─── Advance(Finish) ─────────>│                        │
    │                              │ Request→Draining       │
    │─── NextExecutionPlan() ─────>│ newWriteBackOperation()│
    │<── ExecutionPlan{WriteBack}  │ Request→WritingBack    │
    │─── 异步 Device→Host copy ────>│                        │
    │─── Advance(WriteBackDone) ──>│ Request→Finished→删除  │
```

### 3.3 内存压力下的 Retract 路径

```
    [GPU 显存告急，scheduleDecode 失败]
    │
    │─── NextExecutionPlan() ─────>Scheduler
    │                              │ scheduleRetract(request)
    │                              │ Request→Retracting
    │<── ExecutionPlan{WriteBack(retract=True)}
    │─── 异步 KV Device→Host ─────>GPU
    │─── Advance(WriteBackDone) ──>Scheduler
    │                              │ Request→Retracted
    │
    │  [显存释放后，可再次调度]
    │─── NextExecutionPlan() ─────>Scheduler
    │                              │ scheduleDecodeFromRetracted()
    │                              │ → LoadBack Host→Device
    │<── ExecutionPlan{LoadBack}   │ Request→Decoding（继续）
```

---

## 四、与 vLLM / SGLang 对比分析

### 4.1 架构对比矩阵

| 维度 | TokenSpeed | vLLM | SGLang |
|------|-----------|------|--------|
| **调度控制面** | C++ FSM（编译时类型安全） | Python LLMEngine | Python RadixAttention |
| **并行策略** | local-SPMD + 静态编译器自动生成通信 | 手写 TP/PP/EP | 手写 TP + DeepEP MoE |
| **KV Cache 索引** | RadixTree（C++，rolling hash） | PagedAttention（Python dict） | RadixTree（Python） |
| **KV Cache 层级** | 3 层（GPU/CPU/L3 持久化） | 2 层（GPU/CPU swap） | 2 层（GPU/CPU） |
| **FSM 内存安全** | C++ 类型系统在编译时强制资源所有权 | 运行时引用计数 | 运行时管理 |
| **OOM 处理** | Retract：异步写回 host 继续服务 | Preemption：重新排队 | 类似 vLLM |
| **MLA 支持** | 专项 Blackwell kernel（tokenspeed-mla） | FlashInfer MLA | FlashInfer MLA |
| **Kernel 系统** | 可插拔 Registry + 三维打分 + 缓存 | 单一 backend 切换 | 多 backend 但无统一注册 |
| **Mamba 支持** | HybridPrefixCache（KV + Mamba 联合管理） | 有限 | 有限 |
| **P/D 分离** | 原生支持（Role::kP/kD/kFused） | 实验阶段 | 实验阶段 |
| **目标场景** | Agentic（高并发短 decode，B200） | 通用 | 通用 + 长上下文 |

### 4.2 性能优势（相对 vLLM / SGLang）

**1. C++ 调度器开销更低**

vLLM 和 SGLang 的调度器在 Python 主线程，GIL + Python 解释器开销直接影响调度延迟。TokenSpeed 的调度核心在 C++，Python 只做"调用 `next_execution_plan()`，传 plan 给 model"，调度延迟可降到微秒级，对 agentic 场景（大量短 decode）尤其关键。

**2. 类型系统保证的资源安全 + Retract 机制**

vLLM 的 preemption 需要"重排队并重新 prefill"（浪费 GPU 计算），SGLang 类似。TokenSpeed 的 Retract 将已完成 prefill 的 KV 写回 host、释放 GPU，后续 LoadBack 仅需 Host→Device 拷贝，避免重新计算 prefill，**在长上下文 agentic 场景下能显著减少重新 prefill 的浪费**。

**3. Blackwell 专项 MLA kernel**

- **Prefill**：CuTe DSL JIT + AOT binary 双后端，AOT 版超越 TRT-LLM（NVIDIA 内部 softmax tuning knobs）
- **Decode**：`fold_sq_factor` 优化——将 q_seqlen 折叠进 num_heads axis，提升 BMM1 的 M 维度利用率。对 agentic 场景（q_seqlen=4, num_heads=64）效果显著：在 80K KV 长度下，延迟大幅优于 TRT-LLM 的单 kernel 实现。

**4. 算子层可插拔设计**

三维打分（oracle/objective/priority）+ 运行时 override（env var / yaml / context manager），运维无需改代码即可切换 kernel backend。vLLM 需改代码或 fork；SGLang 没有统一 registry。

**5. local-SPMD 并行编译器**

用户只写单设备模型代码，并行策略由编译器从模块边界注解自动合成——相比 vLLM/SGLang 手写 TP 代码，维护成本低，且可灵活重组并行策略。

### 4.3 劣势与不足（相对 vLLM / SGLang）

**1. 生态成熟度**

vLLM 已是生产主流，支持 100+ 模型架构、AMD/NPU/CPU 等多硬件。SGLang 有活跃社区、完善文档。TokenSpeed 处于 Preview 状态，模型覆盖有限（Kimi K2.5、Qwen 3.6、DeepSeek V4 在建），NPU/Hopper 优化尚未合并主线。

**2. 硬件覆盖窄**

当前核心优化聚焦 NVIDIA Blackwell（B200/B300）。Hopper（H100/H800）优化、AMD MI350 优化均未完成。vLLM 和 SGLang 对 A100/H100 有深度优化，TokenSpeed 在 Hopper 上性能不一定有优势。

**3. 功能完整性**

以下功能 README 明确"进行中"：PD 分离正式版、EPLB（专家负载均衡）、KV Store、VLM 支持、完整 metrics 系统。vLLM/SGLang 这些功能均已 GA。

**4. 调度器黑盒程度**

C++ 调度器通过 nanobind 暴露给 Python，对应用层友好，但调试、定制调度策略比纯 Python 实现难度更高。vLLM 的 Python 调度器更容易 hack 和 override。

**5. 无 continuous batching 的精细 chunked prefill 记录**

vLLM 和 SGLang 对 chunked prefill 的调度策略有成熟的 benchmark 记录，TokenSpeed 的 chunked prefill 逻辑（`schedulePrefillFirstChunk` + `schedulePrefill`）功能完整但公开 benchmark 少。

### 4.4 性能优势聚焦场景

TokenSpeed 的性能优势在以下场景最为显著：

```
1. Agentic workload（高并发，短 decode，长 context）
   - Kimi K2.5 on B200：官方 blog 对比 TRT-LLM Pareto 曲线
   - MLA decode fold_sq_factor 专为小 q_seqlen（1-4）+ 大 KV 设计

2. DeepSeek 系列（MLA 架构）在 Blackwell 上的 prefill/decode
   - AOT MLA binary 超越 TRT-LLM

3. 需要超低调度延迟的场景
   - C++ scheduler 路径：微秒级 NextExecutionPlan，vLLM Python 路径毫秒级

4. 长上下文多 agent 并发（Retract 避免 OOM preemption 重算）
```

---

## 五、总结

TokenSpeed 是 LightSeek 针对 agentic workload + Blackwell 硬件的垂直优化推理引擎。其核心差异点：

1. **C++ FSM 调度器**：类型安全的请求生命周期管理，超低调度延迟
2. **Retract 机制**：不同于 vLLM preemption 重算，Host KV 保留继续服务
3. **TokenSpeed-MLA**：Blackwell 上最快的 MLA 实现之一（fold_sq_factor decode + AOT prefill）
4. **可插拔 Kernel Registry**：三维打分 + 运行时 override，比 vLLM/SGLang 更灵活

代价是：生态不成熟、硬件覆盖窄（聚焦 Blackwell）、若干关键功能（P/D、VLM、EPLB）仍在开发中。
