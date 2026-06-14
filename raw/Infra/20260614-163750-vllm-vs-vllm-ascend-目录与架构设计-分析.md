# vLLM 与 vLLM-Ascend 目录设计、架构设计、关键模块/类关系分析

> 生成时间：2026-06-14
> 范围：代码目录设计 + 架构分层 + 关键模块/类继承与协作关系；vllm 与 vllm-ascend 的「接管/复用/扩展」关系。
> 证据基线：本地代码仓快照——`vllm` @ `0d2961229`（2026-06-13）、`vllm-ascend` @ `8afdf356`（2026-06-13）。所有结论均附源码出处。

---

## 0. TL;DR

1. **vLLM 是「V1 架构」单进程多组件引擎**：请求经 `AsyncLLM`（API 层）→ `EngineCore`（独立进程，含 Scheduler + Executor）→ `Worker`/`GPUModelRunner`（设备执行）→ 输出经 `OutputProcessor` 回流。控制面（调度）与执行面（前向）在进程级解耦，靠 IPC 通信。
2. **vLLM 的硬件抽象靠三个接缝**：`Platform`（平台能力与类选择）、`AttentionBackend`（注意力后端注册）、`config` 驱动的类名字符串（`worker_cls`/`scheduler_cls`/backend cls 全部可被平台替换）。这就是 vllm-ascend 能「插进来」的底层逻辑。
3. **vLLM-Ascend 不 fork vLLM，而是「平台插件 + 继承 + monkey-patch」三件套**：① `NPUPlatform(Platform)` 在 `check_and_update_config` 里把 worker/scheduler/backend 全部指向 Ascend 实现；② `NPUWorker(WorkerBase)` / `NPUModelRunner(GPUModelRunner)` 继承复用 vLLM 执行骨架，只重写设备相关方法；③ `patch/` 目录分 `platform`（worker 启动前）与 `worker`（worker 启动时）两批 monkey-patch，热修 vLLM 内部不可继承的逻辑。
4. **DeepSeek/MoE/投机解码是两仓协同最密集的区域**：vLLM 提供模型骨架 + 算子接口 + spec_decode 框架，vllm-ascend 提供 NPU 算子实现（DSA/MLA/mHC/fused_moe）、ACL Graph 运行时、xlite C++ 整图旁路、以及 EP/MC2 通算融合。

---

## 1. vLLM 目录设计

### 1.1 顶层布局（`vllm/`）

按「职责域」而非「层级」组织，关键目录：

| 目录 | 职责 | 关键内容 |
|------|------|----------|
| `v1/` | **当前主架构（V1 engine）** | engine / core / worker / executor / attention / spec_decode / sample / kv_cache_* |
| `engine/` | 旧版 + arg 解析入口 | `arg_utils.py`、`llm_engine.py`（V0 残留 + 公共 arg） |
| `entrypoints/` | 对外入口 | `llm.py`（离线 LLM）、`openai/`、`anthropic/`、`api_server.py`、`cli/` |
| `config/` | **全量配置对象**（每个子系统一个文件） | `model.py`/`cache.py`/`parallel.py`/`scheduler.py`/`speculative.py`/`compilation.py`/`vllm.py`(顶层 `VllmConfig`) |
| `model_executor/` | 模型与算子实现 | `models/`（各架构）、`layers/`（attention/linear/fused_moe/mla/mhc/quantization/rotary…）、`model_loader/` |
| `attention/` + `v1/attention/` | 注意力抽象与后端 | `backend.py`、`backends/`（flash_attn/flashinfer/triton/mla/mamba…）、`selector.py` |
| `distributed/` | 并行与通信 | `parallel_state.py`、`device_communicators/`、`kv_transfer/`、`eplb/`、`elastic_ep/` |
| `platforms/` | **硬件抽象** | `interface.py`（`Platform` 基类）、`cuda.py`/`rocm.py`/`tpu.py`/`xpu.py`/`cpu.py` |
| `compilation/` | torch.compile + CUDA Graph | `decorators.py`(`@support_torch_compile`)、`cuda_graph.py`、`piecewise_backend.py`、`passes/` |
| `forward_context.py`、`sequence.py`、`sampling_params.py`、`outputs.py` | 跨层数据契约 | forward 上下文、序列/请求结构、采样参数、输出 |

设计取向：**配置即契约**（`config/` 把每个子系统的可调项对象化，`VllmConfig` 聚合）、**平台无关核心 + 平台特化边缘**（核心在 `v1/`，特化在 `platforms/` + 后端注册）。

### 1.2 V1 子架构（`vllm/v1/`，主战场）

```
v1/
├── engine/      # 引擎编排（进程模型）
│   ├── async_llm.py      AsyncLLM —— 面向 API 的异步客户端
│   ├── core.py           EngineCore / EngineCoreProc / DPEngineCoreProc —— 引擎核心（独立进程）
│   ├── core_client.py    与 EngineCore 通信的客户端
│   ├── llm_engine.py     LLMEngine —— 离线/同步入口
│   ├── output_processor.py / detokenizer.py / logprobs.py  输出后处理
│   └── input_processor.py  输入预处理（tokenize/multimodal）
├── core/        # 调度与 KV 管理（控制面）
│   ├── sched/scheduler.py      Scheduler(SchedulerInterface) —— schedule()/update_from_output()
│   ├── sched/async_scheduler.py  AsyncScheduler —— 异步调度变体
│   ├── kv_cache_manager.py / kv_cache_coordinator.py / block_pool.py  KV 块管理
│   └── single_type_kv_cache_manager.py  单类型 KV（含混合 KV 支持）
├── worker/      # 设备执行（执行面）
│   ├── gpu_worker.py     Worker(WorkerBase) —— 设备/分布式初始化 + execute_model
│   ├── gpu_model_runner.py  GPUModelRunner —— 前向主循环（prepare_inputs/execute_model/capture）
│   ├── gpu_input_batch.py   InputBatch —— 持久化 batch 状态
│   ├── block_table.py / workspace.py  块表与工作区
│   └── cpu_*/xpu_*/tpu_*    其他后端 runner
├── attention/   # 注意力后端选择 + ops
├── spec_decode/ # 投机解码（eagle/medusa/ngram/draft_model/dflash…）
├── sample/      # 采样器
├── executor/    # 进程/Ray 执行器（multiproc/uniproc/ray）
└── kv_cache_interface.py  KV 规格接口
```

### 1.3 V1 运行时数据流（请求生命周期）

```
HTTP/CLI 请求
  └─> entrypoints (openai/anthropic/llm.py)
       └─> AsyncLLM (v1/engine/async_llm.py)        [API 进程]
            └── IPC ──>
       EngineCoreProc (v1/engine/core.py)            [Engine 进程]
            ├─ Scheduler.schedule() → SchedulerOutput
            ├─ Executor (multiproc/ray) 分发到各 Worker
            │     └─> Worker.execute_model()         [每个设备进程]
            │           └─> GPUModelRunner.execute_model()
            │                 ├─ _prepare_inputs (InputBatch → 张量)
            │                 ├─ model.forward (含 CUDA Graph replay)
            │                 ├─ spec_decode propose (若启用)
            │                 └─ Sampler → sampled token ids
            ├─ Scheduler.update_from_output()  ← ModelRunnerOutput
            └── IPC ──> OutputProcessor (detokenize/stop/stream) ──> AsyncLLM ──> 客户端
```

要点：
- **进程边界在 EngineCore 与 API 之间**（`core_client.py`/IPC），调度循环不被 HTTP 阻塞。
- **DP（数据并行）= 多 `DPEngineCoreProc`**（core.py:1714），靠 `coordinator.py` 协同。
- **执行器抽象**（`v1/executor/abstract.py` + `multiproc_executor.py`/`ray_executor.py`/`uniproc_executor.py`）决定 Worker 怎么起、怎么分发。

---

## 2. vLLM 架构分层与硬件抽象接缝

### 2.1 三层 + 三接缝

```
┌─────────────────────────────────────────────────────────┐
│ 入口层  entrypoints/ (openai, anthropic, llm, cli)        │
├─────────────────────────────────────────────────────────┤
│ 编排层  v1/engine (AsyncLLM, EngineCore, OutputProcessor)│
├─────────────────────────────────────────────────────────┤
│ 控制面  v1/core/sched (Scheduler) + v1/core (KV manager) │
├─────────────────────────────────────────────────────────┤
│ 执行面  v1/worker (Worker, GPUModelRunner, InputBatch)   │
│         + model_executor (models, layers, loader)        │
├─────────────────────────────────────────────────────────┤
│ 抽象层  platforms/ (Platform) | attention (Backend) |     │
│         distributed (通信) | compilation (图)            │  ← 三接缝
└─────────────────────────────────────────────────────────┘
```

### 2.2 接缝① `Platform`（最关键，vllm-ascend 的入口）

`vllm/platforms/interface.py` 的 `Platform` 基类暴露一组 classmethod 供平台重写，核心是 `check_and_update_config(vllm_config)`——在引擎初始化早期被调用，**平台可在此改写整个 `VllmConfig`**，包括把 `parallel_config.worker_cls`、`scheduler_config.scheduler_cls` 改成自己的实现；以及 `get_attn_backend_cls(...)` 返回平台注意力后端类路径。vLLM 通过 `current_platform` 全局单例分派（cuda/rocm/tpu/xpu/cpu），第三方平台用插件机制注册。

> 底层逻辑：vLLM 把「用哪个 worker / 哪个 scheduler / 哪个 attention backend」做成**运行时可替换的类名字符串**，而不是硬编码 import。这是整个硬件适配生态（昇腾/XPU/第三方）的可行性根基。

### 2.3 接缝② `AttentionBackend`

`vllm/v1/attention/backend.py` 定义 `AttentionBackend`（+ `AttentionMetadata`/`AttentionMetadataBuilder`/`AttentionImpl`）；`backends/` 下各实现（flash_attn、flashinfer、triton、`mla/`、mamba 系）。`selector.py` 按 platform + dtype + 特性选后端。MLA 单独成子目录，因其元数据与 KV 布局特殊。

### 2.4 接缝③ `compilation`（torch.compile + 图捕获）

`@support_torch_compile`（`compilation/decorators.py`）装饰模型类（如 DeepSeek 的 `DeepseekV4Model`），驱动 piecewise compile 与 CUDA Graph 捕获（`cuda_graph.py`/`piecewise_backend.py`）。平台可替换图后端——vllm-ascend 即在此挂 ACL Graph。

---

## 3. vLLM-Ascend 目录设计

### 3.1 顶层布局（`vllm_ascend/`）

按「与 vLLM 接缝对位」组织——**每个目录基本对应 vLLM 的一个可替换点**：

| 目录/文件 | 对位 vLLM | 职责 |
|-----------|-----------|------|
| `platform.py` (`NPUPlatform`) | `platforms/interface.py` | **总入口**：类选择、config 改写、backend 分派 |
| `worker/` (`NPUWorker`, `NPUModelRunner`) | `v1/worker/` | 继承 GPU worker/runner，重写设备执行 |
| `attention/` | `v1/attention/backends/` | `AscendAttentionBackend`/`MLA`/`DSA`/`SFA`/`FA3` |
| `models/` | `model_executor/models/` | `deepseek_v4`、`deepseek_v4_mtp` 等 NPU 特化模型 |
| `ops/` | `model_executor/layers/` | NPU 算子：`fused_moe/`、`mla`、`mhc`、`dsa`、`rope`、`layernorm`、`linear` |
| `quantization/` | `model_executor/layers/quantization/` | W8A8/W4A8/FP8/modelslim 量化方法 |
| `compilation/` | `compilation/` | `acl_graph.py`（ACL Graph 全图）、`passes/`、融合 pass 管理 |
| `distributed/` | `distributed/` | NPU 通信、`kv_transfer/`（PD 分离）、EP |
| `core/` | `v1/core/sched/` | 调度增强：`scheduler_dynamic_batch`、`recompute_scheduler`、`profiling_chunk` |
| `spec_decode/` | `v1/spec_decode/` | NPU 投机解码 proposer（mtp/eagle/dflash/ngram…） |
| `eplb/` | `distributed/eplb/` | 专家负载均衡 |
| `xlite/` | （无对位）**新增** | C++ 整图 runtime 旁路（decode 单调用） |
| `patch/` | （无对位）**热修机制** | monkey-patch vLLM 内部不可继承逻辑 |
| `ascend_config.py` | （扩展 config） | `additional_config` 解析（xlite_graph/weight_prefetch/profiling_chunk…） |
| `ascend_forward_context.py` | `forward_context.py` | NPU forward 上下文（flash_comm、mc2 等运行期标志） |
| `_cann_ops_custom/`、`csrc/`、`meta_registration.py` | — | AscendC 自定义算子注册（`torch.ops._C_ascend`） |

### 3.2 设计取向

- **零 fork**：不复制 vLLM 源码，靠继承 + patch 跟随上游；代码里大量 `vllm_version_is(...)` 分支吸收上游版本差异（model_runner_v1.py 多处）。
- **算子下沉到 csrc + torch.ops**：`torch.ops._C_ascend.npu_hc_pre`/`inplace_partial_rotary_mul`/`dsa_kv_compress_scatter` 等，Python 层只编排。
- **运行时双轨**：默认 ACL Graph（compilation/），可选 xlite C++ 整图（xlite/）。

---

## 4. 两仓接管/复用/扩展关系（核心）

### 4.1 接管点：`NPUPlatform.check_and_update_config`

`vllm_ascend/platform.py:422` 起的 `check_and_update_config` 是「vLLM 把控制权交给昇腾」的总开关。证据（platform.py:610-664）：

```python
if parallel_config.worker_cls == "auto":
    if <310p>:
        parallel_config.worker_cls = "vllm_ascend._310p.worker_310p.NPUWorker310"
    elif <xlite enabled>:
        parallel_config.worker_cls = "vllm_ascend.xlite.xlite_worker.XliteWorker"
    else:
        parallel_config.worker_cls = "vllm_ascend.worker.worker.NPUWorker"
...
scheduler_config.scheduler_cls = (... vllm_ascend 调度增强 ...)
```

`get_attn_backend_cls`（platform.py:747）按模型/特性返回 `AscendAttentionBackend`/`AscendMLABackend`/`AscendDSABackend`/`AscendSFABackend`/`AscendFABackend` 之一。

### 4.2 复用点：继承链

```
vLLM                                vLLM-Ascend
WorkerBase (v1/worker/worker_base)  ←── NPUWorker (worker/worker.py:82)
                                          └── XliteWorker (xlite/xlite_worker.py:22)  [换 model_runner]
                                          └── NPUWorker310 (_310p/)
GPUModelRunner (v1/worker/gpu_model_runner) ←── NPUModelRunner (worker/model_runner_v1.py:255)
                                          └── XliteModelRunner (xlite/xlite_model_runner.py:25)
AttentionBackend (v1/attention/backend) ←── AscendAttentionBackend / MLA / DSA / SFA / FA3
SchedulerInterface ←── (core/ 调度增强类)
```

要点：
- **NPUModelRunner 重写而非重造**：`execute_model`/`load_model`/`initialize_kv_cache`/`_prepare_inputs`/`_dummy_run` 等，其余复用 GPUModelRunner（甚至直接 `GPUModelRunner.capture_model(self)`，model_runner_v1.py:4824）。
- **XliteModelRunner 极薄**（仅 56 行）：只重写 `load_model`（包 `XliteWrapper`）、`get_model`（unwrap）、`initialize_kv_cache`（注册 kv）、`_should_build_dummy_attn_metadata`——其余全继承 NPUModelRunner。

### 4.3 扩展点：`patch/` 两批 monkey-patch

`patch/__init__.py` 注释明确两阶段（platform.py:19-24）：
- `patch/platform/`：在 worker 启动**前**由 `NPUPlatform.pre_register_and_update()` 调用——改全局/进程级逻辑（分布式初始化、KV cache 接口、scheduler、多进程 executor、各模型 tool_call parser、DeepSeek V4 thinking/tool 解析、profiling_chunk…）。
- `patch/worker/`：worker 启动**时**由各 worker `__init__` 调用——改设备侧逻辑（triton、block_table、input_batch、model_state、gdn_attn、qwen3vl、weight_utils…）。`patch_v2/` 为新接口版本。

> 底层逻辑：能继承的走继承（worker/runner/backend），不能继承的（vLLM 内部函数、第三方 parser、上游未开放 hook 的点）走 patch。patch 数量（platform 20+，worker 10+）本身就是「跟随上游而不 fork」策略的成本与抓手。

### 4.4 协同最密集区：DeepSeek V4 全链路

| 环节 | vLLM 侧 | vLLM-Ascend 侧 |
|------|---------|----------------|
| 模型骨架 | `models/deepseek_v4/`（config/quant_config/ops） | `models/deepseek_v4.py`（`AscendDeepseekV4ForCausalLM`，mHC 融合算子 `npu_hc_pre/post`） |
| 注意力 | `layers/sparse_attn_indexer.py`、`layers/mla.py` | `attention/dsa_v1.py`（`AscendDSABackend` + 多流 CV 并行 prolog） |
| MoE | `layers/fused_moe/` | `ops/fused_moe/`（experts_selector、moe_comm_method、moe_stage_*；MC2 通算融合） |
| 量化 | `layers/quantization/` | `quantization/`（W4A8/W8A8、modelslim） |
| 投机解码(MTP) | `v1/spec_decode/`（eagle/draft_model 框架） | `spec_decode/llm_base_proposer.py`（merged draft、parallel drafting）、`models/deepseek_v4_mtp.py` |
| 图运行时 | `compilation/`（torch.compile/CUDA Graph） | `compilation/acl_graph.py`（ACL 全图 + GraphParams 原位更新）；或 `xlite/`（C++ 整图，**当前不支持 DSA**） |

---

## 5. 关键类关系图（文字版）

### 5.1 vLLM V1 引擎

```
AsyncLLM ──持有──> EngineCoreClient ──IPC──> EngineCoreProc(EngineCore)
EngineCore ──持有──> Scheduler(SchedulerInterface)
                └──> Executor(abstract) ──> {Multiproc|Ray|Uniproc}Executor ──> [Worker...]
Worker(WorkerBase) ──持有──> GPUModelRunner
GPUModelRunner ──持有──> InputBatch, AttentionBackend.Impl, Sampler, SpecDecode Proposer
GPUModelRunner ──读──> VllmConfig (model/cache/parallel/scheduler/speculative/compilation)
EngineCore ──产出──> ModelRunnerOutput ──> OutputProcessor ──> AsyncLLM
```

### 5.2 vLLM-Ascend 接管后

```
NPUPlatform.check_and_update_config ──改写──> VllmConfig.worker_cls/scheduler_cls/attn_backend
WorkerBase ←── NPUWorker ←── XliteWorker
GPUModelRunner ←── NPUModelRunner ←── XliteModelRunner
AttentionBackend ←── Ascend{Attention|MLA|DSA|SFA|FA3}Backend
NPUModelRunner ──持有──> ACLGraphWrapper(compilation/acl_graph) [默认]
XliteModelRunner ──持有──> XliteWrapper ──> xlite._C.{Runtime,Model} [旁路, MHA-only]
patch/platform/* ──pre_register_and_update()──> 热修 vLLM 全局逻辑
patch/worker/*   ──worker.__init__()──> 热修设备侧逻辑
torch.ops._C_ascend.* ──csrc/_cann_ops_custom──> AscendC 算子
```

---

## 6. 设计对比与评述

| 维度 | vLLM | vLLM-Ascend |
|------|------|-------------|
| 组织原则 | 职责域 + 平台无关核心 | 与 vLLM 接缝对位 + 算子下沉 |
| 硬件适配机制 | Platform/Backend/config 类名注入 | 实现这些接缝 + 继承 worker/runner + patch |
| 控制/执行解耦 | 进程级（EngineCore IPC） | 完全复用 |
| 图运行时 | torch.compile + CUDA Graph | ACL Graph（默认）+ xlite C++ 整图（旁路） |
| 跟随上游策略 | — | 零 fork，继承 + `vllm_version_is` 分支 + patch |
| 主要复杂度来源 | V1 多进程编排、KV/调度 | patch 维护、双图运行时、DSA/MoE/MTP 协同 |

**架构评述**：
- vLLM 的可扩展性根基是「**把硬件选择延迟到 config 改写时刻**」（接缝②③本质都依赖接缝① Platform）。这让 vllm-ascend 能以插件姿态存在。
- vllm-ascend 的工程取舍是「**继承优先、patch 兜底**」：好处是跟随上游快、不背 fork 包袱；代价是 patch 集合是隐性耦合面，上游接口变动时维护成本集中在 `patch/` 与 model_runner 的版本分支。
- **xlite 是架构上的「第二条执行轨」**：它绕开 vLLM 逐层 dispatch（Python）换 C++ 整图，但以 worker_cls 切换 + 极薄 runner 子类的方式接入，复用了 vLLM 的服务面与 KV/调度——这是「在不破坏主架构前提下引入专用 runtime」的范例（亦见 [[20260611-010803-vllm-ascend-deepseek-v4-runtime算子协同优化-方案设计]] 对其 DSA 缺口的分析）。

---

## 7. 源码证据索引

| 主题 | 位置 |
|------|------|
| vLLM V1 引擎类 | `vllm/v1/engine/core.py:96/867/1714`、`async_llm.py:70`、`llm_engine.py:47` |
| vLLM 调度器 | `vllm/v1/core/sched/scheduler.py:66/357/1395`、`async_scheduler.py` |
| vLLM worker/runner | `vllm/v1/worker/gpu_worker.py:117/806`、`gpu_model_runner.py:418/4000/5082/7242` |
| vLLM 平台抽象 | `vllm/platforms/interface.py`(`Platform`)、`v1/attention/backend.py` |
| vLLM 配置聚合 | `vllm/config/vllm.py`(`VllmConfig`) + `config/*.py` |
| vLLM compile/图 | `vllm/compilation/decorators.py`(`@support_torch_compile`)、`cuda_graph.py` |
| Ascend 平台接管 | `vllm_ascend/platform.py:134/422/610-664/747` |
| Ascend worker/runner 继承 | `vllm_ascend/worker/worker.py:82`、`worker/model_runner_v1.py:255`、`xlite/xlite_worker.py:22`、`xlite/xlite_model_runner.py:25` |
| Ascend attention backends | `vllm_ascend/attention/{attention_v1,mla_v1,dsa_v1,sfa_v1,fa3_v1}.py` |
| Ascend patch 两阶段机制 | `vllm_ascend/patch/__init__.py:19-24` + `patch/platform/*`、`patch/worker/*` |
| Ascend config 扩展 | `vllm_ascend/ascend_config.py`(XliteGraphConfig/WeightPrefetchConfig/ProfilingChunkConfig) |
| DeepSeek V4 协同 | `vllm_ascend/models/deepseek_v4.py`、`attention/dsa_v1.py`、`ops/fused_moe/`、`spec_decode/llm_base_proposer.py` |
