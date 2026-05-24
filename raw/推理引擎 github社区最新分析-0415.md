Now I have comprehensive data. Let me compile the full report.

---

# 开源推理引擎周报（2026-04-09 ~ 2026-04-16）

## 一、最新版本发布概览

| 引擎 | 最新版本 | 发布日期 | 核心亮点 |
|------|---------|---------|---------|
| **vLLM** | v0.19.0 | 2026-04-03 | 448 commits, 197 contributors; Zero-bubble async+投机解码; MRV2 成熟; TurboQuant 2-bit KV; CPU KV offload; DBO 泛化; B300/GB300 支持 |
| **SGLang** | v0.5.10.post1 | 2026-04-09 | Piecewise CUDA Graph 默认开启; Elastic EP 部分容错; HiSparse 稀疏注意力; SGLang-Diffusion 扩展; FA4 官方支持; MXFP8 内核; GLM-5/DeepSeek V3.2 优化 |
| **vLLM-Ascend** | v0.18.0rc1 | 2026-04-01 | DeepSeek-V3.1 C8; A5 芯片 DeepSeek; Flash Comm V1 VL+MLA; Triton 编译优化; GDN prefill 优化 |

---

## 二、本周重点合入 PR

### vLLM（本周合入 ~112 PRs）

| PR | 标题 | 类别 | 影响 |
|----|------|------|------|
| [#38479](https://github.com/vllm-project/vllm/pull/38479) | **TurboQuant: 2-bit KV cache compression, 4x capacity** | 量化 | 在线 KV 压缩，无需离线校准。k8v4 压缩 2.6x, 3-bit 最高 4.9x。GSM8K 精度 0.78-0.86 (baseline 0.90)。**101 条讨论，本周最热 PR** |
| [#36162](https://github.com/vllm-project/vllm/pull/36162) | Mamba FlashInfer selective_state_update | 模型 | Mamba/GDN 混合模型关键算子 |
| [#36644](https://github.com/vllm-project/vllm/pull/36644) | KV offload + HMA: Remove block_size from KVEvents | 架构 | KV 卸载与分层内存管理解耦 |
| [#37206](https://github.com/vllm-project/vllm/pull/37206) | Unified memory layout for offloading workers | 内存 | KV 卸载统一内存布局 |
| [#39773](https://github.com/vllm-project/vllm/pull/39773) | Disable piecewise cudagraph fallback for eagle draft decodes (MRV2) | 投机 | MRV2 下 Eagle draft 图模式修正 |
| [#38372](https://github.com/vllm-project/vllm/pull/38372) | Simplify accepted token counting for hybrid models | 投机 | 混合模型(Mamba+Attention)投机解码简化 |
| [#35549](https://github.com/vllm-project/vllm/pull/35549) | Refactor ZeroExpertFusedMoE | MoE | MoE 框架重构 |
| [#39510](https://github.com/vllm-project/vllm/pull/39510) | TRTLLM GEN NVFP4 MoE for non-512-aligned hidden dims | MoE | NVFP4 MoE 扩展兼容性 |
| [#36029](https://github.com/vllm-project/vllm/pull/36029) | SPEED-bench support for benchmarking CLI | 投机 | 投机解码标准化评测工具 |

### SGLang（本周合入 ~50+ PRs）

| PR | 标题 | 类别 | 影响 |
|----|------|------|------|
| [#22604](https://github.com/sgl-project/sglang/pull/22604) | **Standalone Rollout API + Denoising Backpass for T2I Post-Training** | Diffusion+RL | 1305 行。为扩散模型 RL 后训练提供独立 rollout API，支持 SP 对齐 log-prob |
| [#22392](https://github.com/sgl-project/sglang/pull/22392) | **Eliminate nvjet memset bubbles via CUTLASS FP8 GEMM** | 性能 | 消除 112 次 memset (~2.2ms/forward)，用 CUTLASS 替换 nvjet cooperative GEMM |
| [#21985](https://github.com/sgl-project/sglang/pull/21985) | **Eliminate attention DtoD copy via pre-allocated FA output** | 性能 | 消除 28 次/forward 的 DtoD memcpy (~392µs)。vLLM 已有类似优化 |
| [#21734](https://github.com/sgl-project/sglang/pull/21734) | Optimize PCG inductor path for FP8 models | 性能 | FP8 编译图模式优化 |
| [#22316](https://github.com/sgl-project/sglang/pull/22316) | DeepSeek-R1-0528-w4a8: DeepEP Low Latency FP8 Communication | 通信 | W4A8 场景 FP8 通信加速 |
| [#20736](https://github.com/sgl-project/sglang/pull/20736) | Share expert fusion with router experts for Qwen3.5 BF16/FP8 | MoE | Qwen3.5 共享专家与路由专家融合 |
| [#21858](https://github.com/sgl-project/sglang/pull/21858) | Decoupled LoRA MoE backend with Marlin support | LoRA | MoE LoRA 解耦+Marlin 加速 |
| [#18016](https://github.com/sgl-project/sglang/pull/18016) | Add SiMM as HiCache Storage backend | 缓存 | 新的 KV cache 分层存储后端 |
| [#21232](https://github.com/sgl-project/sglang/pull/21232) | EPLB performance optimization | 负载均衡 | 专家负载均衡性能优化 |
| [#22667](https://github.com/sgl-project/sglang/pull/22667) | Diffusion: Ltx 2.3 two-stage ti2v support | Diffusion | 新扩散模型支持 |

### vLLM-Ascend（本周合入 ~20 PRs）

| PR | 标题 | 类别 |
|----|------|------|
| [#8035](https://github.com/vllm-project/vllm-ascend/pull/8035) | EPLB Swift balancer supports mix placement | 负载均衡 |
| [#7779](https://github.com/vllm-project/vllm-ascend/pull/7779) | Fuse W4A8 dispatch+FFN+combine into single kernel | 性能 |
| [#8004](https://github.com/vllm-project/vllm-ascend/pull/8004) | Fix Qwen3.5 MoE flash comm v1 shared expert shape error (MTP) | Bug修复 |
| [#8143](https://github.com/vllm-project/vllm-ascend/pull/8143) | Restrict Layer Sharding to PD P-node | PD分离 |
| [#7945](https://github.com/vllm-project/vllm-ascend/pull/7945) | Fix model_runner_v2 full graph mode errors | MRV2 |
| [#8263](https://github.com/vllm-project/vllm-ascend/pull/8263) | Fix MLA PrefillNoCache state for short prompt | 注意力 |

---

## 三、最热讨论 Issues/PRs

| 仓库 | # | 标题 | 讨论量 | 焦点 |
|------|---|------|--------|------|
| sglang | [#6017](https://github.com/sgl-project/sglang/issues/6017) | DeepSeek Large-scale PD+EP Instruction | **565 条** | 大规模部署最佳实践 |
| sglang | [#21569](https://github.com/sgl-project/sglang/pull/21569) | Upgrade transformers to 5.5.3 | 114 条 | 生态兼容性 |
| sglang | [#22217](https://github.com/sgl-project/sglang/pull/22217) | Eagle Beta Test Failure | 112 条 | 投机解码稳定性 |
| sglang | [#21985](https://github.com/sgl-project/sglang/pull/21985) | Eliminate attention DtoD copy | 75 条 | 微秒级性能优化 |
| vllm | [#38479](https://github.com/vllm-project/vllm/pull/38479) | TurboQuant 2-bit KV cache | **101 条** | 极致 KV 压缩 |
| vllm | [#36487](https://github.com/vllm-project/vllm/pull/36487) | CPU Replace OMP initialization | 67 条 | CPU 推理基础设施 |
| vllm | [#30566](https://github.com/vllm-project/vllm/pull/30566) | Update to transformers v5 | 64 条 | 生态迁移 |
| vllm | [#29772](https://github.com/vllm-project/vllm/pull/29772) | AFD (Async Fetch+Decode) | 45 条 | 异步调度创新 |
| ascend | [#5318](https://github.com/vllm-project/vllm-ascend/issues/5318) | vLLM Ascend Roadmap Q1 2026 | 38 条 | 路线图讨论 |

---

## 四、未来规划的加速特性

### vLLM 规划中

| 特性 | Ref | 状态 | 说明 |
|------|-----|------|------|
| Full CudaGraph for drafter | #33341 | Tracking | Eagle draft 全图捕获 |
| DFlash Parallel Drafting | #32206 | WIP | 扩散式并行投机 |
| NGram-GPU | #29184 | In Review | GPU 加速 n-gram 投机 |
| Hybrid ngram-eagle | #24344 | In Review | 混合投机策略 |
| DSL 动态投机长度 | #36657 | RFC | 置信度自适应 |
| MineDraft 批间并行 | #38003 | RFC | 吞吐 +75% |
| Proposer 接口统一 | #36219 | RFC | 架构重构 |
| Multi-MTP 层 | #31204 | RFC | 多层 MTP 联合推理 |
| HMA 分层内存管理 | #36644 系列 | 进行中 | KV offload 统一化 |
| Process-level Fault Tolerance | #28296 | Open | 进程级容错 |

### SGLang 规划中

| 特性 | 说明 |
|------|------|
| CUTLASS FP8 替换 nvjet | 消除 GEMM 前 memset 气泡（#22392 进行中） |
| HiCache 多后端 | SiMM/Mooncake/Ascend HiXL 存储后端 |
| Elastic EP 强化 | GPU P2P 专家权重交换 + EPLB 再平衡 |
| Diffusion RL Post-Training | Rollout API + 去噪环境反向传播 |
| 更多稀疏注意力 | HiSparse 扩展到更多模型 |

### vLLM-Ascend 规划中

| 特性 | Ref | 说明 |
|------|-----|------|
| DFlash 支持 | #8188/#8118 | 3 个 PR 竞争中 |
| MRV2 完善 | #5208 | Eagle/ngram/suffix/Triton 算子 |
| Proposer 重构 | #6881 | 对齐上游统一接口 |
| Eagle3 图模式 | #5459 | Draft 全图捕获 |
| W4A8 Fused Kernel | #7779 | dispatch+FFN+combine 融合 |

---

## 五、三大引擎对比分析

| 维度 | vLLM v0.19.0 | SGLang v0.5.10 | vLLM-Ascend v0.18.0rc1 |
|------|-------------|---------------|----------------------|
| **投机解码** | Zero-bubble async+spec; MRV2 rejection sampler; Eagle3 多模态; DFlash 基础支持; SPEED-bench 评测 | Eagle beta testing; FA4+spec decode; 投机解码稳定性仍在修复 | MTP 图模式(FULL_DECODE_ONLY); DFlash 3个PR开发中; Eagle/MTP 合并完成 |
| **KV Cache** | TurboQuant 2-bit (4x容量); CPU offload 通用化; HMA 分层管理; FP8 KV | HiCache 多后端(SiMM/Mooncake); IndexCache 10%+提升; HiSparse 稀疏注意力 | INT8 C8 (DeepSeek-V3.1); Prefix caching |
| **MoE 优化** | ZeroExpert 重构; NVFP4 MoE; GPT-OSS; 在线 MXFP8 | 共享专家融合(Qwen3.5); DeepEP FP8 通信; LoRA MoE; EPLB 优化 | Flash Comm V1; EPLB Swift; W4A8 fused kernel |
| **硬件支持** | NVIDIA B300/GB300; ROCm 7.2.1; Intel XPU; TPU; CPU | NVIDIA SM100/SM103; ROCm; Apple MLX; NPU (文档) | 昇腾 910B/A2/A3/A5; 310P |
| **PD 分离** | NIXL connector; Mooncake 异构 TP; Mamba N-1 prefill | GPU Staging Buffer (5x TPS); Elastic NIXL-EP 容错 | PD shape 对齐修复; Layer Sharding 限制 |
| **Diffusion** | — | **领先**: Rollout API+RL; LTX-2/Hunyuan3D/Helios; 1.5x 性能提升; macOS | — |
| **编译优化** | Mega AOT; Triton 自动调优默认; inductor cache | Piecewise CUDA Graph 默认; PCG inductor FP8 优化 | npugraph_ex; Triton 重编译优化 |
| **模型覆盖** | Gemma 4; NemotronH; ASR; 40+ 修复 | GLM-5; Nemotron-3-Super; Mistral Small 4; LFM2-VL; Voxtral | GLM-5.1(开发中); Qwen3.5; DeepSeek V3.1 |
| **生态** | transformers v5 兼容; speculators 训练库 | transformers 5.5.3; sglang-kernel 0.4.1; MLX | transformers 5.2.0; 社区周会 |

---

## 六、演进趋势分析

### 趋势 1：KV Cache 极致压缩成为新战场

vLLM TurboQuant (2-bit, 4x 容量) 是本周最热 PR (101 条讨论)。在 KV cache 成为长上下文推理瓶颈的背景下，从 FP16→FP8→INT4→2-bit 的压缩路径正在加速。SGLang 走的是 HiCache 多层级存储(GPU→CPU→SSD)路线。两种思路正交且可叠加。

### 趋势 2：微秒级 kernel 优化进入深水区

SGLang 本周两个高讨论 PR (#22392 消除 memset 气泡 ~2.2ms, #21985 消除 DtoD copy ~392µs) 代表了推理优化正在从宏观架构优化转向**逐 kernel 微秒级**的精细调优。vLLM 在编译层面（AOT, Triton autotuning）走类似路线。

### 趋势 3：投机解码从"能用"走向"工程成熟"

- vLLM: Zero-bubble async + spec decode 合入; SPEED-bench 标准化评测; MRV2 rejection sampler; 正在追求 full graph drafter
- SGLang: Eagle beta testing 暴露 112 条讨论的稳定性问题; FA4+spec decode
- vLLM-Ascend: DFlash 3个 PR 开发中; MTP 图模式已稳定
- 整体方向：**DSL(动态长度) + DFlash(并行投机) + 图模式(低开销)** 三者融合是下一个里程碑

### 趋势 4：SGLang 在 Diffusion + RL 领域建立差异化优势

SGLang-Diffusion 本周合入 Rollout API + RL 后训练支持，这是 vLLM 完全没有涉及的领域。配合 LTX-2/Hunyuan3D/Helios 等模型支持，SGLang 正成为**推理+训练一体化**的平台，而非单纯推理引擎。

### 趋势 5：Elastic EP + 容错成为大规模部署刚需

SGLang 的 Elastic NIXL-EP (GPU 故障后重新分配专家权重) 和 vLLM 的 Process-level Fault Tolerance RFC 都指向同一个方向：大规模 MoE 部署必须具备**局部故障不停服**的能力。

### 趋势 6：生态统一化 — transformers v5 迁移

vLLM (#30566, 64 条讨论) 和 SGLang (#21569, 114 条讨论) 都在推进 transformers v5 升级，这是本周两个仓库讨论量最高的 PR 之一。HuggingFace 生态的版本统一正在倒逼推理引擎快速适配。

### 趋势 7：NPU/多硬件适配加速但仍有代差

vLLM-Ascend 本周以 bugfix 和文档为主，功能性 PR 较少。SGLang NPU 支持仍以文档和 offloading 为主。相比 NVIDIA 端的 TurboQuant/FA4/CUTLASS FP8 等前沿优化，NPU 侧整体仍有 **1-2 个版本的特性延迟**。

---

## 七、能力矩阵速查

| 能力 | vLLM | SGLang | vLLM-Ascend |
|------|:----:|:------:|:-----------:|
| Zero-bubble Async+Spec | **已合入** | 部分 | 待适配 |
| 2-bit KV Cache | **已合入** | — | — |
| CPU KV Offload | **已合入** | HiCache 多后端 | — |
| DFlash 投机 | 基础支持 | — | 开发中 |
| MRV2 投机解码 | **成熟** | — | 进行中 |
| Elastic EP 容错 | 基础 | **领先** | 基础 |
| Diffusion 模型 | — | **领先** | — |
| FA4 | 支持 | **默认 SM100** | — |
| MXFP8 | 在线支持 | FlashInfer 内核 | — |
| LoRA MoE | 基础 | **成熟** | — |
| Apple MLX | — | **已支持** | — |