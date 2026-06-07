# SGLang 与 sgl-kernel-npu 对比分析

> 分析日期：2026-05-24  
> 代码仓：`sglang/`、`sgl-kernel-npu/`

---

## 1. 定位与关系

| 维度 | sglang | sgl-kernel-npu |
|------|--------|----------------|
| 定位 | 完整推理框架（调度器、HTTP服务、模型执行、分布式） | SGLang 在昇腾 NPU 上的**算子库**（kernel library） |
| 角色 | 上层框架，消费 kernel | 底层 kernel，供 SGLang/外部调用 |
| 硬件目标 | 主打 NVIDIA GPU（CUDA），兼容 ROCm/CPU/Metal | 专为华为昇腾 NPU（Atlas A2/A3）设计 |
| 独立性 | 独立运行，包含 sgl-kernel（GPU 算子库） | 独立发布的 whl，通过 `torch.ops.load_library` 挂载 |

**核心关系**：sgl-kernel-npu 是 sglang 框架的 NPU 算子后端，提供与 `sgl-kernel`（GPU）同层次的 kernel，通过 SGLang 的 plugin 系统接入框架。

---

## 2. sglang 整体架构

### 2.1 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                   HTTP / gRPC 服务层                         │
│           (entrypoints/, grpc/, sgl-model-gateway)           │
├─────────────────────────────────────────────────────────────┤
│                     调度与会话管理                            │
│        (managers/, session/, disaggregation/, eplb/)         │
├─────────────────────────────────────────────────────────────┤
│                   模型执行层                                  │
│   (model_executor/, model_runner, forward_batch_info)        │
│   ┌───────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│   │  models/  │  │  layers/ │  │speculative│  │   lora/  │  │
│   └───────────┘  └──────────┘  └──────────┘  └──────────┘  │
├─────────────────────────────────────────────────────────────┤
│                 硬件抽象 / Platform 层                        │
│     (platforms/ — SRTPlatform 接口 + CUDA/ROCm 实现)         │
├─────────────────────────────────────────────────────────────┤
│                      Kernel 层                               │
│     sgl-kernel (CUDA/ROCm)  │  sgl-kernel-npu (Ascend NPU)  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Platform 插件系统

SGLang 通过 setuptools entry_points 实现 OOT（Out-of-Tree）硬件平台扩展：

- `sglang.srt.platforms` 组：注册平台插件（`SRTPlatform` 子类）
- `sglang.srt.plugins` 组：注册通用插件（hook 注入、类替换等）
- 环境变量 `SGLANG_PLATFORM` 控制平台选择，避免加载非目标平台的依赖

`SRTPlatform` 抽象接口（`platforms/interface.py`）提供：
- `get_default_attention_backend()` — attention 后端
- `get_graph_runner_cls()` — 图执行器（对应 NPU 的 `NPUGraph`）
- `get_mha_kv_pool_cls()` / `get_mla_kv_pool_cls()` — KV cache 池
- `get_compile_backend()` — 编译后端（CPU/GPU 返回 inductor，NPU 可返回 `npugraph_ex`）

### 2.3 NPU 适配点（sglang 主仓中已有的 NPU 代码）

sglang 主仓对 NPU 有 **732 处**引用，主要位于：

| 文件 | 功能 |
|------|------|
| `compilation/npu_piecewise_backend.py` | `NPUPiecewiseBackend`，继承 CUDA 后端，将 `torch.npu.NPUGraph` 替换 CUDAGraph |
| `platforms/interface.py` | `SRTPlatform` 基类，NPU 平台通过 plugin 实现子类 |
| `server_args.py` | NPU 相关的 server arg 参数 |
| `disaggregation/` | 分离式推理中 NPU KV transfer 接口 |

---

## 3. sgl-kernel（GPU）架构

路径：`sglang/sgl-kernel/`

### 3.1 模块组成

```
sgl-kernel/csrc/
├── allreduce/          # custom AllReduce（NCCL bypass），MSCCLPP 加速
├── attention/          # MLA Decode (CUTLASS), merge attention states
├── elementwise/        # RMSNorm, SwiGLU, RoPE, GeLU
├── gemm/               # FP8/INT8/AWQ GEMM, DeepSeek V3 fused GEMM
├── moe/                # MoE routing/dispatch (CUTLASS), topk sigmoid/softmax
├── quantization/       # GGUF, FP8 blockwise, GPTQ
├── speculative/        # Eagle tree, ngram, speculative sampling
├── mamba/              # CausalConv1D, SSM state update
├── kvcacheio/          # KV cache IO (跨节点 KV 传输)
├── memory/             # 内存分配/管理
├── grammar/            # 语法约束采样 (logits bitmap)
└── lora/               # LoRA BGMV/SGMV
```

### 3.2 技术栈

- **CUDA C++**：核心 kernel（`.cu`/`.cuh`）
- **CUTLASS 3.x**：MLA decode、MoE GEMM 的高性能矩阵运算
- **FlashInfer**：部分 attention 路径
- **Torch Library**：通过 `TORCH_LIBRARY_FRAGMENT(sgl_kernel, m)` 注册到 `torch.ops.sgl_kernel`

---

## 4. sgl-kernel-npu 架构

路径：`sgl-kernel-npu/`

### 4.1 整体结构

```
sgl-kernel-npu/
├── csrc/                    # C++ / AscendC kernel 源码
│   ├── pytorch_extensions.cpp  # 统一算子注册入口
│   ├── attentions/          # 注意力（AscendC 实现 + Triton fallback）
│   ├── mla_preprocess/      # MLA 端到端融合预处理
│   ├── deepep/              # DeepEP-Ascend (MoE Expert Parallelism)
│   ├── assign_cache_op/     # KV cache 赋值
│   ├── alloc_extend/        # Paged KV 页分配
│   ├── cache_location_assign/  # 请求 → 物理 token 位置映射
│   ├── transfer_kv_dim_exchange/ # Host↔Device KV 传输
│   ├── lora/                # LoRA BGMV/SGMV/SGEMMV
│   ├── build_tree/          # 投机解码树构建
│   ├── speculative/         # Greedy tree verify
│   ├── causal_conv1d/       # Mamba SSM
│   ├── lightning_indexer/   # 稀疏 Top-K 索引
│   ├── mega_chunk_gdn/      # Flash Linear Attention chunk
│   ├── catlass/             # CATLASS（华为等效 CUTLASS）
│   └── ...
├── python/
│   ├── sgl_kernel_npu/      # Python whl 包
│   │   ├── attention/       # decode_attention.py (Triton 实现 MLA/GQA)
│   │   ├── fla/             # Flash Linear Attention (Triton)
│   │   ├── moe/             # MoE 计算 (Triton)
│   │   ├── norm/            # RMSNorm, split QKV+RMSNorm+RoPE (Triton)
│   │   ├── mamba/           # Causal Conv1D (Triton)
│   │   ├── activation/      # SwiGLU, SwiGLU-INT8 (Triton)
│   │   ├── speculative.py   # 投机解码树
│   │   └── kvcacheio.py     # KV cache IO
│   └── deep_ep/             # DeepEP-Ascend Python 包
```

### 4.2 技术栈

- **AscendC**：华为昇腾专用算子编程语言（类 CUDA C++），用于高性能 kernel
- **Triton**：多数 Python 端 kernel 用 Triton 实现（更易移植，NPU Triton 后端）
- **torch_npu**：PyTorch NPU 扩展（`PrivateUse1` device dispatch）
- **算子注册**：通过 `TORCH_LIBRARY_FRAGMENT(npu, m)` + `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)` 注册到 `torch.ops.npu`

### 4.3 两大子系统

#### DeepEP-Ascend

| 特性 | A3（Atlas A3） | A2（Atlas A2） |
|------|--------------|--------------|
| 通信 | 全网格 HCCS（节点内+节点间） | HCCS 节点内 + RDMA 节点间 |
| Normal Mode | 最大 65536 tokens/batch | 最大 8192 tokens/batch |
| Low-Latency Mode | <150us（128 tokens/batch） | — |
| 量化 | INT8/FP8/BF16 dispatch & combine | INT8/FP8/BF16 |

#### SGLang-Kernel-NPU

主要 kernel 功能同 GPU 版本一一对应，但实现路径不同：

| 功能 | GPU (sgl-kernel) | NPU (sgl-kernel-npu) |
|------|-----------------|---------------------|
| MLA Decode | CUTLASS SM100 | AscendC + Triton (Paged MLA) |
| GQA Decode | FlashInfer | Triton Paged GQA |
| MLA Preprocess | 分步实现 | AscendC 端到端融合（RMSNorm→Dequant→MatMul→RoPE→Cache） |
| RMSNorm | CUDA kernel | Triton kernel |
| AllReduce | NCCL / MSCCLPP | HCCS（通过 DeepEP dispatch/combine） |
| LoRA | BGMV/SGMV CUDA | BGMV/SGMV/SGEMMV AscendC |
| Speculative | CUDA tree build/verify | AscendC tree build + Triton verify |
| Mamba Conv1D | CUDA | AscendC |
| KV 管理 | CUDA | AscendC |

---

## 5. 核心设计差异

### 5.1 编程模型差异

| 维度 | sgl-kernel (GPU) | sgl-kernel-npu |
|------|-----------------|----------------|
| 核心语言 | CUDA C++ | AscendC（算子）+ Triton（Python 端融合）|
| 矩阵库 | CUTLASS 3.x | CATLASS（华为移植版）|
| 注册命名空间 | `torch.ops.sgl_kernel` | `torch.ops.npu`（PrivateUse1 dispatch）|
| 图执行 | CUDA Graph | NPU Graph (`torch.npu.NPUGraph`)  |
| 编译基础设施 | nvcc + CUTLASS | CANN + Ascend Compiler |

### 5.2 Attention 实现路径

**GPU MLA：** CUTLASS SM100 warp-level tile（`cutlass_mla_kernel.cu`），硬件近距离调度，极致延迟

**NPU MLA：**
1. **decode_attention.py**：Triton 实现 Paged MLA，支持 BF16/FP16，逻辑等价
2. **mla_preprocess**：AscendC 端到端融合，一次 kernel 完成 RMSNorm→Dequant→MatMul→RoPE→写 KV cache，减少多次读写显存的开销

### 5.3 MoE Expert Parallelism

- **GPU**：SGLang 内置 MoE dispatch，AllReduce 通过 MSCCLPP 或 NCCL
- **NPU**：DeepEP-Ascend 提供专用 MoE All-to-All，利用昇腾 HCCS 高速互联（类比 NVLink），支持 A3 全网格拓扑和 A2 的 RDMA 跨节点通信；Low-Latency Mode 专门为在线推理的 decode 阶段优化（sub-150us）

### 5.4 KV Cache 管理

两者都实现了：
- Paged Attention（block_table 映射）
- alloc_extend（动态页分配）
- cache_location_assign（请求→物理位置映射）
- transfer_kv_dim_exchange（Host↔Device KV 数据搬移，支持 disaggregated prefill）

NPU 版本额外考虑了 NPU ↔ CPU 内存的 HCCL 传输路径。

### 5.5 Flash Linear Attention（FLA）

NPU 版本有独立的 `fla/` 目录，包含：
- Gated Delta Rule (`chunk_delta_h.py`, `fused_gdn_gating.py`)
- Chunk attention 系列（`chunk.py`, `chunk_o.py`, `chunk_scaled_dot_kkt.py`）
- `mega_chunk_gdn.py`：AscendC 大 chunk 级 GDN 融合

GPU 版本（sgl-kernel）也有 FLA 支持，路径在 `sglang/python/sglang/srt/layers/` 中通过 `recurrent_gated_delta_rule` kernel。

---

## 6. sglang 中如何集成 NPU kernel

```
安装 sgl_kernel_npu whl
         │
         ▼
import sgl_kernel_npu        ← __init__.py 自动 dlopen libsgl_kernel_npu.so
         │
         ▼
torch.ops.npu.* 注册完成       ← TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
         │
         ▼
NPU Platform Plugin 加载        ← setuptools entry_points: sglang.srt.platforms
         │
         ▼
SRTPlatform.get_attention_backend()  → "triton_attn" / "ascendc_attn"
SRTPlatform.get_graph_runner_cls()   → NPUGraphRunner
SRTPlatform.get_mla_kv_pool_cls()   → NPUMLAKVPool
         │
         ▼
ModelRunner 使用 NPU 专用后端执行 forward
```

Python 端 kernel（Triton）在模型 layer 代码中通过 `from sgl_kernel_npu.attention import ...` 直接调用。

---

## 7. 总结

| 维度 | sglang（框架） | sgl-kernel-npu（NPU 算子库） |
|------|--------------|---------------------------|
| 功能完整性 | 完整推理框架：调度、serving、分布式、模型管理 | 仅 kernel 层，不含框架逻辑 |
| 硬件覆盖 | CUDA 为主，ROCm/CPU/Metal 通过 plugin | 昇腾 NPU 专用（A2/A3） |
| 接入方式 | 上层框架 | 通过 torch.ops.npu + Platform Plugin 接入 sglang |
| 实现语言 | Python + CUDA C++ + Triton | AscendC + Triton + C++ |
| MoE 通信 | MSCCLPP / NCCL AllReduce | DeepEP-Ascend HCCS All-to-All（专用 EP 通信） |
| 图执行 | CUDA Graph | NPU Graph |
| 对应关系 | 等价于 GPU 完整栈 | 等价于 GPU 栈中的 sgl-kernel 子组件 |

**设计哲学**：sgl-kernel-npu 不是 sglang 的 fork，而是通过 SGLang 标准 Plugin 接口实现的**硬件适配层**，最大限度复用 sglang 的框架逻辑（调度、serving、模型抽象），只在 kernel 层做 NPU 专用实现。
