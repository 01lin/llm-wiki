# TokenSpeed 昇腾 Ascend A2/A3 适配方案与可行性分析

> 版本：2026-05-24
> 上游代码仓：[lightseekorg/tokenspeed](https://github.com/lightseekorg/tokenspeed)
> 对照参考：[vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend)
> 关联文档：[[20260524-215754-tokenspeed-architecture-analysis-分析]]

---

## TL;DR

**结论：整体可行，建议采用「Plugin + Platform 最小侵入」路径，不 fork TokenSpeed 主仓。**

| 维度 | 结论 |
|------|------|
| 主仓改动 | 约 5 行（`platform.py` 增加 `"ascend"` vendor 分支） |
| 外挂适配包 | `tokenspeed-kernel-ascend`（参考 vllm-ascend 命名范式） |
| 性能上限预估 | A3 上 MLA decode 约为 B200 的 **40-60%** |
| 总工期 | 约 11 周（4 个 Phase） |
| 最大风险 | `tokenspeed-mla` Blackwell 专属 CuTe DSL kernel 无法移植，需用 CANN `npu_fused_infer_attention_score` 替代 |
| 最小依赖 | `torch_npu 2.10 + triton-ascend 3.2.1 + CANN 9.0.0` |

---

## 一、可行性分层评估

### 1.1 分层难度矩阵

| 层次 | 可移植性 | 工作量 | 关键障碍 |
|------|---------|--------|---------|
| C++ 调度器（`tokenspeed-scheduler`） | ✅ 完全可移植 | 极低 | 无，纯 CPU C++ 逻辑，零硬件依赖 |
| Python runtime 主干（模型、引擎、调度） | ⚠️ 大部分可移植 | 中等 | `torch.cuda.*` stream/event/graph 需替换为 `torch.npu.*` |
| Kernel 层（`tokenspeed-kernel` Triton 路径） | ✅ 可移植 | 中等 | 需要 `triton-ascend==3.2.1`（华为官方 Triton fork） |
| Kernel 层（vendor-specific：trtllm/flashinfer/deepep） | ❌ 不可移植 | 高 | CUDA-only 库，需用 CANN 等价物替换 |
| MLA Kernel（`tokenspeed-mla`） | ❌ 不可移植 | 极高 | Blackwell SM100/CuTe DSL 专属，需全新 CANN 实现 |
| CUDA Graph | ⚠️ 有替代路径 | 中等 | 替换为 ACLGraph（`torch.npu.NPUGraph`），vllm-ascend 有完整参考 |
| 量化（NVFP4/FP8） | ⚠️ 部分可移植 | 高 | FP8 有昇腾支持，NVFP4 无直接等价，需用 W8A8/W4A16 量化代替 |
| 通信层（NCCL） | ⚠️ 有替代路径 | 低 | 替换为 HCCL，vllm-ascend `pyhccl.py` 已有完整实现 |

### 1.2 平台检测层硬约束（必改点）

`tokenspeed-kernel/python/tokenspeed_kernel/platform.py` 当前的 `_detect_platform()` 仅支持 CUDA 和 ROCm：

```python
if torch.cuda.is_available():
    if hasattr(torch.version, "hip") and torch.version.hip:
        return _detect_rocm_platform()
    return _detect_cuda_platform()
raise RuntimeError("tokenspeed-kernel requires an NVIDIA CUDA or AMD ROCm GPU.")
```

`PlatformInfo.vendor` 当前仅含 `"nvidia"` 和 `"amd"`。`CapabilityRequirement.satisfied_by()` 的 `vendors` 字段控制 kernel 选择——若不扩展，NPU 平台所有 kernel 全部被过滤。

### 1.3 最关键的性能瓶颈：MLA Kernel

TokenSpeed 在 Blackwell 上 540 TPS 的核心是 `tokenspeed-mla`：

- 基于 **CuTe DSL + SM100 UTCMMA 指令**
- `fold_sq_factor`、Split-KV、2CTA 优化均为 NVIDIA GPU 专属原语
- **昇腾无直接等价物**

昇腾的 MLA 路径（vllm-ascend `mla_v1.py` 已验证）：
- 调用 CANN `torch_npu.npu_fused_infer_attention_score`
- A3 还支持 `flash_attn_npu_v3`（FA3 on Ascend）

---

## 二、整体架构设计

### 2.1 顶层设计哲学：外挂式 Plugin，不污染主仓

TokenSpeed 架构本身提供了完整扩展点：
- `tokenspeed_kernel.plugins` Python entry_points 机制
- `CapabilityRequirement.vendors` 多 vendor 支持
- `Priority.PLUGIN` band（16-19）专门留给外部插件覆盖

利用这些扩展点做外挂式适配包，比 fork 主仓维护成本低一个数量级。

### 2.2 分层适配架构图

```
┌────────────────────────────────────────────────────────────────────┐
│  tokenspeed-kernel-ascend/   (独立 pip 包，~95% 新代码)               │
│  ├── tokenspeed_kernel_ascend/                                      │
│  │   ├── platform_ascend.py  ← 昇腾 PlatformInfo + A2/A3 differential│
│  │   ├── register.py         ← entry_point 入口                      │
│  │   ├── ops/                                                       │
│  │   │   ├── attention/                                             │
│  │   │   │   ├── npu_mla.py        ← CANN MLA封装                   │
│  │   │   │   ├── npu_fa.py         ← Flash Attention NPU            │
│  │   │   │   └── triton_mha.py     ← triton-ascend MHA              │
│  │   │   ├── gemm/  npu_gemm.py    ← torch_npu mm/matmul            │
│  │   │   ├── moe/   npu_moe.py     ← npu_moe_init_routing_custom    │
│  │   │   ├── layernorm/             ← triton-ascend RMSNorm/RoPE    │
│  │   │   ├── quantization/  w4a16/w8a8                              │
│  │   │   └── sampling/   npu_sampling                               │
│  │   └── communication/  pyhccl_wrapper.py                          │
└────────────────────────────────────────────────────────────────────┘
                                  ↑ 通过 entry_point 注册到主仓
┌────────────────────────────────────────────────────────────────────┐
│  tokenspeed 主仓 (最小改动 ~50 行)                                    │
│  ├── tokenspeed-kernel/platform.py     ← +ascend vendor 分支         │
│  ├── python/tokenspeed/runtime/                                     │
│  │   ├── execution/model_executor.py   ← device-agnostic stream     │
│  │   ├── execution/cuda_graph_wrapper.py ← +ACLGraph 分支            │
│  │   └── distributed/comm_backend/                                  │
│  │       ├── auto.py                   ← 自动选择 NCCL/HCCL          │
│  │       └── hccl.py                   ← 新增 HCCL backend           │
└────────────────────────────────────────────────────────────────────┘
                                  ↓
┌────────────────────────────────────────────────────────────────────┐
│  tokenspeed-scheduler/ (C++ 层，零改动)                              │
│  纯 CPU FSM 调度，与硬件解耦                                          │
└────────────────────────────────────────────────────────────────────┘
```

### 2.3 关键改动逐层说明

#### Layer 1：platform.py vendor 扩展（主仓，~5 行）

```python
# tokenspeed-kernel/python/tokenspeed_kernel/platform.py
def _detect_platform() -> PlatformInfo:
    if torch.cuda.is_available():
        if hasattr(torch.version, "hip") and torch.version.hip:
            return _detect_rocm_platform()
        return _detect_cuda_platform()
    
    # 新增 Ascend 检测分支
    try:
        import torch_npu  # noqa
        if torch.npu.is_available():
            from tokenspeed_kernel_ascend.platform_ascend import _detect_ascend_platform
            return _detect_ascend_platform()
    except ImportError:
        pass
    
    raise RuntimeError("tokenspeed-kernel requires NVIDIA CUDA, AMD ROCm, or Ascend NPU.")
```

`PlatformInfo.vendor` 增加 `"ascend"`，新增 `is_ascend`、`is_a2`、`is_a3` property。

#### Layer 2：Plugin 包入口注册

`tokenspeed-kernel-ascend/pyproject.toml`：
```toml
[project.entry-points."tokenspeed_kernel.plugins"]
ascend = "tokenspeed_kernel_ascend.register:register"
```

`register.py` 在 `discover_plugins()` 时被调用，触发所有 Ascend kernel 注册：

```python
@register_kernel(
    family="attention", mode="decode_with_kvcache",
    capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
    priority=Priority.PLUGIN,  # 16，确保覆盖默认 fallback
)
def npu_mla_decode(...):
    return torch_npu.npu_fused_infer_attention_score(...)
```

#### Layer 3：MLA Attention（核心性能）

**A2（Atlas 800T A2，soc_version 220-225）**：
- CANN `npu_fused_infer_attention_score` 原生支持 MLA（vllm-ascend `mla_v1.py` L19-60 已验证）
- decode：封装 `torch_npu._C._fused_attention` 系列算子
- prefill：使用 Flash Attention NPU（`flash_attn_npu` 或 FA3 variant）

**A3（Atlas 800T A3，soc_version 250-255）**：
- FA3（FlashAttention v3 on Ascend）已在 vllm-ascend 支持（`Dockerfile.a3` 中 `flash_attn_npu_v3`）
- A3 的 Cube Core 数量更多，MLA decode 吞吐更高
- HCCL 默认模式为 AICPU，需设置 `HCCL_BUFFSIZE` 环境变量

**性能差距对齐**：
- `fold_sq_factor` 优化（Blackwell UTCMMA 专属）**无法直接移植**
- 替代方向：昇腾 CANN SplitFusedAttention（SFA）中的 batch grouping 可类似地提升小 q_len 利用率（vllm-ascend `sfa_v1.py`）
- **预估性能上限：A3 上 MLA decode 约为 B200 的 40-60%**（参考 vllm-ascend benchmark 数据）

#### Layer 4：CUDA Graph → ACLGraph

`cuda_graph_wrapper.py` 需做 device-conditional 分支：

```python
# 通过 platform abstraction 层包装
if current_platform.is_nvidia or current_platform.is_amd:
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
elif current_platform.is_ascend:
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
```

vllm-ascend `compilation/acl_graph.py` 中 `ACLGraphWrapper` 提供完整参考实现：
- `torch.npu.NPUGraph()` capture/replay 接口
- graph pool 管理（`current_platform.get_global_graph_pool()`）
- input address validation
- BatchDescriptor 字典缓存

#### Layer 5：通信层 HCCL

`distributed/comm_backend/hccl.py` 新增，参考 vllm-ascend `pyhccl.py`：
- HCCL so 文件发现（`find_hccl_library()`）
- `PyHcclCommunicator.all_reduce()` / `all_gather()` 等接口
- ACLGraph capture 下的 HCCL 兼容

参考代码段（`vllm_ascend/distributed/device_communicators/pyhccl.py`）：
```python
from vllm_ascend.distributed.device_communicators.pyhccl_wrapper import (
    HCCLLibrary, aclrtStream_t, buffer_type, hcclComm_t,
    hcclDataTypeEnum, hcclRedOpTypeEnum, hcclUniqueId,
)

class PyHcclCommunicator:
    def __init__(self, group, device, library_path=None):
        self.hccl = HCCLLibrary(library_path)
        ...
```

#### Layer 6：MoE Dispatch

vllm-ascend 中 `vllm_ascend/ops/fused_moe/fused_moe.py` 有完整昇腾 MoE 实现：
- `torch.ops._C_ascend.npu_moe_init_routing_custom` 替代 deepep dispatch
- `torch_npu.npu_grouped_matmul` 替代 expert grouped GEMM
- `vllm_ascend/eplb/` 提供 Expert Parallel Load Balancing

#### Layer 7：量化方案

| TokenSpeed 原始量化 | 昇腾等价方案 | 工作量 | 性能影响 |
|---------|---------|------|--------|
| NVFP4 权重 | W4A16（INT4 GPTQ/AWQ） | 高（需 checkpoint 转换） | 略低于 NVFP4 |
| FP8 KV Cache | FP8 KV Cache | 中（CANN 支持 FP8） | 接近 |
| FP8 MLA kernel | INT8 W8A8 / FP16 | 中（A2 Cube Core INT8 加速） | A2 性能可比 FP8 |

---

## 三、A2 vs A3 硬件差异对照

| 维度 | A2（Atlas 800T A2） | A3（Atlas 800T A3） |
|------|---------------------|---------------------|
| 芯片 | Ascend 910B | Ascend 910C |
| soc_version | 220-225 | 250-255 |
| HBM 带宽 | ~2 TB/s | ~4 TB/s |
| Cube Core | 较少 | ~2x A2 |
| HCCL 默认模式 | PCIE 模式 | AICPU（需 env 调优） |
| FA3 支持 | 部分（依赖配置） | 完整支持 `flash_attn_npu_v3` |
| ACLGraph | 支持（eager mode 更稳定） | 完整支持 |
| 预估相对于 B200 性能 | ~25-35% | ~40-60% |

vllm-ascend 代码中明确区分 A2/A3 的位置：
- `vllm_ascend/utils.py` L820-870：`AscendDeviceType.A2=0, A3=1`，通过 `soc_version` 区分
- `vllm_ascend/cpu_binding.py` L23-24：A3 用 `GLOBAL_SLICE_MODE`，A2 用 `TOPO_AFFINITY_MODE`
- `vllm_ascend/utils.py` L648-651：A3 HCCL 默认 AICPU，需调优避免显著性能下降
- `vllm_ascend/envs.py` L70：`HCCL_INTRA_PCIE_ENABLE=1` 仅 A2 生效

---

## 四、性能逼近策略

为最大化逼近 540 TPS 的 Blackwell 性能，按重要性排序的优化抓手：

### 4.1 第一性能抓手：MLA Kernel 优化

| 优化项 | Blackwell 实现 | 昇腾对标实现 | 预估收益恢复率 |
|--------|---------------|--------------|---------------|
| fold_sq_factor | CuTe DSL + UTCMMA tile 折叠 | CANN SFA batch grouping | ~50-70% |
| Split-KV 两 kernel | CuTe DSL + workspace | CANN split + reduce 双 kernel | ~80% |
| 2CTA UTCMMA | Blackwell 指令 | A3 Cube Core 双 die 协同 | ~60% |
| FP8 KV decode | CuTe FP8 | CANN FP8 / INT8 W8A8 | ~70-90% |

### 4.2 第二性能抓手：调度器零搬运成本

TokenSpeed C++ FSM Scheduler 是纯 CPU 逻辑，**直接拿来用，零损失**：
- Radix Tree prefix cache → agentic 多轮对话 KV 高命中率
- Retract 机制 → preemption 不浪费算力
- C++ 控制面延迟 < 1ms

这是相比 vllm-ascend 的核心差异化优势。

### 4.3 第三性能抓手：通信优化

- AllReduce + RMSNorm 融合：在 HCCL 上需重写 fused kernel（triton-ascend 实现）
- DeferredReduceOp：与硬件无关的 placement 优化，直接复用
- Ring AllGather + ReduceScatter：HCCL 默认支持

### 4.4 第四性能抓手：投机解码（EAGLE3）

- EAGLE3 draft model 与硬件无关，只需 draft attention backend 切换
- agentic 场景 accept rate 高，A3 上理论可保留 2-3x decode 加速
- vllm-ascend 已有 `spec_decode/dflash_proposer.py` 参考

### 4.5 第五性能抓手：ACLGraph

- decode batch 固定形状用 ACLGraph capture
- 与 CUDA Graph 等效，但需注意 A2/A3 stream API 差异
- vllm-ascend `ACLGraphWrapper` 是直接参考

---

## 五、实施路线图（4 Phase，约 11 周）

### Phase 1：Runtime 层可运行（2 周）

| 任务 | 关键文件 | 验收标准 |
|------|---------|---------|
| platform.py 增加 ascend vendor | `tokenspeed-kernel/platform.py` | `current_platform().is_ascend == True` |
| device-agnostic stream | `runtime/execution/model_executor.py` | torch.cuda 和 torch.npu 路径都能跑 |
| ACLGraph 适配 | `runtime/execution/cuda_graph_wrapper.py` | decode batch capture/replay 通过 |
| HCCL backend | `runtime/distributed/comm_backend/hccl.py` | 4卡 TP AllReduce 通过 |
| **里程碑** | — | Qwen3 dense FP16 在 A2 跑通 prefill+decode |

### Phase 2：Kernel 层性能（3 周）

| 任务 | 关键文件 | 验收标准 |
|------|---------|---------|
| 创建 plugin 包骨架 | `tokenspeed-kernel-ascend/` | pip install 后自动注册 |
| triton-ascend RMSNorm/RoPE | `ops/layernorm/triton_rms.py` | 数值对齐误差 < 1e-3 |
| CANN attention 封装 | `ops/attention/npu_mla.py` | MLA decode 通过 |
| MoE dispatch 封装 | `ops/moe/npu_moe.py` | MoE 数值对齐 |
| **里程碑** | — | Qwen3.5-MoE 在 A2 benchmark 对比 vllm-ascend ≥ 100% |

### Phase 3：MLA 性能优化（4 周）

| 任务 | 关键文件 | 验收标准 |
|------|---------|---------|
| CANN MLA prefill | `ops/attention/npu_mla_prefill.py` | A3 prefill TPS 达到 vllm-ascend × 1.2 |
| CANN MLA decode + batch grouping | `ops/attention/npu_mla_decode.py` | 探索 SFA batch grouping 优化 |
| A3 FA3 backend | `ops/attention/npu_fa3.py` | 集成 `flash_attn_npu_v3` |
| FP8 KV Cache | `ops/kvcache/npu_fp8.py` | KV 内存减半 |
| ACLGraph decode capture | `runtime/execution/cuda_graph_wrapper.py` | decode kernel launch 开销 → 0 |
| **里程碑** | — | DeepSeek V3/Kimi K2.5 在 A3 上 agentic benchmark ≥ B200 性能 50% |

### Phase 4：量化与生产化（2 周）

| 任务 | 关键文件 | 验收标准 |
|------|---------|---------|
| W4A16 量化路径 | `ops/quantization/w4a16.py` | checkpoint 转换工具 + 推理通过 |
| EPLB 集成 | `runtime/moe/eplb_algorithms/ascend.py` | 多轮 expert 负载均衡 |
| A2/A3 differential config | `platform_ascend.py` | 自动按 soc_version 调优 |
| Dockerfile 生产化 | `Dockerfile.ascend-a2`, `Dockerfile.ascend-a3` | 镜像可一键部署 |
| **里程碑** | — | 完整 EAGLE3 + W4A16 + EPLB 端到端可用 |

---

## 六、关键参考文件索引

| 功能 | vllm-ascend 参考路径 | TokenSpeed 需改路径 |
|------|---------------------|---------------------|
| 平台检测 | `vllm_ascend/platform.py` | `tokenspeed-kernel/platform.py` |
| MLA attention | `vllm_ascend/attention/mla_v1.py` | `tokenspeed-kernel-ascend/ops/attention/npu_mla.py` |
| FA3 attention | `vllm_ascend/attention/fa3_v1.py` | `tokenspeed-kernel-ascend/ops/attention/npu_fa3.py` |
| SFA attention | `vllm_ascend/attention/sfa_v1.py` | `tokenspeed-kernel-ascend/ops/attention/npu_sfa.py` |
| DSA attention | `vllm_ascend/attention/dsa_v1.py` | `tokenspeed-kernel-ascend/ops/attention/npu_dsa.py` |
| ACLGraph | `vllm_ascend/compilation/acl_graph.py` | `runtime/execution/cuda_graph_wrapper.py` |
| HCCL 通信 | `vllm_ascend/distributed/device_communicators/pyhccl.py` | `runtime/distributed/comm_backend/hccl.py` |
| MoE dispatch | `vllm_ascend/ops/fused_moe/fused_moe.py` | `tokenspeed-kernel-ascend/ops/moe/npu_moe.py` |
| Triton kernels | `vllm_ascend/ops/triton/` | `tokenspeed-kernel-ascend/ops/layernorm/` 等 |
| A2/A3 差异 | `vllm_ascend/utils.py` L820-870 | 新增 `platform_ascend.py` |
| Spec decode | `vllm_ascend/spec_decode/` | `runtime/spec_decode/` 端到端联调 |
| EPLB | `vllm_ascend/eplb/` | `runtime/moe/eplb_algorithms/ascend.py` |
| KV transfer | `vllm_ascend/distributed/kv_transfer/` | `runtime/cache/transfer/ascend.py` |
| Dockerfile | `Dockerfile.a3` | 新建 `Dockerfile.ascend-a2/a3` |
| 软件栈 | `requirements.txt` | torch_npu 2.10 + triton-ascend 3.2.1 + CANN 9.0.0 |

---

## 七、验证计划

### 7.1 单元验证

```python
# 验证 plugin 注册
from tokenspeed_kernel import discover_plugins
from tokenspeed_kernel.selection import select_kernel
discover_plugins()

# 在 NPU 环境下应该选到 npu_mla 而非默认 fallback
kernel = select_kernel(family="attention", mode="decode_with_kvcache")
assert "npu" in kernel.name.lower()
```

### 7.2 正确性验证

| 模型 | 量化 | 参考实现 | 对齐标准 |
|------|------|---------|--------|
| Qwen3-8B | FP16 | vLLM CUDA 输出 | 数值误差 < 1e-2，BLEU ≥ 0.95 |
| Qwen3.5-MoE-32B | W4A16 | vllm-ascend | 完全一致输出（greedy）|
| DeepSeek V3 | FP8 KV | TokenSpeed CUDA | 误差 < 1e-2 |

### 7.3 性能验证

- 数据集：SWE-smith agentic dataset（与 TokenSpeed agentic_bench.sh 一致）
- 基线 1：vllm-ascend 同硬件 TPS
- 基线 2：TokenSpeed on B200 TPS
- 目标：
  - vs vllm-ascend：**≥ 120%**（C++ Scheduler + Radix Tree 优势）
  - vs B200：**A3 ≥ 50%，A2 ≥ 30%**

### 7.4 稳定性验证

- 多轮对话压测：1000 轮 conversation，无内存泄漏
- Radix Tree prefix cache 在 NPU 上正确性
- Retract 机制：host KV write-back + load-back 闭环
- HCCL 在 ACLGraph capture 下稳定

---

## 八、风险与依赖

### 8.1 主要风险

| 风险 | 等级 | 缓解策略 |
|------|------|---------|
| MLA kernel 性能远低于预期 | 高 | 与华为算子团队拉通，定制 CANN MLA 算子 |
| triton-ascend 不支持某些 Triton 特性 | 中 | 用 CANN ops 兜底，PR 提交给 triton-ascend |
| ACLGraph 与 HCCL 互不兼容（特定场景） | 中 | 参考 vllm-ascend 已踩坑路径 |
| NVFP4 → W4A16 量化精度下降 | 中 | 校准数据集 + 混合精度补救 |
| DeepEP 无昇腾等价物 | 低 | 用 CANN MoE 算子 + Triton 自实现 |

### 8.2 外部依赖

- CANN 9.0.0+（必需）
- torch_npu 2.10+
- triton-ascend 3.2.1+
- Mooncake（KV store 跨节点传输，A3 Dockerfile 已包含）
- HCCL（系统库）

### 8.3 团队协同

- 华为算子团队：MLA kernel 性能调优
- 内部 vllm-ascend 团队：经验复用、坑点规避
- TokenSpeed 上游：platform.py 改动 upstream PR

---

## 九、决策建议

**推荐路径：Phase 1+2 优先，做出 MVP 后再决定 Phase 3 投入。**

- Phase 1+2（5 周）即可达到 vllm-ascend 性能上限并集成 TokenSpeed C++ scheduler 优势
- Phase 3（4 周）是性能进一步逼近 B200 的关键，但 MLA 优化效益有不确定性
- Phase 4（2 周）是生产化必需，可与 Phase 3 部分并行

**ROI 测算**：
- 投入：约 11 人周
- 产出：A3 集群获得 ≥ 50% B200 性能的推理引擎，agentic 场景 TPS 显著超 vllm-ascend
- 战略价值：为昇腾客户提供 TokenSpeed 同款能力，覆盖国产化推理需求
