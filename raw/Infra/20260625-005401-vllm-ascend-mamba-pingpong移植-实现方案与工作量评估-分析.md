# 移植 SGLang Mamba Ping-Pong 到 vLLM/vllm-ascend：实现方案、工作量与可行性分析

> 生成时间：2026-06-25
> 走读版本：vllm `0d2961229` / vllm-ascend `8afdf356` / sglang `b5e0965b07`
> 目标：基于 vllm/vllm-ascend（model_runner_v1 入口）实现 SGLang 式 ping-pong（影子 state + verify 后按真实接受步提交），解决 hybrid 动态投机精度异常
> 纪律：基于代码走读，行号 grep 实测；不足处标 ❓。
> 前序：[[20260625-003658-vllm-ascend-hybrid动态投机精度异常-最终方案设计-分析]]

---

## 〇、关键前提：vLLM 已有基础设施，移植是"改造"不是"从零造"

> 这是工作量评估的核心结论——先讲清楚现状，避免高估。

### 现状（已坐实）

vLLM 主线**已有一套投机 mamba state 的拷贝基础设施**，但**机制与 SGLang 不同**：

| 组件 | vLLM 现状 | 代码位置 |
|------|----------|----------|
| 投机 state 上下文 | `MambaSpecDecodeGPUContext` | [mamba_utils.py:251](vllm/vllm/v1/worker/mamba_utils.py) |
| 拷贝描述符 | `MambaCopyBuffers`（src_ptrs/dst_ptrs/sizes） | mamba_utils.py:219 |
| buffer 容器 | `MambaBuffers.postprocess_align` | mamba_utils.py:542 |
| 提交 kernel | `postprocess_mamba_align_gpu` | mamba_utils.py（kernel） |
| 每模型 state 拷贝函数 | `gated_delta_net_state_copy_func` 等 | [mamba_utils.py:356](vllm/vllm/model_executor/layers/mamba/mamba_utils.py) |

**机制差异（gap 的本质）**：
- **vLLM 现状 = "持久 state 内 block 间 offset 重排"**：align kernel 把 state 从 `src_block` 按 `accept_token_bias` 偏移拷到 `dest_block`（mamba_utils.py:115-119），**没有独立影子 buffer**，靠 block_table 复用做"伪版本"。`MambaSpecDecodeGPUContext` 管理的全是元数据/staging buffer，**无一份 intermediate state tensor**（已坐实：字段全是 addrs/strides/staging）。
- **SGLang = "独立 intermediate(影子) state + 按 accept_lens-1 提交"**：投机写影子、持久 state 不动、verify 后提交真实接受步（spec_utils.py:577/616）。

> **判断**：vLLM 的精度风险正源于"block offset 重排"对 `num_draft+num_accepted+block_size` 对齐的脆弱（前篇存疑 A）。移植 ping-pong = **把"block offset 重排"替换为"影子 buffer + 真实接受步提交"**，复用现有的 staging/元数据/copy_func 框架，但**新增影子 state 存储 + 改提交语义**。

---

## 一、实现方案设计（基于 model_runner_v1 入口）

### 1.1 总体改造点（4 个插入点）

E2E 投机一个 step 在 model_runner_v1 的链路 + ping-pong 插入点：

```
_prepare_inputs (gpu_model_runner)
  └─ stage num_draft/num_computed/num_scheduled → GPU buffer   [现有]
execute_model
  ├─ 主模型 forward
  │   └─ GDN layer forward (qwen_gdn_linear_attn.py)
  │       ├─ 读 initial_state (has_initial_state, :1300)        ← 【插入点①】读"持久 state"做 initial
  │       └─ 写 state                                            ← 【插入点②】写"影子 buffer"而非持久
  ├─ drafter.propose (EagleProposer/MTP)  循环 num_spec 步
  │   └─ 每步 GDN forward 同样走影子                              ← 【插入点②】
  ├─ rejection_sampler → num_accepted                          [现有]
  └─ postprocess_mamba_align_gpu  (mamba_utils.py:1520)         ← 【插入点③】改为"按 accept_lens-1 从影子提交持久"
                                                                ← 【插入点④】block 跨界显式提交
```

### 1.2 各插入点的具体改造

**插入点①②：state 读写重定向到影子 buffer**

GDN forward 现在直接读写 `self_kv_cache`（持久 state，[qwen_gdn_linear_attn.py:1311-1313](vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py)）。改造：投机阶段的 state 写入重定向到影子 buffer（每个 draft step 一份），持久 state 只读不写。

参考 SGLang 的做法是 backend 层切 pool（frozen_kv view），vLLM 对应改 GDN forward 的 state tensor 来源：
```python
# qwen_gdn_linear_attn.py forward 内
# 现状：conv_state/ssm_state 直接指向 self_kv_cache 持久块
# 改造：投机阶段指向 shadow_state[draft_step]，initial_state 仍读持久块
conv_state = shadow_conv_state[draft_step] if in_spec_draft else self_kv_cache[0]
```

**插入点③：提交语义改造（核心）**

把 `postprocess_mamba_align_gpu`（mamba_utils.py:1520）的"block offset 重排"改为"从影子按 `num_accepted-1` 提交"：
```python
# 新 kernel / 改造现有 kernel：
# 现状：src_block → dest_block 的 offset 拷贝（依赖 block_size 对齐，脆弱）
# 改造：commit shadow_state[num_accepted-1] → persistent_state
for req_idx:
    k = num_accepted[req_idx]                  # 真实接受数（同步下 = output!=-1 sum, gpu_model_runner.py:1513）
    persistent_state[req_idx] = shadow_state[req_idx, k - 1]   # 取第 k 步影子提交
```
这步**复用现有 `num_accepted_tokens` staging**（mamba_utils.py:76 已有），只改"拷贝源"从 block-offset 变成 shadow-slot。

**插入点④：block 跨界显式提交**

移植 SGLang interval-crossing（spec_utils.py:631-650）：若 `seq_lens + num_accepted` 跨过 mamba block 边界，在跨界点额外提交一次 state。这替代 align kernel 里 `aligned_new_computed // block_size` 的隐式对齐（存疑 A 的脆弱点）。

### 1.3 影子 buffer 分配

在 `MambaSpecDecodeGPUContext`（mamba_utils.py:251）新增影子 state tensor：
```python
# 每个 mamba 层、每个 state type（conv/temporal）、每个 req、每个 draft step 一份
shadow_conv_state: torch.Tensor   # [num_layers, max_reqs, num_spec, conv_dim, conv_width]
shadow_ssm_state:  torch.Tensor   # [num_layers, max_reqs, num_spec, ...ssm_shape]
```
容量 = 持久 state 的 `num_spec` 倍（对标 SGLang `mamba_ping_pong_track_buffer`，decode.py:209）。

---

## 二、涉及模块/算子范围（明确边界）

| 模块 | 文件 | 改动类型 | 说明 |
|------|------|---------|------|
| **mamba state 上下文** | `vllm/v1/worker/mamba_utils.py` | 改造 + 新增字段 | `MambaSpecDecodeGPUContext` 加影子 buffer；`MambaBuffers` 分配 |
| **提交 kernel** | `vllm/v1/worker/mamba_utils.py`（Triton kernel） | 重写 | `postprocess_mamba_align_gpu` 从 block-offset 改 shadow-commit |
| **GDN forward** | `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` | 改造 | state 读写重定向影子（:1311-1313 区域） |
| **model_runner** | `vllm/v1/worker/gpu_model_runner.py` | 改造 | postprocess 调用点（:1520）传影子；staging 扩展 |
| **state copy func** | `vllm/model_executor/layers/mamba/mamba_utils.py` | 可能复用 | `gated_delta_net_state_copy_func`（:356）已有，提交时复用 |
| **配置** | `vllm/config/cache.py` | 新增 | mamba_cache_mode 加 "pingpong" 档 或复用 align |
| **vllm-ascend GDN 算子** | `vllm-ascend/csrc/attention/recurrent_gated_delta_rule/` | ❓ 需核对 | NPU state 提交路径是否复用主线 postprocess |
| **vllm-ascend model_runner** | `vllm-ascend/vllm_ascend/worker/model_runner_v1.py` | ❓ 可能复用 | 复用主线 SpecDecodeMetadata（:96），提交逻辑可能继承 |

### 算子层范围（关键判断）

- **GPU(Triton) 侧**：提交 kernel 在 `mamba_utils.py` 的 Triton kernel，**纯 Python/Triton，改动可控**。
- **vllm-ascend NPU 侧**：`recurrent_gated_delta_rule`（C++/AscendC 算子）是 **GDN 前向计算**算子，**不是 state 提交算子**。state 提交在 model_runner 的 postprocess（Python 层）。
  - ❓ **需坐实**：vllm-ascend 是否复用主线 `postprocess_mamba_align_gpu`（Triton，NPU 上能否跑），还是 NPU 另有 state 提交 C++ 算子。**这决定 NPU 侧是改 Python 还是改 AscendC 算子——工作量差异巨大。**

---

## 三、工作量评估

> 前提：vLLM 已有 staging/元数据/copy_func 框架，主要新增影子 buffer + 改提交语义。

| 工作项 | 范围 | 工作量 | 风险 |
|--------|------|--------|------|
| 影子 buffer 分配 | mamba_utils.py 加字段 + 分配 | 小（1-2d） | 显存预算（num_spec 倍 state） |
| 提交 kernel 重写 | Triton kernel 从 offset→shadow-commit | 中（3-5d） | kernel 正确性，需对拍 |
| GDN forward state 重定向 | qwen_gdn forward 改 state 来源 | 中（2-3d） | 不能破坏非投机/prefill 路径 |
| block 跨界提交 | 移植 interval-crossing | 中（2-3d） | 边界 case 多，易漏 |
| model_runner 串联 | gpu_model_runner postprocess 调用改造 | 小（1-2d） | — |
| 单测 + 对拍 | mtp3↔mtp1 切换 bit-exact | 中（3-5d） | mamba 递归，错一步必现 |
| **GPU 小计** | | **~2-3 周** | |
| **vllm-ascend NPU 适配** | 取决于 NPU 提交路径 | **❓ 1 周(复用Python) ~ 3-4 周(改AscendC)** | 见 §二 ❓ |

> **总工作量**：GPU 侧 **2-3 周**（1 人）；NPU 侧 **额外 1~4 周**，取决于是否需改 AscendC 算子（未坐实，最大不确定性）。

---

## 四、可行性分析

### 4.1 高可行的依据

1. **框架已存在**：`MambaSpecDecodeGPUContext` + `MambaCopyBuffers` + `gated_delta_net_state_copy_func` 都已有，不是从零建。
2. **提交点单一**：state 提交集中在 `postprocess_mamba_align_gpu` 一处（mamba_utils.py:1520），改造面收敛。
3. **有参照实现**：SGLang 的 ping-pong 是经验证的正确解，逻辑可对照移植。
4. **vllm-ascend 复用主线**：model_runner_v1 复用主线 SpecDecodeMetadata（:96/97），若提交在 Python 层，NPU 自动受益。

### 4.2 风险/不确定性（诚实标注）

1. **❓ NPU 提交路径未坐实**（最大风险）：若 vllm-ascend 的 mamba state 提交不走主线 Triton kernel 而是 AscendC 算子，NPU 工作量翻倍。**落地前必须先坐实这一条。**
2. **显存代价**：影子 buffer = num_spec 倍 state。GDN state 比 KV 小（固定大小），但层数多时仍需预算。mtp3 即 3 倍。
3. **cudagraph 兼容**：影子 buffer 寻址进 graph 捕获，需保证 shape 固定（按 max num_spec 预留，对标现有 decode_cudagraph_max_bs，mamba_utils.py:118）。
4. **不破坏现有路径**：prefill / 非投机 decode / mamba_cache_mode != align 的路径不能受影响——改造需加 mode 分支。

### 4.3 可行性结论

> **GPU 侧高度可行**（框架齐全、提交点单一、有参照），2-3 周可落地。**NPU 侧可行性取决于一个未坐实的关键事实**（提交路径在 Python 还是 AscendC）——这是落地前必须先 close 的前置项。

---

## 五、落地路径建议（闭环）

1. **先 close NPU 提交路径存疑**（§二 ❓）：走读 `recurrent_gated_delta_rule.cpp` + vllm-ascend model_runner 的 mamba postprocess 调用，确认 state 提交在 Python(Triton) 还是 AscendC——**这决定 NPU 工作量，是排期前置**。
2. **GPU 侧先行**：按 §1 改造，单测用 mtp3↔mtp1 切换 bit-exact 验收（mamba 递归错一步必现，是强验收）。
3. **NPU 侧跟进**：据步骤 1 的结论，复用 Python 提交 / 或改 AscendC 算子。
4. **对照退路**：若 ping-pong 工作量超预算，退回前篇阶段 1（修 align kernel 的 block 对齐）+ 阶段 2（断言设防）作为短期止血。

---

## 六、待核实/存疑

1. **❓ NPU mamba state 提交路径**（最高优先级）：vllm-ascend 是否复用主线 `postprocess_mamba_align_gpu`（Triton），还是另有 AscendC 提交算子 —— 决定 NPU 工作量。
2. **现有 align kernel 是否真用 block 复用做"伪影子"**：本文判断它是"持久 state 内 block-offset 重排"无独立影子，依据是 `MambaSpecDecodeGPUContext` 无 intermediate state 字段；但 kernel 是否借 dest_block 形成了某种版本隔离，需逐行确认 kernel 的 src/dst block 是否物理不同。
3. **GDN state 影子化对 cudagraph 捕获的影响**：影子 buffer 进 graph 是否需要额外 padding/固定 shape 处理，未验证。
4. **mamba_cache_mode 是否需新增档位**：复用 "align" 改造 vs 新增 "pingpong" 档，影响配置兼容性，需定。
