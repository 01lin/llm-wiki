# 问题 A 说明与 Kernel 索引修复文档

> 适用路径：`vllm-latest` + `vllm_ascend` AscendC custom op `RecurrentGatedDeltaRule`  
> 关联文档：[gdn_problem_a_indexing.md](./gdn_problem_a_indexing.md)（问题分析）、[dynamic_specdec_gdn_issues.md](./dynamic_specdec_gdn_issues.md)（动态投机背景）  
> 修复日期：2026-06-24

---

## 目录

1. [问题 A 是什么](#1-问题-a-是什么)
2. [根因：两套索引 vs 一种 flatten 布局](#2-根因两套索引-vs-一种-flatten-布局)
3. [触发条件与后果](#3-触发条件与后果)
4. [与 GPU 参考实现的对比](#4-与-gpu-参考实现的对比)
5. [修复方案总览](#5-修复方案总览)
6. [每一处代码修改（修改前 / 修改后 / 原理）](#6-每一处代码修改修改前--修改后--原理)
7. [修复后的数值 Walkthrough](#7-修复后的数值-walkthrough)
8. [未修改的部分与后续工作](#8-未修改的部分与后续工作)
9. [编译与验证](#9-编译与验证)

---

## 1. 问题 A 是什么

在 **MTP 投机解码 + GDN recurrent** 路径下，Ascend custom kernel `npu_recurrent_gated_delta_rule` 通过 flatten 后的 `ssm_state_indices` 查找每个 token 对应的 **物理 SSM cache block id**。

Python 侧传入的 flatten 表语义是 **按 req 分行、每行固定 `num_spec + 1` 列**（逻辑 slot 0..num_spec），与 GPU Triton `fused_recurrent.py` 一致：

```text
flatten 布局（num_spec=3 → 每 req 4 列）:

下标:  0   1   2   3 |  4   5   6   7 | ...
      [r0_s0..r0_s3  |  r1_s0..r1_s3  | ...]
```

而 batch 内 **token 布局**由本步 `uql = 1 + draft_len` 决定，动态投机下可以 **小于** `num_spec + 1`：

```text
2 个 spec req，本步 uql=2:

token 绝对下标:  0, 1  |  2, 3
req0 seq0=0      req1 seq0=2
```

**问题 A**：修复前 kernel 用 **batch 内 token 绝对下标** 去索引上述 **按 req 分行的 slot 表**，在「多 req + uql < num_spec+1」时，req1 的 token 2、3 会落到 **req0 行的 slot 2、3**，而不是 req1 行的 slot 0、1。

这与 `num_accept <= uql` 的 clamp/return（问题 B）是 **独立** 的：即使 num_accept 合法，写路径仍可能错行。

---

## 2. 根因：两套索引 vs 一种 flatten 布局

### 2.1 Python 如何构造 `ssm_state_indices`

```python
# vllm/v1/attention/backends/gdn_attn.py
spec_state_indices_tensor = block_table_tensor[
    spec_sequence_masks_cpu, : self.num_spec + 1
]  # shape: [num_spec_decodes, num_spec + 1]

# vllm_ascend/ops/gdn.py
ssm_state_indices=spec_state_indices_tensor.flatten(),  # 行优先 flatten
```

### 2.2 修复前 kernel 的两套规则

| 操作 | 修复前公式 | 使用的「坐标系」 |
|------|-----------|-----------------|
| **读**初始 state | `GetValue(seq0 + num_accept - 1)` | batch token 累计空间 |
| **写**回 state | `GetValue(seq_i)` | batch token 绝对下标 0,1,2,… |

读、写都调用 `GetValue(...)`，但传入的下标 **含义不一致**，且写路径 **未使用 `batch_i`** 参与 req 行偏移。

### 2.3 何时「看起来没问题」

当每个 spec req 本步 **uql == num_spec + 1**（静态投机 batch）时：

```text
token:   [0,1,2,3 | 4,5,6,7]
flatten: [r0:0-3   | r1:4-7]
```

此时 token 绝对下标与 flatten 下标 **碰巧对齐**，问题 A 不暴露。

### 2.4 典型错位场景（batch 4→8，uql 从 4 变 2）

```text
flatten: [B00,B01,B02,B03 | B10,B11,B12,B13]
下标:     0   1   2   3     4   5   6   7

req1: token 2,3 → 修复前写 GetValue(2), GetValue(3) → B02,B03（req0 的 slot 2,3）✗
                  修复后写 GetValue(4), GetValue(5) → B10,B11（req1 的 slot 0,1）✓
```

---

## 3. 触发条件与后果

### 3.1 触发条件

| 条件 | 是否触发问题 A |
|------|----------------|
| 单 spec req，任意 uql | **否**（token 与 r0 行内 slot 一致） |
| 多 spec req，uql = num_spec+1 | **否** |
| 多 spec req，uql < num_spec+1（动态 draft） | **是** |

### 3.2 后果

1. **req 间 state 污染**：req1 的 SSM 更新写入 req0 的 slot。
2. **req1 自身 slot stale**：本步未写入 req1 对应 slot，下一步可能从错误块读 state。
3. **与问题 B 叠加**：clamp 只改读侧 slot 选择，**不修写错行**；精度异常可能部分来自 A。

---

## 4. 与 GPU 参考实现的对比

GPU `vllm/model_executor/layers/fla/ops/fused_recurrent.py` 对 spec decode 使用 **req 行 + req 内 slot 列**：

**读：**

```python
i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1   # 列 = num_accept - 1
state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t)
```

**写：**

```python
final_state_idx = tl.load(
    ssm_state_indices + i_n * stride_indices_seq + i_t   # i_t = 0..T-1，req 内局部 token
)
```

其中 `stride_indices_seq = num_spec + 1`（2D tensor 的行 stride）。

| | GPU | Ascend（修复前） | Ascend（修复后） |
|--|-----|----------------|-----------------|
| 读 | `i_n * (num_spec+1) + (num_accept-1)` | `seq0 + num_accept - 1` | `batch_i * stride + (num_accept-1)` |
| 写 | `i_n * (num_spec+1) + local_t` | **`seq_i`（全局 token）** | `batch_i * stride + local_t` |
| 多 req + uql=2 | 正确 | **写错行** | 正确 |

---

## 5. 修复方案总览

### 5.1 统一索引公式

读写均通过同一 helper `GetSsmStateFlatIndex`：

```text
flat_idx = batch_i * ssmStateIndicesStride + local_slot
```

| 场景 | `local_slot` |
|------|----------------|
| 读 | `num_accept - 1` |
| 写 | `seq_i - seq0`（req 内 0..uql-1） |

`ssmStateIndicesStride` 在 host tiling 阶段计算：

```text
ssmStateIndicesStride = len(ssm_state_indices.flatten()) / batch_size
                      = num_spec + 1   （spec 路径）
```

### 5.2 修改文件清单

| 文件 | 作用 |
|------|------|
| `recurrent_gated_delta_rule_tiling_data.h` | 新增 tiling 字段 `ssmStateIndicesStride` |
| `recurrent_gated_delta_rule.h` | kernel：统一索引 helper + 读写改造 |
| `recurrent_gated_delta_rule_tiling.h` | host：`FillTilingShapeData` 增加 `ssmStateShape` 参数 |
| `recurrent_gated_delta_rule_tiling.cpp` | host：运行时计算 stride 并写入 tiling |

**两份物理路径（内容同步）：**

- Custom OPP：`vllm-latest/vllm_ascend/_cann_ops_custom/.../recurrent_gated_delta_rule/`
- vLLM-Ascend csrc：`huawei20.2/vllm-ascend/csrc/attention/recurrent_gated_delta_rule/`

**Python / metadata 无需修改**（仍 `spec_state_indices_tensor.flatten()`）。

### 5.3 兼容性

- **非 spec decode**（无 `num_accepted_tokens`）：`hasAcceptedTokens_ == false`，仍用全局 `seq_i`，行为不变。
- **旧 tiling binary**（`ssmStateIndicesStride == 0`）：回退到修复前 `seq0 + local_slot` / `seq_i` 逻辑。

---

## 6. 每一处代码修改（修改前 / 修改后 / 原理）

---

### 修改 1：Tiling 结构体新增 `ssmStateIndicesStride`

**文件：**

- `vllm_ascend/_cann_ops_custom/.../recurrent_gated_delta_rule_tiling_data.h`
- `huawei20.2/vllm-ascend/csrc/attention/recurrent_gated_delta_rule/op_kernel/recurrent_gated_delta_rule_tiling_data.h`

**修改前：**

```cpp
    uint32_t hasGama;
    uint32_t hasGamaK;
    uint32_t hasAcceptedTokens;
};
```

**修改后：**

```cpp
    uint32_t hasGama;
    uint32_t hasGamaK;
    uint32_t hasAcceptedTokens;
    // Per-req row width of flattened ssm_state_indices (= num_spec + 1 for spec decode).
    uint32_t ssmStateIndicesStride;
};
```

**原理：**  
Kernel 在 device 侧需要知道 flatten 表 **每 req 占几列**。该值无法在 kernel 内从 `B_` 和 `T_` 可靠推导（动态 uql 时 `T_/B_ ≠ num_spec+1`），因此由 host tiling 在 launch 前从 `ssm_state_indices` 张量 shape 写入 tiling GM。

---

### 修改 2：Kernel 构造函数读取 stride

**文件：** `recurrent_gated_delta_rule.h`（custom opp 与 csrc 各一份）

**修改前：**

```cpp
        hasGamaK_ = (tilingData->hasGamaK == 1);
        useAddFoldReduce_ = (RGDR_ENABLE_ADD_FOLD_REDUCE != 0);
        vStep_ = tilingData->vStep;
```

**修改后：**

```cpp
        hasGamaK_ = (tilingData->hasGamaK == 1);
        useAddFoldReduce_ = (RGDR_ENABLE_ADD_FOLD_REDUCE != 0);
        ssmStateIndicesStride_ = tilingData->ssmStateIndicesStride;
        vStep_ = tilingData->vStep;
```

**原理：**  
将 host 计算的 stride 缓存到 kernel 成员 `ssmStateIndicesStride_`，供 `Process` / `ProcessHead` 统一使用。

---

### 修改 3：新增成员变量 `ssmStateIndicesStride_`

**文件：** `recurrent_gated_delta_rule.h`

**修改前：**

```cpp
    bool hasGamaK_;
    bool useAddFoldReduce_;
    float gama_;
```

**修改后：**

```cpp
    bool hasGamaK_;
    bool useAddFoldReduce_;
    uint32_t ssmStateIndicesStride_;
    float gama_;
```

**原理：**  
与 tiling 字段对应的 runtime 缓存。

---

### 修改 4：新增统一索引 helper `GetSsmStateFlatIndex`

**文件：** `recurrent_gated_delta_rule.h`（`private:` 区，`CopyInQKV` 之前）

**修改前：**  
（无此函数；读写在不同位置各自计算下标。）

**修改后：**

```cpp
    // Spec decode: batch_i * stride + local_slot (aligned with GPU fused_recurrent).
    // Non-spec decode: global token index seq_i.
    __aicore__ inline int32_t GetSsmStateFlatIndex(uint64_t batch_i, int32_t seq0, int32_t seq_i,
                                                   int32_t local_slot) const
    {
        if (hasAcceptedTokens_ && ssmStateIndicesStride_ > 0) {
            (void)seq0;
            (void)seq_i;
            return static_cast<int32_t>(batch_i * ssmStateIndicesStride_ + local_slot);
        }
        if (hasAcceptedTokens_) {
            return seq0 + local_slot;
        }
        return seq_i;
    }
```

**原理：**

| 分支 | 行为 |
|------|------|
| spec + stride 已知 | **GPU 对齐**：`batch_i * stride + local_slot` |
| spec + stride=0（旧 tiling） | 回退：`seq0 + local_slot`（修复前读逻辑） |
| 非 spec | 每 token 一个 index：`seq_i`（decode 路径不变） |

这是本次修复的 **核心**：读写共用同一函数、同一 `local_slot` 语义。

---

### 修改 5：`Process()` 读初始 state 路径

**文件：** `recurrent_gated_delta_rule.h`，`Process()` 内 per-head 首次 `CopyInGamaBeta` 之前。

**修改前：**

```cpp
                if (copyFlag == 1) {
                    int32_t stateTokenIdx = seq0;
                    if (hasAcceptedTokens_) {
                        int32_t acceptedTokenNum = numAcceptedTokensGm_.GetValue(batch_i);
                        if (acceptedTokenNum <= 0 || acceptedTokenNum > seqLen) {
                            return;
                        }
                        stateTokenIdx = seq0 + acceptedTokenNum - 1;
                    }
                    stateOffset = ssmStateIndicesGm_.GetValue(stateTokenIdx);
                    CopyInGamaBeta(seq0, seq1);
                }
                ProcessHead(seq0, seq1, head_i, stateOffset);
```

**修改后：**

```cpp
                if (copyFlag == 1) {
                    int32_t localSlot = 0;
                    if (hasAcceptedTokens_) {
                        int32_t acceptedTokenNum = numAcceptedTokensGm_.GetValue(batch_i);
                        if (acceptedTokenNum <= 0) {
                            return;
                        }
                        if (ssmStateIndicesStride_ > 0) {
                            if (acceptedTokenNum > static_cast<int32_t>(ssmStateIndicesStride_)) {
                                return;
                            }
                        } else if (acceptedTokenNum > seqLen) {
                            return;
                        }
                        localSlot = acceptedTokenNum - 1;
                    }
                    int32_t stateFlatIdx = GetSsmStateFlatIndex(batch_i, seq0, seq0 + localSlot, localSlot);
                    stateOffset = ssmStateIndicesGm_.GetValue(stateFlatIdx);
                    CopyInGamaBeta(seq0, seq1);
                }
                ProcessHead(batch_i, seq0, seq1, head_i, stateOffset);
```

**原理：**

1. **读**时使用 `localSlot = num_accept - 1`，与 GPU `i_t` 一致。
2. **上界检查**：有 stride 时用 `num_accept <= stride`（允许 uql=2 时读 slot 3），不再用 `num_accept <= uql` 阻断合法 commit 点读取（与问题 B 部分解耦）。
3. `ProcessHead` 增加 `batch_i` 参数，供写路径使用同一 indexing 规则。

---

### 修改 6：`ProcessHead()` 写回 state 路径

**文件：** `recurrent_gated_delta_rule.h`

**6a. 函数签名**

**修改前：**

```cpp
    __aicore__ inline void ProcessHead(int32_t seq0, int32_t seq1, uint64_t head_i, uint64_t stateOffset)
```

**修改后：**

```cpp
    __aicore__ inline void ProcessHead(uint64_t batch_i, int32_t seq0, int32_t seq1, uint64_t head_i,
                                       uint64_t stateOffset)
```

**6b. 循环内写 state 索引**

**修改前：**

```cpp
            for (uint64_t seq_i = seq0; seq_i < seq1; seq_i++) {
                ...
                uint64_t attnOffset = (seq_i * NV_ + head_i) * realV_ + v_i;
                uint64_t curStateOutOffset =
                    ((ssmStateIndicesGm_.GetValue(seq_i) * NV_ + head_i) * realV_ + v_i) * realK_;
```

**修改后：**

```cpp
            for (uint64_t seq_i = seq0; seq_i < seq1; seq_i++) {
                ...
                uint64_t attnOffset = (seq_i * NV_ + head_i) * realV_ + v_i;
                int32_t localSlot = static_cast<int32_t>(seq_i - seq0);
                int32_t stateFlatIdx = GetSsmStateFlatIndex(batch_i, seq0, seq_i, localSlot);
                uint64_t curStateOutOffset =
                    ((ssmStateIndicesGm_.GetValue(stateFlatIdx) * NV_ + head_i) * realV_ + v_i) * realK_;
```

**原理：**

- **写**时 `localSlot = seq_i - seq0`，对应 GPU 循环变量 `i_t`（req 内第几个 token）。
- `GetValue(stateFlatIdx)` 取到正确的物理 block id，再换算 `finalState` GM 偏移（与修复前相同，仅 flat 下标计算方式改变）。
- `attnOffset` 仍用全局 `seq_i`（Q/K/V 在 batch token 序列中连续排列），**不受** 此修改影响。

---

### 修改 7：Host tiling 头文件函数签名

**文件：** `huawei20.2/vllm-ascend/csrc/attention/recurrent_gated_delta_rule/op_host/recurrent_gated_delta_rule_tiling.h`

**修改前：**

```cpp
    void FillTilingShapeData(const gert::Shape &queryShape, const gert::Shape &valueShape, const gert::Shape &stateShape,
                             const gert::Shape &cuSeqlensShape);
```

**修改后：**

```cpp
    void FillTilingShapeData(const gert::Shape &queryShape, const gert::Shape &valueShape, const gert::Shape &stateShape,
                             const gert::Shape &cuSeqlensShape, const gert::Shape &ssmStateShape);
```

**原理：**  
`FillTilingShapeData` 需要读取 `ssm_state_indices` 的 1D 长度以计算 stride。

---

### 修改 8：Host tiling 计算 stride

**文件：** `huawei20.2/vllm-ascend/csrc/attention/recurrent_gated_delta_rule/op_host/recurrent_gated_delta_rule_tiling.cpp`

**8a. `FillTilingShapeData` 实现**

**修改前：**

```cpp
void RecurrentGatedDeltaRuleTiling::FillTilingShapeData(const gert::Shape &queryShape, const gert::Shape &valueShape,
                                                         const gert::Shape &stateShape,
                                                         const gert::Shape &cuSeqlensShape)
{
    tilingData_.t = queryShape.GetDim(DIM_0);
    tilingData_.nk = queryShape.GetDim(DIM_1);
    tilingData_.dk = queryShape.GetDim(DIM_2);
    tilingData_.nv = valueShape.GetDim(DIM_1);
    tilingData_.dv = valueShape.GetDim(DIM_2);
    tilingData_.sBlockNum = stateShape.GetDim(DIM_0);
    tilingData_.b = cuSeqlensShape.GetDim(DIM_0) - 1;
}
```

**修改后：**

```cpp
void RecurrentGatedDeltaRuleTiling::FillTilingShapeData(const gert::Shape &queryShape, const gert::Shape &valueShape,
                                                         const gert::Shape &stateShape,
                                                         const gert::Shape &cuSeqlensShape,
                                                         const gert::Shape &ssmStateShape)
{
    tilingData_.t = queryShape.GetDim(DIM_0);
    tilingData_.nk = queryShape.GetDim(DIM_1);
    tilingData_.dk = queryShape.GetDim(DIM_2);
    tilingData_.nv = valueShape.GetDim(DIM_1);
    tilingData_.dv = valueShape.GetDim(DIM_2);
    tilingData_.sBlockNum = stateShape.GetDim(DIM_0);
    tilingData_.b = cuSeqlensShape.GetDim(DIM_0) - 1;
    if (tilingData_.b > 0 && ssmStateShape.GetDimNum() == SSM_STATE_INDICES_DIM_NUM) {
        const uint64_t indicesLen = static_cast<uint64_t>(ssmStateShape.GetDim(DIM_0));
        tilingData_.ssmStateIndicesStride = static_cast<uint32_t>(indicesLen / tilingData_.b);
    } else {
        tilingData_.ssmStateIndicesStride = 0;
    }
}
```

**8b. `RuleFillTilingShapeData` 调用处**

**修改前：**

```cpp
    const auto &cuSeqlensShape = context_->GetInputShape(CUSEQLENS_INDEX)->GetOriginShape();
    FillTilingShapeData(queryShape, valueShape, stateShape, cuSeqlensShape);
    return ge::GRAPH_SUCCESS;
```

**修改后：**

```cpp
    const auto &cuSeqlensShape = context_->GetInputShape(CUSEQLENS_INDEX)->GetOriginShape();
    const auto &ssmStateShape = context_->GetInputShape(SSM_STATE_INDICES_INDEX)->GetOriginShape();
    FillTilingShapeData(queryShape, valueShape, stateShape, cuSeqlensShape, ssmStateShape);
    return ge::GRAPH_SUCCESS;
```

**原理：**

```text
spec 路径: indices 长度 = num_spec_decodes * (num_spec + 1)
           stride = indices_len / B = num_spec + 1

单测路径（per-token arange）: indices 长度 = T = B * mtp
           stride = T / B = mtp（与均匀 seqLen 一致，行为与修复前等价）

非 spec decode: 通常不传 num_accepted_tokens；即使 stride 被写入，
                kernel 走 hasAcceptedTokens_==false 分支，仍用 seq_i
```

注意：`sBlockNum`（state 张量 dim0 = 物理 cache 块总数）**不是** indices stride，不能复用。

---

## 7. 修复后的数值 Walkthrough

**场景：** 2 req，`num_spec=3`，本步 `uql=2`，flatten = `[B00,B01,B02,B03 | B10,B11,B12,B13]`，`stride=4`。

### req0（batch_i=0, seq0=0, seqLen=2）

| 操作 | local_slot | flat_idx | 物理块 |
|------|------------|----------|--------|
| 写 token0 | 0 | 0*4+0=0 | B00 |
| 写 token1 | 1 | 0*4+1=1 | B01 |
| 读 num_accept=2 | 1 | 0*4+1=1 | B01 |

### req1（batch_i=1, seq0=2, seqLen=2）

| 操作 | local_slot | flat_idx | 物理块 | 修复前 flat_idx |
|------|------------|----------|--------|-----------------|
| 写 token2 | 0 | 1*4+0=**4** | B10 | 2 → B02 ✗ |
| 写 token3 | 1 | 1*4+1=**5** | B11 | 3 → B03 ✗ |
| 读 num_accept=2 | 1 | 1*4+1=**5** | B11 | 3 → B03 ✗ |
| 读 num_accept=4 | 3 | 1*4+3=**7** | B13 | 5 → B11（错 slot）✗ |

---

## 8. 未修改的部分与后续工作

### 8.1 本次未改动的代码

| 模块 | 说明 |
|------|------|
| `vllm_ascend/ops/gdn.py` | 仍 `spec_state_indices_tensor.flatten()`，stride 由 host 自动推导 |
| `gdn_attn.py` | metadata 构造不变 |
| `mamba_utils.py` preprocess | uql 变化时的 state 拷贝逻辑未动 |
| conv1d custom op | 独立索引路径，不在本次范围 |

### 8.2 问题 A 修复后不自动解决的事项（问题 B 等）

1. **uql 缩短时的 state 语义**：上步 commit 在 slot 3，本步 uql=2 只写 slot 0、1；若上步 state 未在 slot 0/1，可能需要 preprocess 对齐。
2. **stale num_accept**：仍依赖上层 clamp / 写回逻辑。
3. **conv1d 与 recurrent 一致性**：需单独验证。

---

## 9. 编译与验证

### 9.1 必须同时重建

1. **Custom OPP kernel**（`ASCEND_CUSTOM_OPP_PATH` → `_cann_ops_custom` 下的 `RecurrentGatedDeltaRule`）
2. **vllm-ascend csrc** 中的 `RecurrentGatedDeltaRule` op_host（tiling）+ op_kernel

仅重建 kernel 而 host tiling 仍为旧版时，`ssmStateIndicesStride == 0`，spec 多 req 会 **回退到修复前逻辑**。

### 9.2 建议验证

1. 单测：`huawei20.2/vllm-ascend/tests/e2e/.../test_recurrent_gated_delta_rule.py`（均匀 mtp，应与修复前一致）
2. 动态投机 batch 4→8 精度压测（多 req + uql=2）
3. 对比 `ignore_eos=false` 的 SQL 评测集

---

## 附录：修改前后索引对照表

| 项目 | 修复前 | 修复后 |
|------|--------|--------|
| 读 flat 下标 | `seq0 + num_accept - 1` | `batch_i * stride + (num_accept - 1)` |
| 写 flat 下标 | `seq_i` | `batch_i * stride + (seq_i - seq0)` |
| num_accept 上界（有 stride） | `<= seqLen (uql)` | `<= stride (num_spec+1)` |
| 与 GPU 对齐 | 否 | 是（spec 路径） |
| Python 改动 | — | 无 |

---

*文档版本：v1.0 — 对应 Ascend GDN recurrent 问题 A kernel 索引修复。*
