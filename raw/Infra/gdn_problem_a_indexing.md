# 问题 A：Ascend GDN Recurrent Kernel 的 `ssm_state_indices` 索引错位

> 适用：`vllm_ascend` custom op `npu_recurrent_gated_delta_rule`  
> 文件：`vllm_ascend/_cann_ops_custom/.../ascendc/recurrent_gated_delta_rule/recurrent_gated_delta_rule.h`  
> 场景：MTP 投机解码 + `spec_state_indices_tensor` + 动态 uql（`1 + draft_len` 可变）

---

## 1. 问题一句话

Ascend recurrent kernel 对 flatten 后的 `ssm_state_indices` 使用了 **两套不一致的下标规则**：

- **读**初始 state：`GetValue(seq0 + num_accept - 1)`（batch token 累计空间 + accept 偏移）
- **写**回 state：`GetValue(seq_i)`（batch 内 **token 绝对下标**）

而 Python 传入的 flatten 表布局是 **`[num_spec_reqs × (num_spec+1)]` 按 req 分行**，不是「每个 batch token 一个 entry」。

当 **batch 内多个 spec req** 且 **本步 uql < num_spec+1**（动态投机典型）时，**写会落到错误 req 的 slot 行**，读也可能偏离 GPU 语义。  
这与 kernel 是否检查 `num_accept <= uql` **是独立问题**。

---

## 2. 背景：`ssm_state_indices` 从哪来

### 2.1 Python / Metadata

```python
# vllm/v1/attention/backends/gdn_attn.py
spec_state_indices_tensor = block_table_tensor[
    spec_sequence_masks_cpu, : self.num_spec + 1
]
# shape: [num_spec_decodes, num_spec + 1]
```

每行是一个 spec req 的 **num_spec+1 个物理 cache block id**（逻辑 slot 0..num_spec）。

### 2.2 传入 kernel

```python
# vllm_ascend/ops/gdn.py
core_attn_out_spec = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
    ...
    ssm_state_indices=spec_state_indices_tensor.flatten(),
    num_accepted_tokens=num_accepted_tokens.to(torch.int32),
)
```

flatten **按行优先**：req0 的 4 个 slot，再 req1 的 4 个 slot，…

```text
num_spec = 3 → 每 req 4 列

flatten 下标:  0   1   2   3 |  4   5   6   7 |  8   9  ...
              [r0_s0..r0_s3  |  r1_s0..r1_s3  |  r2_...]
```

宽度 **固定为 num_spec+1**，不随本步 draft 长度缩短（见 `dynamic_specdec_gdn_issues.md`）。

### 2.3 batch 内 token 布局（uql 可变）

```python
# gpu_model_runner._prepare_inputs
# query_start_loc / cu_num_tokens → 每 req 本步 token 数 = uql = 1 + draft_len
```

动态投机下，同一 batch 内各 req 的 uql 可能为 2 或 4；**token 在 flat batch 里紧挨排列**：

```text
req0: uql=2 → token 下标 0, 1
req1: uql=2 → token 下标 2, 3
req2: uql=4 → token 下标 4, 5, 6, 7
```

---

## 3. Ascend kernel 的两套索引（代码依据）

### 3.1 读初始 SSM state（每个 batch_i、每个 head 一次）

```cpp
// recurrent_gated_delta_rule.h :: Process()
int32_t seq0 = seq1;           // 本 req 在 batch token 序列中的起始
int32_t seqLen = ...;          // 本 req 本步 uql
seq1 += seqLen;

int32_t stateTokenIdx = seq0;
if (hasAcceptedTokens_) {
    int32_t acceptedTokenNum = numAcceptedTokensGm_.GetValue(batch_i);
    stateTokenIdx = seq0 + acceptedTokenNum - 1;
}
stateOffset = ssmStateIndicesGm_.GetValue(stateTokenIdx);
ProcessHead(seq0, seq1, head_i, stateOffset);
```

- `batch_i`：第几个 spec req（0, 1, 2, …）
- `seq0`：**累计 token 偏移**（前面所有 req 的 uql 之和）
- `stateTokenIdx`：用来 **查 flatten 表** 的下标

### 3.2 写回 SSM state（每个 token、每个 head）

```cpp
// recurrent_gated_delta_rule.h :: ProcessHead()
for (uint64_t seq_i = seq0; seq_i < seq1; seq_i++) {
    ...
    curStateOutOffset =
        ((ssmStateIndicesGm_.GetValue(seq_i) * NV_ + head_i) * realV_ + v_i) * realK_;
    ...
    CopyOutState(curStateOutOffset, ...);
}
```

- `seq_i`：从 `seq0` 到 `seq1-1`，即 **batch 内 token 绝对下标**（0, 1, 2, 3, …）
- **直接用 `seq_i` 作为 flatten 表下标** 取物理 block id

### 3.3 对比：同一 req 内读/写用的下标

| 操作 | 公式 | 本 req uql=2, seq0=0 时用的 flatten 下标 |
|------|------|------------------------------------------|
| 写 token0 | `GetValue(seq0+0)` = `GetValue(0)` | 0 |
| 写 token1 | `GetValue(seq0+1)` = `GetValue(1)` | 1 |
| 读 num_accept=2 | `GetValue(seq0+2-1)` = `GetValue(1)` | 1 |
| 读 num_accept=4 | `GetValue(seq0+4-1)` = `GetValue(3)` | 3 |

**单 req、uql=2**：写用 0,1；读最多到 3 —— 仍在 **同一行 r0** 的 slot 0..3 内，**自洽**。

---

## 4. 何时对齐、何时错位

### 4.1 对齐条件（设计隐含假设）

flatten 下标与 token 绝对下标一致，当且仅当：

```text
对每个 spec req i：
  在 batch token 序列中占用的长度 == num_spec + 1
  且前面所有 req 占用的 token 数之和 == i * (num_spec + 1)
```

即：**每个 req 本步 uql 恒等于 num_spec+1**（静态投机 batch 典型情况）。

**例：2 req，uql 均为 4，num_spec=3**

```text
token:     [0,1,2,3 | 4,5,6,7]
flatten:   [r0:0-3   | r1:4-7]

req0 写 GetValue(0..3)  → r0 slot 0..3  ✓
req1 写 GetValue(4..7)  → r1 slot 0..3  ✓
req1 读 seq0=4, num_accept=4 → GetValue(7) → r1 slot 3  ✓
```

此时 **问题 A 不暴露**，与 `num_accept <= uql` 检查无关。

### 4.2 错位：多 req + 动态 uql=2（batch 4→8 典型）

**2 req，uql 均为 2，num_spec=3**

```text
token 下标:     0, 1  |  2, 3
flatten 语义:  r0_s0..3 | r1_s0..3
flatten 下标:  0  1  2  3 | 4  5  6  7
```

| req | 写用的 GetValue | 实际指向 | 应指向 |
|-----|-----------------|----------|--------|
| req0 | 0, 1 | r0_s0, r0_s1 | r0_s0, r0_s1 ✓ |
| req1 | **2, 3** | **r0_s2, r0_s3** | **r1_s0, r1_s1（下标 4, 5）** ✗ |

**req1 的 SSM 更新写进了 req0 的 slot 2、3**，req1 自身 slot 0、1 未更新。

读（num_accept=2）：

```text
req1: seq0=2, stateTokenIdx = 2+1 = 3 → GetValue(3) = r0_s3  ✗（不是 r1 的任何 slot）
```

读（num_accept=4，若放宽 > seqLen 检查）：

```text
req1: stateTokenIdx = 2+3 = 5 → GetValue(5) = r1_s1
```

落在 r1 行，但是 **slot 1**，不是 GPU 语义下的 **slot 3（commit 点）**。

### 4.3 单 req 动态 uql=2 —— 无问题 A

仅 1 个 spec req 时，token 下标 0,1 与 flatten 下标 0,1 同属 r0 一行，**写/read 都在 r0 内**，不跨 req。

batch 4→8 压测若 **并发 req 多**，问题 A 才会显现；单 req 可能 **只复现 num_accept vs uql（return/clamp）问题，看不到 A**。

---

## 5. 与 GPU Triton 路径对比

GPU `fused_recurrent.py` 使用 **req 行 + req 内 slot 列**，与 flatten 布局一致：

**读：**

```python
i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1   # slot 列
state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t)  # 行 i_n
```

**写：**

```python
final_state_idx = tl.load(
    ssm_state_indices + i_n * stride_indices_seq + i_t   # i_t = 0..T-1，req 内局部 token
)
```

| | GPU | Ascend recurrent |
|--|-----|------------------|
| 读 | `i_n * (num_spec+1) + (num_accept-1)` | `seq0 + num_accept - 1` |
| 写 | `i_n * (num_spec+1) + local_t` | **`seq_i`（全局 token 下标）** |
| uql=4 多 req | 行/列对齐 | 与 GPU 等价 |
| uql=2 多 req | 行/列对齐 | **写错行** |

Ascend 更合理的写法应对齐 GPU：

```cpp
// 伪代码
int32_t row = batch_i;
int32_t local_t = seq_i - seq0;                    // 0 .. seqLen-1
int32_t idx_read  = row * (MAX_MTP + 1) + (num_accept - 1);
int32_t idx_write = row * (MAX_MTP + 1) + local_t;
GetValue(idx_read);   // 读
GetValue(idx_write);  // 写
```

当前实现 **未** 使用 `batch_i` 参与 `GetValue(seq_i)`，是问题 A 的根因。

---

## 6. 与 `num_accept <= uql` 检查的关系

### 6.1 该检查管什么

```cpp
if (acceptedTokenNum <= 0 || acceptedTokenNum > seqLen) {
    return;
}
```

- 约束：**读** 时 `seq0 + num_accept - 1` 是否被允许（相对 **本步 uql**）
- **不修改** 写路径 `GetValue(seq_i)`
- **不保证** flatten 行与 token 步长对齐

### 6.2 常见误解

> 「只有 kernel 限制 num_accept <= uql 才不会有问题 A」

**错误。** 对照表：

| 场景 | num_accept≤uql | 问题 A |
|------|----------------|--------|
| 单 req，uql=2 | ✓ 可跑 | **无** |
| 多 req，uql=4（静态） | ✓ | **无** |
| 多 req，uql=2 | ✓ 可跑 | **有**（写 GetValue(2,3) 踩 r0） |
| 多 req，uql=2，num_accept>uql | return | **仍有**（若 return 前已写/或改检查后写仍错） |

问题 A 的触发条件是 **多 req + uql ≠ num_spec+1 的 token 步长与 flatten 行宽不一致**，不是 num_accept 与 uql 的大小关系。

### 6.3 两个独立问题

```text
问题 B（num_accept vs uql）:
  读起点相对「本步 uql 窗口」是否合法 → kernel return / clamp

问题 A（索引坐标系）:
  写/读 flatten 表时，token 下标 vs [req][slot] 行布局是否一致
  → 多 req + 动态短 uql 时写错 req 行
```

batch 4→8 可能 **同时** 遇到 B 和 A；只修 B（clamp 或改 kernel 上界）**不修 A**。

---

## 7. 后果

1. **req 间 state 污染**：req1 的 forward 更新 req0 的 slot 2、3（uql=2 时）。  
2. **req1 自身 slot 0、1  stale**：本步未写入，下一步可能从错误块读。  
3. **与 num_accept/clamp 叠加**：clamp 只改读侧 slot 选择，**不修复写错行**；精度/吞吐异常可能部分来自 A。  
4. **单 req 或全 uql=4 多 req**：现象不明显，易误判为「仅 num_accept 问题」。

---

## 8. 修复方向（概要）

| 方向 | 说明 |
|------|------|
| **改 kernel 索引** | 写/读均用 `batch_i * (num_spec+1) + local_slot`，与 GPU 一致 |
| **强制 uql = num_spec+1** | padding 到 4 token，丧失动态 draft 长度收益 |
| **batch 内仅 1 spec req** | 规避多 req，不解决通用场景 |
| **仅 clamp / 仅改 num_accept 检查** | 不解决 A |

推荐长期方案：**kernel 索引对齐 GPU** + 保留 `num_accept <= num_spec+1` 越界保护 + uql 变化时的 state 语义对齐（见 `dynamic_specdec_gdn_issues.md` 4.5 节）。

---

## 9. 代码索引

| 内容 | 路径 |
|------|------|
| Ascend 读/写索引 | `.../recurrent_gated_delta_rule/recurrent_gated_delta_rule.h`：`Process()` L181-189，`ProcessHead()` L470-476 |
| Python 调用 | `vllm_ascend/ops/gdn.py`：`npu_recurrent_gated_delta_rule` |
| flatten 来源 | `vllm/v1/attention/backends/gdn_attn.py`：`spec_state_indices_tensor` |
| GPU 参考实现 | `vllm/model_executor/layers/fla/ops/fused_recurrent.py` L105-116, L154-156 |
| token layout | `vllm/v1/worker/gpu_model_runner.py`：`_prepare_inputs` → `query_start_loc` |
| 动态 uql 背景 | `docs/dynamic_specdec_gdn_issues.md` |

---

## 10. 附录：数值 walkthrough（2 req，uql=2，num_spec=3）

```text
flatten: [B00,B01,B02,B03 | B10,B11,B12,B13]
          idx 0  1  2  3     4  5  6  7

batch_i=0: seq0=0, seqLen=2
  写 seq_i=0,1 → GetValue(0), GetValue(1) → B00,B01  OK

batch_i=1: seq0=2, seqLen=2
  写 seq_i=2,3 → GetValue(2), GetValue(3) → B02,B03  WRONG (应为 B10,B11 @ idx 4,5)
  读 num_accept=2: GetValue(2+1=3) → B03            WRONG (应为 B10 或 GPU 的 B11)
  读 num_accept=4: GetValue(2+3=5) → B11            在 r1 行但 slot 1，非 commit slot 3
```

---

*文档版本：vllm-latest Ascend GDN recurrent 索引分析。*
