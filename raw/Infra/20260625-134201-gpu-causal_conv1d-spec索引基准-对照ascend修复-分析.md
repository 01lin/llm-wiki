# GPU causal_conv1d Spec Decode 索引基准 —— 对照 Ascend conv1d 修复（③-A 深挖）

> 生成时间：2026-06-25
> 走读：本地 vllm `0d2961229` `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`
> 目的：把 GPU conv_state 在"多 req + 动态 uql + 投机"下的正确索引完整推导，作为 Ascend conv1d custom op 修复的精确对照基准（复核文档 ③-A 的 P0 风险）
> 纪律：行号 grep 实测；Ascend 侧服务器代码无法读，给"应对齐什么"基准 + 待核实点。
> 关联：[[20260625-131524-gdn问题A修复复核与遗漏精度bug排查-分析]]

---

## 〇、核心结论

1. **GPU conv1d 有两个 kernel**：主路径（causal_conv1d.py:13 起）和 **spec decode 专用路径（:755 起，带 `num_accepted_tokens`）**。投机走第二个。
2. **GPU conv1d 的行寻址用 `idx_seq`（req 索引）× stride，结构上不会犯 ssm 那种"全局 token 下标当行"的错** —— 因为它是 per-program 处理一个 req（`program_id` → `idx_seq`）。
3. **但 conv_state 有 ssm 没有的"滚动（rolling）"语义**：接受 k 个 token 后，conv 窗口要丢弃 k 个最老 history、追加 k 个 draft —— 靠 `conv_state_token_offset = num_accepted - 1` 实现（:850-851）。**Ascend conv1d 若没正确实现这个滚动偏移，会精度异常。**
4. **GPU conv1d 有 NULL_BLOCK 守卫**（:815-817），padding 块直接 return。

> **对 Ascend 的判断**：③-A 的风险**不在"行错位"**（conv1d 用 idx_seq 行寻址，结构正确），**而在三处可能遗漏**：① num_accepted 驱动的 conv 窗口滚动偏移；② 读/写用不同列（读 `conv_state_init`、写 `current_last_index`）；③ NULL_BLOCK 守卫。下面给精确基准。

---

## 一、GPU spec conv1d kernel 的完整索引（对照基准）

### 1.1 行寻址：用 req 索引 idx_seq（结构正确）

spec kernel（causal_conv1d.py:755 起）每个 program 处理一个 req，`idx_seq` = req 索引。conv_state_indices 寻址：
```python
# 读初始 state（:810-811）
conv_states_input_coord = tl.load(
    conv_state_indices_ptr + idx_seq * stride_state_indices + conv_state_init
)
# 写回 state（:922-923）
conv_states_offset = tl.load(
    conv_state_indices_ptr + idx_seq * stride_state_indices + current_last_index
)
```
**行偏移 = `idx_seq * stride_state_indices`** —— 每个 req 一行，`idx_seq` 是 req 号不是全局 token 下标。

> **对照 Ascend**：这正是 ssm recurrent kernel **缺失**而被问题 A 修复补上的"行偏移"。conv1d 的 GPU 实现**本来就有**。Ascend conv1d 若也按 req 寻址（program 处理一个 req），则不会有 ssm 那种错位——**但需确认 Ascend conv1d 是 per-req program 还是 per-token**。❓

### 1.2 列寻址：读与写用不同的列偏移（关键差异）

| 操作 | 列偏移 | 含义 | 行号 |
|------|--------|------|------|
| **读**初始 state | `conv_state_init`（APC 时 `initial_state_idx[idx_seq]`，否则 0） | 上一步留下的 state 块 | :810/803 |
| **写**回 state | `current_last_index`（APC 时 `block_idx_last_scheduled_token[idx_seq]`，否则 0） | 本步要写入的 state 块 | :922/804 |

非 APC（prefix caching 关）时，`conv_state_init = 0`、`current_last_index = 0`（:806-808）——读写同一块（slot 0）。

> **对照 Ascend**：conv_state 用的是 `spec_state_indices_tensor[:, 0]`（[qwen_gdn:1350](vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py)，**只传 slot 0 列**），即非 APC 路径读写都在 slot 0。**Ascend 若也只用 slot 0，则列偏移无错**；但 APC 开启时读/写列不同，需对齐。

### 1.3 conv 窗口滚动：num_accepted 驱动（conv 独有，ssm 无）

这是 conv_state 最容易被遗漏的精度点。kernel 注释（:838-849）直述：
```
forward 前: [history1, history2, ..., historyM]
forward 后: [history2, ..., historyM, draft1, ..., draftN]
接受 k 个:   [history(k+1), ..., historyM, draft1, ..., draftk]
```
实现（:850-851）：
```python
if IS_SPEC_DECODING:
    conv_state_token_offset = tl.load(num_accepted_tokens_ptr + idx_seq).to(tl.int64) - 1
else:
    conv_state_token_offset = 0
```
即 conv 窗口要按 `num_accepted-1` 做**token 维度的滚动偏移**——把接受的 draft token 滚进 conv 滑窗、丢弃等量最老 history。

> **对照 Ascend（最关键）**：conv_state 不是"选一个 slot"，而是"滑窗内容按接受数滚动"。**Ascend conv1d 若把 conv_state 当成和 ssm 一样的"按 num_accept 选块"，而没做窗口内的 token 滚动偏移，则 conv 滑窗内容错位 → 精度异常。** 这是 ③-A 最可能的真实 bug 形态——不是行错位，是**滚动语义缺失/错误**。

### 1.4 NULL_BLOCK 守卫

```python
if HAS_NULL_BLOCK:                              # :814
    if conv_states_input_coord == null_block_id:  # :815
        return                                   # :816 padding 块直接跳过
```
cudagraph padding（`fill_(NULL_BLOCK_ID)`）下，取到 padding 块直接 return，不读写。

> **对照 Ascend**：需确认 Ascend conv1d 有等价 NULL 守卫（与复核文档 ③-B 同源）。

---

## 二、Ascend conv1d 应对齐的精确基准（修复参照）

| 维度 | GPU 基准 | Ascend 需核对 |
|------|---------|--------------|
| **行寻址** | `idx_seq(req) * stride_state_indices`（:811） | program 是否 per-req？行偏移是否用 req 号而非全局 token？ |
| **读列** | `conv_state_init`（APC:`initial_state_idx`，否则 0） | 非 APC 是否用 slot 0；APC 是否对齐 |
| **写列** | `current_last_index`（APC:`block_idx_last`，否则 0） | 读写列是否区分（APC 下） |
| **窗口滚动（核心）** | `conv_state_token_offset = num_accepted - 1`（:851），滑窗内容滚动 | **是否实现了 num_accepted 驱动的 conv 窗口滚动**——最易漏 |
| **NULL 守卫** | `conv_states_input_coord == null_block_id → return`（:815） | 是否跳过 padding 块 |
| **varlen 处理** | `state_len/seqlen` 按 query_start_loc 修正（:820-826） | 动态 uql 下 seqlen 是否正确取 query_end-query_start |

---

## 三、综合判断：③-A 的真实风险形态

复核文档（[[20260625-131524-...]]）把 ③-A 描述为"可能和 ssm 一样行错位"。**深挖后修正**：

- **conv1d 的行寻址结构本就正确（用 idx_seq）**，所以**不太可能犯 ssm 那种"写错 req 行"的错**（前提：Ascend conv1d 也是 per-req program）。
- **真实高风险点是 conv 独有的「窗口滚动」**：`num_accepted-1` 的 token 偏移（§1.3）。conv_state 是滑动窗口，接受数变化要滚动窗口内容；动态投机 mtp3→mtp1 时 num_accepted 变化，**若 Ascend 没做或做错这个滚动，conv 滑窗就含错误的 history/draft 组合 → 精度异常**。
- 次高：**varlen seqlen 修正**（§二）——动态 uql 下 conv kernel 的 seqlen 必须按 query_start_loc 实际取，否则滑窗长度错。

> **结论**：③-A 大概率**不是"行错位 bug"（那是 ssm 的问题 A），而是"conv 窗口滚动 / varlen seqlen"的适配缺失**。这两个是 conv1d 独有、ssm 修复不会顺带解决的。**问题 A 文档完全没碰 conv1d，所以这块大概率是真实残留。**

---

## 四、建议的排查动作（给服务器侧）

1. **核对 Ascend conv1d custom op 是否 per-req program**：确认行偏移用 req 号（对齐 §1.1）。若是 per-token，则 conv 也有行错位（与 ssm 同病）。
2. **核对 conv 窗口滚动**（P0）：Ascend conv1d 是否按 `num_accepted-1` 做滑窗 token 滚动（§1.3）。**这是最可能的残留 bug，优先验。**
3. **核对 varlen seqlen**：动态 uql 下 conv 的 seqlen 是否按 query_start_loc 取（§二末行）。
4. **核对 NULL 守卫**（③-B 同源）。
5. **分离验证**：固定 conv_state（或单步关 conv 更新）压测，若精度恢复 → 坐实 conv1d 是残留 bug 源。

---

## 五、待核实/存疑（服务器侧 AscendC，本地无法直读）

1. **Ascend conv1d 是 per-req 还是 per-token program**（§1.1）：决定是否有行错位。
2. **Ascend conv 窗口滚动是否实现**（§1.3，最高优先级）：num_accepted 驱动的滑窗滚动，是 conv 独有难点。
3. **Ascend conv 读/写列是否区分**（§1.2，APC 路径）。
4. **Ascend conv NULL 守卫**（§1.4）。
5. **APC（prefix caching）是否在该部署开启**：决定 conv_state_init/current_last_index 是否非 0、读写列是否需区分——若 APC 关，则读写都 slot 0，简化很多。
