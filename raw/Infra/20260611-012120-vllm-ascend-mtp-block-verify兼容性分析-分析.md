# Block Verify + Entropy Verify：代码级分析与 MTP 默认 argmax 的兼容性

> 范围：vllm-ascend `vllm_ascend/sample/rejection_sampler.py` + `vllm_ascend/spec_decode/llm_base_proposer.py`
> 问题：当前 MTP 默认用 argmax 取 top-1 token（不输出 prob），能触发 Block Verify / Entropy Verify 吗？
> 结论：**不能。** 有双层阻断——① Block Verify 要求 `max_spec_len >= 3`，而 DSv4-Flash 默认 `k=1`；② 即使 k>=3，argmax 路径在 `all_greedy` 分支提前 return，从不进入 Block Verify 代码。
> 生成日期：2026-06-11

---

## 1. 控制流完整追踪

### 1.1 入口：rejection_sampler.py:RejectionSampler.forward (line 155-182)

```python
# line 155: target_logits 来自主模型 logits_processor
target_logits = apply_sampling_constraints(...)  # 全量 [num_tokens, vocab_size]

output_token_ids = rejection_sample(
    metadata.draft_token_ids,
    metadata.num_draft_tokens,
    metadata.max_spec_len,
    metadata.cu_num_draft_tokens,
    draft_probs,       # MTP: None! (constraint: argmax draft, no probs)
    target_logits,     # raw float32 logits from target model
    bonus_token_ids,
    sampling_metadata,
    ori_target_logits=raw_target_logits,  # 原始 logits（未经 processor/clamp），给 entropy verify 用
)
```

关键：`draft_probs` 对 MTP 为 `None`（草稿用 argmax 生成，无概率）；`target_logits` 是全量 logits 而非 prob——Block Verify 需要概率。

### 1.2 rejection_sample 内部（line 310-443）

```python
# line 351-352 — 前置条件检查
using_block_verify = max_spec_len >= 3 and bool(get_ascend_config().rejection_sampler_config.enable_block_verify)
using_entropy_verify = bool(get_ascend_config().rejection_sampler_config.enable_entropy_verify)

# line 368-371 — entropy verify 需要 ori_target_logits
if using_entropy_verify and ori_target_logits is not None:
    ori_target_probs = ori_target_logits.softmax(dim=-1, dtype=torch.float32)
else:
    ori_target_probs = None
```

ori_target_logits 存在（调用方 pass 了），所以 Entropy Verify **数据上**可以工作——但还要看它走不走得到。

### 1.3 关键分叉（line 403-443）

```python
# 不是 all_random → 走 greedy/argmax 路径
if not sampling_metadata.all_random:
    target_argmax = target_logits.argmax(dim=-1)  # 只取 argmax!
    # ... 调用 rejection_greedy_sample_with_triton 或 pytorch 版本 ...
    if sampling_metadata.all_greedy:  # <— 所有请求都是 greedy
        return output_token_ids       # <— 直接 return! 不往下走
    # 只有部分请求不是 greedy 时，才 fallthrough 到后续 random 路径
```

**这就是你要的证据。** 当前 MTP 配置（temperature=0, argmax）下 `all_greedy=True`，代码在 **line 442-443 直接 return**，根本不进入后面包含 Block Verify（line 531-582）和 Entropy Verify 的分支。

---

## 2. Block Verify 的双层阻断

| 阻断层 | 代码位置 | 条件 | DSv4-Flash 默认值 | 结果 |
|--------|----------|------|--------------------|------|
| **第 1 层** | `line 352` | `max_spec_len >= 3` | `num_speculative_tokens = 1` → max_spec_len=1 | `using_block_verify = False` |
| **第 2 层** | `line 442-443` | `if all_greedy → return` | `temperature=0, argmax` → all_greedy=True | **提前 return，永不进入 Block Verify 分支** |

> **即使用 k=3+ 且打开 enable_block_verify，只要所有请求是 greedy (temperature=0)，第 2 层仍会阻断。** 因为 greedy 路径只做 token 比对（`target_argmax == draft_token_ids[k]`），不涉及概率——而 Block Verify 的核心是**累积概率乘积** `∏ P_target(token_k)`，必须在有 prob 的路径里才工作。

---

## 3. Block Verify 需要的概率数据

Block Verify 的 Triton kernel 签名（line 535-559）接收：

```python
target_probs,       # [num_tokens, vocab_size] — 需要全量概率！
draft_probs,        # [num_tokens, vocab_size] — 可选(None)
ori_target_probs,   # [num_tokens, vocab_size] — 原始概率，entropy verify 用
uniform_probs,      # [num_tokens, max_spec_len] — 均匀随机数
```

Block Verify 的数学本质是：对 `target_probs[token_k]` 的累积乘积做块级判断——这要求**每条草稿 token 都有对应的主模型概率**，而 argmax 路径只产出一个 int。

**当前 MTP 根本不生成这些 prob 数据**：
- draft 端：line 1069 `draft_token_ids = logits.argmax(dim=-1)` — 只用 argmax，不保留 logits 或 softmax
- target 端：line 408 `target_argmax = target_logits.argmax(dim=-1)` — 同上

---

## 4. Entropy Verify 的独立阻断

Entropy Verify 理论上可以在 greedy 路径里工作吗？看一下 `using_entropy_verify` 的使用位置：

- **只在 random 分支里传递**（line 502, 525, 554, 577, 644, 667, 694）——全都是 `rejection_random_sample_*` 函数
- greedy 路径里的 `rejection_greedy_sample_*` 调用**完全不接收 ENTROPY_VERIFY 参数**（line 411, 425, 432）

所以即使 `enable_entropy_verify=True` 且传了 `ori_target_logits`，greedy 路径也不会用它——只有 random 路径才会把 `ENTROPY_VERIFY` 和 `POSTERIOR_THRESHOLD/ALPHA` 传给 kernel。

---

## 5. 总结：要启用 Block/Entropy Verify 需要改什么

| 需要修改 | 当前状态 | 目标状态 |
|----------|----------|----------|
| `num_speculative_tokens` | `1` | `>= 3`（Block Verify 硬限制） |
| `enable_block_verify` | `False` | `True` |
| `enable_entropy_verify` | `False` | `True`（如需） |
| 采样模式 | `temperature=0, all_greedy=True` | **至少部分请求用 `temperature > 0`，走 random 分支** |
| draft token logits | 不保留（argmax 后丢弃） | 草稿端保留 `compute_logits` 输出并 softmax 成 `draft_probs` |
| target token probs | 不保留（argmax 后丢弃） | 在 `all_random` 或混合模式下保留 `target_probs = softmax(logits)` |

**最关键的工程问题**：Block Verify 需要**概率数据**——而 MTP argmax 的整个设计哲学就是避免出概率（kv_cache 不存 probs、减少通信）。要改就是 trade-off：用概率换取更高接受率。

---

## 6. 为什么 MTP 默认设计成 argmax 不出 prob？

回顾 `llm_base_proposer.py` 的 MTP 草稿生成流程（line 1042-1069）：

```python
# enable_reduce_sample (默认 False) → 不进 reduce_sample 分支
# → 进 line 1064-1069：
logits = self.model.compute_logits(sample_hidden_states)
# ... lmhead TP 处理 ...
draft_token_ids = logits.argmax(dim=-1)  # 只取 argmax，logits 丢弃
```

这是有意为之——MTP 的目标是**轻量草稿**（单层，无 standalone model），reduce_sample 是额外开销。

**但这不等于永远不能开 Block Verify。** 正确的启用条件组合是：

```bash
--speculative-config '{"num_speculative_tokens": 3, "method": "mtp", ...}'
--additional-config '{"enable_reduce_sample": true, "rejection_sampler_config": {"enable_block_verify": true, "enable_entropy_verify": true, ...}}'
```

同时采样参数使用 `temperature > 0` 或 mixed greedy/random，让请求不全部走 all_greedy 快路径。

**代价是**：reduce_sample 拉 path 从 argmax→full logits softmax，增加计算量和通信；Block Verify 受限于 `max_spec_len >= 3` 所以至少要 k=3。

---

## 7. 建议与优先级

1. **优先改 k=1→3**：当前代码多步 MTP 已实现，只需改配置+验证接受率。k>1 时 max_spec_len >= 3 才能解锁 Block Verify 前置条件。
2. **确保有 random 采样请求**：纯 greedy 的 workload 永远不会进入 Block Verify 路径。需要 temperature>0 或 `top_p/top_k` 混合。
3. **enforce_eager→false + enable_reduce_sample→true**：草稿端出 logits、target 端保留 prob，这是 Block/Entropy Verify 的数据前置。
4. **实测接受率 vs 通信/计算开销的 trade-off**：reduce_sample 增加 logits 计算和 TP 通信（all_gather logits），Block Verify 减少拒绝→提高有效吞吐。拐点在哪里需要实测算。
