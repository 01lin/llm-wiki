# DSpark 推理 e2e 深度展开：draft 预测 / 置信度计算 / 筛选判断逐行走读

> 本文是 [[DSpark_analysis]] 的**补充深度篇**，聚焦三件事：
> 1. **draft token 怎么被预测出来**（并行骨干 → Markov 串行修正）
> 2. **置信度 c_k 怎么算、累积生存概率 a_j 在哪里出现**
> 3. **筛选判断**（静态阈值截断 + rejection sampling 验收）
> 并把这三段缝进**一轮 e2e 完整数据流**。
>
> **所有引用均以本地一手源码 `/Users/linyi/code/Documents/code/DeepSpec/` 为准，行号 grep 实测。** 凡基于公式/论文的逻辑推演，文中标注「【推演】」与一手代码区分。原 `DSpark_analysis.md` 不动，本文为其 e2e 展开。

---

## 0. 速览：一轮推理的 7 个函数落点

```
generate_decoding_sample      base_evaluator.py:308   ← e2e 主循环
  └─ propose()                evaluator.py:99
       ├─ forward_dspark_draft_block   draft_ops.py:22   ← ① 并行骨干 single forward
       └─ build_dspark_proposal        draft_ops.py:96   ← ②③④ 采样+置信度+截断
            ├─ compute_logits          modeling.py:290   ← lm_head → base logits U_k
            ├─ sample_draft_tokens     modeling.py:310   ← Markov 串行采样 x_k
            ├─ _predict_confidence_logits  draft_ops.py:57 ← c_k
            └─ _confident_prefix_length    draft_ops.py:82 ← 单点阈值截断
  └─ verify_draft_tokens      base_evaluator.py:186  ← ⑤ rejection sampling 验收
```

一句话：**「一次并行前向出全部 base logits → Markov 把它串成自回归序列 → 置信度头给每位打分 → 单点阈值砍尾 → target 拒绝采样定生死」**。

---

## 1. draft token 预测：两段式（并行出料 + 串行修正）

DSpark 的反直觉点在于：**并行骨干一次前向就吐出整个 block 的 base logits，但 token 之间没有依赖**；真正注入「前一个 token 影响后一个」的依赖，是在采样阶段由 Markov 头**串行**补上的。两段分开看。

### 1.1 第一段 — 并行骨干 single forward（`forward_dspark_draft_block`，[draft_ops.py:22](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/draft_ops.py)）

```python
block_hidden = model._forward_backbone(
    target_hidden_states=target_hidden_states,          # target 5 层 hidden 拼接
    noise_embedding=model.embed_tokens(draft_input_ids), # [anchor, mask, mask, ...]
    position_ids=draft_position_ids,
    attention_mask=None,                                 # 推理走 SDPA 因果，非 FlexAttention
    past_key_values=past_key_values_draft,
    use_cache=True,
    is_causal=False,
)
past_key_values_draft.crop(start)                        # 关键：用完即裁，KV 不污染下一轮
```

`draft_input_ids` 的构造在 [evaluator.py:109](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/evaluator.py)：

```python
draft_input_ids = torch.full((B, max_proposal_tokens), mask_token_id, ...)
draft_input_ids[:, 0] = output_ids[:, start]   # 第 0 位 = anchor（target 刚确定的 token）
```

→ 输入形如 `[anchor, MASK, MASK, ..., MASK]`（共 block_size=7 个位置）。

`_forward_backbone`（[modeling.py:362](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/qwen3/modeling.py)）内部：
- `target_hidden_states = hidden_norm(fc(target_hidden_states))`（[modeling.py:374](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/qwen3/modeling.py)）：5 层 target hidden 经 `fc`（`5*d → d`，[modeling.py:241](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/qwen3/modeling.py)）压回单份，作为 K/V 的 context 来源。
- 每层 `Qwen3DSparkDecoderLayer`：Q 只来自 draft noise，K/V = `[target_context | draft_noise]` 拼接（这是 [[DSpark_analysis]] §2.2 已坐实的 KV 注入）。
- 输出 `block_hidden`：`[B, block_size, d]`，即 7 个位置各一份 hidden `h_1..h_7`。

**这一步产出的就是 base 表征。还没 token，只有 hidden。**

### 1.2 base logits U_k（`compute_logits`，build_dspark_proposal 内 [draft_ops.py:107](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/draft_ops.py)）

```python
proposal_hidden_states = block_hidden[:, :block_size, :]   # h_1..h_γ
base_draft_logits = model.compute_logits(proposal_hidden_states)  # = lm_head(h)  → U_1..U_γ
```

`compute_logits` 就是冻结共享的 `lm_head`（[modeling.py:290-291](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/qwen3/modeling.py)）。此刻每个位置 k 都有一份**只依赖 anchor、彼此独立**的 base logits `U_k`——这正是「并行 draft 缺 token 间依赖」的根源。

### 1.3 第二段 — Markov 串行采样把并行变自回归（`sample_block_tokens`，[markov_head.py:55](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/markov_head.py)）

这是把「7 份独立 logits」缝成「一条有依赖的 token 链」的核心循环：

```python
prev_token_ids = first_prev_token_ids.long()        # = anchor (draft_input_ids[:,0])
for step_idx in range(proposal_len):                 # γ=7 次
    step_logits = self.apply_step_logits(            # U_k + B_k
        base_logits[:, step_idx, :],
        token_ids=prev_token_ids,                    # ← 串行依赖：上一步采出的 token
        hidden_states=step_hidden,
    )
    next_token_ids = sample_tokens(step_logits.unsqueeze(1), temperature).squeeze(1)
    prev_token_ids = next_token_ids                  # ← 喂回下一步
```

其中 transition bias `B_k`（VanillaMarkov，[markov_head.py:26-32](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/markov_head.py)）：

```python
def compute_step_bias(self, token_ids, hidden_states):
    return self.project_bias(self.get_prev_embeddings(token_ids))
    #      = markov_w2( markov_w1[x_{k-1}] )   即 B = W1[x_{k-1}] @ W2  （低秩 r=256）
```

**关键解读**：
- `base_logits[:, step_idx, :]`（`U_k`）是并行算好的、固定的；循环里**不重新跑 backbone**，只做一次 `embedding lookup (W1) + 一次 r→V 投影 (W2) + 一次加法`。所以串行成本是 `O(γ)` 但每步极廉价（[[DSpark_analysis]] §0 的「轻量顺序头」）。
- 依赖链 `prev_token_ids = next_token_ids` 把第 k 步采出的真实 token 注入第 k+1 步的 bias —— 这就是缓解 suffix decay 的机制：靠后的位置不再只看 anchor，而是看到了前面**真采出来的** token。
- `temperature` 透传：贪心(0)走 argmax，否则按温度采样。两路都返回 `corrected_logits`（U_k+B_k 的结果），后续验收要用它当 draft 分布 q。

返回 `sampled_tokens [B,γ]` + `corrected_logits [B,γ,V]`。三种头变体（Vanilla/Gated/RNN）只是 `B_k` 的算法不同，循环骨架一致；生产默认 Vanilla（[[DSpark_analysis]] §2.3）。

> 【对照训练】训练侧不跑这个串行循环，走 `apply_block_logits`（[markov_head.py:43](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/markov_head.py)）teacher forcing：用 ground-truth 的 `prev token` 一次性并行加 bias。**训练并行、推理串行**是同一套 W1/W2 参数的两种用法。

---

## 2. 置信度计算：单点 c_k 与累积生存 a_j 的真实落点

这里有个**最容易被文档糊过去的边界**，必须讲清：开源代码里 c_k（单点）和 a_j（累积乘积）**出现在两个不同路径**，截断只用单点，累积乘积只在离线诊断里。

### 2.1 单点条件接受概率 c_k（`_predict_confidence_logits`，[draft_ops.py:57](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/draft_ops.py)）

```python
prev_token_ids = torch.cat([draft_input_ids[:, :1], sampled_tokens[:, :-1]], dim=1)
# = [anchor, x_1, x_2, ..., x_{γ-1}]   注意是右移一位：位置 k 的 c_k 依赖 x_{k-1}
confidence_pred = model.predict_confidence_step(proposal_hidden_states, prev_token_ids=prev_token_ids)
return confidence_pred.float().reshape(B, block_size, -1)[:, :, 0]   # [B, γ]
```

`predict_confidence_step`（[modeling.py:293](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/qwen3/modeling.py)）：

```python
prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids)  # W1[x_{k-1}]
features = torch.cat([hidden_states, prev_embeddings], dim=-1)          # [h_k ; W1[x_{k-1}]]
return self.confidence_head(features).float()                          # 输出 logit（未 sigmoid）
```

`AcceptRatePredictor`（[common.py:43-49](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/common.py)）就是**一个 `Linear(input_dim, 1)`**，`input_dim = hidden_size + markov_rank = d + 256`（[modeling.py:265-267](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/qwen3/modeling.py)，因 `confidence_head_with_markov=True`）。

→ `c_k 的 logit = w^T [h_k ; W1[x_{k-1}]]`，sigmoid 外置（在用的时候才 `.sigmoid()`）。

**语义（建模目标，§3 损失坐实）**：c_k = **「前 k-1 位全部被接受的前提下，第 k 位通过 target 验证的条件概率」**。是条件概率而非边际概率，这是后面能用连乘求生存概率的前提。

### 2.2 累积生存概率 a_j = Π c_i —— 只在离线诊断路径出现

[[DSpark_analysis]] §4.1 把 L2 列为「累积生存概率 + 静态阈值截断」，但**走读代码后要纠正一个精度**：

- **截断路径**（生产可用、`build_dspark_proposal`）**不算累积乘积**，只用**单点** `c_k < threshold`（见 §3.1）。
- **累积乘积 a_j = Π c_i 真正出现在离线诊断** `ConfidenceHeadRecorder.observe`（[confidence_head.py:366](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/confidence_head.py)）：

```python
step_probs = torch.sigmoid(confidence_logits[:, :effective_length]).squeeze(0)
cumprod_pred = step_probs.to(torch.float64).cumprod(dim=0)        # ← a_j = Π_{i≤j} c_i
prefix_label = verification.accept_prefix_mask[:, :effective_length].squeeze(0)  # 真实接受前缀
self.dataset_metrics.update(probs=cumprod_pred, targets=prefix_label)  # 算 ECE/AUROC/Brier
```

它把**预测的累积生存** `cumprod_pred` 与**实测的累积接受** `accept_prefix_mask`（rejection sampling 的 cumprod，见 §3.2）对齐，做逐位置校准评测（ECE/AUROC/Brier，[confidence_head.py:31](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/confidence_head.py)）。

> **结论修正**：开源里 a_j 的连乘是**校准诊断用**，不是在线截断决策用。在线截断用单点近似。论文 L3（贪心 `Θ=τ·SPS(B)`，[[DSpark_analysis]] §4.3）才把 a_j 连乘搬进调度决策——**那部分未开源**。这条边界是 [[code-grounded-no-speculation]] 的典型踩坑点，文档原 §4.2 写「L2 用单点近似」是对的，但「累积生存概率」标在 L2 容易让人误以为截断在连乘。

---

## 3. 筛选判断：两道闸门

draft 出来的 7 个 token 不是全送验证、也不是全被接受。两道独立的闸门：

### 3.1 闸门一：静态阈值截断（draft 侧自筛，`_confident_prefix_length`，[draft_ops.py:82](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/draft_ops.py)）

```python
def _confident_prefix_length(confidence_logits, *, block_size, threshold):
    if threshold <= 0.0:
        return block_size                              # threshold=0 → 不截断，全送
    below_threshold = confidence_logits.sigmoid() < threshold
    if not below_threshold[0].any():
        return block_size                              # 全部达标 → 全送
    return int(torch.nonzero(below_threshold[0])[0].item())  # 第一个掉线位置 = 前缀长度
```

**逻辑**：从前往后扫，**第一个** `sigmoid(c_k) < threshold` 的位置就是截断点，只保留它之前的前缀。这是 [[DSpark_analysis]] 的「verify smarter, not longer」——把大概率会被拒的尾巴提前砍掉，省 target 的 batch capacity。

在 `build_dspark_proposal`（[draft_ops.py:127](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/draft_ops.py)）调用，截断后只打包前缀：

```python
proposal_draft_tokens = _confident_prefix_length(confidence_logits, ...)
if proposal_draft_tokens == 0:
    return _empty_dspark_proposal(draft_input_ids)     # 整块都不自信 → 退化成纯自回归一步
verify_input_ids = cat([draft_input_ids[:,:1], sampled_tokens[:, :proposal_draft_tokens]], dim=1)
draft_probs = logits_to_probs(draft_logits[:, :proposal_draft_tokens, :], temperature)  # q 分布
```

注意 `threshold` 来自 `args.confidence_threshold`，约束 `0.0 ≤ threshold ≤ 1.0`（[evaluator.py:81](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/evaluator.py)）；**默认 0.0 即不截断**，截断是离线扫描诊断时才打开的旋钮。

### 3.2 闸门二：rejection sampling 验收（target 侧定生死，`verify_draft_tokens`，[base_evaluator.py:186](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/base_evaluator.py)）

target 一次前向验证 `[anchor, x_1..x_ℓ]`（`verify_length = ℓ+1`，[base_evaluator.py:215](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/base_evaluator.py)），然后标准投机采样的拒绝采样：

```python
proposed_tokens = proposal.verify_input_ids[:, 1:]                       # x_1..x_ℓ
selected_target_probs = gather_token_probs(target_probs[:, :-1, :], proposed_tokens)  # p(x_k)
selected_draft_probs  = gather_token_probs(proposal.draft_probs, proposed_tokens).clamp_min(1e-8)  # q(x_k)
accept_prob = torch.clamp(selected_target_probs / selected_draft_probs, max=1.0)      # min(1, p/q)
accept_mask = (torch.rand_like(accept_prob) < accept_prob).to(torch.int64)            # 抛硬币
accept_prefix_mask = accept_mask.cumprod(dim=1)                          # ← 一旦拒绝，后面全断
accepted_draft_tokens = int(accept_prefix_mask.sum(dim=1)[0].item())     # 接受前缀长度
```

**`cumprod` 是无损保证的关键**（[base_evaluator.py:257](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/base_evaluator.py)）：投机采样要求「接受是前缀性的」——第 j 位被拒，j 之后无论硬币结果如何全部作废。`cumprod` 把 `[1,1,0,1,1]` 变成 `[1,1,0,0,0]`，`sum` 即接受长度。

接受长度确定后：
- **有拒绝**（`accepted < draft_token_count`）：在拒绝位用 `sample_residual`（残差分布 `(p-q)_+` 归一化）补一个 token（[base_evaluator.py:280](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/base_evaluator.py)）——保证整体分布严格等于 target。
- **全接受**：用 target 在最后一位的分布直接采样下一个 bonus token（[base_evaluator.py:285](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/base_evaluator.py)）。

→ 每轮至少前进 1 个 token（bonus），最多前进 `ℓ+1` 个。EOS 命中则提前 `terminated_by_stop_token`（[base_evaluator.py:264-276](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/base_evaluator.py)）。

---

## 4. 一轮 e2e 完整数据流（缝合）

把上面三段按主循环 `generate_decoding_sample`（[base_evaluator.py:385](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/base_evaluator.py) 的 `while start < max_length`）串起来，单轮：

```
【prefill / 上一轮收尾】
  target 产出 anchor x_0 = output_ids[:, start]                       base_evaluator.py:355 / 411
  context.target_hidden_states = 5 层 target hidden 拼接              evaluator.py:93 / _update:139

【① 并行骨干 single forward】  propose → forward_dspark_draft_block   draft_ops.py:35
  输入 [x_0, MASK×(γ-1)] + target_context  →  block_hidden h_1..h_γ
  past_key_values_draft.crop(start)   # KV 用完即裁                    draft_ops.py:44

【② base logits】  U_k = lm_head(h_k)                                  draft_ops.py:107

【③ Markov 串行采样】  for k: x_k ~ softmax(U_k + W1[x_{k-1}]W2)       markov_head.py:76
  产出 sampled_tokens x_1..x_γ + corrected_logits（=draft 分布 q 的来源）

【④ 置信度 + 截断】
  c_k logit = w^T[h_k ; W1[x_{k-1}]]                                  modeling.py:306 / draft_ops.py:69
  ℓ = 第一个 sigmoid(c_k)<threshold 的位置（threshold=0 则 ℓ=γ）       draft_ops.py:90
  打包 verify_input_ids=[x_0, x_1..x_ℓ], draft_probs=q[:ℓ]           draft_ops.py:136

【⑤ target 验证 + 拒绝采样】  verify_draft_tokens                     base_evaluator.py:217
  target([x_0, x_1..x_ℓ]) → p 分布
  accept_prob = min(1, p/q);  accept_mask 抛硬币
  accept_prefix_mask = cumprod(accept_mask)   # 前缀性                base_evaluator.py:257
  accepted = sum(prefix_mask)
  next_token = sample_residual (有拒绝) / target argmax-sample (全接受) base_evaluator.py:280/285

【⑥ 提交 + 推进】
  output_ids[start : start+accepted+1] = 接受前缀                     base_evaluator.py:411
  output_ids[start+accepted+1]        = next_token (bonus)            base_evaluator.py:421
  start += accepted + 1;  target KV crop(start)                      base_evaluator.py:424-425
  update(context): 刷新 target_hidden_states 到接受前缀+1            base_evaluator.py:426 / evaluator.py:139

【⑦ 离线诊断（可选 hook）】  post_verify → recorder.observe          evaluator.py:157
  a_j = cumprod(sigmoid(c_k)) vs accept_prefix_mask → ECE/AUROC/Brier confidence_head.py:366
```

**每轮净产出 token 数 = accepted + 1**（接受前缀 + 一个 bonus/残差 token），这就是 acceptance length 指标的来源（`acceptance_lengths.append(accepted_draft_tokens + 1)`，[base_evaluator.py:423](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/base_evaluator.py)）。

---

## 4.5 投机长度变长吗？step-level 还是 request-level？

这是 DSpark 调度粒度的核心问题，**答案取决于看「开源实现」还是「论文生产调度」两个层面，两者结论不同**，必须拆开。

### 4.5.1 投机长度是变化的 —— 逐 step 变长

每个 step 的投机长度 `proposal_draft_tokens` 由置信度截断动态决定（[draft_ops.py:127](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/draft_ops.py)），第一个 `sigmoid(c_k) < threshold` 处砍尾（[draft_ops.py:90](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/draft_ops.py)），所以同一请求不同 step 的投机长度 ∈ `[0, γ]` 之间浮动 —— **确实变长**，不是固定 γ。截断为 0 时退化纯自回归（§5.3）。

### 4.5.2 step-level vs request-level：分两层

| 层面 | 答案 | 一手依据 |
|------|------|---------|
| **开源 eval 实现** | **谈不上层级**（bsz=1，单请求串行） | [base_evaluator.py:331](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/base_evaluator.py) `assert input_ids.size(0) == 1, "only bsz=1 is supported"`；[draft_ops.py:105](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/draft_ops.py) `requires batch_size=1`；[confidence_head.py:360](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/confidence_head.py) 注释亦确认 bsz=1 |
| **论文生产调度（Algorithm 1）** | **request-level，且细到 (request, token) 粒度** | [[DSpark_analysis]] §4.3 |

**开源侧**：整个 hardware-aware scheduler（Algorithm 1）**未开源**，eval 主循环强制 bsz=1 单请求串行。所以「一个 step 内多请求要不要对齐长度」这个矛盾在开源代码里根本不存在——只有一个请求自己逐 step 变长截断。**不要被开源代码误导成「DSpark 就是单请求变长」**，那只是离线诊断工具。

**论文侧**：DSpark 的核心卖点恰恰是**反对 step-level 统一长度**。Algorithm 1 输出 `ℓ_r ← j`（[[DSpark_analysis]] §4.3），下标 `r` 即 per-request——每个并发请求**各自**一条投机长度 `ℓ_r`，互不要求相同。而且决策候选是 `(r, j)` 二元组（请求 r 的第 j 位），全局按累积生存概率 `a_{r,j}` 降序贪心 admit：

```
E ← {(r,j) | a_{r,j}>0}, 按 a_{r,j} 降序             # 候选 = (请求, 位置) 二元组
for (r,j) in E:  ℓ_r ← j;  B←B+1;  τ*←τ*+a_{r,j}     # 逐请求逐位置分配验证预算
```

→ 既不是 step-level（不要求同 step 各请求等长），是 **request-level 的细粒度形态**：每个请求长度可不同，且按「每请求每位置的边际接受增益 `a_{r,j}`」动态分配。无损性靠 `a_{r,j}` 关于 j 单调非增 + early-stop 因果屏障保证（[[DSpark_analysis]] §4.3）。

### 4.5.3 与 DSL 的分水岭（对齐既有工作判断框架）

这正是 [[DSpark_analysis]] §7 那条对照的本质：

- **DSL（既有工作）**：按 concurrency 调**全局**单一 spec length（一个阈值管所有请求）→ **step-level / batch-level**。
- **DSpark 调度器**：per-request `ℓ_r` + per-token 实时 SPS → **request-level**。

**「step-level vs request-level」恰好是 DSL 与 DSpark 的分水岭**：DSpark 的创新点本身，就是把 DSL 的 step-level 全局长度，做成 request-level 逐请求逐 token 的动态长度。

### 4.5.4 代价：request-level 变长 → 撞变长 query kernel

request-level 各请求 `ℓ_r` 不同 = 送 target 验证的是变长 batch，需物理执行与逻辑序列解耦、token flatten + marker tensor 注入稀疏注意力（[[DSpark_analysis]] §4.4）。这是 DSpark 生产落地最大工作量项，也是 Ascend 移植 §6 把变长 kernel 标 **P0** 的原因。**结论闭环：变长是真的；开源单请求看不出层级；论文生产是 request-level（per-request ℓ_r + per-token 粒度）；代价是变长 kernel。**

> 这一节是 [[ascend-cluster-5to10x-architecture-spine]] 调度粒度决策的直接素材：要复现 DSpark 的 request-level 收益，调度器和 attention kernel 必须同时支持逐请求变长，二者缺一则退化回 DSL 式的 step-level 全局长度。

---

## 5. 三个容易踩的实现细节（走读补充，文档未展开）

1. **置信度的 prev_token 右移**（[draft_ops.py:65](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/draft_ops.py)）：c_k 用 `[anchor, x_1..x_{γ-1}]` 而非 `[x_1..x_γ]`。即位置 k 的接受概率依赖**它前一个**采出的 token，与 Markov bias `B_k` 用 `x_{k-1}` 同构——保证 c_k 与采样过程同源，是「条件接受概率」语义自洽的前提。

2. **draft KV 即用即裁**（[draft_ops.py:44](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/draft_ops.py) `crop(start)` / [base_evaluator.py:425](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/base_evaluator.py) target 侧 `crop(start)`）：draft backbone 的 KV 每轮算完立刻裁回 `start`，不跨轮累积——因为下一轮 anchor 变了，block 要重算。target KV 则按真实接受长度推进。两个 cache 推进节奏不同，移植时易错。

3. **截断为 0 的退化**（[draft_ops.py:133](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/draft_ops.py)）：若 `_confident_prefix_length` 返回 0（首位就不自信），`_empty_dspark_proposal` 让 `verify_input_ids = draft_input_ids[:,:1]`（只有 anchor），`draft_token_count=0` → verify 阶段走 `else` 分支直接 target 采一个 token（[base_evaluator.py:285](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/base_evaluator.py)）→ **退化为纯自回归**，仍保证前进 1 token，不死锁。

---

## 6. 与 [[DSpark_analysis]] 的衔接 / 修订

| 原文档位置 | 本文展开/修订 |
|-----------|--------------|
| §2.3 串行采样伪代码 | 补全 `sample_block_tokens` 真实循环（[markov_head.py:76](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/markov_head.py)）+ U_k 不重算、只加 bias 的成本结构 |
| §4.1 L2「累积生存概率 + 静态阈值截断」 | **修订**：在线截断只用单点 `c_k<threshold`；累积乘积 a_j 仅在离线诊断 `recorder.observe`（[confidence_head.py:366](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/confidence_head.py)）。L3 连乘进调度才是论文未开源部分 |
| §4.2 截断伪代码 | 补「截断为 0 退化纯自回归」「threshold 默认 0」两个边界 |
| §4.5 推理一轮数据流 | 本文 §4 给出带行号的完整缝合版 + 每轮净产出 = accepted+1 |
| §4.3 调度器 / §7 DSL 对照 | 本文 §4.5 新增：投机长度变长性 + step-level vs request-level 两层结论 + DSL 分水岭 + 变长 kernel 代价 |
| 置信度头 §2.4 | 补 prev_token 右移、input_dim=d+256、sigmoid 外置三个落点 |

> 移植 Ascend 时（[[DSpark_analysis]] §6），§5 的三个细节直接对应 P0 风险：串行循环（torchair 防 recompile）、双 KV cache 异节奏推进、截断退化分支的控制流——都需在图模式下显式处理。

---

## 7. 源码索引（可点击跳转 · 行号 grep 实测）

> 本地仓 `DeepSpec/` 在工作目录下，下列链接为 VS Code 跳转（点击直达对应代码行）。仓库根：`/Users/linyi/code/Documents/code/DeepSpec/`。

### 7.1 推理 e2e 主路径

| 环节 | 函数 / 关键行 | 跳转 |
|------|--------------|------|
| e2e 主循环 | `generate_decoding_sample` | [base_evaluator.py:308](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/base_evaluator.py) |
| bsz=1 约束 | `assert input_ids.size(0)==1` | [base_evaluator.py:331](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/base_evaluator.py) |
| propose 入口 | `Qwen3DSparkEvaluator._propose` | [evaluator.py:99](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/evaluator.py) |
| draft_input 构造 | `draft_input_ids[:,0]=anchor` | [evaluator.py:109](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/evaluator.py) |
| context 刷新 | `_update` | [evaluator.py:134](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/evaluator.py) |

### 7.2 ① 并行骨干 + ② base logits

| 环节 | 函数 / 关键行 | 跳转 |
|------|--------------|------|
| 并行骨干 single forward | `forward_dspark_draft_block` | [draft_ops.py:22](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/draft_ops.py) |
| draft KV 即用即裁 | `past_key_values_draft.crop(start)` | [draft_ops.py:44](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/draft_ops.py) |
| backbone 内核 | `_forward_backbone` | [modeling.py:362](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/qwen3/modeling.py) |
| base logits U_k | `compute_logits=lm_head` | [modeling.py:290](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/qwen3/modeling.py) |

### 7.3 ③ Markov 串行采样

| 环节 | 函数 / 关键行 | 跳转 |
|------|--------------|------|
| 串行采样循环 | `sample_block_tokens` for 循环 | [markov_head.py:55](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/markov_head.py) |
| transition bias B_k | `compute_step_bias = W2(W1[x])` | [markov_head.py:26](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/markov_head.py) |
| 训练 teacher-forcing | `apply_block_logits` | [markov_head.py:43](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/markov_head.py) |
| sample 派发 | `sample_draft_tokens` | [modeling.py:310](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/qwen3/modeling.py) |

### 7.4 ④ 置信度 + 截断

| 环节 | 函数 / 关键行 | 跳转 |
|------|--------------|------|
| c_k 预测(prev_token 右移) | `_predict_confidence_logits` | [draft_ops.py:57](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/draft_ops.py) |
| c_k feature 拼接 | `cat([h_k; W1[x_{k-1}]])` | [modeling.py:306](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/qwen3/modeling.py) |
| confidence head (Linear) | `AcceptRatePredictor` | [common.py:43](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/common.py) |
| 静态阈值截断 | `_confident_prefix_length` | [draft_ops.py:82](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/draft_ops.py) |
| proposal 打包 | `build_dspark_proposal` | [draft_ops.py:96](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/draft_ops.py) |

### 7.5 ⑤ 验收 + 离线诊断

| 环节 | 函数 / 关键行 | 跳转 |
|------|--------------|------|
| rejection sampling 验收 | `verify_draft_tokens` | [base_evaluator.py:186](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/base_evaluator.py) |
| 接受前缀(无损保证) | `accept_mask.cumprod(dim=1)` | [base_evaluator.py:257](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/base_evaluator.py) |
| 残差采样补 token | `sample_residual` | [base_evaluator.py:280](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/base_evaluator.py) |
| 累积生存 a_j(离线) | `cumprod_pred` ECE/AUROC/Brier | [confidence_head.py:366](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/confidence_head.py) |
| 诊断 hook | `ConfidenceHeadRecorder.observe` | [confidence_head.py:345](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/confidence_head.py) |

### 7.6 mask / 损失 / 配置

| 环节 | 函数 / 关键行 | 跳转 |
|------|--------------|------|
| 双向 block mask | `create_dspark_attention_mask` | [common.py:78](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/common.py) |
| 接受率计算(TV→accept) | `_compute_accept_rate_3d` | [loss.py:60](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/loss.py) |
| 接受率软标签 | `1-0.5*‖p^d-p^t‖_1` | [loss.py:69](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/loss.py) |
| Qwen3-4B 配置 | `block_size=7, markov_rank=256` | [dspark_qwen3_4b.py:11](file:///Users/linyi/code/Documents/code/DeepSpec/config/dspark/dspark_qwen3_4b.py) |
