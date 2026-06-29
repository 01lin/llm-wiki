# DSpark 投机解码深度分析

> 论文：《DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation》(DeepSeek-AI & 北大, 2026-06)
> 仓库：https://github.com/deepseek-ai/DeepSpec （MIT，含 DSpark / DFlash / Eagle3 三种 draft）
> 本文档用于 Claude Code 后续方案设计与 Ascend 移植分析的输入材料。代码引用基于仓库 `deepspec/` 实际实现。

---

## 0. 一句话定位

DSpark = **DFlash 并行骨干**（高吞吐 draft）+ **轻量顺序头**（注入 block 内 token 依赖，缓解 suffix decay）+ **置信度头**（估计逐位置接受概率）+ **硬件感知前缀调度器**（按系统负载动态截断验证长度）。

线上效果（vs MTP-1 基线，同吞吐）：V4-Flash 每用户生成速度 +60%~85%，V4-Pro +57%~78%。离线 accepted length vs Eagle3 +26.7%~30.9%，vs DFlash +16.3%~18.4%（Qwen3-4B/8B/14B）。

**重要边界**：开源代码实现到「静态阈值截断」（离线诊断），生产级「全局吞吐贪心调度 + ZOS 异步」只在论文描述，**未开源**（耦合 DeepSeek 内部 HAI-LLM 引擎）。

---

## 1. 性能与核心创新点速查

| 维度 | 内容 |
|------|------|
| 解决的两大瓶颈 | (1) 并行 draft 缺 token 间依赖 → suffix acceptance decay；(2) 盲目全长验证在高并发下浪费 batch capacity |
| 创新 1：半自回归 | 并行 backbone（O(1) draft 延迟）+ 轻量串行头（O(γ) 但极廉价）|
| 创新 2：置信度调度 | confidence head 估接受概率 + hardware-aware scheduler 动态截断 |
| 无损保证 | rejection sampling 保留 target 分布；调度器靠 early-stop / 异步因果屏障维持 non-anticipating 性质 |
| 关键反直觉发现 | 并行/半自回归 drafter 的 accepted length 常**高于**全自回归 Eagle3，因 position-1 容量优势（深网络）杠杆效应最大 |

---

## 2. 模型结构（对照代码）

### 2.1 整体模块

```
target model (frozen, 共享 embedding + lm_head)
  └─ 提取 target_layer_ids 的 hidden states  → context feature
DSpark draft:
  ├─ fc + hidden_norm        : 多层 target hidden 拼接投影
  ├─ N×DSparkDecoderLayer     : 并行骨干 (config num_draft_layers=5)
  ├─ lm_head (frozen 共享)     : base logits U_k
  ├─ markov_head              : 顺序头, 注入 transition bias B_k
  └─ confidence_head          : 接受概率 c_k
```

### 2.2 并行骨干：KV 注入 + 双向 block mask

文件 `deepspec/modeling/dspark/qwen3/modeling.py` — `Qwen3DSparkAttention.forward`：

```python
# Q 仅来自 draft noise embedding；K/V = [target context | draft noise] 拼接
k_ctx = self.k_proj(target_hidden_states); k_noise = self.k_proj(hidden_states)
v_ctx = self.v_proj(target_hidden_states); v_noise = self.v_proj(hidden_states)
k = torch.cat([k_ctx, k_noise], dim=1)   # [B, ctx_len + q_len, ...]
v = torch.cat([v_ctx, v_noise], dim=1)
```

注意力 mask（`common.py` — `create_dspark_attention_mask`，FlexAttention mask_mod）：
- context 侧：draft 只看 anchor **之前**的 token（`kv_idx < anchor_pos`，单向因果）
- draft 侧：同 block 内**双向**（`q_block_id == kv_block_id`）
- 跨 block 完全隔离 → 并行训练多个 anchor block

> 论文对 DFlash 的小改：输入 `anchor + (γ-1) mask`，anchor 本身作为第一个预测位，输出 γ 个 logit（原 DFlash 是 anchor + γ mask 只取 mask 位）。节省一次预测计算。

### 2.3 顺序头三变体（`markov_head.py`）

因果块分布（论文 Eq.4）：
```
P(X|x0) = Π_k p_k(x_k | x0, x_<k)
p_k(v|·) = softmax_v( U_k(v) + B_k(x0, x_<k, v) )
```
U_k 来自并行骨干；B_k 是顺序头注入的 prefix-dependent transition bias。

**VanillaMarkov（论文默认 / 生产用）** — 一阶转移，低秩分解 B = W1 W2：
```python
self.markov_w1 = nn.Embedding(vocab_size, markov_rank)   # W1 ∈ R^{V×r}, r=256
self.markov_w2 = nn.Linear(markov_rank, vocab_size, bias=False)  # W2 ∈ R^{r×V}
# B(x_{k-1}, ·) = W1[x_{k-1}] @ W2   (论文 Eq.5)
```
推理串行采样循环（把并行 logit 变自回归的关键）：
```python
prev_token_ids = first_prev_token_ids  # anchor
for step_idx in range(proposal_len):
    step_logits = base_logits[:, step_idx, :] + W1[prev_token_ids] @ W2
    next_token_ids = sample_tokens(step_logits, temperature)
    prev_token_ids = next_token_ids     # 串行依赖
```

**GatedMarkovHead** — 用 backbone hidden 做门控：`B = (σ(W_g[h_k; W1[x_{k-1}]]) ⊙ W1[x_{k-1}]) @ W2`

**RNNHead** — GRU-like 状态跨 block 全前缀记忆（论文 Eq.6）：
```python
z = cat([state, prev_emb, h_k])
gate, candidate, output = joint_proj(z).chunk(3)
new_state = σ(gate)*state + (1-σ(gate))*tanh(candidate)
bias = W2 @ tanh(output)
```
实验：RNN 仅在长 γ(=15) 时比 Markov 多 ~5%；γ=7 几乎持平 → 复杂度/部署劣势，**默认 Vanilla**。

训练用 teacher forcing 的 `apply_block_logits`（并行加 bias，不展开串行），用 ground-truth prev token。

### 2.4 置信度头（`common.py: AcceptRatePredictor`）

```python
self.proj = nn.Linear(input_dim, 1)   # input_dim = hidden_size(+markov_rank)
# c_k = σ(w^T [h_k; W1[x_{k-1}]])      (论文 Eq.7)  —— 输出 logit, sigmoid 外置
```
建模**条件**接受概率：前面全接受的前提下位置 k 通过验证的概率。这是累积乘积成立的前提。

---

## 3. 损失函数（`loss.py`，论文 Eq.9-12）

三项加权（config 默认 α_ce=0.1, α_tv=0.9, α_conf=1.0）：

| 项 | 公式 | 代码 |
|----|------|------|
| CE | `-Σ w_k log p_k^d(x_k*)` | `F.cross_entropy(...) * weights` |
| TV (L1) | `Σ w_k ‖p_k^d - p_k^t‖_1` | `(draft_probs - target_probs).abs().sum(-1)` |
| Conf | BCE(c_k, c*_k), soft label | `binary_cross_entropy_with_logits(conf, accept_rate_3d)` |

接受率软标签 c*_k = 1 - 0.5·‖p^d - p^t‖_1（论文 Eq.8，TV 距离直接代理接受率）：
```python
accept_rate_3d = (1.0 - 0.5*(draft_probs - target_probs).abs().sum(-1)).clamp_(0,1)
```
位置衰减权重 w_k = exp(-(k-1)/γ_decay)（config `loss_decay_gamma=4.0`）：强化前缀早期位置（前缀验证下早期权重最高）。

---

## 4. 推理流程 + Hardware-Aware Prefix Scheduler（核心）

### 4.1 三层递进

```
L1 单点置信度 c_k                                     [开源 ✓]
L2 累积生存概率 a_j = Π_{i≤j} c_i + 静态阈值截断       [开源 ✓ 用单点近似]
L3 全局吞吐贪心 (Algorithm 1) + ZOS 异步 top-K        [仅论文, 未开源]
```

### 4.2 L1/L2 开源实现（`eval/dspark/draft_ops.py`）

```python
# c_k 预测：prev_token = [anchor, x_1..x_{γ-1}]
prev_token_ids = cat([draft_input_ids[:, :1], sampled_tokens[:, :-1]], dim=1)
confidence_pred = model.predict_confidence_step(proposal_hidden_states, prev_token_ids)

# 静态阈值截断：第一个 c_k < threshold 处截断
def _confident_prefix_length(confidence_logits, *, block_size, threshold):
    if threshold <= 0.0: return block_size
    below = confidence_logits.sigmoid() < threshold
    if not below[0].any(): return block_size
    return int(torch.nonzero(below[0])[0].item())
```
截断后 `build_dspark_proposal` 只打包前缀送验证 → "verify smarter, not longer"。
验证 `verify_draft_tokens`（`base_evaluator.py`）：rejection sampling + `accept_prefix_mask = accept_mask.cumprod(dim=1)`。

### 4.3 L3 生产调度器（论文 Algorithm 1）

**全局吞吐最大化建模**：R 个并发请求
- batch 总 token：`B = Σ_r (1 + ℓ_r)`
- 期望接受：`τ = Σ_r (1 + Σ_{j≤ℓ_r} a_{r,j})`
- SPS(B)：引擎初始化时 profile 的 steps-per-second 查找表（O(1) 查）
- 目标：`max Θ = τ · SPS(B)`

**贪心可解的关键**：a_{r,j} 关于 j 单调非增 → 把请求 r 从 j-1 扩到 j 的边际接受增益恰为 a_{r,j} → 全局按 a_{r,j} 降序贪心 admit 即尊重前缀依赖。

```
1: a_{r,j} ← Π_{i≤j} c_{r,i}                     # 累积生存概率
4: E ← {(r,j)|a>0}, 按 a 降序排序
5-6: ℓ_r←0; B←R; τ*←R; Θ_best←R·SPS(R)
7: for (r,j) in E (sorted):
8:    ℓ_r←j; B←B+1; τ*←τ*+a_{r,j}
9:    Θ ← τ*·SPS(B)
10:   if Θ > Θ_best: Θ_best←Θ; ℓ*_r←ℓ_r
12:   else: break                                # early-stop = 因果屏障，保无损
```
early-stop 保证 admit 决策只依赖已处理前缀，不泄漏未来 token（non-anticipating，论文 Appendix A 反例）。

### 4.4 生产落地的两大冲突与解法（论文 Sec.5.2-5.3）

| 冲突 | 解法 |
|------|------|
| SPS 曲线离散阶梯（非平滑单峰）| 去掉 early-stop break，做无约束全局搜索跨 cliff |
| 动态调度 vs ZOS（需提前知道 next batch size）| **异步两步延迟**：用 t-2 步 confidence 估容量 K（隐藏调度延迟）；当前步 token 仍按 t 步实时累积置信度排序 → 等价动态 top-K |
| 无损性（无约束搜索会泄漏未来）| 异步天然隔离：决策只用 t-2 历史 → **异步设计本身形成因果屏障**，既跨 cliff 又无损 |
| 变长 query kernel | 物理执行与逻辑序列解耦，token flatten 成独立元素，依赖经 marker tensor 注入稀疏注意力；V4 上**仅改 index-attention + compress kernel** |

### 4.5 推理一轮数据流

```
1. target 产 anchor x_0
2. 并行骨干 single forward → U_1..U_γ, h_1..h_γ      [_forward_backbone]
3. markov 串行采样: x_k ~ softmax(U_k + W1[x_{k-1}]W2) [sample_block_tokens]
4. confidence: c_k = σ(w^T[h_k; W1[x_{k-1}]])         [predict_confidence_step]
5. 累积生存 a_j = Π c_i
6a 离线: 静态阈值截断                                  [_confident_prefix_length]
6b 生产: 贪心 Θ=τ·SPS(B) + ZOS异步 top-K → ℓ_r        [Algorithm 1, 未开源]
7. target 验证前缀 (变长 kernel) + rejection sampling  [verify_draft_tokens]
```

---

## 5. 关键配置（`config/dspark/`）

```python
# dspark_qwen3_4b.py（8b 同构，仅 target 不同）
block_size=7              # γ=7
num_draft_layers=5        # 5 层 backbone（与 DFlash 同）
target_layer_ids=[1,9,17,25,33]   # Qwen3 36 层中取 5 层 context feature
markov_rank=256           # 低秩 r
markov_head_type='vanilla'
confidence_head_with_markov=True  # conf 输入拼 Markov embedding (input_dim=hidden+256)
loss_decay_gamma=4.0; ce_loss_alpha=0.1; l1_loss_alpha=0.9; confidence_head_alpha=1.0
# gemma4-12b: target_layer_ids=[5,17,29,41,46], mask_token_id=4
# 训练: lr=6e-4, warmup 0.04, bf16, global_batch 512, 10 epoch, torch_compile=True
# train 注意力实现: flex_attention (TRAIN_ATTN_IMPLEMENTATION)
```
生产部署（V4 论文 Sec.5.1）：backbone = 3 层 MoE + mHC + sliding window 128，block_size=5（V4 本体强，draft 要求更易满足）。

数据：Open-PerfectBlend 1.3M（chat 17.6% / math 39.4% / code 38.9% / IF 4.1%），仅用 prompt，response 由各 target 重新生成。target cache 默认 Qwen3-4B 约 **38TB**。

---

## 6. Ascend 移植要点（结合 vllm-ascend 上下文）

| 组件 | 移植要点 | 优先级 |
|------|---------|--------|
| FlexAttention mask_mod (训练) | Ascend FA 不完整支持 custom mask_mod，需转 4D mask / BlockDiagonal；或训练在 GPU、仅推理移植 | P0 |
| Markov 串行采样循环 | γ 次 embedding lookup + vocab 维加法；torchair 图模式下 for-loop 展开需防多余 recompile | P0 |
| 变长 query kernel | Ascend FA 变长支持有限，marker tensor 走 FIA 或自定义 AscendC，最大工作量 | P0 |
| SPS(B) 查找表 | 910B/A2/A3 重新 profile，cliff 位置/陡度与 H800 不同（HBM 带宽差异）| P1 |
| ZOS 两步延迟调度 | 对应 vllm-ascend zero-sync；t-2 历史预测正是绕开 DSL PR#35301 的 `.item()` GPU→CPU sync 的标准解法 | P1 |
| target hidden 跨 EP 通信 | 提取 5 层 hidden 需 AllGather，通信量 5×d×seq，与 MoE All2All 做流水线 overlap | P1 |
| 累积乘积 + top-K | 纯 AIV vector 算子，无难度 | P2 |
| confidence head | 单 Linear，可与 lm_head 融合到同 cube 批次 | P2 |

---

## 7. 与既有工作的关系（供方案对比）

- **DSL（你的工作）**：DSL 按 concurrency 调全局 spec length（阈值 ~12-16）；DSpark 调度器是 per-request/per-token 级 + 耦合实时 SPS，粒度更细，可看作 DSL 的硬件感知泛化。
- **MTP+suffix 混合（你的设计）**：同样拒绝"纯并行 vs 纯串行"二选一；DSpark 的 confidence head 比 early-exit masking 更精细（连续接受概率而非 0/1 mask）。
- **Eagle3 / DFlash**：DSpark backbone 直接复用 DFlash；相对 Eagle3 用 TTT，DSpark 用并行 + 轻量串行修正。
- **Domino (CausalEncoder) / DFlare**：并发工作，Domino 的 CausalEncoder 概念上类似 DSpark RNN head。

---

## 8. 已知 Limitation（论文自述）

draft 侧固定成本不可恢复：复杂 query（低接受率）仍需先并行生成完整 γ-block，这部分 draft 算力浪费。论文展望：difficulty-aware early-exit，让此类请求跳过全块生成。→ **这是你可切入的优化点**（与你的 draft vocab pruning / early-exit 思路契合）。

---

## 附：仓库文件索引（便于 Claude Code 定位）

```
deepspec/modeling/dspark/qwen3/modeling.py  : Qwen3DSparkModel/Attention/DecoderLayer
deepspec/modeling/dspark/gemma4/modeling.py : Gemma4 版
deepspec/modeling/dspark/markov_head.py     : VanillaMarkov / GatedMarkovHead / RNNHead
deepspec/modeling/dspark/common.py          : mask 构造 / AcceptRatePredictor / anchor 采样
deepspec/modeling/dspark/loss.py            : compute_dspark_loss (CE+TV+Conf)
deepspec/eval/dspark/draft_ops.py           : 推理 proposal 构造 + 静态阈值截断
deepspec/eval/dspark/confidence_head.py     : 离线 ECE/AUROC/Brier 校准评测
deepspec/eval/base_evaluator.py             : verify_draft_tokens (rejection sampling)
deepspec/trainer/dspark_trainer.py          : 训练循环
config/dspark/*.py                          : Qwen3-4B/8B/14B, Gemma4-12B 配置
DSpark_paper.pdf                            : 技术报告（Algorithm 1 / Sec.5 生产调度）
```
