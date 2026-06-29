---
title: "DSpark 半自回归投机解码"
tldr: "DSpark = 并行骨干（一次 forward 出 γ=7 个 base logits）+ 轻量 Markov 顺序头（串行补 token 间依赖，每步只做 embedding lookup+投影+加法）+ 置信度头（c_k 单点阈值截断）+ hardware-aware 调度器（request-level 逐请求动态截断，未开源）"
date_created: 2026-06-28
date_modified: 2026-06-28
type: concept
tags: [speculative-decoding, dspark, semi-autoregressive, markov-head, confidence-scheduling]
sources: ["[[20260628-150200-dspark-analysis-投机解码-来源]]"]
explored: false
confidence: high
---

# DSpark 半自回归投机解码

## 解决的两大瓶颈

1. **并行 draft 缺 token 间依赖**：纯并行骨干（如 DFlash）每个位置只看 anchor，位置 k 不知道位置 k-1 猜了什么，导致 suffix acceptance decay
2. **盲目全长验证浪费 batch capacity**：高并发下把置信度低的 tail token 全送 target 验证是多余开销

## 半自回归的精髓

**并行骨干**（O(1) 延迟）产出 γ 份独立 base logits `U_k`，**Markov 顺序头**（O(γ) 但每步极廉价）把它们串成自回归序列：

```python
prev = anchor
for k in range(7):
    x_k = sample(U_k + W1[prev] @ W2)   # U_k 是骨干算好的，不重跑
    prev = x_k
```

每步只做 embedding lookup（W1，词表×256）+ 投影（W2，256×词表）+ 加法，不重跑 5 层骨干。

## 置信度头与截断

`c_k = sigmoid(w^T [h_k ; W1[x_{k-1}]])` - 一个 `Linear(2816→1)` 预测第 k 位的条件接受概率。

截断规则：从前往后，第一个 `c_k < threshold` 处砍尾，只送前缀到 target 验证。`threshold=0`（默认）不截断。

**注意**：累积生存概率 `a_j = Π c_k` 只在离线诊断用（ECE/AUROC/Brier），在线截断只看单点。

## 投机长度调度粒度

- **开源 eval**：强制 bsz=1，单请求串行，谈不上层级
- **论文生产（未开源）**：request-level，per-request `ℓ_r` + per-token `(r,j)` 二元组，按累积生存概率 `a_{r,j}` 降序贪心分配验证预算

这正是与 [[dynamic-speculation-length-概念]]（DSL）的分水岭：DSL 是 step-level 全局统一长度；DSpark 是 request-level 逐请求各自 `ℓ_r`。

## Ascend 移植的 P0 风险

1. Markov 串行循环在 torchair 图模式下需显式处理（防 recompile）
2. 双 KV cache 异节奏推进（draft KV 每轮 crop，target KV 按接受长度推进）——易错
3. request-level 变长 → 需变长 query kernel（flatten + marker tensor），是最大工作量

## 与相关概念的关联

- [[speculative-decoding-概念]] - DSpark 是 spec decode 的新型变体
- [[dflash-概念]] - DSpark 骨干改自 DFlash
- [[eagle-概念]] - DSpark vs Eagle3（accepted length +26.7~30.9%）
- [[dynamic-speculation-length-概念]] - DSL vs DSpark 调度粒度对比

## Counter-arguments

- Markov 顺序头增加了串行开销，γ 较大时可能成为瓶颈（RNN 头仅在 γ=15 时有优势）
- 硬件感知调度器（核心差异化）未开源，外部复现难度高
- confidence head 的训练目标（TV 距离→接受率软标签）能否泛化到训练分布外的模型/温度未验证

## Data gaps

- DSpark 在 MoE 大模型（V4/Qwen3.5-397B，而非 4B/8B/14B）的 accepted length 数据
- Ascend 移植后变长 kernel 的实际吞吐影响（与 GPU 的变长注意力开销差异未量化）
