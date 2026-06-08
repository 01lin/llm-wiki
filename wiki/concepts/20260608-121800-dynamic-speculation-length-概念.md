---
title: "Dynamic Speculation Length (DSL, 动态投机长度)"
tldr: "Vary the number of speculated tokens K per step based on draft confidence: speculate more when confident, early-exit when not. vLLM PR #35301 / issue #36657 report up to 3.6x vs target. Orthogonal to entropy-adaptive rejection (EARS), so they stack. Open concern: longer K visibly inflates KV cache usage."
date_created: 2026-06-08
date_modified: 2026-06-08
type: concept
tags: [inference, speculative-decoding, mtp]
sources: ["[[20260608-120800-glm5-1-mtp-投机性能优化方案-来源]]", "[[20260608-120900-qwen3-5-mtp-ascend-适配分析-来源]]"]
explored: false
confidence: medium
---

# Dynamic Speculation Length (DSL)

**DSL** makes the speculation length K adaptive instead of fixed: when the drafter is confident, speculate more tokens; when confidence drops below a `draft_confidence_threshold`, exit early to avoid wasted verify cost. Tracked upstream in vLLM issue #36657 and PR #35301 (open draft), with reported gains up to **3.6x** vs target. It is **orthogonal** to entropy-adaptive rejection sampling (EARS), so the two stack.

A short raw note (`动态投机长度-分析`) flags an open question: **increasing K appears to noticeably increase KV cache occupancy**, which can interact badly with batch size and prefix-cache hit rate.

Caveat from [[20260608-120900-qwen3-5-mtp-ascend-适配分析-来源]]: the current vLLM implementation still has GPU->CPU sync overhead, and it has not yet been combined with Ascend-specific bottlenecks. It is a "medium-term" lever, after system-layer convergence.

## Data gaps
- Quantified cache-vs-K tradeoff is unmeasured in these notes.
- No Ascend validation of DSL yet.
