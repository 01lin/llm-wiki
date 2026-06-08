---
title: "Source (paper): Speculative Speculative Decoding (SSD / Saguaro) (2603.03251)"
tldr: "Stanford/Princeton/Together paper (arXiv 2603.03251, Mar 2026). Speculative decoding has a sequential dependence: verification must finish before the next speculation begins. SSD parallelizes them - while verification runs (on the target), the draft model (on separate hardware) PREDICTS likely verification outcomes and pre-speculates for all of them; if the real outcome is in the predicted set, tokens return immediately with zero draft overhead. Lossless. Saguaro is the optimized algorithm: ~30% faster than the strongest SD baselines, up to 5x vs AR."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [paper, speculative-decoding, ssd, parallel, inference, draft-model]
sources: []
original_url: "https://arxiv.org/abs/2603.03251"
explored: false
confidence: high
---

# Source (paper): Speculative Speculative Decoding (SSD / Saguaro)

arXiv **2603.03251v2** (Tanishq Kumar, Tri Dao, Avner May; Stanford / Princeton / Together AI, 22 Mar 2026, 34 pages; code at github.com/tanishqkumar/ssd).

## Idea
[[20260608-120000-speculative-decoding-概念|Speculative decoding (SD)]] removes the target's per-token sequentiality but introduces its *own* sequential dependence: the draft model must **wait** for verification to finish before the next speculation round. **Speculative Speculative Decoding (SSD)** parallelizes drafting and verification: while a verification is ongoing, the draft model **predicts the likely verification outcomes** and **pre-speculates for all of them in parallel**. If the actual outcome is in the predicted set, the pre-speculated tokens are returned immediately, **eliminating drafting overhead entirely**. Like SD, SSD is **lossless**; unlike SD, the draft model sits on **distinct hardware** from the target.

## Saguaro & results
Three challenges (anticipating the verification outcome - how many tokens accepted + the bonus token; preparing for multiple outcomes; hardware placement) are solved by **Saguaro**, an optimized SSD algorithm. Implementation is on average **~30% faster than the strongest SD baselines** and **up to 5x faster than autoregressive decoding** with open-source engines (Llama-3.1-70B, TP=4 H100s, batch 1, greedy, Llama-3.2-1B draft).

## Significance
The concrete realization of the "Speculative Speculative Decoding (SSD)" direction noted as vLLM issue #36037 in [[20260608-120800-glm5-1-mtp-投机性能优化方案-来源]]. Complements [[20260608-121800-dynamic-speculation-length-概念|dynamic speculation length]] and zero-bubble async speculative work; conceptually similar to the "predict & prepare" framing in [[20260608-124000-hybrid-speculation-mtp-suffix-概念|hybrid speculation]].

## Counter-arguments / Data gaps
- Requires a **separate device** for the draft model (1xH100 in the figure) - extra hardware, mainly a batch-1 low-latency win.
- Gains depend on accurately predicting verification outcomes; high-entropy targets shrink the benefit.
- Benchmarks are batch=1 greedy; behavior at high batch / sampling not captured (first 3 of 34 pages read).
