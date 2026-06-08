---
title: "Source (paper): How Transformers Learn to Plan via Multi-Token Prediction (2604.11912)"
tldr: "UCLA/SJTU/UPenn/RIKEN paper (arXiv 2604.11912, Apr 2026). Theory + experiments on why MTP beats next-token prediction (NTP) for reasoning/planning. MTP consistently outperforms NTP on graph path-finding, Countdown, SAT. On a 2-layer transformer / star-graph task, MTP provably induces a two-stage REVERSE reasoning process (attend to end node, then trace path backward) via a gradient-decoupling property that gives a cleaner training signal than NTP."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [paper, mtp, reasoning, planning, theory, training, transformers]
sources: []
original_url: "https://arxiv.org/abs/2604.11912"
explored: false
confidence: high
---

# Source (paper): How Transformers Learn to Plan via Multi-Token Prediction

arXiv **2604.11912v1** (Jianhao Huang, Zhanpeng Zhou et al., UCLA / SJTU / UPenn / RIKEN, 13 Apr 2026, 31 pages).

## Thesis
[[20260608-121100-mtp-multi-token-prediction-概念|Multi-token prediction (MTP)]] is usually framed as an *inference* speedup (drafting), but this paper studies it as a *training objective* and argues it produces better **reasoning/planning** than standard next-token prediction (NTP). NTP with teacher forcing captures local patterns but struggles with global structure and long-term dependencies.

## Findings
- **Empirical**: MTP consistently outperforms NTP on synthetic graph path-finding and realistic reasoning benchmarks (Countdown, boolean satisfiability).
- **Theoretical**: on a simplified two-layer Transformer / star-graph task, MTP provably induces a **two-stage reverse reasoning** process - the model first attends to the end/goal node, then reconstructs the path by tracing intermediate nodes backward. This turns hard forward searches (large branching) into easy backward steps.
- **Mechanism**: a **gradient-decoupling property** of MTP gives a cleaner training signal than NTP, biasing optimization toward robust, interpretable reasoning circuits. The MTP architecture used: shared backbone + multiple independent output heads (Gloeckle et al. 2024 formulation), as in DeepSeek-V3.

## Significance
Complements the engineering view of MTP (acceleration) with a **quality/reasoning** rationale - relevant to why models like DeepSeek-V3 adopt MTP and to the train-inference-gap discussion in [[20260608-120800-glm5-1-mtp-投机性能优化方案-来源]]. Relates to [[20260412-194030-reinforcement-learning-概念]] / reasoning.

## Counter-arguments / Data gaps
- Theory is on a simplified 2-layer transformer / star graph - extrapolation to large LLMs is suggestive, not proven.
- "Reverse reasoning" is shown on structured planning tasks; generality to open-ended reasoning is an open question.
- Only first 4 of 31 pages read; full benchmark suite and proofs not captured here.
