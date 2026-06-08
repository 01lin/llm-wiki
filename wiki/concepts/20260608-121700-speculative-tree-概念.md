---
title: "Speculative Tree / Tree Attention"
tldr: "Instead of a linear draft chain, expand multiple candidate paths as a tree and verify many at once via tree attention, raising mean accepted length. 2026 research moves from balanced trees to anisotropic (SMART: big trees can give negative speedup; Goose: deep chains for good tokens, wide branches for weak ones). (stub)"
date_created: 2026-06-08
date_modified: 2026-06-08
type: concept
tags: [inference, speculative-decoding, tree-attention]
sources: ["[[20260608-120600-spec-decoding-arxiv-survey-2026-03-04-来源]]", "[[20260608-120800-glm5-1-mtp-投机性能优化方案-来源]]"]
explored: false
confidence: medium
---

# Speculative Tree / Tree Attention

Rather than a single linear draft chain (mean accepted length ~1.5-1.8 for GLM5.1 MTP3), **tree-based speculation** expands multiple candidate paths and verifies them together using **tree attention**, raising mean accepted length to 3-5+. Classic results: Sequoia (arXiv:2402.12374, DP-optimal trees, 9.5x) and OPT-Tree (TACL 2025, adaptive, mean accept length ~10).

2026 research refines tree shape (see [[20260608-120600-spec-decoding-arxiv-survey-2026-03-04-来源]]):
- **SMART** (2604.09731): a big tree can give *negative* wall-clock speedup under batch/hardware saturation; expand a node only if its marginal benefit/cost beats the current tree speedup. Training-free controller for MSD/EAGLE.
- **Goose** (2604.02047): **anisotropic** trees - high-quality tokens form deep chains, low-quality tokens fan into wide branches (1.9-4.3x, +12-33% over balanced trees).

vLLM already supports `tree_attn` in the EAGLE path. Tree MTP is listed as an exploratory long-term lever for GLM5.1.

## Data gaps
- Stub. No MTP-tree implementation upstream yet.
