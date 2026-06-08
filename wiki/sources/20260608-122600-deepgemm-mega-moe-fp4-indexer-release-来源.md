---
title: "Source: DeepGEMM Public Release 26/04 - Mega MoE, FP4 Indexer (PR #304)"
tldr: "DeepSeek DeepGEMM release (PR #304 by LyricZhao, 2026-04-16): Mega MoE fuses dispatch/linear1/SwiGLU/linear2/combine into one mega-kernel overlapping NVLink comm with tensor-core compute (FP8xFP4, EP<=8, PyTorch>=2.9); plus FP4 Indexer (MQA logits) with larger MTP support, FP8xFP4 GEMM, PDL, faster JIT."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [deepgemm, deepseek, moe, fp4, quantization, kernel, mtp]
sources: []
original_url: "https://github.com/deepseek-ai/DeepGEMM/pull/304"
explored: false
confidence: high
---

# Source: DeepGEMM Release 26/04 - Mega MoE, FP4 Indexer

GitHub PR #304 on **deepseek-ai/DeepGEMM** by [[20260608-122900-lyriczhao-实体]] (published 2026-04-16). A kernel-library release (explicitly *not* a model release).

## New features
- **Mega MoE**: fuses & overlaps dispatch / linear1 / SwiGLU / linear2 / combine into a single **mega-kernel**, overlapping NVLink communication with tensor-core computation.
  - Only FP8 x FP4 MoE supported; only EP <= 8 tested; requires PyTorch >= 2.9. Perf numbers to be posted later. Still under development.
- **FP4 Indexer (MQA logits)** with **larger MTP support** - ties into [[20260608-121100-mtp-multi-token-prediction-概念]] / sparse-attention indexers.
- FP8 x FP4 GEMM, PDL, GEMM heuristics refactor, faster JIT, GEMM optimizations (swap A/B, faster MoE GEMM), DeepEPv2 MoE GEMM layout.

## Bug fixes
- JIT crash on distributed FS; some kernel hangs and IMA.

Relates to [[20260412-194111-quantization-概念]] (FP4/FP8), [[20260412-194210-deepseek-实体]], and DeepSeek-V4's FP4 indexer / sparse-attention path.

## Counter-arguments / Data gaps
- No performance numbers in the release ("posted later") - speedup claims are unquantified.
- FP8xFP4-only and EP<=8 limits mean it is not yet a general MoE path.
- Disclaimer states no relation to internal model releases.
