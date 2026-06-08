---
title: "Source: 大模型投机推理加速论文调研报告 (arXiv survey, Mar-Apr 2026)"
tldr: "Survey of 16 speculative-decoding papers (arXiv, 2026-03-20 to 04-13) grouped into 8 trends: relaxed verification, anisotropic trees, multi-drafter, self-speculation, diffusion-model SD, multimodal/Video-LLM, system-level, and image-generation SD."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [speculative-decoding, survey, research, inference]
sources: []
original_url: "https://export.arxiv.org/ (arXiv API search)"
explored: false
confidence: medium
---

# Source: 大模型投机推理加速论文调研报告

A self-compiled arXiv survey (research date 2026-04-15) covering **16 speculative-decoding papers** published 2026-03-20 to 2026-04-13. Classified into eight research directions.

> Note: two raw files hold this same survey - `20260415-022212-arxiv-speculative-decoding-survey-笔记.md` and `20260415-022302-大模型投机推理加速论文调研报告-笔记.md` (a clippings export of the same content). Treated as one source; flag for LINT dedup.

## The 16 papers by direction

**Core method / verification & trees**
- **SMART** (2604.09731) - hardware-aware tree expansion; big trees can give negative speedup. +20% MLLM, +15.4% LLM, training-free controller for MSD/EAGLE.
- **Goose** (2604.02047) - anisotropic trees; 1.9-4.3x lossless, +12-33% over balanced.
- **DIVERSED** (2604.07622) - dynamic ensemble verifier; relaxes strict distribution match.
- **Cactus** (2604.04987) - constrained-acceptance sampling with controllable divergence.
- **MetaSD** (2604.05417) - multi-drafter with alignment feedback, drafter selection as multi-armed bandit.
- **SpecMoE** (2604.10152) - self-assisted SD for MoE, no extra training, up to 4.30x on memory-bound systems.

**Diffusion / block-diffusion LLM**
- **S2D2** (2603.25702) - training-free self-speculation for diffusion LLMs (block-size=1 degenerates to AR), up to 4.7x.
- **DualDiffusion** (2604.05250) - SD for masked diffusion models.

**Multimodal / Video-LLM**
- **ParallelVLM** (2603.19610) - parallel SD, 3.36x on LLaVA-OneVision-72B.
- **LVSpec** (2604.05650) - loose SD for Video-LLMs, >99.8% target perf, +136% accept length.
- **Fast-dVLM** (2604.06832) - block-diffusion VLM, >6x with SGLang+FP8.

**Multi-token / image / AR-diffusion**
- **MARS** (2604.07023) - teach AR models to emit multiple tokens, 1.71x on Qwen2.5-7B, no extra params.
- **Drift-AR** (2603.28049) - entropy-informed SD + single-step visual decode, 3.8-5.5x.
- **SJD-VP** (2603.27115) - speculative Jacobi decoding with verification prediction (image gen).

**System / hardware**
- **ConfigSpec** (2604.09722) - edge-cloud config selection; no single config optimizes all objectives (goodput vs energy vs cost conflict; energy/cost converge at K=2).
- **A-IO** (2604.09752) - adaptive inference orchestration for memory-bound NPUs (Ascend 910B); "model scaling paradox"; fine-grained SD has core-sync overhead under graph compilation.

## Five trend insights
1. **Relaxation** becomes mainstream (strict match seen as unnecessary).
2. Trees move from **balanced to anisotropic**.
3. **Self-speculation** removes extra-model overhead.
4. SD spreads to **non-AR architectures** (MoE, diffusion, video, image).
5. **System view** matters - algorithmic speedup alone is insufficient.

Connects to [[20260608-120000-speculative-decoding-概念]], [[20260608-121700-speculative-tree-概念]].

## Counter-arguments
- Summaries are abstract-level only; claimed speedups (e.g. 6x, 4.7x) are author-reported, not independently verified.
- Many results are on small/mid models (7B-33B) or specific modalities; generalization to large MoE on NPU is unproven (echoed by A-IO's NPU caveats).

## Data gaps
- No code inspected. Several of these (Sequoia-lineage tree work) overlap with the 5 ingested PDFs that still need full reads.
- The survey is a snapshot of one month; older foundational SD papers (vanilla SD, EAGLE-1/2, Medusa) are not covered here.
