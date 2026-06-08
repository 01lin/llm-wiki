---
title: "Source (paper): Nemotron 3 Super - Hybrid Mamba-Transformer MoE (2604.12374)"
tldr: "NVIDIA paper (arXiv 2604.12374, Apr 2026). Nemotron 3 Super: 120B total / 12B active hybrid Mamba-Attention MoE for agentic reasoning. First Nemotron 3 to (1) pretrain in NVFP4, (2) use LatentMoE (better accuracy per FLOP and per param), (3) include MTP layers for native speculative decoding. 25T-token pretrain + SFT + RL, up to 1M context, 2.2x / 7.5x higher inference throughput than GPT-OSS-120B / Qwen3.5-122B. Open-sourced."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [paper, nemotron, nvidia, moe, mamba, hybrid, mtp, nvfp4, agentic, quantization]
sources: []
original_url: "https://arxiv.org/abs/2604.12374"
explored: false
confidence: high
---

# Source (paper): Nemotron 3 Super

arXiv **2604.12374v1** (NVIDIA, 14 Apr 2026, 51 pages).

## What it is
**Nemotron 3 Super**: a **120B total / 12B active** hybrid **Mamba-Attention Mixture-of-Experts** model for agentic reasoning. Three firsts in the Nemotron 3 family: (1) **pre-trained in NVFP4** (stable low-precision pretrain), (2) **LatentMoE** (a new MoE architecture optimizing accuracy per FLOP *and* per parameter), (3) **MTP layers** for inference acceleration via native [[20260608-120000-speculative-decoding-概念|speculative decoding]] (see [[20260608-121100-mtp-multi-token-prediction-概念]]).

## Training & results
Pre-trained on **25T tokens** (2 phases: 20T broad coverage, 5T high-quality), then SFT + RL with heavy emphasis on agentic capabilities (scaled RL environments). Supports up to **1M context**. Comparable/better benchmark accuracy than GPT-OSS-120B and Qwen3.5-122B (IFBench, HMMT, SWE-Bench, HLE, Terminal-Bench Hard, Tau-Bench v2, RULER@1M) while achieving **up to 2.2x (vs GPT-OSS-120B) and 7.5x (vs Qwen3.5-122B) higher inference throughput** (8k in / 64k out). Base, post-trained, and quantized checkpoints open-sourced on HuggingFace.

Combines the MoE and hybrid-Mamba-Attention directions (cf. [[20260608-132100-gdn-kvcache-cooptimization-概念|hybrid GDN/Mamba models]] like Qwen3.5). Relates to [[20260412-194210-nvidia-实体]], [[20260412-194111-quantization-概念]] (NVFP4), [[20260608-130100-inference-engine-github-weekly-0409-0416-来源]] (lists Nemotron-3-Super in SGLang model coverage). RL post-training connects to [[20260412-194030-reinforcement-learning-概念]].

## Counter-arguments / Data gaps
- Vendor paper; throughput claims are NVIDIA-measured at a specific ISL/OSL setting.
- LatentMoE details and NVFP4 stability evidence are in the body (only first 4 of 51 pages read here).
- 7.5x vs Qwen3.5-122B is partly architecture (hybrid Mamba) + partly quantization (NVFP4 vs BF16) - not isolated.
