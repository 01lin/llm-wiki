---
title: "Source (paper): ESS - Offload-Centric Latent-Cache Management for DeepSeek-V3.2-Exp (2512.10576)"
tldr: "Baidu Baige AI paper (arXiv 2512.10576, Dec 2025). DeepSeek-V3.2's sparse attention cuts latency but the decode stage is bottlenecked by Latent-Cache growing linearly with context vs fixed GPU memory, capping batch size. ESS offloads Latent-Cache to CPU (exploiting Top-2K temporal locality + a GPU sparse memory pool), using UVA, LRU + warmup, and layer-wise compute/comm overlap. Simulations: +69.4% throughput at 32K, +123% at 128K."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [paper, kv-cache, offload, deepseek, dsa, pd-disaggregation, long-context, inference]
sources: []
original_url: "https://arxiv.org/abs/2512.10576"
explored: false
confidence: high
---

# Source (paper): ESS - Offload-Centric Latent-Cache Management for DeepSeek-V3.2-Exp

arXiv **2512.10576v1** (Baidu Baige AI Team, 11 Dec 2025, 9 pages).

## Problem
DeepSeek-V3.2-Exp's [[20260608-125000-deepseek-sparse-attention-dsa-概念|sparse attention]] greatly reduces long-context latency, but under [[20260412-194111-prefill-decode-disaggregation-概念|PD disaggregation]] the **decode stage** is the bottleneck: the **Latent-Cache** grows linearly with sequence length while GPU memory is fixed, capping batch size and suppressing decode throughput.

## Method (ESS = Extended Sparse Server)
Selectively **offload Latent-Cache to CPU memory** while keeping latency-critical components on GPU, decoupling batch-size scaling from GPU memory. Key insight: the **Top-2K selected Latent-Cache exhibits strong temporal locality**, so a GPU **Sparse Memory Pool** holds a dynamically-updated subset. Three offload-prefetch techniques: (1) **UVA** (Unified Virtual Addressing) to mitigate low bandwidth from small-granularity transfers; (2) **LRU replacement + warm-up** to cut cache misses; (3) **layer-wise overlap** for a compute-comm pipeline. Setting: DeepSeek-V3.2-Exp, MTP=2 (accept ratio 1.7), 4 nodes, TP=1/EP=32, FlashMLA, two-batch overlap, PCIe 5.

## Results
High-fidelity simulation: **+69.4% throughput at 32K context, up to +123% at 128K**. Positioned as a practical/scalable long-context serving solution.

Relates to [[20260608-130300-kv-cache-compression-概念]] (offload-tiering axis), [[20260412-194030-kv-cache-概念]], [[20260412-194210-deepseek-实体]]. Contrast with [[20260412-193923-hisparse-turbocharging-sparse-attention-with-hierarchical-memory-来源|HiSparse]] (also offloads inactive KV to host for sparse attention).

## Counter-arguments / Data gaps
- Results are from **simulation**, not a deployed system - real PCIe/UVA behavior may differ.
- Specific to DeepSeek-V3.2-Exp's Top-2K sparse pattern; generalization to other sparse schemes unverified.
- Only the first 5 of 9 pages were read; detailed ablations/overheads not fully captured.
