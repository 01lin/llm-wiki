---
title: "KV Cache Compression (KV 缓存压缩)"
tldr: "Shrinking the KV cache footprint to fit more/longer sequences: the precision path FP16->FP8->INT4->2-bit, plus tiered offload (GPU->CPU->SSD). vLLM TurboQuant does online 2-bit KV (4x capacity, no offline calibration); SGLang HiCache does multi-tier storage. The two approaches are orthogonal and stackable. (stub)"
date_created: 2026-06-08
date_modified: 2026-06-08
type: concept
tags: [kv-cache, quantization, inference, long-context, memory]
sources: ["[[20260608-130100-inference-engine-github-weekly-0409-0416-来源]]"]
explored: false
confidence: medium
---

# KV Cache Compression

As [[20260412-194030-kv-cache-概念|KV cache]] is the memory bottleneck for long-context and high-concurrency serving, **compressing it** is a fast-moving frontier. Two orthogonal axes:
- **Precision**: FP16 -> FP8 -> INT4 -> 2-bit. vLLM **TurboQuant** (PR #38479) does online 2-bit KV (4x capacity, no offline calibration; k8v4 ~2.6x, 3-bit up to 4.9x; GSM8K 0.78-0.86 vs 0.90 baseline) - the hottest inference PR of mid-April 2026.
- **Tiering / offload**: keep hot KV on GPU, spill to CPU/SSD. SGLang **HiCache** multi-tier storage (SiMM/Mooncake backends); vLLM CPU KV offload + HMA.

The two are stackable. Related to [[20260412-194111-quantization-概念|quantization]] (weights/activations) but distinct - this compresses the *cache*. See [[20260608-130100-inference-engine-github-weekly-0409-0416-来源]].

## Counter-arguments / Data gaps
- Stub. 2-bit KV costs accuracy (GSM8K -0.04 to -0.12 vs baseline) - the quality/capacity tradeoff is real.
- No Ascend KV-compression data point yet (NPU lags here).
