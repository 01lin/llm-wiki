---
title: "Speculative Decoding (投机解码)"
tldr: "Draft-then-verify inference acceleration: a cheap drafter proposes K tokens, the target model verifies them in one parallel pass, accepting a prefix. Lossless when verification preserves the target distribution. End-to-end speedup depends on acceptance rate, draft cost, and system efficiency - not acceptance alone."
date_created: 2026-06-08
date_modified: 2026-06-08
type: concept
tags: [inference, speculative-decoding, llm-serving, acceleration]
sources: ["[[20260608-120600-spec-decoding-arxiv-survey-2026-03-04-来源]]", "[[20260608-120700-vllm-vllm-ascend-spec-decode-support-来源]]", "[[20260608-120800-glm5-1-mtp-投机性能优化方案-来源]]", "[[20260608-120900-qwen3-5-mtp-ascend-适配分析-来源]]"]
explored: false
confidence: high
---

# Speculative Decoding (投机解码)

**Speculative decoding** accelerates autoregressive LLM inference by splitting each step into a cheap **draft** stage and a single parallel **verify** stage. A lightweight drafter proposes K candidate tokens; the expensive target model then verifies all K in one forward pass and accepts the longest correct prefix. Because the target still scores every accepted token, the technique is **lossless** when the verification rule preserves the target distribution (rejection sampling). The win comes from amortizing the memory-bandwidth-bound decode step over multiple tokens per target forward pass.

## The core performance equation

A recurring framing across these sources: end-to-end speedup is

> `f(acceptance rate, tokens speculated per round K, draft cost, verify cost, system utilization)`

A critical, repeated lesson: **high acceptance rate does not guarantee end-to-end speedup**. vLLM issue #36498 and #36037 document cases where acceptance is high but decode is still slower than no speculation, because draft/verify overhead and host-device sync eat the gains. [[20260608-120900-qwen3-5-mtp-ascend-适配分析-来源]] makes the same point bluntly: on Ascend, the bottleneck for Qwen3.5-397B was system/operator layer, not algorithm.

## Families of methods

vLLM exposes these via `speculative_config.method` (see [[20260608-120700-vllm-vllm-ascend-spec-decode-support-来源]]):

- **[[20260608-121100-mtp-multi-token-prediction-概念]]** (`mtp`) - draft layers built into the model weights; no extra model. Highest-value path for modern models (DeepSeek, GLM-4.x, Qwen3.5).
- **[[20260608-121200-eagle-概念]] / EAGLE-3** (`eagle`, `eagle3`) - auxiliary drafter fusing hidden states + token embeddings; needs trained EAGLE weights.
- **[[20260608-121400-draft-model-spec-概念]]** (`draft_model`) - independent small model; flexible but heavy.
- **[[20260608-121500-medusa-概念]]** (`medusa`) - extra LM heads on the target.
- **[[20260608-121600-ngram-suffix-decoding-概念]]** (`ngram`, `suffix`) - training-free pattern matching; low gain, good for repetitive output.
- **[[20260608-121300-dflash-概念]]** (`dflash`) - DeepSeek Flash MTP path.

## Algorithmic frontiers (2026)

From the [[20260608-120600-spec-decoding-arxiv-survey-2026-03-04-来源]] (16 papers, Mar-Apr 2026), four trends stand out:

1. **Relaxing verification** - drop strict distribution match for controllable divergence or semantic-equivalence matching (DIVERSED, Cactus, LVSpec). Strict exact-match is treated as an unnecessary constraint.
2. **[[20260608-121700-speculative-tree-概念]] from balanced to anisotropic** - SMART shows big trees can give *negative* speedup under hardware saturation; Goose builds asymmetric trees (deep chains for high-quality tokens, wide branches for low-quality).
3. **Self-speculation** - the model drafts for itself with no extra weights (SpecMoE, S2D2), or emits multiple tokens directly (MARS).
4. **Spreading to non-AR architectures** - MoE, masked/block diffusion, Video-LLM, image generation.

## Acceptance-rate improvement levers

From [[20260608-120800-glm5-1-mtp-投机性能优化方案-来源]] (GLM5.1 MTP3 at ~50-60% acceptance):
- **FastMTP** train-test alignment (self-distillation) to close the train-inference gap - reportedly the root-cause fix, targeting 75-85% acceptance.
- **EARS** entropy-adaptive rejection sampling (relax threshold when target is uncertain) - ~+14.6% throughput, ~30 lines.
- **[[20260608-121800-dynamic-speculation-length-概念]] (DSL)** - vary K by draft confidence; early-exit on low confidence.
- L-MTP skip-prediction, multi-layer MTP, tree MTP (OPT-Tree/Sequoia).

## Counter-arguments

- **Acceptance is a vanity metric.** Multiple production reports (vLLM #36498) show high acceptance with no wall-clock gain. System overhead (host-device sync, graph capture, kernel launch) can dominate.
- **Big trees can hurt.** SMART proves the "bigger tree = faster" assumption false under batch/hardware saturation.
- **Relaxed verification trades losslessness for speed.** DIVERSED/Cactus accept tokens off the strict target distribution; "lossless" no longer holds, only "quality-controlled."
- **Hardware-coupled gains.** On NPUs, gains depend on quantization path (Ascend MTP fast path only opens at `w8a8_dynamic`) and operator integer limits (MTP <= 15).

## Data gaps

- No head-to-head benchmark in these sources comparing MTP vs EAGLE-3 vs draft-model on the *same* model/hardware.
- The survey summarizes abstracts only; the 5 ingested PDFs (Sequoia 2402.12374 etc.) need full reads to verify claimed speedups.
- Little data on speculative decoding interaction with prefix cache and PD-disaggregation at high concurrency (flagged but unresolved in vLLM #38182).
