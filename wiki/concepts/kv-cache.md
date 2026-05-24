---
title: "KV Cache"
tldr: "Key-Value cache stores precomputed attention keys and values during LLM inference to avoid redundant computation, but its memory footprint is the primary bottleneck for long-context and high-concurrency serving"
date_created: 2026-04-12
date_modified: 2026-04-12
type: concept
tags: [inference, memory, attention, serving]
sources: ["[[lmsys-2026-hisparse]]", "[[lmsys-2026-hisparse-tweet]]", "[[vllm-2026-llm-compressor]]"]
explored: false
confidence: high
---

# KV Cache

The **KV cache** stores precomputed key and value tensors from previous tokens during autoregressive LLM inference, allowing each new token's attention computation to reuse prior results rather than recomputing from scratch. While essential for efficient generation, the KV cache's memory footprint grows linearly with sequence length and batch size, making it the primary memory bottleneck in LLM serving.

## Key Ideas

- KV cache size scales with: sequence length x number of layers x number of heads x head dimension x 2 (keys + values) x precision
- For long contexts (32K+ tokens) and high concurrency, KV cache can consume the majority of GPU HBM
- **Memory management strategies**: [[vLLM]]'s PagedAttention, [[SGLang]]'s RadixAttention, and now [[lmsys-2026-hisparse|HiSparse]]'s hierarchical offloading
- [[Quantization]] of KV cache (e.g., FP8, NVFP4) reduces per-token memory but doesn't change the linear scaling
- [[Sparse Attention]] reduces compute but the full KV cache must still reside in memory unless hierarchical management is used

## How It Connects

- [[Sparse Attention]] - reduces which KV entries are accessed but not which are stored
- [[Quantization]] - reduces the precision/size of stored KV entries
- [[Inference Optimization]] - KV cache management is central to serving efficiency
- [[Prefill-Decode Disaggregation]] - architectural approach that separates KV cache generation from consumption

## Counter-arguments

- KV cache is a solved problem for short contexts - the bottleneck only matters at scale
- Alternative architectures (linear attention, state-space models) avoid KV cache entirely

## Data gaps

- Comparative analysis of KV cache management strategies (PagedAttention vs RadixAttention vs HiSparse)
- Impact of different KV cache compression techniques on output quality
