---
title: "Quantization"
tldr: "Reducing model weight and activation precision (FP16->FP8->FP4) to decrease memory footprint and increase inference throughput, with tools like vLLM's llm-compressor enabling quantize-once-serve-anywhere workflows"
date_created: 2026-04-12
date_modified: 2026-04-12
type: concept
tags: [inference-optimization, compression, fp8, nvfp4]
sources: ["[[vllm-2026-llm-compressor]]", "[[lmsys-2026-hisparse]]"]
explored: false
confidence: medium
---

# Quantization

**Quantization** reduces the numerical precision of model weights and activations (e.g., FP16 to FP8 or FP4), decreasing memory footprint and increasing inference throughput. It is complementary to [[Sparse Attention]], which reduces the number of tokens processed.

## Key Ideas

- **FP8** - 8-bit floating point, widely supported, good quality-efficiency tradeoff
- **NVFP4** - NVIDIA's 4-bit format, pushing the precision frontier lower
- [[vLLM]]'s llm-compressor enables "quantize once, serve with vLLM" workflow, supporting [[Gemma]] 4 and [[Qwen]] 3.5
- KV cache quantization is a specific application that reduces the dominant memory bottleneck during inference

## How It Connects

- [[KV Cache]] - quantizing KV cache reduces the main memory bottleneck
- [[Sparse Attention]] - complementary optimization approach
- [[vLLM]] - primary serving framework with quantization support
- [[Inference Optimization]] - quantization is one of several approaches

## Counter-arguments

- Lower precision can degrade output quality, especially for reasoning-heavy tasks
- Quantization effects are model-specific - some architectures are more robust than others

## Data gaps

- Systematic quality comparisons across FP16/FP8/FP4 for different model families and tasks
- How quantization interacts with sparse attention and other optimizations
