---
title: "vLLM LLM Compressor 3K Stars"
tldr: "vLLM's llm-compressor tool hits 3K GitHub stars, supports NVFP4 and FP8 quantization for Gemma 4 and Qwen 3.5 models for faster inference"
date_created: 2026-04-12
date_modified: 2026-04-12
type: source
tags: [quantization, vllm, inference-optimization, fp8, nvfp4]
source_type: tweet
source_file: "[[Thread by @vllm_project]]"
original_url: "https://x.com/vllm_project/status/2042244885001200059"
explored: false
confidence: medium
---

# vLLM LLM Compressor 3K Stars

## Summary

[[vLLM]] announced that their **llm-compressor** tool (github.com/vllm-project/llm-compressor) reached 3,000 GitHub stars. The tool enables model [[Quantization]] with a "quantize once, serve with vLLM" workflow.

Key supported formats:
- **NVFP4** - NVIDIA's 4-bit floating point format
- **FP8** - 8-bit floating point quantization

Already supports latest models including [[Gemma]] 4 and [[Qwen]] 3.5 with pre-quantized checkpoints available.

Community discussion touched on NVFP4 compatibility with the DGX Spark platform and requests for additional quantization methods like TurboQuant.

## Key Takeaways

- Quantization is becoming a standard part of the LLM deployment pipeline, not an optimization afterthought
- The "quantize once, serve anywhere" approach reduces friction in deploying quantized models
- NVFP4 represents a push toward even lower precision (4-bit) inference

## Concepts & Entities Mentioned

- [[vLLM]] - serving framework
- [[Quantization]] - core technique
- [[Gemma]] - Google's model family (Gemma 4)
- [[Qwen]] - Alibaba's model family (Qwen 3.5)
- [[NVIDIA]] - hardware vendor (NVFP4 format, DGX Spark)

## Counter-arguments

- Lower precision quantization (FP4) may introduce meaningful quality degradation for some tasks
- The "quantize once" claim assumes the serving framework handles all edge cases - in practice, quantization can expose numerical instabilities

## Data gaps

- No quality benchmarks comparing quantized vs full-precision models provided in the tweet
- No performance (tokens/sec) numbers for quantized serving
- Unclear how NVFP4 quality compares to FP8 across different model families
