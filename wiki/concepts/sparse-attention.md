---
title: "Sparse Attention"
tldr: "Attention mechanism that attends to only a selected subset of KV caches, reducing compute costs while retaining modeling capability - but doesn't inherently solve the memory capacity bottleneck"
date_created: 2026-04-12
date_modified: 2026-04-12
type: concept
tags: [attention, inference-optimization, kv-cache]
sources: ["[[lmsys-2026-hisparse]]", "[[lmsys-2026-hisparse-tweet]]", "[[vllm-2026-llm-compressor]]"]
explored: false
confidence: high
---

# Sparse Attention

**Sparse attention** is an efficient attention mechanism that reduces the quadratic compute and memory/IO cost of self-attention by attending to only a selected subset of KV cache entries at each decoding step. It retains strong modeling capability while avoiding the sharp cost increase that regular (dense) attention faces as context grows.

## Key Ideas

- **Top-k selection** is the typical approach: at each decoding step, only the top-k most relevant KV cache entries are used for attention computation
- Despite reducing compute, sparse attention does **not eliminate the memory capacity bottleneck** - the full [[KV Cache]] for the entire context must remain in GPU HBM for fast access, even though only a fraction is active
- This makes sparse attention systems often **capacity-bound rather than compute-bound**, limiting batch sizes and throughput
- [[lmsys-2026-hisparse|HiSparse]] addresses this by offloading inactive KV cache to host memory, achieving 3-5x throughput gains

**DeepSeek Sparse Attention (DSA)** is a specific implementation used by DeepSeek-V3.2 and GLM-5.1, and is currently the primary model family supported by HiSparse.

## How It Connects

- [[KV Cache]] - sparse attention's memory bottleneck centers on KV cache management
- [[Inference Optimization]] - sparse attention is one of several approaches to efficient LLM serving
- [[Quantization]] - complementary approach that reduces memory per token rather than reducing token count
- [[SGLang]] - serving framework implementing HiSparse for sparse attention acceleration

## Counter-arguments

- Sparse attention introduces approximation error - some information is lost by not attending to all tokens
- The "sparse" property is model-dependent; not all architectures exhibit the access patterns that make sparse attention effective
- For short contexts, the overhead of top-k selection may exceed the compute savings

## Data gaps

- Quality degradation measurements across different sparsity levels and tasks
- How sparse attention interacts with other optimization techniques (quantization, speculative decoding)
- Whether sparse attention patterns are consistent enough across inputs for effective caching strategies
