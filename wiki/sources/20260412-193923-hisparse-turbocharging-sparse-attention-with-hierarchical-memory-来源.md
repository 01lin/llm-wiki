---
title: "HiSparse: Turbocharging Sparse Attention with Hierarchical Memory"
tldr: "LMSYS introduces HiSparse, a hierarchical memory system for sparse attention that offloads inactive KV cache to host memory, achieving 3-5x throughput gains on long-context LLM inference"
date_created: 2026-04-12
date_modified: 2026-04-12
type: source
tags: [sparse-attention, kv-cache, inference-optimization, sglang, cuda]
source_type: article
source_file: "[[20260412-193653-hisparse-turbocharging-sparse-attention-with-hierarchical-memory-归档]]"
original_url: "https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/"
explored: false
confidence: high
---

# HiSparse: Turbocharging Sparse Attention with Hierarchical Memory

## Summary

**Sparse attention** reduces compute costs by attending to only a subset of KV caches, but the full KV cache still resides in GPU HBM, making inference capacity-bound rather than compute-bound. HiSparse addresses this by introducing a hierarchical memory system that proactively offloads inactive KV cache entries to host memory while maintaining a hot device buffer on GPU HBM for frequently accessed regions.

The system achieves **3x throughput** at 256 concurrent requests versus baseline on 8xH200 (32K input, 8K output), and up to **5x throughput** on long-context scenarios using two H20 PD-disaggregated deployment. At low concurrency, HiSparse introduces modest overhead since the extra I/O from sparse KV loading outweighs memory savings - the gains become pronounced as concurrency increases.

Key technical components:
- **Proactive KV cache offloading** - inactive entries moved to host memory, freeing GPU HBM for larger batch sizes
- **Hot device buffer** - frequently accessed KV regions kept on-device to minimize swap-in latency
- **Custom CUDA kernel** - performs top-k miss detection, LRU eviction, and page table updates in a single pass
- Larger hot device buffers (4096 vs 2048 slots) with LRU eviction substantially reduce miss counts

Currently supports [[DeepSeek Sparse Attention]] (DSA) model families: DeepSeek-V3.2 and GLM-5.1. Implemented as part of [[20260412-194210-sglang-实体|SGLang]].

![[20260412-193614-hisparse-throughput-concurrency-图片.png]]
![[20260412-193615-hisparse-overview-图片.png]]

## Key Takeaways

- Sparse attention alone doesn't solve the memory capacity bottleneck - the full KV cache must still reside in GPU HBM
- Hierarchical memory management (GPU HBM + host memory) unlocks the actual throughput potential of sparse attention
- The approach builds on prior [[HiCache]] work from LMSYS
- Future work targets reducing I/O overhead through better overlap and leveraging higher CPU-GPU bandwidth on Grace Blackwell systems
- Plans to extend to hybrid model architectures beyond DSA

## Concepts & Entities Mentioned

- [[20260412-194030-sparse-attention-概念|Sparse Attention]] - core technique being optimized
- [[20260412-194030-kv-cache-概念|KV Cache]] - the memory bottleneck being addressed
- [[20260412-194210-sglang-实体|SGLang]] - the serving framework implementing HiSparse
- [[20260412-194210-lmsys-实体|LMSYS]] - research org behind the work
- [[20260412-194210-deepseek-实体|DeepSeek]] - model family supported (DeepSeek-V3.2)
- [[20260412-194210-glm-实体|GLM]] - model family supported (GLM-5.1)
- [[20260412-194111-prefill-decode-disaggregation-概念|Prefill-Decode Disaggregation]] - deployment architecture used in benchmarks

## Counter-arguments

- At low concurrency, HiSparse adds overhead rather than improving performance - it only helps at scale
- Currently limited to DSA model families, not applicable to dense attention models or other sparse attention implementations
- Host memory bandwidth is still a bottleneck - the approach pushes the problem from GPU memory capacity to CPU-GPU bandwidth
- Grace Blackwell systems with unified memory may make this approach less necessary

## Data gaps

- No latency (TTFT, TPOT) numbers reported, only throughput
- No comparison with other KV cache compression techniques (e.g., quantized KV cache, attention sinks)
- Unclear how the hot buffer sizing interacts with different workload distributions
- No data on quality degradation from the sparse attention approximation itself
