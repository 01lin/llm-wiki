---
title: "LMSYS HiSparse Announcement Thread"
tldr: "LMSYS Twitter thread announcing HiSparse blog with key results: 3x throughput at 256 concurrent requests, 5x on long-context, supports DeepSeek-V3.2 and GLM-5.1"
date_created: 2026-04-12
date_modified: 2026-04-12
type: source
tags: [sparse-attention, kv-cache, sglang]
source_type: tweet
source_file: "[[Thread by @lmsysorg]]"
original_url: "https://x.com/lmsysorg/status/2042683003730801147"
explored: false
confidence: high
---

# LMSYS HiSparse Announcement Thread

## Summary

Twitter announcement by [[LMSYS]] of the HiSparse blog post. The thread summarizes the key results and techniques from [[lmsys-2026-hisparse|the full blog post]].

Key claims from the thread:
- 3x throughput at 256 concurrent requests vs baseline (32K input, 8K output on 8xH200)
- Up to 5x throughput on long-context scenarios (two H20 PD-disaggregated deployment)
- Proactive offloading of inactive KV cache to host memory
- Hot device buffer keeps frequently accessed KV regions on-device
- Custom CUDA kernel: top-k miss detection + LRU eviction + page table updates in one pass
- Supports DeepSeek-V3.2 and GLM-5.1

Community response included a question comparing HiSparse to prior work (arxiv 2512.10576) on computation and communication overlapping.

## Key Takeaways

- This is a promotional tweet for the full blog - see [[lmsys-2026-hisparse]] for detailed analysis
- Community engagement suggests interest in how this differs from prior KV cache offloading work

## Concepts & Entities Mentioned

- [[Sparse Attention]], [[KV Cache]], [[SGLang]], [[LMSYS]]
- [[DeepSeek]], [[GLM]]

## Counter-arguments

- See [[lmsys-2026-hisparse]] for full counter-arguments

## Data gaps

- Community member raised comparison to arxiv 2512.10576 - relationship unclear
