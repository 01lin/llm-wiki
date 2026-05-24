---
title: "Prefill-Decode Disaggregation"
tldr: "Deployment architecture that separates LLM inference into dedicated prefill and decode instances, enabling independent scaling and optimization of each phase"
date_created: 2026-04-12
date_modified: 2026-04-12
type: concept
tags: [inference, serving, architecture]
sources: ["[[lmsys-2026-hisparse]]"]
explored: false
confidence: medium
---

# Prefill-Decode Disaggregation

**Prefill-decode disaggregation** (PD disaggregation) separates LLM inference into dedicated prefill instances (processing input tokens) and decode instances (generating output tokens). Used in HiSparse benchmarks with two H20 nodes achieving up to 5x throughput improvement.

## How It Connects

- [[KV Cache]] - prefill generates KV cache, decode consumes it
- [[Sparse Attention]] - HiSparse optimizes the decode side
- [[SGLang]] - implements PD disaggregation

## Data gaps

- Latency overhead of transferring KV cache between instances
