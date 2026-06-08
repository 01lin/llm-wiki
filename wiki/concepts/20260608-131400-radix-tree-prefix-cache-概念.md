---
title: "RadixTree Prefix Cache"
tldr: "A radix/prefix tree that indexes KV cache by token sequence (page-granularity rolling hash) so requests sharing a prefix reuse cached KV. TokenSpeed implements it in C++ (O(prefix_len) match) with a HybridPrefixCache co-managing KV + Mamba state; SGLang has RadixAttention (Python); vLLM uses a block-hash chain. (stub)"
date_created: 2026-06-08
date_modified: 2026-06-08
type: concept
tags: [kv-cache, prefix-cache, radix-tree, inference]
sources: ["[[20260608-131100-tokenspeed-architecture-analysis-来源]]", "[[20260608-131200-tokenspeed-vs-vllm-ascend-comparison-来源]]"]
explored: false
confidence: medium
---

# RadixTree Prefix Cache

A **radix (prefix) tree** that indexes [[20260412-194030-kv-cache-概念|KV cache]] by token sequence: tokens are rolling-hashed at page granularity, so requests sharing a prefix reuse cached KV across requests (cutting prefill). [[20260608-131000-tokenspeed-实体|TokenSpeed]] implements it in **C++** with O(prefix_len) match and a **HybridPrefixCache** that co-manages KV cache and Mamba/SSM state via eviction callbacks, spanning its 3-level store (GPU/host/L3). [[20260412-194210-sglang-实体|SGLang]] popularized the idea as RadixAttention (Python); vLLM uses a block-hash chain (larger constant factor). See [[20260608-131200-tokenspeed-vs-vllm-ascend-comparison-来源]].

## Data gaps
- Stub. No measured hit-rate/latency numbers in these sources; NPU prefix-match validation is listed as TODO in the TokenSpeed-Ascend plan.
