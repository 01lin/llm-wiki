---
title: "DSA (DeepSeek Sparse Attention)"
tldr: "DeepSeek's sparse-attention design used in V3.2 and V4 (and GLM-5/5.1). A lightning indexer computes per-query Top-K compressed-KV indices (index_topk), and sparse attention attends only to selected KV via per-layer compression ratios (C4/C128). On Ascend the indexer Top-K is hard-capped at 512 in the operator, blocking DeepSeek-V4-Pro's index_topk=1024."
date_created: 2026-06-08
date_modified: 2026-06-08
type: concept
tags: [sparse-attention, dsa, deepseek, ascend, kv-cache, inference]
sources: ["[[20260608-125100-deepseek-v4-pro-vllm-ascend-gap-analysis-来源]]"]
explored: false
confidence: medium
---

# DSA (DeepSeek Sparse Attention)

**DSA** is DeepSeek's [[20260412-194030-sparse-attention-概念]] design, used in DeepSeek-V3.2, DeepSeek-V4 (Flash/Pro), and adopted by GLM-5/GLM-5.1. Instead of full attention, a **lightning indexer** computes per-query Top-K indices over compressed KV (`index_topk`), and sparse attention attends only to the selected keys/values. Attention layers carry a per-layer **`compress_ratios`** pattern (e.g. C4 = compress ratio 4, C128 = ratio 128) controlling how aggressively KV is compressed.

## Key structural parameters (DeepSeek-V4)
- `head_dim=512`, `index_head_dim=128`, `sliding_window=128` (consistent across Flash/Pro and current Ascend impl).
- `index_topk`: **512 (Flash)** vs **1024 (Pro)**.
- `compress_ratios`: per-layer list whose length must match `num_hidden_layers`; values seen include 0, 1, 4, 128.

## The Ascend Top-K hard cap
On vLLM-Ascend, the operator `sparse_attn_sharedkv` requires the compressed-KV sparse-index last dim to equal a compile-time `TOPK_LIMIT = 512` (`sparse_attn_sharedkv_tiling.h:125`, checked in `.cpp:1029-1034`). The Python path (`dsa_v1.py`) also hardcodes `index_topk = 512` and `sparse_count = 512`. So DeepSeek-V4-**Pro** (`index_topk=1024`) does not close at the operator level - **framework and kernel must both change**; patching Python alone fails while csrc still caps at 512. See [[20260608-125100-deepseek-v4-pro-vllm-ascend-gap-analysis-来源]].

Relates to [[20260412-194030-kv-cache-概念]], [[20260412-194210-deepseek-实体]], [[20260412-194210-glm-实体]].

## Counter-arguments / Data gaps
- Analysis is from one repo snapshot (2026-04-29); the 512 cap may be lifted in later CANN/kernel versions.
- The semantics of `compress_ratio==0` layers (special layers) are not clearly handled in the Ascend impl - a runtime risk, not yet validated.
- No measured accuracy/perf for DSA on V4-Pro (blocked by the cap).
