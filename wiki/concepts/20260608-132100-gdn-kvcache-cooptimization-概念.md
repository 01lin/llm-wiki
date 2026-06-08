---
title: "GDN-KVCache Co-optimization (Mamba CoW + branching_seqlen)"
tldr: "For hybrid GDN/Mamba + attention models, reuse the SSM recurrent state across requests the way prefix cache reuses KV: on a prefix hit, copy-on-write the aligned SSM snapshot and continue GDN recurrence from branching_seqlen instead of re-recurring from token 0 (O(full)->O(~10%)). Pairs with a Host SSM L2 cache (offload inactive SSM state) to lift concurrency. TokenSpeed's technique, proposed for migration into vLLM-Ascend."
date_created: 2026-06-08
date_modified: 2026-06-08
type: concept
tags: [gdn, mamba, ssm, kv-cache, prefix-cache, hybrid-model, ascend, inference]
sources: ["[[20260608-132200-tokenspeed-gdn-kvcache-migration-to-vllm-ascend-来源]]"]
explored: false
confidence: medium
---

# GDN-KVCache Co-optimization

Hybrid models (e.g. Qwen3.5-397B = Gated Delta Net / Mamba layers + MoE attention) keep both a [[20260412-194030-kv-cache-概念|KV cache]] (attention) and a per-request **SSM recurrent state** (conv state + ssm_h) for the linear-attention layers. Standard prefix caching reuses KV but **not** the SSM state, so even with 90% prefix hit the GDN layers re-recur from token 0 (O(full sequence)). GDN-KVCache co-optimization fixes this:

- **Mamba CoW + branching_seqlen**: on a prefix hit, find the deepest block-aligned SSM snapshot, copy-on-write it into the working slot, and continue GDN recurrence from `branching_seqlen = align(match_depth, mamba_block_size)` - cutting GDN compute from O(full) to O(~10%). Requires `mamba_cache_mode="align"` + GDN `initial_state` support (both already in vLLM-Ascend; the missing piece is the signal chain Scheduler -> SchedulerOutput -> ModelRunner -> GDN forward).
- **Host SSM L2 cache**: SSM state is large (~200-500MB/request for 94 GDN layers), so it caps concurrency. Offload inactive SSM state D->H (bulk) and reload H->D (per-layer, PD-pipeline-friendly) via pinned memory, freeing GPU for 3-5x more concurrent requests.
- **MTP O(1) SSM index update**: after speculative verify, update integer slot indices (O(batch)) instead of copying SSM tensors (O(L*D)).
- **TP-deterministic scheduling**: sort candidates by `request_id` (string, cross-rank stable) and pick OOM victims by longest sequence - avoids HCCL deadlock across TP ranks.

This is [[20260608-131000-tokenspeed-实体|TokenSpeed]]'s GDN-KVCache co-optimization, proposed for migration into [[20260608-122850-vllm-ascend-实体|vLLM-Ascend]] for Qwen3.5-397B. See [[20260608-132200-tokenspeed-gdn-kvcache-migration-to-vllm-ascend-来源]]. Relates to [[20260608-131400-radix-tree-prefix-cache-概念]], [[20260608-121100-mtp-multi-token-prediction-概念]].

## Counter-arguments / Data gaps
- Gains assume high prefix-hit workloads (system prompt / long-prefix reuse); low for single-shot long context.
- Ascend pinned-memory D<->H async semantics differ from CUDA - flagged as a risk, not yet validated.
- Projected 3-5x throughput is an estimate dependent on workload.
