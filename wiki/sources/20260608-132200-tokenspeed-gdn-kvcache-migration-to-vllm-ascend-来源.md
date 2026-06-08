---
title: "Source: TokenSpeed GDN-KVCache 协同优化迁移至 vllm-ascend"
tldr: "Migration plan (2026-05-28) to port TokenSpeed's GDN-KVCache co-optimization techniques INTO vllm-ascend (not build a separate engine) for Qwen3.5-397B hybrid GDN+MoE on Ascend A2/A3. 5 phases: Mamba CoW+branching_seqlen, Host SSM L2 cache, TP-deterministic tiebreak, MTP O(1) SSM update, PD layer-wise SSM transfer. Targets 3-5x over current vllm-ascend baseline. ~3 months, 2 SE."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [tokenspeed, vllm-ascend, gdn, mamba, ssm, kv-cache, qwen, ascend, migration, mtp]
sources: []
original_url: ""
explored: false
confidence: high
---

# Source: TokenSpeed GDN-KVCache 协同优化迁移至 vllm-ascend

Migration plan (2026-05-28) - distinct from building TokenSpeed-Ascend: here the goal is to **port TokenSpeed's GDN-KVCache techniques into [[20260608-122850-vllm-ascend-实体|vLLM-Ascend]] itself** for Qwen3.5-397B-A17B (hybrid Gated Delta Net + MoE) on A2/A3, raising the baseline 3-5x. TokenSpeed hits 580 tok/s on B200 (4-node) for this model via [[20260608-132100-gdn-kvcache-cooptimization-概念|GDN-KVCache co-optimization]].

## What vLLM-Ascend already has (reusable)
Qwen3.5 hybrid routing (`patch_qwen3_5.py`), GDN operators with `initial_state` support (A3 `chunk_gated_delta_rule` + `npu_recurrent_gated_delta_rule`; 310P pytorch path), `mamba_cache_mode="align"` (block-aligned SSM snapshots), CPU KV offload framework (KV only), MTP/DFlash proposer framework, RecomputeScheduler (LIFO preemption), Mooncake layer-wise connector.

## The 5 gaps -> 5 phases
1. **Mamba CoW + branching_seqlen** (P0, 3-4wk): the snapshot exists and GDN `initial_state` works, but the signal chain (Scheduler -> SchedulerOutput -> ModelRunner -> GDN forward) is missing. Add `mamba_cow_src_indices` / `mamba_branching_seqlens` to SchedulerOutput; compute CoW src + `align(match_depth, mamba_block_size)`; in GDN forward copy SSM/conv state src->working and continue from branching_seqlen. -> TTFT -60-80%.
2. **Host SSM L2 cache** (P0, 2-3wk): new `mamba_host_cache.py` with `pin_memory=True` buffers; `backup_all_layers` (D->H bulk) / `load_layer` (H->D per-layer); LRU. SSM ~200-500MB/request -> offload lifts concurrency 3-5x.
3. **TP-deterministic tiebreak** (P1, 1wk): sort candidates by `(priority, request_id)`, OOM victim by `(num_computed_tokens, request_id)` - eliminate HCCL deadlock at TP=8/16.
4. **MTP O(1) SSM update** (P1, 2-3wk): replace `postprocess_mamba_fused_kernel` O(L*D) copy with O(batch) integer index write via `@torch.compile(dynamic=True)`. -> TPOT -20-30%.
5. **PD layer-wise SSM transfer** (P2, 4-5wk): extend Mooncake connector with SSM channel for prefill/decode disaggregation. -> cluster throughput +20-40%.

## Projected (Ascend, all stacked)
~3-5x over current vllm-ascend baseline (200-250 -> ~150-300 tok/s single A3 8-card node, approaching B200-class config), **conditional on 90%+ prefix hit, prefix caching on, align mode on, TP>=8, multi-node for PD**. ~3 months / 2 SE.

Relates to [[20260608-132100-gdn-kvcache-cooptimization-概念]], [[20260608-131000-tokenspeed-实体]], [[20260608-120900-qwen3-5-mtp-ascend-适配分析-来源]], [[20260412-194111-prefill-decode-disaggregation-概念]].

## Counter-arguments / Data gaps
- 3-5x assumes high prefix-hit agentic workloads; low-reuse workloads gain little.
- Risks flagged: Ascend pinned-memory semantics, GDN `initial_state` precision drift at large batch, `@torch.compile` operator coverage on Ascend, TP-tiebreak upstream compatibility.
- Based on a 2026-05-28 code snapshot; "non-architectural-speculation" per the note but still unimplemented.
