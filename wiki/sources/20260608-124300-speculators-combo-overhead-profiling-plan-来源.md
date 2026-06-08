---
title: "Source: 组合投机单步开销性能分析与打点方案 (Qwen3.5-122B, Ascend 910B3)"
tldr: "Profiling plan (2026-05-10) to explain why combo speculation (MTP7+suffix3, TPOT 17.64ms) is ~18% slower than pure MTP10 (14.92ms) at bs=8 on Ascend 910B3, despite similar acceptance. Maps the model_runner_v1/eagle_proposer/acl_graph code path, lists 6 prime suspects (metadata sync, pre-replay synchronize, bookkeeping-before-draft, padding, suffix serialization, copy), and a two-tier (lightweight + msprof) instrumentation scheme."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [speculative-decoding, hybrid-speculation, profiling, vllm-ascend, qwen, ascend]
sources: []
original_url: ""
explored: false
confidence: medium
---

# Source: 组合投机单步开销性能分析与打点方案

Profiling plan (2026-05-10) for the combo-speculation slowdown puzzle on Qwen3.5-122B / Ascend 910B3 / [[20260608-122850-vllm-ascend-实体]] `model_runner_v1`.

## The puzzle
`MTP7+suffix3, bs=8`: TPOT 17.64ms, accept 7.32. `MTP10, bs=8`: TPOT 14.92ms, accept 7.77. Acceptance can't explain ~18% slower - so the cost is in fixed per-step / scheduling / sync overhead or graph-path degradation.

## Code path mapped
`execute_model()` (target verify), `sample_tokens()` (sampling, bookkeeping, draft), `propose_draft_token_ids()`, `eagle_proposer._propose()` (MTP graph main path), `_run_merged_draft()` (per-step MTP forward), `ACLGraphWrapper.__call__()` (replay). With line anchors.

## 6 prime suspects (ranked)
1. metadata builder sync near `eagle_proposer.py:691` (`FIXME: causes synchronization`).
2. `acl_graph.py:210` pre-replay `torch.npu.current_stream().synchronize()` - breaks overlap, amplified under concurrency.
3. `_bookkeeping_sync()` runs *before* draft (TODO to move it after) - CPU bookkeeping may block next-round draft prep.
4. combo path inflates `num_tokens_padded` / `batch_descriptor` vs pure MTP.
5. suffix tail build / hybrid compose serialized under concurrency.
6. `_copy_draft_token_ids_to_cpu` / valid-count copy amplified.

## Instrumentation
- L1 always-on lightweight: `time.perf_counter_ns()` for CPU spans, `torch.npu.Event` for NPU spans, aggregate per window (no per-step synchronize), emit JSONL every 100-500 steps. New `utils/spec_profile.py` helper.
- L2 heavy timeline: Ascend profiler / msprof / `record_function`, short windows only, to see stream idle and replay gaps.
- Experiment matrix: {MTP7, MTP10, suffix7/10, MTP7+suffix1/2/3} x bs{1,2,4,8} x graph{on,off} x suffix{disabled, force-miss, real-match, static-match}.

## Interim policy
Keep combo at low concurrency; fall back to pure MTP at bs>=8; test suffix tail 1/2, pause large-scale suffix3 until root cause found.

Feeds [[20260608-124000-hybrid-speculation-mtp-suffix-概念]]; profiles the design in [[20260608-124200-speculators-fixed-mtp3-suffix-tail-design-来源]].

## Counter-arguments / Data gaps
- A plan, not results - the 2.72ms gap is not yet attributed.
- Verify-cost model (does `T_verify` track `M + suffix_hit_len`, or does padding eat the advantage?) is hypothesized, unverified.
