---
title: "Source: 混合投机工程实现分析 (vLLM-Ascend model_runner_v1, MTP graph, async schedule)"
tldr: "Engineering analysis of why dynamic hybrid speculation breaks vLLM-Ascend's async-schedule/graph contracts (spec_num_tokens is a static shape in 4 places) and the recommended fix: static K_verify_cap + dynamic K_eff via valid_len/source_map, with MTP execution length decoupled from verify length (bucket graphs mtp{0,1,2,4}). 5-phase rollout."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [speculative-decoding, hybrid-speculation, mtp, suffix, vllm-ascend, async-schedule]
sources: []
original_url: ""
explored: false
confidence: medium
---

# Source: 混合投机工程实现分析 (async schedule)

Engineering design doc (2026-05-09) for [[20260608-124000-hybrid-speculation-mtp-suffix-概念]] on [[20260608-122850-vllm-ascend-实体]] `model_runner_v1`.

## Root cause
`spec_num_tokens` simultaneously plays four roles: (1) algorithm semantics, (2) buffer width (`future_input_map`, sampling/grammar buffers), (3) graph shape (verify/draft capture metadata `max_seq_len_q`), (4) scheduler protocol (`decode_input_tokens`, reserve-token rules). So runtime-dynamic length breaks async-schedule, overlap, graph replay, and scheduler reserve contracts at once.

## Options A-D
- **A** fully dynamic length: feasible but extreme complexity, conflicts with async/graph - **not recommended**.
- **B** multi-graph buckets `K in {1,2,4}`: feasible but mixing buckets in async batches is hard - not enough as first version.
- **C** static `K_cap` + dynamic `K_eff <= K_cap` via mask: most compatible, smallest change - **recommended main line**.
- **D** static verify cap + decoupled MTP execution length: C plus letting suffix truly replace MTP compute - **recommended performance version**.

## Four-layer design
L1 Static Capacity (graph/buffer/scheduler all key off `K_cap`), L2 Effective Spec State (`effective_spec_len`, `suffix_spec_len`, `mtp_spec_len`, `draft_valid_mask`, `draft_source`), L3 Hybrid Draft Composer (emits fixed `[bs, K_cap]`), L4 Verification Semantics (valid-length-aware verify - only accept first `K_eff`).

## Key principles
- Scheduler only ever sees `K_cap`, never `K_eff` (input capacity fixed, output length dynamic via `accept_len`) - matches existing `make_update_reserve_tokens_event`.
- MTP graph compatibility != MTP cost reduction: must run only the tail `K_mtp_exec`, not full `mtp6`. Use MTP bucket graph (`mtp{0,1,2,4}`), micro-graph, or (Phase-1) run full + take first K.
- Policy at batch-bucket granularity (ctx_len x bs), not per-request; update every N steps / T seconds with EMA, not every round.

## 5-phase rollout
Phase 0 observability -> Phase 1 static `K_cap` + MTP-only dynamic effective length -> Phase 2 decouple MTP exec length (bucket graphs) -> Phase 3 add suffix/hybrid -> Phase 4 policy controller (EMA/bandit) -> Phase 5 micro-graph + roofline-aware refinement.

Feeds [[20260608-124000-hybrid-speculation-mtp-suffix-概念]], relates to [[20260608-124200-speculators-fixed-mtp3-suffix-tail-design-来源]].

## Counter-arguments / Data gaps
- Design only, no measured results.
- Suffix-after-MTP distribution drift flagged as a risk; `K_mtp<=2/3` limit proposed but unvalidated.
