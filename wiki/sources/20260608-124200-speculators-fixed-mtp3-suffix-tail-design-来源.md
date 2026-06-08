---
title: "Source: Fixed Verify-6 + Fixed MTP-3 + Dynamic Suffix-Tail(0..3) 方案设计"
tldr: "Detailed design (2026-05-09) for the simplest practical hybrid-speculation version: verify width fixed at 6, MTP graph fixed at 3, suffix fills 0-3 tail slots. Cuts drafter cost from mtp6 to mtp3 immediately while keeping all async-schedule contracts static. Includes buffers, valid-length verify pseudocode, metrics, and a 4-phase plan."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [speculative-decoding, hybrid-speculation, mtp, suffix, vllm-ascend]
sources: []
original_url: ""
explored: false
confidence: medium
---

# Source: Fixed Verify-6 + Fixed MTP-3 + Dynamic Suffix-Tail(0..3)

Detailed design (2026-05-09) for the recommended first practical version of [[20260608-124000-hybrid-speculation-mtp-suffix-概念]].

## Core idea
Split the speculative chain: first 3 tokens always from a fixed `mtp3` graph, last 0-3 from suffix. Fixed: `K_verify=6`, `K_mtp_fixed=3`, scheduler/`future_input_map`/output-processor protocols. Dynamic: `K_suffix_tail in [0,3]`, so `K_eff = 3 + K_suffix_tail`. Layout `[mtp][mtp][mtp][suffix?][suffix?][suffix?]`.

## Why this version
- Directly fixes the biggest pain: drafter cost upper bound drops from `mtp6` to `mtp3`; suffix can't push it back up.
- Avoids the complexity of dynamic MTP: no multi-bucket graph, no graph manager, no per-batch MTP routing, no dynamic `K_mtp_exec`.
- Keeps every static contract (verify width, scheduler decode width, future_input_map slots, overlap protocol, output slicing) untouched.

## Gains & limits
Gain A: `T_mtp` from cost(mtp6) to cost(mtp3) (most certain). Gain B: suffix raises effective length 3 -> 3..6. Limit: even when suffix could supply 2-3 good tokens, MTP still runs 3 steps - forfeits "stronger suffix -> shorter MTP" (needs later dynamic MTP bucket upgrade). May degrade to `mtp3+verify6` with frequently-invalid tail, but still usually beats redundant `mtp6`.

## Implementation
Buffers: `draft_tokens[bs,6]`, `draft_valid_len[bs]`, `draft_source_map[bs,6]` (INVALID=0/MTP=1/SUFFIX=2). Verify stays width-6 but loops `pos<draft_valid_len` then breaks (valid-length-aware verify; "must-reject token" is a fragile short-term alternative). New modules: `drafter/hybrid_fixed_mtp_suffix.py`, `drafter/suffix_tail.py`, `spec_decode/fixed_mtp_suffix_policy.py`. Metrics: accept_len, effective_spec_len, suffix_tail_hit_rate, acceptance_by_position/source, T_mtp/T_suffix/T_verify (P50/P90/P99).

## Phases
P0 observability (quantify mtp6 cost) -> P1 fixed mtp3 main line -> P2 dynamic suffix tail + valid-length verify -> P3 policy (when to add tail) -> P4 evolve to dynamic MTP bucket if suffix proves strong.

Feeds [[20260608-124000-hybrid-speculation-mtp-suffix-概念]]; follow-on to [[20260608-124100-speculators-hybrid-async-schedule-design-来源]].

## Counter-arguments / Data gaps
- Explicitly "not theoretically optimal," a complexity/stability tradeoff.
- Tail (pos 4-6) acceptance assumed but unmeasured; design only, no results.
