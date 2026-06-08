---
title: "Hybrid Speculation (MTP + Suffix, 混合投机)"
tldr: "Combine MTP draft tokens with suffix-decoding tokens in one speculative window on vLLM-Ascend. Core engineering insight: keep target-verify SHAPE static (K_verify_cap) while making only the effective length / source split dynamic via valid_len + source_map masks, so async-schedule and ACL graph contracts aren't broken. Real gain requires decoupling MTP execution length from verify length."
date_created: 2026-06-08
date_modified: 2026-06-08
type: concept
tags: [speculative-decoding, mtp, suffix, vllm-ascend, ascend, hybrid-speculation]
sources: ["[[20260608-124100-speculators-hybrid-async-schedule-design-来源]]", "[[20260608-124200-speculators-fixed-mtp3-suffix-tail-design-来源]]", "[[20260608-124300-speculators-combo-overhead-profiling-plan-来源]]"]
explored: false
confidence: medium
---

# Hybrid Speculation (MTP + Suffix)

**Hybrid speculation** mixes [[20260608-121100-mtp-multi-token-prediction-概念]] draft tokens with [[20260608-121600-ngram-suffix-decoding-概念]] (suffix) tokens inside a single speculative window, so cheap high-confidence suffix tokens can substitute for some expensive MTP draft steps. This is active engineering work on [[20260608-122850-vllm-ascend-实体]] `model_runner_v1`.

## The core engineering problem

In vLLM/vLLM-Ascend, `spec_num_tokens` is not just an algorithm parameter - it is baked into runtime **shape**: buffer width (`future_input_map`), graph capture metadata (verify/draft), scheduler `decode_input_tokens`, and the async-schedule/overlap protocol. Changing the speculative length at runtime breaks all four contracts.

**Resolution**: don't make every layer support arbitrary dynamic length. Instead fix a static **`K_verify_cap`** (e.g. 6) for verify/scheduler/buffer/graph, and express dynamism only as a **semantic mask** - `K_eff = K_suffix + K_mtp <= K_cap`, plus per-position `valid_len` and `source_map` (INVALID/MTP/SUFFIX). Verify runs at fixed width but only accepts the first `K_eff` positions (valid-length-aware verify).

## Why static shape isn't enough

Fixing `K_cap` gives *compatibility* but not *savings*: if the MTP graph still expands to `mtp6`, then `suffix2+mtp1` costs nearly as much as `mtp6` on the drafter side. So the real lever is **decoupling MTP execution length from verify length** - via MTP bucket graphs (`mtp{0,1,2,4}`), micro-graphs, or the simplest practical version:

**`Fixed Verify-6 + Fixed MTP-3 + Dynamic Suffix-Tail(0..3)`** - MTP always runs a fixed `mtp3` graph (drafter cost cut from mtp6 to mtp3), suffix fills only the 0-3 tail slots. Layout: `[mtp][mtp][mtp][suffix?][suffix?][suffix?]`. Recommended as the first PoC because it cuts the biggest waste with the least complexity.

## Layout convention
Only `prefix-suffix + tail-MTP` (or fixed-MTP + tail-suffix) chains are supported - no interleaving, no per-request trees, no mixed verify widths. This keeps verify a single continuous prefix chain and acceptance decomposable by position.

## A real performance puzzle
Measured on Qwen3.5-122B / Ascend 910B3: `MTP7+suffix3` (TPOT 17.64ms, accept 7.32) is ~18% *slower* than `MTP10` (TPOT 14.92ms, accept 7.77) at bs=8 - acceptance can't explain it. Prime suspects: metadata-builder sync (`eagle_proposer.py:691`), pre-replay `stream.synchronize()` (`acl_graph.py:210`), bookkeeping running before draft, and combo-path padding inflating `num_tokens_padded`. See [[20260608-124300-speculators-combo-overhead-profiling-plan-来源]]. Relates to [[20260608-121800-dynamic-speculation-length-概念]].

## Counter-arguments
- Suffix tail (positions 4-6) is harder to predict than the prefix - higher `K_eff` may not raise real acceptance.
- The simple fixed-mtp3 version forfeits the "stronger suffix -> shorter MTP" optimum.

## Data gaps
- The 18% combo slowdown root cause is unresolved (profiling plan only, no measured breakdown yet).
- No production results for the proposed phased rollout.
