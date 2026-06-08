---
title: "Source: TokenSpeed vs vllm-ascend 方案设计对比分析"
tldr: "Deep comparison (2026-05-24): TokenSpeed = from-scratch engine (10K C++ scheduler + 100K Py runtime + 44K Py kernel) vs vllm-ascend = vLLM hardware adapter (95K Py + 220K C++/AscendC, 963 kernel files). TokenSpeed wins control-plane latency (<1ms vs 5-20ms), KV management, placement compiler; vllm-ascend wins kernel coverage, HW breadth (A2/A3/310p/A5), ecosystem. Recommended TokenSpeed-Ascend path: reuse vllm-ascend kernels + keep TokenSpeed C++ scheduler."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [tokenspeed, vllm-ascend, comparison, scheduler, kernel, ascend, moe]
sources: []
original_url: ""
explored: false
confidence: high
---

# Source: TokenSpeed vs vllm-ascend 对比分析

Design comparison (2026-05-24) of [[20260608-131000-tokenspeed-实体|TokenSpeed]] and [[20260608-122850-vllm-ascend-实体|vLLM-Ascend]].

## Positioning
TokenSpeed: a complete new engine (Scheduler + Runtime + Kernel + MLA), ~10K C++ + 100K Python runtime + 44K Python kernel, "TRT-LLM perf + vLLM usability," fully self-evolving. vLLM-Ascend: a hardware backend adapter for vLLM (monkey-patch injection), ~95K Python + **220K C++/AscendC (963 kernel files)**, must follow vLLM upstream cadence. "Building from foundation" vs "installing an elevator in an existing building."

## Scheduler (biggest difference)
TokenSpeed C++ FSM (10K LOC, `std::variant` type-safe states, RAII KV ownership, <1ms/step, exposes NextExecutionPlan/Advance) vs vLLM-Ascend Python scheduler (inherits vLLM, subclasses for profiling/recompute/dynamic-batch, 5-20ms/step under GIL). TokenSpeed uses [[20260608-131400-radix-tree-prefix-cache-概念|RadixTree]] (O(prefix_len)) + Retract state machine vs vLLM's block-hash trie + recompute.

## Kernel & parallelism
TokenSpeed: `@register_kernel` with priority bands + plugin entry_points + trait selection + auto numerics/benchmark; vendor-neutral public API. vLLM-Ascend: AttentionBackendEnum + 23 monkey-patches; 963 csrc kernels (FA3/SFA/DSA, lightning indexer, KV-quant sparse attn). Parallelism: TokenSpeed declarative Placement (REPLICATE/SHARD/PARTIAL) + static compiler auto-inserting CommOp (FusedReduceNormOp etc.) vs vLLM-Ascend imperative distributed primitives + flashcomm2.

## What each uniquely has
TokenSpeed: C++ scheduler, type-safe KV, placement compiler, Retract, tokenspeed-mla (fold_sq_factor), SMG AsyncLLM, plugin registry. vLLM-Ascend: 963 AscendC kernels, HCCL, ACLGraph, profiling/recompute/dynamic-batch schedulers, Mooncake KV connectors, EPLB multi-strategy, 310p/A5 support, NUMA CPU binding, DeepSeek-V3.2 indexer.

## TokenSpeed-Ascend recommendation
"Borrow, don't rebuild": (1) reuse vllm-ascend kernels via plugin (`vllm_ascend.ops.*`), don't write AscendC; (2) keep TokenSpeed's C++ scheduler (the real differentiator); (3) reuse vllm-ascend's HW engineering (HCCL/ACLGraph/NUMA); (4) don't chase kernel count - focus agentic critical path. Projected TokenSpeed-Ascend ~200-300 TPS (A3) vs vllm-ascend ~150-200; main edge is control-plane latency + KV mgmt, not MLA kernels.

Feeds [[20260608-131000-tokenspeed-实体]], [[20260608-131300-tokenspeed-ascend-self-evolving-loop-design-来源]]; relates to [[20260608-125000-deepseek-sparse-attention-dsa-概念]].

## Counter-arguments / Data gaps
- TPS estimates are projections; B200 540 TPS is cross-hardware vs A3.
- The note's own asides note "don't be fooled by LOC" - architectural leverage vs engineering depth is a judgement call.
