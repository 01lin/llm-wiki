---
title: "Source: TokenSpeed 架构深度分析"
tldr: "Code-level architecture analysis (2026-05-25) of TokenSpeed: 4-layer stack (Entrypoint/Scheduler-C++/Modeling-local-SPMD/Kernels), C++ FSM request lifecycle with Retract, 3-axis kernel selection, 3-level KV (GPU/host/L3) with RadixTree prefix cache, and execution timing diagrams for prefill/decode/retract. Includes a vLLM/SGLang comparison matrix."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [tokenspeed, architecture, inference, scheduler, kv-cache, blackwell]
sources: []
original_url: ""
explored: false
confidence: high
---

# Source: TokenSpeed 架构深度分析

Code-level analysis (2026-05-25) of the [[20260608-131000-tokenspeed-实体|TokenSpeed]] repo (Preview/LightSeek Foundation, Blackwell/agentic).

## Repo layout
`python/tokenspeed` (runtime + CLI), `tokenspeed-kernel` (kernel abstraction: registry/selection/platform + ops + thirdparty TRT-LLM/FlashInfer/DeepGEMM/CuTe DSL), `tokenspeed-scheduler` (C++ via nanobind: scheduler/fsm/resource), `tokenspeed-mla` (Blackwell MLA).

## Four layers
1. Entrypoint: AsyncLLM + SMG gateway.
2. Scheduler (C++): `NextExecutionPlan()` -> ExecutionPlan; lifecycle Submitted->Prefilling->PrefillDone->Decoding->Draining->Finished, plus Retracting->Retracted->LoadBack.
3. Modeling (local-SPMD): static compiler auto-generates collectives from module boundaries; runs FlatForwardOperation (mixed prefill+decode batch).
4. Kernels: KernelRegistry -> `select_kernel()` cached by family+mode+dtype+arch+objective+features+traits.

## Key mechanisms
- **FSM + Retract**: under GPU OOM, async write decoding KV to host (Retract), reload later (LoadBack) - avoids losing requests; key for long-context agentic.
- **3-axis kernel scoring**: `(oracle [0,20), objective {0,1}, priority [0,20))` lexicographic; bands REFERENCE=0/PORTABLE=4/PERFORMANT=8/SPECIALIZED=12/PLUGIN=16; hot path = one dict lookup.
- **3-level KV**: L1 GPU / L2 host / L3 persistent (WIP), indexed by [[20260608-131400-radix-tree-prefix-cache-概念|RadixTree]] (page rolling-hash); HybridPrefixCache co-manages KV + Mamba.

## vs vLLM/SGLang (highlights)
C++ FSM scheduler (compile-time type safety) vs Python engines; local-SPMD auto-comm vs hand-written TP/PP/EP; 3-level KV vs 2-level; Retract vs preemption-recompute; native P/D (kP/kD/kFused) vs experimental. Weaknesses: ecosystem maturity, narrow HW (Blackwell), WIP features (P/D/VLM/EPLB/metrics), harder-to-hack C++ scheduler.

Feeds [[20260608-131000-tokenspeed-实体]], [[20260608-131200-tokenspeed-vs-vllm-ascend-comparison-来源]].

## Counter-arguments / Data gaps
- Architectural analysis from a code read; few public benchmarks (chunked prefill etc.).
- Preview status - feature completeness claims reflect README "WIP" markers.
