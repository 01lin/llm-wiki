---
title: "Source: TileRT vs vLLM / SGLang - 差异与 runtime 消除机制"
tldr: "Comparison (2026-05-27): vLLM/SGLang are general online-serving runtimes (dynamic scheduling, continuous batching, KV management, multi-model) that keep the runtime resident in the per-step loop; TileRT is a model/hardware-specific native decode runtime that moves those decisions out of the hot path into prepare/capture/weight-conversion. 'Runtime elimination' = remove dynamic per-token decisions, not all runtime. Decode step breakdown shows which time terms TileRT cuts."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [tilert, vllm, sglang, comparison, low-latency, runtime, decode]
sources: []
original_url: ""
explored: false
confidence: high
---

# Source: TileRT vs vLLM / SGLang - runtime 消除机制

Comparison (2026-05-27) of [[20260608-133400-tilert-实体|TileRT]] vs [[20260412-194210-vllm-实体|vLLM]] / [[20260412-194210-sglang-实体|SGLang]].

## The core distinction
vLLM/SGLang are **general online-serving runtimes**: dynamic request management, continuous batching, KV cache management, prefix cache, spec decode, LoRA, structured output, multimodal, PD disaggregation. They optimize the hot path (CUDA graph, cache, scheduler) but keep the general runtime resident, so every step still does many dynamic decisions. TileRT is a **model/hardware/latency-specific native decode runtime**: model structure, weight layout, buffers, KV cache, MTP state, sampling state, and the native execution plan are all fixed up front; the decode hot path keeps only a thin control input, then `libtilert.so` runs the whole step.

> Not "who has CUDA graph," but: vLLM/SGLang optimize a general runtime in place; TileRT removes the general runtime from the low-latency decode hot path. See [[20260608-133500-runtime-elimination-fixed-decode-概念]].

## "Runtime elimination" = decisions moved earlier
Not zero runtime - the dynamic per-token work is pushed to init/compile/weight-conversion/native-prepare. Eliminated/weakened from the hot path: dynamic request scheduler, dynamic batch construction, dynamic KV block allocation, dynamic attention-metadata build, dynamic graph op scheduling, frequent tensor allocation, generic sampler/logprob/grammar, dynamic weight-layout adaptation, some CPU/GPU sync, exposed communication.

## Decode-step time decomposition
`T_total = T_scheduler + T_batch_prepare + T_kv_mapping + T_metadata + T_graph_or_dispatch + T_kernel + T_memory_io + T_communication + T_sampler + T_sync + T_output_update`. TileRT cuts T_scheduler (batch=1 fixed loop), T_batch_prepare (fixed temp_vars/MTP seqlen), T_kv_mapping (fixed cache vars), T_metadata (native fixed ABI), T_graph_or_dispatch (prepare_money then show_hands), T_kernel_launch (fused native op), T_memory_io (fused + continuous buffer), T_communication (fused down/allreduce), T_sampler (fixed sampling state). Real gain = lower step latency x higher accepted-tokens-per-step; MTP is the multiplier (e.g. 3ms/accept1 ~333 tok/s vs 5ms/accept2.7 ~540 tok/s).

## Trade-offs
Gives up general model support, dynamic batch, multi-tenant serving, dynamic sampling, dynamic shapes, hardware independence, runtime auditability (closed native). Gains lowest decode overhead, deep fusion, stable native replay, MTP-native acceleration, higher single-request tok/s.

Feeds [[20260608-133400-tilert-实体]], [[20260608-133500-runtime-elimination-fixed-decode-概念]]; Ascend recreation in [[20260608-133100-vllm-ascend-tilert-like-self-evolving-loop-来源]].

## Counter-arguments / Data gaps
- vLLM/SGLang's "generality cost" is a feature for production serving - the comparison is latency-specialization vs general-serving, not better/worse.
- TPS example numbers are illustrative, not measured side-by-side.
