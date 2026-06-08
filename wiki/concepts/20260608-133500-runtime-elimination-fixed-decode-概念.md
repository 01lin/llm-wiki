---
title: "Runtime Elimination (Fixed Decode Pipeline)"
tldr: "Move dynamic inference-runtime decisions (request scheduling, batch construction, KV block mapping, attention metadata, op dispatch, tensor allocation, sampler/grammar) OUT of the per-token decode hot path and into init/compile/weight-conversion/native-prepare. The decode step becomes a fixed-shape, fixed-address, native-driven pipeline. TileRT's core idea; transferable to Ascend via ACLGraph + a fixed runtime-state table."
date_created: 2026-06-08
date_modified: 2026-06-08
type: concept
tags: [inference, low-latency, runtime, cuda-graph, decode, tilert]
sources: ["[[20260608-133300-tilert-vs-vllm-sglang-runtime-elimination-来源]]", "[[20260608-133200-tilert-architecture-runtime-elimination-来源]]"]
explored: false
confidence: medium
---

# Runtime Elimination (Fixed Decode Pipeline)

**Runtime elimination** is the principle behind [[20260608-133400-tilert-实体|TileRT]]: don't optimize the general runtime's hot path - **remove** the general runtime from the low-latency decode hot path. General engines ([[20260412-194210-vllm-实体|vLLM]]/[[20260412-194210-sglang-实体|SGLang]]) keep a full runtime resident in the per-step loop (request scheduling, dynamic batch construction, KV block/token-pool mapping, attention-metadata build, op dispatch, tensor allocation, generic sampler/grammar) - costs that persist even with CUDA Graph because they must support dynamic requests/batches/models. Runtime elimination pushes all of that to init / compile / weight-conversion / native-prepare, so the decode step is a **fixed-shape, fixed-address, native-driven pipeline**.

## What gets eliminated from the hot path
Dynamic scheduler, batch construction/padding, KV block mapping, attention metadata, PyTorch op dispatch, frequent tensor allocation, generic sampler/logprob/grammar, dynamic weight-layout adaptation, and some CPU/GPU sync and exposed communication.

## What replaces it
A specialized native runtime with: a fixed ABI ([[20260608-133400-tilert-实体|temp_vars]] table), continuous address-stable buffers, a captured/fixed execution plan (CUDA Graph / native), offline-converted kernel-friendly weights, model-specific fused ops, and a device-side MTP state machine. The performance win is multiplicative: lower step latency x higher accepted-tokens-per-step (MTP). Even if an MTP step is heavier, a high mean accept length (e.g. 2.7) raises effective tok/s (e.g. ~333 -> ~540).

## Transfer to Ascend
Not by porting TileRT, but by recreating the method on [[20260608-122850-vllm-ascend-实体|vLLM-Ascend]]: keep the serving shell, add a low-latency special path = `prepare runtime state -> ACLGraph/NPUGraph capture -> replay fixed decode step`, an Ascend `temp_vars`-style state table, device-side MTP accept, AscendC/Triton fused MLA-DSA/MoE/sampler, offline weight conversion, MC2/HCCL overlap. Priority: D2H sync cleanup -> shape buckets + persistent buffers -> FULL_DECODE_ONLY graph replay -> device-side MTP -> DSA/MLA + MoE/MC2 overlap. See [[20260608-133100-vllm-ascend-tilert-like-self-evolving-loop-来源]].

## Counter-arguments / Data gaps
- Sacrifices generality: dynamic batch, multi-tenant serving, dynamic sampling params (need recapture), dynamic shapes, hardware independence.
- TileRT's native runtime is closed, so the internal tile scheduling can't be audited.
- Best for batch=1 low-latency; not a high-throughput serving design.
