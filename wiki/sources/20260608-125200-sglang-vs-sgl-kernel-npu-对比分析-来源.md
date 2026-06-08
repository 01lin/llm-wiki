---
title: "Source: SGLang 与 sgl-kernel-npu 对比分析"
tldr: "Analysis (2026-05-24) of how SGLang (full inference framework, CUDA-first) integrates Ascend NPU support via sgl-kernel-npu, a separate kernel library plugged in through SGLang's Platform/plugin system (torch.ops.npu, PrivateUse1). NPU kernels use AscendC + Triton + CATLASS; MoE comm via DeepEP-Ascend HCCS All-to-All. Not a fork - a hardware adapter layer."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [sglang, sgl-kernel-npu, ascend, npu, kernel, moe, deepep]
sources: []
original_url: ""
explored: false
confidence: high
---

# Source: SGLang 与 sgl-kernel-npu 对比分析

Code-repo analysis (2026-05-24) of [[20260412-194210-sglang-实体]] vs [[20260608-125400-sgl-kernel-npu-实体]].

## Relationship
SGLang is the full inference framework (scheduler, HTTP/gRPC serving, model execution, distributed), CUDA-first (ROCm/CPU/Metal via plugin). **sgl-kernel-npu** is SGLang's Ascend NPU **kernel library** - the NPU-side equivalent of `sgl-kernel` (the GPU kernel lib). It is a separately published whl, loaded via `torch.ops.load_library`, and plugged into SGLang through the standard Platform plugin system - **not a fork**, a hardware adapter that reuses SGLang's framework logic.

## Plugin mechanism
SGLang uses setuptools entry_points for OOT hardware: `sglang.srt.platforms` (register `SRTPlatform` subclass) and `sglang.srt.plugins`. `SGLANG_PLATFORM` selects platform. `SRTPlatform` exposes `get_default_attention_backend`, `get_graph_runner_cls` (NPU -> NPUGraph), KV pool classes, `get_compile_backend` (NPU -> `npugraph_ex`). SGLang main repo already has ~732 NPU references (e.g. `NPUPiecewiseBackend` swapping CUDAGraph for `torch.npu.NPUGraph`).

## NPU kernel stack
- **AscendC** (Huawei's CUDA-C++-like operator language) for high-perf kernels; **Triton** for most Python-side fused kernels (portable, NPU Triton backend); **CATLASS** (Huawei's CUTLASS equivalent); registered to `torch.ops.npu` via `TORCH_LIBRARY_IMPL(npu, PrivateUse1)`.
- Two subsystems: **DeepEP-Ascend** (MoE expert-parallel All-to-All over HCCS; A3 full-mesh, A2 HCCS+RDMA; Low-Latency mode <150us for decode) and the kernel lib (MLA decode, GQA, MLA preprocess fused, RMSNorm, LoRA, speculative tree, Mamba conv1d, FLA/gated-delta-rule, lightning indexer).

## GPU vs NPU mapping
MLA decode: CUTLASS SM100 -> AscendC+Triton Paged MLA. MLA preprocess: stepwise -> AscendC end-to-end fused (RMSNorm->Dequant->MatMul->RoPE->Cache). AllReduce: NCCL/MSCCLPP -> HCCS via DeepEP. Graph: CUDA Graph -> NPU Graph.

Relates to [[20260608-125300-sglang-ascend-deployment-perf-来源]], [[20260608-122850-vllm-ascend-实体]] (sibling Ascend serving stack).

## Counter-arguments / Data gaps
- Architectural mapping, not a performance comparison vs GPU or vs vLLM-Ascend.
- "Logic equivalent" Triton MLA claims aren't benchmarked here.
