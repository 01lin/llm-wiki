---
title: "Source: SGLang on Ascend NPU - 部署指南与推理性能"
tldr: "Deployment guide + official best-practice perf data (2026-05-24) for SGLang on Ascend A2/A3. Docker quickstart, key env vars (MLAPO, FIA_NZ, SPEC_V2, DeepEP), and launch configs for DeepSeek-V3/R1/V3.2, GLM-5/5.1, Qwen3 series with NEXTN spec decoding. Perf: DeepSeek-R1 19-20ms TPOT (A3 32-card PD), Qwen3-8B 5ms, Qwen3.5-397B 22ms; DeepEP low-latency sub-150us."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [sglang, ascend, npu, deployment, performance, deepseek, qwen, glm, prefill-decode-disaggregation]
sources: []
original_url: ""
explored: false
confidence: high
---

# Source: SGLang on Ascend NPU - 部署指南与推理性能

Deployment guide + official best-practice performance data (2026-05-24), sourced from `sglang/docs/platforms/ascend/` and `sgl-kernel-npu`.

## Environment
Hardware: Atlas 800I A2 (Ascend 910B, 64Gx8), Atlas 800I A3 (910_9382, 64Gx16). Software: Ascend HDK 25.0.RC1.1, CANN 8.3.RC1/8.5.0, PyTorch>=2.5.1, torch-npu>=2.5.1-7.0.0. Docker is the recommended path (`quay.io/ascend/sglang:main-cann8.5.0-a3` / `-910b`).

## Key optimization knobs
Env: `SGLANG_NPU_USE_MLAPO=1` (fused MLA preprocess), `SGLANG_USE_FIA_NZ=1`, `SGLANG_ENABLE_SPEC_V2=1` + `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` (spec/forward overlap), `STREAMS_PER_DEVICE=32`, `HCCL_OP_EXPANSION_MODE=AIV`, `HCCL_BUFFSIZE` (decode ~650-720, prefill ~1536), `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`. Launch: `--attention-backend ascend`, `--device npu`, `--moe-a2a-backend deepep`, `--deepep-mode auto/normal/low_latency`, `--quantization modelslim`, `--speculative-algorithm NEXTN` (NextN = [[20260608-121100-mtp-multi-token-prediction-概念]]-style, recommended on NPU; typical `--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`).

## Deployment patterns
DeepSeek-V3/R1 single-machine A3 mixed-PD (W4A8), dual-machine A3 [[20260412-194111-prefill-decode-disaggregation-概念]] (W8A8, separate prefill/decode/router). GLM-5/5.1 (DeepSeek-V3/V3.2 architecture = [[20260608-125000-deepseek-sparse-attention-dsa-概念]] + MTP, "0Day" NPU support, needs transformers 5.3.0). Qwen3-235B / Qwen3.5-397B.

## Performance (official best practice, A3)
- DeepSeek-R1: 19-20ms TPOT (32-card PD, W8A8, 3.5-6K in / 1-1.6K out); DeepSeek-V3.2: 26ms at 128K+1K.
- Qwen3-235B-A22B: 10ms (8-card, 11K+1K, BF16); Qwen3-8B: 5ms (1-card, W8A8); Qwen3-32B: 6ms (8-card, 18K+4K); Qwen3.5-397B-A17B: 22ms (8-card, W4A8).
- DeepEP-Ascend (A3 384 SuperPOD, 4096 tokens, top-8): Normal 8-way 146/125 GB/s dispatch/combine, dropping to 57/81 at 128-way; Low-Latency 8-way 132us/126us (sub-150us).
- Supports DeepSeek-V3/V3.1/V3.2, R1, Qwen3 series, GLM-5/4.5, Kimi-K2-Thinking, ERNIE-4.5, Llama-4, plus VLMs (Qwen-VL, GLM-4.5V, DeepSeek-VL2).

Relates to [[20260608-125200-sglang-vs-sgl-kernel-npu-对比分析-来源]], [[20260608-122850-vllm-ascend-实体]].

## Counter-arguments / Data gaps
- Perf numbers are vendor best-practice (tuned configs), not independent reproductions; no concurrency/goodput dimension shown alongside TPOT.
- Acceptance lengths for NEXTN spec decoding not given here.
