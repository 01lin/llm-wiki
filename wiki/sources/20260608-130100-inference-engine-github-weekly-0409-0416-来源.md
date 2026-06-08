---
title: "Source: 开源推理引擎周报 (2026-04-09 ~ 04-16)"
tldr: "Weekly scan of vLLM (v0.19.0), SGLang (v0.5.10), vLLM-Ascend (v0.18.0rc1). Hottest PR: vLLM TurboQuant 2-bit KV cache (4x capacity, 101 comments). 7 trends: extreme KV compression, microsecond kernel tuning, spec-decode maturing (DSL+DFlash+graph), SGLang's diffusion+RL lead, elastic-EP fault tolerance, transformers v5 migration, NPU still 1-2 versions behind."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [vllm, sglang, vllm-ascend, inference, speculative-decoding, kv-cache, moe, weekly]
sources: []
original_url: ""
explored: false
confidence: high
---

# Source: 开源推理引擎周报 (2026-04-09 ~ 04-16)

Self-compiled weekly report scanning the three major inference engines.

## Releases
- **vLLM v0.19.0** (2026-04-03): 448 commits/197 contributors; zero-bubble async + spec decode; MRV2 mature; TurboQuant 2-bit KV; CPU KV offload; DBO generalization; B300/GB300.
- **SGLang v0.5.10.post1** (2026-04-09): piecewise CUDA Graph default; elastic EP partial fault tolerance; HiSparse; SGLang-Diffusion; FA4; MXFP8; GLM-5/DeepSeek-V3.2 opt.
- **vLLM-Ascend v0.18.0rc1** (2026-04-01): DeepSeek-V3.1 C8; A5 chip; Flash Comm V1 VL+MLA; Triton compile opt; GDN prefill opt.

## Notable PRs
- **vLLM #38479 TurboQuant** (hottest, 101 comments): 2-bit KV, 4x capacity, online (no offline calibration); k8v4 ~2.6x, 3-bit up to 4.9x; GSM8K 0.78-0.86 vs 0.90 baseline. See [[20260608-130300-kv-cache-compression-概念]].
- vLLM #39773 (disable piecewise cudagraph fallback for eagle draft, MRV2), #38372 (hybrid-model accepted-token counting), #36029 (SPEED-bench spec eval CLI).
- SGLang #22392 (CUTLASS FP8 GEMM eliminates ~2.2ms memset bubbles), #21985 (eliminate attention DtoD copy ~392us), #22604 (diffusion RL rollout API), #22217 (Eagle beta test failure, 112 comments - spec stability).
- vLLM-Ascend #7779 (fuse W4A8 dispatch+FFN+combine), #8004 (Qwen3.5 MoE flashcomm shared-expert MTP fix), #7945 (MRV2 full graph fixes).

## Roadmap items
vLLM: full cudagraph drafter (#33341), DFlash parallel drafting (#32206), NGram-GPU (#29184), hybrid ngram-eagle (#24344), [[20260608-121800-dynamic-speculation-length-概念|DSL]] (#36657), MineDraft (#38003, +75%), proposer unification (#36219), multi-MTP (#31204). Ascend: DFlash (#8188/#8118, 3 competing PRs), Eagle3 graph (#5459).

## 7 trends
1. **Extreme KV compression** is the new battleground (TurboQuant 2-bit vs SGLang HiCache tiering - orthogonal, stackable).
2. **Microsecond kernel tuning** (memset bubbles, DtoD copies).
3. **Spec decode maturing**: DSL + DFlash + graph mode convergence is the next milestone.
4. **SGLang leads diffusion + RL** (inference+training platform).
5. **Elastic EP + fault tolerance** for large MoE.
6. **transformers v5 migration** (both repos' hottest ecosystem PRs).
7. **NPU still 1-2 versions behind** NVIDIA frontier (TurboQuant/FA4/CUTLASS FP8).

Relates to [[20260412-194210-vllm-实体]], [[20260412-194210-sglang-实体]], [[20260608-122850-vllm-ascend-实体]], [[20260608-120000-speculative-decoding-概念]], [[20260608-121300-dflash-概念]].

## Counter-arguments / Data gaps
- A curated weekly snapshot - PR selection is the author's judgement; comment counts proxy "heat," not importance.
- Release feature claims are from changelogs, not independently tested.
