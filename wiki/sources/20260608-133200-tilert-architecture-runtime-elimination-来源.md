---
title: "Source: TileRT 总体架构与关键优化点分析"
tldr: "Code-level analysis (2026-05-27) of TileRT (open-source Python control plane only; engine is closed libtilert.so). Decode = `prepare_money` (capture native runtime, bake sampling into the plan) then repeated `show_hands`. Key: 51-slot temp_vars ABI, continuous address-stable storage, offline weight swizzle (FP8/FP16 MMA), fused ops, device-side MTP (accept length ~2.77), fixed 8-device topology."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [tilert, architecture, low-latency, mtp, deepseek, blackwell, runtime]
sources: []
original_url: ""
explored: false
confidence: high
---

# Source: TileRT 总体架构与关键优化点分析

Code-level analysis (2026-05-27) of the [[20260608-133400-tilert-实体|TileRT]] repo (DeepSeek-V3.2 path).

## Open-source boundary
Python = model structure + weight conversion + fixed-ABI buffer prep + native-call wrappers + generation/benchmark control. The engine is the closed `libtilert.so`, loaded via `torch.ops.load_library`, called as `torch.ops.tilert.*` (e.g. `dsa_show_hands_prepare_money`, `dsa_show_hands`, `rmsnorm_proj_qa_kva_ki_op`, `expert_select_up_gate_silu_op`, `top1_allreduce_op`). One-liner: TileRT turns the decode step into a fixed-shape/fixed-address/native-driven pipeline + MTP for more tokens per forward - see [[20260608-133500-runtime-elimination-fixed-decode-概念]].

## Execution
`DSAv32Generator` (batch=1; MTP -> mtp_seq_len=4) -> `ShowHandsDSALayer.from_pretrained` loads per-device weights (`dev_{id}`), builds `Dsa` module, collects params/caches/temp_vars into **continuous storage** (1024-byte-aligned views on one uint8 tensor), then per-device `prepare_money` (params + temp_vars + cache_vars + profile_logs + max_seq_len + with_mtp). Sampling params are baked into the captured plan at prepare time -> changing them needs teardown + recapture. Decode loop calls native `show_hands(token_id.cpu(), ...)`; MTP decode reads ACCEPTED/PREDICTED/NEXT_DRAFT tokens, `cur_pos += num_accepted`.

## temp_vars ABI (51 fixed slots)
Q/KV/KI, attention/MoE intermediates, LOGITS_OUT/TOKEN_OUT, CUR_POS/TOKEN_ID, DRAFT/PREDICTED/ACCEPTED/NEXT_DRAFT_TOKENS, X_QUANT/X_SCALE, MTP outputs, SAMPLING_SEED/CONFIG, TOP_P. Python and native share state by fixed index - no dynamic lookup; easy to capture.

## Model & weights
`Dsa` = dense `MlpBlock` layers then `MoeBlock` layers (each with `Mla`) + `RMSNormHeadProj`; MTP = `MTPPreprocessLayer` + `MoeBlock` + head. Each op has a WeightsConverter doing offline reshape/transpose/MMA-swizzle/FP8-FP16-BF16 selection/expert reorder - "weights become kernel-friendly offline, not in the hot path."

## Key optimizations
Native persistent runtime (less Python/dispatch overhead), prepare_money pre-capture, fixed temp_vars ABI, continuous storage (address stability), fused ops (fewer launches + HBM round-trips), weight swizzle/FP8MMA (tensor-core utilization), MTP (effective_tps = accepted/decode_time), fused allreduce (less exposed comm), profile logs (feedback for self-optimization).

Feeds [[20260608-133400-tilert-实体]], [[20260608-133500-runtime-elimination-fixed-decode-概念]]; Ascend transfer in [[20260608-133100-vllm-ascend-tilert-like-self-evolving-loop-来源]].

## Counter-arguments / Data gaps
- Native source closed - tile scheduler / graph capture / kernel fusion can't be audited.
- Hardware-bound (8x B200, num_devices=8 hardcoded); not general.
- Python generation still has demo sync points (`.item()`, `.cpu()`) that an extreme path would push device-side.
