---
title: "Source: vLLM / vLLM-Ascend 投机解码算法支持度分析"
tldr: "Code-level analysis of vLLM + vLLM-Ascend speculative decoding (2026-04-15). Catalogs 9 methods, 14 MTP model types, the v1 spec_decode module layout, and how Ascend reuses the EAGLE proposer for MTP. GLM-4.x and Qwen3.5 MTP fully supported; NPU has MTP<=15 and capture-size limits."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [vllm, vllm-ascend, speculative-decoding, mtp, ascend]
sources: []
original_url: "https://github.com/vllm-project/vllm + https://github.com/vllm-project/vllm-ascend (main)"
explored: false
confidence: high
---

# Source: vLLM / vLLM-Ascend 投机解码算法支持度分析

Code-level analysis (2026-04-15) of speculative-decoding support across vLLM main and vLLM-Ascend main, focused on GLM-4.x / Qwen3.5 MTP support, effectiveness, and roadmap.

## Methods (`speculative_config.method`)
`mtp`, `eagle`, `eagle3` (EagleProposer), `draft_model`, `medusa`, `ngram`/`ngram_gpu`, `suffix`, `dflash`, `mlp_speculator` (v1 disabled), `extract_hidden_states`. See [[20260608-120000-speculative-decoding-概念]] for the family overview.

## MTP coverage
14+ types: `deepseek_mtp` (V3/V3.2), `glm4_moe_mtp` (GLM-4.5/4.6), `glm4_moe_lite_mtp` (GLM-4.7), `glm_ocr_mtp`, `qwen3_5_mtp`, `qwen3_next_mtp`, `mimo_mtp`, `ernie_mtp`, `nemotron_h_mtp`, `exaone_moe_mtp`, `longcat_flash_mtp`, `step3p5_mtp`, `pangu_ultra_moe_mtp`. Auto-detection rewrites `speculative_config` based on `model_type`/`architectures`.

## Module layout
`vllm/v1/spec_decode/`: `eagle.py` (core, >500 lines), `draft_model.py`, `medusa.py`, `ngram_proposer(_gpu).py`, `suffix_decoding.py`, `dflash.py`, `metadata.py`, `metrics.py`.

## vLLM-Ascend enhancements
- `vllm_ascend/spec_decode/`: routes `ngram/suffix/medusa/eagle/eagle3/mtp/draft_model` to Ascend proposers. **Key finding: MTP reuses `AscendEagleProposer`** - shares NPU tree attention, ACL Graph.
- `sample/rejection_sampler.py`: NPU Triton kernels (`rejection_greedy_sample_with_triton`, `rejection_random_sample_kernel`).
- Worker patches: `patch_deepseek_mtp.py`, `patch_draft_quarot.py`, `patch_rejection_sampler.py`.
- TP + speculative-parallel split, but `draft_tensor_parallel_size` limited to 1.

## Known limits
- Pipeline parallelism incompatible with SD in vLLM <= 0.15.0.
- NPU `num_speculative_tokens` max **15** (`npu_fused_infer_attention_score` int limit).
- Fullgraph: capture sizes must be multiples of `(K+1)`.
- DeepSeek MTP v3.2 needs `enforce_eager=True` (cudaGraph issue).

## Architecture detail
GLM4 MoE MTP (`glm4_moe_mtp.py`, 366 lines): enorm/hnorm -> eh_proj (2H->H) -> Glm4MoeDecoderLayer (MHA + FusedMoE) -> shared_head. Qwen3.5 MTP (`qwen3_5_mtp.py`, 452 lines): fc (2H->H) -> reuses `Qwen3_5DecoderLayer`, supports torch.compile + multimodal.

Feeds [[20260608-121100-mtp-multi-token-prediction-概念]], [[20260608-121200-eagle-概念]], [[20260608-121300-dflash-概念]].

## Counter-arguments
- A code-reading analysis, not a benchmark - "fully supported" means registered + wired, not necessarily performant (see [[20260608-120900-qwen3-5-mtp-ascend-适配分析-来源]] for the gap between support and stable speedup).

## Data gaps
- No measured acceptance rates or throughput numbers; effectiveness scores (1-5) are the author's judgement.
- `mlp_speculator` disabled in v1 - status unclear.
