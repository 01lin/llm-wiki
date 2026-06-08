---
title: "Source: MindSpeed-LLM DeepSeek4 SFT 与 PP 并行适配分析"
tldr: "Code analysis (2026-05-28): MindSpeed-LLM implements DeepSeek-V4-Flash fixed-length pretrain; SFT exists as preview (DeepSeek4SFTTrainer via posttrain_gpt.py, full-finetune marked DOING). No Pro-specific scripts. The Flash SFT example uses plain non-interleaved 1F1B (PP=4, no VPP/dualpipev) + MHC tensor-shape patch + noop layer 43. Catalogs 8 PP schedule options and what's DeepSeek4-specific vs generic MindSpeed."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [mindspeed-llm, deepseek, deepseek-v4, sft, pipeline-parallel, mhc, training, ascend]
sources: []
original_url: ""
explored: false
confidence: high
---

# Source: MindSpeed-LLM DeepSeek4 SFT 与 PP 并行适配分析

Code analysis (2026-05-28) of [[20260608-134000-mindspeed-llm-实体|MindSpeed-LLM]] for DeepSeek-V4 training.

## Status
The repo's explicit target is **DeepSeek-V4-Flash** fixed-length pretrain; Flash and Pro were released together but only Flash is implemented. SFT exists: `tune_deepseek4_flash_4k_A3_ptd.sh` -> `posttrain_gpt.py` (`--stage sft --prompt-type deepseek4`) -> `AutoTrainer` -> `DeepSeek4SFTTrainer` -> builds `DeepSeek4Model`. But the README capability matrix still marks full fine-tune **DOING** and there are **no Pro-specific** scripts/CI. Conversion (`model_cfg.json`): `deepseek4_base` (inherits deepseek32; pack_mla, multi_latent_attention, qk_layernorm, router_bias, enable_dsa_indexer, first_k_dense_replace=0) and `deepseek4` (adds tie_mtp_embeddings_and_lmhead) - one generic type, not Flash/Pro-named.

## PP schedule options (8)
1. **Plain non-interleaved 1F1B** - Flash SFT actual path (PP=4, no VPP/dualpipev). 2. **VPP/interleaved 1F1B** (`--num-layers-per-virtual-pipeline-stage`) - POC pretrain used PP=2/VPP=11. 3. **DualPipeV** (`--schedules-method dualpipev`) - framework + DeepSeek4 converter handle the layout (vpp_size=2, MTP/post weights at pp_rank0/vpp_last) but no DeepSeek4 SFT validation. 4. **RiPipe** (recompute-in-advance) - generic. 5. **Noop layers** (Flash uses `--noop-layers 43` with NUM_LAYERS=44 to pad for even PP split). 6. **num-layer-list** (uneven split; mutually exclusive with VPP/dualpipev/noop). 7. **U-shape LDT** (`--layerwise-disaggregated-training`) - Qwen2.5/Qwen3 only, no MoE. 8. P2P/SendRecv/unaligned-pipeline comm optimizations - generic.

## The DeepSeek4-specific PP adaptation
[[20260608-134300-mhc-multi-head-channel-概念|MHC]] is the biggest: with `--enable-mhc`, PP-sent activation goes `[S,B,H] -> [S,B,hc_mult,H]`; `MHCFeature` patches `get_tensor_shapes`, and under VPP patches `forward_backward_pipelining_with_interleaving` -> the MHC interleaving variant. MHC activation shape `(seq, micro_batch, hc_mult, hidden)`.

## Pro SFT recommendation
Reuse the `deepseek4` generic path (model type, conversion, template, trainer all keyed on `deepseek4`, not "flash"). Step 1: load Pro weights via generic conversion, minimally edit the Flash script (swap weights/tokenizer/model-scale/parallel-scale). Step 2: first version plain 1F1B (no VPP/dualpipev), keep MHC patch + noop layer, validate tokenizer/loss-mask/MTP-loss/MHC-shape/MoE-EP. Step 3: then VPP; dualpipev needs separate checkpoint-layout/MTP-placement/embedding-share validation. Re-confirm Pro: num_layers, hidden, heads, experts, EP/TP/PP, compress_ratios, MTP layers, HF keys, FP8 dequant, tokenizer tokens.

Relates to [[20260608-134200-mindspeed-llm-deepseek4-architecture-parallel-来源]], [[20260608-125100-deepseek-v4-pro-vllm-ascend-gap-analysis-来源]].

## Counter-arguments / Data gaps
- "SFT supported" overstates it - preview scripts exist, capability matrix says DOING; no Pro validation.
- DualPipeV/VPP for DeepSeek4 SFT are framework-present but unvalidated.
