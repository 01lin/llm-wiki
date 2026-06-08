---
title: "Source: DeepSeek-V4 (Flash/Pro) 结构与 vLLM-Ascend 适配缺口分析"
tldr: "Code-level analysis (2026-04-29) of DeepSeek-V4-Flash vs Pro configs and the gaps blocking Pro on vLLM-Ascend. Pro's index_topk=1024 hits a hard 512 cap in both operator (TOPK_LIMIT) and Python; compress_ratios is overwritten by a hardcoded dev array masking real HF config mismatches; KV spec binds C4/C128 by layer parity not per-layer ratios. Fix needs framework + kernel changes, ~4-6 person-weeks."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [deepseek, deepseek-v4, dsa, vllm-ascend, ascend, sparse-attention, moe]
sources: []
original_url: "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro"
explored: false
confidence: high
---

# Source: DeepSeek-V4 (Flash/Pro) 结构与 vLLM-Ascend 适配缺口

Code-level analysis (2026-04-29) of DeepSeek-V4 architecture and the gaps blocking DeepSeek-V4-Pro on [[20260608-122850-vllm-ascend-实体]]. Consolidates three raw notes (v4-pro gap analysis, operator-to-module table, and a forward-flow mermaid diagram).

## Flash vs Pro config
| dim | Flash | Pro |
|---|---|---|
| hidden_size | 4096 | 7168 |
| num_hidden_layers | 43 | 61 |
| num_attention_heads | 64 | 128 |
| q_lora_rank | 1024 | 1536 |
| n_routed_experts | 256 | 384 |
| index_topk | 512 | **1024** |
| compress_ratios (first 2) | [0,0,...] | [128,128,...] |

## The blocking gaps for Pro
1. **Operator Top-K cap**: `sparse_attn_sharedkv` requires sparse-index last dim == `TOPK_LIMIT=512` (csrc tiling). Pro's 1024 doesn't close at the kernel level - the deepest constraint (see [[20260608-125000-deepseek-sparse-attention-dsa-概念]]). `quant_lightning_indexer` `sparse_count=512` likewise.
2. **Python hardcodes 512**: `dsa_v1.py` sets `index_topk=512`, `self.index_topk=512`, ignoring `hf_config.index_topk`. Framework + operator must change together.
3. **compress_ratios overwritten**: `DeepseekV4Config.__init__` reassigns a hardcoded dev array `[1,1,4,128,...]`, overriding values read from `config.json` - so even with HF Pro/Flash weights the in-memory pattern is wrong. This *masks* problems (e.g. `initialize_kv_state` assumes a `compress_ratio==1` layer; real HF Flash has only 0/4/128 -> would KeyError).
4. **KV spec by layer parity**: `model_runner_v1` binds C4/C128 by `layer_id % 2` and forces layers 0/1 into `runner_only`, not by per-layer `compress_ratios` - conflicts with Pro's first-two-layers=128 and Flash's first-two=0.
5. **`compress_ratio==0`** (in both HF Flash/Pro) falls into the "C128-like" else branch with no dedicated handling - runtime risk.

## Forward flow (DSA)
Embedding -> mHC pre (`npu_hc_pre`) -> input_layernorm -> DSA attention (Q: wq_a/q_norm/wq_b/RoPE; SWA KV branch; compress_ratio switch: 1=SWA-only, 4=C4 indexer/compressor/lightning_indexer, 128=C128 compressor -> `npu_sparse_attn_sharedkv` with topk) -> O projection -> mHC post -> MoE (router + SharedFusedMoE) -> hc_head -> final RMSNorm. Operator table maps each step to `torch.ops._C_ascend` / `torch_npu` ops.

## Effort estimate
Framework-only (config + per-layer KV spec + DSA Python using `hf_config.index_topk` + fix `compress_ratio==0` + E2E): ~1-2 engineers x 1-2 weeks. Plus operator (relax/reconfigure `sparse_attn_sharedkv` Top-K, sync `quant_lightning_indexer`, regress A3 BF16/FP8): +2-4 weeks, gated on CANN/kernel scheduling.

Relates to [[20260608-125000-deepseek-sparse-attention-dsa-概念]], [[20260412-194210-deepseek-实体]], [[20260412-194111-quantization-概念]].

## Counter-arguments / Data gaps
- The first raw note opens mid-sentence (truncated/garbled header) - reconstructed from context.
- One repo snapshot; later versions may already lift the 512 cap.
- No measured V4-Pro accuracy/perf (blocked).
