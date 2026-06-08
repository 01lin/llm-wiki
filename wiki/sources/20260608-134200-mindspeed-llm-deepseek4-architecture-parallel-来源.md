---
title: "Source: MindSpeed-LLM DeepSeek V4 Flash/Pro 架构与 TP/PP/EP/CP 切分分析"
tldr: "Code analysis (2026-05-29): DeepSeek-V4 Flash/Pro share DeepseekV4ForCausalLM (MoE + G2 sparse shared-KV attention + MHC + MTP). Canonical Flash/Pro param table; how TP/PP/EP/CP shard each matrix (linear_q/kv TP-replicated, q_up/o Column/Row TP, EP shards routed experts, CP shards sequence). Pro = same deepseek4 spec, remap params (hidden 7168, 128 heads, 384 experts, o_groups 16, index_topk 1024) + revalidate."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [mindspeed-llm, deepseek-v4, architecture, tensor-parallel, pipeline-parallel, expert-parallel, context-parallel, mhc, dsa]
sources: []
original_url: "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base"
explored: false
confidence: high
---

# Source: MindSpeed-LLM DeepSeek V4 架构与 TP/PP/EP/CP 切分

Code analysis (2026-05-29) of [[20260608-134000-mindspeed-llm-实体|MindSpeed-LLM]]'s DeepSeek-V4 model + parallelism. Flash/Pro are the same `DeepseekV4ForCausalLM` mapped to `DeepSeek4Model` + `deepseek4_spec`.

## Canonical param table (training side)
| | Flash | Pro | MindSpeed arg |
|---|---:|---:|---|
| hidden | 4096 | 7168 | --hidden-size |
| layers | 43 | 61 | --num-layers (+noop) |
| heads | 64 | 128 | --num-attention-heads |
| KV heads | 1 | 1 | shared KV |
| head dim | 512 | 512 | --qk-head-dim |
| q LoRA | 1024 | 1536 | --q-lora-rank |
| o LoRA | 1024 | 1024 | --o-lora-rank |
| o groups | 8 | 16 | --o-groups |
| routed experts | 256 | 384 | --num-experts |
| experts/token | 6 | 6 | --moe-router-topk |
| MoE intermediate | 2048 | 3072 | --moe-ffn-hidden-size |
| index_topk | 512 | 1024 | --index-topk |
| MHC | hc_mult=4 | hc_mult=4 | --enable-mhc |
| MTP | 1 layer | 1 layer | --mtp-num-layers 1 |
| vocab | 129280 | 129280 | --vocab-size |

## Model topology
Per layer (see [[20260608-134300-mhc-multi-head-channel-概念|MHC]]): hc_repeat -> attn MHC pre -> RMSNorm -> **G2 sparse shared-KV attention** -> MHC post -> mlp MHC pre -> RMSNorm -> MoE MLP -> MHC post. **G2 attention** (`g2_attention.py`): shared KV (`linear_kv: H->head_dim`, not per-head), per-head query (`wq_b` Column TP), RoPE only on the last rope_head_dim=64, always keeps a local window (`--g2-window-size 128`), [[20260608-125000-deepseek-sparse-attention-dsa-概念|DSA indexer]] at compress_ratio=4 / static index at 128; the MTP layer's attention disables indexer (compress_ratio=0). Compressor: gate-softmax-weighted block KV compression (overlap at ratio=4).

## Sharding rules
- **TP**: linear_q/linear_kv are LinearNoTP (TP-replicated); q_up_proj Column TP (by head, needs heads%TP==0); o_down_proj Column TP (needs o_groups%TP==0); o_up_proj Row TP; lm_head Column TP. Flash TP in {1,2,4,8} (heads 64, o_groups 8); Pro up to 16 (heads 128, o_groups 16). Sequence-parallel shards the seq dim; KV/compressed-KV gather_from_sp_cp when global view needed.
- **PP**: Flash uses num_layers=44, noop=43, PP=4 (11 logical layers/stage). Pro padding suggestions: PP=2 (62, noop 61), PP=4 (64, noop 61-63). MHC patches inter-stage shape; VPP requires (num_layers/PP)%vpp_stage==0.
- **EP**: shards routed experts only; `num_experts % EP == 0`, `DP*CP % EP == 0`, `DP = world/(TP*PP*CP)`. Flash: world=128, TP1/PP4/CP1 -> DP=32, EP=32 (8 experts/rank). Pro EP factors of 384: 24/32/48/64/96/128. EP not "bigger is better" - finer all-to-all + smaller expert GEMM.
- **CP**: shards sequence/context for long seq; `q_len_global = q_len*cp_size` so RoPE/sparse-index use global positions; Ulysses or kvallgather algos. Flash CP marked DOING; default CP=1 ulysses.

## Pro SFT param suggestions
Swap hidden 7168, heads 128, experts 384, moe-ffn 3072, q-lora 1536, o-groups 16, index-topk 1024, Pro's 61-layer compress_ratios. PP=2/4 with noop; TP=2/4 for memory; EP recomputed from DP*CP. Risks: o_groups=16 TP divisibility, moe_intermediate 3072 (not 2048), index_topk 1024 raises DSA indexer memory, MHC changes PP shape (single-step validate), CP+pack/variable-length high-risk (DOING).

Relates to [[20260608-134100-mindspeed-llm-deepseek4-sft-pp-parallel-来源]], [[20260608-125100-deepseek-v4-pro-vllm-ascend-gap-analysis-来源]] (inference-side V4 gap, same index_topk 512-vs-1024 issue).

## Counter-arguments / Data gaps
- Pro is param-mapped, not validated; all Pro configs are recommendations.
- Flash MHC scripts still carry legacy `--kv-lora-rank 512 --v-head-dim 128` though the V4 attention main path uses `linear_kv: H->head_dim` - a config/impl mismatch to verify.
