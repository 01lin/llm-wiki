---
title: "MHC (DeepSeek-V4 multi-branch channel expansion)"
tldr: "DeepSeek-V4 mechanism: after embedding, expand hidden [S,B,H] -> [S,B,hc_mult,H] (hc_mult=4); each attention/MLP is wrapped by MHC pre (mix hc branches -> [S,B,H] for the module) and post (recombine module output with the hc residual), using pre/post/comb weights from a Sinkhorn-normalized split. A final hc_head folds back to [S,B,H]. Key training impact: PP inter-stage activation carries the extra hc dimension, so pipeline schedules must be patched."
date_created: 2026-06-08
date_modified: 2026-06-08
type: concept
tags: [deepseek, deepseek-v4, mhc, training, pipeline-parallel, architecture]
sources: ["[[20260608-134200-mindspeed-llm-deepseek4-architecture-parallel-来源]]", "[[20260608-134100-mindspeed-llm-deepseek4-sft-pp-parallel-来源]]"]
explored: false
confidence: medium
---

# MHC (DeepSeek-V4 multi-branch channel expansion)

**MHC** is a [[20260412-194210-deepseek-实体|DeepSeek]]-V4 architecture mechanism (`--enable-mhc`, default `hc_mult=4`, `sinkhorn_iters=20`). After embedding, the hidden state is expanded by `hc_repeat`: `[S,B,H] -> [S,B,hc_mult,H]`. Each attention and MLP block is wrapped by:
- **MHC pre**: flatten `[S,B,hc,H] -> [S,B,hc*H]`, apply `hc_fn` to get `pre/post/comb`; the `pre` weights mix the hc branches down to `[S,B,H]` for the module to compute normally.
- **MHC post**: use `post` and `comb` to recombine the module output `[S,B,H]` with the `[S,B,hc,H]` residual back into `[S,B,hc,H]`.
- The `pre/post/comb` come from `hc_split_sinkhorn`; `comb` (`[B,S,hc,hc]`) is Sinkhorn-normalized, acting as an inter-branch redistribution/mixing matrix.
- A final **hc_head** on the last stage folds `[S,B,hc,H]` back to `[S,B,H]` before MTP/final-norm/vocab head.

## Why it matters for pipeline parallelism
Plain Megatron PP passes `[S,B,H]` between stages; with MHC the real inter-layer activation is `[S,B,hc,H]`. So [[20260608-134000-mindspeed-llm-实体|MindSpeed-LLM]] patches `get_tensor_shapes` -> `get_tensor_shapes_in_mhc` (which also divides S by CP and TP/SP, then appends hc_mult), and under VPP patches the interleaving schedule. Any new PP/VPP/DualPipeV combination must first pass a single-step shape/communication check. See [[20260608-134100-mindspeed-llm-deepseek4-sft-pp-parallel-来源]].

Part of the DeepSeek-V4 stack alongside [[20260608-125000-deepseek-sparse-attention-dsa-概念|G2/DSA attention]] and [[20260608-121100-mtp-multi-token-prediction-概念|MTP]].

## Counter-arguments / Data gaps
- Inferred from MindSpeed-LLM training code, not a DeepSeek paper - the precise role/benefit of MHC isn't documented here.
- hc_mult=4 the same for Flash and Pro; the compute/memory overhead of the 4x channel expansion isn't quantified.
