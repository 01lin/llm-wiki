---
title: "MTP (Multi-Token Prediction, 多 Token 预测)"
tldr: "Speculative-decoding method where draft layers are built into the model weights (1-4 lightweight decoder layers after the final hidden layer), each predicting a future token. No extra model to load. Native in DeepSeek-V3/V3.2, GLM-4.x, Qwen3.5. On vLLM-Ascend, MTP reuses the EAGLE proposer path."
date_created: 2026-06-08
date_modified: 2026-06-08
type: concept
tags: [inference, speculative-decoding, mtp, vllm]
sources: ["[[20260608-120700-vllm-vllm-ascend-spec-decode-support-来源]]", "[[20260608-120800-glm5-1-mtp-投机性能优化方案-来源]]", "[[20260608-120900-qwen3-5-mtp-ascend-适配分析-来源]]"]
explored: false
confidence: high
---

# MTP (Multi-Token Prediction)

**MTP** is a [[20260608-120000-speculative-decoding-概念]] method where the drafter is **built into the main model's weights** rather than being a separate model. One to four lightweight decoder layers ("draft layers" / "nextn predict layers") sit after the model's final hidden layer; each predicts one future token. Because the MTP weights ship inside the model checkpoint, there is no extra model to load and quality is native.

## How it works (vLLM v1)

1. Main model forward pass produces hidden states.
2. The MTP module (a lightweight decoder layer) takes `hidden_states + input embeddings`, projects the concatenation down (`eh_proj` / `fc`), runs a decoder block (attention + MLP or FusedMoE), and emits a draft token.
3. The main model verifies all `(1 + K)` tokens in one pass via rejection sampling.

vLLM auto-detects MTP: when `model_type` is `deepseek_v3/v32`, `glm4_moe_dsa`, `qwen3_5(_moe)`, etc., it rewrites `speculative_config` to the right MTP class and reads `K` from `num_nextn_predict_layers` (GLM) or `mtp_num_hidden_layers` (Qwen3.5).

## Model coverage

14+ MTP types registered in vLLM, including `deepseek_mtp`, `glm4_moe_mtp`, `glm4_moe_lite_mtp` (GLM-4.7), `qwen3_5_mtp`, `qwen3_next_mtp`, `longcat_flash_mtp`, `step3p5_mtp`, `ernie_mtp`, `mimo_mtp`. See [[20260608-120700-vllm-vllm-ascend-spec-decode-support-来源]] for the full table.

## On Ascend (vLLM-Ascend)

- MTP **reuses the [[20260608-121200-eagle-概念]] proposer** (`AscendEagleProposer`) - so it inherits NPU-optimized tree attention and ACL Graph.
- `num_speculative_tokens` max is **15** (operator integer limit in `npu_fused_infer_attention_score`).
- Fullgraph mode requires capture sizes to be multiples of `(K+1)`.
- MTP decode fast path only opens when `mtp_quantize == "w8a8_dynamic"` - gains are bound to the quantization path.
- `async_scheduling` recommended to overlap NPU transfer latency.

## The train-inference gap

The dominant accuracy limiter: MTP heads are trained on ground-truth features but run on their own predictions at inference, causing distribution shift. GLM5.1 MTP3 sits at only ~50-60% acceptance (vs DeepSeek's reported 85-90% single-layer) because steps 2 and 3 degrade. [[20260608-120800-glm5-1-mtp-投机性能优化方案-来源]] identifies FastMTP self-distillation as the root-cause fix.

## Counter-arguments

- **MTP can lose end-to-end** even with decent acceptance: vLLM #36498 and Ascend #7231 report Qwen3.5-397B MTP *degrading* throughput in some scenarios.
- **Multi-layer MTP is underused**: vLLM's EagleProposer often uses only layer 1, wasting trained layers (vLLM #31204 RFC).
- **MoE MTP** pays expert-routing overhead, lowering speedup vs dense.

## Data gaps

- How much of DeepSeek's 85-90% is reproducible on GLM/Qwen with retraining is unverified.
- Interaction of MTP with hybrid KV cache / block_size / prefix cache is flagged but unresolved (vLLM #38182).
