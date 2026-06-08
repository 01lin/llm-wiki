---
title: "Source: SpecForge Issue #455 - trained DFlash has extremely low acceptance rate"
tldr: "GitHub issue (sgl-project/SpecForge #455, opened 2026-01-27 by BAI Fan). Multiple users train DFlash drafters that converge normally (loss/accuracy fine) but get inference acceptance length of only ~1.02-1.29 - effectively failed. Official ZLab training details unreleased; SpecForge reverse-engineers from inference code. Suspected causes: chat-template mismatch, CE vs KL loss, data regeneration, token-ID/mask_token_id alignment."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [speculative-decoding, dflash, specforge, sglang, training, acceptance-rate]
sources: []
original_url: "https://github.com/sgl-project/SpecForge/issues/455"
explored: false
confidence: high
---

# Source: SpecForge #455 - trained DFlash low acceptance rate

GitHub issue on **sgl-project/SpecForge** (#455, opened 2026-01-27 by BAI Fan / @baifanxxx), a community-reported reproducibility problem training [[20260608-121300-dflash-概念]] drafters.

## The bug
Training a DFlash drafter (Qwen3-8B target) converges normally - loss drops, eval accuracy 70%+, only mild overfitting. But loading the trained weights into the **official DFlash benchmark** (z-lab/dflash) gives acceptance length **1.29/(1+3)** - effectively a training failure. The reporter had previously trained EAGLE-3 successfully, so the pipeline/eval setup is presumably correct. Notably, even **overfitting on two repeated samples to loss=0** still yields near-zero acceptance at inference - pointing to a train/inference *misalignment*, not a training-quality issue.

## Reproduced by multiple users
ggg-s (1.02-1.04), yuyangxie96 (1.02-1.05), Ximingwang-09 (output length ~0). Confirms it is not isolated.

## Maintainer (xiaomin-D) diagnosis & workarounds
- **ZLab has not released official training details**; SpecForge reverse-engineers from inference code, so it cannot fully align with the official pipeline yet.
- Likely levers: (1) **align chat template** train vs inference (DFlash disables it by default, SpecForge enables) - big impact on acceptance length; (2) **loss choice** - CE default, but **KL loss** can be better; (3) **regenerate / enlarge dataset**; (4) **token-ID / `mask_token_id` (151669) and `target_layer_ids` [1,9,17,25,33] alignment** between train and inference.
- With aligned template + KL loss, ~3 tokens average achievable (still below official).

## What worked partially
- wengsnow: full perfectblend, regenerate without thinking, 2 epochs -> GSM8K 2.92, Alpaca 2.14, HumanEval 4.50, MT-Bench 2.50 (old code, no sampling anchor / loss decay). Internal business data hit ~5.0.
- **Official weights match the paper** once chat_template is aligned (SGLang DFlash impl, PR #16818) - so inference is correct; the gap is the **training recipe**.

## Significance
Concrete evidence for the claim in [[20260608-122500-投机优化方向-dflash-训练-想法-来源]] that there is no reliable open-source DFlash training recipe (especially for large MoE). The blocker is recipe/alignment, not the inference engine.

Relates to [[20260608-123100-specforge-实体]], [[20260412-194210-sglang-实体]], [[20260412-194210-qwen-实体]].

## Counter-arguments
- Maintainer argues this is "not a bug" absent a concrete implementation defect - more a recipe/objective mismatch given no reference implementation.
- Results vary widely by dataset (ShareGPT ~1.5, UltraChat ~2.5, internal ~5.0), so "failure" is partly a data/template artifact.

## Data gaps
- Root cause never definitively isolated in-thread; resolution waits on ZLab's official release.
- No large-MoE DFlash data point here (all Qwen3-8B dense).
