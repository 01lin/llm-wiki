---
title: "N-gram & Suffix Decoding"
tldr: "Training-free speculative-decoding drafters based on pattern matching: find n-gram / suffix matches in the prompt and already-generated text and propose them. Plug-and-play, no model, but low speedup - best for highly repetitive output. vLLM: ngram/ngram_gpu/suffix. (stub)"
date_created: 2026-06-08
date_modified: 2026-06-08
type: concept
tags: [inference, speculative-decoding, vllm]
sources: ["[[20260608-120700-vllm-vllm-ascend-spec-decode-support-来源]]"]
explored: false
confidence: medium
---

# N-gram & Suffix Decoding

Training-free [[20260608-120000-speculative-decoding-概念]] drafters that need no model at all: they look for **n-gram or suffix matches** in the prompt and already-generated text and propose the continuation. vLLM offers `ngram` (CPU), `ngram_gpu` (GPU), and `suffix` (`SuffixDecodingProposer`). Plug-and-play but low benefit; useful mainly for highly repetitive output (code, structured text). vLLM-Ascend ships NPU-optimized versions.

## Data gaps
- Stub. The related "suffix" method and Prompt-LookUp Decoding face core-sync overhead under NPU graph compilation (see A-IO, 2604.09752 in the survey).
