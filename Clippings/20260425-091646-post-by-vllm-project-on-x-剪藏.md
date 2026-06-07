---
title: "Post by @vllm_project on X"
source: "https://x.com/vllm_project/status/2047520252851105796"
author:
  - "[[@vllm_project]]"
published: 2026-04-24
created: 2026-04-25
description: "Day-0 support for @deepseek_ai V4 Pro and Flash on vLLM — a new generation of DeepSeek model, purpose-built for tasks up to 1M tokens. Along"
tags:
  - "clippings"
---
🎉 Day-0 support for @deepseek\_ai V4 Pro and Flash on vLLM — a new generation of DeepSeek model, purpose-built for tasks up to 1M tokens. Alongside the release, we're publishing a first-principles walkthrough of the new long-context attention and how we implemented it in vLLM.

The new attention mechanism, in four moves:

• Shared K/V + inverse RoPE → 2× memory savings

• c4a / c128a KV compression → 4×–128× savings

• DeepSeek Sparse Attention over compressed tokens

• Short sliding window for locality across compression boundaries

At 1M context, per-layer KV state is ~8.7× smaller than a DeepSeek V3.2-style 61-layer stack (9.62 GiB vs 83.9 GiB, bf16). fp8 attention cache + fp4 indexer cache shrink it further.

vLLM side:

• Unified hybrid KV cache — single logical block size (256 native positions) across all compression rates; compressor state folded into the SWA KV cache spec so prefix caching, disagg prefill, CUDA graphs and MTP reuse the same abstraction

• Three page-size buckets for the full 5-way cache stack → no cross-kind fragmentation

• Fused kernels: compressor + RMSNorm + RoPE + cache insert (1.4–3×), inverse RoPE + fp8 quant (2–3×), Q-norm + KV RoPE + K insert (10–20×)

• Multi-stream overlap of indexer vs main-KV compression vs SWA insertion

Disaggregated serving is supported out of the box and strongly recommended for best performance.

Follow our recipes site for verified commands for @nvidia Blackwell (B200, B300, GB200, GB300) and Hopper (H100/H200/H20) systems.

Thanks to the @deepseek\_ai team for open-sourcing DeepSeek V4, and to @inferact for landing day-0 support 🤝

📝 Blog: http://vllm.ai/blog/deepseek-v4…

📖 Recipes: https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro…

🤗 https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro…

> **DeepSeek @deepseek\_ai** · 2026-04-24
> 
> 🚀 DeepSeek-V4 Preview is officially live & open-sourced! Welcome to the era of cost-effective 1M context length.
> 
> 🔹 DeepSeek-V4-Pro: 1.6T total / 49B active params. Performance rivaling the world's top closed-source models.
> 
> 🔹 DeepSeek-V4-Flash: 284B total / 13B active params.
> 
> ![Image](https://pbs.twimg.com/media/HGpAvCnacAAF0G5?format=jpg&name=large) ![Image](https://pbs.twimg.com/media/HGo9ronbUAAKlRk?format=jpg&name=large)

---

## Comments

> **Jason Fleagle @jjfleagle** · [2026-04-24](https://x.com/jjfleagle/status/2047747974223000035)
> 
> This is the kind of systems work that quietly determines whether long-context agents stay a demo or become a product. A 1M-token headline matters less than what the serving stack does to memory pressure, cache behavior, and the cost of repeated verification passes. When long

> **anonymous @youyouAllen** · [2026-04-25](https://x.com/youyouAllen/status/2047840178350567476)
> 
> I have actually tested the two methods they used, RoPE and shared KV, and the results prove that a significant amount of signal is lost after compression. This is because these two compression methods require the KV vectors to follow a Gaussian distribution to achieve a perfect

> **Sean Pianka @seanpianka** · [2026-04-24](https://x.com/seanpianka/status/2047544863910973887)
> 
> Any ETA on avilability of vllm-mlx support?

> **anonymous @youyouAllen** · [2026-04-24](https://x.com/youyouAllen/status/2047822544443703615)
> 
> Wow, cool, I can further compress it by 1.8-2.3x on this basis.
> 
> [github.com LLM-KV--Cache-compress/reports/paper/kakeyalattice.pdf at main · FluffyAIcode/LLM-KV--Cache-compress](https://t.co/bEGCpRyrOD)