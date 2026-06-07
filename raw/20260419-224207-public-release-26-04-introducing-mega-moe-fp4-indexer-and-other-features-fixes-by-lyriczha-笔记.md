---
title: "[Public release 26/04] Introducing Mega MoE, FP4 Indexer and other features/fixes by LyricZhao · Pull Request #304 · deepseek-ai/DeepGEMM"
source: "https://github.com/deepseek-ai/DeepGEMM/pull/304"
author:
  - "[[LyricZhao]]"
published: 2026-04-16
created: 2026-04-19
description: "New features Mega MoE, fusing & overlapping dispatch/linear 1/SwiGLU/linear 2/combine into a single mega-kernel, overlapping NVLink communi"
tags:
  - "clippings"
---
### New features

- Mega MoE, fusing & overlapping dispatch/linear 1/SwiGLU/linear 2/combine into a single mega-kernel, overlapping NVLink communication and tensor core computation
	- Performance number will be posted later
		- Only FP8 x FP4 MoE is supported
		- Only EP <= 8 is tested
		- Requires PyTorch >= 2.9
- FP4 Indexer (MQA logits) with larger MTP support
- FP8 x FP4 GEMM
- PDL
- Refactors on GEMM heuristics
- Faster JIT compilation
- GEMM optimizations (Swap A/B, much faster MoE GEMM)
- DeepEPv2 MoE GEMM layout

### Bug fixes

- JIT may crash on distributed FS
- Some kernel hangs and IMA

### Contributors

- Mega MoE: [@LyricZhao](https://github.com/LyricZhao) [@zheanxu](https://github.com/zheanxu) [@bucket-xv](https://github.com/bucket-xv) [@RayWang96](https://github.com/RayWang96) [@interestingLSY](https://github.com/interestingLSY) [@kurisu6912](https://github.com/kurisu6912) [@xay5421](https://github.com/xay5421) [@yukuai26](https://github.com/yukuai26)
- FP4 Indexer: [@zheanxu](https://github.com/zheanxu) [@xay5421](https://github.com/xay5421) [@interestingLSY](https://github.com/interestingLSY) [@kurisu6912](https://github.com/kurisu6912)
- GEMM, PDL, JIT and bug fixes: [@zheanxu](https://github.com/zheanxu) [@bucket-xv](https://github.com/bucket-xv) [@xay5421](https://github.com/xay5421) [@yukuai26](https://github.com/yukuai26) [@LyricZhao](https://github.com/LyricZhao)

### Additional notes

Mega MoE is still under development and optimizations, stay tuned and optimization ideas are welcome!  
**Disclaimer**: this release is only related to DeepGEMM's development, has nothing to do with internal model release.