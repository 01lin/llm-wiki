---
title: "LMSYS HiSparse Announcement Thread"
source: "https://x.com/lmsysorg/status/2042683003730801147"
original_url: "https://x.com/lmsysorg/status/2042683003730801147"
author: "LMSYS Org"
published: 2026-04-10
created: 2026-04-12
fetched: 2026-04-12
source_tool: obsidian-web-clipper
description: "LMSYS announces HiSparse blog - sparse attention with hierarchical memory, 3x throughput at 256 concurrent requests"
media:
  - "raw/assets/images/20260412-193617-hisparse-lmsys-tweet-图片.png"
tags:
  - "sparse-attention"
  - "kv-cache"
  - "sglang"
---
**LMSYS Org** @lmsysorg [2026-04-10](https://x.com/lmsysorg/status/2042683003730801147)

🚀 New blog is out: HiSparse — Turbocharging Sparse Attention with Hierarchical Memory!

Sparse attention cuts compute costs, but the full KV cache still sits in GPU HBM, making it capacity-bound.

HiSparse fixes this.

Results:

⚡️ 3× throughput at 256 concurrent requests vs. baseline (32K input, 8K output on 8×H200)

🚀 Up to 5× throughput on long-context scenarios (two H20 PD-disaggregated deployment)

Key techniques include:

💾 Proactively offloads inactive KV cache to host memory, freeing GPU HBM for larger batch sizes

🧠 Hot device buffer keeps frequently accessed KV regions on-device to minimize swap-in latency

🔧 Custom CUDA kernel: top-k miss detection + LRU eviction + page table updates in one pass

Currently supports DeepSeek Sparse Attention (DSA) models: DeepSeek-V3.2 and GLM-5.1.

Thanks to @Zhiqiang\_Xie and the team for this great contribution!

![Image](https://pbs.twimg.com/media/HFkPsf_akAMHRUi?format=png&name=large)

---

**LMSYS Org** @lmsysorg [2026-04-10](https://x.com/lmsysorg/status/2042683105774047597)

Read full blog:

---

**MANISH** @OrbitHigher [2026-04-11](https://x.com/OrbitHigher/status/2042937494296211652)

Excellent and interesting!

IMHO the real systems challenge becomes:

how do you co-design sparse attention with a hierarchical KV memory manager so hot regions stay device-resident while cold regions tier out efficiently?

Feels like the next steps in long-context inference is not

---

**?** @lightthgil2 [2026-04-11](https://x.com/lightthgil2/status/2043108666711191955)

What's the difference between this and https://arxiv.org/abs/2512.10576? Maybe the computation and communication overlapping?

---

**Troll Warlord** @Wrowsla [2026-04-11](https://x.com/Wrowsla/status/2042779774267908172)

checking rn

---

**Alibaba Cloud** @alibaba\_cloud

Dreams moving at the speed of imagination.

Meet 4yo Irsyad from MFDMalaysia. Powered by Qwen3.6-Plus and Wan2.7, we’ve turned his "Sun-Kissed Road Trip" into a high-fidelity cinematic MV.

This is AIforGood: Making every child's silent story visible.