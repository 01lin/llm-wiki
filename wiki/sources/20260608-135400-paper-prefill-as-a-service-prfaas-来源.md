---
title: "Source (paper): Prefill-as-a-Service - Cross-Datacenter KVCache (2604.15039)"
tldr: "Moonshot AI / Tsinghua paper (arXiv 2604.15039, Apr 2026). PD disaggregation is bounded by KVCache transfer, which ties prefill+decode to one RDMA domain. Hybrid-attention models shrink KVCache enough to make cross-cluster transport plausible - but smaller KV alone isn't enough (bursty load, skewed lengths, uneven prefix caches, fluctuating bandwidth). PrfaaS selectively offloads long-context prefill to compute-dense prefill clusters and ships KVCache over commodity Ethernet to local PD clusters, with bandwidth-aware scheduling + cache-aware placement. +54% vs homogeneous PD, +32% vs naive heterogeneous on a 1T hybrid model."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [paper, pd-disaggregation, kv-cache, moonshot, mooncake, cross-datacenter, hybrid-attention, serving]
sources: []
original_url: "https://arxiv.org/abs/2604.15039"
explored: false
confidence: high
---

# Source (paper): Prefill-as-a-Service (PrfaaS)

arXiv **2604.15039v1** (Ruoyu Qin, Weiran He et al., Moonshot AI / Tsinghua, 16 Apr 2026, 16 pages).

## Problem
[[20260412-194111-prefill-decode-disaggregation-概念|PD disaggregation]] is the standard large-scale serving architecture, but its deployment boundary is set by **KVCache transfer**. Dense-attention models generate huge KVCache traffic that keeps prefill and decode tightly coupled inside one high-bandwidth (RDMA) domain, blocking heterogeneous deployment. Recent **hybrid-attention** architectures (cf. [[20260608-132100-gdn-kvcache-cooptimization-概念|GDN/Mamba hybrids]]) shrink KVCache enough to make **cross-cluster** transport plausible - but smaller KV alone isn't sufficient: workloads are bursty, request lengths skewed, prefix caches unevenly distributed, inter-cluster bandwidth fluctuates; naive full prefill externalization causes congestion and poor utilization.

## Method (PrfaaS)
A **cross-datacenter** serving architecture that **selectively offloads long-context prefill** to standalone compute-dense prefill clusters and transfers the resulting KVCache over **commodity Ethernet** to local PD clusters for decode. Combines model-side KV efficiency with system-side **selective offloading + bandwidth-aware scheduling + cache-aware request placement**, removing the requirement that heterogeneous accelerators share one low-latency RDMA fabric and enabling independent scaling of prefill vs decode. Builds on Moonshot's **Mooncake** (KVCache as a first-class resource; integrated with vLLM/SGLang/Dynamo). Aligns with hardware trends (NVIDIA Rubin CPX for prefill, Groq LPU for decode).

## Results
Case study on an internal **1T-parameter hybrid model**: a PrfaaS heterogeneous deployment achieves **+54% throughput vs homogeneous PD** and **+32% vs naive heterogeneous**, using only modest cross-datacenter bandwidth.

Relates to [[20260412-194210-sglang-实体]], [[20260608-130300-kv-cache-compression-概念]], [[20260608-125400-sgl-kernel-npu-实体|Mooncake/KV connectors]].

## Counter-arguments / Data gaps
- Single internal-model case study; generalization across models/workloads unverified.
- Benefits depend on hybrid-attention's smaller KVCache - dense models still need RDMA coupling.
- Only first 3 of 16 pages read; scheduler/placement algorithm details and failure modes not captured.
