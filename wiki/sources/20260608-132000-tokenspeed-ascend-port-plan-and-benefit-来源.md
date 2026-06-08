---
title: "Source: TokenSpeed-Ascend 适配方案与控制面收益分析"
tldr: "Two-doc plan (2026-05-24) to port TokenSpeed to Ascend A2/A3 via 'borrow vllm-ascend kernels + keep TokenSpeed C++ scheduler' (~7-8 weeks, ~4K-LOC adapter, 0 new AscendC). Projected vs vllm-ascend: control-plane latency 5-20ms->–<1ms (10-20x), prefix hit 60-75%->85-95%, preemption recovery ~20-125x cheaper (Retract), agentic TPS +50-100%. MLA kernel stays equal (same operators); A3 ~40-60% of B200."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [tokenspeed, vllm-ascend, ascend, scheduler, kv-cache, agentic, migration]
sources: []
original_url: "https://github.com/lightseekorg/tokenspeed"
explored: false
confidence: high
---

# Source: TokenSpeed-Ascend 适配方案与控制面收益分析

Consolidates two 2026-05-24 design docs: the A2/A3 adaptation feasibility study and the control-plane implementation + benefit analysis. Both conclude: **port [[20260608-131000-tokenspeed-实体|TokenSpeed]] to Ascend by borrowing [[20260608-122850-vllm-ascend-实体|vLLM-Ascend]]'s kernels and keeping TokenSpeed's C++ scheduler** - don't fork, don't rewrite AscendC.

## Layering & portability
- **C++ scheduler** (~10K LOC): 100% portable, zero hardware dependency - the differentiator.
- **Python runtime**: mostly portable (`torch.cuda.*` -> `torch.npu.*`).
- **Triton kernels**: portable via `triton-ascend==3.2.1`.
- **vendor kernels** (trtllm/flashinfer/deepep) and **tokenspeed-mla** (Blackwell CuTe DSL / SM100 UTCMMA, `fold_sq_factor`): **not portable** - replace with CANN `npu_fused_infer_attention_score` / FA3 / SFA. CUDA Graph -> ACLGraph; NCCL -> HCCL; NVFP4 -> W4A16.

## Two implementation framings
1. **Full adaptation** (~11 weeks, 4 phases): minimal main-repo change (~5 lines in `platform.py` adding `ascend` vendor) + a `tokenspeed-kernel-ascend` plugin pip package via entry_points + `Priority.PLUGIN` band. Stack: torch_npu 2.10 + triton-ascend 3.2.1 + CANN 9.0.0.
2. **Borrow control-plane** (~7-8 weeks, ~4.2K LOC adapter, 0 AscendC): C++ scheduler reused as-is; a ~3K-LOC Python adapter translates `ExecutionPlan` <-> vllm-ascend ops (`AscendCommonAttentionMetadata`), imports `vllm_ascend.attention.mla_v1`, `ops.fused_moe`, `pyhccl`, `acl_graph`, `kv_offload/cpu_npu`. Biggest risk: wrapping vLLM-dependent vllm-ascend models (ForwardContext/AttentionMetadata) - ~2 person-weeks.

## Projected benefit vs vllm-ascend (same A3)
| dim | vllm-ascend | TokenSpeed-Ascend | gain |
|---|---|---|---|
| control-plane latency | 5-20ms | <1ms | 10-20x |
| prefix hit (agentic multi-turn) | 60-75% | 85-95% | +25pts ([[20260608-131400-radix-tree-prefix-cache-概念|RadixTree]] vs block-hash) |
| preemption recovery | 100% recompute | ~5% IO ([[20260608-131000-tokenspeed-实体|Retract]]) | 20-125x |
| MLA kernel / layer | baseline | equal (same operators) | 0 |
| agentic TPS (Qwen3.5-MoE SWE) | ~150-220 | ~250-400 | +50-100% |

Gain decomposition: scheduling +25-40%, RadixTree +20-35%, Retract +15-25%, comm fusion (FusedReduceNorm) +5-10%, prefetch overlap +5-10%. A3 ceiling ~40-60% of B200 (540 TPS) - the gap is **MLA kernel hardware**, not scheduling.

## Where it does NOT help
Single-layer kernel latency (equal operators), single long-context prefill (~equal), and it loses vllm-ascend-only features (profiling/dynamic-chunk scheduling, DSA/lightning indexer, Mooncake connectors, 23 monkey-patch model compatibility). It is an **agentic-workload specialization, not a vllm-ascend replacement**.

Relates to [[20260608-131200-tokenspeed-vs-vllm-ascend-comparison-来源]], [[20260608-125000-deepseek-sparse-attention-dsa-概念]], [[20260412-194111-prefill-decode-disaggregation-概念]].

## Counter-arguments / Data gaps
- All TPS/latency figures are projections (some from TokenSpeed internal benchmarks); no Ascend measurements yet.
- Raw notes carry injected "PUA生效" asides - environment artifacts, not technical content.
- MLA performance recovery (40-60% of B200) is the largest uncertainty; gated on CANN operator support.
