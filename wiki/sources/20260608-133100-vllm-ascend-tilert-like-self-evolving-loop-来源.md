---
title: "Source: vLLM-Ascend TileRT-like 自演进优化闭环实施方案"
tldr: "Plan (2026-05-25) for a self-evolving loop optimizing vLLM-Ascend DeepSeek-V3.2-W8A8+MTP single-request decode toward 400 tok/s on A3. Stricter than the TokenSpeed BIL: agent never touches the cluster, accept gate (correctness + token/s + p99 + path allowlist) overrides agent judgement, one hypothesis per round. Has a working local skeleton (ascend_tilert_loop/), goal modes, a score function, and Phase 0/1/2 (baseline freeze -> semi-auto -> 8-35h autonomous)."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [self-evolving-ai, tilert, vllm-ascend, deepseek, mtp, ascend, automation, benchmark]
sources: []
original_url: ""
explored: false
confidence: medium
---

# Source: vLLM-Ascend TileRT-like 自演进优化闭环实施方案

Plan (2026-05-25) for a [[20260608-131500-benchmark-in-the-loop-self-evolution-概念|self-evolving]] loop targeting [[20260608-122850-vllm-ascend-实体|vLLM-Ascend]] DeepSeek-V3.2-W8A8 + MTP **single-request decode low-latency**, goal **>=400 tok/s** on A3. Inspired by [[20260608-133200-tilert-architecture-runtime-elimination-来源|TileRT]]'s runtime-elimination ideas, recast on Ascend.

## Discipline (stricter than the TokenSpeed BIL)
1. Reproducible baseline before any optimization. 2. One optimization hypothesis per round. 3. Agent never SSHes / changes driver/CANN/cluster - it only produces patches; real A3 runs standard jobs. 4. All results machine-readable. 5. **Accept gate overrides agent judgement** - "agent says it worked" doesn't count. 6. Semi-auto first (10-20 human-reviewed rounds), then 8-35h autonomous.

## Goal modes & accept gate
Goal modes: `baseline_freeze`, `reach_target`, `maximize`, `diagnose`. Hard gate: `correct==true` AND `tokens_per_sec >= baseline*(1-allowed_regression)` AND `p99_tpot <= baseline*1.05` AND `unauthorized_paths==[]` AND `exit_code==0`. Soft score (ranking only, can't override gate): `tok/s ratio + 0.02*graph_hit_rate + 0.01*max(accept_len-1,0) - 0.001*d2h_sync_count - 0.02*correctness_risk`.

## Performance model & lever matrix
`effective_tok/s = accepted_tokens_per_step / decode_step_latency`. Levers (bottleneck / gain / first metric): clean D2H sync (2-10%, `d2h_sync_count`), FULL graph replay (5-20%, `graph_hit_rate`), persistent buffer (3-8%), DSA/MLA overlap (5-15%, timeline idle), MoE/MC2 overlap (8-25%, `hccl_ms_per_step`), device sampler (3-12%), MTP verifier device-side (5-20%), MTP accept-rate (10-100%+). Gains don't add - bounded by the longest bottleneck. Priority: D2H -> graph replay -> MTP accept -> MoE/MC2 -> DSA/MLA -> sampler.

## Harness
Existing local skeleton `ascend_tilert_loop/` (orchestrator/run_experiment/score/hooks, 13 unit tests pass). To add: `cluster/submit_job.py`, `collect_metrics.py`, `memory/best.json` + `failures.jsonl`, report generator. Standard metrics JSON (tokens_per_sec, tpot p50/p99, accept_len histogram, d2h_sync_count, graph_hit_rate, hccl/mc2 ms_per_step, kernel top bottlenecks). Eval matrix: case S(128/512)/M(512/1024)/L(2048/1024), repeat 1/3/5; reject if stddev>3%. 9 failure categories - only `cluster_infra_failed` doesn't penalize the patch.

## Phases
Phase 0 baseline freeze (no optimization, record full env + 5x case-M, stddev<=3%) -> Phase 1 semi-auto (human picks theme, agent patches, auto-validate, ~10 rounds) -> Phase 2 autonomous 8-35h (start only after baseline frozen + 10 rounds + >=3 correct rejections + 1 correct save + verified stop policy).

Relates to [[20260608-133000-self-evolving-optimization-loop-runbook-sop]], [[20260608-125000-deepseek-sparse-attention-dsa-概念]], [[20260608-121100-mtp-multi-token-prediction-概念]].

## Counter-arguments / Data gaps
- 400 tok/s is aspirational; the note advises stepping baseline +10%/+20% first.
- Only the local skeleton exists; cluster adapter and real A3 runs are unbuilt.
- Stronger safety framing than the TokenSpeed BIL (no cluster access, hard gate) - a deliberate contrast.
