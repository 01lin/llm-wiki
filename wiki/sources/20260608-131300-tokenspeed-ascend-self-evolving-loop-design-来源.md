---
title: "Source: TokenSpeed-Ascend 自演进闭环系统设计"
tldr: "Design (2026-05-24) for autonomously porting TokenSpeed to Ascend A3 via a Claude Code Benchmark-In-the-Loop: Orchestrator (Opus) + parallel Implementer/QA/Profiler/Evaluator (Sonnet) iterating Plan->Code->Build->Test->Bench->Eval->Reflect until SWE-smith TPS>=300. ~35-72h / 10-20 iters, ~$3,400, claimed 6x ROI. Hard precondition: pin 3 oracles (baseline/target/numeric)."
date_created: 2026-06-08
date_modified: 2026-06-08
type: source
tags: [tokenspeed, ascend, self-evolving-ai, claude-code, automation, agentic, benchmark]
sources: []
original_url: "https://qwen.ai/blog?id=qwen3.7"
explored: false
confidence: medium
---

# Source: TokenSpeed-Ascend 自演进闭环系统设计

Design doc (2026-05-24) to autonomously port [[20260608-131000-tokenspeed-实体|TokenSpeed]] to Ascend A3 using a Claude-Code-driven [[20260608-131500-benchmark-in-the-loop-self-evolution-概念|Benchmark-In-the-Loop]], modeled on Qwen3.7's "35h autonomous evolution."

## Target & shape
Goal: SWE-smith agentic TPS >= 300 on A3 8-card; ~2-4h per iteration; ~35-72h total (10-20 iters); ~90% automated. Why feasible: quantifiable target, change confined to a ~4K-LOC adapter layer, vllm-ascend baseline exists, bottleneck is engineering integration (good Claude-Code fit).

## Agent topology
Orchestrator (Opus, tmux 24h+, plan/terminate), Implementer x N (Sonnet, parallel code via Task tool), QA (tests + numeric align), Profiler (msprof/py-spy), Evaluator (rules + Sonnet judge), Researcher. Runs via a self-paced auto-iteration loop with `<loop-pause>`/`<loop-abort>` signals; AskUserQuestion disabled.

## Per-iteration artifacts
`iter_NNN/`: PLAN.md (gap, prior-bottleneck, hypotheses, action items, success criteria), DIFF.patch, BENCH_RESULT.json, PROFILE_OUTPUT/, EVAL_REPORT.md, RETROSPECTIVE.md. CI (GitHub Actions self-hosted) builds C++ scheduler + Python + Docker, runs `benchmark/run_iteration.sh` (evalscope perf vs locked vllm-ascend baseline). Evaluator -> FAIL_HARD / DONE / STALL / CONTINUE; 3x STALL -> human pause.

## Guardrails & cost
6h/iter cap, $500/day API, 90% disk, never push main without approval, restart NPU process per iter (avoid OOM accumulation), bench and profile in separate runs (avoid msprof distortion), lock baseline image SHA. Cost ~$3,400 for a 35h run; claimed ROI ~6x vs ~8 person-weeks.

## Phased path
Phase 0 baseline (manual, 4-6h) -> Phase 1 runtime runs (8-15h: platform_ascend, torch.npu, ACLGraph, HCCL, Qwen3 adapter, numeric align) -> Phase 2 kernel integration to ~110% baseline TPS (15-25h: MLA/MoE backends, scheduler_bridge opt, input_buffer pool, ACLGraph capture, Retract IO) -> Phase 3 agentic tuning to TPS>=300 (12-30h).

Feeds [[20260608-131500-benchmark-in-the-loop-self-evolution-概念]]; builds on [[20260608-131200-tokenspeed-vs-vllm-ascend-comparison-来源]]; relates to [[20260412-194111-self-evolving-ai-概念]].

## Counter-arguments / Data gaps
- A proposal, not an executed run - all timings/ROI are estimates.
- Acknowledged failure modes: numeric misalignment, NPU OOM, profile distortion, false-positive optimizations (race conditions), local-minimum stall, baseline drift.
- The raw note carries injected "PUA生效" asides (environment artifact), not part of the technical design.
