---
title: "Benchmark-In-the-Loop (BIL) Self-Evolution"
tldr: "Closed-loop Plan->Code->Build->Test->Bench->Eval->Reflect where Claude Code agents iterate on an optimization target (e.g. TPS>=300) until met, with benchmark as the oracle. Works precisely when the target is quantifiable and the bottleneck is engineering integration, not algorithm invention. Requires 3 pinned oracles (baseline, target, numeric alignment) or it degenerates into directionless churn."
date_created: 2026-06-08
date_modified: 2026-06-08
type: concept
tags: [self-evolving-ai, agentic, optimization, claude-code, automation, ascend]
sources: ["[[20260608-131300-tokenspeed-ascend-self-evolving-loop-design-来源]]"]
explored: false
confidence: medium
---

# Benchmark-In-the-Loop (BIL) Self-Evolution

**BIL** is a closed-loop autonomous-optimization pattern: a Claude Code **Orchestrator** (Opus) drives parallel **Implementer** (Sonnet) agents plus QA, Profiler, and Evaluator agents through `Plan -> Code -> Build -> Test -> Bench -> Eval -> Reflect`, looping until a quantified target is met (or auto-pausing on stall). It is a concrete instance of [[20260412-194111-self-evolving-ai-概念|self-evolving AI]], explicitly modeled on Qwen3.7's "35h autonomous evolution."

## Why it works (and when it doesn't)
It fits when (1) the goal is **quantifiable** (TPS/TPOT), (2) changes are **localized** (e.g. ~4K LOC adapter), and (3) the **bottleneck is engineering integration, not algorithm invention** - so the benchmark is a clean oracle and Claude Code acts as a tireless P7-level engineer. The hard precondition: **three oracles must be pinned in advance - baseline (locked image SHA), target metric, and numeric alignment (<1e-3)** - otherwise it is "oracle-less blind churn."

## Mechanics
- Per-iteration artifact dir (`iter_NNN/`): PLAN.md (gap + hypotheses + actions + success criteria), DIFF.patch, BENCH_RESULT.json, PROFILE_OUTPUT, EVAL_REPORT.md, RETROSPECTIVE.md (feeds next PLAN).
- Evaluator returns FAIL_HARD / DONE / STALL / CONTINUE; 3x STALL -> human pause.
- Guardrails: 6h/iter cap, $500/day API, 90% disk, never push to main without approval.
- Cost: ~$3,400 for a 35h run (A3 + Claude API), claimed ~6x ROI vs 8 person-weeks.

Applied to porting [[20260608-131000-tokenspeed-实体|TokenSpeed]] to Ascend (target SWE-smith TPS>=300 on A3 8-card). See [[20260608-131300-tokenspeed-ascend-self-evolving-loop-design-来源]]. Relates to [[20260412-194030-agentic-ai-概念]] and [[20260608-122400-benchmark-reward-hacking-概念]] (a gameable benchmark oracle would corrupt the loop).

## Counter-arguments / Data gaps
- The 6x ROI / 35-72h projection is an estimate, not a completed run.
- Failure modes acknowledged: numeric misalignment, NPU OOM accumulation, profiler distortion, false-positive "optimizations" with race conditions, local-minimum stall, baseline drift.
- A gameable benchmark (see reward-hacking) would let the loop "improve" the metric without real gains.
