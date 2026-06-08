---
title: "SOP: Autonomous Self-Evolving Optimization Loop (Claude Code + Ascend)"
tldr: "Operational runbook for the TokenSpeed-Ascend Benchmark-In-the-Loop: 5-day manual setup (lock baseline SHA, build CI + guardrails) then 35-72h autonomous Claude-Code evolution. Includes the Mission Brief, the iteration four-piece templates (PLAN/RETROSPECTIVE/EVAL/PROFILE), Implementer task template, preflight checklist, and 11 on-call failure SOPs."
date_created: 2026-06-08
date_modified: 2026-06-08
type: sop
tags: [self-evolving-ai, claude-code, ascend, automation, benchmark, runbook, sop]
sources: []
original_url: ""
explored: false
confidence: medium
---

# SOP: Autonomous Self-Evolving Optimization Loop

The executable runbook backing [[20260608-131500-benchmark-in-the-loop-self-evolution-概念|Benchmark-In-the-Loop]] / [[20260608-131300-tokenspeed-ascend-self-evolving-loop-design-来源|TokenSpeed-Ascend self-evolution]]. Consolidates the construction manual and the template/emergency-SOP companion. Pattern: **5 days manual to build "baseline + loop + guardrails," then let Claude Code run 35-72h.**

## Phase 0: 5-day setup
- **Day 0**: resources (A3 8-card 910C + CPU control node + 10TB NVMe), passwordless SSH, Anthropic API reachability, local Docker registry.
- **Day 1**: CANN 9.0 + torch_npu + triton-ascend on A3; **lock vLLM-Ascend baseline commit SHA** and build/push baseline image; download model weights.
- **Day 2**: SWE-smith dataset (128 convs); run 8-concurrency baseline benchmark; **write `/share/ORACLE.json`** (baseline_sha, baseline_image, target_metric, target_value=300, numeric_tol=1e-3).
- **Day 3**: install Claude Code + skills; GitHub Actions self-hosted runners (CPU build + A3 bench); Prometheus/Grafana.
- **Day 4**: repo skeleton (`adapter/`, `iterations/`, `benchmark/`, `profiling/`); write `CLAUDE.md` (mission, mandatory per-iter workflow, red lines, resource caps); CI workflow (`iteration.yml`: build C++ scheduler -> unit/numeric tests -> docker -> bench -> profile -> upload artifact); `run_iteration.sh` (evalscope vs locked baseline), `run_profile.sh` (py-spy + msprof), `aggregate_bench.py`.
- **Day 5**: `start_evolve.sh` (tmux + claude + Mission Brief), `preflight_check.sh` (11 checks all PASS before launch).

## Red lines (in CLAUDE.md / Mission Brief)
Never push to main without approval; never modify ORACLE.json / baseline; never `--no-verify` or skip hooks; never comment out failing tests; never fabricate benchmark data; never bypass the numeric oracle.

## Iteration four-piece templates
- **PLAN.md**: current gap, prior bottleneck, <=3 falsifiable hypotheses, action items (each assigned to an Implementer), hard success criteria, rollback.
- **RETROSPECTIVE.md**: results table, hypothesis verification, attribution, residual problems (ranked), next hypotheses.
- **EVAL_REPORT.md**: hard checks (tests/numeric/no-crash) + soft metrics + decision (DONE/CONTINUE/STALL) + stall counter.
- **PROFILE_ANALYSIS.md**: py-spy scheduler hotspots + msprof NPU operators + cache metrics + mandatory next-iteration levers.
- **Implementer task template**: scoped files, references, acceptance (unit test + numeric align), `[DONE]`/`[BLOCKED]` signaling.

## 11 on-call failure SOPs
build failure, 3x STALL (switch battlefield / add resources / lower target), A3 crash/reboot, API cost over budget (downgrade Opus->Sonnet), numeric misalignment (git bisect), HBM OOM, HCCL hang (disable ACLGraph), profiler distortion (run msprof separately), Claude looping on one file (force-switch hotspot), dependency conflict (lock + rebuild), manual takeover.

Phases: Sprint A (iter 001-008, runtime + numeric align) -> Sprint B (009-020, kernel integration, TPS>=200) -> Sprint C (021+, agentic tuning, TPS>=300). Cost ~$3,400 / 35h run.

Relates to [[20260608-131500-benchmark-in-the-loop-self-evolution-概念]], [[20260608-133100-vllm-ascend-tilert-like-self-evolving-loop-来源]], [[20260608-130000-vllm-ascend-mtp-prometheus-grafana-monitoring-sop]].

## Counter-arguments / Data gaps
- Runbook is unexecuted; commands/timings are from the design, not a completed run.
- References internal skills (`pua:loop`, `superpowers:*`) whose availability is environment-specific.
- Raw notes carry injected "PUA生效" asides - environment artifacts, not part of the procedure.
