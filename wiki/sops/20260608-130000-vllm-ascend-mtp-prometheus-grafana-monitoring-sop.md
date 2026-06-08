---
title: "SOP: vLLM (Ascend + MTP) Prometheus + Grafana Monitoring"
tldr: "Drop-in monitoring stack for vLLM on Ascend: docker-compose (Prometheus+Grafana), scrape config, alert rules (TTFT/TPOT/queue/KV/preemption/prefix-cache/spec-acceptance), and a 5-screen Grafana dashboard (overview, latency breakdown, request shape & cache, spec-decode/MTP, health/MFU). Includes a launch + /metrics verification checklist."
date_created: 2026-06-08
date_modified: 2026-06-08
type: sop
tags: [monitoring, vllm, ascend, mtp, prometheus, grafana, observability, sop]
sources: []
original_url: ""
explored: false
confidence: high
---

# SOP: vLLM (Ascend + MTP) Prometheus + Grafana Monitoring

A directly-runnable monitoring template for vLLM on Ascend with [[20260608-121100-mtp-multi-token-prediction-概念|MTP]]. Consolidated from two raw notes (full config files + the 5-screen indicator list, alerts, and launch checklist).

## Layout
```
monitoring/
  docker-compose.yaml          # prometheus + grafana
  prometheus/prometheus.yml    # scrape vllm /metrics (host.docker.internal:8000)
  prometheus/rules/vllm-alerts.yml
  grafana/dashboards/vllm-ascend-mtp-dashboard.json
```
Deploy: `docker compose up -d`; Prometheus :9090, Grafana :3000 (admin/admin), add Prometheus datasource `http://prometheus:9090`, import the dashboard JSON.

## Alert rules (key thresholds)
- **TTFT P99 > 3s**, **TPOT P99 > 250ms**, **Queue P95 > 1s** (10m).
- **KV cache usage > 90%** (15m), **waiting requests > 20**, **preemption rate > 1/s**.
- **Prefix cache hit rate < 20%** (30m, info).
- **Spec acceptance < 30%** (warning); **Spec likely ineffective** = acceptance < 15% AND TPOT P95 > 200ms (critical) - operationalizes the "[[20260608-120000-speculative-decoding-概念|acceptance != speedup]]" lesson.

## Dashboard (5 screens)
1. **Service overview**: request QPS, prompt/generation tok/s, running vs waiting, KV cache %, preemptions, corrupted requests.
2. **Latency breakdown**: E2E / TTFT / TPOT(ITL) / queue / prefill / decode P50-P99.
3. **Request shape & cache**: prompt/generation length histograms, prefix cache hit ratio (local + external), cached vs recomputed tokens, KV block lifetime/idle/reuse-gap.
4. **Spec decode / MTP**: draft/accepted rates, acceptance ratio, accepted-tokens-per-position, acceptance vs TPOT, acceptance vs throughput.
5. **Health / HTTP / MFU**: HTTP QPS+5xx, RSS/FDs, estimated FLOPs/read/write per GPU, LoRA info.

## Launch + verify checklist
Serve with `--kv-cache-metrics --kv-cache-metrics-sample 0.01 --cudagraph-metrics --enable-mfu-metrics --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":2}'`. Then `curl /metrics | egrep` to confirm `vllm:request_success`, latency buckets, `kv_block_*`, `prefix_cache_*`, `spec_decode_num_*`, `estimated_*_per_gpu_total` exist. Confirm Prometheus targets UP and Grafana env/cluster/model_name variables resolve.

Relates to [[20260608-122850-vllm-ascend-实体|vLLM-Ascend]], [[20260412-194111-prefill-decode-disaggregation-概念]].

## Counter-arguments / Data gaps
- Metric names/labels vary by vLLM version - the note warns to `curl /metrics` first to confirm `model_name`, `finished_reason`, `draft_position`, `status` labels.
- Thresholds are starting defaults, not tuned to a specific SLA.
- Some metrics (`spec_decode_num_accepted_tokens_per_pos`, MFU estimates) may not exist on older builds.
