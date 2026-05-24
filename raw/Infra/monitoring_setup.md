# vLLM Prometheus + Grafana 监控落地文件

目录建议如下：

```text
monitoring/
├── docker-compose.yaml
├── prometheus/
│   ├── prometheus.yml
│   └── rules/
│       └── vllm-alerts.yml
└── grafana/
    └── dashboards/
        └── vllm-ascend-mtp-dashboard.json
```

下面是四个完整文件内容。

## 1. `monitoring/docker-compose.yaml`

```yaml
version: "3.9"

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    extra_hosts:
      - "host.docker.internal:host-gateway"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./prometheus/rules:/etc/prometheus/rules:ro
    command:
      - --config.file=/etc/prometheus/prometheus.yml
      - --storage.tsdb.path=/prometheus
      - --web.enable-lifecycle

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    restart: unless-stopped
    depends_on:
      - prometheus
    ports:
      - "3000:3000"
    volumes:
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
```

## 2. `monitoring/prometheus/prometheus.yml`

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  scrape_timeout: 10s

rule_files:
  - /etc/prometheus/rules/vllm-alerts.yml

scrape_configs:
  - job_name: vllm
    metrics_path: /metrics
    static_configs:
      - targets:
          - host.docker.internal:8000
        labels:
          service: vllm
          env: prod
          cluster: prod-ascend
          model: qwen35-397b-a17b
```

## 3. `monitoring/prometheus/rules/vllm-alerts.yml`

```yaml
groups:
  - name: vllm-latency
    rules:
      - alert: VLLMHighTTFTP99
        expr: |
          histogram_quantile(
            0.99,
            sum by (le, cluster, model_name) (
              rate(vllm:time_to_first_token_seconds_bucket[5m])
            )
          ) > 3
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "TTFT P99 > 3s"
          description: "cluster={{ $labels.cluster }} model={{ $labels.model_name }}"

      - alert: VLLMHighTPOTP99
        expr: |
          histogram_quantile(
            0.99,
            sum by (le, cluster, model_name) (
              rate(vllm:inter_token_latency_seconds_bucket[5m])
            )
          ) > 0.25
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "TPOT P99 > 250ms"
          description: "cluster={{ $labels.cluster }} model={{ $labels.model_name }}"

      - alert: VLLMHighQueueTimeP95
        expr: |
          histogram_quantile(
            0.95,
            sum by (le, cluster, model_name) (
              rate(vllm:request_queue_time_seconds_bucket[5m])
            )
          ) > 1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Queue time P95 > 1s"
          description: "cluster={{ $labels.cluster }} model={{ $labels.model_name }}"

  - name: vllm-capacity
    rules:
      - alert: VLLMKVCacheNearFull
        expr: |
          avg by (cluster, model_name) (
            vllm:kv_cache_usage_perc
          ) > 0.9
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "KV cache usage > 90%"
          description: "cluster={{ $labels.cluster }} model={{ $labels.model_name }}"

      - alert: VLLMWaitingTooHigh
        expr: |
          sum by (cluster, model_name) (
            vllm:num_requests_waiting
          ) > 20
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Waiting requests > 20"
          description: "cluster={{ $labels.cluster }} model={{ $labels.model_name }}"

      - alert: VLLMPreemptionRateHigh
        expr: |
          sum by (cluster, model_name) (
            rate(vllm:num_preemptions[5m])
          ) > 1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Preemption rate > 1/s"
          description: "cluster={{ $labels.cluster }} model={{ $labels.model_name }}"

  - name: vllm-cache
    rules:
      - alert: VLLMPrefixCacheHitRateTooLow
        expr: |
          (
            sum by (cluster, model_name) (rate(vllm:prefix_cache_hits[15m]))
            /
            clamp_min(sum by (cluster, model_name) (rate(vllm:prefix_cache_queries[15m])), 1)
          ) < 0.2
        for: 30m
        labels:
          severity: info
        annotations:
          summary: "Prefix cache hit rate < 20%"
          description: "cluster={{ $labels.cluster }} model={{ $labels.model_name }}"

  - name: vllm-specdecode
    rules:
      - alert: VLLMSpecAcceptanceLow
        expr: |
          (
            sum by (cluster, model_name) (rate(vllm:spec_decode_num_accepted_tokens[15m]))
            /
            clamp_min(sum by (cluster, model_name) (rate(vllm:spec_decode_num_draft_tokens[15m])), 1)
          ) < 0.3
        for: 20m
        labels:
          severity: warning
        annotations:
          summary: "Spec decode acceptance < 30%"
          description: "cluster={{ $labels.cluster }} model={{ $labels.model_name }}"

      - alert: VLLMSpecNoBenefit
        expr: |
          (
            (
              sum by (cluster, model_name) (rate(vllm:spec_decode_num_accepted_tokens[15m]))
              /
              clamp_min(sum by (cluster, model_name) (rate(vllm:spec_decode_num_draft_tokens[15m])), 1)
            ) < 0.15
          )
          and
          (
            histogram_quantile(
              0.95,
              sum by (le, cluster, model_name) (
                rate(vllm:inter_token_latency_seconds_bucket[15m])
              )
            ) > 0.2
          )
        for: 20m
        labels:
          severity: critical
        annotations:
          summary: "Spec decode likely ineffective"
          description: "cluster={{ $labels.cluster }} model={{ $labels.model_name }}"
```

## 4. `monitoring/grafana/dashboards/vllm-ascend-mtp-dashboard.json`

```json
{
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": {
          "type": "grafana",
          "uid": "-- Grafana --"
        },
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "editable": true,
  "fiscalYearStartMonth": 0,
  "graphTooltip": 1,
  "panels": [
    {
      "collapsed": false,
      "gridPos": { "h": 1, "w": 24, "x": 0, "y": 0 },
      "id": 1,
      "panels": [],
      "title": "Screen 1 - Service Overview",
      "type": "row"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "reqps" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 6, "x": 0, "y": 1 },
      "id": 2,
      "targets": [
        {
          "expr": "sum(rate(vllm:request_success{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m]))",
          "legendFormat": "request qps",
          "refId": "A"
        }
      ],
      "title": "Request QPS",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "ops" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 6, "x": 6, "y": 1 },
      "id": 3,
      "targets": [
        {
          "expr": "sum(rate(vllm:prompt_tokens{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m]))",
          "legendFormat": "prompt tok/s",
          "refId": "A"
        }
      ],
      "title": "Prompt Throughput",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "ops" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 6, "x": 12, "y": 1 },
      "id": 4,
      "targets": [
        {
          "expr": "sum(rate(vllm:generation_tokens{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m]))",
          "legendFormat": "generation tok/s",
          "refId": "A"
        }
      ],
      "title": "Generation Throughput",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "percent" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 6, "x": 18, "y": 1 },
      "id": 5,
      "targets": [
        {
          "expr": "avg(vllm:kv_cache_usage_perc{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}) * 100",
          "legendFormat": "kv cache %",
          "refId": "A"
        }
      ],
      "title": "KV Cache Usage",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "none" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 6, "x": 0, "y": 9 },
      "id": 6,
      "targets": [
        {
          "expr": "sum(vllm:num_requests_running{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"})",
          "legendFormat": "running",
          "refId": "A"
        },
        {
          "expr": "sum(vllm:num_requests_waiting{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"})",
          "legendFormat": "waiting",
          "refId": "B"
        }
      ],
      "title": "Running vs Waiting",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "ops" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 6, "x": 6, "y": 9 },
      "id": 7,
      "targets": [
        {
          "expr": "sum(rate(vllm:num_preemptions{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m]))",
          "legendFormat": "preemptions/s",
          "refId": "A"
        }
      ],
      "title": "Preemption Rate",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "short" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 6, "x": 12, "y": 9 },
      "id": 8,
      "targets": [
        {
          "expr": "sum(increase(vllm:corrupted_requests{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m]))",
          "legendFormat": "corrupted / 5m",
          "refId": "A"
        }
      ],
      "title": "Corrupted Requests",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "short" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 6, "x": 18, "y": 9 },
      "id": 9,
      "targets": [
        {
          "expr": "sum by (finished_reason) (rate(vllm:request_success{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m]))",
          "legendFormat": "{{finished_reason}}",
          "refId": "A"
        }
      ],
      "title": "Request Success by Reason",
      "type": "timeseries"
    },

    {
      "collapsed": false,
      "gridPos": { "h": 1, "w": 24, "x": 0, "y": 17 },
      "id": 10,
      "panels": [],
      "title": "Screen 2 - Latency Breakdown",
      "type": "row"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "s" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 8, "x": 0, "y": 18 },
      "id": 11,
      "targets": [
        {
          "expr": "histogram_quantile(0.50, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])))",
          "legendFormat": "P50",
          "refId": "A"
        },
        {
          "expr": "histogram_quantile(0.95, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])))",
          "legendFormat": "P95",
          "refId": "B"
        },
        {
          "expr": "histogram_quantile(0.99, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])))",
          "legendFormat": "P99",
          "refId": "C"
        }
      ],
      "title": "E2E Request Latency",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "s" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 8, "x": 8, "y": 18 },
      "id": 12,
      "targets": [
        {
          "expr": "histogram_quantile(0.50, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])))",
          "legendFormat": "P50",
          "refId": "A"
        },
        {
          "expr": "histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])))",
          "legendFormat": "P95",
          "refId": "B"
        },
        {
          "expr": "histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])))",
          "legendFormat": "P99",
          "refId": "C"
        }
      ],
      "title": "TTFT",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "s" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 8, "x": 16, "y": 18 },
      "id": 13,
      "targets": [
        {
          "expr": "histogram_quantile(0.50, sum by (le) (rate(vllm:inter_token_latency_seconds_bucket{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])))",
          "legendFormat": "P50",
          "refId": "A"
        },
        {
          "expr": "histogram_quantile(0.95, sum by (le) (rate(vllm:inter_token_latency_seconds_bucket{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])))",
          "legendFormat": "P95",
          "refId": "B"
        },
        {
          "expr": "histogram_quantile(0.99, sum by (le) (rate(vllm:inter_token_latency_seconds_bucket{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])))",
          "legendFormat": "P99",
          "refId": "C"
        },
        {
          "expr": "sum(rate(vllm:inter_token_latency_seconds_sum{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])) / sum(rate(vllm:inter_token_latency_seconds_count{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m]))",
          "legendFormat": "Mean",
          "refId": "D"
        }
      ],
      "title": "TPOT / Inter-token Latency",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "s" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 8, "x": 0, "y": 26 },
      "id": 14,
      "targets": [
        {
          "expr": "histogram_quantile(0.50, sum by (le) (rate(vllm:request_queue_time_seconds_bucket{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])))",
          "legendFormat": "P50",
          "refId": "A"
        },
        {
          "expr": "histogram_quantile(0.95, sum by (le) (rate(vllm:request_queue_time_seconds_bucket{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])))",
          "legendFormat": "P95",
          "refId": "B"
        },
        {
          "expr": "histogram_quantile(0.99, sum by (le) (rate(vllm:request_queue_time_seconds_bucket{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])))",
          "legendFormat": "P99",
          "refId": "C"
        }
      ],
      "title": "Queue Time",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "s" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 8, "x": 8, "y": 26 },
      "id": 15,
      "targets": [
        {
          "expr": "histogram_quantile(0.50, sum by (le) (rate(vllm:request_prefill_time_seconds_bucket{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])))",
          "legendFormat": "P50",
          "refId": "A"
        },
        {
          "expr": "histogram_quantile(0.95, sum by (le) (rate(vllm:request_prefill_time_seconds_bucket{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])))",
          "legendFormat": "P95",
          "refId": "B"
        },
        {
          "expr": "histogram_quantile(0.99, sum by (le) (rate(vllm:request_prefill_time_seconds_bucket{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])))",
          "legendFormat": "P99",
          "refId": "C"
        }
      ],
      "title": "Prefill Time",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "s" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 8, "x": 16, "y": 26 },
      "id": 16,
      "targets": [
        {
          "expr": "histogram_quantile(0.50, sum by (le) (rate(vllm:request_decode_time_seconds_bucket{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])))",
          "legendFormat": "P50",
          "refId": "A"
        },
        {
          "expr": "histogram_quantile(0.95, sum by (le) (rate(vllm:request_decode_time_seconds_bucket{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])))",
          "legendFormat": "P95",
          "refId": "B"
        },
        {
          "expr": "histogram_quantile(0.99, sum by (le) (rate(vllm:request_decode_time_seconds_bucket{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])))",
          "legendFormat": "P99",
          "refId": "C"
        }
      ],
      "title": "Decode Time",
      "type": "timeseries"
    },

    {
      "collapsed": false,
      "gridPos": { "h": 1, "w": 24, "x": 0, "y": 34 },
      "id": 17,
      "panels": [],
      "title": "Screen 3 - Request Shape and Cache",
      "type": "row"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "none" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 8, "x": 0, "y": 35 },
      "id": 18,
      "targets": [
        {
          "expr": "sum by (le) (increase(vllm:request_prompt_tokens_bucket{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[$__rate_interval]))",
          "legendFormat": "{{le}}",
          "refId": "A"
        }
      ],
      "title": "Prompt Length Distribution",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "none" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 8, "x": 8, "y": 35 },
      "id": 19,
      "targets": [
        {
          "expr": "sum by (le) (increase(vllm:request_generation_tokens_bucket{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[$__rate_interval]))",
          "legendFormat": "{{le}}",
          "refId": "A"
        }
      ],
      "title": "Generation Length Distribution",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "short" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 8, "x": 16, "y": 35 },
      "id": 20,
      "targets": [
        {
          "expr": "sum(rate(vllm:request_prefill_kv_computed_tokens_sum{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])) / sum(rate(vllm:request_prefill_kv_computed_tokens_count{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m]))",
          "legendFormat": "avg computed kv tokens",
          "refId": "A"
        }
      ],
      "title": "Prefill KV Computed Tokens Avg",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "percentunit" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 8, "x": 0, "y": 43 },
      "id": 21,
      "targets": [
        {
          "expr": "sum(rate(vllm:prefix_cache_hits{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])) / clamp_min(sum(rate(vllm:prefix_cache_queries{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])), 1)",
          "legendFormat": "local hit ratio",
          "refId": "A"
        },
        {
          "expr": "sum(rate(vllm:external_prefix_cache_hits{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])) / clamp_min(sum(rate(vllm:external_prefix_cache_queries{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])), 1)",
          "legendFormat": "external hit ratio",
          "refId": "B"
        }
      ],
      "title": "Prefix Cache Hit Ratio",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "ops" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 8, "x": 8, "y": 43 },
      "id": 22,
      "targets": [
        {
          "expr": "sum(rate(vllm:prompt_tokens_cached{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m]))",
          "legendFormat": "cached tok/s",
          "refId": "A"
        },
        {
          "expr": "sum(rate(vllm:prompt_tokens_recomputed{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m]))",
          "legendFormat": "recomputed tok/s",
          "refId": "B"
        }
      ],
      "title": "Cached vs Recomputed Prompt Tokens",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "s" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 8, "x": 16, "y": 43 },
      "id": 23,
      "targets": [
        {
          "expr": "histogram_quantile(0.95, sum by (le) (rate(vllm:kv_block_lifetime_seconds_bucket{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])))",
          "legendFormat": "lifetime P95",
          "refId": "A"
        },
        {
          "expr": "histogram_quantile(0.95, sum by (le) (rate(vllm:kv_block_idle_before_evict_seconds_bucket{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])))",
          "legendFormat": "idle before evict P95",
          "refId": "B"
        },
        {
          "expr": "histogram_quantile(0.95, sum by (le) (rate(vllm:kv_block_reuse_gap_seconds_bucket{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])))",
          "legendFormat": "reuse gap P95",
          "refId": "C"
        }
      ],
      "title": "KV Residency Metrics",
      "type": "timeseries"
    },

    {
      "collapsed": false,
      "gridPos": { "h": 1, "w": 24, "x": 0, "y": 51 },
      "id": 24,
      "panels": [],
      "title": "Screen 4 - Speculative Decoding / MTP",
      "type": "row"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "ops" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 8, "x": 0, "y": 52 },
      "id": 25,
      "targets": [
        {
          "expr": "sum(rate(vllm:spec_decode_num_drafts{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m]))",
          "legendFormat": "drafts/s",
          "refId": "A"
        },
        {
          "expr": "sum(rate(vllm:spec_decode_num_draft_tokens{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m]))",
          "legendFormat": "draft tok/s",
          "refId": "B"
        },
        {
          "expr": "sum(rate(vllm:spec_decode_num_accepted_tokens{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m]))",
          "legendFormat": "accepted tok/s",
          "refId": "C"
        }
      ],
      "title": "Spec Decode Rates",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "percentunit" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 8, "x": 8, "y": 52 },
      "id": 26,
      "targets": [
        {
          "expr": "sum(rate(vllm:spec_decode_num_accepted_tokens{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])) / clamp_min(sum(rate(vllm:spec_decode_num_draft_tokens{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])), 1)",
          "legendFormat": "acceptance ratio",
          "refId": "A"
        }
      ],
      "title": "Acceptance Ratio",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "ops" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 8, "x": 16, "y": 52 },
      "id": 27,
      "targets": [
        {
          "expr": "sum by (draft_position) (rate(vllm:spec_decode_num_accepted_tokens_per_pos{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m]))",
          "legendFormat": "pos {{draft_position}}",
          "refId": "A"
        }
      ],
      "title": "Accepted Tokens Per Position",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "short" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 60 },
      "id": 28,
      "targets": [
        {
          "expr": "sum(rate(vllm:spec_decode_num_accepted_tokens{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])) / clamp_min(sum(rate(vllm:spec_decode_num_draft_tokens{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])), 1)",
          "legendFormat": "acceptance ratio",
          "refId": "A"
        },
        {
          "expr": "sum(rate(vllm:inter_token_latency_seconds_sum{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])) / sum(rate(vllm:inter_token_latency_seconds_count{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m]))",
          "legendFormat": "TPOT mean",
          "refId": "B"
        }
      ],
      "title": "Acceptance Ratio vs TPOT Mean",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "short" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 60 },
      "id": 29,
      "targets": [
        {
          "expr": "sum(rate(vllm:spec_decode_num_accepted_tokens{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])) / clamp_min(sum(rate(vllm:spec_decode_num_draft_tokens{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m])), 1)",
          "legendFormat": "acceptance ratio",
          "refId": "A"
        },
        {
          "expr": "sum(rate(vllm:generation_tokens{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m]))",
          "legendFormat": "generation tok/s",
          "refId": "B"
        }
      ],
      "title": "Acceptance Ratio vs Generation Throughput",
      "type": "timeseries"
    },

    {
      "collapsed": false,
      "gridPos": { "h": 1, "w": 24, "x": 0, "y": 68 },
      "id": 30,
      "panels": [],
      "title": "Screen 5 - Health / HTTP / MFU / Process",
      "type": "row"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "reqps" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 8, "x": 0, "y": 69 },
      "id": 31,
      "targets": [
        {
          "expr": "sum(rate(http_requests_total{env=~\"$env\",cluster=~\"$cluster\"}[5m]))",
          "legendFormat": "http qps",
          "refId": "A"
        },
        {
          "expr": "sum(rate(http_requests_total{env=~\"$env\",cluster=~\"$cluster\",status=~\"5..\"}[5m]))",
          "legendFormat": "http 5xx",
          "refId": "B"
        }
      ],
      "title": "HTTP Rate and Errors",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "s" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 8, "x": 8, "y": 69 },
      "id": 32,
      "targets": [
        {
          "expr": "histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket{env=~\"$env\",cluster=~\"$cluster\"}[5m])))",
          "legendFormat": "HTTP P95",
          "refId": "A"
        }
      ],
      "title": "HTTP Latency",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "bytes" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 8, "x": 16, "y": 69 },
      "id": 33,
      "targets": [
        {
          "expr": "avg(process_resident_memory_bytes{env=~\"$env\",cluster=~\"$cluster\"})",
          "legendFormat": "rss",
          "refId": "A"
        },
        {
          "expr": "avg(process_open_fds{env=~\"$env\",cluster=~\"$cluster\"})",
          "legendFormat": "open_fds",
          "refId": "B"
        }
      ],
      "title": "Process Memory and FDs",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": { "unit": "ops" }, "overrides": [] },
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 77 },
      "id": 34,
      "targets": [
        {
          "expr": "sum(rate(vllm:estimated_flops_per_gpu_total{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m]))",
          "legendFormat": "flops/s",
          "refId": "A"
        },
        {
          "expr": "sum(rate(vllm:estimated_read_bytes_per_gpu_total{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m]))",
          "legendFormat": "read bytes/s",
          "refId": "B"
        },
        {
          "expr": "sum(rate(vllm:estimated_write_bytes_per_gpu_total{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}[5m]))",
          "legendFormat": "write bytes/s",
          "refId": "C"
        }
      ],
      "title": "MFU Estimated Rates",
      "type": "timeseries"
    },
    {
      "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
      "fieldConfig": { "defaults": {}, "overrides": [] },
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 77 },
      "id": 35,
      "targets": [
        {
          "expr": "vllm:lora_requests_info{env=~\"$env\",cluster=~\"$cluster\",model_name=~\"$model_name\"}",
          "legendFormat": "lora info",
          "refId": "A"
        }
      ],
      "title": "LoRA Requests Info",
      "type": "table"
    }
  ],
  "refresh": "30s",
  "schemaVersion": 39,
  "style": "dark",
  "tags": ["vllm", "prometheus", "grafana", "ascend", "mtp"],
  "templating": {
    "list": [
      {
        "current": {
          "selected": true,
          "text": "Prometheus",
          "value": "Prometheus"
        },
        "hide": 0,
        "includeAll": false,
        "label": "datasource",
        "name": "DS_PROMETHEUS",
        "options": [],
        "query": "prometheus",
        "refresh": 1,
        "regex": "",
        "type": "datasource"
      },
      {
        "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
        "definition": "label_values(vllm:request_success, env)",
        "hide": 0,
        "includeAll": true,
        "label": "env",
        "multi": true,
        "name": "env",
        "query": {
          "query": "label_values(vllm:request_success, env)",
          "refId": "env"
        },
        "refresh": 2,
        "type": "query"
      },
      {
        "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
        "definition": "label_values(vllm:request_success{env=~\"$env\"}, cluster)",
        "hide": 0,
        "includeAll": true,
        "label": "cluster",
        "multi": true,
        "name": "cluster",
        "query": {
          "query": "label_values(vllm:request_success{env=~\"$env\"}, cluster)",
          "refId": "cluster"
        },
        "refresh": 2,
        "type": "query"
      },
      {
        "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
        "definition": "label_values(vllm:request_success{env=~\"$env\",cluster=~\"$cluster\"}, model_name)",
        "hide": 0,
        "includeAll": true,
        "label": "model_name",
        "multi": true,
        "name": "model_name",
        "query": {
          "query": "label_values(vllm:request_success{env=~\"$env\",cluster=~\"$cluster\"}, model_name)",
          "refId": "model_name"
        },
        "refresh": 2,
        "type": "query"
      }
    ]
  },
  "time": {
    "from": "now-6h",
    "to": "now"
  },
  "timepicker": {},
  "timezone": "browser",
  "title": "vLLM Production Dashboard for Ascend + MTP",
  "uid": "vllm-ascend-mtp",
  "version": 1,
  "weekStart": ""
}
```

部署方式很简单：

1. 启动 vLLM，并确认 `/metrics` 可访问。
2. 在 `monitoring/` 目录执行：

```bash
docker compose up -d
```

3. 打开：
- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3000>

4. Grafana 默认账号通常是：
- 用户名：`admin`
- 密码：`admin`

5. 在 Grafana 里添加 Prometheus 数据源：
- URL 填 `http://prometheus:9090`

6. 导入 dashboard：
- 上传 `monitoring/grafana/dashboards/vllm-ascend-mtp-dashboard.json`

补一句：这套文件是“可直接跑起来的落地模板”，但你上线前最好先用 `curl /metrics` 看一下你当前 vLLM 版本的真实 label，尤其是：
- `model_name`
- `finished_reason`
- `draft_position`
- `status`

如果需要，我还可以继续整理一份“vLLM 启动命令模板大全”，分别对应：
- 基础模式
- 开 KV cache metrics
- 开 MTP
- 开 LoRA
- Ascend 推荐生产参数组合
