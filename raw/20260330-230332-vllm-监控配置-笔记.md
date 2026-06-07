**一、五个屏完整指标列表**

**屏 1：服务总览**

- Request QPS：sum(rate(vllm:request_success{env="$env",cluster="$cluster",model_name="$model_name"}[5m]))
- Prompt tok/s：sum(rate(vllm:prompt_tokens{...}[5m]))
- Generation tok/s：sum(rate(vllm:generation_tokens{...}[5m]))
- Running Requests：sum(vllm:num_requests_running{...})
- Waiting Requests：sum(vllm:num_requests_waiting{...})
- KV Cache Usage %：avg(vllm:kv_cache_usage_perc{...}) * 100
- Corrupted Requests / 5m：sum(increase(vllm:corrupted_requests{...}[5m]))
- Preemption Rate：sum(rate(vllm:num_preemptions{...}[5m]))

**屏 2：延迟分解**

- E2E P50/P95/P99：vllm:e2e_request_latency_seconds_bucket
- TTFT P50/P95/P99：vllm:time_to_first_token_seconds_bucket
- TPOT / ITL P50/P95/P99：vllm:inter_token_latency_seconds_bucket
- Queue Time P50/P95/P99：vllm:request_queue_time_seconds_bucket
- Prefill Time P50/P95/P99：vllm:request_prefill_time_seconds_bucket
- Decode Time P50/P95/P99：vllm:request_decode_time_seconds_bucket
- TPOT Mean：rate(sum)/rate(count) on vllm:inter_token_latency_seconds

**屏 3：请求形态与 Cache**

- Prompt Length Heatmap：vllm:request_prompt_tokens_bucket
- Generation Length Heatmap：vllm:request_generation_tokens_bucket
- max_tokens Distribution：vllm:request_params_max_tokens_bucket
- n Distribution：vllm:request_params_n_bucket
- Prefill KV Computed Tokens Avg：vllm:request_prefill_kv_computed_tokens
- Prefix Cache Hit Ratio：rate(vllm:prefix_cache_hits)/rate(vllm:prefix_cache_queries)
- External Prefix Cache Hit Ratio：rate(vllm:external_prefix_cache_hits)/rate(vllm:external_prefix_cache_queries)
- Cached Prompt tok/s：rate(vllm:prompt_tokens_cached)
- Recomputed Prompt tok/s：rate(vllm:prompt_tokens_recomputed)
- KV Block Lifetime P95：vllm:kv_block_lifetime_seconds_bucket
- KV Block Idle Before Evict P95：vllm:kv_block_idle_before_evict_seconds_bucket
- KV Block Reuse Gap P95：vllm:kv_block_reuse_gap_seconds_bucket

**屏 4：投机解码 / MTP**

- Draft Rate：rate(vllm:spec_decode_num_drafts)
- Draft tok/s：rate(vllm:spec_decode_num_draft_tokens)
- Accepted tok/s：rate(vllm:spec_decode_num_accepted_tokens)
- Acceptance Ratio：rate(accepted_tokens)/rate(draft_tokens)
- Accepted Tokens Per Position：sum by (draft_position) (rate(vllm:spec_decode_num_accepted_tokens_per_pos[5m]))
- Acceptance Ratio vs TPOT Mean
- Acceptance Ratio vs Generation tok/s

**屏 5：服务健康与基础设施**

- HTTP QPS：sum(rate(http_requests_total{env="$env",cluster="$cluster"}[5m]))
- HTTP 5xx Rate：sum(rate(http_requests_total{status=~"5..",...}[5m]))
- HTTP P95：http_request_duration_seconds_bucket
- RSS Memory：avg(process_resident_memory_bytes{...})
- CPU Seconds Rate：sum(rate(process_cpu_seconds_total{...}[5m]))
- Open FDs：avg(process_open_fds{...})
- LoRA Info：vllm:lora_requests_info table
- MFU FLOPs Rate：rate(vllm:estimated_flops_per_gpu_total)
- MFU Read/Write Rate：rate(vllm:estimated_read_bytes_per_gpu_total), rate(vllm:estimated_write_bytes_per_gpu_total)

**二、三个任务结果**

**1. prometheus.yml**

```
global:  scrape_interval: 15s  evaluation_interval: 15s  scrape_timeout: 10s rule_files:  - /etc/prometheus/rules/vllm-alerts.yml scrape_configs:  - job_name: vllm    metrics_path: /metrics    static_configs:      - targets: ["10.0.0.21:8000"]        labels:          service: vllm          env: prod          cluster: prod-ascend          model: qwen35-397b-a17b 
```

**2. vllm-alerts.yml**

```
groups:  - name: vllm    rules:      - alert: VLLMHighTTFTP99        expr: histogram_quantile(0.99, sum by (le,cluster,model_name) (rate(vllm:time_to_first_token_seconds_bucket[5m]))) > 3        for: 10m        labels: {severity: warning}        annotations: {summary: "TTFT P99 > 3s"}       - alert: VLLMHighTPOTP99        expr: histogram_quantile(0.99, sum by (le,cluster,model_name) (rate(vllm:inter_token_latency_seconds_bucket[5m]))) > 0.25        for: 10m        labels: {severity: warning}        annotations: {summary: "TPOT P99 > 250ms"}       - alert: VLLMHighQueueTimeP95        expr: histogram_quantile(0.95, sum by (le,cluster,model_name) (rate(vllm:request_queue_time_seconds_bucket[5m]))) > 1        for: 10m        labels: {severity: warning}        annotations: {summary: "Queue time P95 > 1s"}       - alert: VLLMKVCacheNearFull        expr: avg by (cluster,model_name) (vllm:kv_cache_usage_perc) > 0.9        for: 15m        labels: {severity: warning}        annotations: {summary: "KV cache usage > 90%"}       - alert: VLLMWaitingTooHigh        expr: sum by (cluster,model_name) (vllm:num_requests_waiting) > 20        for: 10m        labels: {severity: warning}        annotations: {summary: "Waiting requests > 20"}       - alert: VLLMSpecAcceptanceLow        expr: (sum by (cluster,model_name) (rate(vllm:spec_decode_num_accepted_tokens[15m])) / clamp_min(sum by (cluster,model_name) (rate(vllm:spec_decode_num_draft_tokens[15m])), 1)) < 0.3        for: 20m        labels: {severity: warning}        annotations: {summary: "Spec decode acceptance < 30%"} 
```

**3. 上线检查清单**

- 启动参数确认：

```
vllm serve Qwen/Qwen3.5-397B-A17B \  --served-model-name qwen35-397b-a17b \  --kv-cache-metrics \  --kv-cache-metrics-sample 0.01 \  --cudagraph-metrics \  --enable-mfu-metrics \  --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":2}' 
```

- 核对 /metrics 可访问：curl http://<host>:8000/metrics
- 核对核心指标存在：

```
curl -s http://<host>:8000/metrics | egrep 'vllm:(request_success|prompt_tokens|generation_tokens|num_requests_running|num_requests_waiting|kv_cache_usage_perc|time_to_first_token_seconds|inter_token_latency_seconds|request_queue_time_seconds|request_prefill_time_seconds|request_decode_time_seconds)' 
```

- 核对 KV 指标存在：

```
curl -s http://<host>:8000/metrics | egrep 'vllm:(kv_block_lifetime_seconds|kv_block_idle_before_evict_seconds|kv_block_reuse_gap_seconds|prefix_cache_hits|prefix_cache_queries)' 
```

- 核对 MTP 指标存在：

```
curl -s http://<host>:8000/metrics | egrep 'vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens|accepted_tokens_per_pos)' 
```

- 核对 MFU 指标存在：

```
curl -s http://<host>:8000/metrics | egrep 'vllm:estimated_(flops|read_bytes|write_bytes)_per_gpu_total' 
```

- Prometheus Targets 页面确认 UP
- Grafana 变量 env/cluster/model_name 能正常出值
- 第一屏 8 个总览指标全部有线
- 第二屏 6 组 latency 都能出分位数
- 第三屏 cache 命中率不是 NaN
- 第四屏开启 MTP 后 acceptance ratio 有值
- 告警规则加载成功且无语法错误