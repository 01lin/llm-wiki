#!/usr/bin/env bash
# =============================================================================
# E0-E4 基线外收益验证脚本 — 昇腾 910B3 / DeepSeek-V4-Flash bf16
# 目的：把 [512K上下文-基线外5to10x定量推导] 的定量假设转成可执行实验
#       校准 L1/L2/L6/L4自适应 四项基线外技术的真实倍率
# 复用：vllm bench serve + vllm/benchmarks/multi_turn/benchmark_serving_multi_turn.py
# 前置：服务已起（vllm serve ...），本脚本只跑压测 + 抓 metrics
# 证据基础：所有 CLI flag 已对 vllm/vllm/benchmarks/serve.py grep 实测（--input-len/
#   --output-len/--max-concurrency/--goodput/--num-warmups/--request-rate/--result-dir
#   /--metric-percentiles/--base-url/--save-result 均 OK）；multi_turn 参数同样实测。
#   注：vllm bench serve 用 --input-len/--output-len（非 --random-*），无 --num-prompts/
#   --dataset-name（请求总量由 --request-rate × 时长 或 dataset 决定，此处用 random 默认数据集）
# 注意：512K 需多节点；单节点先用 64K-128K，按文档 §1.2 外推
# =============================================================================
set -euo pipefail

# ---- 公共配置（按实际环境改）----
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash-bf16}"
SERVED_NAME="${SERVED_NAME:-dsv4flash}"
URL="${URL:-http://127.0.0.1:8000}"
RESULT_DIR="${RESULT_DIR:-./e2e_results_$(date +%Y%m%d_%H%M%S)}"
CTX_LEN="${CTX_LEN:-131072}"      # 单节点 128K；多节点改 524288 (512K)
mkdir -p "$RESULT_DIR"

# Prometheus metrics 抓取（关键观测量，已坐实于 vllm/v1/metrics/loggers.py）
# vllm:request_time_per_output_token_seconds (TPOT)  -> :823
# vllm:time_to_first_token_seconds (TTFT)
# vllm:prefix_cache_hits / vllm:external_prefix_cache_hits (L1) -> :558/:584
scrape_metrics() {  # $1 = tag
  curl -s "${URL}/metrics" > "${RESULT_DIR}/metrics_${1}.txt" || true
}

# =============================================================================
# E0 基线锚定：扫并发，定 "TPOT 不劣化" 红线
#   判据：找到 TPOT 开始陡升的 concurrency 拐点 = 后续所有实验红线
# =============================================================================
e0_baseline_tpot_knee() {
  echo "=== E0: TPOT 红线扫描 (concurrency 拐点) ==="
  for CC in 4 8 16 28 40 56; do
    vllm bench serve \
      --backend vllm --model "$MODEL_PATH" --served-model-name "$SERVED_NAME" \
      --base-url "$URL" \
      --input-len "$CTX_LEN" --output-len 256 \
      --max-concurrency "$CC" --request-rate inf --num-warmups 1 \
      --metric-percentiles "50,90,99" --percentile-metrics "ttft,tpot,itl" \
      --save-result --result-dir "$RESULT_DIR" --result-filename "E0_cc${CC}.json"
    scrape_metrics "E0_cc${CC}"
  done
  echo "→ 分析各 cc 的 p90 TPOT，找陡升拐点 = TPOT_SLO 红线"
}

# =============================================================================
# E1 L2' NPU 多流重叠（bf16-only 替代原 L5 量化实验）
#   ★bf16-only 约束：原 c8/compress 量化实验作废，改测 NPU 原生多流重叠
#   杠杆（ascend_config.py 默认全 False，bf16 零精度风险）：
#     multistream_dsv4_dsa_overlap / multistream_overlap_shared_expert
#     / multistream_overlap_gate / prefill_comm_compute_overlap
#   注：需用不同 --additional-config 重启服务对比
#   判据：验证多流重叠 ≈1.5-1.7×（bf16-only 重推导 §4）
# =============================================================================
e1_multistream_overlap() {
  echo "=== E1: L2' NPU 多流重叠 (bf16, 需服务端 additional-config 重启) ==="
  local TAG="${1:-ms_off}"   # ms_off | ms_dsa | ms_shared | ms_all
  local CC="${TPOT_SLO_CC:-28}"   # 用 E0 定的红线
  vllm bench serve \
    --backend vllm --model "$MODEL_PATH" --served-model-name "$SERVED_NAME" --base-url "$URL" \
    --input-len "$CTX_LEN" --output-len 256 \
    --max-concurrency "$CC" --request-rate inf --num-warmups 1 \
    --goodput "tpot:50" --percentile-metrics "ttft,tpot,itl" \
    --save-result --result-dir "$RESULT_DIR" --result-filename "E1_${TAG}.json"
  scrape_metrics "E1_${TAG}"
  echo "→ 对比 ms_off vs ms_* 的吞吐比 = NPU 多流重叠真实倍率（同 TPOT 红线）"
}

# =============================================================================
# E2 L4 投机接受率：spec∈{1,2,3} × 数据类型，算接受率 → 有效加速
#   唯一不可参数化项（主文档 §6-2）。需服务端改 num_speculative_tokens 重启
#   判据：num_accept/num_reject (llm_base_proposer.py:868) → 接受率
# =============================================================================
e2_l4_acceptance() {
  echo "=== E2: L4 投机接受率 (服务端改 spec 重启, 抓 accept/reject) ==="
  local SPEC="${1:-3}"
  local CC="${TPOT_SLO_CC:-28}"
  vllm bench serve \
    --backend vllm --model "$MODEL_PATH" --served-model-name "$SERVED_NAME" --base-url "$URL" \
    --input-len "$CTX_LEN" --output-len 512 \
    --max-concurrency "$CC" --request-rate inf --num-warmups 1 \
    --percentile-metrics "ttft,tpot,itl" \
    --save-result --result-dir "$RESULT_DIR" --result-filename "E2_spec${SPEC}.json"
  scrape_metrics "E2_spec${SPEC}"
  echo "→ 接受率 = 总accept / (总draft)；有效加速 = 1 + 接受率 × spec"
}

# =============================================================================
# E3 L1 前缀命中率：multi_turn benchmark，轮次扫描
#   参数已 grep 实测；external_prefix_cache_hits = 全局索引信号
#   判据：验证 §2.1 命中率 H/(H+x)；无全局索引退化值
# =============================================================================
e3_l1_prefix_hit() {
  echo "=== E3: L1 agentic 多轮命中率 (multi_turn) ==="
  for TURNS in 2 4 8 16; do
    python vllm/benchmarks/multi_turn/benchmark_serving_multi_turn.py \
      --model "$MODEL_PATH" --served-model-name "$SERVED_NAME" --url "$URL" \
      --input-file vllm/benchmarks/multi_turn/generate_multi_turn.json \
      --num-clients 4 --max-active-conversations 8 --max-turns "$TURNS" \
      --warmup-percentages 10 \
      --stats-json-output "${RESULT_DIR}/E3_turns${TURNS}.json"
    scrape_metrics "E3_turns${TURNS}"
  done
  echo "→ 命中率 = external_prefix_cache_hits / 总 prompt token；随轮次上升趋势"
  echo "→ 单节点测实例内 Radix 命中下界；多节点(开全局索引 proxy)测跨实例增量"
}

# =============================================================================
# E4 L2 DBO + 地基：DBO 开/关对比 (enable_dbo 默认 False, parallel.py:208)
#   判据：验证 §2.2 DBO ≈1.43×；需服务端 enable_dbo 重启
# =============================================================================
e4_l2_dbo() {
  echo "=== E4: L2 DBO 开/关对比 (服务端 enable_dbo 重启) ==="
  local TAG="${1:-dbo_off}"   # dbo_off | dbo_on
  local CC="${TPOT_SLO_CC:-28}"
  vllm bench serve \
    --backend vllm --model "$MODEL_PATH" --served-model-name "$SERVED_NAME" --base-url "$URL" \
    --input-len "$CTX_LEN" --output-len 256 \
    --max-concurrency "$CC" --request-rate inf --num-warmups 1 \
    --percentile-metrics "ttft,tpot,itl" \
    --save-result --result-dir "$RESULT_DIR" --result-filename "E4_${TAG}.json"
  scrape_metrics "E4_${TAG}"
  echo "→ 对比 dbo_on/dbo_off 的吞吐比 = DBO 真实倍率"
}

# =============================================================================
# 服务端配置模板（每组实验需对应重启服务）— 仅注释，非执行
# -----------------------------------------------------------------------------
# 基线:   vllm serve $MODEL_PATH --served-model-name $SERVED_NAME \
#           --tensor-parallel-size 8 --max-model-len $CTX_LEN \
#           --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp"}'
# E1-多流: 追加 --additional-config '{"multistream_dsv4_dsa_overlap":true,
#          "multistream_overlap_shared_expert":true,"prefill_comm_compute_overlap":true}'
#          （bf16-only：原 enable_sparse_c8 量化实验已作废）
# E2-spec:改 num_speculative_tokens 值
# E4-dbo: 追加 --enable-dbo  (或 parallel config enable_dbo=true)
# =============================================================================

main() {
  case "${1:-all}" in
    e0) e0_baseline_tpot_knee ;;
    e1) e1_multistream_overlap "${2:-ms_off}" ;;
    e2) e2_l4_acceptance "${2:-3}" ;;
    e3) e3_l1_prefix_hit ;;
    e4) e4_l2_dbo "${2:-dbo_off}" ;;
    all) e0_baseline_tpot_knee; e3_l1_prefix_hit ;;  # 单节点免重启的两个先跑
    *) echo "用法: $0 {e0|e1|e2|e3|e4|all} [tag]"; exit 1 ;;
  esac
  echo "结果目录: $RESULT_DIR"
}
main "$@"
