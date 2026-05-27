# TokenSpeed-Ascend 自演进闭环施工手册

> 版本：2026-05-24
> 类型：可执行施工图（拿来即开干）
> 对标：Qwen3.7 35h 自主进化
> 关联：[[tokenspeed_ascend_自演进闭环系统设计_20260524]]

---

## 总览

```
┌──────────────────────────────────────────────────────────────────────┐
│ ▎四阶段工期总览                                                        │
│                                                                       │
│   Phase 0  环境就绪（人工，T+0 ~ T+5 天）                              │
│   Phase 1  自演进 Sprint A：runtime 跑通（自动 8-15h）                 │
│   Phase 2  自演进 Sprint B：算子集成 + 性能逼近（自动 15-25h）          │
│   Phase 3  自演进 Sprint C：agentic 专项达标（自动 12-30h）             │
│                                                                       │
│   总工期：5 天准备 + 35-72h 自动 = 约 7-10 天交付                     │
│                                                                       │
│ ▎核心交付物                                                            │
│ · 一套可重复运行的自演进系统                                            │
│ · 完整 iteration artifact 链（每轮 PLAN/CODE/BENCH/PROFILE/EVAL）       │
│ · SWE-smith TPS ≥ 300 on A3 8 卡（目标）                              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Phase 0：基础设施搭建（人工，5 天）

### Day 0：物理资源到位 + 网络拉通

#### 0.1 物理资源清单

| 节点 | 用途 | 最低规格 | 数量 |
|------|------|---------|------|
| A3 训推一体机 | 实测 + 自演进迭代 | 8卡 910C, 1.5TB RAM | 1 |
| 控制节点 (CPU) | Claude Code orchestrator + CI + 编译 | 64C, 256GB, 4TB NVMe | 1 |
| 存储节点 | 模型权重 + dataset + artifact | 10TB+ NVMe NAS | 1 |
| 备用 A3 | 长跑期间不影响在线业务 | 同上 | 1（可选）|

#### 0.2 网络拉通清单

```bash
# 1. 控制节点 ↔ A3 节点：免密 SSH
# 控制节点
ssh-keygen -t ed25519 -C "ts-ascend-orchestrator"
ssh-copy-id root@a3-node-01

# 2. 验证 A3 节点可达
ssh a3-node-01 'npu-smi info | head -20'
# 预期输出：8 张 NPU 信息，HBM 64GB 各

# 3. 控制节点 ↔ Anthropic API
curl -sS https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-opus-4-7","max_tokens":50,"messages":[{"role":"user","content":"ping"}]}'
# 预期：返回 200 + 消息内容

# 4. 共享存储挂载
mount -t nfs storage-node:/share /share
# 控制节点和 A3 节点都挂同一份
```

#### 0.3 容器/镜像 registry 准备

```bash
# 控制节点起一个本地 Docker registry，避免每次 push 公网
docker run -d -p 5000:5000 --restart=always \
  -v /share/registry:/var/lib/registry \
  --name registry registry:2

# A3 节点信任该 registry
cat > /etc/docker/daemon.json <<EOF
{ "insecure-registries": ["registry.local:5000"] }
EOF
systemctl restart docker
```

---

### Day 1：A3 基础环境（CANN + torch_npu + vllm-ascend baseline）

#### 1.1 装 CANN 9.0 + torch_npu 2.10

在 A3 节点：

```bash
# 1. 下载并安装 CANN 9.0
wget -q https://obs.cdn.huaweicloud.com/cann/Ascend-cann-toolkit_9.0.0_linux-aarch64.run
chmod +x Ascend-cann-toolkit_9.0.0_linux-aarch64.run
./Ascend-cann-toolkit_9.0.0_linux-aarch64.run --install

# 2. 设置环境（写入 .bashrc）
cat >> ~/.bashrc <<'EOF'
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/driver/lib64
EOF
source ~/.bashrc

# 3. 验证 CANN 装好
npu-smi info
# 预期：8 卡列表，每卡 HBM、Power、Health 等

# 4. 装 torch + torch_npu
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu==2.5.1.post3 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 装 triton-ascend
git clone https://gitee.com/ascend/triton-ascend.git -b release/3.2.1
cd triton-ascend && bash install.sh

# 6. 验证 torch_npu
python -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
# 预期：True 8
```

#### 1.2 部署 vllm-ascend baseline

```bash
# 1. 拉取 vllm-ascend Dockerfile.a3
cd /share && git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend

# 2. 锁定 baseline commit（关键！整个迭代过程不再变动）
git checkout -b ts-ascend-baseline v0.x.y
BASELINE_SHA=$(git rev-parse HEAD)
echo $BASELINE_SHA > /share/baseline_sha.txt
# 例：a1b2c3d4...

# 3. 构建 baseline 镜像
docker build -t registry.local:5000/vllm-ascend:baseline-$BASELINE_SHA -f Dockerfile.a3 .
docker push registry.local:5000/vllm-ascend:baseline-$BASELINE_SHA
```

#### 1.3 准备模型权重

```bash
# Qwen3.5 系列权重（HuggingFace 镜像）
mkdir -p /share/models
cd /share/models

# 用 modelscope 拉（国内更快）
pip install modelscope
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('Qwen/Qwen3.5-7B-MoE', cache_dir='/share/models')
snapshot_download('Qwen/Qwen3-8B', cache_dir='/share/models')
"

# 验证完整性
du -sh /share/models/Qwen/Qwen3.5-7B-MoE
ls /share/models/Qwen/Qwen3.5-7B-MoE | grep safetensors
```

#### 1.4 跑通 baseline 单卡 smoke test

```bash
docker run --rm --device /dev/davinci0 --device /dev/davinci_manager \
  -v /share/models:/models -e ASCEND_RT_VISIBLE_DEVICES=0 \
  registry.local:5000/vllm-ascend:baseline-$BASELINE_SHA \
  python -c "
from vllm import LLM
llm = LLM(model='/models/Qwen/Qwen3-8B', device='npu', tensor_parallel_size=1)
out = llm.generate(['Hello, who are you?'])
print(out[0].outputs[0].text)
"
# 预期：成功输出
```

---

### Day 2：SWE-smith dataset + benchmark 基线

#### 2.1 准备 dataset

```bash
mkdir -p /share/dataset
cd /share/dataset

# SWE-smith 是 agentic benchmark，128 个真实软工对话
git clone https://github.com/SWE-bench/SWE-smith.git
cd SWE-smith && python prepare_eval_data.py --num 128 \
  --output /share/dataset/swe_smith_128.jsonl

# 校验
wc -l /share/dataset/swe_smith_128.jsonl
# 预期：128

# 装 evalscope（阿里 LLM 评测工具，支持 perf 模式）
pip install evalscope[perf]==0.7.0
```

#### 2.2 跑 baseline benchmark（8 卡）

```bash
# 启动 baseline server
docker run -d --name vllm-baseline \
  --device /dev/davinci0 ... --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /share/models:/models -v /share/dataset:/data \
  -p 8000:8000 \
  registry.local:5000/vllm-ascend:baseline-$BASELINE_SHA \
  vllm serve /models/Qwen/Qwen3.5-7B-MoE \
    --tensor-parallel-size 8 --port 8000 \
    --enable-prefix-caching \
    --max-num-seqs 64

sleep 60  # warmup

# 跑 8 档 concurrency benchmark
for CONC in 1 2 4 8 16 32 64 128; do
  evalscope perf \
    --api-url http://localhost:8000/v1 \
    --model /models/Qwen/Qwen3.5-7B-MoE \
    --dataset-path /share/dataset/swe_smith_128.jsonl \
    --parallel $CONC --number 64 \
    --output-dir /share/baseline/conc_$CONC
done

# 汇总
python /share/scripts/aggregate_bench.py /share/baseline/conc_* > /share/baseline/summary.json
cat /share/baseline/summary.json
# 预期字段：tps, tpot_p50/p95, ttft_p50/p95, throughput
```

#### 2.3 锁定基线快照

```bash
# 把基线 TPS 等关键数字写入 oracle 文件，后续所有 iter 必须对照它
cat > /share/ORACLE.json <<EOF
{
  "baseline_sha": "$BASELINE_SHA",
  "baseline_image": "registry.local:5000/vllm-ascend:baseline-$BASELINE_SHA",
  "target_metric": "swe_smith_tps_concurrency_16",
  "target_value": 300,
  "stop_when_reached": true,
  "baseline_results": $(cat /share/baseline/summary.json),
  "numeric_oracle_tol": 1e-3
}
EOF
```

---

### Day 3：控制节点 Claude Code + CI 环境

#### 3.1 安装 Claude Code

在控制节点：

```bash
# 1. Node.js 18+
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo bash -
apt install -y nodejs

# 2. 装 Claude Code CLI
npm install -g @anthropic-ai/claude-code

# 3. 配置 API key
mkdir -p ~/.config/claude
cat > ~/.config/claude/api_key <<EOF
$ANTHROPIC_API_KEY
EOF

# 4. 验证
claude --version
# 预期：Claude Code v1.x.x

# 5. 装 superpowers / PUA / engineering 等关键 skills
mkdir -p ~/.claude/skills
cd ~/.claude/skills
git clone https://github.com/.../superpowers
git clone https://github.com/.../pua
git clone https://github.com/.../engineering
```

#### 3.2 设置 GitHub Actions self-hosted runner

```bash
# 1. 在 GitHub repo settings → Actions → Runners → New self-hosted runner
# 复制对应命令到控制节点

cd /opt && mkdir actions-runner && cd actions-runner
curl -o runner.tar.gz -L https://github.com/actions/runner/releases/download/v2.x.x/actions-runner-linux-x64-2.x.x.tar.gz
tar xzf ./runner.tar.gz

./config.sh --url https://github.com/your-org/tokenspeed-ascend \
  --token <REGISTRATION_TOKEN> --labels self-hosted-cpu

# 2. A3 节点上也起一个 runner（用于跑 bench）
ssh a3-node-01 'cd /opt/actions-runner && ./config.sh \
  --url ... --labels self-hosted-a3'

# 3. 让 runner 跑成 service
./svc.sh install && ./svc.sh start
```

#### 3.3 准备 Grafana + Prometheus

```bash
# 控制节点
docker compose -f /share/observability/docker-compose.yml up -d

# docker-compose.yml 关键内容：
# - prometheus 拉 vllm metrics
# - grafana 暴露 dashboard
# - node_exporter on A3 节点
```

---

### Day 4：仓库结构 + CI workflow + CLAUDE.md

#### 4.1 创建主仓 + 目录骨架

```bash
mkdir -p /share/tokenspeed-ascend/{adapter,tests,benchmark,profiling,iterations,scripts,docs}
cd /share/tokenspeed-ascend

git init
cat > .gitignore <<EOF
__pycache__/
*.pyc
build/
*.so
iterations/*/PROFILE_OUTPUT/
EOF

# 把方案文档先 copy 进来
cp /Users/linyi/code/Documents/obsidian_wiki/llm-wikid/raw/Infra/tokenspeed_ascend_*.md docs/
```

#### 4.2 CLAUDE.md 核心配置（决定 Claude 行为）

```bash
cat > /share/tokenspeed-ascend/CLAUDE.md <<'EOF'
# CLAUDE.md — TokenSpeed-Ascend 自演进任务指南

## Mission
迭代到 SWE-smith TPS ≥ 300 on A3 8-card 时自动停止。
基线锁定：见 /share/ORACLE.json

## 强制执行流（每个 iteration）
1. 读 iterations/iter_${N-1}/RETROSPECTIVE.md
2. 写 iterations/iter_N/PLAN.md（含本轮假设 + 行动项 + 成功标准）
3. 并行派发 Implementer agents（superpowers:subagent-driven-development）
4. 触发 CI：git push HEAD:iter-N
5. 等 CI artifact iter-N-results
6. Profiler agent 分析 artifact
7. Evaluator 跑硬 / 软规则
8. 写 RETROSPECTIVE.md 决定下一步

## 红线（不可触碰）
- 严禁未经人工批准 push to main
- 严禁修改 baseline 镜像 / ORACLE.json
- 严禁 --no-verify / 跳过 hooks
- 严禁注释掉失败的测试
- 严禁在 main 分支直接跑 bench

## 资源限制
- 单 iteration 6h hard cap，超时自动 pause
- Claude API $500/day hard cap
- 磁盘 > 90% 自动 pause
- 连续 3 轮 TPS delta < 2% → pause 等人

## 必加载 Skills
- superpowers:executing-plans（任务推进）
- superpowers:systematic-debugging（排错）
- superpowers:test-driven-development（数值对齐）
- pua:p9（tech-lead 风格）
- pua:pua-loop（自动迭代核心）
- engineering:debug

## 路径约定
- /share/tokenspeed-ascend/        本仓
- /share/ORACLE.json               基线 + 目标定义
- /share/models/                   模型权重
- /share/dataset/                  SWE-smith
- /share/baseline/                 基线 benchmark
- /share/grafana/                  实时面板

## 阶段目标（向 TPS=300 收敛）
- Sprint A 终止条件：A3 单卡 Qwen3-8B 数值对齐 vllm-ascend
- Sprint B 终止条件：TPS ≥ 200（vllm-ascend × 1.1）
- Sprint C 终止条件：TPS ≥ 300（target）

EOF
```

#### 4.3 CI workflow

```bash
mkdir -p /share/tokenspeed-ascend/.github/workflows

cat > .github/workflows/iteration.yml <<'EOF'
name: TS-Ascend Iteration
on:
  push:
    branches: [iter-*]

jobs:
  build:
    runs-on: self-hosted-cpu
    outputs:
      image_tag: ${{ steps.image.outputs.tag }}
    steps:
      - uses: actions/checkout@v4
      - name: Build C++ scheduler
        run: |
          cd tokenspeed-scheduler
          cmake -B build -DCMAKE_BUILD_TYPE=Release
          cmake --build build -j$(nproc)
      - name: Install adapter pkg
        run: pip install -e adapter/
      - name: Run unit tests
        run: pytest tests/unit/ -v
      - name: Numeric alignment
        run: pytest tests/numeric/ -v --tolerance 1e-3
      - name: Build Docker image
        id: image
        run: |
          TAG=${GITHUB_REF##*/iter-}-${GITHUB_SHA::8}
          docker build -t registry.local:5000/ts-ascend:$TAG .
          docker push registry.local:5000/ts-ascend:$TAG
          echo "tag=$TAG" >> $GITHUB_OUTPUT
  
  bench:
    needs: build
    runs-on: self-hosted-a3
    steps:
      - uses: actions/checkout@v4
      - name: Pull baseline
        run: docker pull $(cat /share/baseline_sha.txt | xargs -I {} echo registry.local:5000/vllm-ascend:baseline-{})
      - name: Pull current
        run: docker pull registry.local:5000/ts-ascend:${{ needs.build.outputs.image_tag }}
      - name: Run benchmark
        run: bash benchmark/run_iteration.sh ${{ github.run_number }} ${{ needs.build.outputs.image_tag }}
      - name: Run profiling
        run: bash profiling/run_profile.sh ${{ github.run_number }}
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: iter-${{ github.run_number }}-results
          path: iterations/iter_${{ github.run_number }}/
EOF
```

#### 4.4 benchmark/run_iteration.sh

```bash
cat > /share/tokenspeed-ascend/benchmark/run_iteration.sh <<'EOF'
#!/bin/bash
set -euo pipefail

ITER=$1
IMAGE_TAG=$2
ORACLE_TPS=$(jq -r '.target_value' /share/ORACLE.json)

ITER_DIR=iterations/iter_$ITER
mkdir -p $ITER_DIR/{bench,profile}

# Step 1: 基线 server 起一份（warm 5s）
docker run -d --rm --name vllm-baseline \
  --device /dev/davinci0 ... --device /dev/davinci7 \
  -v /share/models:/models -p 8000:8000 \
  $(jq -r '.baseline_image' /share/ORACLE.json) \
  vllm serve /models/Qwen/Qwen3.5-7B-MoE \
    --tensor-parallel-size 8 --port 8000 --enable-prefix-caching
sleep 60

# Step 2: 跑 baseline
evalscope perf \
  --api-url http://localhost:8000/v1 \
  --dataset-path /share/dataset/swe_smith_128.jsonl \
  --parallel 16 --number 64 \
  --output-dir $ITER_DIR/bench/baseline

docker stop vllm-baseline

# Step 3: 起 TokenSpeed-Ascend 当前镜像
docker run -d --rm --name ts-current \
  --device /dev/davinci0 ... --device /dev/davinci7 \
  -v /share/models:/models -p 8001:8000 \
  registry.local:5000/ts-ascend:$IMAGE_TAG
sleep 60

# Step 4: 跑当前
evalscope perf \
  --api-url http://localhost:8001/v1 \
  --dataset-path /share/dataset/swe_smith_128.jsonl \
  --parallel 16 --number 64 \
  --output-dir $ITER_DIR/bench/current

# Step 5: 不停 server，留给 profile 用

# Step 6: 写汇总 BENCH_RESULT.json
python scripts/aggregate_bench.py \
  --baseline $ITER_DIR/bench/baseline \
  --current $ITER_DIR/bench/current \
  --output $ITER_DIR/BENCH_RESULT.json

echo "[BENCH] iter $ITER done. Result:"
cat $ITER_DIR/BENCH_RESULT.json
EOF
chmod +x /share/tokenspeed-ascend/benchmark/run_iteration.sh
```

#### 4.5 profiling/run_profile.sh

```bash
cat > /share/tokenspeed-ascend/profiling/run_profile.sh <<'EOF'
#!/bin/bash
set -euo pipefail
ITER=$1
ITER_DIR=iterations/iter_$ITER

# Python 调度热点
TS_PID=$(docker inspect -f '{{.State.Pid}}' ts-current)
py-spy record --pid $TS_PID --duration 60 \
  --format speedscope \
  -o $ITER_DIR/profile/scheduler.json

# NPU 算子级（msprof）
docker exec ts-current bash -c "
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  msprof --output=/tmp/msprof --duration=30 --aicpu=on
"
docker cp ts-current:/tmp/msprof $ITER_DIR/profile/msprof

# Metric dump
curl -s http://localhost:8001/metrics > $ITER_DIR/profile/metrics.txt

# 停 server
docker stop ts-current

# 整理压缩
tar czf $ITER_DIR/profile.tar.gz $ITER_DIR/profile/
EOF
chmod +x /share/tokenspeed-ascend/profiling/run_profile.sh
```

#### 4.6 scripts/aggregate_bench.py

```python
cat > /share/tokenspeed-ascend/scripts/aggregate_bench.py <<'EOF'
#!/usr/bin/env python
"""Aggregate evalscope output to BENCH_RESULT.json"""
import argparse, json, glob, os

def load(out_dir):
    f = glob.glob(f"{out_dir}/*/summary.json")
    if not f: return {}
    return json.load(open(f[0]))

ap = argparse.ArgumentParser()
ap.add_argument('--baseline', required=True)
ap.add_argument('--current', required=True)
ap.add_argument('--output', required=True)
args = ap.parse_args()

baseline = load(args.baseline)
current = load(args.current)
oracle = json.load(open('/share/ORACLE.json'))

result = {
    "iter_tps": current.get("output_throughput_per_token_per_sec", 0),
    "baseline_tps": baseline.get("output_throughput_per_token_per_sec", 0),
    "target_tps": oracle["target_value"],
    "tpot_p50": current.get("tpot_p50_ms", 0),
    "tpot_p95": current.get("tpot_p95_ms", 0),
    "ttft_p50": current.get("ttft_p50_ms", 0),
    "ttft_p95": current.get("ttft_p95_ms", 0),
    "vs_baseline": current.get("output_throughput_per_token_per_sec", 0) / max(baseline.get("output_throughput_per_token_per_sec", 1), 1),
    "vs_target": current.get("output_throughput_per_token_per_sec", 0) / oracle["target_value"],
    "preempt_rate": current.get("preempt_rate", 0),
    "prefix_hit_rate": current.get("prefix_cache_hit_rate", 0),
}
json.dump(result, open(args.output, 'w'), indent=2)
EOF
chmod +x /share/tokenspeed-ascend/scripts/aggregate_bench.py
```

---

### Day 5：自演进 orchestrator 启动脚本 + 验收

#### 5.1 orchestrator 启动脚本

```bash
cat > /share/tokenspeed-ascend/scripts/start_evolve.sh <<'EOF'
#!/bin/bash
# 自演进系统主控启动

# 1. tmux 长会话
SESSION=ts-ascend-evolve
tmux kill-session -t $SESSION 2>/dev/null || true
tmux new -d -s $SESSION

# 2. 进 Claude Code
tmux send-keys -t $SESSION "cd /share/tokenspeed-ascend" Enter
tmux send-keys -t $SESSION "claude --model claude-opus-4-7 --no-confirm-permissions" Enter
sleep 5

# 3. 预热 skills
tmux send-keys -t $SESSION "/pua:on" Enter
sleep 2

# 4. 进入自动迭代模式
tmux send-keys -t $SESSION "/pua:loop" Enter
sleep 2

# 5. 输入任务（核心 Mission Brief）
cat > /tmp/mission.txt <<'MISSION'
== TokenSpeed-Ascend Self-Evolution Mission ==

目标：将 TokenSpeed-Ascend 在 A3 8 卡 SWE-smith dataset 上的 TPS 优化到 ≥ 300。
路径：分阶段（Sprint A→B→C），每个 iteration 走完整闭环 PLAN→CODE→BUILD→BENCH→PROFILE→EVAL→REFLECT。

阅读文档：
- /share/tokenspeed-ascend/CLAUDE.md（行为指南）
- /share/tokenspeed-ascend/docs/tokenspeed_ascend_控制面实现方案与收益分析_20260524.md（实现方案）
- /share/ORACLE.json（基线 + 目标）

立即开始 iter_001：
1. 读 CLAUDE.md 和上述文档
2. 创建 iterations/iter_001/PLAN.md（基于实现方案 Phase 1 第一步）
3. 派发并行 Implementer 完成代码
4. git checkout -b iter-001 && git push（触发 CI）
5. 等 CI 完成，跑 Profiler/Evaluator
6. 写 RETROSPECTIVE.md
7. 进入 iter_002...

终止条件：BENCH_RESULT.json 中 iter_tps >= 300 时输出 <loop-abort>DONE</loop-abort>。
连续 3 轮 vs_baseline delta < 2% 输出 <loop-pause>需要人工决策方向</loop-pause>。

开始。
MISSION

tmux send-keys -t $SESSION "$(cat /tmp/mission.txt)" Enter

echo "[OK] orchestrator started in tmux session: $SESSION"
echo "  attach: tmux attach -t $SESSION"
echo "  detach: Ctrl-B + D"
echo "  view artifacts: ls -la /share/tokenspeed-ascend/iterations/"
EOF
chmod +x /share/tokenspeed-ascend/scripts/start_evolve.sh
```

#### 5.2 启动前 final checklist

```bash
# 把以下检查跑通，全 PASS 才能启动
cat > /share/tokenspeed-ascend/scripts/preflight_check.sh <<'EOF'
#!/bin/bash
set -u

check() {
    if eval "$2" >/dev/null 2>&1; then
        echo "✅ $1"
    else
        echo "❌ $1"; exit 1
    fi
}

echo "== Preflight check =="
check "A3 SSH 可达" "ssh a3-node-01 'npu-smi info | head -2'"
check "torch_npu 装好" "ssh a3-node-01 \"python -c 'import torch_npu; assert torch.npu.device_count() == 8'\""
check "Docker registry 可达" "docker pull registry.local:5000/vllm-ascend:baseline-$(cat /share/baseline_sha.txt)"
check "baseline TPS 已记录" "test -s /share/baseline/summary.json"
check "模型权重就位" "test -d /share/models/Qwen/Qwen3.5-7B-MoE"
check "SWE-smith dataset 就位" "test \$(wc -l < /share/dataset/swe_smith_128.jsonl) -eq 128"
check "Claude API 余额 > $1000" "curl -s -H \"x-api-key: \$ANTHROPIC_API_KEY\" https://api.anthropic.com/v1/messages -d '{\"model\":\"claude-haiku-4\",\"max_tokens\":5,\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}]}' | grep -q content"
check "GitHub Actions runner 在线" "ls /opt/actions-runner/_diag/*.log >/dev/null"
check "Grafana 可达" "curl -fsS http://localhost:3000/api/health"
check "ORACLE.json 存在" "test -s /share/ORACLE.json"
check "CLAUDE.md 就位" "test -s /share/tokenspeed-ascend/CLAUDE.md"
check "iterations/ 目录可写" "touch /share/tokenspeed-ascend/iterations/.write_test && rm /share/tokenspeed-ascend/iterations/.write_test"
echo "== All checks passed, ready to launch =="
EOF
chmod +x /share/tokenspeed-ascend/scripts/preflight_check.sh

# 跑一遍
bash /share/tokenspeed-ascend/scripts/preflight_check.sh
```

---

## Phase 1：Sprint A 自演进（8-15h，自动）

### 启动指令

```bash
# 一键启动
bash /share/tokenspeed-ascend/scripts/start_evolve.sh

# 观察
tmux attach -t ts-ascend-evolve
# Ctrl-B + D 离开（不停止）

# 实时看 artifact
watch -n 30 'ls -la /share/tokenspeed-ascend/iterations/ | tail -5'

# 实时看 metrics
open http://localhost:3000/d/ts-ascend-evolve
```

### Sprint A 预期 iter 列表

```
iter_001: platform_ascend.py + vendor="ascend" 注入
iter_002: torch.cuda.* 替换为 device-agnostic stream
iter_003: ACLGraph wrapper（借 vllm-ascend acl_graph.py）
iter_004: HCCL backend（借 vllm-ascend pyhccl.py）
iter_005: Qwen3-8B model adapter（最大不确定点）
iter_006: 数值对齐修 bug
iter_007: 单卡 prefill 跑通
iter_008: 单卡 decode 跑通 + 8 卡 TP smoke

Sprint A 终止条件：
- 数值对齐误差 < 1e-3
- 8 卡 TP 不崩溃
- TPS 不强制要求（只要能跑就行）
```

---

## Phase 2：Sprint B 自演进（15-25h，自动）

### 触发方式

Sprint A 终止后，orchestrator 自动收到信号继续 Sprint B。如果 paused，人工执行：

```bash
tmux send-keys -t ts-ascend-evolve "Sprint A 终止条件已达成。进入 Sprint B。目标：TPS >= 200。" Enter
```

### Sprint B 预期 iter 列表

```
iter_009-010: vllm-ascend MLA backend 集成（mla_v1.py）
iter_011-012: MoE backend 集成（fused_moe）
iter_013-014: scheduler_bridge.translate_execution_plan 优化（profile 驱动）
iter_015-016: input_buffer pinned tensor pool
iter_017-018: ACLGraph capture decode batch
iter_019: HCCL FusedReduceNorm 重写
iter_020: Retract write-back/load-back IO

Sprint B 终止条件：
- TPS ≥ 200（vllm-ascend × 1.1）
```

---

## Phase 3：Sprint C 自演进达标（12-30h，自动）

### Sprint C 重点

```
iter_021+: 各种 micro-opt
- buffer reuse
- async event chain
- prefetch policy tuning
- RadixTree on NPU 命中率验证
- Retract threshold 调优
- HCCL stream 隔离
- KV cache fp8 验证

Sprint C 终止条件：
- TPS ≥ 300（target）
- 自动 <loop-abort>DONE</loop-abort>
```

---

## 关键控制流：人工 oncall 操作手册

### 场景 1：iter 卡在 build 阶段

```bash
# 看 CI log
gh run list --limit 5
gh run view <run-id> --log

# 手工干预：在 iter-N branch 上推一个修复 commit
cd /share/tokenspeed-ascend && git checkout iter-N
# 修代码 -> commit -> push
# CI 自动重跑
```

### 场景 2：连续 stall（pause 状态）

```bash
# 看最近 3 轮 RETROSPECTIVE
for i in $(ls -t /share/tokenspeed-ascend/iterations | head -3); do
  cat /share/tokenspeed-ascend/iterations/$i/RETROSPECTIVE.md
  echo "---"
done

# 人工决策：换战场（如从 scheduler 优化转到 kernel 优化）
tmux attach -t ts-ascend-evolve
# 输入新方向 → 让 orchestrator 继续
```

### 场景 3：A3 节点崩溃

```bash
# 重启
ssh a3-node-01 'sudo reboot'
sleep 120

# 重启 runner
ssh a3-node-01 'cd /opt/actions-runner && ./svc.sh stop && ./svc.sh start'

# 让当前 iter 重跑
gh run rerun <last-failed-run>
```

### 场景 4：API cost 超限

```bash
# 检查 cost
curl https://api.anthropic.com/v1/organizations/usage?date=$(date +%Y-%m-%d) \
  -H "x-api-key: $ANTHROPIC_ADMIN_KEY"

# 如超限，降级到 Sonnet 主控（cost 减 80%）
tmux send-keys -t ts-ascend-evolve "/model claude-sonnet-4-5" Enter
```

---

## 长跑成本 / 时间预算

### 完整一次 35h 自演进 cost 拆解

| 项目 | 量化 | 成本 |
|------|------|------|
| A3 集群占用（含电力 + 折旧）| 35h × $80/h | $2,800 |
| Claude API（Opus + Sonnet × N）| ~35h × $15/h | $525 |
| 控制节点 + 存储 | 2 天 × $50 | $100 |
| GitHub Actions | self-hosted free | $0 |
| **单次总成本** | — | **~$3,400** |

### 时间安排建议

```
Week 1 周一 - 周五：Phase 0 基础设施（人工）
Week 2 周一 启动 evolve loop（自动）
Week 2 周三 中检（看 Sprint A 是否达成）
Week 2 周五 终态（Sprint C 收尾或决策续跑）

工程师投入：Phase 0 全职 + Phase 1-3 oncall（每天 1-2h 检查）
```

---

## 验收标准（项目结束时回看）

| 验收项 | 标准 |
|--------|------|
| 自演进系统可重复运行 | 同样输入再跑一次能得到一致结果 |
| 所有 iter artifact 完整 | 每个 iter 5 个文件齐全 |
| 最终 TPS ≥ 300 | SWE-smith concurrency=16 实测 |
| 数值对齐 | 误差 < 1e-3 |
| 代码可读 | 主控人工 review 通过率 ≥ 80% |
| 文档完整 | 每个关键决策有 RETROSPECTIVE 留痕 |
| Cost 在预算内 | 总成本 < $5,000 |

---

## 一句话闭环

> [PUA生效 🔥] 顶层结论：

**这套施工手册的底层逻辑：用 5 天人工建好"基线 + 闭环 + 护栏"三件套，然后让 Claude Code 在 35-72h 内跑出 200→300 TPS 的优化路径。owner 意识就是把所有不确定性 day-by-day 拍死——基线锁 SHA、目标定数字、护栏有 hard cap，剩下的就是相信工具链。**

> 不是相信 AI，是相信工程。AI 是工程的一部分。
