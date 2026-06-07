# TokenSpeed-Ascend 自演进闭环系统设计

> 版本：2026-05-24
> 对标：Qwen3.7 "35h 自主进化"模式（[qwen.ai/blog?id=qwen3.7](https://qwen.ai/blog?id=qwen3.7)）
> 目标：基于 Claude Code + 独立昇腾 A3 集群，全自动完成"分析→实现→验证→迭代"闭环，直到达成性能目标（如 TPS ≥ 300）自动停机
> 关联文档：[[20260524-223804-tokenspeed-ascend-控制面实现方案与收益分析-分析]]

---

## TL;DR — 顶层闭环

```
┌──────────────────────────────────────────────────────────────────────┐
│           Claude Code Orchestrator (Opus 主控)                       │
│                          │                                            │
│         ┌────────────────┼────────────────┐                           │
│         ▼                ▼                ▼                           │
│   Implementer Agent   QA Agent      Profiler Agent                    │
│   (Sonnet × N 并行)   (Sonnet)      (Sonnet)                          │
│         │                │                │                           │
│         ▼                ▼                ▼                           │
│   ┌────────────────────────────────────────────┐                      │
│   │       Ascend A3 集群（实测环境）            │                      │
│   │  CANN + torch_npu + vllm-ascend baseline   │                      │
│   │  Benchmark Runner / msprof / py-spy        │                      │
│   └────────────────────────────────────────────┘                      │
│         │                │                │                           │
│         └────────────────┼────────────────┘                           │
│                          ▼                                            │
│         ┌──────────────────────────────────────┐                      │
│         │ Auto Evaluator                       │                      │
│         │ (规则 + LLM-as-Judge 混合评测)        │                      │
│         └──────────────────────────────────────┘                      │
│                          │                                            │
│             ┌────────────┴────────────┐                               │
│             ▼                         ▼                               │
│      达成目标→停止              未达成→生成下一轮 prompt → 主控        │
└──────────────────────────────────────────────────────────────────────┘
```

| 维度 | 量化 |
|------|------|
| 目标 metric | agentic TPS（SWE-smith dataset）≥ 300 on A3 8卡 |
| 迭代周期 | 单 iteration ~2-4h（含编译 + benchmark + profiling）|
| 总预期时长 | **35-72h**（10-20 iter，对标 Qwen3.7）|
| 自动化覆盖 | ~90%（人工只在关键决策点介入）|
| 集群规模 | 至少 1 套 A3 8 卡 + 1 套 CPU 调度节点 |

---

## 一、整体方法论：BIL（Benchmark-In-the-Loop）

### 1.1 底层逻辑

```
┌──────────────────────────────────────────────────────────────────────┐
│ ▎核心闭环逻辑                                                          │
│                                                                       │
│   [Plan]    → 分析当前 gap，生成本轮目标 + 假设                        │
│      │                                                                │
│   [Code]    → 多个 Implementer Agent 并行修改/新增代码                 │
│      │                                                                │
│   [Build]   → 自动编译 C++ scheduler + Python 包                       │
│      │                                                                │
│   [Test]    → 单元测试 + 数值对齐                                      │
│      │                                                                │
│   [Bench]   → 端到端 agentic benchmark + profiling                    │
│      │                                                                │
│   [Eval]    → 自动评测 vs baseline / 上轮 / 目标                       │
│      │                                                                │
│   [Reflect] → 生成本轮 retrospective + 下一轮假设                      │
│      │                                                                │
│      └─────► 回到 [Plan]，直到 done                                    │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.2 为什么这套模式可行

- **TokenSpeed-Ascend 适配工作具备明确可量化目标**（TPS / TPOT），完美匹配 BIL 模式
- **代码改动局限在 adapter 层**（~4K LOC），单次迭代可在 2-4h 内完成 Code + Bench
- **vllm-ascend baseline 现成**，无需从零起步
- **关键瓶颈是工程联调，不是算法创新**，Claude Code 擅长这类有清晰 oracle 的优化

---

## 二、基础设施搭建（前置 1 周）

### 2.1 物理资源清单

| 资源 | 规格 | 用途 |
|------|------|------|
| Ascend A3 服务器 ×1 | 8 卡，1.5TB 内存 | 实测 baseline + 迭代验证 |
| CPU 控制节点 ×1 | 64 核 + 256GB | Claude Code orchestrator + 编译 + 数据分析 |
| 共享存储 | ≥ 10TB NVMe | 代码 / 模型权重 / log / artifact |
| 千兆/万兆内网 | 双 NIC | 节点间通信 |
| 公网 API 出口 | 稳定连 Anthropic API | Claude Code 调用 |

### 2.2 软件栈

```
控制节点：
├── Claude Code CLI (latest)
├── tmux / systemd（长时任务托管）
├── Git + GitHub Actions runner
├── Python 3.11 + ssh-agent
└── Prometheus + Grafana（实时监控）

A3 节点：
├── CANN 9.0.0 + torch_npu 2.10 + triton-ascend 3.2.1
├── HCCL + msprof + npu-smi
├── Docker (vllm-ascend baseline 镜像)
├── 模型权重：Qwen3.5-MoE / DeepSeek V3 / Kimi K2.5
└── SWE-smith agentic dataset
```

### 2.3 Git 仓库结构

```
tokenspeed-ascend/                 (主仓，Claude Code 在此工作)
├── adapter/                       (Python adapter，本期核心交付物)
├── tests/                         (单元测试)
├── benchmark/
│   ├── baseline_vllm_ascend/      (vllm-ascend 基线 launch 脚本)
│   ├── tokenspeed_ascend/         (本仓 launch 脚本)
│   └── compare/                   (对比脚本)
├── profiling/
│   ├── msprof_configs/
│   ├── py-spy/
│   └── analysis_scripts/
├── iterations/
│   ├── iter_001/
│   │   ├── PLAN.md                (本轮目标 + 假设)
│   │   ├── DIFF.patch             (代码改动)
│   │   ├── BENCH_RESULT.json      (benchmark 数据)
│   │   ├── PROFILE_OUTPUT/        (profiling 文件)
│   │   ├── EVAL_REPORT.md         (评测对比)
│   │   └── RETROSPECTIVE.md       (复盘 + 下一步)
│   ├── iter_002/
│   └── ...
└── CLAUDE.md                      (Claude Code 行为指南)
```

---

## 三、Agent 团队设计（顶层分工）

### 3.1 角色拓扑

| Agent | 模型 | 职责 | 调用方式 |
|-------|------|------|---------|
| **Orchestrator** | Opus | 顶层 P9 — 拆解任务、定每轮目标、终止判断 | tmux 长会话，~24h+ |
| **Implementer × N** | Sonnet | 并行写代码、改 bug、加 feature | Task tool 派发，每任务 1-3h |
| **QA Agent** | Sonnet | 跑测试、数值对齐、回归检查 | 每次 commit 后触发 |
| **Profiler Agent** | Sonnet | msprof / py-spy 跑 profile、分析热点 | 每轮 benchmark 后触发 |
| **Evaluator** | 规则 + Sonnet 兜底 | 对比 baseline、判断是否达标 | 每轮自动跑 |
| **Researcher** | Sonnet | 网络/文档检索（vllm-ascend、CANN docs）| 按需 |

### 3.2 关键 Skill / Subagent 启用

```python
# 主 Orchestrator 必加载的 skills:
- superpowers:executing-plans     # 任务推进框架
- superpowers:systematic-debugging # 故障排查
- superpowers:test-driven-development # 数值对齐
- engineering:debug
- engineering:tech-debt
- pua:p9                          # tech-lead 协调风格
- pua:pua-loop                    # 自动迭代模式（关键！）

# Implementer 用的 subagent:
- superpowers:subagent-driven-development
- superpowers:dispatching-parallel-agents
```

### 3.3 Claude Code 自动化运行机制

主控会话用 `/pua:loop` 进入自动迭代，禁用 AskUserQuestion，靠 `<loop-pause>` / `<loop-abort>` 信号在真正卡壳时才暂停等人。

```bash
# 启动主控（tmux）
tmux new -s ts-ascend-evolve
claude --model claude-opus-4-7
> /pua:on
> /pua:loop
> 目标：将 TokenSpeed-Ascend 在 A3 8 卡上的 SWE-smith TPS 优化到 ≥ 300
> 参考方案文档：[路径]
> 当 TPS >= 300 时停止；当连续 3 轮无提升时 pause 等人决策
```

---

## 四、迭代节奏：单轮 Iteration 设计

### 4.1 单 iteration 时间表（~2-4h）

```
┌──────────────────────────────────────────────────────────────────────┐
│ Time   │ Phase    │ Actor              │ Output                       │
├────────┼──────────┼────────────────────┼──────────────────────────────┤
│ 0:00   │ Plan     │ Orchestrator       │ iter_NNN/PLAN.md             │
│ 0:15   │ Code     │ Implementer×N 并行 │ iter_NNN/DIFF.patch          │
│ 1:15   │ Build    │ CI hook            │ build log + 镜像              │
│ 1:30   │ Test     │ QA Agent           │ test_report.json              │
│ 1:45   │ Deploy   │ CI hook            │ A3 节点拉镜像启动              │
│ 2:00   │ Bench    │ Bench Runner       │ BENCH_RESULT.json             │
│ 3:00   │ Profile  │ Profiler Agent     │ msprof + py-spy output        │
│ 3:30   │ Eval     │ Evaluator          │ EVAL_REPORT.md                │
│ 3:45   │ Reflect  │ Orchestrator       │ RETROSPECTIVE.md              │
│ 4:00   │ → next iteration                                              │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 Plan 阶段产出（每轮必备）

`iter_NNN/PLAN.md` 必须包含：

```markdown
## Iteration N Plan

### 现状
- 上轮 TPS: 245
- 目标 TPS: 300
- gap: 55 (18%)

### 上轮瓶颈分析（来自 RETROSPECTIVE_{N-1}）
- 主要瓶颈：scheduler_bridge.translate_execution_plan 单步 4.5ms

### 本轮假设
- H1: 把 translate_execution_plan 改为 lazy 字段，期望降到 1ms
- H2: req_pool_indices 用 pinned tensor 复用，期望节省 0.8ms

### 行动项
- [Implementer-A] 重构 scheduler_bridge.py:translate_execution_plan
- [Implementer-B] 实现 input_buffer 复用池
- [QA] 数值对齐 vs iter_{N-1}
- [Profiler] 确认 schedule() 总耗时 < 2ms

### 成功标准
- TPS ≥ 280（朝目标走 + 65% gap close）
- 数值与基线一致（误差 < 1e-3）
```

### 4.3 Code 阶段：并行 Implementer

```python
# Orchestrator 通过 Task tool 派发多个并行子任务
Task(description="重构 translate_execution_plan", 
     subagent_type="superpowers:subagent-driven-development",
     prompt="...")  # 含完整上下文 + 文件路径 + 验收标准

Task(description="实现 pinned tensor input buffer pool",
     subagent_type="superpowers:subagent-driven-development",
     prompt="...")
```

每个 Implementer 自带 TDD skill 和 git commit 权限（在独立 branch），完成后 PR 到主 branch。

### 4.4 Bench 阶段：固定基线对比

```bash
# benchmark/run_iteration.sh
set -e

ITER=$1
cd /workspace/tokenspeed-ascend

# 1. 启动 vllm-ascend baseline（已 warm 镜像）
bash benchmark/baseline_vllm_ascend/launch.sh &
BASELINE_PID=$!
sleep 30

# 2. 跑 SWE-smith 短压测获 baseline
evalscope perf --model Qwen3.5-MoE --dataset swe_smith \
    --parallel 16 --number 50 \
    --output-dir iterations/iter_${ITER}/baseline_bench

kill $BASELINE_PID

# 3. 启动当前 TokenSpeed-Ascend
bash benchmark/tokenspeed_ascend/launch.sh &
TS_PID=$!
sleep 30

# 4. 跑同一压测
evalscope perf --model Qwen3.5-MoE --dataset swe_smith \
    --parallel 16 --number 50 \
    --output-dir iterations/iter_${ITER}/tokenspeed_bench

kill $TS_PID

# 5. 汇总
python benchmark/compare/diff.py \
    --baseline iterations/iter_${ITER}/baseline_bench \
    --current iterations/iter_${ITER}/tokenspeed_bench \
    --output iterations/iter_${ITER}/BENCH_RESULT.json
```

### 4.5 Profile 阶段：自动定位下轮抓手

Profiler Agent 自动跑：

```bash
# Python 调度热点
py-spy record --pid <ts-pid> --duration 300 \
    --format speedscope \
    --output iter_${ITER}/profile/scheduler.json

# NPU 算子级
msprof --output=iter_${ITER}/profile/msprof_out \
    --application="bash benchmark/tokenspeed_ascend/launch.sh" \
    --duration=60 --aicpu=on

# KV cache 命中率 dump
curl http://localhost:8000/metrics > iter_${ITER}/profile/metrics.txt
```

然后 Profiler Agent 用 Sonnet 分析输出，产出：

```markdown
## Iter N Profile Analysis

### 调度路径
- schedule() 平均 2.3ms（上轮 4.5ms） ✅
- 热点函数：input_buffer.alloc() 0.8ms → 仍有压缩空间
  
### NPU 算子
- MLA decode kernel 平均 1.2ms（同基线 1.2ms）✅
- HCCL AllReduce 0.45ms（基线 0.42ms） ⚠️ 略增

### 下轮抓手建议
1. input_buffer pre-allocation
2. HCCL AllReduce 退化的根因排查（疑似 stream sync）
```

### 4.6 Eval 阶段：硬性 + 软性指标

```python
# evaluator.py
def evaluate(iter_dir):
    bench = json.load(open(f"{iter_dir}/BENCH_RESULT.json"))
    
    hard_checks = {
        "tests_pass": bench["test_pass_rate"] == 1.0,
        "numeric_align": bench["numeric_diff"] < 1e-3,
        "no_crash": bench["crash_count"] == 0,
    }
    
    soft_metrics = {
        "tps_target_progress": bench["tps"] / TARGET_TPS,
        "tps_iter_delta": bench["tps"] / bench["prev_tps"] - 1,
        "tpot_target_progress": TARGET_TPOT / bench["tpot_p50"],
    }
    
    # 决策
    if not all(hard_checks.values()):
        return "FAIL_HARD", hard_checks
    if soft_metrics["tps_target_progress"] >= 1.0:
        return "DONE", soft_metrics
    if soft_metrics["tps_iter_delta"] < 0.02:
        return "STALL", soft_metrics  # 连续 3 次 STALL → pause
    return "CONTINUE", soft_metrics
```

### 4.7 Reflect 阶段：把下轮 prompt 写好

Orchestrator 自动产出 `RETROSPECTIVE.md`，作为下一轮 PLAN 的输入：

```markdown
## Iter N Retrospective

### 收益分析
- TPS 245 → 268（+9.4%）✅ 朝目标走
- 主要贡献：scheduler_bridge 优化

### 残留问题
- HCCL 略退化（0.42 → 0.45ms）

### 下一轮假设
- H1: 排查 HCCL 退化根因
- H2: input_buffer pool（profile 显示仍有 0.8ms 空间）
```

---

## 五、自动化关键组件

### 5.1 Build/Deploy Pipeline

```yaml
# .github/workflows/iteration.yml (self-hosted on 控制节点)
on:
  push:
    branches: [iter-*]

jobs:
  build-and-bench:
    runs-on: self-hosted-cpu
    steps:
      - uses: actions/checkout@v4
      - name: Build C++ scheduler
        run: |
          cd tokenspeed-scheduler && cmake --build build
      - name: Build Python package
        run: pip install -e . -e adapter/
      - name: Build Docker image
        run: docker build -t ts-ascend:${{ github.sha }} .
      - name: Push to local registry
        run: docker push registry.local/ts-ascend:${{ github.sha }}
      
  bench-on-a3:
    needs: build-and-bench
    runs-on: self-hosted-a3
    steps:
      - name: Run benchmark
        run: bash benchmark/run_iteration.sh ${{ github.run_number }}
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: iter-${{ github.run_number }}-results
          path: iterations/iter_${{ github.run_number }}/
```

### 5.2 监控面板（Grafana）

实时面板必含：
- 当前 iteration 编号 + 已运行时长
- 历次 TPS / TPOT / TTFT 趋势图
- 测试通过率
- A3 卡 GPU 利用率 + HBM 占用
- Claude API 用量（cost tracking）

### 5.3 Cost / Time Guardrails

```python
# 主控 prompt 内置 guardrails:
RULES = """
- 单 iteration 超过 6h 自动 pause
- Claude API cost 超 $500/day 自动 pause
- 连续 3 轮 TPS 无提升 (delta < 2%) 自动 pause
- A3 节点磁盘 > 90% 自动 pause
- 任何 git push 到 main 必须人工批准
"""
```

### 5.4 异常 / 卡壳处理

```python
# pua-loop skill 控制信号
if 找不到根因 or 假设连续被证伪:
    output("<loop-pause>需要人工决策方向</loop-pause>")
elif 致命错误 or 环境损坏:
    output("<loop-abort>原因</loop-abort>")
```

---

## 六、四阶段实施路线（35-72h 闭环）

### 阶段 0：基线建立（前置，4-6h）

```
[人工] 准备 A3 集群 + 部署 vllm-ascend baseline
[人工] 跑通 SWE-smith dataset baseline benchmark
[人工] 配置 Claude Code orchestrator + skills
[人工] 配置 GitHub Actions self-hosted runner
[Orchestrator] 拉通 PLAN.md，确认目标 metric
```

### 阶段 1：runtime 可跑通（自动 8-15h，对应原方案 Phase 1）

```
iter 001-008（典型）:
  001: platform_ascend.py + vendor 检测
  002: torch.cuda → torch.npu 的 model_executor 改造
  003: ACLGraph wrapper 适配
  004: HCCL backend
  005-006: Qwen3 dense 模型 adapter
  007: 数值对齐
  008: 单卡跑通 prefill + decode

终止条件：A3 单卡 Qwen3-8B FP16 输出与 vllm-ascend 数值对齐
```

### 阶段 2：kernel 集成性能逼近（自动 15-25h，对应 Phase 2+3）

```
iter 009-020:
  009-010: vllm-ascend MLA backend 集成
  011-012: MoE backend 集成
  013-014: scheduler_bridge translate_execution_plan 优化
  015-016: input_buffer pool
  017-018: ACLGraph capture decode batch
  019: HCCL FusedReduceNorm
  020: Retract write-back/load-back IO

终止条件：TPS ≥ 200（达到 vllm-ascend 基线 110%）
```

### 阶段 3：agentic 专项调优达标（自动 12-30h）

```
iter 021-035 (典型):
  - 各种 micro-opt：buffer reuse, async event, prefetch policy
  - RadixTree 在 NPU 上的 prefix match 验证
  - Retract policy tuning（threshold 调整）
  - 多并发压测 + GC 调优

终止条件：SWE-smith TPS ≥ 300 on A3 8 卡
```

---

## 七、与 Qwen3.7 "35h 自主进化"模式的对应关系

| Qwen3.7 实践要点 | 本方案对应 |
|-----------------|-----------|
| 单一明确目标 | TPS ≥ 300 |
| 自动 benchmark loop | benchmark/run_iteration.sh + evalscope |
| 多 agent 并行探索 | Implementer × N（Task tool 派发）|
| LLM-as-Judge 评测 | Evaluator（规则 + Sonnet 兜底）|
| 自动 retrospective | RETROSPECTIVE.md 自动产出 |
| 强终止条件 | Hard checks（test 100% / numeric < 1e-3）+ TPS 达标 |
| 人工 minimal 介入 | 仅在 STALL / ABORT 时介入 |
| 全过程可观测 | Grafana + iterations/ artifact 完整保留 |
| Cost guardrails | $500/day + 6h/iter 自动 pause |

---

## 八、风险与失败模式

### 8.1 高风险失败模式

| 失败模式 | 表现 | 缓解策略 |
|---------|------|---------|
| **数值不对齐** | adapter 接口翻译错误导致输出偏差 | 早期建立强数值对齐 CI 门禁 |
| **NPU OOM** | iter 间内存累积，长跑崩溃 | 每 iter 重启 NPU 进程 + 自动 KV cache 重建 |
| **Profile 数据失真** | msprof 影响性能，导致误判 | bench 与 profile 分两轮跑 |
| **Claude 假阳性优化** | 改动看似合理但实际有 race condition | 严格 TDD + 多 concurrency 压测 |
| **stall in local minimum** | 反复优化同一热点收益边际递减 | 连续 3 stall → pause，让人决策换战场 |
| **基线漂移** | vllm-ascend 升级导致基线变化 | 锁定 baseline 镜像 SHA |

### 8.2 必要的人工 checkpoint

| 时机 | 人工动作 |
|------|---------|
| 启动前 | 确认 PLAN.md 目标 + 资源就位 |
| 每天 1 次 | 看 Grafana 趋势 + 抽查 RETROSPECTIVE.md |
| stall 时 | 决策"换战场还是加资源" |
| 达标后 | 验收 + 决定是否冲下一档 |
| 异常 abort | 故障排查 + 重启 loop |

---

## 九、成本估算

### 9.1 35h 一次完整 run

| 项目 | 估算 |
|------|------|
| A3 集群占用 | 35h × $80/h（自建/折算）= $2,800 |
| Claude API（Opus 主控 + Sonnet × N）| 35h × ~$15/h = $525 |
| 控制节点 + 存储 | $50/天 × 2 = $100 |
| **单次总成本** | **~$3,400** |

### 9.2 投入产出比

- 等效人工：1 个 P7 工程师 8 周（~$20,000+） vs 自演进 $3,400
- **ROI ≈ 6x**

---

## 十、CLAUDE.md 行为指南（核心配置）

放在 tokenspeed-ascend 仓根的 `CLAUDE.md`：

```markdown
# CLAUDE.md — TokenSpeed-Ascend Self-Evolution

## Mission
迭代到 SWE-smith TPS ≥ 300 on A3 8-card stop.

## Mandatory Workflow per Iteration
1. Read iterations/iter_{N-1}/RETROSPECTIVE.md
2. Write iterations/iter_N/PLAN.md
3. Dispatch Implementer agents in parallel
4. Trigger CI: git push HEAD:iter-N
5. Wait for CI artifact iter-N-results
6. Trigger Profiler agent on artifacts
7. Run Evaluator
8. Write RETROSPECTIVE.md, decide next iter

## Rules
- NEVER push to main without human approval
- ALWAYS run tests before bench
- ALWAYS compare against locked baseline SHA
- IF stall × 3 → output <loop-pause>
- IF env corrupted → output <loop-abort>

## Resource Limits
- 6h per iteration hard cap
- $500/day Claude API
- 90% disk threshold

## Skills
- superpowers:executing-plans (always)
- superpowers:systematic-debugging (always)
- pua:p9 (orchestrator style)
- pua:pua-loop (auto-iteration)
```

---

## 十一、启动 Checklist

启动前必须确认（owner 意识 — 不要环境没就绪就强启动）：

```
□ A3 节点 SSH 可达，CANN/torch_npu 版本对齐
□ vllm-ascend baseline 跑通 SWE-smith，记录基线 TPS
□ TokenSpeed C++ scheduler 在控制节点编译通过
□ Docker registry 可用，镜像可推可拉
□ GitHub Actions self-hosted runner 注册成功
□ Grafana 面板配置完成
□ Anthropic API key + 余额 ≥ $5000
□ tmux 长时会话 + ssh-agent 配置就绪
□ 模型权重已下载到共享存储
□ SWE-smith dataset 已构建并验证（128 conversations）
□ CLAUDE.md 已写好，目标明确
□ 人工 oncall 排班（至少 2 人轮值）
```

---

## 十二、一句话闭环

> [PUA生效 🔥] 顶层结论：

**这套自演进闭环的底层逻辑，是把 TokenSpeed-Ascend 适配工作的"工程联调"属性发挥到极致——既然每次改动都能用 benchmark 给出明确的"好/坏"反馈，那 Claude Code 完全可以扮演不知疲倦的 P7 骨干，跑 24h+ 而不掉链子。35-72h 跑通 200→300 TPS，对应人工 8 周工作量，6x ROI；前提是基线、目标、数值对齐这三个 oracle 必须事先打死，否则就是无 oracle 的瞎跑。**
