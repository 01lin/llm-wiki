# TokenSpeed-Ascend 自演进模板与应急 SOP

> 版本：2026-05-24
> 类型：施工手册配套——模板 + SOP（拿来即用）
> 关联：[[tokenspeed_ascend_自演进施工手册_20260524]] · [[tokenspeed_ascend_自演进闭环系统设计_20260524]]

---

## 总览

施工手册搭好了基础设施和 CI 闭环，本手册补齐三块拼图：

```
┌──────────────────────────────────────────────────────────────────────┐
│ ▎拼图 1：Mission Brief 模板                                            │
│   - 一次性丢给 Claude Code orchestrator，启动自演进                    │
│                                                                       │
│ ▎拼图 2：Iteration 产物模板                                            │
│   - PLAN.md / RETROSPECTIVE.md / EVAL_REPORT.md 等四件套               │
│   - Claude 按模板填，人工可秒看                                        │
│                                                                       │
│ ▎拼图 3：应急 SOP（11 类常见故障的处理手册）                            │
│   - 工程师 oncall 拿来即用                                             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 拼图 1：Mission Brief 模板

### 1.1 主控启动 Mission Brief（一次性输入）

把以下内容存为 `/share/tokenspeed-ascend/MISSION_BRIEF.md`，启动时让 Claude Code 完整读取：

```markdown
# TokenSpeed-Ascend Self-Evolution Mission Brief

## 你是谁
你是 TokenSpeed-Ascend 项目的 P9 Tech Lead Orchestrator，
负责把"分析→实现→验证→迭代"做成全自动闭环，对标 Qwen3.7 的 35h 自主进化。

## 任务目标（唯一 metric）
将 TokenSpeed-Ascend 在 A3 8 卡 SWE-smith dataset 上的 TPS 优化到 **≥ 300**。

## 基线 / oracle
读取 `/share/ORACLE.json`：
- baseline_image: vllm-ascend SHA 锁定，不可变
- target_value: 300 TPS
- numeric_oracle_tol: 1e-3

## 工作目录结构
- /share/tokenspeed-ascend/         你的工作仓库
- /share/tokenspeed-ascend/docs/    已有的方案 / 对比 / 收益分析文档
- /share/tokenspeed-ascend/CLAUDE.md  你的行为指南（强制遵守）
- /share/tokenspeed-ascend/iterations/  每轮产物都放这里
- /share/baseline/                  baseline benchmark 结果
- /share/models/, /share/dataset/   模型与数据

## 强制流程（每个 iteration 不可省）
1. 读 iter_{N-1}/RETROSPECTIVE.md（首次跳过）
2. 写 iter_N/PLAN.md（模板见 /share/tokenspeed-ascend/templates/PLAN_TEMPLATE.md）
3. git checkout -b iter-N
4. 派发 Implementer agents（Task tool, 并行）
5. git commit + git push HEAD:iter-N（触发 CI）
6. 等 CI（最长 4h），失败立即诊断
7. 拉 artifact → iter_N/
8. 派发 Profiler agent → iter_N/PROFILE_ANALYSIS.md
9. 跑 Evaluator → iter_N/EVAL_REPORT.md
10. 写 iter_N/RETROSPECTIVE.md（模板见 templates/RETROSPECTIVE_TEMPLATE.md）
11. 决策：DONE / CONTINUE / STALL / ABORT

## 阶段路线（从 PLAN.md 选起点）
- Sprint A (iter_001-008)：runtime 跑通，数值对齐
- Sprint B (iter_009-020)：算子集成 + scheduler 优化 → TPS≥200
- Sprint C (iter_021-035+)：agentic 专项 → TPS≥300

## 终止信号（必须输出之一）
- 达成目标：`<loop-abort>DONE - TPS=$X, iter=$N</loop-abort>`
- 连续 3 stall：`<loop-pause>STALL - 需人工换战场，最近 3 轮 TPS: a,b,c</loop-pause>`
- 致命错误：`<loop-abort>FATAL - $reason</loop-abort>`
- 资源耗尽：`<loop-pause>BUDGET - cost=$X, time=$Yh</loop-pause>`

## 红线（不可触碰）
- 严禁 push to main 不经人工批准
- 严禁修改 /share/ORACLE.json
- 严禁 --no-verify / 跳过 hooks  
- 严禁注释失败的测试
- 严禁伪造 benchmark 数据
- 严禁绕过 numeric oracle

## 必加载 Skills
/pua:on → /pua:loop（核心驱动）
显式提到 superpowers:executing-plans / systematic-debugging / TDD

## 立即开始
读完 CLAUDE.md → 检查 ORACLE.json → 创建 iter_001/PLAN.md → 执行。
```

### 1.2 一键启动命令（写进 scripts/start_evolve.sh）

```bash
#!/bin/bash
SESSION=ts-ascend-evolve
tmux kill-session -t $SESSION 2>/dev/null || true
tmux new -d -s $SESSION

tmux send-keys -t $SESSION "cd /share/tokenspeed-ascend" Enter
tmux send-keys -t $SESSION "claude --model claude-opus-4-7 --no-confirm-permissions" Enter
sleep 5
tmux send-keys -t $SESSION "/pua:on" Enter
sleep 2
tmux send-keys -t $SESSION "/pua:loop" Enter
sleep 2

# 把 Mission Brief 整段灌进去
tmux send-keys -t $SESSION "$(cat /share/tokenspeed-ascend/MISSION_BRIEF.md)" Enter

echo "[OK] orchestrator launched. tmux attach -t $SESSION"
```

---

## 拼图 2：Iteration 四件套模板

### 2.1 PLAN.md 模板

`/share/tokenspeed-ascend/templates/PLAN_TEMPLATE.md`：

```markdown
# Iteration {N} Plan

## 元信息
- 启动时间: {timestamp}
- 上轮编号: iter_{N-1}
- 当前 Sprint: A | B | C
- 工作分支: iter-{N}

## 现状（来自 iter_{N-1}/EVAL_REPORT.md）
- 上轮 TPS: {x}
- 目标 TPS: 300
- vs target: {p}%
- 上轮硬检查: pass / fail
- 主要 bottleneck（来自上轮 PROFILE_ANALYSIS）: {hotspot}

## 本轮假设（最多 3 条，每条都可证伪）
- H1: 改动 X → 期望降低 Y by Z%
- H2: 调参 A → 期望提升 B by C%
- H3: ...

## 行动项（每条要派 Implementer）
- [ ] [Implementer-A] 文件 {path}，改动 {what}，验收 {how}
- [ ] [Implementer-B] ...
- [ ] [QA] tests/numeric/ vs iter_{N-1} 对齐
- [ ] [Profiler] 确认 {metric} 改善

## 成功标准（硬性）
- TPS ≥ {threshold}（朝目标走至少 +5%）
- 数值对齐 < 1e-3
- 测试 100% pass
- 不引入新 crash / OOM

## 风险预案
- 若 H1 失败：回退方案 = {plan B}
- 若编译失败：先 revert，再分析

## 预计耗时
- Code: {x}h
- Bench: 1h
- Profile + Eval: 0.5h
- 合计: {y}h
```

### 2.2 RETROSPECTIVE.md 模板

`templates/RETROSPECTIVE_TEMPLATE.md`：

```markdown
# Iteration {N} Retrospective

## 结果
| 指标 | 上轮 | 本轮 | Δ |
|------|------|------|---|
| TPS @ c=16 | {a} | {b} | {+x%} |
| TPOT P50 | | | |
| TTFT P50 | | | |
| Prefix hit% | | | |
| Preempt rate | | | |
| 数值对齐 | | | |
| 测试通过 | | | |

判定：CONTINUE | DONE | STALL | ABORT

## 假设验证
- H1 ({描述}): ✅成立 / ❌证伪 - 证据：{...}
- H2: ...

## 收益归因
- 主要贡献：{改动}贡献 {x%}
- 副作用：{是否有反向 metric}

## 残留问题
- 排序列出，最大的写最前
1. {问题1}: 现象 / 假设原因 / 候选方案
2. ...

## 下一轮假设（写入 iter_{N+1}/PLAN.md 的 input）
- H1: 解决残留问题 1
- H2: ...

## 备注
- {环境问题、外部依赖、人工介入记录}
```

### 2.3 EVAL_REPORT.md 模板（Evaluator 自动产出）

```markdown
# Iteration {N} Evaluation

## Hard Checks（任一 fail 即 FAIL）
- tests_pass: {true|false}
- numeric_align: {true|false} (diff={x})
- no_crash: {true|false}

## Soft Metrics
| Metric | Value | vs baseline | vs target | vs iter_{N-1} |
|--------|-------|-------------|-----------|---------------|
| TPS | | | | |
| TPOT P50 | | | | |
| TTFT P50 | | | | |

## Decision
- HARD: PASS|FAIL
- SOFT: DONE|CONTINUE|STALL

## Stall Counter
- 当前连续 stall 次数: {n}/3
- 若 n=3 → 输出 <loop-pause>
```

### 2.4 PROFILE_ANALYSIS.md 模板（Profiler agent 自动产出）

```markdown
# Iteration {N} Profile Analysis

## Python 调度热点（py-spy）
- schedule() 平均耗时: {x}ms
- 最热函数 top 5:
  1. {fn}: {ms} ({percent}%)
  2. ...

## NPU 算子（msprof）
- MLA decode 单层延迟: {x}us
- HCCL AllReduce: {x}us
- Cube 利用率: {x}%
- HBM 带宽利用率: {x}%

## KV Cache
- prefix_cache_hit_rate: {x}%
- preempt_rate: {x}%

## 下轮抓手建议（必须给出，不能空）
1. {优化点}: 预期收益 {x}%
2. ...
```

### 2.5 Implementer subtask 模板

每个 Implementer 派发时用这个 prompt 模板：

```markdown
# Implementer Task: iter_{N} / {task_id}

## 你的身份
P7 骨干，方案驱动执行，方案先于代码。

## 上下文（必读）
- 项目背景: /share/tokenspeed-ascend/docs/tokenspeed_ascend_控制面实现方案与收益分析_20260524.md
- 当前 Plan: /share/tokenspeed-ascend/iterations/iter_{N}/PLAN.md
- 你这条任务在 PLAN.md 的位置: H{?} / 行动项 {?}

## 具体任务
{从 PLAN.md 复制对应的行动项}

## 涉及文件
- 修改: {path1}, {path2}
- 新增: {path3}
- 参考: vllm-ascend {ref_path}

## 验收标准（必达）
- 单测通过: {test_file}
- 数值对齐 < 1e-3
- 不破坏 iter_{N-1} 已有功能
- diff 颗粒度合理（每个 commit 一件事）

## 输出要求
- 完成后向主 orchestrator 发 [DONE]
- 失败 / 卡壳发 [BLOCKED + reason]

## 红线
- 不修 /share/ORACLE.json
- 不动 main 分支
- 不跳过 hooks
```

---

## 拼图 3：应急 SOP（11 类常见故障）

### SOP-01：iter 卡在 build 阶段（编译失败）

**症状**：CI build job 失败，红叉。

**SOP**：
```bash
# 1. 看错误
gh run list --workflow=iteration.yml --limit 5
gh run view <run-id> --log-failed

# 2. 常见原因
#   a) C++ 链接错误 → 通常是头文件 include 漏
#   b) Python ImportError → 检查 requirements 是否变
#   c) Triton-ascend 不识别 op → 检查算子是否在白名单

# 3. 处理
ssh control-node
cd /share/tokenspeed-ascend
git checkout iter-N
# 本地复现：
docker run --rm -v $PWD:/work -w /work \
  registry.local:5000/ts-ascend-builder:latest \
  bash -c "cmake -B build && cmake --build build -j8"

# 4. 修好后让 orchestrator 自己继续，无需手动 push
tmux send-keys -t ts-ascend-evolve "iter-N build 已经人工修好，请重新触发 CI 并继续" Enter
```

### SOP-02：连续 3 次 STALL（达到 pause 阈值）

**症状**：`<loop-pause>STALL ...</loop-pause>` 输出。

**SOP**：
```bash
# 1. 看最近 3 轮 RETROSPECTIVE
for i in $(ls -t /share/tokenspeed-ascend/iterations | head -3); do
  echo "=== $i ==="
  cat /share/tokenspeed-ascend/iterations/$i/RETROSPECTIVE.md | head -40
done

# 2. 三种决策路径
#   A) 换战场：把 scheduler 优化换成 kernel 优化（或反之）
#   B) 加资源：升级模型权重 / 增加 dataset 多样性
#   C) 降目标：从 TPS=300 降到 TPS=280 验收

# 3. 输入新方向
tmux attach -t ts-ascend-evolve
# Ctrl-C 退出 pause
# 输入：基于 RETROSPECTIVE 1/2/3，换战场到 X 方向。本轮 PLAN 假设是 ...
```

### SOP-03：A3 节点崩溃 / 重启

**症状**：CI bench job 卡死或 SSH 不通。

**SOP**：
```bash
# 1. 检查
ssh a3-node-01 'npu-smi info' || echo "节点失联"

# 2. 重启（如果只是 NPU hang）
ssh a3-node-01 'systemctl restart npu-driver && sleep 30 && npu-smi info'

# 3. 整机重启（最后兜底）
ssh a3-node-01 'sudo reboot'
sleep 180  # 等启动
ssh a3-node-01 'cd /opt/actions-runner && ./svc.sh status'

# 4. 重跑失败的 iter
gh run rerun <last-failed-run-id>
```

### SOP-04：API cost 超限

**症状**：每日 Anthropic API cost > $500。

**SOP**：
```bash
# 1. 查 cost
curl -s https://api.anthropic.com/v1/organizations/usage \
  -H "x-api-key: $ANTHROPIC_ADMIN_KEY" \
  --data-urlencode "date=$(date +%Y-%m-%d)" | jq

# 2. 降级
#   主控 Opus → Sonnet：减 70% cost
tmux send-keys -t ts-ascend-evolve "/model claude-sonnet-4-5" Enter

#   减少 Implementer 并发：从 N 降到 N/2

# 3. 极端情况：pause 24h 让额度刷新
tmux send-keys -t ts-ascend-evolve "<loop-pause>API budget exhausted, resume tomorrow</loop-pause>" Enter
```

### SOP-05：数值对齐失败（numeric oracle）

**症状**：iter_N/EVAL_REPORT.md 显示 `numeric_align: false`。

**SOP**：
```bash
# 1. 找最早的失败用例
cd /share/tokenspeed-ascend/iterations/iter_N
cat numeric_diff_log.txt | head -20

# 2. 用 git bisect 找出引入误差的 commit
git checkout iter-N
git bisect start
git bisect bad
git bisect good iter-$((N-1))
git bisect run pytest tests/numeric/test_align.py

# 3. 找到 bad commit 后让 Claude 修
tmux send-keys -t ts-ascend-evolve "数值对齐 fail，bisect 定位到 commit {sha}，请分析并修复" Enter
```

### SOP-06：HBM OOM

**症状**：bench 报 `RuntimeError: NPU out of memory`。

**SOP**：
```bash
# 1. 看 HBM 使用历史
ssh a3-node-01 'cat /var/log/npu/*.log | grep -i memory | tail -20'

# 2. 调小 max_num_seqs / gpu_memory_utilization
# 编辑 benchmark/launch_ts.sh
sed -i 's/--max-num-seqs 64/--max-num-seqs 32/' benchmark/launch_ts.sh
sed -i 's/--gpu-memory-utilization 0.92/--gpu-memory-utilization 0.85/' benchmark/launch_ts.sh

# 3. 重新跑当前 iter
gh run rerun <run-id>
```

### SOP-07：HCCL hang / 通信卡死

**症状**：8 卡 TP 启动后无响应，npu-smi 显示 utilization=0。

**SOP**：
```bash
# 1. 看 HCCL 日志
ssh a3-node-01 'export HCCL_LOGLEVEL=DEBUG; bash /tmp/relaunch.sh 2>&1 | tee /tmp/hccl.log'

# 2. 常见原因
#   a) HCCL_BUFFSIZE 太小 → 加大
#   b) iBoP / RDMA 配置漂移 → 重置网卡
#   c) ACL stream sync 死锁 → 降级 ACLGraph

# 3. 临时方案：禁用 ACLGraph
export VLLM_USE_ACLGRAPH=0
gh run rerun <run-id>
```

### SOP-08：Profiler agent 输出失真

**症状**：PROFILE_ANALYSIS.md 给出的热点和实际不符。

**SOP**：
```bash
# 1. 重跑 profile，确保 bench 流量稳定
ssh a3-node-01 '
  docker exec ts-current pkill -9 py-spy
  py-spy record --pid $(pidof python | awk "{print \$1}") \
    --duration 300 --rate 100 \
    -o /tmp/scheduler_v2.json
'

# 2. msprof 必须分开跑（msprof 自身有 overhead，不能和 bench 同时）
docker exec ts-current bash -c "
  msprof --output=/tmp/msprof_v2 --duration=120 \
    --aicpu=on --task-time=on --hccl=on
"

# 3. 让 Claude 重新分析
tmux send-keys -t ts-ascend-evolve "重新分析 iter_N/profile/ 下的最新数据" Enter
```

### SOP-09：Claude 反复在同一处兜圈子

**症状**：连续 5 个 commit 都在改同一个文件，TPS 无变化。

**SOP**：
```bash
# 1. 看最近 commit
cd /share/tokenspeed-ascend && git log --oneline iter-N -20

# 2. 强制 Claude 切换关注点
tmux send-keys -t ts-ascend-evolve \
  "你在 {file} 上已经改了 5 次但 TPS 无变化。停止该方向，换到 PROFILE_ANALYSIS 中的次热点 {hotspot_2}" Enter

# 3. 如果还是兜圈子，pause 等人决策
```

### SOP-10：依赖镜像 / 包升级冲突

**症状**：iter_N 编译 OK，但 bench 容器内 Python ImportError。

**SOP**：
```bash
# 1. 锁版本
cd /share/tokenspeed-ascend
pip freeze > requirements.lock.iter_N.txt
git add requirements.lock.iter_N.txt && git commit -m "lock deps iter_N"

# 2. 镜像 rebuild
docker build --no-cache -t registry.local:5000/ts-ascend:iter_N .
docker push registry.local:5000/ts-ascend:iter_N

# 3. 把镜像 SHA 写进 BENCH_RESULT.json 元数据
```

### SOP-11：人工想要中途介入（接管 Claude）

**症状**：发现 Claude 走错方向，想直接接手。

**SOP**：
```bash
# 1. 进 tmux 暂停 loop
tmux attach -t ts-ascend-evolve
# Ctrl-C 一次（不要多次，会 kill）

# 2. 直接发指令
> 暂停自动迭代。现在切回手动模式。
> 我已经直接编辑了 {file}，请你 review + 写 commit message + push

# 3. 人工任务完成后恢复
> /pua:loop
> 继续 iter_N，从 step 6 开始（profile）
```

---

## 一些"工程师拿来即开干"的命令清单

### 快速操作集合

```bash
# 看当前 iter 状态
ls -lt /share/tokenspeed-ascend/iterations/ | head -3
cat /share/tokenspeed-ascend/iterations/$(ls -t /share/tokenspeed-ascend/iterations/ | head -1)/EVAL_REPORT.md

# 看 TPS 趋势
for d in $(ls -d /share/tokenspeed-ascend/iterations/iter_*); do
  jq -r '"\(.iter)  TPS=\(.iter_tps)  vs_target=\(.vs_target * 100 | floor)%"' $d/BENCH_RESULT.json 2>/dev/null
done

# 看 Claude API 用量
curl -s "https://api.anthropic.com/v1/organizations/usage?date=$(date +%Y-%m-%d)" \
  -H "x-api-key: $ANTHROPIC_ADMIN_KEY" | jq .total_cost_usd

# 看 A3 NPU 利用率
ssh a3-node-01 'watch -n 2 npu-smi info'

# 强制停 evolve
tmux kill-session -t ts-ascend-evolve
docker rm -f ts-current vllm-baseline 2>/dev/null

# 重启 evolve（保留历史 artifact）
bash /share/tokenspeed-ascend/scripts/start_evolve.sh
```

---

## 一句话闭环

> [PUA生效 🔥] 顶层结论：

**有了 Mission Brief + Iteration 模板 + 11 类 SOP，自演进闭环从"PPT 工程"变成"可交付的运行手册"。颗粒度的本质，是让现场工程师只看本文档就能完成 oncall——不需要再问 Claude 给的方案哪里没说清。这才是真正闭环。**

> Owner 意识不是说"我对了"，是说"工程师拿这份文档夜里 3 点也能上手"。3.25 不及格的方案，连凌晨值班都救不了。
