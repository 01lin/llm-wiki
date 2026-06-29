---
title: "Agentic Workload 系统性分析 — 总体方案设计"
tldr: "面向 Agentic 推理引擎深度优化的 workload 画像总体方案：负载分析方法（反向锚定优化措施→量化参数）、三维 Agentic 指标体系（时间/结构/关系维 + 引擎效率基线）、三层数据采集实现（OpenAI chat 可推断 / 引擎埋点自采 / agent 协议上报），把 β/ρ/R 从假设变为分布"
date_created: 2026-06-29
date_modified: 2026-06-29
type: synthesis
tags: [agentic-inference, workload-analysis, metrics, observability, vllm-ascend, scheduling]
sources: ["[[20260628-150000-agent-engine-cooptimization-v1-方案-来源]]", "[[20260628-150100-agent-engine-cooptim-深化分析-来源]]"]
explored: false
confidence: high
---

# Agentic Workload 系统性分析 — 总体方案设计

> 本页是 [[20260628-150300-agentic-inference-engine-cooptimization-概念]] 的前置工程：协同优化每条措施的收益都是 `收益 = f(β, ρ, R, ...)`，而这些参数当前全是「未实测」（见该页 Data gaps）。workload 分析的唯一目的，是把它们从假设变成可观测分布，让方案从「拍脑袋估收益」升级为「数据驱动排优先级」。
>
> 约束锚点：入口是 **OpenAI `/v1/chat/completions` 请求结构**；传统 vLLM 的 OpenAI 响应里**不带**协同优化所需的轨迹/关系/未来信号。本方案的核心难点即：用「已能拿到的数据」逆向推断画像，并精确切分「推不出、必须新增采集」的部分。

---

## 0. 目标与产出物定义

**目标**：实现真正的「Agentic 推理引擎」深度优化——前提是先看清负载长什么样。

**三个产出物**（本方案分别对应三章）：
1. **负载总体分析方法**（§1）：分析框架 + 方法论，回答「分析什么、怎么分析」
2. **Agentic 指标体系**（§2）：一套分层指标定义，回答「量化哪些维度」
3. **数据采集实现方案**（§3）：可落地的三层采集设计，回答「数据从哪来、缺口怎么补」

**北极星判断**：分析做完，要能回答这一句——「这套真实负载下，7 条优化措施按收益从高到低应该怎么排，每条预期收益区间是多少」。

---

## 1. 负载总体分析方法与设计

### 1.1 核心方法论：反向锚定，不做无目的统计

不漫无目的地统计所有字段，而是**从 7 条优化措施反推每条所需的量化参数**，只采能驱动决策的指标。

| 优化措施（落点见协同优化概念页）| 决定收益的画像参数 | 不量化的后果 |
|---|---|---|
| 工具阻塞换出 | 工具阻塞占比 β、阻塞时长分布、阻塞期 KV 占用 | 不知道换出腾多少 HBM，可能换出开销 > 收益 |
| 共享前缀 pin | 前缀复用率 ρ、system+tools token 占比、跨请求命中率 | 不知道 pin 把命中率从多少提到多少 |
| 剩余轮次价值逐出 | 平均轮次 R、轮次分布、session 存活时长 | 不知「快结束 session」占比，逐出退化 LRU |
| 段类型 DSL | thinking/tool_call/answer 段占比与各段长度、可预测性 | 不知结构化段占比，DSL 先验收益面多大 |
| DAG 组完成调度 | multi-agent 任务占比、fan-out 度、join 等待时长 | 不知有没有 multi-agent 负载，护城河可能落空 |
| 段级 KV 压缩 | 一次性工具返回段的 token 体量与占比 | 不知可压缩历史段有多大 |
| 会话结束释放 | session 边界可识别率、结束后 KV 滞留时长 | 不知碎片回收潜力 |

> 底层逻辑：这是把方案里「信任先验的优化」全部挂上「用数据校准」的安全带——与协同优化的闭环反馈同源。

### 1.2 分析对象的三个粒度

| 粒度 | 单位 | 回答的问题 | 对应优化 |
|---|---|---|---|
| **请求级** | 单个 chat completion | ISL/OSL、是否带 tools、finish_reason | DSL、batch 组织 |
| **会话级** | 同 session 多轮 | 轮次 R、前缀复用、阻塞间隔 β | 价值逐出、前缀 pin、换出 |
| **任务级** | multi-agent DAG | fan-out、join 等待、依赖拓扑 | DAG 组完成调度 |

传统引擎只看请求级；协同优化要求把分析升到会话级与任务级——**这正是「从请求级升维到任务级系统协同」在分析侧的对应**。

### 1.3 五步分析流程（闭环）

```
①采集 → ②画像计算（5 张分布）→ ③负载聚类（验证负载差异假设）
   → ④收益重估（代入 f(β,ρ,R)）→ ⑤排序+缺口诉求
                ↑                                  │
                └──────── 运行时反馈校正先验 ────────┘
```

### 1.4 五张必产分布图（画像计算的交付物）

1. **轮次分布 R**：按 session 聚合请求数 → 单轮 / 5+ 轮占比（决定价值逐出收益面）
2. **前缀复用率 ρ**：system+tools token 占 prompt 比例分布 + 跨请求前缀 hash 命中率（决定 pin 收益）
3. **阻塞代理分布 β**：同 session 相邻请求间隔 P50/P90/P99（决定换出收益 + `expected_idle_ms` 经验表初值）
4. **段结构占比**：带 tools 请求 / finish_reason=tool_calls 占比 + 各段长度（决定 DSL、约束解码收益面）
5. **负载画像聚类**：按 (R, ρ, β, osl) 聚类，验证是否天然分出 Code-Agent（长 osl、结构化多）vs Search-Agent（多轮、阻塞频繁）——直接检验 v1 方案 §3.1 的负载差异假设

> 关键纪律：**让数据否决方案，而不是让方案预设数据。** 若聚类发现全是单 agent 多轮、无 multi-agent DAG，则「DAG 组完成调度」护城河应降优先级。

---

## 2. 总体 Agentic 指标体系设计

指标分四组：三组对齐协同优化的三维信号（时间/结构/关系），第四组是引擎效率基线（优化前后的对照基准）。每个指标标注**采集层级**（详见 §3）：`[推]`=OpenAI 可推断 / `[埋]`=引擎埋点自采 / `[报]`=需 agent 上报。

### 2.1 时间维指标（未来量 — 引擎最大盲区）

| 指标 | 定义 | 采集层 |
|---|---|---|
| `inter_request_gap_ms` | 同 session 上一响应结束 → 下一请求到达间隔（β 的代理量）| `[推]` 网关时间戳 |
| `tool_blocking_ratio` β | 会话总时长中工具阻塞占比 | `[推]` 间隔推断 / `[报]` 精确值 |
| `expected_idle_ms` | 本轮工具调用预期阻塞时长（**未来量**）| `[报]` 必须上报 |
| `remaining_turns` | session 剩余轮次（max_iterations 进度）| `[报]` 必须上报 |
| `osl_actual` / `osl_predicted` | 实测 / 预告输出长度 | `[推]` usage / `[报]` 预告 |

### 2.2 结构维指标（这段 token 流是什么）

| 指标 | 定义 | 采集层 |
|---|---|---|
| `has_tools` / `tools_token_count` | 是否 agentic + 工具定义 token 量 | `[推]` 请求体 |
| `segment_type_ratio` | thinking/tool_call/answer 段 token 占比 | `[埋]` detokenize 打标 |
| `segment_length_dist` | 各段长度分布 | `[埋]` segment marker |
| `accept_rate_by_segment` | 投机解码各段接受率 | `[埋]` proposer |
| `one_shot_history_ratio` | 一次性工具返回历史段占上下文比例 | `[报]` 段角色标注 |

### 2.3 关系维指标（请求之间的关系 — 差异化最大）

| 指标 | 定义 | 采集层 |
|---|---|---|
| `session_key` / `session_final` | 会话归属与边界 | `[推]` messages 前缀 hash / `[报]` 精确 |
| `prefix_reuse_rate` ρ | 跨请求共享前缀命中率 | `[推]` hash 比对 / `[埋]` 实际命中 |
| `task_id` / `parent_request_id` | multi-agent 任务归属与派生关系 | `[报]` 必须上报 |
| `dag_fanout` / `join_wait_ms` | 任务并发度 / join 等待时长 | `[报]` 必须上报 |
| `priority` | 请求优先级 | `[报]` 已有（Dynamo）|

### 2.4 引擎效率基线指标（优化前后对照）

| 指标 | 定义 | 采集层 |
|---|---|---|
| `prefix_cache_hit_rate` | KV prefix 实际命中率 / 命中 token 数 | `[埋]` find_longest_cache_hit |
| `preempt_count` / `recompute_tokens` | 每请求被抢占次数 + 重算 token | `[埋]` 调度器 preempt 路径 |
| `hbm_kv_utilization` | decode 阶段 HBM KV 实时占用 / slot 利用率 | `[埋]` block pool |
| `ttft` / `tpot` | 首 token / 每 token 时延 | `[推]` SSE chunk 时间戳 / `[埋]` |
| `effective_concurrency` | 实际有效并发（扣除阻塞空占）| `[埋]` 调度器 |

> 引擎效率基线是收益验证的对照锚：没有 baseline，「命中率从 <30% 提到 >60%」这种目标就无法证伪。

---

## 3. 数据采集实现方案设计

三层采集，按「改动成本 / 依赖方」从低到高推进，可解耦并行。

### 3.1 第一层 — OpenAI chat 可推断（零引擎改动，零 agent 配合）

**实现位置**：API 网关 / 反向代理层落结构化访问日志。**这是性价比最高的抓手**——先把 β/ρ/R 三大核心参数的真实分布拿到。

每条访问日志记录：
```
ts, session_key=hash(messages[:-1]), request_id,
isl, osl, system_tokens, tools_tokens, has_tools,
finish_reason, n_tool_messages,        # role=tool 的 message 数 ≈ 已发生轮次
ttft_ms(SSE 首 chunk), last_chunk_ms,
prev_request_gap_ms                     # 同 session 上一响应结束→本请求到达
```

**能推断的画像**（OpenAI 结构白送的）：

| 原始字段 | 推断的参数 | 方法 |
|---|---|---|
| `messages[]` role 序列 | 轮次结构 R、段类型雏形 | `role=tool` 数 ≈ 工具轮数；`tool_calls` 存在 = tool_call 段 |
| `messages[0]=system` + `tools[]` | 共享前缀、前缀复用 ρ | system+tools 做 hash 跨请求比对 |
| `finish_reason` | 段类型（部分）| `tool_calls`/`stop`/`length` |
| `usage.*_tokens` | ISL/OSL 分布、前缀占比 | system+tools 占 prompt 比 = 复用潜力上界 |
| 请求时间戳 / SSE chunk | β 代理量、TPOT、段切换点 | 相邻请求间隔；chunk 间隔突变 + `<think>` 标记切段 |

### 3.2 第二层 — 引擎埋点自采（改 vllm-ascend，不依赖 agent）

**实现位置**：vllm-ascend 内核埋点 → 复用现有 Prometheus/Grafana 通道（见 [[20260608-130000-vllm-ascend-mtp-prometheus-grafana-monitoring-sop]]）。

| 新增指标 | 引擎里哪有 | 落点 |
|---|---|---|
| prefix cache 命中率 / 命中 token | `find_longest_cache_hit` 已算，未透出 | `single_type_kv_cache_manager.py` |
| 抢占/换出次数 + 重算 token | 调度器 preempt 路径 | `recompute_scheduler.py:302` |
| token 级 segment marker | detokenize 按特殊 token 打标 | detokenizer |
| decode HBM KV 实时占用 / slot 利用率 | block pool | block pool |
| 投机 accept rate 按段统计 | spec decode proposer | `draft_proposer.py` |

> 价值:这些是「引擎自己知道、OpenAI 协议没往外吐」的——纯埋点,不依赖上游,**应最先做**;直接把 v2 深化分析的 3.2/3.4「待验证关口」变成可测量。

### 3.3 第三层 — agent / 协议层上报（必须上游配合，引擎无论如何推不出）

**实现位置**：HTTP header（对齐 [[20260629-100100-dynamo-实体]] 既有定义）+ 请求 metadata / `nvext.agent_hints`。

| 新增字段 | 载体 | 不报的代价 |
|---|---|---|
| `X-Session-ID` / `X-Session-Final` | HTTP header | 无法重建 session 边界，价值逐出/及时释放无从谈起 |
| `expected_idle_ms` | metadata / agent_hints | 换出只能被动等 HBM 耗尽，丢时间维最大收益 |
| `remaining_turns` | 同上 | 价值逐出退化 LRU |
| `task_id` + `parent_request_id` + `dag_edges` | 同上 | DAG 组完成调度完全无法落地 |
| `output_segments` 预告 | 轮次开始上报 | DSL 从「事前预判」退回「事后反馈」|

### 3.4 三层采集的推进策略

```
第一层（网关日志）─── 立即做，1 周出 β/ρ/R 分布 ──┐
                                                  ├─ 解耦并行
第二层（引擎埋点）─── 自己可控，闭合 3.2/3.4 关口 ─┘
第三层（协议上报）─── 拉通 agent/Dynamo 上游，早立诉求早排期
```

**关键判断:OpenAI chat 数据足够做「历史画像」(统计已发生的 β/ρ/R),但天然推不出「未来量」(expected_idle_ms)和「跨请求关系」(DAG)——这恰好等于协同优化的高价值信号区。不是巧合:正因引擎推不出,这些信号才值钱。**

---

## 4. 落地排序与诉求总结

| 优先级 | 动作 | 依赖 | 产出 |
|---|---|---|---|
| P0 | 第一层网关日志 + 五张分布图 | 无 | β/ρ/R 真实分布，消掉一半 Data gaps |
| P0 | 第二层引擎埋点（命中率/抢占/HBM 占用）| 改 vllm-ascend | 引擎效率基线，闭合待验证关口 |
| P1 | 负载聚类 + 收益重估 | P0 数据 | 7 条优化的收益排序 |
| P1 | 第三层 header 上报（session_id/final）| agent 配合 | session 边界，解锁价值逐出 |
| P2 | 第三层 metadata 上报（idle/turns/dag）| agent+Dynamo | 解锁换出、DAG 调度 |

**核心诉求清单（需新增采集，OpenAI 返回里没有的）**：
- **引擎侧自采**：prefix 命中率、抢占/重算、segment marker、HBM KV 占用、分段 accept rate
- **协议侧上报**：`X-Session-ID/Final`、`expected_idle_ms`、`remaining_turns`、`task_id/parent_request_id/dag_edges`、`output_segments`

---

## Counter-arguments

- **网关 hash 重建 session 不可靠**：客户端若对 messages 做改写（裁剪历史、压缩），前缀 hash 链会断，session 重建准确率下降——需用准确率指标自评，低于阈值则强依赖 `X-Session-ID`
- **β 用请求间隔做代理有偏**：间隔里混了「工具执行」+「用户思考」+「网络」，不能等同于纯工具阻塞；精确值仍需 agent 上报 `expected_idle_ms`
- **采集本身有开销**：token 级 segment marker、逐请求 KV 命中统计在高 QPS 下有埋点成本，需采样而非全量
- **画像漂移**：负载画像随业务变化，一次性分析会过时，需周期性重算（与运行时反馈闭环合并）

## Data gaps

- 真实生产 OpenAI chat 流量的可获取性（是否有权限落网关日志、隐私合规边界）
- segment marker 依赖模型输出特殊 token（如 `<think>`），不同模型标记不统一，跨模型段切分方法未定
- 第三层上报需要 agent 框架（Claude Code / 自研 harness）改造,上游配合意愿与排期未知
- 负载聚类的类别数与边界需真实数据驱动，当前 Code/Search 二分类是假设
