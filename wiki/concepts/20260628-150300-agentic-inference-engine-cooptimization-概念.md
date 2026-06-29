---
title: "Agentic 推理引擎协同优化"
tldr: "把 agent 执行轨迹信号（工具阻塞、DAG 依赖、剩余轮次、共享前缀）下沉到推理引擎，从无状态请求调度升维到任务级系统协同；核心范式是引擎从被动响应升级为预判式资源编排"
date_created: 2026-06-28
date_modified: 2026-06-28
type: concept
tags: [agentic-inference, scheduling, kvcache, inference-engine]
sources: ["[[20260628-150000-agent-engine-cooptimization-v1-方案-来源]]", "[[20260628-150100-agent-engine-cooptim-深化分析-来源]]"]
explored: false
confidence: high
---

# Agentic 推理引擎协同优化

## 核心问题

传统推理引擎（vLLM、SGLang）是**无状态、请求独立、语义盲**的——只看到 token 序列，看不到背后的 agent 执行逻辑：这一轮之后还有几轮、某个 session 即将阻塞数十秒等工具返回、哪几个请求属于同一个 multi-agent 任务、哪段历史是一次性工具输出可以丢。

判断一条信息「值不值得下沉」的准绳：**引擎能否自己低成本推断出来？** 能（如 token 局部性）则不必下沉；不能（如未来阻塞时长、剩余轮次、请求间 DAG 依赖）才是协同优化的真正价值区。

## 三维信号体系

**时间维**（引擎最大盲区）：
- `expected_idle_ms`：工具调用阻塞时长预告 → 主动换出 KV（高价值）
- `remaining_turns`：剩余轮次预算 → 按剩余复用价值做逐出决策（中价值）
- `osl`：输出长度预期 → batch 组织（中价值）

**结构维**：
- `output_segments`：输出段类型（thinking/tool_call/answer）→ [[dynamic-speculation-length-概念]] 段类型先验
- 历史段语义角色 → 段级 KV 压缩/逐出

**关系维**（差异化空间最大）：
- `pin_prefix`：共享前缀显式声明 → Prefix Cache pin，命中率 <30%→>60%（最易落地）
- `dag_edges`：multi-agent DAG 拓扑 → 组完成调度（最深护城河）
- `session_final`：会话边界 → KV 及时释放

## 与传统引擎的根本区别

| 维度 | 传统 vLLM | 协同优化引擎 |
|------|-----------|-------------|
| 调度单元 | 独立请求，FCFS | 任务（DAG 感知），优先级+组完成 |
| KV 逐出 | LRU（时间局部性）| 剩余复用价值（语义局部性）|
| KV 生命周期 | 被动随请求结束 | 主动预热/阻塞期换出/会话结束释放 |
| 投机长度 | 固定或纯运行时反馈 | 输出段类型先验 + 反馈修正 |
| Prefix 复用 | 被动发现，可能误逐出 | 显式 pin，保障命中率 |

## vllm-ascend 落地可行性（核心结论）

| 优化 | 证据等级 | 落点 |
|------|---------|------|
| 价值逐出（remaining_turns）| **坐实，一行级** | 自有调度器 key lambda |
| 阻塞换出（expected_idle_ms）| **坐实**（图模式待验证）| preempt hook + swap 算子 |
| 前缀 pin（pin_prefix）| 命中路径坐实，pin 位待设计 | 可能 patch 上游 BlockPool |
| DAG 组完成调度 | 框架可承载，最重 | 新增任务级子系统 |

## Dynamo 现状

[[20260629-100100-dynamo-实体]] 已下沉 `priority`（调度+逐出）和 `session ID`（KV 打标签），但所有高价值信号（时序/结构/关系维）均停在 router 层，未进引擎。`X-Dynamo-Session-Final` SGLang 当前版本未处理。

## Counter-arguments

- agent 先验信号质量依赖应用层配合，已有部署的 agent 不会主动上报
- 引入 agent→引擎耦合，换引擎时 L1/L2 层需全部重写
- 先验信号不准时的兜底逻辑本身是额外复杂度

## Data gaps

- 真实 agentic 工作负载的工具阻塞分布 β、前缀复用率 ρ、平均轮次 R 未实测 → workload 画像方案见 [[20260629-100000-agentic-workload-analysis-总体方案设计-综合]]
- 多个优化叠加的非线性交互未研究

## 相关方案

- [[20260629-100000-agentic-workload-analysis-总体方案设计-综合]] - 前置工程：用 OpenAI chat 数据画像 + 三层采集量化本页所需的 β/ρ/R
