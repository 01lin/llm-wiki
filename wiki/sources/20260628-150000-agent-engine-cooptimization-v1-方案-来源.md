---
title: "Agentic 推理引擎协同优化 v1 方案"
tldr: "把 agent 执行轨迹（工具阻塞、剩余轮次、DAG 依赖、共享前缀）作为带外信号下沉到 vllm-ascend，从无状态请求调度升维到任务级系统协同；四层架构 L0-L3，7 条优化信号归类为时间/结构/关系三维"
date_created: 2026-06-28
date_modified: 2026-06-28
type: source
tags: [agentic-inference, vllm-ascend, scheduling, kvcache, speculative-decoding, dynamo]
sources: []
explored: false
confidence: high
---

# Agentic 推理引擎协同优化 v1 方案

原始来源：`raw/Infra/Agentic_Inference/agent-engine-cooptimization.md`（本地方案文档）

## 核心命题

传统推理引擎（含标准 vLLM）是**无状态、请求独立、语义盲**的——只看到 token 序列，看不到背后的 agent 执行逻辑。协同优化的本质：把 agent「知道、但引擎推断不出」的轨迹状态，作为带外信号下沉到引擎，从「被动响应请求」升级为「理解任务轨迹、预判式编排资源」。

## Dynamo 现状与局限

Dynamo 已定义两类信号：
- `nvext.agent_hints`：`priority`（下沉到 SGLang 排队）、`strict_priority`（router only）、`osl`（router only，未下沉引擎）、`speculative_prefill`（预热触发）
- HTTP header：`X-Dynamo-Session-ID`（KV 打标签）、`X-Dynamo-Session-Final`（SGLang 当前版本**未处理**）

关键局限：只下沉了**标量先验**（priority、session ID），未下沉 agent 的时序/结构/关系状态。

## 三维信号体系

**时间维**（引擎最大盲区）：
- `expected_idle_ms` 工具阻塞预告 → KV 主动换出 HBM→Host（高价值）
- `remaining_turns` 剩余轮次预算 → 按剩余复用价值做逐出决策（中价值）
- `osl` 输出长度预期 → batch 组织与显存预留（中价值，Dynamo 已有但未下沉引擎）

**结构维**：
- `output_segments` 输出段类型序列（thinking/tool_call/answer）→ DSL 段类型先验切换投机长度
- 历史段语义角色（一次性工具返回）→ 段级 KV 压缩/逐出
- 生成格式约束（JSON schema）→ 约束解码 + 投机协同

**关系维**：
- `pin_prefix: bool` 共享前缀显式声明 → Prefix Cache pin（最易落地）
- `dag_edges` multi-agent DAG 拓扑 → 组完成调度（最高护城河）
- `session_final` 会话边界 → KV 及时释放

## 四层架构

```
L0  Agentic 应用层（Code/Search/MultiAgent/运维 Agent）
    ↓ 状态信号上报（执行轨迹关键节点）
L1  统一状态协议层（三维信号归一化采集）
    ↓ 语义信号
L2  协同决策层（任务级调度器 | KV 生命周期 | 弹性策略 | 投机调控）
    ↓ 下沉引擎参数/指令  ↑ 运行时反馈
L3  推理引擎执行层（vllm-ascend / SGLang）
    ↓
    昇腾算力底座（910B/C，HBM→DRAM→NVMe 分层）
```

设计原则：①泛化性（应用声明语义，不关心引擎怎么用）②解耦性（采集/决策/执行三层分离）③自适应（运行时反馈校正不盲信先验）

## 落地排序

近期（低-中复杂度）：工具阻塞换出 → 共享前缀 pin → 剩余轮次价值逐出 → DSL 段类型先验  
中长期（差异化护城河）：Multi-Agent DAG 组完成调度 → 段级 KV 语义压缩

## Counter-arguments

- agent 先验信号质量依赖应用层配合，大量已有部署的 agent 不会主动上报
- 协议层增加了 agent→引擎的耦合，违背无状态 HTTP 设计原则
- 工具阻塞预告若不准（估偏 2x+），换出换入本身带来额外 HBM 带宽压力，可能净负收益

## Data gaps

- 真实 agentic 工作负载中工具阻塞占比 β、前缀复用率 ρ、平均轮次 R 的实测分布未知
- 910B3 HBM 带宽实测值存疑（第三方称 392 GB/s，未有官方 datasheet 确认）
- 与 torchair/aclgraph 图捕获的兼容性未在源码层验证
