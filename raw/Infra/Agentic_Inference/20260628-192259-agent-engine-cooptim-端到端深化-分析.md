---
title: Agentic 应用与推理引擎协同优化 — 端到端深化分析（信息→优化→代码可行性）
tags:
  - inference
  - agentic
  - vllm-ascend
  - kvcache
  - scheduling
  - speculative-decoding
created: 2026-06-28
status: draft
hardware: Ascend 910B3 (Atlas 800T A2)
related:
  - "[[agent-engine-cooptimization]]"
---

# Agentic 应用与推理引擎协同优化 — 端到端深化分析

> [!abstract] 本文定位
> 承接 [[agent-engine-cooptimization]]（v1 方案，提出四层架构与信息全集）。本文把重心收敛到三件事，并对每一条做**严谨推演 + vllm-ascend 实际代码可行性核对**：
> 1. **能利用哪些 agent 信息/状态** —— 从 Claude Code 真实可得的轨迹出发，穷尽分类、判断价值；
> 2. **引擎拿到这些信息后能启动哪些优化措施 / 系统协同手段** —— 信息→优化的逐条映射推演；
> 3. **在 vllm-ascend 现有代码里到底改哪里、可行性如何** —— 每条结论锚定到实测源码行，说不清的标注疑问，不强行下结论。
>
> 硬件锚点：**Ascend 910B3 / Atlas 800T A2**（单卡 64GB HBM2e，单机 8 卡）。910B3 不是随意选的——vllm-ascend 的动态批调度器源码注释明确写着「目前 dynamic batch 仅支持 910B3 NPU」（见 §3 引用）。

> [!warning] 证据等级约定
> 本文每个「改哪里」结论标注证据等级：
> - **【坐实】** = 已在 vllm-ascend 真实源码定位到具体类/函数/行，改动点明确；
> - **【框架可承载，待设计】** = 现有框架有对应扩展位，但需要新增逻辑，未在源码看到现成实现；
> - **【疑问/待验证】** = 推演合理但代码层面尚未核实，或存在已知冲突风险，明确标注。
>
> 行号基于本地仓快照（2026-06-28），落 PR 前需重新 grep 校验。

---

## 1. 总体方案：协同优化的底层范式

传统推理引擎（含标准 vLLM）对请求是**无状态、彼此独立、语义盲**的：它只看到 token 序列，看不到序列背后的 agent 执行逻辑——这一轮之后还有几轮、某个 session 即将阻塞几十秒等工具返回、哪几个请求其实属于同一个 multi-agent 任务、哪段历史是一次性工具输出可以丢。

协同优化的范式是：**把 agent「知道、但引擎推断不出」的轨迹状态，作为带外信号下沉到引擎，让引擎从「被动响应请求」升级为「理解任务轨迹、预判式编排资源」。**

判断一条信息「值不值得下沉」的准绳（贯穿全文）：

> **引擎能否自己低成本推断出来？** 能（如 token 局部性、当前 batch 组成）——不必下沉；不能、且只有 agent 应用知道（如未来阻塞时长、剩余轮次、请求间 DAG 依赖）——这才是协同优化的真正价值区。

这条准绳直接砍掉了一批看似有用实则冗余的信号，把注意力集中在引擎的**真盲区**上。

---

## 2. 能利用哪些 Agent 信息与状态（穷尽 + 价值判断）

从 **Claude Code / agentic harness 真实可得**的轨迹出发，而非凭空设计字段。一个 code/search agent 的执行轨迹天然暴露这些节点：轮次边界、工具调用发起与返回、规划（plan）产物、子 agent 派生、上下文拼装方式。按引擎的三类盲区归类：

### 2.1 时间维（未来会发生什么 —— 引擎最大的盲区）

| Agent 信息                          | 采集节点                            | 引擎当前盲在哪                        | 价值量级                       |
| --------------------------------- | ------------------------------- | ------------------------------ | -------------------------- |
| **工具调用阻塞时长预告** `expected_idle_ms` | 工具调用发起时（agent 知道要调外部 API/等命令执行） | 引擎不知道某 seq 即将空闲数十秒，KV slot 被白占 | **高**：agentic 集群有效并发低的头号原因 |
| **剩余轮次预算** `remaining_turns`      | 轮次开始 / 规划后（max_iterations 进度可估） | 不知道哪些 session 快结束、复用价值低        | 中                          |
| **输出长度预期** `osl`                  | 轮次开始（agent 对本轮产物长度有先验）          | 不知道要生成多长，batch 组织与显存预留只能保守     | 中                          |

### 2.2 结构维（这段 token 流的语义是什么）

| Agent 信息 | 采集节点 | 引擎当前盲在哪 | 价值量级 |
|---|---|---|---|
| **输出段类型序列** `output_segments`（thinking / tool_call / answer 占比） | 轮次开始（agent 知道本轮大致会先想后调工具再答） | 对所有输出段一视同仁，投机长度无法按段切换 | 中 |
| **历史段语义角色**（哪些是一次性工具返回、可压缩/可丢） | 上下文拼装时（agent 清楚哪段是 tool result） | 只能逐 token LRU，无法按语义段压缩/逐出 | 中（长上下文下变高） |
| **生成格式约束**（JSON schema / 工具参数结构） | 工具调用前（agent 已知输出 schema） | 被动接收约束解码，未与投机协同 | 中 |

### 2.3 关系维（请求之间的关系 —— GPU 阵营也没做透）

| Agent 信息 | 采集节点 | 引擎当前盲在哪 | 价值量级 |
|---|---|---|---|
| **共享前缀显式声明**（system prompt / 工具定义 / few-shot） | 会话建立时 | Prefix Cache 被动发现，高复用前缀可能被误逐出 | 高（实现简单、收益直接） |
| **会话边界** `session_id` + `session_final` | 会话开始/结束 | 不知道 KV 何时可彻底释放 | 中 |
| **请求优先级** `priority` | 任务派发时 | 默认 FCFS / priority+arrival 二元组 | 中（已部分支持） |
| **Multi-agent DAG 拓扑**（请求间依赖 / join 点） | 规划后 / 子 agent 派生时 | 视每个请求独立，join 前木桶效应 | **高（差异化空间最大）** |
| **请求幂等/可中断语义** | 任务派发时 | 不敢激进抢占 | 低-中 |

> [!note] 价值判断小结
> 真正高杠杆且引擎无法自推的是三条：**① 工具阻塞预告（时间维）、② 共享前缀显式声明（关系维，最易落地）、③ multi-agent DAG 拓扑（关系维，最难但护城河最深）**。其余作为配套。下一章逐条做信息→优化→代码可行性映射。

---

## 3. 信息 → 优化措施 → 代码可行性（核心推演）

> vllm-ascend 是 **upstream vLLM 的硬件插件**（见其 AGENTS.md「Model and Plugin Architecture」）。它通过 **继承**（如 `RecomputeScheduler(Scheduler)`）和 **patch**（`vllm_ascend/patch/`）扩展上游。因此判断可行性时关键看：**改动点落在 vllm-ascend 自有的子类里（干净）还是必须 patch 上游（需架构评审）。** 本章每条都给出这个定位。

### 3.0 共同前提：信号如何进引擎

agent 信号经 L1 协议层归一化后，需进入引擎的 `Request` 元数据。vllm-ascend 沿用上游 `Request`，自定义字段的承载方式有二：

- 复用 Dynamo 已有通道 `nvext.agent_hints`（参考 v1 文档 §0，Dynamo 侧 `lib/llm/src/preprocessor/speculative_prefill.rs`、`kv_router.rs` 已坐实存在 `agent_hints` / `speculative_prefill` / `strict_priority`）；
- 或在 vllm-ascend 侧通过 `SamplingParams.extra_args` / request metadata 透传。

**【框架可承载，待设计】** 字段透传本身不难，难在让调度器/KV 管理器读到它——下面每条会指出具体读取点。

---

### 3.1 工具阻塞预告 → KV 主动换出（HBM→Host）

**信息**：`expected_idle_ms`（本 seq 即将阻塞多久）。
**优化措施**：阻塞期把该 seq 的 KV block 从 HBM 降级到 host DRAM，释放 HBM 给在跑请求；阻塞快结束前提前换回。
**为什么 910B3 上值得做**：单卡仅 64GB HBM，KV slot 是硬约束；agent 工具阻塞动辄数秒~数十秒，被阻塞 seq 白占 slot 直接压低有效并发。

**代码可行性 —— 【坐实】，且基础设施已存在：**

1. **底层 swap 算子已就绪**。[cpu_npu.py:54](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/kv_offload/cpu_npu.py) `CpuNpuOffloadingHandler` 维护独立 D2H / H2D stream（[:67](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/kv_offload/cpu_npu.py)），`transfer_async`（[:142](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/kv_offload/cpu_npu.py)）最终落到 `torch.ops._C_ascend.swap_blocks_batch(...)`（[:217](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/kv_offload/cpu_npu.py)）做批量块搬运。**HBM↔Host 的异步换出换入能力是现成的。**

2. **调度器已有"抢占即换出"的 hook 锚点**。[recompute_scheduler.py:302](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/recompute_scheduler.py)：当 `allocate_slots` 失败需要腾 KV 时，调度器调用 connector 的 `update_state_before_preempt(...)`，返回 `offloaded` 决定是「降级到 host 保留」还是「直接 free 重算」。
   ```python
   # recompute_scheduler.py:303
   preempt_hook = getattr(self.connector, "update_state_before_preempt", None) ...
   offloaded = bool(preempt_hook(recomputed_req, recomputed_block_ids, recomputed_num_computed_tokens))
   ```
   **这正是阻塞换出要接的位置**：现状是「被动等到 HBM 不够才换出最低优先级」，协同优化要做的是把它升级为「主动按 `expected_idle_ms` 标记触发换出」。

3. **connector 框架完整**。[cpu_offload_connector.py:64](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py) `CPUOffloadingConnector(KVConnectorBase_V1)` 已实现 scheduler/worker 分离、`start_load_kv` / `save_kv_layer` / `get_finished`。换入换出的搬运通道齐备。

**改哪里（落地路径）：**
- 调度循环（`recompute_scheduler.py` 的 `schedule()`，主体 [:171](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/recompute_scheduler.py) 起）新增一条：在抢占触发前，**主动**扫描带 `idle` 标记的 running seq，优先选它们走 `update_state_before_preempt` 换出，而非等 HBM 耗尽被动抢占。
- 换入时机预测：在 `expected_idle_ms` 将尽时预取回 HBM，避免换入延迟暴露在关键路径——可在 connector worker 侧加一个基于 deadline 的预取队列。

**难点 / 疑问：**
- **与 torchair graph mode 的兼容性【疑问/待验证】**：动态 swap 改变了 KV block 的物理驻留，是否破坏图捕获（aclgraph/torchair 的静态地址假设）尚未在源码确认。这是最大风险点，需实测。
- `expected_idle_ms` 估不准时的兜底：必须有运行时校正（实际换回早于/晚于预测），不能盲信先验。

---

### 3.2 共享前缀显式声明 → Prefix Pin（不被误逐出）

**信息**：`pin_prefix: bool` 或 prefix hash 声明（system prompt / 工具定义 / few-shot 这类跨请求高复用前缀）。
**优化措施**：给声明的前缀打 pin 标记，逐出时跳过 pinned 节点，保障 Prefix Cache 命中率。
**价值**：agentic 负载里 system prompt + 工具定义往往占前缀几 K token 且每轮复用，被误逐出会反复重算 prefill。**实现简单、收益直接**，是近期最高杠杆项之一。

**代码可行性 —— 【坐实命中路径，pin 位需设计】：**

- 前缀命中发生在 [single_type_kv_cache_manager.py:195](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/single_type_kv_cache_manager.py) `find_longest_cache_hit(...)`：它逐个 `block_hash` 调 `block_pool.get_cached_block(block_hash, ...)`（[:224](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/single_type_kv_cache_manager.py)）沿命中链匹配。命中链本身不做逐出决策——**逐出发生在 BlockPool 层**（哪个 cached block 被回收）。
- 因此 pin 的真正插入点在 **BlockPool 的逐出选择逻辑**（上游 vLLM 的 `block_pool` / free block queue），不在 vllm-ascend 自有的这个 hit 函数里。

**改哪里：**
- BlockPool 维护 pinned block 集合，逐出（free queue 出队）时跳过 pinned。
- pin 配额管理：pin 过多会挤占动态 KV 空间，需上限（如 pin 总量 ≤ X% HBM KV 容量）。

**难点 / 定位结论：**
- **【疑问/待验证】pin 位大概率要 patch 上游 BlockPool**，而非纯 vllm-ascend 子类改动——因为 `find_longest_cache_hit` 只读命中、不管逐出。需进一步核对上游 `block_pool` 是否已有 pin/lock 原语（部分 vLLM 版本有 `BlockPool` 的引用计数，可复用为软 pin）。这一步是落地前必须确认的可行性关口。

---

### 3.3 剩余轮次 → 价值逐出（替代纯 priority+LRU）

**信息**：`remaining_turns`（该 session 还剩几轮）。
**优化措施**：KV 逐出/抢占评分从「纯优先级+到达时间」升级为「剩余复用价值」——快结束的 session 复用价值低，可优先逐出；还要多轮复用的留住。
**价值**：命中率提升，HBM 留给真正还要用的 session。改动小、收益稳。

**代码可行性 —— 【坐实，且是一行级改动】：**

两个 vllm-ascend 自有调度器都把抢占目标选择写成了同一个二元组 key：

```python
# recompute_scheduler.py:349-352
preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))

# scheduler_dynamic_batch.py:242-245   （同样的 key）
preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
```

抢占 key 锚点：[recompute_scheduler.py:351](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/recompute_scheduler.py) 与 [scheduler_dynamic_batch.py:244](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/scheduler_dynamic_batch.py)。

**这就是价值逐出的精确插入点。** 把 key 从 `(r.priority, r.arrival_time)` 改为带剩余价值的评分函数：

```python
key=lambda r: (r.priority, remaining_value(r), r.arrival_time)
# remaining_value(r) 由 r 上透传的 remaining_turns / expected_idle 等导出
```

**为什么干净**：这两个调度器都是 vllm-ascend **自有子类**（`RecomputeScheduler(Scheduler)` @ [recompute_scheduler.py:95](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/recompute_scheduler.py)；`SchedulerDynamicBatch(Scheduler)` @ [scheduler_dynamic_batch.py:122](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/scheduler_dynamic_batch.py)），改它们不动上游，符合 AGENTS.md 的插件规范，无需 patch。

**难点：**
- 评分函数权重调参，与现有 priority 逐出策略融合（priority 仍应是第一关键字，避免破坏 SLO 分层）。
- `remaining_turns` 是估计值，需运行时校正——同 3.1 的不盲信原则。
- **NPU 注意**：评分若涉及 device tensor 的 `.item()` 会触发同步（AGENTS.md 明确警告 `tensor.item()` 在 NPU 热路径的同步开销）。剩余价值计算应全程留在 CPU 标量，避免 D2H 同步。

---

### 3.4 输出段类型 → 动态投机长度（DSL）先验

**信息**：`output_segments: [{type, expected_len}]`（本轮大致 thinking→tool_call→answer 的段序）。
**优化措施**：投机解码 draft 长度按当前生成段类型切换——结构化/answer 段 accept rate 高用长 draft，thinking 段发散用短 draft。把投机加速从「事后运行时反馈」提前为「事前预判」。
**价值**：长结构化输出尤其有效；agent 的 tool_call/JSON 段近乎可预测，投机收益大。

**代码可行性 —— 【框架在，draft 长度调控点待定位】：**

- vllm-ascend 投机栈存在：[draft_proposer.py:8](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/draft_proposer.py) `AscendDraftModelProposer(DraftModelProposer, AscendSpecDecodeBaseProposer)`；MTP 路径 [deepseek_v4_mtp.py](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/models/deepseek_v4_mtp.py)、[speculator.py](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/worker/v2/spec_decode/eagle/speculator.py)；配置 patch [patch_speculative_config.py](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/patch/platform/patch_speculative_config.py)。
- 现状投机长度通常由 `num_speculative_tokens` 配置固定 or 由运行时 accept rate 反馈调整。**段类型先验要做的是：按当前段类型查先验表动态设置 draft 长度，叠加 accept rate 修正。**

**改哪里：**
- 在 proposer 提议步前，依据请求当前所处 `output_segment.type` 选择 draft 长度（短/长两档起步）。
- 段边界检测：何时从 thinking 切到 tool_call——可由 agent 在段切换时更新 hint，或由结构化解码状态机辅助。

**难点 / 疑问：**
- **【疑问/待验证】** 当前 proposer 的 draft 长度是否每步可变、还是 batch 级固定，需进一步读 `draft_proposer` / `speculator` 实现确认。若是 batch 级固定，则按段切档需要更深的调度配合，可行性下降。
- 与现有 DSL 阈值机制（若已有）的融合点未在本次走读中定位，标为待查。

---

### 3.5 历史段语义角色 → 段级 KV 语义压缩/逐出

**信息**：历史段角色标注（哪段是一次性 tool result、可压缩或可丢）。
**优化措施**：按语义段而非 token 粒度做 KV 压缩/逐出——一次性工具返回的历史段整体压缩或降级。
**价值**：超长上下文 KV 占用大降。但复杂度高。

**代码可行性 —— 【与现有压缩机制耦合，需评估】：**

- vllm-ascend 的 KV 路径**已经是压缩感知的**：[single_type_kv_cache_manager.py:217](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/single_type_kv_cache_manager.py) 出现 `compress_ratio`，[:251](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/single_type_kv_cache_manager.py) 注释明确处理 DSv4/DSA 的 `MLAAttentionSpec` + `compress_ratio>1`（压缩态 MLA，与 [[vllm-ascend-pd-transfers-compressed-kv]] 一致）。
- 这意味着段级压缩**不是在空地上建**，而是要与既有的压缩态 KV 机制协同——好处是底层有压缩原语，坏处是语义段边界与物理 block/compress 粒度未必对齐。

**改哪里 / 难点：**
- 需要把「语义段」映射到 block 区间，在逐出/压缩决策时按段聚合——改动深入 KV cache manager。
- **【疑问/待验证】** 语义段边界与 `block_size` / `compress_ratio` 粒度对齐是核心难点，可行性中等偏低，建议作为中长期项，先不投入。

---

### 3.6 Multi-agent DAG 拓扑 → 组完成调度（系统级协同，护城河）

**信息**：`dag_edges`（同一 multi-agent 任务内请求的依赖 / join 点）。
**优化措施**：以「任务」（而非「请求」）为调度单元——同一 DAG 的多请求识别为一组做**组完成调度**，避免 join 前木桶效应（早算完的 worker KV 空占着等慢的）；不同任务间按 priority+剩余价值仲裁。
**价值**：解决 multi-agent 同步等待的 KV 空占和关键路径拖尾。**Dynamo 当前完全没碰，差异化想象空间最大。**

**代码可行性 —— 【框架可承载，但需新增任务级抽象，最重】：**

- 现有两个调度器都是**请求级**循环：[recompute_scheduler.py:171](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/recompute_scheduler.py) `schedule()` 遍历 `self.running` 逐请求 `allocate_slots`；[scheduler_dynamic_batch.py:195](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/scheduler_dynamic_batch.py) 同构。**没有"任务/组"这一层抽象。**
- 组完成调度需要：① 把带相同 task_id + DAG 关系的请求聚合成组；② 抢占/逐出仲裁以组为单位（不让一个组的前置请求被逐出而拖垮整组）；③ join 点的 KV 生命周期协调（前置完成后其 KV 何时可释放，取决于后继是否还需要）。

**改哪里：**
- 在调度器外包一层「任务编排器」（L2 决策层职责），把 DAG 信号翻译成对底层调度器的分组约束与抢占保护，而非直接重写两个调度器的核心循环。
- 这样底层 `RecomputeScheduler` / `SchedulerDynamicBatch` 改动可控（接受一个"组保护集合"），符合插件架构。

**难点 / 定位结论：**
- **【框架可承载，待设计】**：现有调度器有抢占目标选择的可注入点（同 3.3 的 key），可用于实现"组保护"（保护集合内请求不被选为抢占目标）。但「组完成」的完整语义（join 同步、组级 SLO）是新增子系统，源码中无现成实现。
- 复杂度最高，建议作为中长期差异化主攻方向，但要先用 3.1~3.3 的近期项验证信号通路。

---

### 3.7 会话边界 → KV 及时释放（配套，低成本）

**信息**：`session_final`（会话结束）。
**优化措施**：会话结束主动释放 KV，及时回收减少碎片。
**代码可行性 —— 【坐实通路】**：会话结束即请求生命周期结束，KV 释放走现有 `kv_cache_manager.free(...)`（[recompute_scheduler.py:332](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/recompute_scheduler.py)、[scheduler_dynamic_batch.py:252](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/scheduler_dynamic_batch.py) 均有调用）。难点不在释放本身，而在**判定"会话真结束"**——Dynamo 已定义 `X-Dynamo-Session-Final` header 但 SGLang 当前版本未处理（参考 v1 文档 §0.2）。属低成本配套项。

---

## 4. 系统级协同手段（把散点优化升维）

逐点优化之上，三个系统级抓手把它们串成「任务级系统协同」：

1. **任务级调度单元**（承 3.6）：引入 task 抽象，调度/逐出/抢占从请求级升到任务级。这是与传统 vLLM 最根本的范式差异——传统引擎天花板是「单请求算得多快」，协同引擎打开「整个任务轨迹编排得多高效」的空间。

2. **多 agent 负载聚合自适应**：L2 决策层聚合所有在跑 agent 的三维信号，算集群混合负载画像，动态调 PD 配比 / 并行 / batch 策略，而非为每种应用硬编码。Code Agent（输出长、结构化多）与 Search Agent（轮次多、阻塞频繁）归一化为统一信号后聚合。

3. **闭环反馈校正先验**：agent 先验必然有不准（`expected_idle_ms` 估偏、`osl` 估错、段类型边界漂移）。运行时指标（accept rate / KV 命中率 / 实际阻塞时长 / 队列深度）必须回流校正上层先验——**所有"信任先验"的优化都要有这条安全带**，否则先验一旦失真会反噬性能。这一点在 3.1/3.3/3.4 都被反复强调。

---

## 5. 代码可行性总表与落地排序

| # | 优化项 | 关键信息 | 改动点（实测锚点） | 落点类型 | 证据等级 | 杠杆 |
|---|---|---|---|---|---|---|
| 3.1 | 工具阻塞换出 | `expected_idle_ms` | [recompute_scheduler.py:302](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/recompute_scheduler.py) preempt hook + [cpu_npu.py:217](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/kv_offload/cpu_npu.py) swap 算子 | 自有调度器+connector | **坐实**（图模式兼容待验证） | 高 |
| 3.2 | 共享前缀 pin | `pin_prefix` | [single_type_kv_cache_manager.py:195](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/single_type_kv_cache_manager.py) 命中路径 → BlockPool 逐出层 | **可能需 patch 上游** | 坐实命中/pin位待定 | 高 |
| 3.3 | 剩余轮次价值逐出 | `remaining_turns` | [recompute_scheduler.py:351](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/recompute_scheduler.py) + [scheduler_dynamic_batch.py:244](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/scheduler_dynamic_batch.py) 的 `key=lambda` | 自有调度器（一行级） | **坐实** | 中高 |
| 3.4 | 段类型 DSL | `output_segments` | [draft_proposer.py:8](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/draft_proposer.py) proposer 调长 | 自有投机栈 | 框架在/调控点待定位 | 中 |
| 3.5 | 段级 KV 压缩 | 历史段角色 | [single_type_kv_cache_manager.py:217](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/single_type_kv_cache_manager.py) compress 路径 | 深入 KV manager | 待评估，粒度对齐难 | 中（长期） |
| 3.6 | DAG 组完成调度 | `dag_edges` | 调度器外包任务编排层 + 复用抢占 key 注入位 | 新增任务级子系统 | 框架可承载/待设计 | 高（护城河） |
| 3.7 | 会话结束释放 | `session_final` | [recompute_scheduler.py:332](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/recompute_scheduler.py) 的 `kv_cache_manager.free` | 通路坐实 | 坐实（判定逻辑待接） | 低（配套） |

**落地排序建议：**

- **近期（坐实、低-中复杂度、收益明确）**：3.3 价值逐出（一行级，先打通信号通路验证）→ 3.2 前缀 pin（先确认上游 BlockPool 是否有 pin 原语）→ 3.1 阻塞换出（基础设施齐，主攻图模式兼容）→ 3.7 会话释放（配套）。
- **中长期（差异化、高复杂度）**：3.6 DAG 组完成调度（最深护城河，需任务级抽象）→ 3.4 段类型 DSL（依赖 proposer 可变长确认）→ 3.5 段级压缩（粒度对齐难，最后做）。

> [!tip] 最大未开发空间
> 把 agent 的**时序信息（未来要等多久、还有几轮）和关系信息（请求间 DAG 依赖）**真正下沉到引擎，让引擎"预判式"编排 KV 和算力。这恰好是 vllm-ascend 已有调度器/connector/swap 基础设施能承载、而 Dynamo+SGLang 阵营还没做透的差异化地带。

---

## 6. 与 910B3 硬件约束的关联（为什么这些优化在 A2 上值得做）

| 约束 | 数值（来源见下，含置信度） | 对优化的含义 |
|---|---|---|
| 单卡 HBM 容量 | 64GB HBM2e（高置信） | KV slot 是硬约束 → 3.1/3.2/3.3 的 HBM 腾挪类优化收益直接放大 |
| 单机卡数 | 8 卡 + HCCS 互联（高置信） | multi-agent 任务常跨卡 → 3.6 DAG 协同有跨卡 KV 编排空间 |
| HBM 带宽 | 文献载 392 GB/s（**低置信，存疑，疑似口径混淆**） | decode 是 HBM-bound → 3.2 前缀命中省的 HBM 读直接影响 TTFT；此数需官方 datasheet 复核后才能用于定量 |
| BF16 算力 | ~313 TFLOPS（中置信） | 投机解码（3.4）的 draft 成本核算需要它 |
| 调度器硬绑定 | [scheduler_dynamic_batch.py:161](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/scheduler_dynamic_batch.py) 注释「dynamic batch 仅支持 910B3」 | **910B3 锚点由源码背书**，非外部假设 |

> [!warning] 定量收益建模的诚实边界
> 用户原始需求里的「吞吐 x 倍 / e2e 时延降 xx%」**不应在此阶段给单一漂亮数字**。原因：(1) HBM 带宽这个决定 decode roofline 的关键数当前来源存疑；(2) 收益强依赖 agentic 负载画像参数（工具阻塞占比 β、前缀复用率 ρ、平均轮次 R），这些需实测；(3) 多个优化点收益非线性叠加。
> 正确做法是把收益表达为 **`收益 = f(β, ρ, R)` 的区间 + 三个待实测输入（HBM 带宽实测值、真实 accept rate、真实阻塞分布）**。这是后续工作项，本文不强行给数。

---

## 7. 待验证清单（落地前必须闭合）

1. **【3.1 最高风险】** 动态 KV swap 与 torchair / aclgraph 图捕获的兼容性——动态 block 驻留是否破坏静态地址假设。需在 910B3 实测。
2. **【3.2 可行性关口】** 上游 vLLM `BlockPool` 是否已有 pin/引用计数原语可复用为软 pin；若无，pin 需 patch 上游，评审成本上升。
3. **【3.4 可行性关口】** `AscendDraftModelProposer` 的 draft 长度是 batch 级固定还是每步可变——决定段类型调档是否可行。
4. **【6 定量基础】** 910B3 单卡 HBM 带宽官方 datasheet 值（392 GB/s 存疑）、真实 accept rate、真实工具阻塞分布——三者是收益建模的实测输入。
5. **【信号通路】** `nvext.agent_hints` / request metadata 能否端到端透传到 vllm-ascend 调度器与 KV manager，是所有优化的前置。

---

## 参考

- v1 方案：[[agent-engine-cooptimization]]
- vllm-ascend 源码（本地快照 2026-06-28，链接为 VS Code 跳转）：
  - [recompute_scheduler.py — preempt hook :302](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/recompute_scheduler.py) / [抢占 key :351](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/recompute_scheduler.py)
  - [scheduler_dynamic_batch.py — 910B3 注释 :161](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/scheduler_dynamic_batch.py) / [抢占 key :244](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/scheduler_dynamic_batch.py)
  - [single_type_kv_cache_manager.py — 命中 :195](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/single_type_kv_cache_manager.py) / [压缩 :217](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/core/single_type_kv_cache_manager.py)
  - [cpu_npu.py — swap 算子 :217](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/kv_offload/cpu_npu.py)
  - [cpu_offload_connector.py — connector :64](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py)
  - [draft_proposer.py — 投机 proposer :8](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/draft_proposer.py)
  - [AGENTS.md — 插件/patch 架构规范](file:///Users/linyi/code/Documents/code/vllm-ascend/AGENTS.md)
- 910B3 规格：[AI柠檬 昇腾 NPU 参数汇总](https://blog.ailemon.net/2025/05/24/huawei-ascend-npu-params-for-ai/)（二手，HBM 带宽数存疑，待官方 datasheet 复核）
