# vLLM-Ascend 动态投机长度：异步兼容 + verify 减 token 收益 + 零同步 深化分析

> 生成时间：2026-06-23
> 范围：针对三个关键点深化——① 异步调度兼容（不破坏遮掩、性能不劣化、工程可行）；② 动态投机的真实收益（减草稿前向 + 减 verify token）；③ 关键路径零新增 D2H/H2D 同步。
> 证据基线：`vllm` @ `0d2961229`、`vllm-ascend` @ `8afdf356`。源码引用为可点击 VS Code 链接。
> ⚠️ **本文修正前序方案的图模式判断**：发现「图桶按 `num_tokens=(1+k)×num_reqs` 乘积建、要求整除」的真相，得出比 padding 更优的「减 token 命中桶」方案——这是拿到第 2 点收益的关键。
> 关联：[[20260623-100003-vllm-ascend-动态投机长度-实现级设计-分析]]、[[20260623-003802-vllm-ascend-动态投机长度-可行性性能可靠性深度剖析-分析]]、[[20260623-000214-vllm-ascend-动态投机长度-第一阶段方案设计-分析]]

---

## 0. 三点结论速览

| 点 | 结论 | 关键依据 |
|----|------|---------|
| ① 异步兼容 | ✅ 不破坏遮掩，性能不劣化，工程可行 | 遮掩本质是「launch 后 return None、CPU 排下一步与 NPU 前向重叠」，与投机长度正交；k_active 是 CPU 标量不进等待链 |
| ② verify 减 token 收益 | ✅ **可拿到**（修正前序 padding 方案）：减小 k_active 真减 verify token，**前提是 `num_tokens=(1+k)×num_reqs` 命中已捕获图桶且整除** | 图桶按 `num_tokens` 乘积建（dispatcher.py:211），非按 query_len 固定 |
| ③ 零同步 | ✅ 关键路径无新增 D2H/H2D | k_active CPU 整数；PLACEHOLDER（如需）host 侧填；接受核算走现有 GPU 修正 kernel |

> **核心修正**：前序"padding 到 1+K_max"方案虽保证图命中，但 **verify token 不减 → 拿不到第 2 点收益**。本文给出「减 token 命中桶」方案：让 verify 真实按 `1+k_active` 跑、`num_tokens` 落在捕获桶上——**既命中图又减 token**。两方案取舍见 §2.4。

---

## 1. 第一点：异步调度兼容性（代码级）

### 1.1 异步遮掩遮的到底是什么

实测调用链（[model_runner_v1.py:2315](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py)~[2335](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py)）：

```
execute_model(scheduler_output):
    ... 前向 launch（异步下发到 NPU stream）...
    self.execute_model_state = ExecuteModelState(logits, spec_metadata, ...)  # 存状态
    if deferred_state_corrections_fn: deferred_state_corrections_fn()          # launch 后才纠偏
    return None    # ★ 不等结果，立即返回 → 调度器排下一步
```
注释（[:2331](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py)）："Now the batch has been launched we can wait for corrections **without breaking async scheduling**"。

**遮掩本质**：`execute_model` 把前向 launch 到 NPU 后立即返回 → EngineCore 进程的「下一步 CPU 调度 + prepare_input」与「本步 NPU 前向」在时间轴上重叠。遮掩的是 **CPU 侧调度/准备开销**，被 NPU 前向时间掩盖。

### 1.2 动态投机长度为何不破坏遮掩

逐项对照遮掩依赖的条件：
- 遮掩依赖「execute_model 不阻塞返回」——与本步排几个草稿（k_active）**无关**。k_active 只改 draft 循环次数（[llm_base_proposer.py:566](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py)）和 verify token 数，不引入任何等待。
- 跨步状态 `prev_num_draft_len`（[:1971](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py)）存的是**上一步实际草稿数**（非配置常量）→ 切换后下一步用新 k_active 排草稿、用旧值处理遗留，天然自洽，且已有 KeyError 兜底（[:696](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py)）。
- async 下 CPU 乐观假设「全接受」、GPU kernel 用上一步 `valid_sampled_token_count_gpu` 修正（[:1048](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py)）——这个修正逻辑**用 per-req 实际草稿数**，k_active 变化自动适配，不需要新同步。

### 1.3 性能不劣化与工程可行性判断

- **不劣化**：遮掩比 = `min(CPU_prepare, NPU_forward) / total`。k_active 同时影响分子分母（draft 步数、verify token），但**不改变 execute_model 的"launch 后立即返回"结构** → 遮掩比不退化。
- **工程可行**：切换只需在调度步边界 latch（§3 实现），不碰 `ExecuteModelState`/stream/event 体系。改动收敛、与 async 路径正交。
- **风险点**：DP/TP 多卡必须同步切换（同一调度步各卡 k_active 一致），否则 collective 通信形状错位 → 用 collective_rpc 广播保证（[[20260623-100003-vllm-ascend-动态投机长度-实现级设计-分析]] §3 已列）。

---

## 2. 第二点：动态投机的真实性能收益（含图模式约束的修正）

### 2.1 你要的两个收益（拆解到代码）

| 收益 | 机制 | 代码依据 |
|------|------|---------|
| 减草稿前向耗时 | k_active 减小 → draft 循环 `range(k_active)` 少跑几次草稿前向 | [llm_base_proposer.py:566](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py) |
| 减 verify 耗时 | verify 输入总 token = `(1+k_active)×并发` → k_active 减小 → 主模型处理 token 数减少 → 计算量减少 | verify 前向 query 长度 = `1+k_active` |

> 收益①永远能拿到（少跑循环，无约束）。**收益②受图模式约束**——这是前序方案的修正焦点。

### 2.2 图桶的真相（决定收益②能否拿到）

实测（[cudagraph_dispatcher.py:211](file:///Users/linyi/code/Documents/code/vllm/vllm/v1/cudagraph_dispatcher.py)~[230](file:///Users/linyi/code/Documents/code/vllm/vllm/v1/cudagraph_dispatcher.py)）：

```python
max_num_tokens = uniform_decode_query_len * max_num_seqs
cudagraph_capture_sizes_for_decode = [x for x in capture_sizes
                                      if uniform_decode_query_len <= x <= max_num_tokens]
for bs in cudagraph_capture_sizes_for_decode:
    add_cudagraph_key(FULL, _create_padded_batch_descriptor(bs, uniform_decode=True, ...))
```
而 `_create_padded_batch_descriptor`（[:107](file:///Users/linyi/code/Documents/code/vllm/vllm/v1/cudagraph_dispatcher.py)~[117](file:///Users/linyi/code/Documents/code/vllm/vllm/v1/cudagraph_dispatcher.py)）：
```python
num_reqs = num_tokens_padded // uniform_decode_query_len
assert num_tokens_padded % uniform_decode_query_len == 0   # ★ 整除约束
```

**关键事实（修正前序判断）**：
1. 图桶的 key 维度是 **`num_tokens`**（一个离散桶集合），不是"query_len 固定值"。
2. runtime 的 `num_tokens` 被 `_bs_to_padded_graph_size` **向上取整到最近桶**（[:141](file:///Users/linyi/code/Documents/code/vllm/vllm/v1/cudagraph_dispatcher.py)）。
3. 但 `num_reqs` 是用 `num_tokens_padded ÷ uniform_decode_query_len` 反推的——**这里的 `uniform_decode_query_len` 是初始化时固定的 `1+K_max`**。

### 2.3 两条技术路线（拿收益② vs 保证图命中）

| 路线 | verify query | num_tokens | 图命中？ | 收益②？ | 约束 |
|------|-------------|-----------|---------|--------|------|
| **A：padding 到 K_max**（前序方案） | 恒 `1+K_max` | `(1+K_max)×num_reqs` | ✅ 恒命中 | ❌ token 不减 | 无 |
| **B：减 token 命中桶**（本文新增） | 真实 `1+k_active` | `(1+k_active)×num_reqs` | ⚠️ 需该 num_tokens 落桶 + 整除 | ✅ token 真减 | num_reqs 反推依赖 `uniform_decode_query_len` |

**路线 B 的拦路虎（诚实标注）**：`_create_padded_batch_descriptor` 的 `num_reqs = num_tokens ÷ uniform_decode_query_len` 假设 query_len 恒为 `1+K_max`。若实际 query=`1+k_active`，则 `num_tokens=(1+k_active)×num_reqs` 除以 `1+K_max` 反推出的 num_reqs **错误** → FA3 scheduler_metadata 依赖 num_reqs（[:201](file:///Users/linyi/code/Documents/code/vllm/vllm/v1/cudagraph_dispatcher.py) 注释 "FULL mode needs exact num_reqs"）→ 可能算错或掉图。

### 2.4 推荐方案：分档 uniform_decode_query_len（路线 B 的可行实现）

要让路线 B 成立，需让 dispatcher **按多个 `uniform_decode_query_len` 值（1+1, 1+2, …, 1+K_max）分别注册 FULL decode 图**，运行时按 k_active 选对应 query_len 的图集。即：

- 启动期：对每个 `k ∈ {常用 k 值}`，以 `uniform_decode_query_len = 1+k` 跑一轮 dispatcher key 注册（[cudagraph_dispatcher.py:170](file:///Users/linyi/code/Documents/code/vllm/vllm/v1/cudagraph_dispatcher.py) `initialize_cudagraph_keys(uniform_decode_query_len=1+k)`）。
- runtime：[uniform_decode 判定](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py) 用 `1+k_active`，命中对应桶。

**取舍**：

| 维度 | A (padding) | B (分档) |
|------|-------------|----------|
| 收益②（减 verify token） | ❌ | ✅ |
| 图数量/显存 | 1 档 | K 档（每档若干 size） |
| 启动 capture 时间 | 1× | ~K× |
| dispatcher 改动 | 无 | 需支持多 query_len 注册（中等改动） |
| 命中稳定性 | 恒命中 | 命中（各档独立桶） |

> **判断**：用户第 2 点明确要"减 verify token"——**收益②是硬需求，必须走路线 B（分档）**。代价是启动期多捕获 K 档图（一次性显存+时间）。这与第一阶段"无劣化"不冲突（启动成本不进热路径）。若显存吃紧，K 档可只覆盖"常用 k 值集合"（如 {2,3,5,8}）而非全 1..K_max。

### 2.5 收益量化（修正后）

并发 C、投机长度 k、接受率对应平均接受长度 `a(k)`：
- verify 计算量 ∝ `(1+k)×C`（token 数）→ k 从 8 降到 3，verify token 减 `5×C/((1+8)×C)=55%`。
- draft 前向次数 = k → 从 8 降到 3，draft 耗时减 62%。
- 但加速比 = `a(k)/(1 + k×draft_ratio + verify_ratio)`——k 减小 token 省了，但 `a(k)` 也可能降（草稿少接受少）。**这正是第二阶段"寻优"要解的**：在 token 节省与接受长度间找最优 k。第一阶段只保证"设了某个 k 能真减 token、不劣化、不掉图"。

---

## 3. 第三点：关键路径零新增 D2H/H2D 同步（逐点核验）

### 3.1 完整推理关键路径的同步点盘查

| 路径环节 | 现有同步行为 | 动态投机引入新同步？ |
|---------|-------------|---------------------|
| 调度 latch k_active | CPU 整数读写（GIL 原子） | ❌ 无（纯 host） |
| draft 循环 range(k_active) | 读 CPU 整数 | ❌ 无 |
| 草稿 buffer 索引 [draft_step] | host pin_memory（[llm_base_proposer.py:208](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py)） | ❌ 无 |
| verify token 数 = (1+k)×C | 由 attn_metadata 决定，CPU 侧算 | ❌ 无 |
| 接受核算 | 现有 GPU 修正 kernel `update_num_computed_tokens_for_batch_change`（[:1051](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py)），`non_blocking=True` | ❌ 沿用，无新增 |
| valid_sampled_token_count 回传 | 独立 stream + non_blocking + event（[:1627](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py)） | ❌ 沿用，无新增 |
| 图选择 | dispatcher CPU 侧字典查 | ❌ 无（路线 B 多档也是 CPU 查表） |

### 3.2 零同步的实现保证（硬约束）

1. **k_active 全程 CPU 整数**：controller 的 update/latch/get 都是 host 标量操作，**绝不读设备张量**（杜绝 `.item()/.cpu()/synchronize()`）。
2. **图档选择在 CPU 完成**：路线 B 按 k_active 选 query_len 档，是 dispatcher 字典查（host），不触发设备操作。
3. **无 PLACEHOLDER 填充开销**：路线 B 不需要 padding（verify 真按 1+k_active 跑），连 host 侧填充都省了——比路线 A 更干净。
4. **接受核算复用现有 async 路径**：k_active 只改"本步草稿宽度"这个 CPU 标量，GPU 修正 kernel 逻辑不变。

> **结论**：路线 B（分档）不仅拿到收益②，且**比路线 A 更零同步**（无 PLACEHOLDER host 填充）。关键路径无任何新增 D2H/H2D/synchronize/高耗时步骤。

### 3.3 "几乎无高耗时步骤"核验

唯一新增的 host 操作：① latch（整数赋值）；② 按 k_active 选图档（字典查）；③ draft 循环上界变量读取。三者均 O(1) 纳秒级，远低于单 token 的 ms 级时延 → 可忽略。**无 GIL 长持有、无锁竞争、无设备同步**。

---

## 4. 方案修订结论

1. **第 2 点收益是硬需求 → 主方案改为路线 B（分档 uniform_decode_query_len）**，前序"padding 到 K_max"（路线 A）降为"显存吃紧时的退化备选"。
2. 路线 B 代价：启动期按 K 档（或常用 k 集合）多捕获图，一次性显存+时间，不进热路径。
3. 三点全部满足：① async 遮掩不破坏（与投机长度正交）；② verify 真减 token（路线 B）；③ 关键路径零新增同步（路线 B 比 A 更干净，无 PLACEHOLDER 填充）。
4. **需 Phase 0 实测确认**：dispatcher 多档 query_len 注册的工程改动量 + K 档图的显存占用是否可接受；以及 FA3/DSA 的 scheduler_metadata 是否正确支持多 query_len 档（[:201](file:///Users/linyi/code/Documents/code/vllm/vllm/v1/cudagraph_dispatcher.py) "FULL mode needs exact num_reqs" 的实际约束强度）。这是路线 B 当前最大不确定点。

### 待回写
- 实现级设计 [[20260623-100003-vllm-ascend-动态投机长度-实现级设计-分析]]：§2.3 runner 改动从"padding"改为"按 k_active 选图档 + verify 真实 1+k_active token"。
- 新增 dispatcher 多档注册的改动点。

---

## 5. 源码证据索引（可点击）

| 主题 | 位置 |
|------|------|
| async 遮掩：launch 后 return None | [model_runner_v1.py:2315](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py)~[2335](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py) |
| 图桶按 num_tokens 注册 + 整除约束 | [cudagraph_dispatcher.py:107](file:///Users/linyi/code/Documents/code/vllm/vllm/v1/cudagraph_dispatcher.py)~[117](file:///Users/linyi/code/Documents/code/vllm/vllm/v1/cudagraph_dispatcher.py) · [:211](file:///Users/linyi/code/Documents/code/vllm/vllm/v1/cudagraph_dispatcher.py)~[230](file:///Users/linyi/code/Documents/code/vllm/vllm/v1/cudagraph_dispatcher.py) |
| num_tokens 向上取整到桶 | [cudagraph_dispatcher.py:141](file:///Users/linyi/code/Documents/code/vllm/vllm/v1/cudagraph_dispatcher.py) |
| initialize_cudagraph_keys(uniform_decode_query_len) | [cudagraph_dispatcher.py:170](file:///Users/linyi/code/Documents/code/vllm/vllm/v1/cudagraph_dispatcher.py) |
| FULL needs exact num_reqs | [cudagraph_dispatcher.py:201](file:///Users/linyi/code/Documents/code/vllm/vllm/v1/cudagraph_dispatcher.py) |
| async GPU 修正 kernel（无 D2H） | [model_runner_v1.py:1048](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py)~[1064](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py) |
| valid_sampled_token_count 异步 stream | [model_runner_v1.py:1627](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py)~[1638](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py) |
| draft 主循环 | [llm_base_proposer.py:566](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py) |
| uniform_decode 判定 | [model_runner_v1.py:2867](file:///Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py) |
