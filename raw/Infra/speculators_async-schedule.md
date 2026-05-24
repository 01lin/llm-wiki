# 混合投机工程实现分析（面向 vLLM-Ascend `model_runner_v1`、MTP 图模式、Async Schedule）

## 1. 问题重述

要让混合投机真正达到预期加速效果，算法层面需要解决两个核心决策：

- 在不同并发、不同上下文长度下，投机总长度 `K_total` 应该取多少。
- 在给定 `K_total` 下，`MTP` 与 `Suffix` 应如何分配，即 `K_total = K_mtp + K_suffix`。

但工程实现上，真正的难点并不是“如何算出最优 `K_total/K_mtp/K_suffix`”，而是：

- 现有 `vllm-ascend / vllm` 的 `model_runner_v1` 执行路径，尤其是 `async schedule`、overlap 调度、graph capture、固定宽度输入输出 buffer、spec verify metadata 等，默认把“投机长度”当作**静态形状参数**而不是“动态运行时参数”。
- 一旦在运行中动态改变 `MTP` 投机长度或 `suffix` 投机长度，就会破坏当前调度、buffer、graph replay、scheduler reserve token 等一系列隐含契约。

因此，本问题本质上是：

**如何在“不破坏 async schedule 和图模式”的前提下，把动态投机策略降维成当前执行框架可以承受的固定形状运行模型。**

本文给出：

- 根因分析
- 可行性评估
- 推荐的软件工程实现方案
- 分阶段落地路径

---

## 2. 现有实现的关键约束

从当前代码栈看，投机长度 `spec_num_tokens` 已经深度嵌入运行时形状。

### 2.1 固定输出宽度约束

当前 `ModelExecutorConfig.output_length` 直接由 `server_args.speculative_num_draft_tokens` 决定，后续会传入：

- `RuntimeStates.future_input_map` 的第二维
- `InputBuffers.fill_input_buffers()` decode 路径读取宽度
- overlap schedule 下下一轮 decode 的输入组织

也就是说，当前系统默认：

- 每个 decode 请求在 speculative 模式下，下一轮输入宽度固定为 `spec_num_tokens`
- 不是“有效长度可变”，而是“槽位宽度固定”

### 2.2 Scheduler 固定 decode 输入宽度

当前 scheduler 初始化时：

- `decode_input_tokens = speculative_num_draft_tokens if speculative_algorithm else 1`

这意味着 C++ scheduler 在设计上也假定：

- speculative decode 每轮的 decode 输入 token 数是固定的
- reserve token 更新、request 状态推进、batch 打包宽度都围绕这个固定值设计

### 2.3 Attention metadata / CUDA graph 固定形状

当前 attention backend 和 CUDA graph capture 也把 `speculative_num_draft_tokens` 编进了 metadata 形状：

- `target_verify_metadata.cu_seqlens_q`
- `target_verify_metadata.page_table`
- `draft_extend_metadata`
- grammar bitmask 的 speculative 宽度
- sampling backend 的 `max_draft_tokens_per_req`

特别是在 graph capture 下：

- graph 不是只对 `bs` 敏感，也对 speculative token 的“静态上界宽度”敏感
- 一旦真实宽度变化，就容易出现 metadata 不匹配、page_table 不一致、buffer 切片越界、replay 语义失配

### 2.4 Overlap / Async Schedule 的时序耦合

overlap 调度的基本假设是：

- 上一轮结果通过 `future_input_map` 提前写好
- 本轮 GPU forward 发射后，CPU 同时处理上一轮结果
- scheduler 根据 `output_length` / `accept_length` / reserve token 状态推进请求

因此它依赖两个事实：

- 下一轮 decode 读取的槽位形状是稳定的
- 上一轮写入与下一轮读取的协议是稳定的

一旦某个请求本轮是 `mtp4`、下轮变成 `mtp2+suffix1`、再下轮变成 `suffix2`，如果直接让“真实宽度”跟着跳：

- `future_input_map` 的内容解释会变化
- `fill_input_buffers()` 的 flatten 逻辑会变化
- scheduler 对 reserve token 的理解会变化
- graph replay 所用 metadata 宽度也会变化

这就是当前“不兼容”的本质。

---

## 3. 根因总结：当前系统把投机长度当成了“形状”，而不是“策略”

可以把问题压缩成一句话：

### 当前实现中的 `spec_num_tokens` 同时扮演了四个角色

1. 算法语义：本轮最多投机多少 token
2. buffer 宽度：`future_input_map`、sampling buffer、grammar buffer 的列宽
3. graph 形状：verify/draft/capture metadata 的 `max_seq_len_q`
4. scheduler 协议：decode 输入宽度与 reserve token 更新规则

而混合投机真正需要的是：

- **算法上动态调整有效投机长度**
- 但**工程上保持运行时形状尽量稳定**

因此正确方向不是“让所有层都支持任意动态长度”，而是：

**把“动态有效长度”限制在固定容量 `K_cap` 内，通过 mask / valid length / source map 实现策略变化，而不改运行时主形状。**

但这里有一个必须补充的现实问题：

### 3.1 当前方案先解决了兼容性，还没有真正解决 MTP 成本

如果采用：

- 总投机长度固定：`K_total = 6`
- 动态调整：`K_total = K_suffix + K_mtp`
- 单链、串行、MTP graph mode 开启

那么 target verify 侧的开销本来就会近似固定：

- verify 一次仍按 `K_total=6` 的固定宽度执行

这一点很难通过 `suffix/mtp` 比例调整直接优化。

真正的问题在于 draft 侧：

- 现有 MTP graph 往往按 `mtp6` 的最大展开长度 capture
- 即使本轮只需要 `K_mtp=1/2/3`
- 也仍然可能执行整条 `mtp6` 图

于是会出现一个关键失效点：

- `suffix` 的比例提高了
- 但只是降低了**语义上的 MTP 占比**
- 没有降低**执行上的 MTP 图成本**

所以“静态总长度 + 动态调整 `mtp/suffix` 比例”如果只做到 `K_eff` 语义动态，还不足以兑现收益。

下一步必须进一步做到：

**固定 verify 总长度，但不要让 MTP 执行长度继续绑定 verify 总长度。**

---

## 4. 可行性评估

## 4.1 方案分级

### 方案 A：完全动态长度，运行时实时改 graph / scheduler 宽度

做法：

- 每轮都允许真实 `spec_num_tokens` 改变
- scheduler / buffer / graph / metadata 全部随之动态变化

评估：

- 理论可行
- 工程复杂度极高
- 与 async schedule / CUDA graph / overlap 机制天然冲突
- 对稳定性和维护性都不友好

结论：

- **不建议**

### 方案 B：多 graph bucket，按几个固定长度档位切换

做法：

- 预设少量长度 bucket，例如 `K in {1, 2, 4}`
- 不在每轮任意改长度，只在 bucket 间切换
- 每个 bucket 预捕获 graph、预分配 metadata

评估：

- 可行
- 比方案 A 现实很多
- 但仍要处理：
  - scheduler decode_input_tokens 宽度切换
  - `future_input_map` 协议切换
  - overlap 下不同 bucket 请求混批问题

结论：

- **局部可行，但作为首期主方案仍不够稳**

### 方案 C：固定容量 `K_cap`，动态有效长度 `K_eff <= K_cap`

做法：

- 整个系统只暴露一个静态上界 `K_cap`
- graph / scheduler / buffer / sampling / grammar 全部按 `K_cap` 建立
- 算法层只改变：
  - 本轮有效 verify 长度 `K_eff`
  - `K_mtp`
  - `K_suffix`
  - 每个位置 token 来源和 mask

评估：

- 与 async schedule 最兼容
- 与 graph capture 最兼容
- 对现有 `model_runner_v1` 改动最小
- 能满足动态投机决策的核心需求

结论：

- **推荐主方案**

### 方案 D：固定 verify 容量 + 解耦的 MTP 执行容量

做法：

- verify / scheduler / future_input_map 仍固定到总 speculative 容量
- 但 MTP drafter 不再默认跑满总长度
- 对 MTP 单独定义运行时执行长度 `K_mtp_exec`
- `K_mtp_exec` 只覆盖 tail 段，而不是整个 speculative window
- 通过少量 graph bucket 或 micro-graph 串接来执行 `K_mtp_exec`

评估：

- 比方案 C 多一层 drafter 执行复杂度
- 但这是“让 suffix 真正替代掉一部分 MTP 计算”的关键
- 外层 async schedule 协议仍可保持稳定

结论：

- **推荐作为方案 C 的性能增强版，也是后续主推荐**

---

## 5. 推荐总体架构：静态 verify 容量 + 动态有效长度 + 解耦的 MTP 执行容量

## 5.1 基本原则

首期必须把“动态长度问题”重写成“固定宽度下的动态 mask 问题”。

定义：

- `K_verify_cap`：系统级静态 verify 容量，例如 4、5 或 6
- `K_eff`：某个 batch / bucket 当前实际使用的有效 speculative 长度
- `K_suffix`：前缀中由 suffix 提供的有效 token 数
- `K_mtp`：后缀中由 MTP 提供的有效 token 数
- `K_mtp_exec`：本轮 MTP 实际执行步数
- `K_mtp_graph_cap`：某个 MTP graph bucket 的静态执行上界

满足：

- `K_eff = K_suffix + K_mtp`
- `K_eff <= K_verify_cap`
- 通常 `K_mtp_exec = K_mtp`
- `K_mtp_exec <= K_mtp_graph_cap <= K_verify_cap`

工程上 verify / target 侧静态结构只认 `K_verify_cap`：

- `future_input_map[:, K_verify_cap]`
- scheduler `decode_input_tokens = K_verify_cap`
- verify metadata `max_seq_len_q = K_verify_cap`
- grammar / sampling backend 宽度 = `K_verify_cap`
- verify graph 只按 `K_verify_cap` capture

而 drafter/MTP 侧改成：

- 不再把 `K_verify_cap` 直接等同于 MTP 执行长度
- MTP 只负责 tail 段的 `K_mtp_exec`
- `K_mtp_exec` 由 bucket graph 或 micro-graph 实现

运行时策略只修改：

- 本轮前 `K_eff` 个位置是否有效
- 每个位置是 `suffix` 还是 `mtp`
- verify 后实际接受长度 `accept_len <= K_eff`

## 5.2 为什么这个方案与 async schedule 兼容

因为 overlap / async schedule 真正需要的是：

- 下一轮输入槽位数固定
- 当前轮 forward 的图形状固定
- scheduler 对每个 decode 周期的输入宽度理解固定

固定 `K_cap` 后，这些条件都还成立。

变化的只是：

- 某些槽位是“无效 draft 位”
- verify 时这些无效位不应该参与真实接受链判断

这就变成一个**语义 mask**问题，而不是**执行形状**问题。

---

## 6. 工程实现设计

## 6.1 分层解耦

建议把混合投机实现拆成四层。

### L1. Static Capacity Layer

职责：

- 定义系统静态容量 `K_cap`
- 保证 graph / buffer / scheduler / sampling backend 都只依赖 `K_cap`

涉及改造点：

- `ModelExecutorConfig.output_length`
- `RuntimeStates.future_input_map`
- scheduler `decode_input_tokens`
- attention metadata `speculative_num_draft_tokens`
- grammar / sampling backend 初始化参数

### L2. Effective Spec State Layer

职责：

- 为每个 request / batch 保存本轮有效 speculative 语义

建议新增状态：

```python
effective_spec_len: int
suffix_spec_len: int
mtp_spec_len: int
draft_valid_mask: [K_cap]
draft_source: [K_cap]   # 0=invalid, 1=suffix, 2=mtp
```

注意：

- 这些是**语义状态**
- 不改变图形状，只改变 verify/sampling 解释方式

### L3. Hybrid Draft Composer Layer

职责：

- 生成固定宽度 `K_cap` 的 draft chain
- 前 `K_eff` 为有效 token
- 后 `K_cap - K_eff` 用占位 token 填充

建议输出：

```python
draft_tokens: [bs, K_cap]
draft_valid_lens: [bs]
draft_source_map: [bs, K_cap]
```

生成规则：

- 前 `K_suffix` 位来自 suffix
- 接下来 `K_mtp` 位来自 MTP
- 剩余位填 `pad draft token`

### L4. Verification Semantics Layer

职责：

- 在固定 `K_cap` verify 宽度下，只对前 `K_eff` 位做真实接受语义
- 后续无效位必须被强制视为不可接受

这层是整个工程方案的关键。

---

## 6.2 Verify 语义改造

当前 verify 默认假定：

- `K_cap` 个位置都是有效 draft token

但混合投机需要：

- 只有前 `K_eff` 个位置有效
- 当 `K_eff < K_cap` 时，第 `K_eff + 1` 位之后不能继续“误接受”

推荐两种实现方式。

### 方式 1：valid-length aware verify

在 sampling / verify backend 中增加：

- `draft_valid_lens[bs]`

然后 verify kernel 仅在 `pos < draft_valid_len[req]` 时参与接受判断。

优点：

- 语义最干净
- 不依赖特殊占位 token

缺点：

- 要改 verify backend 接口与 kernel

### 方式 2：invalid 位强制不可接受

思路：

- 无效位填入一个“必拒绝 draft token”
- 或让无效位 `draft_probs = 0`
- verify 逻辑遇到这些位自然停止接受

优点：

- 改动较小

缺点：

- 语义更脆弱
- 不同 sampling backend 上行为可能不完全一致

结论：

- 首期可先用方式 2 验证链路
- 正式版本建议做方式 1

---

## 6.3 `future_input_map` 协议改造

当前 `future_input_map` 的问题不是“宽度不够”，而是“缺少语义元数据”。

建议保留：

- `future_input_map[req, :K_cap]`

同时增加：

- `future_input_valid_len[req]`
- `future_input_source_map[req, :K_cap]`

含义：

- `future_input_map`：下一轮要读的 token 槽位
- `future_input_valid_len`：其中前多少个是真实 draft token
- `future_input_source_map`：这些位置来自 suffix 还是 mtp

这样 `InputBuffers.fill_input_buffers()` 仍然可以固定宽度读取：

- 始终读 `K_cap`
- 但下游 verify / drafter / stats 会根据 `valid_len/source_map` 解释

这能最大化保持 overlap schedule 的时序不变。

---

## 6.4 Scheduler 兼容方案

当前 scheduler 侧最麻烦的点是：

- `decode_input_tokens` 在初始化时固定
- overlap 下 reserve token 更新也围绕固定 decode 周期推进

推荐原则：

### scheduler 永远只看 `K_cap`，不看 `K_eff`

原因：

- 如果 scheduler 也感知 `K_eff`，它就必须和 graph/buffer 一样动态化
- 这会把复杂性扩散到 C++ 调度器层

更好的方案是：

- scheduler 仍认为 speculative decode 输入宽度是 `K_cap`
- 真正的有效接受由 GPU verify 层和 output processor 层决定
- reserve token 更新用真实 `accept_len`

也就是说：

- **输入容量固定**
- **产出长度动态**

这和当前 `make_update_reserve_tokens_event(rid, output_length)` 的语义是一致的，天然更容易兼容。

### 需要特别注意的点

由于 output processor 当前在 speculative decode 场景下使用：

- `pt += self.spec_num_tokens`

来跳过固定宽度切片，所以保持 `K_cap` 固定后，这里仍然成立。

变化只在于：

- `output_length` 可以小于 `K_cap`
- `model_output_ids = output_tokens[pt : pt + output_length]`

因此 output packing 逻辑本身不需要大改。

---

## 6.5 MTP 图模式兼容方案

问题核心：

- 当前 MTP / drafter graph 路径也是按固定 speculative 宽度建图

但这里要强调：

- “MTP 图模式兼容” 和 “MTP 图模式真的降低成本” 不是一回事
- 如果 MTP 图仍固定展开到 `mtp6`
- 那么 `suffix2+mtp1` 和 `mtp6` 在 drafter 侧可能依然接近同样昂贵

所以这里只做固定 `K_cap` + 动态 `K_eff` 还不够，还必须把 verify 容量和 MTP 执行容量拆开。

推荐做法：

### 推荐主方案：固定 verify 容量，MTP 只跑 tail bucket

即：

- verify 仍按固定总宽度 `K_verify_cap` 建图
- suffix 先填充 prefix 段
- MTP 只负责 `tail_start = K_suffix` 之后的 tail 段
- MTP 不再默认跑满 `K_verify_cap`
- 而是只执行 `K_mtp_exec`

例如：

- 总 verify 长度固定 `K_verify_cap = 6`
- 本轮动作是 `suffix3 + mtp1`
- verify 仍然验证 6 个槽位中的前 4 个有效位
- 但 drafter 只应执行 `mtp1`，而不是整条 `mtp6`

具体实现可以有三种。

#### 方案 1：MTP bucket graph（推荐）

例如预置：

- `mtp_bucket in {0, 1, 2, 4}`

映射方式：

- `K_mtp_exec=0`：纯 suffix，不跑 MTP
- `K_mtp_exec=1`：跑 `mtp1` graph
- `K_mtp_exec=2`：跑 `mtp2` graph
- `K_mtp_exec=3`：跑 `mtp4` graph，但只消费前 3 个输出
- `K_mtp_exec=4`：跑 `mtp4` graph

优点：

- 真正降低 MTP 执行成本
- graph mode 仍然保留
- bucket 数量可控

缺点：

- 需要多套 graph capture / metadata
- drafter 内部需要按 action 做路由或子批

#### 方案 2：MTP micro-graph 串接

优点：

- 与 graph 完全兼容
- bucket 数量更少
- 天然适合单链串行

缺点：

- replay 次数增多
- hidden state / KV 衔接更复杂
- 如果单次 graph 太小，调度开销可能抵消收益

#### 方案 3：完整跑满固定宽度，只取前 `K_mtp` 个有效输出

优点：

- 与当前结构最兼容
- 控制逻辑最简单

缺点：

- 多跑的 MTP step 有额外算力浪费
- 无法兑现“suffix 替代 MTP 降本”的核心收益

建议：

- **如果目标只是先跑通主链路，Phase 1 可以先用方案 3**
- **如果目标已经转向性能收益，优先级应切到方案 1**
- **推荐首批 bucket 选 `mtp{0,1,2,4}`，不要一开始铺满所有长度**

因为在你当前这个问题里，真正的瓶颈已经不是“能不能混合”，而是“混合以后 MTP 有没有真的少算”。

### 为什么这仍兼容 async schedule

因为 async schedule 需要稳定的是：

- `future_input_map` 的固定槽位协议
- verify 的固定宽度元数据
- output processor 的固定切片步长
- scheduler 对 decode 周期容量的固定理解

而 MTP bucket / micro-graph 的变化可以被限制在 drafter 内部：

- 对外仍输出固定宽度 `[bs, K_verify_cap]`
- `valid_len/source_map` 协议不变
- output / reserve token 语义不变

也就是说：

- **外层协议固定**
- **内层 MTP 执行按 bucket 变化**

这是在“开启 MTP 图模式前提 + 兼容异步调度”下，性价比最高的降本方向。

---

## 6.6 Suffix 与 MTP 的组合方式

在工程上，推荐只支持一种组合范式：

- `prefix suffix + tail mtp`

不要首期支持：

- 交错式 `suffix-mtp-suffix`
- request 内树状多分支 speculative
- request 级不同 verify 宽度混杂

推荐链式布局：

```text
[ suffix ... ][ mtp ... ][ invalid ... ]
  K_suffix       K_mtp     K_cap-K_eff
```

这样有几个明显好处：

- verify 始终是一条连续前缀链
- output packing 不变
- acceptance 统计可以按前缀位置自然分解
- suffix 只占前缀，符合其“便宜且高置信”的定位

---

## 6.7 更简化的工程版本：固定 MTP 长度 + 动态 Suffix Tail 补齐

如果进一步追求“最小实现难度 + 保住 MTP 图模式收益”，那么比 `动态 K_mtp + bucket graph` 更简单的方案是：

- 总投机长度固定：`K_total = 6`
- verify / async schedule / scheduler / `future_input_map` 全部仍按 `6` 固定
- MTP 图模式固定长度：`K_mtp_fixed = 3`
- suffix 只负责补剩余的 tail 段，补充数量 `K_suffix_tail in [0, 3]`

也就是说，运行时不再动态调整 MTP 长度，而是固定为：

```text
[ mtp ][ mtp ][ mtp ][ suffix? ][ suffix? ][ suffix? ]
   3 个固定 MTP         0~3 个动态 suffix tail
```

更准确地说：

- 前 3 个 draft token 恒由 MTP 产生
- 后 0-3 个 draft token 由 suffix 检索或构造补齐
- 若 suffix 未补满，则剩余位置为 invalid / pad

因此每轮的真实有效长度为：

- `K_eff = K_mtp_fixed + K_suffix_tail`
- 其中 `K_mtp_fixed = 3`
- `K_suffix_tail ∈ {0, 1, 2, 3}`

### 6.7.1 这套方案为什么更简单

因为它把最复杂的动态性直接砍掉了一半：

- MTP 长度不再动态
- MTP graph 不再需要多 bucket
- drafter 不再需要按 `mtp1/mtp2/mtp4` 路由
- async batch 中也不需要处理不同 MTP graph 混排

系统只保留一个固定的 MTP 图：

- 固定 replay `mtp3`
- 然后由 suffix 补 tail
- 最终仍然输出固定宽度 `6` 的 speculative buffer

所以它保留了：

- 外层固定协议
- MTP 图模式收益
- 动态 suffix 补齐能力

同时避开了：

- `mtp6` 冗余计算
- 多 graph bucket 管理复杂度
- 动态 MTP 长度和异步调度的耦合

### 6.7.2 性能收益判断

这套方案的收益不是“全局最优”，但很可能是“工程性价比最高”。

#### 收益 1：MTP 成本从 `mtp6` 直接降到固定 `mtp3`

相对当前“总长 6 下 MTP 实际按 `mtp6` 成本执行”的情况，这个方案最大的直接收益是：

- drafter 侧 MTP 成本上界直接减半
- `suffix` 再多，也不会把 MTP 成本重新抬回 `mtp6`

这一步本身就足以解决当前最痛的问题。

#### 收益 2：用低成本 suffix 去冲击第 4-6 位 acceptance

这套方案的第二层收益来自：

- 先用固定 `mtp3` 稳定覆盖前 3 位
- 再用 suffix 低成本补 4-6 位
- 如果 suffix 质量足够好，就能提高平均 `K_eff`

因此总收益主要来自两部分：

- `T_mtp: mtp6 -> mtp3`
- `K_eff: 3 -> 3~6`

#### 收益 3：收益更稳定

因为 MTP 路径完全静态，所以：

- drafter 时延更稳定
- graph replay 更稳定
- profiling 结果更容易解释
- 策略调优时只需要关注 suffix 是否值得补、补多少

### 6.7.3 这套方案的局限

它不是理论最优，主要限制是：

- 当 suffix 已经能高质量补出 2-3 个 token 时，MTP 仍然固定跑 3 步
- 无法继续把 MTP 再压缩到 1 或 0
- 所以放弃了“suffix 越强，MTP 越短”的极致收益

但这是一个典型的工程折中：

- 放弃一部分最优性
- 换取显著更低的实现复杂度和更高的稳定性

### 6.7.4 工程实现方式

这套方案可以看作是在当前框架里加一个简单的 `tail composer`。

#### Draft 组织方式

建议固定生成：

```text
[ mtp0 ][ mtp1 ][ mtp2 ][ suffix0? ][ suffix1? ][ suffix2? ]
```

对应状态：

- `draft_tokens: [bs, 6]`
- `draft_valid_len: [bs]`
- `draft_source_map: [bs, 6]`

其中：

- `source_map[0:3] = MTP`
- `source_map[3:3+K_suffix_tail] = SUFFIX`
- 剩余位置为 `INVALID`

#### 执行时序

每轮可按如下顺序执行：

1. 固定 replay 一次 `mtp3` graph
2. 基于当前上下文和 `mtp3` 输出，尝试补 `0~3` 个 suffix tail token
3. 组装为固定宽度 `6` 的 draft buffer
4. verify 仍按宽度 `6` 执行
5. verify 只对前 `3 + K_suffix_tail` 个位置生效

这与现有 async schedule 的契约是兼容的，因为：

- 输入 buffer 仍固定宽度 6
- scheduler 仍认为 speculative decode 周期宽度是 6
- output processor 仍按固定步长切片
- 真正动态的只有 `valid_len` 和 tail 段的 `source_map`

#### Verify 语义

推荐仍使用：

- `valid-length aware verify`

即：

- 前 3 位恒有效
- 后 0-3 位按 suffix 是否命中决定是否有效
- 未补到的位置必须强制 invalid，不能继续参与 acceptance

### 6.7.5 改造范围

相对 `动态 MTP bucket graph` 方案，这套简化版的改造点更集中：

#### 必做

- 保留固定 `K_total = 6`
- 保留固定 `mtp3` graph
- 增加 suffix tail composer
- 增加 `draft_valid_len`
- 增加 `draft_source_map`
- verify 正确处理 tail invalid 位
- 增加 suffix tail 命中率和 acceptance 统计

#### 可以不做

- 不需要多套 MTP graph bucket
- 不需要 MTP graph manager
- 不需要 per-batch MTP 路由
- 不需要动态 `K_mtp_exec`

### 6.7.6 可行性判断

如果目标是：

- 先做一个真的能落地的版本
- 明确降低当前 MTP 图模式冗余计算
- 又不想一下子引入 bucket graph 的维护复杂度

那么这套方案的可行性我认为是：

- **工程可行性：高**
- **实现复杂度：低到中**
- **性能收益确定性：高于当前方案**
- **理论最优性：低于动态 MTP bucket 方案**

一句话总结：

**这是一个“不是最优，但最像 first practical version”的方案。**

### 6.7.7 推荐定位

建议把它作为：

- **PoC 首选方案**
- **Phase 1.5 / Phase 2 的优先落地版本**

推荐命名：

- `Fixed Verify-6 + Fixed MTP-3 + Dynamic Suffix-Tail(0..3)`

如果后续验证发现：

- suffix tail 质量一般
- 或第 4-6 位 acceptance 很差

再进一步演进到：

- `Fixed Verify-6 + Dynamic MTP Bucket + Dynamic Suffix`

会更自然，也更安全。

---

## 7. 动态决策器如何落地

## 7.1 不要按 request 粒度实时改策略

如果每个 request 单独决定 `K_eff/K_mtp/K_suffix`，会有两个问题：

- batch 内部语义差异过大，debug 和 profiling 都困难
- graph 虽然仍可复用，但 runtime 解释逻辑会非常复杂

建议：

### 按 batch bucket 或时间窗口做动作选择

例如按以下 bucket：

- `ctx_len`: `<4K`, `4K-16K`, `16K-64K`, `64K+`
- `bs`: `1-4`, `5-8`, `9-16`, `17+`

然后每个 bucket 选一个动作：

- `OFF`
- `suffix1`
- `suffix2`
- `mtp2`
- `mtp3`
- `suffix1+mtp2`
- `suffix2+mtp2`

如果引入 MTP bucket graph，建议再做一层执行映射：

- policy 动作：决定 `K_eff / K_suffix / K_mtp`
- 执行动作：决定 `K_mtp_exec / mtp_bucket_id`

例如：

- `suffix2+mtp1` -> verify 语义是 3 个有效位，但 drafter 只跑 `mtp1`
- `suffix2+mtp3` -> verify 语义是 5 个有效位，但 drafter 路由到 `mtp4` bucket 并只消费前 3 个输出

这样 policy 层仍然关注算法收益，执行层只关注 graph 复用和算力开销，职责更清晰。

这样可以保证：

- 同一时间窗口内，大多数 batch 的语义比较一致
- 在线统计更稳定
- 工程实现更容易维护

## 7.2 动作更新频率

不建议逐轮变。

建议：

- 每 `N` 个 decode step
- 或每 `T` 秒
- 或每收集到足够样本后

再更新一次 bucket 的默认动作。

这样可以避免：

- overlap 调度下前后两轮状态抖动
- 统计未收敛就频繁切动作

---

## 8. 可行性判断

## 8.1 首期可行

以下目标我认为是**高可行**的：

- 固定 `K_verify_cap=4/5/6`
- verify / scheduler / buffer 都按 `K_verify_cap` 固定
- 动态改变 `K_eff`
- 支持 `suffix prefix + mtp tail`
- 支持 batch bucket 粒度的动作选择
- 支持 overlap / async schedule 不退化
- MTP 至少支持少量 bucket graph，如 `mtp{0,1,2,4}`

原因：

- 不需要重写 scheduler 协议
- 不需要彻底重写 graph capture
- 不需要 request 内多分支 verify
- 只是在固定宽度链路上增加“有效长度语义”

## 8.2 中期可行

以下目标是**中期可行**：

- MTP micro-graph 串接
- request 级更细粒度 policy
- suffix 来源多样化
- 更精确的 valid-length aware verify kernel

这些可以在主链路稳定后逐步上。

## 8.3 暂不建议

以下目标短期不建议：

- 每 request 任意动态 `spec_num_tokens`
- graph 宽度随请求实时变化
- scheduler `decode_input_tokens` 动态变化
- 混合 speculative 树状分支 verify

这是高风险、低性价比方向。

---

## 9. 推荐实现方案

## 9.1 总体方案

推荐采用：

### 方案主线：`Static K_verify_cap + Dynamic K_eff + Prefix-Suffix + Tail-MTP + Bucketed MTP Exec`

如果从“最快落地、最小改造、先解决当前 `mtp6` 冗余成本”出发，则更推荐先落简化版：

### 当前优先推荐：`Fixed Verify-6 + Fixed MTP-3 + Dynamic Suffix-Tail(0..3)`

即：

- verify 容量固定 6
- MTP 图固定 3
- suffix 只补 tail 的 0-3 位
- `valid_len/source_map` 表达真实有效长度和来源

这版不是算法最优，但工程风险最低，也最容易验证真实收益。

具体为：

1. 系统配置静态 verify 容量 `K_verify_cap`
2. scheduler / verify graph / buffer / sampling backend 全部固定到 `K_verify_cap`
3. 新增 request/batch 级 spec 语义状态：
   - `K_eff`
   - `K_suffix`
   - `K_mtp`
   - `K_mtp_exec`
   - `mtp_bucket_id`
   - `valid_mask`
   - `source_map`
4. `HybridDrafter` 固定输出 `[bs, K_verify_cap]`
5. suffix 负责 prefix，MTP 只负责 tail 段执行
6. MTP graph manager 按 `mtp_bucket_id` 选择最小可用 graph
7. verify 只对前 `K_eff` 生效
8. output processor 仍用固定步长切片，但真实 `output_length=accept_len`
9. policy 按 bucket 更新动作，而不是逐 request 动态跳

## 9.2 关键模块改造建议

建议新增或改造：

- `runtime/execution/drafter/hybrid.py`
- `runtime/execution/drafter/suffix.py`
- `runtime/execution/drafter/mtp_graph_manager.py`
- `runtime/spec_decode/hybrid_policy.py`
- `runtime/spec_decode/hybrid_state.py`

建议扩展：

- `RuntimeStates`
- `InputBuffers`
- `SamplingBackend.verify(...)`
- `ModelExecutor._forward_step()`
- `GenerationOutputProcessor`

## 9.3 最小改动优先级

### 必做

- 固定 `K_verify_cap`
- 新增 `valid_len/source_map`
- 混合 drafter
- verify 对无效位的正确处理
- MTP bucket graph 或等价的 tail-only 执行路径
- runtime stats 与 profiling

### 次优先

- MTP micro-graph
- 更复杂 suffix 召回
- 更细粒度在线 bandit

---

## 10. 分阶段落地路线

## Phase 0：可观测性

目标：

- 先把问题量化清楚

实现：

- 统计 `accept_len` 分布
- 统计按位置 acceptance
- 分开统计 suffix 段和 mtp 段 acceptance
- 统计 `T_verify / T_mtp / suffix_build_us`
- 统计 bucket 维度收益

## Phase 1：固定 `K_verify_cap` 的 MTP-only 动态有效长度

目标：

- 不引入 suffix，先验证“固定容量 + 动态有效长度”主链路

实现：

- `K_verify_cap=4`
- policy 只在 `mtp2/mtp3/mtp4/OFF` 中选
- verify 支持 `valid_len`

成功标准：

- async schedule 正常
- graph replay 正常
- output / reserve token 正常

## Phase 2：把 MTP 执行长度从 verify 总长度中解耦

目标：

- 证明 `K_mtp` 下降时，drafter 成本会真实下降

实现：

- 保持 `K_verify_cap` 固定
- 引入 `mtp{0,1,2,4}` bucket graph
- 统计不同 `K_mtp_exec` 下的 `T_mtp`
- 验证 `suffix` 增加后总 draft 时间是否下降

成功标准：

- `suffix2+mtp1` 明显低于 `mtp4/mtp6`
- graph replay 与 async schedule 不冲突
- batch bucket 切换不会破坏稳定性

## Phase 3：接入 suffix-only 与 hybrid

目标：

- 引入 `suffix1/2`
- 支持 `suffix2+mtp2` 等组合

实现：

- `HybridDrafter`
- `source_map`
- suffix acceptance 单独统计

## Phase 4：策略控制器

目标：

- 按 bucket 动态选 `K_eff/K_suffix/K_mtp`

实现：

- EMA 或 bandit
- 周期性更新动作

## Phase 5：性能精修

目标：

- 继续降低 MTP bucket 带来的剩余浪费

实现：

- MTP micro-graph
- 更轻量的 invalid 位处理
- 更精准 roofline-aware 策略

---

## 11. 风险与应对

### 风险 1：MTP graph bucket 仍有 padding 浪费

应对：

- bucket 设计成 `0/1/2/4` 这类稀疏集合，而不是把所有长度全量铺开
- 优先压缩高频动作覆盖到最少 bucket
- 如果 `mtp1` 仍偏贵，再进一步考虑 micro-graph

### 风险 2：invalid 位处理不严谨导致误接受

应对：

- 尽快落地 valid-length aware verify
- 不把“特殊 token 必拒绝”当长期方案

### 风险 3：suffix 前缀后接 MTP 导致分布漂移

应对：

- 首期限制 `K_mtp <= 2/3`
- 单独统计“suffix 后接 mtp”的位置接受率

### 风险 4：同一批次内 MTP bucket 过多，导致 drafter 路由开销增加

应对：

- policy 只在 batch bucket 粒度选动作
- 限制同一轮可出现的 graph bucket 数
- 必要时做 drafter 子批，而不是 request 级完全离散

### 风险 5：bucket 动作切换过于频繁导致收益不稳定

应对：

- 限制最小驻留时间
- 用 EMA 而不是瞬时值

---

## 12. 结论

在 `vllm-ascend` 的 `model_runner_v1`、MTP 图模式和 async schedule 约束下，真正可落地的方向不是“让 speculative 长度完全动态”，而是：

- **把 target verify 执行形状固定在 `K_verify_cap`**
- **把算法自由度保留在 `K_eff/K_suffix/K_mtp`**
- **把 MTP 执行长度从 verify 总长度里解耦出来**
- **用 valid length 和 source map 表达混合投机语义**

这样做的价值是：

- 不破坏 async schedule 的固定协议
- 不破坏 graph mode 的固定形状假设
- 不需要把复杂性扩散到 scheduler C++ 层
- 还能让 `suffix` 真正替代掉一部分昂贵 MTP 计算

一句话总结：

**动态策略要做，但动态的应该是“有效长度”“来源分配”和“MTP tail 执行长度”，不应该是 target verify 的 runtime 主形状。**
