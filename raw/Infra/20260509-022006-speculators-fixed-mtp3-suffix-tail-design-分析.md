# 固定 Verify-6 + 固定 MTP-3 + 动态 Suffix-Tail(0..3) 方案设计

## 1. 背景与目标

当前混合投机方案已经基本明确一个事实：

- 在 `async schedule`、固定宽度 buffer、固定 verify metadata、MTP graph mode 前提下
- `total speculative length` 更适合固定，而不是在运行时频繁动态变化

但如果采用：

- 总投机长度固定为 `6`
- 动态调整 `mtp/suffix` 比例

又会遇到一个新的性能问题：

- verify 开销天然仍按固定 `6` 执行
- 如果 MTP graph 也按 `mtp6` 固定展开
- 那么即使本轮只需要 `mtp1/2/3`
- drafter 侧仍然可能以接近 `mtp6` 的成本运行

这会导致：

- suffix 虽然语义上替代了一部分 MTP
- 但执行上没有真正减少 MTP 计算
- 混合投机的主要收益被吞掉

因此本设计的目标不是追求算法最优，而是优先实现一个：

- 工程上简单
- 明确可落地
- 能真实降低当前 MTP 冗余成本
- 与 `async schedule + MTP graph mode` 兼容

的方案。

本文推荐的主方案为：

`Fixed Verify-6 + Fixed MTP-3 + Dynamic Suffix-Tail(0..3)`

---

## 2. 核心思想

将投机链拆成两段：

- 前 3 个 token 固定由 MTP graph 生成
- 后 0 到 3 个 token 由 suffix 动态补齐

固定点：

- 总 verify 宽度固定：`K_verify = 6`
- MTP graph 长度固定：`K_mtp_fixed = 3`
- scheduler / future_input_map / output processor 协议固定

动态点：

- suffix tail 命中长度：`K_suffix_tail in [0, 3]`
- 本轮真实有效 speculative 长度：
  - `K_eff = 3 + K_suffix_tail`

最终 draft buffer 形态固定为：

```text
[ mtp0 ][ mtp1 ][ mtp2 ][ suffix0? ][ suffix1? ][ suffix2? ]
```

其中：

- 前 3 位恒有效
- 后 0 到 3 位按 suffix 是否命中决定是否有效
- 未命中的位置填 invalid / pad

---

## 3. 为什么选择这个方案

## 3.1 直接解决当前最痛点

当前最大问题不是“能不能混合”，而是：

- MTP graph 仍被最大长度绑死
- `suffix` 提升后没有换来真实 drafter 降本

固定 `mtp3` 后，drafter 开销上界立刻从 `mtp6` 降到 `mtp3`：

- 不再为 `mtp4/5/6` 预留执行
- `suffix` 不会把 MTP 成本重新抬高

## 3.2 明显简化工程复杂度

相比动态 `K_mtp` 或 `MTP bucket graph` 方案，这个版本不需要：

- 多套 MTP graph capture
- graph bucket 路由
- batch 内不同 MTP 图混排
- 动态 `K_mtp_exec`
- drafter graph manager

因此复杂度集中在：

- suffix tail 构造
- `valid_len/source_map`
- verify 对 invalid tail 的正确处理

## 3.3 保住当前框架最重要的静态契约

本方案不触碰下面这些关键假设：

- verify 宽度固定
- scheduler `decode_input_tokens` 固定
- `future_input_map` 槽位固定
- overlap / async schedule 的输入输出协议固定
- output processor 固定步长切片

所以它非常适合作为首个工程版本。

---

## 4. 方案定义

## 4.1 固定参数

- `K_verify = 6`
- `K_mtp_fixed = 3`
- `K_suffix_tail_max = 3`

## 4.2 运行时变量

- `K_suffix_tail`
  - 当前轮 suffix 实际补到的 token 数
  - 取值范围 `[0, 3]`
- `K_eff`
  - 当前轮真实有效 speculative 长度
  - `K_eff = 3 + K_suffix_tail`

## 4.3 draft token 来源布局

```text
pos:       0      1      2       3         4         5
source:   mtp    mtp    mtp   suffix?   suffix?   suffix?
valid:     1      1      1      0/1       0/1       0/1
```

约束：

- suffix 只允许补 tail，不允许改前 3 位
- verify 始终按连续前缀解释有效长度
- 无效尾位必须强制终止 acceptance

---

## 5. 预期收益

## 5.1 主要收益来源

总收益主要来自两部分：

### 收益 A：MTP 成本下降

从当前潜在的：

- `T_mtp ~= cost(mtp6)`

变成：

- `T_mtp ~= cost(mtp3)`

这是最确定的收益。

### 收益 B：suffix 提升有效 speculative 长度

通过 suffix tail，把有效长度从固定 3 提高到 3 到 6：

- 当 `K_suffix_tail=0` 时，退化为 `mtp3`
- 当 `K_suffix_tail=1/2/3` 时，分别获得更长的 speculative chain

如果 suffix tail 质量不错，能够带来：

- 更高平均 `K_eff`
- 更高平均 acceptance
- 更少 target-only decode 轮数

## 5.2 收益画像

这个方案的性能画像大致如下：

- verify 成本：固定，按 6
- MTP 成本：固定，按 3
- suffix 成本：低且动态
- 总收益稳定性：高

它不是“理论最优”，但大概率是“首期最稳、收益最可控”的版本。

## 5.3 收益边界

这套方案放弃了一部分潜在极致收益：

- 即使 suffix 已经能补出 2 到 3 个高质量 token
- MTP 仍然固定跑 3 步

所以它无法做到：

- `suffix 强 -> MTP 继续缩短`

这部分收益需要后续升级到动态 MTP bucket 才能拿到。

---

## 6. 风险与限制

## 6.1 suffix tail 质量可能偏弱

因为 suffix 补的是第 4 到 6 位，通常比补前缀更难：

- 上下文更长
- 分布更敏感
- acceptance 可能低于前 3 位 MTP

影响：

- `K_eff` 虽然上去了，但真实 acceptance 不一定同步提升

## 6.2 可能退化成 `mtp3 + verify6`

如果 suffix 经常补不到，系统会长期处于：

- MTP 固定 3
- verify 仍跑 6
- tail 段经常无效

这种情况下收益上限有限，但仍然通常优于当前 `mtp6` 冗余方案。

## 6.3 invalid tail 处理必须严格

如果 verify 对无效尾位处理不严谨，会导致：

- 误接受
- acceptance 统计失真
- output packing 语义错误

因此 `valid-length aware verify` 是强推荐项。

---

## 7. 执行时序设计

每轮 speculative decode 建议按如下顺序执行：

1. 读取当前请求上下文和前一轮状态
2. 固定 replay 一次 `mtp3` graph
3. 得到前 3 个 MTP draft token
4. 基于当前上下文和 MTP 结果尝试构造 suffix tail
5. suffix 最多补 3 个 token
6. 组装固定宽度 `6` 的 draft buffer
7. 生成 `draft_valid_len`
8. 生成 `draft_source_map`
9. 进入固定宽度 `verify6`
10. verify 只对前 `K_eff` 位做真实 acceptance 语义
11. output processor 仍按固定步长处理，但真实输出长度取 `accept_len`

---

## 8. Buffer 与状态设计

## 8.1 Draft Buffer

建议输出：

```python
draft_tokens: Tensor[bs, 6]
draft_valid_len: Tensor[bs]
draft_source_map: Tensor[bs, 6]
```

语义：

- `draft_tokens`
  - 固定宽度 speculative token buffer
- `draft_valid_len`
  - 当前请求前多少个位置有效
- `draft_source_map`
  - 标记每个位置来源

建议枚举：

```python
INVALID = 0
MTP = 1
SUFFIX = 2
```

## 8.2 Runtime State

建议新增状态：

```python
mtp_fixed_len: int = 3
suffix_tail_len: int
effective_spec_len: int
```

可选统计状态：

```python
accept_len: int
accept_mtp_prefix_len: int
accept_suffix_tail_len: int
```

## 8.3 `future_input_map`

仍保留固定宽度：

```python
future_input_map: Tensor[num_reqs, 6]
future_input_valid_len: Tensor[num_reqs]
future_input_source_map: Tensor[num_reqs, 6]
```

对 async schedule 的影响：

- 输入槽位宽度不变
- 时序协议不变
- 只增加语义解释字段

---

## 9. Verify 语义设计

## 9.1 基本原则

verify 仍按静态宽度 6 运行，但只对前 `K_eff` 位生效。

即：

- `pos < draft_valid_len` 的位置参与 acceptance 判断
- `pos >= draft_valid_len` 的位置必须视为 invalid

## 9.2 推荐实现

推荐直接支持：

```python
draft_valid_len: Tensor[bs]
```

在 verify backend 中加入长度判断：

```python
for pos in range(6):
    if pos >= draft_valid_len[req]:
        reject_and_stop()
    else:
        normal_verify()
```

## 9.3 不推荐长期使用的替代方案

可以短期用“特殊 token 必拒绝”做链路验证，但不建议长期依赖，因为：

- 语义脆弱
- backend 相关性强
- 容易在不同实现上出现边界行为差异

---

## 10. 伪代码

```python
def build_hybrid_draft(req_state) -> HybridDraft:
    mtp_tokens = run_fixed_mtp3_graph(req_state)

    suffix_tokens = build_suffix_tail(
        req_state=req_state,
        prefix_tokens=mtp_tokens,
        max_tail_len=3,
    )

    suffix_tail_len = len(suffix_tokens)
    effective_spec_len = 3 + suffix_tail_len

    draft_tokens = [PAD] * 6
    draft_source_map = [INVALID] * 6

    for i in range(3):
        draft_tokens[i] = mtp_tokens[i]
        draft_source_map[i] = MTP

    for j, token in enumerate(suffix_tokens):
        pos = 3 + j
        draft_tokens[pos] = token
        draft_source_map[pos] = SUFFIX

    return HybridDraft(
        draft_tokens=draft_tokens,
        draft_valid_len=effective_spec_len,
        draft_source_map=draft_source_map,
    )
```

verify 侧：

```python
def verify_draft(target_logits, draft_tokens, draft_valid_len):
    accept_len = 0
    for pos in range(6):
        if pos >= draft_valid_len:
            break
        if accept(target_logits[pos], draft_tokens[pos]):
            accept_len += 1
        else:
            break
    return accept_len
```

---

## 11. 模块改造建议

## 11.1 建议新增

- `runtime/execution/drafter/hybrid_fixed_mtp_suffix.py`
- `runtime/execution/drafter/suffix_tail.py`
- `runtime/spec_decode/fixed_mtp_suffix_policy.py`

## 11.2 建议扩展

- `RuntimeStates`
- `InputBuffers`
- `SamplingBackend.verify(...)`
- `ModelExecutor._forward_step()`
- `GenerationOutputProcessor`

## 11.3 不需要新增的复杂模块

本版本可以不引入：

- `mtp_graph_manager`
- 多 bucket graph capture 路由
- 动态 `K_mtp_exec`

---

## 12. 指标与观测

## 12.1 必须统计

- `avg_accept_len`
- `avg_effective_spec_len`
- `suffix_tail_hit_rate`
- `suffix_tail_len_distribution`
- `acceptance_by_position`
- `acceptance_by_source`
- `T_mtp`
- `T_suffix`
- `T_verify`
- `end_to_end_decode_step_time`

## 12.2 推荐分位统计

- `P50/P90/P99(T_mtp)`
- `P50/P90/P99(T_suffix)`
- `P50/P90/P99(T_verify)`

## 12.3 建议重点看两类 acceptance

### 按位置

- pos 0-2：MTP acceptance
- pos 3-5：suffix tail acceptance

### 按来源

- `accept_len_mtp_prefix`
- `accept_len_suffix_tail`

---

## 13. Phase 计划

## Phase 0：可观测性

目标：

- 先量化当前 `mtp6` 路径的真实代价

输出：

- `T_mtp6`
- verify6 成本
- 各位置 acceptance

## Phase 1：固定 `mtp3` 主链路

目标：

- 在不引入 suffix 的前提下，先把 `mtp6 -> mtp3` 打通

实现：

- 固定 verify6
- 固定 replay `mtp3`
- `draft_valid_len = 3`

成功标准：

- async schedule 正常
- graph replay 正常
- output / reserve token 正常

## Phase 2：接入动态 suffix tail

目标：

- 支持 `suffix_tail_len in [0, 3]`

实现：

- suffix tail composer
- `draft_source_map`
- valid-length aware verify

成功标准：

- 无效 tail 不误接受
- suffix hit/miss 统计正确
- E2E latency 可稳定复现

## Phase 3：策略控制

目标：

- 决定什么时候值得补 suffix tail

实现：

- 基于 bucket 的 on/off 或长度策略
- 例如只在某些上下文长度或 batch 大小时补 tail

成功标准：

- suffix tail 引入后收益稳定为正

## Phase 4：后续演进

如果后续发现：

- suffix tail 很强
- 但固定 `mtp3` 仍有可压缩空间

再升级到：

- 动态 MTP bucket graph

---

## 14. 方案结论

`Fixed Verify-6 + Fixed MTP-3 + Dynamic Suffix-Tail(0..3)` 的定位非常明确：

- 不是理论最优
- 但工程风险低
- 与当前框架契合度高
- 能真实解决当前 `mtp6` 冗余执行问题

它的本质是一个工程折中：

- 放弃“suffix 强时继续压缩 MTP”的最优性
- 换取“先把 MTP 冗余成本砍掉一半，再用 suffix 低成本扩有效长度”的稳定收益

如果当前目标是：

- 先做出一版可落地、可测、可复用的混合投机实现

那么这套方案应作为首选。
