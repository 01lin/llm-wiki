# vLLM / vLLM Ascend Step 级动态投机长度 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` or `superpowers:executing-plans` to
> implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 在 vLLM/vllm-ascend 中实现基于实际 decode 并发范围的多候选 MTP
长度选择，使同一调度 step 的所有请求使用统一策略长度，并保持 async scheduling、
FULL ACL Graph、standard rejection sampling 和固定 MTP-K 的稳态性能路径。

**Architecture:** `num_speculative_tokens` 继续表示最大容量。启动时将并发范围编译成
只读 O(1) LUT，并缓存每个候选长度的不可变 step plan。scheduler 在形成
`SchedulerOutput` 后统计真实 decode 并发并盖章；worker 只读取当前 output 的 plan。
target verification 使用已有实际 draft，proposer 使用本 step 选择的 K 生成下一步
draft，不保存请求 cohort，不排空异步队列。

**Tech Stack:** Python 3.10+、msgspec/dataclass、vLLM V1 Scheduler、
AsyncScheduler、PyTorch、torch-npu、Ascend ACL Graph、pytest、NPU profiler。

---

## 0. 基线和边界

### 0.1 代码基线

- vLLM：`0d29612292c6b1e312af42ac00cf649af16a438b`
- vllm-ascend：`8afdf356f6a2496bedfc538253366ef1a8c0d9aa`
- 权威设计：
  `/Users/linyi/code/Documents/obsidian_wiki/llm-wikid/raw/Infra/20260623-120055-vllm-vllm-ascend-dynamic-speculative-length-step-policy-phase1-方案.md`

开始实现前执行：

```bash
git -C /Users/linyi/code/Documents/code/vllm status --short
git -C /Users/linyi/code/Documents/code/vllm-ascend status --short
git -C /Users/linyi/code/Documents/code/vllm rev-parse HEAD
git -C /Users/linyi/code/Documents/code/vllm-ascend rev-parse HEAD
```

Expected：

- 两个工作树无未确认修改。
- commit 与上面的基线一致；若不一致，先重新定位符号和测试，不直接套用行号。

### 0.2 首批准入配置

```json
{
  "method": "mtp",
  "num_speculative_tokens": 8,
  "dynamic_speculative_length": {
    "enabled": true,
    "candidate_lengths": [1, 3, 5, 8],
    "default_length": 1,
    "policy": {
      "type": "concurrency_table",
      "rules": [
        {"min_concurrency": 1, "max_concurrency": 4, "speculative_length": 8},
        {"min_concurrency": 5, "max_concurrency": 8, "speculative_length": 5},
        {"min_concurrency": 9, "max_concurrency": 16, "speculative_length": 3},
        {"min_concurrency": 17, "max_concurrency": 64, "speculative_length": 1}
      ]
    },
    "strict_graph_mode": true
  },
  "rejection_sample_method": "standard"
}
```

### 0.3 不实施

- 请求级长度和请求 cohort。
- 等待所有活跃请求进入同一个安全点。
- 切换时排空 async queue。
- 在线 ACL Graph capture。
- 始终执行最大 K 再屏蔽。
- 接受率/上下文/TPOT 在线寻优。
- block verify、entropy verify 的算法修改。
- 未验证的跨独立 scheduler 共享 proposer collective 拓扑。

## 1. 文件结构

### 1.1 vLLM 新增

| 文件 | 单一职责 |
|---|---|
| `vllm/v1/spec_decode/dynamic_length.py` | 配置运行时类型、并发 LUT 策略、step plan |
| `tests/v1/spec_decode/test_dynamic_length.py` | 范围校验、LUT、计划缓存 |
| `tests/v1/core/test_dynamic_spec_scheduler.py` | step 盖章、升降档、并发统计 |
| `benchmarks/overheads/benchmark_dynamic_spec_policy.py` | scheduler CPU 查表与分配开销 |

### 1.2 vLLM 修改

| 文件 | 修改内容 |
|---|---|
| `vllm/config/speculative.py` | 动态配置、规则校验、hash |
| `vllm/config/__init__.py` | 导出配置类型 |
| `vllm/forward_context.py` | `BatchDescriptor.uniform_query_len` |
| `vllm/v1/cudagraph_dispatcher.py` | 多 query length 图键与 padding |
| `vllm/v1/core/sched/output.py` | `SpecDecodeStepPlan` 字段 |
| `vllm/v1/core/sched/scheduler.py` | 统计 decode 并发、查表、截断、盖章 |
| `vllm/v1/core/sched/async_scheduler.py` | 候选 placeholder bank |
| `vllm/platforms/interface.py` | 动态 step plan 能力门禁 |
| `tests/test_config.py` | 配置回归 |
| `tests/v1/cudagraph/test_cudagraph_dispatch.py` | 多长度图键 |
| `tests/v1/core/test_async_scheduler.py` | output 级 placeholder |

### 1.3 vllm-ascend 新增

| 文件 | 单一职责 |
|---|---|
| `vllm_ascend/spec_decode/dynamic_length.py` | Ascend 候选执行计划注册表 |
| `tests/ut/spec_decode/test_dynamic_length.py` | registry、active proposer 长度 |
| `tests/ut/compilation/test_dynamic_acl_graph.py` | ACL Graph 复合键 |
| `tests/e2e/pull_request/one_card/spec_decode/test_dynamic_mtp.py` | 动态精度与图命中 |
| `benchmarks/scripts/benchmark_dynamic_spec_length.py` | fixed/dynamic 性能对照 |

### 1.4 vllm-ascend 修改

| 文件 | 修改内容 |
|---|---|
| `vllm_ascend/platform.py` | 声明支持条件 |
| `vllm_ascend/worker/model_runner_v1.py` | target/draft 计划路由、最大 stride |
| `vllm_ascend/spec_decode/llm_base_proposer.py` | 显式 active length |
| `vllm_ascend/compilation/acl_graph.py` | `GraphParamKey` |
| `vllm_ascend/compilation/compiler_interface.py` | 多长度 capture 规格 |
| `vllm_ascend/patch/worker/patch_cudagraph.py` | 新图键签名 |
| `vllm_ascend/attention/context_parallel/attention_cp.py` | 实际 query lengths |
| `vllm_ascend/attention/attention_v1.py` | 使用复合 GraphParamKey |
| `vllm_ascend/attention/mla_v1.py` | 使用复合 GraphParamKey |
| `vllm_ascend/attention/context_parallel/mla_cp.py` | 使用复合 GraphParamKey |
| `vllm_ascend/ops/gdn.py` | 使用复合 GraphParamKey |
| `vllm_ascend/_310p/ops/fla/gdn_310.py` | 保持 GraphParamKey 接口一致，310P 不纳入首批准入 |
| `tests/ut/worker/a2/test_model_runner_v1.py` | runner 路由和 stride |

## 2. 核心类型契约

### 2.1 配置类型

文件：`vllm/config/speculative.py`

```python
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ConcurrencySpecLengthRule:
    min_concurrency: int
    max_concurrency: int
    speculative_length: int


@dataclass(frozen=True)
class ConcurrencyTablePolicyConfig:
    type: Literal["concurrency_table"]
    rules: tuple[ConcurrencySpecLengthRule, ...]


@dataclass(frozen=True)
class DynamicSpeculativeLengthConfig:
    enabled: bool
    candidate_lengths: tuple[int, ...]
    default_length: int
    policy: ConcurrencyTablePolicyConfig
    strict_graph_mode: bool = True
```

校验函数固定职责：

```python
def validate(self, *, max_num_seqs: int, max_spec_length: int) -> None:
    candidates = self.candidate_lengths
    if tuple(sorted(set(candidates))) != candidates:
        raise ValueError(
            "candidate_lengths must be strictly increasing and unique"
        )
    if not candidates or candidates[0] <= 0:
        raise ValueError("candidate_lengths must contain positive integers")
    if candidates[-1] != max_spec_length:
        raise ValueError(
            "num_speculative_tokens must equal max(candidate_lengths)"
        )
    if self.default_length not in candidates:
        raise ValueError("default_length must be a candidate length")

    coverage = [False] * (max_num_seqs + 1)
    for rule in self.policy.rules:
        if rule.min_concurrency < 1:
            raise ValueError("min_concurrency must be at least 1")
        if rule.max_concurrency < rule.min_concurrency:
            raise ValueError("max_concurrency must be >= min_concurrency")
        if rule.max_concurrency > max_num_seqs:
            raise ValueError("rule exceeds max_num_seqs")
        if rule.speculative_length not in candidates:
            raise ValueError("rule length must be a candidate length")
        for concurrency in range(
            rule.min_concurrency, rule.max_concurrency + 1
        ):
            if coverage[concurrency]:
                raise ValueError("concurrency rules must not overlap")
            coverage[concurrency] = True
```

首期允许未覆盖范围回落到 `default_length`，但启动日志必须输出缺口。

### 2.2 运行时类型

文件：`vllm/v1/spec_decode/dynamic_length.py`

```python
import msgspec


class SpecDecodeStepPlan(msgspec.Struct, frozen=True):
    policy_version: int
    selected_mtp_length: int
    decode_concurrency: int


class ConcurrencyTableSpecLengthPolicy:
    def __init__(
        self,
        *,
        max_num_seqs: int,
        candidate_lengths: tuple[int, ...],
        default_length: int,
        rules: tuple[tuple[int, int, int], ...],
        policy_version: int = 1,
    ) -> None:
        lut = [default_length] * (max_num_seqs + 1)
        for min_concurrency, max_concurrency, length in rules:
            for concurrency in range(min_concurrency, max_concurrency + 1):
                lut[concurrency] = length
        self._length_by_concurrency = tuple(lut)
        self._plans = {
            concurrency: SpecDecodeStepPlan(
                policy_version=policy_version,
                selected_mtp_length=self._length_by_concurrency[concurrency],
                decode_concurrency=concurrency,
            )
            for concurrency in range(max_num_seqs + 1)
        }
        self._candidate_lengths = candidate_lengths

    def select(self, decode_concurrency: int) -> SpecDecodeStepPlan:
        try:
            return self._plans[decode_concurrency]
        except KeyError as exc:
            raise RuntimeError(
                f"decode concurrency {decode_concurrency} exceeds prepared range"
            ) from exc
```

计划按并发值缓存而不是只按 K 缓存，因为观测字段 `decode_concurrency` 不同。热路径
只做 tuple/dict 索引，不构造对象、不加锁。

### 2.3 SchedulerOutput

文件：`vllm/v1/core/sched/output.py`

```python
spec_decode_step_plan: SpecDecodeStepPlan | None = None
```

约束：

- `make_empty()` 设置 `None`。
- 动态功能关闭时保持 `None`。
- output 序列化和 worker 广播包含该字段。

## 3. Task 1：配置与 LUT 策略

**Files:**

- Modify: `vllm/config/speculative.py`
- Modify: `vllm/config/__init__.py`
- Create: `vllm/v1/spec_decode/dynamic_length.py`
- Modify: `tests/test_config.py`
- Create: `tests/v1/spec_decode/test_dynamic_length.py`

- [ ] **Step 1: 写配置失败测试**

```python
import pytest


@pytest.mark.parametrize(
    ("rules", "message"),
    [
        (((1, 8, 3), (8, 16, 1)), "must not overlap"),
        (((0, 4, 8),), "at least 1"),
        (((9, 8, 3),), "must be >="),
        (((1, 4, 7),), "must be a candidate"),
    ],
)
def test_dynamic_spec_concurrency_rules_reject_invalid_ranges(
    rules, message
):
    config = make_dynamic_spec_config(
        candidates=(1, 3, 5, 8),
        default_length=1,
        rules=rules,
    )
    with pytest.raises(ValueError, match=message):
        config.validate(max_num_seqs=64, max_spec_length=8)
```

- [ ] **Step 2: 写 LUT 边界测试**

```python
def test_concurrency_policy_selects_all_range_boundaries():
    policy = ConcurrencyTableSpecLengthPolicy(
        max_num_seqs=64,
        candidate_lengths=(1, 3, 5, 8),
        default_length=1,
        rules=((1, 4, 8), (5, 8, 5), (9, 16, 3), (17, 64, 1)),
    )
    expected = {
        1: 8,
        4: 8,
        5: 5,
        8: 5,
        9: 3,
        16: 3,
        17: 1,
        64: 1,
    }
    for concurrency, length in expected.items():
        assert policy.select(concurrency).selected_mtp_length == length
```

- [ ] **Step 3: 运行并确认失败**

```bash
cd /Users/linyi/code/Documents/code/vllm
pytest -q tests/test_config.py -k dynamic_spec_concurrency
pytest -q tests/v1/spec_decode/test_dynamic_length.py
```

Expected：FAIL，配置类型和策略尚不存在。

- [ ] **Step 4: 实现最小配置和策略**

按第 2 节类型契约实现；在 speculative config 的 hash 因子中加入所有动态配置字段。
动态功能未配置时不创建 policy。

- [ ] **Step 5: 运行测试**

```bash
pytest -q tests/test_config.py -k "speculative or dynamic_spec"
pytest -q tests/v1/spec_decode/test_dynamic_length.py
```

Expected：PASS。

- [ ] **Step 6: 提交**

```bash
git add vllm/config/speculative.py vllm/config/__init__.py \
  vllm/v1/spec_decode/dynamic_length.py tests/test_config.py \
  tests/v1/spec_decode/test_dynamic_length.py
git commit -m "feat(spec-decode): add concurrency length policy"
```

## 4. Task 2：SchedulerOutput 与 step 盖章

**Files:**

- Modify: `vllm/v1/core/sched/output.py`
- Modify: `vllm/v1/core/sched/scheduler.py`
- Modify: `tests/v1/core/utils.py`
- Create: `tests/v1/core/test_dynamic_spec_scheduler.py`

- [ ] **Step 1: 写真实 decode 并发统计测试**

```python
def test_policy_counts_only_scheduled_decode_requests():
    scheduler = make_scheduler_with_dynamic_policy(
        rules=((1, 4, 8), (5, 8, 3)),
        running_decode_requests=5,
        waiting_requests=7,
        scheduled_prefill_requests=2,
    )
    output = scheduler.schedule()

    assert output.spec_decode_step_plan.decode_concurrency == 5
    assert output.spec_decode_step_plan.selected_mtp_length == 3
```

- [ ] **Step 2: 写单 step 唯一长度测试**

```python
def test_one_scheduler_output_has_one_selected_length():
    scheduler = make_scheduler_with_dynamic_policy(
        rules=((1, 8, 5),),
        running_decode_requests=8,
        waiting_requests=0,
        scheduled_prefill_requests=0,
    )
    output = scheduler.schedule()

    assert output.spec_decode_step_plan.selected_mtp_length == 5
    assert {
        output.spec_decode_step_plan.selected_mtp_length
        for request_id in output.num_scheduled_tokens
        if request_id in output.scheduled_spec_decode_tokens
    } == {5}
```

- [ ] **Step 3: 运行并确认失败**

```bash
pytest -q tests/v1/core/test_dynamic_spec_scheduler.py \
  -k "counts_only or one_scheduler_output"
```

Expected：FAIL，output 字段和 scheduler policy 尚未接入。

- [ ] **Step 4: 实现统计 helper**

在 scheduler 内新增纯 CPU 方法：

```python
def _count_scheduled_decode_requests(
    self,
    num_scheduled_tokens: dict[str, int],
) -> int:
    count = 0
    for request_id, num_tokens in num_scheduled_tokens.items():
        if num_tokens <= 0:
            continue
        request = self.requests[request_id]
        if request.is_prefill_chunk:
            continue
        count += 1
    return count
```

在 `SchedulerOutput` 完成现有调度字段后、返回前：

```python
if self.dynamic_spec_length_policy is not None:
    decode_concurrency = self._count_scheduled_decode_requests(
        scheduler_output.num_scheduled_tokens
    )
    if decode_concurrency > 0:
        scheduler_output.spec_decode_step_plan = (
            self.dynamic_spec_length_policy.select(decode_concurrency)
        )
```

- [ ] **Step 5: 验证固定路径**

```python
def test_fixed_scheduler_does_not_stamp_dynamic_plan():
    scheduler = make_fixed_spec_scheduler(num_speculative_tokens=3)
    output = scheduler.schedule()
    assert output.spec_decode_step_plan is None
```

- [ ] **Step 6: 运行 scheduler 测试**

```bash
pytest -q tests/v1/core/test_dynamic_spec_scheduler.py
pytest -q tests/v1/core/test_scheduler.py
```

Expected：PASS。

- [ ] **Step 7: 提交**

```bash
git add vllm/v1/core/sched/output.py vllm/v1/core/sched/scheduler.py \
  tests/v1/core/utils.py tests/v1/core/test_dynamic_spec_scheduler.py
git commit -m "feat(scheduler): stamp speculative step plans"
```

## 5. Task 3：升档、降档和 AsyncScheduler

**Files:**

- Modify: `vllm/v1/core/sched/scheduler.py`
- Modify: `vllm/v1/core/sched/async_scheduler.py`
- Modify: `tests/v1/core/test_dynamic_spec_scheduler.py`
- Modify: `tests/v1/core/test_async_scheduler.py`

- [ ] **Step 1: 写降档测试**

```python
def test_downshift_truncates_all_existing_drafts_uniformly():
    scheduler = make_decode_scheduler(
        request_draft_lengths=(8, 8, 8, 8),
        policy_length=3,
    )
    output = scheduler.schedule()

    assert {
        len(tokens)
        for tokens in output.scheduled_spec_decode_tokens.values()
    } == {3}
    assert output.spec_decode_step_plan.selected_mtp_length == 3
```

- [ ] **Step 2: 写升档测试**

```python
def test_upshift_does_not_expand_current_drafts():
    scheduler = make_decode_scheduler(
        request_draft_lengths=(3, 3, 3, 3),
        policy_length=8,
    )
    output = scheduler.schedule()

    assert {
        len(tokens)
        for tokens in output.scheduled_spec_decode_tokens.values()
    } == {3}
    assert output.spec_decode_step_plan.selected_mtp_length == 8
```

- [ ] **Step 3: 写 async FIFO 不可变测试**

```python
def test_async_queued_outputs_keep_their_own_step_plan():
    scheduler = make_async_scheduler_with_concurrency_sequence((8, 9))
    first = scheduler.schedule()
    second = scheduler.schedule()

    assert first.spec_decode_step_plan.decode_concurrency == 8
    assert first.spec_decode_step_plan.selected_mtp_length == 5
    assert second.spec_decode_step_plan.decode_concurrency == 9
    assert second.spec_decode_step_plan.selected_mtp_length == 3
```

- [ ] **Step 4: 运行并确认失败**

```bash
pytest -q tests/v1/core/test_dynamic_spec_scheduler.py -k "shift"
pytest -q tests/v1/core/test_async_scheduler.py -k "step_plan"
```

Expected：FAIL。

- [ ] **Step 5: 实现统一截断**

选择 plan 后，对当前 output 中所有 draft：

```python
def _truncate_scheduled_drafts(
    scheduled_spec_decode_tokens: dict[str, list[int]],
    selected_length: int,
) -> None:
    for request_id, draft_token_ids in (
        scheduled_spec_decode_tokens.items()
    ):
        if len(draft_token_ids) > selected_length:
            scheduled_spec_decode_tokens[request_id] = (
                draft_token_ids[:selected_length]
            )
```

不能在升档时添加 placeholder 到当前 verification 列表。

- [ ] **Step 6: 实现 placeholder bank**

AsyncScheduler 初始化：

```python
self._spec_token_placeholder_bank = {
    length: [-1] * length
    for length in dynamic_config.candidate_lengths
}
```

`_update_after_schedule()`：

```python
plan = scheduler_output.spec_decode_step_plan
next_draft_length = (
    self.num_spec_tokens
    if plan is None
    else plan.selected_mtp_length
)
request.spec_token_ids = self._spec_token_placeholder_bank[
    next_draft_length
]
```

`request.num_output_placeholders` 继续使用当前实际 scheduled draft 数，不能改成
`next_draft_length`。

- [ ] **Step 7: 运行回归**

```bash
pytest -q tests/v1/core/test_dynamic_spec_scheduler.py
pytest -q tests/v1/core/test_async_scheduler.py
pytest -q tests/v1/core/test_scheduler.py
```

Expected：PASS。

- [ ] **Step 8: 提交**

```bash
git add vllm/v1/core/sched/scheduler.py \
  vllm/v1/core/sched/async_scheduler.py \
  tests/v1/core/test_dynamic_spec_scheduler.py \
  tests/v1/core/test_async_scheduler.py
git commit -m "feat(scheduler): switch speculative length per step"
```

## 6. Task 4：通用多 query-length 图键

**Files:**

- Modify: `vllm/forward_context.py`
- Modify: `vllm/v1/cudagraph_dispatcher.py`
- Modify: `vllm/config/compilation.py`
- Modify: `tests/v1/cudagraph/test_cudagraph_dispatch.py`

- [ ] **Step 1: 写碰撞测试**

```python
def test_uniform_graph_keys_include_query_length():
    mtp1 = BatchDescriptor(
        num_tokens=18,
        num_reqs=9,
        uniform=True,
        uniform_query_len=2,
    )
    mtp3 = BatchDescriptor(
        num_tokens=18,
        num_reqs=4,
        uniform=True,
        uniform_query_len=4,
    )
    assert mtp1 != mtp3
```

- [ ] **Step 2: 写多长度 dispatch 测试**

```python
@pytest.mark.parametrize(
    ("spec_length", "query_length"),
    [(1, 2), (3, 4), (5, 6), (8, 9)],
)
def test_dispatches_prepared_uniform_query_lengths(
    spec_length, query_length
):
    dispatcher = make_dispatcher(
        speculative_lengths=(1, 3, 5, 8)
    )
    descriptor = dispatcher.dispatch(
        num_tokens=8 * query_length,
        num_reqs=8,
        uniform_query_len=query_length,
    )
    assert descriptor.uniform_query_len == query_length
```

- [ ] **Step 3: 运行并确认失败**

```bash
pytest -q tests/v1/cudagraph/test_cudagraph_dispatch.py \
  -k "query_length or prepared_uniform"
```

Expected：FAIL。

- [ ] **Step 4: 扩展 BatchDescriptor**

```python
uniform_query_len: int | None = None
```

构造时确保：

```python
if uniform and uniform_query_len is None:
    raise ValueError("uniform batches require uniform_query_len")
if not uniform and uniform_query_len is not None:
    raise ValueError("non-uniform batches cannot set uniform_query_len")
```

- [ ] **Step 5: 扩展 dispatcher**

将固定 `uniform_decode_query_len` 改为启动期 tuple，并在 capture、padding map 和
dispatch 中显式传递 query length。动态关闭时 tuple 只包含现有固定值。

- [ ] **Step 6: 运行测试**

```bash
pytest -q tests/v1/cudagraph/test_cudagraph_dispatch.py
pytest -q tests/v1/cudagraph
```

Expected：PASS。

- [ ] **Step 7: 提交**

```bash
git add vllm/forward_context.py vllm/v1/cudagraph_dispatcher.py \
  vllm/config/compilation.py \
  tests/v1/cudagraph/test_cudagraph_dispatch.py
git commit -m "feat(cudagraph): key decode graphs by query length"
```

## 7. Task 5：Ascend execution registry 与 ACL Graph 复合键

**Files:**

- Create: `vllm_ascend/spec_decode/dynamic_length.py`
- Modify: `vllm_ascend/compilation/acl_graph.py`
- Modify: `vllm_ascend/compilation/compiler_interface.py`
- Modify: `vllm_ascend/patch/worker/patch_cudagraph.py`
- Create: `tests/ut/compilation/test_dynamic_acl_graph.py`
- Create: `tests/ut/spec_decode/test_dynamic_length.py`

- [ ] **Step 1: 写 registry 测试**

```python
def test_registry_builds_one_plan_per_candidate():
    registry = AscendSpecExecutionPlanRegistry((1, 3, 5, 8))
    assert [registry.get(k).draft_steps for k in (1, 3, 5, 8)] == [
        1, 3, 5, 8
    ]
    assert [
        registry.get(k).target_uniform_query_len
        for k in (1, 3, 5, 8)
    ] == [2, 4, 6, 9]
```

- [ ] **Step 2: 写 GraphParamKey 测试**

```python
def test_graph_params_do_not_collide_on_equal_num_tokens():
    key_mtp1 = GraphParamKey(num_tokens=18, uniform_query_len=2)
    key_mtp3 = GraphParamKey(num_tokens=18, uniform_query_len=4)
    assert key_mtp1 != key_mtp3
```

- [ ] **Step 3: 运行并确认失败**

```bash
cd /Users/linyi/code/Documents/code/vllm-ascend
pytest -q tests/ut/spec_decode/test_dynamic_length.py
pytest -q tests/ut/compilation/test_dynamic_acl_graph.py
```

Expected：FAIL。

- [ ] **Step 4: 实现 registry**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class AscendSpecExecutionPlan:
    spec_length: int
    target_uniform_query_len: int
    draft_steps: int


class AscendSpecExecutionPlanRegistry:
    def __init__(self, candidate_lengths: tuple[int, ...]) -> None:
        self._plans = {
            length: AscendSpecExecutionPlan(
                spec_length=length,
                target_uniform_query_len=length + 1,
                draft_steps=length,
            )
            for length in candidate_lengths
        }

    def get(self, length: int) -> AscendSpecExecutionPlan:
        try:
            return self._plans[length]
        except KeyError as exc:
            raise RuntimeError(
                f"speculative length {length} was not prepared"
            ) from exc
```

- [ ] **Step 5: 实现复合参数键**

```python
@dataclass(frozen=True)
class GraphParamKey:
    num_tokens: int
    uniform_query_len: int | None
```

将 `_graph_params` 和 `_draft_graph_params` 的创建、更新、replay 查询统一改为该 key。
不允许一处仍按 `num_tokens` 查询。

- [ ] **Step 6: 启动期准备候选图**

compiler interface 从动态配置读取：

```python
target_query_lengths = tuple(
    length + 1 for length in candidate_lengths
)
```

严格模式下任一候选图准备失败即启动失败。

- [ ] **Step 7: 运行测试**

```bash
pytest -q tests/ut/spec_decode/test_dynamic_length.py
pytest -q tests/ut/compilation/test_dynamic_acl_graph.py
pytest -q tests/ut/compilation
```

Expected：PASS。

- [ ] **Step 8: 提交**

```bash
git add vllm_ascend/spec_decode/dynamic_length.py \
  vllm_ascend/compilation/acl_graph.py \
  vllm_ascend/compilation/compiler_interface.py \
  vllm_ascend/patch/worker/patch_cudagraph.py \
  tests/ut/spec_decode/test_dynamic_length.py \
  tests/ut/compilation/test_dynamic_acl_graph.py
git commit -m "feat(ascend): prepare dynamic speculative graph plans"
```

## 8. Task 6：NPUModelRunner target/draft 路由

**Files:**

- Modify: `vllm_ascend/worker/model_runner_v1.py`
- Modify: `tests/ut/worker/a2/test_model_runner_v1.py`

- [ ] **Step 1: 写 transition 路由测试**

```python
def test_runner_routes_verify3_and_draft8_independently():
    runner = make_dynamic_runner(candidate_lengths=(1, 3, 5, 8))
    output = make_scheduler_output(
        request_draft_lengths=(3, 3),
        selected_mtp_length=8,
    )
    runner.execute_model(output)

    runner.target_dispatcher.assert_called_once_with(
        uniform_query_len=4
    )
    runner.drafter.propose.assert_called_once_with(
        active_spec_length=8
    )
```

- [ ] **Step 2: 写降档路由测试**

```python
def test_runner_routes_verify3_and_draft3_after_truncation():
    runner = make_dynamic_runner(candidate_lengths=(1, 3, 5, 8))
    output = make_scheduler_output(
        request_draft_lengths=(3, 3),
        selected_mtp_length=3,
    )
    runner.execute_model(output)

    runner.target_dispatcher.assert_called_once_with(
        uniform_query_len=4
    )
    runner.drafter.propose.assert_called_once_with(
        active_spec_length=3
    )
```

- [ ] **Step 3: 运行并确认失败**

```bash
pytest -q tests/ut/worker/a2/test_model_runner_v1.py \
  -k "routes_verify"
```

Expected：FAIL。

- [ ] **Step 4: 实现 plan 读取**

```python
step_plan = scheduler_output.spec_decode_step_plan
active_spec_length = (
    self.num_spec_tokens
    if step_plan is None
    else step_plan.selected_mtp_length
)
execution_plan = (
    None
    if self.dynamic_spec_plan_registry is None
    else self.dynamic_spec_plan_registry.get(active_spec_length)
)
```

target query length必须从实际 `scheduled_spec_decode_tokens` 推导，不能从
`active_spec_length` 推断。

- [ ] **Step 5: 保持最大 stride**

所有前一批 draft 索引：

```python
start = prev_index * self.num_spec_tokens
```

这里 `self.num_spec_tokens` 是最大容量，不能替换为当前 active length。

- [ ] **Step 6: 运行测试**

```bash
pytest -q tests/ut/worker/a2/test_model_runner_v1.py \
  -k "spec or draft or routes"
```

Expected：PASS。

- [ ] **Step 7: 提交**

```bash
git add vllm_ascend/worker/model_runner_v1.py \
  tests/ut/worker/a2/test_model_runner_v1.py
git commit -m "feat(ascend): route target and draft plans per step"
```

## 9. Task 7：MTP proposer 准确执行 K 步

**Files:**

- Modify: `vllm_ascend/spec_decode/llm_base_proposer.py`
- Modify: `tests/ut/spec_decode/test_dynamic_length.py`

- [ ] **Step 1: 写准确 forward 次数测试**

```python
@pytest.mark.parametrize("active_length", (1, 3, 5, 8))
def test_proposer_runs_exact_active_number_of_steps(active_length):
    proposer = make_mock_mtp_proposer(max_spec_length=8)
    proposer.propose(
        valid_sampled_token_ids=make_sampled_tokens(batch_size=2),
        active_spec_length=active_length,
    )
    assert proposer.model.forward.call_count == active_length
```

- [ ] **Step 2: 写 buffer 容量不变测试**

```python
def test_proposer_keeps_maximum_physical_width():
    proposer = make_mock_mtp_proposer(max_spec_length=8)
    proposer.propose(
        valid_sampled_token_ids=make_sampled_tokens(batch_size=2),
        active_spec_length=3,
    )
    assert proposer.draft_token_ids_tensor.shape[0] == 8
```

- [ ] **Step 3: 运行并确认失败**

```bash
pytest -q tests/ut/spec_decode/test_dynamic_length.py \
  -k "proposer"
```

Expected：FAIL。

- [ ] **Step 4: 显式传递 active length**

proposer 入口增加关键字参数：

```python
def propose(
    self,
    valid_sampled_token_ids: torch.Tensor,
    *,
    active_spec_length: int,
) -> torch.Tensor:
    if not 1 <= active_spec_length <= self.num_speculative_tokens:
        raise ValueError("active_spec_length is outside prepared capacity")
    return self._propose(
        valid_sampled_token_ids,
        active_spec_length=active_spec_length,
    )
```

所有循环、metadata list、token indices 和有效输出切片使用
`active_spec_length`；分配和 stride 使用 `self.num_speculative_tokens`。

- [ ] **Step 5: 禁止最大长度伪执行**

测试中为 model forward 加计数；K=3 时必须为固定 MTP3 的次数，不能为 8。

- [ ] **Step 6: 运行测试**

```bash
pytest -q tests/ut/spec_decode/test_dynamic_length.py
pytest -q tests/ut/spec_decode
```

Expected：PASS。

- [ ] **Step 7: 提交**

```bash
git add vllm_ascend/spec_decode/llm_base_proposer.py \
  tests/ut/spec_decode/test_dynamic_length.py
git commit -m "feat(ascend): execute active MTP draft length"
```

## 10. Task 8：Attention、CP 与拒绝采样兼容

**Files:**

- Modify: `vllm_ascend/attention/context_parallel/attention_cp.py`
- Modify: graph param 访问相关 attention/ops 文件
- Modify: `tests/ut/attention/a2/test_attention_cp_precision.py`
- Modify: `tests/ut/worker/a2/test_model_runner_v1.py`

- [ ] **Step 1: 写实际 query length 测试**

```python
def test_cp_metadata_uses_query_start_locations():
    builder = make_cp_builder(decode_threshold=9)
    metadata = builder.build(
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        num_decodes=2,
    )
    assert metadata.actual_seq_lengths_q == [4, 8]
```

- [ ] **Step 2: 写拒绝采样实际 draft 测试**

```python
def test_rejection_sampler_uses_incoming_draft_count():
    runner = make_dynamic_runner(candidate_lengths=(1, 3, 5, 8))
    output = make_scheduler_output(
        request_draft_lengths=(3, 3),
        selected_mtp_length=8,
    )
    runner.execute_model(output)
    assert runner.rejection_sampler.last_num_draft_tokens == [3, 3]
```

- [ ] **Step 3: 运行并确认失败**

```bash
pytest -q tests/ut/attention/a2/test_attention_cp_precision.py \
  -k "query_start"
pytest -q tests/ut/worker/a2/test_model_runner_v1.py \
  -k "incoming_draft"
```

Expected：至少 CP 测试失败。

- [ ] **Step 4: 修复 CP metadata**

用 `query_start_loc` 的 cumulative 值构造实际序列长度，不使用
`decode_threshold * (i + 1)`。

- [ ] **Step 5: 保持拒绝采样调用**

`num_draft_tokens` 继续来自当前实际 scheduler output。禁止使用
`selected_mtp_length` 覆盖它。standard rejection sampling 的函数签名、公式和调用
顺序不改变。

- [ ] **Step 6: 静态扫描同步**

```bash
rg -n "\\.item\\(|npu\\.synchronize|event\\.synchronize|wait_stream" \
  vllm_ascend/worker/model_runner_v1.py \
  vllm_ascend/spec_decode \
  vllm_ascend/attention
```

Expected：新增 diff 中没有动态策略导致的新同步调用。

- [ ] **Step 7: 运行测试并提交**

```bash
pytest -q tests/ut/attention/a2/test_attention_cp_precision.py
pytest -q tests/ut/worker/a2/test_model_runner_v1.py -k "spec or draft"
git add vllm_ascend/attention vllm_ascend/worker/model_runner_v1.py \
  tests/ut/attention/a2/test_attention_cp_precision.py \
  tests/ut/worker/a2/test_model_runner_v1.py
git commit -m "fix(ascend): use actual speculative query metadata"
```

## 11. Task 9：能力门禁、并行与观测

**Files:**

- Modify: `vllm/platforms/interface.py`
- Modify: `vllm_ascend/platform.py`
- Modify: `vllm/v1/spec_decode/metrics.py`
- Modify: `vllm/v1/metrics/stats.py`
- Create: `tests/platforms/test_dynamic_spec_capability.py`
- Modify: `tests/v1/metrics/test_stats.py`
- Modify: `vllm-ascend/tests/ut/test_platform.py`

- [ ] **Step 1: 写 unsupported backend 测试**

```python
def test_dynamic_spec_requires_step_plan_capability():
    platform = FakePlatform(
        supports_dynamic_spec_step_plan=False
    )
    with pytest.raises(NotImplementedError, match="step plan"):
        validate_dynamic_spec_platform(platform)
```

- [ ] **Step 2: 写共享 collective 拒绝测试**

```python
def test_rejects_independent_schedulers_sharing_draft_collective():
    topology = FakeTopology(
        independent_scheduler_count=2,
        shared_mtp_collective=True,
    )
    with pytest.raises(NotImplementedError, match="shared MTP collective"):
        validate_dynamic_spec_topology(topology)
```

- [ ] **Step 3: 实现能力接口**

通用平台默认：

```python
@classmethod
def supports_dynamic_spec_step_plan(cls) -> bool:
    return False
```

Ascend 仅在已验证的 V1 MTP、支持图模式和安全并行拓扑返回 `True`。

- [ ] **Step 4: 增加低开销指标**

至少：

```text
dynamic_spec_policy_hit_total{length}
dynamic_spec_length_switch_total{from,to}
dynamic_spec_decode_concurrency
dynamic_spec_selected_length
dynamic_spec_graph_dispatch_total{kind,length,result}
dynamic_spec_plan_mismatch_total
```

指标更新不读取设备状态，不逐 step 输出日志。

- [ ] **Step 5: 运行 vLLM 测试并提交**

```bash
cd /Users/linyi/code/Documents/code/vllm
pytest -q tests/platforms/test_dynamic_spec_capability.py
pytest -q tests/v1/metrics/test_stats.py -k "dynamic_spec"
git add vllm/platforms/interface.py vllm/v1/spec_decode/metrics.py \
  vllm/v1/metrics/stats.py \
  tests/platforms/test_dynamic_spec_capability.py \
  tests/v1/metrics/test_stats.py
git commit -m "feat(spec-decode): expose dynamic step metrics"
```

- [ ] **Step 6: 运行 Ascend 能力测试并提交**

```bash
cd /Users/linyi/code/Documents/code/vllm-ascend
pytest -q tests/ut/test_platform.py -k "dynamic_spec"
git add vllm_ascend/platform.py tests/ut/test_platform.py
git commit -m "feat(ascend): gate dynamic speculative step plans"
```

## 12. Task 10：端到端精度

**Files:**

- Create:
  `vllm-ascend/tests/e2e/pull_request/one_card/spec_decode/test_dynamic_mtp.py`

- [ ] **Step 1: 建立 fixed/dynamic greedy 对照**

测试参数：

```python
@pytest.mark.parametrize(
    ("concurrency", "expected_length"),
    [(4, 8), (8, 5), (16, 3), (32, 1)],
)
def test_dynamic_matches_fixed_mtp(
    concurrency, expected_length, model_path
):
    prompts = build_prompt_batch(concurrency)
    fixed = run_fixed_mtp(
        model_path=model_path,
        prompts=prompts,
        speculative_length=expected_length,
        temperature=0,
    )
    dynamic = run_dynamic_mtp(
        model_path=model_path,
        prompts=prompts,
        rules=((1, 4, 8), (5, 8, 5), (9, 16, 3), (17, 64, 1)),
        temperature=0,
    )
    assert [item.token_ids for item in dynamic] == [
        item.token_ids for item in fixed
    ]
```

- [ ] **Step 2: 建立连续切换测试**

按负载阶段制造：

```text
K sequence = 8 -> 5 -> 3 -> 1 -> 3 -> 5 -> 8
```

与非投机 target baseline 比较完整 token ids。

- [ ] **Step 3: 覆盖边界**

- EOS；
- stop string；
- prefix cache hit/miss；
- short/medium/long context；
- 全接受、全拒绝、混合接受；
- preemption 和 abort；
- async scheduling 开/关。

- [ ] **Step 4: 运行**

```bash
pytest -q \
  tests/e2e/pull_request/one_card/spec_decode/test_dynamic_mtp.py \
  -s
```

Expected：全部 greedy token ids 一致，无 crash、hang、online capture。

- [ ] **Step 5: 提交**

```bash
git add tests/e2e/pull_request/one_card/spec_decode/test_dynamic_mtp.py
git commit -m "test(ascend): validate dynamic MTP precision"
```

## 13. Task 11：性能、抖动和长稳

**Files:**

- Create:
  `vllm/benchmarks/overheads/benchmark_dynamic_spec_policy.py`
- Create:
  `vllm-ascend/benchmarks/scripts/benchmark_dynamic_spec_length.py`
- Create:
  `vllm-ascend/benchmarks/tests/dynamic-spec-length-tests.json`

- [ ] **Step 1: CPU policy microbenchmark**

分别运行 100 万次固定路径和动态 LUT，记录 ns/op、对象分配和锁等待。

验收：

```text
select() 不分配 Python 对象
select() 不获取锁
无设备调用
```

- [ ] **Step 2: 稳态 fixed/dynamic 矩阵**

每个点使用相同请求：

| 并发 | 动态 K | 固定对照 |
|---:|---:|---:|
| 4 | 8 | MTP8 |
| 8 | 5 | MTP5 |
| 16 | 3 | MTP3 |
| 32 | 1 | MTP1 |

采集：

- TPOT p50/p90/p99；
- throughput；
- target/draft 分解耗时；
- proposer forward 次数；
- graph hit；
- host wait、D2H scalar、device synchronize；
- CPU allocation。

- [ ] **Step 3: 性能门禁**

```text
TPOT p50/p90 degradation <= 1%
TPOT p99 degradation <= 2%
throughput degradation <= 1%
FULL graph hit rate 不下降
proposer forward 次数与 fixed K 相等
无新增 D2H scalar 和 device synchronize
```

- [ ] **Step 4: 阈值振荡**

构造：

```text
8 <-> 9 concurrency
16 <-> 17 concurrency
```

记录 switch ratio、transition step TPOT 和稳态恢复时间。该测试用于量化，不在第一
阶段增加迟滞。

- [ ] **Step 5: 长稳**

```text
CI: 24 hours
release qualification: 72 hours
length switches: >= 10,000
```

验收：

- 无 crash/hang/错 token；
- 无运行期 graph capture；
- host/NPU 内存达到平台；
- 无日志洪泛；
- graph registry 数量不增长。

- [ ] **Step 6: 提交 benchmark**

```bash
git add benchmarks
git commit -m "bench(ascend): compare fixed and dynamic MTP"
```

## 14. Task 12：最终回归和交付

- [ ] **Step 1: vLLM CPU 回归**

```bash
cd /Users/linyi/code/Documents/code/vllm
pytest -q tests/test_config.py
pytest -q tests/v1/spec_decode
pytest -q tests/v1/core/test_dynamic_spec_scheduler.py
pytest -q tests/v1/core/test_async_scheduler.py
pytest -q tests/v1/cudagraph
```

- [ ] **Step 2: vllm-ascend 单元回归**

```bash
cd /Users/linyi/code/Documents/code/vllm-ascend
pytest -q tests/ut/spec_decode
pytest -q tests/ut/compilation
pytest -q tests/ut/worker/a2/test_model_runner_v1.py
pytest -q tests/ut/attention/a2/test_attention_cp_precision.py
```

- [ ] **Step 3: 固定路径回归**

分别启动固定 MTP1、MTP3、MTP5、MTP8，确认动态配置关闭时：

- `SchedulerOutput.spec_decode_step_plan is None`；
- 图键、proposer 次数、拒绝采样和 TPOT 未变化；
- 不创建 dynamic policy 和 placeholder bank。

- [ ] **Step 4: profiler 复核**

对 fixed MTP3 与 dynamic(C→3) 比较 timeline：

- scheduler 到 worker 没有新同步 RPC；
- accepted-token correction 位置不变；
- 无新增 stream wait；
- target/draft graph 与固定 MTP3 相同；
- proposer 只运行 3 步。

- [ ] **Step 5: 文档**

更新用户文档：

- 配置示例；
- 并发定义；
- step 级统一语义；
- 升档一拍延迟；
- 候选图启动成本；
- 不支持拓扑；
- 性能对照方法。

- [ ] **Step 6: 最终检查**

```bash
git -C /Users/linyi/code/Documents/code/vllm status --short
git -C /Users/linyi/code/Documents/code/vllm-ascend status --short
git -C /Users/linyi/code/Documents/code/vllm log --oneline -12
git -C /Users/linyi/code/Documents/code/vllm-ascend log --oneline -12
```

Expected：只包含本功能预期文件，提交粒度与 Task 对齐。

## 15. 实现不变量检查表

- [ ] 动态长度是 step 属性，不是请求属性。
- [ ] 同一 output 只有一个 `selected_mtp_length`。
- [ ] 不存在 `_spec_plan_version_by_req` 或等价 cohort map。
- [ ] 不等待所有活跃请求同时被调度。
- [ ] 不排空 async queue。
- [ ] worker 不读取 scheduler 全局 active length。
- [ ] 升档不扩展当前 draft。
- [ ] 降档只统一截断当前 draft。
- [ ] verification 使用实际 incoming draft。
- [ ] proposer 使用 output 的 selected K。
- [ ] K=3 不执行 K=8 的多余前向。
- [ ] 最大容量和 stride 不随 step 改变。
- [ ] 图在启动期准备，运行期不 capture。
- [ ] standard rejection sampling 未修改。
- [ ] 无新增 CPU-NPU 同步。
- [ ] fixed 模式不创建动态热路径对象。
- [ ] dynamic(C→K) 与 fixed K 达到精度和性能门禁。

## 16. 风险与停止条件

| 风险 | 处理 |
|---|---|
| target/draft 不能独立 replay | 预捕获可达 `(verify K, draft K)` 组合 |
| 候选图内存过大 | 减少候选集合，不允许运行期 eager |
| DP+EP shared collective 长度不一致 | 能力门禁拒绝，不新增逐 step barrier |
| 动态 K=3 比 fixed K=3 多执行算子 | 停止准入，修复路由后再测 |
| profiler 出现新增同步 | 停止准入，定位并移除 |
| greedy 输出不一致 | 停止性能测试，先修复正确性 |
| 边界振荡收益为负 | 第一阶段记录；第二阶段设计迟滞/成本模型 |

## 17. 完成定义

只有同时满足以下结果才完成第一阶段：

1. 配置支持多并发范围和多候选长度。
2. 每个 step 的 decode 请求使用统一策略长度。
3. 请求可跨 step 改变长度且无 cohort。
4. async scheduling、FULL graph、拒绝采样和通信路径正常。
5. dynamic(C→K) 与 fixed K greedy 输出一致。
6. 稳态 TPOT/throughput 满足门禁。
7. profiler 证明无新增 CPU-NPU 同步。
8. 24/72 小时长稳通过。
9. 动态关闭时固定路径无回归。
