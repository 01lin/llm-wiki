# vLLM / vLLM Ascend 动态投机长度第一阶段 Implementation Plan

> **修订状态：已由
> `20260623-120055-vllm-vllm-ascend-dynamic-speculative-length-step-policy-phase1-实现计划.md`
> 替代。** 新计划不再实现请求 cohort、全体请求安全点和以手工切换为主的状态机。
> 本文保留为源码落点与历史风险分析参考，不应直接执行。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不重启推理服务的情况下，按调度批次在预声明的 MTP 候选长度之间切换，同时保持 vLLM async scheduling、Ascend FULL ACL Graph、既有设备侧 accepted-token 校正和通信快速路径，并使动态稳态性能与相同固定长度基线持平。

**Architecture:** `num_speculative_tokens` 继续表示最大资源容量；新增 CPU-only 长度控制器和随 `SchedulerOutput` 下发的不可变批次计划戳。vLLM 通用层负责配置、状态机、调度安全点和多 query-length 图键，vllm-ascend 负责候选 NPU 执行计划、ACL Graph 复合缓存键、target/draft 路由和 attention 元数据。切换只改变下一轮 draft 长度，当前 verification 始终依据实际 scheduled draft token。

**Tech Stack:** Python 3.10+、Pydantic/dataclass/msgspec、vLLM V1 Scheduler/AsyncLLM、PyTorch、torch-npu、Ascend ACL Graph、pytest、vLLM benchmark serving、NPU profiler。

---

## 0. 基线、前置条件与实施边界

### 0.1 代码基线

- vLLM：`0d29612292c6b1e312af42ac00cf649af16a438b`
- vllm-ascend：`8afdf356f6a2496bedfc538253366ef1a8c0d9aa`
- 设计规格：
  `/Users/linyi/code/Documents/obsidian_wiki/llm-wikid/raw/Infra/20260623-010010-vllm-vllm-ascend-dynamic-speculative-length-phase1-方案.md`
- 设计规格 SHA-256：
  `0cf1f4a958b7bb4b4a55df374cda7714143ab14a023ae1bc0ebf48c195253442`

所有行号均以以上 commit 为准。开始实现前若仓库基线变化，应先重新运行本计划中的定位命令并更新行号，不能直接套用补丁。

### 0.2 第一阶段运行配置

首个生产准入组合：

```json
{
  "method": "mtp",
  "num_speculative_tokens": 8,
  "dynamic_speculative_length": {
    "candidate_lengths": [1, 3, 8],
    "initial_length": 3,
    "strict_graph_mode": true
  },
  "disable_padded_drafter_batch": false,
  "rejection_sample_method": "standard"
}
```

模型与环境：

- DeepSeek V4 Flash + MTP。
- Ascend 910B3 / 910C。
- V1 model runner。
- async scheduling 开启。
- `FULL_DECODE_ONLY`。
- greedy/temperature=0 作为严格精度门禁。

### 0.3 明确不做

- 不在运行中扩大候选集合或最大长度。
- 不进行在线 ACL Graph 捕获。
- 不实现请求粒度策略。
- 不把第二阶段自动寻优逻辑耦合进 scheduler 或 model runner。
- 不修改 standard rejection sampling。
- 不以始终执行最大 MTP8 再裁剪的方式伪装 MTP1/MTP3。
- 不为 PCP/DCP 做未经 profiler 验证的“无同步”生产声明。

### 0.4 对设计规格的实现细化

设计规格把“没有更早版本的 SchedulerOutput 等待提交”列为保守安全点条件。
实现阶段为保留 async batch queue 的遮掩效果，不主动排空已经按 FIFO
顺序提交的旧批次。安全性由以下条件共同保证：

1. 每个 `SchedulerOutput` 携带不可变版本戳。
2. executor 与 batch queue 按提交顺序执行和回收。
3. worker 只能读取当前 output 的计划，不能读取 scheduler 的新全局状态。
4. transition output 之前的旧 output 按旧计划完成；之后不再产生旧版本 output。
5. 若任何 executor backend 不能保证 FIFO，则该 backend 在能力门禁中返回不支持。

因此切换时不增加一次强制 queue drain，也不会破坏稳态和切换附近的异步重叠。

## 1. 关键技术结论

### 1.1 运行时长度分成两个概念

```text
verify length = len(scheduled_spec_decode_tokens[request_id])
next draft length = scheduler_output.spec_decode_batch_plan.next_draft_length
```

前者是当前批已存在的事实；后者是本批 target 完成后要执行的 proposer 计划。两者在切换批可以不同。

### 1.2 计划对象必须随 SchedulerOutput 有序下发

禁止 worker 读取一个可热修改的全局长度。否则已经排队的异步批会读取新值，出现旧 draft + 新 verify 假设的竞态。

`SpecDecodeBatchPlan` 在版本不变期间复用同一只读对象，不在每批创建新的控制对象。

### 1.3 最大长度只负责容量

以下容量始终按 `max(candidate_lengths)` 准备：

- KV lookahead。
- `_draft_token_ids` 的物理行宽。
- proposer step buffer。
- position、slot mapping、seq lens、query start location buffer。
- sampler 输出。
- MC2/FlashComm workspace 上界。

运行期只切换有效前缀、循环上限和图键。

### 1.4 FULL Graph 需要按 query length 分开

候选长度 `{1, 3, 8}` 对应 target uniform query length `{2, 4, 9}`。

`BatchDescriptor(num_tokens=36, uniform=True)` 无法区分：

- 18 个请求 × query length 2。
- 9 个请求 × query length 4。
- 4 个请求 × query length 9。

因此图键必须包含 `uniform_query_len`。

### 1.5 Ascend GraphParams 也必须采用复合键

当前 `ACLGraphWrapper` 使用 `BatchDescriptor` 缓存图，但 `_graph_params`、`_draft_graph_params` 仍按 `num_tokens` 保存 event、workspace、handle 和 attention 参数。

动态长度下相同 `num_tokens` 可以对应不同图。只扩展 `BatchDescriptor` 而不扩展 GraphParams，会造成图与更新参数错配。

正式键定义：

```python
@dataclass(frozen=True)
class GraphParamKey:
    num_tokens: int
    uniform_query_len: int | None
```

非 uniform batch 使用 `uniform_query_len=None`。

## 2. 文件结构与职责

### 2.1 vLLM 新增文件

| 文件 | 职责 |
|---|---|
| `vllm/v1/spec_decode/dynamic_length.py` | 运行时状态、批次计划、控制器；纯 CPU、无设备依赖 |
| `tests/v1/spec_decode/test_dynamic_length.py` | 控制器状态机、版本、幂等和失败行为 |
| `tests/v1/core/test_dynamic_spec_scheduler.py` | scheduler/async scheduler 切换与安全点 |
| `benchmarks/overheads/benchmark_dynamic_spec_length.py` | controller 稳态 CPU 开销与分配数 |

### 2.2 vLLM 修改文件

| 文件 | 修改内容 |
|---|---|
| `vllm/config/speculative.py` | 启动配置、校验、候选长度辅助属性、hash |
| `vllm/config/__init__.py` | 导出动态配置类型 |
| `vllm/config/compilation.py` | 多 query-length capture size 归一化 |
| `vllm/forward_context.py` | `BatchDescriptor.uniform_query_len` |
| `vllm/v1/cudagraph_dispatcher.py` | 多长度 key 初始化、分长度 padding map、dispatch |
| `vllm/v1/core/sched/output.py` | `SchedulerOutput.spec_decode_batch_plan` |
| `vllm/v1/core/sched/scheduler.py` | controller、计划盖章、安全点、请求 cohort 版本 |
| `vllm/v1/core/sched/async_scheduler.py` | placeholder bank 与动态模板选择 |
| `vllm/v1/engine/core.py` | prepare/commit/get utility 方法 |
| `vllm/v1/engine/core_client.py` | sync/async/DP 控制 API |
| `vllm/v1/engine/async_llm.py` | 用户可调用的异步 API |
| `vllm/v1/engine/llm_engine.py` | 同步 LLMEngine API |
| `vllm/entrypoints/llm.py` | `LLM` 同步控制 API |
| `vllm/platforms/interface.py` | 默认关闭动态投机长度执行能力 |
| `tests/test_config.py` | 配置合法性与兼容性 |
| `tests/v1/cudagraph/test_cudagraph_dispatch.py` | 多长度图键与 padding |
| `tests/v1/engine/test_async_llm.py` | API 透传 |
| `tests/v1/engine/test_llm_engine.py` | 同步 API 透传 |
| `tests/v1/distributed/test_async_llm_dp.py` | DP prepare/commit |

### 2.3 vllm-ascend 新增文件

| 文件 | 职责 |
|---|---|
| `vllm_ascend/spec_decode/dynamic_length.py` | NPU 候选执行计划注册表 |
| `tests/ut/spec_decode/test_dynamic_length.py` | plan registry 与 proposer active length |
| `tests/ut/compilation/test_dynamic_acl_graph.py` | ACL Graph 与 GraphParams 复合键 |
| `tests/e2e/pull_request/one_card/spec_decode/test_dynamic_mtp.py` | 单卡动态切换精度和图命中 |
| `benchmarks/scripts/benchmark_dynamic_spec_length.py` | 固定/动态稳态和切换性能采集 |
| `benchmarks/tests/dynamic-spec-length-tests.json` | 并发、上下文、输出长度矩阵 |

### 2.4 vllm-ascend 修改文件

| 文件 | 修改内容 |
|---|---|
| `vllm_ascend/worker/model_runner_v1.py` | target plan 路由、active query length、最大 stride |
| `vllm_ascend/platform.py` | 声明 Ascend V1 MTP 动态长度能力 |
| `vllm_ascend/spec_decode/llm_base_proposer.py` | active draft length、候选图、buffer 前缀 |
| `vllm_ascend/compilation/acl_graph.py` | `GraphParamKey` 与复合参数仓 |
| `vllm_ascend/compilation/compiler_interface.py` | 多长度 decode capture size |
| `vllm_ascend/patch/worker/patch_cudagraph.py` | patch 签名与新图键兼容 |
| `vllm_ascend/attention/context_parallel/attention_cp.py` | 使用实际 query_start_loc |
| `vllm_ascend/attention/attention_v1.py` | GraphParams 复合 key 访问 |
| `vllm_ascend/attention/mla_v1.py` | GraphParams 复合 key 访问 |
| `vllm_ascend/attention/context_parallel/mla_cp.py` | GraphParams 复合 key 访问 |
| `vllm_ascend/ops/gdn.py` | GraphParams 复合 key 访问 |
| `vllm_ascend/_310p/ops/fla/gdn_310.py` | 保持接口一致；310P 不列入首期生产准入 |
| `tests/ut/worker/a2/test_model_runner_v1.py` | target routing、placeholder、stride |
| `tests/ut/attention/a2/test_attention_cp_precision.py` | 实际 query length 精度 |

## 3. 类型与接口契约

### 3.1 配置类型

```python
@config
class DynamicSpeculativeLengthConfig:
    candidate_lengths: tuple[int, ...]
    initial_length: int
    strict_graph_mode: bool = True

    @model_validator(mode="after")
    def _verify_args(self) -> Self:
        if not self.candidate_lengths:
            raise ValueError("candidate_lengths must not be empty")
        if any(length <= 0 for length in self.candidate_lengths):
            raise ValueError("candidate_lengths must contain positive integers")
        if tuple(sorted(set(self.candidate_lengths))) != self.candidate_lengths:
            raise ValueError(
                "candidate_lengths must be strictly increasing and unique"
            )
        if self.initial_length not in self.candidate_lengths:
            raise ValueError("initial_length must be present in candidate_lengths")
        return self
```

`SpeculativeConfig` 增加：

```python
dynamic_speculative_length: DynamicSpeculativeLengthConfig | None = None

@property
def runtime_speculative_lengths(self) -> tuple[int, ...]:
    config = self.dynamic_speculative_length
    if config is None:
        return (self.num_speculative_tokens,)
    return config.candidate_lengths

@property
def initial_runtime_speculative_length(self) -> int:
    config = self.dynamic_speculative_length
    if config is None:
        return self.num_speculative_tokens
    return config.initial_length
```

校验规则：

```python
if dynamic := self.dynamic_speculative_length:
    if max(dynamic.candidate_lengths) != self.num_speculative_tokens:
        raise ValueError(
            "num_speculative_tokens must equal max(candidate_lengths)"
        )
    if self.disable_padded_drafter_batch:
        raise ValueError(
            "dynamic speculative length requires padded drafter batches"
        )
    if self.rejection_sample_method != "standard":
        raise ValueError(
            "dynamic speculative length phase 1 requires standard rejection sampling"
        )
```

vLLM 通用配置与 scheduler 状态机保持 proposer 类型无关，便于独立 CPU
测试；具体执行后端必须在服务启动时声明能力。第一阶段
`vllm-ascend` 只在 `method == "mtp"` 时返回支持，否则启动失败。

候选集合影响图结构，必须加入 `SpeculativeConfig.compute_hash()`：

```python
factors.append(
    None
    if self.dynamic_speculative_length is None
    else (
        self.dynamic_speculative_length.candidate_lengths,
        self.dynamic_speculative_length.strict_graph_mode,
    )
)
```

### 3.2 运行时类型

文件：`vllm/v1/spec_decode/dynamic_length.py`

```python
from enum import StrEnum
from threading import Lock

import msgspec


class DynamicSpecLengthState(StrEnum):
    APPLIED = "applied"
    PENDING = "pending"
    READY = "ready"
    TRANSITION = "transition"
    FAILED = "failed"


class SpecDecodeBatchPlan(msgspec.Struct, frozen=True):
    version: int
    next_draft_length: int


class DynamicSpecLengthStatus(msgspec.Struct, frozen=True):
    state: DynamicSpecLengthState
    requested_length: int
    applied_length: int
    version: int
    candidate_lengths: tuple[int, ...]
    reason: str | None = None
```

控制器对外方法固定为：

| 方法 | 返回类型 | 行为 |
|---|---|---|
| `request_length(length)` | `DynamicSpecLengthStatus` | 单 engine 创建 pending 更新 |
| `prepare_length(length)` | `DynamicSpecLengthStatus` | DP 校验候选并进入 prepared/ready 流程 |
| `commit_prepared(version)` | `DynamicSpecLengthStatus` | 提交已准备版本 |
| `abort_prepared(version, reason)` | `DynamicSpecLengthStatus` | 取消准备，保持旧 applied plan |
| `select_batch_plan(can_start_transition, has_active_decode_requests, verifies_transition_version)` | `SpecDecodeBatchPlan` | scheduler 每批选择只读计划 |
| `get_status()` | `DynamicSpecLengthStatus` | 返回 CPU 状态快照 |
| `candidate_lengths` | `tuple[int, ...]` | 只读候选集合 |
| `applied_plan` | `SpecDecodeBatchPlan` | 只读缓存计划，供测试和观测 |

controller 采用 scheduler 单写者模型：

- scheduler loop 是 applied/pending/transition 状态的唯一修改者。
- utility handler 只发布不可变的控制请求。
- 稳态 `select_batch_plan()` 直接返回缓存引用，不获取锁。
- `_control_lock` 只在发布/接收控制请求和切换状态变化时使用。
- 锁内禁止设备调用、RPC 和等待。

### 3.3 SchedulerOutput 契约

在 `SchedulerOutput` 现有字段末尾增加：

```python
spec_decode_batch_plan: SpecDecodeBatchPlan | None = None
```

`make_empty()` 保持 `None`。未启用动态功能时固定路径仍为 `None`，worker 不执行新增逻辑。

### 3.4 请求 cohort 元数据

策略仍然是批次级。scheduler 内部需要追踪请求最后一次生成 draft 时采用的计划版本，用于证明切换安全：

```python
self._spec_plan_version_by_req: dict[str, int] = {}
```

该映射不是请求级策略：

- 用户不能单独设置某个请求。
- 同一个调度批只会写入一个版本。
- 请求 finish/abort 时立即删除。
- 新请求首次进入 draft 时使用当前 applied 版本。

### 3.5 图键

```python
@dataclass(frozen=True)
class BatchDescriptor:
    num_tokens: int
    num_reqs: int | None = None
    uniform: bool = False
    uniform_query_len: int | None = None
    has_lora: bool = False
    num_active_loras: int = 0
```

不变量：

```python
if self.uniform:
    assert self.uniform_query_len is not None
else:
    assert self.uniform_query_len is None
```

### 3.6 Ascend 执行计划

文件：`vllm_ascend/spec_decode/dynamic_length.py`

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
                f"Speculative length {length} was not prepared at startup"
            ) from exc
```

### 3.7 测试局部 helper 约束

后续测试代码片段中的 `make_*`、`build_*`、`install_*` 名称均为同一测试
文件内新建的私有 helper，不依赖隐藏的外部测试框架。实现行为固定如下：

| Helper | 文件 | 固定行为 |
|---|---|---|
| `make_dynamic_decode_scheduler` | `tests/v1/core/test_dynamic_spec_scheduler.py` | 调用扩展后的 `create_scheduler()`，完成一次 prefill/update，将请求放入 RUNNING decode，并安装指定长度的 `spec_token_ids` |
| `install_worker_drafts` | 同上 | 为每个 req_id 构造长度为 `draft_length` 的 token 列表，封装为 `DraftTokenIds` 并调用 `scheduler.update_draft_token_ids()` |
| `make_running_decode_request` | `tests/v1/core/test_async_scheduler.py` | 使用 `create_requests()` 创建请求，完成 prefill 后设置给定 draft token |
| `make_full_decode_dispatcher` | `tests/v1/cudagraph/test_cudagraph_dispatch.py` | 复用 `_create_vllm_config()`，创建 `FULL_DECODE_ONLY` dispatcher |
| `make_async_llm_with_mock_engine` | `tests/v1/engine/test_async_llm.py` | 使用 `AsyncLLM.__new__()`，只安装 `AsyncMock` engine core |
| `make_dp_client_with_two_engines` | `tests/v1/distributed/test_async_llm_dp.py` | 使用 `DPLBAsyncMPClient.__new__()`，安装两个假 engine identity 和可检查调用的 `AsyncMock` |
| `prepared_status` / `failed_status` | 同上 | 直接构造 `DynamicSpecLengthStatus`，不包含额外状态 |
| `abort_call_was_sent_to_prepared_engine` | 同上 | 检查 `_call_utility_async.mock_calls` 中是否存在对应 version 的 `abort_speculative_length` |
| `make_runner_with_dynamic_graph_dispatcher` | `tests/ut/worker/a2/test_model_runner_v1.py` | 使用 `NPUModelRunner.__new__()`，只创建 `_determine_batch_execution_and_padding()` 所需 CPU numpy 状态与 mock dispatcher |
| `make_mock_mtp_proposer` | `tests/ut/spec_decode/test_dynamic_length.py` | 使用 `AscendSpecDecodeBaseProposer.__new__()`，预分配 max-length CPU tensor，并以 `MagicMock` model 统计 forward 次数 |
| `make_draft_inputs` | 同上 | 返回 `_run_merged_draft()` 所需的最小 CPU tensor 参数字典，batch size 固定为 2 |
| `make_dynamic_runner` | 同上 | 使用 `NPUModelRunner.__new__()`，安装 registry、mock target dispatcher 和 mock drafter |
| `make_scheduler_output` | 同上 | 构造包含实际 draft 列表和 `SpecDecodeBatchPlan` 的 `SimpleNamespace` |
| `make_attention_cp_builder` | `tests/ut/attention/a2/test_attention_cp_precision.py` | 复用该文件既有 builder fixture，仅覆盖 `decode_threshold` |
| `make_common_metadata` | 同上 | 复用该文件既有 metadata factory，并覆盖 query start location 与 decode 数 |
| `make_registry_with_captured_lengths` | `tests/ut/spec_decode/test_dynamic_length.py` | 构建 registry 后显式调用 target/draft graph ready 标记方法 |
| `build_prompt_matrix` | `tests/e2e/pull_request/one_card/spec_decode/test_dynamic_mtp.py` | 返回固定短、中、长上下文 prompt 列表 |
| `make_dynamic_llm` | `tests/e2e/pull_request/one_card/spec_decode/test_dynamic_mtp.py` | 复用 `VllmRunner`，固定 DeepSeek MTP、async scheduling 和 `FULL_DECODE_ONLY` |
| `set_length` | 同上 | 调用公开 Engine API，并在 timeout 内等待 `applied` |
| `assert_outputs_equal` | 同上 | 比较请求顺序、token ids 和归一化后的 logprobs |

helper 中不得执行设备同步。CPU unit test factory 不创建真实 NPU tensor。

## 4. Task 1：配置与固定路径兼容

**Files:**

- Modify: `vllm/config/speculative.py:74-299,975-1019`
- Modify: `vllm/config/__init__.py:40,115-117`
- Modify: `tests/test_config.py:1477-1494`

- [ ] **Step 1: 写配置失败测试**

```python
def test_dynamic_speculative_length_config_accepts_sorted_candidates():
    config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=8,
        dynamic_speculative_length={
            "candidate_lengths": [1, 3, 8],
            "initial_length": 3,
        },
    )
    assert config.runtime_speculative_lengths == (1, 3, 8)
    assert config.initial_runtime_speculative_length == 3


@pytest.mark.parametrize(
    "dynamic_config,error",
    [
        ({"candidate_lengths": [], "initial_length": 1}, "must not be empty"),
        (
            {"candidate_lengths": [3, 1, 8], "initial_length": 3},
            "strictly increasing",
        ),
        (
            {"candidate_lengths": [1, 3, 3], "initial_length": 3},
            "strictly increasing",
        ),
        (
            {"candidate_lengths": [1, 3, 8], "initial_length": 5},
            "must be present",
        ),
    ],
)
def test_dynamic_speculative_length_config_rejects_invalid_candidates(
    dynamic_config, error
):
    with pytest.raises(ValueError, match=error):
        SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=8,
            dynamic_speculative_length=dynamic_config,
        )


def test_dynamic_speculative_length_requires_max_capacity_match():
    with pytest.raises(ValueError, match="must equal max"):
        SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=3,
            dynamic_speculative_length={
                "candidate_lengths": [1, 3, 8],
                "initial_length": 3,
            },
        )


def test_fixed_speculative_config_has_single_runtime_length():
    config = SpeculativeConfig(method="ngram", num_speculative_tokens=3)
    assert config.runtime_speculative_lengths == (3,)
    assert config.initial_runtime_speculative_length == 3
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
cd /Users/linyi/code/Documents/code/vllm
pytest -q tests/test_config.py -k dynamic_speculative_length
```

Expected: FAIL，错误指向未知配置字段或缺少辅助属性。

- [ ] **Step 3: 实现配置类型、校验和 hash**

按第 3.1 节完整实现。不要修改 `num_speculative_tokens` 的既有语义。

- [ ] **Step 4: 运行配置测试**

Run:

```bash
pytest -q tests/test_config.py -k "dynamic_speculative_length or draft_sample_method"
```

Expected: PASS。

- [ ] **Step 5: 提交 vLLM 配置改动**

```bash
git add vllm/config/speculative.py vllm/config/__init__.py tests/test_config.py
git commit -m "feat(spec-decode): add dynamic length startup config"
```

## 5. Task 2：CPU-only 长度控制器

**Files:**

- Create: `vllm/v1/spec_decode/dynamic_length.py`
- Create: `tests/v1/spec_decode/test_dynamic_length.py`

- [ ] **Step 1: 写状态机失败测试**

```python
def test_controller_is_idempotent_for_applied_length():
    controller = DynamicSpecLengthController((1, 3, 8), initial_length=3)
    before = controller.get_status()
    after = controller.request_length(3)
    assert after == before


def test_controller_rejects_unprepared_length_without_state_change():
    controller = DynamicSpecLengthController((1, 3, 8), initial_length=3)
    with pytest.raises(ValueError, match="not prepared"):
        controller.request_length(5)
    assert controller.get_status().applied_length == 3
    assert controller.get_status().version == 1


def test_controller_transition_uses_cached_batch_plan():
    controller = DynamicSpecLengthController((1, 3, 8), initial_length=3)
    pending = controller.request_length(8)
    assert pending.state == DynamicSpecLengthState.PENDING

    old_plan = controller.select_batch_plan(
        can_start_transition=False,
        has_active_decode_requests=True,
        verifies_transition_version=False,
    )
    assert old_plan.next_draft_length == 3

    transition_plan = controller.select_batch_plan(
        can_start_transition=True,
        has_active_decode_requests=True,
        verifies_transition_version=False,
    )
    assert transition_plan.next_draft_length == 8
    assert controller.get_status().state == DynamicSpecLengthState.TRANSITION

    same_plan = controller.select_batch_plan(
        can_start_transition=False,
        has_active_decode_requests=True,
        verifies_transition_version=False,
    )
    assert same_plan is transition_plan

    applied_plan = controller.select_batch_plan(
        can_start_transition=False,
        has_active_decode_requests=True,
        verifies_transition_version=True,
    )
    assert applied_plan.next_draft_length == 8
    assert controller.get_status().state == DynamicSpecLengthState.APPLIED


def test_idle_controller_applies_without_transition_batch():
    controller = DynamicSpecLengthController((1, 3, 8), initial_length=3)
    controller.request_length(8)
    plan = controller.select_batch_plan(
        can_start_transition=True,
        has_active_decode_requests=False,
        verifies_transition_version=False,
    )
    assert plan.next_draft_length == 8
    assert controller.get_status().state == DynamicSpecLengthState.APPLIED
```

- [ ] **Step 2: 运行测试并确认失败**

```bash
cd /Users/linyi/code/Documents/code/vllm
pytest -q tests/v1/spec_decode/test_dynamic_length.py
```

Expected: FAIL，模块不存在。

- [ ] **Step 3: 实现控制器**

实现要求：

- 初始版本固定为 `1`。
- 同一个版本缓存一个 `SpecDecodeBatchPlan`。
- `request_length()` 只改变 CPU 状态。
- 只允许一个 pending/prepared 更新；新请求覆盖旧 pending 时分配新版本。
- 已进入 transition 后拒绝覆盖，返回明确错误。
- `FAILED` 作为最近一次控制操作结果，不改变 applied plan。
- APPLIED 稳态批不获取 `_control_lock`。
- 锁内不等待。

核心初始化：

```python
self._candidate_lengths = candidate_lengths
self._version = 1
self._applied_length = initial_length
self._requested_length = initial_length
self._state = DynamicSpecLengthState.APPLIED
self._plans = {
    initial_length: SpecDecodeBatchPlan(
        version=self._version,
        next_draft_length=initial_length,
    )
}
self._control_lock = Lock()
```

稳态快速路径：

```python
if self._state == DynamicSpecLengthState.APPLIED:
    return self._applied_plan
```

只有检测到不可变 control request 引用非空时才进入带锁状态转换。

- [ ] **Step 4: 运行状态机测试**

```bash
pytest -q tests/v1/spec_decode/test_dynamic_length.py
```

Expected: PASS。

- [ ] **Step 5: 提交控制器**

```bash
git add vllm/v1/spec_decode/dynamic_length.py \
  tests/v1/spec_decode/test_dynamic_length.py
git commit -m "feat(spec-decode): add batch length controller"
```

## 6. Task 3：SchedulerOutput 计划戳与基础 scheduler 集成

**Files:**

- Modify: `vllm/v1/core/sched/output.py:180-255`
- Modify: `vllm/v1/core/sched/scheduler.py:217-233,959-1014`
- Create: `tests/v1/core/test_dynamic_spec_scheduler.py`
- Modify: `tests/v1/core/utils.py:43-170`

- [ ] **Step 1: 扩展测试 helper**

`create_scheduler()` 增加：

```python
dynamic_speculative_lengths: tuple[int, ...] | None = None
initial_speculative_length: int | None = None
```

构造：

```python
dynamic_config = None
if dynamic_speculative_lengths is not None:
    dynamic_config = {
        "candidate_lengths": dynamic_speculative_lengths,
        "initial_length": initial_speculative_length,
    }

speculative_config = SpeculativeConfig(
    method="ngram",
    num_speculative_tokens=max(dynamic_speculative_lengths),
    dynamic_speculative_length=dynamic_config,
)
```

- [ ] **Step 2: 写计划戳失败测试**

```python
def test_scheduler_stamps_initial_dynamic_plan():
    scheduler = create_scheduler(
        num_speculative_tokens=8,
        dynamic_speculative_lengths=(1, 3, 8),
        initial_speculative_length=3,
    )
    scheduler.add_request(create_requests(1)[0])
    output = scheduler.schedule()
    assert output.spec_decode_batch_plan is not None
    assert output.spec_decode_batch_plan.next_draft_length == 3


def test_fixed_scheduler_does_not_stamp_dynamic_plan():
    scheduler = create_scheduler(num_speculative_tokens=3)
    scheduler.add_request(create_requests(1)[0])
    output = scheduler.schedule()
    assert output.spec_decode_batch_plan is None
```

- [ ] **Step 3: 运行测试并确认失败**

```bash
pytest -q tests/v1/core/test_dynamic_spec_scheduler.py \
  -k "stamps_initial or fixed_scheduler"
```

Expected: FAIL，`SchedulerOutput` 没有该字段。

- [ ] **Step 4: 集成 controller**

Scheduler 初始化：

```python
dynamic_config = (
    speculative_config.dynamic_speculative_length
    if speculative_config is not None
    else None
)
self.dynamic_spec_length_controller = (
    DynamicSpecLengthController(
        dynamic_config.candidate_lengths,
        dynamic_config.initial_length,
    )
    if dynamic_config is not None
    else None
)
self._spec_plan_version_by_req: dict[str, int] = {}
```

构造 `SchedulerOutput` 前：

```python
batch_plan = None
if self.dynamic_spec_length_controller is not None:
    batch_plan = self.dynamic_spec_length_controller.select_batch_plan(
        can_start_transition=self._is_dynamic_spec_transition_safe(
            num_scheduled_tokens
        ),
        has_active_decode_requests=self._has_active_decode_requests(),
        verifies_transition_version=self._batch_verifies_transition_version(
            num_scheduled_tokens
        ),
    )
```

输出：

```python
spec_decode_batch_plan=batch_plan
```

- [ ] **Step 5: 运行基础 scheduler 测试**

```bash
pytest -q tests/v1/core/test_dynamic_spec_scheduler.py \
  -k "stamps_initial or fixed_scheduler"
pytest -q tests/v1/core/test_scheduler.py
```

Expected: PASS。

- [ ] **Step 6: 提交输出契约**

```bash
git add vllm/v1/core/sched/output.py \
  vllm/v1/core/sched/scheduler.py \
  tests/v1/core/utils.py \
  tests/v1/core/test_dynamic_spec_scheduler.py
git commit -m "feat(scheduler): stamp dynamic spec plans on batches"
```

## 7. Task 4：批次安全点、cohort 版本与 3→8 过渡

**Files:**

- Modify: `vllm/v1/core/sched/scheduler.py:299-320,940-1030,1395-1510,1821-1880`
- Modify: `tests/v1/core/test_dynamic_spec_scheduler.py`

- [ ] **Step 1: 写安全点失败测试**

```python
def test_transition_waits_until_all_running_decode_requests_are_scheduled():
    scheduler = make_dynamic_decode_scheduler(
        candidate_lengths=(1, 3, 8),
        initial_length=3,
        max_num_batched_tokens=4,
        request_draft_lengths=(3, 3),
    )
    status = scheduler.request_speculative_length(8)
    assert status.state == DynamicSpecLengthState.PENDING

    output = scheduler.schedule()
    assert len(output.num_scheduled_tokens) == 1
    assert output.spec_decode_batch_plan.next_draft_length == 3
    assert scheduler.get_speculative_length_status().state == (
        DynamicSpecLengthState.PENDING
    )


def test_transition_batch_verifies_old_drafts_and_produces_new_drafts():
    scheduler = make_dynamic_decode_scheduler(
        candidate_lengths=(1, 3, 8),
        initial_length=3,
        max_num_batched_tokens=32,
        request_draft_lengths=(3, 3),
    )
    scheduler.request_speculative_length(8)

    transition = scheduler.schedule()
    assert {
        len(tokens)
        for tokens in transition.scheduled_spec_decode_tokens.values()
    } == {3}
    assert transition.spec_decode_batch_plan.next_draft_length == 8
    assert scheduler.get_speculative_length_status().state == (
        DynamicSpecLengthState.TRANSITION
    )


def test_next_batch_verifies_new_plan_and_marks_applied():
    scheduler = make_dynamic_decode_scheduler(
        candidate_lengths=(1, 3, 8),
        initial_length=3,
        max_num_batched_tokens=32,
        request_draft_lengths=(3, 3),
    )
    scheduler.request_speculative_length(8)
    transition = scheduler.schedule()
    install_worker_drafts(scheduler, transition, draft_length=8)

    applied = scheduler.schedule()
    assert {
        len(tokens)
        for tokens in applied.scheduled_spec_decode_tokens.values()
    } == {8}
    assert applied.spec_decode_batch_plan.next_draft_length == 8
    assert scheduler.get_speculative_length_status().state == (
        DynamicSpecLengthState.APPLIED
    )


def test_queued_old_plan_remains_ordered_before_transition_plan():
    scheduler = make_dynamic_decode_scheduler(
        candidate_lengths=(1, 3, 8),
        initial_length=3,
        max_num_batched_tokens=32,
        request_draft_lengths=(3, 3),
    )
    queued_old = scheduler.schedule()
    scheduler.request_speculative_length(8)
    transition = scheduler.schedule()

    assert queued_old.spec_decode_batch_plan.version == 1
    assert queued_old.spec_decode_batch_plan.next_draft_length == 3
    assert transition.spec_decode_batch_plan.version == 2
    assert transition.spec_decode_batch_plan.next_draft_length == 8
    assert [
        output.spec_decode_batch_plan.version
        for output in (queued_old, transition)
    ] == [1, 2]
```

- [ ] **Step 2: 运行测试并确认失败**

```bash
pytest -q tests/v1/core/test_dynamic_spec_scheduler.py -k transition
```

Expected: FAIL，安全点和 cohort 版本逻辑尚不存在。

- [ ] **Step 3: 实现安全点**

辅助方法固定为：

```python
def _active_decode_request_ids(self) -> set[str]:
    return {
        request.request_id
        for request in self.running
        if request.num_computed_tokens >= request.num_prompt_tokens
        and not request.is_finished()
    }


def _is_dynamic_spec_transition_safe(
    self,
    num_scheduled_tokens: dict[str, int],
) -> bool:
    active_decode_ids = self._active_decode_request_ids()
    return active_decode_ids <= num_scheduled_tokens.keys()
```

还必须拒绝以下批次作为安全点：

- 存在本轮 preempted decode request。
- PP eligibility 使某个 active decode request 未被调度。
- 某请求处于未回填的 structured output placeholder 状态。

这些条件应在一个方法中返回 `False`，不能通过等待 worker 或设备状态判断。
已经提交到 FIFO batch queue 的旧版本 output 不阻止切换，也不能被重写。

- [ ] **Step 4: 记录和清理 cohort 版本**

Scheduler 暴露给 EngineCore 的方法名固定为：

```python
def request_speculative_length(
    self, length: int
) -> DynamicSpecLengthStatus:
    return self.dynamic_spec_length_controller.request_length(length)


def prepare_speculative_length(
    self, length: int
) -> DynamicSpecLengthStatus:
    return self.dynamic_spec_length_controller.prepare_length(length)


def commit_speculative_length(
    self, version: int
) -> DynamicSpecLengthStatus:
    return self.dynamic_spec_length_controller.commit_prepared(version)


def abort_speculative_length(
    self, version: int, reason: str
) -> DynamicSpecLengthStatus:
    return self.dynamic_spec_length_controller.abort_prepared(version, reason)


def get_speculative_length_status(self) -> DynamicSpecLengthStatus:
    return self.dynamic_spec_length_controller.get_status()
```

动态功能未配置时，上述方法抛出明确的 `RuntimeError`。

在 `_update_after_schedule()` 的基础实现中：

```python
plan = scheduler_output.spec_decode_batch_plan
if plan is not None:
    for req_id in scheduler_output.num_scheduled_tokens:
        request = self.requests[req_id]
        if not request.is_prefill_chunk:
            self._spec_plan_version_by_req[req_id] = plan.version
```

finish/abort 清理：

```python
self._spec_plan_version_by_req.pop(request_id, None)
```

- [ ] **Step 5: 运行 scheduler 全量 CPU 测试**

```bash
pytest -q tests/v1/core/test_dynamic_spec_scheduler.py
pytest -q tests/v1/core/test_scheduler.py
pytest -q tests/v1/core/test_async_scheduler.py
```

Expected: PASS。

- [ ] **Step 6: 提交安全点**

```bash
git add vllm/v1/core/sched/scheduler.py \
  tests/v1/core/test_dynamic_spec_scheduler.py
git commit -m "feat(scheduler): switch spec length at batch safe points"
```

## 8. Task 5：AsyncScheduler placeholder bank

**Files:**

- Modify: `vllm/v1/core/sched/async_scheduler.py:10-47`
- Modify: `tests/v1/core/test_async_scheduler.py`

- [ ] **Step 1: 写 placeholder 失败测试**

```python
def test_async_scheduler_uses_transition_draft_length_for_placeholders():
    scheduler = create_scheduler(
        async_scheduling=True,
        num_speculative_tokens=8,
        dynamic_speculative_lengths=(1, 3, 8),
        initial_speculative_length=3,
    )
    request = make_running_decode_request(draft_tokens=[10, 11, 12])
    scheduler.add_request(request)
    scheduler.request_speculative_length(8)

    output = scheduler.schedule()

    assert len(output.scheduled_spec_decode_tokens[request.request_id]) == 3
    assert output.spec_decode_batch_plan.next_draft_length == 8
    assert request.spec_token_ids == [-1] * 8


def test_async_placeholder_lists_are_reused():
    scheduler = create_scheduler(
        async_scheduling=True,
        num_speculative_tokens=8,
        dynamic_speculative_lengths=(1, 3, 8),
        initial_speculative_length=3,
    )
    assert scheduler._spec_token_placeholder_bank[3] is (
        scheduler._spec_token_placeholder_bank[3]
    )
```

- [ ] **Step 2: 运行测试并确认失败**

```bash
pytest -q tests/v1/core/test_async_scheduler.py -k dynamic
```

Expected: FAIL，仍使用最大长度单一 placeholder。

- [ ] **Step 3: 实现 placeholder bank**

初始化：

```python
if self.dynamic_spec_length_controller is None:
    self._spec_token_placeholder_bank = {
        self.num_spec_tokens: [-1] * self.num_spec_tokens
    }
else:
    self._spec_token_placeholder_bank = {
        length: [-1] * length
        for length in self.dynamic_spec_length_controller.candidate_lengths
    }
```

更新：

```python
plan = scheduler_output.spec_decode_batch_plan
next_draft_length = (
    self.num_spec_tokens if plan is None else plan.next_draft_length
)
request.spec_token_ids = self._spec_token_placeholder_bank[next_draft_length]
```

`request.num_output_placeholders` 仍使用当前 `cur_num_spec_tokens`，不能改成 `next_draft_length`。

- [ ] **Step 4: 运行 async 测试**

```bash
pytest -q tests/v1/core/test_async_scheduler.py
pytest -q tests/v1/core/test_dynamic_spec_scheduler.py
```

Expected: PASS。

- [ ] **Step 5: 提交 async 改动**

```bash
git add vllm/v1/core/sched/async_scheduler.py \
  tests/v1/core/test_async_scheduler.py
git commit -m "feat(async-scheduler): reuse per-length draft placeholders"
```

## 9. Task 6：多 query-length CUDAGraph 图键和 padding

**Files:**

- Modify: `vllm/forward_context.py:30-58`
- Modify: `vllm/config/compilation.py:1316-1507`
- Modify: `vllm/v1/cudagraph_dispatcher.py:20-340`
- Modify: `tests/v1/cudagraph/test_cudagraph_dispatch.py`

- [ ] **Step 1: 写图键失败测试**

```python
def test_dynamic_uniform_lengths_create_distinct_full_graph_keys():
    dispatcher = make_full_decode_dispatcher(
        capture_sizes=[2, 4, 8, 9, 16, 18, 32, 36],
        max_num_seqs=8,
    )
    dispatcher.initialize_cudagraph_keys(
        CUDAGraphMode.FULL_DECODE_ONLY,
        uniform_decode_query_lens=(2, 4, 9),
    )

    mode2, key2 = dispatcher.dispatch(
        num_tokens=8,
        uniform_decode=True,
        uniform_decode_query_len=2,
    )
    mode4, key4 = dispatcher.dispatch(
        num_tokens=8,
        uniform_decode=True,
        uniform_decode_query_len=4,
    )

    assert mode2 == mode4 == CUDAGraphMode.FULL
    assert key2.uniform_query_len == 2
    assert key4.uniform_query_len == 4
    assert key2 != key4


def test_uniform_padding_uses_active_query_length_map():
    dispatcher = make_full_decode_dispatcher(
        capture_sizes=[2, 4, 8, 9, 12, 16, 18],
        max_num_seqs=8,
    )
    dispatcher.initialize_cudagraph_keys(
        CUDAGraphMode.FULL_DECODE_ONLY,
        uniform_decode_query_lens=(2, 4, 9),
    )

    _, key4 = dispatcher.dispatch(
        num_tokens=10,
        uniform_decode=True,
        uniform_decode_query_len=4,
    )
    _, key9 = dispatcher.dispatch(
        num_tokens=10,
        uniform_decode=True,
        uniform_decode_query_len=9,
    )
    assert key4.num_tokens == 12
    assert key9.num_tokens == 18
```

- [ ] **Step 2: 运行图测试并确认失败**

```bash
pytest -q tests/v1/cudagraph/test_cudagraph_dispatch.py -k dynamic_uniform
```

Expected: FAIL，API 不支持多个 query length。

- [ ] **Step 3: 扩展 BatchDescriptor**

加入 `uniform_query_len` 和 `__post_init__` 不变量。所有现有 `uniform=True` 测试对象补充 `uniform_query_len=1`。

- [ ] **Step 4: 实现多长度 capture size 归一化**

`CompilationConfig.resolve_cudagraph_mode_and_sizes()` 接收：

```python
uniform_decode_query_lens: int | tuple[int, ...] = 1
```

先归一化：

```python
query_lens = (
    (uniform_decode_query_lens,)
    if isinstance(uniform_decode_query_lens, int)
    else tuple(sorted(set(uniform_decode_query_lens)))
)
```

对每个 query length 独立生成合法 capture sizes，再取 union。SP 场景每个候选长度分别验证与 TP 的公倍数，不用一个全局 LCM 强迫所有图扩大。

- [ ] **Step 5: 实现分长度 padding map**

Dispatcher 新增：

```python
self._uniform_padding_maps: dict[int, list[int]] = {}
```

每个 query length 的 map 只包含可被该长度整除的 FULL capture sizes。PIECEWISE 继续使用既有 `_bs_to_padded_graph_size`。

`dispatch()` 新参数：

```python
uniform_decode_query_len: int | None = None
```

当 `uniform_decode=True` 时必须提供长度或使用单长度 legacy 默认值。

- [ ] **Step 6: 运行通用图测试**

```bash
pytest -q tests/v1/cudagraph/test_cudagraph_dispatch.py
pytest -q tests/compile/test_config.py -k cudagraph
```

Expected: PASS。

- [ ] **Step 7: 提交图键改动**

```bash
git add vllm/forward_context.py \
  vllm/config/compilation.py \
  vllm/v1/cudagraph_dispatcher.py \
  tests/v1/cudagraph/test_cudagraph_dispatch.py \
  tests/compile/test_config.py
git commit -m "feat(cudagraph): dispatch multiple uniform query lengths"
```

## 10. Task 7：Engine、AsyncLLM 与 DP 控制接口

**Files:**

- Modify: `vllm/v1/engine/core.py:630-680,817-824`
- Modify: `vllm/v1/engine/core_client.py:875-950,1101-1197,1443-1452`
- Modify: `vllm/v1/engine/async_llm.py:917-976`
- Modify: `vllm/v1/engine/llm_engine.py:327-367`
- Modify: `vllm/entrypoints/llm.py:787-807`
- Modify: `vllm/platforms/interface.py`
- Modify: `tests/v1/engine/test_async_llm.py`
- Modify: `tests/v1/engine/test_llm_engine.py`
- Modify: `tests/v1/distributed/test_async_llm_dp.py`

- [ ] **Step 1: 写 API 透传失败测试**

```python
@pytest.mark.asyncio
async def test_async_llm_sets_speculative_length_through_engine_core():
    llm = make_async_llm_with_mock_engine()
    expected = DynamicSpecLengthStatus(
        state=DynamicSpecLengthState.PENDING,
        requested_length=8,
        applied_length=3,
        version=2,
        candidate_lengths=(1, 3, 8),
    )
    llm.engine_core.set_speculative_length_async.return_value = expected

    status = await llm.set_speculative_length(8)

    assert status == expected
    llm.engine_core.set_speculative_length_async.assert_awaited_once_with(
        8, False, None
    )
```

- [ ] **Step 2: 实现单 EngineCore API**

EngineCore 初始化阶段先做能力门禁：

```python
dynamic_config = (
    self.vllm_config.speculative_config.dynamic_speculative_length
    if self.vllm_config.speculative_config is not None
    else None
)
if (
    dynamic_config is not None
    and not current_platform.supports_dynamic_speculative_length()
):
    raise NotImplementedError(
        "The current platform does not implement dynamic speculative length"
    )
```

`Platform` 基类方法返回 `False`，防止其他 backend 接收配置后静默忽略批次计划。

```python
def prepare_speculative_length(
    self, length: int
) -> DynamicSpecLengthStatus:
    return self.scheduler.prepare_speculative_length(length)


def commit_speculative_length(
    self, version: int
) -> DynamicSpecLengthStatus:
    return self.scheduler.commit_speculative_length(version)


def abort_speculative_length(
    self, version: int, reason: str
) -> DynamicSpecLengthStatus:
    return self.scheduler.abort_speculative_length(version, reason)


def get_speculative_length_status(self) -> DynamicSpecLengthStatus:
    return self.scheduler.get_speculative_length_status()
```

单 engine client 的 `set_speculative_length_async()` 顺序：

```python
prepared = await self.call_utility_async("prepare_speculative_length", length)
return await self.call_utility_async(
    "commit_speculative_length", prepared.version
)
```

同步 client 使用相同 prepare/commit 顺序，通过已有 `call_utility()` 完成，
不进入逐批推理热路径。

- [ ] **Step 3: 写 DP 部分 prepare 失败测试**

```python
@pytest.mark.asyncio
async def test_dp_dynamic_length_aborts_all_prepared_engines_on_failure():
    client = make_dp_client_with_two_engines()
    client._call_utility_async.side_effect = [
        prepared_status(version=2),
        RuntimeError("graph plan missing"),
        failed_status(version=2),
    ]

    with pytest.raises(RuntimeError, match="graph plan missing"):
        await client.set_speculative_length_async(8)

    assert abort_call_was_sent_to_prepared_engine(client, version=2)
```

- [ ] **Step 4: 实现 DP prepare/commit**

DP client 必须覆盖专用方法，不能复用“只返回第一个结果”的通用 `call_utility_async()`：

```python
async def set_speculative_length_async(
    self,
    length: int,
    wait: bool = False,
    timeout: float | None = None,
) -> DynamicSpecLengthStatus:
    prepared = await gather_prepare_from_all_engines(length)
    assert_same_version_and_candidates(prepared)
    try:
        committed = await gather_commit_to_all_engines(prepared[0].version)
    except BaseException:
        await gather_abort_on_all_engines(prepared[0].version)
        raise
    assert_same_requested_length(committed)
    return committed[0]
```

DP 应用安全点采用一次控制面 quiescent handshake：

1. 各 scheduler 到达本地安全点后进入 `READY`。
2. `prepare` Future 完成。
3. coordinator 收齐 READY 后发送 commit。
4. commit 后各 scheduler 才发 transition batch。

READY 等待期间允许 prefill 和旧长度 decode 继续到达安全点，但不得提前发新长度 draft。

- [ ] **Step 5: 实现 AsyncLLM API**

```python
async def set_speculative_length(
    self,
    length: int,
    *,
    wait: bool = False,
    timeout: float | None = None,
) -> DynamicSpecLengthStatus:
    return await self.engine_core.set_speculative_length_async(
        length, wait, timeout
    )


async def get_speculative_length_status(
    self,
) -> DynamicSpecLengthStatus:
    return await self.engine_core.get_speculative_length_status_async()
```

`wait=True` 只轮询 CPU 状态或等待 scheduler Future，不读取设备状态。

同步 `LLMEngine` 与 `LLM` 暴露同名方法：

```python
def set_speculative_length(
    self,
    length: int,
    *,
    wait: bool = False,
    timeout: float | None = None,
) -> DynamicSpecLengthStatus:
    return self.engine_core.set_speculative_length(
        length, wait, timeout
    )


def get_speculative_length_status(self) -> DynamicSpecLengthStatus:
    return self.engine_core.get_speculative_length_status()
```

- [ ] **Step 6: 运行 engine 测试**

```bash
pytest -q tests/v1/engine/test_async_llm.py -k speculative_length
pytest -q tests/v1/engine/test_llm_engine.py -k speculative_length
pytest -q tests/v1/distributed/test_async_llm_dp.py -k speculative_length
```

Expected: PASS。

- [ ] **Step 7: 提交控制接口**

```bash
git add vllm/v1/engine/core.py \
  vllm/v1/engine/core_client.py \
  vllm/v1/engine/async_llm.py \
  vllm/v1/engine/llm_engine.py \
  vllm/entrypoints/llm.py \
  vllm/platforms/interface.py \
  tests/v1/engine/test_async_llm.py \
  tests/v1/engine/test_llm_engine.py \
  tests/v1/distributed/test_async_llm_dp.py
git commit -m "feat(engine): expose dynamic speculative length control"
```

## 11. Task 8：vLLM 通用层回归门禁

**Files:**

- Modify: `tests/v1/e2e/spec_decode/test_async_spec_decode.py`
- Modify: `tests/v1/e2e/general/test_async_scheduling.py`

- [ ] **Step 1: 增加无新增 host sync 结构测试**

在现有 `sync_tracker` 基础上增加动态调用：

```python
outputs_before = llm.generate(
    ["Hello, my name is"],
    SamplingParams(temperature=0, max_tokens=10),
)
asyncio.run(llm.llm_engine.set_speculative_length(3, wait=True))
outputs_after = llm.generate(
    ["The capital of France is"],
    SamplingParams(temperature=0, max_tokens=10),
)
assert outputs_before
assert outputs_after
sync_tracker.assert_no_sync()
```

该 CUDA 测试只证明通用代码没有新增已知 lazy sync；Ascend 最终结论以 NPU profiler 为准。

- [ ] **Step 2: 运行通用测试集**

```bash
pytest -q tests/test_config.py -k dynamic_speculative_length
pytest -q tests/v1/spec_decode/test_dynamic_length.py
pytest -q tests/v1/core/test_dynamic_spec_scheduler.py
pytest -q tests/v1/core/test_async_scheduler.py
pytest -q tests/v1/cudagraph/test_cudagraph_dispatch.py
pytest -q tests/v1/engine/test_async_llm.py -k speculative_length
```

Expected: 全部 PASS。

- [ ] **Step 3: 增加 scheduler CPU 微基准**

文件：`benchmarks/overheads/benchmark_dynamic_spec_length.py`。

连续调用 1,000,000 次稳态 `select_batch_plan()`，比较缓存计划直接读取：

```python
assert controller.select_batch_plan(
    can_start_transition=False,
    has_active_decode_requests=True,
    verifies_transition_version=False,
) is controller.applied_plan
```

门禁：

```text
steady-state controller lookup p50 <= 0.5 microsecond
no per-call object allocation
no lock acquisition in profiler
```

- [ ] **Step 4: 运行固定路径回归**

```bash
pytest -q tests/v1/spec_decode/test_mtp.py
pytest -q tests/v1/e2e/spec_decode/test_spec_decode.py -k mtp
pytest -q tests/v1/e2e/general/test_async_scheduling.py
```

Expected: 全部 PASS；未配置动态功能时 `SchedulerOutput.spec_decode_batch_plan is None`。

- [ ] **Step 5: 提交 CPU 微基准**

```bash
git add benchmarks/overheads/benchmark_dynamic_spec_length.py
git commit -m "bench(spec-decode): measure dynamic length control overhead"
```

- [ ] **Step 6: 记录 vLLM 兼容 commit**

```bash
git status --short
git log --oneline -7
```

Expected: 只有计划内提交，无未提交文件。

## 12. Task 9：Ascend 候选执行计划注册表

**Files:**

- Create: `vllm_ascend/spec_decode/dynamic_length.py`
- Create: `tests/ut/spec_decode/test_dynamic_length.py`
- Modify: `vllm_ascend/worker/model_runner_v1.py:454-490`
- Modify: `vllm_ascend/platform.py`
- Modify: `tests/ut/test_platform.py`

- [ ] **Step 1: 写 registry 失败测试**

```python
def test_registry_builds_one_plan_per_candidate():
    registry = AscendSpecExecutionPlanRegistry((1, 3, 8))
    assert registry.get(1).target_uniform_query_len == 2
    assert registry.get(3).target_uniform_query_len == 4
    assert registry.get(8).target_uniform_query_len == 9


def test_registry_rejects_unprepared_runtime_length():
    registry = AscendSpecExecutionPlanRegistry((1, 3, 8))
    with pytest.raises(RuntimeError, match="not prepared at startup"):
        registry.get(5)


def test_ascend_platform_declares_dynamic_spec_length_support():
    assert NPUPlatform.supports_dynamic_speculative_length()
```

- [ ] **Step 2: 运行测试并确认失败**

```bash
cd /Users/linyi/code/Documents/code/vllm-ascend
pytest -q tests/ut/spec_decode/test_dynamic_length.py
pytest -q tests/ut/test_platform.py -k dynamic_speculative_length
```

Expected: FAIL，模块不存在。

- [ ] **Step 3: 实现 registry 并接入 runner**

Ascend platform 声明支持：

```python
@classmethod
def supports_dynamic_speculative_length(cls) -> bool:
    return True
```

Runner 启动时进行第一阶段能力约束：

```python
if dynamic_config is not None:
    if self.speculative_config.method != "mtp":
        raise NotImplementedError(
            "Ascend dynamic speculative length phase 1 only supports MTP"
        )
    rejection_config = get_ascend_config().rejection_sampler_config
    if (
        rejection_config.enable_block_verify
        or rejection_config.enable_entropy_verify
    ):
        raise ValueError(
            "Dynamic speculative length requires standard verification"
        )
```

Runner 初始化：

```python
runtime_lengths = (
    self.speculative_config.runtime_speculative_lengths
    if self.speculative_config is not None
    else ()
)
self.dynamic_spec_plan_registry = (
    AscendSpecExecutionPlanRegistry(runtime_lengths)
    if len(runtime_lengths) > 1
    else None
)
self.max_spec_length = self.num_spec_tokens
```

固定路径不创建 registry。

- [ ] **Step 4: 运行测试**

```bash
pytest -q tests/ut/spec_decode/test_dynamic_length.py
```

Expected: PASS。

- [ ] **Step 5: 提交 registry**

```bash
git add vllm_ascend/spec_decode/dynamic_length.py \
  vllm_ascend/worker/model_runner_v1.py \
  vllm_ascend/platform.py \
  tests/ut/spec_decode/test_dynamic_length.py \
  tests/ut/test_platform.py
git commit -m "feat(spec-decode): register Ascend runtime length plans"
```

## 13. Task 10：Ascend ACL Graph 复合参数键

**Files:**

- Modify: `vllm_ascend/compilation/acl_graph.py:53-125,284-418`
- Modify: `vllm_ascend/attention/attention_v1.py`
- Modify: `vllm_ascend/attention/mla_v1.py`
- Modify: `vllm_ascend/attention/context_parallel/attention_cp.py`
- Modify: `vllm_ascend/attention/context_parallel/mla_cp.py`
- Modify: `vllm_ascend/ops/gdn.py`
- Modify: `vllm_ascend/_310p/ops/fla/gdn_310.py`
- Create: `tests/ut/compilation/test_dynamic_acl_graph.py`

- [ ] **Step 1: 写复合键失败测试**

```python
def test_graph_params_separate_same_token_count_by_query_length():
    descriptors = [
        BatchDescriptor(
            num_tokens=36,
            num_reqs=18,
            uniform=True,
            uniform_query_len=2,
        ),
        BatchDescriptor(
            num_tokens=36,
            num_reqs=9,
            uniform=True,
            uniform_query_len=4,
        ),
        BatchDescriptor(
            num_tokens=36,
            num_reqs=4,
            uniform=True,
            uniform_query_len=9,
        ),
    ]
    params = GraphParams.from_batch_descriptors(descriptors)
    assert len(params.keys()) == 3
```

- [ ] **Step 2: 运行测试并确认失败**

```bash
pytest -q tests/ut/compilation/test_dynamic_acl_graph.py
```

Expected: FAIL，GraphParams 只接受 token size。

- [ ] **Step 3: 实现 GraphParamKey**

```python
@dataclass(frozen=True)
class GraphParamKey:
    num_tokens: int
    uniform_query_len: int | None

    @classmethod
    def from_batch_descriptor(
        cls, descriptor: BatchDescriptor
    ) -> "GraphParamKey":
        return cls(
            num_tokens=descriptor.num_tokens,
            uniform_query_len=(
                descriptor.uniform_query_len if descriptor.uniform else None
            ),
        )
```

`GraphParams` 提供集中访问方法：

```python
def key_for_current_context(self, num_tokens: int) -> GraphParamKey:
    descriptor = get_forward_context().batch_descriptor
    if descriptor is None:
        return GraphParamKey(num_tokens, None)
    return GraphParamKey(
        num_tokens,
        descriptor.uniform_query_len if descriptor.uniform else None,
    )
```

所有 backend 从：

```python
graph_params.events[num_tokens]
```

改为：

```python
graph_params.events[
    graph_params.key_for_current_context(num_tokens)
]
```

workspace 更新函数也使用相同 key。

- [ ] **Step 4: 用 capture descriptors 初始化参数仓**

`NPUModelRunner._check_and_update_cudagraph_mode()` 不再只传 `capture_sizes`：

```python
capture_descriptors = [
    descriptor
    for _, descriptors in self.cudagraph_dispatcher.get_capture_descs()
    for descriptor in descriptors
]
set_graph_params(capture_descriptors)
set_draft_graph_params(capture_descriptors)
```

- [ ] **Step 5: 运行 ACL Graph 单测**

```bash
pytest -q tests/ut/compilation/test_dynamic_acl_graph.py
pytest -q tests/ut/compilation/a2/test_acl_graph.py
```

Expected: PASS。

- [ ] **Step 6: 提交复合键**

```bash
git add vllm_ascend/compilation/acl_graph.py \
  vllm_ascend/attention/attention_v1.py \
  vllm_ascend/attention/mla_v1.py \
  vllm_ascend/attention/context_parallel/attention_cp.py \
  vllm_ascend/attention/context_parallel/mla_cp.py \
  vllm_ascend/ops/gdn.py \
  vllm_ascend/_310p/ops/fla/gdn_310.py \
  tests/ut/compilation/test_dynamic_acl_graph.py
git commit -m "fix(aclgraph): key graph parameters by query length"
```

## 14. Task 11：Target runner 动态图路由

**Files:**

- Modify: `vllm_ascend/worker/model_runner_v1.py:700-746,1903-2080,2846-2915,3300-3381,4768-4797`
- Modify: `vllm_ascend/patch/worker/patch_cudagraph.py:6-35`
- Modify: `vllm_ascend/compilation/compiler_interface.py:71-79`
- Modify: `tests/ut/worker/a2/test_model_runner_v1.py`

- [ ] **Step 1: 写 target routing 失败测试**

```python
def test_determine_batch_uses_actual_uniform_query_length():
    runner = make_runner_with_dynamic_graph_dispatcher((2, 4, 9))
    runner.input_batch.num_computed_tokens_cpu[:2] = [100, 100]

    mode, descriptor, *_ = runner._determine_batch_execution_and_padding(
        num_tokens=8,
        num_reqs=2,
        num_scheduled_tokens_np=np.array([4, 4], dtype=np.int32),
        max_num_scheduled_tokens=4,
        use_cascade_attn=False,
        uniform_decode_query_len=4,
    )

    assert mode == CUDAGraphMode.FULL
    assert descriptor.uniform_query_len == 4
```

- [ ] **Step 2: 运行测试并确认失败**

```bash
pytest -q tests/ut/worker/a2/test_model_runner_v1.py \
  -k actual_uniform_query_length
```

Expected: FAIL，函数仍比较固定 `self.uniform_decode_query_len`。

- [ ] **Step 3: 从当前批事实计算 target query length**

在 `execute_model()` 中：

```python
uniform_query_len = (
    max_num_scheduled_tokens
    if np.all(num_scheduled_tokens_np == max_num_scheduled_tokens)
    else None
)
```

只有全 decode 且 uniform 时传入 dispatcher。该值来自 CPU scheduler metadata，不读取 NPU。

函数签名增加：

```python
uniform_decode_query_len: int | None = None
```

判断：

```python
uniform_decode = (
    is_all_decode
    and uniform_decode_query_len is not None
    and max_num_scheduled_tokens == uniform_decode_query_len
    and num_tokens == uniform_decode_query_len * num_reqs
)
```

dispatch：

```python
self.cudagraph_dispatcher.dispatch(
    num_tokens=num_tokens,
    has_lora=has_lora,
    uniform_decode=uniform_decode,
    uniform_decode_query_len=uniform_decode_query_len,
    valid_modes=valid_modes,
    invalid_modes=invalid_modes,
    num_active_loras=num_active_loras,
)
```

- [ ] **Step 4: 参数化 padding 和 dummy run**

以下方法不得再读取固定长度决定本批 shape：

为 `_pad_query_start_loc_for_fia()` 和 `_dummy_run()` 各增加关键字参数
`uniform_query_len: int | None = None`，并保持其余现有参数及返回类型不变。

graph capture 时从 `batch_descriptor.uniform_query_len` 传入。

- [ ] **Step 5: 更新 compiler interface**

`_compute_decode_cudagraph_batch_sizes()` 对每个候选 query length 独立过滤后取 union，不只计算最大长度。

- [ ] **Step 6: 运行 runner 与 graph 测试**

```bash
pytest -q tests/ut/worker/a2/test_model_runner_v1.py
pytest -q tests/ut/compilation/test_dynamic_acl_graph.py
pytest -q tests/ut/compilation/a2/test_acl_graph.py
```

Expected: PASS。

- [ ] **Step 7: 提交 target routing**

```bash
git add vllm_ascend/worker/model_runner_v1.py \
  vllm_ascend/patch/worker/patch_cudagraph.py \
  vllm_ascend/compilation/compiler_interface.py \
  tests/ut/worker/a2/test_model_runner_v1.py
git commit -m "feat(aclgraph): route target batches by active MTP length"
```

## 15. Task 12：MTP proposer active draft length

**Files:**

- Modify: `vllm_ascend/worker/model_runner_v1.py:1641-1871`
- Modify: `vllm_ascend/spec_decode/llm_base_proposer.py:133-245,491-632,643-975,987-1252`
- Modify: `tests/ut/spec_decode/test_dynamic_length.py`

- [ ] **Step 1: 写 proposer 步数失败测试**

```python
@pytest.mark.parametrize("active_length", [1, 3, 8])
def test_proposer_runs_exact_active_number_of_steps(active_length):
    proposer = make_mock_mtp_proposer(max_spec_length=8)
    output = proposer._run_merged_draft(
        **make_draft_inputs(),
        active_spec_length=active_length,
    )
    assert output.shape[1] == active_length
    assert proposer.model.forward.call_count == active_length
```

- [ ] **Step 2: 写过渡组合测试**

```python
def test_runner_uses_verify3_target_plan_and_draft8_proposer_plan():
    runner = make_dynamic_runner()
    scheduler_output = make_scheduler_output(
        scheduled_draft_lengths=[3, 3],
        next_draft_length=8,
    )
    runner.execute_model(scheduler_output)
    assert runner.last_target_uniform_query_len == 4
    assert runner.drafter.last_active_spec_length == 8
```

- [ ] **Step 3: 运行测试并确认失败**

```bash
pytest -q tests/ut/spec_decode/test_dynamic_length.py -k "active or verify3"
```

Expected: FAIL，proposer 固定使用最大长度。

- [ ] **Step 4: 增加 active_spec_length 参数**

完整调用链：

```text
NPUModelRunner.propose_draft_token_ids
  -> drafter._propose
  -> drafter._run_merged_draft
  -> per-step metadata update
```

每层显式传递：

```python
active_spec_length: int
```

禁止在 `_run_merged_draft()` 中用运行期可变全局值代替。

- [ ] **Step 5: 保持最大 buffer，切有效前缀**

初始化一次：

```python
self.draft_token_ids_tensor = torch.empty(
    (
        self.num_speculative_tokens,
        self.runner.max_num_reqs,
    ),
    dtype=self.input_ids.dtype,
    device=self.device,
)
```

运行时：

```python
draft_steps = self.draft_token_ids_tensor[:active_spec_length]
for draft_step in range(active_spec_length - 1):
    run_one_additional_draft_step(
        draft_step=draft_step,
        draft_token_ids_tensor=draft_steps,
        multi_steps_attn_metadata=multi_steps_attn_metadata,
    )
return draft_steps.swapaxes(0, 1)
```

这里的 `run_one_additional_draft_step` 表示把当前
`_run_merged_draft()` 循环体原样提取成私有方法；不得改变模型 forward、
position 递增、slot mapping 更新、采样和 collective 的现有顺序。

slot mapping、seq lens、query start location、attention metadata 使用：

```python
self.slot_mapping_group[:active_spec_length]
self.seq_lens_group[:active_spec_length]
self.query_start_loc_group[:active_spec_length]
multi_steps_attn_metadata[:active_spec_length]
```

- [ ] **Step 6: 区分 draft graph**

draft forward context 的 `BatchDescriptor` 必须携带：

```python
uniform_query_len=active_spec_length + 1
```

同一 `num_tokens` 下不同 active length 不能共享 ACL Graph entry。

- [ ] **Step 7: 保持 async accepted-token 校正不变**

以下路径不得改成 CPU 判断：

- `valid_sampled_token_count_gpu`。
- `prev_num_draft_tokens`。
- `update_num_computed_tokens_for_batch_change()`。
- `valid_sampled_token_count_copy_stream`。

只允许把实际上一轮 draft 数继续写入现有 `prev_num_draft_tokens`。

- [ ] **Step 8: 运行 proposer 测试**

```bash
pytest -q tests/ut/spec_decode/test_dynamic_length.py
pytest -q tests/ut/worker/a2/test_model_runner_v1.py \
  -k "placeholder or dynamic"
```

Expected: PASS。

- [ ] **Step 9: 提交 proposer**

```bash
git add vllm_ascend/worker/model_runner_v1.py \
  vllm_ascend/spec_decode/llm_base_proposer.py \
  tests/ut/spec_decode/test_dynamic_length.py \
  tests/ut/worker/a2/test_model_runner_v1.py
git commit -m "feat(mtp): execute the batch-selected draft length"
```

## 16. Task 13：Attention、CP 与通信兼容

**Files:**

- Modify: `vllm_ascend/attention/context_parallel/attention_cp.py:248-255`
- Modify: `tests/ut/attention/a2/test_attention_cp_precision.py`
- Inspect without unrelated refactor:
  `vllm_ascend/attention/context_parallel/sfa_cp.py`
- Inspect without unrelated refactor:
  `vllm_ascend/worker/pcp_utils.py:787-807`

- [ ] **Step 1: 写实际 query length 失败测试**

```python
def test_mtp_cp_actual_seq_lengths_follow_query_start_loc():
    builder = make_attention_cp_builder(decode_threshold=9)
    metadata = make_common_metadata(
        query_start_loc_cpu=torch.tensor([0, 4, 8, 17], dtype=torch.int32),
        num_decodes=2,
    )
    result = builder.build(0, metadata)
    assert result.actual_seq_lengths_q == [4, 8, 17]
```

- [ ] **Step 2: 运行测试并确认失败**

```bash
pytest -q tests/ut/attention/a2/test_attention_cp_precision.py \
  -k actual_seq_lengths_follow
```

Expected: FAIL，当前结果按最大 `decode_threshold` 生成。

- [ ] **Step 3: 改为实际 cumulative query lengths**

替换固定乘法：

```python
actual_seq_lengths_q = query_start_loc_cpu[1:].tolist()
```

保留 prefill/PCP 已有切分逻辑，避免复制或重新同步 tensor。

- [ ] **Step 4: 检查 SFA/DSA/MLA**

使用源码搜索确认所有生产准入 backend：

```bash
rg -n "decode_threshold \\*|uniform_decode_query_len \\*" \
  vllm_ascend/attention vllm_ascend/worker
```

每一处按以下分类记录：

- 容量上界：保留最大长度。
- decode/prefill 分类阈值：保留最大长度。
- 本批实际序列长度：改用 query lens/query_start_loc。
- 图 shape：改用 batch descriptor。

- [ ] **Step 5: 验证通信 shape**

TP/EP/MC2/FlashComm 测试必须断言 active MTP1/3/8 的实际 token 数传入已有通信选择逻辑。不得新增通信初始化或逐批 collective。

- [ ] **Step 6: 运行 attention 精度测试**

```bash
pytest -q tests/ut/attention/a2/test_attention_cp_precision.py
pytest -q tests/ut/attention/a2/test_attention_v1_precision.py \
  -k mtp
pytest -q tests/ut/attention/a2/test_sfa_v1_precision.py \
  -k mtp
pytest -q tests/ut/attention/a2/test_mla_cp_precision.py \
  -k mtp
```

Expected: PASS。

- [ ] **Step 7: 提交 attention 修复**

```bash
git add vllm_ascend/attention/context_parallel/attention_cp.py \
  tests/ut/attention/a2/test_attention_cp_precision.py
git commit -m "fix(attention): derive MTP sequence lengths from batch metadata"
```

## 17. Task 14：Fail-closed 与图完整性

**Files:**

- Modify: `vllm_ascend/spec_decode/dynamic_length.py`
- Modify: `vllm_ascend/worker/model_runner_v1.py`
- Modify: `vllm_ascend/compilation/acl_graph.py`
- Modify: `tests/ut/spec_decode/test_dynamic_length.py`
- Modify: `tests/ut/compilation/test_dynamic_acl_graph.py`

- [ ] **Step 1: 写缺图拒绝测试**

```python
def test_strict_plan_rejects_switch_when_target_graph_is_missing():
    registry = make_registry_with_captured_lengths(target=(1, 3), draft=(1, 3, 8))
    with pytest.raises(RuntimeError, match="target graph.*length 8"):
        registry.assert_ready(8, strict_graph_mode=True)


def test_strict_plan_rejects_switch_when_draft_graph_is_missing():
    registry = make_registry_with_captured_lengths(target=(1, 3, 8), draft=(1, 3))
    with pytest.raises(RuntimeError, match="draft graph.*length 8"):
        registry.assert_ready(8, strict_graph_mode=True)
```

- [ ] **Step 2: 实现启动完整性检查**

capture 完成后为每个候选长度验证：

```python
registry.mark_target_graph_ready(length, target_keys)
registry.mark_draft_graph_ready(length, draft_keys)
registry.assert_all_candidates_ready(strict_graph_mode=True)
```

检查对象是 uniform steady-state keys；非 uniform fallback 不作为缺图。

- [ ] **Step 3: 禁止运行期捕图**

在 graph capture phase 结束后：

```python
self.dynamic_spec_plan_registry.freeze()
```

freeze 后发现新 `BatchDescriptor`：

```python
raise RuntimeError(
    "Dynamic speculative length attempted an uncaptured graph key"
)
```

不能调用 eager 作为 strict 模式下的静默替代。

- [ ] **Step 4: 运行异常测试**

```bash
pytest -q tests/ut/spec_decode/test_dynamic_length.py -k strict
pytest -q tests/ut/compilation/test_dynamic_acl_graph.py
```

Expected: PASS。

- [ ] **Step 5: 提交 fail-closed**

```bash
git add vllm_ascend/spec_decode/dynamic_length.py \
  vllm_ascend/worker/model_runner_v1.py \
  vllm_ascend/compilation/acl_graph.py \
  tests/ut/spec_decode/test_dynamic_length.py \
  tests/ut/compilation/test_dynamic_acl_graph.py
git commit -m "feat(spec-decode): fail closed on missing runtime plans"
```

## 18. Task 15：单卡端到端精度与切换

**Files:**

- Create: `vllm-ascend/tests/e2e/pull_request/one_card/spec_decode/test_dynamic_mtp.py`

- [ ] **Step 1: 建立固定基线输出**

测试分别启动：

- 非投机 target baseline。
- 固定 MTP1。
- 固定 MTP3。
- 固定 MTP8。

统一：

```python
SamplingParams(
    temperature=0,
    max_tokens=256,
    ignore_eos=False,
    logprobs=5,
)
```

保存每个请求：

```python
{
    "token_ids": output.outputs[0].token_ids,
    "logprobs": normalize_logprobs(output.outputs[0].logprobs),
}
```

- [ ] **Step 2: 写动态切换测试**

```python
def test_deepseek_v4_dynamic_mtp_matches_fixed_greedy_baselines():
    prompts = build_prompt_matrix()
    with make_dynamic_llm(candidate_lengths=(1, 3, 8), initial_length=3) as llm:
        dynamic3 = llm.generate(prompts, GREEDY_PARAMS)
        set_length(llm, 8, wait=True)
        dynamic8 = llm.generate(prompts, GREEDY_PARAMS)
        set_length(llm, 1, wait=True)
        dynamic1 = llm.generate(prompts, GREEDY_PARAMS)

    assert_outputs_equal(dynamic1, fixed1)
    assert_outputs_equal(dynamic3, fixed3)
    assert_outputs_equal(dynamic8, fixed8)
    assert_outputs_equal(dynamic1, target_baseline)
    assert_outputs_equal(dynamic3, target_baseline)
    assert_outputs_equal(dynamic8, target_baseline)
```

- [ ] **Step 3: 覆盖过渡边界**

使用持续请求流，在生成过程中执行：

```text
1 → 3 → 8 → 3 → 1
```

断言：

- 没有丢 token、重复 token。
- token ids 与 target baseline 一致。
- status 经过 pending/transition/applied。
- 调度日志中的组合只出现：
  `verify1/draft3`、`verify3/draft8`、`verify8/draft3` 等合法过渡。

- [ ] **Step 4: 覆盖边界输入**

参数矩阵：

```python
[
    {"concurrency": 1, "input_len": 32, "output_len": 32},
    {"concurrency": 8, "input_len": 1024, "output_len": 256},
    {"concurrency": 16, "input_len": 4096, "output_len": 256},
    {"concurrency": 8, "input_len": max_model_len - 12, "output_len": 16},
]
```

另加 EOS、stop token、structured output、abort、preemption。

- [ ] **Step 5: 覆盖接受长度变化**

用现有 rejection sampler 测试注入可重复的 accepted-token count：

```python
accepted_lengths = [0, 1, 3, 8, 2, 0, 7]
```

验证：

- 上一批实际 draft 数进入 `prev_num_draft_tokens`。
- accepted-token device correction 顺序不变。
- 下一批名义长度仍由 batch plan 决定，不由单请求接受率改变。
- 全接受、全拒绝和混合接受时输出均与 target baseline 一致。

- [ ] **Step 6: 运行单卡 E2E**

```bash
pytest -q \
  tests/e2e/pull_request/one_card/spec_decode/test_dynamic_mtp.py \
  -s
```

Expected: PASS。

- [ ] **Step 7: 提交 E2E**

```bash
git add tests/e2e/pull_request/one_card/spec_decode/test_dynamic_mtp.py
git commit -m "test(spec-decode): verify dynamic MTP greedy correctness"
```

## 19. Task 16：TP、EP、DP 与图模式集成

**Files:**

- Modify: `vllm-ascend/tests/e2e/pull_request/two_card/spec_decode/test_spec_decode.py`
- Create: `vllm-ascend/tests/e2e/pull_request/four_card/spec_decode/test_dynamic_mtp.py`

- [ ] **Step 1: TP/EP 动态切换**

运行：

```text
TP2 + EP + async scheduling + FULL_DECODE_ONLY
```

断言所有 worker 收到相同 `SpecDecodeBatchPlan(version, next_draft_length)`。

- [ ] **Step 2: DP prepare/commit**

构造两个 DP engine：

1. 两侧都 ready，切换成功。
2. 一侧缺图，双方保持旧长度。
3. 一侧安全点超时，双方保持 pending/旧长度。
4. commit 后任一 rank 的 batch plan version 不一致时显式失败。

- [ ] **Step 3: graph hit 断言**

为每个候选长度记录 target/draft replay 计数。稳定运行 20 个 decode step 后：

```python
assert online_capture_count == startup_capture_count
assert target_replay_count[length] > 0
assert draft_replay_count[length] > 0
```

- [ ] **Step 4: 运行多卡测试**

```bash
pytest -q \
  tests/e2e/pull_request/two_card/spec_decode/test_spec_decode.py \
  -k dynamic
pytest -q \
  tests/e2e/pull_request/four_card/spec_decode/test_dynamic_mtp.py \
  -s
```

Expected: PASS。

- [ ] **Step 5: 提交多卡测试**

```bash
git add tests/e2e/pull_request/two_card/spec_decode/test_spec_decode.py \
  tests/e2e/pull_request/four_card/spec_decode/test_dynamic_mtp.py
git commit -m "test(spec-decode): cover dynamic MTP distributed execution"
```

## 20. Task 17：性能与无新增同步验证

**Files:**

- Create: `vllm-ascend/benchmarks/scripts/benchmark_dynamic_spec_length.py`
- Create: `vllm-ascend/benchmarks/tests/dynamic-spec-length-tests.json`
- Modify: `vllm-ascend/benchmarks/README.md`

- [ ] **Step 1: 实现固定/动态成对 benchmark**

每个 case 运行：

```text
fixed MTP K
dynamic candidates=[1,3,8], applied=K
```

必须使用相同：

- prompt token ids。
- concurrency。
- input/output length。
- sampling 参数。
- graph capture sizes。
- TP/EP/DP 配置。
- warmup 次数。
- 测量窗口。

输出 JSON：

```json
{
  "device_type": "910B3",
  "mode": "dynamic",
  "applied_length": 3,
  "concurrency": 8,
  "input_len": 1024,
  "output_len": 256,
  "tpot_ms_p50": 0.0,
  "tpot_ms_p90": 0.0,
  "tpot_ms_p99": 0.0,
  "throughput_tok_s": 0.0,
  "target_graph_replays": 0,
  "draft_graph_replays": 0,
  "online_graph_captures": 0
}
```

- [ ] **Step 2: 建立性能矩阵**

```json
[
  {"concurrency": 1, "input_len": 128, "output_len": 256},
  {"concurrency": 8, "input_len": 1024, "output_len": 256},
  {"concurrency": 16, "input_len": 4096, "output_len": 256},
  {"concurrency": 32, "input_len": 8192, "output_len": 128}
]
```

每个 case 测 MTP1、MTP3、MTP8。

同一矩阵分别在 910B3 和 910C 执行，结果文件必须记录 CANN、
torch-npu、vLLM、vllm-ascend 和模型版本。

- [ ] **Step 3: 性能门禁**

自动计算：

```python
assert dynamic.tpot_p50 / fixed.tpot_p50 <= 1.01
assert dynamic.tpot_p90 / fixed.tpot_p90 <= 1.01
assert dynamic.tpot_p99 / fixed.tpot_p99 <= 1.02
assert dynamic.throughput / fixed.throughput >= 0.99
assert dynamic.online_graph_captures == 0
```

- [ ] **Step 4: NPU profiler 对照**

分别 profile 固定 MTP3 与动态 applied MTP3。

检查：

- 没有新增 `aclrtSynchronizeDevice`。
- 没有新增 `torch.npu.synchronize`。
- 没有由新增 `.item()` 触发的 D2H。
- 没有逐批 utility RPC。
- H2D copy 数量和大小无显著增加。
- async copy stream 与默认 stream 的重叠关系保持。

当前代码为部分 attention backend 已存在 event synchronize；判定标准是动态版本相对相同固定长度没有新增同步点。

- [ ] **Step 5: 提交 benchmark**

```bash
git add benchmarks/scripts/benchmark_dynamic_spec_length.py \
  benchmarks/tests/dynamic-spec-length-tests.json \
  benchmarks/README.md
git commit -m "bench(spec-decode): compare dynamic and fixed MTP lengths"
```

## 21. Task 18：长稳、故障注入与内存平台

**Files:**

- Create: `vllm-ascend/tests/e2e/nightly/single_node/spec_decode/test_dynamic_mtp_soak.py`

- [ ] **Step 1: 实现切换循环**

```python
LENGTH_SEQUENCE = (1, 3, 8, 3, 1)

for switch_index in range(10_000):
    target = LENGTH_SEQUENCE[switch_index % len(LENGTH_SEQUENCE)]
    status = set_length(llm, target, wait=True, timeout=30)
    assert status.applied_length == target
    submit_next_request_batch(mixed_workload[switch_index % len(mixed_workload)])
```

- [ ] **Step 2: 故障注入**

覆盖：

- 非候选长度。
- 重复版本。
- prepare 后 abort。
- 缺少 target graph。
- 缺少 draft graph。
- 安全点超时。
- 请求 abort。
- preemption。
- structured output 裁剪。
- 接近 `max_model_len`。

每次故障后发送健康请求，验证服务仍能输出正确 token。

- [ ] **Step 3: 内存平台断言**

每 100 次切换采集：

- NPU allocated/reserved memory。
- host RSS。
- ACL Graph entry 数。
- GraphParamKey 数。
- pending Future 数。

warmup 后采用线性拟合，增长斜率必须接近 0；ACL Graph 与 GraphParamKey 数量必须保持常量。

- [ ] **Step 4: 运行时长**

- PR smoke：30 分钟、100 次切换。
- nightly：24 小时、至少 10,000 次切换。
- release：72 小时。

- [ ] **Step 5: 提交长稳测试**

```bash
git add tests/e2e/nightly/single_node/spec_decode/test_dynamic_mtp_soak.py
git commit -m "test(spec-decode): add dynamic MTP soak coverage"
```

## 22. Task 19：文档、观测与外部控制适配

**Files:**

- Modify: `vllm/docs/features/spec_decode.md` 或当前 speculative decode 主文档
- Modify: `vllm-ascend/docs/source/user_guide/feature_guide/speculative_decoding.md`
- Create: `vllm-ascend/examples/dynamic_speculative_length.py`

- [ ] **Step 1: 文档配置和限制**

写明：

- 候选集合必须启动期声明。
- `num_speculative_tokens` 等于最大候选长度。
- 第一阶段只支持 MTP。
- 切换以调度批次为粒度。
- 过渡批可能是 verify3/draft8。
- strict graph 模式下缺图拒绝切换。
- stochastic 输出不承诺跨长度 bitwise 一致。

- [ ] **Step 2: 提供 Python 控制示例**

```python
status = await engine.set_speculative_length(8, wait=True, timeout=30)
print(status)
```

- [ ] **Step 3: 外部管理接口保持薄适配**

如果部署需要 HTTP，独立 adapter 只调用 `AsyncLLM`：

```python
@router.put("/admin/speculative-length")
async def set_speculative_length(request: SetLengthRequest):
    return await engine.set_speculative_length(
        request.length,
        wait=request.wait,
        timeout=request.timeout,
    )
```

该 adapter 不直接访问 scheduler、worker、NPU tensor 或插件私有对象。

- [ ] **Step 4: 增加低频指标**

控制事件指标：

```text
dynamic_spec_length_applied
dynamic_spec_length_requested_total{length,result}
dynamic_spec_length_transition_total{from,to,result}
dynamic_spec_length_pending_seconds
spec_plan_mismatch_total
```

图 replay/capture 指标可放在 vllm-ascend。禁止逐批 info 日志。

- [ ] **Step 5: 提交文档**

各仓分别提交对应文件，避免跨仓 commit：

```bash
git commit -m "docs(spec-decode): document dynamic MTP length control"
```

## 23. 跨仓实施顺序

必须按以下依赖顺序执行：

```text
vLLM Task 1-6
    ↓
vLLM Task 7-8
    ↓
锁定 vLLM integration commit
    ↓
vllm-ascend Task 9-14
    ↓
单卡精度 Task 15
    ↓
多卡与图模式 Task 16
    ↓
性能 Task 17
    ↓
长稳 Task 18
    ↓
文档与发布 Task 19
```

vllm-ascend 开发环境必须安装包含 vLLM integration commit 的 editable vLLM，不能依赖旧 wheel：

```bash
cd /Users/linyi/code/Documents/code/vllm
pip install -e .

cd /Users/linyi/code/Documents/code/vllm-ascend
pip install -e .
```

## 24. 提交与评审边界

建议 vLLM 保持 7 个可独立评审 commit：

1. 配置。
2. controller。
3. SchedulerOutput。
4. scheduler 安全点。
5. async placeholders。
6. multi-query graph。
7. Engine API 与测试。

建议 vllm-ascend 保持 8 个可独立评审 commit：

1. plan registry。
2. ACL GraphParamKey。
3. target routing。
4. proposer active length。
5. CP metadata。
6. fail-closed。
7. E2E/多卡。
8. benchmark/长稳/文档。

任何 commit 都不得同时包含无关格式化或邻近重构。

## 25. 回滚策略

### 25.1 配置级回滚

删除 `dynamic_speculative_length` 配置后：

- controller 不创建。
- `SchedulerOutput.spec_decode_batch_plan=None`。
- dispatcher 使用单一 legacy query length。
- Ascend registry 不创建。
- 固定 speculative decode 路径保持原样。

### 25.2 运行时失败

更新失败时：

- 保持旧 applied length。
- 不清理旧图。
- 不修改最大 buffer。
- 不重启服务。
- 返回明确 `failed/pending` 状态。

### 25.3 发布回滚

若动态功能未通过性能或长稳门禁：

- 默认配置关闭。
- 保留通用多 query-length 图键代码，前提是固定回归全部通过。
- 不在生产文档中声明可用。

## 26. 完成验收清单

### 功能

- [ ] 支持候选 `{1,3,8}`。
- [ ] 服务运行中热切换，无重启。
- [ ] 同一个调度批只有一个名义 next draft length。
- [ ] 3→8 时出现合法 `verify3/draft8` 过渡。
- [ ] async scheduling 正常。
- [ ] FULL_DECODE_ONLY 对所有候选长度有 target/draft replay。

### 精度

- [ ] 动态 MTP1 与固定 MTP1、target baseline token/logprob 一致。
- [ ] 动态 MTP3 与固定 MTP3、target baseline token/logprob 一致。
- [ ] 动态 MTP8 与固定 MTP8、target baseline token/logprob 一致。
- [ ] 切换边界、EOS、structured output、max model length 正确。

### 性能

- [ ] p50/p90 TPOT 劣化不超过 1%。
- [ ] p99 TPOT 劣化不超过 2%。
- [ ] throughput 劣化不超过 1%。
- [ ] 无运行期新增 graph capture。
- [ ] 无新增 CPU-NPU 同步点。
- [ ] 无逐批控制 RPC。
- [ ] 相同 applied length 命中固定长度等价图和通信路径。

### 稳定性

- [ ] 缺图拒绝切换，旧计划继续。
- [ ] DP prepare 失败不部分提交。
- [ ] 安全点超时不影响推理。
- [ ] 10,000 次切换无 crash/hang/错 token。
- [ ] host/NPU 内存进入平台。
- [ ] graph entry 和 GraphParamKey 数量恒定。

### 兼容性

- [ ] 未配置动态功能时固定路径测试全绿。
- [ ] prefix cache、abort、preemption、PP 队列无回归。
- [ ] TP/EP 通过。
- [ ] DP 通过后才标记 DP 生产可用。
- [ ] PCP/DCP 通过专项 profiler 后才标记生产可用。

## 27. 最终验证命令

### vLLM CPU/通用层

```bash
cd /Users/linyi/code/Documents/code/vllm

pytest -q tests/test_config.py -k dynamic_speculative_length
pytest -q tests/v1/spec_decode/test_dynamic_length.py
pytest -q tests/v1/core/test_dynamic_spec_scheduler.py
pytest -q tests/v1/core/test_async_scheduler.py
pytest -q tests/v1/cudagraph/test_cudagraph_dispatch.py
pytest -q tests/v1/engine/test_async_llm.py -k speculative_length
pytest -q tests/v1/engine/test_llm_engine.py -k speculative_length
pytest -q tests/v1/distributed/test_async_llm_dp.py -k speculative_length
```

### vllm-ascend 单元测试

```bash
cd /Users/linyi/code/Documents/code/vllm-ascend

pytest -q tests/ut/spec_decode/test_dynamic_length.py
pytest -q tests/ut/compilation/test_dynamic_acl_graph.py
pytest -q tests/ut/worker/a2/test_model_runner_v1.py
pytest -q tests/ut/attention/a2/test_attention_cp_precision.py -k mtp
pytest -q tests/ut/attention/a2/test_attention_v1_precision.py -k mtp
pytest -q tests/ut/attention/a2/test_sfa_v1_precision.py -k mtp
pytest -q tests/ut/attention/a2/test_mla_cp_precision.py -k mtp
```

### Ascend E2E

```bash
pytest -q \
  tests/e2e/pull_request/one_card/spec_decode/test_dynamic_mtp.py \
  -s

pytest -q \
  tests/e2e/pull_request/two_card/spec_decode/test_spec_decode.py \
  -k dynamic -s

pytest -q \
  tests/e2e/pull_request/four_card/spec_decode/test_dynamic_mtp.py \
  -s
```

### 性能

```bash
python benchmarks/scripts/benchmark_dynamic_spec_length.py \
  --config benchmarks/tests/dynamic-spec-length-tests.json \
  --output-json dynamic-spec-length-results.json \
  --fail-on-regression
```

Expected:

```text
all steady-state performance gates passed
online graph captures after startup: 0
```

## 28. 本计划输出时的源码状态

本计划仅描述后续修改，不直接修改本地代码。

验证命令：

```bash
git -C /Users/linyi/code/Documents/code/vllm status --short
git -C /Users/linyi/code/Documents/code/vllm-ascend status --short
```

两条命令均应无输出。
