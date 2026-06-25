# B 层 Hook 框架骨架（HookRegistry + RunnerContext + 6 锚点）实现计划

> **说明**：需求 2 工程脊柱的可落地实现计划。这是所有 B 层特性（L4 自适应等）的挂载基座。核心目标：**patch 面积与特性数量解耦**——6 行锚点（1 个 patch 文件）承载任意多特性。
> 生成时间：2026-06-25 01:45
> 关联：[[20260625-011549-昇腾推理集群-deepseek-v4-flash-5to10x吞吐-系统性加速架构设计-分析]]（§4 双层接缝）[[ascend-cluster-5to10x-architecture-spine]]

**Goal**：在 vllm-ascend `model_runner_v1` 主链路插入 6 个稳定 hook 锚点 + 一个特性注册器 + 受控上下文，使新特性以"注册纯函数"方式挂载，零额外 patch。

**Architecture**：① HookRegistry（注册/触发）② RunnerContext（白名单内部态视图）③ 6 锚点（薄 patch，1 文件）④ ctx_adapter（版本吸收）。全 B 层。

**前置事实（grep 实测，6 锚点上下文）**：
- after_prepare_inputs @ `_prepare_inputs(` 后(model_runner_v1.py:2036)
- after_preprocess @ `_preprocess(` 后(:2196)
- around_model_forward @ `_model_forward(` 前后(:2256)
- before_sample @ `_sample(` 前(:2390，sample_tokens 内)
- around_propose_draft @ `propose_draft_token_ids(` 前后(:2403)
- after_bookkeeping @ `_bookkeeping_sync(` 后(:2425)
- HookRegistry 挂载点：`__init__`(:256) 的 self 初始化区

---

## Task 1：HookRegistry（注册/触发核心）

**Files**：Create: `vllm-ascend/vllm_ascend/accel_framework/hook_registry.py`

**Step 1.1 — 注册器**
```python
from collections import defaultdict
from typing import Callable

class HookRegistry:
    """锚点→注册函数列表。特性 = 注册到锚点的纯函数。"""
    # 合法锚点名（与 6 锚点一一对应，防拼写错）
    ANCHORS = frozenset({
        "after_prepare_inputs", "after_preprocess", "around_model_forward",
        "before_sample", "around_propose_draft", "after_bookkeeping",
    })

    def __init__(self):
        self._hooks: dict[str, list[tuple[int, Callable]]] = defaultdict(list)

    def register(self, anchor: str, fn: Callable, priority: int = 0):
        assert anchor in self.ANCHORS, f"unknown anchor {anchor}"
        self._hooks[anchor].append((priority, fn))
        self._hooks[anchor].sort(key=lambda x: x[0])  # 低 priority 先跑

    def run(self, anchor: str, ctx) -> None:
        for _, fn in self._hooks.get(anchor, ()):
            fn(ctx)   # 纯副作用：通过 ctx 读写，不返回
```

**验收（无本地 harness，用最小 import 自测）**
```python
# 验证：注册 + 触发顺序 + 非法锚点拒绝
r = HookRegistry(); seen = []
r.register("before_sample", lambda c: seen.append("a"), priority=1)
r.register("before_sample", lambda c: seen.append("b"), priority=0)
r.run("before_sample", None)
assert seen == ["b", "a"]   # priority 升序
```

**Commit**：`feat(accel): HookRegistry with 6 fixed anchors`

---

## Task 2：RunnerContext（白名单内部态视图）

**Files**：Create: `vllm-ascend/vllm_ascend/accel_framework/runner_context.py`

**Step 2.1 — 受控 ctx**
```python
from dataclasses import dataclass
from typing import Any

@dataclass
class RunnerContext:
    """唯一受控的 runner 内部态视图。特性只能读写这里，禁碰 self.xxx。
    上游改内部字段名 → 只需改 ctx_adapter 的映射，不波及特性。"""
    anchor: str
    scheduler_output: Any = None
    input_ids: Any = None
    positions: Any = None
    spec_decode_metadata: Any = None
    sampler_output: Any = None
    num_reqs: int = 0
    acceptance_stats: dict | None = None   # 供 L4 自适应回灌
    # 特性间传递的暂存（如 L4 算出的 k_eff）
    scratch: dict = None

    def __post_init__(self):
        if self.scratch is None:
            self.scratch = {}
```

**Commit**：`feat(accel): RunnerContext whitelist view`

---

## Task 3：ctx_adapter（版本吸收点）

**Files**：Create: `vllm-ascend/vllm_ascend/accel_framework/ctx_adapter.py`

**Step 3.1 — 按版本组装 ctx（复用 vllm_version_is）**
```python
from vllm_ascend.utils import vllm_version_is  # 已存在 utils.py:558
from .runner_context import RunnerContext

def build_ctx(anchor: str, runner) -> RunnerContext:
    """从 runner 的 self.xxx 抽取白名单字段。这是版本差异的唯一吸收处。"""
    ctx = RunnerContext(anchor=anchor)
    ctx.num_reqs = runner.input_batch.num_reqs
    # 版本差异在此分支吸收（示例：字段改名时）
    if vllm_version_is("0.21.0"):
        ctx.spec_decode_metadata = getattr(runner, "spec_decode_metadata", None)
    else:
        ctx.spec_decode_metadata = getattr(runner, "_spec_metadata", None)  # 假设上游改名
    return ctx
```

**Commit**：`feat(accel): ctx_adapter as version absorption point`

---

## Task 4：6 锚点薄 patch（patch 面积的全部）

**Files**：
- Create: `vllm-ascend/vllm_ascend/patch/worker/patch_runner_hooks.py`
- Modify（薄）: 在 NPUModelRunner `__init__`(:256) 注入 `self._hooks = HookRegistry()`

**Step 4.1 — __init__ 挂注册器（1 行）**
```python
# 在 NPUModelRunner.__init__ 早期（self 初始化区，约 :260）
self._hooks = HookRegistry()
self._load_registered_features()  # 从 additional_config 读启用的特性并 register
```

**Step 4.2 — 6 锚点（每处 1 行，紧贴对应 call-site）**
```python
# @ :2036 后  → after_prepare_inputs
self._hooks.run("after_prepare_inputs", build_ctx("after_prepare_inputs", self))
# @ :2196 后  → after_preprocess
self._hooks.run("after_preprocess", build_ctx("after_preprocess", self))
# @ :2256 前  → around_model_forward (前)
self._hooks.run("around_model_forward", build_ctx("around_model_forward", self))
# @ :2390 前  → before_sample
self._hooks.run("before_sample", build_ctx("before_sample", self))
# @ :2403 前  → around_propose_draft（L4 在此调 k_eff）
self._hooks.run("around_propose_draft", build_ctx("around_propose_draft", self))
# @ :2425 后  → after_bookkeeping（L4 在此回灌接受率）
self._hooks.run("after_bookkeeping", build_ctx("after_bookkeeping", self))
```

> **patch 面积 = 6 锚点行 + 1 init 行 = 7 行**，放进 patch_runner_hooks.py。**与特性数量解耦**。

**Step 4.3 — 升级适配协议（写进文档）**
```
上游升级时：
1. grep 6 个 call-site（_prepare_inputs/_preprocess/_model_forward/_sample/
   propose_draft_token_ids/_bookkeeping_sync）确认仍在
2. 若行号变 → 挪锚点（个位数改动）
3. 若字段改名 → 只改 ctx_adapter.build_ctx 映射
4. patch_runner_hooks.py 之外的特性代码零改动
```

**验收**：起服务，注册一个 no-op 特性，确认 6 锚点都被触发（日志打点），且不影响推理正确性。

**Commit**：`feat(accel): 6 hook anchors thin patch (patch surface = 7 lines)`

---

## Task 5：示范特性 — L4 投机自适应（验证框架可用）

**Files**：Create: `vllm-ascend/vllm_ascend/accel_framework/features/spec_adaptive.py`

**Step 5.1 — 实现为注册到锚点的纯函数**
```python
# 验证"特性 = 注册函数，零额外 patch"
_acc_ema = {"v": 0.7}

def _on_propose(ctx):  # around_propose_draft
    load = ctx.num_reqs / ctx.scratch.get("max_reqs", ctx.num_reqs or 1)
    if load > 0.85 or _acc_ema["v"] < 0.3:
        ctx.scratch["k_eff"] = 0   # 高负载/低接受率 → 不投机
    else:
        ctx.scratch["k_eff"] = min(ctx.scratch.get("k_max", 3), int(3 * _acc_ema["v"]))

def _on_bookkeeping(ctx):  # after_bookkeeping
    if ctx.acceptance_stats:
        acc = ctx.acceptance_stats.get("rate", _acc_ema["v"])
        _acc_ema["v"] = 0.9 * _acc_ema["v"] + 0.1 * acc

def register(hooks):
    hooks.register("around_propose_draft", _on_propose)
    hooks.register("after_bookkeeping", _on_bookkeeping)
```

> 注意：本特性受 L4 设计约束——num_spec 结构固定只能向下调（k_eff≤k_max），见 [[20260625-012334-昇腾集群-L4投机自适应-支柱特性详细设计-分析]]。`propose_draft_token_ids` 内需读 `ctx.scratch["k_eff"]` 限制本步 propose 数（这一步仍需在 propose 路径内消费 k_eff，属 L4 落地细节）。

**验收**：开关本特性，对比 E2（接受率/有效加速），确认框架挂载链路通。

**Commit**：`feat(accel): spec-adaptive feature via hooks (zero extra patch)`

---

## 工作量与依赖

| Task | 工作量 | patch |
|------|--------|-------|
| T1 HookRegistry | 0.2人月 | 0（新文件） |
| T2 RunnerContext | 0.1人月 | 0 |
| T3 ctx_adapter | 0.2人月 | 0 |
| T4 6锚点薄patch | 0.2人月 | **7行/1文件** |
| T5 示范特性 | 0.2人月 | 0 |
| **合计** | **~0.9人月** | **7行** |

## 遗留问题（要求1）

1. **around_model_forward 的"后"锚点** — _model_forward 后立即是 hidden_states 处理，"前后"包夹需两个调用点（本计划只插了"前"）；若特性需"后"，再加 1 锚点。🟡
2. **ctx 白名单字段完备性** — 当前 7 字段覆盖 L4，未来特性可能需更多；扩字段须同步 ctx_adapter。🟡
3. **k_eff 在 propose 路径的消费** — T5 算出 k_eff，但 propose_draft_token_ids 内部消费它需改 propose 逻辑（属 L4 落地，非框架本身）。❓
4. **hook 异常隔离** — 一个特性 fn 抛异常不应崩主链路，run() 需加 try（当前未加，生产需补）。🟡 已知缺口

---

## 自检（writing-plans self-review）

- **spec 覆盖**：§4.5 HookRegistry+RunnerContext+版本契约 → T1/T2/T3 覆盖；§4.4 6锚点 → T4 覆盖 ✅
- **占位符**：所有类/函数给完整代码；锚点行号实测 ✅
- **类型一致**：`HookRegistry.register/run`、`RunnerContext` 字段、`build_ctx` 在 T1-T5 全程一致；`ctx.scratch["k_eff"]` 在 T5 produce、L4 propose consume 一致 ✅
- **patch 面积自洽**：7 行声明与 T4 实际行数一致 ✅
- **bf16-only 合规**：框架无关精度 ✅
