# vLLM / vLLM-Ascend 动态投机长度 Phase 1：实现与 NPU 验证

## 1. 当前实现范围

本阶段实现的是 V1 Scheduler + vLLM-Ascend V1 ModelRunner 的 step 粒度动态 MTP 长度：

- 按当前 Scheduler step 中实际进入 decode 的并发数选择 MTP 长度。
- 同一个 step 内所有 decode 请求使用同一个长度。
- 不按请求绑定长度，不创建 request cohort。
- 当前 step 可向下截断上一 step 已生成的 draft；向上切换不扩充当前 draft。
- 当前 step 选出的长度用于生成下一 step 的 draft。
- Async Scheduler 直接消费 `SchedulerOutput` 携带的不可变 step plan，不读取可变全局状态。
- NPU runner 每 step 只做一次 CPU 侧 plan 查询，不新增 `.item()`、`synchronize()` 或 CPU→NPU 控制流。
- draft 缓冲按最大 K 启动时预分配，运行时只减少 MTP 前向次数和返回张量有效宽度。
- target ACL graph 使用 `(num_tokens, uniform_query_len)` 区分相同 token 数、不同 MTP 形状。
- ACL graph 参数保留原有整数索引热路径，仅在进入 forward context 时切换一次预创建存储别名。

当前没有在本地宣称 NPU E2E、精度、TPOT 或长稳验证通过；这些结论必须在 910B3/910C 环境补测。

## 2. 代码位置和提交

### vLLM

Worktree：

`/Users/linyi/.config/superpowers/worktrees/vllm/dynamic-spec-step-policy-phase1`

提交：

- `88f27bb66`：并发区间策略配置和 O(1) step plan。
- `873332364`：Scheduler/Async Scheduler step 粒度选择与 draft 截断。
- `c862e760c`：动态 query length 图 key。
- `3c108df28`：候选 query length 专属 padding 表及严格图覆盖校验。

### vLLM-Ascend

Worktree：

`/Users/linyi/.config/superpowers/worktrees/vllm-ascend/dynamic-spec-step-policy-phase1`

提交：

- `964ec513`：无 `torch_npu` 依赖的执行计划注册表。
- `b23eeef6`：NPU runner、MTP proposer、ACL graph 参数接入。

## 3. 基础策略配置示例

以下示例表示：

- 并发 1～4：MTP8
- 并发 5～8：MTP5
- 并发 9～16：MTP3
- 未命中范围：MTP1

```json
{
  "method": "mtp",
  "num_speculative_tokens": 8,
  "dynamic_speculative_length": {
    "enabled": true,
    "candidate_lengths": [1, 3, 5, 8],
    "default_length": 1,
    "strict_graph_mode": true,
    "policy": {
      "type": "concurrency_table",
      "rules": [
        {
          "min_concurrency": 1,
          "max_concurrency": 4,
          "speculative_length": 8
        },
        {
          "min_concurrency": 5,
          "max_concurrency": 8,
          "speculative_length": 5
        },
        {
          "min_concurrency": 9,
          "max_concurrency": 16,
          "speculative_length": 3
        }
      ]
    }
  }
}
```

`num_speculative_tokens` 必须等于最大候选长度，用于确定启动期最大缓冲容量。

## 4. ACL graph capture size

候选 MTP 长度 K 对应 target query length `K + 1`。严格图模式要求：在 `max_cudagraph_capture_size` 范围内，每个策略可达形状都能 padding 到一个可被对应 query length 整除的 capture size。

上例在最大并发 16 下，为尽量避免额外 padding，可把以下尺寸合并进原有 `cudagraph_capture_sizes`：

```text
9, 18, 27, 30, 36, 40, 42, 44, 48, 52, 56, 60, 64
```

实际配置应同时保留框架和其他工作负载需要的 capture sizes。若缺失可达形状，`strict_graph_mode=true` 会在启动期报错，而不是运行中静默退化或触发整除断言。

## 5. 本地验证证据

### vLLM

已覆盖：

- 多并发区间和边界选择。
- 非法候选、重叠范围、超出容量配置。
- 只统计当前 step 的 decode 请求。
- prefill 不进入 decode 并发计数。
- downshift 截断整批 draft。
- upshift 不扩充当前 draft。
- Async Scheduler 使用输出 plan 生成下一 step placeholder。
- 固定长度路径不写入动态 plan。
- 相同 token 数、不同 query length 使用不同图 key。
- 非整除通用 capture size 会跳到下一个兼容尺寸。
- 严格图模式在启动期拒绝未覆盖形状。

### vLLM-Ascend

已在无 NPU 本地覆盖：

- 候选执行计划启动期注册。
- 未准备长度拒绝执行。
- Scheduler plan 到 active K 的解析。
- 固定长度兼容回退。
- target query length 候选查询。
- 图参数 key 区分相同 token 数、不同 query length。
- Ruff、Python compileall 和 diff whitespace 检查。

ACL graph 参数隔离测试因本机没有 `torch_npu` 被跳过，文件为：

`tests/ut/compilation/a2/test_dynamic_graph_params.py`

## 6. 910B3 / 910C 功能与精度验证

### 6.1 必测组合

至少覆盖：

| 维度 | 取值 |
|---|---|
| 硬件 | 910B3、910C |
| 调度 | Async Scheduler 开启；同步调度作为辅助对照 |
| 图模式 | 生产图模式；enforce eager 作为定位对照 |
| MTP | 1、3、5、8 |
| 并发 | 1、4、5、8、9、16，以及每个策略边界前后 |
| 上下文 | 短、中、长上下文 |
| 接受率 | 高、中、低接受长度数据集 |
| 运行 | 固定并发、并发阶跃、持续波动、长稳 |

### 6.2 精度对照

对每个并发点分别启动：

1. 固定 MTP-K 服务。
2. 动态服务，确保该并发命中同一个 K。

固定随机种子、采样参数、输入顺序和模型权重，逐请求比较：

- 输出 token IDs。
- 输出文本。
- finish reason。
- 输出长度。
- 可获得时比较 logprob。

贪心解码要求 token IDs 完全一致。随机采样应验证相同随机状态下结果一致，并额外执行统计回归。

### 6.3 step 一致性观测

增加临时 debug 指标或采样日志，仅记录 CPU plan：

- `policy_version`
- `decode_concurrency`
- `selected_mtp_length`
- target 实际 `uniform_query_len`
- draft 返回宽度

验证同一 SchedulerOutput 中所有 decode 请求的 K 一致；日志不得读取 NPU tensor，不得进入默认性能压测路径。

## 7. 性能验收

### 7.1 无劣化基线

每个 K 和对应固定并发分别比较：

```text
固定 MTP-K vs 动态策略命中 MTP-K
```

保持相同：

- 模型、权重、并行配置。
- Async Scheduler。
- ACL graph 配置及 capture sizes。
- 请求数据和到达模式。
- warmup 次数、测试时长。
- NPU 频率、功耗模式和其他系统负载。

主要指标：

- mean / median / P99 TPOT。
- TTFT。
- output throughput。
- draft model 前向次数。
- target verify token 数。
- ACL graph replay 命中率。
- CPU scheduler 时间。

建议验收口径：动态版本与固定 K 的 TPOT 差异不超过重复实验噪声；可先采用中位数绝对差 ≤1% 且置信区间重叠，最终以项目性能规范为准。

### 7.2 动态收益

以固定最大 MTP8 为对照，在高并发命中较短 K 时确认：

- draft 前向次数随 K 减少。
- target verify 总 token 数随 `并发 × K` 减少。
- TPOT 或吞吐获得正收益。
- 接受长度下降造成的有效 token 损失没有抵消计算收益。

## 8. 无同步阻塞验证

代码静态检查已确认本次 diff 未新增：

- `.item()`
- `torch.npu.synchronize()`
- stream synchronize
- device-to-host 决策拷贝

NPU 上仍需使用 profiler 验证：

- Scheduler plan 解析全部发生在 CPU。
- draft active K 不依赖 NPU tensor 值。
- Async Scheduler 的 CPU 工作被现有异步流水遮掩。
- 动态版本没有新增 host wait、stream wait 或 graph recapture。

## 9. 长稳和异常验证

建议至少执行 8 小时预检和 24 小时正式长稳：

- 并发在各策略范围间周期切换。
- 边界并发反复变化。
- 短请求和长请求混合。
- 高、低接受率交替。
- 请求取消、超时和客户端断连。

观察：

- 服务进程退出、NPU error、ACL graph replay error。
- KV cache、host memory、device memory 是否持续增长。
- graph 数量是否在启动后保持稳定。
- 请求完成数、失败数和超时数。
- 切换后 active K 与策略是否一致。

## 10. 当前限制

- 尚未在真实 910B3/910C 上执行模型 E2E。
- PCP/DCP、DP、LoRA、混合 prefill/decode 等组合虽然保留兼容路径，仍必须分别做 NPU 回归。
- Phase 1 只实现静态并发区间策略；接受长度、上下文长度和在线寻优属于 Phase 2。
- 当前热更新指 step 间按已加载策略切换 K；运行中修改策略配置本身的控制面接口不在本次提交范围。
