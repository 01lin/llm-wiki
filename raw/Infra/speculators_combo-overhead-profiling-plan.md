# 组合投机单步开销性能分析与打点方案

## 1. 背景

当前实测场景：

- 模型：`qwen3.5 122B`
- 硬件：昇腾 `910B3` 单机
- 框架：`vllm-ascend` `model_runner_v1`
- 模式：MTP 投机 + suffix 组合投机，开启 MTP 图模式
- 现象：多并发下组合投机 TPOT 明显慢于纯 MTP

关键样例：

```text
MTP7 + suffix3, bs=8, TPOT=17.64ms, 接受距离=7.32
MTP10,          bs=8, TPOT=14.92ms, 接受距离=7.77
```

这组结果说明：

- acceptance 差异不足以解释 TPOT 慢接近 18%
- 问题更可能在单步固定开销、调度开销、同步开销或图路径退化

本方案目标是把组合投机单步开销拆开，定位：

- 慢在 target verify
- 慢在 rejection sampling
- 慢在 suffix 构造
- 慢在 MTP drafter
- 慢在 metadata / graph replay / copy / overlap

---

## 2. 当前代码链路定位

核心路径在：

- `vllm-ascend/vllm_ascend/worker/model_runner_v1.py`
- `vllm-ascend/vllm_ascend/spec_decode/eagle_proposer.py`
- `vllm-ascend/vllm_ascend/spec_decode/suffix_proposer.py`
- `vllm-ascend/vllm_ascend/compilation/acl_graph.py`

## 2.1 `execute_model()`：target verify 与 logits

入口：

- `NPUModelRunner.execute_model()`

关键阶段：

- `_update_states`
- `_prepare_inputs`
- `_determine_batch_execution_and_padding`
- `_build_attention_metadata`
- `_preprocess`
- target `_model_forward`
- target `compute_logits`

代码位置：

- `model_runner_v1.py:1486`
- `model_runner_v1.py:1529`
- `model_runner_v1.py:1571`
- `model_runner_v1.py:1593`
- `model_runner_v1.py:1676`
- `model_runner_v1.py:1692`
- `model_runner_v1.py:1743`
- `model_runner_v1.py:1803`

## 2.2 `sample_tokens()`：sampling、bookkeeping、draft 生成

入口：

- `NPUModelRunner.sample_tokens()`

关键阶段：

- `_sample`
- rejection sampler
- `_bookkeeping_sync`
- `propose_draft_token_ids`
- `_copy_draft_token_ids_to_cpu`
- `finalize_kv_connector`
- `async_state_update`

代码位置：

- `model_runner_v1.py:1849`
- `model_runner_v1.py:1901`
- `model_runner_v1.py:1930`
- `model_runner_v1.py:1946`
- `model_runner_v1.py:2002`
- `model_runner_v1.py:2036`
- `model_runner_v1.py:2060`

特别注意：

- 当前代码中 `_bookkeeping_sync()` 在 `draft_token` 之前执行
- 代码注释里也保留了 TODO：`_bookkeeping_sync is moved after propose_draft_token_ids`
- 这意味着多并发下可能存在 CPU bookkeeping 影响下一轮 draft 准备的情况

## 2.3 `propose_draft_token_ids()`：MTP drafter 输入准备

入口：

- `NPUModelRunner.propose_draft_token_ids()`

关键阶段：

- `prepare_next_token_ids_padded`
- `_copy_valid_sampled_token_count`
- `prepare_inputs_padded`
- target hidden states gather
- drafter `_propose`

代码位置：

- `model_runner_v1.py:1309`
- `model_runner_v1.py:1351`
- `model_runner_v1.py:1385`
- `model_runner_v1.py:1444`
- `model_runner_v1.py:1463`

## 2.4 `eagle_proposer._propose()`：MTP 图模式主路径

入口：

- `AscendSpecDecodeBaseProposer._propose()`

关键阶段：

- `set_inputs_first_pass`
- cudagraph dispatch
- `_sync_metadata_across_dp`
- graph full padding
- slot / seq_lens / query_start_loc copy
- metadata builder
- multi-step metadata 构建
- run draft graph
- update graph params

代码位置：

- `eagle_proposer.py:524`
- `eagle_proposer.py:557`
- `eagle_proposer.py:578`
- `eagle_proposer.py:586`
- `eagle_proposer.py:601`
- `eagle_proposer.py:678`
- `eagle_proposer.py:691`
- `eagle_proposer.py:797`
- `eagle_proposer.py:817`
- `eagle_proposer.py:845`

特别注意：

- `eagle_proposer.py:691` 附近有注释：`FIXME: The below two ops cause synchronization`
- 这里很可能是组合投机多并发下的固定开销来源之一

## 2.5 `_run_merged_draft()`：MTP 每步模型执行

入口：

- `AscendSpecDecodeBaseProposer._run_merged_draft()`

关键阶段：

- first draft model forward
- first draft `compute_logits`
- remaining draft loop
- per-step input copy
- per-step model forward
- per-step `compute_logits`
- per-step argmax

代码位置：

- `eagle_proposer.py:853`
- `eagle_proposer.py:886`
- `eagle_proposer.py:928`
- `eagle_proposer.py:957`
- `eagle_proposer.py:975`
- `eagle_proposer.py:1004`
- `eagle_proposer.py:1040`
- `eagle_proposer.py:1061`

## 2.6 ACL graph replay

入口：

- `ACLGraphWrapper.__call__()`

关键阶段：

- graph dispatch
- first capture
- replay 前 synchronize
- graph replay

代码位置：

- `acl_graph.py:110`
- `acl_graph.py:130`
- `acl_graph.py:199`
- `acl_graph.py:210`
- `acl_graph.py:212`

特别注意：

- `acl_graph.py:210` 在部分条件下会执行 `torch.npu.current_stream().synchronize()`
- 这会破坏 overlap，并且在多并发下可能被放大
- 需要确认 MTP draft graph 是否命中跳过同步的条件

---

## 3. 性能假设

当前 `MTP7+suffix3` 比 `MTP10` 慢，优先验证以下假设。

## 3.1 组合投机引入额外 host-side 开销

可能来源：

- suffix 检索 / 匹配
- suffix token 拼接
- `valid_len/source_map` 维护
- draft token copy 到 CPU
- bookkeeping 的 request 状态更新

验证指标：

- `T_suffix_build_cpu`
- `T_hybrid_compose_cpu`
- `T_bookkeeping_sync_cpu`
- `T_copy_draft_to_cpu`

## 3.2 MTP 图模式下存在额外同步

可能来源：

- ACL graph replay 前 `synchronize`
- metadata builder 内部同步
- block table / seq_lens / query_start_loc copy
- graph params update 与 replay 顺序约束

验证指标：

- `T_aclgraph_pre_replay_sync`
- `T_aclgraph_replay`
- `T_draft_metadata_build`
- `T_update_full_graph_params`

## 3.3 多并发导致 draft graph padding 或 batch descriptor 变化

可能来源：

- `num_tokens` 被 pad 到更大 graph bucket
- `num_reqs_padded` 大于真实 `num_reqs`
- `batch_descriptor` 选择与纯 MTP 不一致
- combo 路径触发额外 `query_start_loc` padding

验证指标：

- `num_reqs`
- `num_reqs_padded`
- `num_tokens`
- `num_input_tokens`
- `batch_descriptor`
- `aclgraph_runtime_mode`
- `num_tokens_across_dp`

## 3.4 verify token 数与预期不一致

你已经观察到：

- `MTP M + suffix N` 的 verify 平均成本在 `M` 到 `M+N` 之间
- suffix 不足 `N` 时，后续 token 会被舍弃

需要确认：

- `num_draft_tokens`
- `num_sampled_tokens`
- `logits_indices` 长度
- `target_logits_indices` 长度
- `num_tokens_unpadded`
- `num_tokens_padded`

验证目标：

- suffix miss 时 verify 是否真的少算
- suffix hit 时 verify token 数是否和 `M + suffix_hit_len` 对齐
- combo 的 padding 是否把动态有效长度重新放大

## 3.5 overlap 被组合路径破坏

可能来源：

- suffix 必须等待 MTP 输出后才能做
- suffix 构造发生在关键路径
- bookkeeping 在 draft 前执行
- CPU 状态更新阻塞下一轮 draft

验证指标：

- `T_sample_to_draft_gap`
- `T_bookkeeping_before_draft`
- `T_draft_start_delay`
- NPU stream idle gap

---

## 4. 打点原则

建议分两层打点。

## 4.1 L1：轻量 always-on 聚合打点

目标：

- 在线低开销定位大方向
- 可以跑完整 benchmark
- 输出每阶段平均值和 P50/P90/P99

方式：

- CPU 阶段用 `time.perf_counter_ns()`
- NPU 阶段用 `torch.npu.Event(enable_timing=True)` 或同等 NPU event
- 避免每步 `synchronize`
- 只在窗口结束时聚合统计

适合阶段：

- `prepare_input`
- `target_forward`
- `target_compute_logits`
- `sample`
- `bookkeeping`
- `draft_total`
- `draft_prepare`
- `draft_metadata`
- `draft_graph`
- `suffix_build`
- `copy_to_cpu`

## 4.2 L2：重型 timeline profiler

目标：

- 验证 stream idle、graph replay、kernel timeline
- 只跑短窗口

方式：

- 使用 Ascend profiler / `torch_npu` profiler / msprof
- 配合 `record_function` 标签
- 只采集稳定 decode 区间，例如 warmup 后 100-300 step

适合问题：

- graph replay 前后是否有空洞
- NPU stream 是否等待 CPU
- combo 路径是否多了 memcpy / sync
- MTP graph 是否真的按预期 replay

---

## 5. 建议打点字段

## 5.1 每步基础字段

```text
step_id
batch_size
num_reqs
num_spec_reqs
method
concurrency
num_tokens_unpadded
num_tokens_padded
num_input_tokens_draft
num_reqs_padded
batch_descriptor
aclgraph_runtime_mode
use_async_scheduling
use_async_spec_decode
```

## 5.2 投机字段

```text
mtp_len
suffix_max_len
suffix_hit_len_sum
suffix_hit_len_avg
suffix_hit_len_hist[0..N]
num_draft_tokens_sum
num_accepted_tokens_sum
accept_len_avg
acceptance_by_pos
acceptance_by_source
```

## 5.3 时间字段

```text
T_execute_total
T_prepare_input
T_prepare_inputs
T_determine_batch
T_build_attn_metadata
T_preprocess
T_target_forward
T_target_compute_logits
T_sample_total
T_rejection_sample
T_bookkeeping_sync
T_draft_total
T_prepare_next_token_ids
T_prepare_inputs_padded
T_draft_metadata_build
T_draft_graph_dispatch
T_draft_graph_replay
T_update_full_graph_params
T_suffix_build
T_hybrid_compose
T_copy_valid_sampled_count
T_copy_draft_to_cpu
T_finalize_kv_connector
T_async_state_update
```

## 5.4 图模式字段

```text
target_cudagraph_mode
target_batch_descriptor
draft_aclgraph_mode
draft_batch_descriptor
draft_graph_cache_hit
draft_graph_capture_count
draft_graph_replay_count
draft_pre_replay_sync_count
draft_pre_replay_sync_time
```

---

## 6. 轻量打点实现方案

## 6.1 增加局部 profile helper

建议新增一个很薄的 helper，例如：

```text
vllm-ascend/vllm_ascend/utils/spec_profile.py
```

核心能力：

- `span_cpu(name)`
- `span_npu(name)`
- `add_counter(name, value)`
- `add_hist(name, value)`
- 按 step 聚合
- 每 N step 打印一行 JSON

示例：

```python
with spec_prof.span_cpu("bookkeeping_sync"):
    out = self._bookkeeping_sync(...)

with spec_prof.span_npu("target_forward"):
    hidden_states = self._model_forward(...)
```

注意：

- `span_npu` 不能每次退出都 synchronize
- 应记录 start/end event
- 在 flush 时统一同步和计算 elapsed

## 6.2 `execute_model()` 打点

建议插入：

- `_update_states`
- `_prepare_inputs`
- `_determine_batch_execution_and_padding`
- `_build_attention_metadata`
- `_preprocess`
- `_model_forward`
- `compute_logits`

重点采集：

- target verify token 数
- padding 前后 token 数
- graph mode / batch descriptor

## 6.3 `sample_tokens()` 打点

建议插入：

- `_sample`
- `_bookkeeping_sync`
- `propose_draft_token_ids`
- `_copy_draft_token_ids_to_cpu`
- `finalize_kv_connector`
- `_update_states_after_model_execute`

重点判断：

- draft 是否被 bookkeeping 阻塞
- copy_to_cpu 是否放大
- async_state_update 是否进入关键路径

## 6.4 `propose_draft_token_ids()` 打点

建议插入：

- `prepare_next_token_ids_padded`
- `_copy_valid_sampled_token_count`
- `prepare_inputs_padded`
- target token / hidden gather
- `drafter._propose`

重点判断：

- padded batch 是否比 pure MTP 多开销
- valid count copy 是否同步
- token_indices 生成是否异常

## 6.5 `eagle_proposer._propose()` 打点

建议插入：

- `set_inputs_first_pass`
- cudagraph dispatch
- `_sync_metadata_across_dp`
- graph full padding
- slot/seq/query copy
- `builder.build`
- multi-step metadata loop
- `_update_full_graph_params_if_needed`
- `run_draft`

这里是多并发问题最需要细看的区域。

## 6.6 `_run_merged_draft()` 打点

建议插入：

- first model forward
- first compute logits
- first argmax
- per draft step:
  - input copy
  - model forward
  - compute logits
  - argmax

如果不想每步都打太细，至少需要：

- first step
- remaining steps total
- per-step P50/P90

## 6.7 `ACLGraphWrapper.__call__()` 打点

建议插入：

- graph cache miss / capture
- pre replay synchronize
- graph replay

特别要记录：

- `is_draft_model`
- `runtime_mode`
- `batch_descriptor`
- 是否执行了 `torch.npu.current_stream().synchronize()`

如果组合投机比纯 MTP 多了 replay sync，这会非常明显。

---

## 7. 建议输出格式

轻量打点建议每 100 或 500 step 输出一行 JSONL：

```json
{
  "mode": "mtp7_suffix3",
  "concurrency": 8,
  "steps": 500,
  "tpot_ms": 17.64,
  "accept_len_avg": 7.32,
  "target_forward_ms_p50": 8.1,
  "target_forward_ms_p90": 8.8,
  "sample_ms_p50": 0.6,
  "bookkeeping_ms_p50": 0.4,
  "draft_total_ms_p50": 5.2,
  "draft_metadata_ms_p50": 1.1,
  "draft_graph_ms_p50": 3.4,
  "suffix_build_ms_p50": 0.5,
  "copy_to_cpu_ms_p50": 0.2,
  "draft_pre_replay_sync_count": 500,
  "draft_pre_replay_sync_ms_p50": 0.7,
  "num_tokens_padded_avg": 80,
  "num_tokens_unpadded_avg": 73,
  "suffix_hit_len_hist": [0.2, 0.3, 0.3, 0.2]
}
```

---

## 8. 对照实验矩阵

优先做最小矩阵，别一开始扫太大。

## 8.1 主对照

```text
MTP7
MTP10
suffix7
suffix10
MTP7 + suffix1
MTP7 + suffix2
MTP7 + suffix3
```

## 8.2 并发维度

```text
bs=1
bs=2
bs=4
bs=8
```

## 8.3 graph 维度

```text
MTP graph on
MTP graph off
```

如果 graph off 下组合不慢，graph on 下组合慢，优先查 graph replay / metadata / padding。

## 8.4 suffix 维度

```text
suffix disabled
suffix enabled but force miss
suffix enabled with real match
suffix enabled with precomputed/static match
```

这个实验可以拆开：

- suffix 检索开销
- suffix 命中带来的 verify token 变化
- hybrid compose 开销

---

## 9. 下一步建议

## 9.1 第一优先级：解释 18% 慢在哪里

先只关注：

```text
MTP7 + suffix3, bs=8
MTP10,          bs=8
```

必须拆出：

- `T_target_forward`
- `T_sample`
- `T_bookkeeping`
- `T_draft_total`
- `T_draft_metadata`
- `T_draft_graph`
- `T_suffix_build`
- `T_copy_to_cpu`
- `T_graph_sync`

目标：

- 找到 2.72ms 差距的主要来源

## 9.2 第二优先级：确认 verify 成本模型

对每步记录：

- `M`
- `suffix_hit_len`
- `num_draft_tokens`
- `logits_indices_len`
- `num_tokens_unpadded`
- `num_tokens_padded`
- `T_target_forward`

目标：

- 验证 `T_verify` 是否真的随 `M + suffix_hit_len` 变化
- 验证 padding 是否把有效长度优势吃掉

## 9.3 第三优先级：验证 graph replay 同步

在 `ACLGraphWrapper.__call__()` 里统计：

- replay 次数
- synchronize 次数
- synchronize 时间
- draft / target 各自的 batch descriptor

目标：

- 判断 MTP+suffix 是否触发更多同步或 graph bucket miss

## 9.4 第四优先级：收缩策略

在多并发问题查清前，建议线上策略暂时保守：

- 单并发或低并发保留组合投机
- `bs >= 8` 先回退纯 MTP
- suffix tail 先测 `1/2`，暂停大规模使用 `suffix3`

---

## 10. 预期判断逻辑

如果发现：

- `T_draft_graph` 变大
- `T_draft_metadata` 变大
- `T_graph_sync` 变大

说明问题在 MTP 图和 metadata 路径。

如果发现：

- `T_suffix_build` 变大
- `T_hybrid_compose` 变大
- `T_bookkeeping` 变大

说明问题在 CPU / host 侧组合路径。

如果发现：

- `T_target_forward` 变大
- `num_tokens_padded` 明显变大

说明问题在 verify token 数或 graph padding。

如果发现：

- 各段单独都不大
- 但 timeline 上有空洞

说明问题在 overlap 被破坏，需要看 stream wait / event / synchronize。

---

## 11. 当前最可疑点

基于代码结构，当前最可疑的点按优先级排序：

1. `eagle_proposer.py:691` 附近 metadata builder 触发同步
2. `acl_graph.py:210` replay 前 `torch.npu.current_stream().synchronize()`
3. `sample_tokens()` 中 `_bookkeeping_sync()` 在 draft 前执行
4. combo 路径导致 `num_tokens_padded / batch_descriptor` 大于纯 MTP
5. suffix tail 构造或 hybrid compose 在多并发下串行化
6. `_copy_draft_token_ids_to_cpu` 或 valid count copy 放大

建议先用轻量打点把这 6 个点排一遍。
