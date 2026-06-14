# vLLM / vLLM-Ascend 三模块下钻：Scheduler 闭环 · DSA backend · execute_model 主循环

> 生成时间：2026-06-14
> 范围：在 [[20260614-163750-vllm-vs-vllm-ascend-目录与架构设计-分析]] 的整体架构基础上，下钻三个最关键的运行时模块。
> 证据基线：`vllm` @ `0d2961229`、`vllm-ascend` @ `8afdf356`（2026-06-13 快照）。所有行号均为该快照实测。

---

## 模块 A：vLLM Scheduler 调度闭环

### A.1 核心心智模型：没有 prefill/decode 之分

`schedule()` 顶部注释（scheduler.py:359-369）给出了 V1 调度的底层逻辑——**调度器眼里没有「prefill 阶段」和「decode 阶段」**，只有每个请求的两个数：

- `num_computed_tokens`：已算的 token 数
- `num_tokens_with_spec = len(prompt) + len(output) + len(spec_token_ids)`：需要算到的 token 数

每步调度就是「给请求分配 token，让 `num_computed_tokens` 追上 `num_tokens_with_spec`」。这一个统一抽象同时覆盖了 chunked prefill、prefix caching、投机解码、未来的 jump decoding——**这是 V1 相对 V0 最大的设计简化**。

### A.2 `schedule()` 主流程（scheduler.py:357+）

```
schedule():
  current_step += 1
  token_budget = max_num_scheduled_tokens          # 全局 token 预算
  kv_cache_manager.new_step_starts()

  # 阶段1：先调度 RUNNING 队列（decode/continued prefill 优先）
  while running 且 token_budget > 0:
    req = running[i]
    # 跳过条件（async scheduling 关键）：
    #  - num_output_placeholders 已确保达到 max_tokens → 不多排一步
    #  - current_step < req.next_decode_eligible_step → V2+PP 节流
    num_new_tokens = num_tokens_with_spec + placeholders − num_computed_tokens
    num_new_tokens = min(..., long_prefill_threshold, token_budget, max_model_len 余量)
    # 编码器输入预算、mamba 块对齐切分...
    while True:                                      # KV 分配 + 抢占循环
      new_blocks = kv_cache_manager.allocate_slots(req, num_new_tokens, num_lookahead_tokens)
      if new_blocks is not None: break              # 分配成功
      # 失败 → 抢占：PRIORITY 策略抢最低优先级，否则抢队尾（LIFO）
      preempt(lowest or running.pop())

  # 阶段2：再调度 WAITING 队列（新请求/恢复请求，prefix cache 匹配）
  # 阶段3：组装 SchedulerOutput（scheduled_new/resumed/running_reqs,
  #         num_scheduled_tokens, scheduled_spec_decode_tokens, req_to_new_blocks）
```

关键设计点：
- **token 预算是全局的**（不是 per-request），prefill 与 decode 抢同一个预算池——这是 continuous batching 的核算单位。
- **KV 分配与抢占在同一个 while 循环里**（scheduler.py:478+）：分配失败立即抢占再重试，抢占释放的 token 退回 `token_budget`。两种策略：`PRIORITY`（抢 `(priority, arrival_time)` 最大者）vs 默认 FCFS（`running.pop()` 抢队尾）。
- **`continue` 而非 `break`**（scheduler.py:464 注释）：某请求排不下时跳过它而非中断整轮——故意放宽 FCFS，让低优请求有机会被排，提升预算利用率。
- **async scheduling 的提前感知**（scheduler.py:395-411）：用 `num_output_placeholders` 推断「上一步是否已确定达到 max_tokens」，避免多排一步浪费——因为 async 下调度器在上一步结果回来前就排了下一步。

### A.3 `update_from_output()` 回流（scheduler.py:1395+）

模型执行完后，结果经此闭环回灌调度器状态：

```
update_from_output(scheduler_output, model_runner_output):
  sampled_token_ids = output.sampled_token_ids       # 设备采样结果（已 D2H）
  # routed_experts 持久化（MoE，须在 per-req 读取前）
  for req_id, num_scheduled in num_scheduled_tokens.items():   # 热循环（注释自警: 1K+ 请求时是瓶颈）
    request = requests[req_id]
    generated = sampled_token_ids[req_index]
    # === 投机解码接受/拒绝核算（核心）===
    if scheduled_spec_token_ids and generated:
      num_draft = len(scheduled_spec_token_ids)
      num_accepted = max(len(generated) − num_sampled_per_step, 0)
      num_rejected = num_draft − num_accepted
      request.num_computed_tokens   −= num_rejected   # 拒绝的 token 回退计算指针
      request.num_output_placeholders −= num_rejected  # async 占位同步回退
      make_spec_decoding_stats(...)                    # accept_len 统计
    # stop 检查、structured output grammar 推进、logprobs、routed_experts 读取
    _update_request_with_output(request, new_token_ids) → stopped?
```

闭环的本质（与投机解码协同）：
- **接受长度 = `len(generated) − num_sampled_per_step`**（scheduler.py:1477）。draft 排了 `num_draft` 个，实际接受 `num_accepted`，拒绝的部分**回退 `num_computed_tokens`**——这样下一轮 `schedule()` 的 `num_new_tokens` 自然把拒绝的位置重新排进去。投机解码的正确性就靠这个「排 → 验 → 回退指针」的闭环维持。
- **async scheduling 的占位同步**：`num_output_placeholders` 与 `num_computed_tokens` 同步回退——因为 async 下调度提前于结果，占位符是「乐观预排」的记账，拒绝时必须一起修正。
- 热循环明确标注为性能敏感点（scheduler.py:1446 注释），昇腾侧的调度增强（`core/scheduler_dynamic_batch`、`recompute_scheduler`）正是围绕此处。

### A.4 与执行面的契约

`SchedulerOutput`（控制面 → 执行面）携带：`scheduled_*_reqs`、`num_scheduled_tokens`(dict)、`scheduled_spec_decode_tokens`(dict)、`req_to_new_blocks`(KV 块表)。
`ModelRunnerOutput`（执行面 → 控制面）携带：`sampled_token_ids`、`logprobs`、`req_id_to_index`、`spec` 相关、`routed_experts`、`kv_connector_output`。
**两个 dataclass 就是整个控制/执行解耦的接口契约**（跨进程 IPC 序列化的就是它们）。

---

## 模块 B：vLLM-Ascend AscendDSABackend（DeepSeek 稀疏注意力）

### B.1 三级元数据结构（dsa_v1.py:239-330）

DSA 的元数据按「公共 + prefill 专属 + decode 专属」三级组织，对应混合 batch（同一步既有 prefill 又有 decode）：

```
AscendDSAMetadata（公共，dsa_v1.py:277）
├─ num_actual_tokens / slot_mapping / query_start_loc / seq_lens / block_tables
├─ sin / cos（顶层 RoPE 缓存）
├─ num_decodes / num_decode_tokens / num_prefills   ← 混合 batch 的切分点
├─ attn_state（默认 ChunkedPrefill）
├─ hadamard / start_pos（dsv4 indexer 专属）
├─ prefill: AscendDSAPrefillMetadata | None  （dsa_v1.py:224）
│    └─ attn_mask, query/seq/context_lens, compress_sin/cos, sas/qli_metadata,
│       cu_c4/c128_cmp_seqlen_list（压缩 KV 的累积序列长度表）
└─ decode: AscendDSADecodeMetadata | None  （dsa_v1.py:251）
     └─ input_positions, block_table, seq_lens_list, max_seqlen_kv/q,
        compress_sin/cos, batch_seq_mask, sas/qli_metadata
```

设计要点：
- **prefill 与 decode 元数据物理分离**（两个独立 dataclass），因为两条路径的 KV 访问模式、mask、压缩序列布局完全不同——DSA 的稀疏性使 decode 路径有专属的 `sas_metadata`（sparse attention selection）和 `qli_metadata`（query-level index）。
- **`compress_sin/cos` 与顶层 `sin/cos` 并存**：DSA 的 Compressor 对 KV 做压缩，压缩后的 KV 有独立的 RoPE。
- builder 类属性 `aclgraph_support = AttentionCGSupport.UNIFORM_BATCH`（dsa_v1.py:337）：声明该后端**只在 uniform batch（纯 decode、shape 一致）下支持 ACL Graph**——这正是极致时延优化要把 decode 收敛成固定 shape 桶的根因。

### B.2 `build()` 的增量缓存设计（dsa_v1.py:506+）

`build()` 用 `common_ratio_to_sas_metadata` 字典做**跨调用增量缓存**：

```
build(common_prefix_len, common_attn_metadata, **kwargs):
  if common_ratio_to_sas_metadata["num_decodes"] is None:    # 首次：全量计算
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens
        = split_decodes_and_prefills(common_attn_metadata, decode_threshold)
    input_positions = positions[:num_input_tokens].long()
    cos, sin = get_cos_and_sin_dsa(input_positions, <decode_only?>)   # RoPE 计算
    query_lens = query_start_loc_cpu[1:] − query_start_loc_cpu[:-1]
    # 全部塞进 common_ratio_to_sas_metadata 缓存
  else:                                                        # 复用：直接读缓存
    num_decodes, ... = common_ratio_to_sas_metadata[...]
```

底层逻辑：DSA 一次 verify step 内 target 与多步 draft 会**多次 build 元数据**（每次前向一次），把不随 draft step 变的量（decode/prefill 切分、RoPE cos/sin、query_lens）缓存进 `common_ratio_to_sas_metadata`，draft 步直接复用——这是 runtime 侧「减少每步 host 重复计算」的具体抓手，也是 [[20260611-010803-vllm-ascend-deepseek-v4-runtime算子协同优化-方案设计]] 里 H4（元数据构建）优化的现有基础。

### B.3 forward 三条路径（dsa_v1.py:1566+）

```
forward(layer_name, hidden_states, kv_cache, attn_metadata, ...):
  has_prefill = num_prefills > 0; has_decode = num_decodes > 0
  # FlashComm V1（SP）：延迟 allgather —— 纯 prefill 时本地算完再 allgather 小中间量
  if prefill_comm_compute_overlap and 纯prefill: 不提前 gather
  else: hidden_states = maybe_all_gather_and_maybe_unpad(...)
  prefill_hs = hidden_states[decode_tokens:actual_tokens]
  decode_hs  = hidden_states[:decode_tokens]

  if has_prefill: o[decode:actual] = _forward_prefill(...)   # dsa_v1.py:1866
  if has_decode:  o[:decode]       = _forward_decode(...)    # dsa_v1.py:2186
  # 统一尾部：partial rotary mul → wo_a → wo_b（A5 走 MXFP4 动态量化 matmul）
```

### B.4 多流 CV 并行 prolog（dsa_v1.py:1691，算子协同的精华）

`_mla_prolog_multistream` 把 MLA prolog 拆成 3 段 + 尾，主流（Cube/矩阵）与辅流（Vector/向量）交错：

```
Part1: q_quant[V] → q_a_down[C]   ||  kv_quant[V]
Part2: q_norm[V] + q_b_quant[V]   ||  kv_matmul[C]
Part3: q_b_matmul[C]              ||  kv_norm[V] + rope[V] + scatter[AIV]
Tail:  q_rms[V] + rope[V]（仅尾部一次 wait_stream 同步）
```

设计取舍：每段内主辅流数据自洽（无跨流依赖），段间不 sync，只在尾部 `wait_stream` 一次确保 scatter 完成——把昇腾 Cube/Vector/AIV 三类计算单元**在同一段时间窗内并行打满**，这是 vllm-ascend 算子侧已达到 TokenSpeed kernel 水平的直接证据。

---

## 模块 C：NPUModelRunner execute_model → sample_tokens 主循环

### C.1 两段式拆分（async scheduling 的核心机制）

NPU runner 把单步执行拆成两个方法，由 `ExecuteModelState`（NamedTuple，gpu_model_runner.py:402）串联：

```
execute_model(scheduler_output):                   # 第一段：launch 前向
  _update_states(scheduler_output)                 # 持久 batch 状态更新
  _prepare_inputs(...)                             # InputBatch → 张量（H2D）
  model.forward(...)                               # 前向（ACL Graph replay）→ logits
  self.execute_model_state = ExecuteModelState(    # ★ 存ephemeral状态，不立即采样
      scheduler_output, logits, spec_decode_metadata,
      hidden_states, sample_hidden_states, attn_metadata, positions, ...)
  if deferred_state_corrections_fn: fn()           # ★ 此处才等上一步的修正（不阻塞 async）
  return None                                       # ★ 关键：返回 None，前向已 launch

sample_tokens(grammar_output):                      # 第二段：采样 + 记账 + draft
  unpack ExecuteModelState                          # 取出 launch 时存的状态
  if grammar: apply_grammar_bitmask(... 经 CPU ...) # 结构化输出（NPU 不支持 compile 优化）
  sampler_output = _sample(logits, spec_decode_metadata)   # 采样
  if need_accepted_tokens: sampling_done_event.record()    # ★ 记 event 而非 D2H 同步
  _bookkeeping_sync(...)                            # 接受/拒绝核算、valid_sampled_token_ids
  if speculative_config:                            # MTP/Eagle draft 提议
    if use_padded_batch: propose_draft(sampler_output.sampled_token_ids)  # EAGLE 用设备张量
    else: propose_draft(valid_sampled_token_ids)                          # ngram 用 CPU 张量
    finalize_kv_connector()
```

### C.2 为什么这样拆——async scheduling 的底层逻辑

- **`execute_model` 返回 `None` 时前向已 launch**（model_runner_v1.py:2334）：调度器进程不必等结果，可立即 `schedule()` 下一批，CPU 调度与 NPU 前向**重叠**。这是 `use_async_scheduling` 整套机制的支点。
- **`deferred_state_corrections_fn`**（model_runner_v1.py:2332）：上一步的状态修正（如拒绝回退）推迟到本步前向 launch 之后才执行——「先把活派出去，再处理上一单的纠偏」，不让纠偏阻塞流水。
- **`sampling_done_event.record()` 而非 D2H**（model_runner_v1.py:2392）：接受长度需要回 host 才能驱动调度，但这里只**记 event**，真正的 D2H 推迟/异步化——对应 [[20260611-010803-vllm-ascend-deepseek-v4-runtime算子协同优化-方案设计]] 里 H7（接受判定 D2H 同步）的去同步点。
- **draft 提议的双路径**（model_runner_v1.py:2447-2459）：EAGLE 类用设备侧 `sampled_token_ids` 直接提议（不等 bookkeeping），ngram 类用 CPU 侧 `valid_sampled_token_ids`（须等 bookkeeping）——体现「能用设备张量就不等 host」的设计偏好。

### C.3 与 ACL Graph / xlite 的衔接

- 默认路径：`model.forward` 内部经 `ACLGraphWrapper.__call__`（acl_graph.py:152），replay 前 `update_full_graph_params` 在独立 update_stream 上原位刷新 attn 参数，用 `ExternalEvent` 与上一轮 replay 串序（acl_graph.py:259-262）。
- xlite 路径：`XliteModelRunner.load_model` 把 model 包成 `XliteWrapper`，`execute_model` 调到的 `model.forward` 变成 `xlite_model.forward(rt, input_ids, attn_meta, kv_caches, ...)` 一次 C++ 整图调用（xlite.py:781）——但当前 xlite 不支持 DSA、与 spec decode 互斥。

### C.4 一步的完整闭环（端到端）

```
Scheduler.schedule() ─SchedulerOutput─> [IPC] ─> NPUModelRunner.execute_model()
   ├─ _update_states / _prepare_inputs (H2D)
   ├─ model.forward (ACLGraph replay / xlite C++) → logits
   └─ 存 ExecuteModelState, return None  ──┐ (前向已 launch，调度器可继续)
NPUModelRunner.sample_tokens()  <───────────┘
   ├─ _sample → sampler_output
   ├─ _bookkeeping_sync → valid_sampled_token_ids (接受/拒绝)
   ├─ propose_draft_token_ids (MTP/EAGLE 下一轮草稿)
   └─ ModelRunnerOutput ─[IPC]─> Scheduler.update_from_output()
        └─ num_computed_tokens −= num_rejected (回退指针，闭环)
   ─> OutputProcessor (detokenize/stop/stream) ─> AsyncLLM ─> 客户端
```

---

## 三模块协同的一句话总结

Scheduler 用「统一 token 追赶 + 接受长度回退指针」维持投机解码的**逻辑闭环**；DSA backend 用「三级元数据 + 增量缓存 + 多流 CV 并行」实现稀疏注意力的**算子协同**；execute_model 两段式拆分用「launch 后返回 None + event 代替 D2H」实现 CPU 调度与 NPU 前向的**流水重叠**。三者在 `SchedulerOutput`/`ModelRunnerOutput` 两个契约上对齐，构成 vLLM-Ascend DeepSeek V4 的单步运行时闭环。

---

## 源码证据索引

| 主题 | 位置 |
|------|------|
| Scheduler 统一 token 抽象 | `vllm/v1/core/sched/scheduler.py:359-369` |
| schedule() 主流程 / KV 分配抢占 | `scheduler.py:357, 478+` |
| async scheduling 提前感知 | `scheduler.py:395-411` |
| update_from_output 接受核算 | `scheduler.py:1395, 1446(注释), 1477` |
| DSA 三级元数据 | `vllm-ascend/vllm_ascend/attention/dsa_v1.py:224, 251, 277` |
| DSA build 增量缓存 | `dsa_v1.py:506+` |
| DSA forward 三路径 | `dsa_v1.py:1566+` |
| DSA 多流 CV 并行 prolog | `dsa_v1.py:1691+` |
| execute_model 两段式 / ExecuteModelState | `worker/model_runner_v1.py:2314-2334`、`vllm/v1/worker/gpu_model_runner.py:402` |
| sample_tokens / event 代替 D2H / draft 双路径 | `model_runner_v1.py:2336-2459` |
| ACL Graph replay 串序 | `vllm-ascend/vllm_ascend/compilation/acl_graph.py:259-262` |
