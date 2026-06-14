# DeepSeek V4 attention 端到端代码级调用栈分析

> 生成时间：2026-06-14
> 范围：以 DeepSeek V4 的一次 attention 计算为例，串起 engine → executor → worker → model_runner → 模型 forward → DecoderLayer → attention → DSA impl → 算子 的完整 e2e 调用栈，逐跳给出真实函数名与行号。
> 证据基线：`vllm` @ `0d2961229`、`vllm-ascend` @ `8afdf356`（2026-06-13 快照）。**所有行号已用 `grep -n` 逐行重新校验对齐当前本地代码（2026-06-14 修订）。**
> 关联：[[20260614-163750-vllm-vs-vllm-ascend-目录与架构设计-分析]]、[[20260614-164853-vllm-vllm-ascend-三模块下钻-Scheduler闭环-DSA-execute-model-分析]]、[[20260614-175025-vllm-ascend-csrc算子链路与xlite机制-分析]]

---

## 0. 一图速览（13 跳调用栈）

```
[Engine 进程]
 1. EngineCore.step()                              vllm/v1/engine/core.py:452（execute_model:464）
 2. Executor.execute_model()                       vllm/v1/executor/abstract.py:37/48/210（Multiproc/Ray）
      └─ IPC → 各设备进程
[Worker 进程 / 每 NPU]
 3. Worker.execute_model()                         vllm/v1/worker/gpu_worker.py:806（调 model_runner:866，NPUWorker 继承）
 4. NPUModelRunner.execute_model()                 vllm-ascend/.../model_runner_v1.py:1904
      ├─ _prepare_inputs() → 构建 attn_metadata
      └─ set_ascend_forward_context(attn_metadata)  model_runner_v1.py:2233 ★ 注入全局上下文（旁路传递）
 5. _model_forward() → self.model(...)             model_runner_v1.py:2756（调用点 2256；run_model 2776/2783）
[模型 forward / 逐层]
 6. AscendDeepseekV4ForCausalLM.forward            deepseek_v4.py:1265（类 1212）→ DeepseekV4Model.forward:1103（类 1007）
 7. DeepseekV2DecoderLayer.forward                 deepseek_v4.py:984（类 913；hc_pre:972 → self_attn:995 → hc_post:978）
 8. DeepseekV4Attention.forward                    deepseek_v4.py:904 → return self.dsa_attn(...):910（dsa_attn 构造 882）
[DSA 算子层]
 9. AscendDeepseekSparseAttention.forward          ops/dsa.py:157（类 61）→ torch.ops.vllm.dsa_forward(...):172
10. dsa_forward()（custom op 实现）                ops/dsa.py:178（no_compile_layers:185 取回 layer+metadata）
11. AscendDSAImpl.forward                          attention/dsa_v1.py:1574（类 1378）
12. 算子下发（_C_ascend / torch_npu / 多流）       dsa_v1.py:1691（_mla_prolog_multistream）
13. dispatcher(PrivateUse1) → AscendC kernel       csrc/torch_binding.cpp:2126 → op_host tiling → op_kernel
```

---

## 1. 控制面：从 EngineCore 到 Worker（跳 1-3）

### 跳 1 — EngineCore.step（core.py:452-488）

引擎核心的单步循环。注意它**不直接调模型**，而是经 `model_executor`：
```python
def step(self):
    scheduler_output = self.scheduler.schedule()                       # 控制面产出
    future = self.model_executor.execute_model(scheduler_output, non_block=True)  # core.py:464 ★非阻塞
    grammar_output = ...
    model_output = self.model_executor.sample_tokens(grammar_output)   # core.py:472 第二段
    self.scheduler.update_from_output(scheduler_output, model_output)  # 回流闭环
```
要点：`non_block=True` + `execute_model`/`sample_tokens` 两段调用，是 async scheduling 的引擎侧体现（详见 [[20260614-164853-vllm-vllm-ascend-三模块下钻-Scheduler闭环-DSA-execute-model-分析]] 模块 C）。

### 跳 2 — Executor.execute_model（executor/abstract.py:37）

`Executor.get_class`（abstract.py:48）按 `distributed_executor_backend` 选 `MultiprocExecutor`/`RayExecutorV2`/`UniProcExecutor`。它通过 `collective_rpc` 把 `scheduler_output` **广播到每个 Worker 进程**（IPC），是控制面跨进程到执行面的桥。

### 跳 3 — Worker.execute_model（gpu_worker.py:806）

NPUWorker 继承自此（未重写该方法）。职责：PP 通信（irecv intermediate_tensors）→ 调 `self.model_runner.execute_model(scheduler_output, intermediate_tensors)`（gpu_worker.py:866）。`annotate_profile` 包裹用于 msprof 打点。

---

## 2. 执行面准备：NPUModelRunner 与 forward_context 旁路（跳 4-5）

### 跳 4 — NPUModelRunner.execute_model（model_runner_v1.py:1904）

这是 attention metadata 的**生产地**与**注入点**。核心两件事：

1. **构建 attn_metadata**（`_prepare_inputs` → DSA builder，见 [[20260614-164853-vllm-vllm-ascend-三模块下钻-Scheduler闭环-DSA-execute-model-分析]] 模块 B）：把 block_table、slot_mapping、seq_lens、cos/sin、4 类缓存组信息打包成 `AscendDSAMetadata`。

2. **注入全局上下文**（model_runner_v1.py:2233-2256）：
```python
with set_ascend_forward_context(
        attn_metadata, self.vllm_config, num_tokens=..., aclgraph_runtime_mode=cudagraph_mode,
        model_instance=self.model, ...):
    hidden_states = self._model_forward(num_tokens_padded, input_ids, positions, ...)
```

> ★ **关键设计（整篇核心）**：`attn_metadata` **不**作为参数层层透传进模型 forward，而是塞进 `set_ascend_forward_context` 管理的**线程局部全局上下文**。模型 forward 签名只有 `(input_ids, positions, intermediate_tensors, inputs_embeds)`——干净、稳定，且能被 torch.compile / ACL Graph 捕获（捕获时 forward 参数是固定 shape 张量，metadata 在 op 内部取回）。这就是为什么后面跳 10 要从 `forward_context` 反向取回 metadata。

### 跳 5 — _model_forward（model_runner_v1.py:2756）

```python
run_model = partial(self.model, **model_inputs)   # self.model = AscendDeepseekV4ForCausalLM(被 ACLGraphWrapper 包裹)
hidden_states = run_model()                         # decode 稳态 → ACL Graph replay
self._update_full_graph_params_if_needed(...)       # replay 前/后原位刷新 attn 图参数
```
decode 稳态下 `run_model()` 实际触发的是 **ACL Graph replay**（已捕获的图），Python 不再逐层下发——跳 6-13 整条栈在首次 capture 时走一遍，之后被 replay 取代。

---

## 3. 模型 forward：逐层到 attention（跳 6-8）

### 跳 6 — AscendDeepseekV4ForCausalLM.forward（deepseek_v4.py:1265，类 1212）→ DeepseekV4Model.forward（:1103，类 1007）

`@support_torch_compile` 装饰的 `DeepseekV4Model.forward`（deepseek_v4.py:1103）：
```python
hidden_states = self.embed_input_ids(input_ids)            # 或 inputs_embeds
hidden_states = hidden_states.unsqueeze(1).repeat(1, hc_mult, 1)  # mHC 超连接：(b,h)→(b,c,h)
for layer in islice(self.layers, start, end):              # 逐 DecoderLayer
    hidden_states, residual = layer(positions, hidden_states, residual, llama_4_scaling)
hidden_states = self.hc_head(...)                          # mHC 收敛
hidden_states = self.norm(hidden_states)
```

### 跳 7 — DeepseekV2DecoderLayer.forward（deepseek_v4.py:984，类 913）

mHC（多头超连接）夹住 attention：
```python
residual = hidden_states.clone()
hidden_states, post, comb = self.hc_pre(hidden_states, self.hc_attn_fn, ...)  # 调 torch.ops._C_ascend.npu_hc_pre
hidden_states = self.input_layernorm(hidden_states)
hidden_states = self.self_attn(positions=..., hidden_states=..., llama_4_scaling=...)  # ★ 跳 8
hidden_states = self.hc_post(hidden_states, residual, post, comb)              # torch.ops._C_ascend.npu_hc_post
# ... 同样的 hc 夹 MLP/MoE
```
`hc_pre`/`hc_post` 本身就是自研算子调用（deepseek_v4.py:972/978，分别调 `torch.ops._C_ascend.npu_hc_pre`/`npu_hc_post`）——attention 的「前后处理」已经是 csrc 算子，印证 [[20260614-175025-vllm-ascend-csrc算子链路与xlite机制-分析]] 中「norm 类是自研算子密集区」。

### 跳 8 — DeepseekV4Attention.forward（deepseek_v4.py:904，return 在 910）

极简——只做一件事：转发给 DSA 对象。
```python
def forward(self, positions, hidden_states, llama_4_scaling):
    return self.dsa_attn(positions, hidden_states, llama_4_scaling)
```
`self.dsa_attn` 是 `AscendDeepseekSparseAttention`（deepseek_v4.py:882 构造），构造时把所有 Linear（wq_a/wq_b/wkv/wo_a/wo_b）、indexer、compressor、swa_cache 打包成 `DSAModules`（deepseek_v4.py:865）一次性传入——**权重归属在 DeepseekV4Attention，计算逻辑在 DSA 对象**，职责分离。

---

## 4. DSA 算子层：custom op 边界与 metadata 取回（跳 9-11）

### 跳 9 — AscendDeepseekSparseAttention.forward（ops/dsa.py:157）

不直接算，而是**包成 custom op**：
```python
def forward(self, positions, hidden_states, kv_cache=None, attn_metadata=None):
    need_gather_q_kv = get_forward_context().flash_comm_v1_enabled
    output = torch.empty(output_shape, ...)
    # 所有 DSA 路径都跑在 dsa_forward custom op 边界内，
    # 这是 ACL graph 捕获的必要条件（dispatch_key="PrivateUse1"）
    torch.ops.vllm.dsa_forward(hidden_states, need_gather_q_kv, output, self.prefix)  # ops/dsa.py:172
    return output.view(-1, output_shape[-1])
```
> ★ 为什么要包一层 custom op：ACL Graph（和 torch.compile）捕获的是 op 序列。若 attention 直接用 Python 控制流（if prefill/decode），无法被捕获成稳定图。包成 `torch.ops.vllm.dsa_forward` 后，整个 DSA 计算对图捕获器是**一个不透明算子节点**，内部的 Python 分支在 capture 时执行一次、固化进图。

### 跳 10 — dsa_forward() custom op 实现（ops/dsa.py:178）

custom op 的真正实现体，**从 forward_context 反向取回** layer 对象与 metadata：
```python
def dsa_forward(hidden_states, need_gather_q_kv, output, layer_name):
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]        # ★ 用 layer_name 取回 layer 实例
    if forward_context.attn_metadata:
        attn_metadata = filter_metadata(forward_context.attn_metadata, self.prefix)  # 按 prefix 过滤本层
    if attn_metadata is None:                                    # warmup/dummy 路径
        self.dsa_attn.impl.dsa_warmup_with_multistream(hidden_states); output.fill_(0); return
    kv_cache = _build_kv_cache(self, forward_context)
    self.dsa_attn.impl.forward(self.dsa_attn.layer_name, hidden_states, kv_cache,
                               attn_metadata, need_gather_q_kv, output)              # ★ 跳 11
```
这就是跳 4「旁路注入」的对端「旁路取回」——闭合了 attn_metadata 的传递。`no_compile_layers[layer_name]` 是 forward_context 维护的「layer 名 → 实例」映射，让 custom op（只能传张量和字符串）能拿到完整 layer 对象。

### 跳 11 — AscendDSAImpl.forward（dsa_v1.py:1574，类 1378）

DSA 注意力的实际计算入口（结构见 [[20260614-164853-vllm-vllm-ascend-三模块下钻-Scheduler闭环-DSA-execute-model-分析]] 模块 B.3）：
```python
has_prefill = num_prefills > 0; has_decode = num_decodes > 0
hidden_states = maybe_all_gather_and_maybe_unpad(...)        # FlashComm V1 SP allgather
if has_prefill: o[decode:actual] = self._forward_prefill(...)   # dsa_v1.py:1866
if has_decode:  o[:decode]       = self._forward_decode(...)    # dsa_v1.py:2186
torch.ops._C_ascend.inplace_partial_rotary_mul(...)         # 尾部 partial RoPE
output[...] = self.wo_b(self.wo_a(o))                        # 输出投影
```

---

## 5. 算子下发：三类算子混用 + AscendC kernel（跳 12-13）

### 跳 12 — _mla_prolog_multistream（dsa_v1.py:1691）

MLA prolog 的多流 Cube/Vector 并行（精华，见 [[20260614-164853-vllm-vllm-ascend-三模块下钻-Scheduler闭环-DSA-execute-model-分析]] 模块 B.4）。这里三类算子混用：
```python
q_quant, q_scale = self.cv_wq_a.quantize(hidden_states)              # 量化（自研/厂商）
with npu_stream_switch(aux_stream):                                  # 辅流
    kv_quant, kv_scale = self.cv_wkv.quantize(hidden_states)
wq_a_result = self.cv_wq_a.matmul(q_quant, q_scale)                  # Cube matmul
qr, qr_scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(...)   # 自研融合算子
torch.ops._C_ascend.inplace_partial_rotary_mul(kv, cos, sin, ...)    # 自研 RoPE
DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, kv, slot_mapping)  # 自研 KV 压缩写入
q = torch_npu.npu_quant_matmul(q_b_quant, self.wq_b.weight, ...)     # 厂商量化 matmul
```

### 跳 13 — dispatcher → AscendC kernel

以 `torch.ops._C_ascend.inplace_partial_rotary_mul` 为例（链路见 [[20260614-175025-vllm-ascend-csrc算子链路与xlite机制-分析]] §1.6）：
```
torch.ops._C_ascend.inplace_partial_rotary_mul(...)
  → dispatcher [PrivateUse1 key]
  → vllm_ascend::<impl>（csrc torch_adpt：at::Tensor → aclnn）
  → op_host tiling（切块、定核数）
  → op_kernel（arch22/arch35 AscendC kernel）→ AI Core（Cube/Vector/AIV）
```
`torch_npu.npu_*` 走 CANN 内置算子库（同样落到 AI Core，但实现在 torch_npu/CANN 而非本仓 csrc）。

---

## 6. 回流与 decode 稳态（闭环）

attention 算完 → 逐层返回 → `DeepseekV4Model.forward` 出 hidden_states → `_model_forward` 返回 → `execute_model` 存 `ExecuteModelState`、返回 `None`（前向已 launch）→ `sample_tokens` 采样 + draft → `ModelRunnerOutput` 经 IPC 回 `EngineCore` → `Scheduler.update_from_output`（接受/拒绝回退指针）→ `OutputProcessor` → 客户端。

**decode 稳态的真相**：上述跳 6-13 **只在 ACL Graph 首次 capture 时完整执行一次**。之后每个 decode step，跳 5 的 `run_model()` 直接 replay 已捕获的图——Python 侧只剩跳 1-5 的调度与图参数刷新，跳 6-13 全部由 NPU 按图执行。这正是 [[20260611-010803-vllm-ascend-deepseek-v4-runtime算子协同优化-方案设计]] 里「把 host 每步参与压到最小」的现有机制基础，也是 attn_metadata 必须走 forward_context 旁路（而非 forward 参数）的根因——参数透传会破坏图的稳定签名。

---

## 7. 三个关键设计点（这条栈为什么这么绕）

| 设计点 | 跳 | 为什么 |
|--------|----|--------|
| attn_metadata 走 forward_context 全局旁路，不走 forward 参数 | 4→10 | 保持模型 forward 签名稳定，可被 ACL Graph/torch.compile 捕获 |
| DSA 计算包成 `torch.ops.vllm.dsa_forward` custom op | 9 | 把含 Python 分支的 attention 变成图捕获器眼里的单一不透明算子节点 |
| 权重归属（DeepseekV4Attention）与计算（DSA impl）分离，经 DSAModules 打包传递 | 8→11 | 模型定义层管权重加载/量化，算子层管计算编排，各自演进 |
| execute_model 返回 None + sample_tokens 两段 | 4 | async scheduling：前向 launch 后调度器即可排下一步 |

---

## 8. 源码证据索引

| 跳 | 函数 | 位置 |
|----|------|------|
| 1 | EngineCore.step | `vllm/v1/engine/core.py:452-488` |
| 2 | Executor.execute_model / get_class | `vllm/v1/executor/abstract.py:37-83` |
| 3 | Worker.execute_model | `vllm/v1/worker/gpu_worker.py:806（调 model_runner 866）` |
| 4 | NPUModelRunner.execute_model / set_ascend_forward_context | `vllm-ascend/vllm_ascend/worker/model_runner_v1.py:1904, 2233-2256` |
| 5 | _model_forward | `model_runner_v1.py:2756（调用点 2256）` |
| 6 | ForCausalLM / Model.forward | `vllm_ascend/models/deepseek_v4.py:1265(类1212), 1103(类1007)` |
| 7 | DecoderLayer.forward / hc_pre/post | `deepseek_v4.py:984(类913), hc_pre 972 / hc_post 978 / self_attn 995` |
| 8 | DeepseekV4Attention.forward / dsa_attn 构造 / DSAModules | `deepseek_v4.py:904(return 910), dsa_attn 882, DSAModules 865` |
| 9 | AscendDeepseekSparseAttention.forward | `vllm_ascend/ops/dsa.py:157(类61), dsa_forward 调用 172` |
| 10 | dsa_forward custom op 实现 / metadata 取回 | `ops/dsa.py:178（no_compile_layers 185, filter_metadata 187, impl.forward 198）` |
| 11 | AscendDSAImpl.forward | `vllm_ascend/attention/dsa_v1.py:1574（类 1378）` |
| 12 | _mla_prolog_multistream（三类算子混用） | `dsa_v1.py:1691（_forward_prefill 1866 / _forward_decode 2186）` |
| 13 | 算子注册/下发 | `csrc/torch_binding.cpp:2126` + op_host/op_kernel |
