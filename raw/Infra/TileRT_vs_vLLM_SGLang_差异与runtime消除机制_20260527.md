# TileRT vs vLLM / SGLang：差异、runtime 消除机制与性能收益来源

日期：2026-05-27

分析对象：

- `TileRT/`
- `vllm/`
- `sglang/`

重点参考：

- `TileRT/python/models/deepseek_v3_2/modules/end2end.py`
- `TileRT/python/models/deepseek_v3_2/temp_var_indices.py`
- `TileRT/python/models/deepseek_v3_2/modules/dsa.py`
- `TileRT/python/models/deepseek_v3_2/modules/mla.py`
- `TileRT/python/models/base.py`
- `vllm/vllm/v1/core/sched/scheduler.py`
- `vllm/vllm/v1/worker/gpu/model_runner.py`
- `vllm/vllm/compilation/cuda_graph.py`
- `sglang/python/sglang/srt/managers/scheduler.py`
- `sglang/python/sglang/srt/model_executor/model_runner.py`
- `sglang/python/sglang/srt/model_executor/cuda_graph_runner.py`

---

## 1. 一句话结论

vLLM / SGLang 是 **通用在线推理引擎**：它们的核心价值是动态请求管理、continuous batching、KV cache 管理、多模型/多功能 serving、prefix cache、spec decode、LoRA、结构化输出、多模态、PD disaggregation 等。

TileRT 更像是 **面向特定模型、特定硬件、特定低延迟场景的专用 native decode runtime**：它把模型结构、权重布局、临时 buffer、KV cache、MTP 状态、采样状态和 native 执行计划全部提前固定，decode hot path 中只保留最小控制输入，然后由 `libtilert.so` 里的原生 runtime 完成整个 step。

所以最大的区别不是“谁有没有 CUDA graph”，而是：

> vLLM / SGLang 在保留通用 runtime 的同时尽量优化 hot path；TileRT 则把通用 runtime 从低延迟 decode hot path 中拿掉，换成模型/硬件专用的固定执行计划。

---

## 2. vLLM / SGLang 典型推理引擎在做什么

### 2.1 vLLM 的典型职责

从 `vllm/vllm/v1/core/sched/scheduler.py` 和 `vllm/vllm/v1/worker/gpu/model_runner.py` 可以看到，vLLM 的核心 loop 需要处理：

- 请求队列：waiting、running、finished。
- 调度策略：FIFO/priority、max seqs、max batched tokens。
- KV cache manager：block 分配、释放、prefix cache、KV connector。
- Prefill/decode 混合调度。
- Speculative decoding metadata。
- Structured output / grammar。
- Multimodal encoder budget。
- Data parallel / pipeline parallel / context parallel。
- LoRA、pooling、embedding、sampling、logprobs。
- CUDA graph capture/replay。
- Worker output 回传给 scheduler。

这些能力构成了 vLLM 的通用性，但也意味着每个 step 需要持续做大量 runtime 决策。

### 2.2 SGLang 的典型职责

从 `sglang/python/sglang/srt/managers/scheduler.py`、`model_runner.py`、`cuda_graph_runner.py` 可以看到，SGLang 同样是一个完整 serving runtime：

- Scheduler 管理 request lifecycle。
- ScheduleBatch 管理 prefill/decode batch。
- Radix/prefix cache、HiCache、KV transfer、disaggregation。
- Grammar、reasoning parser、LoRA、multimodal、session。
- 多后端 attention、MoE、quantization。
- CUDA graph / piecewise CUDA graph / breakable CUDA graph。
- Speculative decoding、EAGLE、MTP 相关 runner。
- Metrics、profiler、health check、weight update。

SGLang 比 vLLM 更强调 structured generation、前端语言和 serving 组合能力，但本质上也是“通用 runtime + 优化 kernel/graph”的架构。

### 2.3 通用引擎的基本代价

vLLM/SGLang 的 runtime 每一步通常需要处理：

```text
请求状态更新
调度决策
batch 构造
KV block / token pool 映射
input buffer 准备
attention metadata 准备
model forward
sampler / verifier
输出回传
完成条件判断
资源释放/复用
```

即使使用 CUDA graph，很多外层工作依然存在，因为它们必须支持动态请求、动态 batch、不同模型、不同并发、不同功能。

---

## 3. TileRT 在做什么

TileRT 当前开源 Python 层显示出完全不同的设计重心。

### 3.1 加载 native runtime

`TileRT/python/__init__.py` 中通过：

```python
torch.ops.load_library(str(lib_path))
_load_library("libtilert.so")
```

加载原生 runtime。后续核心执行基本都通过 `torch.ops.tilert.*` 调用。

### 3.2 `prepare_money` 和 `show_hands`

`ShowHandsDSALayer` 中的关键 API：

- `dsa_show_hands_prepare_money(...)`
- `dsa_show_hands(...)`
- `dsa_show_hands_reset(...)`
- `dsa_show_hands_go_home(...)`
- `dsa_show_hands_set_sampling_seed(...)`
- `dsa_mtp_e2e_show_hands_set_prefill_valid_tokens(...)`
- `dsa_mtp_e2e_show_hands_set_prefill_mtp_extra_token(...)`

其中：

- `prepare_money` 接收 params、temp_vars、cache_vars、profile_logs。
- `show_hands` 是 decode step 的 native 执行入口。
- sampling 参数变化需要 teardown + recapture。

这说明 TileRT 的关键动作不是每步动态构图，而是先把整套 runtime 环境“准备好”，decode 时只调用一个原生入口。

### 3.3 固定 temp_vars ABI

`temp_var_indices.py` 定义了 51 个固定槽位，包括：

- Q/KV/KI
- attention 中间结果
- MoE 中间结果
- logits/token output
- cur_pos/token_id
- draft/predicted/accepted/next_draft
- FP8 quant buffer
- sampling seed/config
- top-p debug buffer

这些槽位是 Python 和 C++ runtime 的固定 ABI。

TileRT 不是通过 Python dict 或动态图对象在每步传递状态，而是通过固定 index 的 tensor table 传递状态。

### 3.4 连续显存

`generate_params_with_continuous_storage` 会把 temp_vars 重新映射到一块大 `uint8` tensor 上，并按 1024 byte 对齐切 view。

这带来：

- 地址稳定。
- allocator 干扰减少。
- graph/native runtime 更容易 capture/replay。
- native side 可以固定 pointer 访问。

### 3.5 权重和模型专用化

`Dsa` 明确按 DeepSeek-V3.2/GLM 的结构注册：

- 前 `n_dense_layers` 用 `MlpBlock`。
- 后续用 `MoeBlock`。
- 每层都有 `Mla`。
- 末尾是 `RMSNormHeadProj`。
- MTP 额外注册 `MTPPreprocessLayer + MoeBlock + RMSNormHeadProj`。

权重转换器大量做 reshape、transpose、MMA swizzle、FP8/FP16/BF16 algorithm selection。

这意味着 TileRT 把“模型结构知识”提前编进了 runtime 和权重布局，而不是在通用模型 executor 中动态分派。

---

## 4. 最大差异对比

| 维度 | vLLM / SGLang | TileRT |
|---|---|---|
| 目标 | 通用在线 serving，高吞吐，多请求 | 单请求/低延迟极限 decode |
| Runtime | 完整调度 runtime 常驻 hot loop | hot loop 中仅保留极薄控制入口 |
| Batch | 动态 continuous batching | 当前代码明显偏 batch=1 |
| KV | 动态 block/token pool 管理 | 固定 cache vars，模型专用 layout |
| 模型支持 | 多模型、多任务、多后端 | DeepSeek/GLM 专用路径明显 |
| 权重格式 | 加载时适配多后端 | 离线转换到 kernel-friendly layout |
| Graph | CUDA graph/分段 graph 优化通用 runtime | native runtime 预准备固定执行计划 |
| Spec/MTP | 作为 runtime 功能模块接入 | MTP 状态成为 temp_vars ABI 一部分 |
| 采样 | 通用 sampler/logprob/grammar 等 | sampling config 进入 prepare/capture |
| 通信 | 通用 TP/PP/DP/EP 通信模块 | allreduce 等直接融合进 op 序列 |
| 灵活性 | 高 | 低 |
| 极致低延迟潜力 | 受通用 runtime 约束 | 更高 |

---

## 5. “消除 runtime”到底是什么意思

这里的“消除 runtime”不是说完全没有 runtime，而是：

> 消除通用推理框架在 decode hot path 上的大量动态 runtime 决策，把它们提前到初始化、编译、权重转换和 native runtime prepare 阶段。

换句话说，TileRT 不是没有 runtime，而是把 runtime 从：

```text
每个 token 动态调度
```

变成：

```text
初始化时构造固定 native runtime，decode 时只执行固定计划
```

### 5.1 被消除或大幅削弱的 runtime 工作

1. **动态 request scheduler**

   vLLM/SGLang 每步要从 waiting/running 队列中调度请求。TileRT demo 路径基本是单请求固定序列，不需要 continuous batching scheduler。

2. **动态 batch 构造**

   通用框架每步构造 batch、padding、seq lens、slot mapping。TileRT 的 batch/seq/MTP shape 在 temp_vars 初始化时固定。

3. **动态 KV block 分配**

   vLLM/SGLang 需要 block manager/token pool。TileRT 直接为每层创建固定 `ki_cache/kv_cache/pe_cache`。

4. **动态 attention metadata 构造**

   通用框架需要根据 batch/prefill/decode 构造 metadata。TileRT 通过固定 cache 和 cur_pos/token state 让 native runtime 自己按固定 ABI 执行。

5. **动态图 op 调度**

   通用框架即使有 graph，也仍有很多模型 runner/input prep/sampler 逻辑。TileRT 将端到端 step 包成 `show_hands` native op。

6. **频繁 tensor allocation**

   TileRT 把 temp_vars 放在连续大 buffer 中，避免 decode 热路径不断分配中间 tensor。

7. **通用 sampler/logprob/grammar 逻辑**

   TileRT 的 sampling config 是 temp_vars 的固定槽位，且注释显示参数 baked into graph instructions。

8. **动态权重 layout 适配**

   TileRT 在 `weight_converter.py` 和各 op converter 中提前把权重变成 kernel 需要的 layout。

9. **部分 CPU/GPU 同步**

   通过 device-side temp_vars 保存 token/MTP 状态，减少 runtime 外层取值。不过当前 Python generator 仍存在展示用 `.item()` / `.cpu()`，极致路径中可进一步 native 化。

10. **通信暴露时间**

   通过 `DownAllReduce`、`ExpertDownAllReduce`、`UnProjOAllReduce` 等 fused op，把通信纳入模型 op 序列，给 native runtime 做 fusion/overlap 的机会。

---

## 6. 模型 + 编译器/runtime + 硬件紧耦合如何实现

### 6.1 模型紧耦合

TileRT 并不是把任意 HuggingFace 模型直接丢给通用 executor。

它显式知道：

- DeepSeek-V3.2 有多少 dense layer。
- 哪些层是 MoE。
- MLA 由哪些子 op 构成。
- MTP 层如何接入。
- 每个中间状态的 shape、dtype、slot index。
- KV/PE/KI cache 维度。
- head projection 和 vocab 分片。

这些知识体现在：

- `Dsa`
- `Mla`
- `MlpBlock`
- `MoeBlock`
- `MTP`
- `temp_var_indices.py`

### 6.2 编译器/runtime 紧耦合

Python 层把模型序列化成：

```text
params: 权重列表
temp_vars: 固定中间状态列表
cache_vars: KV/PE/KI cache 列表
profile_logs: 原生 profile buffer
```

然后交给：

```python
dsa_show_hands_prepare_money(...)
```

这相当于把模型执行计划交给 native runtime。

之后 decode 只需要：

```python
dsa_show_hands(token_id)
```

或 MTP：

```python
dsa_mtp_e2e_show_hands(draft_tokens)
```

这就是“运行时决策前移”：模型图、buffer、cache、采样、MTP 状态都在 prepare 阶段绑定。

### 6.3 硬件紧耦合

TileRT 代码和 README 中的硬件假设很强：

- README 要求 8x B200。
- `ShowHandsDSALayer.num_devices = 8`。
- 权重文件按 `dev_{device_id}` 加载。
- 多个 converter 做 MMA swizzle。
- FP8MMA/FP16MMA/BF16 algorithm 分支。
- allreduce 融进 op。

这些说明它针对固定硬件拓扑、固定 device 数、固定矩阵 tile 和通信模式做了布局和 runtime 设计。

---

## 7. 核心做了哪些关键优化或改动

### 7.1 从通用 serving loop 改成 single-request fixed loop

通用框架：

```text
schedule -> prepare batch -> forward -> sample -> update requests
```

TileRT：

```text
prepare_money once -> show_hands repeatedly
```

收益：消除调度和 batch 构造开销。

### 7.2 从动态图/模块执行改成 native end-to-end step

通用框架仍在 model runner 中组织 forward、sampler、metadata。

TileRT 把 step 放进 `libtilert.so`。

收益：减少 Python 层、PyTorch dispatcher、小 op launch 开销。

### 7.3 固定 temp_vars ABI

所有中间结果、token state、MTP state 都固定在 tensor table。

收益：减少动态状态管理；便于 native pointer 访问；利于 graph capture。

### 7.4 连续 buffer

所有 temp vars 映射到大 tensor。

收益：地址稳定、少分配、少碎片。

### 7.5 权重离线转换

权重提前变成 kernel layout。

收益：decode 时不做 layout transform。

### 7.6 fused op

融合 RMSNorm、projection、quant、MoE expert select/up/gate/silu/down、allreduce、head projection、sampling。

收益：减少 kernel 数、减少中间 HBM 读写、减少通信暴露。

### 7.7 MTP 内生化

MTP 不是外部临时 proposer，而是 runtime state 和 native execution 的一部分。

收益：提升每次 forward 的 accepted token 数。

### 7.8 Sampling capture

sampling 参数在 prepare 阶段进入固定执行计划。

收益：减少动态 sampler 开销。

代价：动态更新 sampling 参数需要 recapture。

### 7.9 Profile buffer

profile logs 作为 tensor 传入 native op。

收益：低成本记录性能，为继续优化提供数据。

---

## 8. 消除了哪些时间

可以按 decode step 时间拆：

```text
T_total =
  T_scheduler
  + T_batch_prepare
  + T_kv_mapping
  + T_metadata
  + T_graph_or_dispatch
  + T_kernel_compute
  + T_memory_io
  + T_communication
  + T_sampler
  + T_sync
  + T_output_update
```

TileRT 主要削减：

| 时间项 | vLLM/SGLang 中的来源 | TileRT 的处理 |
|---|---|---|
| `T_scheduler` | waiting/running request 调度 | batch=1 固定 loop，基本移除 |
| `T_batch_prepare` | 组 batch、padding、seq lens | 固定 temp_vars / MTP seq len |
| `T_kv_mapping` | block table、slot mapping | 固定 cache vars |
| `T_metadata` | attention metadata、forward context | native runtime 内部固定 ABI |
| `T_graph_or_dispatch` | PyTorch op dispatch、graph mode 判断 | `prepare_money` 后 `show_hands` |
| `T_kernel_launch` | 多小 op launch | fused native op / graph replay |
| `T_memory_io` | 中间 tensor 写回 HBM | 融合 op、连续 buffer |
| `T_communication` | allreduce 暴露 | down/allreduce 融合或 overlap |
| `T_sampler` | 通用 sampler/logprob/grammar | sampling state 固定化 |
| `T_sync` | `.item()`、CPU/GPU 往返 | MTP/token 状态放 temp_vars，仍可继续优化 |
| `T_output_update` | request state、streaming output | demo 路径简化 |

真正的性能提升通常来自组合效果：

```text
更低 step latency + 更高 accepted_tokens_per_step
```

其中 MTP 是乘法项。假设：

- 非 MTP step latency = 3 ms，accepted = 1，约 333 token/s。
- MTP step latency = 5 ms，accepted mean = 2.7，约 540 token/s。

这说明即使 MTP 单步更重，只要平均接受长度足够高，有效 token/s 仍能显著提升。

---

## 9. 代价和边界

TileRT 的设计不是免费午餐。

它牺牲了：

- 通用模型支持。
- 动态 batch 灵活性。
- 多租户 serving 能力。
- 动态 sampling 参数更新。
- 动态 shape 适应能力。
- 硬件无关性。
- runtime 可解释性，因为 native 源码未开。

换来了：

- 极低 decode hot path overhead。
- 固定模型/硬件下更强 fusion。
- 更稳定 graph/native replay。
- MTP 与主 runtime 的深度融合。
- 更高单请求 token/s 上限。

---

## 10. 对 Ascend / vLLM Ascend 的启发

如果要在 Ascend 上复刻 TileRT-like 思想，不建议直接搬 TileRT。

更实际的路径是：

1. 在 vLLM Ascend 中保留 serving 外壳。
2. 为特定模型增加 low-latency special path。
3. 将 decode path 固定为：

   ```text
   prepare runtime state -> ACLGraph/NPUGraph capture -> replay fixed decode step
   ```

4. 建立 Ascend 版 `temp_vars` / runtime state table。
5. 将 MTP accept / next draft / predicted tokens device-side 化。
6. 对 MLA/DSA、MoE、head/sampler 做 AscendC 或 Triton-Ascend fused op。
7. 对权重做离线 layout conversion。
8. 通过 MC2/HCCL fusion/overlap 降低通信暴露。

对应到优先级：

```text
P0: 清理 D2H sync
P1: 固定 decode shape bucket 和 persistent buffers
P2: FULL_DECODE_ONLY graph replay 稳定
P3: MTP accept path device-side
P4: DSA/MLA fused + overlap
P5: MoE dispatch/GMM/combine + MC2 overlap
P6: device-side sampler
```

---

## 11. 最核心的区别总结

vLLM/SGLang 的核心是：

> 用一个通用 runtime 管理复杂在线服务，然后用 kernel、CUDA graph、cache、scheduler 优化把通用性成本降下来。

TileRT 的核心是：

> 放弃一部分通用 runtime 能力，把模型、权重、buffer、cache、MTP、采样、通信和硬件拓扑提前固化成 native decode runtime，让 hot path 只做固定执行。

所以所谓“消除 runtime”，真正消除的是：

- 动态调度时间。
- 动态 batch 构造时间。
- 动态 KV/block/metadata 管理时间。
- 动态 tensor 分配时间。
- PyTorch 小 op dispatch 时间。
- kernel launch 数量。
- 中间 HBM 往返。
- 部分 CPU/GPU 同步。
- 部分通信暴露。
- 通用 sampler/verifier 开销。

而它保留并强化的是：

- 原生 specialized runtime。
- 固定 ABI。
- 固定 buffer。
- 固定 graph/执行计划。
- 固定权重 layout。
- 模型专用 fused op。
- MTP 的设备侧状态机。

这就是 TileRT 相比典型开源推理引擎最大的架构差异。

