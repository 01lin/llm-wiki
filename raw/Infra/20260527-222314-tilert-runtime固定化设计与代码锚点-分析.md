# TileRT runtime 固定化设计与代码锚点

日期：2026-05-27

本文基于本地 `TileRT/`、`vllm-ascend/` 代码做实现视角分析，目标是回答一个更具体的问题：为了尽可能消除 decode 热路径里的 runtime 耗时，TileRT 把哪些内容固定了，这些固定项的范围和含义是什么，对应代码在哪里。

需要先说明边界：当前开源目录中没有看到 native C++/CUDA 实现源码，Python 包通过 `TileRT/python/__init__.py:49` 加载 `libtilert.so`。因此，本文能精确分析 Python 公开层暴露出来的固定 ABI、权重布局、执行顺序、buffer 布局和 native op 入口，但 `prepare_money` 内部到底如何 capture、fusion、tile schedule、stream schedule 和跨卡通信重排，只能从接口形态和外层约束做工程推断。

## 1. 一句话结论

TileRT 的核心思路是把 decode 中“每步动态推导”的内容，尽可能提前变成固定契约：

- 模型结构固定：层数、dense/MoE 分界、head 维度、expert 数、MTP 结构都写成常量。
- 硬件拓扑固定：默认 8 设备，权重和 cache 都按 `dev_{id}` 预切好。
- shape 固定：`batch=1`、decode chunk `seq_len=4`、最大 KV 长度、cache padding、block size 全部提前定死。
- 内存 ABI 固定：51 个 `temp_vars` slot 顺序固定，Python enum 与 C++ `DsaTempVars` 一一对应。
- buffer 地址固定：每个设备的 temp vars 使用一整块连续 storage，再切成固定 dtype/shape view。
- 执行 DAG 固定：每层有哪些 op、op 顺序、参数列表顺序、cache 列表顺序、权重列表顺序全部由 Python 侧注册时固定。
- 权重布局固定：HF 权重在离线转换阶段被 reshape、transpose、swizzle、shard 成 native kernel 期望的布局。
- graph/runtime plan 固定：`prepare_money` 一次性接收 params/temp/cache/profile/shape，然后热路径只调用 `dsa_show_hands(token_id)`。
- sampling 固定：temperature/top_p/top_k/use_topp 被写入 `SAMPLING_CONFIG`，并且代码注释明确说明这些参数在 prepare 阶段 bake 到 CUDA graph 指令中，变化时要 teardown + recapture。
- MTP 状态固定：draft/predicted/accepted/next draft 等 speculative decode 状态都有固定 slot，避免每步构造 Python 对象或动态 verifier 数据结构。

对应热路径里，最理想状态只剩：

```text
token_id -> native show_hands -> 复用固定 params/temp/cache/graph -> 写回固定 output slots
```

这就是“消除 runtime”的具体含义：不是没有 runtime，而是把 runtime 从 Python 调度器、动态 shape 处理、buffer 管理、op dispatch、采样处理、跨卡同步编排，压缩为一个高度专用的 native replay/调度入口。

## 2. 固定化总览表

| 层面 | 固定内容 | 含义 | 消除的 runtime 工作 | 代码位置 |
|---|---|---|---|---|
| native 库 | `libtilert.so` | 高性能核心在 native 扩展中 | Python 不直接拼接 kernel 热路径 | `TileRT/python/__init__.py:28-49` |
| 模型族 | `deepseek_v3_2` / `glm_5` | 只支持明确列出的模型结构 | 避免通用模型图解释 | `TileRT/python/models/deepseek_v3_2/modules/end2end.py:131-139` |
| 模型参数 | hidden、layers、heads、experts、MLA/MoE/MTP 维度 | shape 和 kernel layout 的静态源头 | 避免每步查 config、推 shape | `TileRT/python/models/deepseek_v3_2/model_args.py:51-100` |
| batch | `max_batch_size=1` | 单请求 decode 专用 | 避免动态 batch 调度和 padding 复杂度 | `model_args.py:53` |
| 最大序列 | `max_seq_len=160K`、`kv_cache_pad=8` | cache 预分配上限 | 避免 cache 动态扩容 | `model_args.py:54,95` |
| decode chunk | `forward_max_seq_len=4` | MTP / decode 固定小步长 | 避免每步重建 graph bucket | `end2end.py:139-140` |
| 设备数 | `num_devices=8` | 8 卡 TP/分片固定 | 避免动态拓扑规划 | `end2end.py:139` |
| temp ABI | 51 个 slot，0 到 50 连续 | Python/C++ 共享固定编号 | 避免 dict/name lookup 和动态 tensor 创建 | `temp_var_indices.py:17-74` |
| ABI 校验 | Python enum 对齐 C++ `DsaTempVars` | 启动时发现错位 | 避免 silent mismatch | `temp_var_indices.py:80-122` |
| 连续 storage | `large_tensor` + 1024B 对齐 view | 固定地址、固定 stride | 减少碎片和 graph replay 地址变化 | `end2end.py:254-275` |
| cache 布局 | 每层 `[ki, kv, pe]` | cache list 顺序固定 | 避免 KV cache metadata 动态拼装 | `modules/mla.py:89-107` |
| 外部 cache 注入 | `base_idx=layer_id*3` | prefill-decode 解耦时仍复用固定 cache list | 避免通用 cache manager 查表 | `generator.py:474-485` |
| op 顺序 | `exec_seq` 固定注册 | 模块 DAG 变成 list | 避免动态图遍历和调度决策 | `models/base.py:224-259` |
| 权重 key | `layer_{idx}_{param}_dev_{id}` | 按层/设备/算子固定命名 | 避免运行时权重匹配 | `weight_converter.py:483-495` |
| 权重布局 | 离线 reshape/swizzle/shard | native kernel 直接消费 | 避免运行时 transpose/pack | `weight_converter.py:251-443`，`ops/*` converter |
| prepare | `dsa_show_hands_prepare_money` | 传入固定 params/temp/cache/profile/shape | 把 capture/调度准备移出热路径 | `end2end.py:27-44,382-406` |
| forward | `dsa_show_hands(token_id.cpu())` | 热路径只给 token | 避免 Python 层逐 op 执行 | `end2end.py:418-425` |
| sampling | `SAMPLING_CONFIG` 固定，变更要 recapture | 采样策略进入 graph/native plan | 避免每步 Python sampler 和 graph 参数变化 | `end2end.py:186-242` |
| MTP 状态 | draft/predicted/accepted/next draft slots | speculative verifier 状态固定 | 避免 Python verifier 动态结构 | `dsa.py:123-148` |

## 3. 模型、硬件和 shape 固定项

### 3.1 模型基础参数固定

`ModelArgs` 是所有固定 shape 的源头，关键字段在 `TileRT/python/models/deepseek_v3_2/model_args.py:51-100`：

| 固定项 | 值 | 含义 | 影响 |
|---|---:|---|---|
| `arch_name` | `deepseek_v3_2` | 模型族 | 决定 native op 选择和 layout |
| `max_batch_size` | `1` | 单请求 decode | 放弃通用 batch 调度，追求单请求 TPOT |
| `max_seq_len` | `160 * 1024` | 最大上下文 | KV/cache/attention workspace 按上限准备 |
| `dtype` | `fp8` | 主计算/权重量化目标 | converter 和 native op 按 FP8 路径准备 |
| `vocab_size` | `129280` | 词表 | 每设备 logits 为 `129280 / 8 = 16160` |
| `dim` | `7168` | hidden size | 几乎所有 temp/workspace 的主维度 |
| `inter_dim` | `18432` | dense MLP 中间维 | dense MLP 权重和 kernel shape |
| `moe_inter_dim` | `2048` | MoE expert 中间维 | 每设备 `2048 / 8 = 256` |
| `n_layers` | `61` | 主模型层数 | 外层 DSA 循环固定 61 层 |
| `n_dense_layers` | `3` | 前 3 层 dense | 0-2 用 MLP block，3-60 用 MoE block |
| `n_heads` | `128` | attention heads | 每设备 local heads `16` |
| `n_routed_experts` | `256` | MoE routed experts | routing scores 固定为 256 |
| `n_shared_experts` | `1` | shared expert | 总激活 expert buffer 为 `8+1=9` |
| `n_activated_experts` | `8` | top-k experts | select probs/indices 固定为 8 |
| `q_lora_rank` | `1536` | MLA Q low-rank | Q buffer 和 Wq layout |
| `kv_lora_rank` | `512` | MLA KV low-rank | KV/O/Q_NOPE buffer |
| `qk_nope_head_dim` | `128` | no-PE QK head dim | Q_NOPE_DOWN shape |
| `qk_rope_head_dim` | `64` | RoPE dim | Q_PE/ROPE/PE cache shape |
| `v_head_dim` | `128` | V dim | PROJ_O shape |
| `index_n_heads` | `64` | sparse index heads | IQ/IDX_SCORES shape |
| `index_head_dim` | `128` | index head dim | KI/IQ shape |
| `index_topk` | `2048` | sparse index top-k | IDX_SELECTS shape |
| `kv_cache_pad` | `8` | cache padding | KV/PE/KI cache 长度固定加 8 |
| `block_size` | `128` | quant block | X_SCALE dim 为 `7168 / 128 = 56` |

这些常量意味着：TileRT 不是拿一个通用 transformer graph 来解释执行，而是把具体模型的拓扑、维度、MoE 参数、MLA 参数直接编进执行计划。

### 3.2 硬件拓扑固定

`ShowHandsDSALayer.__init__` 中固定：

- `self.num_devices = 8`，见 `TileRT/python/models/deepseek_v3_2/modules/end2end.py:139`。
- `self.forward_max_seq_len = 4`，见 `end2end.py:140`。
- `multi_devices_results` 按 `torch.cuda.device_count()` 建 list，但实际加载、prepare 都按 `range(self.num_devices)`，见 `end2end.py:146,363-406`。

工程含义：

- 权重转换、权重命名、cache 分配、op 参数列表都假设 8 卡。
- `n_local_heads = n_heads // num_devices = 16`，见 `modules/dsa.py:67`。
- `moe_inter_dim_per_device = moe_inter_dim // num_devices = 256`，见 `modules/dsa.py:77`。
- `vocab_size_per_device = vocab_size // num_devices = 16160`，见 `modules/dsa.py:78`。

消除的 runtime 工作：

- 不需要在 decode 步中判断 TP size。
- 不需要动态计算 local head / local vocab / local expert slice。
- 不需要通用调度器为不同拓扑生成不同 communication plan。

### 3.3 decode shape 固定

当前 decode 层初始化 temp vars 时调用：

```python
dsa.get_temp_vars(
    1,
    self.forward_max_seq_len,
    {...sampling config...},
)
```

位置：`TileRT/python/models/deepseek_v3_2/modules/end2end.py:303-316`。

因此实际 prepare/capture 的核心 shape 是：

- batch `B=1`
- decode/MTP chunk `S=4`
- device `D=8`
- max cache len `160K + 8`

消除的 runtime 工作：

- 不需要每 token 处理 batch compaction。
- 不需要复杂 shape bucket 查找。
- 不需要为不同 `num_tokens` 重做输入 tensor 拼接。
- graph replay 更容易保持 tensor address / shape / stride 不变。

## 4. temp_vars 固定 ABI

### 4.1 ABI 设计

`TileRT/python/models/deepseek_v3_2/temp_var_indices.py:1-5` 明确说明 Python enum 镜像 C++ `DsaTempVars`。`TEMP_VARS_SIZE=51` 在 `temp_var_indices.py:73-74`，`validate_temp_vars_layout()` 会检查：

- enum 数量等于 51。
- index 连续为 0 到 50。
- 如果 native op 可用，调用 `torch.ops.tilert.dsa_temp_vars_size()` 对齐 C++ 侧 size。

位置：`temp_var_indices.py:80-122`。

这说明 `temp_vars` 不是普通 Python list，而是 native runtime 的稳定 ABI。native code 可以写死：

```text
temp_vars[Idx.Q] -> Q workspace
temp_vars[Idx.KV] -> KV workspace
temp_vars[Idx.CUR_POS] -> current position
temp_vars[Idx.SAMPLING_CONFIG] -> sampler config
...
```

### 4.2 51 个 slot 清单

下面的 shape 以 DeepSeek V3.2 默认值、8 卡、`B=1`、`S=4` 为基础。代码来源主要是 `TileRT/python/models/deepseek_v3_2/modules/dsa.py:47-154`。

| idx | 名称 | shape / dtype | 范围 | 含义 | 代码 |
|---:|---|---|---|---|---|
| 0 | `Q` | `[B,S,1536] bf16` | MLA | Q low-rank projection workspace | `dsa.py:82` |
| 1 | `KV` | `[B,S,512] bf16` | MLA/cache | compressed KV workspace | `dsa.py:83` |
| 2 | `KI` | `[B,S,128] bf16` | sparse index | compressed index key | `dsa.py:84` |
| 3 | `Q_NOPE_DOWN` | `[B,S,16,128] bf16` | MLA | local heads no-PE Q down projection | `dsa.py:85-87` |
| 4 | `Q_PE` | `[B,S,16,64] bf16` | MLA/RoPE | local heads RoPE Q | `dsa.py:88` |
| 5 | `IQ` | `[B,S,64,128] bf16` | sparse index | index query | `dsa.py:89` |
| 6 | `IQ_RT` | `[B,S,64,128] bf16` | sparse index | rotated/transformed index query | `dsa.py:90` |
| 7 | `IDX_SCORES` | `[B,S,64] bf16` | sparse index | per index head score | `dsa.py:91` |
| 8 | `IDX_LOGITS` | `[B,S,163848] fp32` | sparse index | long-context index logits | `dsa.py:92-94` |
| 9 | `IDX_SELECTS` | `[B,S,2048] int32` | sparse index | selected cache positions | `dsa.py:95` |
| 10 | `Q_NOPE` | `[B,S,16,512] bf16` | MLA | local no-PE Q expanded to KV rank | `dsa.py:96` |
| 11 | `O` | `[B,S,16,512] bf16` | attention | attention output before projection | `dsa.py:97` |
| 12 | `O_ACC` | `[B,S,16,32,512] fp32` | attention | attention accumulation workspace | `dsa.py:98` |
| 13 | `O_LSE` | `[B,S,16] fp32` | attention | log-sum-exp workspace | `dsa.py:99` |
| 14 | `O_LSE_ACC` | `[B,S,16,32] fp32` | attention | LSE accumulation workspace | `dsa.py:100` |
| 15 | `PROJ_O` | `[B,S,16,128] bf16` | attention | projected V output | `dsa.py:101` |
| 16 | `UNPROJ_O` | `[B,S,7168] bf16` | attention/residual | unproject/allreduce output | `dsa.py:102` |
| 17 | `SCORES` | `[B,S,256] fp32` | MoE | router scores | `dsa.py:103` |
| 18 | `X_MLP_IN` | `[B,S,7168] bf16` | MLP/MoE | post-attention MLP input | `dsa.py:104` |
| 19 | `UP_GATE` | `[B,S,9,256] bf16` | MoE | selected experts up/gate workspace | `dsa.py:105-106` |
| 20 | `SEL_PROBS` | `[B,S,8] fp32` | MoE | selected expert probabilities | `dsa.py:107` |
| 21 | `SEL_INDICES` | `[B,S,8] int32` | MoE | selected expert ids | `dsa.py:108` |
| 22 | `EXP_OUT` | `[B,S,7168] bf16` | MoE | expert output | `dsa.py:109` |
| 23 | `X_RMSNORM` | `[B,S,7168] bf16` | norm | RMSNorm output workspace | `dsa.py:110` |
| 24 | `LOGITS_OUT` | `[B,S,16160] fp32` | head/sampler | per-device logits shard | `dsa.py:111` |
| 25 | `TOKEN_OUT` | `[B,S,1] int32` | sampler | sampled token output | `dsa.py:112` |
| 26 | `EMBEDDING_RMSNORM` | `[B,S,7168] bf16` | MTP | normalized embedding | `dsa.py:114` |
| 27 | `HIDDEN_RMSNORM` | `[B,S,7168] bf16` | MTP | normalized hidden | `dsa.py:115` |
| 28 | `EH_PROJ` | `[B,S,7168] bf16` | MTP | embedding/hidden projection output | `dsa.py:116` |
| 29 | `X_TENSOR` | `[B,S,7168] bf16` | global hidden | current hidden state | `dsa.py:117` |
| 30 | `ROPE_FREQS` | `[B,S,64] fp32` | RoPE | selected RoPE freqs | `dsa.py:118` |
| 31 | `CUR_POS` | `[B] int32` | position | current decode position | `dsa.py:119` |
| 32 | `TOKEN_ID` | `[B,S,1] int32` | input | input token id workspace | `dsa.py:120` |
| 33 | `LAST_HIDDEN_STATES` | `[B,S,7168] bf16` | MTP/prefill-decode | last hidden state injection | `dsa.py:121` |
| 34 | `DRAFT_TOKENS` | `[B,S] int32` | MTP | draft token sequence | `dsa.py:123` |
| 35 | `PREDICTED_TOKENS` | `[B,S,1] int32` | MTP/verifier | main model predicted tokens | `dsa.py:124` |
| 36 | `PREDICTED_HIDDEN` | `[B,S,7168] bf16` | MTP | predicted hidden states | `dsa.py:125` |
| 37 | `ACCEPTED_TOKENS` | `[B] int32` | MTP/verifier | accepted token count | `dsa.py:126` |
| 38 | `NEXT_DRAFT_TOKENS` | `[B,S] int32` | MTP | next round draft tokens | `dsa.py:127` |
| 39 | `X_QUANT` | `[B,S,7168] fp8` | quant | quantized hidden | `dsa.py:129` |
| 40 | `X_SCALE` | `[B,S,56] fp32` | quant | per-block scale | `dsa.py:130-132` |
| 41 | `MOE_UP_GATE` | `[B,S,9,256] bf16` | MoE | MoE up/gate copy/workspace | `dsa.py:133` |
| 42 | `IDX_SEL_WS` | `[B,S,205058] int32` | sparse index | selection workspace | `dsa.py:135-136` |
| 43 | `MTP0_TOKEN_OUT` | `[B,S,1] int32` | MTP | first MTP head token output | `dsa.py:138` |
| 44 | `MTP1_TOKEN_OUT` | `[B,S,1] int32` | MTP | second MTP head token output | `dsa.py:139` |
| 45 | `MTP0_EXP_OUT` | `[B,S,7168] bf16` | MTP/MoE | MTP expert output | `dsa.py:140` |
| 46 | `SAMPLING_SEED` | `[B,S] int64` | sampler | device-side sampler seed | `dsa.py:142` |
| 47 | `SAMPLING_POSITIONS` | `[B,S] int64` | sampler | position-dependent randomness | `dsa.py:143` |
| 48 | `SAMPLING_CONFIG` | `[4] fp32` | sampler | `[temperature, top_p, top_k, use_topp]` | `dsa.py:144-146` |
| 49 | `TOP_P_SCORES` | `[B,S] fp32` | sampler | top-p score/output workspace | `dsa.py:147` |
| 50 | `TOP_P_DEBUG` | `[B,S,16160] fp32` | sampler/debug | top-p debug logits/scores shard | `dsa.py:148` |

### 4.3 temp_vars 固定化带来的收益

这套 ABI 消除或压缩了以下 runtime 开销：

- buffer 动态申请：所有 workspace 在初始化时分配。
- shape 动态推断：slot 的 dtype/shape 在 `get_temp_vars()` 中固定。
- name lookup：native code 只按 index 读写。
- Python object 创建：decode 步不创建 per-layer intermediate。
- graph replay 地址变化：连续 storage 提高指针稳定性。
- verifier/sampler 数据结构：MTP 和 sampler 的状态也放入固定 slot。

## 5. buffer、cache 和内存地址固定

### 5.1 temp vars 连续 storage

`ShowHandsDSALayer.generate_params_with_continuous_storage()` 会：

1. 计算每个 temp tensor 的 `nbytes`。
2. 按 1024 bytes 对齐。
3. 分配一个大的 `torch.zeros(tot_size, dtype=torch.uint8)`。
4. 对每个 slot 切片并 `.view(param.dtype).view(param.shape)`。

位置：`TileRT/python/models/deepseek_v3_2/modules/end2end.py:254-275`。

含义：

- Python 侧仍看到 51 个 tensor。
- 物理上它们来自一块大 storage。
- 对 native graph replay 更友好，因为各 slot 的地址和相对 offset 在 prepare 后稳定。

消除的 runtime 工作：

- 避免每步申请/释放 workspace。
- 避免内存碎片导致 graph replay input address 变化。
- native 侧可以缓存 slot 指针和 offset。

### 5.2 cache layout 固定

MLA 每层 cache 由三个 tensor 组成：

- `ki_cache`: `[max_batch_size, max_seq_len + kv_cache_pad, index_head_dim]`
- `kv_cache`: `[max_batch_size, max_seq_len + kv_cache_pad, kv_lora_rank]`
- `pe_cache`: `[max_batch_size, max_seq_len + kv_cache_pad, qk_rope_head_dim]`

位置：`TileRT/python/models/deepseek_v3_2/modules/mla.py:89-107`。

返回顺序固定为：

```python
return [*super().get_cache_vars(), self.ki_cache, self.kv_cache, self.pe_cache]
```

外部 prefill cache 注入也假设每层 cache 是三元组，且 `base_idx = layer_id * 3`：

- `caches[base_idx + 0]` -> KI
- `caches[base_idx + 1]` -> KV
- `caches[base_idx + 2]` -> PE

位置：`TileRT/python/models/deepseek_v3_2/generator.py:474-485`。

消除的 runtime 工作：

- 不需要通用 block table 查表。
- 不需要按请求动态管理 KV page。
- decode 侧可以按固定 layer offset 访问 cache。

代价：

- 当前设计强绑定单请求、固定 max seq 和特定 cache layout。
- 要支持多请求 continuous batching，需要额外设计静态 batch slots、slot remap 和 device-side metadata。

## 6. 执行 DAG 和 op 顺序固定

### 6.1 `SerializableTileRTModule` 固定 op 序列

基础类维护：

- `exec_seq`
- `prefix_seq`
- `suffix_seq`
- `retain_weights_seq`

位置：`TileRT/python/models/base.py:224-241`。

关键方法：

- `register_op()`：固定子 op 顺序，`base.py:235-241`。
- `get_cache_vars()`：按 `exec_seq` 顺序拼 cache，`base.py:229-233`。
- `get_weights_list()`：按 `exec_seq` 顺序拼 weights，`base.py:255-259`。
- `init_tilert_weights()`：按 prefix/suffix 从 state dict 取固定 key，`base.py:283-297`。

这使得 native 侧接收到的 `params` 和 `caches` 不是松散对象，而是一个顺序固定的数组。

### 6.2 主模型 DSA 顺序固定

`Dsa.__init__`：

- 遍历 `range(model_args.n_layers)`。
- `layer_idx < n_dense_layers` 用 `MlpBlock`。
- 其他层用 `MoeBlock`。
- 每层 prefix 为 `layer_{layer_idx}_`，suffix 为 `_dev_{device_id}`。
- 最后追加 `RMSNormHeadProj`。

位置：`TileRT/python/models/deepseek_v3_2/modules/dsa.py:13-45`。

固定主链路：

```text
layer 0..2:
  MlpBlock = MLA -> MLP

layer 3..60:
  MoeBlock = MLA -> MoE

final:
  RMSNormHeadProj
```

### 6.3 MLA 子图固定

`Mla.__init__` 的固定顺序：

1. `RMSNormProjxWqkvia`
2. `LayerNormRoPERotate`
3. `RmsnormProjqWqib`
4. `ProjxWis`
5. `ProjqWqb`
6. `KVRMSNorm`
7. `ProjoWKVb`
8. `UnProjOAllReduce`

位置：`TileRT/python/models/deepseek_v3_2/modules/mla.py:24-83`。

模型相关分支：

- `glm_5` 使用 `RMSNormProjxWqkviaAlgorithm.DECOUPLED`，否则 `GENERAL`，见 `mla.py:33-37`。
- `glm_5` 的 `RmsnormProjqWqib` 用 `FP16MMA`，否则 `BF16`，见 `mla.py:47-51`。
- `glm_5` 的 `UnProjOAllReduce` 用 `FP16MMA`，否则默认 `FP8MMA`，见 `mla.py:73-83`。

这些分支发生在初始化阶段，不在 decode token 热路径里动态判断。

### 6.4 Dense MLP 固定

`Mlp.__init__` 固定顺序：

1. `RMSNormUpGateSiLU`
2. `DownAllReduce`

位置：`TileRT/python/models/deepseek_v3_2/modules/mlp.py:11-28`。

`MlpBlock.__init__` 固定：

```text
MLA -> MLP
```

位置：`modules/mlp.py:31-47`。

### 6.5 MoE 固定

`Moe.__init__` 固定顺序：

1. `RMSNormExpertProj`
2. `ExpertSelectUpGateSiLU`
3. `ExpertDownAllReduce`

位置：`TileRT/python/models/deepseek_v3_2/modules/moe.py:12-32`。

`MoeBlock.__init__` 固定：

```text
MLA -> MoE
```

位置：`modules/moe.py:35-51`。

### 6.6 MTP 固定

`MTP.__init__` 固定顺序：

1. `MTPPreprocessLayer`
2. `MoeBlock`
3. `RMSNormHeadProj`

MTP layer id 使用 `model_args.n_layers`，即第 61 层之后的 MTP 层，位置：`TileRT/python/models/deepseek_v3_2/modules/mtp.py:19-35`。

`MTP.get_weights_list()` 固定先放：

1. `embed_tokens_weight`
2. `freqs_cis`
3. 其他 MTP 子 op weights

位置：`modules/mtp.py:42-47`。

## 7. native op 入口固定

Python op wrapper 基本都只是把固定参数列表交给 `torch.ops.tilert.*`。可见的 native op 入口包括：

| 范围 | native op | 代码位置 |
|---|---|---|
| end2end prepare | `dsa_show_hands_prepare_money` / `dsa_mtp_e2e_show_hands_prepare_money` / glm5 variants | `modules/end2end.py:27-44` |
| end2end forward | `dsa_show_hands` / `dsa_mtp_e2e_show_hands` / glm5 variants | `modules/end2end.py:47-52` |
| reset/cleanup | `dsa*_show_hands_reset` / `dsa*_show_hands_go_home` | `modules/end2end.py:55-68` |
| sampler seed | `dsa*_show_hands_set_sampling_seed` | `modules/end2end.py:71-82` |
| prefill/MTP | `dsa_mtp_e2e_show_hands_set_prefill_valid_tokens` | `modules/end2end.py:85-99` |
| prefill/MTP | `dsa_mtp_e2e_show_hands_set_prefill_mtp_extra_token` | `modules/end2end.py:102-114` |
| MTP preprocess | `mtp_preprocess_layer` | `modules/mtp_preprocess.py:20-29` |
| MLA Q/KV/KI | `rmsnorm_proj_qa_kva_ki_op` | `ops/rmsnorm_projx_wqkvia.py:57` |
| MLA proj x | `projx_wqkvia_op` / `projx_wqkvia_glm5` | `ops/rmsnorm_projx_wqkvia.py:99-101` |
| RoPE | `layernorm_rope_rotate_op` | `ops/layernorm_rope_rotate.py:71` |
| Q/IQ proj | `rmsnorm_proj_qb_iq_op` / `rmsnorm_proj_qb_iq_glm5_op` | `ops/rmsnorm_projq_wqib.py:38-40` |
| projection | `proj_w_op` / `proj_w_glm5_op` | `ops/projx_wis.py:37-39` |
| projection | `projq_wqb_op` / `proj_qb_glm5_op` | `ops/projq_wqb.py:42-44` |
| KV norm/cache | `rmsnorm_kv_op` | `ops/rmsnorm_kv.py:37` |
| attention output | `projo_wkvb_op` / `proj_ob_glm5_op` | `ops/projo_wkvb.py:43-45` |
| attention allreduce | `unproj_o_allreduce_op` / `unproj_o_allreduce_glm5_op` | `ops/unproj_o_allreduce.py:49-54` |
| dense MLP | `rmsnorm_up_gate_silu_op` | `ops/rmsnorm_up_gate_silu.py:36` |
| dense allreduce | `down_allreduce_op` / `down_allreduce_glm5_op` | `ops/down_allreduce.py:51-86` |
| MoE router | `rmsnorm_expert_proj_op` | `ops/rmsnorm_expert_proj.py:148` |
| MoE select/up/gate | `expert_select_up_gate_silu_op` | `ops/expert_sel_up_gate_silu.py:50` |
| MoE down/allreduce | `expert_down_allreduce_op` / `expert_down_allreduce_glm5_op` | `ops/expert_down_allreduce.py:49-79` |
| sparse MLA | `flash_sparse_mla_op` / `flash_sparse_mla_glm5_op` | `ops/flash_sparse_mla.py:72-87` |
| sparse index | `sparse_index_op` / `sparse_index_glm5_op` | `ops/sparse_index.py:63-65` |
| sparse topk | `sparse_index_topk_glm5_op` | `ops/sparse_index.py:122` |
| sampler | `top_p_op` / `top_p_glm5_op` | `ops/top_p.py:49-51` |
| head | `rmsnorm_head_proj_op` / `rmsnorm_head_proj_glm5_op` | `ops/rmsnorm_head_proj.py:29-66` |

这里的固定化不只是函数名固定，更重要的是参数列表、输入输出 slot、权重 layout 都固定。通用框架里每一层可能经过 Python module call、dispatcher、operator overload、attention backend selection、sampler pipeline。TileRT 则把这些决策提前封装为 native op/plan。

## 8. 权重布局固定

### 8.1 权重命名固定

转换后权重 key 统一为：

```text
layer_{layer_idx}_{param_name}_dev_{device_id}
```

生成位置：`TileRT/python/models/preprocess/weight_converter.py:483-495`。

加载时按 device 过滤：

```python
weights_list = [_k for _k in weight_file_map.keys() if _k.endswith(f"dev_{device_id}")]
```

位置：`TileRT/python/models/deepseek_v3_2/modules/end2end.py:165-166`。

额外固定 key：

- `model.embed_tokens.weight`
- `layer_{n_layers}_lm_head.weight_dev_{device_id}`
- `layer_{n_layers}_model.norm.weight_dev_{device_id}`

位置：`modules/end2end.py:289-297`。

### 8.2 权重转换固定

`weight_converter.py` 把 HF 权重转成 TileRT 布局：

- MLA：`transform_mla()`，见 `weight_converter.py:251-275`。
- MoE：`transform_moe()`，见 `weight_converter.py:277-323`。
- Dense MLP：`transform_mlp()`，见 `weight_converter.py:325-380`。
- MTP：`transform_mtp()`，见 `weight_converter.py:382-411`。
- 按 layer 类型选择 MLP/MoE/MTP：`convert_a_layer()`，见 `weight_converter.py:413-443`。
- head/norm 特殊处理：`__process_head_weights()`，见 `weight_converter.py:445-472`。
- embedding 特殊处理：`__process_embedding_weights()`，见 `weight_converter.py:474-481`。

### 8.3 swizzle / reshape 固定

典型 converter 中存在大量硬件 layout 相关转换：

- `RMSNormProjxWqkvia` 中 `_swizzle_mma_16x32`、`_swizzle_qmma_16x32`，以及大量固定 reshape/transpose，见 `ops/rmsnorm_projx_wqkvia.py:118-683`。
- `RmsnormProjqWqib` 中 Q/IQ 权重按 head/nope/rope/page 重新排列，见 `ops/rmsnorm_projq_wqib.py:98-332`。
- `ExpertSelectUpGateSiLU` 中 expert gate/up 权重 reshape、pages、swizzle，见 `ops/expert_sel_up_gate_silu.py:114-291`。
- `ExpertDownAllReduce` 中 expert down 权重 swizzle 和 scale layout，见 `ops/expert_down_allreduce.py:94-158`。
- `UnProjOAllReduce` 中 unproj weight swizzle、scale pack，见 `ops/unproj_o_allreduce.py:102-207`。
- MTP preprocess 的 `eh_proj.weight` 固定 reshape 为 `[128, 7, 56, 256]` 方向，见 `modules/mtp_preprocess.py:91-104`。

消除的 runtime 工作：

- 不在 decode 步转置权重。
- 不在 kernel 内处理通用 strided layout。
- 不在 runtime 计算每设备 shard。
- native kernel 可以直接按最佳 tile layout 读权重。

这是 TileRT “模型 + 编译器/runtime + 硬件”耦合的核心证据之一：权重不是简单保存成框架默认 layout，而是提前变成 kernel/tile 期望的 layout。

## 9. prepare / graph / replay 固定

### 9.1 prepare_money 是关键分界线

`dsa_show_hands_prepare_money()` 接收：

- `params`
- `temp_vars`
- `cache_vars`
- `profile_logs`
- `forward_max_seq_len`
- `with_mtp`
- `is_glm5`

位置：`TileRT/python/models/deepseek_v3_2/modules/end2end.py:27-44`。

初始化完成后，对每个设备执行：

```python
dsa_show_hands_prepare_money(
    params,
    intermediates,
    caches,
    profile_logs,
    self.forward_max_seq_len,
    self.with_mtp,
    self.is_glm5,
)
```

位置：`modules/end2end.py:382-395`。

如果启用 MTP，还额外 prepare 一份非 MTP module：

```python
dsa_show_hands_prepare_money(
    params[: self._base_params_count],
    intermediates,
    caches[: self._base_caches_count],
    profile_logs,
    self.forward_max_seq_len,
    False,
    self.is_glm5,
)
```

位置：`modules/end2end.py:397-406`。

工程含义：

- prepare 阶段拿到了全部固定资源。
- native runtime 可以缓存指针、生成 graph、准备通信、创建内部 worker/thread、初始化 sampler 状态。
- MTP 和 non-MTP 都可以提前准备，运行时切换成本更低。

### 9.2 forward 热路径极薄

`forward()` 只做：

```python
active_mtp = with_mtp if with_mtp is not None else self.with_mtp
dsa_show_hands(token_id.cpu(), active_mtp, self.is_glm5)
return [self._get_device_result(device_id) for device_id in range(self.num_devices)]
```

位置：`TileRT/python/models/deepseek_v3_2/modules/end2end.py:418-425`。

注意：这里仍有 `token_id.cpu()`，说明 Python wrapper 暴露层还可能存在 host 侧 token 输入接口。真正性能路径中 native 内部是否避免同步，需要看 `libtilert.so`。但从接口设计看，除 token id 以外，其他所有输入都已在 prepare 阶段固定。

### 9.3 sampling config 变化需要 recapture

`update_sampling_config()` 的注释非常关键：

> Sampling parameters are baked into CUDA graph instructions at prepare_money time, so any change requires a full teardown + re-capture cycle.

位置：`TileRT/python/models/deepseek_v3_2/modules/end2end.py:186-193`。

代码流程：

1. 如果新旧 config 相同，直接返回，见 `end2end.py:194-197`。
2. 调用 `go_home` teardown，见 `end2end.py:204-209`。
3. 更新每设备 `intermediates[Idx.SAMPLING_CONFIG]`，见 `end2end.py:217-228`。
4. 对所有设备重新 prepare，见 `end2end.py:230-252`。

这说明 TileRT 为了 runtime 稳定，宁愿把 sampling 参数变化视为重新 capture 事件，也不把 sampling 作为每步动态参数处理。

## 10. sampler 和 MTP 固定

### 10.1 sampler 固定项

固定 slots：

- `SAMPLING_SEED`：`Idx=46`，shape `[B,S] int64`，`dsa.py:142`。
- `SAMPLING_POSITIONS`：`Idx=47`，shape `[B,S] int64`，`dsa.py:143`。
- `SAMPLING_CONFIG`：`Idx=48`，shape `[4] fp32`，`dsa.py:144-146`。
- `TOP_P_SCORES`：`Idx=49`，shape `[B,S] fp32`，`dsa.py:147`。
- `TOP_P_DEBUG`：`Idx=50`，shape `[B,S,16160] fp32`，`dsa.py:148`。

request-level seed 通过 native API 设置：

- `dsa_show_hands_set_sampling_seed()`，见 `end2end.py:71-82`。
- `ShowHandsDSALayer.set_sampling_seed()`，见 `end2end.py:427-437`。

含义：

- sampler 尽量 device-side / native-side。
- Python 不在每步拿 logits 回 CPU 做采样。
- top-p/top-k 参数变化被认为是 graph plan 变化。

### 10.2 MTP 固定项

MTP 相关 slots：

- `DRAFT_TOKENS`
- `PREDICTED_TOKENS`
- `PREDICTED_HIDDEN`
- `ACCEPTED_TOKENS`
- `NEXT_DRAFT_TOKENS`
- `MTP0_TOKEN_OUT`
- `MTP1_TOKEN_OUT`
- `MTP0_EXP_OUT`
- `LAST_HIDDEN_STATES`
- `EH_PROJ`
- `EMBEDDING_RMSNORM`
- `HIDDEN_RMSNORM`

MTP preprocess native op：

- `torch.ops.tilert.mtp_preprocess_layer(params, temp_vars, profile_logs)`，见 `modules/mtp_preprocess.py:20-29`。

MTP prefill 辅助 API：

- `set_prefill_valid_tokens()`，见 `modules/end2end.py:467-476`。
- `set_prefill_mtp_extra_token()`，见 `modules/end2end.py:478-484`。

读回 MTP 状态：

- `get_next_draft_tokens()` 读 `Idx.NEXT_DRAFT_TOKENS`，见 `end2end.py:486-496`。
- `get_num_accepted()` 读 `Idx.ACCEPTED_TOKENS`，见 `end2end.py:498-508`。
- `get_predicted_tokens()` 读 `Idx.PREDICTED_TOKENS`，见 `end2end.py:510-520`。

消除的 runtime 工作：

- 不需要 Python verifier 每步创建复杂结构。
- draft、verify、accept、next draft 都可在固定 buffer 内完成。
- 如果 native 内部实现充分融合，可以避免 logits/token 多次 D2H。

## 11. 仍然变化的内容

TileRT 固定了绝大多数结构，但 decode 当然仍有变化项。关键是这些变化项也被限制在固定槽位或 native 状态中：

| 变化项 | 如何承载 | 代码位置 | 备注 |
|---|---|---|---|
| 当前 token | `forward(token_id)` | `end2end.py:418-425` | wrapper 里传入 `token_id.cpu()` |
| 当前 position | `CUR_POS` 或 native global | `generator.py:489-517` | MTP 走 GPU tensor，non-MTP 走 C++ API |
| KV/cache 内容 | 固定 cache tensor 中的内容变化 | `modules/mla.py:89-107` | 地址和 layout 固定，内容更新 |
| last hidden | `LAST_HIDDEN_STATES` | `generator.py:518-548` | prefill-decode 解耦/MTP 需要 |
| sampling seed | native API / slot | `end2end.py:427-437` | request 级固定 |
| sampling position | `SAMPLING_POSITIONS` | `dsa.py:143` | 每步变化但 buffer 固定 |
| accepted length | `ACCEPTED_TOKENS` | `dsa.py:126` | MTP verifier 输出 |
| predicted/next draft | fixed MTP slots | `dsa.py:123-127` | 内容变化，slot 固定 |
| profile logs | `profile_logs` | `end2end.py:347` | profiling buffer 固定 |

这类设计目标是：变化数据可以变，但不要改变对象身份、地址、shape、dtype、op graph 和调度路径。

## 12. 与 vLLM-Ascend 的对照

vLLM-Ascend 保持更通用的框架形态，TileRT 的固定化方向可以对照出几个可优化点。

### 12.1 ACL graph wrapper 当前不负责 persistent buffers

`vllm-ascend/vllm_ascend/compilation/acl_graph.py:54-60` 注释明确说明：

- `ACLGraphWrapper` 不保存 persistent buffers。
- 不负责把 runtime inputs copy 到这些 buffers。
- 因为 wrapper 不假设 dynamic shape。

这与 TileRT 相反。TileRT 在 `prepare_money` 前已经把 params/temp/cache 绑定好，热路径只传 token。若要在 vLLM-Ascend 上实现 TileRT-like，需要在 graph wrapper 外增加一层专用 decode plan/buffer pool。

### 12.2 Ascend attention 仍依赖 CPU seq_lens

`vllm-ascend/vllm_ascend/worker/v2/input_batch.py:55-64`：

- 创建 `seq_lens_cpu`。
- 注释说明 NPU attention backend still needs seq_lens on CPU side。

`vllm-ascend/vllm_ascend/worker/v2/model_runner.py:108-119`：

- 需要把 `num_computed_tokens` copy back to CPU，用于更新实际 `seq_lens_cpu`。

这正是 TileRT-like 要优先消除的 runtime 开销之一：热路径中 CPU metadata 和 D2H sync 会破坏 graph replay 稳定性和 TPOT。

### 12.3 vLLM-Ascend 仍有较多 runtime mode / batch descriptor 分支

`ACLGraphWrapper.__call__` 每次根据 forward context 判断：

- runtime mode 是否为 NONE。
- runtime mode 是否匹配。
- batch descriptor 是否已有 graph entry。
- 若无则 capture，否则 replay。

位置：`acl_graph.py:110-142`。

这是通用引擎必要的灵活性，但 TileRT-like 单请求 decode 可以把这些分支提前变成：

```text
fixed decode bucket -> fixed graph entry -> fixed input buffers -> replay
```

## 13. 如果迁移到 Ascend，需要固定哪些东西

建议把 TileRT 固定化思想映射为 vLLM-Ascend 上的几个新模块：

### 13.1 `AscendTileRTConfig`

固定模型和硬件配置：

- model family：GLM5.1 / DeepSeek V3.2
- `tp_size=8`
- `batch=1`
- `decode_seq_len=1/2/4`，优先 4 用于 MTP
- `max_seq_len`
- `num_layers`
- `dense/moe layer split`
- MLA dims
- MoE dims
- MTP enabled / draft len
- dtype / quant mode

### 13.2 `AscendTileRTTempVarSpec`

把 TileRT 的 51 个 slot 思路移植为 Ascend NPU tensor ABI：

- 每个 slot 有 name、idx、shape、dtype、alignment、lifetime。
- Python 和 C++/custom op 共用同一份 schema。
- 启动时做 ABI 校验。

目标是替代热路径中的动态 Python metadata。

### 13.3 `AscendTileRTBufferPool`

负责：

- 每设备分配连续 workspace。
- 为每个 slot 建 view。
- 固定 cache tensor。
- 固定 sampling/MTP state。
- 固定 graph input/output tensor。

对应 TileRT 代码参考：

- continuous storage：`end2end.py:254-275`
- temp vars：`dsa.py:47-154`
- cache vars：`mla.py:89-107`

### 13.4 `AscendTileRTWeightPacker`

负责：

- HF/vLLM 权重 -> Ascend cube/vector kernel layout。
- per-device shard。
- NZ/ND、FP8/INT8/BF16 scale layout。
- MoE expert packed layout。
- MLA Q/KV/IQ packed layout。
- head/vocab shard layout。

对应 TileRT 代码参考：

- `weight_converter.py:251-443`
- `ops/rmsnorm_projx_wqkvia.py`
- `ops/rmsnorm_projq_wqib.py`
- `ops/expert_sel_up_gate_silu.py`
- `ops/expert_down_allreduce.py`
- `ops/unproj_o_allreduce.py`

### 13.5 `AscendTileRTDecodePlan`

负责把固定 op DAG 编成 NPU decode plan：

- 61 层固定展开。
- dense/MoE 层分支在初始化展开。
- 每个 op 绑定固定 input/output slot。
- 每个 allreduce/MC2 绑定固定 communication buffer。
- 每个 graph bucket 绑定固定 tensor 地址。

对应 TileRT 代码参考：

- `SerializableTileRTModule.exec_seq`：`models/base.py:224-259`
- `Dsa`：`modules/dsa.py:13-45`
- `Mla/Mlp/Moe/MTP`：`modules/mla.py`、`modules/mlp.py`、`modules/moe.py`、`modules/mtp.py`

### 13.6 `AscendTileRTGraphManager`

负责：

- prepare/capture。
- replay。
- sampling config 变更触发 recapture。
- MTP/non-MTP 双 graph。
- graph hit rate 统计。

对应 TileRT 代码参考：

- `prepare_money`：`modules/end2end.py:27-44,382-406`
- `forward`：`modules/end2end.py:418-425`
- `update_sampling_config`：`modules/end2end.py:186-242`

### 13.7 `AscendTileRTSamplerVerifier`

负责：

- device-side top-k/top-p。
- MTP draft/verify/accept。
- accepted tokens 写固定 slot。
- next draft tokens 写固定 slot。
- 避免 logits/token D2H。

对应 TileRT 代码参考：

- sampler slots：`dsa.py:142-148`
- MTP slots：`dsa.py:123-140`
- MTP APIs：`end2end.py:467-520`

## 14. 固定化优先级

如果目标是在 Ascend A2/A3 上尽可能消除 runtime，建议按收益和可落地性排序：

| 优先级 | 固定化目标 | 原因 | 验证指标 |
|---:|---|---|---|
| P0 | 固定单请求 decode shape 和 persistent input buffers | graph replay 稳定的前提 | graph hit rate、input address stable |
| P0 | 消除 CPU `seq_lens` / D2H sync | D2H 会直接拉高 TPOT | D2H count、stream sync count |
| P0 | 固定 KV/cache layout 和 cur_pos device-side 更新 | decode 每步都访问 | TPOT、ACL graph replay rate |
| P1 | 固定 sampler/device-side sampler | 避免 logits 回 CPU | logits D2H、sampler latency |
| P1 | 固定 MTP verifier state | 400 token/s 级别通常要靠 accepted length 放大 | accept_len_mean、visible step latency |
| P1 | 固定 MoE expert buffer 和 MC2 comm buffer | MoE decode 关键瓶颈 | HCCL/MC2 overlap、expert latency |
| P2 | 权重离线 pack 到 cube 友好 layout | 减少 kernel 内 layout 处理 | matmul kernel latency |
| P2 | MLA + sparse index workspace 固定 | 长上下文 decode 的关键 | MLA latency、sparse index latency |
| P2 | MTP/non-MTP 双 graph 预捕获 | 混合路径切换稳定 | graph miss count |

## 15. 最终判断

从当前 TileRT 开源 Python 层看，它消除 runtime 的主线非常清楚：

1. 用 `ModelArgs` 和固定 8 卡拓扑把模型结构、shape 和分片策略静态化。
2. 用 51 个 `temp_vars` slot 建立 Python/C++ 共享 ABI。
3. 用连续 storage 固定 workspace 地址。
4. 用 `exec_seq` 固定 op DAG、weights list 和 cache list。
5. 用离线 weight converter 把权重变成 native/tile layout。
6. 用 `prepare_money` 把 params/temp/cache/profile/shape 交给 native runtime 一次性准备。
7. 用极薄的 `forward(token_id)` 替代通用框架每步的调度、metadata、采样和图选择。
8. 用 device-side sampler/MTP fixed slots 避免 verifier 和 sampler 回到 Python/CPU。

对 vLLM/SGLang 这类通用框架来说，这些固定化牺牲了泛化能力，但直接瞄准了单请求低延迟 decode 的核心瓶颈。若要在 Ascend A2/A3 上复刻 TileRT-like 效果，最先要做的不是写 mega kernel，而是先把固定 ABI、persistent buffer、device-side metadata、graph replay 和 sampler/verifier 固定下来。否则即使 kernel 很快，runtime metadata、D2H sync、graph miss 和调度分支仍会吃掉低延迟收益。

