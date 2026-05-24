## 总体结论（先于表格）

V4 Flash 与现有 vLL以下为以下为续写并收束的版本：先给出总表**，再点明框架层与算子层的关键结论；其中算子层有一条

正在核对代码与配置，续写被截断的对比分析与总结。

下面是续写并完成收束的版本：接着你上一段被截断处（「算子层有一条……」），把 `index_topk` / Sparse-Attention 的 TopK 上界说清楚，并结合 HuggingFace 上两份真实 `config.json` 逐项对齐。

---

## 1. 总表：Flash vs Pro（HF）与本地实现差异

下表 `HF` 列摘自官方模型配置（你已提供的链接：`DeepSeek-V4-Flash`、`DeepSeek-V4-Pro`）。

|维度|DeepSeek‑V4‑Flash (HF)|DeepSeek‑V4‑Pro (HF)|当前 `vllm-ascend` 中与 Pro/Flash「真实结构」的一致性|
|---|---|---|---|
|`hidden_size`|4096|7168|仅用各自权重 + `DeepseekV4Config`；Pro 规模大，MoE 等路径需在显存/slot 上复核，但与「是否为 Pro」正交。|
|`num_hidden_layers`|43|61|层数由模型决定；`compress_ratios` 列表长度必须与层数匹配（Pro 末尾多 `4,0`）。|
|`num_attention_heads`|64|128|TP 切分时 `n_local_heads` 等会随之变化；DSA 主干仍假定 `head_dim=512`、`index_head_dim=128`，与两份 HF 一致。|
|`q_lora_rank`|1024|1536|权重映射里若假定 1024 会错；需在 `deepseek_v4` 线性层上与 HF 对齐。|
|`n_routed_experts`|256|384|MoE 路由与张量并行策略压力更大；不属于 DSA TopK，但整机压力显著上升。|
|`index_topk`|512|1024|不满足 Pro：见下文算子 512 硬封顶与 Python 仍写死 512。|
|`compress_ratios`（首二层）|`[0, 0, ...]`|`[128, 128, ...]`|不满足两份 HF：见下文 KV spec 与 `runner_only`。|
|`compress_ratios`（末层 Flash）|末项为 `0`|（Pro 最后一项为 `0`）|`compress_ratio==0`：模型分支与 DSA `else`(C128) 不一致风险。|

---

## 
2. 被截断句子的完整表述（算子层「那一条」）

算子层有一条非常关键、且已被代码「钉死」的约束：`sparse_attn_sharedkv` 在 tiling 侧要求 压缩 KV 稀疏索引的最后一维恰好等于 `TOPK_LIMIT`，而该常量为 512：

sparse_attn_sharedkv_tiling.hLines 125-125

constexpr uint32_t TOPK_LIMIT = 512;

对应检查逻辑（`cmpSparseIndices` 的 K 维必须为 512）：

sparse_attn_sharedkv_tiling.cppLines 1029-1034

OP_CHECK_IF((opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(DIM_NUM_THREE - 1) != TOPK_LIMIT),

OP_LOGE(opName_, "K should be %u, but got: %lld ",TOPK_LIMIT,

...

OP_CHECK_IF((opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(DIM_NUM_FOUR - 1) != TOPK_LIMIT),

而 HF Pro 明确为 `index_topk: 1024`——因此在 不修算子 tiling/内核与配套 metadata 形状的前提下，Pro 的 indexer–sparse-attn 整条链路在算子语义上就不闭合，这是比「Python 漏传配置」更深的硬约束。

同一问题在 Python 里也仍有体现：解码侧 metadata 构造里 `index_topk` 被写死为 512，未使用 `hf_config.index_topk`：

dsa_v1.pyLines 1013-1055

index_topk = 512

if self.compressor_ratio == 1:

...

cmp_topk=index_topk,

Indexer 运行时 `sparse_count` 仍为 512（与 HF Pro 1024 冲突）：

dsa_v1.pyLines 1894-1912

topk_idxs, _ = torch.ops._C_ascend.npu_quant_lightning_indexer(

...

sparse_count=512,

实现里还把注意力侧的 `index_topk` 固定成 512（未接 `hf_config`）：

dsa_v1.pyLines 1250-1252

self.index_topk = 512

因此：Framework 与算子需同时改，只改 Python 若 csrc 仍限制 512，Pro 仍跑不通或行为错误。

此外，`csrc/.../quant_lightning_indexer_topk.h` 中存在 `topK = 512` 的默认值/实现向痕迹，与 Pro 的 1024 同样存在潜在冲突，需与 `npu_quant_lightning_indexer_metadata` 的 `sparse_count` 一起做端到端核对。

---

## 3. 框架层：与 HF Flash/Pro 不一致的关键点（带代码）

### 3.1 `compress_ratios` 被本地配置覆盖

`DeepseekV4Config` 在 `__init__` 里 再次赋死 `compress_ratios`，覆盖构造参数里从 `config.json` 读入的值（因此即使用户提供 HF Pro/Flash，内存中的 pattern 仍可能是开发时用的 `[1,1,4,128,...]`）：

deepseek_v4.pyLines 230-236

self.compress_ratios = [

1, 1, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,

128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,

128, 4, 128, 4, 128, 4, 128, 4, 128, 4

]

HF Flash 实际是 `[0,0,...]`，HF Pro 前两层是 `[128,128,...]`，与此完全不一致。

### 3.2 KV cache spec：按 layer_id 奇偶绑定 C4/C128，不看 `compress_ratios`

model_runner_v1.pyLines 3424-3490

elif isinstance(attn_module, DSAAttention):

hf_config = self.model_config.hf_config

if layer_id in [0, 1] or "mtp" in layer_name:

self.runner_only_attn_layers.add(layer_name)

elif layer_id % 2 == 0:

...

compress_ratio=4,

...

elif layer_id % 2 != 0:

...

compress_ratio=128,

后果简述：

- Flash：前两应为 ratio 0（HF），但这里把 0、1 层强行放进 `runner_only` 并走 与「全层 compress 模式」不同的路径；且 偶/奇 与真实 4/128 交错 不一定一致。
- Pro：前两应为 C128，但 0、1 层仍被当作 `runner_only`，且 偶层 C4 与 Pro 的 首层即 128 冲突。

### 3.3 `initialize_kv_state` 假定存在 `compress_ratio == 1` 的层

model_runner_v1.pyLines 2626-2628

c1_layers = compress_ratio_to_layers[1]

c4_layers = compress_ratio_to_layers[4]

c128_layers = compress_ratio_to_layers[128]

HF Flash 的数组里没有 `1`，只有 `0/4/128`：若某日 真正按 HF 填入 `compress_ratios`，此处 `compress_ratio_to_layers[1]` 会 `KeyError`。当前之所以「能跑」，多半仍是因为 上文 3.1 把 ratio 改成了含 `1` 的死数组——这是 掩盖问题 而非 支持 HF。

同时 `runner_only` 断言 C1：

model_runner_v1.pyLines 2706-2708

for layer_name in self.runner_only_attn_layers:

layer_index = extract_layer_index(layer_name)

assert layer_index in c1_layers, "layer_index out of range"

与 Flash 前两层为 0、Pro 前两层为 128 直接矛盾——除非继续使用被覆盖的假 `compress_ratios`。

### 3.4 `compress_ratio == 0`（HF Flash/Pro 均出现）

模型侧 `compress_ratio > 1` 才有 Compressor，且 RoPE 的 `original_max_position_embeddings` 在 `compress_ratio > 0` 才取自 config：

deepseek_v4.pyLines 633-649

self.compress_ratio = config.compress_ratios[layer_idx]

if self.compress_ratio > 1:

...

self.rotary_emb = ComplexExpRotaryEmbedding(

...

original_max_position_embeddings=config.

rope_parameters['original_max_position_embeddings']

if self.compress_ratio > 0 else 0,

DSA 前向一般以 `==1 / ==4 / else→C128` 分支；0 既非 1 也非 4，会掉进 「类 C128」分支，语义是否与 HF「特殊层」一致 未见专门分支，属于 运行时风险点。

---

## 4. 可满足「复用」的部分（双方都常见、与本仓库一致）

- `head_dim=512`、`index_head_dim=128`、`sliding_window=128`：两份 HF 与当前 DSA/Compressor 设计 一致。
- Flash 的 `index_topk=512`：与当前 `TOPK_LIMIT=512`、Python 多处 512 一致；Pro 的 1024 才是突破点。

---

## 5. 总体结论：Pro 在框架层与算子层的主要缺口

|层级|主要不满足 Pro（及真实 HF Flash）的原因|
|---|---|
|配置|`compress_ratios` 被覆盖；`index_topk` 等未可靠贯通。|
|KV / Spec|C4/C128 由 层号奇偶 与 0/1 runner_only 决定，未按每层 `compress_ratios`；与 Pro 前两层 128、Flash 前两层 0 冲突。|
|DSA / Indexer|decode 与 `indexer_select_qli` 仍 512；与 Pro 1024 不符。|
|算子 csrc|`TOPK_LIMIT=512` 等 强制 K=512；不扩展则 Pro 无法合法执行。|

---

## 6. 工作量量级（粗估）

- 仅框架（配置 + KV spec 按层映射 + DSA Python 全改用 `hf_config.index_topk` + 修 `compress_ratio==0` 语义 + E2E）：约 1～2 名熟悉 vLLM KV/调度 的同学 × 1～2 周（视联调与测试环境而定）。
- 加上算子（放宽/重配 `sparse_attn_sharedkv` 的 TopK、同步 `quant_lightning_indexer`、回归 A3 BF16/FP8 路径）：整体上升到 算子+框架 联调级，常需 再 2～4 周（依赖 CANN/内核排期与压测）。

若你希望下一步落到「最小补丁清单（文件级）」或「Pro BF16-only 是否要跳过部分 FP8 路径」，可以说明你目标精度与是否必须严格对齐 HF 的 `compress_ratios`