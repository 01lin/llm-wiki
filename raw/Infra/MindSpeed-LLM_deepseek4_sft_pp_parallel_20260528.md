# MindSpeed-LLM DeepSeek4 SFT 与 PP 并行适配分析

分析对象：`/Users/linyi/code/Documents/code/MindSpeed-LLM/`

结论时间：2026-05-28

## 结论摘要

1. 仓库当前显式落地对象是 `DeepSeekV4-Flash`，不是单独的 `DeepSeekV4-Pro`。`examples/mcore/deepseek4_flash/README.md` 提到 DeepSeekV4-Flash 和 DeepSeekV4-Pro 同时发布，但随后明确说 MindSpeed LLM 当前实现的是 **DeepSeekV4-Flash 定长数据预训练**。
2. SFT 链路已经有代码和脚本：`tune_deepseek4_flash_4k_A3_ptd.sh` 通过 `posttrain_gpt.py` 启动，设置 `--stage sft --prompt-type deepseek4`，`AutoTrainer` 会选择 `DeepSeek4SFTTrainer`，模型 provider 会构造 `DeepSeek4Model`。
3. 但 README 的能力矩阵仍把“全参微调”标成 `DOING`，所以更准确的状态是：**已有 preview 级 SFT/全参微调脚本和代码路径，尚未在文档里标为完整 OK；仓库没有 DeepSeekV4-Pro 专属脚本或验证用例。**
4. DeepSeek4 SFT 官方示例脚本使用的是 **普通 PP 非交错 1F1B**：`PP=4`，没有设置 `--num-layers-per-virtual-pipeline-stage`，也没有设置 `--schedules-method dualpipev`。
5. 对 DeepSeek4 影响最大的 PP 专门适配是 **MHC 在 PP/VPP 下的激活张量形状和 VPP schedule 适配**：开启 `--enable-mhc` 后，pipeline 发送的激活形状从 `[S, B, H]` 变为 `[S, B, hc_mult, H]`；如果同时开启 VPP，则替换 interleaved 1F1B schedule。

## DeepSeek4 SFT 支持链路

### 文档声明

`examples/mcore/deepseek4_flash/README.md`：

- 第 3 行说明当前实现的是 `DeepSeekV4-Flash` 定长数据预训练支持。
- 第 22-28 行能力矩阵中，“全参微调”为 `DOING`，`Lora微调` 为 `TODO`。
- 第 51-53 行说明 `PP` 切分策略为 `OK`。
- 第 136-143 行又提供“全参微调”脚本入口。

因此文档本身呈现的是 preview 状态：训练脚本已经给出，但能力矩阵还没有把全参微调升到 OK。

### SFT 启动脚本

`examples/mcore/deepseek4_flash/tune_deepseek4_flash_4k_A3_ptd.sh`：

- 第 24-29 行：`TP=1, PP=4, EP=32, CP=1, NUM_LAYERS=44`。
- 第 42-57 行：开启 DSA/MHC 相关特性，包括 `--enable-mhc`、`--hc-mult 4`、`--use-triton-mhc`。
- 第 130-147 行：使用 `deepseek4_spec`、`DeepSeek4` mcore 模型、`--pipeline-model-parallel-size ${PP}`、`--noop-layers 43`。
- 第 211-216 行：SFT 关键开关为 `--finetune --stage sft --is-instruction-dataset --prompt-type deepseek4`。
- 第 218-230 行：实际入口是 `posttrain_gpt.py`。

这个脚本没有开启 VPP 或 dualpipev，所以脚本默认走 Megatron/MindSpeed 的普通非交错 PP schedule。

### Trainer 选择

`posttrain_gpt.py` 只做一件事：导入 `mindspeed_llm` 触发 patch，再构造 `AutoTrainer` 并调用 `train()`。

`mindspeed_llm/tasks/posttrain/launcher.py`：

- 第 20-24 行：当 `stage == "sft"` 且 `prompt_type == "deepseek4"` 时，返回 `DeepSeek4SFTTrainer()`。
- 如果开启 `--layerwise-disaggregated-training`，则优先返回 `LDTSFTTrainer()`。

`mindspeed_llm/tasks/posttrain/sft/sft_trainer.py`：

- `SFTTrainer.get_batch()` 处理 `input_ids/attention_mask/labels`，并支持中间 PP stage 只传递必要数据。
- `DeepSeek4SFTTrainer.model_provider()` 在第 258-294 行构造 `DeepSeek4Model`，导入 `args.spec` 指向的 `deepseek4_spec`，并按 `args.enable_mhc` 创建 `hc_head_spec`。

### Prompt/template 支持

`configs/finetune/templates.json` 注册了 `deepseek4` prompt type，`template_class` 为 `DeepSeek4Template`。

`mindspeed_llm/tasks/preprocess/templates.py` 中的 `DeepSeek4Template` 实现了 DeepSeek4 的特殊 token、thinking token、tool-calling 相关渲染逻辑。SFT 脚本传 `--prompt-type deepseek4` 后会走这套模板。

### 权重转换支持

`mindspeed_llm/features_manager/convert_checkpoint/convert_checkpoint.py` 把 `deepseek4` 和 `deepseek4_base` 加入 `--model-type-hf` choices。

`configs/checkpoint/model_cfg.json`：

- `deepseek4_base` 继承自 `deepseek32`，并设置 `qkv_type=pack_mla`、`multi_latent_attention=true`、`qk_layernorm=true`、`router_bias=true`、`enable_dsa_indexer=true`、`first_k_dense_replace=0`。
- `deepseek4` 继承 `deepseek4_base`，并设置 `tie_mtp_embeddings_and_lmhead=true`。

这说明转换层面使用的是通用 `deepseek4` 模型类型，不区分 Flash/Pro 名称。若要跑 Pro，至少需要确认 Pro 的 HF config、权重命名、层数、专家数、MTP、compress_ratios 等是否仍匹配该映射。

## PP 并行适配算法/策略梳理

这里把“PP 并行”分成三类：schedule 算法、层切分/权重布局策略、DeepSeek4 专用张量形状适配。

### 1. 普通 PP 非交错 1F1B

这是 DeepSeek4 SFT 示例实际使用的路径。

触发方式：

- `--pipeline-model-parallel-size ${PP}` 大于 1。
- 不设置 `--num-layers-per-virtual-pipeline-stage`。
- 不设置 `--schedules-method dualpipev`。

证据：

- SFT 脚本 `PP=4`，但没有 VPP/dualpipev 参数。
- `mindspeed_llm/training/training.py` 第 372-390 行显示：只有 `virtual_pipeline_model_parallel_size` 或 `schedules_method == dualpipev` 时才构造 list 型 data iterator；否则走普通 data iterator，即普通 pipeline schedule。

DeepSeek4 额外点：

- 因为 SFT 脚本开启了 `--enable-mhc`，即使普通 PP schedule 不被替换，也会通过 `MHCFeature` patch `get_tensor_shapes`，让 PP 通信张量带上 `hc_mult` 维度。

### 2. Virtual Pipeline Parallel / interleaved 1F1B

触发方式：

- `--num-layers-per-virtual-pipeline-stage N`。

通用文档说明：

- `docs/zh/pytorch/features/mcore/virtual_pipeline_parallel.md` 说明 VPP 将一个物理 PP stage 进一步切成多个 virtual stage，以更多通信换取更低 bubble。

DeepSeek4 专门适配：

- `mindspeed_llm/features_manager/transformer/mhc_feature.py` 第 28-36 行：开启 MHC 时 patch `get_tensor_shapes`；如果设置了 `num_layers_per_virtual_pipeline_stage`，再 patch `forward_backward_pipelining_with_interleaving` 为 `forward_backward_pipelining_with_interleaving_in_mhc`。
- `mindspeed_llm/tasks/models/transformer/deepseek4/mhc/mhc.py` 第 360-364 行：MHC 激活形状为 `(seq_length, micro_batch_size, hc_mult, hidden_size)`。
- 同文件第 368-384 行定义 MHC 版本 interleaved 1F1B schedule；第 514-517 行再次在 schedule 内部按 MHC 形状构造 `tensor_shape`；第 553-572 行使用 Megatron 的 schedule table 管理 virtual microbatch 到 model chunk 的映射。

验证/示例：

- `tests/poc/deepseek4_flash/pretrain_deepseek4_flash_4k_A3_ptd.sh` 使用 `PP=2`、`VPP=11`、`--enable-mhc`，说明 DeepSeek4 Flash 预训练 POC 覆盖过 MHC + PP + VPP。
- 但 `examples/mcore/deepseek4_flash/tune_deepseek4_flash_4k_A3_ptd.sh` 的 SFT 脚本没有开启 VPP。

### 3. DualPipeV / bidirectional pipeline

触发方式：

- `--schedules-method dualpipev`。

代码适配：

- `mindspeed_llm/features_manager/pipeline_parallel/dualpipev_feature.py` 第 16-39 行：当 `args.schedules_method == "dualpipev"` 时，patch：
  - `megatron.training.training.get_model`
  - `train_step`
  - `forward_backward_pipelining_without_interleaving` -> `forward_backward_pipelining_with_cutinhalf`
  - `Float16Module.forward`
  - `get_num_layers_to_build`
  - embedding grad allreduce
  - MTP embedding/output layer setup
- `convert_ckpt_v2.py` 第 91-95 行把 `--schedules-method` 限制为 `dualpipev`。

DeepSeek4 权重转换适配：

- `mindspeed_llm/tasks/checkpoint/convert_ckpt_deepseek4.py` 第 64-72 行：如果 `schedules_method == dualpipev`，则不允许同时设置普通 VPP，内部固定 `vpp_size=2` 并计算每个 virtual pipeline stage 的层数。
- 同文件第 1465-1472 行：dualpipev 下 post weight、norm、lm_head、MTP 层放在 `pp_rank == 0 && vpp_rank == vpp_size - 1`；非 dualpipev 下放在最后一个 PP/VPP stage。

当前 DeepSeek4 SFT 状态：

- 官方 DeepSeek4 Flash SFT 示例没有开启 `dualpipev`。
- 仓库里有 DeepSeek3/deepseek32 的 dualpipev 脚本；DeepSeek4 的转换器已经处理 dualpipev 布局，但没有看到 DeepSeek4 SFT dualpipev 示例或测试。因此这应视为“框架/转换层支持，DeepSeek4 SFT 未见明确验证”。

### 4. RiPipe / recompute-in-advance 类 PP schedule

`features_manager/__init__.py` 第 192-201 行把 `RiPipeSchedulesBubbleFeature()` 和 `RiPipeSchedulesAdvanceFeature()` 加入 PP feature 列表。

`mindspeed_llm/core/pipeline_parallel/schedules.py` 第 45-51 行有 wrapper：当 `args.recompute_in_advance` 且处于 grad-enabled 模式时，将 forward/backward 函数切到 `forward_backward_ripipe_pipelining`。

这是 MindSpeed 通用 PP 能力，不是 DeepSeek4 专用能力。DeepSeek4 SFT 示例没有启用 `--recompute-in-advance`。

### 5. Noop layers

这不是新的 schedule，但它是 PP 层切分/负载均衡的重要适配。

触发方式：

- `--noop-layers ...`

DeepSeek4 SFT 示例：

- `tune_deepseek4_flash_4k_A3_ptd.sh` 第 131 行设置 `--noop-layers 43`。

代码：

- `mindspeed_llm/features_manager/pipeline_parallel/noop_layers.py` 继承 MindSpeed 的 `NoopLayersFeature`，并在开启时 patch FLOPs 统计和 MoE metrics。
- `NumLayerListFeature` 第 35-38 行说明如果同时配置 `noop_layers`，`num_layer_list` 会被禁用。

含义：

- DeepSeek4 Flash 示例以 `NUM_LAYERS=44` 加 `noop-layers 43` 的形式保留一个空层，常见用途是对齐层数、PP/VPP 切分或处理 MTP/模型布局边界。

### 6. Num-layer-list / 非均匀 PP 切分

触发方式：

- `--num-layer-list a,b,c,...`

代码：

- `mindspeed_llm/features_manager/pipeline_parallel/num_layer_list.py` 第 24-34 行校验：
  - 必须 PP > 1；
  - 长度要等于 `pipeline_model_parallel_size`；
  - 不能和 VPP 同时开启；
  - 不能和 `dualpipev` 同时开启；
  - 遇到 `noop_layers` 会禁用。
- 第 50-59 行 patch `get_num_layers_to_build`、layer offset 和 `core_transformer_config_from_args`。

DeepSeek4 转换器：

- `convert_ckpt_deepseek4.py` 第 326-345 行同样校验 `num_layer_list` 与 VPP/noop 的互斥关系。

当前 DeepSeek4 SFT 示例没有使用 `num-layer-list`。

### 7. Layerwise Disaggregated Training / U-shape PP

触发方式：

- `--layerwise-disaggregated-training`
- 通常配合 `--num-layer-list` 和 `--num-virtual-stages-per-pipeline-rank 2`

代码：

- `launcher.py` 第 20-22 行：SFT 且开启 LDT 时，优先选择 `LDTSFTTrainer`。
- `u_shaped_split_feature.py` 第 25-64 行 patch：
  - `get_forward_backward_func`
  - `forward_backward_pipelining_without_interleaving`
  - `initialize_model_parallel`
  - PP p2p `_communicate/send_forward/send_backward`
  - `get_model/train_step/initialize_megatron`

文档：

- `docs/zh/pytorch/features/mcore/layerwise_disaggregated_training.md` 第 14-22 行说明 U-shape 切分：首尾层部署在边侧，中间层部署在云侧，样本不上云。
- 第 44-63 行说明其流水编排：先拆成首层/尾层两个逻辑流水并参考 1F1B，再合并并按 `FS-FE-BS-BE` 重排，目标是减少跨域通信带来的空泡。

限制：

- 文档明确当前支持范围是 Qwen2.5/Qwen3 LLM，暂不支持 MoE。因此这不是 DeepSeek4 Pro/Flash 的可用方案。

### 8. P2P 通信优化 / SendRecv 优化 / Unaligned pipeline

`features_manager/__init__.py` 第 192-201 行还加入了：

- `OptimizeP2PCommFeature`
- `OptimizeSendRecvCommFeature`
- `UnalignedPipelineFeature`

这些来自 `mindspeed.features_manager`，属于 MindSpeed 通用 PP 通信/非均衡 pipeline 能力。MindSpeed-LLM 侧只负责纳入 feature list；本仓库未给 DeepSeek4 SFT 的专门脚本验证。

## 面向 DeepSeekV4-Pro SFT 的判断

从本仓库看，不能直接说“已经完整支持 DeepSeekV4-Pro SFT”。更稳妥的判断如下：

1. **可复用基础链路**：模型类型、权重转换、prompt template、DeepSeek4SFTTrainer、DeepSeek4Model、MHC/DSA/MTP 这些都是以 `deepseek4` 为名实现，不强绑定 `flash` 字符串。
2. **缺少 Pro 专项验证**：没有 `deepseek4_pro` 目录、脚本、CI baseline 或 README 使用方法；README 只说当前实现 DeepSeekV4-Flash 定长数据预训练。
3. **SFT 是 preview 状态**：虽然脚本可跑 `--stage sft`，README 能力矩阵仍标 `全参微调 DOING`。
4. **PP 首选路径**：如果先让 Pro SFT 跑通，应优先沿用 Flash SFT 脚本的普通 PP 非交错 1F1B + MHC tensor shape patch + noop layer 方案；等 loss/数值稳定后再尝试 VPP 或 dualpipev。
5. **需要重新确认的 Pro 参数**：`num_layers`、`hidden_size`、`num_attention_heads`、`num_experts`、`EP/TP/PP`、`compress_ratios`、MTP 层数、HF 权重 key、FP8 反量化路径、tokenizer 特殊 token。

## PP 算法适配清单

| 类别 | 开关/入口 | DeepSeek4 SFT 示例状态 | 结论 |
| --- | --- | --- | --- |
| 普通 PP 非交错 1F1B | `--pipeline-model-parallel-size > 1` | 已使用，`PP=4` | DeepSeek4 SFT 示例实际路径 |
| VPP / interleaved 1F1B | `--num-layers-per-virtual-pipeline-stage` | SFT 示例未用，POC 预训练用 `PP=2,VPP=11` | DeepSeek4 MHC 有专门 schedule 适配 |
| MHC PP tensor shape | `--enable-mhc` | 已使用 | DeepSeek4 专用关键适配 |
| DualPipeV | `--schedules-method dualpipev` | SFT 示例未用 | 框架和 DeepSeek4 转换器支持布局，未见 DeepSeek4 SFT 验证 |
| RiPipe / recompute-in-advance | `--recompute-in-advance` | 未用 | MindSpeed 通用 PP schedule |
| Noop layers | `--noop-layers` | 已使用，`43` | DeepSeek4 示例使用的 PP 层布局辅助 |
| Num-layer-list | `--num-layer-list` | 未用 | 通用非均匀 PP 切分，和 VPP/dualpipe/noop 互斥 |
| LDT/U-shape PP | `--layerwise-disaggregated-training` | 不适用 | 当前文档限制为 Qwen2.5/Qwen3 LLM，暂不支持 MoE/DeepSeek4 |
| P2P/SendRecv 优化 | MindSpeed feature | 未见 DeepSeek4 SFT 专项 | 通用通信优化 |

## 建议

如果目标是“支持 DeepSeekV4-Pro 的 SFT 训练”，建议分三步：

1. 先按 `deepseek4` 通用转换链路接入 Pro 权重，最小改动复用 `tune_deepseek4_flash_4k_A3_ptd.sh`，只替换权重/tokenizer/模型规模/并行规模。
2. 第一版 PP 使用普通非交错 1F1B，不开 VPP/dualpipev；保留 MHC patch 和必要 noop layer，先验证 tokenizer、loss mask、MTP loss、MHC 激活 shape 和 MoE EP 分组。
3. 跑通后再扩展 VPP；dualpipev 需要单独验证 checkpoint 布局、MTP post weight 放置、embedding/lm_head 共享，以及 MHC 是否与 dualpipev schedule 完整兼容。
