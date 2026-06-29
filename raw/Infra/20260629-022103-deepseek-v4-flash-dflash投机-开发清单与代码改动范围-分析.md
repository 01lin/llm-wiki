# DeepSeek V4 Flash 实现 DFlash 投机 · 开发清单与代码改动范围

> 目标：在 vllm/vllm-ascend 上为 DeepSeek V4(Flash) 实现 DFlash 投机解码，评估开发要做哪些事、改哪些代码。
> 一手依据：vllm / vllm-ascend 本地源码 grep 实测 + Explore 探针交叉验证。锚点 vllm-ascend HEAD `a99b3b26`。
> 关联：[[20260629-005422-dspark-on-vllm-ascend-架构设计与实现方案-分析]]（DSpark 落地）。本地源码链接见文末（Obsidian 内可点，对话气泡点不开）。

---

## 0. 两个前置结论（实测纠正）

### 0.1 DFlash 支持哪些模型？—— 框架不限，但 draft 模型类只有 Qwen3 一个

- 分发表只认 `method=="dflash"` 字符串，不限模型（vllm-ascend spec_decode/__init__.py:44）。
- **但 dflash draft 模型类全仓只有 `qwen3_dflash.py` 一个**（`DFlashQwen3ForCausalLM`）；vllm-ascend `load_model` 也只断言这一个类（llm_base_proposer.py:689）。
- **GDN 算子坑（#10380 MAX_MTP=8 乱码 / #10088 task group 卡死）不是 dflash 的问题，是 Qwen3.5/3.6 hybrid 架构本身用了 GDN**。dflash 跑纯 dense 模型不碰 GDN。
- → 准确表述：**dflash 目前只有 Qwen3 draft 实现；GDN 坑源于模型架构，非 dflash 框架。**

### 0.2 DeepSeek V4 架构 —— 纯 MLA + sparse，无 GDN（重大利好）

实测：`grep -rniE "gdn|conv1d|gated_delta|linear_attn|recurrent" vllm/models/deepseek_v4/` **结果为空**。V4 = **MLA（Multi-head Latent Attention）+ SparseAttnIndexer + MoE + 自带 MTP**。

> **结论：V4 做 dflash 完全绕开 #10380/#10088——这俩根因都在 GDN 算子，V4 不用 GDN。这是 V4 路线相对 Qwen 路线最大的优势。**

---

## 1. 顶层判断：V4 做 DFlash = 写一个 MLA 版 draft 骨干

DFlash 的本质是「N 层并行骨干 + 交叉注意力(Q=draft, K/V=[context|draft]) + 预算 context KV」。Qwen3 版骨干用 **dense 注意力**；V4 版骨干要用 **MLA 注意力**。所以核心工作 = **照 `qwen3_dflash.py` 的范式，写一个 `deepseek_v4_dflash.py`，把 dense 注意力换成 V4 的 MLA**。

V4 自带 MTP（nvidia/mtp.py）是「逐步串行预测」，DFlash 是「并行骨干」——**两者范式不同，MTP 可参考但不能直接复用**；DFlash 范式独立，照 Qwen3 模板写。

---

## 2. 开发清单（要做哪些事）

| # | 任务 | 文件 | 说明 | 工作量 |
|---|------|------|------|--------|
| T1 | 新建 V4 DFlash draft 模型 | 新建 `vllm/model_executor/models/deepseek_v4_dflash.py` | 仿 qwen3_dflash，骨干用 V4 MLA 层 | 大(P0) |
| T2 | MLA 版 precompute_and_store_context_kv | 同上 | 把 Qwen3 的 dense KV 预算改成 MLA 的 latent KV 预算 | 大(P0) |
| T3 | combine_hidden_states 多层融合 | 同上 | V4 hidden 含 hc_mult 维度，融合前需 flatten | 中(P0) |
| T4 | 注册模型类 + load_model 断言放开 | 改 vllm 模型注册表 + [llm_base_proposer.py:689](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py:689) | 加 `DFlashDeepseekV4ForCausalLM` 到允许列表 | 小(P0) |
| T5 | proposer 侧适配(若 MLA KV 形态不同) | 可能改 [dflash_proposer.py](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py:63) `set_inputs_first_pass` | MLA KV cache 形态与 dense 不同 | 中(P0) |
| T6 | sparse indexer KV 预算(可选) | 改 T2 | 阶段一可禁用 sparse(compress_ratio=1)先跑通 | 中-高(P1) |
| T7 | 分层 RoPE theta | 改 T2 | V4 不同 compress_ratio 层 RoPE theta 不同 | 中(P1) |
| T8 | 训练 V4 DFlash draft 权重 | DeepSpec/外部 | 需用 V4 target 重生成数据训 draft | 大(并行/外部) |
| T9 | Ascend 算子确认 | vllm-ascend MLA backend | V4 MLA 在 Ascend 的 backend 复用 | 中(P0) |

---

## 3. 代码改动范围（分层）

### 3.1 vllm 侧（模型层，主战场）

**新建 `vllm/model_executor/models/deepseek_v4_dflash.py`**，照 [qwen3_dflash.py](vscode://file/Users/linyi/code/Documents/code/vllm/vllm/model_executor/models/qwen3_dflash.py:217) 范式：

| 部件 | Qwen3 版(模板) | V4 版要改成 |
|------|---------------|-----------|
| 骨干 DecoderLayer | DFlashQwen3DecoderLayer(dense) | 复用 V4 DeepseekV4DecoderLayer(MLA+MoE) |
| 注意力 | DFlashQwen3Attention(QKV) | V4 MLA(q_lora_rank/kv latent/分裂 RoPE) |
| precompute KV | dense qkv_proj split([qwen3_dflash.py:347](vscode://file/Users/linyi/code/Documents/code/vllm/vllm/model_executor/models/qwen3_dflash.py:347)) | MLA latent KV 路径(fused_wqa_wkv) |
| _build_fused_kv_buffers | [qwen3_dflash.py:289](vscode://file/Users/linyi/code/Documents/code/vllm/vllm/model_executor/models/qwen3_dflash.py:289) | 适配 MLA K 投影(两段) + K-norm |
| combine_hidden_states | fc 拼多层([qwen3_dflash.py:561](vscode://file/Users/linyi/code/Documents/code/vllm/vllm/model_executor/models/qwen3_dflash.py:561) 附近) | 加 hc_mult flatten |
| target_layer_ids 多层提取 | [qwen3_dflash.py:260](vscode://file/Users/linyi/code/Documents/code/vllm/vllm/model_executor/models/qwen3_dflash.py:260) | 同机制，配 V4 层号 |

参考但不复用：V4 MTP [nvidia/mtp.py](vscode://file/Users/linyi/code/Documents/code/vllm/vllm/models/deepseek_v4/nvidia/mtp.py)（逐步预测，范式不同）。

### 3.2 vllm-ascend 侧（proposer + 接入，改动小）

| 改动 | 文件 | 内容 |
|------|------|------|
| load_model 允许 V4 dflash | [llm_base_proposer.py:689](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py:689) | isinstance 列表加 `DFlashDeepseekV4ForCausalLM` |
| 模型 patch（如需） | 新建 `patch/worker/patch_deepseek_v4_dflash.py` | 仿 [patch_qwen3_dflash.py:62](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_dflash.py:62) 注入 MLA 版 precompute |
| proposer 复用 | [dflash_proposer.py:15](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py:15) | AscendDflashProposer 大部分可复用；若 MLA KV 形态不同，set_inputs_first_pass 需适配 |

> **关键**：proposer 层(AscendDflashProposer)是模型无关的并行骨干驱动，**大概率可直接复用**，主要工作在模型层(vllm 侧)。这与 DSpark 继承 dflash 是同一逻辑——站在 dflash proposer 肩上。

### 3.3 Ascend 算子侧（确认 > 新增）

| 能力 | V4 现状 | 工作 |
|------|---------|------|
| MLA attention | V4 本体已有 Ascend MLA backend | 复用，draft 骨干共用 | 
| sparse indexer | V4 本体已有 | 阶段一禁用(compress=1)，阶段二复用 |
| **GDN/conv1d** | **V4 不用** | **0——绕开所有 GDN 坑** |
| 采样/embedding/lm_head | 框架层 | 复用 |

> V4 做 dflash **几乎不需要新算子**（MLA/sparse V4 本体已有），这是相对 Qwen 路线(踩 GDN)的根本优势。

---

## 4. 嫁接难点（V4 MLA vs Qwen dense）

| 难点 | 说明 | 阶段 | 工作量 |
|------|------|------|--------|
| MLA ≠ dense KV 预算 | V4 K/V 走 latent(q_lora_rank/kv latent)，precompute 路径要重写 | P0 | 中 |
| sparse indexer KV | sparse 层 KV 是 FP8 量化特殊格式，预算难写 | P1 | 高(可先禁用) |
| hc_mult 维度 | V4 hidden 是 [T, hc_mult, D]，多层融合前 flatten | P0 | 低 |
| 分层 RoPE theta | 不同 compress_ratio 层 theta 不同 | P1 | 中 |
| MoE 路由偏差 | draft 路由可能偏离 target | 固有 | 无(投机固有) |

---

## 5. 实施路线（先跑通后优化）

```
阶段一 MVP（绕开难点，~2-3 周）：
  1. deepseek_v4_dflash.py 骨干复用 V4 MLA，禁用 sparse(compress_ratio=1) → verify: 前向 shape 对
  2. MLA 版 precompute_and_store_context_kv → verify: KV 预算不报错
  3. combine_hidden_states 加 hc_mult flatten → verify: 多层融合 shape 对
  4. 注册模型 + load_model 放开 + method="dflash" 起服务 → verify: 能启动
  5. e2e: V4 + dflash → verify: 接受率>0、无损(输出=target)
  ※ 全程不碰 GDN，绕开 #10380/#10088

阶段二 性能（~2-4 周）：
  6. 支持 sparse KV 预算(compress_ratio=4) → verify: 接受率/吞吐对齐 V4 本体
  7. 分层 RoPE theta → verify: 长上下文精度

阶段三 训练 + 调优（并行/外部）：
  8. 用 V4 target 训 DFlash draft 权重(DeepSpec 流水线) → verify: 接受率达标
```

---

## 6. 与 Qwen 路线对比（为什么 V4 路线更干净）

| 维度 | Qwen3.5/3.6 + dflash | DeepSeek V4 + dflash |
|------|---------------------|---------------------|
| 注意力 | hybrid(GDN + dense) | MLA + sparse |
| GDN 算子坑 | 🔴 #10380 乱码 + #10088 卡死 | ✅ 无 GDN，全绕开 |
| num_spec 上限 | ≤7(MAX_MTP=8 限制) | 不受 GDN 限制(取决 MLA 实现) |
| 新算子 | 需修 GDN tiling | 几乎 0(MLA/sparse 已有) |
| draft 模型 | 已有 qwen3_dflash | 需新写 deepseek_v4_dflash |
| 主要工作量 | 修算子坑 | 写 MLA 版 draft 骨干 |

> 战略判断：**V4 做 dflash 的难点在「写 MLA 版 draft 骨干」(工程量明确)，而非「踩 GDN 算子坑」(坑深且依赖 CANN)**。V4 路线虽然要新写模型，但没有 Qwen 路线那些 open 的算子级雷。

---

## 7. 源码索引（Obsidian 内可点 · 对话气泡点不开）

| 环节 | 跳转 |
|------|------|
| dflash 分发表 | [spec_decode/__init__.py:44](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/__init__.py:44) |
| dflash load_model 模型断言 | [llm_base_proposer.py:689](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py:689) |
| Qwen3 dflash 模型(模板) | [qwen3_dflash.py:217](vscode://file/Users/linyi/code/Documents/code/vllm/vllm/model_executor/models/qwen3_dflash.py:217) |
| precompute KV(模板) | [qwen3_dflash.py:347](vscode://file/Users/linyi/code/Documents/code/vllm/vllm/model_executor/models/qwen3_dflash.py:347) |
| _build_fused_kv_buffers | [qwen3_dflash.py:289](vscode://file/Users/linyi/code/Documents/code/vllm/vllm/model_executor/models/qwen3_dflash.py:289) |
| V4 MLA 注意力 | [deepseek_v4/attention.py:4](vscode://file/Users/linyi/code/Documents/code/vllm/vllm/models/deepseek_v4/attention.py:4) |
| V4 MTP(参考) | [deepseek_v4/nvidia/mtp.py](vscode://file/Users/linyi/code/Documents/code/vllm/vllm/models/deepseek_v4/nvidia/mtp.py) |
| Ascend dflash proposer | [dflash_proposer.py:15](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py:15) |
| Qwen3 模型 patch(模板) | [patch_qwen3_dflash.py:62](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_dflash.py:62) |

> 打开方式：Obsidian 内点链接(过一次"打开外部应用"确认框)即跳 VSCode；或复制路径 VSCode Cmd+P 粘贴 `路径:行号`。
