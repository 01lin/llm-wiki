# 动态投机 @ vLLM/vLLM-Ascend 完整方案设计报告
## Qwen3.5/3.6（GDN Hybrid）与 DeepSeek V4 Flash/Pro（DSA 稀疏）双路线

> 生成时间：2026-06-25
> 走读：vllm `0d2961229` / vllm-ascend `8afdf356`，model_runner_v1 入口
> 整合来源：Codex 分析（[20260625-003651-...GDN-Mamba-state适配设计]）+ 本系列 6 篇（问题A复核 / conv1d基准 / 敏感变量普查 / pingpong移植 等）
> 纪律：每条给实现代码行号实测；Codex 引用已逐条核对；服务器侧 AscendC 代码无法直读处标 ❓。
> 关联：[[20260625-135108-vllm-ascend-投机长度敏感变量普查-精度风险位置识别-分析]] [[20260625-134201-gpu-causal_conv1d-spec索引基准-对照ascend修复-分析]] [[20260625-131524-gdn问题A修复复核与遗漏精度bug排查-分析]]

---

## 〇、核心原则（Codex 与本系列共识，互相坐实）

```text
容量按 max_k（buffer / graph capture 范围 / mamba blocks）—— 静态，合理
运行时按 active_k（metadata view / state 提交 / graph dispatch key）—— 动态
graph key 必须带 active_k（不能只按 num_tokens 命中）
recurrent/稀疏 state 只提交 accepted 前缀
不新增 D2H 同步（active_k 是 CPU 标量）
```

**核心不变量**：`active_D = active_k + 1` 必须贯穿下列全部变量同一轮取值——
`query_start_loc / decode_token_per_req / num_scheduled_tokens / num_draft_tokens / spec_state_indices.size(-1) / num_accepted 上界 / graph key`。任一停留在静态 `max_D` 即精度风险。

> 这一不变量 Codex 与本系列"投机长度敏感变量普查"完全一致，互相印证。

---

## 一、两个模型类型的本质差异（决定适配重点不同）

| 维度 | **Qwen3.5/3.6（GDN Hybrid）** | **DeepSeek V4 Flash/Pro（DSA 稀疏）** |
|------|------------------------------|--------------------------------------|
| KV 形态 | full-attn 层 KV + **GDN 线性层 recurrent state**（conv_state + ssm_state） | **HCS/CSA/SWA 三档稀疏压缩 KV**（位置寻址，无 recurrent state） |
| 投机难点 | 🔴 **recurrent state 必须按 accepted 前缀提交/回滚** | 🟠 稀疏 KV 的 slot_mapping/topk 布局按 active_D 对齐 |
| draft 结构 | MTP 1 层 full-attn（frozen-KV 复用 target KV） | DSA 多步 draft（`build_for_drafting(draft_step)`，dsa_v1.py:1124） |
| state 回滚 | **需要**（mamba 递归，rejected 污染不可逆） | **不需要**（KV 位置寻址，逻辑长度不推进即可） |
| 核心敏感算子 | `causal_conv1d_update` + `npu_recurrent_gated_delta_rule` | `npu_fused_infer_attention_score`（TND）+ DSA indexer |
| 静态投机变量 | `spec_q_per_seq`（gdn.py:443）/ `spec_state_indices.size(-1)` | `spec_multiple`（mla_v1.py:851）/ `actual_seq_lengths` |

> **关键判断（基于代码）**：**GDN 路线的精度风险远高于 DSA 路线**。GDN 有 recurrent state，target verify 处理 active_D 个 token 但只能提交 accepted 前缀，rejected token 的 state 不可逆污染下一轮；DSA 是稀疏 KV 位置寻址，rejected token 逻辑长度不推进即不可见（与 full-attn KV 同理）。**两路线的 RuntimeState 适配共享框架，但 GDN 多一层 recurrent state 提交/回滚的硬要求。**

---

## 二、RuntimeState 三件套适配（按模型类型分列）

### 2.A Qwen3.5/3.6（GDN Hybrid）

#### 2.A.1 Attention Metadata

| 变量 | 位置 | 适配 |
|------|------|------|
| `decode_token_per_req` | model_runner_v1.py:600 区域 / 写入 :3129 | 构造期 `max_D=max_k+1`，每轮传 `active_D` |
| `query_start_loc / gdn_query_start_loc` | model_runner_v1.py:929-940 / gdn_attn.py:224 | `cu_num_tokens` 由 active `num_scheduled_tokens` 生成 |
| `spec_state_indices_tensor` | gdn_attn.py:266/287 切 `:num_spec+1` | **改 `: active_D`**；buffer 按 max_D 申请、运行时 view active_D |
| `spec_token_indx` | gdn_attn.py:253-285 | `spec_token_size = num_spec_decodes * active_D`（非静态 num_spec+1） |
| `num_decode_draft_tokens` | model_runner_v1.py:1296-1322 / gpu_model_runner.py:2159 | 每 spec req 填 `active_k`（已是动态 `draft_len`，✅） |
| `max_query_len`（conv1d） | qwen_gdn_linear_attn.py:1355 | = `spec_state_indices.size(-1)` → 保证 = active_D |

#### 2.A.2 KV Cache + Recurrent State（GDN 路线核心）

| 项 | 位置 | 适配 |
|----|------|------|
| **full-attn KV 本体** | — | 不改（位置寻址，rejected 不推进即不可见） |
| **conv_state（滑窗）** | causal_conv1d spec kernel（GPU:causal_conv1d.py:850-851） | **按 `num_accepted-1` 做窗口滚动**（见 §三 ③-A）—— GDN 独有 |
| **ssm_state（递归）** | npu_recurrent_gated_delta_rule（ops/gdn.py:642-660） | `ssm_state_indices.flatten()` 宽度 = active_D；读用 `num_accepted-1` 列 |
| **Mamba align preprocess** | patch_mamba_utils.py:159-175 | `num_scheduled_tokens = active_D`（错→curr_state_idx 偏移，精度错） |
| **Mamba align postprocess** | mamba_utils.py:75-83 | `num_scheduled=active_D` + `num_draft=active_k` 配套 |
| **mamba 容量** | kv_cache_interface.py:607 / patch_mamba_utils.py:140 | `num_speculative_blocks` 保持 max_k（容量，不缩） |

#### 2.A.3 ACLGraph

| 项 | 位置 | 适配 |
|----|------|------|
| **graph key** | acl_graph.py BatchDescriptor / GraphParams:320 | key 加 `spec_k` / `decode_token_per_req`（不能只 num_tokens） |
| **conv1d_params 分桶** | acl_graph.py:325 `conv1d_params: dict[int,...]` | `[num_actual_tokens]` → `[(num_actual_tokens, active_D)]` |
| **conv1d host padding** | ops/gdn.py:443 `spec_q_per_seq=size(-1)` | 🔴 **R1**：`spec_q_per_seq` 用 active_D / 按 active_k 分桶捕获 |

### 2.B DeepSeek V4 Flash/Pro（DSA 稀疏）

#### 2.B.1 Attention Metadata

| 变量 | 位置 | 适配 |
|------|------|------|
| `spec_multiple` | mla_v1.py:851 `= num_speculative_tokens + 1`（**静态**） | 🟠 **DSA 侧 R1 等价物**：改为 active_D |
| `actual_seq_lengths` | mla_v1.py:853 `[spec_multiple*(i+1) ...]` | 用 active_D 算，否则 seq_lens 布局错位 |
| `decode_token_per_req` | mla_v1.py:626 `graph_pad_size // decode_token_per_req` | 传 active_D（graph padding 依赖） |
| `decode_threshold` | dsa_v1.py:390 / mla_v1.py:264 `+= spec_token_num` | 构造期 max，build/reorder 用 active_D |
| `spec_slot_mapping[draft_step]` | dsa_v1.py:1145/1211/1292 | 多步 draft 每步 slot_mapping 按 active 步数 |

#### 2.B.2 KV Cache（DSA 路线 —— 无 recurrent state，相对简单）

| 项 | 适配 |
|----|------|
| **HCS/CSA/SWA 稀疏 KV 本体** | 不改（位置寻址 + 压缩态，rejected 不推进即不可见） |
| **c4/c128 topk 列布局** | topk 选择按 active_D 的 token 数（draft token 也过 indexer） |
| **draft KV（多步）** | `build_for_drafting(draft_step)` 每步独立 metadata，draft_step 上界 = active_k |
| ❓ **DSA 的 num_accepted 提交** | DSA 无 recurrent state，但 topk/indexer 的 state 是否有跨步累积需核实 |

#### 2.B.3 ACLGraph

| 项 | 适配 |
|----|------|
| graph key | 同 GDN：带 active_k；DSA 用 `actual_seq_lengths` 形状，需 active_D 一致 |
| TND layout 限制 | `decode_threshold <= 16`（attention_v1.py:247 断言）—— active_D 不超限 |

---

## 三、精度异常关键点总结（按风险，已坐实代码）

> 整合本系列坐实的所有独立风险点。**这些是"修一个仍异常"的多 bug 叠加来源。**

### GDN 路线（Qwen）

| # | 风险点 | 代码依据 | 路线 |
|---|--------|---------|------|
| **③-A** | conv_state **窗口滚动**：按 `num_accepted-1` 滚动滑窗（[causal_conv1d.py:850-851](vllm/vllm/model_executor/layers/mamba/ops/causal_conv1d.py)）；Ascend 若未实现/做错 → conv 滑窗含错误 history/draft | GPU 基准坐实，Ascend ❓ | 🔴 GDN |
| **R1** | conv1d ACL Graph `spec_q_per_seq = size(-1) = num_spec+1`（静态，[ops/gdn.py:443](vllm-ascend/vllm_ascend/ops/gdn.py)）→ 动态 uql 时 padding token 布局错位 | ✅ 坐实 | 🔴 GDN |
| **问题A** | ssm recurrent kernel 读/写两套索引坐标系不一致、写缺 batch_i 行偏移（AscendC，服务器侧文档已修） | GPU 基准坐实，Ascend 已修 | 🟠 GDN |
| **R4** | align kernel `aligned_new_computed=(...)//block_size*block_size` 跨 step 对齐截断（[mamba_utils.py:84](vllm/vllm/v1/worker/mamba_utils.py)） | ✅ 坐实变量，跨step风险待动态验证 | 🟡 GDN |
| **③-B** | NULL_BLOCK_ID 写守卫（GPU `if final_state_idx>0`，causal_conv1d.py:815）；Ascend cudagraph padding 取脏块 | GPU 有，Ascend ❓ | 🟠 GDN |
| **③-C** | host stride 用 `len/B` 在 cudagraph padding 下 ≠ num_spec+1 | ✅ 推导坐实 | 🟠 GDN |

### DSA 路线（DeepSeek V4）

| # | 风险点 | 代码依据 | 路线 |
|---|--------|---------|------|
| **R1'** | `spec_multiple = num_speculative_tokens+1`（静态，[mla_v1.py:851](vllm-ascend/vllm_ascend/attention/mla_v1.py)）算 `actual_seq_lengths`/`num_tokens//spec_multiple` → 动态投机 seq_lens 布局错位 | ✅ 坐实 | 🔴 DSA |
| **D1** | `spec_slot_mapping[draft_step]`（dsa_v1.py:1145）多步 draft 的 slot 按静态步数 | ✅ 坐实变量 | 🟠 DSA |
| ❓ | DSA indexer/topk 是否有跨 draft_step 累积 state（类比 mamba） | 未坐实 | ❓ DSA |

> **共性根因**：两路线都是"**静态投机长度（num_spec+1 / spec_multiple）被用于运行时 token 布局**"。GDN 多一层 recurrent state 的提交/滚动/回滚（③-A/问题A/R4），DSA 无 recurrent state 故无此类。

---

## 四、方案架构（采纳 Codex 四层 + 本系列风险点）

```
DynamicSpecController      → 选 active_k（固定阶段返回 fixed_k=3）
DynamicSpecRuntimeState    → CPU 标量 (verify_k, max_k)，无 GPU 同步
DynamicSpecMetadataAdapter → 写 active_D 到 runner/attn/GDN-or-DSA metadata/graph key
DynamicSpecGraphDispatcher → graph key 带 spec_k；GDN conv1d_params 按 (num_tokens, active_D) 分桶
```

**容量 max_k / 运行时 active_k / graph key 带 active_k / state 只提交 accepted 前缀**——四层对两路线通用，差异在 MetadataAdapter 内部：GDN 走 conv/ssm state 提交，DSA 走 slot_mapping/seq_lens 对齐。

---

## 五、落地顺序（Codex 基础 + 双路线分叉）

1. **DynamicSpecRuntimeState（固定 fixed_k=3）** —— 两路线共用。
2. **Runner metadata 适配** —— `decode_token_per_req=active_D`、`num_decode_draft_tokens=active_k`。
3. **分路线 metadata**：
   - GDN：`spec_state_indices[:, :active_D]` + conv/ssm host meta + Mamba preprocess/postprocess。
   - DSA：`spec_multiple→active_D`、`actual_seq_lengths`、`spec_slot_mapping[draft_step]`。
4. **Graph key**：BatchDescriptor 加 spec_k；GDN conv1d_params 按 (num_tokens, active_D) 分桶。
5. **GDN recurrent state 校验**（仅 GDN）：③-A 窗口滚动 + 问题A索引 + R4 block 对齐 + NULL 守卫 —— 这是 GDN 独有、最难、最易残留。
6. **fixed_k=3 等价校验** → 通过后接 acceptance-aware 动态策略。

---

## 六、验证标准（fixed_k=3 等价 + 动态切换）

**等价校验**（两路线）：fixed_k=3 与静态 num_speculative_tokens=3 输出 bit-exact、accepted 分布一致、graph 命中同类、无新增 `.cpu()/.item()/synchronize()`。

**动态切换 mtp3→mtp1**：
- GDN：state postprocess 把 conv/ssm 固化到 accepted 前缀；step t+1 `spec_state_indices.size(-1)=2`、graph key spec_k=1；上轮 rejected state 残留物理块但不被逻辑 index 选中。
- DSA：`spec_multiple` 跟随 active_D；`actual_seq_lengths` 重算；rejected token 逻辑长度不推进。

**分离定位**（区分 bug 源）：
1. ACLGraph on/off → R1/R1'（graph padding）
2. conv vs ssm 分离 → ③-A vs 问题A（仅 GDN）
3. GDN vs DSA 模型对比 → 哪条路线的 bug
4. 固定 vs 动态切换 → 静态投机变量 vs 跨 step state

---

## 七、最终判断

1. **Codex 方案设计可靠、与本系列互相印证**：核心不变量 `active_D` 贯穿、四层架构、容量/运行时分离、async 兼容——结论一致。
2. **Codex 的缺口（本报告补全）**：① 几乎只覆盖 GDN，**DSA 路线（DeepSeek V4）的 `spec_multiple`/`actual_seq_lengths`/`spec_slot_mapping` 敏感点本报告坐实补上**；② conv_state 窗口滚动（③-A）、NULL 守卫、host stride（③-C）本系列已坐实，需并入 GDN 风险清单。
3. **两路线适配的本质差异**：
   - **GDN（Qwen）= 难**：recurrent state 必须按 accepted 前缀提交 + 窗口滚动 + 跨 step 回滚，且 conv/ssm 双 state、ACL Graph conv1d 分桶。精度风险点最多（6 个）。
   - **DSA（DeepSeek V4）= 相对易**：无 recurrent state，主要是 seq_lens/slot_mapping/topk 按 active_D 对齐；rejected token 位置寻址不污染。
4. **最关键原则**（两路线通用）：`容量按 max_k / 运行时按 active_k / graph key 带 active_k / state 只提交 accepted 前缀`。

---

## 八、待核实/存疑（服务器侧 AscendC 或需动态验证）

1. **Ascend conv1d 窗口滚动是否实现**（③-A，GDN P0）：服务器 AscendC，本地无法读。
2. **R1/R1' 的 graph 是否已按投机长度分桶捕获**：决定是否已规避。
3. **DSA indexer/topk 是否有跨 draft_step 累积 state**：DSA 是否真的"无 recurrent state"需在 dsa_v1 forward 坐实。
4. **align kernel block 对齐跨 step（R4）**：需 mtp3→mtp1 单步数值追踪。
5. **num_accepted 跨 step 归属**：当前 step 真实值 vs 上一步残留。
6. **Qwen3.6 / DeepSeek V4 Pro 与 3.5/Flash 的结构差异**：本报告按 3.5(GDN)/V4-Flash(DSA) 坐实，3.6/Pro 若结构有变（层配比/压缩档位）需单独核对 —— ❓ 本地无 3.6/Pro 模型定义，未坐实差异。
