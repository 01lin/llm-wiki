# vLLM-Ascend 投机解码成熟度分析 —— 聚焦 DeepSeek-V4-Flash @ 910B/910C

> 范围：vllm-ascend（HEAD 已 ff 至最新 `0b5223c5`）
> 重点：DeepSeek-V4-Flash 在昇腾 910B(Atlas A2)/910C(Atlas A3) 的 MTP / DFlash 投机解码实现状态与效果
> 证据：源码（`vllm_ascend/spec_decode/`、`ascend_config.py`、`models/deepseek_v4_mtp.py`）+ 官方教程 `docs/.../DeepSeek-V4-Flash.md` + git 提交记录（含 PR 号）
> 生成日期：2026-06-11

---

## 0. 结论速览（TL;DR）

1. **DSv4-Flash 在昇腾上的生产投机路径是 MTP，不是 DFlash。** 官方权重即 `DeepSeek-V4-Flash-w8a8-mtp`，教程推荐 `method=mtp, num_speculative_tokens=1, enforce_eager=true`。**DFlash 当前只接 Qwen3（`DFlashQwen3ForCausalLM`），尚未接 DSv4。**
2. **架构上 MTP/Eagle/Eagle3 已统一到一个 proposer（`AscendEagleProposer`）**，DFlash 以子类扩展（`AscendDflashProposer`）。说明 MTP 复用了成熟的 Eagle 验证/采样链路，成熟度较高。
3. **功能面已相当完整**：单步/多步草稿、async scheduling、PD 分离（MooncakeHybridConnector）、与 DSA-CP/PCP/DCP 并行组合、D2D 草稿权重加载均已落地；近 6 周大量提交在**修复 spec × 并行组合**的边界问题，处于「功能齐备、组合稳定化」阶段。
4. **接受率优化有新抓手**：新引入 **Block Verify + Entropy Verify**（`RejectionSamplerConfig`），在标准拒绝采样之上做块级累积概率验证 + 熵自适应阈值。
5. **成熟度缺口**：DSv4 默认 `num_speculative_tokens=1`（仅 2× 理论上限）、草稿 `enforce_eager=true`（MTP 层尚未默认图捕获，留有开销）、仓内无公开的接受率/吞吐硬指标（评测走 AISBench/lm_eval 方法学）。

---

## 1. 投机解码总体架构与方法矩阵

派发入口 `vllm_ascend/spec_decode/__init__.py: get_spec_decode_method()`：

| method | Proposer 类 | 说明 | 目标模型 |
|--------|-------------|------|----------|
| `ngram` | `AscendNgramProposer` | n-gram 草稿 | 通用 |
| `ngram_gpu` | `AscendNgramProposerNPU` | NPU 上 n-gram，兼容 async scheduler | 通用 |
| `suffix` | `AscendSuffixDecodingProposer` | 后缀投机 | 通用 |
| `medusa` | `AscendMedusaProposer` | Medusa 头 | 通用 |
| **`mtp` / `eagle` / `eagle3`** | **`AscendEagleProposer`** | **三者统一实现** | DSv4(MTP)、各 Eagle 模型 |
| **`dflash`** | **`AscendDflashProposer`**（继承 Eagle） | 交叉注意力并行草稿 | **目前仅 Qwen3** |
| `draft_model` | `AscendDraftModelProposer` | 独立 draft model | 通用 |
| `extract_hidden_states` | `AscendExtractHiddenStatesProposer` | 隐状态抽取式 | 研究/特性 |

**要点**：`mtp` 与 `eagle/eagle3` 走同一个 `AscendEagleProposer`——MTP 在昇腾上本质是「单层草稿 + Eagle 式验证」，因此直接继承了 Eagle 路径的工程成熟度（CUDA/ACL graph、rejection sampler、PCP/DCP 适配）。基类已重构进 `llm_base_proposer.py`（`AscendSpecDecodeBaseProposer`，#9251）。

---

## 2. DeepSeek-V4-Flash @ 昇腾：官方部署与配置现状

> 910B = Atlas 800 A2（64G×8）；910C = Atlas 800 A3（128G×8）。官方教程对 **A2/A3 双系列均给出完整脚本**。

### 2.1 模型与硬件
- 权重：`DeepSeek-V4-Flash-w8a8-mtp`（W8A8 量化 + 内置 MTP 层）。
- A2：1 节点（64G×8）；A3：1 节点（128G×8）可单机部署。

### 2.2 官方推荐投机配置（A2/A3 一致）
```bash
--speculative-config '{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}'
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
--async-scheduling
--block-size 128
--additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},
                      "enable_cpu_binding":true, "enable_dsa_cp":true,
                      "multistream_overlap_shared_expert":true}'
```
解读：
- **MTP 单步（k=1）**：每步主模型验证 1+1 token，接受则 2× 出字；保守但稳。
- **主模型 FULL_DECODE_ONLY 图捕获，草稿 `enforce_eager=true`**：MTP 草稿层目前 eager 执行，**未默认图捕获**——这是明确的性能留量点。
- **与昇腾特性叠加**：DSA-CP（稀疏注意力上下文并行）、shared-expert 多流 overlap、npugraph_ex、CPU 绑核协同启用。

### 2.3 PD 分离（A3 多机 1P1D）
- 经 `MooncakeHybridConnector` 做 KV producer/consumer；prefill `dp4/tp4`、decode `dp16/tp1`，decode 端追加 `recompute_scheduler_enable`。
- 说明 **MTP 投机已可与 PD 分离 + 大 DP decode 组合**，是集群级成熟度的关键信号。

### 2.4 评测方式
- 精度：lm_eval（gsm8k 等）/ AISBench；性能：vLLM benchmark（latency/serve/throughput）/ AISBench。
- **仓内文档未给硬编码的接受率/TPOT/吞吐数字**，为方法学引导 → 量化效果需本地实测（见第 6 节建议）。

---

## 3. MTP 实现状态（DSv4 主路径）

### 3.1 算法链路（`Multi_Token_Prediction` 指南 + 代码）
- 主模型同时算 `1+k` token：第 1 个恒对，第 2 个为 **bonus token**（来自 MTP 草稿，需验证）。
- 验证：`AscendRejectionSampler.forward`，两种策略：
  - **Greedy**：草稿 token 与主模型 argmax 一致则接受。
  - **Rejection Sampling**：`P_target/P_draft ≥ U` 接受；拒绝则从 `Q=max(P_target−P_draft,0)` 重采样。当前 MTP 实现 `P_draft` 缺省为 1，故简化为 `P_target ≥ U`。

### 3.2 多步草稿能力
- `llm_base_proposer.py` 中 `for draft_step in range(1, self.num_speculative_tokens)` 等逻辑表明 **k>1 多步 MTP 在代码层已实现**（含 attn metadata 逐步更新 `attn_update_stack_num_spec_norm`）。
- **但 DSv4-Flash 官方默认仍 k=1**——多步为能力储备，未作为推荐配置（精度/接受率/稳定性权衡）。

### 3.3 接受率优化：Block Verify + Entropy Verify（较新）
`ascend_config.py: RejectionSamplerConfig`：
- **Block Verify**：把多个草稿 token 作为一个 block，用**累积概率乘积**整体验证，提高接受率。
- **Entropy Verify**：按目标分布**熵自适应**调接受阈值——高熵（不确定）放宽、低熵（确定）收紧；参数 `posterior_threshold=0.95`、`posterior_alpha=0.4`。
- 由 `[Feature][SpecDecode] fix Magic MTP and add Entropy Verify (#9772)` 引入，属**接受率维度的新抓手**，默认关闭、可经 `--additional-config` 开启。

---

## 4. DFlash 实现状态

- **Proposer**：`AscendDflashProposer`（继承 Eagle），交叉注意力——context K/V 取自 target hidden states，Q 取自 query embedding（bonus + mask token），即 **parallel drafting（一次前向出多 token）**，对应 `parallel_drafting` 分支。
- **模型绑定**：`from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM` + patch `precompute_and_store_context_kv` → **目前仅 Qwen3 系**。
- **图支持**：`[Feature] Support the merged graph for dflash (#9074)` 已支持合并图；`[BugFix] Keep draft lm_head for DFlash with reduced (d2t) vocab (#9795)` 修复缩减词表场景。
- **结论**：DFlash 工程已成型（parallel drafting + merged graph + d2t 词表），**但尚未接入 DeepSeek-V4**。**对 DSv4-Flash 而言，DFlash 暂不可用，MTP 是唯一生产路径。**

---

## 5. 近期成熟度演进（按提交，spec_decode 相关）

| PR | 内容 | 维度 |
|----|------|------|
| #9270 / #9757 | Support DeepseekV4 / DeepSeek-V4 for **Ascend950** | 模型+硬件落地 |
| #9772 | fix Magic MTP + **Entropy Verify** | 接受率/正确性 |
| #10043 | Reject placeholder draft tokens in sampler | 正确性 |
| #9735 | Reduce-sampling 重构、**消除 patch 行为、同时支持 DFlash 与 MTP** | 架构收敛 |
| #9678 | pcp/dcp + spec decode 的 **irregular mask** 构建 | 并行组合 |
| #10172 | 修 **qwen3.5 + pcp + mtp** 单 batch 错误 | 并行组合 |
| #9531 | DSA context parallel for DSv4（与 spec 协同） | 稀疏注意力并行 |
| #9893 | **D2D netloader 加载草稿权重** | 部署效率 |
| #9717 / #9867 | 草稿 DCP 校验 / token_indices_to_sample 越界修复 | 健壮性 |
| #9703 | 修 Eagle3 + MLA shape 不匹配，加 DSv2 Eagle3 | 模型覆盖 |
| #9074 | DFlash merged graph | 图捕获 |
| #9251 | 抽出 `AscendSpecDecodeBaseProposer` 到 base | 架构 |

**判断**：5 月以来主线从「新增方法」转向「**spec × 并行（CP/PCP/DCP）× PD 分离**的组合稳定化 + 接受率优化」，说明 **MTP 核心功能已过开发期，进入稳定化/调优期**；多并行组合仍在持续修边界（健壮性尚未完全收敛）。

---

## 6. 成熟度评估与缺口

### 已成熟（可生产）
- ✅ DSv4-Flash MTP 单步：A2/A3 单机 + PD 分离均有官方脚本。
- ✅ MTP 复用 Eagle 验证链路 + rejection sampler；async scheduling、block-size 128、W8A8 量化协同。
- ✅ 与昇腾特性叠加：DSA-CP、shared-expert 多流 overlap、npugraph_ex、CPU 绑核。

### 留量 / 缺口
- ⚠️ **草稿 `enforce_eager=true`**：MTP 层未默认图捕获，decode 单步开销未压尽。
- ⚠️ **默认 k=1**：多步 MTP 代码就绪但非推荐配置，理论加速被限制在 ~2×。
- ⚠️ **DFlash 未接 DSv4**：DSv4 暂无 DFlash 这条「一次出多 token」的更高并行草稿路径。
- ⚠️ **无公开硬指标**：仓内无接受率/TPOT/吞吐数字，效果需实测。
- ⚠️ **并行组合仍在修边界**：spec × PCP/DCP/CP 近月多次 bugfix，复杂拓扑下需回归验证。

### 实测建议（补齐效果数据）
1. 固定 A2(910B)/A3(910C) 各一组，对比 **baseline vs MTP(k=1)** 的 TPOT / 吞吐 / 接受率（mean accepted length）。
2. 扫 `num_speculative_tokens ∈ {1,2,3}`，看接受率与端到端收益拐点。
3. 开/关 **Block Verify + Entropy Verify**（`posterior_threshold/alpha` 网格），量化接受率提升与精度影响。
4. PD 分离（A3 1P1D, decode dp16/tp1）下复测，验证集群级收益。
5. 若评估 DFlash 收益，需先推动 DSv4 接入 DFlash（当前仅 Qwen3）。

---

## 7. 一句话总结

> vLLM-Ascend 的投机解码在 DSv4-Flash 上已形成 **MTP 为主、Eagle 链路复用、PD/并行可组合** 的可生产能力，处于「功能齐备、组合稳定化、接受率调优」阶段；**DFlash 工程成型但尚未接 DSv4**。当前最大性能留量在「草稿图捕获（去 enforce_eager）」「多步草稿（k>1）」「Block/Entropy Verify 接受率调优」三处，且效果数据需本地实测补齐。

> 可延伸：① DSv4-Flash MTP 在 vLLM-Ascend vs tokenspeed(MTP 前缀复用 #361) 的实现与收益对比；② 推动 DFlash 接入 DSv4 的改造点评估；③ Block/Entropy Verify 接受率收益实测专题。
