# DeepSeek-V4-Flash bf16 为什么开 DCP 报错：源码级根因分析

> 日期：2026-06-16
> 触发：实测开 `decode-context-parallel-size=2` 报错；前序方案误将"开 DCP"列为"立即可拿"的建议，**未经代码层面确认**。本文用源码坐实根因。
> 代码基线：`vllm-ascend @ a57a8f0`（dsa_cp.py 关键断言 commit `871759a2`，2026-06-10）
> 实测配置：`run_node0.sh` —— `tensor-parallel-size=8` + `enable-expert-parallel` + **`speculative-config: deepseek_mtp, k=1`**（MTP 投机解码开启）

---

## 0. 结论先行（一句话根因）

> **不是 DCP 本身不支持 DeepSeek-V4，而是「DCP × MTP 投机解码」的 draft 元数据构建路径，对压缩层（C4/C128）的 CP 切分尚未实现——只实现了 SWA 层。** 由于 DeepSeek-V4-Flash 必然含 C4(compress_ratio=4)/C128(compress_ratio=128) 层，draft 构建时断言 `compressor_ratio <= 1` 必然失败。

**根因代码（决定性证据）：**
```python
# vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py:354
# 在 AscendDSACPMetadataBuilder.build_for_drafting() 内
assert self.compressor_ratio <= 1, "vLLM-Ascend only support SWA-layer for Deepseek-V4 now."
```

---

## 1. 完整证据链（逐环源码坐实，可复核）

### 环节 1：DeepSeek-V4 DSA 后端走 DSA-CP 路径（DCP>1 时）

`dsa_v1.py` 在 DCP 启用时切到 CP 版 builder/impl：
```python
# vllm-ascend/vllm_ascend/attention/dsa_v1.py:195
from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPMetadataBuilder
# :213
from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPImpl
```
→ 所以 DCP>1 时，DeepSeek-V4 的 attention 元数据由 `AscendDSACPMetadataBuilder` 构建。

### 环节 2：MTP 投机解码走 `build_for_drafting`（draft step 专用路径）

```python
# vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py:1644
attn_metadata = attn_metadata_builder.build_for_drafting(
    draft_step, common_attn_metadata, **extra_attn_metadata_args,
)
```
- `build_for_drafting` **只在投机解码 draft 时被调用**（全仓仅此一处调用 + `dsa_v1.py:1124`/`dsa_cp.py:347` 两处定义）。
- 实测 `run_node0.sh` 开了 `speculative-config: deepseek_mtp, num_speculative_tokens=1` → **必然走这条 draft 路径**。

### 环节 3：draft 路径断言只支持 SWA 层

```python
# dsa_cp.py:347-354  AscendDSACPMetadataBuilder.build_for_drafting()
def build_for_drafting(self, draft_step, common_attn_metadata, ...):
    assert self.compressor_ratio <= 1, \
        "vLLM-Ascend only support SWA-layer for Deepseek-V4 now."
```
其中 `compressor_ratio` 来源：
```python
# dsa_cp.py:186
self.compressor_ratio = getattr(kv_cache_spec, "compress_ratio", 0)
```

### 环节 4：DeepSeek-V4-Flash 必含 compress_ratio=4/128 的层

模型构造链确认各层 compress_ratio：
```python
# vllm-ascend/vllm_ascend/models/deepseek_v4.py:118-170
compress_ratio: int,                       # 层构造参数
self.compress_ratio = compress_ratio
# :128 page_size_padded 依 compress_ratio==4 分支
```
- 资料 [182107 §3.1](20260614-182107-deepseek-v4-flash-kvcache申请占用管理-源码量化深度分析-分析.md) 已坐实：DeepSeek-V4-Flash 43 主层中 **21 层 C4(ratio=4)+ 20 层 C128(ratio=128)**，仅 2 层纯 SWA(ratio≤1)。
- → draft 构建遍历到任一 C4/C128 层时，`compressor_ratio = 4 或 128 > 1`，**断言失败，抛 AssertionError**。

### 证据链闭合

```
DCP>1 → DSA走CP builder(环节1)
  + MTP开启 → 走 build_for_drafting(环节2)
  + 遍历到 C4/C128 层(环节4)
  → compressor_ratio=4/128 > 1
  → assert compressor_ratio<=1 失败(环节3)
  → 报错 "vLLM-Ascend only support SWA-layer for Deepseek-V4 now"
```

---

## 2. 关键澄清：报错边界（哪些组合行，哪些不行）

### 2.1 主路径 build() 对 C4/C128 是有实现的（draft 路径才缺）

非 draft 的主元数据构建 `build()`（`dsa_cp.py:257`）对压缩层有完整处理：
```python
# dsa_cp.py:580  if self.compressor_ratio > 1:  →  layer_name = f"c{compressor_ratio}"
# dsa_cp.py:744/769/796/815  大量 compressor_ratio>1 / ==4 / !=4 分支
```
→ **DCP 的主 decode 路径支持 C4/C128**；缺的只是 **draft（投机）路径**。

### 2.2 报错的精确触发条件

| DCP | MTP/投机 | DeepSeek-V4 压缩层 | 结果 |
|-----|---------|------------------|------|
| =1 | 任意 | C4/C128 | ✅ 不走 CP 路径，正常 |
| >1 | **关闭** | C4/C128 | ✅ 走 `build()` 主路径（有实现）——**理论可行** |
| >1 | **开启(实测)** | C4/C128 | ❌ 走 `build_for_drafting`，断言失败 |

> **校准我之前的错误**：我说"开 DCP 立即可拿"是错的——在**实测配置（MTP 开启）下，DCP 直接报错**。准确表述应是：**DCP 与 MTP 不能同时开**（在当前 dsa_cp 实现下）。

### 2.3 platform.py 的 DCP 校验为什么没拦住

`platform.py:361 _validate_draft_decode_context_parallel_config` 有一段：
```python
# platform.py:379
if draft_model_config.use_mla:
    return   # MLA draft 直接放行，不走 GQA/MQA 的 DCP 校验
```
- DeepSeek-V4 MTP draft 是 MLA → 这里 `return` 提前放行 → **启动期校验通过，不报错**。
- 报错被推迟到**运行期 draft step 构建**（`build_for_drafting` 的 assert）。
- → 这解释了为什么启动看似正常、压测时才崩。

---

## 3. 对方案的影响与正确的应对

### 3.1 推翻"开 DCP 立即可拿"，给出代码确认后的真实选项

| 选项 | 可行性（代码确认） | 代价 |
|------|------------------|------|
| **DCP + 关 MTP** | ✅ 走 `build()` 主路径有实现 | 失去 MTP 加速（实测接受率 92.55%，acceptance length 1.93，≈省 0.93 token/步） |
| **DCP + MTP 同开** | ❌ `dsa_cp.py:354` 断言失败 | 需开发：在 `build_for_drafting` 补 C4/C128 的 CP 切分 |
| **不开 DCP（实测现状）** | ✅ 当前可跑 | KV 不摊卡，单卡 KV 仅 13.30 GiB |

### 3.2 开发选项：补齐 draft 路径的压缩层 CP（若要 DCP+MTP）

> 改动点明确：`dsa_cp.py:347 build_for_drafting`，把当前只支持 SWA（compressor_ratio≤1）的实现，扩展到 C4/C128。
> 难度评估：**高**——draft step 需为每个压缩层正确构造 CP 切分后的 slot_mapping / spec_slot_mapping（`dsa_cp.py:375-378`）和 compressed position（参考主路径 `build()` 的 `:580-815` 压缩层处理逻辑）。这是 kernel-adjacent 的元数据工程，非配置可解。

### 3.3 扩 KV 池的替代手段（DCP 受阻后）

既然 DCP+MTP 当前不可行，扩 KV 池（实测仅 13.30 GiB）的其他抓手：
| 手段 | 机理 | 代码确认状态 |
|------|------|------------|
| 关 MTP + 开 DCP | KV 摊 2 卡 | ✅ 代码支持，但损失 MTP |
| 权重量化(W8A8/FP8) | 权重占用减半，KV 池翻倍 | 需另查 vllm-ascend 量化对 DSV4 支持（下一步） |
| L2 host 卸载（方案 §2.4） | KV 卸 host | 方案核心，omni-cache 零拷贝 |
| 降 gpu-memory-utilization 的反向操作 / 精简 graph capture | 挤出更多 KV 预算 | 实测 capture_sizes=[1,2,4,8,16,32]，精简可省 graph 显存 |

---

## 4. 还需要的数据（反馈给实测方）

为精确量化 DCP/MTP 取舍，建议补测：
1. **DCP=2 + 关 MTP** 的一组：验证 §2.2「DCP 主路径可行」，并量化 KV 池是否翻倍（应从 13.30 → ~26 GiB 量级）、并发提升多少。
2. **关 MTP 单独一组**（DCP=1）：量化 MTP 对 TPOT 的实际贡献（实测 acceptance length 1.93，理论省 ~48% decode 步，但有 draft 开销）。
3. 启动日志完整 KV 段（`Available KV cache memory` / `GPU KV cache size` / `num_blocks`）在 **DCP=2 关 MTP** 下的值，坐实 KV 摊卡收益。

---

## 5. 一页纸总结

| 维度 | 结论 |
|------|------|
| **根因** | `dsa_cp.py:354` 断言 `compressor_ratio<=1`，draft 路径只实现 SWA 层 |
| **触发** | DCP>1 + MTP 开启 + 遍历到 C4/C128(ratio=4/128) → 断言失败 |
| **不是** | 不是 DCP 完全不支持 DeepSeek-V4（主 decode 路径 `build()` 支持压缩层） |
| **是** | 是 **DCP × MTP draft 路径** 对压缩层未实现 |
| **为什么启动不报错** | `platform.py:379` MLA draft 提前放行启动校验，错误推迟到运行期 |
| **我的纠错** | 前序"开 DCP 立即可拿"是**未经代码确认的错误建议**；实测配置（MTP 开）下 DCP 直接崩 |
| **正确选项** | DCP+关MTP（代码支持，损失 MTP）/ 或开发补齐 draft 压缩层 CP（难度高）/ 或走 L2 卸载扩容 |
| **待补数据** | DCP=2+关MTP 的 KV 池/并发/TPOT 实测 |
