# 昇腾 NPU DeepSeek-V4 多流遮掩特性：互斥关系、场景 Recipe 与算子级遮掩图解

> 生成时间：2026-06-25 13:41
> 问题：vllm-ascend 针对 DeepSeek-V4 的多个"多流异步遮掩"开关——互斥关系（代码依据）、场景亲和的最佳组合 recipe、每个开关具体遮掩了哪些算子（可视化）。
> 走读基准：vllm-ascend ascend_config.py / ops/fused_moe/fused_moe.py / attention/dsa_v1.py / utils.py。
> 证据纪律：互斥关系与遮掩算子全部行号 grep 实测。
> 关联：[[20260625-132332-昇腾DBO特性-deepseek-v4-910B3支持性-代码级判定-分析]]（DBO 不支持，本文是其真实替代）[[ascend-cluster-5to10x-architecture-spine]]

---

## 0. 五个多流遮掩开关总览（代码坐实，均默认 False）

| 开关 | 定义 | 第二条流 | 遮掩对象（算子级） |
|------|------|---------|------------------|
| `multistream_dsv4_dsa_overlap` | ascend_config.py:137 | `dsv4_dsa_overlap_stream()`(dsa_v1.py:51) | **MLA prolog 的 CV 并行**：Cube(matmul) ∥ Vector(quantize/norm/rope) |
| `multistream_dsa_preprocess` | ascend_config.py:136 | `attention_calculation_stream()` | DSA 预处理（indexer/kv quant_matmul）与注意力计算重叠 |
| `multistream_overlap_shared_expert` | ascend_config.py:131 | `shared_experts_calculation_stream()`(utils.py:488) | **共享专家计算** ∥ 路由专家 dispatch/combine 通信 |
| `multistream_overlap_gate` | ascend_config.py:132 | `AscendFusedMoE.gate_stream` | 共享专家计算 ∥ gate/router 路径（fused_moe.py:626 在 gate_stream 上跑 shared_experts） |
| `prefill_comm_compute_overlap` | ascend_config.py:138 | （SP 局部计算+延迟 allgather，非独立 stream） | prefill 阶段 wq_a/wkv 局部计算 ∥ allgather 通信 |

> 机制底座：`npu_stream_switch(target_stream, enabled=...)`(utils.py:1015) 切第二条 NPU stream + `record_event`/`wait_event`/`wait_stream` 做依赖同步。这是 NPU 上"计算通信遮掩"的真实实现（**DBO 不支持，见关联文档**）。

---

## 1. 互斥关系（代码依据，逐条坐实）

### 互斥 1：`multistream_dsv4_dsa_overlap` ⊥ `prefill_comm_compute_overlap`（multistream 优先）

**dsa_v1.py:1595-1606**：
```python
# Mutually exclusive with multistream_dsv4_dsa_overlap (multistream wins).
need_prefill_gather = (
    self.prefill_comm_compute_overlap
    and not self.multistream_dsv4_dsa_overlap   # ← 显式互斥：dsv4 开则 prefill_gather 失效
    and need_gather_q_kv
    and has_prefill
    and not has_decode
)
```
> **坐实**：`prefill_comm_compute_overlap` 的延迟 allgather 优化路径，被 `not self.multistream_dsv4_dsa_overlap` 守卫——两者同开时 **multistream_dsv4_dsa_overlap 赢**，prefill_comm 路径被旁路。注释明写 "Mutually exclusive ... multistream wins"。

### 互斥 2：`multistream_overlap_shared_expert` ⊥ `mix_placement` / `enable_shared_expert_dp`

**ascend_config.py:299-301**：
```python
def _check_mix_placement(self):
    if self.mix_placement:
        if self.enable_shared_expert_dp or self.multistream_overlap_shared_expert:
            raise ValueError("Mix placement is not supported with shared expert DP or multistream overlap.")
```
> **坐实**：开了 `mix_placement` 时，`multistream_overlap_shared_expert`（及 `enable_shared_expert_dp`）会 **raise ValueError 启动失败**。三者两两不可与 mix_placement 共存。

### 互斥 3（语义级，非 raise）：`multistream_overlap_gate` 与 `multistream_overlap_shared_expert` 抢同一对象

**fused_moe.py:624-626**（gate 路径）：
```python
with npu_stream_switch(AscendFusedMoE.gate_stream, enabled=self.multistream_overlap_gate):
    shared_out = fc3_context.shared_experts(hidden_states)   # ← gate_stream 上跑 shared_experts
```
**fused_moe.py:748**（shared_expert 路径）：
```python
with npu_stream_switch(shared_experts_calculation_stream(), enabled=self.multistream_overlap_shared_expert):
    ... # 也在另一条 stream 跑 shared_experts
```
> **坐实（语义互斥）**：两个开关**都把"共享专家计算"往各自的第二条 stream 放**（gate_stream vs shared_experts_calculation_stream）。同开会对同一计算产生两套 stream 调度路径——**应二选一**（代码无 raise，但语义冲突；需实测确认哪个生效或是否冲突）。⚠️ 这是组合时的真实坑。

### 非互斥（可叠加）
- `multistream_dsa_preprocess`（注意力预处理流）与 `multistream_overlap_shared_expert`（MoE 共享专家流）**作用在不同模块**（attention vs MoE），无代码守卫互斥 → 可叠加。
- `multistream_dsv4_dsa_overlap`（MLA prolog）与 `multistream_overlap_shared_expert`（MoE）**不同模块** → 可叠加。

---

## 2. 场景亲和的最佳组合 Recipe

> 底层逻辑：多流遮掩的收益 = 被遮掩的通信/串行算子占比 × 第二条流的空闲算力。**不同场景（prefill 重 vs decode 重、EP 规模、是否 mix placement）下，主导瓶颈不同，亲和的开关也不同。**

### 场景 A：Decode 主导 + 大规模 EP（agentic 多轮 decode 阶段，EP64）
**瓶颈**：MoE dispatch/combine all-to-all 通信占比高。
**亲和开关**：
```json
{"multistream_overlap_shared_expert": true,
 "multistream_dsv4_dsa_overlap": true,
 "multistream_dsa_preprocess": true}
```
**理由**：shared-expert 计算遮掩路由专家通信（主收益）；MLA prolog CV 并行 + DSA 预处理遮掩降 attention 串行。**不开 gate**（与 shared_expert 抢对象，互斥3）。**不开 prefill_comm**（decode 阶段无效）。

### 场景 B：Prefill 主导 + 长上下文（512K 首 token / chunked prefill 重段）
**瓶颈**：prefill 的 wq_a/wkv 投影 + allgather 通信。
**亲和开关**：
```json
{"prefill_comm_compute_overlap": true,
 "multistream_overlap_shared_expert": true}
```
**理由**：prefill_comm 延迟 allgather 隐藏通信。**此时不开 dsv4_dsa_overlap**（互斥1，会让 prefill_comm 失效）——若 prefill 段 MLA 投影是瓶颈则反过来：开 dsv4_dsa_overlap 关 prefill_comm，二选一看实测。

### 场景 C：PD 混合 / 通用均衡（P/D 同实例，无 mix_placement）
**亲和开关**：
```json
{"multistream_dsv4_dsa_overlap": true,
 "multistream_overlap_shared_expert": true,
 "multistream_dsa_preprocess": true}
```
**理由**：覆盖 MLA + MoE + DSA 三处遮掩，prefill/decode 都受益；dsv4_dsa_overlap 在 prefill 段也有 CV 并行收益（不依赖 prefill_comm）。

### 场景 D：mix_placement 部署
**亲和开关**：`multistream_overlap_shared_expert` **必须关**（互斥2 raise）。只能开：
```json
{"multistream_dsv4_dsa_overlap": true, "multistream_dsa_preprocess": true}
```

### 场景亲和特征总表

| 场景 | 主瓶颈 | dsv4_dsa | dsa_preprocess | shared_expert | gate | prefill_comm |
|------|--------|----------|----------------|---------------|------|--------------|
| A Decode+大EP | MoE通信 | ✅ | ✅ | ✅ | ❌(互斥3) | ❌(decode无效) |
| B Prefill长上下文 | 投影+allgather | ❌(互斥1,二选一) | ✅ | ✅ | ❌ | ✅ |
| C PD混合均衡 | 混合 | ✅ | ✅ | ✅ | ❌ | ❌(互斥1) |
| D mix_placement | — | ✅ | ✅ | ❌(互斥2 raise) | ❌ | 视情况 |

> **通用推荐（无 mix_placement，最稳）= 场景 C**：`dsv4_dsa_overlap + dsa_preprocess + overlap_shared_expert` 三开，覆盖面最广、互斥最少。gate 与 prefill_comm 按 prefill/decode 占比专项调，且与上面互斥项二选一。

---

## 3. 算子级遮掩逻辑（每个开关遮掩了哪些算子）

### 3.1 `multistream_dsv4_dsa_overlap` — MLA prolog 的 CV(Cube/Vector) 3-block 并行
**dsa_v1.py:1691-1745**（`_mla_prolog_multistream`，注释直接给出分块）：
```
Block 分区（V=Vector, C=Cube, AIV=AI Vector）：
  Part1: q_quant[V] → q_a_down[C]   ∥  kv_quant[V]      (aux流)
  Part2: q_norm[V] + q_b_quant[V]   ∥  kv_matmul[C]      (aux流)
  Part3: q_b_matmul[C]              ∥  kv_norm[V]+rope[V]+scatter[AIV] (aux流)
  Tail:  q_rms[V]+rope[V] (wait aux 完成)
```
> 遮掩本质：**主流跑 Q 路径（Cube matmul 为主），aux 流跑 KV 路径（Vector quantize/norm/rope 为主）**——Q 的 Cube 算力与 KV 的 Vector 算力**在两条流并行**，互不等待（块内数据自包含，仅 Tail wait_stream 同步 scatter）。

### 3.2 `multistream_dsa_preprocess` — DSA 预处理 ∥ 注意力计算
**dsa_v1.py:2287**：`with npu_stream_switch(attention_calculation_stream(), ...)` 内跑 `npu_quant_matmul`（kv 量化矩阵乘）→ 与主流的注意力/indexer 计算重叠。

### 3.3 `multistream_overlap_shared_expert` — 共享专家计算 ∥ 路由专家通信
**fused_moe.py:748/835**：
```
主流：路由专家 dispatch(all-to-all) → expert matmul → combine(all-to-all)
aux流(shared_experts_calculation_stream)：共享专家 gate_up → act → down
```
> 遮掩本质：**共享专家的 matmul（aux 流）遮掩路由专家的 all-to-all 通信（主流）**——这是大规模 EP 下最大收益项（通信占比高）。收尾 `wait_stream`(fused_moe.py:836) 同步。

### 3.4 `multistream_overlap_gate` — 共享专家 ∥ gate/router
**fused_moe.py:624-626**：gate_stream 上跑 `shared_experts(hidden_states)`，与主流 gate/router_logits 计算重叠。⚠️ 与 3.3 抢共享专家（互斥3）。

### 3.5 `prefill_comm_compute_overlap` — prefill 局部计算 ∥ allgather
**dsa_v1.py:1595-1606**：pure-prefill 时 wq_a/wkv 先在 SP 局部分片算，再 allgather 较小的中间结果——**用局部计算遮掩 allgather 通信延迟**。

---

## 4. 遗留问题（要求1）

1. **互斥3（gate vs shared_expert）无 raise，语义冲突** — 两者都调度共享专家到第二流，同开行为未知（哪个生效/是否冲突/是否双跑），**须实测**。❓
2. **各开关真实遮掩率** — 第二流空闲算力 × 被遮掩算子占比，无昇腾实测无法定量。❓
3. **910B3 上 aux-stream × ACL graph 兼容** — dsv4_dsa_overlap 有 aux-stream warmup(dsa_v1.py:1490) 保 graph 捕获，但覆盖率需验证。❓
4. **场景 B 的二选一（dsv4_dsa vs prefill_comm）** — 哪个在 prefill 段收益大，依上下文长度/SP 规模，须实测。❓

---

## 5. 一句话 recipe 结论

- **通用最稳（无 mix_placement）**：`multistream_dsv4_dsa_overlap + multistream_dsa_preprocess + multistream_overlap_shared_expert` 三开。
- **gate 与 shared_expert 二选一**（互斥3，抢共享专家）；
- **dsv4_dsa_overlap 与 prefill_comm 二选一**（互斥1，multistream 优先）；
- **mix_placement 下 shared_expert 必关**（互斥2 raise）。
- 可视化图例见同名 widget / 下节。
