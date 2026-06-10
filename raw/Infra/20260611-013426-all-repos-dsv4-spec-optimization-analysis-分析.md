# DSv4 Pro/Flash 投机推理优化点分析：vllm-ascend 对标业界

> 范围：vllm-ascend（主目标）、vllm 上游、sglang、tokenspeed 四大引擎 DSv4 投机推理
> 方法：从源码逐步剖析关键差异化能力，提出 vllm-ascend 优化点、可行性、预估收益与方案设计
> 代码都更新到最新 HEAD
> 生成日期：2026-06-11

---

## 0. 现状基线（vllm-ascend DSv4-Flash MTP）

再次确认当前状态作为优化起点：

| 维度 | 当前状态 | 源码证据 |
|------|----------|----------|
| 草稿方法 | MTP（`AscendEagleProposer`） | `spec_decode/__init__.py:42` |
| 草稿 token 数 | 默认 k=1（max_spec_len=1） | 教程 `DeepSeek-V4-Flash.md` |
| 采样模式 | argmax only（`all_greedy=True`） | `rejection_sampler.py:404-443` 提前 return |
| 草稿 prob | 不保留（`logits.argmax(dim=-1)`） | `llm_base_proposer.py:1069` |
| 图捕获 | 主模型 FULL_DECODE_ONLY，**草稿 enforce_eager=true** | 官方教程 + `use_cuda_graph` 开关 |
| Block/Entropy Verify | 不可用（max_spec_len<3 + all_greedy） | 上篇分析 |
| DSA MTP index share | **未实现**（DSA 层有 `skip_topk` 入参但 proposer 不用） | `dsa_v1.py:1468` 有参数无人设 |
| Tree drafting / topk>1 | **未实现**（TODO 注记） | `llm_base_proposer.py:1224` |
| DFlash for DSv4 | 未接入（仅 Qwen3） | `dflash_proposer.py` 仅用 `DFlashQwen3ForCausalLM` |
| PD 分离 | 已支持（MooncakeHybridConnector） | 1P1D 官方教程 |
| DSA context parallel | 已支持（`enable_dsa_cp`） | config 示例 + #9531 |

---

## 1. 优化向量 I：DSA Index Share for MTP 迭代（对标 vllm 上游）

### 1.1 问题：DSv4 稀疏注意力的 topk 索引在每步 MTP 重复计算

DSv4 的 DSA（稀疏注意力）每层需要做 indexer 选 topk token。MTP 多步草稿时，每步都需要重新跑 indexer 为 KV 压缩 pick indices——但步骤 1..k 的 index 与 step 0 基本相同（因 token 窗口滑动很小）。**vllm 上游 #44420 已对此做优化。**

### 1.2 vllm 上游方案：index_share_for_mtp_iteration

```python
# vllm/model_executor/models/deepseek_mtp.py: set_skip_topk()
# Step 0: skip_topk=False, 正常算 indexer
# Step 1+: skip_topk=True, 复用 step 0 的 index_buffer

# vllm/v1/spec_decode/llm_base_proposer.py:
# Step 0: model.set_skip_topk(False)
# After step 0: model.set_skip_topk(True)  ← steps 1..k skip indexer entirely
```

### 1.3 vllm-ascend 缺口

DSA 注意力层 `dsa_v1.py` **已经有** `self.skip_topk` 和 `IndexCache` 机制——这是为 multistream overlap 设计的。但 MTP proposer **完全没调用 `set_skip_topk`**（grep 结果：proposer dir 下 0 match）。

### 1.4 可行性评估

| 评估项 | 结论 |
|--------|------|
| 基础设施就绪 | ✅ DSA 层已支持 `skip_topk` + `IndexCache` + `topk_indices_buffer` |
| 仅需 proposer 层改造 | ✅ 在 `llm_base_proposer.py` 的多步循环中仿造 vllm 上游加 `set_skip_topk` toggle |
| Ascend 特有风险 | ⚠️ topk_indices_buffer 在多步间的形状/内存布局需验证（PCP/DCP 下 index 是否跨步一致） |
| 实现量 | 小（~50 行 proposer + 模型 adapter），主要是 index_buffer 生命周期管理 |

### 1.5 预估收益

- 每步跳过 indexer 的 GEMM + topk 约省 **0.3-0.5ms**（DSA indexer 是 attn block 内的重操作）
- k=3 时 net 省 ~1ms/decode step，**decode throughput +5%~8%**
- 注意：reduction 不是 3×，因为 step 0 仍需算 indexer

---

## 2. 优化向量 II：MTP 前缀缓存复用（对标 tokenspeed）

### 2.1 问题：MTP 草稿每步都重新 prefill 一段小前缀

tokenspeed #361 实现了 **MTP 草稿的 KV prefix cache 复用**：草稿 forward 的 `captured_hidden_states` 和 out_cache_loc 从 prefix cache 加载而非每次重算。在 vllm-ascend 中，MTP 每步草稿都从头算 hidden_states。

### 2.2 tokenspeed 方案

```python
# deepseek_v4_mtp.py:322 forward()
def forward(self, ctx, input_ids, positions, out_cache_loc,
            captured_hidden_states=None, spec_step_idx=0, ...):
    # captured_hidden_states 从 scheduler prefix cache 传入
    # 匹配则跳过 embed/attn 前几层
```

核心：scheduler 层（C++）track 草稿 token 的 radix match，命中则传 `captured_hidden_states` 给 draft model forward，skip 前若干层的重算。

### 2.3 可行性

- vllm-ascend 已有 `kv_prefix_cache` 但**未传 capture hidden states 给 MTP proposer**。
- 改造点：(1) scheduler 记录草稿 token 的 radix hash 并传捕获的 hidden states；(2) `llm_base_proposer.py` 在 `_propose` 前查表和注入。
- 复杂度：**中**（需 scheduler → proposer 跨层接口），但 tokenspeed 的实现已验证可行性。

### 2.4 预估收益

- 草稿 prefill 省 60%-80% 的前几层 attention 计算
- 多轮对话/长前缀场景（multi-QA、agent）收益最大，**spec step 延迟 -30%40%**

---

## 3. 优化向量 III：Tree Drafting（对标 sglang）

### 3.1 问题：当前 MTP k>1 是序列化的（每次取 argmax 链式推），而非分支探索

sglang Spec V2 做了 **tree drafting**：每步草稿不只出一个 top-1 token，而是 topk 个分支组成验证树，一次 verify 同时验整棵树，大幅提升接受率。

vllm-ascend 代码中自己也有注释：`# TODO(wenlong): get more than one token for tree attention`（line 1224）。

### 3.2 sglang 方案

- `eagle_info_v2.py: duplicate_prefix_tail_to_draft_branches()` — 复制前缀 page 尾部到各分支首 page
- `verify_tree_greedy_func` — 同时验证整棵树，根到叶 longest chain 为 accept
- 树深度 = `spec_steps + 1`，分支数 = topk

### 3.3 可行性

- vllm-ascend reject sampler 当前不支持树验证（只支持线性 token-by-token）。
- **先导条件**：DSA index_share（向量 I）+ k>=3（多步草稿）落地后，tree 才有意义。
- 复杂度：**高**（需改 rejection sampler kernel 为树验证、proposer 批量 draft extend、page table 分支布局）。
- **优先级**：中长期。tree 可以在线性 MTP 有成熟的 index_share + >1k 接受率底子后再做。

### 3.4 预估收益

- sglang 文档/论文级别的 tree drafting 可提升接受率 **30%-50%**（topk=2, depth=3），equivalent 加速 ~1.3-1.5×。

---

## 4. 优化向量 IV：草稿 ACLGraph 捕获（去 enforce_eager）

### 4.1 问题

当前 MTP 草稿 `enforce_eager=true`，每步 launch overhead 和 Python dispatch 成本高，尤其在 k>1 多步时累积明显。

### 4.2 可行性

- 代码中已预留 `self.use_cuda_graph = self.runner._use_aclgraph() and not self.speculative_config.enforce_eager`
- ACLGraphWrapper 已实现并用于 dflash 的 merged graph（#9074）
- **但目前 MTP k>1 + ACLGraph 的 input buffer 动态形状处理可能不稳定**，需要验证

### 4.3 预估收益

- k>1 多步时，每步省 kernel launch + Python dispatch ~**0.2-0.3ms**
- 保守估计 decode step 延迟 -3%~5%

---

## 5. 优化向量 V：DP Sampling for Spec Decode（对标 tokenspeed）

### 5.1 问题

tokenspeed #232 支持 spec decode 的 DP（data parallel）采样——大 DP 下每 rank 的 sampler 独立采样不需要全局 all-reduce，通信量从 O(vocab_size) 降到 O(tp_size)。

vllm-ascend 当前未做此优化。

### 5.2 可行性

- vllm 上游有类似方向（#39419：大 vocab draft 模型减 TP 通信）
- 对 DSv4 尤其重要（vocab_size 129k+，DP 多组时通信瓶颈）

### 5.3 预估收益

- DP=16 时 TP 通信节省显著，**大 DP 场景 decode 延迟 -10%~15%**

---

## 6. 优化向量 VI：Block/Entropy Verify 启用（已有能力但未用）

### 6.1 问题

上篇已详细分析：当前 MTP 在 argmax+greedy+k=1 下完全不能触发 Block Verify / Entropy Verify。

### 6.2 解决方案

1. k 改 >=3 → 解锁 `max_spec_len >= 3` 前置
2. 采样从纯 greedy 改为 mixed（部分请求 temperature>0）
3. 开 `enable_reduce_sample: true` → 出 target_probs 和 draft_probs
4. 开 `enable_block_verify: true` + `enable_entropy_verify: true`

### 6.3 预估收益

- Block Verify 接受率可比标准拒绝采样提升 **8%~15%**（取决于 distribution entropy）
- Entropy Verify 对高熵 token 放宽接受可再提升 **3%~5%**
- 代价：reduce_sample 的 comm+compute 增加（需实测算 trade-off）

---

## 7. 综合对比表

| 优化点 | vllm-ascend | vllm 上游 | sglang | tokenspeed | 优先级 | 实现量 | 预估收益 |
|--------|:-----:|:-----:|:-----:|:-----:|:----:|:--:|------|
| DSA index_share for MTP | ❌ | ✅ #44420 | —(不同架构) | — | **P0** | 小 | +5~8% decode |
| MTP prefix cache | ❌ | — | ✅(HiCache) | ✅ #361 | **P0** | 中 | -30% draft prefill |
| Block/Entropy Verify | 代码有/config无 | ❌ | ❌ | ❌ | **P1** | 小 | +8~15% 接受率 |
| Draft ACLGraph | ❌(enforce_eager) | ✅(full CUDA graph) | ✅(piecewise) | ✅ | **P1** | 小 | -3~5% step delay |
| Tree drafting | ❌(TODO) | ❌ | ✅(Spec V2) | ❌ | P2 | 高 | +30~50% 接受率 |
| DP sampling for spec | ❌ | ✅(非直接) | ✅ | ✅ #232 | P2 | 中 | -10~15% DP decode |
| k>1 多步 MTP | 不支持(cap 1) | ✅(支持) | ✅ | ✅ | **P0** | 中 | ~1.5-2× decode |
| DFlash for DSv4 | ❌(仅 Qwen3) | ✅ #44586 | ✅ | ❌ | P2 | 高 | +40~60% 接受率(vs linear MTP) |

---

## 8. 初步方案设计：三阶段推进

### 阶段 1（短期，立即可做）

**目标**：MTP k=3 + reduce_sample + Block/Entropy Verify 上线

| 步骤 | 工作量 | 关键文件 |
|------|--------|----------|
| 1a. 配置 k=3 + `enable_reduce_sample=true` | 0 行代码 | server 启动参数 |
| 1b. 修 target_probs 数据在 all_greedy 下的产出 | ~30 行 | `llm_base_proposer.py` line 1064-1069 |
| 1c. 打开 `enable_block_verify + enable_entropy_verify` | 0 行配置 | 条件：resolution 1b |
| 1d. 多步草稿循环稳定性验证（k=3, PCP/DCP 下） | 测试 | `llm_base_proposer.py` loop |
| 1e. 接收率 + TPOT benchmark @ 910B/910C | 评测 | AISBench / vllm benchmark |

### 阶段 2（中期，1-2 周）

**目标**：DSA index_share + MTP prefix cache

| 步骤 | 工作量 | 关键文件 |
|------|--------|----------|
| 2a. DSA index_share: proposer toggle `set_skip_topk` | ~50 行 | `llm_base_proposer.py` + MTP model adapter |
| 2b. topk_indices_buffer shape/cp 验证 | 测试 | `dsa_v1.py` multi-step 兼容 |
| 2c. MTP prefix cache: scheduler → proposer 传 captured_hidden_states | ~150 行 | scheduler + `llm_base_proposer.py` |
| 2d. 组合验收：index_share + prefix cache + Block/Entropy Verify 联调 | 集成测试 | 全链路 |

### 阶段 3（中长期，roadmap）

**目标**：Tree drafting + DFlash for DSv4

| 步骤 | 工作量 | 关键点 |
|------|--------|--------|
| 3a. Tree drafting: rejection sampler → tree verify kernel | ~500 行+ | 复杂，需参考 sglang 架构 |
| 3b. DFlash for DSv4: DFlashQwen3 → generalize 到 DSv4 | ~300 行 | 需 DSv4 cross-attention 适配 |
| 3c. DP sampling for spec | ~200 行 | 大 DP 场景专项 |

---

## 9. 关键发现与建议

1. **DSA index_share 是最容易落地的 P0 优化**：基础设施（skip_topk / IndexCache）已在 dsa_v1.py 中，只是 MTP proposer 没用——proposer 层小改即可，收益明确。
2. **k=3 是解锁其他所有优化的关键闸门**：Block/Entropy Verify、index_share、tree drafting 全依赖 max_spec_len>=3。
3. **vllm 上游和 tokenspeed 各自有 vllm-ascend 可复用的优化**，侧重点不同：vllm 上游 index_share + DFlash，tokenspeed prefix cache + DP sampling。
4. **当前 vllm-ascend 的 DSv4 MTP 处于功能完善但优化未推尽的状态**——核心链路可用，但每个组件都在自己的性能洼地：草稿无图捕获、index 重算、前缀无复用、k 锁死 1。组合跨越这些优化点后，**预估总 decode latency 下降 20%-35%**（最保守估计）。
