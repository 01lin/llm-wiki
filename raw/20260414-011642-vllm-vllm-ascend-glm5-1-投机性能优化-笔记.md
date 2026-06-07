Now I have comprehensive information. Let me compile the analysis.

---

## GLM5.1 MTP 投机性能优化方案（昇腾 910B3 / vllm-ascend）

> 当前基线：MTP3 接受率 ~50-60%，已有开源 MTP 训练权重

---

### 优化思路框架

投机解码端到端性能 = **f(接受率, 单次投机 token 数, draft 开销, 验证开销, 系统利用率)**

50-60% 的接受率意味着 MTP3 平均每轮验证只接受 1.5~1.8 个 token，距离 DeepSeek 作者报告的单层 85-90% 还有显著差距。优化路径分两大方向：**提升接受率**（算法层）和**降低投机开销 / 提升系统效率**（系统层）。

---

### 一、提升接受率（算法层 — 高收益）

| 方案 | 核心思路 | 预期效果 | 实现复杂度 | 社区/学术依据 |
|------|----------|----------|-----------|--------------|
| **1. MTP 训练-推理对齐 (FastMTP)** | 当前 MTP head 训练时用 ground-truth 特征，推理时用自己的预测特征，存在 train-test gap。通过在训练时让 MTP head 递归使用自身输出（self-distillation），对齐推理时的分布 | 接受率可从 50-60% 提升至 70-80%+，论文报告 vanilla MTP → FastMTP **提速 82%** | 中 — 需重新训练 MTP head | [FastMTP (arXiv:2509.18362)](https://arxiv.org/abs/2509.18362)：单 MTP head + 位置共享权重 + 自蒸馏数据，2.03x 加速 |
| **2. 多层 MTP 联合推理** | GLM5.1 如有多个 MTP layer（>1），当前 EagleProposer 仅用第1层，浪费了后续层的预测能力。每层独立预测不同 position 的 token | 每个额外 MTP 层理论上增加一个高接受率位置 | 中 — 需修改 proposer 逻辑 | vllm [#31204](https://github.com/vllm-project/vllm/issues/31204)：Multi-MTP layer 支持 RFC；DeepSeek V3 原始论文中 MTP 设计 |
| **3. L-MTP 跳跃预测** | 不连续预测 token 1/2/3，而是预测 token 1/3/5（跳跃位置），利用概率衰减特性在跳跃位置反而有更高接受率 | 同等 head 数 **+22% 推理加速**，更高的远距 token 接受率 | 低-中 — 修改 MTP head 推理逻辑+重训练 | [L-MTP (2025)](https://arxiv.org/abs/2505.15003)：概率衰减分析证明跳跃预测优于连续预测 |
| **4. EARS 熵自适应拒绝采样** | 当 target 模型不确定时（高熵），动态放松拒绝阈值，避免不必要地拒绝合理的 draft token | GLM-4.7 MTP3 实测 **+14.6% 吞吐** | 低 — 几十行代码改动 | vllm-ascend [PR #7845](https://github.com/vllm-project/vllm-ascend/pull/7845)：已有完整实现，仅因 PR 规范未合入 |
| **5. Dynamic Speculation Length (DSL)** | 根据 draft 置信度动态调整投机长度：高置信度多投机，低置信度提前退出 | vllm 实测最高 **3.6x** vs target | 中 — 需修改 proposer 退出逻辑 | vllm [#36657](https://github.com/vllm-project/vllm/issues/36657) + [PR #35301](https://github.com/vllm-project/vllm/pull/35301)：`draft_confidence_threshold` 参数 |

**优先级建议**：

```
[最高] EARS (低成本高回报, 已有代码)
  ↓
[高] FastMTP 训练对齐 (根本性提升接受率)
  ↓
[高] DSL 动态投机长度 (与 EARS 正交, 可叠加)
  ↓
[中] 多层 MTP (取决于 GLM5.1 模型结构)
  ↓
[探索] L-MTP 跳跃预测 (需重训练)
```

**分析逻辑**：50-60% 的接受率说明 MTP head 在第 2、3 步预测时质量衰减严重（典型的 train-test distribution shift）。FastMTP 通过训练时模拟推理行为直接解决这个根因。EARS 和 DSL 则在不改变 MTP head 质量的前提下，从验证策略和投机策略两端分别减少浪费。三者可以叠加。

---

### 二、降低投机开销（系统层 — 中高收益）

| 方案 | 核心思路 | 预期效果 | 实现复杂度 | 社区/学术依据 |
|------|----------|----------|-----------|--------------|
| **6. Multi-Step Graph (多步图合并)** | 将多轮 MTP draft 推理合并为单张 ACLGraph，消除步间 host-device 同步 | 消除 host-bound 瓶颈，spec token 多时效果显著 | 已有基础框架 | vllm-ascend [#6077](https://github.com/vllm-project/vllm-ascend/issues/6077)（已 close/落地） |
| **7. Zero-Bubble Async Speculative** | 乐观假设上轮全部接受，CPU 不等 GPU 同步即开始准备下轮输入，在 GPU 端修正 | TPOT 降低 ~2-3% | 高 — 需改调度逻辑 | vllm [PR #29957](https://github.com/vllm-project/vllm/pull/29957)：H20 上 Qwen3-235B 验证 |
| **8. npugraph_ex 算子融合** | 在 fx.graph 上做 NPU 亲和的算子融合（norm_quant fusion 等），减少 kernel launch | decode 路径 kernel 开销降低 | 低 — 已支持，需确认 GLM5.1 启用 | vllm-ascend [#4715](https://github.com/vllm-project/vllm-ascend/issues/4715)；GLM5 的 `fuse_muls_add` 已修复 [#6928](https://github.com/vllm-project/vllm-ascend/pull/6928) |
| **9. Triton 算子优化** | rejection_sample、prepare_inputs、prepare_eagle_decode 等关键路径的 Triton 算子 NPU 优化 | 减少投机解码关键路径延迟 | 中 | vllm-ascend [#5208](https://github.com/vllm-project/vllm-ascend/issues/5208) MRV2 tracker 中列出的 Triton 算子优化清单；[PR #8011](https://github.com/vllm-project/vllm-ascend/pull/8011) |
| **10. 动态词表压缩** | 对 MTP head 的 lm_head 做语言感知的动态词表裁剪（仅保留高频 token），减少 draft 计算量 | draft 延迟降低 30-50%，接受率影响可忽略 | 中 | [FastMTP](https://arxiv.org/abs/2509.18362) 中的 language-aware dynamic vocabulary compression |

---

### 三、高级优化（长期/探索）

| 方案 | 核心思路 | 预期效果 | 社区/学术依据 |
|------|----------|----------|--------------|
| **11. Tree-Based MTP** | MTP 不走线性链，而是树状展开多条候选路径，一次验证多条路径 | 平均接受长度可提升到 3-5+（当前线性链约 1.5-1.8） | [Sequoia](https://arxiv.org/abs/2402.12374)：最优树结构 DP 求解，9.5x 加速；[OPT-Tree (TACL 2025)](https://arxiv.org/abs/2408.04628)：自适应树，mean acceptance length=10 |
| **12. EAGLE-3 风格训练** | 用 tri-layer feature fusion + training-time test 策略训练 MTP head，解决 feature prediction 局限 | 3-6.5x 加速，batch 吞吐 +38% | [EAGLE-3 (NeurIPS 2025)](https://arxiv.org/abs/2503.01840)：多层特征融合 + 训练时模拟推理时行为 |
| **13. MineDraft 批间并行** | Batch A 验证 / Batch B 起草交替执行，隐藏 draft 延迟 | 论文：吞吐 +75%, 延迟 -39% | vllm [#38003](https://github.com/vllm-project/vllm/issues/38003)；需等上游设计 |
| **14. QuantSpec (量化KV投机)** | draft 模型使用 4-bit 量化 KV cache + 4-bit 权重，共享 target 架构 | 接受率 >90%, 2.5x 加速 | [QuantSpec (2025)](https://arxiv.org/abs/2502.20579) |
| **15. Speculative Speculative (SSD)** | 两级投机：快速低质量 speculator 做第一层 draft，慢速高质量 speculator 做 fallback | 在 cache miss 场景下优化 | vllm [#36037](https://github.com/vllm-project/vllm/issues/36037) |

---

### 四、推荐实施路径（短/中/长期）

```
┌─────────────────────────────────────────────────────────────────┐
│  短期 (1-2 周，快速见效)                                          │
│                                                                   │
│  [A] EARS 熵自适应拒绝采样                                        │
│      - 改 rejection_sampler.py ~30 行                             │
│      - 环境变量控制，向后兼容                                      │
│      - 预期：吞吐 +10-15%                                        │
│                                                                   │
│  [B] 确认 npugraph_ex + fuse_muls_add 对 GLM5.1 已正确启用       │
│      - 检查 routed_scaling_factor 值是否匹配                      │
│      - 确认 FULL_DECODE_ONLY 图模式正常工作                       │
│                                                                   │
│  [C] DSL 动态投机长度 (cherry-pick vllm #35301)                  │
│      - 设 num_speculative_tokens=5~8, threshold=0.5              │
│      - 高置信度多投机, 低置信度提前退出                            │
│      - 与 EARS 叠加使用                                           │
├─────────────────────────────────────────────────────────────────┤
│  中期 (1-2 月，根本性提升)                                        │
│                                                                   │
│  [D] FastMTP 训练-推理对齐重训练 MTP head                         │
│      - 用 GLM5.1 自蒸馏数据训练 position-shared MTP head          │
│      - 核心收益：从根源提升第 2、3 步接受率                        │
│      - 预期：接受率从 50-60% → 75-85%                             │
│                                                                   │
│  [E] 动态词表压缩                                                 │
│      - 构建中文高频 token 子集                                     │
│      - MTP head 推理时只在子集上 softmax                          │
│      - 降低 draft 计算量, 对接受率影响 <1%                        │
│                                                                   │
│  [F] Multi-Step Graph 验证/优化                                   │
│      - 确认 GLM5.1 MLA 路径是否兼容                               │
│      - 多步合并消除 D/H 同步                                      │
├─────────────────────────────────────────────────────────────────┤
│  长期 (探索性)                                                    │
│                                                                   │
│  [G] Tree-Based MTP (OPT-Tree/Sequoia 风格)                     │
│  [H] EAGLE-3 tri-layer fusion 训练策略                           │
│  [I] Zero-Bubble Async (跟进上游 vllm #29957)                   │
└─────────────────────────────────────────────────────────────────┘
```

---

### 五、量化收益预估

| 方案组合 | 接受率预估 | 平均接受长度 (MTP3) | 相对当前提速 |
|----------|-----------|-------------------|-------------|
| 当前基线 | 50-60% | ~1.5-1.8 | 1.0x |
| +EARS +DSL (短期) | 55-65% | ~1.8-2.2 | **1.15-1.25x** |
| +FastMTP 重训练 (中期) | 75-85% | ~2.5-3.0 | **1.5-1.8x** |
| +FastMTP +Tree MTP (长期) | 80-90% | ~3.5-5.0 | **2.0-2.5x** |

> 注：DeepSeek 作者报告单层 MTP 接受率 85-90%，SGLang 实测 DeepSeek V3 平均接受长度 2.4（MTP1）。GLM5.1 MTP3 达到类似水平是可行的，关键瓶颈在训练-推理分布不对齐。

---

### 六、与社区工作的冲突检查

| 你的优化方向      | 潜在冲突                                          | 规避建议                                  |
| ----------- | --------------------------------------------- | ------------------------------------- |
| EARS        | PR #7845 已关闭但代码完整                             | 直接复用其实现，按社区 PR 规范重新提交                 |
| DSL         | vllm #35301 正在 review                         | Cherry-pick 到 ascend，注意 proposer 接口兼容 |
| FastMTP 重训练 | speculators Q2 规划 MTP 训练支持 (#377)             | 可先独立训练，后续对齐 speculators 格式            |
| 多层 MTP      | vllm #31204 未开始实现                             | 可抢先在 ascend 侧实现，后续贡献上游                |
| Tree MTP    | 上游无 MTP tree 实现                               | 探索性工作，不会冲突                            |
| Triton 算子   | #5208 中 rejection_sample 已有 owner (@lhp-deep) | 需先确认 assignee 再开发                     |