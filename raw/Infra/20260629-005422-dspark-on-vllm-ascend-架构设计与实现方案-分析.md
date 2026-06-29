# DSpark 投机方案 · vllm-ascend 落地架构设计与实现方案

> 目标：在 vllm/vllm-ascend（`model_runner_v1`）推理引擎上实现 DSpark 投机解码（先做**静态阈值截断**版，对齐 DeepSpec 开源实现）。
> 一手依据：vllm-ascend / ops-transformer / DeepSpec 三仓本地源码，行号 grep 实测、Explore 探针交叉验证。
> 前置：DSpark 算法见 [[20260628-201837-dspark-draft-head-structure-图文详解-分析]]（结构）与 [[20260628-191517-dspark-draft-confidence-e2e-deepdive-分析]]（e2e）。
> 本地源码链接为 `vscode://file/<绝对路径>:<行号>`，点击直接在 VSCode 打开对应文件并跳到该行。
> **首次点击 Obsidian 会弹"打开外部应用"确认框，点允许即跳；若完全无反应见文末「附录：让 vscode 链接在 Obsidian 点得动」。** （[[output-docs-need-clickable-code-index]]）

---

## 0. 一句话顶层设计

> **DSpark 落地 = 继承 vllm-ascend 现成的 `AscendDflashProposer`（白嫁 DFlash 并行骨干 + 交叉注意力 + Triton 输入 kernel），只新增三件套：Markov 串行采样头、置信度头、静态阈值截断。**

底层逻辑：DFlash 是 DSpark 的并行骨干前身，vllm-ascend **已有完整 DFlash proposer + patch + 模型**。DSpark 相对 DFlash 的增量正好是那三个挂件——所以**不从零写 proposer**，而是站在 DFlash 肩上做增量。算子层几乎零新增（embedding/linear/采样/置信度全是框架层），唯一可选缺口是 block-diagonal mask。

**工作量基线**：proposer + draft model + patch（3-4 人日）；可选 block-mask 算子适配（1-2 人日）；可选 Markov Triton 融合（2-3 人日）。

---

## 1. 现状地基（实测，DSpark 站在谁的肩上）

### 1.1 proposer 继承链（已坐实）

```
vllm.v1.spec_decode … EagleProposer
        │
AscendSpecDecodeBaseProposer        ← Ascend 投机基类（_propose/_run_merged_draft/load_model/dummy_run）
        │
AscendEagleProposer(EagleProposer, AscendSpecDecodeBaseProposer)   ← pass_hidden_states=True
        │
AscendDflashProposer(AscendEagleProposer)   ← DSpark 直接继承它
        │
AscendDsparkProposer(AscendDflashProposer)  ← 【新增】只加三件套
```

| 类 | 文件:行 | 角色 |
|----|--------|------|
| AscendSpecDecodeBaseProposer | [llm_base_proposer.py:130](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py:130) | 投机骨架：`_propose`/`_run_merged_draft`/`compute_draft_token_ids`/`load_model`/`dummy_run` |
| AscendEagleProposer | [eagle_proposer.py:10](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/eagle_proposer.py:10) | 开 hidden 传递 |
| AscendDflashProposer | [dflash_proposer.py:15](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py:15) | 并行骨干：`set_inputs_first_pass`(交叉注意力)/`build_model_inputs_first_pass`(预算 K/V)/`dummy_run` |

### 1.2 三个挂钩点（实测）

| 挂钩点 | 文件:行 | 现状 | DSpark 要做 |
|--------|--------|------|-------------|
| H1 分发表 | [spec_decode/__init__.py:33](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/__init__.py:33) `get_spec_decode_method` | `elif method=="dflash": AscendDflashProposer` | 加 `elif method=="dspark": AscendDsparkProposer` |
| H2 调用点 | [model_runner_v1.py:1686](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:1686) `propose_draft_token_ids` | eagle/dflash 走 `drafter._propose(...)` 分支 | DSpark 同分支复用（isinstance 命中父类）或加专支 |
| H3 模型 patch | [patch_qwen3_dflash.py:62](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_dflash.py:62) `precompute_and_store_context_kv` | DFlash 模型预算上下文 K/V | DSpark 仿造 `patch_qwen3_dspark.py` |

### 1.3 hidden states 传递（实测）

target 多层 hidden 经 `aux_hidden_states`（[model_runner_v1.py:2318](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:2318)）在 dim=-1 拼接后传给 proposer（[model_runner_v1.py:1880](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:1880) 附近 `torch.cat([h[token_indices] for h in aux_hidden_states], dim=-1)`）。DSpark 的 `target_layer_ids=[1,9,17,25,33]` 对应这里多层拼接——配置侧需让 target 输出这 5 层 aux hidden。

### 1.4 验证（无需改）

rejection sampling 在 [rejection_sampler.py](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/sample/rejection_sampler.py)，proposer 只产出 draft tokens + draft probs，验证由现成 sampler 完成。**DSpark 静态阈值截断在 proposer 内部完成（截断后少报 num_speculative_tokens），对验证透明。**

---

## 2. 类设计（class diagram）

下图为内联 SVG（纯内联属性，Obsidian 直接渲染）。

<svg width="100%" viewBox="0 0 680 540" xmlns="http://www.w3.org/2000/svg" role="img">
<title>DSpark proposer 类继承设计</title>
<desc>AscendDsparkProposer 继承 DFlash proposer，新增 Markov 头、置信度头、阈值截断三件套</desc>
<defs><marker id="cy" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#888"/></marker></defs>

<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="40" y="28" font-weight="500">类继承设计：DSpark 站在 DFlash 肩上（vllm-ascend）</text>

<rect fill="#f0f0ee" stroke="#b8b8b2" stroke-width="1.5" x="200" y="44" width="280" height="50" rx="8"/>
<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="220" y="66" font-weight="500">AscendSpecDecodeBaseProposer</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="220" y="85">_propose / _run_merged_draft / load_model</text>

<line x1="340" y1="94" x2="340" y2="116" stroke="#888" stroke-width="1.5" marker-end="url(#cy)"/>

<rect fill="#dce9fb" stroke="#7aa7e0" stroke-width="1.5" x="200" y="118" width="280" height="48" rx="8"/>
<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="220" y="140" font-weight="500">AscendEagleProposer</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="220" y="158">pass_hidden_states = True</text>

<line x1="340" y1="166" x2="340" y2="188" stroke="#888" stroke-width="1.5" marker-end="url(#cy)"/>

<rect fill="#d3efe8" stroke="#6cc0ac" stroke-width="1.5" x="170" y="190" width="340" height="78" rx="8"/>
<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="190" y="212" font-weight="500">AscendDflashProposer  （并行骨干 · 已存在）</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="190" y="232">set_inputs_first_pass（交叉注意力输入 Triton kernel）</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="190" y="250">build_model_inputs_first_pass（预算上下文 K/V）</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="190" y="266">dummy_run（ACL 图捕获）</text>

<line x1="340" y1="268" x2="340" y2="290" stroke="#888" stroke-width="1.5" marker-end="url(#cy)"/>

<rect fill="#e7dcf5" stroke="#a886d4" stroke-width="1.5" x="120" y="292" width="440" height="110" rx="8"/>
<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="140" y="314" font-weight="500">AscendDsparkProposer  【新增 · 只加三件套】</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="140" y="334">+ confidence_threshold : float</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="140" y="352">override compute_draft_token_ids → Markov 串行采样</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="140" y="370">+ _apply_confidence_cutoff → 单点阈值截断</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="140" y="388">继承：_propose / set_inputs / build_inputs / dummy_run（不动）</text>

<rect fill="#fbeacb" stroke="#dcab4e" stroke-width="1.5" x="120" y="424" width="440" height="96" rx="8"/>
<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="140" y="446" font-weight="500">DsparkQwen3Model（draft model 侧 · 仿 qwen3_dflash）</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="140" y="466">骨干 5 层 + 交叉注意力（复用 DFlash）</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="140" y="484">+ markov_head：sample_draft_tokens（W1 256 → W2 词表）</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="140" y="502">+ confidence_head：predict_confidence_step（Linear 2816→1）</text>

<line x1="340" y1="402" x2="340" y2="424" stroke="#888" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#cy)"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="350" y="418">持有/调用</text>
</svg>

### 2.1 新增 `AscendDsparkProposer`（核心类）

继承 `AscendDflashProposer`，复用其全部并行骨干能力，只覆盖/新增：

| 成员 | 类型 | 职责 | 对应 DeepSpec |
|------|------|------|--------------|
| `markov_head` | 子模块(模型侧) | γ 次串行采样补 token 依赖 | [markov_head.py:55](vscode://file/Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/markov_head.py:55) |
| `confidence_head` | 子模块(模型侧) | Linear(2816→1) 估接受概率 | [common.py:43](vscode://file/Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/common.py:43) |
| `confidence_threshold` | float(config) | 静态阈值，截断点 | [draft_ops.py:82](vscode://file/Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/draft_ops.py:82) |
| `compute_draft_token_ids()` | override | 用 Markov 串行采样替换 DFlash 的逐步 argmax；产出 draft tokens + draft probs | [markov_head.py:76](vscode://file/Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/markov_head.py:76) |
| `_apply_confidence_cutoff()` | new | 算 c_k、单点阈值截断，返回前缀长度 | [draft_ops.py:90](vscode://file/Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/draft_ops.py:90) |
| `set_inputs_first_pass()` | inherit/微调 | 复用 DFlash 交叉注意力输入 | [dflash_proposer.py:63](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py:63) |
| `build_model_inputs_first_pass()` | inherit | 复用 DFlash 预算 K/V | [dflash_proposer.py:253](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py:253) |

### 2.2 模型侧 `DsparkQwen3Model`（仿 `qwen3_dflash`）

在 draft model 里加 `markov_head` / `confidence_head` 两个子模块 + `predict_confidence_step` 方法，骨干（5 层 + 交叉注意力 + `precompute_and_store_context_kv`）直接复用 DFlash 模型结构。

---

## 3. 模块设计（落地文件清单）

| 模块 | 新建/改动文件 | 内容 | 优先级 |
|------|--------------|------|--------|
| proposer | 新建 `vllm_ascend/spec_decode/dspark_proposer.py` | `AscendDsparkProposer` 三件套 | P0 |
| 分发 | 改 [spec_decode/__init__.py:33](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/__init__.py:33) | 加 `dspark` 分支 + import | P0 |
| draft model | 新建 `vllm/model_executor/models/qwen3_dspark.py` | `DsparkQwen3Model` + markov/confidence 头 | P0 |
| 模型 patch | 新建 `vllm_ascend/patch/worker/patch_qwen3_dspark.py` | `precompute_and_store_context_kv`（仿 DFlash） | P0 |
| runner 接入 | 改 [model_runner_v1.py:1686](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:1686) | DSpark proposer 分支（多数可被 isinstance 父类命中，最小改动） | P0 |
| config | 改 SpeculativeConfig patch | 识别 `method=="dspark"`、读 `confidence_threshold`/`markov_rank`/`block_size` | P0 |
| block mask（可选） | 接 ops-transformer `BlockSparseAttention` 或扩 `attention_mask.py` | 同 block 双向、跨 block 隔离 | P1 |
| Markov Triton（可选） | 新建 `ops/triton/spec_decode/markov_sample.py` | γ 步采样融合，减 host-device 同步 | P2 |

> 静态阈值版**可先不做 block mask**：DFlash 现状 `cad.attn_mask=None`（[dflash_proposer.py:147](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py:147)）即全密集交叉注意力，先复用跑通，再上 block-diagonal 精度优化。这是"先跑通再优化"的颗粒度。

---

## 4. 接口设计

### 4.1 proposer 核心接口（继承基类签名，实测）

```python
class AscendDsparkProposer(AscendDflashProposer):
    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        super().__init__(vllm_config, device, runner=runner)
        self.confidence_threshold = vllm_config.speculative_config.confidence_threshold
        # markov_head / confidence_head 在 load_model 后由 draft model 持有

    # 覆盖：把"逐步 argmax"换成"Markov 串行采样 + 置信度截断"
    def compute_draft_token_ids(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

    # 新增：单点阈值截断，返回 confident 前缀长度
    def _apply_confidence_cutoff(
        self, confidence_logits: torch.Tensor, block_size: int, threshold: float
    ) -> int: ...
```

`_propose()` / `_run_merged_draft()` / `load_model()` / `dummy_run()` **全部继承不动**（基类已处理 ACL 图、PCP/DCP、位置编码）。

### 4.2 模型侧接口（仿 DeepSpec，挂到 draft model）

```python
class DsparkQwen3Model(DFlashQwen3Model):   # 复用 DFlash 骨干
    def sample_draft_tokens(self, base_logits, first_prev_token_ids, temperature, hidden_states):
        # = DeepSpec markov_head.sample_block_tokens：γ 步串行
        ...
    def predict_confidence_step(self, hidden_states, prev_token_ids) -> torch.Tensor:
        # = DeepSpec predict_confidence_step：c_k logit
        ...
    # precompute_and_store_context_kv 由 patch 注入（仿 DFlash）
```

### 4.3 config 接口

| 字段 | 来源 | 默认 | 说明 |
|------|------|------|------|
| `method` | SpeculativeConfig | `"dspark"` | 触发分发 |
| `num_speculative_tokens` | SpeculativeConfig | 7 | = block_size γ |
| `confidence_threshold` | 扩展字段 | 0.0 | >0 才截断；离线诊断调 |
| `markov_rank` | draft model config | 256 | 低秩中转维 |

---

## 5. 执行逻辑时序图

<svg width="100%" viewBox="0 0 680 620" xmlns="http://www.w3.org/2000/svg" role="img">
<title>DSpark 一轮 step 执行时序</title>
<desc>model_runner 驱动 target forward、proposer 产 draft、rejection sampler 验证的时序</desc>
<defs><marker id="sq" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#666"/></marker></defs>

<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="40" y="26" font-weight="500">一轮 step 执行时序（vllm-ascend model_runner_v1）</text>

<rect fill="#dce9fb" stroke="#7aa7e0" stroke-width="1.5" x="40" y="42" width="130" height="38" rx="6"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="56" y="65">model_runner</text>
<rect fill="#d3efe8" stroke="#6cc0ac" stroke-width="1.5" x="200" y="42" width="130" height="38" rx="6"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="216" y="65">target model</text>
<rect fill="#e7dcf5" stroke="#a886d4" stroke-width="1.5" x="360" y="42" width="150" height="38" rx="6"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="376" y="65">DsparkProposer</text>
<rect fill="#fbeacb" stroke="#dcab4e" stroke-width="1.5" x="540" y="42" width="110" height="38" rx="6"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="556" y="65">rejection</text>

<line x1="105" y1="80" x2="105" y2="600" stroke="#ccc" stroke-width="1"/>
<line x1="265" y1="80" x2="265" y2="600" stroke="#ccc" stroke-width="1"/>
<line x1="435" y1="80" x2="435" y2="600" stroke="#ccc" stroke-width="1"/>
<line x1="595" y1="80" x2="595" y2="600" stroke="#ccc" stroke-width="1"/>

<line x1="105" y1="108" x2="263" y2="108" stroke="#666" stroke-width="1.5" marker-end="url(#sq)"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="112" y="102">1. forward（target）</text>

<line x1="265" y1="138" x2="107" y2="138" stroke="#666" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#sq)"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="112" y="132">2. logits + aux_hidden(5层)</text>

<rect fill="#f0f0ee" stroke="#b8b8b2" stroke-width="1.5" x="48" y="152" width="120" height="30" rx="4"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="56" y="172">3. 采样得 anchor</text>

<line x1="105" y1="200" x2="433" y2="200" stroke="#666" stroke-width="1.5" marker-end="url(#sq)"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="112" y="194">4. _propose(target_hidden, anchor)</text>

<rect fill="#ece3f7" stroke="#a886d4" stroke-width="1.5" x="345" y="214" width="180" height="138" rx="6"/>
<text font-family="sans-serif" font-size="12" fill="#1a1a1a" x="356" y="234" font-weight="500">proposer 内部</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="356" y="254">4a 骨干前向→U₁..U₇,h₁..h₇</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="356" y="274">（复用 DFlash 交叉注意力）</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="356" y="296">4b Markov 串行采样</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="356" y="314">→ x₁..x₇ + draft probs q</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="356" y="336">4c 算 c_k → 阈值截断 → ℓ</text>

<line x1="433" y1="378" x2="107" y2="378" stroke="#666" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#sq)"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="120" y="372">5. 前 ℓ 个 draft token + q</text>

<line x1="105" y1="408" x2="593" y2="408" stroke="#666" stroke-width="1.5" marker-end="url(#sq)"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="112" y="402">6. 验证（target logits vs q）</text>

<rect fill="#fdf2da" stroke="#dcab4e" stroke-width="1.5" x="536" y="422" width="120" height="74" rx="6"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="548" y="442">6a min(1,p/q)</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="548" y="462">6b cumprod 前缀</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="548" y="482">6c 接受+bonus</text>

<line x1="595" y1="520" x2="107" y2="520" stroke="#666" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#sq)"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="120" y="514">7. accepted tokens + next</text>

<rect fill="#f0f0ee" stroke="#b8b8b2" stroke-width="1.5" x="48" y="536" width="220" height="30" rx="4"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="56" y="556">8. 推进 start += accepted+1，下一轮</text>

<text font-family="sans-serif" font-size="12" fill="#2f7d43" x="40" y="592">DSpark 新增 = 仅 4b/4c；其余复用 DFlash + 现成 rejection sampler</text>
</svg>

### 时序文字说明（一轮 step）

1. **target forward**（[model_runner_v1.py:2313](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:2313)）→ 产出 logits + `aux_hidden_states`(5 层)。
2. **target 采样** → anchor（被接受的真实 token）。
3. **propose_draft_token_ids**（[model_runner_v1.py:1686](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:1686)）→ 调 `drafter._propose`，传 `target_hidden_states`(5层拼接) + anchor。
4. **proposer 内（DSpark 新增逻辑）**：
   - 继承 DFlash：`set_inputs_first_pass` 交叉注意力输入 → backbone 一次前向得 U_1..U_γ + h_1..h_γ。
   - 新增：`compute_draft_token_ids` 走 Markov 串行采样（γ 步 U_k+B_k）得 x_1..x_γ + draft probs q。
   - 新增：算 c_k → `_apply_confidence_cutoff` 单点阈值截断 → 前缀长度 ℓ。
   - 返回前 ℓ 个 draft token + 对应 q。
5. **rejection sampling**（[rejection_sampler.py](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/sample/rejection_sampler.py)）→ target 验证前缀，接受 + bonus，推进。

---

## 6. Ascend 算子缺口与对策（实测结论）

| DSpark 能力 | Ascend 现状 | 缺口 | 对策 |
|------------|------------|------|------|
| 并行骨干 + 交叉注意力 | DFlash 已有（全密集，`attn_mask=None`） | 无（静态版） | 复用 [dflash_proposer.py:63](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py:63) |
| embedding lookup | `AscendVocabParallelEmbedding` | 无 | 复用 |
| Markov logit 加法 | linear/sampler 框架层 | 无 | γ 步框架循环；可选 Triton 融合 |
| 置信度头 Linear+sigmoid | 框架层 | 无 | `torch.sigmoid(Linear(x))` |
| 阈值截断（单点） | 纯 torch | 无 | `cumprod`/比较纯框架 |
| block-diagonal mask | BlockSparseAttention（ops-transformer，需预设 mask）/ `attention_mask.py` 仅静态 causal | **唯一真缺口** | 静态版先用全密集；P1 接 [block_sparse_attention](vscode://file/Users/linyi/code/Documents/code/ops-transformer/attention/block_sparse_attention) |

> 结论：**静态阈值版几乎零新算子**，全部站在 DFlash + 现成采样/embedding/linear 框架层上。block-diagonal mask 是精度优化项，非跑通前提。

---

## 7. 实施路线（先跑通后优化）

```
阶段一（P0，跑通静态版，~4 人日）：
  1. qwen3_dspark.py draft model（仿 qwen3_dflash + markov/confidence 头）→ verify: 单测前向 shape
  2. patch_qwen3_dspark.py（precompute_and_store_context_kv）→ verify: K/V 预算不报错
  3. dspark_proposer.py（继承 DFlash + compute_draft_token_ids + 截断）→ verify: 产出 draft tokens
  4. __init__.py + config patch 接分发 → verify: method="dspark" 能起服务
  5. e2e 小模型 Qwen3-4B → verify: 接受率 >0、输出与 target 一致（无损）

阶段二（P1，精度优化，~2 人日）：
  6. block-diagonal mask 接 BlockSparseAttention → verify: 接受率提升、与 DeepSpec 对齐

阶段三（P2，性能优化，~3 人日）：
  7. Markov γ 步 Triton 融合 → verify: 单 step 延迟下降、无 host-device 同步
```

每步成功判据明确，可独立验证、loop 到通过（[[code-grounded-no-speculation]]）。

---

## 8. 风险与对齐项

| 风险 | 说明 | 缓解 |
|------|------|------|
| Markov 串行 γ 步在 ACL 图下 recompile | for 循环展开易触发多次编译 | 固定 γ、静态 shape；torchair 图内展开 |
| 双 KV cache 异节奏 | draft 即用即裁 vs target 按接受推进 | 复用 DFlash 已处理的 cache 逻辑 |
| 截断为 0 退化 | 首位不自信→纯自回归 | 复用 DeepSpec `_empty_proposal` 语义，保证前进 1 |
| target 5 层 aux hidden | 需 target 输出 [1,9,17,25,33] 层 | 配置 `use_aux_hidden_state_outputs`，对齐 eagle3 多层机制 |

---

## 10. DFlash 地基成熟度与 GitHub issue 风险评估（继承前必读）

> 继承 DFlash 前必须确认地基稳不稳。本节双线核验：本地代码走读 + GitHub 一手 issue。锚点：本地 vllm-ascend HEAD = `a99b3b26`（2026-06-27，main）。

### 10.1 DFlash 成熟度评级：早期可用（6.5/10）

- 仅支持 Qwen3 系列；`_raise_if_multimodal()` 空实现（[dflash_proposer.py:269](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py:269)）；测试薄（1 个 e2e + 1 个 UT，无 correctness 对齐）；2026-04 首入、之后每周级 BugFix，仍在 stabilization 期。
- **无官方 benchmark/加速比公开数据** → DSpark 阶段一须自测 DFlash 基线接受率作对照。

### 10.2 6 个 issue × 本地代码核验（git 实测修复 commit 是否在本地）

| Issue | 类型 | 修复在本地? | 本地是否还有 | 依据 |
|-------|------|-----------|-------------|------|
| #9322 | 精度(FDO attn 排序) | ✅ `de00758e` | 已修复 | git 实测 |
| #10080 | 越界 token_indices | ✅ `e7129a39` | 已修复 | git 实测 |
| #10580 | PCP decode 分类 | ✅ `5bd4e395` | 已修复 | git 实测 |
| **#10380** | **精度乱码 num_spec>7** | ❌ 未合入(open PR) | **🔴 确证仍在** | 根因已挖到算子源码 `MAX_MTP=8`（见 §10.3b），修复未合入 |
| **#10088** | **卡死 FULL+TP4 hcom stall** | ❌ open 无修复 | **🔴 代码层确认根因在本地** | 见 §10.3 |
| **#10622** | **性能倒退 spec=1 更差** | ❌ open 无修复 | 🟡 仍在（设计特征非bug） | dflash 无 spec=1 短路（见 §10.3c） |

### 10.3 #10088 卡死深度根因（本地代码精确定位）

**现象**：Qwen3.6-27B(hybrid GDN) + TP4 + FULL_DECODE_ONLY → 每 step 1 次 ~1秒 hcom_allReduce stall，TPOT 退化 16×（1719ms vs PIECEWISE 105ms）。纯 dense Qwen3-8B 不触发。

**根因链（maintainer 定位 + 本地坐实）**：
1. GDN 的 conv1d 自定义算子需在 NPUGraph 建任务组边界：本地 [gdn.py:467](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/ops/gdn.py:467) `graph_task_group_begin` 包住 `npu_causal_conv1d_custom` → [gdn.py:482](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/ops/gdn.py:482) `_end`。每层 GDN 有两处（spec [:467](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/ops/gdn.py:467) + non-spec [:583](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/ops/gdn.py:583)）。
2. `forward`([gdn.py:297](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/ops/gdn.py:297)) 每层都调 → ~48 层 = ~48+ 个 task group 边界。
3. FULL graph replay 时每个边界触发 HCCL 通信重验证(~14ms)：48 层 ≈ 720ms；个别边界致 HCCL 状态损坏 → 1秒+ stall。
4. **本质**：HCCL 缺 graph-safe API（类比 NCCL `ncclCommGraphRegister`），task group 边界与 HCCL 通信在 graph 内冲突。
5. **难修**：删那两个 API → TPOT 48ms 但输出乱码（API 是 conv1d 正确执行所必需，[gdn.py:243](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/ops/gdn.py:243) host 参数同步）。

**触发三条件（缺一不可）**：hybrid GDN（非 dense） + TP>1（有 HCCL） + FULL graph（replay 边界）。

**修复方案排序**：A 规避——GDN+TP>1 时降级 PIECEWISE（0 改动，立即可用）；B 把 TP 通信移出 task group 边界；C 合并 48 层 task group 减边界；D 根治需 HCCL graph-safe API（依赖 CANN）。

### 10.3b #10380 num_spec>7 乱码深度根因（已挖到算子源码 · CONFIRMED）

**现象**：Qwen3.5/3.6-27B + dflash/suffix，当 `num_speculative_tokens > 7` 时输出乱码。修复 commit `2dff103` **未合入本地**（git log grep 空）→ 本地仍存在。

**根因（亲自实测 C++/算子源码）**：一个硬编码常量 `MAX_MTP = 8` 决定所有 UB（Unified Buffer 片上缓冲）尺寸，但未按实际 spec 数动态分配。

| 位置 | 代码 |
|------|------|
| host tiling 常量 | [recurrent_gated_delta_rule_tiling.cpp:52](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/csrc/attention/recurrent_gated_delta_rule/op_host/recurrent_gated_delta_rule_tiling.cpp:52) `const size_t MAX_MTP = 8;` |
| UB 字节计算 | [同文件:515-530](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/csrc/attention/recurrent_gated_delta_rule/op_host/recurrent_gated_delta_rule_tiling.cpp:515) `usedUbBytes = MAX_MTP * (...)` |
| kernel 常量 | [recurrent_gated_delta_rule.h:28](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/csrc/attention/recurrent_gated_delta_rule/op_kernel/arch35/recurrent_gated_delta_rule.h:28) `constexpr uint64_t MAX_MTP = 8;` |
| 各 buffer 按它 InitBuffer | [同文件:120-130](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/csrc/attention/recurrent_gated_delta_rule/op_kernel/arch35/recurrent_gated_delta_rule.h:120) `InitBuffer(qInQueue_, ..., MAX_MTP * alignK_ * ...)` |
| dflash spec 调此算子 | [gdn.py:653](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/ops/gdn.py:653) `npu_recurrent_gated_delta_rule` |
| spec_q_per_seq = num_spec+1 | [gdn.py:446](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/ops/gdn.py:446) |

**为什么 >7 炸、≤7 安全（推导）**：spec 请求 query 长度 `seqLen = num_spec + 1`。num_spec ≤ 7 → seqLen ≤ 8 = MAX_MTP，buffer 刚好够；num_spec ≥ 8 → seqLen ≥ 9 > 8 → 往 8 元素的 UB buffer 塞 9+ token → 越界写损坏相邻 UB 内存 → attention 出垃圾 → 乱码。临界点 `num_spec=8`（seqLen=9）。

**修复方向**：MAX_MTP 从固定 8 改为按 `num_spec+1` 动态分配（host tiling 传 maxSeqLen 字段 → kernel 按它 InitBuffer）。

**仅 hybrid GDN 模型（Qwen3.5/3.6）触发**，纯 dense（Qwen3-4B/8B）不走此算子、不触发。

### 10.3c #10622 spec=1 性能倒退根因（设计特征，非 bug）

**现象**：Qwen3.5-27B + TP2 + dflash + `spec-num=1` 时，单卡吞吐/TPOT 比不开 dflash 还差，Host 下发间隙大、NPU 空闲率暴涨。issue open 无修复 → 本地仍存在。

**根因（本地代码实测 + 现象自洽）**：dflash_proposer 全程**无 `num_spec==1` 短路特判**（grep 无 `==1`）。spec=1 时照样走完整 draft 流程：set_inputs Triton kernel + backbone 完整前向（[dflash_proposer.py:231/239](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py:231) `self.model(...)`/`self._runnable(...)`) + 预算 K/V。这些固定开销与 num_spec 无关。spec=1 时收益（最多多接受 1 token）摊不掉这笔开销，叠加 `--async-scheduling` host 下发吃紧 → NPU 空闲 → 比不开还慢。

**性质**：投机解码固有特征（固定开销没 token 数去摊），非代码缺陷。DSpark 论文自述"draft 固定成本不可恢复"即此。

**修复方向**：若要支持 spec=1，加短路（退化为不投机）。**更务实：别用 spec=1**——DSpark 价值在 block 化多 token（block_size=7）。

### 10.4 对 DSpark 的硬约束（继承 DFlash 的风险传导）

| 约束 | 原因 | 动作 |
|------|------|------|
| **锁 num_speculative_tokens ≤ 7** | #10380 乱码修复未合入，本地无校验，DSpark 默认 block_size=7 踩边界 | proposer `__init__` 加 `assert num_spec<=7` 或选含 #10380 的版本 |
| **hybrid GDN + TP>1 强制 PIECEWISE** | #10088 卡死在本地 | config 校验：检测到 GDN+TP>1+FULL → 降级或报错 |
| **阶段零先验证纯 DFlash** | 地基未稳，先测 DFlash 接受率/无损 | 继承前跑通 DFlash baseline |
| **锁版本，不用 HEAD** | 关键 BugFix 需齐 | 选含 #9322/#10080/#10380 的 tag |
| **优先纯 dense 模型(Qwen3-4B/8B)** | 规避 GDN 相关 #10088/#10380 | 阶段一用 dense |

---

## 9. 源码索引（可点击）

| 环节 | 跳转 |
|------|------|
| DFlash proposer | [dflash_proposer.py:15](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py:15) |
| 投机基类 | [llm_base_proposer.py:130](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py:130) |
| 分发表 | [spec_decode/__init__.py:33](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/__init__.py:33) |
| runner 调用点 | [model_runner_v1.py:1686](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:1686) |
| DFlash 模型 patch | [patch_qwen3_dflash.py:62](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_dflash.py:62) |
| rejection sampler | [rejection_sampler.py:36](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/sample/rejection_sampler.py:36) |
| DeepSpec markov 采样 | [markov_head.py:55](vscode://file/Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/markov_head.py:55) |
| DeepSpec 置信度截断 | [draft_ops.py:82](vscode://file/Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/draft_ops.py:82) |
| ops-transformer block sparse | [block_sparse_attention](vscode://file/Users/linyi/code/Documents/code/ops-transformer/attention/block_sparse_attention) |
| #10380 根因 host tiling | [recurrent_gated_delta_rule_tiling.cpp:52](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/csrc/attention/recurrent_gated_delta_rule/op_host/recurrent_gated_delta_rule_tiling.cpp:52) |
| #10380 根因 kernel | [recurrent_gated_delta_rule.h:28](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/csrc/attention/recurrent_gated_delta_rule/op_kernel/arch35/recurrent_gated_delta_rule.h:28) |
| GDN spec 调算子 | [gdn.py:653](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/ops/gdn.py:653) |
| #10622 dflash 前向(无spec=1短路) | [dflash_proposer.py:231](vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py:231) |

> 注：对话回复里的链接（Claude desktop 气泡）默认点不开 vscode://，要打开源码用本文档（Obsidian 内）的链接，或复制路径在 VSCode `Cmd+P` 粘贴 `路径:行号`。

---

## 附录：让 vscode 链接在 Obsidian 点得动（实测）

本文所有源码链接格式为 `vscode://file/<绝对路径>:<行号>`，点击直达 VSCode 对应行。系统层已实测可跳（`open "vscode://file/...:467"` 成功）。Obsidian desktop 点击行为按以下处理：

**方式一（默认，无需配置）**：直接点链接 → Obsidian 弹"This link will open an external app, continue?"确认框 → 点 **Open** → VSCode 打开并跳到该行。每次点弹一次框。

**方式二（永久信任，免确认框）**：点确认框时勾选 **"Always allow"**（或 Don't ask again），之后该协议永久放行，点击直接跳。

**方式三（彻底放行 + 行号精准跳转，推荐）**：
1. 确保 VSCode 已装且注册了 `code` 命令（VSCode 内 Cmd+Shift+P → "Shell Command: Install 'code' command in PATH"）。
2. `vscode://` 协议由 VSCode 自带注册，macOS 上无需额外配置；点击链接选 Open 即跳到 `:行号`。

**若完全无反应（极少数）**：
- 确认 VSCode 在 `/Applications`（本机已确认 ✅）。
- 在终端实测：`open "vscode://file/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/ops/gdn.py:467"` —— 能跳说明协议正常，问题在 Obsidian 确认框被忽略，重新点并留意弹框。
- 实在不行：复制链接里的路径，VSCode 里 Cmd+P 粘贴路径 + `:行号` 跳转。

> 取舍说明：`vscode://` 比 `file://` 强在**能跳到具体行号**（file:// 只能打开文件）；代价是首次需过一次 Obsidian 确认框。本文已全部用 `vscode://file/...:行号` 格式（41 处源码链接 + §9 索引表）。
