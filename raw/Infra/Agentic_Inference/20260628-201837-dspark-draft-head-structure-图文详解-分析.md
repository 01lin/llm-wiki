# DSpark 投机头：模型结构与实现 · 图文详解（少公式版）

> 面向「数学公式理解不强」的读者，把 DSpark draft head 的结构、参数、实现逻辑用大白话 + 图讲清。
> 一手代码：`/Users/linyi/code/Documents/code/DeepSpec/`，行号 grep 实测。配置以 Qwen3-4B 为例（[[20260628-191517-dspark-draft-confidence-e2e-deepdive-分析]] 是其 e2e 串讲篇）。
> 图为 Obsidian 内嵌 SVG，直接渲染；源码引用为 file:// 可点击（系统默认程序打开）跳转。

---

## 0. 一句话心智模型

> **DSpark 投机头 = 一个「迷你版 Qwen3」（只有 5 层）+ 两个小挂件（Markov 头、置信度头）。**
> 迷你 Qwen3 一次性猜出 7 个候选词，Markov 头把这 7 个词"前后串起来"修正，置信度头给每个词"打分"决定信不信。

玩「接龙猜词」：target 大模型刚说完一个词（**anchor** 锚点），投机头要一口气猜接下来 7 个词，再让 target 一次性验证，猜对几个就白赚几个 token。

---

## 1. 整体结构（5 个盒子）

```
┌─ ① Target 模型（Qwen3-4B，36层，冻结）────────────────┐
│   跑一次，抽第 1/9/17/25/33 层 hidden（5份）拼成 context │
└───────────────────────┬───────────────────────────────┘
                        ↓  [1, 序列长, 2560×5]
┌─ ② Draft 骨干 = 迷你 Qwen3（仅 5 层 DecoderLayer）─────┐
│   fc 把 5份 hidden 压回 1份(2560) → 当 K/V 上下文        │
│   输入 [anchor, MASK×6]；Q=draft, K/V=[target|draft]     │
│   输出 7 份 hidden h₁..h₇   [1, 7, 2560]                  │
└───────────────────────┬───────────────────────────────┘
                        ↓
┌─ ③ lm_head（冻结共享）：hidden → 词表分数 ─────────────┐
│   7 份基础候选分 U₁..U₇   [1, 7, 151936]                 │
└──────────┬──────────────────────────┬──────────────────┘
          ↓ (Markov)                  ↓ (置信度)
┌─ ④ Markov 头 ──────────┐  ┌─ ⑤ 置信度头 ──────────────┐
│ 把7个独立候选串成自回归 │  │ 给每个词打"会被接受吗"的分 │
│ W1=查表256, W2投回词表  │  │ 一个 Linear(2816→1)        │
└────────────────────────┘  └────────────────────────────┘
```

**关键直觉**：盒子②是并行的（快，但词之间没关系）→ 盒子④把它串行化（补上前后依赖）→ 盒子⑤做质检。这就是"半自回归"：一半并行（骨干）、一半串行（Markov）。

<svg width="100%" viewBox="0 0 680 560" xmlns="http://www.w3.org/2000/svg" role="img">
<title>DSpark 投机头整体结构</title>
<desc>target 模型提供 hidden 给 draft 的 5 层迷你骨干，再经 Markov 头和置信度头输出候选词与分数</desc>
<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#888"/></marker></defs>

<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="40" y="30" font-weight="500">DSpark 投机头 · 整体结构（Qwen3-4B 配置）</text>

<rect fill="#f0f0ee" stroke="#b8b8b2" stroke-width="1.5" x="40" y="48" width="600" height="78" rx="8"/>
<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="60" y="74" font-weight="500">① Target 模型（Qwen3-4B，冻结不训练）</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="60" y="96">跑一次，抽出第 1/9/17/25/33 层的 hidden（5 份），拼成 context 特征</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="60" y="114">输出形状：[1 句, 序列长, 2560×5]  → 喂给下面的 draft</text>
<line x1="340" y1="126" x2="340" y2="150" stroke="#888" stroke-width="1.5" marker-end="url(#ar)"/>

<rect fill="#dce9fb" stroke="#7aa7e0" stroke-width="1.5" x="40" y="152" width="600" height="150" rx="8"/>
<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="60" y="178" font-weight="500">② Draft 骨干 = 迷你 Qwen3（只有 5 层 DecoderLayer）</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="60" y="198">fc：把 5 份 hidden（2560×5）压回 2560 一份 → 当 K/V 的"上下文记忆"</text>
<rect fill="#d3efe8" stroke="#6cc0ac" stroke-width="1.5" x="60" y="212" width="560" height="46" rx="6"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="76" y="231">输入：[anchor, MASK, MASK, ... MASK]  共 7 个位置（block_size=7）</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="76" y="249">每层注意力：Q=draft，K/V=[target上下文 | draft]，同块内双向</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="60" y="284">输出：7 个位置各一份 hidden  h₁..h₇   形状 [1, 7, 2560]</text>
<line x1="340" y1="302" x2="340" y2="326" stroke="#888" stroke-width="1.5" marker-end="url(#ar)"/>

<rect fill="#fbeacb" stroke="#dcab4e" stroke-width="1.5" x="40" y="328" width="600" height="52" rx="8"/>
<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="60" y="352" font-weight="500">③ lm_head（冻结共享）：hidden → 词表分数</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="60" y="370">每个位置出一份"基础候选分" U₁..U₇   形状 [1, 7, 151936]（词表大小）</text>

<line x1="180" y1="380" x2="180" y2="430" stroke="#888" stroke-width="1.5" marker-end="url(#ar)"/>
<line x1="500" y1="380" x2="500" y2="430" stroke="#888" stroke-width="1.5" marker-end="url(#ar)"/>

<rect fill="#d8eedb" stroke="#79bd83" stroke-width="1.5" x="40" y="432" width="270" height="100" rx="8"/>
<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="58" y="456" font-weight="500">④ Markov 头（挂件 A）</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="58" y="476">把 7 个独立候选"前后串起来"</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="58" y="494">查上一个词的偏置，串行修正</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="58" y="512">参数：W1=查表 256 维, W2 投回词表</text>

<rect fill="#e7dcf5" stroke="#a886d4" stroke-width="1.5" x="370" y="432" width="270" height="100" rx="8"/>
<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="388" y="456" font-weight="500">⑤ 置信度头（挂件 B）</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="388" y="476">给每个词打"会被接受吗"的分</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="388" y="494">输入 [hidden ; 上个词的查表向量]</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="388" y="512">参数：就一个 Linear(2816 → 1)</text>
</svg>

---

## 2. 盒子② 迷你 Qwen3 骨干：注意力怎么"偷看" target

普通 transformer 的 Q/K/V 都来自同一串输入；DSpark 不一样（[modeling.py:104-113](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/qwen3/modeling.py)）：

- **Q（查询）只来自 draft**：draft 在问"接下来 7 个词是啥？"
- **K/V（记忆）= target 上下文 + draft 自己 拼起来**：`k = cat(k_proj(target_hidden), k_proj(draft))`。记忆里既有真实历史，又有自己刚猜的。

**两条 mask 规则**（谁能看见谁，[common.py:86-96](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/common.py)）：
1. 看 target 历史：只能看 anchor 之前的真实 token（单向因果，不偷看未来）
2. 看 draft 自己：**同一 block 内 7 个位置互相都能看（双向）**——位置 3 能参考位置 5，比纯左→右更全。

**真实参数（Qwen3-4B，draft `deepcopy(target_config)` 继承维度、仅改层数，[config.py:37](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/qwen3/config.py) / [:40](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/qwen3/config.py)）**：

| 参数               | 值              | 含义                   | 来源                                                                                                      |
| ---------------- | -------------- | -------------------- | ------------------------------------------------------------------------------------------------------- |
| num_draft_layers | **5**          | 迷你 Qwen3 层数（本体 36 层） | [config 实测](file:///Users/linyi/code/Documents/code/DeepSpec/config/dspark/dspark_qwen3_4b.py) |
| block_size γ     | **7**          | 一次猜 7 个词             | [config 实测](file:///Users/linyi/code/Documents/code/DeepSpec/config/dspark/dspark_qwen3_4b.py) |
| target_layer_ids | [1,9,17,25,33] | 抽本体这 5 层当线索          | [config 实测](file:///Users/linyi/code/Documents/code/DeepSpec/config/dspark/dspark_qwen3_4b.py) |
| hidden_size      | 2560           | token 向量宽度           | Qwen3-4B 官方规格（继承）                                                                                       |
| vocab_size       | 151936         | 词表大小                 | Qwen3-4B 官方规格（继承）                                                                                       |

> 维度 2560/词表 151936 是 Qwen3-4B 官方公开规格（draft 继承 target_config）；层数=5/block=7/抽层=[1,9,17,25,33] 是本地 config.py 实测。两者已区分，避免拿官方规格冒充本地实测。

<svg width="100%" viewBox="0 0 680 470" xmlns="http://www.w3.org/2000/svg" role="img">
<title>DSpark 骨干注意力的 KV 拼接</title>
<desc>Q 仅来自 draft，K 和 V 由 target 上下文与 draft 拼接而成，同一 block 内双向可见</desc>
<defs><marker id="ar2" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#888"/></marker></defs>

<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="40" y="28" font-weight="500">放大盒子②：每层注意力怎么"偷看" target（modeling.py forward）</text>

<rect fill="#d3efe8" stroke="#6cc0ac" stroke-width="1.5" x="40" y="46" width="250" height="64" rx="8"/>
<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="58" y="70" font-weight="500">draft 输入（noise）</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="58" y="90">[anchor, MASK, MASK ... ]</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="58" y="106">形状 [1, 7, 2560]</text>

<rect fill="#f0f0ee" stroke="#b8b8b2" stroke-width="1.5" x="390" y="46" width="250" height="64" rx="8"/>
<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="408" y="70" font-weight="500">target 上下文 hidden</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="408" y="90">经 fc 压成一份</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="408" y="106">形状 [1, ctx长, 2560]</text>

<text font-family="sans-serif" font-size="12" fill="#333333" x="165" y="138" text-anchor="middle">只产生 Q</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="515" y="138" text-anchor="middle">只产生 K、V 的"上下文部分"</text>
<line x1="165" y1="110" x2="165" y2="150" stroke="#888" stroke-width="1.5" marker-end="url(#ar2)"/>
<line x1="515" y1="110" x2="515" y2="150" stroke="#888" stroke-width="1.5" marker-end="url(#ar2)"/>

<rect fill="#dce9fb" stroke="#7aa7e0" stroke-width="1.5" x="40" y="152" width="600" height="120" rx="8"/>
<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="60" y="178" font-weight="500">注意力计算（Q 去查 K，取出 V）</text>
<rect fill="#d3efe8" stroke="#6cc0ac" stroke-width="1.5" x="60" y="190" width="170" height="64" rx="6"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="76" y="212" font-weight="500">Q = draft</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="76" y="232">「我想知道下7个词」</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="76" y="248">q_proj(draft)</text>
<rect fill="#fbeacb" stroke="#dcab4e" stroke-width="1.5" x="245" y="190" width="375" height="64" rx="6"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="261" y="212" font-weight="500">K / V = [ target上下文 | draft ] 拼起来</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="261" y="232">k = cat(k_proj(target), k_proj(draft))</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="261" y="248">「记忆 = 真实历史 + 自己刚猜的」</text>
<line x1="340" y1="272" x2="340" y2="298" stroke="#888" stroke-width="1.5" marker-end="url(#ar2)"/>

<rect fill="#e7dcf5" stroke="#a886d4" stroke-width="1.5" x="40" y="300" width="600" height="92" rx="8"/>
<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="60" y="324" font-weight="500">两条 mask 规则（谁能看见谁）</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="60" y="346">· 看 target 历史：只能看 anchor 之前的真实 token（单向，因果）</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="60" y="366">· 看 draft 自己：同一个 block 内 7 个位置互相都能看（双向）</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="60" y="386">→ 双向是关键：位置3能"参考"位置5，比纯左到右信息更全</text>

<rect fill="#d8eedb" stroke="#79bd83" stroke-width="1.5" x="40" y="406" width="600" height="46" rx="8"/>
<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="60" y="430" font-weight="500">输出：7 份 hidden h₁..h₇</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="360" y="430">形状 [1, 7, 2560]，送往 lm_head</text>
</svg>

---

## 3. 盒子④ Markov 头：把"并行"变"自回归"（公式核心）

吓人的公式 `p_k(v) = softmax(U_k(v) + B_k(x_{k-1}, v))` 翻译成大白话：

> **最终给每个词的分数 = 基础分 U_k（骨干算的）+ 偏置 B_k（根据上一个词查出来的修正）。**

### 3.1 偏置 B 怎么来 —— 两步查表（[markov_head.py:17-18](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/markov_head.py)）

1. `W1`：把"上一个词"查成 256 维小向量 —— `Embedding(词表, 256)`
2. `W2`：把这个 256 维向量投回整个词表分数 —— `Linear(256, 词表)`

```python
self.markov_w1 = nn.Embedding(vocab_size, markov_rank)   # 词 → 256
self.markov_w2 = nn.Linear(markov_rank, vocab_size)      # 256 → 词表
# B(上个词) = markov_w2( markov_w1[上个词] )
```

> 为什么要 256 维中转？直接"词→词"转移表是 151936×151936（几百亿，存不下）。拆成 `词→256→词` 只要约 7800 万参数，这叫**低秩分解**，`markov_rank=256` 就是中间宽度。

### 3.2 串行循环 —— 接龙（[markov_head.py:76-89](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/markov_head.py)）

```python
prev = anchor
for k in range(7):
    分数 = U[k] + B(prev)        # 基础分 + 上个词的偏置
    x_k = 采样(分数)
    prev = x_k                   # ← 喂回下一步（这就是"串行依赖"）
```

第 2 步用 x₁、第 3 步用 x₂…… 后面的词"接上了"前面，缓解 suffix decay（并行骨干里 x₃ 不知道 x₂ 猜了啥，容易跑偏）。

### 3.3 成本为什么极低

循环里**不重跑那 5 层骨干**：U₁..U₇ 是骨干一次性算好、固定不变的。每步只做"查 W1 + 过 W2 + 加法"三个廉价操作。所以 7 步串行虽然排队，但每步几乎不耗时——这就是"半自回归"的精髓：骨干并行（快）+ Markov 串行（轻）。

> 三种头变体（[markov_head.py](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/markov_head.py)）：Vanilla（默认，纯查表）/ Gated（加门控）/ RNN（带跨位置记忆）。生产默认 Vanilla，循环骨架一致，只是 B 的算法不同。

<svg width="100%" viewBox="0 0 680 520" xmlns="http://www.w3.org/2000/svg" role="img">
<title>Markov 头如何把并行候选串成自回归</title>
<desc>每一步用上一个采出的词查偏置表，叠加到基础分数上再采样，逐步串联</desc>
<defs>
<marker id="ar3" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#888"/></marker>
<marker id="ar3b" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#3b9a82"/></marker>
</defs>

<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="40" y="28" font-weight="500">放大盒子④：Markov 头把"并行7个独立候选"串成"前后有关系"</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="40" y="48">公式：最终分数 = 基础分 U_k  ＋  偏置 B_k（B_k = 查"上一个词"得到的修正）</text>

<text font-family="sans-serif" font-size="14" fill="#8a6d1e" x="40" y="78" font-weight="500">偏置怎么来：B = W2( W1[上一个词] )  —— 两步查表</text>
<rect fill="#fbeacb" stroke="#dcab4e" stroke-width="1.5" x="40" y="88" width="185" height="58" rx="6"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="58" y="110" font-weight="500">W1：词 → 256维向量</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="58" y="130">查表 Embedding(词表,256)</text>
<line x1="225" y1="117" x2="255" y2="117" stroke="#888" stroke-width="1.5" marker-end="url(#ar3)"/>
<rect fill="#fbeacb" stroke="#dcab4e" stroke-width="1.5" x="258" y="88" width="185" height="58" rx="6"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="276" y="110" font-weight="500">W2：256维 → 词表分数</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="276" y="130">Linear(256, 151936)</text>
<line x1="443" y1="117" x2="473" y2="117" stroke="#888" stroke-width="1.5" marker-end="url(#ar3)"/>
<rect fill="#fbeacb" stroke="#dcab4e" stroke-width="1.5" x="476" y="88" width="164" height="58" rx="6"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="494" y="110" font-weight="500">B = 一份词表偏置</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="494" y="130">"上个词的余味"</text>

<text font-family="sans-serif" font-size="14" fill="#2f7d43" x="40" y="178" font-weight="500">串行循环：7 步，每步把上一步采的词喂进来（markov_head.py for 循环）</text>

<rect fill="#d8eedb" stroke="#79bd83" stroke-width="1.5" x="40" y="192" width="140" height="86" rx="8"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="58" y="214" font-weight="500">第1步</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="58" y="234">上个词 = anchor</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="58" y="252">U₁ + B(anchor)</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="58" y="270">采样 → x₁</text>
<line x1="180" y1="235" x2="212" y2="235" stroke="#3b9a82" stroke-width="2" marker-end="url(#ar3b)"/>
<text font-family="sans-serif" font-size="12" fill="#2f7d43" x="196" y="226" text-anchor="middle">x₁</text>

<rect fill="#d8eedb" stroke="#79bd83" stroke-width="1.5" x="214" y="192" width="140" height="86" rx="8"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="232" y="214" font-weight="500">第2步</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="232" y="234">上个词 = x₁</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="232" y="252">U₂ + B(x₁)</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="232" y="270">采样 → x₂</text>
<line x1="354" y1="235" x2="386" y2="235" stroke="#3b9a82" stroke-width="2" marker-end="url(#ar3b)"/>
<text font-family="sans-serif" font-size="12" fill="#2f7d43" x="370" y="226" text-anchor="middle">x₂</text>

<rect fill="#d8eedb" stroke="#79bd83" stroke-width="1.5" x="388" y="192" width="140" height="86" rx="8"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="406" y="214" font-weight="500">第3步</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="406" y="234">上个词 = x₂</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="406" y="252">U₃ + B(x₂)</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="406" y="270">采样 → x₃</text>
<line x1="528" y1="235" x2="560" y2="235" stroke="#3b9a82" stroke-width="2" marker-end="url(#ar3b)"/>
<text font-family="sans-serif" font-size="12" fill="#2f7d43" x="558" y="226" text-anchor="middle">…</text>

<rect fill="#f0f0ee" stroke="#b8b8b2" stroke-width="1.5" x="562" y="192" width="78" height="86" rx="8"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="580" y="226">直到</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="580" y="246">第7步</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="580" y="266">x₇</text>

<rect fill="#dce9fb" stroke="#7aa7e0" stroke-width="1.5" x="40" y="300" width="600" height="74" rx="8"/>
<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="60" y="324" font-weight="500">为什么这样能缓解"越往后越不准"</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="60" y="346">并行骨干里，x₃ 不知道 x₂ 猜了啥 → 后面容易跑偏（suffix decay）</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="60" y="366">Markov 头让每一步都看到"上一步真正采出的词" → 接上了前后文</text>

<rect fill="#e7dcf5" stroke="#a886d4" stroke-width="1.5" x="40" y="388" width="600" height="110" rx="8"/>
<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="60" y="412" font-weight="500">成本极低：循环里不重跑那 5 层骨干</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="60" y="434">· U₁..U₇ 是骨干一次性算好的，固定不变</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="60" y="454">· 每步只做：查 W1（1次）+ 过 W2（1次）+ 加法（1次）</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="60" y="474">· 所以 7 步串行虽然排队，但每步便宜到几乎不耗时</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="60" y="494">· 这就是"半自回归"：骨干并行（快） + Markov 串行（轻）</text>
</svg>

---

## 3.5 采样过程到底怎么实现（补全前图里的"采样→x_k"黑盒）

每个 draft token 的分数 = 骨干基础分 U_k + Markov 偏置 B_k。这一节把"加完之后怎么采出 x_k"讲到代码级。

### 3.5.1 U_k 和 B_k 在 logits 空间相加（易错点）

`step_logits = U_k + B_k` 发生在 **logits（原始分数）空间，不是概率空间**（[markov_head.py:78-82](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/markov_head.py)）：

```python
step_logits = self.apply_step_logits(
    base_logits[:, step_idx, :],   # U_k：骨干基础分（还没 softmax）
    token_ids=prev_token_ids,      # 上一个词
)   # 内部 = U_k + B_k，两个 [词表] 向量逐元素相加
```

> 为什么强调？softmax 是非线性的。**先加 bias 再 softmax**（DSpark 做法）≠ 两个概率相加。在 logits 空间加偏置，相当于"乘性地"调整每个词的概率比例，这是数学上正确的转移修正方式。

### 3.5.2 sample_tokens 两条路径（[sampling.py:20-27](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/utils/sampling.py)）

```python
def sample_tokens(logits, temperature=0.0):
    if temperature < 1e-5:                       # ① 贪心
        return torch.argmax(logits, dim=-1)      # 取分数最高的词，不算 softmax
    flat = logits.reshape(-1, vocab) / temperature   # ② 随机采样
    probs = torch.softmax(flat, dim=-1)              # 除温度 → softmax
    return torch.multinomial(probs, num_samples=1)   # 按概率掷骰子
```

- **路径①（temperature < 1e-5，贪心）**：直接 `argmax`，取 U_k+B_k 后最高分的词，最省。
- **路径②（temperature ≥ 1e-5，随机）**：`logits/温度 → softmax → multinomial`。温度越大越随机。

<svg width="100%" viewBox="0 0 680 540" xmlns="http://www.w3.org/2000/svg" role="img">
<title>DSpark 单步采样实现</title>
<desc>U_k 与 B_k 在 logits 空间相加后，按温度走贪心 argmax 或 softmax 多项式采样</desc>
<defs><marker id="s1" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#888"/></marker></defs>

<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="40" y="28" font-weight="500">单步采样实现（markov_head.py 循环内 + sampling.py）</text>

<rect fill="#fbeacb" stroke="#dcab4e" stroke-width="1.5" x="40" y="46" width="285" height="58" rx="8"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="58" y="70" font-weight="500">U_k：骨干基础分（lm_head 输出）</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="58" y="90">形状 [词表]，还没 softmax</text>
<rect fill="#d8eedb" stroke="#79bd83" stroke-width="1.5" x="355" y="46" width="285" height="58" rx="8"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="373" y="70" font-weight="500">B_k：Markov 偏置 = W2(W1[上个词])</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="373" y="90">形状 [词表]，也是 logits</text>
<line x1="182" y1="104" x2="320" y2="134" stroke="#888" stroke-width="1.5" marker-end="url(#s1)"/>
<line x1="497" y1="104" x2="360" y2="134" stroke="#888" stroke-width="1.5" marker-end="url(#s1)"/>

<rect fill="#e7dcf5" stroke="#a886d4" stroke-width="1.5" x="170" y="138" width="340" height="56" rx="8"/>
<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="190" y="161" font-weight="500">step_logits = U_k + B_k（在 logits 空间相加）</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="190" y="181">关键：先加再 softmax ≠ 概率相加（softmax 非线性）</text>
<line x1="340" y1="194" x2="340" y2="218" stroke="#888" stroke-width="1.5" marker-end="url(#s1)"/>

<rect fill="#dce9fb" stroke="#7aa7e0" stroke-width="1.5" x="200" y="222" width="280" height="40" rx="8"/>
<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="220" y="247" font-weight="500">sample_tokens(step_logits, temperature)</text>
<line x1="270" y1="262" x2="190" y2="290" stroke="#888" stroke-width="1.5" marker-end="url(#s1)"/>
<line x1="410" y1="262" x2="490" y2="290" stroke="#888" stroke-width="1.5" marker-end="url(#s1)"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="210" y="282">温度≈0</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="470" y="282">温度&gt;0</text>

<rect fill="#d3efe8" stroke="#6cc0ac" stroke-width="1.5" x="40" y="294" width="280" height="92" rx="8"/>
<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="58" y="318" font-weight="500">① 贪心（argmax）</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="58" y="340">直接取分数最高的词</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="58" y="360">不算 softmax，最省</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="58" y="380">temperature &lt; 1e-5 时走这条</text>

<rect fill="#f9d6cd" stroke="#e08c79" stroke-width="1.5" x="360" y="294" width="280" height="92" rx="8"/>
<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="378" y="318" font-weight="500">② 随机采样（multinomial）</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="378" y="340">logits / 温度 → softmax → 概率</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="378" y="360">按概率"掷骰子"抽一个词</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="378" y="380">温度越大越随机</text>

<line x1="180" y1="386" x2="180" y2="410" stroke="#888" stroke-width="1.5" marker-end="url(#s1)"/>
<line x1="500" y1="386" x2="500" y2="410" stroke="#888" stroke-width="1.5" marker-end="url(#s1)"/>
<rect fill="#f0f0ee" stroke="#b8b8b2" stroke-width="1.5" x="40" y="414" width="600" height="40" rx="8"/>
<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="60" y="439" font-weight="500">采出 x_k → 喂回循环下一步当"上个词"（串行依赖）</text>

<rect fill="#dce9fb" stroke="#7aa7e0" stroke-width="1.5" x="40" y="466" width="600" height="58" rx="8"/>
<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="60" y="490" font-weight="500">易错点：采样用 step_logits，但验证用的 draft 概率 q 另算</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="60" y="510">循环还返回 corrected_logits(=step_logits)，之后 logits_to_probs 转成 q 给拒绝采样</text>
</svg>

### 3.5.3 易错点：采样用 step_logits，验证用的 q 另算

采样时用 `step_logits` 采词；但 target 验证（拒绝采样）需要的是 draft 概率分布 q，是**另算**的。循环除了返回采出的词，还返回 `corrected_logits`（每步 step_logits 堆起来，[markov_head.py:83](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/markov_head.py)），之后打包 proposal 时才转成 q（[draft_ops.py:140](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/draft_ops.py)）：

```python
draft_probs = logits_to_probs(draft_logits[:, :proposal_draft_tokens, :], temperature)
```

`logits_to_probs`（[sampling.py:6-11](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/utils/sampling.py)）与采样同源：贪心时是 one-hot，否则 `softmax(logits/温度)`。这个 q 就是拒绝采样里 `min(1, p/q)` 的分母。

> 闭环：**同一份 U_k+B_k 用了两次**——一次采样得候选词 x_k，一次转成概率 q 供 target 验证。贪心下两者天然一致（argmax 选的词在 q 里概率=1）。

### 3.5.4 完整采样链（一步内真实顺序）

```
1. 骨干算 U_k                      （lm_head，7 个位置一次性算好）
2. 查上个词得 B_k = W2(W1[prev])   （compute_step_bias）
3. step_logits = U_k + B_k         （logits 空间相加，apply_step_logits）
4. 采样 x_k：温度≈0→argmax；温度>0→multinomial(softmax(step_logits/温度))
5. prev = x_k                      （喂回，串行依赖）
6. 存 corrected_logits = step_logits → logits_to_probs → draft 概率 q（给验证）
```

---

## 4. 盒子⑤ 置信度头：给每个词打"会被接受吗"的分

结构最简单 —— **就一个 `Linear(2816 → 1)`**（[common.py:44-49](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/common.py)）。

```python
class AcceptRatePredictor(nn.Module):
    def __init__(self, input_dim):
        self.proj = nn.Linear(input_dim, 1)   # input_dim = 2560 + 256 = 2816
    def forward(self, features):
        return self.proj(features).squeeze(-1)  # 输出 1 个数（logit）
```

- **输入**：拼两样东西（[modeling.py:306](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/modeling/dspark/qwen3/modeling.py)）—— 该位置 hidden（2560）+ 上个词查表向量（256）= 2816 维。
- **输出**：过 Linear 得 1 个数，sigmoid 压到 0~1 = `c_k`（接受概率）。
- **用途**：从前往后扫，**第一个低于阈值的位置砍尾**（[draft_ops.py:82-93](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/draft_ops.py)）。例：阈值 0.5，`c₄=0.3` 首次跌破 → 只送前 3 个词验证，省 target 算力。

> `c_k` 是**条件概率**："前面都被接受的前提下，第 k 个会被接受吗"。所以可连乘求"前 j 个全过"概率 `a_j=c₁×…×c_j`——但连乘只在离线诊断用（[confidence_head.py:366](file:///Users/linyi/code/Documents/code/DeepSpec/deepspec/eval/dspark/confidence_head.py)），在线截断只看单点。细节见 [[20260628-191517-dspark-draft-confidence-e2e-deepdive-分析]] §2。

<svg width="100%" viewBox="0 0 680 470" xmlns="http://www.w3.org/2000/svg" role="img">
<title>置信度头打分与截断</title>
<desc>把 hidden 和上个词向量拼起来过一个 Linear 得到分数，sigmoid 后低于阈值就截断</desc>
<defs><marker id="ar4" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#888"/></marker></defs>

<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="40" y="28" font-weight="500">放大盒子⑤：置信度头给每个词打"会被接受吗"的分（AcceptRatePredictor）</text>

<rect fill="#dce9fb" stroke="#7aa7e0" stroke-width="1.5" x="40" y="48" width="285" height="60" rx="8"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="58" y="72" font-weight="500">输入A：该位置的 hidden h_k</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="58" y="92">2560 维（骨干算的）</text>
<rect fill="#fbeacb" stroke="#dcab4e" stroke-width="1.5" x="355" y="48" width="285" height="60" rx="8"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="373" y="72" font-weight="500">输入B：上个词的查表向量 W1[x_{k-1}]</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="373" y="92">256 维（复用 Markov 的 W1）</text>
<line x1="182" y1="108" x2="300" y2="140" stroke="#888" stroke-width="1.5" marker-end="url(#ar4)"/>
<line x1="497" y1="108" x2="380" y2="140" stroke="#888" stroke-width="1.5" marker-end="url(#ar4)"/>

<rect fill="#e7dcf5" stroke="#a886d4" stroke-width="1.5" x="220" y="144" width="240" height="52" rx="8"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="240" y="166" font-weight="500">拼起来 = 2816 维</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="240" y="186">[ h_k ; W1[上个词] ]</text>
<line x1="340" y1="196" x2="340" y2="220" stroke="#888" stroke-width="1.5" marker-end="url(#ar4)"/>

<rect fill="#d8eedb" stroke="#79bd83" stroke-width="1.5" x="220" y="224" width="240" height="64" rx="8"/>
<text font-family="sans-serif" font-size="12" fill="#333333" x="240" y="248" font-weight="500">一个 Linear(2816 → 1)</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="240" y="268">→ 一个数 → sigmoid 压到 0~1</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="240" y="284">= c_k（这个词的接受概率）</text>
<line x1="340" y1="288" x2="340" y2="312" stroke="#888" stroke-width="1.5" marker-end="url(#ar4)"/>

<rect fill="#f0f0ee" stroke="#b8b8b2" stroke-width="1.5" x="40" y="316" width="600" height="134" rx="8"/>
<text font-family="sans-serif" font-size="14" fill="#1a1a1a" x="60" y="340" font-weight="500">用途：从前往后扫，第一个低于阈值的位置就"砍尾"</text>
<rect fill="#d8eedb" stroke="#79bd83" stroke-width="1.5" x="60" y="352" width="80" height="34" rx="4"/><text font-family="sans-serif" font-size="12" fill="#333333" x="78" y="374">c₁=0.9</text>
<rect fill="#d8eedb" stroke="#79bd83" stroke-width="1.5" x="148" y="352" width="80" height="34" rx="4"/><text font-family="sans-serif" font-size="12" fill="#333333" x="166" y="374">c₂=0.8</text>
<rect fill="#d8eedb" stroke="#79bd83" stroke-width="1.5" x="236" y="352" width="80" height="34" rx="4"/><text font-family="sans-serif" font-size="12" fill="#333333" x="254" y="374">c₃=0.7</text>
<rect fill="#f9d6cd" stroke="#e08c79" stroke-width="1.5" x="324" y="352" width="80" height="34" rx="4"/><text font-family="sans-serif" font-size="12" fill="#333333" x="338" y="374">c₄=0.3</text>
<rect fill="#f0f0ee" stroke="#b8b8b2" stroke-width="1.5" x="412" y="352" width="80" height="34" rx="4"/><text font-family="sans-serif" font-size="12" fill="#333333" x="430" y="374">c₅✂</text>
<rect fill="#f0f0ee" stroke="#b8b8b2" stroke-width="1.5" x="500" y="352" width="80" height="34" rx="4"/><text font-family="sans-serif" font-size="12" fill="#333333" x="518" y="374">c₆✂</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="60" y="410">阈值=0.5：c₄=0.3 首次跌破 → 只送前 3 个词去验证，后面砍掉</text>
<text font-family="sans-serif" font-size="12" fill="#333333" x="60" y="430">好处：大概率被拒的尾巴提前扔掉，不浪费 target 的算力（verify smarter）</text>
</svg>

---

## 5. 参数总览（Qwen3-4B）

| 模块 | 参数 / 结构 | 量级 | 训不训 |
|------|------------|------|--------|
| Target 模型 | Qwen3-4B 本体 36 层 | 4B | 冻结 |
| embed_tokens / lm_head | 与 target 共享权重 | 复用 | 冻结 |
| Draft 骨干 | 5 层 DecoderLayer（hidden 2560） | 主要可训参数 | 训练 |
| fc | Linear(2560×5 → 2560) | 中等 | 训练 |
| Markov W1 | Embedding(151936, 256) | ~3900万 | 训练 |
| Markov W2 | Linear(256, 151936) | ~3900万 | 训练 |
| 置信度头 | Linear(2816, 1) | ~2816 | 训练 |

> 真正"重"的是 5 层骨干 + Markov 的 W1/W2；置信度头几乎免费。embedding/lm_head 复用 target、不增参数。

---

## 6. 一句话回顾每个盒子

1. **Target**：冻结大模型，借它 5 层 hidden 当线索。
2. **骨干**：迷你 5 层 Qwen3，一次并行猜 7 个候选（K/V 拼 target+draft，块内双向）。
3. **lm_head**：候选 hidden → 词表基础分 U。
4. **Markov 头**：U + "上个词查出的偏置 B"，串行接龙补依赖（低秩 256 中转）。
5. **置信度头**：[hidden;上个词] 过一个 Linear → 接受分 c，低于阈值砍尾。

完整 e2e 推理流程（含 target 验证、拒绝采样）见姊妹篇 [[20260628-191517-dspark-draft-confidence-e2e-deepdive-分析]]；调度粒度（step/request level）见其 §4.5。
