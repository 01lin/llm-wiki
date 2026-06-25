# vLLM/vllm-ascend Hybrid Attention 动态投机精度异常：最终方案设计

> 生成时间：2026-06-25
> 走读版本：vllm `0d2961229` / vllm-ascend `8afdf356` / sglang `b5e0965b07`
> 约束（用户确认）：① 同 step 内所有 req 投机长度一致，长度只在 step 间切换（mtp3→mtp1）；② **同步调度下也出现精度异常**（非异步独有）。
> 纪律：所有判断给代码依据；依据不足处明确标 ❓ 存疑、不推测下结论。
> 前序分析过程：[[20260624-233518-vllm-ascend-GDN动态投机精度异常-根因定位与修复方案-分析]]（含已被证伪的早期假设，留作过程记录）

---

## 〇、根因定位的确定性边界（诚实分层，先讲清楚"哪些确定、哪些存疑"）

> 经两轮校正，前两版假设（"同 step 列宽错配""异步乐观接受"）均被用户约束证伪。本版只陈述**有代码实锤的事实**，根因的最终单点定位**部分仍存疑**，如实标注。

### ✅ 已坐实的代码事实

1. **hybrid mamba state 在投机后走 "align" 模式的 GPU 重排 kernel**：`postprocess_mamba_align_gpu`（[gpu_model_runner.py:1520](vllm/vllm/v1/worker/gpu_model_runner.py)），当 `mamba_cache_mode == "align"`（[cache.py:37/139](vllm/vllm/config/cache.py)，"only cache the mamba state of the last token of each scheduler step"）。
2. **该 kernel 的 conv_state 重排同时依赖 `num_draft`(投机长度) + `num_accepted`(接受数)**（[mamba_utils.py:82-93](vllm/vllm/v1/worker/mamba_utils.py)）：
   ```python
   num_tokens_running_state = num_computed + num_scheduled - num_draft   # :82
   new_num_computed = num_tokens_running_state + num_accepted - 1        # :83
   aligned_new_computed = (new_num_computed // block_size) * block_size  # :84
   accept_token_bias = aligned_new_computed - num_tokens_running_state   # :92
   dest_block_idx = aligned_new_computed // block_size - 1               # :93
   ```
3. **conv_state 按 `accept_token_bias` 做偏移拷贝**（mamba_utils.py:119/131）：源偏移 `accept_token_bias * inner * elem`，拷贝量 `(conv_width - accept_token_bias) * inner`。**bias 算错 → conv_state 滑窗错位 → 精度异常**。
4. **`num_draft` 是当前 step 真实动态值**（`draft_len = len(draft_token_ids)`，[gpu_model_runner.py:2159](vllm/vllm/v1/worker/gpu_model_runner.py)，来自 scheduler 当前 step 实际调度）—— **这条排除了"num_draft 用了旧/固定值"的假设**。
5. **同步调度也跑这个 kernel**：`postprocess_mamba_align_gpu` 的调用不依赖 `use_async_scheduling`（gpu_model_runner.py:1520 在 `mamba_cache_mode=="align"` 分支，与同步/异步无关）→ **与用户"同步也出错"吻合**。

### ❓ 仍存疑（无充分单点代码证据，不下结论）

- **A. `block_size` 在 align 模式的取值与跨 step 切换的交互**：`aligned_new_computed = (new_num_computed // block_size) * block_size`（mamba_utils.py:84）做了 **block 对齐截断**。mtp3→mtp1 切换时 `num_draft` 从 3→1，`num_tokens_running_state` 与 `num_accepted` 的组合是否会让 `aligned_new_computed` 落到与上一步 state 物理布局不一致的 block？**这一步的对齐截断是最可疑点，但需要构造具体数值/单步追踪才能坐实是否越界，本次静态走读不足以下定论。** ❓
- **B. `needs_copy = aligned_new_computed >= num_tokens_running_state`（mamba_utils.py:88）的边界**：当 `num_accepted` 小、`num_draft` 大时（mtp3 大量被拒），`new_num_computed` 可能 < `num_tokens_running_state`，`needs_copy=False` 直接 return（:90）——**此时 conv_state 不拷贝，但 state 是否已停在正确位置？跨 step 切换时这个 skip 是否遗漏了必要的 state 对齐？** ❓ 需动态验证。
- **C. mamba_cache_mode 实际取值**：用户环境是 `"align"` 还是 `"all"`/`"none"`？不同 mode 走完全不同的 state 管理路径（align 走上述 kernel，all 走全缓存）。**根因分析仅在 align 模式成立**，需先确认实际配置。❓

> **结论**：**已坐实 mamba align kernel 的 conv_state 重排是同步调度下投机精度的强相关点**（bias/对齐/skip 三处都对 num_draft+num_accepted 的组合敏感），但**"mtp3→mtp1 切换必然在此算错"的最终单点定位仍需动态验证（A/B），不静态臆断**。下面方案在此诚实前提下给出——既覆盖已坐实点，也对存疑点设防。

---

## 一、问题1：SGLang ping-pong 的实现细节，如何兼容（异步/同步）调度且无精度异常

> 这是回答"修复点 2"的参照实现。SGLang 用 **intermediate（影子）caches + verify 后按真实接受步提交** 解决，代码实锤如下。

### 1.1 核心机制：投机阶段写影子，verify 后提交真实接受步

`commit_mamba_states_after_verify`（[spec_utils.py:577](sglang/python/sglang/srt/speculative/spec_utils.py)），docstring 直述设计（:8-15）：
> "states in intermediate caches instead of advancing the persistent conv/ssm caches. After acceptance, the state of each request's last accepted step is committed back, plus the interval-crossing state."

拆解：
1. **投机阶段**：draft 的每一步 mamba state 推进，**写到 intermediate（影子）cache，不动持久 conv/ssm cache**。即持久 state 在 verify 前保持"上一轮确认值"不变。
2. **verify 后提交**：取每个请求**最后接受步**的影子 state 提交回持久 cache：
   ```python
   accept_index[req_idx, (accept_lens - 1).to(torch.int64)]   # :616 第 (accept_len-1) 步的影子 state
   ```
   `accept_lens` 是该请求真实接受长度 → **只提交"真实接受到第 k 步"的 state，丢弃第 k 步之后的所有投机推进**。
3. **interval-crossing 处理**（:631-650）：若 verify 后 `seq_lens` 跨过 mamba track interval（`seq_lens_pre // interval != seq_lens_post // interval`），在跨界点额外提交 state（:637 `accept_index[req_idx, to_track_ith] - offset`）——保证跨 block 边界时 state 落到正确 track 点。

### 1.2 双 buffer（ping-pong）容量

`HybridMambaDecodeReqToTokenPool`（[decode.py:183](sglang/python/sglang/srt/disaggregation/decode.py)）：
- `mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1`（[decode.py:209](sglang/python/sglang/srt/disaggregation/decode.py)）—— overlap（异步）时用 2 个 track buffer（一个推进、一个保留），非 overlap 用 1 个。
- `enable_mamba_extra_buffer`（:210）控制是否额外开影子 slot。

### 1.3 为什么能兼容调度模式 + 无精度异常

- **持久 state 永远只前进"真实接受步数"**：投机推进全在影子上，持久 cache 只在 verify 后被"第 k 步影子"覆盖。无论投机长度怎么变（mtp3/mtp1）、无论同步异步，**持久 state 的推进步数 = 真实接受数，恒不变**。
- **投机长度切换无副作用**：影子 buffer 按最大投机长度开，mtp1 只用前 1 个 slot、mtp3 用前 3 个，提交时按 `accept_lens-1` 取对应 slot——**长度变化只影响"用几个影子 slot"，不影响持久 state 的正确性**。
- **block 跨界单独处理**（interval-crossing）：避免了 vLLM align kernel 里 `aligned_new_computed // block_size` 那种"对齐截断可能错位"的风险（对应本文存疑 A）。

> **判断**：SGLang 的设计把"投机的不确定性"完全隔离在影子 cache，持久 state 只接受"已确认"的更新，且显式处理 block 跨界——这是 hybrid+投机的**结构性正确解**，从设计上消除了精度风险，而非靠 kernel 内的 bias 计算去"算对"。

---

## 二、问题2 已纳入：同步调度下也出错 → 根因不在异步乐观接受

用户确认同步也出错，已据此校正：根因**不是**前一版的"异步乐观接受"，而在 **align kernel 的 conv_state 重排（§0 已坐实）**。同步调度下 `num_accepted = (output_token_ids != -1).sum()`（gpu_model_runner.py:1513）是当前 step 真实值，kernel 仍按 §0.2 的 bias 公式重排——**这个重排逻辑本身对"投机长度跨 step 切换"是否完全鲁棒，是存疑 A/B 的核心**。

---

## 三、问题4：最终完整方案设计

> 分四档：① 立即可验证根因的低成本动作 ② 已坐实点的修复 ③ 存疑点的设防 ④ 结构性最优解（移植 SGLang ping-pong）。按"先证实根因、再对症修复"推进，符合不臆断原则。

### 阶段 0：先坐实根因（必做，决定后续方向）

| 动作 | 目的 | 依据 |
|------|------|------|
| **确认 `mamba_cache_mode` 实际取值** | align/all/none 走不同路径，根因仅 align 成立 | 存疑 C |
| **构造 mtp3→mtp1 切换的单步追踪**：打印 `num_draft / num_accepted / num_tokens_running_state / aligned_new_computed / accept_token_bias / dest_block_idx`（mamba_utils.py:82-93） | 坐实切换 step 这些值是否落到错误 block | 存疑 A/B |
| **对照实验**：固定 mtp3 不切换 vs 切换 mtp3→mtp1，对比 conv_state 拷贝前后数值 | 定位是否 bias/对齐/skip 出错 | §0.2/0.3 |

> 这一步把存疑 A/B/C 转成确定结论，是"修改能完全解决"的前提——不跳过。

### 阶段 1：修复已坐实点（align kernel 的 state 重排鲁棒性）

**针对存疑 A（block 对齐截断）**：若阶段 0 确认 `aligned_new_computed` 在切换 step 落错 block，则修复 mamba_utils.py 的对齐逻辑——保证 `dest_block_idx`（:93）始终指向当前 step 真实 state 应在的 block，而非被 `// block_size` 截断到上一步长度对应的 block。

**针对存疑 B（needs_copy skip）**：若确认 mtp3 大量被拒时 `needs_copy=False`（:88）遗漏了必要的 state 对齐，则补充"skip 路径下 state 已在正确位置"的保证或显式对齐。

### 阶段 2：对存疑点设防（不依赖根因 100% 定位也能拦截）

**数值断言（防御性，立即可加）**：在 `postprocess_mamba_align_gpu` 调用前后加断言——
```python
# state 推进后，断言 conv_state 的有效窗口长度 == 真实接受后应有的窗口
# 以及 dest_block_idx 不越过该 req 已分配的 mamba block 范围
assert (dest_block_idx >= 0).all() and (dest_block_idx < num_mamba_blocks_per_req).all()
```
切换 step 一旦越界立即抛错暴露（修复前必触发，修复后恒不触发）—— 这是把"存疑"转成"可观测"的抓手。

### 阶段 3：结构性最优解 —— 移植 SGLang ping-pong（推荐的彻底方案）

若阶段 1 的 kernel 级修复不足以覆盖所有 block 跨界 case（align 模式的对齐截断本质脆弱），则采用 **SGLang 式 intermediate cache + verify 后提交**：

1. **新增影子 state buffer**：mamba conv/ssm 各开 `max_spec` 份 intermediate slot（对标 `enable_mamba_extra_buffer`，decode.py:210）。
2. **投机推进写影子**：draft 各步 state 写 intermediate，持久 cache 不动。
3. **verify 后按真实接受步提交**：取第 `num_accepted-1` 步影子覆盖持久 cache（对标 spec_utils.py:616）。
4. **显式处理 block 跨界**：移植 interval-crossing 逻辑（spec_utils.py:631-650），替代 align kernel 的 `// block_size` 隐式对齐。

**为什么这是彻底解**：持久 state 只前进真实接受步数、且 block 跨界显式处理 → **投机长度怎么跨 step 切换都不影响持久 state 正确性**（§1.3 论证）。代价：每层 max_spec 份影子 state 显存。

### 阶段 4：vllm-ascend NPU 适配

vllm-ascend 的 GDN 走 `recurrent_gated_delta_rule`（[csrc/attention/recurrent_gated_delta_rule/](vllm-ascend/csrc/attention/recurrent_gated_delta_rule/)）+ `fused_gdn_gating`。需确认：
- NPU 是否也走 `mamba_cache_mode=="align"` 的 `postprocess_mamba_align_gpu`，还是 NPU 另有 state 提交路径；
- 若阶段 1/3 修改了 state 提交逻辑，NPU 侧的等价算子/调用需同步。
- ❓ 本次未核对 NPU 的 mamba state 提交路径，需专项走读 `recurrent_gated_delta_rule.cpp` 的 state 更新与上层 align kernel 的关系。

---

## 四、修改清单汇总（按"适配/改变量/修 bug/新模块"分类）

| 类别 | 内容 | 阶段 | 依据/状态 |
|------|------|------|----------|
| **先证实** | 确认 mamba_cache_mode + 单步追踪 bias/对齐 | 0 | 必做，转化存疑 A/B/C |
| **修 bug** | align kernel `aligned_new_computed`/`dest_block_idx` 跨 step 对齐 | 1 | ✅ 坐实敏感点，待阶段0确认 |
| **修 bug** | `needs_copy` skip 路径的 state 对齐遗漏 | 1 | ❓ 待阶段0确认 |
| **改特性（防御）** | postprocess 前后加 dest_block 越界断言 | 2 | 立即可加 |
| **新模块** | mamba 影子 state buffer + verify 后提交（移植 SGLang ping-pong） | 3 | 结构性最优解 |
| **新模块** | block 跨界（interval-crossing）显式提交 | 3 | 对标 spec_utils.py:631 |
| **适配** | vllm-ascend NPU GDN state 提交路径同步 | 4 | ❓ 待核对 NPU 算子 |

---

## 五、待核实/存疑（保留，不推测）

1. **存疑 A**：align kernel `aligned_new_computed` 的 block 对齐截断在 mtp3→mtp1 切换时是否落错 block —— 需阶段 0 单步追踪坐实。
2. **存疑 B**：`needs_copy=False` skip 路径在切换 step 是否遗漏 state 对齐 —— 需动态验证。
3. **存疑 C**：用户环境 `mamba_cache_mode` 实际取值（align/all/none）—— 根因仅 align 成立，需先确认。
4. **NPU 路径**：vllm-ascend 的 GDN state 提交是否复用 `postprocess_mamba_align_gpu`，未核对。
5. **`mamba_track_interval` 在 vLLM 侧是否存在等价物**：SGLang 有 interval-crossing 处理，vLLM align kernel 用 `block_size` 对齐——二者是否等价、vLLM 是否缺失跨界处理，是存疑 A 的关联点，未坐实。
