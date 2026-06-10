# vllm-ascend 面向 DeepSeek V4-Flash/Pro 的「Runtime+算子」协同极致时延优化方案设计

> 生成时间：2026-06-11
> 范围：方案设计（不含代码修改）。参考 TokenSpeed、TileRT 的 runtime+算子协同优化设计，结合 vllm-ascend 现状，给出收益分析、可行性分析与分阶段整体方案。
> 证据来源：本地代码仓 `tokenspeed/`、`TileRT/`、`vllm-ascend/`、`ascend_tilert_loop/`（均为 2026-06 快照），关键结论均附源码出处。

---

## 0. TL;DR

1. **问题本质**：batch=1（或极小 batch）+ MTP 的极致时延 decode 场景下，单步设备计算时间被算子优化和投机解码不断压低（目标 TPOT ≈ 2~3ms 量级），此时 Python 框架的每步动态开销（调度、输入准备、图参数更新、采样、MTP 草稿循环、D2H 同步）从「噪声」变成「主要矛盾」——这正是 TileRT/TokenSpeed 做 runtime+算子协同的底层逻辑。
2. **参考工作的共同设计模式**：① 把每 token 的 host 参与压缩到接近零（TileRT 整个 decode step 是一次 host 调用）；② 采样与 MTP 接受判定下沉设备侧，host 只在请求边界介入；③ 控制面用 C++/编译期状态机替代 Python 动态逻辑（TokenSpeed FSM + ExecutionPlan）；④ 算子按「norm+matmul」「matmul+通信」粒度融合，通信用 P2P 低时延缓冲而非通用集合通信；⑤ 用「接受长度驱动」语义保证 MTP 正确性的同时最小化同步点。
3. **vllm-ascend 现状**：DeepSeek V4 路径已具备相当强的算子侧协同（DSA prolog 三段 Cube/Vector 双流并行、mHC 融合算子、mc2 通算融合、ACL 全图捕获 + 图参数原位更新、async scheduling、merged draft）；同时已有 `xlite` C++ 整图 runtime，但**只支持 MHA 类架构，不支持 MLA/DSA，且与投机解码互斥**——这是面向 DeepSeek V4 的最大结构性缺口。
4. **推荐方案**：双轨四阶段。短期在 vLLM 体系内做「decode 全闭环」（残余 D2H 清零 → 图内采样 → MTP 验证设备侧闭环）；中期扩展 xlite 支持 DSA/MoE-EP/MTP，把 DeepSeek V4 decode 整步降为 1~2 次 host 调用；远期面向专用极低时延部署做 TileRT 式 persistent super-kernel 模式。预期 host 残余开销从 ~2-4ms/step 压至 <0.3ms/step，按 MTP 均接受 2.7、设备侧 6ms/step 估算，**端到端 TPOT 收益约 25%~40%**（需 Phase 0 profiling 校准，详见 §4）。

---

## 1. 背景与问题定义

### 1.1 极致时延场景

目标场景：单请求或极小并发的 decode 主导工作负载（agentic 工具链、AI 编程助手、实时决策），优化目标是 **TPOT（time per output token）/ 单请求 tokens/s**，而非吞吐。对应 `ascend_tilert_loop/agent_prompt.md` 中已立项的目标：

- 模型：DeepSeek-V3.2/V4 系（W8A8/W4A8）+ MTP，Ascend A3 单机
- 主指标：`tokens_per_sec`（目标 ≥400）、`tpot_ms`、`p99_tpot_ms`
- 体检指标：`d2h_sync_count == 0`（热路径）、`graph_hit_rate → 1.0`、MTP `accept_len_mean` 不劣化

vllm-ascend 的 DeepSeek V4 已有 e2e 用例 `gdydems/DeepSeek-V4-Flash-w4a8-mtp`（4 卡，`tests/e2e/pull_request/four_card/test_deepseek_v4.py`）。V4-Pro 与 Flash 同架构族（DSA + mHC + MoE + NextN/MTP），规模与量化格式不同（Flash 为 MXFP4/W4A8 专家，Pro 为更大规模），本方案以 Flash@4 卡为主线，Pro 视作同方案的规模放大（注意 Pro 卡数更多时通信项权重上升，§6 Phase 3 通信融合优先级提高）。

### 1.2 单步时延的分解模型

对每个 decode 迭代（一次 MTP verify step）：

```
T_step = T_host_exposed + T_device
T_host_exposed = T_host_total − T_overlapped(被设备执行掩盖的部分)
有效 TPOT = T_step / E[accept_len]
```

关键推论（runtime+算子协同的「底层逻辑」）：

- **算子优化和 MTP 压低 `T_device`，会放大 `T_host_exposed` 的相对占比**。设备侧从 20ms 优化到 6ms 后，原本可忽略的 2ms host 开销变成 25% 的税。
- **MTP 进一步放大 host 开销**：每个 verify step 包含 target 前向 + 多步 draft 前向，每次前向都要走一遍「元数据构建 → 图参数更新 → 图下发」，host 工作量按图下发次数线性放大；而接受长度判定又引入 D2H 同步点，阻断流水。
- 因此**先做 runtime（缩 host），再做算子（缩 device），再回头做 runtime**是个互为放大器的迭代循环——两边必须协同设计，单边优化收益会快速饱和。

---

## 2. 参考工作的底层逻辑拆解

### 2.1 TileRT：编译器驱动的 tile 级持久 runtime

定位（`TileRT/README.md`）：牺牲通用 serving 能力，换取「百亿/万亿参数模型毫秒级 TPOT」。实测：DeepSeek-V3.2 ~600 tok/s、GLM-5 ~500 tok/s（8×B200），与小米合作在 1T 参数模型上突破 1000 TPS；MTP 平均接受长度 ~2.77。

核心机制（源码证据）：

1. **整个 decode step = 一次 host 调用**。`tilert/models/deepseek_v3_2/modules/end2end.py:557`：`forward()` 仅执行 `dsa_show_hands(token_id.cpu())`，一个 torch custom op 驱动 8 卡完成全部 61 层 + 采样 + MTP 草稿与验证。host 每 token 的参与就是「喂 1 个 token id、读设备侧缓冲区里的接受结果」（`Idx.ACCEPTED_TOKENS`、`Idx.NEXT_DRAFT_TOKENS`、`Idx.PREDICTED_TOKENS`）。
2. **算子按「前后融合 + 通信入核」粒度组织**。`tilert/models/deepseek_v3_2/ops/` 的算子命名直接暴露融合策略：`rmsnorm_projq_wqb`、`rmsnorm_up_gate_silu`（norm 融进 matmul 前端）、`unproj_o_allreduce`、`down_allreduce`、`eh_proj_allreduce`、`expert_down_allreduce`（allreduce 融进 matmul 后端）、`expert_sel_up_gate_silu`（专家选择与专家计算融合）。通信不是独立的 NCCL 调用，而是 kernel 内通过预交换的 **P2P 低时延缓冲**直接写对端 HBM（`end2end.py:491-501`：`ll_buf.data_ptr()` 跨卡交换进 `peer_bufs`）。
3. **资源全静态化**：权重按卡预切分（`*_dev_{0..7}` 离线转换）、中间变量连续大块预分配（`generate_params_with_continuous_storage`，1KB 对齐）、固定 `forward_max_seq_len=4`（=MTP 宽度）。
4. **采样进图**：温度/top-p/top-k/种子放在设备侧常驻张量（`Idx.SAMPLING_CONFIG`），换采样参数要**重新捕获图**（`update_sampling_config` 打印 "Recapturing CUDA graphs"）——这是把动态性换确定性的典型取舍。
5. **代价清单**：每个模型一个预编译后端 `.so`（`libtilert_dsv32.so`），一进程只能跑一个模型；无 continuous batching、无 prefix cache、采样配置变更代价高；环境 ABI 强绑定。

**对本方案的启示**：TileRT 是「runtime 开销归零」的极限参照系，但它是一个**专用推理器**而非 serving 引擎。vllm-ascend 不应整体照搬，而应把「decode 热循环单调用化 + 采样/MTP 设备闭环 + 通信入核」作为可分阶段吸收的机制，保留 vLLM 的服务面。

### 2.2 TokenSpeed：C++ 控制面 + 类型化 FSM + 接受长度驱动

定位（`tokenspeed/README.md`）：面向 agentic 负载的生产级引擎，"C++ control plane and Python execution plane"，请求生命周期/KV 所有权/overlap 时机全部编码为有限状态机，编译期由类型系统保证 KV 资源复用安全。已官方支持 **DeepSeek V4-Flash**（`docs/serving/deepseek-v4.md`）。

核心机制（源码证据）：

1. **调度器整体 C++ 化**（`tokenspeed-scheduler/csrc/`）：`Scheduler::NextExecutionPlan()` 产出操作列表（Prefill/Decode/Prefetch/WriteBack/Retract Operation），Python 执行面消费后通过 `Advance(ExecutionEvent)` 回灌状态；事件按 `fsm/forward_events`、`cache_events`、`pd_events` 分域。Python 只做张量执行，所有每步决策逻辑（含 prefix cache 哈希匹配、页分配模拟）都在 C++。
2. **V4 的多缓存组调度**：V4 注意力携带 4 个独立 paged cache group（SWA / compressed KV / compressor state / CSA indexer state），调度器原生管理 compact group block table（`logical_page - base_offset` 语义）——说明**面向 DSA 类架构，KV 管理本身就是 runtime 协同设计的一部分**，不能只靠通用 block table。
3. **MTP 的「接受长度驱动」语义**（`docs/serving/deepseek-v4.md`）："MTP advances each request by the sampled accepted length, not by a fixed verify width. The next target-verify step must be scheduled only after the scheduler has observed the previous accepted length"。同时明确：草稿多步解码的元数据必须从 `valid_cache_len + accept_len` 推进，禁止草稿读到被拒绝的 verify-tail KV。这是 MTP 正确性的硬约束，也解释了为什么「接受长度」成为天然同步点——优化方向只能是**把这个判定也搬到设备侧**，而非取消它。
4. **算子侧配套**：mega_moe（DeepGEMM `fp8_fp4_mega_moe` 融合专家）、MXFP4 indexer cache、tilelang fast mHC 融合内核、`flash_mla` 稀疏 decode——与调度器同仓演进，kernel 经由统一 registry（`register_kernel`）接入，runtime 不直接依赖第三方 kernel 库。
5. **overlap 的边界管理**：spec decode + paged cache group 同时启用时主动**禁用 overlap 调度**作为 phase-1 边界——印证「正确性边界明确的分阶段交付」比一步到位激进 overlap 更工程可行。

**对本方案的启示**：vLLM 的 Python 调度面短期不可替换，但 TokenSpeed 证明了两件事可移植：① 每步动态决策可以离开 Python 热路径；② MTP 的元数据推进规则可以做成确定性状态机，从而允许「按最大宽度预排、设备侧按实际接受长度修正」的低同步方案。

### 2.3 共同设计模式总结

| # | 模式 | TileRT | TokenSpeed | 对 vllm-ascend 的映射 |
|---|------|--------|------------|----------------------|
| 1 | host 每 token 参与最小化 | 整步单调用 | C++ ExecutionPlan 批量下发 | decode 全图 + 设备侧闭环（§6 P1/P2） |
| 2 | 采样/MTP 判定设备化 | 完全进图 | 接受长度驱动、调度器只读结果 | 图内 sampler + 设备侧 verify（§6 P1.3/P2） |
| 3 | 控制面去 Python 化 | 无控制面（单请求） | C++ FSM 调度器 | xlite 扩展 / C++ 元数据构建器（§6 P2） |
| 4 | 算子融合到「norm+mm」「mm+通信」粒度 | ops 命名即融合表 | mega_moe/mHC tilelang | 已有部分（hc_pre/hc_post、mc2），补 DSA decode 链路（§6 P3） |
| 5 | 通信用 P2P 低时延缓冲 | ll_buf 直写对端 HBM | — | HCCS P2P / mc2 深化（§6 P3） |
| 6 | 资源静态化（shape 桶、连续工作区、权重预布局） | 全静态 | scheduler 模拟分配 | ACL graph capture sizes + workspace 已有，补 MTP 桶（§6 P1.2） |

---

## 3. vllm-ascend DeepSeek V4 现状盘点

### 3.1 已有资产（家底比预期厚）

**模型与算子层**（`vllm_ascend/models/deepseek_v4.py`、`vllm_ascend/attention/dsa_v1.py`、`vllm_ascend/ops/`）：

- DeepSeek V4 全架构支持：DSA（Indexer + Compressor + 稀疏 MLA）、mHC 超连接（自定义融合算子 `npu_hc_pre`/`npu_hc_post`，deepseek_v4.py:837-846）、MoE、MTP（`deepseek_v4_mtp.py`，与主模型共享 `topk_indices_buffer` 和 `_mtp_hidden_buffer` 稳定地址缓冲）。
- **算子级协同已经做得很深**：DSA prolog 三段式 Cube/Vector 双流并行（`_mla_prolog_multistream`，dsa_v1.py:1686+，q 量化/matmul 与 kv 量化/norm/rope/scatter 跨流交错，仅尾部一次 wait_stream）；融合算子 `npu_rms_norm_dynamic_quant`、`inplace_partial_rotary_mul`、`npu_transpose_quant_batchmatmul`、`dsa_kv_compress_scatter`；A5 路径的 MXFP4 动态量化 matmul。csrc 还有 `mla_preprocess`、`mc2`（通算融合）、`gmm` 等 AscendC 内核。
- 旁路优化：weight prefetch（`WeightPrefetchConfig`）、FlashComm1（SP，attention 内 allgather 延迟化，dsa_v1.py:1589+）、`prefill_comm_compute_overlap`、`multistream_dsv4_dsa_overlap`。

**图与运行时层**：

- ACL Graph 全图模式（`compilation/acl_graph.py`）：`ACLGraphWrapper` + 按 capture size 的 `GraphParams`（events/workspaces/handles/attn_params），replay 前经 `update_full_graph_params` 在独立 update_stream 上原位刷新 attention 参数（用 `ExternalEvent` 与上一轮 replay 串序，acl_graph.py:259-262 注释明确了该次序约束）。**draft 模型有独立的 graph params 体系**（`_draft_graph_params`、`_draft_graph_prefill_params`）。
- async scheduling 已接入（`use_async_scheduling`，model_runner_v1.py 多处），`execute_model`/`sample_tokens` 已拆分为状态机。
- spec decode 框架：`AscendSpecDecodeBaseProposer` 支持 merged draft（`_run_merged_draft`）与 `parallel_drafting`，draft 循环 `for draft_step in range(num_speculative_tokens)`（llm_base_proposer.py:544）。

**xlite：现成的 C++ 整图 runtime（关键资产 + 关键缺口）**（`vllm_ascend/xlite/`）：

- 形态：外部二进制组件 `xlite._C`（Runtime/Model/AttnMeta），`XliteWrapper` 替换 vLLM 模型对象，decode 时**整模型前向一次 C++ 调用**（`xlite_model.forward(rt, input_ids, attn_meta, kv_caches, freq_cis, h, stream)`，xlite.py:781），彻底绕开 Python 逐层 dispatch；支持 decode-only 与 full 两种模式。
- 缺口 ①：**架构白名单只有 MHA 类**（Llama/Qwen2/Qwen3/Qwen3-MoE/GLM4-MoE/MiniMax-M2，xlite.py:624-633），`attn_type` 仅 `AttnMHA`——**无 MLA，更无 DSA**，DeepSeek 全系不可用。
- 缺口 ②：**与投机解码硬互斥**（`ascend_config.py:567-570` 直接 raise），与 PP 互斥；xlite.py:751-757 的 TODO 注明 MTP 多 token 场景待解决。
- 缺口 ③：**每步在 host 构建 Python list 元数据**：`xlite_attn_metadata.block_tables_cpu = attn_metadata.block_tables.cpu().tolist()`（xlite.py:771）——热路径上有一次 D2H 同步 + tensor→list 转换，与极致时延目标直接冲突（对照 `ascend_tilert_loop` 的 `d2h_sync_count==0` 指标）。

### 3.2 decode 单步（MTP verify step）时间线解剖：残余动态开销清单

以 V4-Flash w4a8 + MTP(k 步) + ACL 全图 + async scheduling 为基准，host 侧每步仍要做：

| # | 开销项 | 位置 | 性质 |
|---|--------|------|------|
| H1 | EngineCore 调度（Python）：waiting/running 队列、token 预算、spec token 记账 | vLLM v1 scheduler | 每步，Python |
| H2 | scheduler_output 跨进程传递与（条件性）**deepcopy**：async sched + spec 且 `_draft_token_ids is None` 时整份 deepcopy | model_runner_v1.py:1933-1949（TODO 自注 "deepcopy is expensive"） | 每步，Python |
| H3 | `_update_states` + `_prepare_inputs`：持久 batch 状态更新、按请求 numpy 拼装、block table/slot mapping 写入、H2D 拷贝 | model_runner_v1.py:748+ | 每步，Python+numpy |
| H4 | DSA 元数据构建：prefill/decode 双段、**每层 cos/sin 字典**（`attn_metadata.cos[layer_name]`）、4 类缓存组的表 | dsa_v1.py:335-498 builder | 每步，量大 |
| H5 | `update_full_graph_params`：update_stream 上逐 handle 刷新 attn 参数 + event 编排 | acl_graph.py:283+ | 每图每步 |
| H6 | 图下发次数：target 1 次 + draft（merged 后 1 次，否则 k 次）+ 采样/拒绝采样 eager 小算子若干 | runner + proposer | 每步 ×(2~k+1) |
| H7 | 采样与接受判定的 **D2H 同步**：accepted length / sampled ids 回 host，才能驱动 H1 的下一步（TokenSpeed 文档同款硬约束） | sample_tokens → bookkeeping | 每步，同步点 |
| H8 | MTP 草稿输入重建：`propose_draft_token_ids`、draft 元数据按 accept 推进、`_mtp_hidden_buffer` copy_ 管理 | llm_base_proposer.py、deepseek_v4.py:1008+ | 每步 |
| H9 | 输出处理：detokenize（异步线程/进程）、stop 检查、流式返回 | OutputProcessor | 可异步化 |

设备侧的 runtime 相关损耗（不属于纯算子时间）：

| # | 开销项 | 说明 |
|---|--------|------|
| D1 | 图间空隙：target 图、draft 图、采样 eager 段之间的设备空闲 | host 下发节奏决定 |
| D2 | 通用集合通信 vs P2P：HCCL allreduce/a2a 的固定启动成本（小消息时延主导） | 对照 TileRT ll_buf |
| D3 | 未融合的小 Vector 算子缝隙（hc 链路外的 norm/cast/index 类） | 全图内仍存在调度气泡 |

### 3.3 现状结论

vllm-ascend 在「算子协同」维度已接近 TokenSpeed 的 kernel 侧水平（多流 CV 并行、量化融合、通算融合都有），**短板集中在 runtime 维度的最后一公里**：每步 Python 元数据工厂（H3/H4）、MTP 多图下发与接受判定同步（H6/H7/H8）、以及 xlite 这条「C++ 整图捷径」没有打通 DeepSeek（§3.1 缺口①②③）。这与 `ascend_tilert_loop/agent_prompt.md` 列的优先级（去 D2H → shape 桶/常驻缓冲 → FULL_DECODE_ONLY replay → DSA/MLA overlap → MC2 → sampler/MTP 设备化）完全对齐。

---

## 4. Runtime 优化收益分析

### 4.1 收益模型

```
有效 TPOT = (T_host_exposed + T_device_compute + T_device_gap) / E[accept_len]
```

runtime 优化作用于 `T_host_exposed`（H1~H9）与 `T_device_gap`（D1~D3）；算子优化作用于 `T_device_compute` 与 D3。收益相乘不相加。

### 4.2 量级假设（待 Phase 0 校准）

> ⚠️ 以下为基于代码路径结构与同类系统公开数据的**工程估计**，不是实测。Phase 0 的第一交付物就是把这张表换成 msprof/torch_npu profiler 实测值。

V4-Flash@4×A3、batch=1、MTP k=3、W4A8，假设设备纯计算 ~5-7ms/verify-step（与 `ascend_tilert_loop` 目标 tpot 2.39ms × accept 2.7 ≈ 6.5ms/step 自洽）：

| 形态 | host 暴露开销/step | 设备空隙/step | 估算 TPOT（accept=2.7） | 相对收益 |
|------|-------------------|---------------|------------------------|----------|
| F0 同步调度 + 分段图 | 4~8 ms | 0.5~1.5 ms | ~4.3-6 ms | 基线 |
| F1 async sched + ACL 全图（现状可达上限） | 1.5~3 ms | 0.3~1 ms | ~3-3.6 ms | ~25-40% |
| F2 decode 全闭环（残余 D2H 清零 + 图内采样 + MTP 设备 verify + 单/双图下发） | 0.3~0.8 ms | 0.2~0.5 ms | ~2.4-2.8 ms | F1 基础上再 ~15-25% |
| F3 xlite-DSA / persistent super-kernel（整步 1 次调用 + 通信入核） | <0.1 ms | <0.2 ms | ~2.1-2.4 ms | F2 基础上再 ~8-15% |

三个结构性判断（比具体数字更稳）：

1. **收益递减但門槛递增**：F0→F1 是配置与小改动收益（现有能力组合），F1→F2 是中等改造，F2→F3 是架构级投入。**F2 是 ROI 拐点**——它把 host 从「每步多次介入」降到「每步一次介入」，拿走大部分收益，且不牺牲 serving 形态。
2. **MTP 是 runtime 开销的放大器，也是 runtime 优化的放大器**：k=3 时每步最多 4 次前向。把 draft 循环从「k 次（元数据重建 + 图下发）」收敛为 merged/图内一次（H6/H8），收益按 k 放大；接受判定的 D2H（H7）从每步阻塞变为设备侧分支，消掉整条流水线的去同步点。
3. **算子侧每省 1ms，runtime 侧收益权重 +15%**：协同迭代的排序应为「先 F2（runtime），同时算子继续压 T_device，再评估 F3 是否值得」。若 Pro 模型上线（更多卡、更重通信），D2 项权重上升，F3 中「通信入核」子项独立提前。

### 4.3 反面清单（runtime 优化不解决什么）

- prefill/TTFT：本方案聚焦 decode；TTFT 由 chunked prefill 与 prefill 算子决定。
- accept_len 本身：那是 speculators/draft 训练的事（`speculators/` 仓），但 runtime 必须保证优化不劣化 accept_len（验收红线）。
- 大 batch 吞吐：F2/F3 的设计点在 batch≤8；大 batch 下 host 开销天然被摊薄，现有路径已够。

---

## 5. 可行性分析

### 5.1 路线 A：vLLM 体系内深化（→ F2）

做法：不引入新 runtime，在现有 ACL 全图 + async scheduling 上补齐 decode 闭环：

1. 热路径 D2H 清零、DSA 元数据构建增量化/张量化（消 H3/H4 大头，cos/sin 改预计算索引）；
2. 采样进图（贪心/固定 top-k 先行，参照 TileRT 的「采样配置常驻设备张量」方案规避重捕获）；
3. MTP verify 设备闭环：target 图 + merged draft 图固定两次下发，接受长度在设备侧算出并直接驱动 draft 输入（H7 的 D2H 移到步尾与下一步 H1 重叠，借 TokenSpeed 的「按最大宽度预排 + 接受长度修正」语义）；
4. scheduler_output deepcopy 移除（H2，上游 TODO 已挂账）。

- 可行性：**高**。全部在 `vllm_ascend/` 允许路径内（与 `ascend_tilert_loop` 的 allowlist 一致），不依赖外部组件源码。
- 风险：图内采样的灵活性边界（per-request 采样参数变化）；与 vLLM 上游版本演进的 patch 维护成本。
- 工程量：中。预计覆盖 H2/H3/H4/H6/H7/H8 的 70-80%。

### 5.2 路线 B：xlite 扩展 DeepSeek（→ F3 的 serving 兼容形态）

做法：为 xlite 增加 DSA/MLA 支持 + MoE-EP + MTP in-runtime verify，并修掉接入层缺口③（AttnMeta 设备化）。

- 依据：xlite 接入层已验证「C++ 整图 + vLLM serving 共存」模式（decode-only 模式下 prefill 走 Python、decode 走 C++，xlite.py:741-748 dispatch 逻辑现成）；GLM4-MoE/MiniMax-M2 适配器证明 MoE-EP 已在 xlite 内核中存在。
- 缺口与依赖：**xlite._C 源码不在本仓**（二进制组件），DSA（indexer+compressor+稀疏注意力+4 缓存组）和 MTP 循环须由 xlite 所属团队实现或开放共建——这是组织依赖，不是技术不可行。MTP 互斥限制（ascend_config.py:567）需要 xlite 内核支持多 token 输入后才能解除。
- 可行性：**中**（技术可行，取决于 xlite 演进路线是否纳入 MLA/DSA）；工程量：高；收益：F3 级且保留 serving。
- 建议姿势：vllm-ascend 侧先把**接口契约**做好（AttnMeta 设备化、MTP 元数据协议、4 缓存组注册），与路线 A 的产物复用同一套设备侧元数据，使 xlite 就绪后可即插即用。

### 5.3 路线 C：TileRT 式 persistent super-kernel 专用模式（→ F3 极限形态）

做法：面向 batch≤4 的专用部署，把 V4 decode 整步（含 MTP）编译为 AscendC super-kernel/任务图常驻执行，host 仅请求边界介入；通信走 HCCS P2P 直写（对照 TileRT ll_buf；A3 机内 HCCS 带宽与对端内存访问能力具备硬件基础，csrc/mc2 已是通算融合先例）。

- 可行性：**中低**（CANN 上等价物需验证：超大融合核的 Cube/Vector 任务编排、跨卡同步原语、编译期资源静态化工具链——`sgl-kernel-npu`、`ops-transformer`、CATLASS 可作积木）；工程量：很高；且与 serving 特性互斥面大（同 TileRT 的代价清单）。
- 定位：不进 vllm-ascend 主线，作为独立「ultra-low-latency 模式」长线孵化，需求触发（如 Pro 模型 + 高频交易类客户）再立项。`ascend_tilert_loop` 脚手架即为此路线的实验载体。

### 5.4 对比矩阵

| 维度 | A：vLLM 内深化 | B：xlite-DSA | C：persistent super-kernel |
|------|---------------|--------------|---------------------------|
| 目标形态 | F2 | F3（serving 兼容） | F3（专用极限） |
| host 残余/step | 0.3~0.8ms | <0.1ms | ~0 |
| serving 特性保留 | 全部 | 大部分（PD/EPLB 需逐项验证） | 极少 |
| 外部依赖 | 无 | xlite 团队/源码 | CANN 深度特性 |
| MTP 兼容 | 是（设计核心） | 需 xlite 新能力 | 内嵌 |
| 工程量 | 中 | 高 | 很高 |
| 风险 | 低 | 中（组织依赖） | 高 |
| 建议 | **立即做** | **并行启动接口对齐，能力到位即切换** | 需求触发后孵化 |

---

## 6. 整体方案设计（推荐：双轨四阶段）

顶层设计：**主轨 = 路线 A → B 渐进（F1→F2→F3-serving）；副轨 = 路线 C 实验孵化（复用 ascend_tilert_loop）**。每阶段有独立验收，不通过不进下一阶段。

### Phase 0：基线度量与归因（1~2 周量级）

- 工具：msprof timeline + torch_npu profiler + `record_function`（execute_model 已有 "prepare input" 区段标注）。
- 产出：
  1. §3.2 表逐项实测（H1~H9、D1~D3），区分「暴露」与「被掩盖」；
  2. 每 verify-step 的图下发次数、D2H 同步次数（对齐 `d2h_sync_count`）、设备空闲率；
  3. 基线指标冻结：`tokens_per_sec / tpot_ms / p99_tpot_ms / accept_len_mean / GSM8K 快测分`（借用 TokenSpeed 的验收法：GSM8K 5-shot 50 样本，分数跌出基线 ±0.04 即判回归）。
- 验收：归因报告能解释 ≥90% 的 step 墙钟时间。

### Phase 1：host 残余开销压缩（路线 A 前半，→ F1+）

1.1 **热路径 D2H/H2D 清零**：xlite 接入层 `block_tables.cpu().tolist()` 同款问题全链路排查；DSA builder 的 per-layer cos/sin 字典改预计算 + 设备索引。
1.2 **shape 桶与常驻缓冲**：decode 按 MTP 宽度固定 bucket（1+k），杜绝 capture miss（`graph_hit_rate→1.0`）；MTP 草稿输入用稳定地址缓冲（`_mtp_hidden_buffer` 模式推广）。
1.3 **图内采样（第一档）**：贪心与固定参数 top-k/top-p 进 target 图，采样配置走设备常驻张量（TileRT `Idx.SAMPLING_CONFIG` 方案），动态 per-request 参数走慢路径回退。
1.4 **杂项**：H2 deepcopy 移除；`update_full_graph_params` 批量化（减少 per-handle host 调用）。
- 验收：host 暴露开销 ≤1.5ms/step；`d2h_sync_count==0`（步内）；精度快测不回归。

### Phase 2：decode 全闭环（路线 A 后半，→ F2）

2.1 **MTP 设备侧 verify 闭环**：拒绝采样/接受长度在设备侧产出；draft 输入由设备侧按接受结果直接构造（borrow TokenSpeed 语义：按最大 verify 宽度预排 KV slot，接受长度只做掩码与指针推进，保证「草稿不读被拒 KV」约束在 slot 预排时静态满足）；accepted ids 的 D2H 改异步、与下一步 H1 重叠。
2.2 **每步图下发收敛到 2 次**（target 全图 + merged draft 全图），探索进一步合并为 1 次（target 尾部直接级联 draft，需图间张量地址静态化——ACL graph workspace 机制已支持按 size 固定）。
2.3 **调度协议精简**：decode-only 稳态下 scheduler_output 走增量协议（只传 delta），为 xlite/C++ 元数据构建器预留同一接口契约。
- 验收：host 暴露 ≤0.5ms/step；TPOT 较 Phase 0 基线 ≥25% 改善；accept_len_mean 不降；GSM8K 快测在带内。

### Phase 3：协同深化（路线 B 切换 + 算子配套，→ F3-serving）

3.1 **xlite-DSA 接口对齐与切换**（组织依赖项提前谈）：AttnMeta 设备化协议、4 缓存组注册、MTP 多 token 协议；xlite 能力到位后 decode 切 C++ 整图，Phase 2 产物作为回退路径。
3.2 **通信入核**：机内 allreduce/a2a 的 P2P 低时延版本（小消息直写对端 HBM，mc2 经验外推），优先 o_proj 后 allreduce 与 MoE combine——V4-Pro 多卡场景此项权重最高。
3.3 **DSA decode 链路融合补全**：对照 TileRT ops 清单逐段核对（indexer top-k 与稀疏注意力衔接、`rmsnorm+wq_a/wkv` 前融合、`o_proj+allreduce` 后融合），多流 CV 并行从 prolog 推广到全链路。
- 验收：host 暴露 ≤0.1ms/step；TPOT 进入目标带（对照 `ascend_tilert_loop` goal：tokens/s ≥400 量级）。

### 副轨（与主轨并行）：persistent super-kernel 实验

- 载体：`ascend_tilert_loop` 迭代循环（agent 改 → A3 验证 → JSON 评分 → history），单层/单段先行（如「一层 DSA+MoE 的 super-kernel decode 微基准」），验证 CANN 上 tile 级任务编排与跨卡 P2P 原语的真实收益，再决定是否升格立项。

### 风险与开放问题

| 风险 | 影响 | 缓解 |
|------|------|------|
| 图内采样牺牲 per-request 采样灵活性 | 功能回退 | 双路径：固定参数走图，动态参数走 eager 慢路径 |
| 设备侧 verify 改动 MTP 语义引入精度回归 | 红线 | 每阶段 GSM8K 快测门禁 + accept_len 监控；遵守 TokenSpeed 文档列出的 KV 推进约束 |
| xlite 源码不可达/排期不可控 | Phase 3.1 延期 | 接口契约先行；Phase 2 产物自身即 F2 交付，不依赖 B |
| ACL graph 多 capture size 的显存（workspace/图缓存）膨胀 | OOM 风险 | decode 桶收敛到极少数 size；weak_ref workspace 机制已有 |
| 与 vLLM 上游演进冲突（async sched/spec 接口变动快，代码内多处 `vllm_version_is` 分支） | 维护成本 | 改动收敛在 vllm_ascend 层；关键诉求（deepcopy 移除等）推上游 |
| EPLB / PD 分离 / prefix cache 与全闭环 decode 的交互 | 功能矩阵复杂 | 学 TokenSpeed：显式声明 phase 边界（如 Phase 2 先限定非 PD、EPLB 静态） |

---

## 7. 附录：源码证据索引

| 主题 | 位置 |
|------|------|
| TileRT 整步单调用 / 设备侧采样与 MTP | `TileRT/tilert/models/deepseek_v3_2/modules/end2end.py:551-558, 253-311, 617-651` |
| TileRT 融合算子清单（norm+mm、mm+allreduce） | `TileRT/tilert/models/deepseek_v3_2/ops/*.py`（文件名即融合表） |
| TileRT P2P 低时延缓冲交换 | `end2end.py:394-401, 491-501` |
| TokenSpeed C++ 调度器 / FSM / ExecutionPlan | `tokenspeed/tokenspeed-scheduler/csrc/scheduler/scheduler.h:59-62`、`csrc/fsm/*` |
| TokenSpeed V4-Flash 服务配置与 MTP 语义约束 | `tokenspeed/docs/serving/deepseek-v4.md`（接受长度驱动、4 缓存组、GSM8K 验收法） |
| vllm-ascend DeepSeek V4 模型 / mHC 融合算子 | `vllm-ascend/vllm_ascend/models/deepseek_v4.py:837-870, 1008-1030` |
| DSA 多流 CV 并行 prolog | `vllm-ascend/vllm_ascend/attention/dsa_v1.py:1686+` |
| ACL 全图参数原位更新机制 | `vllm-ascend/vllm_ascend/compilation/acl_graph.py:283-330, 259-262` |
| async scheduling 与 deepcopy 痛点 | `vllm-ascend/vllm_ascend/worker/model_runner_v1.py:1933-1949` |
| MTP merged draft / 草稿循环 | `vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py:205, 434, 544` |
| xlite 整图调用与架构白名单 / MHA-only | `vllm-ascend/vllm_ascend/xlite/xlite.py:624-639, 736-803` |
| xlite 与 spec decode 互斥 | `vllm-ascend/vllm_ascend/ascend_config.py:559-582` |
| xlite 热路径 D2H+tolist 问题点 | `vllm-ascend/vllm_ascend/xlite/xlite.py:771` |
| V4-Flash e2e 用例（4 卡 w4a8 mtp） | `vllm-ascend/tests/e2e/pull_request/four_card/test_deepseek_v4.py:46` |
| ascend_tilert_loop 指标体系与优化优先级 | `ascend_tilert_loop/README.md`、`agent_prompt.md` |
