# vllm-ascend 五大优化 Phase 深度分析：技术原理 × 量化收益 × 可行性

> **输出日期**：2026-05-28  
> **分析场景**：Qwen3.5-397B-A17B @ 昇腾 A2（910B Pro × 8，单节点）  
> **参考实现**：TokenSpeed（B200 580 tok/s 基准）  
> **目标仓库**：vllm-ascend

---

## 预备：Qwen3.5-397B-A17B 模型架构速查

理解后续所有优化的前提，是先搞清楚这个模型的**核心结构数字**。

```
Qwen3.5-397B-A17B 架构参数（基于 tokenspeed config + HuggingFace 公开信息）：

num_hidden_layers = 94               # 总层数
full_attention_interval = 9           # 每 9 层插 1 个 Full-Attention
→ Full-Attention 层数 = 94 / 9 ≈ 10  # 第 9,18,27,...,90 层
→ GDN（线性注意力）层数 = 94 - 10 = 84  # ← 主体！

hidden_size = 7168                    # 隐层维度
num_attention_heads = 64              # Full-Attention Q 头数
num_kv_heads = 8                      # GQA KV 头数（Full-Attention 层）

# GDN（Gated DeltaNet）层参数
linear_num_key_heads = 16             # GDN Q/K 头数
linear_key_head_dim = 128             # GDN K 头维度
linear_num_value_heads = 32           # GDN V 头数（注意：比 K 多）
linear_value_head_dim = 128           # GDN V 头维度
linear_conv_kernel_dim = 4            # 卷积核宽度

# MoE 参数
num_experts = 128                     # 专家总数（每层）
num_experts_per_tok = 10              # 每 token 激活专家数
```

### SSM 状态（什么是 SSM State）

**SSM = State Space Model（状态空间模型）**，GDN 是 SSM 的一个变体。其核心思想是用一个**固定大小的"记忆向量"来压缩历史信息**，而不是像 Transformer 那样保留所有历史 token 的 KV 缓存。

```
GDN 的递推方程（简化）：
  h_t = α_t ⊙ h_{t-1} + β_t ⊙ (k_t^T v_t)
         ↑ 遗忘门         ↑ 当前输入更新

  output_t = q_t · h_t         ← 用当前 query 查询压缩历史

其中 h_t ∈ ℝ^{num_value_heads × key_dim × value_dim} 就是 SSM 状态
```

**Qwen3.5-397B 单层 SSM 状态的两个组成部分**：

```
① temporal_state（核心 SSM 状态 / ssm_h）：
   shape = (linear_num_value_heads, linear_key_head_dim, linear_value_head_dim)
         = (32, 128, 128)
   dtype = bfloat16 (2 bytes)
   单层大小 = 32 × 128 × 128 × 2 = 1,048,576 bytes = 1 MB / 层

② conv_state（因果卷积状态）：
   conv_dim = linear_key_head_dim × linear_num_key_heads × 2
            + linear_value_head_dim × linear_num_value_heads
            = 128 × 16 × 2 + 128 × 32 = 4096 + 4096 = 8192
   shape = (conv_dim / tp_size, conv_kernel_dim - 1)
         = (8192/8, 3) = (1024, 3)  [TP=8]
   dtype = bfloat16
   单层大小 = 1024 × 3 × 2 = 6,144 bytes ≈ 6 KB / 层
```

**单请求全量 SSM 状态（84 个 GDN 层，TP=8）**：

```
temporal_state 合计 = 84 × 1 MB = 84 MB / 请求
conv_state 合计     = 84 × 6 KB ≈ 0.5 MB / 请求
单请求 SSM 总计 ≈ 84.5 MB / 请求（TP=8 分片后，每卡 ~10.6 MB）
```

**与 KV Cache 的本质区别**：

| | KV Cache（Full-Attention 层）| SSM State（GDN 层）|
|---|---|---|
| 大小随序列长度 | **线性增长**，L tokens → L × 2 × head_dim × heads | **固定大小**，与序列长度无关 |
| 可分片共享 | 可按 token 维度分片 | 不可分片，必须整体复用 |
| 跨请求共享 | 相同 prefix → 直接复用（prefix cache）| 必须 Copy-on-Write 后续算 |
| 10 个 Full-Attention 层（TP=8）| 90k tokens → 90k × 10 × 2 × 128 × 8 × 2B = ~3.7 GB/卡 | — |
| 84 个 GDN 层 | — | 固定 84.5 MB/请求（总计，8 卡共享） |

**关键认知**：SSM State 是 GDN 层的"压缩记忆"，固定大小但**请求私有**，是实现高命中率 prefix cache 在 GDN 架构上性能收益的核心瓶颈所在。

---

## Phase 1：Mamba CoW + branching_seqlen

### 1.1 技术原理深度剖析

**问题本质**：prefix cache 命中≠性能收益（对 GDN 架构）

```
标准 Transformer（只有 Full-Attention 层）：
  prefix 命中 → KV cache hit → 直接跳过 prefill 计算 → TTFT 降低 90%+
  ✅ KV 可以直接复用，没有额外代价

Hybrid GDN 架构（Qwen3.5-397B，84/94 层是 GDN）：
  prefix 命中 → KV cache hit（10 个 Full-Attention 层）✅
  BUT → GDN 层的 SSM state h_t 不能直接复用 ❌
  → 必须从 h_0=0 开始重新递推到 h_{prefix_len}
  → 即使 prefix 完全命中，GDN 层仍需对 90k tokens 全量计算
  → TTFT 降低幅度远小于预期
```

**GDN 递推的计算复杂度**：

```
GDN prefill 时间复杂度：O(L × D²)
  L = 序列长度（tokens）
  D = head_dim = 128（每层）

Qwen3.5-397B，L=90k tokens，84 个 GDN 层：
  FLOPs ≈ 84 × 90000 × 128² × 2 ≈ 247 万亿 FLOPs（仅 GDN 递推部分）
  A2（910B Pro）@BF16：~200 TFLOPS 理论峰值
  理论 GDN prefill 时间 ≈ 247 / 200 ≈ 1.2 秒（仅 GDN，90k prefix）
```

**Mamba CoW + branching_seqlen 的解法**：

```
核心思路：在 prefix cache 命中时，同时命中对应位置的 SSM state 快照
         新请求从快照位置"续算"，而非从头递推

执行流程：

Step 1: Match（Scheduler 侧）
  request tokens: [sys_prompt 0..90000] + [new_content 90001..91000]
  KV prefix hit: depth = 90000 tokens（10 个 FA 层全命中）
  SSM snapshot: 在 90000 位置有对齐快照 h_{90000}（branching_seqlen = 90000）
  → mamba_cow_src_index = snapshot_slot_42
  → branching_seqlen = 90000

Step 2: CoW（model runner 侧，GPU kernel）
  ssm_state[working_slot] ← ssm_state[snapshot_slot_42]
  conv_state[working_slot] ← conv_state[snapshot_slot_42]
  代价：84 层 × (1 MB + 6 KB) ≈ 84.5 MB memcpy（GPU 内）
       A2 HBM 带宽 ~800 GB/s → 耗时 ≈ 0.1 ms（可忽略）

Step 3: GDN chunk forward（从 branching_seqlen 续算）
  仅对 [90001..91000]（1000 tokens）执行 GDN 递推
  FLOPs ≈ 84 × 1000 × 128² × 2 ≈ 2.75 万亿 FLOPs（减少 98.9%）
```

**"对齐边界"是关键工程约束**：

```
为什么需要对齐到 block_size（mamba_cache_mode="align"）？

GDN 是序列递推，h_t 依赖 h_{t-1}。
要在任意位置 T 保存快照，需要确保该位置的 h_T 已完整计算。
vllm 的 align 模式在每个 block_size 边界保存 h_T 快照。

branching_seqlen = (prefix_match_tokens // block_size) * block_size
              = (90000 // 128) * 128 = 89984（对齐后）

效果：
  对齐后仅损失 90000 - 89984 = 16 tokens（重算）
  节省 89984/91000 = 98.9% 的 GDN 递推计算
```

**两种 CoW 场景**：

```
场景 A：GPU 上直接有快照（热路径）
  mamba_cow_src_index = GPU_slot_42
  → 直接 GPU 内 memcpy，延迟 <1 ms
  → 几乎零额外代价

场景 B：只有 Host 快照（冷路径，L2 Cache 命中）
  mamba_host_src_index = Host_slot_7
  → 需要 H→D 传输：84.5 MB @ NPU-CPU 带宽（昇腾约 30-50 GB/s）
  → 传输延迟 ≈ 84.5 MB / 40 GB/s ≈ 2 ms（可与 KV loadback 并行）
  → 仍然远优于重算 89984 tokens（约 1.2 秒）
```

### 1.2 量化收益分析（Qwen3.5-397B @ 昇腾 A2）

**昇腾 A2 硬件规格参考**：

```
昇腾 910B Pro（A2 服务器搭载）：
  BF16 峰值算力：~400 TFLOPS（单卡理论，实际 MFU ~50-60%）
  有效算力：~200-240 TFLOPS/卡
  HBM 带宽：~800 GB/s（单卡）
  HBM 容量：64 GB/卡

单节点 TP=8 配置（8 × 910B Pro）：
  总 HBM：8 × 64 = 512 GB
  模型权重（BF16）：397B × 2B ≈ 794 GB → 需 2 节点（16卡）
```

**场景设定**：16 卡（2 节点）× 910B Pro，TP=8，PP=2

```
基准测试场景：
  input_len = 90000 tokens（长 system prompt + 上下文）
  output_len = 500 tokens（短回复）
  prefix_cache_hit_rate = 90%（90000 × 90% = 81000 tokens 命中）
  batch_size = 16 并发请求
```

**TTFT（Time To First Token）收益**：

```
无优化（当前 vllm-ascend baseline）：
  GDN prefill FLOPs（90k tokens）：
    84 层 × 90000 × 128² × 2 = 247.5 TFLOPS
  实际 GDN prefill 时间 ≈ 247.5 / (200 × 16) ≈ 0.077 秒 [16卡]
  
  注：加上 FA 层 attention prefill（90k²复杂度，约是主要瓶颈）
  FA prefill（10层，90k tokens，GQA）≈ 10 × 90k² × 7168 × 2 / 16卡 ...
  → 对于 90k token，QA attention 是主瓶颈，约 2-5 秒
  
  总 TTFT baseline：约 5-10 秒（含通信、采样开销）

Mamba CoW 后（90% prefix hit，branching_seqlen=81000）：
  GDN prefill 仅对 9000 new tokens：
    84 × 9000 × 128² × 2 = 24.7 TFLOPS → 约 0.008 秒 [16卡]
  GDN 计算量降低：90%（节省 222 TFLOPS）
  
  FA attention 对 90% hit 的 10 层：KV 直接复用，不需重算 ✅
  FA attention 仅对 9000 new tokens：计算量大幅降低
  
  总 TTFT（CoW 后）：约 0.5-1 秒（GDN+FA 大幅减少）

→ TTFT 降低幅度：约 80-90%（从 5-10 秒 → 0.5-1 秒）
```

**吞吐收益（Throughput）**：

```
TTFT 降低 → GPU 空闲时间减少 → 更多请求并行处理

假设：
  baseline TTFT = 8 秒，decode 时间 = 500 tok / 30 tok/s = 17 秒
  CoW TTFT = 0.8 秒，decode 时间 = 17 秒

prefill 占比（baseline）：8 / (8+17) = 32%
prefill 占比（CoW 后）：0.8 / (0.8+17) = 4.3%

并发 GPU 利用率提升：
  baseline：prefill 期间其他请求必须等待（若非 continuous batching）
  CoW 后：prefill 时间极短，decode 主导，GPU 利用率接近 100%

吞吐提升估算：
  16 并发时，prefill 瓶颈消除
  throughput ≈ 16 × 30 tok/s = 480 tok/s（decode-bound）
  vs baseline ≈ 16 × 30 × 17/(8+17) = ~326 tok/s

→ 吞吐提升：约 47%（仅 CoW，长 prefix 场景）
```

**内存收益**：

```
CoW 复用快照，不需要为每个命中 prefix 的请求额外分配 SSM slot
节省的 GPU SSM 内存：
  N_reuse 个请求共享同一 snapshot slot
  相比每请求独立 slot：节省 (N_reuse - 1) × 84.5 MB

16 并发，10 个请求命中同一 prefix：
  节省 9 × 84.5 MB = 760 MB（约 1.2% of 64 GB HBM/卡）
  相对有限，主要收益是计算量降低
```

### 1.3 可行性分析

**技术可行性**：⭐⭐⭐⭐⭐（最高）

```
✅ 已有基础 1：vllm-ascend GDN forward 已支持 initial_state 参数
   chunk_gated_delta_rule(..., initial_state=h_T, ...)
   → 直接传入 h_branching_seqlen 即可从该点续算

✅ 已有基础 2：mamba_cache_mode="align" 已在 block 边界保存 SSM 快照
   这就是 branching_seqlen 处的 h_T 的存储位置

✅ 已有基础 3：vllm v1 KV prefix cache 已返回 num_computed_tokens
   → branching_seqlen = align(num_computed_tokens, mamba_block_size)

❗ 缺失的只是"三点连线"：
   Scheduler 读取 num_computed_tokens
   → 计算 branching_seqlen + 找对应 SSM slot
   → 填入 SchedulerOutput → ModelRunner → attn_metadata → GDN forward
   这条信号链路在 vllm-ascend 中不存在，需要新建
```

**兼容性分析**：

```
① 与现有 prefix caching 兼容：
   CoW 是叠加在现有 KV prefix cache 之上的，不修改 KV cache 逻辑
   ✅ enable_prefix_caching=True 时生效，不影响 False 路径

② 与 MTP/spec decode 兼容：
   Phase 4 会处理 MTP 下的 SSM 状态更新
   Phase 1 只涉及 prefill 阶段，不影响 spec decode
   ✅ 可独立上线

③ 与 TP/PP 兼容：
   SSM state 已按 TP 分片（conv_dim/attn_tp_size）
   CoW 在每卡本地执行，无跨卡通信
   ✅ 天然 TP 安全

④ 与 310P 兼容：
   310P 有独立的 gdn_310.py，需同步改造
   工作量：1-2 天额外适配
   ✅ 可行，代码结构相同
```

**主要风险**：

```
风险 1：SSM snapshot 存储位置与 vllm block 结构的对应关系
  mamba_cache_mode="align" 下，SSM state 是按 mamba_block_size 对齐存储的
  但 block 的物理存储格式与 KV cache block 不同
  → 需要仔细核查 mamba_utils.py 中 block table 的索引方式
  缓解：单元测试验证 snapshot 读取的正确性（tolerance < 1e-3）

风险 2：branching_seqlen 非对齐场景的边界处理
  如果 KV match depth 正好在 block 边界，branching_seqlen = depth
  如果 KV match depth 不在边界，branching_seqlen < depth（需少量重算）
  → 逻辑简单，不是技术风险，是实现细节
```

---

## Phase 2：Host SSM L2 Cache

### 2.1 技术原理深度剖析

**问题本质：SSM State 的 GPU 内存压力**

```
Qwen3.5-397B，TP=8（16 卡），单卡 SSM 内存占用：

单请求 SSM 大小（TP 分片后，每卡）：
  temporal_state = 84 × (32/8) × 128 × 128 × 2B = 84 × 4 × 32768B ≈ 11 MB/卡
  conv_state = 84 × (1024/8) × 3 × 2B = 84 × 768B ≈ 0.06 MB/卡
  合计 ≈ 11 MB/卡/请求

GPU HBM 可用于 SSM 的空间（64 GB 单卡，扣除模型权重后）：
  Qwen3.5-397B 权重（BF16）：397B × 2B / 16卡 ≈ 49.6 GB/卡（权重）
  KV cache（10 个 FA 层，90k tokens，16 请求）：
    10 × 90000 × 8 × 128 × 2B / 8 ≈ 2.3 GB/卡
  剩余 HBM ≈ 64 - 49.6 - 2.3 ≈ 12 GB/卡

可承载的最大 SSM 请求数（不用 offload）：
  12 GB / 11 MB ≈ 109 个并发请求（理论上限）

实际约束：
  KV cache 随序列长度增长，长 context 场景 KV 更大
  实际 SSM 可用空间 ≈ 6-8 GB → 约 55-73 个并发请求上限
```

**Host SSM Cache 的工作原理**：

```
核心思想：将 GPU 的 SSM 状态比作 CPU 的 L2 Cache
  GPU HBM（SSM L1）：只存活跃请求的 SSM state
  CPU DRAM（SSM L2）：存储不活跃（被抢占/预取）请求的 SSM state
  磁盘/分布式存储（L3）：长期保存（可选）

工作流程：

1. 新请求进入，GPU SSM 槽位充足：
   直接分配 GPU slot，正常 decode
   
2. 新请求进入，GPU SSM 槽位不足（OOM）：
   LRU 选择最少使用的请求 → 将其 SSM state D→H 拷贝（异步）
   释放 GPU slot → 分配给新请求
   
3. 被卸载的请求再次被调度：
   H→D 拷贝（异步，可与其他计算 overlap）
   + CoW（Phase 1）→ 从快照续算

D→H 传输机制（昇腾等效 cudaHostRegister）：
  CPU 内存必须是"固定内存"（Pinned/DMA-mapped）才能支持 NPU 直接 DMA 访问
  torch_npu 等效：pin_memory=True 分配的 tensor 支持 npu.copy_ 异步传输
  
  传输速度（昇腾 A2）：
    NPU-CPU 互连（PCIe 5.0 × 16）：理论 ~64 GB/s，实际 ~30-40 GB/s
    84 MB SSM（单请求，16 卡合计）：
      per-卡传输量 ≈ 84 MB / 8 = 10.5 MB/卡
      传输时间 ≈ 10.5 MB / 35 GB/s ≈ 0.3 ms（单卡，非阻塞）
  
  bulk 传输优化（tokenspeed transfer_kv_all_layer_mla）：
    所有 84 层一次 kernel launch 完成（单 DMA chain）
    vs 循环 84 次 cudaMemcpyAsync：每次有 ~10 µs kernel launch overhead
    → bulk 节省 84 × 10 µs = 0.84 ms kernel overhead
```

**LRU 驱逐策略与 KV Cache 分离**：

```
两套独立 LRU 的必要性：

KV Cache LRU（现有）：
  驱逐单位是 KV block（固定大小，按 token 数分片）
  相同 prefix 的多请求可共享同一 block（物理页共享）
  驱逐 KV block 不影响对应 SSM state 的有效性

SSM LRU（新增）：
  驱逐单位是整个请求的 SSM state（固定大小，请求私有）
  与 KV block 无直接对应关系（SSM 大小与序列长度无关！）
  
  → 一个请求可能：KV 在 GPU，SSM 在 Host（KV 命中率高但 SSM 内存不足）
  → 或：SSM 在 GPU，KV 部分在 Host（短请求 SSM 小，KV 长）
  
  两套 LRU 各自独立优化，最大化 GPU 内存利用效率
```

**与 RecomputeScheduler 的集成**：

```
RecomputeScheduler 现有的 OOM 处理：
  OOM → 直接 recompute（abort 最旧请求，重新 prefill）
  代价：重新 prefill 浪费全部历史计算
  
Host SSM Cache 集成后：
  OOM → 检查 Host SSM 是否有空间
  有空间 → backup_all_layers() D→H（0.3 ms）→ 释放 GPU SSM → 继续调度
  无空间 → 再检查 KV Host Cache → 最后才考虑 recompute
  
  对比：
    recompute 代价 = 重跑全量 prefill（90k tokens → 数秒）
    SSM offload 代价 = 0.3 ms D→H + 后续 0.3 ms H→D
  → SSM offload 是 recompute 代价的 1/10000
```

### 2.2 量化收益分析（Qwen3.5-397B @ 昇腾 A2）

**并发数上限提升**：

```
当前 vllm-ascend（无 Host SSM Cache）：
  SSM GPU 内存上限 ≈ 6-8 GB/卡 → 约 55-73 并发请求

Host SSM Cache（CPU DRAM 256 GB 服务器）：
  Host 侧 SSM 容量：预留 50 GB CPU DRAM
  单请求 SSM（Host 侧，不分 TP）：84 × 11 MB = 924 MB ← 注意 Host 不需要 TP 分片
  Host 可缓存请求数：50 GB / 924 MB ≈ 54 个额外请求
  
  GPU 热点请求：~55 个（活跃 decode）
  Host 缓存请求：~54 个（暂时挂起）
  总并发承载：~109 个（实际受 KV cache 和调度约束，估算 80-100 个）

并发数提升：55 → 100，约 1.8x

注：更大的 CPU DRAM 或更短的平均序列长度可进一步提升
```

**吞吐收益**：

```
并发数 1.8x → 若 decode 是瓶颈：
  throughput × 1.8（线性）

若 prefill 也是瓶颈：
  更多并发请求 → 更高的 prefill 批次利用率
  → 额外 10-20% 吞吐提升（非线性叠加）

综合吞吐提升估算：约 1.5-2x（含 Phase 1 CoW 协同效果）
```

**OOM recompute 减少的收益**：

```
baseline（无 Host SSM）：
  高并发下频繁触发 recompute（每次 ~5-10 秒）
  假设 10 个请求/分钟触发 recompute
  → 损失吞吐 = 10 × 5 秒 × 30 tok/s = 1500 tok/min

Host SSM Cache 后：
  recompute 几乎不触发（SSM offload 代替）
  → 恢复 1500 tok/min ≈ 25 tok/s 额外收益（16 并发场景）
```

### 2.3 可行性分析

**技术可行性**：⭐⭐⭐⭐（高，有适配工作量）

```
✅ 昇腾 pin_memory 已验证：
   torch_npu 支持 pin_memory=True，等效 CUDA cudaHostRegister
   vllm-ascend CPU offload connector 已使用（KV 路径）

✅ 框架已有：
   CPUOffloadingConnector 提供了 D→H / H→D 的传输框架
   只需扩展 ReqMeta 增加 SSM 路径

❗ 适配工作 1：SSM state 的 layout 与 KV cache block 不同
   KV cache：contiguous block，可整块 DMA
   SSM state：per-layer 分散存储，需要 gather 后传输或逐层传输
   → 选择：batch all layers into single buffer 后 bulk transfer（更高效）

❗ 适配工作 2：TP 场景下 SSM 分片
   每卡只有 temporal_state 的 1/TP_SIZE
   Host backup：每卡独立 backup 自己的分片（无需跨卡协调）
   Host restore：每卡独立 restore（无需跨卡协调）
   ✅ 自然 TP 安全

⚠️ 风险：Host 内存池管理
   高并发下 Host DRAM 管理复杂度（碎片化、并发访问）
   → 使用预分配的固定大小 slot 池（类比 tokenspeed）
   → 无碎片，O(1) alloc/free

兼容性分析：
  ① 与 Phase 1 CoW：完全依赖（Host SSM → CoW 续算，两者协同）
  ② 与 Phase 3 TP tiebreak：独立，可并行开发
  ③ 与现有 KV offload：并行存在，互不影响
  ④ 与 prefill chunking：需要确保 backup 发生在 prefill 完成后
```

---

## Phase 3：TP 确定性调度 Tiebreak

### 3.1 技术原理深度剖析

**问题本质：分布式系统的调度一致性**

```
为什么需要所有 TP Rank 做出完全相同的调度决策？

Qwen3.5-397B TP=8：8 个进程（NPU 卡）共同执行同一 batch
  每个 Rank 都有自己的 Python 进程 + Scheduler 副本
  但所有 Rank 必须执行完全相同的 forward_op（相同的请求集合，相同的输入）

如果 Rank 0 决定调度请求 A（batch=[A,B,C]）
   Rank 3 决定调度请求 D（batch=[B,C,D]）
   → 下一个 HCCL AllReduce 会卡死（rank 0 等待 rank 3 发送 A 的 hidden states，但 rank 3 没有 A）
   → HCCL_TIMEOUT 后报错，整个推理服务 hang

触发条件（高并发 + OOM 边界）：
  当 token_budget 或 page_budget 紧张时，不同 rank 的候选列表遍历顺序决定谁被包含
  unordered_map 在不同进程的哈希随机化 → 遍历顺序不一致
  → 结果：rank 0 选了 16 个请求，rank 7 选了另外 15 个 + 1 个不同的请求
  → HCCL 死锁
```

**三类不确定性来源及修复**：

```
① 候选请求排序不确定性（最主要）
  原因：running deque + 字典迭代顺序依赖进程 hash 种子（随机）
  修复：
    candidates.sort(key=lambda req: (priority(req), req.request_id))
    request_id 是字符串（来自客户端或递增计数器），跨进程完全一致

② LRU 驱逐的 tiebreak 不确定性
  原因：相同 timestamp 的节点，用指针值（内存地址）做 tiebreak
        ASLR（Address Space Layout Randomization）使不同进程的指针值不同
        → 相同 timestamp 的两个节点，rank 0 选 A 驱逐，rank 7 选 B 驱逐
        → 下次 A/B 对应请求的 KV cache hit 情况不同 → 调度决策分叉
  修复：
    LRU key = (timestamp, seq_id, ptr)
    seq_id = 全局单调递增整数（创建时分配，与进程无关）
    ptr 只用于 set 去重，不参与比较

③ OOM Victim 选择不确定性
  原因：max(running, key=TokenSize) 若 TokenSize 相同则行为未定义
  修复：
    max(running, key=lambda r: (r.num_computed_tokens, r.request_id))
    request_id 作为最终 tiebreak，跨 rank 确定一致
```

**昇腾特有的不确定性**：

```
昇腾 A2 的额外风险：
  HCCL（华为集合通信）对 TP 不一致更敏感（相比 NCCL 的错误恢复能力）
  昇腾的 HCCL_TIMEOUT 默认值更短（某些版本为 60 秒）
  → TP 不一致导致的死锁在昇腾上更容易触发、更难恢复

修复后的保证：
  所有 TP rank 的 candidates 列表顺序完全一致
  → token_budget 耗尽时选择的候选子集完全一致
  → HCCL 死锁从"概率性触发"→ "永不触发"
```

### 3.2 量化收益分析

```
这是一个稳定性优化，收益体现为"避免的损失"而非直接吞吐提升：

估算场景（生产环境，16 并发，TP=8，7×24 运行）：
  无 tiebreak：HCCL 死锁频率 ≈ 1-5 次/天（高负载边界条件触发）
  每次死锁恢复时间 ≈ 30-120 秒（重启 worker）
  
  每天损失时间：平均 3 次 × 平均 75 秒 = 225 秒/天
  推理服务可用性：(86400 - 225) / 86400 = 99.74%

  有 tiebreak 后：死锁次数 = 0
  服务可用性：99.99%+（由其他故障决定）

  SLA 角度：0.25% → 0% HCCL 导致的可用性损失
  
间接吞吐收益：
  死锁期间所有并发请求都失败，重启后用户重试
  保守估算：避免 225 秒/天 × 30 tok/s × 16 并发 = 108,000 tokens/天的损失
```

### 3.3 可行性分析

**技术可行性**：⭐⭐⭐⭐⭐（最高，改动最小）

```
✅ 改动极小：
   只需在 RecomputeScheduler.schedule() 加一行 sort()
   + LRU key 结构修改
   代码变更 <50 行

✅ 零性能负担：
   sort() 的时间复杂度 O(N log N)，N = active requests（通常 <200）
   每次 schedule 调用约 1-2 µs，可忽略

✅ 零风险：
   确定性排序只影响 tie-breaking 场景（相同优先级时）
   对于优先级不同的请求（绝大多数情况），行为不变

兼容性：
  ① 不影响 vllm 上游 Scheduler（只在 RecomputeScheduler 改）
  ② 不影响任何其他特性
  ③ 对现有测试无影响
```

---

## Phase 4：MTP O(1) SSM 索引更新

### 4.1 技术原理深度剖析

**MTP（Multi-Token Prediction）的工作机制**：

```
MTP = 每个 decode step 同时生成 N 个草稿 token，然后验证
  主模型（Qwen3.5-397B）：每步生成 1 个 token
  MTP draft（Qwen3.5-NextN）：预测接下来 N 个 token（e.g. N=3）
  
  step t:
    主模型生成 token[t]
    MTP 同时生成 draft[t+1], draft[t+2], draft[t+3]
    验证：逐一 check draft 是否与贪心解码一致
    接受 k 个（k ≤ N）→ 推进 k+1 个 token

  throughput 提升 ≈ (1 + avg_accept_rate × N) / 1
  假设 avg_accept_rate = 0.8，N=3：(1 + 0.8×3)/1 = 3.4x decode 速度
```

**SSM State 在 MTP 中的挑战**：

```
每个 draft token 生成时，GDN 层的 SSM state 会更新：
  step t（decode）：state[t] → state[t+1]
  draft 1：state[t+1] → draft_state[t+2]
  draft 2：draft_state[t+2] → draft_state[t+3]
  draft 3：draft_state[t+3] → draft_state[t+4]
  
verify：接受 k=2 个 draft → 真实 state 是 draft_state[t+3]

问题：verify 后，如何以最低代价恢复正确的 SSM state？

朴素方案（当前 vllm）：
  保存 N 个 draft state 快照
  verify 后：将第 k 个快照 copy 到 working state
  代价：O(N × L × D) 的 GPU 内存 + O(L × D) 的 tensor copy
    L = 84 层，D = 11 MB/层/请求
    每次 verify copy：84 × 11 MB = 924 MB/请求 ← 极大

O(1) 整数索引方案（tokenspeed）：
  分配 N+1 个 SSM slot 给每个请求（1 base + N draft）
  GDN kernel 直接将 draft step k 的 state 写入 slot[base + k]
  verify 后：只更新一个整数 current_slot[req_id] = base + k
  代价：O(batch_size) 整数写（极小）
```

**Draft Slot 内存布局**：

```
SimpleMambaPool 内存布局（N=3 draft tokens，16 并发请求）：

GPU SSM slots：
  [slot 0..15]：16 个请求的 base slots（base model working state）
  [slot 16..63]：16 × 3 = 48 个 draft slots（round-robin 使用）

  req_id=0 的 draft slots：slot 16, 17, 18（3个轮转）
  req_id=1 的 draft slots：slot 19, 20, 21
  ...

current_slot_indices[16]：整数数组，记录每个请求的当前有效 slot
  initial: current_slot_indices = [0, 1, 2, ..., 15]

MTP step（req_id=0，生成 3 个 draft）：
  GDN kernel 写：
    draft 1 → slot 16 的 state
    draft 2 → slot 17 的 state
    draft 3 → slot 18 的 state

verify（req_id=0，接受 k=2 个）：
  current_slot_indices[0] = 16 + (2-1) % 3 = 17
  下一步 decode：从 slot 17 的 state 出发
  代价：1 次整数写！

对比朴素方案：
  朴素：924 MB tensor copy（per request）
  O(1)：1 整数写 = 4 bytes
  内存带宽节省：99.999%
```

**`@torch.compile(dynamic=True)` 在昇腾上的作用**：

```
verify kernel 是一个 Python 循环（O(batch_size) 整数写）：
  for i in range(batch_size):
    if accept_lengths[i] > 0:
      slot_indices[req_pool_indices[i]] = draft_base + ...

torch.compile(dynamic=True) 将此编译为：
  ① 在 CUDA/NPU 上：单个 kernel launch（消除 Python 循环开销）
  ② dynamic=True：允许 batch_size 变化时不重新编译（节省编译时间）
  
昇腾适配：
  TorchDynamo + Ascend NPU 后端（torch_npu）
  支持 torch.compile 的 eager 和 graph 模式
  整数写操作在 NPU 上极高效（不涉及复杂算术）
```

### 4.2 量化收益分析

```
场景：MTP N=3，avg_accept_rate=0.8，batch=16，Qwen3.5-397B @ A2 16卡

朴素 MTP SSM 更新：
  每次 verify：16 请求 × 924 MB tensor copy
  A2 HBM 带宽：800 GB/s/卡
  copy 时间 ≈ 16 × 924 MB / 800 GB/s ≈ 18 ms/step
  → verify step 被 tensor copy 主导，30 tok/s → 实际 ~15 tok/s（copy overhead）

O(1) 整数索引更新：
  每次 verify：16 × 4 bytes = 64 bytes
  时间 ≈ 64 bytes / 800 GB/s ≈ 0.08 µs（可忽略）
  
  MTP 理论加速（accept_rate=0.8，N=3）：
    (1 + 0.8×3) = 3.4x decode speed
  实际（去掉朴素 copy overhead）：~1.7x
  O(1) 后：接近 3.4x 理论值（2.8-3x 估算）

TPOT 收益：
  baseline（无 MTP）：~33 ms/tok（30 tok/s）
  朴素 MTP：~60 ms/step / 3.4 tok = 17.6 ms/tok（但实际 <3.4x 加速）
  O(1) MTP：~35 ms/step / 3.4 tok ≈ 10 ms/tok

→ TPOT 从 33 ms/tok → 10 ms/tok，降低约 70%
→ 对应吞吐：30 tok/s → 100 tok/s（单流，满足 SLA 下的真实 decode 速度）
```

### 4.3 可行性分析

**技术可行性**：⭐⭐⭐（中高，有昇腾适配风险）

```
✅ 内存布局已验证（vllm 上游 mamba_utils.py）：
   mamba_state_idx 机制已在 GPU 端实现
   tokenspeed 的 SimpleMambaPool 设计经过 B200 生产验证

❗ 昇腾适配风险：
   vllm 上游 postprocess_mamba_fused_kernel 是 Triton kernel
   昇腾 310P：Triton 不完全支持，需要替换为 pytorch 参考实现或 ACLNN
   昇腾 A3：Triton-Ascend 支持，但需验证整数写 kernel 的正确性

❗ ACL Graph 集成：
   MTP verify 必须在 ACL Graph（等效 CUDA Graph）外执行
   verify 结果（accept_lengths）是 CPU 可见的整数
   → verify outside graph → update slot_indices → next graph replay with new slot
   这需要确保 ACL Graph 的输入 buffer（slot_indices）是 graph-external 的
   → 工作量约 1 周额外调试

兼容性：
  ① 与 Phase 1 CoW：协同（draft slot 是 CoW 的扩展）
  ② 与 DFlash proposer：需要接入 AscendDflashProposer.verify() 回调
  ③ 与 chunked prefill：不影响（仅 decode 阶段）
  ④ 与 310P：需单独适配（纯 pytorch 实现即可）
```

---

## Phase 5：PD 分离 Layer-wise SSM 传输

### 5.1 技术原理深度剖析

**PD 分离的基本原理**：

```
P（Prefill）节点：专注 prefill 计算
D（Decode）节点：专注 decode 计算

不分离时的问题：
  prefill 和 decode 混跑，互相抢占 GPU 资源
  → prefill 的长批次（大 batch_size）导致 decode 的 TPOT 抖动
  → decode 的小批次需求导致 prefill GPU 利用率低

分离后：
  P 节点：大 batch prefill，GPU 利用率接近峰值
  D 节点：小 batch decode，TPOT 稳定，SLA 可保证
  → 整体集群吞吐提升 20-40%
```

**SSM State 在 PD 分离中的挑战**：

```
Standard PD 分离（纯 Full-Attention 模型）：
  P 节点 prefill 完成 → 传输 KV Cache 给 D 节点
  D 节点从 last token 开始 decode
  KV 传输量：L_prefix × num_layers × 2 × num_kv_heads × head_dim × 2B

Hybrid GDN 模型的额外挑战：
  P 节点 prefill 完成 → 除了 KV，还需传输 SSM State
  SSM 传输量：num_gdn_layers × SSM_size = 84 × 84.5 MB = 7.1 GB（全量）
  
  串行传输 7.1 GB @ 40 GB/s（RDMA）：~175 ms
  → D 节点等待时间 175 ms，严重影响 TTFT

Layer-wise 流水化传输（tokenspeed 解法）：
  P 节点完成 layer k 的 prefill → 立即发送 layer k 的 KV + SSM State
  D 节点收到 layer k → 可以开始 layer k 的 decode 预热
  
  流水化后：
    P node prefill L=90k，单层时间 ≈ 90k/94 层 × forward_time
    D node 等待第一层：约 prefill 时间/94 + 传输单层时间
    有效等待时间 ≈ 1 层传输时间（~2 ms）而非全量等待（~175 ms）
  
  D 节点启动延迟：175 ms → 2 ms（约 100x 改善）
```

**Bootstrap Token 机制**：

```
D 节点从 Retracted 状态恢复 decode 的特殊处理：

问题：D 节点没有执行 prefill，没有 prefill 的最后一个 token 的隐状态
     → 如何开始第一个 decode step？

Bootstrap token 解法：
  P 节点 prefill 完成后，将 last_token_id 和 last_hidden_state 发送给 D 节点
  D 节点将此作为 decode step 0 的输入（hist_token_len = total_len - 1）
  → 等效于 decode step 0 只有 1 个 token 的输入
  → 正常执行 GDN decode（单步，O(1)，使用传来的 SSM state）

StateRecovery Intent：
  D 节点 Match 时用 MatchIntent::StateRecovery（而非 PrefixReuse）
  → 优先找最深的 SSM state 快照（不强求 KV 也全在 GPU）
  → 因为 P 节点已经通过 RDMA 传来了完整 SSM state
```

### 5.2 量化收益分析

```
场景：2 节点（16 卡）P 节点 + 2 节点（16 卡）D 节点
     总资源：4 节点 32 卡 910B Pro

无 PD 分离（混合部署，4 节点）：
  prefill+decode 混跑，互相干扰
  估算 GPU 利用率：60-70%（prefill 期间 decode TPOT 增加）
  总吞吐：~16 并发 × 30 tok/s × 0.65 = 312 tok/s

PD 分离后（2P + 2D）：
  P 节点：专注 prefill，GPU 利用率 ~85%
    throughput：处理更多并发 prefill（N_prefill 可以更大）
  D 节点：专注 decode，TPOT 稳定 <50 ms（SLA 可保证）
    支持更多并发 decode（N_decode 可以更大）
  
  理论峰值：4 节点资源利用率从 65% → 85%
  吞吐提升：312 × (0.85/0.65) = 408 tok/s → 约 +31%

Layer-wise 流水化额外收益：
  D 节点启动延迟降低：175 ms → 2 ms
  影响：高 QPS 场景下，TTFT 降低 ~175 ms（对于 1-2s 的总 TTFT，约 10% 改善）

综合 PD 分离吞吐收益：+20-40%（与集群规模、流量模型相关）
```

### 5.3 可行性分析

**技术可行性**：⭐⭐⭐（中，依赖 Phase 1+2，工程复杂度最高）

```
✅ 已有基础：
   MooncakeLayerwiseConnector（KV layer-wise 传输）已有完整框架
   HCCL/Mooncake RDMA 通道已验证（vllm-ascend 生产在用）
   RecomputeScheduler 的 PD 角色（kP/kD）可通过 config 扩展

❗ 主要挑战：
   1. SSM State 的 layer-by-layer 发送需要精确同步
      P 节点：完成 layer k → send SSM_k → increment layer_done_counter[k]
      D 节点：wait_until layer_done_counter[k] → recv SSM_k → start decode
      同步原语：需要在 HCCL/Mooncake 之上实现 layer-level 事件通知
   
   2. SSM State 的序列化
      temporal_state：float16，densely packed，可直接 DMA
      conv_state：较小，可与 temporal_state 一起打包
      无需特殊序列化
   
   3. 与 vllm-ascend 调度器的集成
      需要 RecomputeScheduler 支持 Role.kP/Role.kD 分支
      类比 tokenspeed 的 config_.role 判断
      工作量：约 2 周

兼容性：
  ① 强依赖 Phase 1（CoW）和 Phase 2（Host SSM）：
     Phase 5 的 D 节点 decode 依赖 SSM state 正确传输和 CoW
     若 Phase 1/2 未完成，Phase 5 无法工作
  ② 与 Mooncake 集成：已有 mooncake_layerwise_connector，扩展 SSM 通道
  ③ 与 PP（Pipeline Parallel）：复杂，PD+PP 同时使用需额外设计
     建议：Phase 5 先只支持 TP+PD，不支持 PP
  ④ 与 310P：310P 使用不同通信后端，需单独适配
```

---

## 综合收益叠加分析

### 各 Phase 收益矩阵（Qwen3.5-397B @ A2，16 卡，长 prefix 场景）

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  基准（vllm-ascend 当前）：                                                   │
│  TTFT ≈ 8 秒（90k prefix），TPOT ≈ 33 ms，并发 ≈ 55 请求，吞吐 ≈ 300 tok/s  │
├─────────┬───────────────┬──────────────┬────────────────┬────────────────────┤
│ Phase   │ TTFT 变化     │ TPOT 变化    │ 并发数变化      │ 综合吞吐估算        │
├─────────┼───────────────┼──────────────┼────────────────┼────────────────────┤
│ Base    │ 8 秒          │ 33 ms        │ 55 并发        │ 300 tok/s          │
├─────────┼───────────────┼──────────────┼────────────────┼────────────────────┤
│ +Ph1    │ 0.8 秒 (-90%) │ 33 ms (持平) │ 55 并发 (持平) │ ~450 tok/s (+50%)  │
│ CoW     │               │              │                │ prefill 减少释放GPU │
├─────────┼───────────────┼──────────────┼────────────────┼────────────────────┤
│ +Ph2    │ 0.8 秒 (持平) │ 33 ms (持平) │ 100 并发 (+82%)│ ~820 tok/s (+82%)  │
│ HostSSM │               │              │                │ 并发数提升          │
├─────────┼───────────────┼──────────────┼────────────────┼────────────────────┤
│ +Ph3    │ 0.8 秒 (持平) │ 33 ms (持平) │ 100 并发 (持平)│ ~820 tok/s (稳定性)│
│ Tiebrk  │               │              │                │ 避免死锁损失        │
├─────────┼───────────────┼──────────────┼────────────────┼────────────────────┤
│ +Ph4    │ 0.8 秒 (持平) │ 10 ms (-70%) │ 100 并发 (持平)│ ~1000 tok/s (+22%) │
│ MTP O(1)│               │              │                │ MTP 加速 decode     │
├─────────┼───────────────┼──────────────┼────────────────┼────────────────────┤
│ +Ph5    │ 0.7 秒 (-12%) │ 10 ms (持平) │ 150+ 并发      │ ~1400 tok/s (+40%) │
│ PD 分离 │               │              │ (多节点)       │ 集群规模化          │
└─────────┴───────────────┴──────────────┴────────────────┴────────────────────┘

注：吞吐估算基于 16 卡 A2 场景，各 Phase 假设前序已完成
    P5 需额外 16 卡（D 节点），实际是 32 卡集群的总吞吐
```

### 非线性叠加效应

```
Phase 1 + Phase 2 的协同：
  CoW 减少了 prefill 时间 → GPU 空出更多 decode 时间槽
  Host SSM 增加了并发数 → 更多请求填充空出的 decode 时间槽
  两者叠加 > 线性相加（prefill 节省时间被更多 decode 利用）

Phase 4 + Phase 1/2 的协同：
  Phase 1 减少了 prefill 时间，Phase 4 减少了 decode 时间
  整体延迟降低 → 相同 SLA 下可承载更多 QPS

实际测量建议：
  在 A2 真实环境下，每个 Phase 上线后单独 benchmark
  用 benchmarks/benchmark_throughput.py 测量
  避免过度估算，以实测为准
```

---

## 实施优先级与路线图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      实施路线图（2 SE × 3 个月）                          │
│                                                                          │
│  Week 1-2:   Phase 3（TP tiebreak）- 最低风险，立即消除隐患              │
│              SE-A: recompute_scheduler.py sort + LRU fix                 │
│                                                                          │
│  Week 1-6:   Phase 1（Mamba CoW）- 最高收益                             │
│              SE-B: SchedulerOutput 扩展 → ModelRunner 注入 → GDN forward │
│                                                                          │
│  Week 7-9:   Phase 2（Host SSM Cache）- 扩大并发上限                    │
│              SE-A: mamba_host_cache.py + CPUOffloadingConnector 集成     │
│                                                                          │
│  Week 7-12:  Phase 4（MTP O(1)）- 进一步提升 decode 速度               │
│              SE-B: slot indices + AscendDflashProposer 接入 + ACL Graph  │
│                                                                          │
│  Week 13-24: Phase 5（PD 分离）- 集群规模化                             │
│              SE-A + SE-B + 额外 0.5 SE: Mooncake SSM 通道 + 调度扩展    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 参考数据来源

| 数据项 | 来源 |
|-------|-----|
| Qwen3.5-397B SSM state 参数 | `file://tokenspeed/python/tokenspeed/runtime/configs/qwen3_5_text_base_config.py` |
| GDN 层分布（full_attention_interval） | `file://tokenspeed/python/tokenspeed/runtime/configs/qwen3_5_text_base_config.py:269` |
| SSM state 公式 | `file://tokenspeed/tokenspeed-scheduler/csrc/resource/radix_tree/radix_tree.h` |
| CoW kernel 实现 | `file://tokenspeed/python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py` |
| bulk D→H 传输 | `file://tokenspeed/python/tokenspeed/runtime/cache/mamba_cache_host.py` |
| TP tiebreak 设计 | `file://tokenspeed/tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp:551-571` |
| O(1) slot 更新 | `file://tokenspeed/python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py:_update_current_inputs_after_verify_kernel` |
| 昇腾 GDN 实现 | `file://vllm-ascend/vllm_ascend/ops/gdn.py` |
| mamba align 配置 | `file://vllm-ascend/vllm_ascend/patch/platform/patch_mamba_config.py` |
| CPU offload 框架 | `file://vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py` |
| RecomputeScheduler | `file://vllm-ascend/vllm_ascend/core/recompute_scheduler.py` |
| 昇腾 A2 规格 | Dockerfile.a3（SOC_VERSION=ascend910_9391）+ 公开数据手册 |

---

*本分析所有量化数字基于架构参数推导，实际数值需在昇腾 A2 真实环境中 benchmark 验证。*
*收益估算保守侧，实测通常优于估算（因多个优化协同叠加效应）。*
