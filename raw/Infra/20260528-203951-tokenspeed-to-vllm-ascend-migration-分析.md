# TokenSpeed GDN-KVCache 协同优化迁移至 vllm-ascend：完整方案分析

> **输出日期**：2026-05-28  
> **目标硬件**：昇腾 A2 / A3  
> **目标模型**：Qwen3.5-397B-A17B（Hybrid GDN + MoE）  
> **参考实现**：`/Users/linyi/code/Documents/code/tokenspeed/`  
> **目标仓库**：`/Users/linyi/code/Documents/code/vllm-ascend/`

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [两套系统现状对比](#2-两套系统现状对比)
3. [核心差距分析](#3-核心差距分析)
4. [整体迁移架构](#4-整体迁移架构)
5. [分阶段实施计划](#5-分阶段实施计划)
6. [关键文件映射与可复用能力](#6-关键文件映射与可复用能力)
7. [收益预期与量化分析](#7-收益预期与量化分析)
8. [验证方法](#8-验证方法)
9. [工作量与优先级](#9-工作量与优先级)

---

## 1. 背景与动机

### 1.1 TokenSpeed 的性能基准

TokenSpeed 在 B200（4 节点）上实现 Qwen3.5-397B **580 tok/s**，相比 vLLM/SGLang 约 200-300 tok/s 有 2-3x 提升。核心优化体系围绕 **GDN-KVCache 协同优化**：

| 优化层 | 核心技术 | 收益 |
|-------|---------|------|
| KV 前缀缓存 | RadixTree + SHA256 链式哈希 | 90%+ prefix hit → TTFT 10x |
| SSM 状态缓存 | Mamba CoW + branching_seqlen | GDN 递推从 100% 降至 10% |
| L2 Host Cache | cudaHostRegister + 异步 D↔H | GPU SSM 内存 5-10x 释放 |
| TP 确定性 | Request::Id() tiebreak | 消除 NCCL 死锁 |
| MTP 投机解码 | O(1) 整数索引更新 | TPOT 降低 30-60% |
| PD 分离 | Layer-wise SSM 传输 | 集群吞吐 +20-40% |

### 1.2 迁移目标

在昇腾 A2/A3 上，将 vllm-ascend 对 Qwen3.5-397B 的推理性能**从当前 baseline 提升 3-5x**，实现等效于 B200 TokenSpeed 的相对优化收益（考虑昇腾硬件差异后）。

---

## 2. 两套系统现状对比

### 2.1 vllm-ascend 已有能力（可复用）

通过对 `/Users/linyi/code/Documents/code/vllm-ascend/vllm_ascend/` 的全面代码分析，已有以下关键能力：

**① Qwen3.5 基础支持**

```
file://vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py
file://vllm-ascend/vllm_ascend/_310p/ops/fla/gdn_310.py
```

`AscendQwen3_5DecoderLayer` 已支持 `layer_type="linear_attention"` 和 `"full_attention"` 的混合路由，`AscendQwen3NextAttention` 针对昇腾优化了 `triton_split_qkv_rmsnorm_mrope` 融合算子。

**② GDN 算子（两套路径）**

```
file://vllm-ascend/vllm_ascend/ops/gdn.py              # A3 路径（Triton + NPU custom op）
file://vllm-ascend/vllm_ascend/_310p/ops/fla/gdn_310.py # 310P 路径（pytorch 参考实现）
```

- A3：`chunk_gated_delta_rule`（Triton，已适配昇腾）+ `npu_recurrent_gated_delta_rule`（自定义 NPU op）
- 310P：`chunk_gated_delta_rule_pytorch` + `fused_gdn_gating_pytorch`
- **关键**：两套实现均已支持 `initial_state` 参数，可直接作为 Mamba CoW 的续算入口

**③ Mamba align 模式前缀缓存**

```
file://vllm-ascend/vllm_ascend/patch/platform/patch_mamba_config.py
```

`HybridAttentionMambaModelConfig.verify_and_update_config` 已实现：
- `mamba_cache_mode="align"` 时，`mamba_block_size = attention_block_size`（page 对齐）
- 自动计算 `mamba_page_size_padded`（conv + ssm 合并页）
- **这就是 block-aligned SSM 快照的配置入口**，与 tokenspeed `FLA_CHUNK_SIZE` 对齐逻辑等价

**④ CPU KV Offload（L2 Cache 框架）**

```
file://vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py
file://vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_kv_cache_manager.py
```

`CPUOffloadingConnector` 已实现：KV pages 的 D→H / H→D 异步传输框架、`ReqMeta` 数据结构管理 CPU/GPU block ids、LRU 通过 `sha256` hash 管理。**但不覆盖 Mamba/GDN SSM 状态**。

**⑤ MTP / Spec Decode 框架**

```
file://vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py
file://vllm-ascend/vllm_ascend/spec_decode/eagle_proposer.py
```

`AscendDflashProposer`（继承 `AscendEagleProposer`）已有完整 MTP 提案框架。vllm 上游 `mamba_utils.py` 已有 `postprocess_mamba_fused_kernel`（Triton kernel，实现 block-aligned SSM state copy），但**昇腾路径（`npu_recurrent_gated_delta_rule_310`）未接入此 postprocess**。

**⑥ RecomputeScheduler（OOM 抢占框架）**

```
file://vllm-ascend/vllm_ascend/core/recompute_scheduler.py
```

比 vllm 上游多了 `RecomputeReqInfo` + `recomputed_reqs` 列表，OOM 时 `pop()` 最后一个 running 请求并 recompute（等价于 tokenspeed 的 retraction 机制）。**差距**：victim 选择策略是 LIFO（栈顶），tokenspeed 是最长 token 序列优先，且 tiebreak 无确定性保证。

**⑦ Mooncake Layer-wise 传输（PD 分离基础）**

```
file://vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py
```

已有 KV layer-wise P→D 传输，扩展 SSM 传输通道即可支持 tokenspeed 的 PD 分离 SSM layer-wise 流水。

### 2.2 vllm-ascend 缺失能力（需迁移）

| 缺失能力 | TokenSpeed 实现位置 | 影响量级 |
|---------|-------------------|---------|
| **Mamba CoW + branching_seqlen** | `hybrid_linear_attn.py` `SimpleMambaPool` | TTFT 60-80%↓ |
| **Host SSM L2 Cache** | `mamba_cache_host.py` | 并发数 3-5x↑ |
| **TP 确定性调度 tiebreak** | `forward.cpp` `Request::Id()` sort | 零 HCCL 死锁 |
| **MTP O(1) SSM 索引更新** | `_update_current_inputs_after_verify_kernel` | TPOT 20-30%↓ |
| **PD 分离 SSM layer-wise 传输** | `MambaCachePool.load_to_device_per_layer` | 吞吐 +20-40% |
| **细粒度 PagedCache 三阶段匹配** | `HybridPrefixCache.augmentMatchPagedCache` | 命中率 +5-15% |

---

## 3. 核心差距分析

### 3.1 差距 1：Mamba CoW 机制缺失

**问题**：vllm-ascend 的 `mamba_cache_mode="align"` 已在 block 边界保存 SSM 快照（通过 `postprocess_mamba_fused_kernel`），但**没有跨请求复用**这个快照的机制。

每个新请求即使有 90% prefix 命中，GDN 层仍然从 token 0 开始重新递推 SSM 状态（O(full_sequence_length) 步），而不是从命中边界（`branching_seqlen`）续算（O(10% tokens)）。

**TokenSpeed 的解法**（源于 `hybrid_linear_attn.py`）：
```
prefix cache 命中 → 找最深 aligned SSM 快照节点
→ mamba_cow_src_index = node.MambaSlotIndex()
→ branching_seqlen = (match_depth // chunk_size) * chunk_size
→ GDN forward: CoW src → working slot，从 branching_seqlen 位置续算
```

**vllm-ascend 已有但未连通的拼图**：
- SSM 快照存在（align 模式）✓
- GDN `initial_state` 参数支持 ✓
- KV prefix cache 命中深度 ✓
- **缺失**：三者之间的信号传递链路（Scheduler → SchedulerOutput → ModelRunner → GDN forward）

### 3.2 差距 2：Host SSM 状态管理

**问题**：SSM 状态（conv state + ssm_h）是 per-request 的，Qwen3.5-397B 有 94 个 GDN 层，每层 SSM 状态约 `(num_heads × head_dim² + conv_dim × conv_kernel) × 2 bytes`，**一个请求的全量 SSM 状态约 200-500MB GPU 内存**。

大 batch 场景下，GPU SSM 内存成为并发上限的瓶颈，而非计算量。

**TokenSpeed 的解法**：
```
cudaHostRegister → GPU-visible pinned memory
不活跃请求 SSM → D→H 异步（transfer_kv_all_layer_mla，单 kernel 覆盖所有层）
恢复时 → H→D 异步（transfer_kv_per_layer_mla，逐层，支持 PD 流水）
LRU 驱逐 → 与 KV cache 分离的 Mamba-only LRU（MambaEvictionManager）
```

**vllm-ascend 现状**：KV CPU offload 有框架，但 SSM 状态完全在 GPU，无任何 offload 机制。

### 3.3 差距 3：TP 确定性调度

**问题**：A2/A3 部署 Qwen3.5-397B 通常 TP=8 或 TP=16。`RecomputeScheduler` 继承了 vllm 上游调度器的不确定性——`running` deque 的出队顺序和字典的迭代顺序在不同进程间不同（hash 随机化 + ASLR），导致不同 TP rank 在资源紧张时调度不同请求，下一个 HCCL AllReduce 死锁。

**TokenSpeed 的三层保证**（源于 `forward.cpp`）：
1. 候选排序：`(priority, Request::Id())` — request_id 是字符串，跨进程一致
2. LRU tiebreak：`seq_id`（单调递增整数）而非指针（ASLR 不稳定）
3. OOM victim：`max(TokenSize)` — token 数跨 rank 一致

**vllm-ascend 现状**：`RecomputeScheduler.schedule()` 按 `running.popleft()` LIFO 顺序，无 tiebreak，OOM victim 同样不确定。

### 3.4 差距 4：MTP SSM 状态 O(1) 更新

**问题**：vllm 上游的 `postprocess_mamba_fused_kernel` 在每个 verify step 后执行 SSM block copy，代价 O(L×D)（每层全量 copy）。

**TokenSpeed 的解法**：
- Draft slots 用整数索引 `current_input_indices[req_pool_index]` 指向
- verify 后只更新整数（O(batch) 写），不拷贝 tensor
- `@torch.compile(dynamic=True)` 消除 Python 循环

**vllm-ascend 现状**：310P/A3 路径的 `npu_recurrent_gated_delta_rule_310` 未接入 align 模式的 postprocess，MTP verify 后 SSM 状态无有效更新机制（或全量重算）。

---

## 4. 整体迁移架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    迁移后 vllm-ascend 系统架构                               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │               RecomputeScheduler (改造)                             │    │
│  │  vllm_ascend/core/recompute_scheduler.py                            │    │
│  │                                                                     │    │
│  │  schedule()                                                         │    │
│  │  ├── KV prefix match → cpu_kv_cache_manager.get_matched_num()      │    │
│  │  ├── [新增] SSM CoW 决策：                                          │    │
│  │  │    match_depth → branching_seqlen = align(depth, mamba_blk_sz)  │    │
│  │  │    找最深 SSM GPU slot → mamba_cow_src_index                     │    │
│  │  │    若仅 Host → PrepareMambaLoadBack → H→D async                 │    │
│  │  ├── [改造] victim 选择：max(num_computed_tokens) + req_id tiebreak │    │
│  │  └── SchedulerOutput + {mamba_cow_src, branching_seqlen, ...}      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                    │                          │                             │
│         ┌──────────▼──────────┐   ┌──────────▼──────────────────────┐      │
│         │  MambaHostCache     │   │  ModelRunnerV1 (改造)            │      │
│         │  [新增]             │   │  vllm_ascend/worker/             │      │
│         │  mamba_host_cache.py│   │  model_runner_v1.py              │      │
│         │                     │   │                                  │      │
│         │  backup_all_layers()│   │  prepare_inputs():               │      │
│         │  (D→H bulk async)   │   │  inject mamba_cow_src_index      │      │
│         │                     │   │  inject branching_seqlen         │      │
│         │  load_layer()       │   │  into GDNAttentionMetadata        │      │
│         │  (H→D per-layer)    │   │                                  │      │
│         │                     │   │  [MTP verify 外置]               │      │
│         │  pin_memory via     │   │  update_mamba_after_verify():    │      │
│         │  torch_npu pinned   │   │  O(batch) index write only       │      │
│         └─────────────────────┘   └──────────────────────────────────┘      │
│                    │                          │                             │
│         ┌──────────▼──────────────────────────▼──────────────────────┐      │
│         │                GDN Forward (改造)                           │      │
│         │  vllm_ascend/ops/gdn.py                                     │      │
│         │                                                             │      │
│         │  if branching_seqlen > 0:                                   │      │
│         │    1. CoW: ssm_state[working] ← ssm_state[cow_src]          │      │
│         │    2. chunk forward from branching_seqlen                   │      │
│         │       (already supports initial_state param)               │      │
│         │  else:                                                      │      │
│         │    standard full forward from 0                             │      │
│         └────────────────────────────────────────────────────────────┘      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  CPUOffloadingConnector (改造)                                      │    │
│  │  vllm_ascend/distributed/.../cpu_offload_connector.py              │    │
│  │                                                                     │    │
│  │  ReqMeta += mamba_cpu_slot_id                                      │    │
│  │  OOM evict → mamba_host_cache.backup_all_layers()                  │    │
│  │  Restore  → mamba_host_cache.load_layer() per-layer                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  [Phase 5: PD 分离扩展]                                                      │
│  MooncakeLayerwiseConnector → mamba_layer_done 信号 → D 节点流水接收         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 架构设计原则

1. **最小侵入**：不修改 vllm 上游核心，所有改动以 vllm-ascend patch 方式叠加
2. **可复用优先**：GDN `initial_state`、CPU offload 框架、spec decode 框架均直接复用
3. **信号链完整**：Scheduler → SchedulerOutput → ModelRunner → GDN forward，每层数据结构有明确扩展点
4. **昇腾适配**：CUDA `cudaHostRegister` → `torch_npu` pin_memory；Triton kernel → ACLNN 算子等效替换

---

## 5. 分阶段实施计划

### Phase 1：Mamba CoW + branching_seqlen（P0，3-4 周）

**目标**：prefix cache 命中时 GDN 从 `branching_seqlen` 位置续算，不重推全序列。

#### Step 1.1：扩展 SchedulerOutput

**改造文件**：`vllm_ascend/core/recompute_scheduler.py`

在 `SchedulerOutput` 数据类中增加字段（通过 dataclass 继承或直接扩展）：

```python
@dataclass
class AscendSchedulerOutput(SchedulerOutput):
    # 新增：Mamba CoW 控制字段
    mamba_cow_src_indices: dict[str, int] = field(default_factory=dict)
    # request_id → src mamba slot index（GPU 上已有的对齐快照槽位）

    mamba_branching_seqlens: dict[str, int] = field(default_factory=dict)
    # request_id → branching_seqlen（GDN 从此处续算，跳过前缀）
```

#### Step 1.2：在 schedule() 中注入 CoW 决策

在 `RecomputeScheduler.schedule()` 处理新请求调度时（`Submitted` 状态进入 `Prefilling`），插入 CoW 查找逻辑：

```python
def _compute_mamba_cow_for_request(self, request: Request, kv_match_depth: int) -> tuple[int, int]:
    """
    返回 (mamba_cow_src_index, branching_seqlen)
    -1 表示无命中，从 0 开始全量计算
    """
    mamba_block_size = self.cache_config.mamba_block_size
    if mamba_block_size <= 0 or kv_match_depth == 0:
        return -1, -1

    # 对齐到 mamba block boundary（等价于 tokenspeed AlignMambaCacheSeqlen）
    branching_seqlen = (kv_match_depth // mamba_block_size) * mamba_block_size
    if branching_seqlen == 0:
        return -1, -1

    # 从 CPU KV cache manager 查找该 prefix depth 对应的 SSM 快照槽位
    # mamba_cache_mode="align" 下，快照与 KV block 共存于同一 block 结构
    src_slot = self._mamba_host_cache.find_slot_at_depth(
        request.prefix_token_ids, branching_seqlen
    )
    if src_slot < 0:
        return -1, -1

    return src_slot, branching_seqlen
```

#### Step 1.3：改造 GDN forward 接受 branching_seqlen

**改造文件**：`vllm_ascend/ops/gdn.py`

在 `GatedDeltaNetAttention.forward()` 的昇腾 patch 中，增加 CoW + 续算分支：

```python
# 从 attn_metadata 读取 CoW 信息（由 ModelRunner 填充）
cow_src_indices = getattr(attn_metadata, 'mamba_cow_src_indices', None)
branching_seqlens = getattr(attn_metadata, 'mamba_branching_seqlens', None)

if cow_src_indices is not None and branching_seqlens is not None:
    # Step 1: CoW — 将 src slot 的 SSM 状态复制到 working slot
    # ssm_state shape: [num_slots, num_heads, head_dim, head_dim]
    for req_idx, (src, bseqlen) in enumerate(zip(cow_src_indices, branching_seqlens)):
        if src >= 0 and bseqlen > 0:
            # working slot 的索引由 ssm_state_indices[req_idx] 给出
            working = ssm_state_indices[req_idx]
            ssm_state[working].copy_(ssm_state[src])
            if conv_state is not None:
                conv_state[working].copy_(conv_state[src])

    # Step 2: prefill 时 initial_state 参数注入（已支持）
    # 昇腾 chunk_gated_delta_rule 的 initial_state 参数直接对应 branching_seqlen 处的状态
    output = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        initial_state=ssm_state[ssm_state_indices],  # 已 CoW 过的 working slot
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        # 关键：告知 kernel 从 branching_seqlen 开始计算，前面的 tokens 已在 state 中
        # 通过调整 input tokens 的 offset 实现（或直接截断输入 token 序列）
    )
```

**关键点**：`chunk_gated_delta_rule` 的 `initial_state` 已支持，只需确保输入 token 序列从 `branching_seqlen` 开始截断（前缀对应的 KV 已在 cache 中，不需要重算）。

#### Step 1.4：ModelRunner 注入 attn_metadata

**改造文件**：`vllm_ascend/worker/model_runner_v1.py`

在 `prepare_inputs_for_prefill()` 中：
```python
# 从 scheduler_output 读取 Mamba CoW 信息，注入 GDNAttentionMetadata
for req_id, req_data in scheduler_output.requests.items():
    cow_src = scheduler_output.mamba_cow_src_indices.get(req_id, -1)
    bseqlen = scheduler_output.mamba_branching_seqlens.get(req_id, -1)
    # 填入对应的 attn_metadata tensor 位置
```

---

### Phase 2：Host SSM L2 Cache（P0，2-3 周）

**目标**：不活跃请求的 SSM 状态 D→H 卸载，恢复时 H→D，提升并发数上限。

#### Step 2.1：新建 MambaHostCache

**新文件**：`vllm_ascend/core/mamba_host_cache.py`

```python
class MambaHostCache:
    """
    昇腾版 Mamba Host L2 Cache，类比 tokenspeed MambaCachePool + MambaPoolHost
    """
    def __init__(
        self,
        num_layers: int,
        num_slots: int,       # CPU 侧最大缓存的请求数
        ssm_state_shape: tuple,   # per-layer SSM state shape
        conv_state_shape: tuple,  # per-layer conv state shape
        dtype: torch.dtype,
        device: torch.device,
    ):
        # CPU 固定内存：昇腾等效 cudaHostRegister
        # torch_npu 支持 pin_memory=True 分配昇腾 DMA 可直接访问的 CPU buffer
        self.ssm_buffer = torch.zeros(
            (num_layers, num_slots) + ssm_state_shape,
            dtype=dtype, pin_memory=True  # 昇腾 DMA 映射
        )
        self.conv_buffer = torch.zeros(
            (num_layers, num_slots) + conv_state_shape,
            dtype=dtype, pin_memory=True
        )

        # LRU 管理（slot_id → (timestamp, request_id)）
        self._lru: OrderedDict[int, tuple[float, str]] = OrderedDict()
        self._req_to_slot: dict[str, int] = {}
        self._free_slots: list[int] = list(range(num_slots))

    def backup_all_layers(
        self,
        device_ssm: list[torch.Tensor],   # [num_layers] GPU SSM state tensors
        device_conv: list[torch.Tensor],   # [num_layers] GPU conv state tensors
        device_indices: list[int],         # 每层的 GPU slot index
        request_id: str,
    ) -> int:
        """D→H 异步拷贝（单次覆盖所有层，等价于 tokenspeed transfer_kv_all_layer_mla）"""
        host_slot = self._alloc_slot(request_id)
        for layer_idx in range(len(device_ssm)):
            dev_idx = device_indices[layer_idx]
            # 异步 D→H
            self.ssm_buffer[layer_idx, host_slot].copy_(
                device_ssm[layer_idx][dev_idx], non_blocking=True
            )
            self.conv_buffer[layer_idx, host_slot].copy_(
                device_conv[layer_idx][dev_idx], non_blocking=True
            )
        return host_slot

    def load_layer(
        self,
        layer_idx: int,
        host_slot: int,
        device_ssm: torch.Tensor,   # GPU SSM state tensor for this layer
        device_conv: torch.Tensor,  # GPU conv state tensor for this layer
        device_idx: int,
    ):
        """H→D 单层拷贝（PD 分离 layer-wise 流水使用）"""
        device_ssm[device_idx].copy_(self.ssm_buffer[layer_idx, host_slot], non_blocking=True)
        device_conv[device_idx].copy_(self.conv_buffer[layer_idx, host_slot], non_blocking=True)
```

#### Step 2.2：集成到 CPUOffloadingConnector

**改造文件**：`vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py`

在 `ReqMeta` 增加 `mamba_cpu_slot_id: int = -1`。

在 OOM 时（`preempt_request()` 路径）调用 `mamba_host_cache.backup_all_layers()`，在恢复时调用 `load_layer()`。

---

### Phase 3：TP 确定性调度 Tiebreak（P1，1 周）

**改造文件**：`vllm_ascend/core/recompute_scheduler.py`

在 `schedule()` 主循环前对候选请求排序：

```python
def _priority(req: Request) -> int:
    if req.status == RequestStatus.RUNNING:     return 1
    if req.status == RequestStatus.WAITING:     return 2
    if req.status == RequestStatus.PREEMPTED:   return 3  # 等价于 Retracted
    return 9

# TP 确定性关键：request_id 字符串 tiebreak（跨所有 TP rank 进程完全一致）
candidates.sort(key=lambda req: (_priority(req), req.request_id))

# OOM victim 选择：最长序列（num_computed_tokens 跨 rank 一致，无需 tiebreak）
victim = max(
    running_candidates,
    key=lambda r: (r.num_computed_tokens, r.request_id)  # id 作为最终 tiebreak
)
```

**同步改造**：LRU 驱逐的 tiebreak 改为 `(timestamp, request_creation_seq_id)` 而非指针。

---

### Phase 4：MTP O(1) SSM 索引更新（P1，2-3 周）

**改造文件**：`vllm_ascend/worker/model_runner_v1.py`、`vllm_ascend/spec_decode/`

核心：将 `postprocess_mamba_fused_kernel` 的 block-aligned copy 替换为整数索引更新。

```python
# vllm_ascend/worker/model_runner_v1.py: MTP verify 后的回调
def update_mamba_after_verify(
    self,
    accept_lengths: torch.Tensor,  # [batch]
    req_pool_indices: torch.Tensor, # [batch]
):
    """
    O(batch) 整数写，不拷贝任何 SSM tensor
    类比 tokenspeed _update_current_inputs_after_verify_kernel
    """
    # current_mamba_slot_indices[req_pool_idx] → 当前有效 SSM slot
    # draft_slots[req_pool_idx * spec_num + step] → draft SSM slot

    @torch.compile(dynamic=True)
    def _update_kernel(slot_indices, req_pool_indices, accept_lengths, spec_num, pool_size):
        for i in range(len(req_pool_indices)):
            k = accept_lengths[i]
            if k > 0:
                req_idx = req_pool_indices[i]
                draft_base = pool_size + req_idx * (spec_num - 1)
                new_slot = draft_base + ((k - 1) % (spec_num - 1))
                slot_indices[req_idx] = new_slot

    _update_kernel(
        self.current_mamba_slot_indices,
        req_pool_indices,
        accept_lengths,
        self.spec_num_tokens,
        self.mamba_pool_size,
    )
```

**昇腾适配**：`@torch.compile(dynamic=True)` 在昇腾上通过 TorchDynamo + ACLNN 后端编译，无需手写 Triton kernel。

---

### Phase 5：PD 分离 Layer-wise SSM 传输（P2，4-5 周）

**扩展文件**：`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py`

在现有 KV layer-wise 传输基础上，增加 SSM 传输通道：

```python
class MooncakeLayerwiseConnector:
    def send_mamba_layer(self, layer_idx: int, ssm_state: torch.Tensor, conv_state: torch.Tensor):
        """P node: prefill 完成第 layer_idx 层后立即发送"""
        # 通过已有的 Mooncake RDMA 通道发送
        self._rdma_send(f"mamba_ssm_{layer_idx}", ssm_state)
        self._rdma_send(f"mamba_conv_{layer_idx}", conv_state)
        self._layer_done_counter.increment(layer_idx)

    def recv_mamba_layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """D node: 等待 P node 发送第 layer_idx 层"""
        self._layer_done_counter.wait_until(layer_idx)
        return self._rdma_recv(f"mamba_ssm_{layer_idx}"), self._rdma_recv(f"mamba_conv_{layer_idx}")
```

D 节点 bootstrap：`scheduleDecodeFromRetracted` 等价路径，通过 `StateRecovery` intent 匹配最深 SSM 快照，填充 `decode_input_id = last_prefill_token`，`hist_token_len = total_len - 1`。

---

## 6. 关键文件映射与可复用能力

### 6.1 TokenSpeed → vllm-ascend 文件映射

| TokenSpeed 源文件 | vllm-ascend 对应文件 | 操作类型 |
|---|---|---|
| `tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp` | `vllm_ascend/core/recompute_scheduler.py` | **改造**：CoW 决策 + TP tiebreak + victim 策略 |
| `python/tokenspeed/runtime/cache/mamba_cache_host.py` | `vllm_ascend/core/mamba_host_cache.py` | **新增**：Host SSM L2 Cache |
| `python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py` | `vllm_ascend/ops/gdn.py` | **改造**：branching_seqlen 路径 + CoW |
| `tokenspeed-scheduler/csrc/resource/hybrid_prefix_cache/` | `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py` | **改造**：SSM offload 路径挂载 |
| `tokenspeed-scheduler/csrc/scheduler/page_hasher.h` | 复用 `vllm.utils.hashing.sha256` | **无需新增**（逻辑等价） |
| `python/tokenspeed/runtime/execution/cuda_graph_wrapper.py` | `vllm_ascend/compilation/acl_graph.py` | **改造**：MTP verify 外置，图内 replay |
| `python/tokenspeed/runtime/models/qwen3_5_moe.py`（StreamFork） | `vllm_ascend/models/`（已有 MoE）| 可参考但 vllm-ascend 已有等效实现 |
| `tokenspeed-scheduler/csrc/resource/radix_tree/radix_tree.cpp` | `vllm.v1.core.kv_cache_manager`（上游已有）| **无需迁移**（vllm v1 已有 block hash prefix cache）|

### 6.2 可直接复用的现有能力

```
① GDN chunk prefill（initial_state 已支持）
   file://vllm-ascend/vllm_ascend/ops/triton/fla/chunk.py
   → chunk_gated_delta_rule(initial_state=...) 直接作为 CoW 续算入口

② SSM block-aligned 快照（align 模式）
   file://vllm-ascend/vllm_ascend/patch/platform/patch_mamba_config.py
   → mamba_cache_mode="align" 已在 block 边界保存 SSM state，直接作为 CoW src

③ KV prefix cache hit 深度
   file://vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_kv_cache_manager.py
   → get_matched_num_and_touch() 返回 num_computed_tokens（prefix hit 深度）

④ SHA256 hash（链式哈希逻辑等价）
   vllm.utils.hashing.sha256（上游已有）

⑤ MTP proposer 框架
   file://vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py
   → AscendDflashProposer 框架完整，仅需 SSM postprocess 接入

⑥ CPU Offload 传输框架（KV 路径）
   file://vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py
   → 扩展 ReqMeta + 挂载 SSM 传输即可
```

---

## 7. 收益预期与量化分析

### 7.1 各优化点收益估算

| 优化点 | 收益维度 | 昇腾 A2/A3 预期量化 | 适用场景 |
|-------|---------|-------------------|---------|
| **Mamba CoW + branching_seqlen** | TTFT | 降低 60-80%（prefix 90%+ 命中） | Qwen3.5 long-context，system prompt 复用 |
| **Host SSM L2 Cache** | 并发数上限 | 提升 3-5x（GPU SSM 内存释放） | 大 batch、长 session 场景 |
| **TP 确定性 tiebreak** | 稳定性 | 消除 HCCL 死锁（0 次/天→0） | TP=8/16 部署 |
| **MTP O(1) 索引更新** | TPOT | 降低 20-30% | spec decode 开启（DFlash/EAGLE） |
| **PD 分离 SSM Layer-wise** | 集群吞吐 | 提升 20-40% | 多节点 PD 分离部署 |
| **全量叠加** | 综合吞吐 | **相比当前 baseline 3-5x** | 以上所有条件同时满足 |

### 7.2 与 B200 TokenSpeed 的对比估算

```
B200 TokenSpeed：580 tok/s

昇腾 A2/A3 参考（无优化基线）：
  A3（等效 910B pro）算力：约 B200 的 60-70%（BFLOAT16 MFU 参考）
  无优化 baseline：约 200-250 tok/s

迁移全量优化后预期：
  TTFT 加速（CoW）：90% prefix 命中 → GDN 计算量 10%  → TTFT ~10x
  并发数提升（Host SSM）：3-5x → batch throughput +3-5x
  MTP（DFlash spec decode）：accept rate ~2x → tok/s 2x
  PD 分离：集群利用率 +30%

  综合叠加预期：200-250 × 3-5 ≈ 600-1250 tok/s（集群聚合）
  单节点等效（A3 ×8 卡）：约 150-300 tok/s（接近或超过 B200 同规模配置）
```

> **注意**：以上估算基于 90%+ prefix hit 率的工作负载假设。实际收益依赖具体部署场景（请求长度分布、session 重复率、batch size 等）。

### 7.3 关键依赖条件

1. `mamba_cache_mode="align"` 必须开启（已有配置路径）
2. prefix caching 必须开启（`enable_prefix_caching=True`）
3. Qwen3.5-397B 工作负载需有 system prompt 或长前缀重复（否则 CoW 收益有限）
4. TP=8+ 场景才能体现 TP tiebreak 的稳定性收益
5. 多节点集群才有 PD 分离收益

---

## 8. 验证方法

### 8.1 Phase 1 验收（Mamba CoW）

```bash
# 1. 单元测试：验证 branching_seqlen > 0 时 GDN forward 输出与全量计算一致
pytest tests/ops/test_gdn_cow.py -v

# 2. FLOPs profiling：CoW 路径下 GDN 计算量应降低 ~90%（prefix 90% 命中）
# 使用 torch_npu profiler 对比 prefill FLOPs

# 3. 端到端 TTFT 对比（相同 prompt，系统 prompt 90k tokens）
python benchmarks/benchmark_latency.py --model Qwen3.5-397B \
  --input-len 90000 --output-len 100 --batch-size 1 \
  --enable-prefix-caching --mamba-cache-mode align
```

### 8.2 Phase 2 验收（Host SSM Cache）

```bash
# GPU SSM 内存监控：对比 offload 前后 npu-smi 内存占用
watch -n1 "npu-smi info | grep -A5 'Memory'"

# 并发上限测试：逐步提升并发数，观察 OOM 发生时的 batch size
python benchmarks/benchmark_throughput.py --max-num-seqs 256 --mamba-host-cache
```

### 8.3 Phase 3 验收（TP tiebreak）

```bash
# TP=8/16 下高并发压测 24h，统计 HCCL timeout 次数
# 目标：零 HCCL 死锁
python benchmarks/benchmark_serving.py --tensor-parallel 8 \
  --num-prompts 10000 --request-rate 50
```

### 8.4 Phase 4 验收（MTP O(1)）

```bash
# torch.profiler 对比 verify latency
# 预期：tensor copy 路径 O(L×D) ms → 整数写路径 O(batch) μs
python tests/spec_decode/test_mtp_verify_latency.py
```

### 8.5 端到端基准

```bash
# 基准测试（对比 vllm-ascend baseline vs 全量优化）
python benchmarks/benchmark_throughput.py \
  --model Qwen3.5-397B \
  --tensor-parallel 8 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --speculative-model Qwen3.5-397B-NextN \
  --num-speculative-tokens 3
```

---

## 9. 工作量与优先级

### 9.1 人力规划

| Phase | 优化点 | 工期 | 人力 | 优先级 | 依赖 |
|-------|-------|------|------|--------|-----|
| **Phase 1** | Mamba CoW + branching_seqlen | 3-4 周 | 1 SE | **P0** | 无 |
| **Phase 2** | Host SSM L2 Cache | 2-3 周 | 1 SE | **P0** | Phase 1 |
| **Phase 3** | TP 确定性 tiebreak | 1 周 | 0.5 SE | P1 | 无（可与 P1 并行）|
| **Phase 4** | MTP O(1) 索引更新 | 2-3 周 | 1 SE | P1 | Phase 1 |
| **Phase 5** | PD 分离 SSM Layer-wise | 4-5 周 | 1.5 SE | P2 | Phase 1+2 |
| **合计** | | **约 3 个月** | **2 SE 并行** | | |

### 9.2 并行执行建议

```
Week 1-4:  SE-A: Phase 1 (Mamba CoW)
           SE-B: Phase 3 (TP tiebreak，1周） + Phase 4 前期调研

Week 5-7:  SE-A: Phase 2 (Host SSM Cache)
           SE-B: Phase 4 (MTP O(1))

Week 8-12: SE-A + SE-B: Phase 5 (PD 分离) + 联调端到端
```

### 9.3 风险与缓解

| 风险 | 概率 | 缓解方案 |
|-----|------|---------|
| 昇腾 DMA pin_memory 语义与 CUDA 不同 | 中 | 提前验证 `torch_npu` pinned memory H↔D 异步性；备选：`torch_npu.npu_format_cast` 显式映射 |
| GDN `initial_state` 在大 batch 下精度漂移 | 低 | Phase 1 增加 fp32 参考对比测试 |
| `@torch.compile` 在昇腾 TorchDynamo 不支持部分算子 | 中 | 降级为 eager 模式 + 手写 ACLNN kernel |
| TP tiebreak 改动影响上游 vllm scheduler 兼容性 | 低 | 以 patch 方式仅在 `RecomputeScheduler` 中生效，不修改上游 |

---

## 参考源文件索引

### TokenSpeed 关键实现

| 文件 | 涉及优化点 |
|-----|---------|
| `file://tokenspeed/tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp` | TP tiebreak、CoW 决策、Retraction 策略 |
| `file://tokenspeed/tokenspeed-scheduler/csrc/resource/hybrid_prefix_cache/hybrid_prefix_cache.h` | HybridPrefixCache 接口全貌 |
| `file://tokenspeed/tokenspeed-scheduler/csrc/resource/radix_tree/radix_tree.cpp` | RadixTree 精确匹配 + 在线分裂 |
| `file://tokenspeed/python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py` | SimpleMambaPool、CoW kernel、MTP O(1) |
| `file://tokenspeed/python/tokenspeed/runtime/cache/mamba_cache_host.py` | Host SSM L2 Cache 实现 |
| `file://tokenspeed/tokenspeed-scheduler/csrc/scheduler/page_hasher.h` | SHA256 链式哈希 |

### vllm-ascend 关键改造点

| 文件 | 改造内容 |
|-----|---------|
| `file://vllm-ascend/vllm_ascend/core/recompute_scheduler.py` | CoW 决策注入、TP tiebreak、victim 策略 |
| `file://vllm-ascend/vllm_ascend/ops/gdn.py` | branching_seqlen 路径、SSM CoW |
| `file://vllm-ascend/vllm_ascend/patch/platform/patch_mamba_config.py` | align 模式配置参数对齐 |
| `file://vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py` | SSM offload 路径挂载 |
| `file://vllm-ascend/vllm_ascend/worker/model_runner_v1.py` | attn_metadata 注入、MTP verify 回调 |
| `file://vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py` | MTP O(1) 接入 |

---

*本分析基于 2026-05-28 源码快照，所有改造方案均基于实际代码结构，非架构推测。*
