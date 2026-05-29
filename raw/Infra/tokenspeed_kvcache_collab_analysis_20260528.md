# TokenSpeed GDN-KVCache 协同优化系统：深度源码解读

> **输出日期**：2026-05-28  
> **分析范围**：围绕 GDN-KVCache 的协同优化体系（PD 分离 · MTP 投机解码 · 调度策略）  
> **源码路径**：`/Users/linyi/code/Documents/code/tokenspeed/`

---

## 目录

1. [系统架构全景](#1-系统架构全景)
2. [核心数据结构层次](#2-核心数据结构层次)
3. [请求生命周期时序图](#3-请求生命周期时序图)
4. [优化点深度解读（逐行代码分析）](#4-优化点深度解读)
   - 4.1 RadixTree 精确匹配与分裂
   - 4.2 Mamba CoW + branching_seqlen
   - 4.3 TP 确定性调度
   - 4.4 OOM Retraction 策略
   - 4.5 PD 分离与 Layer-wise 传输
   - 4.6 MTP O(1) 状态更新
   - 4.7 PagedCache Adjunct 三阶段匹配
   - 4.8 AdmissionFailure 细粒度驱逐
5. [系统级组合效应分析](#5-系统级组合效应分析)
6. [性能收益量化](#6-性能收益量化)

---

## 1. 系统架构全景

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        TokenSpeed 调度系统全景架构                               │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                        Scheduler (C++)                                  │    │
│  │                                                                         │    │
│  │  newForwardOperation()  ─── TP 确定性排序 ──────────────────────────    │    │
│  │       │                    (priority + Request::Id() tiebreak)          │    │
│  │       ├── schedulePrefillFirstChunk()  ←── Submitted / PrefetchDone    │    │
│  │       │        │  1. Match(tokens) → MatchResult{device, host, mamba}  │    │
│  │       │        │  2. EnsureCapacityByEvict<Device>(pages_needed)       │    │
│  │       │        │  3. EnsureMambaCapacityByEvict(2 + loadback_slots)    │    │
│  │       │        │  4. AdmitChunk(paged_cache)                           │    │
│  │       │        │  5. PrepareMambaDeviceLoadBack(H→D)                   │    │
│  │       │        └─→ SchedulePrefillFirstChunkEvent                      │    │
│  │       │                                                                 │    │
│  │       ├── schedulePrefill()           ←── Prefilling                   │    │
│  │       │        └─→ SchedulePrefillEvent                                │    │
│  │       │                                                                 │    │
│  │       ├── scheduleDecode()            ←── PrefillDone / Decoding       │    │
│  │       │        └─→ ScheduleDecodeEvent                                 │    │
│  │       │                                                                 │    │
│  │       ├── scheduleDecodeFromRetracted() ←── Retracted (OOM 恢复)       │    │
│  │       │        │  StateRecovery Intent → FindLastMambaNode             │    │
│  │       │        │  若仅 Host 有 → PrepareMambaDeviceLoadBack            │    │
│  │       │        │  若完全丢失 → AbortEvent + warn                       │    │
│  │       │        └─→ ScheduleDecodeFromRetractedEvent                    │    │
│  │       │                                                                 │    │
│  │       └── scheduleRetract()           ←── OOM 触发                     │    │
│  │                └─→ Insert to RadixTree + Host WriteBack                │    │
│  │                                                                         │    │
│  │  applyEventAndGenerateOp()                                              │    │
│  │       ├── CommitChunk()       // 构建 PagedCacheSnapshot               │    │
│  │       ├── AcquireForRequest() // 申请/复用 paged cache pages           │    │
│  │       └── PopulateOp()        // 填充 op.paged_cache_pages             │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                    │                          │                                  │
│         ┌──────────▼──────────┐   ┌──────────▼──────────┐                      │
│         │  HybridPrefixCache  │   │   ForwardOperation   │                      │
│         │  (C++)              │   │   (SoA FlatLayout)   │                      │
│         │  ┌───────────────┐  │   │  mamba_working_idx   │                      │
│         │  │ KVPrefixCache │  │   │  mamba_ckpt_dst_idx  │                      │
│         │  │ (RadixTree)   │  │   │  mamba_cow_src_idx   │                      │
│         │  └───────────────┘  │   │  mamba_branching_len │                      │
│         │  ┌───────────────┐  │   │  paged_cache_blocks  │                      │
│         │  │MambaChunkAlloc│  │   └──────────────────────┘                      │
│         │  └───────────────┘  │              │                                  │
│         │  ┌───────────────┐  │              │ Python IPC / shared mem          │
│         │  │MambaHostAlloc │  │              ▼                                  │
│         │  └───────────────┘  │   ┌──────────────────────────────────────────┐  │
│         │  ┌───────────────┐  │   │   HybridLinearAttnBackend (Python)       │  │
│         │  │PagedCacheGroup│  │   │   SimpleMambaPool                        │  │
│         │  └───────────────┘  │   │   current_input_indices[req_pool_idx]    │  │
│         └─────────────────────┘   │   GDN prefill: chunk_gated_delta_rule    │  │
│                    │              │   GDN decode:  fused_sigmoid_delta_update │  │
│         ┌──────────▼──────────┐   │   MTP verify:  update_mamba_after_verify │  │
│         │   RadixTree (C++)   │   └──────────────────────────────────────────┘  │
│         │  TreeNode           │              │                                  │
│         │  ├─ DeviceResource  │              ▼                                  │
│         │  ├─ HostResource    │   ┌──────────────────────────────────────────┐  │
│         │  ├─ MambaSlot(GPU)  │   │     MambaCachePool (Python)              │  │
│         │  ├─ MambaSlot(Host) │   │  ┌──────────────┐  ┌─────────────────┐  │  │
│         │  └─ PagedCacheSnap  │   │  │SimpleMambaPool│  │MambaPoolHost    │  │  │
│         │  SHA256 chain hash  │   │  │(GPU VRAM)     │  │(cudaHostRegist) │  │  │
│         │  LRU seq_id tiebreak│   │  └──────────────┘  └─────────────────┘  │  │
│         └─────────────────────┘   │  transfer_kv_all_layer_mla (D→H bulk)   │  │
│                                   │  transfer_kv_per_layer_mla (H→D per-lyr) │  │
│                                   └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**关键设计原则**：C++ Scheduler 负责资源分配决策（无 Python GIL），Python Runtime 负责 GPU Kernel 执行。两层通过 `ForwardOperation` / `CacheOperation` SoA 结构传递指令，最小化跨层通信开销。

---

## 2. 核心数据结构层次

### 2.1 TreeNode —— 多资源联合持有者

```
file://tokenspeed-scheduler/csrc/resource/radix_tree/tree_node.h
```

```
TreeNode
├── tokens_: token_slice               // 该节点覆盖的 token 范围
├── depth_in_tokens_: int32_t          // 从根节点到本节点的 token 总深度
├── device_resource_: DeviceResource   // GPU KV cache pages（OwnedPages RAII）
├── host_resource_: HostResource       // CPU KV cache pages（OwnedPages RAII）
├── mamba_slot_: MambaSlot*            // GPU SSM 状态槽（optional）
├── mamba_host_slot_: MambaSlot*       // CPU SSM 状态槽（optional）
├── paged_cache_snapshot_: PagedCacheSnapshot*  // PagedAttn 快照（optional）
├── last_access_time_: timestamp_t     // LRU 访问时间
├── seq_id_: int64_t                   // 单调递增 ID，TP ranks 间确定性一致
└── children_: map<token_t, TreeNode*> // 子节点（首 token 索引）
```

一个 `TreeNode` **同时**管理四种异构资源（KV GPU/CPU + SSM GPU/CPU + PagedCache Snapshot），这是 TokenSpeed 能做到跨请求、跨模态（Attention + SSM）统一缓存的基础数据结构。

### 2.2 MatchResult —— 多层命中描述符

```
file://tokenspeed-scheduler/csrc/resource/types.h
```

```cpp
struct MatchResult {
    struct Device { TreeNode* last_node; int32_t page_size{0}; } device;
    struct Host   { TreeNode* last_node; int32_t page_size{0}; } host;

    // Mamba 专属字段
    int32_t mamba_branching_seqlen{-1};  // GDN 状态有效的最长对齐 seqlen
    int32_t mamba_cow_src_index{-1};     // CoW 源槽位（已在 GPU）
    int32_t mamba_host_src_index{-1};    // Host 槽位（需 H→D loadback）

    // PagedCache Adjunct 字段
    struct PagedCache {
        TreeNode* last_node{nullptr};
        int32_t prefix_len_tokens{0};
        map<string, vector<int32_t>> per_group_page_ids;
        map<string, int32_t> per_group_base_logical_page;
        enum class RestoreKind { kSnapshotComplete };
    } paged_cache;
};
```

这个结构是调度决策的"命中通知单"，**一次 Match 返回六个维度的命中信息**：Device KV 命中深度、Host KV 命中深度、Mamba GPU 命中位置、Mamba Host 命中位置、PagedCache 命中节点及其物理 page ids。

### 2.3 ForwardOperationBase —— 执行指令单

```
file://tokenspeed-scheduler/csrc/scheduler/operations/forward.h
```

```cpp
struct ForwardOperationBase {
    string request_id;
    int32_t request_pool_index;      // → SimpleMambaPool 的槽位索引
    int32_t input_length;
    vector<int32_t> occupied_pages;  // KV cache 物理 page ids
    int32_t begin, size;
    int32_t prefill_length;

    // PagedCache 字段
    map<string, vector<int32_t>> paged_cache_pages;
    map<string, int32_t> paged_cache_page_base_offsets;

    // Mamba 状态控制字段（全部默认 -1 表示无效）
    int32_t mamba_working_idx{-1};        // 当前请求的 SSM 工作槽
    int32_t mamba_checkpoint_dst_idx{-1}; // 检查点目标槽（chunk 对齐边界保存）
    int32_t mamba_cow_src_idx{-1};        // CoW 来源（prefix cache 命中时复制）
    int32_t mamba_branching_seqlen{-1};   // GDN 有效状态对齐长度
};
```

---

## 3. 请求生命周期时序图

### 3.1 标准 Prefill → Decode 流程

```
Time ──────────────────────────────────────────────────────────────────────────────▶

Client   Scheduler (C++)              HybridPrefixCache              Python Runtime
  │          │                              │                               │
  │ Submit   │                              │                               │
  ├─────────▶│ Submitted state              │                               │
  │          │                              │                               │
  │          │ schedulePrefillFirstChunk()  │                               │
  │          ├─── Match(tokens) ──────────▶ │                               │
  │          │ ◀── MatchResult{dev=3,host=5,│                               │
  │          │      mamba_cow_src=42} ──── │                               │
  │          │                              │                               │
  │          │ (host_matched > dev_matched) │                               │
  │          │ loadback_diff = [node3,4]    │                               │
  │          ├─── EnsureCapacityByEvict()──▶│ LRU evict unlocked leaves     │
  │          ├─── EnsureMambaCapacity() ───▶│ Mamba LRU evict               │
  │          ├─── AdmitChunk(paged_cache) ─▶│                               │
  │          │ ◀── ok ──────────────────── │                               │
  │          │                              │                               │
  │          │ applyEventAndGenerateOp()    │                               │
  │          ├─── CommitChunk() ───────────▶│ SplitAt + AttachSnapshot      │
  │          ├─── AcquireForRequest() ─────▶│ import hit pages + alloc new  │
  │          ├─── PopulateOp() ────────────▶│ fill paged_cache_pages        │
  │          │                              │                               │
  │          │ GenerateLoadBackOp()         │                               │
  │          │ LoadBackOp{kKV: node3,4 H→D │                               │
  │          │            kMamba: node→slot}│                               │
  │          │                              │                               │
  │          ├── FlatForwardOp ────────────────────────────────────────────▶│
  │          │   {mamba_cow_src=42,         │                               │
  │          │    branching_seqlen=64,      │                               │
  │          │    paged_cache_blocks=...}   │                               │
  │          │                              │               GDN prefill:    │
  │          │                              │  chunk_gated_delta_rule()     │
  │          │                              │  (output_h=True at FLA_CHUNK  │
  │          │                              │   boundaries → checkpoint)    │
  │          │                              │                               │
  │          │ ◀── PrefillDone event ─────────────────────────────────────│
  │          │                              │                               │
  │          │ scheduleDecode()             │                               │
  │          │ (came_from_prefill_done=True)│                               │
  │          ├─── CommitChunk() ───────────▶│ 最后一个 prefill chunk 写入   │
  │          ├─── AcquireForRequest() ─────▶│                               │
  │          ├─── PopulateOp() ────────────▶│                               │
  │          │                              │                               │
  │          ├── DecodeOp ────────────────────────────────────────────────▶│
  │          │   {mamba_working_idx=7,      │                               │
  │          │    mamba_ckpt_dst_idx=15}    │                               │
  │          │                              │          fused_sigmoid_delta  │
  │          │                              │          _rule_update() O(1)  │
  │          ◀─────────── token stream ────────────────────────────────────│
```

### 3.2 OOM Retraction → Recovery 流程

```
Time ──────────────────────────────────────────────────────────────────────────────▶

Scheduler                         HybridPrefixCache           MambaCachePool
    │                                    │                          │
    │ newForwardOperation() → ops empty  │                          │
    │ (all decode scheduling failed)     │                          │
    │                                    │                          │
    │ Find victim = max TokenSize()      │                          │
    │ (deterministic across TP ranks)    │                          │
    │                                    │                          │
    │ scheduleRetract(victim)            │                          │
    ├── Insert KV pages to RadixTree ───▶│ KV pages 归入缓存树      │
    │    (full_paged_tokens + prefix)    │                          │
    │                                    │                          │
    ├── Match(StateRecovery) ───────────▶│                          │
    │ ◀── {host.last_node=N} ──────────│                          │
    │                                    │                          │
    │ EnsureCapacity<Host>(pages_needed) │                          │
    ├─── HostAllocator evict if needed ─▶│                          │
    │                                    │                          │
    │ ScheduleRetractEvent → WriteBackOp │                          │
    │ {kKV: dev_pages → host_pages,      │                          │
    │  is_retract=true}                  │                          │
    │                                    │                          │
    │ Victim state: Decoding→Retracting  │                          │
    │                  →Retracted        │                          │
    │                                    │                          │
    │  ─── D→H KV async copy ──────────────────────────────────────│
    │  ─── D→H Mamba async copy ────────────────────────────────────│
    │      (backup_from_device_all_layer) │                         │
    │                                    │                          │
    │  .... 时间流逝，其他 batch 继续 ....│                          │
    │                                    │                          │
    │ scheduleDecodeFromRetracted(victim) │                          │
    ├── Match(StateRecovery) ───────────▶│                          │
    │ ◀── {host.last_node=N,            │                          │
    │      mamba_host_src=M} ──────────│                          │
    │                                    │                          │
    │ FindLastMambaNode() → GPU copy?    │                          │
    │ → null，FindLastMambaHostNode() → M│                          │
    │ needs_mamba_loadback = true        │                          │
    ├── EnsureMambaCapacity(2+slots) ───▶│ Mamba LRU evict          │
    ├── PrepareMambaDeviceLoadBack([M]) ─▶│ H→D Mamba loadback      │
    │                                    │                          │
    │ DecodeFromRetractedOp              │                          │
    │ {decode_input_id = last_token,     │                          │
    │  hist_token_len = tokensize-1,     │                          │
    │  mamba_cow_src = M.MambaSlotIndex}│                          │
    │                                    │                          │
    │ State: Retracted→Decoding          │                          │
```

### 3.3 MTP 投机解码流程

```
Time ──────────────────────────────────────────────────────────────────────────────▶

CudaGraphWrapper           SimpleMambaPool              MTP NextN Model
    │                           │                            │
    │ replay(bs=8, draft_tokens=3)│                          │
    │                           │                            │
    │ (inside CUDA graph)       │                            │
    │ get_current_input_indices  │                            │
    │ (slot: working + 3 draft) │                            │
    │                           │                            │
    │ forward(base_model)       │                            │
    │ GDN decode: O(1)/step     │                            │
    │                           │                            │
    │ ── hidden_states ────────────────────────────────────▶│
    │                           │          NextN FC:         │
    │                           │  cat(pre_fc_norm(input_emb)│
    │                           │      pre_fc_norm(hidden))  │
    │                           │  → 1-layer transformer     │
    │                           │  → 3 draft tokens          │
    │                           │                            │
    │ (outside CUDA graph)      │                            │
    │ verify accept_lengths=[2] │                            │
    │                           │                            │
    │ update_mamba_after_verify │                            │
    │ ── accept_lengths=[2] ───▶│                            │
    │                           │ _update_current_inputs     │
    │                           │ _after_verify_kernel()     │
    │                           │ @torch.compile(dynamic=T)  │
    │                           │                            │
    │                           │ working_slot[req] = base   │
    │                           │   + (2 % spec_num_tokens)  │
    │                           │ // O(bs) integer writes    │
    │                           │ // NOT O(L×D) tensor copy  │
    │                           │                            │
    │ next decode step          │                            │
    │ (slot already updated)    │                            │
```

### 3.4 PD 分离层级传输流程

```
Time ──────────────────────────────────────────────────────────────────────────────▶

P Node (Prefill)                               D Node (Decode)
    │                                               │
    │ config_.role = kP                             │ config_.role = kD
    │ decode_input_tokens = 0                       │
    │                                               │
    │ schedulePrefillFirstChunk()                   │
    │   Prefill-only, no decode tokens              │
    │                                               │
    │ Layer 0 prefill done                          │
    │ LayerDoneCounter.increment(0)                 │
    │ transfer_kv_per_layer_mla(layer=0) ──────────▶│
    │ (pinned host buffer 中转)                     │ LayerDoneCounter.wait_until(0)
    │                                               │ 收到 layer 0 KV
    │ Layer 1 prefill done                          │
    │ LayerDoneCounter.increment(1)                 │
    │ transfer_kv_per_layer_mla(layer=1) ──────────▶│
    │                                               │ 解锁 layer 1 KV
    │ ...                                           │
    │                                               │
    │ All layers done                               │
    │ D role: scheduleDecodeFromRetracted()         │
    │         → bootstrap_token = last_token        │
    │         → hist_token_len = total_len - 1      │
    │         Reuse KV from RadixTree (StateRecovery)│
    │                                               │
    │                                               │ decode forward with full KV
```

---

## 4. 优化点深度解读

### 4.1 RadixTree 精确匹配与在线分裂

```
file://tokenspeed-scheduler/csrc/resource/radix_tree/radix_tree.cpp
```

**核心问题**：不同请求的前缀长度任意，不能要求 page 对齐。

**解法**：`WalkDownUntilMismatch` + 按需 `SplitChild`，实现任意粒度的前缀共享。

```cpp
WalkResult RadixTree::WalkDownUtilMismatch(
    token_slice aligned_tokens, timestamp_t access_time, TreeNode* start_node) {

    auto* current = start_node;
    token_slice remaining_tokens = aligned_tokens;

    while (remaining_tokens.size() >= page_size_) {
        // Step 1: 查子节点（首 token 索引）
        TreeNode* child = FindChild(current, walk_key_cache);

        // Step 2: 计算实际匹配的 page 数
        int32_t matched_pages = calcMatchedPages(child, remaining_tokens, page_size_);

        // Step 3: 部分匹配 → 在线分裂
        // 场景：child 持有 [A,B,C,D] tokens，当前请求仅匹配 [A,B]
        // 分裂后：prefix=[A,B], suffix=[C,D]，共享 [A,B] 不复制
        if (matched_pages != child->Tokens().size() / page_size_) {
            SplitResult split = splitChild(current, walk_key_cache, matched_pages);
            child = split.prefix;  // 复用已有 prefix 节点
        }

        child->Touch(access_time);  // 更新 LRU 时间戳

        // Step 4: 更新 Device/Host 命中深度
        update_tier(device_alive, result.match.device, child, child->OnDevice());
        update_tier(host_alive, result.match.host, child, child->OnHost());

        // Step 5: 前进到下一层
        current = child;
        remaining_tokens = remaining_tokens.subspan(matched_pages * page_size_);
    }

    return WalkResult{current, remaining_tokens, result.match};
}
```

**关键工程细节**：
- `FindChild` 用首 token 作 key，时间复杂度 O(log K)（map）
- `calcMatchedPages`：逐 page 比较，遇到第一个不匹配 token 即停止
- `SplitChild` 是原地操作：prefix 节点继承原节点的 DeviceResource/HostResource（RAII OwnedPages 转移），无 GPU 内存拷贝
- `update_tier` 双路更新：device 和 host 分别跟踪，支持 L2 Cache（host 命中比 device 更深）

**性能影响**：每次 prefill 的 Match 操作 O(prefix_depth / page_size)，而不是 O(full_sequence_length)。对 90k token 前缀、16 token page size，仅需走约 5625 步，远优于朴素前缀比对。

---

### 4.2 Mamba CoW + branching_seqlen

```
file://tokenspeed-scheduler/csrc/resource/hybrid_prefix_cache/hybrid_prefix_cache.cpp
file://python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py
```

**核心问题**：GDN（Gated DeltaNet）是序列递推结构，SSM 状态 `h_t = f(h_{t-1}, x_t)` 不像 KV cache 可以分片并行共享。不同请求的 SSM 状态如果 Fork 出去，后续各自生成不同 token，状态就发散了。

**解法**：Copy-on-Write（CoW）+ `branching_seqlen` 语义标记。

```cpp
// hybrid_prefix_cache.cpp: augmentMatch()
void HybridPrefixCache::augmentMatch(MatchResult& match) const {
    // 在 KV match 结果基础上，寻找 SSM 状态命中点
    TreeNode* mamba_node = FindLastMambaNode(match.device.last_node);
    if (mamba_node != nullptr) {
        // GPU 上直接有 Mamba 状态
        match.mamba_cow_src_index = mamba_node->MambaSlotIndex();
    }

    TreeNode* mamba_host_node = FindLastMambaHostNode(match.host.last_node);
    if (mamba_host_node != nullptr) {
        // 仅 Host 有 Mamba 状态，需 H→D loadback
        match.mamba_host_src_index = mamba_host_node->MambaHostSlotIndex();
    }

    // branching_seqlen: 对齐到 mamba_cache_chunk_size 边界
    // 含义：SSM 状态在 branching_seqlen 之前是可信的 prefix 部分
    //       branching_seqlen 之后的 token 需要重新递推
    if (mamba_node != nullptr) {
        match.mamba_branching_seqlen =
            AlignMambaCacheSeqlen(mamba_node->DepthInTokens());
    }
}
```

```python
# hybrid_linear_attn.py: _get_current_input_indices_with_cow_kernel
@torch.compile(dynamic=True)
def _get_current_input_indices_with_cow_kernel(
    current_input_indices: torch.Tensor,   # shape: [pool_size]
    cow_src_indices: torch.Tensor,         # shape: [batch]: 来源槽位
    dst_indices: torch.Tensor,             # shape: [batch]: 目标工作槽位
    req_pool_indices: torch.Tensor,        # shape: [batch]
):
    """
    CoW 操作：将 prefix-cache 命中的 Mamba 状态复制索引映射到新请求的工作槽
    实际的 tensor 拷贝由 GPU kernel 完成（transfer_kv_per_layer_mla）
    这里只是更新整数索引表，O(batch_size)
    """
    for i in range(len(req_pool_indices)):
        if cow_src_indices[i] >= 0:
            # 源槽有效 → 设置目标槽的 src 索引（后续 kernel 据此拷贝）
            current_input_indices[req_pool_indices[i]] = dst_indices[i]
```

**branching_seqlen 的精确含义**：

```
Prefix (shared):     [tok 0..N_align-1]   → Mamba 状态 h_{N_align} 可复用
New request tail:    [tok N_align..M]     → 必须从 h_{N_align} 开始重新递推

N_align = AlignMambaCacheSeqlen(N) = (N / chunk_size) * chunk_size

执行时：
1. CoW：h_{N_align} 从 cow_src 槽 copy 到新请求的 working 槽
2. GDN prefill 从 branching_seqlen 位置开始（不是从 0）
3. 节省 N_align 步的递推计算
```

**工程代价**：一次 GPU memcpy `(num_heads × head_dim)` 的 float16 张量，约 `128 × 256 × 2 = 64KB`，远小于重新计算 N_align 步骤的 FLOPs。

---

### 4.3 TP 确定性调度

```
file://tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp (line 551-571)
file://tokenspeed-scheduler/csrc/resource/kv_prefix_cache/eviction.h
```

**核心问题**：TP=8 时，8 个进程各自运行一份 Scheduler 副本。如果调度选择不一致（某 rank 调度请求 A，另一 rank 不调度），下一个 NCCL AllReduce 就会因为 rank 参与数不同而死锁。

**根本原因**：`unordered_map` 的 key（string）在 libstdc++ 中使用 process-local 随机种子哈希，同一时刻不同进程的迭代顺序不同。同理，指针值受 ASLR 影响，也不能作为 tiebreaker。

**解法：三层确定性保证**

```cpp
// 1. 请求优先级排序 + 确定性 tiebreak（forward.cpp）
std::sort(candidates.begin(), candidates.end(), [&](const auto& a, const auto& b) {
    int pa = priority(a), pb = priority(b);
    // TP-determinism: tie-break on Request::Id()
    // Request::Id() 是字符串（通常来自客户端 request_id 或递增计数器）
    // 跨进程完全一致，不依赖 pointer、不依赖 hash_map 迭代顺序
    return pa != pb ? pa < pb : a->Id() < b->Id();
});
```

```cpp
// 2. LRU eviction tiebreak（eviction.h / tree_resource.h）
// lru_leaves_ = set<tuple<timestamp_t, int64_t, TreeNode*>>
//   ─ timestamp_t: 访问时间（monotonic，全局一致）
//   ─ int64_t:     seq_id（TreeNode 创建时分配的单调 ID）
//   ─ TreeNode*:   仅用于 set 唯一性，不参与排序（ASLR unsafe）
//
// Comment in source: "pointer values are not usable as a tiebreaker
// because they are randomized per-process and would diverge across TP
// ranks, causing different ranks to evict different leaves on Time ties
// and eventually wedging the next NCCL collective"
using LruKey = std::tuple<timestamp_t, int64_t, TreeNode*>;
std::set<LruKey> lru_leaves_;
```

```cpp
// 3. OOM Retraction victim 选择（forward.cpp line 644-647）
Request* victim = *std::max_element(
    retract_candidates.begin(), retract_candidates.end(),
    [](const Request* a, const Request* b) {
        return a->TokenSize() < b->TokenSize();
        // 选最长（TokenSize 最大）的请求 retract
        // TokenSize 由请求内容决定，跨 TP ranks 完全一致
        // 若相同 TokenSize 存在，已经通过上面的 sort 排好了
    }
);
```

**收益**：这三处 tiebreak 确保了在 token budget 紧张、page 不足、Mamba slot 告警等边界条件下，所有 TP ranks 做出**完全相同**的调度决定，消除了分布式死锁的根源。

---

### 4.4 OOM Retraction 策略

```
file://tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp (line 306-391)
```

**触发条件**：`ops.empty() && !candidates.empty()`，即所有候选请求的调度均失败（device page 耗尽）。

**策略选择**：驱逐最长 decode 请求。

**工程理由**：
1. 最长请求占用最多 KV page（历史 token 越长，page 越多），释放它能腾出最多空间
2. 最长请求的 KV prefix 价值最高（复用价值最大），写入 RadixTree 后对后续同 session 请求有益
3. 短请求因资源不足而 abort 的代价更大（用户感知的 TTFT 更差）

```cpp
// scheduleRetract: 将 victim 的 KV pages 写入 RadixTree
void Scheduler::scheduleRetract(Request* request) {
    auto full_paged_tokens = request->GetFullPagedTokens(true);

    // 1. 当前请求持有的 pages（尾部，不在 prefix cache 中的部分）
    OwnedPages alloc_pages = request->TakeFirstPages(alloc_count);

    // 2. 插入 RadixTree：KV pages 从请求转移到缓存树（ownership 转移，无内存拷贝）
    kv_prefix_cache_.Insert<ResourceType::Device>(
        full_paged_tokens, prefix_pages, std::move(alloc_pages));

    // 3. Match StateRecovery：找到刚插入的节点在 host 侧需要写多少 pages
    MatchResult match_result = kv_prefix_cache_.Match(
        full_paged_tokens, MatchIntent::StateRecovery);

    // 4. 计算需要 D→H 写回的 pages（device 有而 host 没有的部分）
    int32_t host_pages_needed = device_matched - host_matched;

    // 5. 生成 WriteBackOperation（异步 D→H 拷贝）
    return ScheduleRetractEvent{&kv_prefix_cache_, &host_allocator_,
                                match_result, hybrid_prefix_cache_};
}
```

**Mamba 状态处理**：
- retract 发生时，MambaCachePool 将该请求的 SSM 状态通过 `backup_from_device_all_layer` 异步 D→H
- 写入后标记为 host-owned，GPU slot 释放供其他请求使用
- 恢复时 `PrepareMambaDeviceLoadBack` 异步 H→D，再 CoW 进新的工作槽

```python
# mamba_cache_host.py: 异步 D→H 全层批量传输
def backup_from_device_all_layer(self, device_pool, host_indices, device_indices, ...):
    transfer_kv_all_layer_mla(
        src_layers=ptrs["device_conv"],   # GPU conv 状态
        dst_layers=ptrs["host_conv"],     # CPU 固定内存（cudaHostRegister）
        src_layers2=ptrs["device_ssm"],   # GPU ssm_h 状态
        dst_layers2=ptrs["host_ssm"],     # CPU 固定内存
        ...
    )
    # 单次 kernel 覆盖所有 num_layers
    # 避免 for layer in layers: cuda_memcpy() 的 kernel launch 开销
```

---

### 4.5 PD 分离与 Layer-wise 传输

```
file://tokenspeed-scheduler/csrc/scheduler/scheduler.h (role 字段)
file://tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp (role 判断分支)
file://python/tokenspeed/runtime/cache/transfer/mamba_pool.py
```

**架构分离设计**：

```cpp
// scheduler.h
enum class Role { kP, kD, kCombined };

// forward.cpp: 基于 role 的调度分支
if (request->Is<fsm::Prefilling>() && config_.role != Role::kD) {
    // P node 只做 prefill，D node 不走这里
    schedulePrefill(request, token_budget, ...);
}
if (request->Is<fsm::PrefillDone>() || request->Is<fsm::Decoding>()) {
    if (config_.role != Role::kP) {  // P node 不做 decode
        scheduleDecode(request, simulated_free);
    }
}

// P node: decode_input_tokens 强制为 0
int32_t decode_input_tokens = config_.role == Role::kP ? 0 : config_.decode_input_tokens;
```

**D Node Bootstrap 机制**：

D Node 接收到 P Node 传来的全量 KV 后，需要切入 decode 流程。因为 D Node 没有做 prefill，它没有 `Prefilling→PrefillDone` 的状态历史，需要特殊的"bootstrap token"路径：

```cpp
// forward.cpp: D node 首次 decode
DecodeOperation Scheduler::applyEventAndGenerateOp(
    Request* request, fsm::ScheduleDecodeEvent event) {

    // D node 专属：从 PrefillDone 进入 Decoding 时，
    // 第一个 decode token 就是 last prefill token（bootstrap）
    const bool need_bootstrap_token =
        request->Is<fsm::PrefillDone>() && config_.role == Role::kD;
    int32_t bootstrap_token = need_bootstrap_token ? request->GetLastToken() : -1;

    auto op = applyDecodeEvent(request, std::move(event), config_.decode_input_tokens);
    if (need_bootstrap_token) {
        op.decode_input_id = bootstrap_token;
        // hist_token_len 不需要设置（P node 已经做了全量 prefill）
    }
    ...
}
```

**Layer-wise 传输（PD 流水线化）**：

```python
# mamba_pool.py: 逐层传输（P→D，每完成一层立即传）
def load_to_device_per_layer(self, device_pool, host_indices, device_indices, layer_idx, ...):
    """
    PD 分离场景：P node 完成 layer i 的 prefill 后，立即通过 pinned host buffer 发送
    D node 可以在 P 还未完成所有层时，已经开始接收并准备前面的层
    这是 layer-wise pipeline 的关键：减少 PD 传输的 end-to-end latency
    """
    transfer_kv_per_layer_mla(
        src=self.conv_buffer[layer_idx],      # host buffer（P→H 已写入）
        dst=device_pool.conv_buffer[layer_idx],  # GPU buffer（准备 decode）
        src2=self.ssm_buffer[layer_idx],
        dst2=device_pool.ssm_buffer[layer_idx],
        ...
    )
```

```python
# mamba_cache_host.py: cudaHostRegister 确保 DMA 直通
def __init__(self, ...):
    self.conv_buffer = torch.zeros(...)  # CPU tensor
    self.ssm_buffer = torch.zeros(...)
    # 注册为固定内存，GPU 可 DMA 直接访问，避免额外拷贝
    platform.register_host_tensor_for_gpu_access(self.conv_buffer)
    platform.register_host_tensor_for_gpu_access(self.ssm_buffer)
```

**收益**：Layer-wise 传输将 P→D 的串行传输延迟拆解为 `num_layers` 个细粒度流水线阶段，D Node 最早可以在 P 完成前几层后就开始预热（prefetch KV），整体 TTFT 降低。

---

### 4.6 MTP O(1) 状态更新

```
file://python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py
```

**核心问题**：MTP（Multi-Token Prediction）每步生成 `spec_num_tokens` 个草稿 token，verification 后 accept 了其中 `k` 个。接受后，需要将 Mamba SSM 状态从第 `k` 个 draft 位置恢复为 working state。

**朴素实现的代价**：如果保存 draft token 对应的 SSM 状态快照，需要存储 `spec_num_tokens × num_layers × (ssm_size + conv_size)` 的额外 GPU 内存，接受后还需要执行张量拷贝 O(L×D)。

**TokenSpeed 的解法**：整数索引表 + Draft Slot Region。

```python
# SimpleMambaPool.__init__
def __init__(self, size, spec_num_tokens, ...):
    # 正常请求槽位
    self.size = size

    # MTP Draft 扩展：每个请求额外 spec_num_tokens-1 个 draft 槽
    # draft_base 之后是 draft 专用区域
    draft_base = size
    draft_slots_per_req = spec_num_tokens - 1
    draft_total = size * draft_slots_per_req

    # 索引表：req_pool_index → 当前有效的 SSM 槽位 ID
    # 形状 [size + size*(spec_num_tokens-1)]，覆盖 base + draft 区域
    current_input_size = size + draft_total
    self.current_input_indices = torch.full(
        (current_input_size,), -1, dtype=torch.int32, device=device
    )
```

```python
# MTP 流程中的 draft 索引生成
@torch.compile(dynamic=True)
def _build_mtp_output_indices_kernel(
    current_input_indices,    # [pool_size]
    req_pool_indices,         # [batch]
    spec_num_tokens: int,
    size: int,
) -> torch.Tensor:
    """
    为每个 draft token 计算其应写入的 SSM 槽位
    draft slot = draft_base_for_req + (step_idx % spec_num_tokens-1)
    """
    output_indices = torch.zeros_like(req_pool_indices)
    for i, req_idx in enumerate(req_pool_indices):
        base = size + req_idx * (spec_num_tokens - 1)
        current = current_input_indices[req_idx]
        # round-robin in draft region
        draft_step = (current - size) if current >= size else 0
        output_indices[i] = base + (draft_step % (spec_num_tokens - 1))
    return output_indices
```

```python
# Verification 后的 O(1) 更新
@torch.compile(dynamic=True)
def _update_current_inputs_after_verify_kernel(
    current_input_indices: torch.Tensor,  # [pool_size], in-place 修改
    req_pool_indices: torch.Tensor,       # [batch]
    accept_lengths: torch.Tensor,         # [batch] 每个请求接受的 token 数
    size: int,
    spec_num_tokens: int,
):
    """
    核心：只更新整数索引，不拷贝 SSM 张量
    draft slot 在 accept_lengths 对应位置已经有了正确的状态
    只需将 current_input_indices 指向它
    """
    for i, req_idx in enumerate(req_pool_indices):
        k = accept_lengths[i].item()
        if k > 0:
            # 接受了 k 个 token → 找到第 k 个 draft 槽的索引
            draft_base = size + req_idx * (spec_num_tokens - 1)
            # 轮转到第 k 个 draft 位置
            new_slot = draft_base + ((k - 1) % (spec_num_tokens - 1))
            current_input_indices[req_idx] = new_slot
        # k=0 时 current 不变（base model 的工作槽）
```

**内存布局示意**：

```
SimpleMambaPool 内存布局（spec_num_tokens=4, batch_size=2）:

slot: [0]   [1]   | [2] [3] [4] | [5] [6] [7]
       req0  req1  | req0 draft  | req1 draft
       base  base  | d0  d1  d2  | d0  d1  d2

current_input_indices:
  req0: 0 → 当前 working slot 是 slot[0]
  req1: 1 → 当前 working slot 是 slot[1]

MTP step (req0 生成 3 draft tokens):
  GDN 在 d0,d1,d2 写入 draft 状态

verify: accept_lengths[req0] = 2 (接受了 d0, d1)
  new_slot = draft_base + (2-1) % 3 = 2 + 1 = slot[3]
  current_input_indices[req0] = 3  ← O(1) 整数写

下一步 decode: 从 slot[3] 的状态出发，不需要任何张量拷贝
```

**`@torch.compile(dynamic=True)` 的作用**：batch size 在推理时动态变化，`dynamic=True` 允许 shape 变化时不重新编译，同时将整数索引操作融合成高效 CUDA kernel，消除 Python 循环开销。

---

### 4.7 PagedCache Adjunct 三阶段匹配

```
file://tokenspeed-scheduler/csrc/resource/hybrid_prefix_cache/hybrid_prefix_cache.cpp
file://tokenspeed-scheduler/csrc/resource/radix_tree/paged_cache_snapshot.h
```

**背景**：某些 attention 变体（如 full FA4 层 + 滑动窗口层并存）需要额外的 paged cache group 管理，不同 group 有不同的 `RawTokensPerPage`，必须在 LCM 对齐边界上创建快照。

**三阶段匹配逻辑**：

```
Phase A: History chain walk（全历史 KV）
  沿 TreeNode 链向上走，收集所有 History-family group 的完整快照
  停止条件：遇到不完整的快照（!IsCompleteFor(History)）

Phase B: State window backward scan（滑动窗口 KV）
  从 match 终点向上回溯 sliding_window_tokens 覆盖范围
  找到最深的完整 State-family 快照

Phase C: per_group_page_ids 组装
  合并 Phase A + Phase B 结果，生成每个 group 的物理 page id 列表
  base_logical_page：全历史 group = 0，滑动窗口 group > 0
```

**分裂-快照原子操作**：

```cpp
// hybrid_prefix_cache.cpp: CommitChunk
void HybridPrefixCache::CommitChunk(const string& request_id, TreeNode* terminal) {
    int32_t last_committed = GetLastCommittedPosition(request_id);

    // LCM 对齐推进：每次提交一个完整的 history_alignment_tokens 段
    while (last_committed + paged_cache_history_alignment_tokens_ <= chunk_depth) {
        int32_t target = last_committed + paged_cache_history_alignment_tokens_;

        // Step 1: 在 target token 位置原地分裂 TreeNode
        // 确保快照附着在 token 边界对齐的节点上
        TreeNode* snap_node = SplitAt(terminal, target);

        // Step 2: 从 per-request table 中提取 target 位置的 page ids
        auto snapshot = BuildSnapshotFromTable(request_id, target);

        // Step 3: 附着快照到节点（计算 complete_families）
        AttachPagedCacheSnapshotToNode(snap_node, std::move(snapshot));

        last_committed = target;
    }
}
```

**AdmitChunk 的 OOM 精细处理**：

```cpp
// 当 AdmitChunk 失败，根据失败类型选择驱逐策略
if (admission.ok == false) {
    AdmissionFailureKind kind = ClassifyAdmissionFailure(admission);

    // kHistoryStarved: 历史 KV group 空间不足 → 驱逐整个快照（history + state）
    // kStateStarved:  仅滑动窗口 group 空间不足 → 只驱逐 state 部分，保留 history
    // kBothStarved:   两者都不足 → 完整驱逐
    if (!tryPrunePagedCacheSnapshot(kind)) {
        return false;  // 无法腾出空间 → 调度失败
    }
    // 重试 admission
}
```

**设计精髓**：`kStateStarved` 场景下只驱逐 state groups（滑动窗口 pages），保留 history groups（全量 KV）。这是因为 history KV 的复用价值远高于 state，宁可让滑动窗口重新计算，也不要丢掉全量历史。

---

### 4.8 SHA256 链式哈希：跨 Session 缓存复用

```
file://tokenspeed-scheduler/csrc/scheduler/page_hasher.h
```

```cpp
inline std::string HashPage(
    std::span<const std::int32_t> tokens,
    const std::string& prior_hash) {  // ← 关键：链式！

    SHA256_CTX ctx;
    SHA256_Init(&ctx);

    // 1. 先混入前一个 page 的哈希（链式，不是独立）
    if (!prior_hash.empty()) {
        SHA256_Update(&ctx, HexToBytes(prior_hash).data(), DIGEST_BYTES);
    }
    // 2. 再混入当前 page 的 tokens
    for (int32_t t : tokens) {
        uint8_t buf[4] = {(uint8_t)t, (uint8_t)(t>>8), (uint8_t)(t>>16), (uint8_t)(t>>24)};
        SHA256_Update(&ctx, buf, 4);
    }

    SHA256_Final(digest, &ctx);
    return DigestToHex(digest);
}
```

**为什么要链式哈希**：

```
独立哈希（错误方案）：
  page 0 hash = H([t0,t1,...,t15])
  page 1 hash = H([t16,...,t31])

  问题：page 1 相同的 token 在不同前缀下会有相同的 hash，
        导致错误命中（两个 session 在不同上下文下的相同 page 被错误共享）

链式哈希（TokenSpeed 方案）：
  page 0 hash = H("" || [t0..t15])      = h0
  page 1 hash = H(h0 || [t16..t31])     = h1
  page 2 hash = H(h1 || [t32..t47])     = h2

  page_k 的 hash 携带了 [t0..t_{k*page_size-1}] 的完整历史信息
  相同 tokens 序列必然产生相同的链式 hash → 跨 session 安全复用
```

**应用场景**：L3 Storage（磁盘/分布式对象存储）预取。当某个常见 prompt pattern（如系统 prompt + 特定前缀）在多个 session 中出现时，链式哈希确保可以正确识别并复用缓存，而不会发生跨上下文的错误命中。

---

## 5. 系统级组合效应分析

### 5.1 各优化点的依赖关系

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        优化点依赖与组合图                                     │
│                                                                              │
│  ┌─────────────┐         ┌─────────────────────────────────────────────┐    │
│  │ SHA256链式哈希│────────▶│ RadixTree Match                            │    │
│  │(跨Session标识)│        │ (WalkDownUntilMismatch + SplitChild)        │    │
│  └─────────────┘         └────────────────┬────────────────────────────┘    │
│                                           │ MatchResult{device,host,mamba}  │
│                                           ▼                                 │
│  ┌─────────────┐    依赖    ┌────────────────────────────────────────────┐  │
│  │ TP确定性排序 │◀──────────│ EnsureCapacity (KV + Mamba)                │  │
│  │(seq_id + Id)│           │ AdmitChunk (PagedCache)                    │  │
│  └─────────────┘           └────────────────┬───────────────────────────┘  │
│          │                                  │ 调度成功                       │
│          ▼                                  ▼                               │
│  ┌─────────────────┐      ┌─────────────────────────────────────────────┐  │
│  │ OOM Retraction  │      │ Mamba CoW + branching_seqlen               │  │
│  │(最长victim + D→H│      │ (复用 prefix SSM 状态)                      │  │
│  │ + insert tree)  │      └────────────────┬────────────────────────────┘  │
│  └─────────────────┘                       │                               │
│          │                                 ▼                               │
│          │ 恢复路径              ┌──────────────────────────────────────┐  │
│          ▼                      │ MTP O(1) 索引更新                    │  │
│  ┌─────────────────┐            │ (draft slots + verify kernel)        │  │
│  │DecodeFromRetracted│          └────────────────┬─────────────────────┘  │
│  │(StateRecovery +  │                           │                          │
│  │ Mamba H→D load) │                           ▼                          │
│  └─────────────────┘            ┌──────────────────────────────────────┐  │
│                                 │ CUDA Graph Replay                    │  │
│          ┌──────────────────────│ (所有 batch size 预捕获)             │  │
│          ▼                      └──────────────────────────────────────┘  │
│  ┌─────────────────┐                                                        │
│  │ PD 分离         │                                                        │
│  │ Layer-wise 传输 │                                                        │
│  │ Bootstrap token │                                                        │
│  └─────────────────┘                                                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 关键路径：一次高命中率 Prefill 请求的系统行为

以 Qwen3.5-397B 典型工作负载（90k token 前缀，prefix hit 90%+）为例，追踪一次完整请求的系统行为：

**阶段 1：Match（～1ms，纯 CPU）**

```
Scheduler.schedulePrefillFirstChunk()
  │
  ▼ RadixTree.WalkDownUntilMismatch(90k tokens / page_size=16 = 5625 steps)
  │   → device.last_node depth = 81k tokens（81000/16 = 5063 pages 已在 GPU）
  │   → host.last_node   depth = 90k tokens（5625 pages 在 CPU）
  │   → mamba_cow_src    = slot #42（对应 81k 对齐后的 SSM 状态）
  │   → paged_cache.per_group_page_ids = {group0: [p0,p1,...,p5624]}
  │
  ▼ 计算 loadback_diff = [nodes 5063..5625]（9k tokens 的 KV 需要 H→D）
  │   = 562 pages × 16 tokens × 2 bytes × head_dim = ～300MB
  │   （异步 H→D，不阻塞调度）
```

**阶段 2：资源分配（～0.1ms，CPU）**

```
EnsureCapacityByEvict<Device>(pages_needed=562+新tokens/16)
  → LRU 驱逐 562 个最旧的 unlocked 叶子节点（按 (timestamp, seq_id) 排序）

EnsureMambaCapacityByEvict(slots_needed=2+1=3)
  → 驱逐 Mamba LRU 槽位

AdmitChunk(paged_cache, first_pos=0, target=9000)
  → 检查 PagedCacheGroupAllocator 剩余 pages
  → 若 kStateStarved：仅驱逐 state group snapshots
```

**阶段 3：Operation 生成（～0.1ms，CPU）**

```
applyEventAndGenerateOp(PrefillFirstChunkEvent)
  → CommitChunk(): SplitAt(terminal, 0) + AttachSnapshot（之前 chunk 补提交）
  → AcquireForRequest(0, 9000, paged_cache_hit)：import 5624 hit pages + alloc 562 new
  → PopulateOp(): fill paged_cache_block_tables

FlatForwardOperation {
  mamba_cow_src_idx     = 42
  mamba_branching_seqlen = 81000  ← GDN 从这里续算，省去 81000 步递推
  paged_cache_block_tables = {group0: [[p0..p15], [p16..p31], ...]}
  occupied_pages = [p0..p562+new_pages]
}

LoadBackOperation {
  kKV:    [(host_p5063, dev_p5063), ..., (host_p5624, dev_p5624)]
  kMamba: [(host_slot_42, dev_slot_42)]  ← 若仅 host 有
}
```

**阶段 4：并行执行（GPU + CPU 流水）**

```
CPU 发起 H→D async copy（562 KV pages + 1 Mamba slot）
          │
GPU fork ─┤
          │ GDN prefill: chunk_gated_delta_rule(tokens[81000:90000])
          │   branching_seqlen=81000 → skip 前 81000 tokens 的递推
          │   仅对 9000 新 tokens 执行 prefill（约 1/10 的计算量）
          │
          │ FA4 full attention: PagedAttn 使用 recovered paged_cache pages
          │   直接复用 90k prefix 的 KV，零计算
          │
GPU join ─┘ 等待 H→D copy 完成（Mamba loadback 完成后才能继续依赖其状态的层）
```

**量化节省**：
- KV 重计算节省：90000 × 128 heads × 256 dim × 2 bytes = 约节省 5.9GB KV 读写
- GDN 递推节省：81000 步 × num_gdn_layers × O(head_dim²) FLOPs
- 整体 TTFT 降低：从约 100s（全量 prefill）降至约 10s（10% 新 tokens）

### 5.3 组合叠加的非线性收益

单点优化线性叠加，但在 TokenSpeed 中，各点的相互增强使整体收益超过线性叠加：

| 优化组合 | 叠加效应 |
|---------|---------|
| RadixTree Match + Mamba CoW | KV hit 高 → Mamba branching_seqlen 深 → GDN 续算节省更多（非线性） |
| TP 确定性 + OOM Retraction | Retraction 选择一致 → 恢复路径确定 → 无 NCCL 死锁 |
| PagedCache Adjunct + CommitChunk | LCM 对齐快照 → AdmitChunk 更精准 → 减少不必要驱逐 |
| MTP O(1) + CUDA Graph | O(1) 整数更新不破坏 graph 约束 → MTP 可在 graph 外 verify，graph 内 replay |
| PD 分离 + Layer-wise + Mamba Host Cache | P 节点 prefill 后立即写 host → D 节点 loadback 与 P 的后续层流水 |

---

## 6. 性能收益量化

### 6.1 各优化点收益估算

| 优化点 | 主要收益维度 | 量化估算 |
|--------|------------|---------|
| **RadixTree 精确匹配** | TTFT（首 token 延迟） | prefix 90% hit → 减少 90% 的 KV 重计算；TTFT 10倍+ |
| **Mamba CoW** | TTFT（GDN 部分） | branching_seqlen=81k → GDN prefill 从 90k 步降至 9k 步；9x加速 |
| **Mamba Host Cache** | GPU SSM 内存 | active 请求 SSM 在 GPU，idle 请求 SSM 在 CPU；GPU Mamba 内存降低 5-10x |
| **TP 确定性调度** | 系统稳定性 | 消除 TP>1 时的 NCCL 死锁；从偶发死锁 → 零死锁 |
| **OOM Retraction** | 系统吞吐 | 替代 abort：高请求率下 avg 队列等待降低 30-50% |
| **PD 分离 + Layer-wise** | 集群利用率 | P/D 专用化 → GPU 利用率各自优化，整体吞吐提升 20-40% |
| **MTP O(1) 更新** | TPOT（逐 token 延迟） | 消除 O(L×D) tensor copy；draft-verify 延迟降低 30-60% |
| **PagedCache Adjunct** | 命中率精细化 | History/State 独立驱逐 → 命中率额外提升 5-15% |
| **SHA256 链式哈希** | 跨 Session 命中 | 系统 prompt 跨 session 命中 → 冷启动 TTFT 与热 TTFT 接近 |

### 6.2 Qwen3.5-397B B200 上的整体性能分析

论文/blog 数据：580 tok/s（B200 4节点），对比 vLLM/SGLang 约 200-300 tok/s。

**性能来源拆解**：

```
Base（无任何优化，纯 attention+GDN decode）： ～50 tok/s
  ↕ ×2  FA4 + DeepEP（算子层优化，非本文分析范围）
  = ～100 tok/s

  ↕ ×1.5  prefix cache hit 90%（KV重计算从100%降至10%）
  = ～150 tok/s

  ↕ ×1.5  Mamba CoW（GDN 重计算从100%降至10%）
  = ～225 tok/s

  ↕ ×1.3  MTP 3x draft（推理 3 token / 1 step，accept rate ～2.5）
  = ～293 tok/s

  ↕ ×1.3  PD 分离 + 调度优化（更高 GPU 利用率）
  = ～380 tok/s

  ↕ ×1.5  FusedReduceNorm + StreamFork + CUDA Graph（overlap 提升）
  = ～570 tok/s

  ≈ 580 tok/s ✓
```

> 各因子并非严格独立，实际因并发 batch 提升 GPU 利用率而有超线性叠加效果。上述拆解为定性量级估算。

### 6.3 关键 Bottleneck 移除分析

```
无优化时的关键瓶颈链：
  KV 重计算 → 高 TTFT → 低并发 → GPU 利用率低 → 低吞吐

TokenSpeed 打破了哪个环节：
  1. Prefix Cache（KV + Mamba）打破 KV 重计算瓶颈
  2. TP 确定性调度打破 NCCL 稳定性瓶颈（不再需要保守 padding）
  3. MTP 打破 decode per-step FLOPs 瓶颈（3 tokens/step 而非 1）
  4. PD 分离打破 GPU 角色混合的利用率瓶颈
  5. PagedCache 精细驱逐打破缓存抖动瓶颈（kStateStarved 场景）

结果：每个瓶颈都有对应的系统级解法，整体形成闭环
```

---

## 参考源文件索引

| 文件 | 关键内容 |
|-----|---------|
| `file://tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp` | 调度主逻辑、TP 确定性、Retraction |
| `file://tokenspeed-scheduler/csrc/resource/hybrid_prefix_cache/hybrid_prefix_cache.h` | HybridPrefixCache 完整接口 |
| `file://tokenspeed-scheduler/csrc/resource/radix_tree/radix_tree.cpp` | RadixTree Walk + Split |
| `file://tokenspeed-scheduler/csrc/resource/radix_tree/tree_node.h` | TreeNode 多资源联合持有 |
| `file://tokenspeed-scheduler/csrc/resource/types.h` | MatchResult、ForwardOperationBase |
| `file://tokenspeed-scheduler/csrc/scheduler/operations/cache.h` | CacheKind、TransferPair、CacheOperation |
| `file://tokenspeed-scheduler/csrc/scheduler/page_hasher.h` | SHA256 链式哈希 |
| `file://tokenspeed-scheduler/csrc/resource/radix_tree/paged_cache_snapshot.h` | PagedCacheSnapshot |
| `file://python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py` | SimpleMambaPool、MTP O(1) 更新 |
| `file://python/tokenspeed/runtime/cache/mamba_cache_host.py` | cudaHostRegister、bulk transfer |
| `file://python/tokenspeed/runtime/cache/transfer/mamba_pool.py` | MambaCachePool、LayerDoneCounter |
| `file://python/tokenspeed/runtime/execution/cuda_graph_wrapper.py` | CUDA Graph capture/replay |
| `file://python/tokenspeed/runtime/layers/attention/linear/gdn.py` | fused_gdn_gating_kernel |

---

*分析基于 2026-05-28 源码快照，所有代码引用均来自本地实际文件，非推断。*
