# TokenSpeed Prefix Cache 90%+ 命中率深度分析

> 分析日期：2026-05-28  
> 源码根路径：`/Users/linyi/code/Documents/code/tokenspeed/`  
> 分析框架：**两层解构** — ①如何提升命中率；②如何把高命中率转化为推理性能收益

---

## 整体架构一句话

TokenSpeed 的 prefix cache 本质是一棵 **C++ RadixTree（token 粒度）** 上挂载了三类资源——KV pages（Device/Host 双层）+ **Mamba/GDN SSM state**（Device L1 + Host L2）+ **PagedCache snapshot**（for FA 层 state 组/history 组）——由调度器统一管理，供 Python 运行时零拷贝复用。

agentic 工作负载（50K 首轮 + 800 后续轮次）命中 90%+ 的根本原因：**工作负载本身高度重复**（相同 system prompt + tool call 结果反复出现）+ **系统对这类重复的感知和利用做到了极致**。

---

## 第一层：如何提升命中率

### 1.1 RadixTree：token 粒度的最长前缀匹配

📄 [radix\_tree.h](file:///Users/linyi/code/Documents/code/tokenspeed/tokenspeed-scheduler/csrc/resource/radix_tree/radix_tree.h) | [radix\_tree.cpp](file:///Users/linyi/code/Documents/code/tokenspeed/tokenspeed-scheduler/csrc/resource/radix_tree/radix_tree.cpp)

传统 KV cache 以 request 为粒度管理——请求结束就释放，下次请求必须全量 prefill。TokenSpeed 用 RadixTree 共享任意长度的公共前缀。

**关键数据结构**：

```cpp
// tree_node.h — 每个节点持有 token 序列 + KV/Mamba/PagedCache 资源
class TreeNode {
    token_vec_t tokens_;           // 该节点代表的 token 序列（page-aligned）
    std::size_t depth_in_tokens_;  // 从根到此节点的累计 token 数

    // 三类资源（可独立存在，形成多维前缀树）
    std::unique_ptr<DeviceResource> device_resource_;  // KV pages on GPU
    std::unique_ptr<HostResource>   host_resource_;    // KV pages on CPU
    std::unique_ptr<MambaSlot>      mamba_slot_;       // GDN/Mamba SSM state (GPU)
    std::unique_ptr<MambaSlot>      mamba_host_slot_;  // GDN/Mamba SSM state (CPU, L2)
    std::unique_ptr<PagedCacheSnapshot> paged_cache_snapshot_;  // FA 层额外 state

    timestamp_t last_access_time_;  // LRU eviction 排序键
    seq_id_t seq_id_;               // ASLR 安全的单调递增 id（跨 TP rank 一致驱逐）
};
```

**WalkDownUntilMismatch** — 每次请求进来时精确走到最深匹配节点：

```cpp
// radix_tree.cpp
WalkResult RadixTree::WalkDownUtilMismatch(token_slice aligned_tokens,
                                           TreeNode::timestamp_t access_time,
                                           TreeNode* start_node) {
    // 逐 page 匹配：取 page[0] 的 token 序列作为 key 查 children hash map
    while (result.remaining_tokens.size() >= page_size_) {
        walk_key_cache.assign(remaining, remaining + page_size_);
        TreeNode* child = FindChild(current, walk_key_cache);
        if (child == nullptr) break;

        // 精确计算匹配页数（子节点可能包含多页）
        std::int32_t matched_pages = calcMatchedPages(child, remaining_tokens, page_size_);
        if (matched_pages == 0) break;

        // 如果部分匹配：SplitChild 把节点分裂成 prefix + suffix
        if (matched_pages != child->Tokens().size() / page_size_) {
            SplitResult split = splitChild(current, walk_key_cache, matched_pages);
            child = split.prefix;
        }

        child->Touch(access_time);  // 更新 LRU 时间戳

        // 分别追踪 Device / Host 两层的最深命中节点
        update_tier(device_alive, result.match.device, child, child->OnDevice());
        update_tier(host_alive,   result.match.host,   child, child->OnHost());

        current = child;
        result.remaining_tokens = result.remaining_tokens.subspan(matched_pages * page_size_);
    }
    return result;
}
```

**命中率关键**：`SplitChild` 让节点在"部分匹配"时动态分裂——即使新请求只与历史请求有 3/4 的前缀重叠，仍能共享那 3/4。这是相比朴素 page-level hash 的核心优势。

### 1.2 SHA256 链式 page hash：跨会话内容寻址

📄 [page\_hasher.h](file:///Users/linyi/code/Documents/code/tokenspeed/tokenspeed-scheduler/csrc/scheduler/page_hasher.h)

```cpp
// SHA256(prior_hash + page_tokens) — 链式哈希，每页的 hash 包含历史
inline std::string HashPage(std::span<const std::int32_t> tokens,
                             const std::string& prior_hash) {
    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    if (!prior_hash.empty()) {
        // 把前一页的 hash 喂入当前 SHA256，形成 prefix-aware 的链式指纹
        auto prior_bytes = HexToBytes(prior_hash);
        SHA256_Update(&ctx, prior_bytes.data(), prior_bytes.size());
    }
    for (std::int32_t t : tokens) {
        uint8_t buf[4] = {(uint8_t)t, (uint8_t)(t>>8), (uint8_t)(t>>16), (uint8_t)(t>>24)};
        SHA256_Update(&ctx, buf, 4);
    }
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256_Final(digest, &ctx);
    return DigestToHex(digest);
}

// 批量计算一段 token_pages 的 hash 链
inline std::vector<std::string> ComputePagedHashes(
    const std::vector<std::span<const std::int32_t>>& token_pages,
    const std::string& prior) { ... }
```

**意义**：每个 page 的 hash 不只是该 page token 的 hash，而是**从根到此 page 的完整路径哈希**。相同 token 序列必然产生相同 hash，这使得：
- **跨会话复用**：不同 session 发来相同 system prompt，hash 一致，直接命中
- **防误碰撞**：不同前缀下相同 token 内容产生不同 hash（因链式传递 prior）

`TreeNode::block_hashes_`（uint64_t 版本）用于内存内快速比较，`page_hashes_`（string 版本）用于跨进程/持久化存储匹配。

### 1.3 HybridPrefixCache：KV + Mamba 统一管理

📄 [hybrid\_prefix\_cache.h](file:///Users/linyi/code/Documents/code/tokenspeed/tokenspeed-scheduler/csrc/resource/hybrid_prefix_cache/hybrid_prefix_cache.h) | [hybrid\_prefix\_cache.cpp](file:///Users/linyi/code/Documents/code/tokenspeed/tokenspeed-scheduler/csrc/resource/hybrid_prefix_cache/hybrid_prefix_cache.cpp)

Qwen3.5 是混合架构：FA 层有 KV cache，GDN 层有 SSM state。朴素做法只缓存 KV，每次请求都要全量 prefill 所有 GDN 层的 SSM state（O(seq_len) 递推）。TokenSpeed 把 Mamba/GDN state 也挂到 RadixTree 节点上：

```cpp
class HybridPrefixCache {
    KVPrefixCache& kv_prefix_cache_;         // 底层共享同一棵 RadixTree
    MambaChunkAllocator* mamba_allocator_;   // GDN SSM state GPU 分配器
    MambaHostAllocator* mamba_host_allocator_; // GDN SSM state CPU L2 分配器
    MambaEvictionManager mamba_eviction_manager_; // 独立的 Mamba LRU 管理

    // 页对齐一致性：Mamba state 必须挂在 page-aligned 的 TreeNode 上
    void InsertMamba(TreeNode* terminal_node, std::unique_ptr<MambaSlot> slot) {
        // 检查节点必须 page-aligned（与 FLA_CHUNK_SIZE 对齐）
        if (terminal_node->DepthInTokens() % page_size_ != 0) {
            throw std::logic_error("terminal node is not block-aligned");
        }
        terminal_node->AttachMamba(std::move(slot));
        mamba_eviction_manager_.TrackNode(terminal_node);
    }
};
```

**Match 增强**（`augmentMatch`）——KV match 完成后，向上查找最深的 Mamba 节点：

```cpp
void HybridPrefixCache::augmentMatch(MatchResult& match) const {
    TreeNode* kv_terminal   = match.device.last_node;  // KV 命中深度
    TreeNode* device_mamba  = FindLastMambaNode(kv_terminal);    // 沿父链找最深 Mamba (GPU)
    TreeNode* host_mamba    = FindLastMambaHostNode(match.host.last_node); // 找 CPU L2

    // 优先选更深的（device vs host）
    const bool prefer_host_mamba = host_mamba_depth > device_mamba_depth;

    if (device_mamba_node != nullptr) {
        if (!prefer_host_mamba) {
            match.mamba_cow_src_index = device_mamba_node->MambaSlotIndex(); // CoW 复用
        }
    }
    if (host_mamba_node != nullptr) {
        match.mamba_host_src_index = host_mamba_node->MambaHostSlotIndex(); // H→D 加载
    }

    // KV 深度 > Mamba 深度时：记录 branching_seqlen（告诉运行时从哪里开始 chunk replay）
    if (kv_depth > mamba_depth) {
        const int32_t aligned = AlignMambaCacheSeqlen(kv_depth * page_size);
        if (aligned > mamba_depth * page_size) {
            match.mamba_branching_seqlen = aligned; // GDN 需要从这里开始 replay
        }
    }
}
```

**这是命中率的第二层抓手**：KV 命中 → GDN state 也命中。agentic 场景中 50K token 的 system prompt，一旦 KV 命中，对应的 GDN SSM state 也可能命中（挂在同一 TreeNode 上），完全避免 GDN prefill 的 O(n) 递推。

### 1.4 Mamba L2（Host）：Device 容量不够也能命中

📄 [hybrid\_prefix\_cache.cpp](file:///Users/linyi/code/Documents/code/tokenspeed/tokenspeed-scheduler/csrc/resource/hybrid_prefix_cache/hybrid_prefix_cache.cpp)

GPU 内存有限，Mamba state 会被驱逐。TokenSpeed 实现了 Mamba state 的 Device→Host 写回（L2 cache），使命中范围扩展到主机内存：

```cpp
// D→H 写回：为一批 TreeNode 的 Mamba state 申请 host slot，触发异步传输
std::vector<TransferPair> HybridPrefixCache::PrepareMambaHostWriteBack(
    const std::vector<TreeNode*>& nodes) {
    std::vector<TransferPair> transfers;
    for (TreeNode* node : nodes) {
        // 只处理：有 device Mamba 但还没 host copy 的节点
        if (!node->HasMamba() || node->HasMambaOnHost()) continue;

        auto slot = mamba_host_allocator_->Allocate();
        const int32_t device_idx = node->MambaSlotIndex();
        const int32_t host_idx   = slot->Index();

        // pending_mamba_host_writebacks_：写回完成前标记为"进行中"
        pending_mamba_host_writebacks_.emplace(node, std::make_unique<MambaSlot>(*slot));
        transfers.push_back({CacheKind::kMamba, device_idx, host_idx});
    }
    return transfers;
}

// H→D 加载：prefix cache 命中 host Mamba，申请 device slot 触发加载
std::vector<TransferPair> HybridPrefixCache::PrepareMambaDeviceLoadBack(
    const std::vector<TreeNode*>& nodes) {
    for (TreeNode* node : nodes) {
        if (!node->HasMambaOnHost() || node->HasMamba()) continue;
        auto slot = mamba_allocator_->Allocate();
        const int32_t host_idx   = node->MambaHostSlotIndex();
        const int32_t device_idx = slot->Index();
        node->AttachMamba(std::make_unique<MambaSlot>(*slot));
        transfers.push_back({CacheKind::kMamba, host_idx, device_idx}); // H→D 方向
    }
    return transfers;
}

// Device copy 完成后自动降级（GPU 内存释放）
void HybridPrefixCache::DemoteIdleMambaDeviceCopiesPresentOnHost() {
    for (TreeNode* node : mamba_host_writeback_done_nodes_) {
        if (node->OnDevice() && node->Device().RefCount() != 0) continue; // 仍在用
        OnKVDeviceDemote(node); // 释放 device copy，保留 host copy
    }
}
```

**命中率意义**：Device L1 容量有限（TP8 每卡 ~80GB），但 Host L2 可以更大（CPU 内存通常 TB 级）。高频 system prompt 可以在 Device 上保留热副本，历史 session 降级到 Host。

### 1.5 LRU 驱逐：跨 TP rank 确定性，保护热数据

📄 [eviction.h](file:///Users/linyi/code/Documents/code/tokenspeed/tokenspeed-scheduler/csrc/resource/kv_prefix_cache/eviction.h) | [tree\_resource.h](file:///Users/linyi/code/Documents/code/tokenspeed/tokenspeed-scheduler/csrc/resource/radix_tree/tree_resource.h)

```cpp
// LRU 排序：(timestamp, seq_id, node*) — seq_id 是跨 TP rank 一致的确定性 tiebreaker
std::set<std::tuple<timestamp_t, std::int64_t, TreeNode*>> lru_leaves_;

// 关键说明（代码注释原文）：
// "pointer values are not usable as a tiebreaker because they are randomized
//  per-process and would diverge across TP ranks, causing different ranks to
//  evict different leaves on Time ties and eventually wedging the next NCCL collective"
// TP 8 个进程 ASLR 导致指针值不同 → 用单调 seq_id 代替

// 驱逐时跳过被锁定的节点（正在使用中）
std::vector<TreeNode*> ResourceManager<RType>::Evict(std::int32_t num_pages) {
    while (evicted < num_pages && !lru_leaves_.empty()) {
        auto leaf = lru_leaves_.begin();  // oldest first

        if (!GetResource<RType>(leaf).IsEvictable()) {
            // 活跃请求持有 → 放入 deferred，跳过本次不驱逐
            deferred_locked.push_back({ts, leaf});
            continue;
        }
        auto resource_ptr = leaf->DetachResource<RType>();
        // 驱逐后触发 eviction_callback_（Mamba/PagedCache 资源联动清理）
        if (eviction_callback_) eviction_callback_(leaf);
        // 父节点可能成为新的 leaf — 立刻加入 LRU
        updateLeaf(leaf->Parent());
    }
    // 恢复被延迟的 locked 节点，用 node->Time() 而非 saved ts（Touch 可能改变了时间）
    for (auto& [ts, node] : deferred_locked) {
        lru_leaves_.insert({node->Time(), node->SeqId(), node});
    }
}
```

**命中率关键**：
- `deferred_locked`：活跃请求的 KV 不被驱逐，避免正在使用的前缀被错误淘汰
- TP rank 一致驱逐：所有 TP worker 驱逐同一个 leaf，避免 NCCL 死锁
- `updateLeaf(parent)`：驱逐 leaf 后立刻将父节点加入 LRU 候选——子树完全消失后父节点才变成可驱逐 leaf（保护前缀完整性）

### 1.6 PagedCache Adjunct：FA 层 sliding window + history 快照

📄 [hybrid\_prefix\_cache.h](file:///Users/linyi/code/Documents/code/tokenspeed/tokenspeed-scheduler/csrc/resource/hybrid_prefix_cache/hybrid_prefix_cache.h) | [paged\_cache\_snapshot.h](file:///Users/linyi/code/Documents/code/tokenspeed/tokenspeed-scheduler/csrc/resource/radix_tree/paged_cache_snapshot.h)

Qwen3.5 的 FA 层（Blackwell FA4）支持两种 cache 组：

```cpp
// paged_cache_snapshot.h
struct PagedCacheSnapshot {
    std::int32_t prefix_len_tokens{0};
    std::map<std::string, PagedCacheGroupSnapshot> groups;
    // 按 family 追踪完整性
    std::set<PagedCacheGroupFamily> complete_families;
    // History: 全量 KV 历史（适合短序列 FA 层）
    // State:   滑动窗口 KV 状态（适合 SWA 层）
    bool IsCompleteFor(PagedCacheGroupFamily f) const { ... }
};
```

`CommitChunk`——每完成 `history_alignment_tokens`（各 group `RawTokensPerPage` 的 LCM）个 token，提交一个 snapshot 到对应 TreeNode：

```cpp
void HybridPrefixCache::CommitChunk(const std::string& request_id, TreeNode* terminal) {
    const int32_t lcm = paged_cache_history_alignment_tokens_;
    while (last_committed + lcm <= chunk_depth) {
        const int32_t target = last_committed + lcm;

        // SplitAt：确保 target 深度有一个精确对齐的 TreeNode（没有则动态分裂）
        TreeNode* attach_node = kv_prefix_cache_.GetRadixTree().SplitAt(terminal, target);

        // 对每个 required group，提交 History 或 State 类型 snapshot
        for (const auto& gid : required_groups) {
            auto& table = tables[gid];
            auto result = (cfg.family == History)
                ? table.CommitHistoryToSnapshot(target)   // 全量提交
                : table.CheckpointStateToSnapshot(target); // 只保留 sliding window
            snapshot->groups.emplace(gid, std::move(result));
        }
        AttachPagedCacheSnapshotToNode(attach_node, std::move(snapshot));
        last_committed = target;
    }
}
```

**Match 阶段增强**（`augmentMatchPagedCache`）：

```cpp
void HybridPrefixCache::augmentMatchPagedCache(MatchResult& match) const {
    // Phase A: History chain — 从根到叶，连续检查每 `align` token 对齐的节点是否有完整 History snapshot
    TreeNode* deepest_history = nullptr;
    std::int32_t expected_depth = align;
    for (TreeNode* n : path) {
        if (n->DepthInTokens() != expected_depth) break;
        if (!snap->IsCompleteFor(PagedCacheGroupFamily::History)) break;
        deepest_history = n;
        expected_depth += align;
    }

    // Phase B: State window — 反向找最深 D' 使得 trailing `segments_needed` 个节点都有完整 State
    for (int end_idx = history_chain.size()-1; end_idx >= 0; --end_idx) {
        const int start_idx = max(0, end_idx - segments_needed + 1);
        bool ok = all(history_chain[start_idx..end_idx], has_complete_State_snapshot);
        if (ok) { usable_node = history_chain[end_idx]; break; }
    }

    // Phase C: Per-group page ids 组装（History: 全链; State: trailing window slice）
    match.paged_cache.last_node = usable_node;
    match.paged_cache.prefix_len_tokens = usable_node->DepthInTokens();
    // 填充 per_group_page_ids，供运行时直接 attach 到 FA4 attention metadata
}
```

**命中率关键**：FA 层的 attention 不需要重算前缀，但 Blackwell FA4 需要 KV history + sliding window state 都完整才能复用。`AdmissionFailureKind` 分类精细驱逐策略：

```cpp
enum class AdmissionFailureKind {
    kNone,
    kHistoryStarved,  // 只缺 history → full cascade 驱逐
    kStateStarved,    // 只缺 state → 只驱逐 state groups（保留 history）
    kBothStarved      // 都缺 → full cascade
};
```

`tryPrunePagedCacheSnapshot(kStateStarved)` 只删 state groups 而保留 history，避免因为 state 不够而连带删掉 history 导致命中率大幅下降。

---

## 第二层：基于高命中率，拿到推理性能收益

### 2.1 KV Prefix 命中：跳过 prefill，直接 decode

**匹配结果到运行时的传递**：`MatchResult` 中的 `device.last_node` / `host.last_node` 的 `DepthInPage()` 给出命中的 page 数，调度器直接用这些 page id 作为 req 的初始 KV 状态：

```cpp
// types.h — MatchResult 传递给调度器，调度器转为 ForwardOperation
struct MatchResult {
    struct Device {
        TreeNode* last_node;   // 命中深度的节点
        int32_t page_size{0};
        int32_t DepthInPage() const;  // 返回命中的 page 数
    } device;
    // ...
};
```

命中的 page 直接复用，运行时的 `extend_with_prefix=True` + `extend_prefix_lens` 告知 attention 后端前多少 token 已有 KV，只对 `remaining_tokens` 做 prefill。

**收益**：50K token 首轮 system prompt，90%+ 命中时实际 prefill 只有 ~5K token（gap + new content），prefill FLOP 降低 ~10×。

### 2.2 Mamba CoW 命中：GDN state 直接复用

📄 [hybrid\_linear\_attn.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py)

```python
@staticmethod
@torch.compile(dynamic=True)
def _get_current_input_indices_with_cow_kernel(
    req_pool_indices, mamba_pool_indices,
    mamba_cow_src_indices,      # 来自 MatchResult.mamba_cow_src_index
    mamba_branching_seqlens,    # 来自 MatchResult.mamba_branching_seqlen
    current_input_indices,
    ...
):
    # mamba_cow_src_index != -1 时：把 src slot 的 SSM state copy 到新 dst slot
    # 然后从 branching_seqlen 处继续 GDN chunk replay
    # mamba_cow_src_index == -1 时：直接从 current_input_indices 读（普通路径）
```

`mamba_branching_seqlen`：KV 命中深度 > Mamba state 深度时（FA 层命中更深），告诉 GDN 运算从 `AlignMambaCacheSeqlen(kv_depth * page_size)` 处开始 chunk replay——只重算 Mamba 未命中的那段，而不是全量重跑。

**收益**：GDN prefill 是 O(seq_len) 的 chunked 计算。命中 90% 意味着 GDN prefill 从 50K tokens → ~5K tokens，约 10× 加速（叠加在 FA KV 命中收益之上）。

### 2.3 MTP × Prefix Cache 的乘法效应

MTP（3-4 draft tokens/step）和 prefix cache 的交互是 580 tok/s 的关键放大器：

```
命中后的 decode 阶段：
  每 step 主模型 forward bs 很小（agentic 多轮，后续轮次 800 token）
  → CUDA Graph decode 路径（batch_size ∈ [1..160]，精确命中 graph）
  → MTP 每 step 产出 ~3 token（accepted_length ≈ 3）
  → O(1) Mamba state 指针更新（update_current_inputs_after_verify）
  → 有效吞吐 = 实际 tok/s × MTP accept_rate ≈ tok/s × 3
```

prefix cache 的作用是大幅压缩 TTFT（首 token 时间），让 decode 阶段占比更高，而 decode 阶段 MTP 提速效果最明显。

### 2.4 PagedCache Snapshot 命中：FA4 层无需 replay

📄 [hybrid\_prefix\_cache.cpp](file:///Users/linyi/code/Documents/code/tokenspeed/tokenspeed-scheduler/csrc/resource/hybrid_prefix_cache/hybrid_prefix_cache.cpp)

FA4 的 attention forward 需要完整的 KV history 和 state。`PagedCacheSnapshot` 在每 `history_alignment_tokens` 处保存快照，命中时直接注入：

```cpp
// augmentMatchPagedCache 输出
match.paged_cache.per_group_page_ids["kv_history"] = [p0, p1, p2, ...p_N]
match.paged_cache.per_group_page_ids["kv_state"]   = [p_last_W, ...]  // sliding window

// 运行时注入 CUDA Graph replay 的 block tables
// CudaGraphWrapper.__call__() 中：
paged_cache_block_tables = {
    "kv_history": tensor([p0, p1, ..., p_N]),
    "kv_state":   tensor([p_last_W, ...]),
}
```

`RestoreKind::kSnapshotComplete`：完整快照路径，运行时直接 attach，`replay_start_tokens = 0`（无需 replay 任何 token）。Phase 2 将支持 `kReplay` 变体（partial snapshot + replay 剩余段），进一步提升深度较长时的命中率。

### 2.5 KV Host → Device 预取：异步 pipeline，命中不等待

📄 [kv\_prefix\_cache.h](file:///Users/linyi/code/Documents/code/tokenspeed/tokenspeed-scheduler/csrc/resource/kv_prefix_cache/kv_prefix_cache.h)

```cpp
// 命中在 Host tier 时，异步将 KV pages 传输到 Device
void KVPrefixCache::EnqueueTransfer(TreeNode* last_node) { ... }

// 只释放 Device 资源，保留 Host 资源（Host 是 L2 cache）
std::vector<TreeNode*> KVPrefixCache::ReleaseDeviceResourcesPresentOnHost(
    TreeNode* last_node, std::function<void(TreeNode*)> on_release) { ... }
```

`MatchResult` 的 `host.last_node` 和 `device.last_node` 可以不同深度——device 命中少、host 命中多时，调度器可以提前触发 H2D 传输，在 decode 第一步开始前完成 warm-up。

### 2.6 SMG tokenizer L0/L1 cache：命中在请求编码层

📄 [serve\_smg.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/cli/serve_smg.py)

```python
# SMG gateway 默认启用 L0/L1 tokenizer cache
# 相同 prompt text → 相同 token ids，不重复 tokenize
# 精确保证 RadixTree 上的 hash 一致性
"--tokenizer-l0-cache-config", json.dumps(l0_cache_config),
"--tokenizer-l1-cache-config", json.dumps(l1_cache_config),
```

tokenizer cache 确保相同文本内容产生完全一致的 token id 序列，这是 hash 匹配的前提条件——如果不同请求的同一 system prompt 产生不同 token ids（比如不同 tokenizer 版本），RadixTree 无法匹配。

---

## 整体架构图

```
请求进入
    │
    ▼
SMG tokenizer L0/L1 cache (文本 → token ids，确保一致性)
    │
    ▼
RadixTree.WalkDownUntilMismatch(token_ids)
    │
    ├── KV Device hit  → device.last_node.DepthInPage() × page_size = 命中 token 数
    ├── KV Host hit    → EnqueueTransfer() 异步 H2D，同时用 host 数据开始 prefill
    ├── Mamba GPU hit  → mamba_cow_src_index, CoW 复用 SSM state
    ├── Mamba Host hit → mamba_host_src_index, 触发 H→D Mamba state 加载
    └── PagedCache hit → per_group_page_ids 直接注入 FA4 block tables
              │
              ▼
    调度器生成 ForwardOperation
    (prefix_len = 命中 token 数, remaining = 新增 token 数)
              │
              ▼
    Python 运行时
    ├── extend_with_prefix=True: attention 后端只对 remaining 做 prefill
    ├── mamba_cow_src_indices: GDN 层从 branching_seqlen 处开始 chunk replay
    └── paged_cache_block_tables: FA4 直接 attach snapshot pages
              │
              ▼
    decode 阶段 (CUDA Graph + MTP)
    ├── 每 step O(1) GDN state 更新
    ├── MTP 每 step 3-4 token
    └── verify 后 O(1) 指针更新 (_update_current_inputs_after_verify)
              │
              ▼
    KV/Mamba state 写回 RadixTree（供后续请求复用）
    └── CommitChunk: 每 lcm token 提交 PagedCacheSnapshot
```

---

## 源码文件索引

| 层次 | 文件 | 核心职责 |
|------|------|---------|
| 命中率 | [radix\_tree.h/.cpp](file:///Users/linyi/code/Documents/code/tokenspeed/tokenspeed-scheduler/csrc/resource/radix_tree/radix_tree.cpp) | RadixTree + WalkDown + SplitAt |
| 命中率 | [tree\_node.h](file:///Users/linyi/code/Documents/code/tokenspeed/tokenspeed-scheduler/csrc/resource/radix_tree/tree_node.h) | KV/Mamba/PagedCache 三资源节点 |
| 命中率 | [page\_hasher.h](file:///Users/linyi/code/Documents/code/tokenspeed/tokenspeed-scheduler/csrc/scheduler/page_hasher.h) | SHA256 链式 page hash |
| 命中率 | [hybrid\_prefix\_cache.h/.cpp](file:///Users/linyi/code/Documents/code/tokenspeed/tokenspeed-scheduler/csrc/resource/hybrid_prefix_cache/hybrid_prefix_cache.cpp) | KV+Mamba+PagedCache 统一管理 |
| 命中率 | [kv\_prefix\_cache.h](file:///Users/linyi/code/Documents/code/tokenspeed/tokenspeed-scheduler/csrc/resource/kv_prefix_cache/kv_prefix_cache.h) | 底层 KV 管理 + EnqueueTransfer |
| 命中率 | [eviction.h](file:///Users/linyi/code/Documents/code/tokenspeed/tokenspeed-scheduler/csrc/resource/kv_prefix_cache/eviction.h) | LRU 驱逐，TP 一致 seq_id |
| 命中率 | [tree\_resource.h](file:///Users/linyi/code/Documents/code/tokenspeed/tokenspeed-scheduler/csrc/resource/radix_tree/tree_resource.h) | NodeResource RefCount + LRU |
| 命中率 | [paged\_cache\_snapshot.h](file:///Users/linyi/code/Documents/code/tokenspeed/tokenspeed-scheduler/csrc/resource/radix_tree/paged_cache_snapshot.h) | PagedCacheSnapshot per-group 结构 |
| 命中率 | [mamba\_slot.h](file:///Users/linyi/code/Documents/code/tokenspeed/tokenspeed-scheduler/csrc/resource/radix_tree/mamba_slot.h) | MambaSlot RAII（index + releaser） |
| 命中率 | [serve\_smg.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/cli/serve_smg.py) | tokenizer L0/L1 cache（hash 一致性） |
| 性能收益 | [hybrid\_linear\_attn.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py) | CoW + branching_seqlen + O(1) MTP 指针 |
| 性能收益 | [cuda\_graph\_wrapper.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/execution/cuda_graph_wrapper.py) | CUDA Graph replay + Mamba padding |
| 性能收益 | [mamba\_cache\_host.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/cache/mamba_cache_host.py) | GPU-visible pinned host memory |
| 性能收益 | [mamba\_pool.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/cache/transfer/mamba_pool.py) | 层序传播 PD disagg |

---

## 关键洞察

1. **命中率的底层逻辑是 SplitChild**：RadixTree 能在"部分匹配"时动态分裂节点，相比 page-level hash 可以获得更细粒度的前缀共享，agentic 场景中即使不同对话有细微差异（比如上下文不同轮次）也能共享尽可能多的前缀。

2. **Mamba state 的 prefix cache 是 tokenspeed 的独特优势**：普通框架只缓存 KV，Mamba/GDN 的 SSM state 每次都全量重算。tokenspeed 把 SSM state 也挂到 RadixTree 节点，实现了"混合架构的全量 prefix cache"——这对 Qwen3.5 这类大部分层是 GDN 的模型尤为关键。

3. **AdmissionFailureKind 分类驱逐**：驱逐策略区分 kHistoryStarved / kStateStarved，避免为了释放 state group 而连带删除更宝贵的 history group，精细控制驱逐粒度，最大化命中率。

4. **MTP + Prefix Cache 的乘法效应**：prefix cache 压缩 TTFT（首 token 时间）→ decode 占比提高 → MTP 效果放大。90%+ 命中时 prefill 降到 ~10%，而 MTP 在 decode 阶段提供 3-4× 加速，两者叠加是 580 tok/s 的核心来源。

5. **TP rank 一致的 seq_id 设计**：代码注释里显式说明"pointer values would diverge across TP ranks due to ASLR and eventually wedge the next NCCL collective"，用单调 seq_id 替代 pointer 做 tiebreaker，这是生产级分布式系统的细节正确性。
