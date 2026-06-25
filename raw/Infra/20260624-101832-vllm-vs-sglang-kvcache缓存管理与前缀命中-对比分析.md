# vLLM/vllm-ascend vs SGLang/sgl-kernel-npu：KV Cache 缓存管理与前缀命中 对比分析

> 生成时间：2026-06-24
> 走读版本：vllm `0d2961229` / vllm-ascend `8afdf356` / sglang `b5e0965b07` / sgl-kernel-npu(同期)
> 聚焦：**KV cache 缓存管理 + 前缀命中** 两个功能点，重点 **超长上下文 + 多轮交互** 场景
> 原则：结论基于代码走读，行号 grep 实测；缺口/存疑明确标注。
> 关联：[[20260624-012451-deepseek-v4-flash-昇腾A2-长上下文agentic-kvcache与PD弹性-深度分析]] [[20260624-011205-sglang-vs-vllm-kvcache系统级优化-大规模集群视角-深度分析]]

---

## 〇、一句话定性（先给结论）

| | vLLM 系 | SGLang 系 |
|---|---------|-----------|
| **核心数据结构** | **哈希块表**（hash → block） | **Radix 前缀树**（TreeNode 树） |
| **命中粒度** | block 对齐（page_size 整块） | token 级（树边可任意长度） |
| **设计哲学** | 显存管理优先，前缀命中是块复用的副产品 | 前缀复用是一等公民，缓存即树 |
| **分层(HBM↔host↔远端)** | 块级 offloading（connector 外挂） | **树节点内生分层**（TreeNode 自带 host_value） |
| **超长上下文** | 按 attention 类型多套 manager（full/SWA/hybrid） | 按 attention 类型多套 radix（radix/swa/mamba/unified） |

> **底层逻辑差异**：vLLM 把"块"做成原子单位、hash 表查命中——工程上简洁、与 PagedAttention 天然契合；SGLang 把"前缀树"做成核心抽象——多轮共享前缀的语义表达更自然、分层更内生。这决定了后面所有优劣。

---

## 一、vLLM/vllm-ascend 实现方案

### 1.1 数据结构：哈希块表 + 块池

- **`BlockHashToBlockMap`**（[block_pool.py:34](vllm/vllm/v1/core/block_pool.py)）：`BlockHash → KVCacheBlock` 哈希表，O(1) 查某个块是否已缓存。
- **`BlockPool`**（block_pool.py:130）：块池 + free queue（LRU），核心方法：
  - `get_cached_block`（:184）：按 hash 查命中块；
  - `cache_full_blocks`（:211）：把算完的满块注册进 hash 表（变成可复用前缀）；
  - `touch`（:402）：命中复用时给块加引用计数（防被驱逐）；
  - `_maybe_evict_cached_block`（:365）：从 free queue 驱逐时同步删 hash 表项。

### 1.2 前缀命中：block hash 链 + extra_keys 隔离

- **block hash 算法**（[kv_cache_utils.py](vllm/vllm/v1/core/kv_cache_utils.py)）：`BlockHash` 是 32 字节（:43），每块 hash = f(前块 hash, 本块 token)，**链式**——保证前缀一致才命中（前缀属性）。`NONE_HASH` 随机初始化（:111，防跨进程碰撞攻击）。
- **命中匹配**：`KVCacheManager.get_computed_blocks`（[kv_cache_manager.py:202](vllm/vllm/v1/core/kv_cache_manager.py)）逐块查 hash 表，返回最长命中块数。
- **多轮/多模态隔离（关键）**：`generate_block_hash_extra_keys`（kv_cache_utils.py:525）把 **mm（多模态 identifier）、lora_name、cache_salt** 拌进 hash（:417/484/494）——`need_extra_keys`（:397）。**这是多轮交互场景做"会话/租户隔离"的根基**：不同 cache_salt 的相同 token 不会误命中。

### 1.3 超长上下文：按 attention 类型多套 manager

`single_type_kv_cache_manager.py` 里 **`find_longest_cache_hit` 有 5+ 个变体**（:398/523/601/811/971/1263），对应：
- `FullAttentionManager`：标准全量；
- `SlidingWindowManager`（:582）：滑窗——**只需命中窗口内的块**，超长上下文下不必匹配全程；
- Hybrid / Mamba / cross 等。

> 即 vLLM 对超长上下文的前缀命中是**按 attention 类型分治**的，SWA 模型只匹配窗口块、省匹配成本。

### 1.4 跨请求共享：common prefix

`get_num_common_prefix_blocks`（kv_cache_manager.py:507）算 running 请求间的公共前缀块——配合 cascade attention，公共 system prompt 只算一次。

### 1.5 昇腾适配

- prefix cache 主体**复用 vLLM 主线**（block hash 那套，平台无关）。
- vllm-ascend 增量：`core/recompute_scheduler.py`（重计算调度）、`scheduler_profiling_chunk.py`（chunk profiling）。
- KV 搬运算子在 connector 侧（前篇坐实 PD 传压缩态）。

---

## 二、SGLang/sgl-kernel-npu 实现方案

### 2.1 数据结构：Radix 前缀树

- **`TreeNode`**（[radix_cache.py:223](sglang/python/sglang/srt/mem_cache/radix_cache.py)）：
  - `children: defaultdict(TreeNode)` + `parent` —— 树形（:228/229）；
  - `key: RadixKey` + `value`（device KV）—— 一条树边存一段连续 token 的 KV（:230/231）；
  - `lock_ref`（引用计数，:232）+ `last_access_time`（LRU，:233）；
  - **`host_value` + `host_ref_counter`**（:241/239）—— **节点内生支持 HBM↔host 分层**（vLLM 没有的关键差异）。
- **`RadixKey`**（:57）：`token_ids` + **`extra_key`**（:60/72）—— extra_key 做会话/租户隔离（对标 vLLM 的 cache_salt）。还支持 `is_bigram`（:58，bigram 视图，用于某些匹配优化）。

### 2.2 前缀命中：树遍历最长匹配

- **`match_prefix`**（radix_cache.py:360）→ `_match_prefix_helper`（:653）：从根沿 children 走，**token 级最长前缀匹配**——不像 vLLM 必须 block 对齐，树边可任意长度，**部分块也能命中**（命中粒度更细）。
- **写回**：`cache_unfinished_req`（:493，多轮关键——请求没结束也把已算 KV 插树，下一轮立即可复用）+ `cache_finished_req`（:442）。
- **`insert` / `_insert_helper`**（:420/709）：插入时自动**分裂节点**（公共前缀变父节点）——多轮共享前缀天然形成树结构。
- **驱逐**：`evict`（:568）按 `last_access_time` LRU + `lock_ref` 保护在用节点；`evictable_size`/`protected_size`（:633/636）。

### 2.3 超长上下文：HiRadix 分层 + 多种 radix 变体

**(a) HiRadixCache 三级分层命中**（[hiradix_cache.py:72](sglang/python/sglang/srt/mem_cache/hiradix_cache.py)）：
- `match_prefix`（:1438）覆盖 **device → host → storage** 三级——命中可命中到 host/远端冷层；
- `init_load_back` / `load_back`（:1212/1140）：命中冷层时回载 device；
- `prefetch_from_storage`（:1471）：预取隐藏回载延迟；
- `ready_to_load_host_cache` / `writing_check`（:1270/906）：异步回写/回载的就绪检查。

**(b) 按 attention 类型的 radix 变体**（超长上下文核心）：
- `swa_radix_cache.py`（滑窗，超长上下文按窗口存）；
- `mamba_radix_cache.py` + `hi_mamba_radix_cache.py`（线性注意力/状态空间，KV 是状态而非 token）；
- `unified_radix_cache.py`（统一）；
- 还有 V4 专用 `DeepSeekV4*Pool`（前篇坐实的 HCS/CSA/SWA 三档压缩稀疏）。

### 2.4 昇腾适配

- sgl-kernel-npu KV 算子：`assign_cache_op`（KV 槽位赋值）、`kvcacheio.py`（KV IO）、`mem_cache/`——**Radix 树管理在 Python 侧，KV 实际搬运/赋值落到昇腾算子**。
- 命中逻辑（match_prefix）是平台无关的 Python 树遍历，昇腾复用。

---

## 三、差异与优劣势对比（聚焦超长上下文 + 多轮）

### 3.1 前缀命中粒度

| | vLLM | SGLang |
|---|------|--------|
| 粒度 | **block 对齐**（page_size 整块，如 16/128 token） | **token 级**（树边任意长度） |
| 影响 | 不足一整块的尾部前缀**命中不了**，多轮场景边界 token 浪费 | 部分块也能命中，命中率天然更高 |

> **多轮场景判断**：多轮对话每轮长度不规整，vLLM 的 block 对齐会在每轮边界损失"不足一块"的命中；SGLang token 级匹配更充分。**SGLang 在多轮命中率上有结构性优势**。

### 3.2 分层（超长上下文显存外溢）

| | vLLM | SGLang |
|---|------|--------|
| 机制 | 块级 offloading，**外挂 connector**（kv_offload/offloading_connector） | **TreeNode 内生 host_value**，HiRadix 三级 |
| 命中冷层 | 命中判定与分层是两套（hash 命中 + 单独 offload 逻辑） | **match_prefix 一次遍历直接命中到 host/远端**（hiradix_cache.py:1438） |

> **超长上下文判断**：SGLang 的分层是"树语义内生"的——同一次前缀匹配就知道命中在 device/host/storage 哪一级，且按前缀树结构冷热分层更自然。vLLM 的 offloading 是块级外挂，命中与分层耦合度低。**SGLang 在超长上下文分层复用上更内聚**。

### 3.3 多轮隔离

| | vLLM | SGLang |
|---|------|--------|
| 机制 | hash 拌入 `cache_salt`/lora/mm（kv_cache_utils.py:525） | `RadixKey.extra_key`（radix_cache.py:60） + session radix |
| 评价 | 隔离拌进 hash，干净；mm/lora 隔离更体系化（专门 extra_hash_keys） | extra_key 简洁；有 `SessionRadixCacheMixin` 做会话级 |

> 两边都解决了多轮隔离，**vLLM 的多模态/lora 隔离更体系化**（专门的 mm extra hash keys），SGLang 的会话级树（session radix）对纯文本多轮更直接。打平偏 vLLM。

### 3.4 内存管理开销

| | vLLM | SGLang |
|---|------|--------|
| 查命中 | hash 表 **O(1)/块** | 树遍历 **O(depth)** |
| 驱逐 | free queue LRU，块级 | 树 LRU + 节点分裂/合并，有树维护开销 |
| 评价 | 查命中更快、结构更简单、内存碎片可控 | 树维护（分裂/锁/分层）开销更大，但换来更高命中率 |

> **工程成本判断**：vLLM 结构简单、开销可预测、与 PagedAttention 契合好；SGLang 树维护更重，但在高复用场景（多轮/共享前缀）用命中率赚回来。**纯吞吐/低复用场景 vLLM 更省，高复用场景 SGLang 更值**。

### 3.5 超长上下文的 attention 类型适配

两边**都做了按 attention 类型分治**（vLLM 多套 manager / SGLang 多套 radix），打平。SGLang 的 mamba/hi_mamba radix 对线性注意力模型覆盖更细；vLLM 的 hybrid manager 体系化。

---

## 四、综合判断

### 4.1 超长上下文 + 多轮场景，谁更强？

**SGLang 在这个特定场景结构性占优**，三个根本原因：
1. **token 级命中**（vs vLLM block 对齐）—— 多轮不规整长度下命中率更高；
2. **分层内生**（TreeNode.host_value + HiRadix 三级 match）—— 超长上下文显存外溢后，命中冷层是同一次树遍历完成，更内聚；
3. **cache_unfinished_req** —— 请求未结束就插树，多轮连续请求复用更及时。

**vLLM 的优势**在另一面：结构简单、查命中 O(1)、与 PagedAttention/cascade 契合好、多模态&lora 隔离更体系化、内存开销可预测。**通用性和工程稳健性更强**。

### 4.2 一句话

> **vLLM = 块表，工程稳健、通用、开销可控；SGLang = 前缀树，多轮/超长上下文命中率与分层内生性更强。** 超长上下文 + agentic 多轮这个特定场景，**SGLang 的 Radix/HiRadix 架构是更贴合的设计**——这也是 SGLang 当初以 RadixAttention 立身的根本。

### 4.3 昇腾侧（两套都成立）

- 缓存管理 + 前缀命中**主体都是平台无关的 Python 逻辑**（hash 表 / 树遍历），昇腾复用。
- 差异在 KV 实际搬运：vllm-ascend 走 connector + recompute_scheduler；sgl-kernel-npu 走 assign_cache/kvcacheio 算子。
- 两边昇腾适配**都不影响上层命中算法的优劣对比**，差异在前缀命中这层不体现。

---

## 五、待核实/存疑

1. **vLLM block hash 是否支持"部分块"命中**：本次看是 block 对齐，但 hybrid/SWA manager 是否有更细粒度命中未逐一深读 —— ⚠️ 倾向"块对齐"，未穷尽 5 个 manager 变体。
2. **SGLang HiRadix 远端(storage)命中的实际延迟**：match_prefix 能命中到 storage，但回载延迟是否会抵消复用收益，需 benchmark，代码侧只能确认机制存在。
3. **两边命中率的量化对比**：本文是机制层对比，token 级 vs block 级的实际命中率差距需实测 —— 无 benchmark 数据，不臆测数字。
4. **昇腾 sgl-kernel-npu 是否有 radix 树的算子级加速**：看到的是 KV IO/assign 算子，树遍历本身是否有 npu 加速未确认。
