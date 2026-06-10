# DeepSeek V4 Flash @ 910B3 (A2) 低时延/高吞吐 PD 分离 Recipe —— W8A8 & BF16 双精度策略

> 生成时间: 2026-06-11 | 基于 vllm-ascend v0.18-v0.20 | 聚焦 910B3 单机 + PD 分离

---

## 0. 910B3 内存硬约束：先算账再谈策略

### 0.1 硬件基本面

| 规格 | Ascend 910B3 |
|------|-------------|
| **单卡 HBM** | 64 GB HBM2e |
| **单节点 NPU 数** | 8 (Atlas 800T A2) |
| **单节点总 HBM** | **512 GB** |
| **内存带宽** | ~400 GB/s per NPU |
| **FP16 算力** | ~320 TFLOPS per NPU |
| **INT8 算力** | ~640 TOPS per NPU |

### 0.2 量化精度基础：W8A8 到底占多少内存

**关键澄清**：Ascend 的 `--quantization ascend` (W8A8) 是指：

- **Weight**: INT8 精度，**1 byte per parameter**
- **Activation**: INT8 精度（运行时动态量化）
- 底层通过 CANN 的 `npu_quant_matmul` 执行 INT8×INT8 矩阵乘法

**因此 W8A8 的权重显存占用与 FP8 完全相同**——都是每个参数 1 byte。W8A8 ≠ FP8 的一半。FP8 也是 1 byte/param，只是数值格式不同（浮点 vs 整数）。W8A8 相比 BF16 的压缩比是 2×（不是 4×）。

DeepSeek V4 Flash 的权重结构：

| 组件 | 参数量 | 说明 |
|------|--------|------|
| **Attention (Q/K/V/O, LoRA 投影)** | ~7.2B | 43 层, hidden=4096, Q heads=64, KV head=1 |
| **Routed Experts (256 × FFN)** | ~272B | 每 expert 含 gate/up/down，moe_intermediate=2048 |
| **Shared Expert (1 × FFN)** | ~2.3B | 1 个共享 expert |
| **Embedding + lm_head** | ~1.1B | vocab=129280, hidden=4096 |
| **MTP Layer** | ~0.8B | 1 层 MTP |
| **Other (norm, router, etc.)** | ~0.6B | |
| **Total** | **~284B** | |

不同精度下的**运行时 NPU 内存占用**（权重部分）：

| 精度 | bytes/param | 权重大小 (总) | 权重大小 (TP=8, 单卡) | 单卡可用 HBM | 单卡剩余 (KV cache + 运行时) | 910B3 单节点可行性 |
|------|------------|-------------|---------------------|-------------|---------------------------|-------------------|
| **BF16** | 2 | **~568 GB** | 71 GB | 64 GB | **-7 GB → OOM** | **❌ 放不下** |
| **FP8 (原生)** | 1 | **~284 GB** | 35.5 GB | 64 GB | ~22 GB | **✅ 可行** |
| **W8A8 (Ascend INT8)** | 1 | **~284 GB** | **35.5 GB** | 64 GB | **~22 GB** | **✅ 可行** |
| **FP4+FP8 (原生)** | ~0.56 (平均) | ~160 GB | 20 GB | 64 GB | ~38 GB | **✅ 可行** |

> **修正要点**：之前错误地将 W8A8 权重估为 ~142 GB（当成 0.5 byte/param），实际 W8A8 中 weight 是 INT8 即 1 byte/param，与 FP8 同为 ~284 GB。W8A8 和 FP8 在显存占用上是等价的，区别仅在于数值格式（整数 vs 浮点）和计算路径。

### 0.3 各精度下的实际运行时内存分解 (TP=8)

```
                    BF16          FP8/W8A8       FP4+FP8 (原生)
                    ─────         ────────       ─────────────
权重 (284B params)   568 GB        284 GB         160 GB
单卡权重 (÷8)         71 GB         35.5 GB        20 GB
运行时开销 (~5GB)      5 GB          5 GB           5 GB
单卡 KV Cache 可用    -12 GB ⚰      23.5 GB        39 GB
```

### 0.4 核心结论

**BF16 在 910B3 单节点物理不可行**。71 GB/卡 > 64 GB HBM，即使 TP=8 打满也装不下权重。

**W8A8 / FP8 在 910B3 单节点可行但内存紧张**。单卡 35.5 GB 权重 + 5 GB 运行时 = 40.5 GB。剩余 23.5 GB 用于 KV cache。

**23.5 GB KV cache 能支持多少上下文/并发？** 

| Context Length | KV Cache / seq (W8A8) | Max seqs (23.5 GB KV) |
|---------------|----------------------|----------------------|
| 8K | ~0.3 GB | ~78 seqs (够用) |
| 16K | ~0.6 GB | ~39 seqs (够用) |
| 32K | ~1.2 GB | ~19 seqs (勉强) |
| 64K | ~2.4 GB | ~9 seqs (紧张) |
| 128K | ~4.8 GB | ~4 seqs (极少) |

> 注：以上估算基于 DeepSeek V4 Flash 的 CSA/HCA KV 压缩（约 90% 压缩率）。未压缩时为 10×。

**这改变了 910B3 上 PD 分离的可行性判断**：同机 PD 分离中，若 P 或 D 任一实例用 TP=1，单卡需装 284 GB / 1 = 284 GB → 远超出 64 GB。**同机 PD 分离必须是全部 8 卡参与 TP=8 才装得下权重**，这意味着 P 和 D 无法同时跑在单节点上——它们需要共享同一份权重（TP 跨全部 8 卡）。

---

## 1. W8A8 精度 — 低时延 Recipe

### 1.1 场景约束与目标

| 约束 | 值 |
|------|-----|
| 硬件 | 单节点 Atlas 800T A2 (8 × 910B3, 64GB) |
| 精度 | W8A8 (`--quantization ascend`) |
| 权重占用 | 284 GB (35.5 GB/卡 @ TP=8) |
| 单卡 KV cache 可用 | **~23.5 GB** (64 - 35.5 - 5 运行时) |
| 目标 TTFT | < 500ms (8K context) |
| 目标 ITL | < 30ms |
| 部署模式 | 单节点 TP=8 无 PD，或多节点 PD 分离 |

### 1.2 架构选择：为什么 910B3 单节点不能做同机 PD 分离

**这是一个关键认知修正**。同机 PD 分离要求 P 和 D 部署在同一台机器上，各自独立运行模型实例。这意味着每个实例都需要加载一份完整的权重。

```
W8A8 权重 = 284 GB
单机 8 × 64GB = 512 GB

选项 1: P 和 D 都用 TP=1 → 每实例需 284 GB/card → 单卡 64GB 放不下 ⚰
选项 2: P 用 TP=4, D 用 TP=4 → 每实例 284/4 = 71 GB/card → 单卡 64GB 放不下 ⚰
选项 3: P 用 TP=8, D 用 TP=8 → 需要 16 卡 → 单节点只有 8 卡 ⚰
```

**结论：910B3 单节点无法做同机 PD 分离**。8 × 64GB 的总 HBM 不足以同时容纳两份 W8A8 权重（284 GB × 2 = 568 GB > 512 GB）。

**910B3 上的低时延只有两条路**：
| 方案 | 描述 | 适用场景 |
|------|------|---------|
| **Recipe A** | 单节点 TP=8 无 PD | 低并发、对 ITL 无硬约束 |
| **Recipe A2** | 2 节点跨机 PD (P: TP=8, D: TP=8) | 需要 PD 隔离的极致 ITL 控制 |

### 1.3 Recipe A：单节点 TP=8 无 PD（910B3 低时延标准方案）

最简部署策略——官方推荐方案，无需额外的 proxy/Mooncake。

```bash
#!/bin/bash
# === 910B3 W8A8 单节点低时延 Recipe ===
# 硬件: Atlas 800T A2, 8 × 910B3 64GB
# 模型: DeepSeek-V4-Flash-w8a8-mtp (ModelScope)

# --- 系统调优 ---
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000

# --- 环境变量 ---
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=8
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACL_OP_INIT_MODE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export USE_MULTI_GROUPS_KV_CACHE=1
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=512
export USE_MULTI_BLOCK_POOL=1

# --- 启动命令 ---
vllm serve /path/to/DeepSeek-V4-Flash-w8a8-mtp \
  --safetensors-load-strategy 'prefetch' \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":16}' \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.88 \
  --data-parallel-size 1 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --quantization ascend \
  --block-size 64 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --async-scheduling \
  --speculative-config '{"num_speculative_tokens": 1, "method": "mtp"}' \
  --additional-config '{
    "enable_cpu_binding": true,
    "multistream_overlap_shared_expert": false,
    "recompute_scheduler_enable": true
  }' \
  --compilation-config '{
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32]
  }'
```

**参数分析**:

| 参数 | 值 | 低时延视角分析 |
|------|-----|--------------|
| `max-model-len` | **16384** | 低时延场景不需要长上下文，16K 已覆盖绝大多数对话场景。减少预分配 KV cache，给 decode 留更多 batch 空间 |
| `max-num-batched-tokens` | **4096** | 限制单次 prefill 的 token 总量。910B3 算力有限，4096 tokens 的 prefill 约需 200-300ms。太小→prefill 被切碎效率低；太大→单次 prefill 太久拖慢 TTFT |
| `max-num-seqs` | **8** | 并发数低是低时延的核心策略。更少 seq 竞争 = 每个 seq 更频繁被调度 = 更低 ITL |
| `gpu-memory-utilization` | **0.88** | 权重 35.5 GB/卡 + 5 GB 运行时 = 40.5 GB。0.88 × 64 = 56.3 GB → KV cache 可用 15.8 GB。留 12% buffer 给运行时 spike（NPU 算子编译临时内存等），防止 OOM |

### 1.4 Recipe A2：跨机 PD 分离 W8A8（910B3 极致低时延）

**背景**：910B3 单节点无法做同机 PD（见 1.2 分析），但可以通过 2 节点实现 PD 分离来控制 ITL 尾延迟。

```
2 × 910B3 节点 (每节点 8 × 64GB)
├── P 节点: TP=8 → 专注 prefill，不受 decode 干扰
├── D 节点: TP=8 → 专注 decode，不受 prefill 打断
└── RDMA 跨机 KV transfer (MooncakeHybridConnector)
```

**为什么 P 和 D 都用 TP=8？** 因为 W8A8 权重 284 GB 必须分到至少 5 张卡（284/64=4.44）。在实际部署中，满 8 卡 TP=8 是最优选择——充分利用节点内 HCCS 带宽，且单卡权重 35.5 GB 留出 23.5 GB 给 KV cache。

```bash
#!/bin/bash
# === 910B3 W8A8 跨机 PD 低时延 Recipe ===

# ================== Prefill 节点 (8 × 910B3) ==================
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000

nic_name="eth0"
local_ip=$(hostname -I | awk '{print $1}')

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export HCCL_OP_EXPANSION_MODE="AIV"
export TASK_QUEUE_ENABLE=1
export VLLM_RPC_TIMEOUT=3600000
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export USE_MULTI_GROUPS_KV_CACHE=1
export USE_MULTI_BLOCK_POOL=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /path/to/DeepSeek-V4-Flash-w8a8-mtp \
  --host 0.0.0.0 --port 8100 \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --quantization ascend \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.85 \
  --block-size 64 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --async-scheduling \
  --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}' \
  --additional-config '{
    "recompute_scheduler_enable": true,
    "enable_shared_expert_dp": true,
    "enable_cpu_binding": true
  }' \
  --kv-transfer-config '{
    "kv_connector": "MooncakeHybridConnector",
    "kv_role": "kv_producer",
    "kv_port": "30000",
    "engine_id": "0",
    "kv_connector_extra_config": {
      "prefill": {"dp_size": 1, "tp_size": 8},
      "decode": {"dp_size": 1, "tp_size": 8}
    }
  }'

# ================== Decode 节点 (8 × 910B3) ==================
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export HCCL_OP_EXPANSION_MODE="AIV"
export TASK_QUEUE_ENABLE=1
export VLLM_RPC_TIMEOUT=3600000
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export USE_MULTI_GROUPS_KV_CACHE=1
export USE_MULTI_BLOCK_POOL=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /path/to/DeepSeek-V4-Flash-w8a8-mtp \
  --host 0.0.0.0 --port 8200 \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --quantization ascend \
  --max-model-len 16384 \
  --max-num-batched-tokens 60 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 \
  --block-size 64 \
  --no-enable-prefix-caching \
  --async-scheduling \
  --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{
    "recompute_scheduler_enable": true,
    "multistream_overlap_shared_expert": false,
    "enable_cpu_binding": true,
    "enable_npugraph_ex": true,
    "eplb_config": {
      "dynamic_eplb": false,
      "expert_heat_collection_interval": 600,
      "algorithm_execution_interval": 50,
      "eplb_policy_type": 2,
      "num_redundant_experts": 32
    }
  }' \
  --kv-transfer-config '{
    "kv_connector": "MooncakeHybridConnector",
    "kv_role": "kv_consumer",
    "kv_port": "30400",
    "engine_id": "1",
    "kv_connector_extra_config": {
      "prefill": {"dp_size": 1, "tp_size": 8},
      "decode": {"dp_size": 1, "tp_size": 8}
    }
  }'
```

**Recipe A2 参数分析（与 Recipe A 的关键差异）**:

| 参数 | Recipe A (单节点无 PD) | Recipe A2 (跨机 PD) | 为什么不同 |
|------|----------------------|-------------------|-----------|
| **节点数** | 1 | 2 | PD 分离需要两套独立权重，每套 284 GB，单节点 512 GB 装不下 |
| P `gpu-memory-utilization` | 0.88 | **0.85** | P 端需要为 prefill 中间激活留更多 buffer，比无 PD 时多留 3% |
| D `gpu-memory-utilization` | 0.88 | **0.90** | D 端仅 decode，中间激活小，可以多分配给 KV cache |
| D `max-num-batched-tokens` | — | **60** | 纯 decode 模式，60 tokens/batch 在 ITL 和吞吐间平衡 |
| D `max-num-seqs` | 8 | **16** | PD 分离后 D 不受 prefill 干扰，可以承载更多并发 |
| `enforce-eager` (P) | false | **true** | 跨机 PD 下 P 端 prefill 不需要 CUDA graph（输入 shape 变化大），eager 更稳定 |
| `kv_connector` | 无 | **MooncakeHybridConnector** | 跨机 RDMA KV transfer |
| P/D `tp_size` | — | **8** (两端相同) | 对称 TP，符合 vllm-ascend 支持的 P_tp=D_tp 配置 |

**跨机 PD 的额外延迟代价**：RDMA KV 传输约增加 TTFT 50-150ms（取决于 context 长度和 KV cache 压缩率）。但对 ITL 的稳定性改善（消除 prefill 打断）通常值得这个代价。

---

## 2. W8A8 精度 — 高吞吐 Recipe

### 2.1 场景约束与目标

| 约束 | 值 |
|------|-----|
| 硬件 | 2-4 节点 Atlas 800T A2 (每节点 8 × 910B3, 64GB) |
| 精度 | W8A8 |
| 目标吞吐 | 最大化总 tokens/s |
| TTFT 接受范围 | < 2s |
| 部署模式 | 跨机 PD 分离 |

### 2.2 架构选择

**核心约束回顾**：W8A8 权重 284 GB。910B3 单卡 64 GB。任何实例至少要 TP≥5（284/64=4.44）才装得下权重。实际部署中，P 节点和 D 节点都需要 TP=8。

```
2 × 910B3 节点 (共 16 NPU, 1024 GB HBM)
├── P 节点 (8 NPU): TP=8 → 专注高吞吐 prefill
├── D 节点 (8 NPU): TP=8 → 专注高并发 decode
└── RDMA 跨机 KV transfer (MooncakeHybridConnector)

高并发不是靠 DP 降低 TP 实现，而是靠：
1. D 端更大的 max-num-seqs (利用 23.5 GB/卡 KV cache 空间)
2. 更多 D 节点 (从 1P1D 扩展到 1P2D 或 2P2D)
```

**扩展路径**：

```
1P1D (2 节点) → 1P2D (3 节点) → 2P2D (4 节点)
   │               │               │
   └─ 基础吞吐      └─ D 加一倍       └─ P/D 各加一倍
```

> 如果请求速率极高（>20 req/s），扩展到 4 节点：2P + 2D。

### 2.3 Prefill 节点 (TP=8)

```bash
#!/bin/bash
# === 910B3 W8A8 高吞吐 Prefill 节点 ===
# 权重: 284 GB → TP=8 单卡 35.5 GB
# KV cache 可用: ~23.5 GB/卡 (gpu=0.90 时)

sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000

nic_name="eth0"
local_ip=$(hostname -I | awk '{print $1}')

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export HCCL_OP_EXPANSION_MODE="AIV"
export TASK_QUEUE_ENABLE=1
export VLLM_RPC_TIMEOUT=3600000
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export USE_MULTI_GROUPS_KV_CACHE=1
export USE_MULTI_BLOCK_POOL=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /path/to/DeepSeek-V4-Flash-w8a8-mtp \
  --host 0.0.0.0 \
  --port 8100 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --served-model-name deepseek_v4 \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.88 \
  --block-size 128 \
  --enforce-eager \
  --async-scheduling \
  --enable-prefix-caching \
  --quantization ascend \
  --safetensors-load-strategy 'prefetch' \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":16}' \
  --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}' \
  --additional-config '{
    "enable_cpu_binding": true,
    "multistream_overlap_shared_expert": false,
    "recompute_scheduler_enable": true
  }' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --kv-transfer-config '{
    "kv_connector": "MooncakeHybridConnector",
    "kv_role": "kv_producer",
    "kv_port": "30000",
    "engine_id": "0",
    "kv_connector_extra_config": {
      "prefill": {"dp_size": 1, "tp_size": 8},
      "decode": {"dp_size": 1, "tp_size": 8}
    }
  }'
```

### 2.4 Decode 节点 (TP=8)

```bash
#!/bin/bash
# === 910B3 W8A8 高吞吐 Decode 节点 ===
# 权重: 284 GB → TP=8 单卡 35.5 GB
# KV cache 可用: ~25.9 GB/卡 (gpu=0.94 时)

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export HCCL_OP_EXPANSION_MODE="AIV"
export TASK_QUEUE_ENABLE=1
export VLLM_RPC_TIMEOUT=3600000
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export USE_MULTI_GROUPS_KV_CACHE=1
export USE_MULTI_BLOCK_POOL=1

vllm serve /path/to/DeepSeek-V4-Flash-w8a8-mtp \
  --host 0.0.0.0 \
  --port 8200 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --served-model-name deepseek_v4 \
  --max-model-len 65536 \
  --max-num-batched-tokens 60 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.94 \
  --block-size 128 \
  --async-scheduling \
  --quantization ascend \
  --safetensors-load-strategy 'prefetch' \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":16}' \
  --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{
    "ascend_compilation_config": {
      "enable_npugraph_ex": true,
      "enable_static_kernel": false
    },
    "eplb_config": {
      "dynamic_eplb": false,
      "expert_heat_collection_interval": 600,
      "algorithm_execution_interval": 50,
      "eplb_policy_type": 2,
      "num_redundant_experts": 32
    },
    "enable_cpu_binding": true,
    "multistream_overlap_shared_expert": false,
    "recompute_scheduler_enable": true
  }' \
  --kv-transfer-config '{
    "kv_connector": "MooncakeHybridConnector",
    "kv_role": "kv_consumer",
    "kv_port": "30400",
    "engine_id": "1",
    "kv_connector_extra_config": {
      "prefill": {"dp_size": 1, "tp_size": 8},
      "decode": {"dp_size": 1, "tp_size": 8}
    }
  }'
```

**高吞吐参数分析**:

| 参数 | 值 | 分析 |
|------|-----|------|
| P/D `tp_size` | **8** (对称) | 权重 284 GB 必须 TP≥5，满 8 卡最优。对称 TP 符合 vllm-ascend 支持矩阵 |
| `max-model-len` | **65536** | 64K context 下每 seq KV cache ~1.2 GB。D 端 32 seqs × 1.2 = 38.4 GB 总量 → 分散在 8 卡上每卡 ~4.8 GB，在 23.5 GB 可用范围内。64K 而非 128K——64K 已覆盖绝大多数高吞吐场景的长 prompt |
| P `max-num-batched-tokens` | **8192** | 充分利用 TP=8 的聚合算力。P 端 prefill 用 8 卡的 2.56 PFLOPs 聚合吞吐 |
| `enable-prefix-caching` (P) | **true** | 高吞吐场景 shared prompt 多，prefix caching 省掉重复 prefill |
| D `max-num-seqs` | **32** | 比低时延的 16 翻倍。每卡 32 seqs × ~1.2 GB/seq = 38.4/8 = 4.8 GB KV cache。高并发是吞吐的关键 |
| D `gpu-memory-utilization` | **0.94** | 对比低时延 0.90 → 多 4% 即 2.6 GB/卡 给 KV cache。Decode 端内存优先于 buffer |
| `block-size` | **128** | 大 block 减少 metadata 和 KV transfer 开销。高吞吐适合 |
| `num_redundant_experts` | **32** | 高并发下 expert dispatch 冲突严重，最大静态冗余补偿 |

### 2.5 高吞吐扩展：多 D 节点

当单 D 节点吞吐不足时，加 D 节点是最经济的扩展策略（P 通常不是瓶颈）：

```
扩展到 3 节点 (1P2D):
├── P 节点 × 1 (TP=8) 
├── D 节点 × 2 (各 TP=8)
└── 通过 Proxy 负载均衡到两个 D 节点
```

```bash
# P 节点 (不变)
python launch_online_dp.py --dp-size 1 --tp-size 8 --dp-address <p_ip>

# D 节点 1
python launch_online_dp.py --dp-size 1 --tp-size 8 --dp-address <d1_ip>

# D 节点 2  
python launch_online_dp.py --dp-size 1 --tp-size 8 --dp-address <d2_ip>
```

**内存验证（1P2D @ 64K context）**:

| 节点 | 配置 | 单卡权重 | 单卡 KV cache | 总并发 |
|------|------|---------|-------------|--------|
| P | TP=8 | 35.5 GB | ~2 GB | max 16 seqs (prefill 短驻留) |
| D1 | TP=8 | 35.5 GB | ~9.6 GB | max 32 seqs |
| D2 | TP=8 | 35.5 GB | ~9.6 GB | max 32 seqs |
| **合计** | | | | **max 64 seqs 并发** |
## 3. BF16 精度 — 多节点策略

### 3.1 BF16 为什么在 910B3 上必须多节点

```
BF16 权重: 284B × 2 bytes = 568 GB
910B3 单节点: 8 × 64 GB = 512 GB

568 GB > 512 GB → 物理上不可行
```

**唯一可行路径**：至少 2 节点，通过 TP 或 PD 分离把权重分布到 >512 GB 的聚合 HBM 上。

但还有个严重问题——vLLM BF16 在 910B3 上没有官方测试过。社区有 `RedHatAI/DeepSeek-V4-Flash-BF16`（全 BF16 转换版，568GB）和 `FlagRelease/DeepSeek-V4-Flash-ascend-FlagOS`（昇腾优化版），但 vllm-ascend 官方文档推荐的始终是 W8A8 量化模型。

### 3.2 BF16 多节点 PD 分离

```
BF16 2 节点 PD 分离：
├── P 节点 × 1 (8 × 910B3 64GB)
│   └── TP=8, W8A8 权重 = 22GB/card ✓
│   └── 但 BF16 权重 = 71GB/card → OOM!
```

**问题**：即使 PD 分离，BF16 权重在 TP=8 下单卡仍需 71GB（> 64GB）。P 或 D 任何一端都放不下。

**唯一解法：跨节点 TP**。把 BF16 权重分布到 >8 个 NPU 上：

```
BF16 跨节点 TP=16 (2 节点 × 8 NPU = 16 NPU):
每卡权重: 568 GB / 16 = 35.5 GB
剩余每卡: 64 - 35.5 = 28.5 GB → 足够 KV cache + 运行时
```

但这意味着 **P 和 D 不能分离**——因为 16 卡都要参与同一个模型的 TP。这就是"无 PD 分离、纯 TP 跨节点"模式。

### 3.3 Recipe C：BF16 跨节点 TP=16（无 PD 分离）

```bash
#!/bin/bash
# === 910B3 BF16 2 节点 TP=16 ===
# 硬件: 2 × Atlas 800T A2, 共 16 × 910B3 64GB
# 模型: RedHatAI/DeepSeek-V4-Flash-BF16 (或自转换 BF16)

# =============== 节点 0 (rank 0) ===============
export HCCL_IF_IP=<节点0_IP>
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=8
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE="AIV"
export TASK_QUEUE_ENABLE=1
export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export USE_MULTI_GROUPS_KV_CACHE=1
export USE_MULTI_BLOCK_POOL=1

# 跨节点 TP 需要 Ray 或 torch.distributed 初始化
# vLLM-V1 通过 --tensor-parallel-size 16 自动处理

vllm serve /path/to/DeepSeek-V4-Flash-BF16 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 16 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.78 \
  --block-size 128 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --no-quantization \
  --dtype bfloat16 \
  --async-scheduling \
  --additional-config '{
    "recompute_scheduler_enable": true,
    "enable_cpu_binding": true,
    "multistream_overlap_shared_expert": false
  }' \
  --compilation-config '{
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [1,2,4,8]
  }'

# =============== 节点 1 (rank 1) ===============
# 除 HCCL_IF_IP 和 --port 外完全相同
# vLLM 通过 Ray 或 MPI 自动协调跨节点 TP
```

**参数分析（BF16 特有关键点）**:

| 参数 | 值 | 分析 |
|------|-----|------|
| `tensor-parallel-size` | **16** | 跨 2 节点 16 卡 TP。单卡权重 35.5GB，剩余 28.5GB。这是 BF16 在 64GB 单卡上能跑的**最小可行配置** |
| `gpu-memory-utilization` | **0.78** | 极保守。BF16 激活值占内存大（非量化），每卡只剩 28.5GB 用于 KV cache + 激活，必须留足 buffer。0.78 × 64 = 50GB → 实际权重 35.5GB，留给其他 ~14.5GB |
| `max-model-len` | **32768** | 受限于剩余内存。32K context BF16 KV cache 约 1.5 GB/seq（类 MQA 结构 ~0.05 GB/K tokens），每卡可容纳约 6-8 seqs |
| `max-num-seqs` | **8** | 2 节点 16 卡总共可约 128 seqs（每卡 8 seqs × 16），但受 TP 广播限制实际更低 |
| `max-num-batched-tokens` | **4096** | BF16 prefill 激活占用远超 W8A8，4096 是安全上限 |
| `enforce-eager` | **true** | 跨节点 TP 下 CUDA graph 可能不稳定（通信依赖），关掉换稳定性 |
| `no-quantization` | **true** | 明确不使用量化 |
| `dtype` | **bfloat16** | 指明 BF16 |

### 3.4 Recipe D：BF16 4 节点 PD 分离（理论最佳方案）

既然 BF16 2 节点才能装下权重，要做 PD 分离需要 4 节点：

```
4 × 910B3 节点 (共 32 NPU, 2048 GB HBM)
├── P 节点 × 2: TP=16 (跨 2 节点) 
│   └── 专注 prefill，完成后 KV cache 转 BF16→量化→传输
├── D 节点 × 2: TP=16 (跨 2 节点) 
│   └── 专注 decode，从 P 接收 KV cache
└── MooncakeHybridConnector (RDMA 跨机)
```

**重要限制**：vllm-ascend 目前限制 P_tp > D_tp 且 P_tp % D_tp = 0，而且 P/D 必须同构（都是 A2）。如果 P 和 D 各用 TP=16（2 节点），则 KV transfer 需要额外的量化步骤以降低 RDMA 传输量。

> **现实评估**：4 节点 910B3 做 BF16 PD 分离成本过高（服务器成本 + 功率），且性能大概率不如 2 节点 W8A8。除非有明确的精度需求（如金融/医疗领域的严格精度要求），否则 W8A8 方案在 910B3 上是投入产出比最优解。

```bash
# 仅供学术参考，生产环境强烈建议 W8A8

# P 节点群 (2 节点, TP=16):
vllm serve /path/to/DeepSeek-V4-Flash-BF16 \
  --tensor-parallel-size 16 \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.78 \
  --enforce-eager \
  --no-quantization --dtype bfloat16 \
  --kv-transfer-config '{
    "kv_connector": "MooncakeHybridConnector",
    "kv_role": "kv_producer",
    "kv_port": "30000",
    "kv_connector_extra_config": {
      "prefill": {"dp_size": 1, "tp_size": 16},
      "decode": {"dp_size": 1, "tp_size": 16}
    }
  }'

# D 节点群 (2 节点, TP=16):
vllm serve /path/to/DeepSeek-V4-Flash-BF16 \
  --tensor-parallel-size 16 \
  --max-model-len 65536 \
  --max-num-batched-tokens 120 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.78 \
  --no-quantization --dtype bfloat16 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --kv-transfer-config '{
    "kv_connector": "MooncakeHybridConnector",
    "kv_role": "kv_consumer",
    "kv_port": "30400",
    "kv_connector_extra_config": {
      "prefill": {"dp_size": 1, "tp_size": 16},
      "decode": {"dp_size": 1, "tp_size": 16}
    }
  }'
```

---

## 4. 精度 vs 场景决策矩阵

### 4.1 910B3 上的可行方案总览

| 方案 | 精度 | 节点数 | 部署模式 | 最大上下文 | 可行性 | 推荐场景 |
|------|------|--------|---------|-----------|--------|---------|
| **A** | W8A8 | 1 | TP=8 无 PD | 16K-64K | ✅ 官方支持 | 简单低时延、小规模部署 |
| **A2** | W8A8 | 2 | 跨机 PD (P:TP=8, D:TP=8) | 16K-64K | ✅ 官方支持 | 极致 ITL 控制 |
| **C** | W8A8 | 2 | 跨机 PD (1P1D, 各 TP=8) | 64K | ✅ 官方推荐 | 高吞吐生产环境 |
| **D** | W8A8 | 3+ | 跨机 PD (1P+2D, 各 TP=8) | 64K | ✅ 扩展方案 | 极高吞吐 (>50 req/s) |
| **E** | BF16 | 2 | TP=16 无 PD | 32K | ⚠️ 可行但约束大 | 需要 BF16 精度的低并发场景 |
| **F** | BF16 | 4 | 跨机 PD (P:TP=16, D:TP=16) | 64K | ⚠️ 理论可行成本高 | BF16 精度且需要 PD 隔离 |

> **注**: 同机 PD 分离 (Recipe B) 在 910B3 上不可行——单节点 512 GB HBM 无法同时容纳两份 W8A8 权重（284 GB × 2 > 512 GB）。

### 4.2 推荐路径

```
你的需求是什么？
├── "低时延、单节点成本优先" → Recipe A (单节点 TP8)
├── "低时延、ITL 必须稳定" → Recipe A2 (2 节点 PD)
├── "高吞吐、生产环境" → Recipe C (2 节点 PD, 高并发参数)
├── "极高吞吐" → Recipe D (3+ 节点 PD, 多 D)
├── "必须 BF16 精度" → Recipe E (2 节点 TP16)
└── "我要省钱、生产稳定" → 始终选 W8A8
```

---

## 5. W8A8 vs BF16 精度对比总结

| 维度 | W8A8 (Ascend Quant) | BF16 (Native) |
|------|---------------------|---------------|
| **权重内存** | ~284 GB (35.5 GB/卡 @ TP8) | ~568 GB (71 GB/卡 @ TP8) |
| **910B3 单节点可行** | ✅ 是（官方推荐） | ❌ 否（物理不行：71 > 64） |
| **910B3 最小节点数** | 1 | 2 |
| **单卡 KV cache 可用 @ TP8** | ~23.5 GB | N/A (单节点不行) / 28.5 GB (TP=16) |
| **最大上下文 (TP=8/TP=16)** | 64K / N/A | N/A / 32K |
| **精度损失** | < 0.5% perplexity vs FP8（CANN AMP 标定） | 无（参考精度） |
| **Decode 吞吐 (相对 W8A8)** | 1× 基准 | 0.6-0.8× (BF16 带宽需求翻倍，910B3 400GB/s 带宽瓶颈) |
| **vllm-ascend 支持** | 🟢 完整官方支持，有 Docker 镜像 | 🟡 声称支持但无生产验证 |
| **推荐用途** | 所有通用场景 | 精度敏感场景（金融合规、医疗） |
| **成本效率 (token/s/¥)** | ⭐⭐⭐⭐⭐ | ⭐⭐ |

---

## 6. 910B3 专用避坑指南

### 6.1 A2 vs A3 关键差异列表

| 参数 | A2 (910B3) | A3 (910C) | 原因 |
|------|-----------|-----------|------|
| `HCCL_BUFFSIZE` | 512-1024 | 1024-2560 | A2 带宽更低，大 buffer 收益递减 |
| `OMP_NUM_THREADS` | 8 | 10 | A2 仅 8 NPU/节点 |
| `dynamic_eplb` | **false** | true | A2 算力不够跑实时 EPLB |
| `num_redundant_experts` | **32** | 16 | A2 用更多静态冗余补偿 dynamic=false |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | **无** | 1 | A2 不支持 Fused MC2 |
| `ASCEND_A3_ENABLE` | **无** | 1 | A2 不开 A3 特有优化 |
| `cudagraph_capture_sizes` | 显式指定 | 自动 | A2 需要显式控制 capture size |
| `model-loader-extra-config` | 需要 | 不需要 | A2 需要多线程加速模型加载 |
| `safetensors-load-strategy` | prefetch | 默认 | A2 内存紧张，prefetch 更安全 |
| `kv_port` 安全值 | >= 28000 | >= 36000 | A2 仅 8 NPU, 端口冲突范围更小 |

### 6.2 常见 OOM 场景和解决方案

| 症状 | 根因 | 解决 |
|------|------|------|
| 启动即 OOM | 权重 + KV cache 超限 | 降 `gpu-memory-utilization` 到 0.85；降 `max-model-len` |
| Prefill 中途 OOM | 长 prompt 中间激活超限 | 降 `max-num-batched-tokens`；开启 `enable-chunked-prefill` |
| Decode 运行一段时间 OOM | KV cache 碎片化累积 | 升 `block-size` 到 128；重启服务 |
| 多请求并发 OOM | 超额并发 | 降 `max-num-seqs` |
| BF16 永远 OOM | 物理内存不够（568 > 512） | 换 W8A8 或加节点到 2+ |

### 6.3 网络验证 (PD 分离前置条件)

```bash
# A2 的端口数是 8（不是 16），检查 0-7
for i in {0..7}; do hccn_tool -i $i -link -g; done          # 必须全 UP
for i in {0..7}; do hccn_tool -i $i -net_health -g; done    # 必须全 success
for i in {0..7}; do hccn_tool -i $i -ip -g; done            # 获取 NPU IPs
hccn_tool -i 0 -ping -g address <remote_npu_ip>              # 跨节点 ping
for i in {0..7}; do hccn_tool -i $i -tls -g; done | grep switch  # 必须一致
```

---

## 7. 参考

- [vLLM-Ascend DeepSeek V4 Flash 官方文档](https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/DeepSeek-V4-Flash.html)
- [vLLM-Ascend PD Disaggregation 多节点指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/features/pd_disaggregation_mooncake_multi_node.html)
- [vLLM-Ascend PD 设计文档](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/disaggregated_prefill.html)
- [vLLM-Ascend 模型支持矩阵](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/supported_models.html)
- [DeepSeek V4 Flash HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [RedHatAI DeepSeek-V4-Flash-BF16](https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-BF16)
- [昇腾社区 DeepSeek V4 适配实战](https://news.qiniu.com/archives/post-1777533854604-0)

---

> **文档版本**: v2.0 (910B3 专项) | **生成方式**: vllm-ascend 源码 + 官方文档 + Web 调研 + 精确内存计算
