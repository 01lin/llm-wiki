# DeepSeek V4 Flash @ 昇腾 910B3/910C PD 分离部署最佳 Recipe

> 生成时间: 2026-06-11 | 基于 vllm v0.18+ / vllm-ascend v0.18-v0.20

---

## 1. 模型与硬件基线

### 1.1 DeepSeek V4 Flash 模型规格

| 参数 | 值 |
|------|-----|
| **总参数量** | 284B |
| **每 token 激活参数** | ~13B |
| **层数** | 43 |
| **Hidden Size** | 4096 |
| **Attention 类型** | MQA (64 Q heads, 1 KV head) + LoRA 投影 (q_lora_rank=1024) + CSA/HCA 压缩 |
| **MoE 配置** | 256 routed experts, top_k=6, 1 shared expert, moe_intermediate=2048 |
| **Attention Head Dim** | 512 (QK RoPE dim=64) |
| **Vocab Size** | 129,280 |
| **最大上下文** | **1,048,576 (1M tokens)** |
| **RoPE** | theta=10000, YARN scaling (factor=16), per-layer compress_rope_theta=160000 |
| **精度** | 原生 FP4 expert weights + FP8 其他参数 (e4m3, 128×128 weight blocks) |
| **MTP** | 1 speculative token (deepseek_mtp) |
| **Sliding Window** | 128 |

> **关键架构洞察**: V4 Flash 不再使用 V2/V3 的 MLA（Multi-head Latent Attention），而是采用 MQA + LoRA + CSA/HCA KV 压缩的混合方案。KV 缓存大幅缩减：1M context 下约 9.62 GiB (bf16)，相比 V3.2 减少 ~90%。

### 1.2 昇腾硬件规格

| 规格 | Ascend 910B3 (A2) | Ascend 910C (A3) |
|------|-------------------|---------------------|
| **HBM 内存** | 64 GB HBM2e | 128 GB HBM2e |
| **内存带宽** | ~400 GB/s | **3.2 TB/s** |
| **FP16 算力** | ~320 TFLOPS | ~787 TFLOPS |
| **INT8 算力** | ~640 TOPS | ~1,600 TOPS |
| **单服务器 NPU 数** | 8 (Atlas 800T A2) | 16 (Atlas 800T A3) |
| **单服务器总 HBM** | 512 GB | 2,048 GB |
| **互联** | HCCS intra-node + RDMA (RoCE) inter-node | HCCS intra-node + RDMA (RoCE) inter-node |
| **对标 H100** | ~30-40% | ~60-80% |

### 1.3 模型内存需求估算

| 配置 | 权重大小 | KV Cache (单 seq, 8K ctx) | KV Cache (单 seq, 128K ctx) |
|------|---------|--------------------------|----------------------------|
| W8A8 quantized (Ascend) | ~71 GB | ~0.08 GB | ~1.2 GB |
| FP8 native | ~142 GB | ~0.08 GB | ~1.2 GB |
| BF16 | ~284 GB | ~0.15 GB | ~2.4 GB |

> **结论**: 910B 单卡 64GB 需要 W8A8 量化 + TP≥8 才能勉强装下权重。910C 单卡 128GB 可以用 FP8 单卡装下权重，但 PD 分离需要额外 KV cache 空间。

---

## 2. PD 分离架构总览

### 2.1 核心概念

PD 分离（Prefill-Decode Disaggregation）将推理过程拆分为两个独立服务：

```
Client Request → Proxy → Prefill Server (P Node)
                      ↘ Decode Server (D Node)
                              ↓
                      KV Cache Transfer (via Mooncake/RDMA)
```

- **P Node (Prefiller)**: 处理 prompt prefill，产生 KV cache → 传输给 D Node。侧重计算吞吐（compute-bound），需要高 TP 并行度。
- **D Node (Decoder)**: 接收 KV cache，执行自回归 decode。侧重内存带宽（memory-bound），需要高 DP 并行度以支持多并发。
- **KV Transfer**: 通过 MooncakeConnector (pull) 或 MooncakeLayerwiseConnector (push) 在 P/D 之间传输 KV cache，底层使用 RDMA (AscendDirectTransport)。

### 2.2 两种连接器模式

| 特性 | MooncakeConnector (Pull) | MooncakeLayerwiseConnector (Push) |
|------|--------------------------|----------------------------------|
| **传输方向** | D 节点从 P 节点拉取 KV cache | P 节点逐层推送 KV cache 到 D 节点 |
| **代理路由** | 请求先到 P，P 完成 prefill 后转发到 D | 请求先到 D，D 触发 P 做 prefill |
| **适用场景** | 通用场景，P/D 对称或接近对称的 TP | P 完成 prefill 立即释放，适合 P 资源紧张 |
| **优点** | 实现简单，调度逻辑清晰 | P 节点内存压力更小，prefill 完成即释放 |
| **缺点** | P 节点需保持 KV cache 直到 D 拉取完成 | 层间同步开销，实现更复杂 |

### 2.3 vLLM-Ascend PD 分离能力矩阵

| 能力 | 状态 |
|------|------|
| A2 (910B) 硬件 | 🟢 Functional |
| A3 (910C) 硬件 | 🟢 Functional |
| 对称 TP (P_tp = D_tp) | 🟢 Functional |
| 非对称 TP (P_tp > D_tp) | 🟢 Functional (需 P_tp % D_tp = 0) |
| MLA 模型 (V2/V3) | 🟢 Functional |
| GQA/MQA 模型 (V4 Flash) | 🟢 Functional |
| Expert Parallel (EP) | 🟢 Functional |
| MTP 投机解码 + PD | 🟢 Functional |
| Dynamic EPLB | 🟢 Functional (Decode 端) |
| 异构 P/D 硬件 (A2+A3) | 🔴 不支持 |

---

## 3. 场景一：极致低时延

### 3.1 场景定义

- **目标**: TTFT < 200ms, TPOT (ITL) < 15ms
- **典型应用**: 实时对话、代码补全、交互式 Copilot
- **核心矛盾**: PD 分离天然增加一次网络传输时延，需通过架构策略和参数调优抵消

### 3.2 架构策略

**核心思想**: P/D 同机部署 + Layerwise Push + 极小 TP + MTP 加速

```
单台 910C (A3) 服务器 (16 × 128GB)
├── P Instance × 2 (每实例 TP=1, DP=2)
│   └── 负责 prefill，产 KV cache 后 layerwise push
├── D Instance × 4 (每实例 TP=1, DP=4)
│   └── 负责 decode，低延迟消费 KV cache
└── Localhost RDMA (无跨机网络延迟)
```

**为什么不跨机？** 极致低时延场景下，跨机 RDMA 延迟（~5-20μs per layer × 43 layers ≈ 0.2-0.9ms）对 TTFT 有影响。同机内 HCCS 互联延迟几乎为零。

### 3.3 推荐配置

#### 硬件拓扑

| 角色 | 硬件 | NPU 数 | 说明 |
|------|------|--------|------|
| Prefill | 910C × 4 | 4 | TP=1, 每 NPU 独立处理一个 prefill 请求 |
| Decode | 910C × 8 | 8 | TP=1, 每 NPU 运行多个 decode 实例 |

> **优选 910C**: 128GB HBM 在 W8A8 下可容纳模型权重 (~71GB) + KV cache + 运行时开销。910B 的 64GB 在单卡 TP=1 时内存极度紧张，不推荐用于极致低时延场景。

#### Prefill 节点启动参数

```bash
#!/bin/bash
# Prefill Node (同机，kv_producer)
export USE_MULTI_BLOCK_POOL=1
export USE_MULTI_GROUPS_KV_CACHE=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

vllm serve /path/to/DeepSeek-V4-Flash-w8a8-mtp \
  --host 0.0.0.0 \
  --port 8100 \
  --tensor-parallel-size 1 \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --quantization ascend \
  --max-model-len 8192 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.85 \
  --block-size 64 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}' \
  --additional-config '{
    "recompute_scheduler_enable": true,
    "enable_shared_expert_dp": true,
    "enable_cpu_binding": true
  }' \
  --kv-transfer-config '{
    "kv_connector": "MooncakeLayerwiseConnector",
    "kv_role": "kv_producer",
    "kv_port": "36000",
    "kv_connector_extra_config": {
      "prefill": {"dp_size": 2, "tp_size": 1},
      "decode": {"dp_size": 4, "tp_size": 1}
    }
  }'
```

**参数解读**:
- `tensor-parallel-size 1`: 极致低时延下避免 TP 通信开销，单 NPU 独立 prefill
- `max-num-batched-tokens 4096`: 限制单次 prefill batch，防止 prefill 阻塞过久
- `max-model-len 8192`: 低时延场景通常上下文较短，减少 KV cache 预分配
- `block-size 64`: 更小的 block 粒度减少 prefill 调度等待
- `enforce-eager`: 跳过 CUDA graph 编译，加速启动和首次推理
- `MooncakeLayerwiseConnector`: Push 模式下 P 完成即释放，减少 P 节点内存压力

#### Decode 节点启动参数

```bash
#!/bin/bash
# Decode Node (同机，kv_consumer)
export USE_MULTI_BLOCK_POOL=1
export USE_MULTI_GROUPS_KV_CACHE=1
export DYNAMIC_EPLB="true"                # 动态专家负载均衡
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

vllm serve /path/to/DeepSeek-V4-Flash-w8a8-mtp \
  --host 0.0.0.0 \
  --port 8200 \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --quantization ascend \
  --max-model-len 8192 \
  --max-num-batched-tokens 120 \
  --max-num-seqs 24 \
  --gpu-memory-utilization 0.88 \
  --block-size 64 \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{
    "recompute_scheduler_enable": true,
    "multistream_overlap_shared_expert": true,
    "enable_cpu_binding": true,
    "enable_npugraph_ex": true,
    "eplb_config": {
      "dynamic_eplb": true,
      "expert_heat_collection_interval": 100,
      "algorithm_execution_interval": 20,
      "eplb_policy_type": 2,
      "num_redundant_experts": 8
    }
  }' \
  --kv-transfer-config '{
    "kv_connector": "MooncakeLayerwiseConnector",
    "kv_role": "kv_consumer",
    "kv_port": "36100",
    "kv_connector_extra_config": {
      "prefill": {"dp_size": 2, "tp_size": 1},
      "decode": {"dp_size": 4, "tp_size": 1}
    }
  }'
```

**参数解读**:
- `max-num-batched-tokens 120`: 解码端极小的 batch tokens 数，优先保证 ITL
- `cudagraph_mode: FULL_DECODE_ONLY`: 仅对 decode 阶段启用 CUDA graph，消除 launch overhead
- `block-size 64`: 更小 block → 更低碎片 → decode 更高效
- `MTP num_speculative_tokens 1`: 投机解码降低有效 ITL
- `enable_npugraph_ex`: NPU 算子级图优化，减少 kernel launch 开销

### 3.4 低时延关键 tuning 参数

| 参数 | 推荐值 | 作用 |
|------|--------|------|
| `HCCL_BUFFSIZE` | 1024 | 更大的 HCCL buffer 减少通信次数 |
| `block-size` | 64 (D), 64 (P) | 小 block 减少调度粒度 |
| `max-num-batched-tokens` | P:4096, D:120 | P 限制 batch 防阻塞，D 极小化保证 ITL |
| `enable_npugraph_ex` | true (D) | NPU 图优化消除 kernel launch overhead |
| `MTP speculative tokens` | 1 | 每步推测 1 token，降低有效 ITL |
| `DYNAMIC_EPLB` | true (D) | 动态专家负载均衡，减少 expert dispatch 时延 |
| `num_redundant_experts` | 8 | 低时延场景少冗余，减少 expert 切换开销 |

---

## 4. 场景二：高吞吐

### 4.1 场景定义

- **目标**: 最大化 tokens/s，TTFT < 1s 可接受
- **典型应用**: 批量评估、离线推理、数据合成
- **核心矛盾**: 需要在 P 和 D 之间取得最优的吞吐配比，避免任一端成为瓶颈

### 4.2 架构策略

**核心思想**: 跨机 PD 分离 + 高 DP Decode + 大 Batch Prefill

```
2 × 910C (A3) 服务器 + 2 × 910C (A3) 服务器
├── P 集群: 2 节点, 每节点 16 NPU
│   ├── DP=2 (跨节点), TP=8 (节点内)
│   └── 专注大 batch prefill，高吞吐 KV 产出
├── D 集群: 2 节点, 每节点 16 NPU
│   ├── DP=32, TP=1
│   └── 高并发 decode，最大化 token 产出
└── RDMA 跨机 KV transfer (MooncakeConnector Pull)
```

**为什么跨机？** 高吞吐场景需要 P 和 D 各自独立扩展。P 需要高 TP 以加速 prefill（受计算限制），D 需要高 DP 以容纳更多并发（受内存带宽限制）。独立扩展避免资源竞争。

### 4.3 推荐配置

#### Prefill 集群 (2 × A3)

```bash
#!/bin/bash
# Prefill Node (kv_producer) - 每个 Prefill 节点
export USE_MULTI_BLOCK_POOL=1
export USE_MULTI_GROUPS_KV_CACHE=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=2560
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export ASCEND_A3_ENABLE=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

# 环境特定参数
export HCCL_IF_IP=<本机 RDMA IP>
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0

vllm serve /path/to/DeepSeek-V4-Flash-w8a8-mtp \
  --host 0.0.0.0 \
  --port 8100 \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank <0 或 1> \
  --data-parallel-address <对方节点 IP> \
  --data-parallel-rpc-port 12321 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --quantization ascend \
  --max-model-len 131072 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.92 \
  --block-size 128 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}' \
  --additional-config '{
    "recompute_scheduler_enable": true,
    "enable_shared_expert_dp": true,
    "enable_cpu_binding": true,
    "multistream_overlap_shared_expert": true
  }' \
  --kv-transfer-config '{
    "kv_connector": "MooncakeConnectorV1",
    "kv_role": "kv_producer",
    "kv_port": "36000",
    "kv_connector_extra_config": {
      "prefill": {"dp_size": 2, "tp_size": 8},
      "decode": {"dp_size": 32, "tp_size": 1}
    }
  }'
```

**参数解读**:
- `data-parallel-size 2, data-parallel-size-local 1`: 跨 2 节点数据并行，每个节点 1 个 DP 实例
- `tensor-parallel-size 8`: 节点内 8 卡 TP，充分利用 910C 算力加速 prefill
- `max-num-batched-tokens 16384`: 大 prefill batch，提高算力利用率
- `HCCL_BUFFSIZE 2560`: 最大化 HCCL buffer 以支持大 batch 通信
- `MooncakeConnectorV1 (Pull)`: D 节点按需拉取，避免 P 节点等待，适合高吞吐异步模型
- `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`: 优化 all-reduce，减少 TP 通信开销

#### Decode 集群 (2 × A3)

```bash
#!/bin/bash
# Decode Node (kv_consumer) - 每个 Decode 节点 (共 2 节点)
export USE_MULTI_BLOCK_POOL=1
export USE_MULTI_GROUPS_KV_CACHE=1
export DYNAMIC_EPLB="true"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export ASCEND_A3_ENABLE=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

export HCCL_IF_IP=<本机 RDMA IP>
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0

vllm serve /path/to/DeepSeek-V4-Flash-w8a8-mtp \
  --host 0.0.0.0 \
  --port 8200 \
  --data-parallel-size 16 \
  --data-parallel-size-local 16 \
  --data-parallel-start-rank <0 或 16> \
  --data-parallel-address <对方节点 IP> \
  --data-parallel-rpc-port 12321 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --quantization ascend \
  --max-model-len 131072 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.94 \
  --block-size 128 \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{
    "recompute_scheduler_enable": true,
    "multistream_overlap_shared_expert": false,
    "enable_cpu_binding": true,
    "enable_npugraph_ex": true,
    "finegrained_tp_config": {"lmhead_tensor_parallel_size": 16},
    "eplb_config": {
      "dynamic_eplb": true,
      "expert_heat_collection_interval": 600,
      "algorithm_execution_interval": 50,
      "eplb_policy_type": 2,
      "num_redundant_experts": 16
    }
  }' \
  --kv-transfer-config '{
    "kv_connector": "MooncakeConnectorV1",
    "kv_role": "kv_consumer",
    "kv_port": "36200",
    "kv_connector_extra_config": {
      "prefill": {"dp_size": 2, "tp_size": 8},
      "decode": {"dp_size": 32, "tp_size": 1}
    }
  }'
```

**参数解读**:
- `data-parallel-size 32 (total)`: 极高 DP 并行度，每 NPU 独立 decode 实例，总并发可达 128 seqs
- `max-num-batched-tokens 256`: 适度增加 decode batch，在 ITL 可接受范围内提高吞吐
- `gpu-memory-utilization 0.94`: 最大化 KV cache 可用空间
- `multistream_overlap_shared_expert false`: 高吞吐下关闭 multi-stream overlap——减少 context switch，让计算更连续
- `num_redundant_experts 16`: 更多冗余 expert 副本降低 dispatch 冲突
- `finegrained_tp_config lmhead_tensor_parallel_size 16`: lm_head 用更高 TP 分片，减轻单卡内存（vocab=129K）

### 4.4 高吞吐关键 tuning 参数

| 参数 | 推荐值 | 作用 |
|------|--------|------|
| **P 节点数** | 2-4 | 根据请求速率线性扩展 |
| **D 节点数** | 2-4 | 根据并发数线性扩展 |
| **P : D 节点比** | 1:1 ~ 1:2 | DeepSeek V4 Flash 的 prefill 相对轻量，D 更易成为瓶颈 |
| `max-num-batched-tokens` | P: 16384-32768, D: 256-512 | P 大 batch，D 适度 batch |
| `max-num-seqs` | P: 8-32, D: 64-256 | D 端高并发 |
| `DYNAMIC_EPLB` | true (D 端) | 高并发下 expert dispatch 冲突显著，动态 EPLB 必要 |
| `num_redundant_experts` | 16-32 | 高并发下更多冗余降低 dispatch 排队 |
| `block-size` | 128 | 平衡内存利用率和调度开销 |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | 1 | 优化 TP 通信，对 P 端收益大 |
| `gpu-memory-utilization` | D: 0.94, P: 0.90 | D 端优先保证 KV cache 容量 |

---

## 5. 场景三：超长上下文

### 5.1 场景定义

- **目标**: 支持 128K-1M 上下文，单请求吞吐 ≥ 5 tokens/s
- **典型应用**: 长文档分析、代码库理解、Full-context RAG
- **核心矛盾**: KV cache 随上下文长度线性增长，1M context 下单个请求 KV cache 约 9.62 GB (bf16)，即使是 128GB 的 910C 也面临内存压力

### 5.2 DeepSeek V4 Flash 的 CSA/HCA KV 压缩

这是 V4 Flash 对超长上下文的关键利好：

```
注意力机制层级结构:
├── Sliding Window (128 tokens) → 局部注意力，KV 不压缩
├── C4A (4× compression) → KV 索引压缩 4× (部分层)
├── C128A (128× compression) → KV 索引压缩 128× (部分层)
└── Total: 73% FLOPs 减少, ~90% KV cache 减少 vs V3.2
```

| Context Length | 无压缩 KV Cache | C4A+C128A 后 KV Cache | 单次 Prefill FLOPs |
|---|---|---|---|
| 128K | ~1.2 GB | ~0.12 GB | ~6.6T |
| 256K | ~2.4 GB | ~0.24 GB | ~13.2T |
| 512K | ~4.8 GB | ~0.48 GB | ~26.4T |
| 1M | ~9.6 GB | ~0.96 GB | ~52.8T |

### 5.3 架构策略

**核心思想**: 单机 910C (128GB×16) + TP=8 + 动态 block 分配 + Prefix Caching

```
单台 910C (A3) 服务器 (16 × 128GB)
├── P Node (TP=8, EP enabled)
│   ├── 大 TP 加速 prefill（超长上下文 prefill FLOPs 大）
│   └── block_size=128 平衡调度和内存
├── D Node (TP=1, DP=8, EP enabled)
│   ├── 每 NPU 独立 decode，分散 KV cache 存储
│   └── Full decode CUDA graph + MTP
└── Localhost KV transfer (Layerwise Push)
```

**为什么单机优先？** 超长上下文下 KV cache 量虽已大幅压缩，但跨机 RDMA 传输仍是大开销（尤其是 128× compression 层需要传输更多层）。单机 PD 分离避免了跨机 KV 传输瓶颈。

**为什么仍做 PD 分离？** 即便同机部署，PD 分离也有两个关键收益：
1. Prefill 的超长上下文计算（52.8T FLOPs @ 1M）会严重干扰 decode 的 ITL
2. P 和 D 可以用不同的内存策略：P 用完释放，D 持久保存 KV cache

### 5.4 推荐配置

#### 910C 单机 PD (128K-256K 上下文)

```bash
#!/bin/bash
# ========== Prefill Node (同机, kv_producer) ==========
export USE_MULTI_BLOCK_POOL=1
export USE_MULTI_GROUPS_KV_CACHE=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=2560
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export ASCEND_A3_ENABLE=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

vllm serve /path/to/DeepSeek-V4-Flash-w8a8-mtp \
  --host 0.0.0.0 \
  --port 8100 \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --quantization ascend \
  --max-model-len 262144 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.85 \
  --block-size 128 \
  --enforce-eager \
  --enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}' \
  --additional-config '{
    "recompute_scheduler_enable": true,
    "enable_shared_expert_dp": true,
    "enable_cpu_binding": true,
    "multistream_overlap_shared_expert": true
  }' \
  --kv-transfer-config '{
    "kv_connector": "MooncakeLayerwiseConnector",
    "kv_role": "kv_producer",
    "kv_port": "36000",
    "kv_connector_extra_config": {
      "prefill": {"dp_size": 1, "tp_size": 8},
      "decode": {"dp_size": 8, "tp_size": 1}
    }
  }'

# ========== Decode Node (同机, kv_consumer) ==========
export USE_MULTI_BLOCK_POOL=1
export USE_MULTI_GROUPS_KV_CACHE=1
export DYNAMIC_EPLB="true"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export ASCEND_A3_ENABLE=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

vllm serve /path/to/DeepSeek-V4-Flash-w8a8-mtp \
  --host 0.0.0.0 \
  --port 8200 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --quantization ascend \
  --max-model-len 262144 \
  --max-num-batched-tokens 120 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.92 \
  --block-size 128 \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{
    "recompute_scheduler_enable": true,
    "multistream_overlap_shared_expert": true,
    "enable_cpu_binding": true,
    "enable_kv_cache_swap": true,
    "swap_space": 64,
    "eplb_config": {
      "dynamic_eplb": true,
      "expert_heat_collection_interval": 300,
      "algorithm_execution_interval": 30,
      "eplb_policy_type": 2,
      "num_redundant_experts": 8
    }
  }' \
  --kv-transfer-config '{
    "kv_connector": "MooncakeLayerwiseConnector",
    "kv_role": "kv_consumer",
    "kv_port": "36100",
    "kv_connector_extra_config": {
      "prefill": {"dp_size": 1, "tp_size": 8},
      "decode": {"dp_size": 8, "tp_size": 1}
    }
  }'
```

**参数解读**:
- `enable-prefix-caching` (P 端): 超长上下文场景通常有大量 prefix 重用（如 system prompt、共享文档），prefix caching 可显著减少 prefill 计算
- `max-model-len 262144`: 256K 上下文，适配大部分长文档场景
- `enable_kv_cache_swap true, swap_space 64`: 允许 KV cache 溢出到 CPU 内存（64GB），防止 OOM
- `block-size 128`: 较大 block 更适合长上下文的连续分配
- `DYNAMIC_EPLB`: 长上下文下不同位置的 expert 热度差异大，动态负载均衡更关键
- `recompute_scheduler_enable true`: 允许在内存紧张时重计算部分 attention（以计算换内存）

#### 910C 多机 PD (512K-1M 上下文)

当上下文超过 256K 时，单机内存不足以同时存放权重 + 大规模 KV cache。需要 PD 完全分离到不同物理节点：

```bash
# ========== Prefill 节点 (独立 910C × 16) ==========
# 关键变化:
# - TP=16 (用满 16 卡) 加速超长 prefill
# - max-model-len 1048576 (1M)
# - max-num-batched-tokens 32768 (更大 prefill chunk)
# - max-num-seqs 2 (超长上下文下并发数必须低)

export HCCL_BUFFSIZE=4096       # 超长上下文 prefill 通信量更大
export HCCL_OP_EXPANSION_MODE="AIV"

vllm serve /path/to/DeepSeek-V4-Flash-w8a8-mtp \
  --tensor-parallel-size 16 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --quantization ascend \
  --max-model-len 1048576 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.80 \
  --block-size 256 \
  --enforce-eager \
  --enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}' \
  --additional-config '{
    "recompute_scheduler_enable": true,
    "enable_shared_expert_dp": true,
    "enable_cpu_binding": true
  }' \
  --kv-transfer-config '{
    "kv_connector": "MooncakeConnectorV1",
    "kv_role": "kv_producer",
    "kv_port": "36000",
    "kv_connector_extra_config": {
      "prefill": {"dp_size": 1, "tp_size": 16},
      "decode": {"dp_size": 16, "tp_size": 1}
    }
  }'

# ========== Decode 节点 (独立 910C × 16) ==========
# 关键变化:
# - DP=16, TP=1 (最大内存可用量 per NPU)
# - max-num-seqs 8 (1M 上下文下每 seq KV cache ~1GB，8 seqs ~8GB per NPU)
# - swap_space 128 (大量 CPU swap 空间)

export DYNAMIC_EPLB="true"

vllm serve /path/to/DeepSeek-V4-Flash-w8a8-mtp \
  --tensor-parallel-size 1 \
  --data-parallel-size 16 \
  --enable-expert-parallel \
  --quantization ascend \
  --max-model-len 1048576 \
  --max-num-batched-tokens 120 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.92 \
  --block-size 256 \
  --no-enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{
    "recompute_scheduler_enable": true,
    "enable_cpu_binding": true,
    "enable_kv_cache_swap": true,
    "swap_space": 128,
    "eplb_config": {
      "dynamic_eplb": true,
      "expert_heat_collection_interval": 300,
      "algorithm_execution_interval": 50,
      "eplb_policy_type": 2,
      "num_redundant_experts": 8
    }
  }' \
  --kv-transfer-config '{
    "kv_connector": "MooncakeConnectorV1",
    "kv_role": "kv_consumer",
    "kv_port": "36200",
    "kv_connector_extra_config": {
      "prefill": {"dp_size": 1, "tp_size": 16},
      "decode": {"dp_size": 16, "tp_size": 1}
    }
  }'
```

### 5.5 超长上下文关键 tuning 参数

| 参数 | 128K-256K | 512K-1M | 分析 |
|------|-----------|---------|------|
| **部署模式** | 单机 PD | 多机 PD | 256K+ 单机内存难以同时容纳权重和 KV cache |
| `block-size` | 128 | 256 | 更大上下文用更大 block 减少 metadata 开销 |
| `max-num-seqs` (D) | 16 | 4-8 | KV cache 大小与 seq 数成正比 |
| `max-num-seqs` (P) | 4 | 1-2 | 超长 prefill 极耗内存，需极力限制并发 |
| `gpu-memory-utilization` (P) | 0.85 | 0.80 | P 端需为 prefill 中间激活预留更多空间 |
| `enable-prefix-caching` | true | true | 长上下文 shared prefix 复用率高 |
| `swap_space` (GB) | 32-64 | 64-128 | CPU swap 作为安全网，防止 OOM |
| `enable_kv_cache_swap` | true | true | 启用 KV cache CPU offload |
| `kv_connector` | Layerwise | MooncakeConnectorV1 | 单机用 layerwise 降延迟，跨机用 pull 更可靠 |
| P `tp_size` | 8 | 16 | 更大 TP 加速长 prefill，TP 通信开销低于 prefill 计算收益 |

---

## 6. 910B3 (A2) 适配方案

### 6.1 910B3 的约束

- **64 GB HBM**: W8A8 下权重 ~71GB，单卡放不下 → **必须 TP≥2**
- **内存带宽 ~400 GB/s**: decode 阶段 memory-bound，ITL 显著高于 910C
- **8 卡/节点**: TP 最大 8，DP 灵活性受限

### 6.2 910B3 推荐方案

| 场景 | 硬件 | 配置 | 预期性能 |
|------|------|------|---------|
| **低时延** | 2 × A2 (不推荐单机) | P: TP=2, DP=1; D: TP=1, DP=8 | TTFT~500ms, ITL~30ms |
| **高吞吐** | 4 × A2 | P: TP=8, DP=2 (跨节点); D: TP=1, DP=16 | ~800 tok/s total |
| **超长上下文** | 2 × A2 (≤64K context) | P: TP=8; D: TP=1, DP=4 | ~8 tok/s single seq @ 64K |

```bash
# 910B3 关键适配参数
--gpu-memory-utilization 0.88 \          # 64GB 下更保守
--block-size 64 \                         # 更小 block 减少碎片
--max-num-batched-tokens 8192 \           # P 端 batch 限制
--max-num-seqs 8 \                        # D 端并发限制
--quantization ascend \                    # 必须 W8A8
--additional-config '{
  "recompute_scheduler_enable": true,      # 内存紧张，启用重计算
  "enable_shared_expert_dp": true
}'
```

> **910B3 性能预估**: 受限于 64GB 内存和 400GB/s 带宽，单 seq decode 速率约为 910C 的 25-35%。对极致低时延场景，强烈建议使用 910C。

---

## 7. 参数全景分析

### 7.1 KV Transfer Config 参数矩阵

```json
{
  "kv_connector": "MooncakeConnectorV1|MooncakeLayerwiseConnector",
  "kv_role": "kv_producer|kv_consumer",
  "kv_port": "<port>",
  "engine_id": "<unique_id>",
  "kv_connector_extra_config": {
    "prefill":  {"dp_size": <p_dp>, "tp_size": <p_tp>},
    "decode":   {"dp_size": <d_dp>, "tp_size": <d_tp>}
  }
}
```

| 参数 | 低时延 | 高吞吐 | 超长上下文 |
|------|--------|--------|-----------|
| `kv_connector` | Layerwise | MooncakeConnectorV1 | Layerwise (单机) / V1 (多机) |
| P `dp_size` | 2 | 2 | 1 |
| P `tp_size` | 1 | 8 | 8-16 |
| D `dp_size` | 4 | 32 | 8-16 |
| D `tp_size` | 1 | 1 | 1 |
| `kv_port` | 36000+ | 36000+ | 36000+ |

### 7.2 环境变量决策树

| 变量 | 作用 | 低时延 | 高吞吐 | 超长上下文 | 必选? |
|------|------|--------|--------|-----------|-------|
| `VLLM_USE_V1=1` | V1 引擎 | ✅ | ✅ | ✅ | **必须** |
| `HCCL_OP_EXPANSION_MODE=AIV` | HCCL AIV 模式 | ✅ | ✅ | ✅ | **必须** |
| `TASK_QUEUE_ENABLE=1` | 任务队列 | ✅ | ✅ | ✅ | **必须** |
| `LD_PRELOAD=libjemalloc` | 内存分配器 | ✅ | ✅ | ✅ | 强烈推荐 |
| `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` | NPU 内存扩展 | ✅ | ✅ | ✅ | 强烈推荐 |
| `DYNAMIC_EPLB=true` | 动态 Expert 负载均衡 | ✅ (D) | ✅ (D) | ✅ (D) | 高并发推荐 |
| `USE_MULTI_BLOCK_POOL=1` | 多 block 池 | ✅ | ✅ | ✅ | 推荐 |
| `USE_MULTI_GROUPS_KV_CACHE=1` | 多组 KV cache | ✅ | ✅ | ✅ | 推荐 |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` | FlashComm 优化 | ❌ (TP=1 无需) | ✅ | ✅ | TP>1 推荐 |
| `VLLM_ASCEND_ENABLE_FUSED_MC2=1` | Fused MC2 | ❌ | ✅ | ✅ | A3 推荐 |
| `ASCEND_A3_ENABLE=1` | A3 特定优化 | ✅ | ✅ | ✅ | A3 必须 |
| `HCCL_BUFFSIZE` | HCCL buffer 大小 | 1024 | 2560 | 2560-4096 | 按需调整 |
| `OMP_NUM_THREADS` | OpenMP 线程数 | 10 | 10 | 10 | 推荐 |

### 7.3 vLLM 服务参数决策树

| 参数 | 低时延 | 高吞吐 | 超长上下文 | 单位 | 分析 |
|------|--------|--------|-----------|------|------|
| `tensor-parallel-size` (P) | 1 | 8 | 8-16 | cards | TP 增加加速 prefill 但增加通信 |
| `tensor-parallel-size` (D) | 1 | 1 | 1 | cards | Decode 是 memory-bound，TP 收益低 |
| `data-parallel-size` (P) | 2 | 2-4 | 1-2 | instances | P 的 DP 用于负载均衡，非加速 |
| `data-parallel-size` (D) | 4 | 32 | 8-16 | instances | D 的 DP 决定并发容量 |
| `max-model-len` | 8192 | 131072 | 262144-1048576 | tokens | 直接影响 KV cache 预分配 |
| `max-num-batched-tokens` (P) | 4096 | 16384 | 16384-32768 | tokens | Prefill batch 大小上限 |
| `max-num-batched-tokens` (D) | 120 | 256 | 120 | tokens | Decode batch 大小上限 |
| `max-num-seqs` (P) | 4 | 16 | 2-4 | seqs | P 端并发限制 |
| `max-num-seqs` (D) | 24 | 128 | 8-16 | seqs | D 端并发限制 |
| `gpu-memory-utilization` (P) | 0.85 | 0.90 | 0.80 | ratio | P 需中间激活空间 |
| `gpu-memory-utilization` (D) | 0.88 | 0.94 | 0.92 | ratio | D 需 KV cache 空间 |
| `block-size` | 64 | 128 | 128-256 | tokens | 小=低延迟，大=高内存效率 |
| `compilation-config.cudagraph_mode` | 无 (P) / FULL_DECODE_ONLY (D) | 同上 | 同上 | - | D 端 CUDA graph 收益显著 |
| `enforce-eager` | true (P) | true (P) | true (P) | - | P 端 skip graph compilation |
| `enable-prefix-caching` | false | false | true | - | 仅长上下文有 prefix 重用价值 |
| `speculative_config.num_speculative_tokens` | 1 | 1 | 1 | tokens | MTP 投机解码 |

### 7.4 Additional Config 参数详解

```json
{
  // === 调度与内存 ===
  "recompute_scheduler_enable": true,
  // 允许 scheduler 在内存不足时重计算（非 re-materialization，而是重新调度 prefill）
  // 建议: 所有场景开启

  // === Expert Parallel ===
  "enable_shared_expert_dp": true,
  // Shared expert 也参与 DP，减少单卡 shared expert 计算量
  // 建议: DP > 1 时开启

  "eplb_config": {
    "dynamic_eplb": true,
    // 动态 Expert Parallel Load Balancing
    // 建议: 高并发 Decode 场景开启

    "expert_heat_collection_interval": 600,
    // Expert 热度收集间隔（秒）。越长越稳定，越短越灵敏
    // 低时延: 100-300, 高吞吐: 600, 长上下文: 300

    "algorithm_execution_interval": 50,
    // 重平衡算法执行间隔（步数）
    // 低时延: 20, 高吞吐: 50, 长上下文: 30-50

    "eplb_policy_type": 2,
    // 策略类型: 1=static, 2=dynamic heat-based

    "num_redundant_experts": 16
    // 冗余 expert 副本数。更多=更低 dispatch 冲突，但更多内存
    // 低时延: 8, 高吞吐: 16, 长上下文: 8
  },

  // === 计算优化 ===
  "multistream_overlap_shared_expert": true,
  // Shared expert 和 routed expert 的 multi-stream overlap
  // 低时延: true, 高吞吐: false (减少 context switch), 长上下文: true

  "enable_cpu_binding": true,
  // CPU 核心绑定，减少 NUMA 跨 socket 访问
  // 建议: 所有场景开启

  "enable_npugraph_ex": true,
  // NPU 算子级图优化，减少 kernel launch overhead
  // 低时延: true, 高吞吐: true, 长上下文: false (内存优先)

  "finegrained_tp_config": {
    "lmhead_tensor_parallel_size": 16
  },
  // lm_head 的细粒度 TP（vocab=129K，lm_head 参数量大）
  // 高吞吐/长上下文: 推荐

  // === 超长上下文专用 ===
  "enable_kv_cache_swap": true,
  "swap_space": 64,
  // KV cache CPU offload，64-128 GB
  // 仅超长上下文场景需要
}
```

---

## 8. 部署前置检查清单

### 8.1 软件栈版本要求

| 组件 | 最低版本 | 推荐版本 |
|------|---------|---------|
| CANN | 7.0 | 8.0.RC2+ / 9.0.0 |
| NPU Driver | 25.5 | Latest |
| vLLM-Ascend | v0.18.0 | v0.20.2rc1+ |
| Python | 3.8 | 3.12 |
| Mooncake | v0.3.9 | v0.3.9+ |
| Docker Image | `quay.io/ascend/vllm-ascend:deepseekv4` | Latest |

### 8.2 跨节点通信验证 (PD 分离必须)

```bash
# 1. 检查 RDMA 链路状态 (A3: 16 ports, A2: 8 ports)
for i in {0..15}; do hccn_tool -i $i -link -g; done
# 全部必须 UP

# 2. 检查 net_health
for i in {0..15}; do hccn_tool -i $i -net_health -g; done
# 全部必须 success

# 3. 跨节点 PING
hccn_tool -i 0 -hccs_ping -g address <remote_npu_ip>

# 4. TLS 一致性
for i in {0..15}; do hccn_tool -i $i -tls -g; done | grep switch
# 所有节点必须一致
```

### 8.3 kv_port 避坑

Mooncake AscendDirectTransport 会占用 `[20000, 20000 + npu_per_node × 1000)` 端口范围。

| NPU 数 | 保留端口范围 | 推荐 kv_port |
|--------|------------|-------------|
| 8 (A2) | 20000-27999 | >= 28000 |
| 16 (A3) | 20000-35999 | >= 36000 |

### 8.4 系统调优

```bash
# NUMA 优化
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0

# 文件描述符限制
ulimit -n 1048576

# 关闭 CPU 频率调节
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

---

## 9. 场景总结对照表

| 维度 | 极致低时延 | 高吞吐 | 超长上下文 |
|------|-----------|--------|-----------|
| **核心目标** | TTFT<200ms, ITL<15ms | Max tokens/s | 支持 128K-1M context |
| **硬件选择** | 910C 单机 (16 NPU) | 910C × 4-8 节点 | 910C × 1-3 节点 |
| **PD 分离** | 同机 Layerwise Push | 跨机 Pull (MooncakeConnectorV1) | 同机 Push / 跨机 Pull |
| **P TP** | 1 | 8 | 8-16 |
| **P DP** | 2 | 2 | 1-2 |
| **D TP** | 1 | 1 | 1 |
| **D DP** | 4 | 32 | 8-16 |
| **Decode CUDA graph** | FULL_DECODE_ONLY | FULL_DECODE_ONLY | FULL_DECODE_ONLY |
| **MTP** | 1 token | 1 token | 1 token |
| **Dynamic EPLB** | 轻量 (8 redundant) | 重量 (16 redundant) | 中等 (8 redundant) |
| **Prefix Caching** | Off | Off | On |
| **KV Cache Swap** | Off | Off | On (64-128 GB) |
| **block_size** | 64 | 128 | 128-256 |
| **max-model-len** | 8K | 128K | 256K-1M |
| **P max-num-seqs** | 4 | 16 | 2-4 |
| **D max-num-seqs** | 24 | 128 | 8-16 |

---

## 10. 参考与致谢

- [vLLM Disaggregated Prefill 文档](https://docs.vllm.ai/en/latest/features/disagg_prefill.html)
- [vLLM-Ascend PD Disaggregation 多节点部署指南](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/features/pd_disaggregation_mooncake_multi_node.html)
- [vLLM-Ascend DeepSeek V4 Flash 部署文档](https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/DeepSeek-V4-Flash.html)
- [vLLM-Ascend PD Disaggregated Prefill 设计文档](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/disaggregated_prefill.html)
- [DeepSeek V4 Flash HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [Mooncake KV Transfer](https://github.com/kvcache-ai/Mooncake)
- [AISBench Benchmark Tool](https://github.com/AISBench/benchmark)
- [昇腾社区 DeepSeek V4 适配实战](https://www.hiascend.com/zh/developer/techArticles/20260425-1)

---

> **文档版本**: v1.0 | **生成方式**: 基于 vllm/vllm-ascend 源码分析 + 社区文档 + Web 调研综合
