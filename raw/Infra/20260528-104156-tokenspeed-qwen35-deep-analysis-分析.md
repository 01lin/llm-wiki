# TokenSpeed Qwen3.5-397B-A17B 深度优化源码解读

> 分析日期：2026-05-28  
> 源码根路径：`/Users/linyi/code/Documents/code/tokenspeed/`  
> 分析对象：B200 TP8 NVFP4 **580 tok/s** agentic 工作负载（50K首轮 + 800后续，90%+ KV命中率）

---

## 源码文件索引（可跳转）

| 模块 | 文件 | 主要职责 |
|------|------|---------|
| 模型配置 | [qwen3\_5\_text\_base\_config.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/configs/qwen3_5_text_base_config.py) | HybridLayerType、head\_dim=256、mamba2\_cache\_params |
| 多模态配置 | [qwen3\_5\_config.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/configs/qwen3_5_config.py) | Qwen3\_5Config、Qwen3\_5MoeConfig、rope\_parameters |
| 混合注意力后端 | [hybrid\_linear\_attn.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py) | SimpleMambaPool、HybridLinearAttnBackend、MTP O(1)指针 |
| GDN gating kernel | [gdn.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/attention/linear/gdn.py) | fused\_gdn\_gating\_kernel（Triton） |
| MTP draft 模型 | [qwen3\_5\_nextn.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/models/qwen3_5_nextn.py) | Qwen3\_5ForConditionalGenerationNextN |
| MoE block | [qwen3\_5\_moe.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/models/qwen3_5_moe.py) | StreamFork 双流并行、DeepEP 路径 |
| 主模型 | [qwen3\_5.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/models/qwen3_5.py) | Qwen3\_5ForCausalLM |
| 通信算子 | [comm\_ops.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/models/base/comm_ops.py) | FusedReduceNormOp、DeferredReduceOp |
| LayerNorm | [layernorm.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/layernorm.py) | GemmaRMSNorm weight+1、forward\_with\_allreduce\_fusion |
| CUDA Graph | [cuda\_graph\_wrapper.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/execution/cuda_graph_wrapper.py) | CudaGraphWrapper capture/replay |
| Mamba host cache | [mamba\_cache\_host.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/cache/mamba_cache_host.py) | MambaPoolHost、cudaHostRegister、all-layer bulk transfer |
| Mamba transfer pool | [mamba\_pool.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/cache/transfer/mamba_pool.py) | MambaCachePool、逐层 PD disagg transfer |
| CUDA stream fork | [cuda\_stream.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/utils/cuda_stream.py) | StreamFork.scope / branch |
| MoE layer | [layer.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/moe/layer.py) | MoELayer、ep\_num\_redundant\_experts |

---

## 一、混合架构感知路由（Hybrid Architecture）

### 1.1 层类型定义

📄 [qwen3\_5\_text\_base\_config.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/configs/qwen3_5_text_base_config.py)

```python
class HybridLayerType(enum.Enum):
    full_attention    = "attention"          # 标准 MHA，走 paged KV cache
    linear_attention  = "linear_attention"   # GDN，走 SSM state
    swa_attention     = "swa_attention"      # Sliding Window Attention
    mamba2            = "mamba2"             # Mamba2 状态空间层
```

`layers_block_type` 属性按 `full_attention_interval` 周期性插入全注意力层：

```python
@property
def layers_block_type(self):
    types = []
    for i in range(self.num_hidden_layers):
        if (i + 1) % self.full_attention_interval == 0:
            types.append(HybridLayerType.full_attention)
        else:
            types.append(HybridLayerType.linear_attention)
    return types
```

Qwen3.5-397B 关键参数（同一文件）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `head_dim` | **256** | FA4 Blackwell 原生宽度（Qwen3 为 128） |
| `linear_conv_kernel_dim` | 4 | GDN conv 核宽度 |
| `linear_key_head_dim` | 128 | GDN KV 维度 |
| `linear_value_head_dim` | 128 | GDN value 维度 |
| `num_experts` | 512 | MoE 专家总数 |
| `num_experts_per_tok` | 10 | 每 token 激活专家数 |
| `shared_expert_intermediate_size` | 512 | 共享专家中间维度 |

### 1.2 后端路由实现

📄 [hybrid\_linear\_attn.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py)

```python
class HybridLinearAttnBackend(AttentionBackend):
    def _backend_for_layer(self, layer_id: int):
        # O(1) set lookup — 核心路由判断
        if self.linear_attn_backend is None or layer_id in self.full_attn_layers:
            return self.full_attn_backend   # FA3/FA4 + paged KV
        return self.linear_attn_backend     # GDN/Mamba SSM state
```

`forward()` 统一入口，根据 `layer_id` 透明分发，调用方不感知后端切换。

---

## 二、GDN（Gated DeltaNet）线性注意力

### 2.1 GDN Gating 融合内核

📄 [gdn.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/attention/linear/gdn.py)

朴素 gating 计算：`a` → `softplus(a + dt_bias)` → `× (-exp(A_log))` = `g`，需要 3 次 DRAM round-trip。

TokenSpeed 将其合并为单个 Triton kernel：

```python
@triton.jit
def fused_gdn_gating_kernel(
    g, A_log, a, dt_bias, seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,       # softplus beta 系数
    threshold: tl.constexpr,  # 数值稳定性阈值（20.0）
    BLK_HEADS: tl.constexpr,
):
    # Step 1: a + dt_bias（in register）
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    # Step 2: softplus — 大于 threshold 时直接 x（避免 exp overflow）
    softplus_x = tl.where(
        beta * x <= threshold,
        (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    # Step 3: g = -exp(A_log) * softplus(x)（3步融合，1次写）
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
```

**效果**：3 次 DRAM 读写 → 1 次，对 GDN recurrent decode 路径（小 batch）影响显著。

### 2.2 GDN Prefill（chunk 并行）

GDN prefill 走 `chunk_gated_delta_rule`（来自 `tokenspeed_kernel._triton`），每 `FLA_CHUNK_SIZE=64` token 为一块的 DeltaNet chunked 并行实现：

```python
# hybrid_linear_attn.py — forward_extend 路径
from tokenspeed_kernel._triton import chunk_gated_delta_rule

h, o = chunk_gated_delta_rule(
    q=q, k=k, v=v, g=g,
    initial_state=initial_state,   # 从 SimpleMambaPool 读取历史状态
    output_final_state=True,       # 写回更新后的状态
    output_h=need_h_track,         # 是否输出中间 h（checkpoint 追踪所需）
    chunk_size=self.chunk_size,    # FLA_CHUNK_SIZE
)
```

### 2.3 GDN Decode（O(1) recurrent update）

```python
# hybrid_linear_attn.py — forward_decode 路径
from tokenspeed_kernel._triton import fused_sigmoid_gating_delta_rule_update
from tokenspeed_kernel._triton import causal_conv1d_update

# conv state O(1) 更新
new_conv = causal_conv1d_update(x_conv, conv_state, conv_weight, ...)

# SSM h state O(1) 更新
new_h = fused_sigmoid_gating_delta_rule_update(q, k, v, g, ssm_state, ...)
```

decode 阶段每步 O(1)（不随序列长度增长），是相比 Transformer O(n) KV 扫描的根本算法优势。

---

## 三、SimpleMambaPool：O(1) MTP 状态指针

📄 [hybrid\_linear\_attn.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py)

### 3.1 问题背景

MTP 投机解码时，每个 spec step 推进 Mamba/GDN 状态。verify 后 `accepted_length` 确定，需要把对应 step 的状态设为当前状态。

- **朴素做法**：拷贝 `accepted_length` 对应的 ssm_state 张量 → O(L × state_dim) 带宽
- **TokenSpeed 做法**：整数索引表 `current_input_indices` 记录"哪个 slot 是当前有效状态"，只更新整数 → **O(bs) 整数写**

### 3.2 Pool 布局

```python
class SimpleMambaPool:
    def __init__(self, size, num_mamba_layers, conv_shape, ssm_shape, device, spec_num_tokens=1):
        self.draft_base        = size
        self.draft_slots_per_req = spec_num_tokens - 1  # 每 req 的 draft slot 数

        total_size = size + size * self.draft_slots_per_req  # 主区 + draft 区

        # [num_layers, total_size, *state_shape]
        self.conv_state = torch.full((num_mamba_layers, total_size, *conv_shape), ...)
        self.ssm_state  = torch.full((num_mamba_layers, total_size, *ssm_shape),  ...)

        # 核心索引表：req_pool slot → 当前有效 state slot
        self.current_input_indices = torch.full(
            (self.current_input_size,), -1, dtype=torch.int32, device=device
        )
```

内存布局：
```
[  0 .. size-1  ][  size .. size + size*(spec_steps-1) - 1  ]
   主区（正常 req）        draft 区（每 req 有 spec_steps-1 个额外 slot）
```

### 3.3 O(1) verify 后状态更新（@torch.compile）

```python
@staticmethod
@torch.compile(dynamic=True)   # fuse 成单个 CUDA kernel
def _update_current_inputs_after_verify_kernel(
    req_pool_indices,       # [bs] req 在 pool 中的位置
    output_indices,         # [bs, spec_steps] 每步对应的 state slot
    accepted_lengths,       # [bs] 每个 req 实际 accept 的长度
    current_input_indices,  # [pool_size] 要更新的索引表（in-place）
    max_col: int,
):
    # 取出 accepted_length 对应列的 slot index
    idx      = (accepted_lengths.clamp(min=1, max=max_col) - 1).to(torch.int64)
    rows     = torch.arange(n, device=req_pool_indices.device)
    selected = output_indices[rows, idx].to(torch.int32)
    # in-place 写回索引表 — O(bs) 整数写，非 O(bs × state_dim) 张量拷贝
    current_input_indices[req_pool_indices.to(torch.int64)] = selected
```

同文件中 4 个关键函数均加 `@torch.compile(dynamic=True)`，将 Python 循环 + 中间张量 fuse 成单 CUDA kernel：

| 函数 | 作用 |
|------|------|
| `_build_mtp_output_indices_kernel` | 构建 draft 区 slot 索引矩阵 |
| `_get_current_input_indices_kernel` | 取出当前有效 state slot |
| `_get_current_input_indices_with_cow_kernel` | CoW 版本（prefix cache 命中路径） |
| `_update_current_inputs_after_verify_kernel` | verify 后 O(1) 指针更新 |

---

## 四、MTP（Multi-Token Prediction）投机解码

📄 [qwen3\_5\_nextn.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/models/qwen3_5_nextn.py)

### 4.1 Draft 模型结构

```python
class Qwen3_5ForConditionalGenerationNextN(nn.Module):
    def __init__(self, config, mapping, quant_config=None):
        if self.is_multimodal:
            config = config.text_config          # 多模态 checkpoint 取 text 部分

        if quant_config and quant_config.get_name() == "nvfp4":
            quant_config = None                  # MTP 部分保持 BF16（不量化）

        # 融合门：[hidden*2] → [hidden]
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        # 前置 GemmaRMSNorm（weight+1 语义）
        self.pre_fc_norm_embedding = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_fc_norm_hidden    = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)

        # 核心：只跑 1 层，且强制为全注意力层
        config.num_hidden_layers    = 1
        config.full_attention_interval = 1
        self.model = Qwen3_5ForCausalLM(config, mapping, quant_config)
```

### 4.2 MTP Forward 流程

```python
def forward(self, ctx, input_ids, positions, ..., captured_hidden_states=None):
    # 主模型 forward 时通过 CaptureHiddenMode.FULL 截获 hidden_states，传入此处

    # Step 1: token embedding
    input_embeds = self.model.embed_tokens(input_ids)

    # Step 2: 双路 norm + concat → [T, 2H]
    input_embeds  = self.pre_fc_norm_embedding(input_embeds)
    hidden_states = self.pre_fc_norm_hidden(captured_hidden_states)
    hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)

    # Step 3: FC gate 压缩 → [T, H]
    hidden_states = self.fc(hidden_states)

    # Step 4: 单层 transformer（全注意力，BF16）
    hidden_states, _ = self.model(input_ids, positions, ctx, out_cache_loc,
                                  input_embeds=hidden_states)

    # Step 5: logits → draft token
    return self.logits_processor(input_ids, hidden_states, self.lm_head, ...)
```

---

## 五、StreamFork：共享专家与路由专家双流并行

### 5.1 StreamFork 核心实现

📄 [cuda\_stream.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/utils/cuda_stream.py)

```python
class StreamFork:
    def __init__(self, aux_stream: torch.cuda.Stream | None):
        self.aux_stream  = aux_stream
        self.fork_event  = torch.cuda.Event() if aux_stream else None
        self.join_event  = torch.cuda.Event() if aux_stream else None

    @contextmanager
    def scope(self, *, enable: bool):
        """进入 fork scope：主流 record fork_event，结束时等 join_event"""
        self._active = enable and self.aux_stream is not None
        if self._active:
            self._current = torch.cuda.current_stream()
            self.fork_event.record(self._current)   # 主流打分叉点
        try:
            yield self
        finally:
            if self._active:
                self.join_event.wait(self._current) # 主流等 aux 完成后汇合

    @contextmanager
    def branch(self):
        """aux 流的工作区：等 fork 点，执行，record join_event"""
        if not self._active:
            yield; return
        with torch.cuda.stream(self.aux_stream):
            self.fork_event.wait(self.aux_stream)  # aux 等主流到达 fork 点
            yield                                   # 共享专家 GEMM 在此执行
            self.join_event.record(self.aux_stream) # aux 完成，通知主流
```

### 5.2 MoE 中的双流并行

📄 [qwen3\_5\_moe.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/models/qwen3_5_moe.py)

```python
class Qwen3_5MoeSparseMoeBlock(nn.Module):
    def __init__(self, ..., alt_stream: torch.cuda.Stream | None = None):
        self.stream_fork = StreamFork(alt_stream)

    def _forward_tp(self, hidden_states, ...):
        # 条件：共享专家存在 + 有 token + 正在做 CUDA Graph capture
        with self.stream_fork.scope(
            enable=(
                self.shared_expert is not None
                and hidden_states.shape[0] > 0
                and get_is_capture_mode()   # ← 只在 graph capture 时启用
            )
        ) as fork:
            with fork.branch():
                # ═══ aux 流：共享专家 GEMM ═══
                shared_output = self.shared_expert(hidden_states)

            # ═══ 主流：TopK routing + 路由专家 GEMM ═══
            topk_output = self.topk(hidden_states, router_logits)
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                topk_output=topk_output, ...
            )

        # join 后合并（shared_output 已由 join_event 保证就绪）
        if shared_output is not None:
            # fused gate + sigmoid + mul + add（tokenspeed-kernel Triton op）
            fused_gate_sigmoid_mul_add(
                hidden_states, self.shared_expert_gate.weight.squeeze(0),
                shared_output, final_hidden_states,
            )
```

**设计约束**：`get_is_capture_mode()` 来自 📄 [cuda\_graph\_wrapper.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/execution/cuda_graph_wrapper.py) 中的全局标志，capture 时录入双流模式，replay 时自动复现，非 capture 阶段不触发（避免流同步开销）。

---

## 六、DeepEP Expert Parallel

📄 [qwen3\_5\_moe.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/models/qwen3_5_moe.py)

```python
class Qwen3_5MoeSparseMoeBlock(nn.Module):
    def __init__(self, ...):
        # DeepEP 条件：all2all 后端是 deepep AND MoE 后端是 flashinfer_cutedsl
        self.use_deepep = (
            get_all2all_backend().is_deepep()
            and get_moe_backend().is_flashinfer_cutedsl()
        )

    def _forward_deepep(self, hidden_states, ...):
        # Gate 计算在本地 token（无需 AllGather，节省 O(T×H×TP) 通信）
        router_logits, _ = self.gate(hidden_states)

        # 共享专家在本地 token，TP 下显式 AllReduce
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            if self.mapping.dense.has_tp:
                shared_output = all_reduce(shared_output, tp_rank, tp_group)

        # TopK 在本地 token（路由结果）
        topk_output = self.topk(hidden_states, router_logits)

        # executor 内部处理 DeepEP dispatch → GEMM → combine（RDMA GPU-to-GPU）
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output, ...
        )
```

**冗余专家配置**（📄 [layer.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/moe/layer.py)）：

```python
self.experts = MoELayer(
    num_experts=config.num_experts + global_server_args_dict["ep_num_redundant_experts"],
    # ↑ 为热门专家配置冗余副本，缓解 expert imbalance 气泡
    ...
)
```

---

## 七、Fused AllReduce + Residual + RMSNorm

📄 [comm\_ops.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/models/base/comm_ops.py)

### 7.1 问题

每层 `RowParallelLinear` 输出后：`AllReduce → residual add → RMSNorm`，3 步 = 2× kernel launch + 3× DRAM round-trip。

### 7.2 FusedReduceNormOp

```python
class FusedReduceNormOp(CommOp):
    def _should_fuse(self, num_tokens: int) -> bool:
        return (
            use_all_reduce_mode
            and has_parallel                             # TP > 1
            and global_server_args_dict["enable_allreduce_fusion"]
            and 0 < num_tokens <= comm_fusion_max_num_tokens  # decode 小 batch 触发
        )

    def forward(self, hidden_states, residual, ctx):
        if self._should_fuse(hidden_states.shape[0]):
            # 单次调用完成：AllReduce + residual + RMSNorm
            hidden_states, residual, *_ = self.norm_module.forward_with_allreduce_fusion(
                self._rank, self._group, hidden_states, residual
            )
        else:
            # fallback：显式 AllReduce + unfused norm（prefill 大 batch）
            if self._has_parallel:
                hidden_states = all_reduce(hidden_states, self._rank, self._group)
            hidden_states, residual = self.norm_module(hidden_states, residual)
        return hidden_states, residual
```

`DeferredReduceOp`（零开销 pass-through）将 AllReduce 推迟到下游的 `FusedReduceNormOp` 一起执行，避免 AllReduce + norm 中间多一次 kernel。

### 7.3 GemmaRMSNorm weight+1 处理

📄 [layernorm.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/layernorm.py)

Qwen3.5 用 GemmaRMSNorm（`weight+1` 语义）。标准 `allreduce_residual_rmsnorm` kernel 不支持这个语义，TokenSpeed 预计算 `gemma_weight = weight + 1.0` buffer，传给标准 kernel 实现零修改复用：

```python
class GemmaRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        # (Chen-0210) 预计算 gemma_weight = weight + 1，load weight 时同步更新
        self.register_buffer("gemma_weight", self.weight.data + 1.0, persistent=False)

    def _weight_loader(self, param, loaded_weight):
        param.data.copy_(loaded_weight)
        self.gemma_weight = param.data + 1.0   # 同步更新 buffer

    def forward_with_allreduce_fusion(self, rank, group, x, residual=None, ...):
        # 传 gemma_weight（= weight + 1）而非 weight
        # 让 trtllm_allreduce_residual_rmsnorm 计算 x * (1 + weight)
        fused_result = allreduce_residual_rmsnorm(
            input_tensor=x,
            residual=residual,
            weight=self.gemma_weight,   # ← 关键：gemma 语义
            rank=rank,
            group=_get_process_group(group),
            ...
        )
```

底层实现：NVIDIA → `trtllm_allreduce_residual_rmsnorm`，AMD → `triton_allreduce_residual_rmsnorm`。

---

## 八、CUDA Graph 全路径覆盖

📄 [cuda\_graph\_wrapper.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/execution/cuda_graph_wrapper.py)

### 8.1 Batch Size 覆盖矩阵

```python
def get_batch_sizes_to_capture(config):
    if config.disable_cuda_graph_padding:
        # 无填充模式：精确覆盖，1-32 + 64/96/128/160
        capture_bs = list(range(1, 33)) + [64, 96, 128, 160]
    else:
        # 有填充模式：2的幂 + 8的倍数（减少 capture 数）
        capture_bs = [1, 2, 4] + [i * 8 for i in range(1, 21)]
    return [bs for bs in capture_bs if bs <= effective_max]
```

### 8.2 Capture 流程（含 StreamFork 激活）

```python
def _capture_one(self, bs: int):
    global _is_capture_mode, global_graph_memory_pool

    # 预热 4 次（稳定 CUDA 缓存）
    for _ in range(4):
        torch.cuda.synchronize(); dist.barrier()
        run_once()

    # DeepEP 设为 low-latency dispatch 模式
    self.deepep_adapter.capture()

    _is_capture_mode = True    # ← StreamFork.scope(enable=get_is_capture_mode()) 在此激活
    with torch.cuda.graph(graph, pool=global_graph_memory_pool, stream=self.stream):
        out = run_once()       # 录制 forward（含双流 MoE）
    _is_capture_mode = False

    global_graph_memory_pool = graph.pool()  # 传递给下一个 bs 复用显存池
    return graph, out
```

### 8.3 Mamba 状态在 Graph 中的处理

```python
# __call__ — use_graph + 需要 padding 时，同步 pad Mamba 相关 tensor
if use_graph and padded_bs != bs:
    pad = padded_bs - bs
    if mamba_pool_indices is not None:
        mamba_pool_indices    = F.pad(mamba_pool_indices,    (0, pad), value=0)
    if mamba_cow_src_indices is not None:
        mamba_cow_src_indices = F.pad(mamba_cow_src_indices, (0, pad), value=-1)
    if mamba_branching_seqlens is not None:
        mamba_branching_seqlens = F.pad(mamba_branching_seqlens, (0, pad), value=-1)

# Graph replay 后，O(1) 指针更新（不触发 graph 重录）
if self.drafter is not None and ctx.forward_mode.is_decode():
    accept_lengths = result[1]
    self.attn_backend.update_mamba_state_after_mtp_verify(accept_lengths, None)
```

---

## 九、Mamba 状态 L2 卸载与 PD 解耦

### 9.1 GPU-Visible 主机内存

📄 [mamba\_cache\_host.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/cache/mamba_cache_host.py)

```python
class MambaPoolHost:
    def __init__(self, ...):
        # 分配 pinned memory
        self.conv_buffer = torch.zeros((num_layers, size, *conv_shape), pin_memory=True)
        self.ssm_buffer  = torch.zeros((num_layers, size, *ssm_shape),  pin_memory=True)

        # cudaHostRegister — GPU kernel 可直接通过 PCIe 读写主机内存（零拷贝）
        platform.register_host_tensor_for_gpu_access(self.conv_buffer)
        platform.register_host_tensor_for_gpu_access(self.ssm_buffer)
```

### 9.2 全层批量写回（D→H，一次 kernel）

```python
def backup_from_device_all_layer(self, device_pool, host_indices, device_indices, ...):
    ptrs = self._ensure_kernel_ptr_tables(device_pool)  # 构建所有层的指针数组
    # 一次 kernel 完成所有层的 conv_state 转移（避免 N 次 kernel launch）
    transfer_kv_all_layer_mla(
        src_layers=ptrs["device_conv"], dst_layers=ptrs["host_conv"],
        src_indices=device_indices, dst_indices=host_indices, ...
    )
    transfer_kv_all_layer_mla(
        src_layers=ptrs["device_ssm"], dst_layers=ptrs["host_ssm"], ...
    )
```

### 9.3 逐层加载（H→D，PD 解耦关键）

📄 [mamba\_pool.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/cache/transfer/mamba_pool.py)

```python
class MambaCachePool:
    def __init__(self, device_pool, host_pool, io_backend):
        # 层序计数器：prefill 完成第 L 层后立即通知 decode，不等全部层完成
        self._counter = LayerDoneCounter(self.num_layers())
        device_pool.register_layer_transfer_counter(self._counter)

    def loadback(self, src_indices, dst_indices, layer_idx):
        # 单层转移（逐层触发，配合 wait_until 实现流水线）
        self.host_pool.load_to_device_per_layer(
            self.device_pool, src_indices, dst_indices, layer_idx, ...
        )
```

对应 [hybrid\_linear\_attn.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py) 中的等待逻辑：

```python
def get_mamba_params(self, layer_id):
    internal_idx = self.mamba_map[layer_id]
    # 等待第 internal_idx 层的 host→device 转移完成（prefill→decode 层序解耦）
    self.layer_transfer_counter.wait_until(internal_idx)
    return self.conv_state[internal_idx], self.ssm_state[internal_idx]
```

---

## 十、Copy-On-Write（CoW）Prefix Cache

📄 [hybrid\_linear\_attn.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py)

```python
@staticmethod
@torch.compile(dynamic=True)   # fuse 成单 CUDA kernel
def _get_current_input_indices_with_cow_kernel(
    req_pool_indices,
    mamba_pool_indices,         # 目标 slot（新分配）
    mamba_cow_src_indices,      # CoW 源 slot（prefix cache 共享 slot）
    mamba_branching_seqlens,    # 分支点序列长度
    current_input_indices,
    draft_base, draft_slots_per_req, ...
):
    # 命中 prefix cache 时：
    #   1. 将 src slot 的 Mamba 状态 copy 到 dst slot（CoW trigger）
    #   2. 将 dst slot 设为当前有效状态（写隔离）
    # 未命中时：直接从 current_input_indices 读（无额外开销）
```

90%+ prefix cache 命中场景下，CoW 确保每个请求有独立的 Mamba 状态写入空间，同时不为未命中请求引入额外开销。

---

## 十一、Checkpoint 追踪（Mamba Prefix Cache）

📄 [hybrid\_linear\_attn.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py)

FA 层 KV cache 天然支持按 token 的 prefix cache。GDN 的 SSM h 状态是累积的，TokenSpeed 在 prefill 时按 `FLA_CHUNK_SIZE=64` 边界保存中间 h，使后续请求可复用前缀的 Mamba 状态：

```python
# forward_extend（prefill）路径
need_h_track = any([track_ssm_h_src, track_ssm_h_dst])

h, o = chunk_gated_delta_rule(
    q, k, v, g,
    initial_state=initial_state,
    output_final_state=True,
    output_h=need_h_track,    # ← 输出 chunk-aligned 中间 h（checkpoint）
    chunk_size=FLA_CHUNK_SIZE,
)

# MambaForwardMetadata 中追踪的字段：
# track_ssm_h_src / track_ssm_h_dst  — SSM h 状态的 src/dst slot
# track_conv_indices                 — conv state 的保存位置
# track_ssm_final_src / dst          — 末层状态
```

空间开销：O(num_pages × head_dim × value_head_dim)，而非 O(seq_len)。

---

## 十二、FA4 支持（Blackwell head_dim=256）

📄 [qwen3\_5\_text\_base\_config.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/configs/qwen3_5_text_base_config.py)

```python
class Qwen3_5BaseTextConfig(PretrainedConfig):
    head_dim: int = 256   # FA4 在 Blackwell (B200) 上原生支持 256
    # Qwen3 为 128；Qwen3.5 特意升到 256 以充分利用 Blackwell WMMA 指令
```

📄 [layernorm.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/attention/backends/mha.py)（MHA backend 注册）：

```python
ATTN_BACKEND_MAP = {
    ...
    "fa4": "fa4",    # Flash Attention 4 Blackwell backend
    ...
}
```

视觉 encoder 也已独立支持 FA4（`mm_encoder_attention.py` 中 `vision_attn_fa4`）。

---

## 十三、通信算子编译时自动插入

📄 [comm\_ops.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/models/base/comm_ops.py)

"层编译器"（`compiler.py`）基于 `Placement` 标注自动推导通信需求，插入对应 `CommOp`：

| CommOp | 语义 | 触发条件 |
|--------|------|---------|
| `AllReduceOp` | Partial → Replicate | TP 标准路径 |
| `ReduceScatterOp` | Partial → Shard | 序列并行（RSAG 模式） |
| `AllGatherOp` | Shard → Replicate | RSAG → AR 切换 |
| `FusedReduceNormOp` | AllReduce + residual + RMSNorm | decode 小 batch |
| `DeferredReduceOp` | 零开销 pass-through | 标记推迟 AllReduce |
| `ResidualSliceOp` | 裁剪 residual | AR → RSAG 模式切换 |
| `FinalNormOp` | 最终 norm + 可选 AllGather | LM head 前恢复完整 token |

---

## 十四、整体性能叠加模型

```
Baseline（朴素 PyTorch MoE 推理） = 1×
│
├── [+] FA4 head_dim=256（Blackwell WMMA 原生）    ≈ 1.3-1.5×  FLOP utilization
├── [+] GDN O(1) decode（vs Transformer O(n)）     ≈ 1.4-2.0×  memory bandwidth
├── [+] NVFP4 量化（weight bandwidth ÷2）          ≈ 1.5-2.0×  weight bandwidth
├── [+] CUDA Graph 全路径（消除 kernel launch）     ≈ 1.1-1.2×  dispatch overhead
├── [+] Fused AllReduce+Residual+RMSNorm          ≈ 1.05-1.1× per-layer comm
├── [+] StreamFork 双流并行（shared+routed expert）≈ 1.1-1.15× MoE compute overlap
├── [+] DeepEP EP + 冗余专家（负载均衡）            ≈ 1.1-1.2×  expert load balance
├── [+] MTP 3-4 token/step（命中率相关）           ≈ 1.5-2.0×  throughput
├── [+] Prefix Cache 90%+ hit（KV + Mamba CoW）   ≈ 2-3×      TTFT / recompute
├── [+] torch.compile index fusion（MTP verify）  ≈ 1.02-1.05× verify overhead
└── [+] Mamba L2 卸载 + PD 逐层解耦               ≈ 1.1-1.3×  prefill-decode pipeline

综合（正交叠加，非完全乘法）：
B200 TP8 NVFP4 agentic（50K首轮+800后续）= 580 tok/s
```

---

## 十五、源码验证矩阵

| 技术点 | 源码文件 | 关键类/函数 | ✓ |
|--------|---------|------------|---|
| Hybrid 层类型路由 | [qwen3\_5\_text\_base\_config.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/configs/qwen3_5_text_base_config.py) | `HybridLayerType`, `layers_block_type` | ✅ |
| 后端路由分发 | [hybrid\_linear\_attn.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py) | `HybridLinearAttnBackend._backend_for_layer` | ✅ |
| GDN fused gating kernel | [gdn.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/attention/linear/gdn.py) | `fused_gdn_gating_kernel`（Triton） | ✅ |
| GDN O(1) decode | [hybrid\_linear\_attn.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py) | `fused_sigmoid_gating_delta_rule_update` | ✅ |
| SimpleMambaPool 布局 | [hybrid\_linear\_attn.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py) | `SimpleMambaPool.__init__` | ✅ |
| O(1) MTP state 指针 | [hybrid\_linear\_attn.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py) | `_update_current_inputs_after_verify_kernel` | ✅ |
| torch.compile index ops ×4 | [hybrid\_linear\_attn.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py) | `@torch.compile(dynamic=True)` | ✅ |
| MTP NextN draft 模型 | [qwen3\_5\_nextn.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/models/qwen3_5_nextn.py) | `Qwen3_5ForConditionalGenerationNextN` | ✅ |
| StreamFork CUDA stream | [cuda\_stream.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/utils/cuda_stream.py) | `StreamFork.scope / branch` | ✅ |
| MoE StreamFork 使用 | [qwen3\_5\_moe.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/models/qwen3_5_moe.py) | `_forward_tp` fork scope | ✅ |
| DeepEP 路径 | [qwen3\_5\_moe.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/models/qwen3_5_moe.py) | `_forward_deepep` | ✅ |
| 冗余专家 EP | [layer.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/moe/layer.py) | `ep_num_redundant_experts` | ✅ |
| fused AllReduce+RMSNorm | [comm\_ops.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/models/base/comm_ops.py) | `FusedReduceNormOp` | ✅ |
| DeferredReduceOp | [comm\_ops.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/models/base/comm_ops.py) | `DeferredReduceOp.forward`（pass-through） | ✅ |
| GemmaRMSNorm weight+1 | [layernorm.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/layernorm.py) | `gemma_weight = weight + 1.0` | ✅ |
| fused allreduce+norm（GemmaRMS）| [layernorm.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/layernorm.py) | `forward_with_allreduce_fusion` | ✅ |
| CUDA Graph capture | [cuda\_graph\_wrapper.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/execution/cuda_graph_wrapper.py) | `CudaGraphWrapper.capture / _capture_one` | ✅ |
| CUDA Graph Mamba padding | [cuda\_graph\_wrapper.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/execution/cuda_graph_wrapper.py) | `__call__` mamba tensor F.pad | ✅ |
| DeepEP Graph adapter | [cuda\_graph\_wrapper.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/execution/cuda_graph_wrapper.py) | `DeepEPCudaGraphRunnerAdapter` | ✅ |
| Pinned host Mamba cache | [mamba\_cache\_host.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/cache/mamba_cache_host.py) | `register_host_tensor_for_gpu_access` | ✅ |
| 全层 bulk D→H transfer | [mamba\_cache\_host.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/cache/mamba_cache_host.py) | `backup_from_device_all_layer` | ✅ |
| 逐层 PD disagg H→D | [mamba\_pool.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/cache/transfer/mamba_pool.py) | `MambaCachePool.loadback` | ✅ |
| 层序等待 counter | [mamba\_pool.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/cache/transfer/mamba_pool.py) | `LayerDoneCounter` | ✅ |
| CoW prefix cache | [hybrid\_linear\_attn.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py) | `_get_current_input_indices_with_cow_kernel` | ✅ |
| Checkpoint h 追踪 | [hybrid\_linear\_attn.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py) | `track_ssm_h_src/dst`, `track_conv_indices` | ✅ |
| head\_dim=256 FA4 | [qwen3\_5\_text\_base\_config.py](file:///Users/linyi/code/Documents/code/tokenspeed/python/tokenspeed/runtime/configs/qwen3_5_text_base_config.py) | `head_dim: int = 256` | ✅ |

---

## 关键洞察

1. **GDN 是算法基础**：从 O(n) KV decode 到 O(1) SSM state decode，其余所有工程优化都是在这个基础上叠加的。

2. **MTP × Prefix Cache 的乘法效应**：90%+ Mamba/KV prefix cache 命中 × MTP 3-4× token/step，二者相乘产生远超单一优化的效果，这是 agentic 580 tok/s 的核心来源。

3. **CUDA Graph 的设计约束**：`StreamFork` 只在 `get_is_capture_mode()=True` 时激活（capture 期间），replay 时双流模式已录入 graph 自动复现，capture 外不触发——graph-aware 设计范式。

4. **GemmaRMSNorm weight+1 预计算**：预计算 `gemma_weight` buffer，不修改底层 `trtllm_allreduce_residual_rmsnorm` kernel，以最小改动复用 NVIDIA 通信融合优化——"在基础设施约束内以最小代价适配"的典型工程决策。

5. **PD 解耦的层序传播**：`LayerDoneCounter` 使 prefill→decode 的 Mamba 状态传递从 all-or-nothing 变为逐层流水线，每完成一层立即通知，最小化 pipeline bubble。
