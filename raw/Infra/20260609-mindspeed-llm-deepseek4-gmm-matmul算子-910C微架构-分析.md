# MindSpeed-LLM DeepSeek V4 (Flash/Pro) 预训练 — GMM 与 MatMul 算子实现分析（基于昇腾 910C 微架构）

> 续接：`20260606-203000-mindspeed-llm-deepseek4-flash-pretrain-execution-flow-分析.md`
> 本文聚焦 **matmul / grouped_matmul(GMM)** 一类算子：先做全量盘点（Part 1），再选 **GMM** 在 **910C** 上做逐行剖析（Part 2）。
> 配置来源：`examples/mcore/deepseek4_flash/pretrain_deepseek4_flash_4k_A3_ptd.sh`

---

## ⚠️ Part 0 — 一个必须先纠正的前提：910C ≠ arch35

上一篇分析在 7.x 节把 AscendC 算子的 `arch35` 目录当作「910C」实现来剖析。这是**不准确的**，会直接影响本次「基于 910C」的结论。实测代码仓的 SOC→ARCH 映射如下：

`ops-transformer/CMakeLists.txt:55`
```
# SOC和ARCH目录对应关系:
#   ascend310p:arch20, ascend910b:arch22, ascend910_93:arch22,
#   ascend950:arch35, mc62:arch38, kirinx90:arch22, kirin9030:arch22
```

| SOC 字符串 | 产品 | 芯片 | ARCH 目录 | AI Core 宏 |
|-----------|------|------|----------|-----------|
| `ascend910b` | Atlas A2 | 910B | **arch22** | `__CCE_AICORE__==220` |
| **`ascend910_93`** | **Atlas A3** | **910C** | **arch22** | `__CCE_AICORE__==220` |
| `ascend950` | Atlas A5 / 950 系列 | 950PR/DT | **arch35** | `__CCE_AICORE__==310` |

旁证：
- 本次预训练启动脚本是 **A3**（`pretrain_deepseek4_flash_4k_A3_ptd.sh`）→ 编译目标 `ascend910_93` → **910C**。
- `cmake/scripts/util/ascendc_impl_build.py:655` 的 arch 选择逻辑：`ascend310p→arch31`、`ascend950→arch35`、**其余（含 910_93）→ arch22**。
- `docs/zh/invocation/quick_op_invocation.md`：「当设置为 ascend950 时会运行 arch35 目录下的示例文件」——arch35 是 **950**，不是 910C。

**结论：910C 跑的 GMM/MatMul 是顶层 `op_kernel/*.h`（arch22, `__CCE_AICORE__==220`）这条路径，不是 `arch35/`。** 本文后续所有 910C 的剖析都以此为准。

> 910C（910_93）与 910B（910b）共用 arch22 源码，少量差异由 `__NPU_ARCH__`（如 `3003/3113`）宏区分（核数、L2 容量、个别 prefetch 优化）。GEMM 的核内数据流（L1/L0A/L0B/L0C/Cube/Fixpipe）两者一致。

---

## Part 1 — DeepSeek V4 Pro 预训练全部 MatMul 算子盘点

### 1.1 关键 shape 参数（A3 单卡视角）

| 参数 | 值 | 参数 | 值 |
|------|----|------|----|
| hidden `H` | 4096 | seq `S` | 4096 |
| micro-batch `MBS` | 1 → 每 microbatch token `T=4096` | layers | 44（含 1 个 noop 层 43） |
| TP / PP / EP / CP | 1 / 4 / 32 / 1 | 本卡 local experts | 256/32 = **8** |
| n_heads | 64 | qk head_dim | 512 |
| qk_pos_emb(rope) | 64 | v_head_dim | 128 |
| q_lora_rank | 1024 | kv_lora/head_dim | 512 |
| o_lora_rank | 1024 | o_groups | 8 |
| num_experts | 256 | moe_router_topk | 6 |
| moe_ffn_hidden | 2048（SwiGLU→fc1 出 2×2048=4096） | shared_expert_inter | 2048 |
| index n_heads / dim / topk | 64 / 128 / 512 | hc_mult(MHC) | 4 |
| vocab | 129280 | dtype | bf16（router fp32） |

> 约定：矩阵乘记为 `[M,K]×[K,N]→[M,N]`。`T=4096` 是每个 microbatch 的 token 数（TP=1，SP 不切分）。MoE 的 `M_total` = group_list 累加和 = 本卡 8 个专家收到的 token 总数（数据相关，动态）。

### 1.2 算子全清单（按调用位置 + 数学 shape + 底层 kernel）

下表把「一层 TransformerLayer」内出现的全部矩阵乘列全。除非特别说明，**前向每项在反向都对应 2 个 GEMM**（`dX=dY·Wᵀ`、`dW=Xᵀ·dY`）。

#### A. MLA 注意力 — 线性投影（普通 MatMul，走 Cube）

| # | 算子/层 | 文件:行 | 数学 shape `[M,K]×[K,N]` | 底层 kernel | 类型 |
|---|--------|--------|--------------------------|------------|------|
| A1 | `linear_q` (wq_a) | g2_attention.py:127 | `[4096,4096]×[4096,1024]` | aclnnMatmul / MatMulV3 | LinearNoTP |
| A2 | `linear_kv` (wkv_a) | g2_attention.py:139 | `[4096,4096]×[4096,512]` | aclnnMatmul | LinearNoTP |
| A3 | `linear_q_up_proj` (wq_b) | g2_attention.py:165 | `[4096,1024]×[1024,32768]` (64头×512) | aclnnMatmul | ColumnParallel |
| A4 | `linear_o_down_proj` (wo_a) | g2_attention.py:486-493 | **einsum `sbgd,gld→sbgl`**，按 g=8 组的 batched GEMM | BMM / aclnnMatmul | ColumnParallel(分组) |
| A5 | `linear_o_up_proj` (wo_b) | g2_attention.py:191/494 | `[4096,8192]×[8192,4096]` + all-reduce | aclnnMatmul + HCCL | RowParallel |

#### B. DSA Lightning Indexer（投影 + Cube 内 QKᵀ）

| # | 算子 | 文件:行 | 数学 shape | 底层 kernel | 类型 |
|---|-----|--------|-----------|------------|------|
| B1 | index `wq`/`wk`/`weights_proj` 投影 | dsa_indexer.py | `[4096,*]→[*, 64×128]` 等 | aclnnMatmul | Linear |
| B2 | **Q×Kᵀ（indexer 打分）** | lightning_indexer kernel `ComputeMm1` | per head `[S,128]×[128,S]`，topk=512 稀疏 | AscendC Cube（lightning_indexer） | 自定义算子 |

#### C. Compressor（KV 压缩，fp32）

| # | 算子 | 文件:行 | 数学 shape | 底层 kernel |
|---|-----|--------|-----------|------------|
| C1 | `wkv` | compressor.py | `[T,*]×[*, coff·head_dim]` (fp32) | aclnnMatmul |
| C2 | `wgate` | compressor.py | 同上 (fp32) | aclnnMatmul |

#### D. Core Attention（G2 稀疏 Flash Attention）

| # | 算子 | 文件:行 | 数学 shape | 底层 kernel |
|---|-----|--------|-----------|------------|
| D1 | **QKᵀ** | g2_attention_kernel.py:47 | per head `[S,512]×[512,S]` | npu_fusion_attention / SparseFlashAttentionTriton（Cube） |
| D2 | **P×V** | g2_attention_kernel.py:79 | per head `[S,S]×[S,128]` | 同上（Cube） |

> 训练实际走 `--use-sparse-flash-attn` 的融合 kernel；上面是数学等价的两段 matmul。

#### E. MHC（Multi-Head Combining，每层 2 次：attn 前 + mlp 前）

| # | 算子 | 文件:行 | shape | 底层 kernel | 类型 |
|---|-----|--------|------|------------|------|
| E1 | `hc_fn` | mhc.py | `[T,4096]×[4096,(2+hc_mult)·hc_mult]`（小投影） | aclnnMatmul | LinearNoTP |
| E2 | `hc_pre_bmm` | mhc/pre_bmm.py:14 | batched，`H_pre×x` | **Triton BMM** | 自定义 |
| E3 | `hc_post_bmm1` | mhc/post_bmm1.py:14 | batched，`H_post×x` | **Triton BMM** | 自定义 |
| E4 | `hc_post_bmm2` | mhc/post_bmm2.py:21 | batched，`H_res×residual` | **Triton BMM** | 自定义 |

#### F. MoE（本文重点：GMM 出现处）

| # | 算子 | 文件:行 | 数学 shape | 底层 kernel | 类型 |
|---|-----|--------|-----------|------------|------|
| F1 | **router gate** | moe gating | `[4096,4096]×[4096,256]` (fp32) | aclnnMatmul | 普通 GEMM |
| F2 | **GMM1 / FC1 (gate_up)** | grouped_mlp `gmm_op(...,group_type=0)` | per expert `[m_e,4096]×[4096,4096]`，8 组 → `M_total` | **`npu_grouped_matmul` → grouped_matmul(GMM)** | 分组 GEMM |
| — | SwiGLU（非 matmul） | — | `[M_total,4096]→[M_total,2048]` | npu_swiglu（Vector） | — |
| F3 | **GMM2 / FC2 (down)** | grouped_mlp `gmm_op(...,group_type=0)` | per expert `[m_e,2048]×[2048,4096]`，8 组 | **grouped_matmul(GMM)** | 分组 GEMM |
| F4 | shared expert FC1/FC2 | mlp | `[T,4096]×[4096,4096]`；`[T,2048]×[2048,4096]` | aclnnMatmul / GMM(单组) | dense |

> GMM 反向（`grouped_mlp_with_comp_and_comm_overlap_all2all.py`）：
> - dgrad：`gmm_op(grad, W, group_type=0)`（同前向的 group_type=0，按 M 切）
> - wgrad：`gmm_op(Xᵀ, grad, group_type=2)`（**group_type=2，按 K 切**）

#### G. 输出 / 词表 / MTP

| # | 算子 | 文件:行 | 数学 shape | 底层 kernel |
|---|-----|--------|-----------|------------|
| G1 | **output_layer（词表投影）** | deepseek4_model.py | `[4096,4096]×[4096,129280]` | aclnnMatmul | ColumnParallel |
| G2 | `hc_head`（MHC 输出） | deepseek4_model.py | 小投影 + BMM | Triton/Matmul |
| G3 | MTP block | mtp | 复用一整套 A~F 的 GEMM（再来一层） | 同上 |

### 1.3 按「底层 kernel」归并

| 底层 kernel | 命中算子 | 是否本文深剖 |
|-------------|---------|:---:|
| **`grouped_matmul`（GMM, AscendC）** | F2 FC1、F3 FC2、F4 shared（及其反向 dgrad/wgrad） | ✅ **Part 2** |
| `aclnnMatmul`/`MatMulV3`（普通 Cube GEMM） | A1-A5、B1、C1-C2、E1、F1、G1 | 见 Part 2 末「与普通 MatMul 的关系」 |
| `npu_fusion_attention` / SparseFlashAttn | D1、D2 | 上一篇已覆盖 |
| `lightning_indexer`（AscendC Cube） | B2 | 上一篇已覆盖 |
| Triton BMM | E2-E4、G2 | 上一篇已覆盖 |

**算力占比直觉**：MoE 的 GMM（F2+F3）是单层最大的算力开销——256 专家、topk=6，等效把 `T×6=24576` 个 token-expert 对喂进两段 `K≈4096/2048, N≈4096` 的 GEMM。因此 **选 GMM 作为 910C 深剖对象**，最具代表性。

---

## Part 2 — GMM（grouped_matmul）在 910C 上的逐行实现剖析

### 2.1 为什么是 GMM，且只看 BF16 / 非量化路径

- 预训练是 `--bf16`，专家权重 bf16 → 命中 **`GMM_FLOAT`（非量化）** 分支。
- `npu_grouped_matmul` 的 BF16 实现 = `BF16GMMFunction`（`MindSpeed/mindspeed/core/transformer/moe/grouped_matmul_util.py:34`）。
- 910C = arch22 = `__CCE_AICORE__==220`，且 BF16 非量化路径是 **`GROUPED_MATMUL_CUBE_ONLY`（纯 Cube/AIC，AIV 不参与）**——见 `op_kernel/grouped_matmul_tiling_key.h:351`：
  `SET_GMM_NO_QUANT_TPL_ARGS(ASCENDC_TPL_MIX_AIC_1_0, GMM_TPL_BF16, 0, 0, GROUPED_MATMUL_CUBE_ONLY)`。
  `ASCENDC_TPL_MIX_AIC_1_0` = 1 AIC : 0 AIV → **整个 kernel 只在 Cube 核上跑**。

这让 910C 的微架构故事变得纯粹：**只用 Cube 流水线**。

### 2.2 910C（arch22）AI Core 微架构图

910C 的 AI Core 是 **Cube 核(AIC)** 与 **Vector 核(AIV)** 物理分离的「达芬奇」架构（一般 1 AIC : 2 AIV）。BF16 GMM 只用 AIC：

```
            昇腾 910C — 单个 AIC（Cube 核）数据通路（BF16 GMM 实际使用）
┌──────────────────────────────────────────────────────────────────────────────┐
│  HBM (GM)  ────────────────────  L2 Cache (片上, 多核共享)  ───────────────────  │
│     │  x[M,K]  weight[K,N]  group_list                                          │
│     │                                                                          │
│     │  ╔═══════════ MTE2（GM/L2 → L1 搬运引擎，PIPE_MTE2）═══════════╗          │
│     ▼  ▼                                                            ║          │
│  ┌────────────────────────────  L1 Buffer (≈512KB, 输入缓存) ──────────────────┐│
│  │   A 子块(bf16) [baseM×baseK]        B 子块(bf16) [baseK×baseN]               ││
│  │   depthA1=8 深度预取               depthB1=8 深度预取                        ││
│  └───────┬───────────────────────────────────┬─────────────────────────────────┘│
│          │  ╔═══ MTE1（L1 → L0A/L0B，含 ND→NZ/fractal 重排，PIPE_MTE1）═══╗      │
│          ▼  ▼                                  ▼                              ║  │
│   ┌──────────────┐   ┌──────────────┐                                          │
│   │   L0A (64KB) │   │   L0B (64KB) │   ← 双缓冲 dbL0A=dbL0B=2（乒乓预取）       │
│   │ [baseM×baseK]│   │ [baseK×baseN]│      16×16 fractal packing (C0=16)        │
│   └──────┬───────┘   └──────┬───────┘                                          │
│          │ (左矩阵)         │ (上矩阵)                                          │
│          ▼                  ▼                                                   │
│   ┌────────────────────────────────────────┐                                   │
│   │   CUBE MAC 阵列  16×16×16 (M0×N0×K0)     │  PIPE_M                          │
│   │   每条 cube 指令: L0C[16×16] += A·B      │  fp32 累加                        │
│   └───────────────────┬────────────────────┘                                   │
│                       ▼                                                         │
│             ┌────────────────────┐                                             │
│             │   L0C (≈128KB,fp32) │  累加器 [baseM×baseN]                        │
│             └─────────┬──────────┘                                             │
│                       │  ╔═══ FIXPIPE（L0C→GM，bias 加、fp32→bf16 cast，PIPE_FIX）═══╗
│                       ▼                                                         │
│     ───────────────────────────  GM: y[M,N] (bf16)  ────────────────────────── │
│                                                                                │
│   [Scalar 单元 + 同步]  PIPE_S 标量算地址/循环；MTE2→MTE1→M→FIX 之间用            │
│                         SetFlag/WaitFlag(EVENT_ID) 串起流水线，重叠搬运与计算    │
│                                                                                │
│   [AIV / UB]  本 kernel 不使用（CUBE_ONLY）。SwiGLU 激活是独立 Vector 算子。     │
└──────────────────────────────────────────────────────────────────────────────┘
```

**6 条并行指令流水（PIPE）**：`MTE2`(GM→L1) / `MTE1`(L1→L0) / `M`(Cube) / `FIX`(L0C→GM) / `V`(Vector，本 kernel 空) / `S`(Scalar)。GEMM 高性能的本质 = 让 **MTE2 预取下一块、MTE1 喂下一块、Cube 算当前块、Fixpipe 写上一块** 四件事在时间上重叠。

### 2.3 端到端调用链

```
[Python 训练层] MindSpeed-LLM MoE
  BF16GMMFunction (MindSpeed/.../grouped_matmul_util.py)
    → torch_npu.npu_grouped_matmul(x, weight, group_list=group_list, split_item=..., group_type=0)
        │
[CANN Host] aclnnGroupedMatmulGetWorkspaceSize() ── 计算 Tiling（baseM/N/K、coreNum、group 切分）
            aclnnGroupedMatmul()                 ── 下发 kernel 到 AIC 队列
        │
[AscendC Device, ops-transformer/gmm/grouped_matmul]
  op_kernel/grouped_matmul.cpp  __global__ grouped_matmul<...>()    ← kernel 入口/模板分派
    └─(GMM_FLOAT, TRANS_A=0,TRANS_B=0, CUBE_ONLY)→ GMM_CUBE_IMP(GMMProcess, ...)
        ├─ GMMProcess<...>::Process()      ← 遍历专家(group)、把输出切块分到各 Cube 核
        └─ GMMCompute<...>::MMCompute()    ← 每个 tile：SetTensorA/B + IterateAll
              └─ mm.IterateAll()  ← AscendC Matmul API：在 Cube 上跑 K 循环（L1/L0/MAC/Fixpipe）
```

### 2.4 Host Tiling：baseM / baseN / baseK 怎么定（910C 的 L0 容量反推）

GMM 不在 device 上自己算 L0 切分，而是 Host 端按 **L0 buffer 容量 + 双缓冲** 反推（`op_host/op_tiling/.../grouped_no_quant_matmul_tiling.cpp:45-55`，arch35 写法，arch22 等价逻辑走 `grouped_matmul_tiling.cpp` 的 matmul tiling API）：

```cpp
// baseK：L0B 一半（双缓冲）能放下多少列 bf16
baseK_ = (l0BSize / DB_SIZE) / (baseN_ * sizeof(bf16));     // DB_SIZE=2(双缓冲)
baseK_ = baseK_ & ~15;                                      // 16 对齐下取整
// maxBaseM：L0C(fp32) 能放下的最大行数
uint32_t maxBaseM = l0CSize / (baseN_ * sizeof(fp32));      // L0C 是 fp32 累加器
// baseM：L0A 一半能放下的行数，且不超过 maxBaseM，且封顶 BASE_M_DEFAULT(=128)
baseM_ = min( (l0ASize / DB_SIZE) / (baseK_ * sizeof(bf16)), maxBaseM );
if (baseM_ > BASE_M_DEFAULT) baseM_ = BASE_M_DEFAULT;
```

代入 910C（L0A=L0B=64KB，L0C≈128KB，bf16=2B，fp32=4B）取一组**代表性数值**：
- 设 `baseN=256` → `baseK = (64KB/2)/(256×2) = 64`；
- `maxBaseM = 128KB/(256×4) = 128`；`baseM = min(64KB/2/(64×2), 128)=min(256,128)=128`。
- **代表 tile：`baseM=128, baseN=256, baseK=64`**。校验：L0C=128×256×4=128KB ✓；L0A=128×64×2×2(db)=32KB ✓；L0B=64×256×2×2=64KB ✓。

静态 tiling 常量（`op_kernel/grouped_matmul_utils.h:177`）：`depthA1=depthB1=8`（L1 深 8 块）、`stepKa=stepKb=4`（K 方向预取 4 块）、`dbL0A=dbL0B=2`、`dbL0C=1`。

**group_list 语义**：长度 = local experts = 8 的累加和数组（cumsum）。`group_type=0` 表示「沿 M（token 维）切分专家」——每个专家拿到 `m_e = group_list[i]-group_list[i-1]` 个 token，K、N 固定（FC1: K=4096,N=4096；FC2: K=2048,N=4096）。

### 2.5 Kernel 入口与模板分派 — `grouped_matmul.cpp`

```cpp
// grouped_matmul.cpp:382  模板参数把「数据类型/转置/group类型/AIV:AIC比」全编进 kernel
template <int D_T_A, int D_T_B, int D_T_Y, int TRANS_A, int TRANS_B, int GROUP_LIST_TYPE,
          int IS_STATIC_TILING_API, int A8W4_KERNEL_TEMPLATE, int A16W8_KERNEL_TEMPLATE,
          int AIV_AIC_RATIO, bool IS_ENABLE_FIXED_AXIS>
__global__ __aicore__ void grouped_matmul(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, ...,
                                          GM_ADDR groupList, ..., GM_ADDR y,
                                          GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe tPipe;                                  // 389: 本核的流水/buffer 管理器
    AscendCUtils::SetOverflow(1);                 // 390: 打开溢出标志
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY); // 391: 声明本 kernel 只在 AIC(Cube) 上跑 ★
    GM_ADDR user1 = GetUserWorkspace(workspace);  // 392: 取设备侧 workspace
```
逐行：
- `391` `KERNEL_TYPE_AIC_ONLY`：**整张 kernel 只调度到 Cube 核**，与 2.1 的 `CUBE_ONLY` 呼应——AIV 不会被唤醒。
- BF16 非量化命中 `#elif defined(GMM_FLOAT)`（`grouped_matmul.cpp:594`）：

```cpp
#elif defined(GMM_FLOAT)            // 594: NO_QUANT 路径
  if (IS_STATIC_TILING_API == 0 && A8W4_KERNEL_TEMPLATE == ..._NONE) {
    GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling); // 597
    if constexpr (TRANS_A==0 && TRANS_B==0 && AIV_AIC_RATIO==GROUPED_MATMUL_CUBE_ONLY) { // 598 ★命中
      if (GROUP_LIST_TYPE==..._SPARSEM && gmmBaseParams_.groupType==0) {
        GMM_CUBE_IMP(GMMGroupMSparseProcess, false, false, false, matmulCFGUnitFlag); // 600
      } else {
        GMM_CUBE_IMP(GMMProcess, false, false, false, matmulCFGUnitFlag);             // 602 ★典型走这
      }
    } ...
```
逐行：
- `598`：前向 FC1/FC2 都是 **不转置 A、不转置 B、纯 Cube** → 命中。
- `602`：用 **`GMMProcess`** 作为「调度类」、`matmulCFGUnitFlag` 作为 Matmul 配置，展开 `GMM_CUBE_IMP` 宏。`matmulCFGUnitFlag`（utils.h:164）= `{doMultiDataLoad=true, enUnitFlag=true, enableKdimReorderLoad=true}`——开启「多数据加载 / UnitFlag 流水 / K 维重排加载」三项 Cube 优化。

### 2.6 `GMM_CUBE_IMP` 宏 — 把 Matmul 对象接上算子（grouped_matmul.cpp:173-191）

```cpp
#define GMM_CUBE_IMP(processClass, transA, transB, sync, cfg)                       \
  do {                                                                              \
    if ASCEND_IS_AIV { return; }                                                    \ //175 AIV 核直接返回(不参与)
    using matmulType = MMImplType<xType<transA>, weightType<transB>, yType, biasType, cfg>; \ //178 组装 Matmul 类型
    matmulType::MT mm;                                                              \ //179 实例化 Matmul 对象 mm
    GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);   \ //180 取基础参数(m,k,n,coreNum...)
    GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);     \ //181 取 TCubeTiling(baseM/N/K...)
    GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);    \ //182 取 m/k/n 列表地址
    mm.SetSubBlockIdx(0);                                                           \ //183
    mm.Init(&mmTilingData_, &tPipe);                                                \ //184 用 tiling 初始化 Matmul(分配 L1/L0)
    GMMCompute<matmulType, sync> computeOp(mm);                                     \ //185 计算类(持有 mm)
    computeOp.Init(x, weight, bias, scale, ..., y, user1, &gmmBaseParams_, &mmTilingData_, &tPipe); \ //186
    processClass<decltype(computeOp)> op(computeOp);                               \ //188 调度类(GMMProcess)
    op.Init(&gmmBaseParams_, &mmTilingData_, gmmArrayAddr_, groupList, tiling);     \ //189
    op.Process();                                                                  \ //190 ★进入主循环
  } while (0)
```
要点：
- `178` `MMImplType<A,B,C,Bias,cfg>`：A=`xType`(GM,ND,bf16)、B=`weightType`(GM,ND/NZ,bf16)、C=`yType`(GM,ND,bf16)、累加内部 fp32。这就是 AscendC 高阶 **Matmul 模板**——它内部封装了 L1/L0 切分、MTE 搬运、Cube 迭代。
- `184` `mm.Init(tiling)`：依据 `baseM/baseN/baseK/depth/step/db` 在 L1、L0A、L0B、L0C 上**划好 buffer 与双缓冲**。
- `190` `op.Process()`：真正开始算。

### 2.7 `GMMProcess::Process()` — 专家循环 + 输出分块的多核调度（grouped_matmul.h:278）

这是 GMM「**Grouped**」的灵魂：在一个 kernel 里顺序遍历 8 个专家，把每个专家的 `[m_e×n]` 输出切成 `baseM×baseN` 小块，**round-robin 分给所有 Cube 核**。

```cpp
__aicore__ inline void GMMProcess<ComputeType>::Process() {
  MNConfig mnConfig;                                       // 279 当前 group 的 m/k/n/base/idx 等
  ...
  AscendC::WaitPreTaskEnd();                               // 286 与上一个 kernel 做 task 级同步
  uint32_t groupListInnerShape = (groupListType==SPARSE)?2:1; // 289 group_list 形状 [e] 或 [e,2]
  uint32_t groupListShapeSize = groupNum * groupListInnerShape;
  for (uint32_t groupIdx=0, count=0; groupIdx<groupListShapeSize; groupIdx+=groupListInnerShape) { // 291 ★遍历专家
    UpdateMnConfig(mnConfig);                              // 292 累加上一个专家的 x/y/w 基址偏移
    int32_t splitValue = GetSplitValueFromGroupList(groupIdx, preOffset, gmmBaseParams, groupListGm); // 293 ★m_e=本专家token数(由cumsum差分)
    SetMNConfig(splitValue, groupIdx, mnConfig);           // 297 m=m_e, k/n 取自 tiling 列表
    if (mnConfig.m<=0 || k<=0 || n<=0) continue;           // 298 空专家跳过(MoE 常见)
    mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM); // 301 M 方向块数
    mnConfig.blockDimN = Ceil(mnConfig.n, mnConfig.singleN); // 302 N 方向块数
    uint32_t curCount = count + blockDimM*blockDimN;        // 304 全局块计数推进
    uint32_t curBlock = coreIdx>=count ? coreIdx : coreIdx + coreNum; // 305 本核领到的首个全局块号
    uint32_t thresholdM_dimN = thresholdBlockNum * blockDimN; // 306
    while (curBlock < curCount) {                          // 308 ★本核负责的块: 步长=coreNum
      MNBlockIdxCompute(mnConfig, curBlock, count, thresholdM_dimN); // 309 全局块号→(mIdx,nIdx)
      computeOp.MMCompute(groupIdx, mnConfig, coreIdx);    // 310 ★算这一块 GEMM
      computeOp.VectorCompute(mnConfig);                   // 311 BF16: 空操作
      curBlock += gmmBaseParams->coreNum;                  // 312 跳到下一个属于本核的块
    }
    count = curCount % gmmBaseParams->coreNum;              // 314 跨专家继承余数→块在专家间连续分配
  }
  computeOp.PostCompute();                                 // 316
  AscendC::SetNextTaskStart();                             // 317 放行下一个 kernel
}
```
关键设计逐条解释：
- **293 `GetSplitValueFromGroupList`**：`m_e` 来自 group_list 的相邻差分（cumsum）。这正是「专家收到多少 token 是动态的」的体现。
- **305 + 312 round-robin**：所有专家的所有输出块被拉平成一个全局序列，`curBlock += coreNum` 让 48/64 个 Cube 核**跨专家**均分块。**好处**：哪怕某专家只有 3 个 token（半块），也不会让一个核空转——块在专家边界**连续**分配（`314` 用 `count = curCount % coreNum` 把余数带到下一个专家）。这解决了 MoE 负载不均（专家 token 数差异大）的核心痛点。
- **309 `MNBlockIdxCompute`（对角线分块）**：大 shape（`blockDimM > thresholdDimM=5`）时不用简单行优先，而用「对角线」映射 `mIdx/nIdx`（grouped_matmul.h:95-108）。**好处**：相邻核访问的 weight/x 子块在 GM/L2 上错开，提升 **L2 命中与 HBM 带宽利用**，避免所有核同时抢同一段 weight。

### 2.8 `GMMCompute::MMCompute()` — 单块 GEMM 的 tensor 装配（grouped_matmul.h:546）

```cpp
__aicore__ inline void GMMCompute<...>::MMCompute(uint32_t groupIdx, MNConfig& mnConfig, uint32_t coreIdx, uint32_t listIndex) {
  if (subBlockIdx != 0) return;                            // 548 AIC 子块号(纯 cube 取 0)
  uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;       // 551 本块在 N 上的起始列
  uint32_t curSingleN = (nIdx<blockDimN-1)? singleN : n-tailN;   // 552 末块裁尾(N)
  uint32_t curSingleM = (mIdx<blockDimM-1)? singleM : m-mIdx*singleM; // 553 末块裁尾(M)
  uint64_t xOffset   = mnConfig.mIdx * mnConfig.singleM * mnConfig.k; // 555 A 子块在 x 的偏移
  uint64_t outOffset = mnConfig.mIdx*singleM*n + tailN;    // 559 C 子块在 y 的偏移
  // ── 装配 A(x)：单一大 tensor + 偏移(singleX==1) 或 分 group 指针(singleX==0)
  if (singleX==0) xGm.SetGlobalBuffer(GetTensorAddr<AT>(groupIdx, xTensorPtr));        // 565
  else            xGm.SetGlobalBuffer(GetTensorAddr<AT>(0, xTensorPtr)+mnConfig.xBaseOffset); // 568
  GlobalTensor<BT> weightGmLocal = SetGlobalBufferW(groupIdx, tailN, mnConfig);        // 570 ★定位本专家 weight 子块
  // ── 告诉 Matmul：整矩阵 shape & 本块 shape & A/B 地址
  mm.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);      // 571 原始 M,N,K
  mm.SetSingleShape(curSingleM, curSingleN, mnConfig.k);   // 572 ★本核这一块 = [curSingleM × curSingleN], K 整跑
  mm.SetTensorA(xGm[xOffset], transposeX);                 // 573 A = x 子块
  mm.SetTensorB(weightGmLocal, transposeW);                // 574 B = weight 子块
  SetGlobalBufferBias(groupIdx, tailN, mnConfig);          // 583 (本模型 disable-bias-linear, 无 bias)
  // ── 定位输出
  if (singleY==0) yGm.SetGlobalBuffer(GetTensorAddr<CT>(groupIdx, yTensorPtr));        // 588
  else            yGm.SetGlobalBuffer(GetTensorAddr<CT>(0, yTensorPtr)+mnConfig.yBaseOffset); // 591
  mm.template IterateAll<sync>(yGm[outOffset], 0);         // 597 ★★执行整块 GEMM(K 循环)→写 y
}
```
逐行要点：
- **552/553 裁尾**：`m_e` 一般不是 `baseM=128` 整数倍，最后一块行数 `< 128`，这里把 `curSingleM` 收窄，避免越界与无效算力。
- **570 `SetGlobalBufferW`**（grouped_matmul.h:521）：按 `groupIdx` 找到**本专家**的 weight 段（`wBaseOffset`，按 `k×n` 或 NZ 的 `AlignUp16(k)×AlignUp16(n)` 累加），再加 `tailN` 列偏移。这就是 grouped 与普通 matmul 的**唯一本质差别**：A 和 C 沿 M 用 group_list 偏移、B 用每专家独立权重段。
- **571/572** `SetOrgShape` vs `SetSingleShape`：前者是整专家 `[m_e,k,n]`（让 Matmul 知道 GM 上的 stride），后者是**本核这一块**实际要算的 `[curSingleM, curSingleN, k]`。
- **597 `IterateAll`**：把这一块的 **K 维整跑完**（FC1: K=4096 → `ceil(4096/64)=64` 次 K 迭代），结果直接写回 `yGm`。这一行是 Cube 真正干活的入口。

### 2.9 `mm.IterateAll()` 在 910C Cube 上的执行（微架构落地）

`IterateAll` 是 AscendC Matmul 模板的封装，对一块 `[curSingleM × curSingleN]`、K=`mnConfig.k` 的 GEMM，在 **单个 AIC** 上展开成如下流水（结合 2.2 的图）：

```
对 1 个输出 tile  C[baseM=128, baseN=256]，K=4096，baseK=64 → kLoop = 64：
┌── 预热：MTE2 把 A[128×64]、B[64×256] 从 GM 读进 L1 (PIPE_MTE2)
│
└── for ki in 0..63:                                   # K 方向主循环
      (MTE2) GM → L1   : 预取 A[:,ki+1],B[ki+1,:]      # depthA1/B1=8, 提前 stepK=4 块
      (MTE1) L1 → L0A  : A 子块 ND→NZ 16×16 fractal    # 喂 cube 左
      (MTE1) L1 → L0B  : B 子块 ND→NZ 16×16 fractal    # 喂 cube 上
      (M/Cube) L0A×L0B → L0C(+=)                       # 16×16×16 MAC, fp32 累加
        ▲ dbL0A=dbL0B=2: 算 ki 这块时, MTE1 已把 ki+1 灌进另一组 L0A/L0B(乒乓)
# K 循环结束, L0C 持有完整 C[128×256] (fp32)
(FIX/Fixpipe) L0C(fp32) → GM y 子块 : cast fp32→bf16 (+bias, 本模型无)
```
- **算力**：cube 每周期推进 16×16×16 MAC。一个 128×256×4096 的 tile ≈ `128×256×4096 / (16×16×16)` 个 cube 步 = 32768 步（被 K 循环×双缓冲流水化）。
- **延迟隐藏**：`depthA1=8、stepKa=4` 让 MTE2 的 GM→L1（最慢，HBM 延迟）提前 4 个 K 块发起；`dbL0A=dbL0B=2` 让 MTE1 的 L1→L0 与 Cube 计算乒乓重叠。理想情况下 **Cube 不空等**，吞吐逼近 MAC 峰值。
- **`enUnitFlag=true`（matmulCFGUnitFlag）**：用 UnitFlag 硬件流水标志替代部分手工 SetFlag/WaitFlag，降低同步开销。
- **`enableKdimReorderLoad=true`**：K 维加载顺序重排，让连续 K 块在 L1/L2 上更友好。

### 2.10 一个具体例子：FC1 在 910C 上跑一遍

设本卡某 microbatch 路由后，8 个专家收到 token：`group_list = [120, 200, 0, 90, 300, 64, 150, 180]`（`M_total=1104`）。FC1：`K=4096, N=4096(gate_up)`，`base=128×256×64`，假设 `coreNum=48`（AIC 数，举例）。

1. Host tiling：算出 `baseM=128,baseN=256,baseK=64`，把 `m/k/n` 列表与 cumsum group_list 写进 tiling。
2. `Process()` 遍历 8 专家：
   - 专家0：`m=120` → `blockDimM=ceil(120/128)=1, blockDimN=ceil(4096/256)=16` → 16 块；其末行块 `curSingleM=120`。
   - 专家2：`m=0` → `continue`（空专家，0 开销）。
   - 16+25+0+1×16+...（各专家块数）拉平成全局序列，`count` 跨专家继承余数 → 48 核 round-robin 连续领块。
3. 每核领到的块 → `MMCompute`：`SetTensorA(x 子块)`、`SetTensorB(专家自己的 w1 子块)`、`IterateAll` 跑 K=4096（64 次 baseK=64 迭代）→ 写 `y[*, 256列]`。
4. 全部专家算完 → `y[1104, 4096]`。下游 `npu_swiglu`（Vector 算子，独立 kernel）把它降到 `[1104, 2048]`，再进 **FC3=GMM2**（`K=2048,N=4096`）。

> 反向：dgrad 用同样 `group_type=0` 的 GMM（`grad_y[1104,4096]×W1ᵀ`）；wgrad 用 **`group_type=2`**（按 K 切，`xᵀ[4096,1104]×grad`，每专家独立累加权重梯度）——见 `grouped_mlp_with_comp_and_comm_overlap_all2all.py:133/221`。

### 2.11 GMM 与「普通 MatMul」(A1-A5/F1/G1) 的关系

普通 MatMul（`aclnnMatmul`/`MatMulV3`，对应 `MindSpeed/.../grouped_matmul.py` 之外的线性层）在 910C 上**内核数据流与 2.9 完全一致**（同样 L1/L0A/L0B/L0C/Cube/Fixpipe、同样双缓冲），区别只有三点：

| 维度 | 普通 MatMul (A1-A5,F1,G1) | GMM (F2/F3) |
|------|--------------------------|-------------|
| group 循环 | 无，单个 `[M,K]×[K,N]` | **有**，`Process()` 遍历 8 专家 |
| A/C 寻址 | 连续 | 沿 M 用 **group_list cumsum** 偏移 |
| B(weight) | 单权重 | **每专家独立权重段**（`SetGlobalBufferW`） |
| 末块/空块 | 常规裁尾 | 额外处理 **空专家(m=0) continue** |

也就是说：**GMM = MatMul 内核 + 一层「按 group_list 切 M / 按专家切 weight」的外层调度**。理解了 2.7-2.9，A1-A5、F1、G1 这些普通 GEMM 就是 GMM 去掉专家循环的特例。

---

## 三、结论与勘误

1. **勘误**：910C = `ascend910_93` = **arch22 / `__CCE_AICORE__==220`**；`arch35` 是 **Ascend 950**。上一篇 7.x 用 arch35 当 910C 的部分，结论应迁移到 arch22 顶层 kernel（数据流一致，差异在核数/L2 与个别 `__NPU_ARCH__` 宏优化）。
2. **全量 matmul 清单**：见 Part 1.2，共 7 组（MLA 投影 / DSA Indexer / Compressor / Core-Attn / MHC / MoE / 输出），底层归并为 5 类 kernel（Part 1.3）。**GMM 是单层最大算力**。
3. **GMM 在 910C 的实现链路**：`npu_grouped_matmul`(BF16) → `aclnnGroupedMatmul`(Host tiling) → `grouped_matmul.cpp`(`GMM_FLOAT`,`CUBE_ONLY`) → `GMMProcess::Process`(专家循环+对角线分块+round-robin 多核) → `GMMCompute::MMCompute`(装 A/B/C) → `IterateAll`(Cube K 循环：MTE2→L1→MTE1→L0A/L0B→Cube MAC→L0C→Fixpipe→GM，双缓冲流水)。
4. **910C 微架构要点**：BF16 GMM **纯 Cube(AIC)**；性能来自 6 条 PIPE 的搬运/计算重叠（`depthA1=8/stepK=4` 隐藏 HBM 延迟，`dbL0A=dbL0B=2` 隐藏 L1→L0 延迟）；MoE 负载不均由「跨专家连续 round-robin + 空专家跳过 + 末块裁尾」消化。

### 源码索引（便于复核）
- Kernel 入口/分派：`ops-transformer/gmm/grouped_matmul/op_kernel/grouped_matmul.cpp:382,391,594,602`
- 宏展开：`.../grouped_matmul.cpp:173-191`（`GMM_CUBE_IMP`）
- 调度循环：`.../op_kernel/grouped_matmul.h:278`（`Process`）、`:90`（对角线 `MNBlockIdxCompute`）
- 单块计算：`.../grouped_matmul.h:546`（`MMCompute`）、`:521`（`SetGlobalBufferW`）
- Host tiling：`.../op_host/op_tiling/grouped_matmul_tiling.cpp`、`.../arch35/grouped_no_quant_matmul_tiling.cpp:45`（公式参考）
- 静态常量：`.../op_kernel/grouped_matmul_utils.h:164,177-179`
- SOC→ARCH：`ops-transformer/CMakeLists.txt:55`；`cmake/scripts/util/ascendc_impl_build.py:655`
- Python 侧：`MindSpeed/mindspeed/core/transformer/moe/grouped_matmul_util.py:34`（`BF16GMMFunction`）、`grouped_mlp_with_comp_and_comm_overlap_all2all.py:49/60/133/221`
- 模型投影：`MindSpeed-LLM/mindspeed_llm/tasks/models/transformer/deepseek4/g2_attention.py:127-202`
- 启动配置：`MindSpeed-LLM/examples/mcore/deepseek4_flash/pretrain_deepseek4_flash_4k_A3_ptd.sh`
