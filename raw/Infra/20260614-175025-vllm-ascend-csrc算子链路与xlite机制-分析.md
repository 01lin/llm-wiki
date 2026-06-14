# vLLM-Ascend csrc 算子链路与 xlite 整图 runtime 机制分析

> 生成时间：2026-06-14
> 范围：① csrc 与 vllm_ascend/ops 的关系、算子编译/执行/注册全链路；② xlite 整图 runtime 的架构、核心优化、收益。
> 证据基线：`vllm-ascend` @ `8afdf356`（2026-06-13 快照）。行号/数量均为实测。
> 关联：[[20260614-163750-vllm-vs-vllm-ascend-目录与架构设计-分析]]、[[20260614-164853-vllm-vllm-ascend-三模块下钻-Scheduler闭环-DSA-execute-model-分析]]、[[20260611-010803-vllm-ascend-deepseek-v4-runtime算子协同优化-方案设计]]

---

## 第一部分：csrc 与 ops/ 的关系，以及算子编译/执行/注册逻辑

### 1.1 一句话定位

- **`csrc/`** = C++/AscendC **算子实现层**（host 端 tiling/shape + device 端 kernel），编译产物经 `TORCH_LIBRARY` 注册进 `torch.ops._C_ascend.*`。
- **`vllm_ascend/ops/`** = Python **算子编排/封装层**，三种调用对象混用：① 自研 `torch.ops._C_ascend.*`（来自 csrc）；② 厂商 `torch_npu.npu_*`（CANN 内置）；③ 自定义编排 op `torch.ops.vllm.*`（Python 实现 + `direct_register_custom_op`，主要用于通信/前后处理且要可被 torch.compile 捕获）。

底层逻辑：**csrc 管「单个昇腾算子怎么算得快」，ops/ 管「这些算子怎么在模型里编排、怎么和通信/图捕获拉通」**。两层通过 PyTorch 的 dispatcher（`torch.ops` 命名空间）解耦——csrc 改 kernel 实现不动 Python，ops/ 改编排不动 C++。

### 1.2 算子三层来源（在 ops/ 里混用）

| 来源 | 调用形式 | 实现位置 | 数量（实测） | 典型 |
|------|---------|----------|------|------|
| 自研 AscendC | `torch.ops._C_ascend.<op>` | `csrc/` | **65 个 ops.impl** | `npu_hc_pre/post`、`mla_preprocess`、`npu_add_rms_norm_bias`、`grouped_matmul_swiglu_quant`、`sparse_flash_attention`、`lightning_indexer` |
| 厂商 CANN | `torch_npu.npu_<op>` | torch_npu（外部） | — | `npu_rms_norm`、`npu_add_rms_norm`、`npu_quant_matmul`、`npu_transpose_batchmatmul` |
| Python 编排 | `torch.ops.vllm.<op>` | `ops/*.py` + `register_custom_ops.py` | **11 个 direct_register** | `mla_forward`、`maybe_all_gather_and_maybe_unpad`、`maybe_pad_and_reduce`、`matmul_and_reduce`、`prefetch_preprocess` |

证据（`ops/layernorm.py:73-82`）——同一函数里自研与厂商算子并存，按是否有 bias 分流：
```python
if <has bias>:  x, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(...)   # 自研
else:           x, _, residual = torch_npu.npu_add_rms_norm(...)                   # 厂商
```
取舍逻辑：厂商算子覆盖标准场景（稳定、免维护），自研算子补厂商缺的融合/量化/稀疏变体（如带 bias 的 add_rms_norm、mHC、DSA 稀疏注意力）。

### 1.3 单个自研算子的目录结构（标准 Ascend 三件套 + torch 适配）

以 `csrc/mla_preprocess/` 与 `csrc/attention/sparse_flash_attention/` 为样本，目录是固定范式：

```
csrc/<family>/<op>/
├── op_host/                      # host 侧（CPU 跑）
│   ├── <op>_def.cpp              # 算子原型定义（输入输出/属性）
│   ├── <op>_infershape.cpp       # shape 推导
│   ├── <op>_tiling.cpp/.h        # tiling：把问题切成 AI Core 能处理的块（性能关键）
│   └── op_api/aclnn_<op>.cpp/.h  # aclnn 调用接口（CANN 单算子 API）
├── op_kernel/                    # device 侧（AI Core 跑）
│   ├── <op>.cpp                  # kernel 入口
│   ├── arch22/ ...               # ★ 按微架构分实现（910B 等）
│   ├── arch35/ ...               # ★ 不同 SoC 不同 kernel（910_93/950 等）
│   │   └── service_cube_mla.h / service_vector_mla.h  # Cube/Vector 单元分别实现
│   └── <op>_template_tiling_key.h
├── <op>_torch_adpt.h             # ★ torch 适配：at::Tensor ↔ aclnn，可选参数兜底
├── CMakeLists.txt
└── docs/ examples/ README.md
```

三个设计要点：
1. **op_host / op_kernel 分离**是 AscendC 编程模型的硬性结构——host 算 tiling（怎么切块、用几个核），device 跑 kernel。tiling 质量直接决定算子性能（对照 §1.6 TileRT 也是 tile 级编排）。
2. **按微架构分目录**（`arch22`/`arch35`）：同一算子在 910B、910_93、950 上 kernel 实现不同（Cube/Vector 单元能力、寄存器、内存层级有别），这是「极致时延需要为具体 SoC 定制 kernel」的工程体现。`mla_preprocess` 的 `op_kernel/kernel/iterators/*.inc`（gm→l1→l0、l0c→ub 等）直接暴露了昇腾多级内存（GM/L1/L0/UB/FB）的搬运迭代器——kernel 优化的本质就是这套内存搬运 + Cube/Vector 流水的编排。
3. **torch_adpt.h 是 C++ 侧的「胶水」**（`mla_preprocess_torch_adpt.h:22+`）：把 PyTorch 的 `at::Tensor`/`c10::optional` 转成 aclnn 需要的形式，对 optional 参数做空张量兜底（`descale0.has_value() ? ... : at::empty(...)`），再调 `op_host/<op>.h` 的实现。

### 1.4 注册：从 C++ 到 `torch.ops._C_ascend`

`csrc/torch_binding.cpp` 是总注册入口：

```cpp
#include "moe/.../<op>_torch_adpt.h"          // 1. 包含所有算子的 torch 适配头
...
TORCH_LIBRARY_EXPAND(CONCAT(_C, _ascend), ops) {   // 2. 注册到 _C_ascend 命名空间 (line 2126/2158)
    ops.def("mla_preprocess(Tensor hiddenState, ...) -> (...)");   // schema 声明
    ops.impl("mla_preprocess", torch::kPrivateUse1, &vllm_ascend::mla_preprocess);  // 绑实现 (PrivateUse1=NPU)
}
```
- `torch::kPrivateUse1` 是 PyTorch 给「非官方后端」预留的 dispatch key，昇腾 NPU 即注册在此。
- `#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS` / `#else` 两套注册体（torch_binding.cpp:2126 vs 2158）：ATB+直调 kernel 路径 与 通用路径分别注册不同算子集——编译期决定哪些算子可用。
- **meta/fake 注册**（`meta_registration.py:44`）：`Library("_C_ascend", "IMPL")` 为部分算子补 meta kernel（如 `get_masked_input_and_mask_meta`），让 torch.compile/ACL Graph 在「假执行」推 shape 时不真正下发——这是图捕获能工作的前提。

### 1.5 编译：build.sh → CMake → wheel

链路（`setup.py` + `csrc/build.sh` + `csrc/CMakeLists.txt`）：
```
pip install / setup.py build_ext
  └─ cmake_build_ext.configure (setup.py:259)   # 传 SOC_VERSION、CXX_COMPILER、CANN 路径
       └─ csrc/build_aclnn.sh (setup.py:224)     # 先生成 aclnn 接口
       └─ csrc/CMakeLists.txt
            ├─ include(cmake/opbuild.cmake) → gen_aclnn_with_opdef()  # 由 op_def 自动生成 aclnn (line 128-132)
            ├─ add_library(opsproto SHARED)        # 算子原型库 (line 211)
            ├─ add_library(op_host_aclnn SHARED)   # host 侧 aclnn 库 (line 186)
            ├─ add_subdirectory(<op>/op_host)      # 各算子 host 编译 (line 285)
            └─ install(...) → ascendc/common/act 等  # 装进 vendor 目录 (line 329)
```
- **按 SoC 编译**（build.sh:46-47）：`ASCEND_SOC_UNITS="ascend910b"`，支持 `ascend310p/910b/910_93/950/kirinx90`——`--soc=ascend910b` 决定编哪套 arch kernel。这解释了为什么 §1.3 要分 arch 目录。
- 产物：`vllm_ascend_C` 扩展（`.so`）+ vendor 算子包，随 wheel 分发或运行时 load。
- **依赖 CANN toolkit**（build.sh:70-73，`Ascend/ascend-toolkit/latest`）：编译期需要 CANN 的 aclnn 工具链与头文件。

### 1.6 执行：运行时调用链（以 DeepSeek V4 mHC 为例）

```
模型 forward (models/deepseek_v4.py:837 hc_pre)
  └─ torch.ops._C_ascend.npu_hc_pre(x, hc_fn, ...)      # Python 调 dispatcher
       └─ [dispatcher: PrivateUse1 key]
            └─ vllm_ascend::npu_hc_pre (csrc torch_adpt)  # C++ 适配层
                 └─ op_host tiling → 下发 op_kernel        # host 算切块, device 跑 AI Core
                      └─ AscendC kernel (Cube/Vector/AIV)   # 实际计算
```
编排级 op 的执行链（以 SP allgather 为例）则停在 Python：
```
attention forward → torch.ops.vllm.maybe_all_gather_and_maybe_unpad (dsa_v1.py)
  └─ _maybe_all_gather_and_maybe_unpad_impl (register_custom_ops.py:40)  # 纯 Python + 集合通信
     （配 _fake 版本 line 108，供图捕获推 shape）
```

### 1.7 ops/ 与 csrc 协同强度（实测）

各 ops 文件对自研算子的依赖密度：`fused_moe.py` 11 处 `_C_ascend`、`layernorm.py` 7 处、`mla/dsa/rope_dsv4/linear` 各 ~1 处自研 + 大量 torch_npu。说明 **MoE 与 norm 是自研算子最密集区**（融合诉求最强：gmm+swiglu+quant、add_rms_norm+bias），而 MLA/DSA 更多复用厂商 matmul + 少量自研稀疏/preprocess 算子。

---

## 第二部分：xlite 整图 runtime 机制

### 2.1 定位：vLLM-Ascend 的「第二条执行轨」

xlite 是一个**外部 C++ 二进制组件**（`xlite._C`，源码不在本仓），通过 `from xlite._C import AttnMeta, AttnMHA, Model, ModelConfig, Runtime, ScoringFuncSigmoid, ScoringFuncSoftmax`（xlite.py:34）暴露。它把**整个模型前向用一次 C++ 调用完成**，绕开 vLLM 的 Python 逐层 dispatch——对标 TileRT 的「整步单调用」，但保留 vLLM 的服务面（调度/KV/采样）。

### 2.2 接入架构：三个适配层 + 一个 dispatch

```
NPUPlatform (worker_cls = XliteWorker)              # platform.py:618 选择
  └─ XliteWorker(NPUWorker)                          # xlite_worker.py:22 (仅重写 init_device)
       └─ XliteModelRunner(NPUModelRunner)           # xlite_model_runner.py:25 (56 行极薄)
            └─ load_model(): model = XliteWrapper(model, vllm_config)  # 包装替换
```

`XliteWrapper.__call__`（xlite.py:708）是 dispatch 核心——按 batch 形态决定走 C++ 整图还是回退 Python：
```python
with_prefill = attn_state not in [DecodeOnly, SpecDecoding]
if not full_mode and dp_size > 1:
    use_xlite_graph = num_tokens <= num_reqs        # DP 下按 token/req 比判定纯decode
else:
    use_xlite_graph = not with_prefill or full_mode # 纯decode 或 full_mode 走 C++
if use_xlite_graph:
    self.xlite_model.forward(rt, input_ids, attn_meta, kv_caches, freq_cis, h, stream)  # ★ C++ 整图
else:
    return self.runnable(...)                        # 回退 vLLM 原生逐层
```
两种工作模式（xlite.py:741-748 注释）：
- **full_mode**：prefill 与 decode 都走 C++ 整图。
- **decode-only**：prefill 走 Python runnable，decode 走 C++ 整图（最常用——decode 才是时延敏感的热循环）。

### 2.3 模型构建：vLLM 权重 → xlite Model（适配器模式）

`XliteModel` 抽象基类（xlite.py:44）+ 架构特化子类（`LlamaXliteModel`/`QwenMoeXliteModel`/`Glm4MoeXliteModel`/`MiniMaxM2XliteModel`），职责是把 vLLM 已加载的 `nn.Module` 权重**零拷贝映射**进 `xlite._C.Model`：
```
initialize():
  _build_model_config()   # vLLM hf_config → xlite ModelConfig (vocab/hidden/heads/tp/ep/moe...)
  _build_model()          # 遍历 vLLM layers, 把 weight tensor 引用塞进 xlite_model 字段
  xlite_model.init(config, rank)
  _precompute_freqs_cis() # RoPE cache 预计算到 NPU
```
关键：`_build_model` 直接取 vLLM 的 `named_parameters()` 张量引用（如 `xlite_model.mha_qkv = [layer.self_attn.qkv_proj.weight ...]`），**不复制权重**——xlite 与 vLLM 共享同一份 HBM 权重，省一倍显存。量化场景额外处理 deq_scale（`_prepare_deq_scale_weights`，xlite.py:339，按 fixpipe 硬件要求把 FP32 拆成 TF32 布局）。

### 2.4 核心优化（机制层面）

| # | 优化 | 机制 | 证据 |
|---|------|------|------|
| 1 | **整模型前向单次 C++ 调用** | 全部 layer + attention + MoE 在 C++ runtime 内循环，Python 每 decode step 只 1 次调用 | xlite.py:781 `xlite_model.forward(...)` |
| 2 | **绕开 Python 逐层 dispatch** | 不走 vLLM 的 `for layer in layers` + 每算子 torch.ops dispatch，消除 per-op Python 开销 | xlite.py:750 use_xlite_graph 分支 |
| 3 | **常驻 tensor pool** | `Runtime.init_tensor_pool(size)` 预分配工作区，前向复用不重分配 | xlite.py:665-669 |
| 4 | **稳定地址 hidden_states 缓冲** | `self.hidden_states = torch.empty(max_num_tokens, hidden_size)` 一次分配, 每步 `h[:num_tokens]` 复用 | xlite.py:671-672 |
| 5 | **零拷贝权重共享** | 取 vLLM 张量引用而非复制 | xlite.py:228+ `_build_model` |
| 6 | **MoE EP/TP 原生支持** | xlite ModelConfig 直接配 moe_ep_size/moe_tp_size，C++ 内做专家路由 | xlite.py:417-419 |

与 ACL Graph（默认路径）的区别：ACL Graph 是「捕获 Python 下发的算子序列成图再 replay」，仍需每步 Python 构图参数（`update_full_graph_params`）；xlite 是「直接 C++ 实现整个前向」，连构图都不需要 Python——更彻底，但灵活性更低。

### 2.5 主要收益与代价

**收益**（对照极致时延目标，参见 [[20260611-010803-vllm-ascend-deepseek-v4-runtime算子协同优化-方案设计]] 的 host 开销清单）：
- 消除每 decode step 的 Python 逐层 dispatch 与多次 torch.ops 调用开销（H6 图下发的 Python 侧）——这是 batch=1 极小 batch 下 host 暴露开销的大头。
- 常驻 pool + 稳定缓冲消除每步内存分配抖动（对应「shape 桶/常驻缓冲」诉求）。
- 零拷贝权重省显存。

**代价/缺口**（实测限制）：
1. **架构白名单仅 MHA 类**（xlite.py:624-633）：Llama/Qwen2/Qwen3/Qwen3-MoE/GLM4-MoE/MiniMax-M2，`attn_type` 只有 `AttnMHA`——**无 MLA/DSA，DeepSeek V4 全系不可用**（这是面向 DeepSeek 极致时延的最大结构性缺口）。
2. **与投机解码硬互斥**（ascend_config.py:567-570 直接 raise）、与 PP 互斥（pp_size>1 raise）。
3. **block_size 强约束 128**（ascend_config.py:578，否则 warning）。
4. **热路径仍有一次 D2H + tolist**（xlite.py:771 `block_tables.cpu().tolist()`）——与 `d2h_sync_count==0` 目标冲突，是接入层待优化点。
5. 源码闭源（`xlite._C` 二进制），DSA/MTP 支持依赖 xlite 团队排期（组织依赖）。

### 2.6 与 csrc 算子的关系

xlite **不复用 csrc 的 `_C_ascend` 算子**——它是独立的 C++ runtime，内部有自己的 kernel 实现（通过 `xlite._C`）。两者是 vLLM-Ascend 的两套并行执行后端：
- 默认路径：vLLM Python 编排 → `torch.ops._C_ascend.*`（csrc）+ `torch_npu.*` + ACL Graph。
- xlite 路径：vLLM 仅做服务面 → xlite C++ runtime 包办整个前向（含其内部算子）。

> 这意味着 csrc 算子的协同优化（DSA 多流 prolog、mHC 融合）与 xlite 是**两条独立演进线**——前者服务默认路径与 DeepSeek V4，后者服务 MHA 类模型的极致时延。要让 DeepSeek V4 享受 xlite 式整图收益，要么扩 xlite 支持 DSA（路线 B），要么在默认路径内做 decode 全闭环（路线 A），见方案设计文档。

---

## 源码证据索引

| 主题 | 位置 |
|------|------|
| `_C_ascend` 注册（65 个 ops.impl） | `csrc/torch_binding.cpp:2126/2158` + 各 `ops.def/impl` |
| 算子三件套目录范式 | `csrc/mla_preprocess/{op_host,op_kernel}`、`csrc/attention/sparse_flash_attention/` |
| 按微架构分 kernel | `csrc/.../op_kernel/arch22/`、`arch35/`（service_cube/vector） |
| torch 适配胶水 | `csrc/mla_preprocess/mla_preprocess_torch_adpt.h:22+` |
| meta/fake 注册 | `vllm_ascend/meta_registration.py:44` |
| CMake/build 链路 | `setup.py:224/259`、`csrc/build.sh:46-73`、`csrc/CMakeLists.txt:128/186/211/285/329` |
| ops/ 三类算子混用 | `ops/layernorm.py:73-82`、`ops/fused_moe/fused_moe.py`(11×_C_ascend) |
| Python 编排 op（11 个） | `vllm_ascend/ops/register_custom_ops.py:40/108/220+` |
| xlite 导入符号 | `vllm_ascend/xlite/xlite.py:34` |
| xlite 接入三层 | `xlite/xlite_worker.py:22`、`xlite_model_runner.py:25`、`xlite.py:642-672` |
| xlite dispatch / 整图调用 | `xlite.py:708-803`（use_xlite_graph 判定、forward） |
| xlite 模型构建（零拷贝权重） | `xlite.py:223-309`（_build_model） |
| xlite 架构白名单 / MHA-only | `xlite.py:624-633` |
| xlite 互斥限制 | `vllm_ascend/ascend_config.py:559-582` |
| xlite 热路径 D2H+tolist | `xlite.py:771` |
