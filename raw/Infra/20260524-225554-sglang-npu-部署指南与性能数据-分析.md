# SGLang on Ascend NPU — 部署指南与推理性能

> 分析日期：2026-05-24  
> 代码来源：`sglang/docs/platforms/ascend/`、`sgl-kernel-npu/`

---

## 1. 环境要求

### 1.1 硬件支持

| 硬件 | 型号 | 说明 |
|------|------|------|
| Atlas 800I A2 | Ascend 910B（64G×8） | 支持大部分主流模型 |
| Atlas 800I A3 | Ascend 910_9382（64G×16） | 旗舰推理节点，支持全功能 |

### 1.2 软件依赖

| 组件 | 版本要求 |
|------|----------|
| Driver | Ascend HDK 25.0.RC1.1 |
| CANN | 8.3.RC1 / 8.5.0（推荐） |
| Python | >= 3.9 |
| PyTorch | >= 2.5.1 |
| torch-npu | >= 2.5.1-7.0.0 |
| pybind11 | 任意（`pip install pybind11`） |

---

## 2. 快速启动（推荐：Docker）

**这是最简路径，无需手动编译任何 kernel。**

### 2.1 拉取镜像

```bash
# Atlas 800I A3
docker pull quay.io/ascend/sglang:main-cann8.5.0-a3

# Atlas 800I A2（镜像 tag 不同）
docker pull quay.io/ascend/sglang:main-cann8.5.0-910b
```

### 2.2 启动容器

```bash
# A3 — davinci0~davinci15 (16卡)
docker run -it --rm --privileged --network=host --ipc=host --shm-size=16g \
    --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci8 --device=/dev/davinci9 --device=/dev/davinci10 --device=/dev/davinci11 \
    --device=/dev/davinci12 --device=/dev/davinci13 --device=/dev/davinci14 --device=/dev/davinci15 \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --volume /usr/local/sbin:/usr/local/sbin \
    --volume /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    --volume /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    --volume /etc/ascend_install.info:/etc/ascend_install.info \
    --volume /var/queue_schedule:/var/queue_schedule \
    --volume ~/.cache/:/root/.cache/ \
    --entrypoint=bash \
    quay.io/ascend/sglang:main-cann8.5.0-a3

# A2 — davinci0~davinci7 (8卡)
docker run -it --rm --privileged --network=host --ipc=host --shm-size=16g \
    --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci_manager --device=/dev/hisi_hdc \
    --volume /usr/local/sbin:/usr/local/sbin \
    --volume /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    --volume /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    --volume /etc/ascend_install.info:/etc/ascend_install.info \
    --volume /var/queue_schedule:/var/queue_schedule \
    --volume ~/.cache/:/root/.cache/ \
    --entrypoint=bash \
    quay.io/ascend/sglang:main-cann8.5.0-910b
```

### 2.3 验证安装

```bash
# 容器内
pip show sglang
python -c "import sgl_kernel_npu; print(sgl_kernel_npu.__path__)"
```

### 2.4 启动服务（最简示例）

```bash
# 设置镜像源（国内环境）
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=<your_token>

# 启动服务（以 Qwen2.5-7B 为例）
sglang serve \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --attention-backend ascend \
    --device npu &

# 测试接口
curl -X POST http://localhost:30000/generate \
    -H "Content-Type: application/json" \
    -d '{"text": "The capital of France is", "sampling_params": {"temperature": 0, "max_new_tokens": 16}}'
```

---

## 3. 从源码编译（高级）

仅在需要定制 kernel 或 Docker 镜像不可用时使用。

### 3.1 编译 sgl-kernel-npu

```bash
cd sgl-kernel-npu

# 1. 设置 CANN 环境
_CANN_PATH=$(cat /etc/Ascend/ascend_cann_install.info | grep Toolkit_InstallPath | awk -F= '{print $2}')
source ${_CANN_PATH}/set_env.sh

# 2. 编译全量（包含 DeepEP + kernels + attentions）
bash build.sh                     # 默认 A3（Ascend910_9382）

# A2 单独编译 DeepEP（ops2 路径）
bash build.sh -a deepep2          # A2 DeepEP

# 只编译算子库（不含 DeepEP）
bash build.sh -a kernels

# 3. 安装 whl
pip install output/sgl_kernel_npu*.whl
pip install output/deep_ep*.whl        # 如需 DeepEP
```

**编译产物**：
- `output/sgl_kernel_npu*.whl` — 主算子包（MLA/GQA/LoRA/Mamba/RMSNorm 等）
- `output/deep_ep*.whl` — DeepEP MoE 通信包
- `output/attentions*.whl` — AscendC 高性能 Attention 插件

### 3.2 编译选项说明

| 参数 | 说明 |
|------|------|
| `build.sh` | 全量编译（A3 deepep + kernels + attentions） |
| `-a deepep` | 仅 DeepEP-A3（Ascend910_9382，full mesh HCCS） |
| `-a deepep2` | 仅 DeepEP-A2（Ascend910B1，HCCS+RDMA） |
| `-a kernels` | 仅 sgl_kernel_npu 算子库 |
| `-a memory-saver` | 仅 torch_memory_saver |
| `-d` | Debug 模式（开启日志输出） |

### 3.3 关键环境变量

编译时 `build.sh` 会自动读取 `/etc/Ascend/ascend_cann_install.info` 定位 CANN 路径，如果路径非标准需手动设置：

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/8.5.0
```

---

## 4. 核心优化参数

启动服务时关键的 NPU 专属参数和环境变量：

### 4.1 必设环境变量

```bash
# NPU 内存分配
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# 多流并发（提高设备利用率）
export STREAMS_PER_DEVICE=32

# CANN 环境（每次启动前必须）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### 4.2 性能优化环境变量

```bash
# MLA 融合预处理算子（RMSNorm→MatMul→RoPE→Cache 一步完成，重要！）
export SGLANG_NPU_USE_MLAPO=1

# FIA NZ 格式（MLA 专用内存优化）
export SGLANG_USE_FIA_NZ=1

# 投机解码 overlap（spec + forward 并行流水）
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

# 多流（进一步并发）
export SGLANG_NPU_USE_MULTI_STREAM=1

# HCCL AllReduce 优化
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1000     # MoE decode 调小（约 650-720），prefill 调大（1536）

# CPU 亲和性绑定（生产环境）
export SGLANG_SET_CPU_AFFINITY=1
unset ASCEND_LAUNCH_BLOCKING

# 高性能 CPU 调度（系统级，需 root）
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
```

### 4.3 关键 launch_server 参数

| 参数 | 说明 |
|------|------|
| `--attention-backend ascend` | 使用昇腾 attention backend（必须） |
| `--device npu` | 指定 NPU 设备（必须） |
| `--moe-a2a-backend deepep` | MoE AlltoAll 使用 DeepEP-Ascend（MoE 模型必须） |
| `--deepep-mode auto/normal/low_latency` | DeepEP 模式：auto 自动选择；normal 高吞吐；low_latency 低延迟 |
| `--quantization modelslim` | 华为 msmodelslim 量化格式 |
| `--enable-dp-attention` | 开启 DP Attention（多 DP 并行时必须） |
| `--enable-dp-lm-head` | DP LM Head（decode 高并发时开启） |
| `--speculative-algorithm NEXTN` | NextN 投机解码（NPU 上推荐） |

---

## 5. SOTA 模型部署示例

### 5.1 DeepSeek-V3/R1（单机 A3，混合 PD 模式）

适用场景：1台 Atlas 800I A3（16卡），W4A8 量化，最经济部署。

模型权重：`Modelers_Park/DeepSeek-R1-0528-w4a8`

```bash
# 系统优化
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0; sysctl -w kernel.numa_balancing=0

# CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 关键环境变量
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export HCCL_BUFFSIZE=1600
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32

python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --tp 16 \
    --trust-remote-code \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --watchdog-timeout 9000 \
    --cuda-graph-bs 8 16 24 28 32 \
    --mem-fraction-static 0.68 \
    --max-running-requests 128 \
    --context-length 8188 \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --max-prefill-tokens 16384 \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --enable-dp-attention \
    --dp-size 4 \
    --enable-dp-lm-head \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --dtype bfloat16
```

### 5.2 DeepSeek-V3/R1（双机 A3，PD 分离模式）

适用场景：2台 Atlas 800I A3（共 32 卡），W8A8 量化，低延迟生产部署。

模型权重：`State_Cloud/Deepseek-R1-bf16-hfd-w8a8`

**Prefill 节点（2台中的 P 节点）：**

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export ASCEND_MF_STORE_URL="tcp://<PREFILL_HOST_IP>:24669"
export HCCL_BUFFSIZE=1536
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1
export TASK_QUEUE_ENABLE=2
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

python -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --host $PREFILL_HOST_IP \
    --port 8000 \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port 8996 \
    --disaggregation-transfer-backend ascend \
    --trust-remote-code \
    --nnodes 1 --node-rank 0 \
    --tp-size 16 \
    --mem-fraction-static 0.6 \
    --attention-backend ascend --device npu \
    --quantization modelslim \
    --max-running-requests 8 \
    --context-length 8192 \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --max-prefill-tokens 28680 \
    --moe-a2a-backend deepep --deepep-mode normal \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --dp-size 2 --enable-dp-attention \
    --disable-shared-experts-fusion \
    --dtype bfloat16
```

**Decode 节点（D 节点）：**

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export ASCEND_MF_STORE_URL="tcp://<PREFILL_HOST_IP>:24669"
export HCCL_BUFFSIZE=720
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=88
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1
unset TASK_QUEUE_ENABLE
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

python -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --disaggregation-mode decode \
    --host $DECODE_HOST_IP \
    --port 8001 \
    --trust-remote-code \
    --nnodes 1 --node-rank 0 \
    --tp-size 16 --dp-size 16 \
    --mem-fraction-static 0.8 \
    --max-running-requests 352 \
    --attention-backend ascend --device npu \
    --quantization modelslim \
    --moe-a2a-backend deepep \
    --enable-dp-attention \
    --deepep-mode low_latency \
    --enable-dp-lm-head \
    --cuda-graph-bs 8 10 12 14 16 18 20 22 \
    --disaggregation-transfer-backend ascend \
    --watchdog-timeout 9000 \
    --context-length 8192 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --disable-shared-experts-fusion \
    --dtype bfloat16 \
    --tokenizer-worker-num 4
```

**Router（PD 路由）：**

```bash
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://<PREFILL_HOST_IP>:8000 8996 \
    --decode http://<DECODE_HOST_IP>:8001 \
    --host 127.0.0.1 \
    --port 6688
```

### 5.3 GLM-5.1（单机 A3，W4A8，16卡）

模型权重：`Eco-Tech/GLM-5-w4a8`（ModelScope）  
GLM-5 采用 DeepSeek-V3/V3.2 架构（DSA + MTP），NPU 0Day 支持。

```bash
# 系统优化
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0; sysctl -w kernel.numa_balancing=0
export SGLANG_SET_CPU_AFFINITY=1
unset ASCEND_LAUNCH_BLOCKING

# CANN
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 优化参数
export STREAMS_PER_DEVICE=32
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_NPU_USE_MULTI_STREAM=1
export HCCL_BUFFSIZE=1000
export HCCL_OP_EXPANSION_MODE=AIV
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1

python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --attention-backend ascend \
    --device npu \
    --tp-size 16 --nnodes 1 --node-rank 0 \
    --chunked-prefill-size 16384 \
    --max-prefill-tokens 280000 \
    --trust-remote-code \
    --host 127.0.0.1 \
    --mem-fraction-static 0.7 \
    --port 8000 \
    --served-model-name glm-5 \
    --cuda-graph-bs 16 \
    --quantization modelslim \
    --moe-a2a-backend deepep \
    --deepep-mode auto
```

> Docker 镜像（GLM-5 专版）：
> ```bash
> # A3
> docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann8.5.0-a3-glm5
> # A2
> docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann8.5.0-910b-glm5
> ```
> 注意：需更新 transformers 至 5.3.0：`pip install transformers==5.3.0`

### 5.4 GLM-5.1（双机 A3，BF16，32卡多节点）

```bash
# 两台节点上运行相同脚本，脚本自动判断本机 IP

P_IP=('node0_ip' 'node1_ip')
P_MASTER="${P_IP[0]}:5000"

LOCAL_HOST1=`hostname -I | awk '{print $1}'`
LOCAL_HOST2=`hostname -I | awk '{print $2}'`

for i in "${!P_IP[@]}"; do
    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]]; then
        python3 -m sglang.launch_server \
            --model-path $MODEL_PATH \
            --attention-backend ascend --device npu \
            --tp-size 32 --nnodes 2 --node-rank $i --dist-init-addr $P_MASTER \
            --chunked-prefill-size 16384 --max-prefill-tokens 131072 \
            --trust-remote-code \
            --host 127.0.0.1 \
            --mem-fraction-static 0.8 \
            --port 8000 \
            --cuda-graph-max-bs 32 \
            --moe-a2a-backend deepep \
            --deepep-mode auto \
            --disable-radix-cache
        break
    fi
done
```

### 5.5 Qwen3-235B-A22B（A3 8卡，混合 PD）

低延迟目标：11K+1K，TPOT 10ms

```bash
# 详见 best_practice 中 Qwen3-235B 配置
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-235B-A22B \
    --attention-backend ascend --device npu \
    --tp-size 8 \
    --mem-fraction-static 0.8 \
    --quantization modelslim \
    --moe-a2a-backend deepep --deepep-mode auto \
    --enable-dp-attention --dp-size 1 \
    --cuda-graph-bs 4 8 12 16 \
    --chunked-prefill-size 16384 \
    --max-prefill-tokens 65536 \
    --dtype bfloat16
```

---

## 6. 推理性能数据（官方 Best Practice）

### 6.1 DeepSeek 系列 — 昇腾 A3（Atlas 800I A3）

#### 低延迟场景（TPOT 目标）

| 模型 | 硬件 | 卡数 | 部署模式 | 输入+输出 | TPOT | 量化 |
|------|------|------|----------|-----------|------|------|
| DeepSeek-R1 | A3 | 32 | PD 分离 | 6K+1.6K | **20ms** | W8A8 INT8 |
| DeepSeek-R1 | A3 | 32 | PD 分离 | 3.9K+1K | **19ms** | W8A8 INT8 |
| DeepSeek-R1 | A3 | 32 | PD 分离 | 3.5K+1.5K | **19ms** | W8A8 INT8 |
| DeepSeek-R1 | A3 | 32 | PD 分离 | 3.5K+1K | **19ms** | W8A8 INT8 |
| DeepSeek-V3.2 | A3 | 32 | PD 分离 | 128K+1K | **26ms** | W8A8 INT8 |

#### 高吞吐场景（TPOT 50ms）

| 模型 | 硬件 | 卡数 | 部署模式 | 输入+输出 | TPOT | 量化 |
|------|------|------|----------|-----------|------|------|
| DeepSeek-R1 | A3 | 32 | PD 分离 | 3.5K+1.5K | 50ms | W8A8 INT8 |
| DeepSeek-R1 | A3 | 24 | PD 分离 | 2K+2K | 50ms | W8A8 INT8 |
| DeepSeek-R1 | A3 | 16 | PD 分离 | 2K+2K | 50ms | W4A8 INT8 |
| DeepSeek-R1 | A3 | 16 | PD 分离 | 3.5K+1.5K | 50ms | W4A8 INT8 |
| DeepSeek-R1 | A3 | **8** | PD 混合 | 2K+2K | 50ms | W4A8 INT8 |
| DeepSeek-R1 | A3 | **8** | PD 混合 | 3.5K+1.5K | 50ms | W4A8 INT8 |

### 6.2 Qwen 系列

#### 低延迟场景

| 模型 | 硬件 | 卡数 | 输入+输出 | TPOT | 量化 |
|------|------|------|-----------|------|------|
| Qwen3-235B-A22B | A3 | 8 | 11K+1K | **10ms** | BF16 |
| Qwen3-32B | A3 | 4 | 4K+1.5K | **11ms** | BF16 |
| Qwen3-32B | A3 | 4 | 6K+1.5K | 18ms | BF16 |
| Qwen3-32B | A3 | 8 | 18K+4K | **6ms** | BF16 |
| Qwen3-32B | A3 | 2 | 1K+0.3K | 12ms | W8A8 |
| Qwen3-8B | A3 | 1 | 3.5K+1.5K | **5ms** | W8A8 |
| Qwen3-8B | A3 | 1 | 1K+0.3K | **7ms** | W8A8 |
| Qwen3-30B-A3B | A3 | 1 | 6K+1.5K | **10ms** | W8A8 |
| Qwen3-14B | A3 | 1 | 3.5K+1.5K | **9ms** | W8A8 |
| Qwen3.5-397B-A17B | A3 | 8 | 3.5K+1.5K | 22ms | W4A8 |

#### 高吞吐场景（TPOT 50ms）

| 模型 | 硬件 | 卡数 | 部署模式 | 输入+输出 | TPOT |
|------|------|------|----------|-----------|------|
| Qwen3-235B-A22B | A3 | 24 | PD 分离 | 3.5K+1.5K | 50ms |
| Qwen3-235B-A22B | A3 | 8 | PD 混合 | 3.5K+1.5K | 50ms |
| Qwen3-Coder-480B | A3 | 24 | PD 分离 | 3.5K+1.5K | 50ms |
| Qwen3-Coder-480B | A3 | 16 | PD 混合 | 3.5K+1.5K | 50ms |
| Qwen3-Coder-480B | A3 | 8 | PD 混合 | 3.5K+1.5K | 50ms |
| Qwen3-32B | A2 | 8 | PD 混合 | 3.5K+1.5K | 50ms |

### 6.3 DeepEP-Ascend MoE 通信性能（A3 384 SuperPOD）

测试条件：4096 tokens/batch，7168 hidden size，top-8 experts，INT8 dispatch + BF16 combine

#### Normal Mode（高吞吐）

| EP 规模 | Dispatch 带宽 | Combine 带宽 |
|---------|--------------|-------------|
| 8-way intra | 146 GB/s | 125 GB/s |
| 16-way intra | 107 GB/s | 103 GB/s |
| 32-way intra | 102 GB/s | 95 GB/s |
| 64-way intra | 81 GB/s | 91 GB/s |
| 128-way intra | 57 GB/s | 81 GB/s |

#### Low-Latency Mode（128 tokens/batch，生产推理 decode 阶段）

| EP 规模 | Dispatch 延迟 | Dispatch 带宽 | Combine 延迟 | Combine 带宽 |
|---------|--------------|--------------|-------------|-------------|
| 8-way | 132 us | 58 GB/s | 126 us | 116 GB/s |
| 16-way | 139 us | 55 GB/s | 135 us | 109 GB/s |
| 32-way | 153 us | 49 GB/s | 151 us | 97 GB/s |

> sub-150us 延迟已满足大多数生产在线推理 SLA

---

## 7. 支持的模型列表（部分）

### LLM

| 模型 | A2 | A3 |
|------|:---:|:---:|
| DeepSeek-V3/V3.1/V3.2-W8A8 | ✓ | ✓ |
| DeepSeek-R1-0528-W8A8 | ✓ | ✓ |
| Qwen3-235B-A22B-W8A8 | ✓ | ✓ |
| Qwen3-Coder-480B-A35B-W8A8 | ✓ | ✓ |
| Qwen3-32B / Qwen3-8B | ✓ | ✓ |
| Qwen3.5-397B-A17B | ✓ | ✓ |
| Kimi-K2-Thinking | ✓ | ✓ |
| GLM-5 / GLM-5-w4a8 | ✓ | ✓ |
| GLM-4.5 | ✓ | ✓ |
| ERNIE-4.5 系列 | ✓ | ✓ |
| Llama-4-Scout-17B-16E | ✓ | ✓ |
| Qwen2.5-7B-Instruct | ✓ | ✓ |

### VLM（多模态）

| 模型 | A2 | A3 |
|------|:---:|:---:|
| Qwen2.5-VL-72B-Instruct | ✓ | ✓ |
| Qwen3-VL-235B-A22B | ✓ | ✓ |
| GLM-4.5V-106B | ✓ | ✓ |
| DeepSeek-VL2 | ✓ | ✓ |
| Kimi-VL-A3B | ✓ | ✓ |

---

## 8. 常见问题

| 问题 | 解决方案 |
|------|----------|
| `import sgl_kernel_npu` 失败 | 检查 `libsgl_kernel_npu.so` 是否在 whl 的 lib/ 目录，重新 pip install |
| CANN 环境变量未生效 | 每次启动前执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| DeepEP 编译失败 | A2 用 `-a deepep2`，A3 用 `-a deepep`，SOC 版本不能混用 |
| NPUGraph capture 慢 | 正常现象，首次 warmup 会编译 graph；后续请求走缓存路径 |
| MoE 推理显存不足 | 调低 `--mem-fraction-static`（0.6~0.7），关闭 radix cache |
| 多节点 HCCL 初始化超时 | 设置正确的 `HCCL_SOCKET_IFNAME`（ifconfig 查看实际网卡名） |
| GLM-5 推理报错 | 更新 transformers 至 5.3.0，使用 GLM-5 专版 docker 镜像 |

---

## 9. 参考来源

- `sglang/docs/platforms/ascend/ascend_npu_quick_start.md`
- `sglang/docs/platforms/ascend/ascend_npu_best_practice.md`
- `sglang/docs/platforms/ascend/ascend_npu_deepseek_example.md`
- `sglang/docs/platforms/ascend/ascend_npu_glm5_examples.md`
- `sglang/docs/platforms/ascend/ascend_npu_support_models.md`
- `sgl-kernel-npu/README.md`
- `sgl-kernel-npu/build.sh`
- `sgl-kernel-npu/python/sgl_kernel_npu/README.md`
- `sgl-kernel-npu/python/deep_ep/README.md`
