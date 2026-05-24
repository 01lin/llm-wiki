# Qwen3.5 MTP 在 vLLM / vLLM-Ascend 上的昇腾适配分析报告
更新时间：2026-03-30  
重点范围：`vllm`、`vllm-ascend`、昇腾 `910B3/A2` 为主，兼顾 `910C`  
重点模型：`Qwen3.5-397B-A17B`，补充 `Qwen3.5-122B-A10B`、`Qwen3.5-35B-A3B`

## 1. 执行摘要
`Qwen3.5 MTP` 在 `vllm` / `vllm-ascend` 上已经完成了基础功能接入，但在昇腾 `910B3/A2` 上，当前主要矛盾已经从“是否支持”转变为“是否稳定、正确、持续获得 TPOT 正收益”。

从仓库分工看，`vllm` 负责上游 speculative decoding 框架、`qwen3_5_mtp` 模型类型接入和后续算法演进；`vllm-ascend` 则承担昇腾侧 GDN/Mamba 路径、图模式、MoE 通信、PD 分离和量化 fast path 的实际落地。因此，针对 `Qwen3.5-397B-A17B` 的性能优化和问题收敛，主战场明确在 `vllm-ascend`。

截至 2026 年 3 月底，公开信息显示，`Qwen3.5-397B-A17B-w8a8-mtp` 在 `910B3/A2` 上仍存在以下几类关键问题：
- `mtp + graph` 无法稳定拉起服务
- `PD 分离 + 高并发` 下服务卡死、崩溃、超时
- `EP/MoE dispatch` 在 A2 平台存在 expert/card 数量约束
- `FULL_DECODE_ONLY`、`hybrid KV cache`、`block_size`、`prefix cache` 等机制之间还存在显著耦合
- 开启 MTP 后并不保证端到端 TPOT 必然改善，某些场景甚至出现明显性能劣化

已合入工作中，最重要的性能优化是 `vllm-ascend` PR [#7487](https://github.com/vllm-project/vllm-ascend/pull/7487)，它改善的是 GDN prefill 路径的 host/device 同步开销，在 prefill-heavy、高 batch 场景下带来更好的吞吐和 TTFT。但这不等价于 `397B-A17B` 在线服务场景下 MTP 已经稳定正收益。相反，多条 issue 表明，目前 `Qwen3.5-397B-A17B` 的主要瓶颈仍然是系统层和后端层，而不是纯算法层。

对下一步方向的建议是：短期优先围绕 `vllm-ascend` 现有 `qwen3_5_mtp` 路径做收敛，重点投入 GDN/Mamba 路径、图模式、A2 平台 MoE dispatch/flash comm、PD 分离稳定性与 hybrid KV cache 机制；中期再跟进 `vllm` 上游的动态 speculation length 等算法优化。

---

## 2. 背景与仓库定位

### 2.1 `vllm` 的定位
`vllm` 已经在上游框架中接入 `qwen3_5_mtp`，相关入口在 [speculative.py](/Users/linyi/Documents/code/vllm/vllm/config/speculative.py)。这说明：
- `Qwen3.5 MTP` 已被纳入通用 speculative decoding 框架
- 上游后续对 speculative 调度、动态长度、page mapping、prefix cache 的改进，会直接影响 Ascend 落地效果

同时，`FULL_DECODE_ONLY` 等编译相关能力位于 [compilation.py](/Users/linyi/Documents/code/vllm/vllm/config/compilation.py)，属于 `Qwen3.5 MTP` 在图模式下的重要基础能力。

### 2.2 `vllm-ascend` 的定位
`vllm-ascend` 是昇腾适配主仓。`Qwen3.5` 相关官方文档已经直接给出推荐配置：
- [Qwen3.5-27B.md](/Users/linyi/Documents/code/vllm-ascend/docs/source/tutorials/models/Qwen3.5-27B.md)
- [Qwen3.5-397B-A17B.md](/Users/linyi/Documents/code/vllm-ascend/docs/source/tutorials/models/Qwen3.5-397B-A17B.md)

文档中明确推荐：
- `qwen3_5_mtp`
- `FULL_DECODE_ONLY`
- `async_scheduling`
- 某些场景下配合 `multistream_overlap_shared_expert`

同时文档也明确提示：
- hybrid KV cache 会拉大 `block_size`
- `block_size` 过大可能降低 prefix cache hit rate
- 这会直接影响长上下文场景和端到端 TPOT

Ascend 侧的模型专项 patch 在 [patch_qwen3_5.py](/Users/linyi/Documents/code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py)，代码中保留了自定义 `_forward_core`，原因是当前 `torch_npu` 的 recurrent gated delta rule 路径对所需状态类型支持仍不完整。这是当前很多性能、稳定性问题的底层背景。

另一个关键实现点在 [utils.py](/Users/linyi/Documents/code/vllm-ascend/vllm_ascend/utils.py)：Ascend 上 `mtp` 的 decode fast path 目前只在 `mtp_quantize == "w8a8_dynamic"` 时打开。这意味着 MTP 加速收益和量化路径强绑定。

---

## 3. 现状判断

### 3.1 已具备的能力
当前已经可以明确认为：
- `Qwen3.5 MTP` 在 `vllm` 上游框架中已正式接入
- `vllm-ascend` 已经支持 `Qwen3.5-27B`、`Qwen3.5-397B-A17B` 的 MTP 运行路径
- `vllm-ascend` 已经开始针对 GDN/Mamba、flash comm、PD 分离、graph 模式进行专项修复和优化
- `Qwen3.5-397B-A17B-w8a8-mtp` 已经被纳入 nightly acc/perf 持续验证方向，见 PR [#7745](https://github.com/vllm-project/vllm-ascend/pull/7745)

### 3.2 未收敛的核心问题
当前尚不能认为 `Qwen3.5-397B-A17B` 在 `910B3/A2` 上已经达到生产级 MTP 加速状态。主要原因包括：
- 图模式兼容性未收敛
- 高并发 PD 分离场景下稳定性差
- A2 平台 MoE dispatch 存在硬件/算子级约束
- 某些业务场景下开 MTP 性能反降
- 采样参数、`min_tokens`、cache 机制对结果影响较大，表明调度策略还未稳定

### 3.3 910B3 与 910C 的情况
截至 2026-03-30，公开检索到的 `Qwen3.5 MTP` 问题、PR 和 roadmap 信息明显以 `910B3/A2` 为主。  
`910C` 尚未看到一条独立、成体系的专项验证线，因此当前更合理的策略是：
1. 先以 `910B3/A2` 建立收敛基线
2. 再横向验证 `910C` 是否复现相同问题或具备更优表现

---

## 4. PR / Issue / Roadmap 详细附录

## 4.1 功能问题
| 仓库          | 类型  | 编号                                                         | 标题                                                         | 设备/模型                    | 状态                    | 发起时间     | 结束时间     | 主要影响                            | 备注                                                         |
| ------------- | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------- | ----------------------- | ------------ | ------------ | ----------------------------------- | ------------------------------------------------------------ |
| `vllm-ascend` | Issue | [#7598](https://github.com/vllm-project/vllm-ascend/issues/7598) | `qwen3.5 mtp+graph cannot start vllm service`                | `Qwen3.5-397B-A17B-w8a8-mtp` | open                    | 未直出       | -            | 图模式启动失败                      | `FULL_DECODE_ONLY` 下 `KeyError: 60`                         |
| `vllm-ascend` | Issue | [#7532](https://github.com/vllm-project/vllm-ascend/issues/7532) | `qwen3.5 开启 MTP 偶发崩溃`                                  | `Qwen3.5-27B`                | open                    | 未直出       | -            | 服务稳定性                          | 大规模 pod 压测下暴露                                        |
| `vllm-ascend` | Issue | [#7061](https://github.com/vllm-project/vllm-ascend/issues/7061) | `think 标签无法正常解析`                                     | `Qwen3.5-397B-A17B-w8a8-mtp` | issue，已有上游修复路径 | 未直出       | 未直出       | reasoning 结果异常                  | 依赖上游 PR [#34779](https://github.com/vllm-project/vllm/pull/34779) |
| `vllm`        | PR    | [#34779](https://github.com/vllm-project/vllm/pull/34779)    | `Fix Qwen3/Qwen3.5 Reasoning Parser`                         | `Qwen3.5`                    | merged                  | `2026-02-18` | `2026-02-22` | 修 reasoning parser                 | 解决 `think` 标签与 streaming 解析                           |
| `vllm`        | Issue | [#38106](https://github.com/vllm-project/vllm/issues/38106)  | `tool_choice=required + speculative decoding` 导致 failed tool calls | `Qwen3.5-397B-A17B`          | open                    | 未直出       | -            | structured output/tool-calling 错误 | speculative 输出 XML 而不是 JSON                             |

### 小结
功能问题里，`397B-A17B` 最重要的 blocker 是 `mtp + graph` 无法正常启动。  
这说明在 Ascend 上，图模式还不是 `Qwen3.5 MTP` 的稳定加速手段，而是当前主要风险源之一。

---

## 4.2 精度问题
| 仓库          | 类型  | 编号                                                         | 标题                                                         | 设备/模型                     | 状态   | 发起时间     | 结束时间     | 主要影响                                       | 备注                                                         |
| ------------- | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------- | ------ | ------------ | ------------ | ---------------------------------------------- | ------------------------------------------------------------ |
| `vllm-ascend` | PR    | [#7364](https://github.com/vllm-project/vllm-ascend/pull/7364) | `A2 MOE method && layerwise MTP bugfix && Mamba gdn_metadata bugfix` | `A2/910B3`, `Qwen3.5-35B-A3B` | merged | `2026-03-17` | `2026-03-17` | PD + MTP 精度修复                              | 修 `num_decode_draft_tokens` / `spec_tokens_padding` / `gdn_metadata` |
| `vllm-ascend` | PR    | [#7506](https://github.com/vllm-project/vllm-ascend/pull/7506) | `fix padding error on FullGraph mode && fix layerwise connector mamba accuracy` | `Qwen3.5`                     | merged | `2026-03-20` | `2026-03-24` | FullGraph 下 NaN/状态污染                      | review 中明确仍是 workaround                                 |
| `vllm-ascend` | Issue | [#7604](https://github.com/vllm-project/vllm-ascend/issues/7604) | `Qwen3.5-397B Prefill Decode Disaggregation fail...`         | `397B-A17B-w8a8-mtp`          | open   | 未直出       | -            | 非零 temperature 或默认 temperature 时请求失败 | 被列入 release blocker                                       |
| `vllm-ascend` | Issue | [#7517](https://github.com/vllm-project/vllm-ascend/issues/7517) | `Qwen3.5-122B-A10B 过度思考`                                 | `122B`                        | open   | 未直出       | -            | 输出质量/可用性                                | 122B 需要独立质量矩阵                                        |
| `vllm`        | PR    | [#35777](https://github.com/vllm-project/vllm/pull/35777)    | `fused_sigmoid_gating_delta_rule_update kernel`              | 上游 `Qwen3.5`                | merged | `2026-03-02` | `2026-03-09` | 精度基本不变                                   | PR 内 GSM8K 显示 `Qwen3.5 MTP` 精度未明显下降                |

### 小结
精度问题的关键不是“模型天然不准”，而是：
- `PD + MTP` 调度元数据处理
- `FullGraph` 下 padding/状态写入逻辑
- 采样路径与 speculative/PD 联动
- 122B 在 reasoning 场景下的输出行为

这类问题说明当前 Ascend 的系统实现仍可能引入行为回归。

---

## 4.3 性能优化
| 仓库          | 类型  | 编号                                                         | 标题                                                         | 设备/模型                | 状态       | 发起时间     | 结束时间     | 主要影响                               | 备注                               |
| ------------- | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------ | ---------- | ------------ | ------------ | -------------------------------------- | ---------------------------------- |
| `vllm-ascend` | PR    | [#7487](https://github.com/vllm-project/vllm-ascend/pull/7487) | `Optimize Qwen3.5/Qwen3Next GDN prefill by prebuilding chunk metadata` | `Qwen3.5-0.8B/35B/397B`  | merged     | `2026-03-20` | `2026-03-22` | 提升 prefill-heavy/high-BS 吞吐与 TTFT | 当前最重要已合入性能 PR            |
| `vllm-ascend` | PR    | [#7540](https://github.com/vllm-project/vllm-ascend/pull/7540) | `Cache GDN chunk metadata and mask qwen states`              | `Qwen3.5`                | open       | `2026-03-23` | -            | 继续压缩 metadata 准备开销             | review 指出线程安全问题            |
| `vllm-ascend` | Issue | [#7231](https://github.com/vllm-project/vllm-ascend/issues/7231) | `Qwen3.5-397B 开启 MTP 多模态测试性能劣化`                   | `397B-A17B`              | open       | 未直出       | -            | 开 MTP 性能反降                        | 1/2/3 speculative tokens 都劣化    |
| `vllm-ascend` | Issue | [#7653](https://github.com/vllm-project/vllm-ascend/issues/7653) | `无 min_tokens 的测试性能高于设置 min_tokens`                | `397B-A17B-w8a8-mtp`     | open       | 未直出       | -            | 吞吐差可达 2 倍                        | 调度/采样参数强影响性能            |
| `vllm`        | PR    | [#35301](https://github.com/vllm-project/vllm/pull/35301)    | `dynamic speculation length with confidence-threshold early exit` | 上游算法                 | open draft | `2026-02-25` | -            | 算法层潜在大收益                       | 当前实现仍有 GPU->CPU sync 开销    |
| `vllm`        | Issue | [#36498](https://github.com/vllm-project/vllm/issues/36498)  | `acceptance 高但 decode 仍慢于不开 MTP`                      | `35B/122B`               | open       | 未直出       | -            | acceptance 不等于端到端收益            | 还伴随 0.17.0 高并发非法访问       |
| `vllm`        | PR    | [#35703](https://github.com/vllm-project/vllm/pull/35703)    | `Map multiple FullAttn layers to a single page`              | `Qwen3.5-397B` 类 hybrid | open       | `2026-03-02` | -            | 降低 block_size，改善 prefix cache hit | 对 `397B` 长上下文很关键           |
| `vllm`        | Issue | [#38182](https://github.com/vllm-project/vllm/issues/38182)  | `MTP 与 prefix cache hit rate 的关系`                        | `Qwen3.5-35B-A3B`        | open       | 未直出       | -            | cache 指标解释复杂                     | 暴露 cache 与 speculative 交互问题 |

### 小结
性能相关的关键信号是：
- 已合入优化主要改善 prefill 和系统气泡，不代表 decode/TPOT 必然全面优化
- 对 `397B-A17B`，MTP 仍存在明显场景依赖性
- 上游动态 speculation length 很值得关注，但还未与 Ascend 特定瓶颈结合验证
- `hybrid KV cache + block_size + prefix cache` 很可能是长上下文场景下的核心性能因子

---

## 4.4 算子与通信
| 仓库          | 类型  | 编号                                                         | 标题                                                      | 设备/模型                       | 状态   | 发起时间     | 结束时间     | 主要影响                         | 备注                                        |
| ------------- | ----- | ------------------------------------------------------------ | --------------------------------------------------------- | ------------------------------- | ------ | ------------ | ------------ | -------------------------------- | ------------------------------------------- |
| `vllm-ascend` | PR    | [#7486](https://github.com/vllm-project/vllm-ascend/pull/7486) | `Qwen3.5 MoE supports flash comm`                         | `Qwen3.5 MoE`                   | merged | `2026-03-19` | `2026-03-25` | MoE 通信优化                     | 多模态 first all-gather skip                |
| `vllm-ascend` | PR    | [#7644](https://github.com/vllm-project/vllm-ascend/pull/7644) | `Qwen3.5 MoE supports flashcomm v1`                       | release branch                  | merged | `2026-03-25` | `2026-03-25` | 发布分支同步                     | 说明 flashcomm 被视作必要能力               |
| `vllm-ascend` | PR    | [#7683](https://github.com/vllm-project/vllm-ascend/pull/7683) | `Fix flash comm v1 bugs of Qwen3.5 series on A2 platform` | `A2/910B3`                      | open   | `2026-03-26` | -            | A2 专项 bugfix                   | 说明 flashcomm 在 A2 仍未完全稳定           |
| `vllm-ascend` | PR    | [#7396](https://github.com/vllm-project/vllm-ascend/pull/7396) | `chunk gated delta rule`                                  | GDN/Mamba                       | open   | `2026-03-17` | -            | 底层 GDN 路径优化                | 价值高但还未合入                            |
| `vllm-ascend` | Issue | [#7360](https://github.com/vllm-project/vllm-ascend/issues/7360) | `Qwen3.5-27B A2 单机 8 卡出现 aclnnCausalConv1d 错误`     | `Atlas 800T A2 / 910B3`         | open   | 未直出       | -            | 关键算子兼容性风险               | 明确是 `910B3` 场景                         |
| `vllm-ascend` | Issue | [#7553](https://github.com/vllm-project/vllm-ascend/issues/7553) | `910B3 dual-machine...MoeDistributeDispatchV4 failed`     | `910B3`, `397B-A17B-w8a8-mtp`   | open   | 未直出       | -            | MoE dispatch 限制导致启动失败    | `moeExpertNum 32 > 24`                      |
| `vllm-ascend` | Issue | [#7635](https://github.com/vllm-project/vllm-ascend/issues/7635) | `A2 双机 qwen3.5 开 EP 拉起失败`                          | `A2`, `397B-A17B-w8a8`          | open   | 未直出       | -            | EP/MoE dispatch 失败             | 与 `#7553` 同类瓶颈                         |
| `vllm-ascend` | Issue | [#7651](https://github.com/vllm-project/vllm-ascend/issues/7651) | `PD 分离部署高并发卡住或崩溃`                             | `A2 910B`, `397B-A17B-w8a8-mtp` | open   | 未直出       | -            | 高并发 PD 稳定性与算子/HCCL 超时 | `aclnnSigmoid`、HCCL、FFTS timeout 同时出现 |
| `vllm`        | PR    | [#35777](https://github.com/vllm-project/vllm/pull/35777)    | `fused gating delta kernel`                               | 上游 kernel                     | merged | `2026-03-02` | `2026-03-09` | 上游 kernel 融合演进             | 对 Ascend 有方法学参考价值                  |

### 小结
算子与通信是当前 `910B3/A2` 上最现实、最硬的瓶颈：
- `aclnnCausalConv1d`
- `aclnnSigmoid`
- `MoeDistributeDispatchV4`
- HCCL/FFTS timeout

这表明对 `397B-A17B` 的优化，不能只看算法 acceptance，而必须结合算子稳定性和通信约束一起评估。

---

## 4.5 Roadmap / 发布收敛
| 仓库          | 类型              | 编号                                                         | 标题                                        | 状态    | 发起时间            | 结束时间 | 主要内容                                                     | 结论                               |
| ------------- | ----------------- | ------------------------------------------------------------ | ------------------------------------------- | ------- | ------------------- | -------- | ------------------------------------------------------------ | ---------------------------------- |
| `vllm-ascend` | Roadmap           | [#5318](https://github.com/vllm-project/vllm-ascend/issues/5318) | `vLLM Ascend Roadmap Q1 2026`               | roadmap | Q1 2026             | -        | `aclgraph Full mode`、KV connector、KV cache quant、Eagle3、PCP/DCP | 与 `Qwen3.5 MTP` 依赖高度一致      |
| `vllm-ascend` | Release checklist | [#7634](https://github.com/vllm-project/vllm-ascend/issues/7634) | `Release checklist for v0.18.0rc1`          | open    | 发布日 `2026-03-27` | -        | `#7604/#7598/#7595` 仍列为未解决 bug                         | `Qwen3.5` 线尚未完全 release-ready |
| `vllm-ascend` | PR                | [#7745](https://github.com/vllm-project/vllm-ascend/pull/7745) | `add nightly ... Qwen3.5-397B-w8a8-mtp`     | open    | `2026-03-27`        | -        | nightly 增加 `Qwen3.5-397B-w8a8-mtp` acc/perf                | 持续验证体系还在建设               |
| `vllm`        | Feature issue     | [#36037](https://github.com/vllm-project/vllm/issues/36037)  | `Supports Speculative Speculative Decoding` | open    | 未直出              | -        | SSD 路线                                                     | 更远期 speculative 方向            |

### 小结
从 roadmap 和 release checklist 看，`Qwen3.5 MTP` 仍处于“持续补洞和加固”阶段，而不是“已经完全稳定的默认加速路径”。

---

## 5. 重点结论：面向 Qwen3.5-397B-A17B

### 5.1 当前最关键的已落地成果
当前最值得明确记住的一条已合入性能成果是：
- `vllm-ascend` PR [#7487](https://github.com/vllm-project/vllm-ascend/pull/7487)

它的意义是：
- GDN prefill 路径的 metadata 准备从热路径中挪出
- 减少 host/device 同步
- 提升 prefill-heavy、高 batch 场景吞吐
- 降低 TTFT
- acceptance 基本保持稳定

但它更像是“让系统不被无谓同步拖垮”，而不是“已经解决 `397B` 的所有 MTP 问题”。

### 5.2 当前最重要的风险簇
对 `Qwen3.5-397B-A17B-w8a8-mtp`，当前最重要的风险有三簇：

第一簇是图模式相关：
- `mtp + graph` 启动失败
- `FullGraph` 下 padding/状态处理存在历史问题
- `FULL_DECODE_ONLY` 在高并发 PD 场景下会触发崩溃

第二簇是 A2/910B3 上的 MoE/EP 通信约束：
- `MoeDistributeDispatchV4` 存在 experts/card 限制
- EP 打开后容易在 A2 双机场景失败
- flashcomm 虽已合入，但 A2 专项 bugfix 仍在 open

第三簇是 PD 分离与服务稳定性：
- 非零 temperature 时请求失败
- 高并发下 HCCL/FFTS timeout
- `min_tokens`、cache、调度参数会显著改变结果

### 5.3 当前最值得投入的优化方向
短期建议的优化优先级：
1. GDN/Mamba 路径与相关算子
2. `FULL_DECODE_ONLY` / graph 模式兼容
3. A2 平台 MoE dispatch 与 flash comm
4. hybrid KV cache / block size / prefix cache
5. PD 分离下的高并发稳定性

中期建议跟进的上游算法方向：
- [vllm PR #35301](https://github.com/vllm-project/vllm/pull/35301) 的动态 speculation length

---

## 6. 结论
截至 2026 年 3 月 30 日，`Qwen3.5 MTP` 在 `vllm` / `vllm-ascend` 上已经具备可运行基础，且 `vllm-ascend` 已开始围绕 GDN、MoE 通信、graph 模式和 PD 分离进行系统性优化。但对昇腾 `910B3/A2` 而言，`Qwen3.5-397B-A17B-w8a8-mtp` 还没有达到“稳定、可预测、持续 TPOT 正收益”的成熟状态。

现阶段更准确的判断是：
- 功能层：已打通
- 工程层：正在加固
- 性能层：已有局部正收益，但端到端收益仍强依赖场景和配置
- 稳定性层：仍有明显 blocker
- 算法层：上游仍在继续演进，尚未完全与 Ascend 系统瓶颈结合

因此，如果要在昇腾上推进 `Qwen3.5-397B-A17B` 的 MTP 生产落地，最现实的路径不是盲目扩大 speculative token 或直接依赖 graph，而是围绕 `vllm-ascend` 现有 `qwen3_5_mtp` 路径，优先收敛 GDN/Mamba、A2 平台 MoE dispatch、PD 分离和 hybrid cache 问题，再叠加上游动态 speculation length 等算法收益。