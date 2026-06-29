# Agentic_Inference

「Agentic 推理引擎」深度优化专题的工作目录。本主题下后续所有分析输出、方案设计、采集脚本、画像数据等中间产物，统一放置于此。

## 目录定位

- 这是 **raw 工作区**（原始素材 + 中间产物），不是已编译的 wiki。
- 沉淀稳定后的结论会被 INGEST 到 `wiki/`（concepts / syntheses / sources）。
- 已编译入库的对应页：
  - `wiki/concepts/20260628-150300-agentic-inference-engine-cooptimization-概念.md`
  - `wiki/concepts/20260628-150400-dspark-semi-autoregressive-spec-概念.md`
  - `wiki/syntheses/20260629-100000-agentic-workload-analysis-总体方案设计-综合.md`

## 默认归档约定（用户指定）

**本主题后续所有分析输出件，默认放置于本目录 `raw/Infra/Agentic_Inference/`。**
- 分析脚本、报告、方案设计 → 本目录根
- 原始输入数据（数据集拉取的样例）→ `raw-data/` 子目录；大文件先瘦身到分析所需字段再归档（如 gaia2 从 77M 瘦身为 events-only 183K），避免大二进制进 git。

## 已沉淀产物索引

| 文件 | 类型 | 说明 |
|------|------|------|
| `20260629-100000-agentic-workload-analysis-总体方案设计-综合.md` | 方案 | workload 总体方案（正本在 wiki/syntheses，此为副本）|
| `20260629-004700-swe-trajectory-workload-画像分析-脚本.py` | 脚本 | SWE 单 agent 画像（R/ρ/段结构）|
| `20260629-005000-swe-trajectory-样例实例化分析-报告.md` | 报告 | SWE 实例化分析（fan-out=1，关系维空）|
| `20260629-010000-gaia2-multiagent-dag-画像分析-脚本.py` | 脚本 | gaia2 multi-agent DAG 还原画像 |
| `20260629-011000-gaia2-multiagent-dag-验证分析-报告.md` | 报告 | DAG 验证（fan-out 5.3 / join 67%）|
| `20260629-013000-workload-可视化生成-脚本.py` | 脚本 | 纯标准库 SVG 可视化生成（→ figures/）|
| `20260629-014000-指标体系实例化映射-脚本.py` | 脚本 | 20 指标×两数据集实例化映射（9✅/4△/7✗）|
| `20260629-015000-指标体系与推理优化与轻量采集-深化报告.md` | 报告 | 三问深化：指标映射+补缺 / 优化推导 / 轻量采集（18/20 复用，引擎改~20行）|
| `20260629-020000-指标联合驱动推理优化-代码级深度推导.md` | 报告 | **5 条指标联合优化的代码级深度推导**：每条=指标联合→vllm-ascend真实代码决策点→case收益→可行性/风险，含 num_spec_tokens 静态 shape 约束等否定性发现 |
| `20260629-021500-价值逐出特性-可施工级方案设计.md` | 报告 | **单特性深挖到可施工**：价值逐出四问（指标含义/方案+时序+代码步骤/case收益/复杂度可行性泛化性），~10行落2自有文件不patch上游，所有行号 grep 实测 |
| `20260629-023000-按优化领域展开-agentic指标驱动优化全景.md` | 报告 | **按领域组织的优化全景**：KV管理(pin/换出/段压缩)+动态投机(段DSL)+调度(价值逐出/组完成)，每特性=采集指标→优化思路→怎么用指标→执行时序步骤+代码依据；3 个跨领域统一抓手 |
| `20260629-024500-agentic推理优化切入点全景盘点-收益代入.md` | 报告 | **infra 专家视角穷尽 16 个切入点**（请求全链路×资源栈两轴扫描），从 3 领域扩到计算/显存/通信/调度/存储/编解码；case 数据代入估收益+P0-P3 优先级；发现 4 个暗角(持久化前缀库/预测预热/KV量化换并发/任务亲和路由)，#15 持久化前缀库应提 P0 |
| `figures/*.svg` | 图 | 6 张分布/DAG/对照可视化 |
| `raw-data/20260629-swe-rows-100-原始数据.json` | 数据 | SWE 100 行原始输入 |
| `raw-data/20260629-gaia2-exec-30-events-only-原始数据.json` | 数据 | gaia2 30 scenario（events-only 瘦身）|

## 命名约定

沿用 vault 规范：`YYYYMMDD-HHMMSS-{task}-{result}.{ext}`，Latin 词用小写 kebab-case，CJK 保留语义。

示例：
- `20260629-xxxxxx-workload-画像计算-脚本.py`
- `20260629-xxxxxx-gateway-log-schema-设计.md`
- `20260629-xxxxxx-收益重估-分析.md`

## 主题范围

- agent 信号下沉的协同优化方案（v1/v2，已入库）
- agentic workload 系统性分析（画像方法、指标体系、三层采集，总体方案已入库）
- DSpark 投机解码及其 Ascend 移植
- 后续：网关日志采集、五张分布图计算、引擎埋点、收益重估
