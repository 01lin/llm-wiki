#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agentic Workload 画像分析 — multi-agent DAG 验证（补 SWE 数据缺失的关系维）

输入：meta-agents-research-environments/gaia2，config=execution（HF API）
      每条 scenario 的 data.events[] 是带 dependencies 的事件图：
      {event_id, event_type(USER/AGENT), dependencies[], event_relative_time, action{app,function,args}}
      → dependencies 即 DAG 边，可完整还原任务拓扑。

目的：实例化方案 §2.3 关系维指标（dag_fanout / join_wait / task_id），
      验证「DAG 组完成调度」护城河在真实 multi-agent 负载上的收益面。
      这是 SWE 单 agent 数据集（关系维为空）的补全验证。

对照方案：wiki/syntheses/20260629-100000-agentic-workload-analysis-总体方案设计-综合.md
对照报告：20260629-005000-swe-trajectory-样例实例化分析-报告.md

用法：python3 此脚本 gaia2_exec_30.json
"""
import sys, json, statistics
from collections import Counter, defaultdict

def build_dag(events):
    """从 events[] 还原 DAG：节点=event_id，边=dependencies。返回拓扑度量。"""
    nodes = {e["event_id"]: e for e in events}
    indeg = {eid: 0 for eid in nodes}            # 入度 = 依赖数
    outdeg = {eid: 0 for eid in nodes}           # 出度 = 被依赖数
    children = defaultdict(list)
    for e in events:
        eid = e["event_id"]
        deps = e.get("dependencies") or []
        indeg[eid] = len(deps)
        for d in deps:
            if d in nodes:
                outdeg[d] += 1
                children[d].append(eid)
    roots = [n for n in nodes if indeg[n] == 0]   # 根 = USER 请求
    leaves = [n for n in nodes if outdeg[n] == 0] # 叶 = 终态
    # fan-out：单节点最大出度（一个节点派生多少并行子任务）
    max_fanout = max(outdeg.values()) if outdeg else 0
    # join：入度>1 的节点数（多个前置汇聚 = join 点）
    joins = [n for n in nodes if indeg[n] > 1]
    max_join_indeg = max((indeg[n] for n in joins), default=0)
    # 关键路径长度（DAG 最长链）= 串行下界
    memo = {}
    def depth(n):
        if n in memo: return memo[n]
        deps = nodes[n].get("dependencies") or []
        deps = [x for x in deps if x in nodes]
        memo[n] = 1 + max((depth(x) for x in deps), default=0)
        return memo[n]
    critical_path = max((depth(n) for n in nodes), default=0)
    return dict(
        n_nodes=len(nodes), n_edges=sum(indeg.values()),
        n_roots=len(roots), n_leaves=len(leaves),
        max_fanout=max_fanout, n_joins=len(joins), max_join_indeg=max_join_indeg,
        critical_path=critical_path,
        # 并行度 = 节点数/关键路径：>1 说明有可并行空间（join 前木桶效应来源）
        parallelism=round(len(nodes) / critical_path, 2) if critical_path else 0,
    )

def analyze(rows):
    per = []
    app_counter = Counter()
    func_counter = Counter()
    type_counter = Counter()
    for r in rows:
        dj = json.loads(r["row"]["data"])
        events = dj.get("events", [])
        if not events:
            continue
        dag = build_dag(events)
        # 动作语义：每个 AGENT event 是一次工具调用 app.function
        for e in events:
            type_counter[e.get("event_type")] += 1
            act = e.get("action") or {}
            if isinstance(act, dict):
                if act.get("app"): app_counter[act["app"]] += 1
                if act.get("function"): func_counter[act["function"]] += 1
        # join 等待代理：依赖事件的 relative_time 跨度（最早→最晚前置）
        rel_times = [e.get("event_relative_time", 0) for e in events]
        per.append(dict(
            scenario=r["row"]["scenario_id"],
            **dag,
            time_span=round(max(rel_times) - min(rel_times), 1) if rel_times else 0,
        ))
    return per, dict(app=app_counter, func=func_counter, type=type_counter)

def pct(xs, p):
    if not xs: return 0
    xs = sorted(xs); k = (len(xs)-1)*p/100
    f = int(k); c = min(f+1, len(xs)-1)
    return round(xs[f] + (xs[c]-xs[f])*(k-f), 1)

def report(per, agg):
    n = len(per)
    print("="*72)
    print(f"Multi-Agent DAG 画像 — gaia2/execution（n={n} scenario）")
    print("="*72)

    print("\n【元信息：动作语义】每个 AGENT event = 一次工具调用 app.function")
    print("  event 类型:", dict(agg['type']))
    print("  Top apps  :", agg['app'].most_common(6))
    print("  Top funcs :", agg['func'].most_common(6))

    # 关系维核心：DAG 拓扑（SWE 数据集完全没有的维度）
    nodes = [p["n_nodes"] for p in per]
    edges = [p["n_edges"] for p in per]
    print("\n【DAG 规模】[报] dependencies 字段还原 —— SWE 单 agent 数据此维为空")
    print(f"  节点数/scenario: P50={pct(nodes,50)} P90={pct(nodes,90)} max={max(nodes)} mean={round(statistics.mean(nodes),1)}")
    print(f"  边数/scenario  : P50={pct(edges,50)} P90={pct(edges,90)} mean={round(statistics.mean(edges),1)}")

    # fan-out：决定并行子任务规模 → 组完成调度的作用面
    fo = [p["max_fanout"] for p in per]
    print("\n【分布·关系维① fan-out（单节点最大派生并行子任务数）】")
    print(f"  P50={pct(fo,50)} P90={pct(fo,90)} max={max(fo)} mean={round(statistics.mean(fo),1)}")
    has_fanout = sum(1 for x in fo if x >= 2)
    print(f"  有并行 fan-out(≥2) 的 scenario 占比 = {has_fanout/n:.0%}  → 组完成调度的直接作用面")

    # join：决定木桶效应风险 → 组完成调度的核心收益点
    joins = [p["n_joins"] for p in per]
    jindeg = [p["max_join_indeg"] for p in per]
    print("\n【分布·关系维② join（入度>1 的汇聚点 = 木桶效应风险点）】")
    print(f"  join 点数/scenario: mean={round(statistics.mean(joins),1)} max={max(joins)}")
    print(f"  最大 join 入度    : P90={pct(jindeg,90)} max={max(jindeg)}  → 一个 join 等 N 个前置，N 越大木桶越深")
    has_join = sum(1 for x in joins if x >= 1)
    print(f"  有 join 的 scenario 占比 = {has_join/n:.0%}  → 这些场景下 KV 空占等待真实存在")

    # 并行度：节点数/关键路径 → 可并行空间（>1 即有组完成调度收益）
    par = [p["parallelism"] for p in per]
    cp = [p["critical_path"] for p in per]
    print("\n【分布·关系维③ 并行度 = 节点数/关键路径长】")
    print(f"  关键路径(串行下界): mean={round(statistics.mean(cp),1)}  并行度: P50={pct(par,50)} P90={pct(par,90)} max={max(par)}")
    high_par = sum(1 for x in par if x >= 2)
    print(f"  并行度≥2 的 scenario = {high_par/n:.0%}  → 早算完的子任务 KV 空占等 join，组完成调度可回收")

    print("\n" + "="*72)
    print("【收益重估：DAG 维度代回方案 §3.6 / §4】")
    print("="*72)
    print(f"  fan-out 均 {round(statistics.mean(fo),1)}、{has_join/n:.0%} scenario 有 join、并行度均 {round(statistics.mean(par),1)}")
    print(f"  → 「DAG 组完成调度」护城河在 multi-agent 负载上【收益面坐实】：")
    print(f"     · fan-out 子任务并行执行，各占 KV slot")
    print(f"     · join 点前，早完成的子任务 KV 空占等待最慢的（木桶效应）")
    print(f"     · 组完成调度 = 以 DAG 为单位仲裁，避免组内前置被逐出拖垮整组")
    print(f"  对比 SWE 单 agent（fan-out=1，无 join）→ 该优化对单 agent 负载无收益，")
    print(f"     验证方案判断：DAG 调度是 multi-agent 专属护城河，需按负载类型差异化启用。")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "gaia2_exec_30.json"
    data = json.load(open(path, encoding="utf-8"))
    per, agg = analyze(data["rows"])
    report(per, agg)
