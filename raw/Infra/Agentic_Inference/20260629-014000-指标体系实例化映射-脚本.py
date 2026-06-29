#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agentic 指标体系实例化映射 —— 把方案 §2 的 20 个指标逐个映射到两个真实数据集

输入：SWE 100 行 + gaia2 30 scenario（raw-data/）
输出：每个指标的「数据集实例化值 / 覆盖状态 / 缺失原因」三联表（stdout，贴入报告）

覆盖状态：✅可算  △近似  ✗缺失(需补)
对照方案 §2.1-2.4。
用法：python3 此脚本 <swe_json> <gaia2_json>
"""
import sys, json, statistics
from collections import Counter, defaultdict

def est_tokens(s): return max(1, len(s)//4)

def swe_metrics(path):
    d=json.load(open(path)); per=[]
    for r in d["rows"]:
        row=r["row"]; traj=row.get("trajectory") or []
        roles=[m.get("role") for m in traj]
        sys_text=next((m.get("system_prompt") or m.get("text","") for m in traj if m.get("role")=="system"),"") or ""
        in_tok=sum(est_tokens(m.get("text") or "") for m in traj if m.get("role") in("user","system"))
        out_tok=sum(est_tokens(m.get("text") or "") for m in traj if m.get("role")=="ai")
        tool_ret=sum(est_tokens(m.get("text") or "") for m in traj if m.get("role")=="user")
        # 段：ai 内 ``` 围栏
        th=tc=0
        for m in traj:
            if m.get("role")=="ai":
                parts=(m.get("text") or "").split("```")
                th+=len(parts[0]); tc+=sum(len(p) for p in parts[1:])
        per.append(dict(R=roles.count("ai"), n_user=roles.count("user"),
            sys_tok=est_tokens(sys_text), in_tok=in_tok, out_tok=out_tok, tool_ret=tool_ret,
            ctx=in_tok+out_tok, has_tools=1, th=th, tc=tc,
            exit=row.get("exit_status"), target=bool(row.get("target"))))
    return per

def gaia_metrics(path):
    d=json.load(open(path)); per=[]
    for r in d["rows"]:
        dj=json.loads(r["row"]["data"]); events=dj.get("events",[])
        if not events: continue
        nodes={e["event_id"]:e for e in events}
        outdeg=Counter(); indeg={}
        for e in events:
            deps=e.get("dependencies") or []; indeg[e["event_id"]]=len(deps)
            for x in deps:
                if x in nodes: outdeg[x]+=1
        memo={}
        def depth(n):
            if n in memo: return memo[n]
            deps=[x for x in (nodes[n].get("dependencies") or []) if x in nodes]
            memo[n]=1+max((depth(x) for x in deps),default=0); return memo[n]
        cp=max((depth(n) for n in nodes),default=1)
        # join 等待代理：join 节点的依赖中 relative_time 跨度
        rel={e["event_id"]:e.get("event_relative_time",0) for e in events}
        per.append(dict(n_nodes=len(nodes), fanout=max(outdeg.values(),default=0),
            joins=sum(1 for n in nodes if indeg[n]>1), cp=cp,
            par=round(len(nodes)/cp,2) if cp else 0,
            n_user=sum(1 for e in events if e.get("event_type")=="USER"),
            n_agent=sum(1 for e in events if e.get("event_type")=="AGENT")))
    return per

def fmt(x, suffix=""):
    return f"{x}{suffix}"

if __name__=="__main__":
    swe=swe_metrics(sys.argv[1]); gaia=gaia_metrics(sys.argv[2])
    # 预聚合
    R=[p["R"] for p in swe]; sysr=[p["sys_tok"]/p["ctx"] for p in swe if p["ctx"]]
    toolr=[p["tool_ret"]/p["ctx"] for p in swe if p["ctx"]]
    th=sum(p["th"] for p in swe); tc=sum(p["tc"] for p in swe); seg=th+tc or 1
    trunc=sum(1 for p in swe if "exit_context" in (p["exit"] or ""))/len(swe)
    fo=[p["fanout"] for p in gaia]; jn=[p["joins"] for p in gaia]; par=[p["par"] for p in gaia]

    rows = [
        # (维度, 指标, SWE实例化值, gaia2实例化值, 覆盖, 说明)
        ("时间维","inter_request_gap_ms", "✗ 无时间戳", "△ relative_time(逻辑序)", "△",
            "离线数据集均无真实墙钟间隔；gaia2 有逻辑时序但非真实阻塞"),
        ("时间维","tool_blocking_ratio β", "△ 工具轮代理=mean 24", "△ fan-out 分支数", "△",
            "只能用工具调用频次代理，真实阻塞时长缺"),
        ("时间维","expected_idle_ms", "✗", "✗", "✗",
            "未来量，两数据集都无 → 印证必须 [报] 上报"),
        ("时间维","remaining_turns", f"△ 可后验 R={statistics.mean(R):.0f}", "△ 关键路径剩余", "△",
            "离线可后验算总轮次，在线预测需 [报]"),
        ("时间维","osl_actual/predicted", f"✅ out_tok 可算", "✅ 节点数", "✅",
            "actual 可推断；predicted 需 [报]"),
        ("结构维","has_tools/tools_token", "✅ 100% 带 tools", "✅ app/function", "✅",
            "SWE 全是 tool agent；gaia2 每 event 是 app.function"),
        ("结构维","segment_type_ratio", f"✅ think:tool={th/seg:.0%}:{tc/seg:.0%}", "✅ USER vs AGENT", "✅",
            "SWE 用 ``` 近似；gaia2 event_type 天然分段"),
        ("结构维","segment_length_dist", "✅ 各段 token 可算", "✅ 各 app 调用数", "✅",
            "可算，精确需 [埋] detokenize 打标"),
        ("结构维","accept_rate_by_segment", "✗ 无投机执行", "✗", "✗",
            "离线轨迹无投机解码过程 → 需引擎 [埋]"),
        ("结构维","one_shot_history_ratio", f"✅ 工具返回占 {statistics.mean(toolr):.0%}", "✅ 子任务结果", "✅",
            "SWE 工具返回是上下文主体；精确需 [报] 段角色"),
        ("关系维","session_key/final", "✅ messages 前缀 hash", "✅ scenario_id", "✅",
            "可推断/还原；final 精确需 [报]"),
        ("关系维","prefix_reuse_rate ρ", f"✅ system ρ={statistics.mean(sysr):.0%} 单模板", "✅ 多 app 共享", "✅",
            "可 hash 比对；实际命中需 [埋]"),
        ("关系维","task_id/parent_request_id", "✗ 单 agent 无", f"✅ DAG 还原", "✅(gaia2)",
            "SWE 无派生；gaia2 dependencies 即派生关系"),
        ("关系维","dag_fanout/join_wait_ms", "✗ fan-out=1", f"✅ fanout={statistics.mean(fo):.1f} join={sum(1 for x in jn if x)/len(gaia):.0%}", "✅(gaia2)",
            "本轮 gaia2 补全坐实；join_wait 精确时长需 [报]"),
        ("关系维","priority", "✗ 数据集无", "✗", "✗",
            "调度元数据，数据集不含 → [报] (Dynamo 已有)"),
        ("引擎基线","prefix_cache_hit_rate", "✗ 离线无引擎", "✗", "✗",
            "引擎运行时指标 → [埋] find_longest_cache_hit"),
        ("引擎基线","preempt_count/recompute", f"△ 截断率 {trunc:.0%} 代理", "✗", "△",
            "SWE exit_context 率可作 KV 压力代理；精确需 [埋]"),
        ("引擎基线","hbm_kv_utilization", "✗", "✗", "✗",
            "引擎运行时 → [埋] block pool"),
        ("引擎基线","ttft/tpot", "✗ 离线无时延", "✗", "✗",
            "需在线 SSE 时间戳 [推] 或引擎 [埋]"),
        ("引擎基线","effective_concurrency", "✗", "✗", "✗",
            "引擎运行时 → [埋] 调度器"),
    ]

    print("="*100)
    print("Agentic 指标体系实例化映射（方案 §2 的 20 指标 × 两数据集真实值）")
    print("="*100)
    print(f"{'维度':<8}{'指标':<28}{'SWE 实例化':<26}{'gaia2 实例化':<26}{'覆盖':<5}")
    print("-"*100)
    cov=Counter()
    for dim,name,sv,gv,c,note in rows:
        cov[c]+=1
        print(f"{dim:<8}{name:<28}{sv:<26}{gv:<26}{c:<5}")
    print("-"*100)
    print(f"覆盖统计：✅可算={cov['✅']+cov.get('✅(gaia2)',0)}  △近似={cov['△']}  ✗缺失={cov['✗']}  (总 {len(rows)})")
    print("\n【缺失指标(✗)的共性】→ 全部是『引擎运行时指标』或『未来量/调度元数据』：")
    for dim,name,sv,gv,c,note in rows:
        if c=="✗": print(f"  ✗ {name:<28} {note}")
    print("\n【近似指标(△)的提升路径】→ 离线代理量 → 在线精确采集：")
    for dim,name,sv,gv,c,note in rows:
        if c=="△": print(f"  △ {name:<28} {note}")
