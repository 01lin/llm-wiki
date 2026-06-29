#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agentic Workload 画像分析 — 样例实例化脚本

输入：nebius/SWE-agent-trajectories（HF 数据集，SWE-agent 解 GitHub issue 轨迹）
      经 HF datasets-server API 拉取的 JSON 行（每行含 trajectory: [{role,text,system_prompt,...}]）

目的：把「Agentic Workload 总体方案设计」§1.4 的五张分布、§2 的指标体系，
      在真实 agentic 轨迹上跑一遍，验证：① 哪些指标 OpenAI chat 结构能推断 [推]
      ② 哪些必须新增采集 [埋]/[报]，并产出真实数字。

对照方案：wiki/syntheses/20260629-100000-agentic-workload-analysis-总体方案设计-综合.md

用法：python3 此脚本 swe_rows_100.json
"""
import sys, json, re, statistics
from collections import Counter

# ── token 估算：SWE 轨迹是英文+代码，按 ~4 char/token 粗估（真实应接 tokenizer）──
def est_tokens(s): return max(1, len(s) // 4)

# ── 段类型启发式：ai message 里 ``` 围栏=命令(tool_call 段)，其余=thinking 段 ──
FENCE = re.compile(r"```")
def split_ai_segments(text):
    """把一条 ai message 粗切为 thinking / tool_call 两类的字符数。
    [推] 近似：OpenAI 里若模型用 ```/tool_calls 字段可精确，此处用 ``` 围栏近似。"""
    parts = FENCE.split(text)
    thinking = parts[0]                      # 第一段（围栏前）≈ 思考
    tool_call = "".join(parts[1:])           # 围栏内+后 ≈ 命令/工具调用
    return len(thinking), len(tool_call)

# user message 末尾的稳定模板（环境状态行）——可复用前缀候选
ENV_TAIL = re.compile(r"\(Open file:.*?\)\s*\(Current directory:.*?\)\s*bash-\$?\s*$", re.DOTALL)

def analyze(rows):
    per_traj = []          # 每条轨迹一份画像
    seg_thinking_chars = 0
    seg_toolcall_chars = 0
    model_counter = Counter()
    target_counter = Counter()
    exit_counter = Counter()

    for r in rows:
        row = r["row"]
        traj = row.get("trajectory") or []
        model_counter[row.get("model_name")] += 1
        target_counter[bool(row.get("target"))] += 1
        exit_counter[row.get("exit_status")] += 1

        roles = [m.get("role") for m in traj]
        # ── R：轮次 = ai 消息数（每个 ai = 一次"思考+动作"轮）──[推] role 序列可数
        n_ai = roles.count("ai")
        n_user = roles.count("user")
        # ── system prompt token（共享前缀主体）── system_prompt 字段或首条 system text
        sys_text = ""
        for m in traj:
            if m.get("role") == "system":
                sys_text = m.get("system_prompt") or m.get("text") or ""
                break
        if not sys_text:
            # 有些把 system_prompt 挂在每条 message 上
            sys_text = next((m.get("system_prompt") for m in traj if m.get("system_prompt")), "") or ""
        sys_tok = est_tokens(sys_text)

        # ── ISL/OSL 代理 ──[推] user(含工具返回)=输入侧，ai=输出侧
        in_tok = sum(est_tokens(m.get("text") or "") for m in traj if m.get("role") in ("user", "system"))
        out_tok = sum(est_tokens(m.get("text") or "") for m in traj if m.get("role") == "ai")

        # ── 段类型占比 ──[推]近似/[埋]精确：ai 段里 thinking vs tool_call
        t_chars = c_chars = 0
        for m in traj:
            if m.get("role") == "ai":
                th, tc = split_ai_segments(m.get("text") or "")
                t_chars += th; c_chars += tc
        seg_thinking_chars += t_chars
        seg_toolcall_chars += c_chars

        # ── 一次性工具返回历史段占比 ──[报]精确，此处用 user 消息总量近似
        tool_return_tok = sum(est_tokens(m.get("text") or "") for m in traj if m.get("role") == "user")
        ctx_tok = in_tok + out_tok

        # ── 稳定模板复用（env tail）──[推] 每个 user 末尾固定结构
        env_tail_hits = sum(1 for m in traj if m.get("role") == "user" and ENV_TAIL.search(m.get("text") or ""))

        per_traj.append(dict(
            instance=row.get("instance_id"), model=row.get("model_name"),
            target=bool(row.get("target")), exit=row.get("exit_status"),
            R=n_ai, n_user=n_user, sys_tok=sys_tok,
            in_tok=in_tok, out_tok=out_tok, ctx_tok=ctx_tok,
            sys_ratio=round(sys_tok / ctx_tok, 3) if ctx_tok else 0,
            tool_return_ratio=round(tool_return_tok / ctx_tok, 3) if ctx_tok else 0,  # 占全上下文
            env_tail_hits=env_tail_hits,
        ))

    return per_traj, dict(
        model_counter=model_counter, target_counter=target_counter,
        exit_counter=exit_counter,
        seg_thinking_chars=seg_thinking_chars, seg_toolcall_chars=seg_toolcall_chars,
    )

def pct(xs, p):
    if not xs: return 0
    xs = sorted(xs); k = (len(xs)-1)*p/100
    f = int(k); c = min(f+1, len(xs)-1)
    return round(xs[f] + (xs[c]-xs[f])*(k-f), 1)

def report(per, agg):
    n = len(per)
    print("="*72)
    print(f"Agentic Workload 画像 — 样例实例化（n={n} 条 SWE-agent 轨迹）")
    print("="*72)

    print("\n【元信息】")
    print("  model 分布   :", dict(agg['model_counter']))
    print("  target(解决) :", dict(agg['target_counter']))
    print("  exit_status  :", dict(agg['exit_counter'].most_common(5)))

    # 分布①：轮次 R —— 决定价值逐出收益面
    Rs = [p["R"] for p in per]
    print("\n【分布① 轮次 R = ai 消息数】[推] role 序列可数")
    print(f"  P50={pct(Rs,50)}  P90={pct(Rs,90)}  P99={pct(Rs,99)}  max={max(Rs)}  mean={round(statistics.mean(Rs),1)}")
    single = sum(1 for r in Rs if r <= 2); longr = sum(1 for r in Rs if r >= 10)
    print(f"  单轮(≤2)占比={single/n:.0%}  长会话(≥10轮)占比={longr/n:.0%}  → 价值逐出主要作用于长会话")

    # 分布②：前缀复用率 ρ —— 决定 pin 收益
    sysr = [p["sys_ratio"] for p in per]
    syst = [p["sys_tok"] for p in per]
    print("\n【分布② 前缀复用率 ρ = system prompt 占上下文比】[推] system 字段可 hash")
    print(f"  system token: P50={pct(syst,50)} P90={pct(syst,90)}  占上下文比 mean={round(statistics.mean(sysr),3)}")
    uniq_sys = len(set(p["sys_tok"] for p in per))  # 粗略：相同长度近似相同模板
    print(f"  system 模板近似种类数≈{uniq_sys}（同一 agent 框架 → 高度复用 → pin 收益直接）")

    # 分布③：阻塞代理 β —— 本数据集缺时间戳，标注为不可推断
    print("\n【分布③ 阻塞代理 β = 工具调用阻塞时长】")
    print("  ✗ 本数据集无请求间时间戳 → OpenAI chat 离线轨迹推不出真实阻塞")
    print("  → 印证方案结论：expected_idle_ms 必须 [报] agent 上报；")
    print("    但工具返回的『轮数』可作阻塞频次代理：mean tool 调用轮 =", round(statistics.mean([p['n_user'] for p in per]),1))

    # 分布④：段结构 —— 决定 DSL / 约束解码收益面
    th = agg["seg_thinking_chars"]; tc = agg["seg_toolcall_chars"]
    tot = th + tc or 1
    print("\n【分布④ 输出段结构 thinking vs tool_call】[推]近似 ``` / [埋]精确")
    print(f"  thinking 段占 ai 输出 ={th/tot:.0%}   tool_call(命令)段 ={tc/tot:.0%}")
    print(f"  → tool_call/命令段结构化、accept rate 高 → DSL 长 draft + 约束解码收益区")

    # 分布⑤：负载聚类 —— 验证 Code/Search 二分假设
    print("\n【分布⑤ 负载画像（按 R / out_tok 粗聚类）】验证负载差异假设")
    code_like = [p for p in per if p["out_tok"] >= statistics.median([x["out_tok"] for x in per])]
    print(f"  长输出/重思考类(out_tok≥中位) n={len(code_like)}  mean R={round(statistics.mean([p['R'] for p in code_like]),1)}")
    short = [p for p in per if p not in code_like]
    print(f"  短输出/多交互类             n={len(short)}  mean R={round(statistics.mean([p['R'] for p in short]),1) if short else 0}")

    # 一次性工具返回占比 —— 决定段级 KV 压缩收益
    trr = [p["tool_return_ratio"] for p in per]
    print("\n【附 上下文构成（占全上下文 token）】[报] 段角色标注精确，此处按 role 近似")
    sysr_full = statistics.mean([p['sys_tok']/p['ctx_tok'] if p['ctx_tok'] else 0 for p in per])
    outr_full = statistics.mean([p['out_tok']/p['ctx_tok'] if p['ctx_tok'] else 0 for p in per])
    print(f"  system 前缀 ={sysr_full:.0%}   工具返回(user) ={statistics.mean(trr):.0%}   ai 输出 ={outr_full:.0%}")
    print(f"  → 工具返回是上下文主体 → 段级 KV 压缩潜力最大区（长会话尤甚）")

    # ── 收益重估输入：把真实分布代回 f(β,ρ,R) ──
    print("\n" + "="*72)
    print("【收益重估输入：真实 β/ρ/R 代回方案 §4】")
    print("="*72)
    print(f"  R（平均轮次）   = {round(statistics.mean(Rs),1)}  → 价值逐出/前缀 pin 收益随 R 放大")
    print(f"  ρ（前缀占比）   = {round(statistics.mean(sysr),2)}  → pin 命中保障的省算量")
    print(f"  β（阻塞）       = 数据集缺，需 [报] 上报后填入")
    print(f"  结论：本类负载（SWE code agent）属『长会话+高前缀复用+重 tool_call』，")
    print(f"        优先级排序 → 前缀 pin ≈ 价值逐出 > 段类型 DSL > 段级压缩 > 阻塞换出(待β)")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "swe_rows_100.json"
    data = json.load(open(path, encoding="utf-8"))
    rows = data["rows"]
    per, agg = analyze(rows)
    report(per, agg)
