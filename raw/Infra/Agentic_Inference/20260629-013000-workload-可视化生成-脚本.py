#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Workload 分析可视化生成 — 纯标准库 SVG（无 matplotlib 依赖，Obsidian/desktop 直接渲染）

产出：
  1. 数据集关键信息 + 统计明细（stdout，可贴入报告）
  2. SVG 图：分布直方图、DAG 拓扑图、SWE vs gaia2 对照条形图
  3. ASCII 直方图（嵌报告用）

用法：python3 此脚本 <swe_json> <gaia2_json> <out_svg_dir>
"""
import sys, json, statistics
from collections import Counter, defaultdict

# ── 纯 SVG 绘图原语（无第三方库）──────────────────────────────
def svg_header(w, h, title=""):
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" font-family="-apple-system,Segoe UI,Roboto,sans-serif">'
def svg_text(x, y, s, size=13, anchor="start", weight="normal", fill="#222"):
    s = (s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">{s}</text>'
def svg_rect(x, y, w, h, fill, rx=2):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"/>'
def svg_line(x1,y1,x2,y2,stroke="#bbb",w=1):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{w}"/>'
def svg_circle(cx,cy,r,fill,stroke="#fff",sw=2):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

def histogram_svg(values, bins, title, xlabel, color="#4a90d9", w=520, h=240):
    """画分布直方图 SVG。"""
    lo, hi = min(values), max(values)
    if hi == lo: hi = lo + 1
    edges = [lo + (hi-lo)*i/bins for i in range(bins+1)]
    counts = [0]*bins
    for v in values:
        idx = min(int((v-lo)/(hi-lo)*bins), bins-1)
        counts[idx] += 1
    maxc = max(counts) or 1
    pad_l, pad_b, pad_t = 40, 40, 36
    plot_w, plot_h = w-pad_l-20, h-pad_b-pad_t
    bw = plot_w/bins
    out = [svg_header(w,h)]
    out.append(svg_text(w/2, 22, title, 14, "middle", "bold"))
    # y 轴
    out.append(svg_line(pad_l, pad_t, pad_l, pad_t+plot_h, "#888"))
    out.append(svg_line(pad_l, pad_t+plot_h, pad_l+plot_w, pad_t+plot_h, "#888"))
    for i,c in enumerate(counts):
        bh = c/maxc*plot_h
        x = pad_l + i*bw + 1
        y = pad_t + plot_h - bh
        out.append(svg_rect(x, y, bw-2, bh, color))
        if c: out.append(svg_text(x+bw/2-1, y-3, str(c), 10, "middle", fill="#555"))
    # x 轴刻度（首中尾）
    for frac in (0, 0.5, 1.0):
        xe = lo + (hi-lo)*frac
        out.append(svg_text(pad_l+plot_w*frac, pad_t+plot_h+16, f"{xe:.0f}", 10, "middle", fill="#666"))
    out.append(svg_text(pad_l+plot_w/2, h-6, xlabel, 11, "middle", fill="#444"))
    out.append(svg_text(pad_l-6, pad_t+plot_h, "0", 10, "end", fill="#666"))
    out.append(svg_text(pad_l-6, pad_t+8, str(maxc), 10, "end", fill="#666"))
    out.append("</svg>")
    return "\n".join(out)

def dag_svg(events, title, w=560, h=360):
    """画一条 scenario 的 DAG 拓扑图（按依赖分层）。"""
    nodes = {e["event_id"]: e for e in events}
    indeg = {n:len(nodes[n].get("dependencies") or []) for n in nodes}
    # 分层：BFS 按最长依赖深度
    memo={}
    def depth(n):
        if n in memo: return memo[n]
        deps=[d for d in (nodes[n].get("dependencies") or []) if d in nodes]
        memo[n]=0 if not deps else 1+max(depth(d) for d in deps)
        return memo[n]
    layers=defaultdict(list)
    for n in nodes: layers[depth(n)].append(n)
    maxd=max(layers) if layers else 0
    pad_t, pad_b = 50, 20
    layer_h = (h-pad_t-pad_b)/(maxd+1) if maxd>=0 else h
    pos={}
    out=[svg_header(w,h)]
    out.append(svg_text(w/2, 24, title, 14, "middle", "bold"))
    # 算坐标
    for d in range(maxd+1):
        ns=layers[d]
        for i,n in enumerate(ns):
            x=(w/(len(ns)+1))*(i+1)
            y=pad_t+layer_h*d+layer_h/2
            pos[n]=(x,y)
    # 画边
    for n in nodes:
        for dep in (nodes[n].get("dependencies") or []):
            if dep in pos:
                x1,y1=pos[dep]; x2,y2=pos[n]
                out.append(svg_line(x1,y1+12,x2,y2-12,"#bbb",1.5))
    # 画节点
    for n in nodes:
        x,y=pos[n]
        et=nodes[n].get("event_type")
        color="#e07b39" if et=="USER" else ("#4a90d9" if indeg[n]<=1 else "#7b4ad9")  # join 紫
        out.append(svg_circle(x,y,12,color))
        act=nodes[n].get("action") or {}
        lbl = "USER" if et=="USER" else (act.get("function","")[:10] if isinstance(act,dict) else "")
        out.append(svg_text(x, y+26, lbl, 9, "middle", fill="#555"))
    # 图例
    out.append(svg_circle(20,h-12,6,"#e07b39")); out.append(svg_text(30,h-8,"USER根",10))
    out.append(svg_circle(100,h-12,6,"#4a90d9")); out.append(svg_text(110,h-8,"AGENT子任务",10))
    out.append(svg_circle(210,h-12,6,"#7b4ad9")); out.append(svg_text(220,h-8,"join汇聚点",10))
    out.append("</svg>")
    return "\n".join(out)

def bar_compare_svg(pairs, title, w=520, h=260):
    """SWE vs gaia2 对照条形图。pairs=[(label, swe_val, gaia_val), ...]"""
    pad_l, pad_t, pad_b = 110, 40, 30
    plot_w, plot_h = w-pad_l-60, h-pad_t-pad_b
    n=len(pairs); row_h=plot_h/n
    maxv=max(max(p[1],p[2]) for p in pairs) or 1
    out=[svg_header(w,h)]
    out.append(svg_text(w/2,22,title,14,"middle","bold"))
    for i,(lbl,a,b) in enumerate(pairs):
        y=pad_t+i*row_h
        out.append(svg_text(pad_l-6,y+row_h/2,lbl,11,"end",fill="#333"))
        ba=a/maxv*plot_w; bb=b/maxv*plot_w
        out.append(svg_rect(pad_l,y+4,ba,row_h/2-6,"#9aa7b5"))
        out.append(svg_text(pad_l+ba+4,y+row_h/2-2,f"{a:g}",10,fill="#666"))
        out.append(svg_rect(pad_l,y+row_h/2+1,bb,row_h/2-6,"#7b4ad9"))
        out.append(svg_text(pad_l+bb+4,y+row_h-4,f"{b:g}",10,fill="#666"))
    out.append(svg_circle(pad_l+10,h-10,5,"#9aa7b5")); out.append(svg_text(pad_l+20,h-6,"SWE单agent",10))
    out.append(svg_circle(pad_l+150,h-10,5,"#7b4ad9")); out.append(svg_text(pad_l+160,h-6,"gaia2多agent",10))
    out.append("</svg>")
    return "\n".join(out)

def ascii_hist(values, bins, label, width=40):
    lo,hi=min(values),max(values)
    if hi==lo: hi=lo+1
    counts=[0]*bins
    for v in values:
        counts[min(int((v-lo)/(hi-lo)*bins),bins-1)]+=1
    maxc=max(counts) or 1
    lines=[f"  {label}  (n={len(values)}, range [{lo:g},{hi:g}])"]
    for i,c in enumerate(counts):
        e0=lo+(hi-lo)*i/bins; e1=lo+(hi-lo)*(i+1)/bins
        bar="█"*int(c/maxc*width)
        lines.append(f"  [{e0:5.0f},{e1:5.0f}) {bar} {c}")
    return "\n".join(lines)

# ── 数据加载 + 统计 ──────────────────────────────────────────
def est_tokens(s): return max(1,len(s)//4)

def load_swe(path):
    d=json.load(open(path)); per=[]
    for r in d["rows"]:
        row=r["row"]; traj=row.get("trajectory") or []
        roles=[m.get("role") for m in traj]
        sys_text=next((m.get("system_prompt") or m.get("text","") for m in traj if m.get("role")=="system"),"")
        per.append(dict(R=roles.count("ai"), n_user=roles.count("user"),
            sys_tok=est_tokens(sys_text or ""),
            out_tok=sum(est_tokens(m.get("text") or "") for m in traj if m.get("role")=="ai"),
            target=bool(row.get("target")), exit=row.get("exit_status"), model=row.get("model_name")))
    return per

def load_gaia(path):
    d=json.load(open(path)); per=[]; raw_events=[]
    for r in d["rows"]:
        dj=json.loads(r["row"]["data"]); events=dj.get("events",[])
        if not events: continue
        nodes={e["event_id"]:e for e in events}
        outdeg=Counter(); indeg={}
        for e in events:
            deps=e.get("dependencies") or []; indeg[e["event_id"]]=len(deps)
            for d2 in deps:
                if d2 in nodes: outdeg[d2]+=1
        memo={}
        def depth(n):
            if n in memo: return memo[n]
            deps=[x for x in (nodes[n].get("dependencies") or []) if x in nodes]
            memo[n]=1+max((depth(x) for x in deps),default=0); return memo[n]
        cp=max((depth(n) for n in nodes),default=1)
        per.append(dict(n_nodes=len(nodes), max_fanout=max(outdeg.values(),default=0),
            n_joins=sum(1 for n in nodes if indeg[n]>1), critical_path=cp,
            parallelism=round(len(nodes)/cp,2) if cp else 0))
        raw_events.append(events)
    return per, raw_events

if __name__=="__main__":
    swe_path, gaia_path, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    import os; os.makedirs(outdir, exist_ok=True)
    swe=load_swe(swe_path); gaia,gaia_events=load_gaia(gaia_path)

    print("="*70); print("数据集关键信息与统计明细"); print("="*70)
    print(f"\n[SWE-agent] n={len(swe)} 轨迹")
    print(f"  model: {dict(Counter(p['model'] for p in swe))}")
    print(f"  解决率: {sum(p['target'] for p in swe)}/{len(swe)} = {sum(p['target'] for p in swe)/len(swe):.0%}")
    print(f"  exit: {dict(Counter(p['exit'] for p in swe).most_common(3))}")
    Rs=[p['R'] for p in swe]
    print(f"  轮次 R: min={min(Rs)} P50={statistics.median(Rs):.0f} mean={statistics.mean(Rs):.1f} max={max(Rs)} stdev={statistics.pstdev(Rs):.1f}")
    print("\n" + ascii_hist(Rs, 8, "SWE 轮次 R 分布"))

    print(f"\n[gaia2/execution] n={len(gaia)} scenario")
    fo=[p['max_fanout'] for p in gaia]; jn=[p['n_joins'] for p in gaia]; par=[p['parallelism'] for p in gaia]
    print(f"  节点: mean={statistics.mean([p['n_nodes'] for p in gaia]):.1f}")
    print(f"  fan-out: min={min(fo)} mean={statistics.mean(fo):.1f} max={max(fo)}")
    print(f"  join 占比: {sum(1 for x in jn if x>=1)/len(gaia):.0%}  并行度: mean={statistics.mean(par):.1f}")
    print("\n" + ascii_hist(fo, 7, "gaia2 fan-out 分布"))

    # 生成 SVG
    files=[]
    def dump(name, content):
        p=os.path.join(outdir,name); open(p,"w").write(content); files.append(name)
    dump("fig1-swe-轮次R分布.svg", histogram_svg(Rs, 12, "SWE-agent 轮次 R 分布", "R (ai 消息数/轨迹)", "#4a90d9"))
    dump("fig2-swe-输出长度分布.svg", histogram_svg([p['out_tok'] for p in swe], 12, "SWE-agent 输出 token 分布", "ai 输出 token", "#4a90d9"))
    dump("fig3-gaia2-fanout分布.svg", histogram_svg(fo, 8, "gaia2 DAG fan-out 分布", "max fan-out/scenario", "#7b4ad9"))
    dump("fig4-gaia2-并行度分布.svg", histogram_svg(par, 8, "gaia2 DAG 并行度分布", "并行度 (节点/关键路径)", "#7b4ad9"))
    # 找一条有 join 的 scenario 画 DAG
    best=max(range(len(gaia)), key=lambda i: gaia[i]['n_joins']*100+gaia[i]['max_fanout'])
    dump("fig5-gaia2-DAG拓扑示例.svg", dag_svg(gaia_events[best], f"gaia2 DAG 拓扑示例 (fan-out={gaia[best]['max_fanout']}, join={gaia[best]['n_joins']})"))
    # 对照图
    dump("fig6-swe-vs-gaia2对照.svg", bar_compare_svg([
        ("fan-out", 1, round(statistics.mean(fo),1)),
        ("join数", 0, round(statistics.mean(jn),1)),
        ("并行度", 1, round(statistics.mean(par),1)),
    ], "SWE 单agent vs gaia2 多agent — 关系维对照"))

    print("\n" + "="*70); print(f"已生成 {len(files)} 张 SVG 图到 {outdir}/")
    for f in files: print("  -", f)
