# ruff: noqa: E501
"""Offline spectrum/held-out scatter explorer from the audited joined CSV.

Color identifies student family; marker identifies added stage. Every point is
one run/step/panel; selecting multiple input modes does not create independent
teacher samples. All numbers are measured, no fitted values enter the scatter.
"""

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "spectrum-expanded-v4"
sys.path.insert(0, str(ROOT.parents[2] / "dotfiles/.agents/skills/figure-viz/assets/mpl"))
import fvstyle as fv  # noqa: E402


def main():
    original = (
        ROOT.parents[1] / "examples/mirroropd/docs/experiments/teacher-transfer/fixed-state-results/explorer.html"
    )
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", original.read_text(), flags=re.S)
    plotly_bundle = next(s for s in scripts if "plotly.js" in s[:500].lower())
    rows = list(csv.DictReader((OUT / "heldout-feature-join.csv").open()))
    numeric = [
        "initial_rkl",
        "final_rkl",
        "remaining",
        "absolute_decrease",
        "late_relative_decrease",
        "log_absorption",
        "norm_over_base",
        "stable_rank_fraction",
        "rank90_fraction",
        "top1pct",
        "top5pct",
        "kappa_spec",
        "rho_sigma",
        "weight_stable_rank",
        "weight_top16",
        "weight_rank90_fraction",
    ]
    for r in rows:
        for k in numeric:
            r[k] = float(r[k]) if r[k] else None
        r["step"] = int(r["step"])
    options = {
        "norm_over_base": "Teacher delta norm / initial norm",
        "stable_rank_fraction": "Teacher delta stable rank / matrix rank",
        "rank90_fraction": "Teacher delta 90% energy rank fraction",
        "top1pct": "Teacher delta top-1% energy",
        "top5pct": "Teacher delta top-5% energy",
        "kappa_spec": "Teacher delta base-frame kappa",
        "rho_sigma": "Teacher delta singular-value displacement ratio",
        "weight_stable_rank": "Teacher full-weight stable rank",
        "weight_top16": "Teacher full-weight top16 energy",
        "weight_rank90_fraction": "Teacher full-weight 90% rank fraction",
        "initial_rkl": "Initial held-out RKL",
    }
    page = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>Teacher spectrum × held-out</title>
<style>body{font:15px system-ui;margin:24px;color:#25323b;background:#fff}h1{font-size:24px}label{display:inline-block;margin:8px}select{padding:5px;max-width:380px}#plot{height:620px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.note{color:#596975;max-width:1050px;line-height:1.7}</style>
<h1>Teacher spectrum × held-out</h1><p class="note">精确 CPU FP64 SVD，首／中／末层 k_proj 与 v_proj 共六个矩阵的中位数。参数 delta 相对实际 initial student；跨尺寸条件仅提供完整权重谱。每点对应一条原始曲线，同 teacher 的输入模式是相关观测。RKL 使用原固定状态与 H1024 分母。</p>
<label>指标 <select id="feature"></select></label><label>结果 <select id="target"><option value="remaining">剩余比例 Kt/K0</option><option value="final_rkl">绝对终点 RKL</option><option value="absolute_decrease">绝对下降 K0−Kt</option><option value="late_relative_decrease">末64步相对下降</option><option value="log_absorption">对数吸收量 log10(K0/Kt)</option></select></label>
<label>更新 <select id="step"><option>64</option><option>320</option></select></label><label>面板 <select id="panel"><option>All held-out</option><option>Math</option><option>Code</option><option>IF</option><option>Training scored</option></select></label><label>Student／模式 <select id="student"><option value="all">全部</option></select></label><label><input id="direct" type="checkbox">仅已登记直接父型号</label><label><input id="logx" type="checkbox" checked>log X</label><label><input id="logy" type="checkbox" checked>log Y</label><button id="download">下载当前 CSV</button><div id="count"></div><div id="plot"></div><table><thead><tr><th>Student / teacher</th><th>指标</th><th>初始 KL</th><th>终点 KL</th><th>剩余比例</th><th>末64步下降</th></tr></thead><tbody id="rows"></tbody></table><script>__PLOTLY__</script><script>
const DATA=__DATA__, FEATURES=__FEATURES__;
const el=id=>document.getElementById(id),esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
Object.entries(FEATURES).forEach(([k,v])=>el('feature').add(new Option(v,k)));
[...new Set(DATA.map(r=>r.student))].forEach(s=>el('student').add(new Option(s,s)));
let selected=[];
function render(){const x=el('feature').value,y=el('target').value;selected=DATA.filter(r=>r.step===+el('step').value&&r.panel===el('panel').value&&(el('student').value==='all'||r.student===el('student').value)&&(!el('direct').checked||r.lineage==='published_direct_parent')&&r[x]!==null&&r[y]!==null);
const family=r=>r.student.startsWith('r')?'R1':r.student.startsWith('q')?'Qwen':'Llama',colors=__COLORS__;
let traces=[];Object.keys(colors).forEach(f=>{let a=selected.filter(r=>family(r)===f);traces.push({type:'scatter',mode:'markers',name:f,x:a.map(r=>r[x]),y:a.map(r=>r[y]),text:a.map(r=>esc(r.student+' / '+r.label)),customdata:a.map(r=>[r.initial_rkl,r.final_rkl,r.remaining,r.lineage,r.added_stage]),marker:{size:11,color:colors[f],symbol:a.map(r=>r.added_stage==='added SFT'?'square':r.added_stage.startsWith('RL')?'circle':'triangle-up')},hovertemplate:'%{text}<br>X=%{x:.5g}<br>Y=%{y:.5g}<br>K0=%{customdata[0]:.5g}<br>Kt=%{customdata[1]:.5g}<br>remaining=%{customdata[2]:.4f}<br>%{customdata[3]}<br>%{customdata[4]}<extra></extra>'});});
Plotly.react('plot',traces,{margin:{t:25,r:20,b:70,l:85},xaxis:{title:FEATURES[x],type:el('logx').checked?'log':'linear'},yaxis:{title:el('target').selectedOptions[0].text,type:el('logy').checked?'log':'linear'},legend:{orientation:'h'},paper_bgcolor:'white',plot_bgcolor:'white'},{responsive:true});
el('count').textContent=`${selected.length} 条曲线，${new Set(selected.map(r=>r.teacher_path)).size} 个 teacher 权重。方形=SFT；圆形=RL；三角=其他训练历史。`;
const fmt=v=>v===null?'—':v.toPrecision(5);el('rows').innerHTML=[...selected].sort((a,b)=>a.remaining-b.remaining).map(r=>'<tr><td>'+esc(r.student+' / '+r.label)+'</td>'+[r[x],r.initial_rkl,r.final_rkl,r.remaining,r.late_relative_decrease].map(v=>'<td>'+fmt(v)+'</td>').join('')+'</tr>').join('');}
['feature','target','step','panel','student','direct','logx','logy'].forEach(id=>el(id).onchange=render);
el('target').onchange=()=>{el('logy').checked=['remaining','final_rkl'].includes(el('target').value);render();};
el('download').onclick=()=>{let keys=['student','teacher','step','panel',el('feature').value,'initial_rkl','final_rkl','remaining','late_relative_decrease','lineage'];let csv=[keys.join(','),...selected.map(r=>keys.map(k=>JSON.stringify(r[k])).join(','))].join('\\n');let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));a.download='spectrum-heldout.csv';a.click();URL.revokeObjectURL(a.href);};render();</script></html>"""
    palette = fv.series_colors("paper", 4)
    colors = dict(zip(["R1", "Qwen", "Llama"], [palette[0], palette[1], palette[3]], strict=True))
    page = (
        page.replace("__PLOTLY__", plotly_bundle)
        .replace("__COLORS__", json.dumps(colors))
        .replace("__DATA__", json.dumps(rows).replace("<", "\\u003c"))
        .replace("__FEATURES__", json.dumps(options))
    )
    (OUT / "explorer.html").write_text(page)
    print("Offline explorer built", len(rows), "rows")


if __name__ == "__main__":
    main()
