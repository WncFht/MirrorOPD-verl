"""Ledger: each bar is one completed 128-prompt x16 validation run.
Axes are full validation seconds, actual generated tokens/s, sampled NVML busy,
and observed whole-device peak GiB; colors identify execution configurations.
Natural-EOS runs; repeat means are explicit, TPS uses total tokens/total time, memory uses maximum. Not a deterministic fixed-token workload.
"""
from pathlib import Path
import json,sys,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/dotfiles/.agents/skills/figure-viz/assets/mpl')
import fvstyle as fv
P=Path(__file__).resolve().parent.parent
allruns=json.loads((P/'analysis.json').read_text())['runs'];keys=['valid-w32-s32-20261022-a','valid-w128-s32-20261022-a','valid-w128-s64-20261022-a'];rows=[];labels=[]
for key in keys:
 matches=[v for k,v in allruns.items() if k.rsplit('-',1)[0]==key.rsplit('-',1)[0]]
 row=dict(matches[0])
 for field in ['testing_seconds','gpu_busy_mean']:row[field]=statistics.mean(r[field] for r in matches)
 row['output_tokens_per_second']=sum(r['tokens'] for r in matches)/sum(r['testing_seconds'] for r in matches)
 row['gpu_memory_max_gib']=max(r['gpu_memory_max_gib'] for r in matches)
 rows.append(row)
 labels.append(key.split('-')[1].upper()+' / '+'C'+key.split('-')[2][1:]+('\n(n=2)' if len(matches)==2 else ''))
fv.apply_rc('paper');colors=fv.series_colors('paper',3)
fig,axes=plt.subplots(2,2,figsize=(5.5,4.2))
for ax,(k,label,scale) in zip(axes.flat,[('testing_seconds','Validation time (s)',1),('output_tokens_per_second','Output tokens / second',1),('gpu_busy_mean','GPU busy (%)',1),('gpu_memory_max_gib','Whole-device peak (GiB)',1)]):
 ax.bar(labels,[r[k]*scale for r in rows],color=colors,width=.6)
 ax.set_ylabel(label);ax.tick_params(axis='x',labelsize=7);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 if k=='gpu_busy_mean':ax.set_ylim(0,100)
fig.text(.5,.025,'4 x H800; W = prompt window; C = per-engine concurrency; identical evaluation budget.',ha='center',fontsize=7,color=fv.NEUTRALS['baseline'])
fig.tight_layout(rect=(0,.07,1,1));fig.canvas.draw();audit=fv.audit(fig);(P/'figure-audit.json').write_text(json.dumps(audit));fig.savefig(P/'validation-tuning.png',dpi=220);fig.savefig(P/'validation-tuning.pdf');print(audit)
