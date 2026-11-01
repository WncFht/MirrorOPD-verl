"""Keep equal-input512 comparisons and allocator capacity evidence in one figure."""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
sys.path.insert(0, os.environ.get('FIGURE_VIZ_MPL','/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/dotfiles/.agents/skills/figure-viz/assets/mpl'))
import fvstyle as fv
import matplotlib.pyplot as plt

work = Path(__file__).resolve().parent
s = json.loads((work/'summary.json').read_text())
keys = ['baseline-size512','partial-checkpoint-size512','partial-expandable-size512','lean-combo-size512']
labels = ['Full checkpointing\nBaseline','Selective combo\nDefault allocator','Same selective combo\nExpandable segments','Full checkpointing\nBudget65536 / diagnostics off']
fv.apply_rc('paper')
fig, axes = plt.subplots(1,2,figsize=(7.8,4.3),sharey=True,gridspec_kw={'width_ratios':[1.25,1]})
ys=[];times=[];alloc=[];reserved=[]
for i,k in enumerate(keys):
    if k not in s['cases']:
        if k in s['failed_cases']:
            for ax in axes:ax.text(2,i,'OOM',va='center',fontsize=9,color=fv.HUES['vcoral'])
        continue
    r=s['cases'][k];ys.append(i);times.append(r['median_seconds']);alloc.append(r['allocated_peak_gib']);reserved.append(r['reserved_peak_gib'])
    line,=axes[0].plot([min(r['measured_seconds']),max(r['measured_seconds'])],[i,i],color=fv.NEUTRALS['vink'],lw=1.2,zorder=3);fv.mark_neutral(line)
    dot=axes[0].scatter(r['measured_seconds'],[i]*2,s=12,color=fv.NEUTRALS['vink'],zorder=4);fv.mark_neutral(dot)
    axes[0].text(max(r['measured_seconds'])+2,i,f"{r['median_seconds']:.1f}",va='center',fontsize=9)
axes[0].barh(ys,times,color=fv.NEUTRALS['baseline'],height=.48)
axes[1].scatter(alloc,[y-.12 for y in ys],color=fv.HUES['vteal'],marker='o',s=32,label='Allocated peak')
axes[1].scatter(reserved,[y+.12 for y in ys],color=fv.HUES['vorange'],marker='s',s=32,label='Reserved peak')
axes[1].legend(loc='upper center',bbox_to_anchor=(.48,1.22),fontsize=8,ncol=1,frameon=False)
axes[0].set_yticks(range(4),labels,fontsize=9)
axes[0].set_ylim(3.5,-.5)
axes[0].set_xlim(0,135);axes[1].set_xlim(0,85)
axes[0].set_xlabel('Actor time for 4 updates (s)');axes[1].set_xlabel('Independent memory peaks (GiB)')
axes[0].set_ylabel('512 fixed trajectories')
for ax in axes:ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
fig.text(.5,.050,'Two H800s; 587,956 input tokens; lr0; two calls per finite setting. Ranges are raw calls, not confidence intervals.',ha='center',fontsize=8,color=fv.NEUTRALS['baseline'])
fig.text(.5,.012,'Selective combo: 14/28 checkpointed layers, no-sync, diagnostics off. Only allocator differs between rows 2 and 3.',ha='center',fontsize=8,color=fv.NEUTRALS['baseline'])
fig.tight_layout(rect=(0,.09,1,.94))
fig.canvas.draw();audit=str(fv.audit(fig));print(audit)
for ext in ('png','pdf'):fig.savefig(work/('capacity-comparison.'+ext),dpi=180,bbox_inches='tight')
(work/'capacity-comparison.audit.txt').write_text(audit+'\n')
