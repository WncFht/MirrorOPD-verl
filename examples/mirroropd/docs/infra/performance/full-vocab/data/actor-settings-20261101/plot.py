"""Plot measured settings at fixed256 trajectories; dots/ranges are raw calls."""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
style = Path(os.environ.get('FIGURE_VIZ_MPL', '/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/dotfiles/.agents/skills/figure-viz/assets/mpl'))
sys.path.insert(0, str(style))
import fvstyle as fv
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--summary', type=Path, default=Path(__file__).with_name('summary.json'))
parser.add_argument('--output', type=Path, default=Path(__file__).with_name('settings-comparison'))
args = parser.parse_args()
summary = json.loads(args.summary.read_text())
labels = {
    'baseline':'Baseline', 'no-sync':'No-sync', 'budget65536':'Token budget 65536',
    'diagnostics-off':'Diagnostics off', 'chunk1024':'Head chunk 1024',
    'checkpoint-off':'Checkpoint off / 32768',
    'no-sync-budget65536':'No-sync + budget 65536', 'baseline-recheck':'Baseline recheck',
    'checkpoint-off-budget24576':'Checkpoint off / 24576',
    'combined-checkpoint-off24576':'Checkpoint off / 24576\n+ no-sync + diagnostics off',
    'partial-checkpoint':'Checkpoint 14/28 layers',
    'partial-checkpoint-combined':'Checkpoint 14/28 layers\n+ no-sync + diagnostics off',
    'lean-combo':'Full checkpointing\n+ budget 65536 + diagnostics off',
}
order = [k for k in labels if k in summary['cases'] or k in summary['failed_cases']]
values = {k:summary['cases'][k] for k in order if k in summary['cases']}
best = min(values, key=lambda k:values[k]['median_seconds'])
fv.apply_rc('paper')
fig, axes = plt.subplots(1,2,figsize=(7.8,7.6),sharey=True,gridspec_kw={'width_ratios':[1.35,1]})
for i,key in enumerate(order):
    color = fv.NEUTRALS['baseline'] if key.startswith('baseline') else fv.HUES['vteal']
    if key == best:
        color = fv.HUES['vorange']
    if key not in values:
        for ax in axes:
            ax.text(1,i,'OOM',va='center',ha='left',color=fv.HUES['vcoral'],fontsize=9)
        continue
    row = values[key]
    seconds = row['median_seconds']
    times = row['measured_seconds']
    # Bars are drawn as one categorical series below.
    line, = axes[0].plot([min(times),max(times)],[i,i],color=fv.NEUTRALS['vink'],lw=1.1)
    fv.mark_neutral(line)
    dots = axes[0].scatter(times,[i]*len(times),s=12,color=fv.NEUTRALS['vink'],zorder=3)
    fv.mark_neutral(dots)
    axes[0].text(max(times)+1.2,i,f'{seconds:.1f}',va='center',fontsize=8)
    # Memory is one categorical series as well.
    axes[1].text(row['allocated_peak_gib']+1,i,f"{row['allocated_peak_gib']:.1f}",va='center',fontsize=8)
finite = [k for k in order if k in values]
colors = [fv.HUES['vorange'] if k == best else fv.NEUTRALS['baseline'] if k.startswith('baseline') else fv.HUES['vteal'] for k in finite]
axes[0].barh([order.index(k) for k in finite],[values[k]['median_seconds'] for k in finite],height=.52,color=colors,zorder=1)
axes[1].barh([order.index(k) for k in finite],[values[k]['allocated_peak_gib'] for k in finite],height=.52,color=colors,zorder=1)
axes[0].set_yticks(range(len(order)),[labels[k] for k in order],fontsize=9)
axes[0].set_ylim(len(order)-.5,-.5)
axes[0].set_xlim(0,max(max(v['measured_seconds']) for v in values.values())*1.18)
axes[1].set_xlim(0,max(v['allocated_peak_gib'] for v in values.values())*1.2)
axes[0].set_xlabel('Actor time for 4 updates (s)')
axes[1].set_xlabel('Allocated peak (GiB)')
axes[0].set_ylabel('Setting')
for ax in axes:
    ax.grid(axis='x',alpha=.2)
    ax.set_axisbelow(True)
fig.text(.5,.035,'Two H800s; fixed 256 trajectories; median of two calls. Dots/range: raw calls, not confidence intervals.',ha='center',fontsize=8,color=fv.NEUTRALS['baseline'])
fig.text(.5,.008,'Warmup/profile excluded. Orange: fastest finite timing in this scan. Ray/TQ, rollout and teacher preparation excluded.',ha='center',fontsize=8,color=fv.NEUTRALS['baseline'])
fig.tight_layout(rect=(0,.08,1,1))
fig.canvas.draw()
audit = str(fv.audit(fig))
args.output.parent.mkdir(parents=True,exist_ok=True)
for extension in ('png','pdf'):
    fig.savefig(args.output.with_suffix('.'+extension),dpi=180,bbox_inches='tight')
args.output.with_suffix('.audit.txt').write_text(audit+'\n')
args.output.with_suffix('.json').write_text(json.dumps({'source':str(args.summary),'rows':{k:values.get(k,{'status':'OOM'}) for k in order},'audit':audit},indent=2)+'\n')
print(audit)
