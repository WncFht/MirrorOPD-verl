# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Replot E41 comparisons from the local aggregated tables.

Semantic ledger: red = original-KL training; teal = stop-aligned training.
Curves are observed values, not smoothed trends. The raw KL metric is identical
across methods; optimization objectives differ and are explicitly labeled.
Geometry ledger: outer/update coordinates are numerical; all four heatmaps
share one linear color scale and equal-width response deciles. Stop-probability
points are 10 fixed diagnostic prefixes, not a random-sample confidence interval.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'data/figure-data.json').read_text())
STYLE=json.loads((ROOT/'data/style.json').read_text())
try:
    import fvstyle as fv
    fv.apply_rc('paper',fontsize=8.5)
    FULL_AUDIT=True
except ImportError:
    # Portable rendering does not require a skill installation.
    plt.rcParams.update(STYLE['rc'])
    fv=SimpleNamespace(HUES=STYLE['hues'],NEUTRALS=STYLE['neutrals'])
    FULL_AUDIT=False
plt.rcParams.update({'axes.labelsize':8.5,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'lines.linewidth':1.35})
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
ARMS=('baseline','aligned');LABEL={'baseline':'Original KL','aligned':'Stop-aligned KL'}
COLOR={'baseline':fv.HUES['vcoral'],'aligned':fv.HUES['vteal']}
AUDIT={}

def select(key,arm):return [r for r in DATA[key] if r['arm']==arm]

def frame(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color=fv.NEUTRALS['vmist'],linestyle='--',linewidth=.6,alpha=.7)
    ax.set_axisbelow(True)

def outer_axis(ax):
    ax.set_xlim(0,21);ax.set_xticks([1,5,10,15,20]);ax.set_xlabel('Outer step')

def legend(fig,handles=None,labels=None,ncol=2,y=.98):
    if handles is None:handles=[Line2D([],[],color=COLOR[a],lw=1.6) for a in ARMS];labels=[LABEL[a] for a in ARMS]
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,y),ncol=ncol,frameon=False)

def pair(height=2.9):
    fig,axes=plt.subplots(1,2,figsize=(5.5,height));fig.subplots_adjust(left=.11,right=.98,bottom=.28,top=.76,wspace=.38)
    for ax in axes:frame(ax)
    return fig,axes

def save(fig,name,caption):
    fig.text(.5,.025,caption,ha='center',va='bottom',fontsize='small',color=fv.NEUTRALS['baseline'])
    fig.canvas.draw()
    if FULL_AUDIT:
        issues=[{'severity':'error' if fv.RENDER_RULES.get(s.partition(' ')[0],'error')=='error' else 'warning','message':s} for s in fv.audit(fig,print_slot='full')]
    else:
        renderer=fig.canvas.get_renderer();issues=[]
        for ax in fig.axes:
            box=ax.get_window_extent(renderer)
            if box.width<=0 or box.height<=0:issues.append({'severity':'error','message':'Empty axes geometry'})
    AUDIT[name]={'auditor':'figure-viz' if FULL_AUDIT else 'portable basic geometry','issues':issues}
    for ext in ('png','svg'):
        target=OUT/f'{name}.{ext}'
        fig.savefig(target,dpi=240,bbox_inches='tight')
        if ext=='svg':target.write_text('\n'.join(line.rstrip() for line in target.read_text().split('\n')))
    plt.close(fig)

# 1. Length: means over 64 responses and the full late-window distribution.
fig,axes=pair()
for arm in ARMS:
    rows=select('steps',arm);axes[0].plot([r['outer'] for r in rows],[r['mean_length']/1000 for r in rows],color=COLOR[arm],marker='.',label=LABEL[arm])
    values=np.sort([r['length']/1000 for r in select('responses',arm) if r['outer']>=16])
    axes[1].step(np.r_[0,values],np.r_[0,np.arange(1,len(values)+1)*100/len(values)],where='post',color=COLOR[arm],label=LABEL[arm])
outer_axis(axes[0]);axes[0].set_ylim(0,7.6);axes[0].set_ylabel('Mean length (k tokens)')
axes[1].set_xlabel('Response length (k tokens)');axes[1].set_ylabel('Cumulative responses (%)');axes[1].set_ylim(-2,102);axes[1].set_xlim(0,7.5)
legend(fig);save(fig,'length','Left: each outer step. Right: all 320 responses from outer steps 16–20.')

# 2. Two different definitions of repetition, same method colors.
fig,axes=pair()
for arm in ARMS:
    rows=select('steps',arm)
    for ax,key in zip(axes,['loop_pct','repeat8_pct']):ax.plot([r['outer'] for r in rows],[r[key] for r in rows],color=COLOR[arm],marker='.',label=LABEL[arm])
for ax in axes:outer_axis(ax);ax.set_ylim(-4,104);ax.set_yticks([0,25,50,75,100])
axes[0].set_ylabel('Responses with long loops (%)');axes[1].set_ylabel('Repeated 8-grams (%)')
legend(fig);save(fig,'repetition','Exact loops: at least 128 tokens and four periods. Repeated n-grams can include valid formatting.')

# 3. Do not silently relabel an altered optimization loss as raw vocabulary KL.
fig,axes=pair()
for arm in ARMS:
    rows=select('updates',arm)
    for ax,key in zip(axes,['raw_kl','optimization_loss']):ax.plot([r['update'] for r in rows],[r[key] for r in rows],color=COLOR[arm],label=LABEL[arm])
for ax in axes:ax.set_xlabel('Optimizer update');ax.set_xlim(0,321);ax.set_xticks([1,80,160,240,320]);ax.set_ylim(0,.59)
axes[0].set_ylabel('Raw vocabulary KL (nats/token)');axes[1].set_ylabel('Optimization loss (nats/token)')
legend(fig);save(fig,'kl-loss','Left: the same raw-KL metric. Right: different optimization objectives. Values precede updates.')

fig,axes=pair(3.0)
ymax=max(r['raw_kl'] for r in DATA['reuse'])*1.08
for ax,window,title in zip(axes,['early','late'],['Outer steps 1–5','Outer steps 16–20']):
    for arm in ARMS:
        rows=sorted([r for r in select('reuse',arm) if r['window']==window],key=lambda r:r['reuse_index'])
        ax.plot([r['reuse_index'] for r in rows],[r['raw_kl'] for r in rows],color=COLOR[arm],marker='.',label=LABEL[arm])
    ax.set_xlabel('Observation within reuse');ax.set_ylabel('Batch-mean raw KL (nats/token)');ax.set_title(title)
    ax.set_xlim(.4,16.6);ax.set_xticks([1,4,8,12,16]);ax.set_ylim(0,ymax)
legend(fig);save(fig,'reuse-kl','Five batches per curve. Prefixes are fixed within each batch; all observations precede updates.')

# 4. A shared heat scale is required for visual comparison across both runs.
fig,axes=plt.subplots(2,2,figsize=(5.5,5.1),sharex=True,sharey=True)
fig.subplots_adjust(left=.13,right=.84,bottom=.18,top=.9,wspace=.2,hspace=.4)
matrices={}
for arm in ARMS:
    for phase in (1,16):
        matrix=np.full((20,10),np.nan)
        for r in select('positions',arm):
            if r['reuse_index']==phase:matrix[r['outer']-1,r['bin']]=r['mean'] if r['mean'] is not None else np.nan
        matrices[arm,phase]=matrix
vmax=max(float(np.nanmax(m)) for m in matrices.values())
cmap=LinearSegmentedColormap.from_list('position-kl',[fv.NEUTRALS['paper'],fv.HUES['vteal']]);cmap.set_bad(fv.NEUTRALS['vmist'])
for row,arm in enumerate(ARMS):
    for col,phase in enumerate((1,16)):
        ax=axes[row,col];im=ax.imshow(np.ma.masked_invalid(matrices[arm,phase]),origin='lower',aspect='auto',extent=(0,100,.5,20.5),cmap=cmap,vmin=0,vmax=vmax,interpolation='nearest')
        ax.set_title(('First' if phase==1 else 'Sixteenth')+' observation')
        ax.set_xticks([0,50,100]);ax.set_yticks([1,5,10,15,20])
        if col==0:ax.set_ylabel(LABEL[arm]+'\nOuter step')
        if row==1:ax.set_xlabel('Response position (%)')
bar=fig.colorbar(im,cax=fig.add_axes([.89,.23,.025,.59]));bar.set_label('Raw KL (nats/token)')
save(fig,'full-kl-position','All panels share one color scale. Bins use each response’s actual length; means are token-weighted.')

# 5. Profiles retain small late-window differences hidden by a global heat scale.
fig,axes=pair(3.15);fig.subplots_adjust(top=.76,bottom=.26)
for ax,window,title in zip(axes,['early','late'],['Outer steps 1–5','Outer steps 16–20']):
    for arm in ARMS:
        for phase,ls,marker in [(1,'-','o'),(16,'--','s')]:
            rows=sorted([r for r in select('profiles',arm) if r['window']==window and r['reuse_index']==phase],key=lambda r:r['bin'])
            ax.plot([5+10*r['bin'] for r in rows],[r['mean'] for r in rows],color=COLOR[arm],linestyle=ls,marker=marker,markersize=3,label=LABEL[arm]+f' · obs {phase}')
    ax.set_title(title);ax.set_xlabel('Response position (%)');ax.set_xlim(0,100);ax.set_xticks([0,50,100]);ax.set_ylim(-.025,.68);ax.set_yticks([0,.2,.4,.6]);ax.set_ylabel('Raw KL (nats/token)')
handles,labels=axes[0].get_legend_handles_labels();legend(fig,handles,labels,ncol=2,y=1.01)
save(fig,'position-profiles','Solid: first observation; dashed: sixteenth. Five batches are pooled by token sums and counts.')

# 6. Ending behavior and paired fixed-prefix probabilities.
fig,axes=pair(3.0)
for arm in ARMS:
    rows=select('steps',arm);axes[0].plot([r['outer'] for r in rows],[r['cap_pct'] for r in rows],color=COLOR[arm],marker='.',label=LABEL[arm])
outer_axis(axes[0]);axes[0].set_ylabel('Responses reaching length cap (%)');axes[0].set_ylim(-4,104)
states=['initial','baseline','aligned','teacher'];colors=[fv.NEUTRALS['baseline'],COLOR['baseline'],COLOR['aligned'],fv.HUES['vgold']];state_handles=[]
for i,(state,color) in enumerate(zip(states,colors)):
    rows=sorted([r for r in DATA['stops'] if r['model']==state],key=lambda r:r['uid']);y=np.array([r['probability']*100 for r in rows])
    state_handles.append(axes[1].scatter(i+np.linspace(-.12,.12,len(y)),y,color=color,s=15,zorder=3))
    line=axes[1].plot([i-.2,i+.2],[np.median(y)]*2,color=fv.NEUTRALS['vink'],lw=1.3)[0]
    if FULL_AUDIT:fv.mark_neutral(line)
axes[1].set_yscale('log');axes[1].set_ylim(1e-10,300);axes[1].set_yticks([1e-9,1e-6,1e-3,1,100]);axes[1].set_xticks(range(4),['Initial','Original','Aligned','Teacher']);axes[1].tick_params(axis='x',labelsize=7.5)
axes[1].set_xlabel('Model on identical ending prefixes');axes[1].set_ylabel('Total stop probability (%)');axes[1].set_xlim(-.5,3.5)
legend(fig,state_handles,['Initial student','Original KL','Stop-aligned KL','Teacher'],ncol=4)
save(fig,'termination','Right: 10 fixed diagnostic prefixes; dots are cases, horizontal marks are medians. No confidence intervals.')

# 7. Optimization health and the deliberately limited training-score evidence.
fig,axes=pair()
for arm in ARMS:
    u=select('updates',arm);s=select('steps',arm)
    axes[0].plot([r['update'] for r in u],[r['grad_norm'] for r in u],color=COLOR[arm],label=LABEL[arm])
    axes[1].plot([r['outer'] for r in s],[r['score_pct'] for r in s],color=COLOR[arm],marker='.',label=LABEL[arm])
ref=axes[0].axhline(1,color=fv.NEUTRALS['baseline'],ls=':',lw=.8)
if FULL_AUDIT:fv.mark_neutral(ref)
axes[0].set_yscale('log');axes[0].set_ylabel('Pre-clip gradient norm');axes[0].set_xlabel('Optimizer update');axes[0].set_xlim(0,321);axes[0].set_xticks([1,80,160,240,320])
outer_axis(axes[1]);axes[1].set_ylabel('Training-problem score (%)');axes[1].set_ylim(-5,105);axes[1].set_yticks([0,25,50,75,100])
legend(fig);save(fig,'training-health','Gradient clipping threshold: 1. Scores concern the training questions, not independent validation.')

# 8. Prefix intervention: longer exact repetitions can make the teacher agree with loops.
fig,axes=pair(3.0);cycles=np.array([0,1,4,16,64]);x=np.log1p(cycles)
for arm in ARMS:
    rows=sorted(select('loop_prefix',arm),key=lambda r:r['cycles'])
    axes[0].plot(x,[100*r['student_next_median'] for r in rows],color=COLOR[arm],marker='.',label=LABEL[arm])
    axes[1].plot(x,[r['raw_kl_median'] for r in rows],color=COLOR[arm],marker='.',label=LABEL[arm])
rows=sorted(select('loop_prefix','baseline'),key=lambda r:r['cycles'])
axes[0].plot(x,[100*r['teacher_next_median'] for r in rows],color=fv.HUES['vgold'],marker='.',label='Teacher')
for ax in axes:ax.set_xticks(x,[str(i) for i in cycles]);ax.set_xlim(-.15,x[-1]+.15);ax.set_xlabel('Repeated cycles (log1p spacing)')
axes[0].set_ylim(-4,104);axes[0].set_ylabel('Probability of next loop token (%)')
axes[1].set_yscale('log');axes[1].set_ylabel('Raw KL (nats/token)')
handles,labels=axes[0].get_legend_handles_labels();legend(fig,handles,labels,ncol=3)
save(fig,'loop-prefix','Medians over the same 10 loop prefixes. This tests forced prefixes, not spontaneous loop frequency.')

(OUT/'audit.json').write_text(json.dumps(AUDIT,ensure_ascii=False,indent=2)+'\n')
errors=[issue for record in AUDIT.values() for issue in record['issues'] if issue['severity']=='error']
print(json.dumps(AUDIT,ensure_ascii=False))
if errors:raise SystemExit(1)
