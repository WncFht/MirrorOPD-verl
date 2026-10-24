#!/usr/bin/env python3
"""Audit E16/E17 scalar convergence; no token-position reconstruction.

Ledger: each update point is a pre-optimizer full-KL measurement; one outer
batch fixes the student-generated prefixes. First/last curves are distinct
within-batch observations, not two held-out evaluations. One color per run;
line styles distinguish first versus last. Rolling means use 20 observed
outer batches. The reuse panel normalizes each batch by its own first loss.
"""
import argparse
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def analyze(path, label, expected_epochs):
    rows = [json.loads(line) for line in (path / 'history.jsonl').read_text().splitlines()]
    defs = [json.loads(line) for line in (path / 'metric-definitions.jsonl').read_text().splitlines()]
    assert any(d.get('name') == 'distill/full_kl' and d.get('step_metric') == 'step/update' for d in defs)
    keys = sorted({key for row in rows for key in row})
    updates, grouped, outer = [], defaultdict(list), {}
    for r in rows:
        if 'rollout/response_length/mean' in r:
            outer[int(r['step/outer'])] = r
        if 'distill/full_kl' not in r:
            continue
        assert r['optim/actor/update_applied'] == 1 and r['optim/actor/update_skipped'] == 0
        u = {'run': label, 'outer': int(r['step/outer']), 'update': int(r['step/update']),
             'epoch_index': int(r['rollout/data_reuse/epoch_index']),
             'full_kl': r['distill/full_kl'], 'actor_loss': r['optim/actor/loss'],
             'grad_norm': r['optim/actor/grad_norm_preclip']}
        assert math.isclose(u['full_kl'], u['actor_loss'], rel_tol=1e-5, abs_tol=1e-8)
        updates.append(u)
        grouped[u['outer']].append(u)
    assert len({u['update'] for u in updates}) == len(updates)
    batches = []
    for step, us in sorted(grouped.items()):
        us.sort(key=lambda u: u['update'])
        assert [u['epoch_index'] for u in us] == list(range(expected_epochs))
        first, last = us[0]['full_kl'], us[-1]['full_kl']
        o = outer.get(step, {})
        batches.append({'run': label, 'outer': step, 'first_update': us[0]['update'],
                        'last_update': us[-1]['update'], 'first_kl': first, 'last_kl': last,
                        'within_drop_pct': 100 * (1 - last / first),
                        'last_transition_drop_pct': 100 * (1 - last / us[-2]['full_kl']),
                        'last_grad_norm': us[-1]['grad_norm'],
                        'response_length': o.get('rollout/response_length/mean'),
                        'length_stop_fraction': (o['rollout/finish_reason/length_count'] / o['rollout/response_count']) if o else None})
    def window(lo, hi):
        bs = [b for b in batches if lo <= b['outer'] <= hi]
        result = {'first_outer': lo, 'last_outer': hi, 'batches': len(bs)}
        for k in ['first_kl', 'last_kl', 'within_drop_pct', 'last_transition_drop_pct', 'last_grad_norm', 'response_length', 'length_stop_fraction']:
            vs = [b[k] for b in bs if b[k] is not None]
            result[k + '_mean'] = st.mean(vs) if vs else None
        ratios = [grouped[b['outer'] + 1][0]['full_kl'] / b['last_kl'] for b in bs if b['outer'] + 1 in grouped]
        result['next_batch_first_over_previous_last_mean'] = st.mean(ratios) if ratios else None
        result['normalized_reuse_profile'] = [st.mean(grouped[b['outer']][i]['full_kl'] / b['first_kl'] for b in bs) for i in range(expected_epochs)]
        return result
    end = batches[-1]['outer']
    windows = {f'{lo}-{hi}': window(lo, hi) for lo, hi in [(1,20),(21,40),(81,100),(121,140),(141,160),(end-39,end-20),(end-19,end)]}
    evaluation = [{k: v for k, v in r.items() if k == 'step/outer' or k in ['eval/math500/acc/mean','eval/amc23/acc/mean','eval/aime25/acc/mean']} for r in rows if 'eval/math500/acc/mean' in r]
    result = {'source': json.load(open(path/'source.json')), 'epochs': expected_epochs,
              'history_events': len(rows), 'metric_count': len(keys), 'update_count': len(updates),
              'last_update_outer': end, 'last_complete_outer': max(outer),
              'last_point': updates[-1], 'windows': windows, 'evaluation': evaluation,
              'training_kl_keys': [k for k in keys if k.startswith('distill/')],
              'loss_comparison_max_abs_error': max(abs(u['full_kl']-u['actor_loss']) for u in updates)}
    return result, batches, updates, keys


def plot(data, out):
    import fvlocate
    fvlocate.ensure()
    import fvstyle as fv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fv.apply_rc('paper', fontsize=10)
    colors = dict(zip(data, fv.series_colors('paper', len(data))))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color=fv.NEUTRALS['vmist'], lw=.6)
    def rolling(values):
        return [st.mean(values[max(0,i-19):i+1]) for i in range(len(values))]
    for label, (summary, batches) in data.items():
        xs = [b['outer'] for b in batches]
        for k, style, desc in [('first_kl','-','first'),('last_kl','--','last')]:
            ys = rolling([b[k] for b in batches])
            for ax in axes[:2]:
                valid = [(x,y) for x,y in zip(xs,ys) if ax is axes[0] or x >= 100]
                ax.plot(*zip(*valid), color=colors[label], ls=style, lw=1.8, label=f'{label} {desc}')
        profile = summary['windows']['141-160']['normalized_reuse_profile']
        axes[2].plot(range(1,len(profile)+1), profile, color=colors[label], marker='o', markersize=3, label=label)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Outer batch')
    axes[0].set_ylabel('Full KL (nats/token, log scale)')
    axes[0].set_title('Whole run: 20-batch rolling mean')
    axes[1].set_xlabel('Outer batch (late window)')
    axes[1].set_ylabel('Full KL (nats/token)')
    axes[1].set_title('Late training: 20-batch rolling mean')
    axes[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    axes[2].set_xlabel('Within-batch observation index')
    axes[2].set_ylabel('KL / first observed KL')
    axes[2].set_title('Reuse: outer batches 141-160')
    axes[2].set_xticks([1,4,8,12,16])
    axes[2].legend(loc='upper right', frameon=False, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False)
    fig.tight_layout(rect=(0,.12,1,.85),w_pad=2.5)
    fig.text(.5,.025,'All KL measurements precede optimizer updates. E17 is a partial snapshot through batch 178.',ha='center',fontsize=9,color=fv.NEUTRALS['baseline'])
    fig.canvas.draw()
    issues=fv.audit(fig,print_slot=fig.get_figwidth())
    (out/'convergence.audit.txt').write_text('\n'.join(fv.format_audit_lines(issues))+'\n' if issues else 'PASS\n')
    if any(fv.problem_severity(i)=='error' for i in issues):
        raise RuntimeError(fv.format_audit_lines(issues))
    for ext in ['png','pdf']:
        fig.savefig(out/f'convergence.{ext}',dpi=170,facecolor=fv.NEUTRALS['paper'])
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--e16',type=Path,required=True)
    p.add_argument('--e17',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--plot',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    bundle={};plotdata={};all_batches=[];all_updates=[]
    for label,path,epochs in [('E16',args.e16,4),('E17',args.e17,16)]:
        summary,batches,updates,keys=analyze(path,label,epochs)
        bundle[label]=summary;plotdata[label]=(summary,batches)
        all_batches.extend(batches);all_updates.extend(updates)
        (args.out/f'{label}-recorded-keys.txt').write_text('\n'.join(keys)+'\n')
    dump(args.out/'analysis.json',bundle)
    write_csv(args.out/'batches.csv',all_batches);write_csv(args.out/'updates.csv',all_updates)
    if args.plot:plot(plotdata,args.out)
    print(json.dumps({k:{'last':v['last_point'],'windows':v['windows']} for k,v in bundle.items()},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
