#!/usr/bin/env python3
"""Summaries on measures fixed independently of training-source choices.

Uncertainty uses paired prompt bootstrap, with the two fresh-generation seeds
averaged within each prompt. It is not uncertainty across training seeds.
"""
import argparse
import collections
import json
import random
import statistics as st
from pathlib import Path


def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def ci(vals):
    rng=random.Random(42);b=sorted(st.mean(rng.choices(vals,k=len(vals))) for _ in range(1000));return [b[24],b[974]]


def fixed(rows):
    out={}
    for r in rows:
        for band,v in r['stats'].items():
            key=(r['prompt_id'],r['domain'],r['source'],band)
            out[key]={'horizon_rkl':v['rkl']/1024,'token_rkl':v['rkl']/v['alive'] if v['alive'] else None,
                      'horizon_js':v['js']/1024,'horizon_tv':v['tv']/1024,'token_tv':v['tv']/v['alive'] if v['alive'] else None,
                      'overlap16':v['overlap16']/v['alive'] if v['alive'] else None,'overlap64':v['overlap64']/v['alive'] if v['alive'] else None,
                      'length':v['alive']}
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args();root=a.data
    summary=[];fresh=[];counts=[];training=[];tasks=[]
    folders=list(root.glob('*-*-k16'))+[f for f in root.glob('selection-*') if f.is_dir()]
    for folder in sorted(folders):
        if not (folder/'protocol.json').exists():continue
        cfg=read(folder/'protocol.json');teacher=cfg['teacher'];baseline=root/f'baseline-{teacher}'
        if not (baseline/'done').exists():continue
        base=fixed(read(baseline/'fixed.json'))
        for f in sorted(folder.glob('fixed-*.json')):
            rows=read(f);step=rows[0]['step'];current=fixed(rows)
            assert set(base)==set(current)
            for domain in ['math','code','if']:
                for source in ['student','justrl','polaris','balanced']:
                    for band in ['all','early','middle','late']:
                        keys=[k for k in base if k[1]==domain and k[3]==band and (source=='balanced' or k[2]==source)]
                        if not keys:continue
                        before=collections.defaultdict(list);after=collections.defaultdict(list)
                        for k in keys:
                            before[k[0]].append(base[k]['horizon_rkl']);after[k[0]].append(current[k]['horizon_rkl'])
                        bs=[st.mean(before[k]) for k in sorted(before)];ats=[st.mean(after[k]) for k in sorted(before)]
                        delta=[b-c for b,c in zip(bs,ats)]
                        valid=[k for k in keys if base[k]['length']>0]
                        summary.append(dict(arm=folder.name,teacher=teacher,step=step,domain=domain,source=source,band=band,
                            prompts=len(bs),before=st.mean(bs),after=st.mean(ats),drop=1-st.mean(ats)/st.mean(bs) if st.mean(bs) else None,
                            absolute_drop_ci=ci(delta),improved_fraction=st.mean(d>0 for d in delta),
                            js_before=st.mean(base[k]['horizon_js'] for k in keys),js_after=st.mean(current[k]['horizon_js'] for k in keys),
                            tv_before=st.mean(base[k]['horizon_tv'] for k in keys),tv_after=st.mean(current[k]['horizon_tv'] for k in keys),
                            overlap16_before=st.mean(base[k]['overlap16'] for k in valid) if valid else None,
                            overlap16_after=st.mean(current[k]['overlap16'] for k in valid) if valid else None))
        initial=read(baseline/'fresh.json');bi={(r['id'],r['seed']):r for r in initial}
        for f in sorted(folder.glob('fresh-*.json')):
            rs=read(f);step=rs[0]['step'];by=collections.defaultdict(list);bj=collections.defaultdict(list)
            if len(rs)!=64:continue  # the second generation seed may still be running
            for r in rs:
                by[r['id']].append(r['stats']['all']['rkl']);bj[r['id']].append(bi[(r['id'],r['seed'])]['stats']['all']['rkl'])
            delta=[st.mean(bj[k])-st.mean(by[k]) for k in sorted(by)]
            fresh.append(dict(arm=folder.name,teacher=teacher,step=step,prompts=len(by),responses=len(rs),
                sequence_kl_before=st.mean(r['stats']['all']['rkl'] for r in initial),
                sequence_kl_after=st.mean(r['stats']['all']['rkl'] for r in rs),
                conditional_js_horizon_after=st.mean(r['stats']['all']['js']/1024 for r in rs),
                conditional_tv_horizon_after=st.mean(r['stats']['all']['tv']/1024 for r in rs),
                sample_gap_after=st.mean(r['stats']['all']['sample_gap'] for r in rs),
                conditional_minus_sample_gap_ci=ci([st.mean(x['stats']['all']['rkl']-x['stats']['all']['sample_gap'] for x in rs if x['id']==k) for k in by]),
                absolute_drop_ci=ci(delta),length_before=st.mean(len(r['response_ids']) for r in initial),
                length_after=st.mean(len(r['response_ids']) for r in rs),stop_fraction=st.mean(r['finish_reason']=='stop' for r in rs)))
        for f in sorted(folder.glob('rollout-*.json')):
            rs=read(f);outer=int(f.stem.split('-')[-1]);n=sum(len(r['response_ids']) for r in rs)
            for k in [8,16,64]:
                counts.append(dict(arm=folder.name,outer=outer,k=k,tokens=n,
                    category_counts=[sum(r['stats']['all'][f'category{k}_{c}'] for r in rs) for c in range(4)],
                    mass_p=[sum(r['stats']['all'][f'mass_p{k}_{c}'] for r in rs)/n for c in range(4)],
                    mass_q=[sum(r['stats']['all'][f'mass_q{k}_{c}'] for r in rs)/n for c in range(4)],
                    overlap=sum(r['stats']['all'][f'overlap{k}'] for r in rs)/n,
                    rejected_proposals=sum(r['proposal_rejections'] for r in rs),empty_intersections=sum(r['empty_intersections'] for r in rs)))
        if (folder/'training.json').exists():
            tr=read(folder/'training.json');training.append(dict(arm=folder.name,updates=len(tr),clip_count=sum(r['grad_norm']>1 for r in tr),
                last_kl=tr[-1].get('pre_update_kl',tr[-1].get('loss')),max_grad=max(r['grad_norm'] for r in tr),
                peak_gib=max((r['peak_gib'] for r in tr if 'peak_gib' in r),default=None)))
    for f in root.glob('task-*.json'):
        if f.name in ['task-prompts.json','task-paired-contrasts.json']:continue
        rows=read(f)
        if len(rows)!=32 or not all('grade' in r for r in rows):continue
        tasks.append(dict(arm=f.stem[5:],n=len(rows),correct=sum(bool(r['grade']['acc']) for r in rows),
            stop=sum(r['finish_reason']=='stop' for r in rows),mean_length=st.mean(len(r['response_ids']) for r in rows),
            prediction_missing=sum(not r['grade'].get('pred') for r in rows)))
    dump(root/'summary-fixed.json',summary);dump(root/'summary-fresh.json',fresh);dump(root/'summary-categories.json',counts)
    dump(root/'summary-training.json',training);dump(root/'summary-task.json',tasks)
    print(json.dumps(training,indent=2))
    for r in summary:
        if r['step']==64 and r['source']=='balanced' and r['band']=='all':print(r['arm'],r['domain'],'fixed drop',round(100*r['drop'],2))
    for r in fresh:print(r['arm'],r['step'],'fresh J',round(r['sequence_kl_before'],3),'->',round(r['sequence_kl_after'],3))


if __name__=='__main__':main()
