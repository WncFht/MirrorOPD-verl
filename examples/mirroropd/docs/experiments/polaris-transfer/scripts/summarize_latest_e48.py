#!/usr/bin/env python3
"""Summarize frozen completed E48/E20 history without older report inputs."""
import argparse
import collections
import json
import statistics as st
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args();out={}
    for name,folder in [('E48','run'),('E20','E20')]:
        rows=[json.loads(s) for s in (a.data/folder/'history.jsonl').open() if s.strip()]
        updates=[r for r in rows if 'distill/full_kl' in r];outer=[r for r in rows if 'rollout/response_tokens' in r]
        groups=collections.defaultdict(list)
        for r in updates:groups[r['step/outer']].append(r)
        assert len(updates)==4096 and len(groups)==256 and [r['step/update'] for r in updates]==list(range(1,4097))
        windows=[]
        for lo in range(1,257,32):
            batches=[groups[i] for i in range(lo,lo+32)];rs=[r for r in outer if lo<=r['step/outer']<lo+32]
            w=dict(first_outer=lo,last_outer=lo+31,first_kl=st.mean(b[0]['distill/full_kl'] for b in batches),
                last_kl=st.mean(b[-1]['distill/full_kl'] for b in batches),within_drop=st.mean(1-b[-1]['distill/full_kl']/b[0]['distill/full_kl'] for b in batches),
                gradient_mean=st.mean(r['optim/actor/grad_norm_preclip'] for b in batches for r in b),
                response_length=st.mean(r['rollout/response_length/mean'] for r in rs),score=st.mean(r['rollout/score/mean'] for r in rs),
                length_stop_fraction=sum(r['rollout/finish_reason/length_count'] for r in rs)/sum(r['rollout/response_count'] for r in rs))
            if name=='E48':
                for side,j in [('first',0),('last',-1)]:
                    w['positions_'+side]=[sum(b[j][f'distill/full_kl_position/p{i:02d}_{i+10:02d}/sum'] for b in batches)/sum(b[j][f'distill/full_kl_position/p{i:02d}_{i+10:02d}/token_count'] for b in batches) for i in range(0,100,10)]
            windows.append(w)
        ev=[{k:v for k,v in r.items() if k.startswith(('step/','eval/'))} for r in rows if 'eval/math500/acc/mean' in r]
        out[name]=dict(history=len(rows),updates=len(updates),outer=len(groups),windows=windows,evals=ev,
            nonfinite=sum(r['optim/actor/loss_nonfinite']+r['optim/actor/grad_norm_preclip_nonfinite'] for r in updates),
            skipped=sum(r['optim/actor/update_skipped'] for r in updates),clipped=sum(r['optim/actor/grad_norm_preclip']>1 for r in updates),
            within_decreases=sum(y['distill/full_kl']<x['distill/full_kl'] for b in groups.values() for x,y in zip(b,b[1:])),
            learning_rates=sorted({r['optim/actor/lr'] for r in updates}),
            batch_curve=[dict(outer=k,first=b[0]['distill/full_kl'],last=b[-1]['distill/full_kl']) for k,b in groups.items()])
    (a.data/'latest-run-summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print({k:(v['outer'],v['updates'],v['windows'][-1]['first_kl']) for k,v in out.items()})


if __name__=='__main__':main()
