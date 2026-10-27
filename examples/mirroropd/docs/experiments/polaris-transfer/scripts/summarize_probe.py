#!/usr/bin/env python3
"""Summarize fixed probes and teacher-only temperature counterfactuals.

Temperatures change the objective; reduced initial KL is not evidence that
training or task accuracy improves. Logits are optional reproducible caches.
"""
import argparse
import json
import statistics as st
from pathlib import Path
import torch


def main():
    a=argparse.ArgumentParser();a.add_argument('--probe',type=Path,required=True);a.add_argument('--cache',type=Path,required=True);args=a.parse_args();torch.set_num_threads(4)
    rows=json.loads((args.probe/'results.json').read_text());out={}
    student=torch.load(args.cache/'DeepSeek-R1-Distill-Qwen-1.5B.pt',weights_only=True)
    for name in ['JustRL-DeepSeek-1.5B','Polaris-7B-Preview']:
        rs=[r for r in rows if r['teacher']==name]
        out[name]={'means':{k:st.mean(r[k] for r in rs) for k in rs[0] if isinstance(rs[0][k],(int,float,bool)) and k not in ('case','position','length_bin','student_top1','teacher_top1')},
                   'count':len(rs),'max_support_floor':max(r['support_floor'] for r in rs)}
        top=sorted(rs,key=lambda r:r['kl'],reverse=True)[:31]
        out[name]['largest_31_positions_kl_fraction']=sum(r['kl'] for r in top)/sum(r['kl'] for r in rs)
        out[name]['top1_agreement_positions_kl_fraction']=sum(r['kl'] for r in rs if r['top1_agree'])/sum(r['kl'] for r in rs)
        teacher=torch.load(args.cache/(name+'.pt'),weights_only=True)
        temps=[]
        for temp in [1.,1.25,1.5,2.]:
            kls=[];ent=[];tailgrad=0.;totalgrad=0.
            for sl,tl in zip(student,teacher):
                lp=sl.log_softmax(-1);p=lp.exp();lt=(tl/temp).log_softmax(-1)
                kl=(p*(lp-lt[:,:p.shape[-1]])).sum(-1);kls.extend(kl.tolist());ent.extend((-(lt.exp()*lt).sum(-1)).tolist())
                g=p*(lp-lt[:,:p.shape[-1]]-kl[:,None]);mask=torch.ones_like(p,dtype=torch.bool).scatter_(1,p.topk(32,-1).indices,False)
                tailgrad+=float(g[mask].abs().sum());totalgrad+=float(g.abs().sum())
            temps.append(dict(teacher_temperature=temp,mean_kl=st.mean(kls),mean_teacher_entropy=st.mean(ent),
                              pooled_tail32_abs_logit_gradient_fraction=tailgrad/totalgrad))
        out[name]['temperature_counterfactuals']=temps
    (args.probe/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
