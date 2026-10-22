import json,collections
from pathlib import Path
import numpy as np
P=next(p for p in Path(__file__).resolve().parents if (p/"verl/trainer/main_ppo.py").is_file())/"outputs/rollout-tuning-20261022"
files={p.parent.parent.parent.name:p for p in P.glob('valid-*/training/validation/0.jsonl') if (p.parent.parent.parent/'exit.json').exists()}
groups={}
for name,p in files.items():
 d=collections.defaultdict(list)
 for l in p.read_text().splitlines():
  r=json.loads(l);d[r['input']].append(r['score'])
 assert len(d)==128 and all(len(v)==16 for v in d.values()),(name,len(d))
 groups[name]={k:np.mean(v) for k,v in d.items()}
base=groups['valid-w32-s32-20261022-a'];keys=sorted(base);rng=np.random.default_rng(20261022);idx=rng.integers(0,len(keys),(10000,len(keys)));out={}
for name,d in groups.items():
 assert set(d)==set(base)
 delta=np.array([d[k]-base[k] for k in keys]);boot=delta[idx].mean(axis=1)
 out[name]={'mean_score':float(np.mean(list(d.values()))),'paired_difference_pp':float(delta.mean()*100),'prompt_cluster_bootstrap_95_percentile_pp':(np.quantile(boot,[.025,.975])*100).tolist(),'prompts':128,'responses_per_prompt':16}
(P/'quality-paired-data.json').write_text(json.dumps({'prompt_inputs':keys,'run_prompt_scores':{name:[float(d[k]) for k in keys] for name,d in groups.items()}},ensure_ascii=False))
(P/'quality-check.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
