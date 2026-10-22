import json
from pathlib import Path
P=next(p for p in Path(__file__).resolve().parents if (p/"verl/trainer/main_ppo.py").is_file())/"outputs/rollout-tuning-20261022"
runs=json.loads((P/'analysis.json').read_text())['runs']
def flat(d,p=''):
 result={}
 for k,v in d.items():
  kk=p+'.'+k if p else k
  if isinstance(v,dict):result.update(flat(v,kk))
  else:result[kk]=v
 return result
base=flat(runs['valid-w32-s32-20261022-a']['config']);allowed={'data.val_batch_size','actor_rollout_ref.rollout.max_num_seqs','actor_rollout_ref.rollout.cudagraph_capture_sizes','actor_rollout_ref.rollout.trace.experiment_name','trainer.experiment_name','trainer.default_local_dir','trainer.validation_data_dir','ray_kwargs.ray_init._temp_dir'}
out={}
for name,run in runs.items():
 cfg=flat(run['config']);diff={k:[base.get(k),cfg.get(k)] for k in base.keys()|cfg.keys() if base.get(k)!=cfg.get(k)}
 extra=set(diff)-allowed
 if extra:raise AssertionError((name,extra))
 assert run['responses']==2048 and run['prompt_groups']==128 and run['responses_per_group']==16
 assert all(v==0 for k,v in run['quality_metrics'].items() if k.endswith('failed_count'))
 out[name]=diff
(P/'config-differences.json').write_text(json.dumps(out,indent=2))
print('PASS',len(runs),'runs: expected config differences, complete responses, zero failed scoring')
