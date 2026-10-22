import csv,json,re,statistics,subprocess
from pathlib import Path
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2
root=next(p for p in Path(__file__).resolve().parents if (p/"verl/trainer/main_ppo.py").is_file());P=root/"outputs/rollout-tuning-20261022";results={}
for p in P.glob('rollout-tuning-train-*'):
 if not (p/'exit.json').exists() or json.loads((p/'exit.json').read_text())['exit_code']!=0:continue
 log=(p/'launch.log').read_text();rid=re.findall(r'offline-run-\d{8}_\d{6}-([a-z0-9]+)',log)[0]
 wp=next((root/'wandb').glob(f'offline-run-*-{rid}/run-{rid}.wandb'));store=DataStore();store.open_for_scan(str(wp));rows=[]
 while True:
  data=store.scan_data()
  if data is None:break
  r=wandb_internal_pb2.Record();r.ParseFromString(data)
  if r.HasField('history'):rows.append({i.key or '/'.join(i.nested_key):json.loads(i.value_json) for i in r.history.item})
 (p/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 outer=[r for r in rows if 'perf/timing/outer_step_seconds' in r];updates=[r for r in rows if 'optim/actor/grad_norm_preclip' in r]
 assert len(outer)==4 and len(updates)==64,(len(outer),len(updates))
 assert all(r['step/update']==r['step/optimizer']==i+1 for i,r in enumerate(updates))
 steady=outer[1:];keys=[k for k in steady[0] if k.startswith(('perf/','rollout/'))]
 summary={k:statistics.mean(r[k] for r in steady if k in r) for k in keys}
 summary['response_tokens_per_second']=sum(r['rollout/response_tokens'] for r in steady)/sum(r['perf/timing/outer_step_seconds'] for r in steady)
 summary['grad_norm_max']=max(r['optim/actor/grad_norm_preclip'] for r in updates)
 summary['nonfinite_max']=max(r.get('policy/current_behavior/nonfinite_fraction',0) for r in updates)
 summary['wall_seconds']=json.loads((p/'exit.json').read_text())['wall_seconds'];summary['run_id']=rid
 summary['outer_steps']=len(outer);summary['updates']=len(updates);results[p.name]=summary
 print(p.name,json.dumps(summary,indent=2))
(P/'training-analysis.json').write_text(json.dumps(results,indent=2))
