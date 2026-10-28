import csv,gzip,json,re,statistics,math
from pathlib import Path
from collections import Counter
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2
from omegaconf import OmegaConf
root=Path('/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/MirrorOPD-verl');base=root/'outputs/rollout-tuning-20261022';result={'runs':{}}
for p in sorted(base.glob('valid-*')):
 if not (p/'exit.json').exists() or json.loads((p/'exit.json').read_text())['exit_code']!=0:continue
 log=(p/'launch.log').read_text();rid=re.findall(r'offline-run-\d{8}_\d{6}-([a-z0-9]+)',log)[0]
 wp=next((root/'wandb').glob(f'offline-run-*-{rid}/run-{rid}.wandb'));store=DataStore();store.open_for_scan(str(wp));rows=[]
 while True:
  data=store.scan_data()
  if data is None:break
  r=wandb_internal_pb2.Record();r.ParseFromString(data)
  if r.HasField('history'):rows.append({i.key or '/'.join(i.nested_key):json.loads(i.value_json) for i in r.history.item})
 (p/'history.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 rows=[r for r in rows if 'perf/timing/validation_seconds' in r];assert len(rows)==1
 row=rows[0];seconds=row['perf/timing/validation_seconds'];end=row['_timestamp'];start=end-seconds
 outputs=[json.loads(l) for l in (p/'training/validation/0.jsonl').read_text().splitlines()]
 groups=Counter(x['uid'].rsplit('_',2)[0] for x in outputs);assert len(outputs)==2048 and len(groups)==128 and set(groups.values())=={16}
 metrics=[json.loads(l) for l in (p/'engine-metrics.jsonl').read_text().splitlines()];latest={r['address']:r for r in metrics if r.get('metrics')};assert len(latest)==4
 sums={k:sum(r['metrics'][k] for r in latest.values()) if all(k in r['metrics'] for r in latest.values()) else None for k in ['vllm:request_success_total','vllm:generation_tokens_total','vllm:request_generation_tokens_sum','vllm:num_preemptions_total']}
 assert sums['vllm:request_success_total']==2048,sums
 assert sums['vllm:generation_tokens_total']==sums['vllm:request_generation_tokens_sum'],sums
 gpu=[r for r in csv.DictReader((p/'gpu.csv').open()) if start+1<=float(r['timestamp'])<=end-1]
 cfg=OmegaConf.to_container(OmegaConf.load(p/'training/resolved_config.yaml'),resolve=True)
 r={'run_id':rid,'testing_seconds':seconds,'rollout_span_seconds':row.get('perf/timing/validation_rollout_seconds'),'tokens':sums['vllm:generation_tokens_total'],'output_tokens_per_second':sums['vllm:generation_tokens_total']/seconds,'responses':len(outputs),'prompt_groups':len(groups),'responses_per_group':16,'score_mean':statistics.mean(x['score'] for x in outputs),'quality_metrics':{k:v for k,v in row.items() if k.startswith('eval/')},'engine_counters':sums,'max_engine_waiting':max(x['metrics'].get('vllm:num_requests_waiting',0) for x in metrics if x.get('metrics')),'max_kv_fraction':max(x['metrics'].get('vllm:kv_cache_usage_perc',0) for x in metrics if x.get('metrics')),'gpu_busy_mean':statistics.mean(float(x['busy_percent']) for x in gpu),'gpu_memory_mean_gib':statistics.mean(float(x['memory_gib']) for x in gpu),'gpu_memory_max_gib':max(float(x['memory_gib']) for x in gpu),'gpu_probe_rows':len(gpu),'config':cfg,'history':row}
 # Each monitor round has one sample from each of the four TP1 HTTP servers.
 rounds=[]
 for i in range(0,len(metrics)-3,4):
  block=metrics[i:i+4]
  if len({x['address'] for x in block})!=4 or not all(x.get('metrics') for x in block):continue
  rounds.append((max(x['time'] for x in block),sum(x['metrics'].get('vllm:request_success_total',0) for x in block)))
 window=cfg['data']['val_batch_size']*16;tails=[]
 for upper in range(window,2049,window):
  threshold=upper-window+math.ceil(.9*window)
  t90=next(t for t,n in rounds if n>=threshold);t100=next(t for t,n in rounds if n>=upper)
  tails.append({'window_responses':window,'completion_90_to_100_seconds':t100-t90})
 r['decode_tail_samples']=tails
 active_rounds=[]
 for i in range(0,len(metrics)-3,4):
  block=metrics[i:i+4]
  if len({x['address'] for x in block})!=4 or not all(x.get('metrics') for x in block):continue
  if not start<=max(x['time'] for x in block)<=end:continue
  active=[x['metrics'].get('vllm:num_requests_running',0) for x in block]
  waiting=[x['metrics'].get('vllm:num_requests_waiting',0) for x in block]
  active_rounds.append({'active':sum(active),'waiting':sum(waiting),'any_idle':any(x==0 for x in active)})
 r['queue_samples']={'count':len(active_rounds),'active_mean':statistics.mean(x['active'] for x in active_rounds),'fraction_all_waiting_empty':statistics.mean(x['waiting']==0 for x in active_rounds),'fraction_active_lt16':statistics.mean(x['active']<16 for x in active_rounds),'fraction_any_idle':statistics.mean(x['any_idle'] for x in active_rounds)}
 r['wall_seconds']=json.loads((p/'exit.json').read_text())['wall_seconds']
 result['runs'][p.name]=r
 print(p.name,{k:v for k,v in r.items() if k in ['testing_seconds','tokens','output_tokens_per_second','gpu_busy_mean','gpu_memory_max_gib','max_kv_fraction','max_engine_waiting','score_mean','decode_tail_samples']})
(base/'analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
