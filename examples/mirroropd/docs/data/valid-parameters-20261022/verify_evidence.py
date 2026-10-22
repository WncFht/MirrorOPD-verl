"""Recompute validation counters, timer and NVML peak from archived samples."""
import csv,gzip,json,math,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent
runs=json.loads((P/'analysis.json').read_text())['runs']
previous=json.loads((P/'previous-full-analysis.json').read_text())['runs']
for name,r in {**runs,**previous}.items():
 d=P/'runs'/name
 assert json.loads((d/'exit.json').read_text())['exit_code']==0
 rows=[json.loads(l) for l in (d/'history.jsonl').read_text().splitlines()]
 rows=[x for x in rows if 'perf/timing/validation_seconds' in x];assert len(rows)==1
 h=rows[0];seconds=h['perf/timing/validation_seconds'];assert math.isclose(seconds,r['testing_seconds'])
 latest={}
 with gzip.open(d/'engine-metrics.jsonl.gz','rt') as f:
  for l in f:
   x=json.loads(l)
   if x.get('metrics'):latest[x['address']]=x['metrics']
 assert len(latest)==4
 counters={k:sum(x[k] for x in latest.values()) for k in ['vllm:request_success_total','vllm:generation_tokens_total','vllm:request_generation_tokens_sum','vllm:num_preemptions_total']}
 assert counters==r['engine_counters']
 assert counters['vllm:request_success_total']==r['prompt_groups']*16==r['responses']
 assert counters['vllm:generation_tokens_total']==counters['vllm:request_generation_tokens_sum']==r['tokens']
 assert math.isclose(r['tokens']/seconds,r['output_tokens_per_second'])
 assert all(v==0 for k,v in r['quality_metrics'].items() if k.endswith('failed_count'))
 with gzip.open(d/'gpu.csv.gz','rt') as f:
  samples=[x for x in csv.DictReader(f) if h['_timestamp']-seconds+1<=float(x['timestamp'])<=h['_timestamp']-1]
 assert math.isclose(max(float(x['memory_gib']) for x in samples),r['gpu_memory_max_gib'])
 c=r['config'];assert c['trainer']['val_only'] and c['actor_rollout_ref']['rollout']['val_kwargs']['n']==16
 print('PASS',name,round(seconds,2),round(r['output_tokens_per_second'],2),counters['vllm:num_preemptions_total'])
subprocess.run([sys.executable,str(P/'calculate_capacity.py')],check=True,stdout=subprocess.DEVNULL)
q=json.loads((P/'capacity-estimates.json').read_text());assert q['kv_bytes_per_token']==28672
assert math.isclose(q['sequence_kv_rows'][0]['kv_gib_C128'],28)
assert math.isclose(q['sequence_kv_rows'][1]['kv_gib_C192'],84)
print('PASS capacity formulas; 9 archived GPU runs verified')
