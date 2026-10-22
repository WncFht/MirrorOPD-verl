import ast,json,re,time,requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
root=Path('/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/MirrorOPD-verl/outputs/rollout-tuning-20261022')
end=time.monotonic()+14400
names={'vllm:generation_tokens_total','vllm:prompt_tokens_total','vllm:request_success_total','vllm:num_requests_running','vllm:num_requests_waiting','vllm:kv_cache_usage_perc','vllm:num_preemptions_total','vllm:request_generation_tokens_sum','vllm:request_generation_tokens_count'}
known={};closed=set()
def fetch(address):
 try:
  r=requests.get('http://'+address+'/metrics',timeout=2);r.raise_for_status();metrics={}
  for line in r.text.splitlines():
   if line.startswith('#'):continue
   m=re.match(r'([\w:]+)(?:\{[^}]*\})?\s+([\d.eE+\-]+)',line)
   if m and m[1] in names:metrics[m[1]]=metrics.get(m[1],0)+float(m[2])
  return {'address':address,'time':time.time(),'metrics':metrics}
 except Exception as e:return {'address':address,'time':time.time(),'error':str(e)[:180]}
while time.monotonic()<end and not (root/'stop-engine-monitor').exists():
 for p in root.glob('valid-*'):
  if not p.is_dir() or p.name in closed:continue
  if (p/'exit.json').exists():closed.add(p.name);continue
  if p.name not in known and (p/'launch.log').exists():
   text=(p/'launch.log').read_text(errors='replace');m=re.search(r'LLMServerManager: (\[[^\n]+\])',text)
   if m:known[p.name]=ast.literal_eval(m[1])
  if p.name in known:
   with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(fetch,known[p.name]))
   with (p/'engine-metrics.jsonl').open('a') as f:
    for r in results:f.write(json.dumps(r)+'\n')
 time.sleep(2)
print('monitor finished')
