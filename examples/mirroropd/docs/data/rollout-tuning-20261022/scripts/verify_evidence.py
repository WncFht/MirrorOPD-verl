"""Recompute experimental counts and rates from portable observations, without GPUs."""
import collections,gzip,json,math
from pathlib import Path
P=Path(__file__).resolve().parent.parent
for p in sorted(P.glob('valid-*')):
 s=json.loads((p/'summary.json').read_text());rows=json.loads((p/'response-audit.json').read_text())
 assert len(rows)==len({r['uid'] for r in rows})==2048
 groups=collections.Counter(r['uid'].rsplit('_',2)[0] for r in rows)
 assert len(groups)==128 and set(groups.values())=={16}
 assert sum(r['score'] for r in rows)/len(rows)==s['score_mean']
 with gzip.open(p/'engine-metrics.jsonl.gz','rt') as f:metrics=[json.loads(l) for l in f]
 last={r['address']:r['metrics'] for r in metrics if r.get('metrics')};assert len(last)==4
 for k,v in s['engine_counters'].items():assert sum(r[k] for r in last.values())==v
 assert s['engine_counters']['vllm:request_success_total']==2048
 assert s['tokens']==s['engine_counters']['vllm:generation_tokens_total']==s['engine_counters']['vllm:request_generation_tokens_sum']
 assert math.isclose(s['tokens']/s['testing_seconds'],s['output_tokens_per_second'])
 h=[json.loads(l) for l in (p/'history.jsonl').read_text().splitlines()];h=next(r for r in h if 'perf/timing/validation_seconds' in r)
 assert h['perf/timing/validation_seconds']==s['testing_seconds']
 assert sum(v for k,v in h.items() if k.endswith('/response_count') and k.startswith('eval/'))==2048
 assert all(v==0 for k,v in h.items() if k.endswith('/failed_count') and k.startswith('eval/'))
 assert json.loads((p/'exit.json').read_text())['exit_code']==0
 print(p.name,'PASS counts, unique sessions, scoring, engine counters, TPS, exit')
