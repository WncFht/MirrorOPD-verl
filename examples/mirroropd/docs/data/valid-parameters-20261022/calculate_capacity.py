"""Capacity estimates for TP1 BF16 Qwen2; no GPU required; all extrapolations labeled."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
c=json.loads((P/'student-config.json').read_text())
b=2*c['num_hidden_layers']*c['num_key_value_heads']*(c['hidden_size']//c['num_attention_heads'])*2
assert b==28672
R=json.loads((P/'analysis.json').read_text())['runs']
r=R['valid-full-s128-u0.65-t8192-a']
H=79.10;pool=47.22;peak=r['gpu_memory_max_gib'];busy=pool*r['max_kv_fraction']
out={'assumptions':{'tp':1,'dtype_bytes':2,'runtime_device_gib_rounded':H,'anchor_util':.65,'anchor_kv_pool_gib_rounded':pool,'anchor_nvml_peak_gib':peak,'non_kv_profile_and_other_allocations_held_constant':True,'concurrency_extrapolation_assumes_unchanged_active_length_distribution':True},'kv_bytes_per_token':b,'weight_gib_parameter_data_only':1777088000*2/2**30,'observed_peak_kv_gib_approx':busy,'capacity_rows':[],'sequence_kv_rows':[],'pressure_extrapolation':[]}
for u in [.5,.65,.7,.75,.8,.85,.9]:
 m=pool+(u-.65)*H
 out['capacity_rows'].append({'util':u,'kv_pool_gib_estimate':m,'kv_tokens_estimate':m*2**30/b,'nvml_peak_gib_estimate_fixed_concurrency128':peak+(u-.65)*H,'mean_live_context_budget_at_C128':m*2**30/b/128,'mean_live_context_budget_at_C192':m*2**30/b/192})
for L in [8192,16384,31744,35841]:
 out['sequence_kv_rows'].append({'context_tokens':L,'kv_gib_one_sequence':L*b/2**30,**{f'kv_gib_C{C}':C*L*b/2**30 for C in [96,128,160,192,256]}})
for C,u in [(128,.75),(160,.75),(192,.8),(192,.85),(256,.9)]:
 demand=busy*C/128;avail=pool+(u-.65)*H
 out['pressure_extrapolation'].append({'concurrency':C,'util':u,'estimated_peak_kv_gib':demand,'estimated_pool_gib':avail,'estimated_pool_fraction':demand/avail})
(P/'capacity-estimates.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
