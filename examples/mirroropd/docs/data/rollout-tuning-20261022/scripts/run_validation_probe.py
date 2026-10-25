import argparse,csv,json,os,subprocess,threading,time
from pathlib import Path
import psutil,pynvml
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['eager','graph'],required=True);p.add_argument('--attempt',default='1');p.add_argument('--window',type=int,default=16);p.add_argument('--seqs',type=int,choices=[32,64,128],default=32);args=p.parse_args()
root=Path('/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/MirrorOPD-verl')
python='/root/.local/share/mirroropd/venvs/profile-cu128/bin/python'
name=f'valid-w{args.window}-s{args.seqs}-20261022-{args.attempt}'
out=root/'outputs/rollout-tuning-20261022'/name
if out.exists(): raise SystemExit(f'Already exists: {out}')
pynvml.nvmlInit();handles=[pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(4)]
for i,h in enumerate(handles):
 if pynvml.nvmlDeviceGetMemoryInfo(h).used>1024**3:raise SystemExit(f'GPU {i} busy; refusing launch')
out.mkdir(parents=True)
env=os.environ.copy();settings={
'CUDA_VISIBLE_DEVICES':'0,1,2,3','STUDENT_GPUS':'4','TEACHER_GPUS':'0',
'DISTILLATION_EXECUTION_MODE':'shared_sync','ROLLOUT_TP':'1','TRAIN_BATCH_SIZE':'64',
'PYTHON_BIN':python,'VERL_USE_UV':'0','MODEL_ROOT':str(root.parent/'models'),'DATA_ROOT':str(root.parent/'MD-OPD/data'),
'VAL_BATCH_SIZE':str(args.window),'VAL_N':'16','TEST_FREQ':'-1','SAVE_FREQ':'-1','WANDB_MODE':'offline','RAY_ADDRESS':'local','RAY_DEDUP_LOGS':'0','RAY_USAGE_STATS_ENABLED':'0',
'ROLLOUT_ENFORCE_EAGER':'True' if args.mode=='eager' else 'False',
'EXPERIMENT_NAME':name,'RUN_DIR':str(out/'training'),'OMP_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false',
'NVTX_INJECTION64_PATH':'/nonexistent/verl-disabled-nvtx-injection.so'}
settings.update({k:str(root/'outputs/rollout-tuning-20261022/subsets'/v) for k,v in [('MATH_FILE','MATH-500.parquet'),('AMC_FILE','AMC23.parquet'),('AIME_FILE','AIME25.parquet')]})
settings['ROLLOUT_MAX_NUM_SEQS']=str(args.seqs)
env.update(settings)
cmd=['bash','examples/mirroropd/run_math_opd_ablation_fsdp2.sh','custom','--method','sampled',
 'trainer.total_training_steps=1','trainer.val_before_train=True','trainer.val_only=True','actor_rollout_ref.rollout.disable_log_stats=False',
 '++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL=INFO',
 '++ray_kwargs.ray_init.num_gpus=4','++ray_kwargs.ray_init.num_cpus=64',
 '++ray_kwargs.ray_init.include_dashboard=False','++ray_kwargs.ray_init.object_store_memory=4294967296',
 f'++ray_kwargs.ray_init._temp_dir=/tmp/rt22-v{args.window}s{args.seqs}-{args.attempt}',
 'actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,32]']
cmd[-1]='actor_rollout_ref.rollout.cudagraph_capture_sizes='+str([1,2,4,8,16,32]+([64,128] if args.seqs==128 else [64] if args.seqs==64 else [])).replace(' ','')
(out/'launch.json').write_text(json.dumps({'command':cmd,'environment_overrides':settings},indent=2)+'\n')
owned={};stop=threading.Event()
with (out/'launch.log').open('w') as log, (out/'gpu.csv').open('w') as f:
 writer=csv.writer(f);writer.writerow(['timestamp','gpu','busy_percent','memory_busy_percent','memory_gib','power_w','sm_clock_mhz'])
 proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 parent=psutil.Process(proc.pid)
 def monitor():
  while not stop.is_set():
   try:
    for child in parent.children(recursive=True):owned[child.pid]=child.create_time()
   except psutil.Error:pass
   for i,h in enumerate(handles):
    u=pynvml.nvmlDeviceGetUtilizationRates(h);m=pynvml.nvmlDeviceGetMemoryInfo(h)
    writer.writerow([time.time(),i,u.gpu,u.memory,m.used/1024**3,pynvml.nvmlDeviceGetPowerUsage(h)/1000,pynvml.nvmlDeviceGetClockInfo(h,pynvml.NVML_CLOCK_SM)])
   f.flush();stop.wait(1)
 thread=threading.Thread(target=monitor,daemon=True);thread.start();begin=time.time()
 try:code=proc.wait(timeout=3600)
 except subprocess.TimeoutExpired:
  os.killpg(proc.pid,15);code=124
 finally:
  time.sleep(3);stop.set();thread.join()
  # Only reap descendants observed under this launcher, checking PID identity.
  survivors=[]
  for pid,created in owned.items():
   try:
    child=psutil.Process(pid)
    if child.create_time()==created:child.terminate();survivors.append(child)
   except psutil.Error:pass
  _,alive=psutil.wait_procs(survivors,timeout=10)
  for child in alive:
   try:child.kill()
   except psutil.Error:pass
  (out/'exit.json').write_text(json.dumps({'exit_code':code,'wall_seconds':time.time()-begin,'finished_at':time.time()},indent=2)+'\n')
pynvml.nvmlShutdown();print(out,code,flush=True);raise SystemExit(code)
