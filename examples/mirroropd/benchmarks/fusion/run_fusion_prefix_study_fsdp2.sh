#!/usr/bin/env bash
# E20-derived prefix comparison: fixed learning budget, explicit execution knobs.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../.." && pwd)
WORKSPACE=$(dirname -- "$REPO")
worker=false
if [[ ${1:-} == __worker ]]; then worker=true; shift; fi
PAIR=${1:?J or P}
ARM=${2:?S0, S, LM, TM, or T}
LABEL=${3:?unique run label}
MODE=${4:-formal}
[[ $LABEL =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Run label must contain only letters, numbers, underscores or hyphens' >&2; exit 2; }
case "$PAIR" in J|P) ;; *) exit 2 ;; esac
case "$ARM" in S0|S|LM|TM|T) ;; *) exit 2 ;; esac
case "$MODE" in formal|profile|smoke) ;; *) exit 2 ;; esac
FUSION_GPUS=${FUSION_GPUS:-7}
case "$FUSION_GPUS" in 6|7|6,7) ;; *) echo 'Use allocated GPUs 6, 7, or 6,7' >&2; exit 2 ;; esac
read_only=false
if [[ ${DRY_RUN:-0} == 1 || ${CONFIG_ONLY:-0} == 1 ]]; then read_only=true; fi
if ! $worker; then
    if $read_only; then
        IFS=',' read -r -a selected <<< "$FUSION_GPUS"
        export STUDENT_GPUS=${#selected[@]} CUDA_VISIBLE_DEVICES=""
        export PYTHON_BIN="$WORKSPACE/envs/mirroropd-fusion/bin/python" VERL_USE_UV=0 RAY_ADDRESS=local
        export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
        export PYTHONPYCACHEPREFIX=/root/.cache/mirroropd-fusion/pycache OMP_NUM_THREADS=1
        export VERL_ROLLOUT_IPC_NAMESPACE="prefix-config-${BASHPID}"
        exec bash "$0" __worker "$PAIR" "$ARM" "$LABEL" "$MODE"
    fi
    exec bash "$HERE/run_fusion_local_fsdp2.sh" bash "$0" __worker "$PAIR" "$ARM" "$LABEL" "$MODE"
fi
if ! $read_only; then
    [[ ${CUDA_VISIBLE_DEVICES:-} == "$FUSION_GPUS" && -n ${FUSION_RESOURCE_PARENT:-} ]] || exit 2
    IFS=',' read -r -a selected <<< "$FUSION_GPUS"
    for device in "${selected[@]}"; do
        fd=9
        if [[ $device == 6 ]]; then fd=8; fi
        [[ $(readlink "/proc/$$/fd/$fd") == "$WORKSPACE/runs/vllm-fusion-20261029/gpu$device.lock" ]] || exit 2
    done
fi
export FUSION_GPUS
export STUDENT_MODEL="$WORKSPACE/model/DeepSeek-R1-Distill-Qwen-1.5B"
if [[ $PAIR == J ]]; then
    export TEACHER_MODEL="$WORKSPACE/model/JustRL-DeepSeek-1.5B"
    seqs=${FUSION_MAX_NUM_SEQS:-32}
    kv=${FUSION_KV_BYTES:-25769803776}
else
    export TEACHER_MODEL="$WORKSPACE/model/Polaris-7B-Preview"
    seqs=${FUSION_MAX_NUM_SEQS:-16}
    kv=${FUSION_KV_BYTES:-17179869184}
fi
export TRAIN_FILE="$WORKSPACE/data/math/DAPO/train.parquet" DATA_ROOT="$WORKSPACE/data"
export TRAIN_BATCH_SIZE=64 PPO_EPOCHS=16 LEARNING_RATE=1e-6 SEED=42 TOTAL_EPOCHS=1
export MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=7168 MAX_VAL_RESPONSE_LENGTH=31744
export TEST_FREQ=-1 SAVE_FREQ=-1 DISTILLATION_EXECUTION_MODE=shared_sync TEACHER_GPUS=0
export PARAM_OFFLOAD=True OPTIMIZER_OFFLOAD=True CONTINUOUS_TOKEN_MODEL_FAMILY=deepseek
export FULL_KL_TERMINATION_TOKEN_IDS='[]' MIRROR_DIAGNOSTICS=False
export FULL_KL_BACKEND=${PREFIX_FULL_KL_BACKEND:-precompute}
export FULL_KL_TOKEN_CHUNK=${PREFIX_TOKEN_CHUNK:-512} FULL_KL_VOCAB_CHUNK=${PREFIX_VOCAB_CHUNK:-8192}
export FULL_KL_ASYNC_WRITE=True FULL_KL_PREFETCH=True
export PPO_MAX_TOKEN_LEN_PER_GPU=${PREFIX_PPO_TOKEN_BUDGET:-32768}
export TEACHER_FSDP_TOKEN_BUDGET=${PREFIX_TEACHER_TOKEN_BUDGET:-32768}
export ROLLOUT_MAX_NUM_SEQS=${PREFIX_ROLLOUT_SEQS:-128}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${PREFIX_ROLLOUT_MEMORY_FRACTION:-0.65}
export WANDB_MODE=${WANDB_MODE:-offline} VERL_USE_UV=0
unset TRAIN_PRECISION
export TOTAL_TRAINING_STEPS=40
if [[ $MODE == profile ]]; then export TOTAL_TRAINING_STEPS=${PREFIX_PROFILE_OUTERS:-2}; fi
if [[ $MODE == smoke ]]; then
    export TOTAL_TRAINING_STEPS=2 TRAIN_BATCH_SIZE=2 MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=32
fi
# RolloutConfig validates this window even when no validation is requested.
export MAX_VAL_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH
context=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
export EXPERIMENT_NAME="prefix-${MODE}-${PAIR}-${ARM}-${LABEL}"
export RUN_DIR="$REPO/outputs/$EXPERIMENT_NAME"
export VERL_FILE_LOGGER_PATH="$RUN_DIR/metrics.jsonl"
tp=${FUSION_TP:-1}
if [[ $ARM != LM && $tp != 1 ]]; then
    echo 'FUSION_TP applies to LM; other prefix modes currently use TP1 replicas' >&2
    exit 2
fi
case "$tp" in 1|2) ;; *) exit 2 ;; esac
(( STUDENT_GPUS % tp == 0 )) || { echo 'Fusion TP must divide allocated GPUs' >&2; exit 2; }
policy_mode=single
source=student
binding=current
weight=0
case "$ARM" in
    S0) binding=initial ;;
    S) policy_mode=trajectory_mixture ;;
    LM) policy_mode=token_mixture; weight=0.5 ;;
    TM) policy_mode=trajectory_mixture; weight=0.5 ;;
    T) source=teacher; weight=1 ;;
esac
PROFILE_ARGS=()
if [[ ${PREFIX_TORCH_TRACE:-0} == 1 ]]; then
    [[ $MODE == profile ]] || { echo 'Torch traces are only allowed in dedicated profile runs' >&2; exit 2; }
    trace_outer=${PREFIX_TRACE_OUTER:-2}
    [[ $trace_outer =~ ^[1-9][0-9]*$ ]] && (( trace_outer <= TOTAL_TRAINING_STEPS )) || exit 2
    PROFILE_ARGS=(
        'global_profiler.tool=torch' "global_profiler.steps=[$trace_outer]"
        "global_profiler.save_path=$RUN_DIR/traces"
        'actor_rollout_ref.actor.profiler.enable=True'
        'actor_rollout_ref.actor.profiler.all_ranks=False'
        'actor_rollout_ref.actor.profiler.ranks=[0]'
        'actor_rollout_ref.actor.profiler.tool_config.torch.contents=[cuda,shapes]'
        "actor_rollout_ref.actor.profiler.tool_config.torch.discrete=${PREFIX_TRACE_DISCRETE:-false}"
        "actor_rollout_ref.actor.profiler.tool_config.torch.schedule.skip_first=${PREFIX_TRACE_SKIP:-0}"
        "actor_rollout_ref.actor.profiler.tool_config.torch.schedule.active=${PREFIX_TRACE_UPDATES:-2}"
        'actor_rollout_ref.actor.profiler.tool_config.torch.schedule.repeat=1'
        'actor_rollout_ref.rollout.profiler.enable=False'
    )
fi
if ! $read_only; then
    [[ ! -e $RUN_DIR ]] || { echo "Run already exists: $RUN_DIR" >&2; exit 2; }
    "$PYTHON_BIN" "$HERE/monitor.py" --parent "$FUSION_RESOURCE_PARENT" \
        --out "$RUN_DIR/resources.jsonl" >/dev/null 2>&1 8>&- 9>&- &
fi
cd "$REPO"
exec bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E20 --method full_kl \
    'trainer.val_before_train=False' 'trainer.test_freq=-1' 'trainer.save_freq=-1' \
    'trainer.resume_mode=disable' 'trainer.logger=[console,file]' \
    "trainer.default_local_dir=$RUN_DIR/runtime" 'data.val_files=[]' \
    'data.seed=42' 'data.shuffle=True' 'data.train_max_samples=-1' \
    'algorithm.filter_groups.enable=False' 'trainer.v1.sampler.sync_refill_failed_groups=False' \
    'actor_rollout_ref.actor.fsdp_config.forward_precision=null' \
    'actor_rollout_ref.actor.fsdp_config.model_dtype=fp32' \
    'actor_rollout_ref.actor.fsdp_config.dtype=bfloat16' \
    'actor_rollout_ref.actor.fsdp_config.seed=42' 'actor_rollout_ref.actor.data_loader_seed=42' \
    'actor_rollout_ref.rollout.seed=42' "actor_rollout_ref.rollout.max_model_len=$context" \
    'actor_rollout_ref.rollout.temperature=1.0' 'actor_rollout_ref.rollout.top_p=1.0' \
    'actor_rollout_ref.rollout.top_k=-1' 'actor_rollout_ref.rollout.n=1' \
    'actor_rollout_ref.rollout.logprobs_mode=processed_logprobs' \
    'actor_rollout_ref.rollout.stop_token_ids=[151643]' \
    'actor_rollout_ref.rollout.full_determinism=False' \
    "actor_rollout_ref.rollout.agent.num_workers=${PREFIX_AGENT_WORKERS:-8}" \
    "reward.num_workers=${PREFIX_REWARD_WORKERS:-8}" \
    "transfer_queue.backend.SimpleStorage.num_data_storage_units=${PREFIX_STORAGE_UNITS:-8}" \
    "++distillation.shared_teacher.forward_only_cpu_offload=${PREFIX_TEACHER_LAYER_OFFLOAD:-false}" \
    "rollout_policy.mode=$policy_mode" "rollout_policy.source=$source" \
    "rollout_policy.student=$binding" "rollout_policy.teacher_weight=$weight" \
    'rollout_policy.execution.backend=vllm_fusion' \
    "rollout_policy.execution.tensor_parallel_size=$tp" \
    "rollout_policy.execution.max_num_seqs=$seqs" \
    "rollout_policy.execution.kv_cache_memory_bytes=$kv" \
    "rollout_policy.execution.enforce_eager=${FUSION_ENFORCE_EAGER:-false}" \
    "++ray_kwargs.ray_init.num_gpus=$STUDENT_GPUS" \
    "++ray_kwargs.ray_init.num_cpus=${PREFIX_RAY_CPUS:-32}" \
    '++ray_kwargs.ray_init.include_dashboard=False' \
    "++ray_kwargs.ray_init.object_store_memory=${PREFIX_OBJECT_STORE_BYTES:-17179869184}" \
    "++ray_kwargs.ray_init._temp_dir=${SMOKE_RAY_TMPDIR:-/tmp/mirroropd-prefix-config-$$}" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_ROLLOUT_IPC_NAMESPACE='$VERL_ROLLOUT_IPC_NAMESPACE'" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH='$VERL_FILE_LOGGER_PATH'" \
    "${PROFILE_ARGS[@]}"
