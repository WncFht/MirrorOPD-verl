#!/usr/bin/env bash
# Two real training outers followed by checkpoint restore on the allocated GPU.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../.." && pwd)
WORKSPACE=$(dirname -- "$REPO")
PAIR=${1:?dense or moe}
BINDING=${2:?current or initial}
PREFIX=${3:?unique run prefix}
RESUME_FROM=${FUSION_RESUME_FROM:-}
FUSION_MODEL_ROOT=${FUSION_MODEL_ROOT:-$WORKSPACE/model}
export STUDENT_MODEL="$FUSION_MODEL_ROOT/Qwen3-1.7B-Base"
case "$PAIR" in
    dense) export TEACHER_MODEL="$FUSION_MODEL_ROOT/Qwen3-8B" ;;
    moe) export TEACHER_MODEL="$FUSION_MODEL_ROOT/Qwen3-30B-A3B-Instruct-2507" ;;
    *) exit 2 ;;
esac
case "$BINDING" in current|initial) ;; *) exit 2 ;; esac
DATA=${FUSION_DATA_DIR:-$WORKSPACE/runs/vllm-fusion-20261029/lifecycle-data}
[[ -f $DATA/manifest.json ]] || { echo "Run prepare_vllm_workload.py first" >&2; exit 2; }
export TRAIN_FILE="$DATA/DAPO.parquet" MATH_FILE="$DATA/MATH-500.parquet" AMC_FILE="$DATA/AMC23.parquet"
export AIME24_FILE="$DATA/AIME24.parquet" AIME_FILE="$DATA/AIME25.parquet" AIME26_FILE="$DATA/AIME26.parquet"
"$WORKSPACE/envs/mirroropd-fusion/bin/python" "$HERE/resource_guard.py" validate --gpus "${FUSION_GPUS:-7}"
IFS=',' read -r -a fusion_devices <<< "${FUSION_GPUS:-7}"
fusion_cpus=$((12 * ${#fusion_devices[@]}))
export TRAIN_BATCH_SIZE=${FUSION_TRAIN_BATCH_SIZE:-2} PPO_EPOCHS=1 VAL_N=1 SMOKE_RAY_CPUS=${SMOKE_RAY_CPUS:-$fusion_cpus}
export MAX_PROMPT_LENGTH=${FUSION_PROMPT_CAP:-1024} MAX_RESPONSE_LENGTH=${FUSION_RESPONSE_CAP:-32} MAX_VAL_RESPONSE_LENGTH=32
export PARAM_OFFLOAD=True OPTIMIZER_OFFLOAD=True
export WANDB_MODE=offline
# Three distinct batches are required across the two-outers + restore sequence.
"$WORKSPACE/envs/mirroropd-fusion/bin/python" - "$DATA/manifest.json" "$TRAIN_BATCH_SIZE" "$MAX_PROMPT_LENGTH" <<'PYVERIFY'
import json
import sys
manifest = json.load(open(sys.argv[1]))
if len(manifest["sources"]["DAPO"]["row_indices"]) < 3 * int(sys.argv[2]):
    raise SystemExit("Lifecycle data must cover three complete training batches")
if manifest["max_prompt_length"] > int(sys.argv[3]):
    raise SystemExit("Prepared prompt cap exceeds the runtime cap; regenerate the frozen workload")
PYVERIFY
mkdir -p "$WORKSPACE/runs/vllm-fusion-20261029/$PREFIX"
POLICY=('rollout_policy.execution.backend=vllm_fusion'
    "rollout_policy.execution.enforce_eager=${FUSION_ENFORCE_EAGER:-true}"
    "rollout_policy.execution.tensor_parallel_size=${FUSION_TP:-1}"
    "rollout_policy.execution.max_num_seqs=${FUSION_MAX_NUM_SEQS:-2}"
    "rollout_policy.execution.kv_cache_memory_bytes=${FUSION_KV_BYTES:-1073741824}")
if [[ $BINDING == initial ]]; then
    POLICY+=('rollout_policy.teacher_weight={kind:linear,start:1.0,end:0.0,steps:2,clock:outer_step}')
fi
run() {
    local name=$1
    shift
    [[ ! -e $REPO/outputs/$name ]] || { echo "Run exists: $name" >&2; exit 2; }
    EXPERIMENT_NAME=$name bash "$HERE/run_fusion_local_fsdp2.sh" \
        bash "$REPO/examples/mirroropd/smoke/smoke_rollout_fusion.sh" token_mixture "$BINDING" \
        "${POLICY[@]}" 'trainer.test_freq=1' 'trainer.save_freq=2' \
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1' \
        'actor_rollout_ref.rollout.agent.num_workers=2' 'reward.num_workers=2' \
        'transfer_queue.backend.SimpleStorage.num_data_storage_units=1' \
        'data.train_max_samples=-1' 'data.val_max_samples=-1' "$@" \
        > "$WORKSPACE/runs/vllm-fusion-20261029/$PREFIX/$name.log" 2>&1
}
if [[ -n $RESUME_FROM ]]; then
    run "$PREFIX" 'trainer.total_training_steps=3' 'trainer.resume_mode=resume_path' \
        "trainer.resume_from_path=$RESUME_FROM"
else
    run "$PREFIX" 'trainer.total_training_steps=2'
    run "$PREFIX-resume" 'trainer.total_training_steps=3' 'trainer.resume_mode=resume_path' \
        "trainer.resume_from_path=$REPO/outputs/$PREFIX/checkpoints/global_step_2"
fi
