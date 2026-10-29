#!/usr/bin/env bash
# Local rollout-policy functional acceptance. Native overrides go last.
set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
WORKSPACE=$(dirname -- "$REPO_ROOT")
MODE=${1:-token_mixture}
BINDING=${2:-current}
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi
case "$MODE" in single|trajectory_mixture|token_mixture) ;; *) exit 2 ;; esac
case "$BINDING" in current|initial) ;; *) exit 2 ;; esac
: "${CUDA_VISIBLE_DEVICES:?Set the GPUs allocated to this run explicitly}"
IFS=',' read -r -a FUSION_VISIBLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
export STUDENT_GPUS=${STUDENT_GPUS:-${#FUSION_VISIBLE_GPUS[@]}}
export STUDENT_MODEL=${STUDENT_MODEL:-$WORKSPACE/model/Qwen3-1.7B-Base}
export TEACHER_MODEL=${TEACHER_MODEL:-$WORKSPACE/model/Qwen3-8B}
export CONTINUOUS_TOKEN_MODEL_FAMILY=qwen3
export OMP_NUM_THREADS=1
export VLLM_USE_V2_MODEL_RUNNER=0
export PYTHON_BIN=${PYTHON_BIN:-$WORKSPACE/envs/mirroropd-fusion/bin/python}
COMPAT_DIR=$WORKSPACE/cache/fusion-runtime/cuda-compat-12-9/usr/local/cuda-12.9/compat
if [[ -d $COMPAT_DIR ]]; then
    export LD_LIBRARY_PATH=$COMPAT_DIR:${LD_LIBRARY_PATH:-}
fi
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-fusion-$MODE-$BINDING-$(date +%Y%m%d-%H%M%S)}
export VERL_FILE_LOGGER_PATH=${VERL_FILE_LOGGER_PATH:-$REPO_ROOT/outputs/$EXPERIMENT_NAME/metrics.jsonl}
if [[ ${FUSION_MONITOR:-1} == 1 ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/examples/mirroropd/benchmarks/fusion/monitor.py" \
        --parent "$$" --out "$REPO_ROOT/outputs/$EXPERIMENT_NAME/resources.jsonl" >/dev/null 2>&1 &
fi
exec bash "$SCRIPT_DIR/smoke_full_reverse_kl.sh" \
    "rollout_policy.mode=$MODE" "rollout_policy.student=$BINDING" \
    'actor_rollout_ref.rollout.stop_token_ids=[151643,151645]' \
    '++actor_rollout_ref.rollout.engine_kwargs.vllm.logits_processors=["verl.experimental.rollout_policy.vllm_support:TokenizerSupport"]' \
    "rollout_policy.micro_batch_size=${FUSION_MICRO_BATCH_SIZE:-2}" \
    'trainer.logger=[console,file]' \
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH='$VERL_FILE_LOGGER_PATH'" "$@"
