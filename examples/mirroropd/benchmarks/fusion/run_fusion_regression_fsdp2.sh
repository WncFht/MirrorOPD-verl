#!/usr/bin/env bash
# Legacy fixed-weight regression; each variant gets its own protected local allocation.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CURRENT=$(cd -- "$HERE/../../../.." && pwd)
WORKSPACE=$(dirname -- "$CURRENT")
if [[ ${1:-} != worker ]]; then
    BASELINE=${1:?Pass a detached baseline checkout}
    OUT=${2:-$WORKSPACE/runs/rollout-fusion-20261028/default-regression}
    [[ ! -e $OUT ]] || { echo "Refusing existing regression output: $OUT" >&2; exit 2; }
    mkdir -p "$OUT/support"
    cp "$CURRENT/verl/experimental/rollout_policy/vllm_support.py" "$OUT/support/tokenizer_support.py"
    for variant in baseline current; do
        ROOT=$CURRENT
        if [[ $variant == baseline ]]; then ROOT=$BASELINE; fi
        bash "$HERE/run_fusion_local_fsdp2.sh" bash "$0" worker "$ROOT" "$OUT/$variant" "$OUT/support" \
            > "$OUT/$variant-launch.log" 2>&1
    done
    exit
fi
[[ -n ${FUSION_RESOURCE_PARENT:-} ]] || { echo "Internal worker requires the protected allocation" >&2; exit 2; }
case "$CUDA_VISIBLE_DEVICES" in 6|7|6,7) ;; *) exit 2 ;; esac
ROOT=${2:?source root}
export RUN_DIR=${3:?output directory}
SUPPORT=${4:?support directory}
export PYTHONPATH="$ROOT:$SUPPORT"
export EXPERIMENT_NAME="fusion-default-regression-$(basename -- "$RUN_DIR")"
export VERL_FILE_LOGGER_PATH="$RUN_DIR/metrics.jsonl"
[[ ! -e $RUN_DIR ]] || { echo "Refusing existing regression run" >&2; exit 2; }
export OMP_NUM_THREADS=1 PYTHONHASHSEED=0
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_PLUGINS=""
export PYTHON_BIN=${PYTHON_BIN:-$WORKSPACE/envs/mirroropd-fusion/bin/python}
export LD_LIBRARY_PATH=$WORKSPACE/cache/fusion-runtime/cuda-compat-12-9/usr/local/cuda-12.9/compat:${LD_LIBRARY_PATH:-}
export STUDENT_MODEL=$(python3 "$CURRENT/examples/mirroropd/assets/paths.py" --alias Qwen3-1.7B-Base --model-root "${MODEL_ROOT:-$WORKSPACE/model}") TEACHER_MODEL=$(python3 "$CURRENT/examples/mirroropd/assets/paths.py" --alias Qwen3-8B --model-root "${MODEL_ROOT:-$WORKSPACE/model}")
export DATA_ROOT=$WORKSPACE/data CONTINUOUS_TOKEN_MODEL_FAMILY=qwen3
export TRAIN_FILE=$WORKSPACE/runs/rollout-fusion-20261028/default-regression-prompts.parquet
export TRAIN_BATCH_SIZE=64 MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=7168 MAX_VAL_RESPONSE_LENGTH=128
export ROLLOUT_ENFORCE_EAGER=False ROLLOUT_MAX_NUM_SEQS=32 ROLLOUT_GPU_MEMORY_UTILIZATION=0.65
export ROLLOUT_TOKEN_BUDGET=8192 PPO_MAX_TOKEN_LEN_PER_GPU=11264 TEACHER_FSDP_TOKEN_BUDGET=11264
export SMOKE_STEPS=4 PPO_EPOCHS=1
"$PYTHON_BIN" "$HERE/monitor.py" --parent "$FUSION_RESOURCE_PARENT" \
    --out "$RUN_DIR/resources.jsonl" >/dev/null 2>&1 8>&- 9>&- &
cd "$ROOT"
exec bash "$ROOT/examples/mirroropd/smoke/smoke_full_reverse_kl.sh" \
        'data.shuffle=False' 'trainer.critic_warmup=100' 'trainer.val_before_train=False' \
        'trainer.logger=[console,file]' 'actor_rollout_ref.rollout.full_determinism=True' \
        'actor_rollout_ref.rollout.stop_token_ids=[151643,151645]' \
        '++actor_rollout_ref.rollout.engine_kwargs.vllm.logits_processors=["tokenizer_support:TokenizerSupport"]' \
        '++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.inductor_compile_config.combo_kernels=true' \
        '++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.inductor_compile_config.benchmark_combo_kernel=false' \
        "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_ROLLOUT_IPC_NAMESPACE='$VERL_ROLLOUT_IPC_NAMESPACE'" \
        "trainer.rollout_data_dir=$RUN_DIR/generations" \
        "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONHASHSEED='0'" \
        "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_PLUGINS=''" \
        "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH='$VERL_FILE_LOGGER_PATH'"
