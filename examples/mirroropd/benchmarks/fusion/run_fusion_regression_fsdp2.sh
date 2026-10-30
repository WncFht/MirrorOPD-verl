#!/usr/bin/env bash
# Four fixed-weight full batches: first warmup, then three measured repeats.
# Compare the unchanged current-student rollout path at two source revisions.
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CURRENT=$(cd -- "$HERE/../../../.." && pwd)
WORKSPACE=$(dirname -- "$CURRENT")
BASELINE=${1:?Pass a detached baseline checkout}
OUT=${2:-$WORKSPACE/runs/rollout-fusion-20261028/default-regression}
mkdir -p "$OUT/support"
cp "$CURRENT/verl/experimental/rollout_policy/vllm_support.py" "$OUT/support/tokenizer_support.py"
export CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 PYTHONHASHSEED=0
export VLLM_USE_V2_MODEL_RUNNER=0 STUDENT_GPUS=4
export PYTHON_BIN=${PYTHON_BIN:-/tmp/mirroropd-fusion-venv/bin/python}
export LD_LIBRARY_PATH=$WORKSPACE/cache/fusion-runtime/cuda-compat-12-9/usr/local/cuda-12.9/compat:${LD_LIBRARY_PATH:-}
export STUDENT_MODEL=$(python3 "$CURRENT/examples/mirroropd/assets/paths.py" --alias Qwen3-1.7B-Base --model-root "${MODEL_ROOT:-$WORKSPACE/model}") TEACHER_MODEL=$(python3 "$CURRENT/examples/mirroropd/assets/paths.py" --alias Qwen3-8B --model-root "${MODEL_ROOT:-$WORKSPACE/model}")
export DATA_ROOT=$WORKSPACE/data CONTINUOUS_TOKEN_MODEL_FAMILY=qwen3
export TRAIN_FILE=$WORKSPACE/runs/rollout-fusion-20261028/default-regression-prompts.parquet
export TRAIN_BATCH_SIZE=64 MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=7168 MAX_VAL_RESPONSE_LENGTH=128
export ROLLOUT_ENFORCE_EAGER=False ROLLOUT_MAX_NUM_SEQS=32 ROLLOUT_GPU_MEMORY_UTILIZATION=0.65
export ROLLOUT_TOKEN_BUDGET=8192 PPO_MAX_TOKEN_LEN_PER_GPU=11264 TEACHER_FSDP_TOKEN_BUDGET=11264
export SMOKE_STEPS=4 PPO_EPOCHS=1
{
for variant in baseline current; do
    root=$CURRENT
    if [[ $variant == baseline ]]; then root=$BASELINE; fi
    export PYTHONPATH=$root:$OUT/support
    export EXPERIMENT_NAME=fusion-default-regression-$variant RUN_DIR=$OUT/$variant
    export VERL_FILE_LOGGER_PATH=$RUN_DIR/metrics.jsonl
    if [[ -e $RUN_DIR ]]; then
        echo "Refusing to overwrite $RUN_DIR" >&2
        exit 2
    fi
    "$PYTHON_BIN" "$CURRENT/examples/mirroropd/benchmarks/fusion/monitor.py" \
        --parent "$$" --out "$RUN_DIR/resources.jsonl" >/dev/null 2>&1 &
    monitor_pid=$!
    bash "$root/examples/mirroropd/smoke/smoke_full_reverse_kl.sh" \
        'data.shuffle=False' 'trainer.critic_warmup=100' 'trainer.val_before_train=False' \
        'trainer.logger=[console,file]' 'actor_rollout_ref.rollout.full_determinism=True' \
        'actor_rollout_ref.rollout.stop_token_ids=[151643,151645]' \
        '++actor_rollout_ref.rollout.engine_kwargs.vllm.logits_processors=["tokenizer_support:TokenizerSupport"]' \
        '++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.inductor_compile_config.combo_kernels=true' \
        '++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.inductor_compile_config.benchmark_combo_kernel=false' \
        "trainer.rollout_data_dir=$RUN_DIR/generations" \
        "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONHASHSEED='0'" \
        "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH='$VERL_FILE_LOGGER_PATH'" \
        > "$OUT/$variant-launch.log" 2>&1
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
done
exit
}
