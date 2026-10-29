#!/usr/bin/env bash
# Fixed-weight default single path: one warmup and three complete measured batches.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CURRENT=$(cd -- "$HERE/../../../.." && pwd)
WORKSPACE=$(dirname -- "$CURRENT")
if [[ ${1:-} == worker ]]; then
    ROOT=${2:?source root}
    export RUN_DIR=${3:?output directory}
    SUPPORT=${4:?support directory}
    export PYTHONPATH="$ROOT:$SUPPORT" VLLM_PLUGINS="" PYTHONHASHSEED=0
    export STUDENT_GPUS=1 SMOKE_RAY_CPUS=12 WANDB_MODE=offline
    export STUDENT_MODEL="$WORKSPACE/model/Qwen3-1.7B-Base" TEACHER_MODEL="$WORKSPACE/model/Qwen3-8B"
    export DATA_ROOT="$WORKSPACE/data" CONTINUOUS_TOKEN_MODEL_FAMILY=qwen3
    export TRAIN_FILE="$SUPPORT/prompts.parquet"
    export TRAIN_BATCH_SIZE=64 MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=7168 MAX_VAL_RESPONSE_LENGTH=128
    export ROLLOUT_ENFORCE_EAGER=False ROLLOUT_MAX_NUM_SEQS=32 ROLLOUT_GPU_MEMORY_UTILIZATION=0.65
    export ROLLOUT_TOKEN_BUDGET=8192 PPO_MAX_TOKEN_LEN_PER_GPU=11264 TEACHER_FSDP_TOKEN_BUDGET=11264
    export SMOKE_STEPS=$((1 + ${FUSION_REGRESSION_REPEATS:-3})) PPO_EPOCHS=1
    export EXPERIMENT_NAME="$(basename -- "$RUN_DIR")" VERL_FILE_LOGGER_PATH="$RUN_DIR/metrics.jsonl"
    [[ ! -e $RUN_DIR ]] || { echo "Refusing existing regression run" >&2; exit 2; }
    if [[ ${DRY_RUN:-0} != 1 ]]; then
        "$PYTHON_BIN" "$HERE/monitor.py" --parent "${FUSION_RESOURCE_PARENT:-$$}" \
            --out "$RUN_DIR/resources.jsonl" >/dev/null 2>&1 8>&- 9>&- &
    fi
    cd "$ROOT"
    exec bash "$ROOT/examples/mirroropd/smoke/smoke_full_reverse_kl.sh" \
        'data.shuffle=False' 'trainer.critic_warmup=100' 'trainer.val_before_train=False' \
        'trainer.logger=[console,file]' 'actor_rollout_ref.rollout.full_determinism=True' \
        'actor_rollout_ref.rollout.stop_token_ids=[151643,151645]' \
        'actor_rollout_ref.rollout.agent.num_workers=2' 'reward.num_workers=2' \
        'transfer_queue.backend.SimpleStorage.num_data_storage_units=1' \
        '++actor_rollout_ref.rollout.engine_kwargs.vllm.logits_processors=["tokenizer_support:TokenizerSupport"]' \
        '++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.inductor_compile_config.combo_kernels=true' \
        '++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.inductor_compile_config.benchmark_combo_kernel=false' \
        "trainer.rollout_data_dir=$RUN_DIR/generations" \
        "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONHASHSEED='0'" \
        "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_PLUGINS=''" \
        "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_ROLLOUT_IPC_NAMESPACE='$VERL_ROLLOUT_IPC_NAMESPACE'" \
        "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH='$VERL_FILE_LOGGER_PATH'"
fi
BASELINE=${1:?isolated baseline source with the common IPC isolation patch}
PREFIX=${2:?unique regression prefix}
SUPPORT="$WORKSPACE/runs/vllm-fusion-20261029/default-regression-support"
OUT="$WORKSPACE/runs/vllm-fusion-20261029/$PREFIX"
[[ -f $SUPPORT/prompts.parquet && -f $SUPPORT/baseline.json ]] || exit 2
mkdir "$OUT"
for variant in baseline current; do
    ROOT=$CURRENT
    if [[ $variant == baseline ]]; then ROOT=$BASELINE; fi
    bash "$HERE/run_fusion_local_fsdp2.sh" bash "$0" worker "$ROOT" "$OUT/$variant" "$SUPPORT" \
        > "$OUT/$variant-launch.log" 2>&1
done
