#!/usr/bin/env bash
# math / code / IF 共用的单教师 sampled-token OPD 启动逻辑。
# 默认共享 8 张卡：rollout → 冻结 teacher FSDP2 评分 → actor 更新。
# DISTILLATION_EXECUTION_MODE=standalone 可恢复原 student 4 + teacher 4 路径。
# 所有常用参数均可由同名环境变量覆盖；末尾还可追加原生 Hydra overrides。
# 用法：
#   DRY_RUN=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh custom --method sampled
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VERL_USE_UV=0 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh custom --method sampled
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXAMPLE_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$EXAMPLE_DIR/../.." && pwd)
WORKSPACE=$(dirname -- "$REPO_ROOT")
cd "$REPO_ROOT"

# Native launch defaults belong to the shared recipe, including scheduler launches.
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# ── 1. 模型与数据路径 ────────────────────────────────────────────────
MODEL_ROOT=${MODEL_ROOT:-$WORKSPACE/model}
DATA_ROOT=${DATA_ROOT:-$WORKSPACE/data}
STUDENT_MODEL=${STUDENT_MODEL:-$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-1.5B}
DOMAIN=${DOMAIN:-math}
VAL_FILES=()
case "$DOMAIN" in
    math)
        TEACHER_MODEL=${TEACHER_MODEL:-$MODEL_ROOT/JustRL-DeepSeek-1.5B}
        TRAIN_FILE=${TRAIN_FILE:-$DATA_ROOT/math/DAPO/train.parquet}
        MATH_FILE=${MATH_FILE:-$DATA_ROOT/math/MATH-500/test.parquet}
        AMC_FILE=${AMC_FILE:-$DATA_ROOT/math/AMC23/test.parquet}
        AIME_FILE=${AIME_FILE:-$DATA_ROOT/math/AIME25/test.parquet}
        AIME24_FILE=${AIME24_FILE:-$DATA_ROOT/math/AIME24/test.parquet}
        AIME26_FILE=${AIME26_FILE:-$DATA_ROOT/math/AIME26/test.parquet}
        VAL_FILES=("$MATH_FILE" "$AMC_FILE" "$AIME24_FILE" "$AIME_FILE" "$AIME26_FILE")
        ;;
    code)
        TEACHER_MODEL=${TEACHER_MODEL:-$MODEL_ROOT/DeepCoder-1.5B-Preview}
        TRAIN_FILE=${TRAIN_FILE:-$DATA_ROOT/code/TACO/train.parquet}
        CODE_VAL_FILE=${CODE_VAL_FILE:-$DATA_ROOT/code/LCBv6/test.parquet}
        VAL_FILES=("$CODE_VAL_FILE")
        ;;
    if)
        TEACHER_MODEL=${TEACHER_MODEL:-$MODEL_ROOT/UltraData-IF-1.5B}
        TRAIN_FILE=${TRAIN_FILE:-$DATA_ROOT/if/MultiIF/train.parquet}
        # MultiIF 的正式验证包含三轮对话，不能由 single_turn_agent 代替。
        TEST_FREQ=${TEST_FREQ:--1}
        VAL_N=${VAL_N:-1}
        MAX_VAL_RESPONSE_LENGTH=${MAX_VAL_RESPONSE_LENGTH:-${MAX_RESPONSE_LENGTH:-7168}}
        ;;
    *) echo "Unknown DOMAIN=$DOMAIN; expected math, code or if" >&2; exit 2 ;;
esac

# ── 2. GPU 与推理资源 ────────────────────────────────────────────────
# shared_sync 的 STUDENT_GPUS 是整个共享池；TEACHER_GPUS=0 表示没有独立池。
# 小规模 smoke test 可显式设置 STUDENT_GPUS=2；正式默认使用 8 卡。
DISTILLATION_EXECUTION_MODE=${DISTILLATION_EXECUTION_MODE:-shared_sync}
case "$DISTILLATION_EXECUTION_MODE" in
    shared_sync)
        STUDENT_GPUS=${STUDENT_GPUS:-8}
        TEACHER_GPUS=${TEACHER_GPUS:-0}
        TEACHER_NNODES=0
        ;;
    standalone)
        STUDENT_GPUS=${STUDENT_GPUS:-4}
        TEACHER_GPUS=${TEACHER_GPUS:-4}
        TEACHER_NNODES=1
        ;;
    *) echo "Unknown DISTILLATION_EXECUTION_MODE=$DISTILLATION_EXECUTION_MODE" >&2; exit 2 ;;
esac
ROLLOUT_TP=${ROLLOUT_TP:-1}
TEACHER_TP=${TEACHER_TP:-1}
# Math defaults come from full 570 x 16 validation on four H800s.
# These are per-engine limits: keep them fixed when changing the GPU count.
if [[ $DOMAIN == math ]]; then
    ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}
    ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-128}
else
    ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}
    ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
fi
ROLLOUT_TOKEN_BUDGET=${ROLLOUT_TOKEN_BUDGET:-8192}
TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.4}
TEACHER_MAX_NUM_SEQS=${TEACHER_MAX_NUM_SEQS:-64}
TEACHER_TOKEN_BUDGET=${TEACHER_TOKEN_BUDGET:-8192}
# Compiled CUDA Graph rollout is the tested default. True remains an explicit
# fallback for diagnosing compilation or backend compatibility problems.
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
TEACHER_ENFORCE_EAGER=${TEACHER_ENFORCE_EAGER:-True}
# 共享 teacher 的 packed forward 预算，与 vLLM TEACHER_TOKEN_BUDGET 独立。
TEACHER_FSDP_TOKEN_BUDGET=${TEACHER_FSDP_TOKEN_BUDGET:-32768}
TEACHER_FSDP_DTYPE=${TEACHER_FSDP_DTYPE:-bfloat16}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-True}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-True}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}
CONTINUOUS_TOKEN_MODEL_FAMILY=${CONTINUOUS_TOKEN_MODEL_FAMILY:-deepseek}

# ── 3. 训练与验证预算 ────────────────────────────────────────────────
# batch 是全局 prompt 数；n=1、mini-batch=batch、ppo_epochs=1，每批更新一次。
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_EPOCHS=${PPO_EPOCHS:-1}
OPD_ADVANTAGE_SOURCE=${OPD_ADVANTAGE_SOURCE:-current}
OPD_POLICY_LOSS_MODE=${OPD_POLICY_LOSS_MODE:-vanilla}
OPD_CLIP_EPS=${OPD_CLIP_EPS:-0.2}
source "$SCRIPT_DIR/_distillation_args.sh"
LEARNING_RATE=${LEARNING_RATE:-1e-6}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SEED=${SEED:-42}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-7168}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}

# 验证允许更长的回答；VAL_N 是每题采样次数，VAL_BATCH_SIZE 是提交窗口题数。
MAX_VAL_RESPONSE_LENGTH=${MAX_VAL_RESPONSE_LENGTH:-31744}
VAL_N=${VAL_N:-16}
# null submits the full validation set; positive integers retain windowed evaluation.
# The submission window is independent of the per-engine active sequence limit.
if [[ $DOMAIN == math ]]; then
    VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-null}
    VAL_REPETITION_DETECTION=${VAL_REPETITION_DETECTION:-true}
else
    VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-32}
    VAL_REPETITION_DETECTION=${VAL_REPETITION_DETECTION:-false}
fi
# Selected from the replay tradeoff; it reduces observed harm, not to zero.
# Only validation receives this setting. False restores natural stopping.
VAL_REPETITION_MIN_PATTERN=${VAL_REPETITION_MIN_PATTERN:-16}
VAL_REPETITION_MAX_PATTERN=${VAL_REPETITION_MAX_PATTERN:-128}
VAL_REPETITION_MIN_COUNT=${VAL_REPETITION_MIN_COUNT:-8}
case ${VAL_REPETITION_DETECTION,,} in
    true|1)
        for name in VAL_REPETITION_MIN_PATTERN VAL_REPETITION_MAX_PATTERN VAL_REPETITION_MIN_COUNT; do
            if [[ ! ${!name} =~ ^[1-9][0-9]*$ ]]; then
                echo "$name must be a positive integer" >&2
                exit 2
            fi
        done
        if (( VAL_REPETITION_MIN_PATTERN > VAL_REPETITION_MAX_PATTERN || VAL_REPETITION_MIN_COUNT < 2 )); then
            echo "Validation repetition detection requires min pattern <= max pattern and min count >= 2" >&2
            exit 2
        fi
        VAL_REPETITION_PARAMS="{min_pattern_size:$VAL_REPETITION_MIN_PATTERN,max_pattern_size:$VAL_REPETITION_MAX_PATTERN,min_count:$VAL_REPETITION_MIN_COUNT}"
        ;;
    false|0) VAL_REPETITION_PARAMS=null ;;
    *) echo "VAL_REPETITION_DETECTION must be true or false" >&2; exit 2 ;;
esac
TEST_FREQ=${TEST_FREQ:-20}
SAVE_FREQ=${SAVE_FREQ:-50}
if [[ $DOMAIN == if ]] && (( TEST_FREQ > 0 )); then
    echo "MultiIF requires separate three-turn evaluation; set TEST_FREQ=-1" >&2
    exit 2
fi

# ── 4. 启动前检查与派生值 ────────────────────────────────────────────
for name in STUDENT_GPUS ROLLOUT_TP TEACHER_TP ROLLOUT_MAX_NUM_SEQS TRAIN_BATCH_SIZE PPO_EPOCHS TEACHER_FSDP_TOKEN_BUDGET \
    MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH MAX_VAL_RESPONSE_LENGTH VAL_N PPO_MAX_TOKEN_LEN_PER_GPU; do
    if [[ ! ${!name} =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer" >&2
        exit 2
    fi
done
if [[ $VAL_BATCH_SIZE != null && ! $VAL_BATCH_SIZE =~ ^[1-9][0-9]*$ ]]; then
    echo "VAL_BATCH_SIZE must be null or a positive integer" >&2
    exit 2
fi
if [[ $DISTILLATION_EXECUTION_MODE == shared_sync ]]; then
    if [[ $TEACHER_GPUS != 0 || $ROLLOUT_TP != 1 ]]; then
        echo "shared_sync requires TEACHER_GPUS=0 and ROLLOUT_TP=1" >&2
        exit 2
    fi
    for name in PARAM_OFFLOAD OPTIMIZER_OFFLOAD; do
        case "${!name}" in
            True|true|TRUE) ;;
            *) echo "shared_sync requires $name=True" >&2; exit 2 ;;
        esac
    done
    if (( TEACHER_FSDP_TOKEN_BUDGET < MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH )); then
        echo "Teacher FSDP token budget must hold a complete training sequence" >&2
        exit 2
    fi
elif [[ ! $TEACHER_GPUS =~ ^[1-9][0-9]*$ ]]; then
    echo "standalone requires positive TEACHER_GPUS" >&2
    exit 2
fi
if (( TRAIN_BATCH_SIZE % STUDENT_GPUS != 0 || STUDENT_GPUS % ROLLOUT_TP != 0 || TEACHER_GPUS % TEACHER_TP != 0 )); then
    echo "Require batch % student GPUs == 0 and each inference pool divisible by its TP" >&2
    exit 2
fi
if (( PPO_MAX_TOKEN_LEN_PER_GPU < MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH )); then
    echo "PPO token budget must hold a complete training sequence" >&2
    exit 2
fi
if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
    IFS=',' read -r -a VISIBLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
    if [[ $CUDA_VISIBLE_DEVICES == -1 ]] || (( ${#VISIBLE_GPUS[@]} < STUDENT_GPUS + TEACHER_GPUS )); then
        echo "CUDA_VISIBLE_DEVICES does not cover student + teacher pools" >&2
        exit 2
    fi
fi

# Student generation needs prompt + response. The scoring-only vLLM teacher
# requests one dummy output token after the complete training trajectory.
STUDENT_CONTEXT=$((MAX_PROMPT_LENGTH + (MAX_RESPONSE_LENGTH > MAX_VAL_RESPONSE_LENGTH ? MAX_RESPONSE_LENGTH : MAX_VAL_RESPONSE_LENGTH)))
TEACHER_CONTEXT=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))
# Bound capture to the configured decode concurrency rather than unrelated
# context/token budgets. Native Hydra overrides at the end can still replace it.
ROLLOUT_CAPTURE_SIZES="["
rollout_capture_size=1
while (( rollout_capture_size < ROLLOUT_MAX_NUM_SEQS )); do
    ROLLOUT_CAPTURE_SIZES+="$rollout_capture_size,"
    rollout_capture_size=$((rollout_capture_size * 2))
done
ROLLOUT_CAPTURE_SIZES+="$ROLLOUT_MAX_NUM_SEQS]"
EXPERIMENT_NAME=${EXPERIMENT_NAME:-$DOMAIN-full-sampled-$(date +%Y%m%d-%H%M%S)-$$}
RUN_DIR=${RUN_DIR:-$REPO_ROOT/outputs/$EXPERIMENT_NAME}

# SDK 在 WANDB_DIR 下创建 wandb/；固定仓库根目录，避免 Ray 的工作目录改变保存位置。
export WANDB_DIR="$REPO_ROOT"
export WANDB_MODE=${WANDB_MODE:-offline}

# ── 5. 按职责组织原生 Hydra 参数 ──────────────────────────────────────

# 按领域选择数据；IF 的空验证列表表示只训练、独立做三轮评测。
VAL_FILES_OVERRIDE="["
for path in "${VAL_FILES[@]}"; do
    VAL_FILES_OVERRIDE+="'$path',"
done
VAL_FILES_OVERRIDE="${VAL_FILES_OVERRIDE%,}]"
DATA_ARGS=(
    "data.train_files=['$TRAIN_FILE']"
    "data.val_files=$VAL_FILES_OVERRIDE"
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.val_batch_size=$VAL_BATCH_SIZE"
    "data.max_prompt_length=$MAX_PROMPT_LENGTH"
    "data.max_response_length=$MAX_RESPONSE_LENGTH"
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=True
    data.validation_shuffle=False
    "data.seed=$SEED"
    data.dataloader_num_workers=0
    # R1-Distill has a Qwen2 architecture but uses the DeepSeek chat protocol.
    "data.continuous_token.model_family=$CONTINUOUS_TOKEN_MODEL_FAMILY"
)

# FSDP2 在 student 池内分片；动态 token batching 与 offload 控制训练显存。
ACTOR_ARGS=(
    "actor_rollout_ref.model.path='$STUDENT_MODEL'"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=False
    "actor_rollout_ref.model.enable_gradient_checkpointing=$GRADIENT_CHECKPOINTING"
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU"
    "actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE"
    "actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS"
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    "actor_rollout_ref.actor.optim.lr=$LEARNING_RATE"
    actor_rollout_ref.actor.optim.betas=[0.9,0.999]
    actor_rollout_ref.actor.optim.weight_decay=0.01
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0.0
    "actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIMIZER_OFFLOAD"
)

# Student 用 vLLM 生成训练回答；与 actor 共池，分阶段切换。
ROLLOUT_ARGS=(
    actor_rollout_ref.rollout.name=vllm
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP"
    "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION"
    "actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS"
    "actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_TOKEN_BUDGET"
    "actor_rollout_ref.rollout.max_model_len=$STUDENT_CONTEXT"
    "actor_rollout_ref.rollout.enforce_eager=$ROLLOUT_ENFORCE_EAGER"
    "actor_rollout_ref.rollout.cudagraph_capture_sizes=$ROLLOUT_CAPTURE_SIZES"
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    "actor_rollout_ref.rollout.calculate_log_probs=$OPD_CALCULATE_LOG_PROBS"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU"
)

# 验证单独设置采样与长度；verifier 计算任务分数，不把它混入蒸馏训练 loss。
VALIDATION_ARGS=(
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    "actor_rollout_ref.rollout.val_kwargs.n=$VAL_N"
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9
    "actor_rollout_ref.rollout.val_kwargs.max_tokens=$MAX_VAL_RESPONSE_LENGTH"
    "actor_rollout_ref.rollout.val_kwargs.repetition_detection=$VAL_REPETITION_PARAMS"
    "reward.custom_reward_function.path='$EXAMPLE_DIR/rewards/reward.py'"
    reward.custom_reward_function.name=reward_func
)

# 固定 teacher 权重；两种执行模式使用相同 token 与蒸馏 loss。
DISTILLATION_ARGS=(
    distillation.enabled=True
    "distillation.execution_mode=$DISTILLATION_EXECUTION_MODE"
    "distillation.nnodes=$TEACHER_NNODES"
    "distillation.n_gpus_per_node=$TEACHER_GPUS"
    "distillation.teacher_models.teacher_model.model_path='$TEACHER_MODEL'"
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.temperature=1.0
    "distillation.teacher_models.teacher_model.inference.enforce_eager=$TEACHER_ENFORCE_EAGER"
    "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=$TEACHER_TP"
    "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=$TEACHER_GPU_MEMORY_UTILIZATION"
    "distillation.teacher_models.teacher_model.inference.max_model_len=$TEACHER_CONTEXT"
    "distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=$TEACHER_TOKEN_BUDGET"
    "distillation.teacher_models.teacher_model.inference.max_num_seqs=$TEACHER_MAX_NUM_SEQS"
    "${DISTILLATION_OBJECTIVE_ARGS[@]}"
)
if [[ $DISTILLATION_EXECUTION_MODE == shared_sync ]]; then
    DISTILLATION_ARGS+=(
        "distillation.shared_teacher.infer_max_token_len_per_gpu=$TEACHER_FSDP_TOKEN_BUDGET"
        "distillation.shared_teacher.model_dtype=$TEACHER_FSDP_DTYPE"
        "distillation.shared_teacher.mixed_precision={param_dtype:$TEACHER_FSDP_DTYPE,reduce_dtype:float32}"
    )
fi

# 沿用当前主线的 sampled-token k1 + PG 路径，保持 reward / IS 语义不变。
# task rewards 已禁用，grpo 不承担蒸馏 advantage 的构造；冻结 reward 方案另行讨论。
ALGORITHM_ARGS=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.rollout_correction.bypass_mode=False
    algorithm.rollout_correction.rollout_is=null
    algorithm.rollout_correction.rollout_rs=null
)

# V1 同步训练；按 global step 验证、保存，输出集中到同一个 RUN_DIR。
TRAINER_ARGS=(
    trainer.use_v1=True
    trainer.v1.trainer_mode=sync
    trainer.nnodes=1
    "trainer.n_gpus_per_node=$STUDENT_GPUS"
    "trainer.total_epochs=$TOTAL_EPOCHS"
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.test_freq=$TEST_FREQ"
    trainer.val_before_train=False
    trainer.resume_mode=disable
    trainer.project_name=MirrorOPD-verl
    "trainer.experiment_name='$EXPERIMENT_NAME'"
    'trainer.logger=[console,wandb]'
    "trainer.default_local_dir='$RUN_DIR/checkpoints'"
    "trainer.validation_data_dir='$RUN_DIR/validation'"
    "hydra.run.dir='$RUN_DIR/hydra'"
)

# 原生参数只在这里汇总；调用者的 "$@" 最后追加，保留 Hydra 覆盖优先级。
ARGS=(
    "${DATA_ARGS[@]}"
    "${ACTOR_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${VALIDATION_ARGS[@]}"
    "${DISTILLATION_ARGS[@]}"
    "${ALGORITHM_ARGS[@]}"
    "${TRAINER_ARGS[@]}"
)

# ── 6. Python 环境与执行模式 ──────────────────────────────────────────
# 原生模式默认使用本仓库虚拟环境的绝对路径，driver 与 Ray worker 保持一致。
# PYTHON_BIN 可显式指定其他绝对路径；uv 模式需显式 VERL_USE_UV=1。
PYTHON_BIN=${PYTHON_BIN:-$REPO_ROOT/.venv-native/bin/python}
LAUNCH=("$PYTHON_BIN")
RAY=("ray_kwargs.ray_init.runtime_env.py_executable='$PYTHON_BIN'")
if [ "${VERL_USE_UV:-0}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
    LAUNCH=(uv run --frozen --all-packages --extra fsdp --extra vllm --extra math python3)
    RAY=('ray_kwargs.ray_init.runtime_env.py_executable="uv run --frozen --all-packages --extra fsdp --extra vllm --extra math"')
fi
RAY+=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR='$WANDB_DIR'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_MODE='$WANDB_MODE'"
)
# Keep LCB limits identical in the driver and Ray reward workers.
if [[ $DOMAIN == code ]]; then
    export MIRROROPD_LCB_RUN_ID=${MIRROROPD_LCB_RUN_ID:-$EXPERIMENT_NAME}
    export MIRROROPD_LCB_AUDIT_DIR=${MIRROROPD_LCB_AUDIT_DIR:-$RUN_DIR/lcb-audit}
fi
for name in OPENCOMPASS_ROOT NLTK_DATA \
    MIRROROPD_LCB_BACKEND MIRROROPD_LCB_CGROUP_PARENT MIRROROPD_LCB_MEMORY_MIB \
    MIRROROPD_LCB_ADDRESS_SPACE_MIB MIRROROPD_LCB_OUTPUT_MIB MIRROROPD_LCB_DIAGNOSTIC_MIB \
    MIRROROPD_LCB_INPUT_MIB MIRROROPD_LCB_MAX_CONCURRENT MIRROROPD_LCB_QUEUE_TIMEOUT_S \
    MIRROROPD_LCB_STARTUP_TIMEOUT_S MIRROROPD_LCB_POLL_S MIRROROPD_LCB_SLOT_DIR \
    MIRROROPD_LCB_AUDIT_DIR MIRROROPD_LCB_RUN_ID; do
    if [[ -n ${!name:-} ]]; then
        RAY+=("+ray_kwargs.ray_init.runtime_env.env_vars.$name='${!name}'")
    fi
done
export RAY_ADDRESS=${RAY_ADDRESS:-local}
CMD=("${LAUNCH[@]}" -m verl.trainer.main_ppo)
# 只打印命令 / 只解析配置时，不创建训练输出目录，也不启动 Ray。
if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf '%q ' "${CMD[@]}" "${ARGS[@]}" "${RAY[@]}" "$@"
    printf '\n'
    exit 0
fi
if [[ ${CONFIG_ONLY:-0} == 1 ]]; then
    exec "${CMD[@]}" --cfg job --resolve "${ARGS[@]}" "${RAY[@]}" "$@"
fi
# ── 7. 正式运行与实验记录 ─────────────────────────────────────────────
# GPU 由调用者明确分配；不自动选择卡或停止已有进程。
[[ -n ${CUDA_VISIBLE_DEVICES:-} ]] || {
    echo "Set CUDA_VISIBLE_DEVICES to GPUs allocated to this run (student + teacher pools)." >&2
    exit 2
}
for path in "$TRAIN_FILE" "${VAL_FILES[@]}"; do
    if [[ ! -f $path ]]; then
        echo "Missing dataset: $path" >&2
        exit 2
    fi
done
if [[ $DOMAIN == if ]]; then
    if [[ -z ${OPENCOMPASS_ROOT:-} || ! -d $OPENCOMPASS_ROOT/opencompass/datasets/MultiIF ]]; then
        echo "Set OPENCOMPASS_ROOT to a source tree containing opencompass/datasets/MultiIF" >&2
        exit 2
    fi
    "${LAUNCH[@]}" -c 'from verl.utils.reward_score.one_shot_opd.if_eval import _ensure_opencompass_modules; _ensure_opencompass_modules()'
fi
if [[ -e $RUN_DIR ]]; then
    echo "Use a new RUN_DIR; already exists: $RUN_DIR" >&2
    exit 2
fi
# 同时记录源码状态、完整命令和解析配置，便于回溯每次实验。
mkdir -p "$RUN_DIR"
# tee 同时保存 stdout / stderr；pipefail 保留 Python 失败时的退出码。
export PYTHONUNBUFFERED=1
# Parse the pipeline and exit together. If this source file is updated while
# Python runs, Bash must not resume reading newly appended/repositioned code.
{
    {
        printf 'Training log: %s\n' "$RUN_DIR/train.log"
        if [[ -d $REPO_ROOT/.git || -f $REPO_ROOT/.git ]] && command -v git >/dev/null 2>&1; then
            git rev-parse HEAD > "$RUN_DIR/source_commit.txt"
            git diff --binary HEAD > "$RUN_DIR/source_changes.patch"
        else
            printf 'unavailable (source archive without Git metadata)\n' > "$RUN_DIR/source_commit.txt"
            : > "$RUN_DIR/source_changes.patch"
        fi
        printf '%q ' "${CMD[@]}" "${ARGS[@]}" "${RAY[@]}" "$@" > "$RUN_DIR/command.sh"
        printf '\n' >> "$RUN_DIR/command.sh"
        "${CMD[@]}" --cfg job --resolve "${ARGS[@]}" "${RAY[@]}" "$@" > "$RUN_DIR/resolved_config.yaml"
        exec "${CMD[@]}" "${ARGS[@]}" "${RAY[@]}" "$@"
    } 2>&1 | tee "$RUN_DIR/train.log"
    exit
}
