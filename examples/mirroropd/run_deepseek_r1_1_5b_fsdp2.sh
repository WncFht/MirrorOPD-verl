#!/usr/bin/env bash
# 单教师 sampled-token OPD：DAPO 全量训练，MATH-500 / AMC23 / AIME25 验证。
# 默认占用 8 张卡：student 4 卡（训练与 rollout 共用），teacher 4 卡。
# 所有常用参数均可由同名环境变量覆盖；末尾还可追加原生 Hydra overrides。
# 用法：
#   DRY_RUN=1 bash examples/mirroropd/run_deepseek_r1_1_5b_fsdp2.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VERL_USE_UV=0 bash examples/mirroropd/run_deepseek_r1_1_5b_fsdp2.sh
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
WORKSPACE=$(dirname -- "$REPO_ROOT")
cd "$REPO_ROOT"

# ── 1. 模型与数据路径 ────────────────────────────────────────────────
MODEL_ROOT=${MODEL_ROOT:-$WORKSPACE/models}
DATA_ROOT=${DATA_ROOT:-$WORKSPACE/MD-OPD/data}
STUDENT_MODEL=${STUDENT_MODEL:-$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-1.5B}
TEACHER_MODEL=${TEACHER_MODEL:-$MODEL_ROOT/JustRL-DeepSeek-1.5B}
TRAIN_FILE=${TRAIN_FILE:-$DATA_ROOT/math/DAPO/train.parquet}
MATH_FILE=${MATH_FILE:-$DATA_ROOT/math/MATH-500/test.parquet}
AMC_FILE=${AMC_FILE:-$DATA_ROOT/math/AMC23/test.parquet}
AIME_FILE=${AIME_FILE:-$DATA_ROOT/math/AIME25/test.parquet}

# ── 2. GPU 与推理资源 ────────────────────────────────────────────────
# 两个 Ray 资源池互不重叠，总卡数 = STUDENT_GPUS + TEACHER_GPUS。
# TP 是单个推理副本占用的卡数；此处 DP/PP 均为 1，副本数 = 资源池卡数 / TP。
# 1.5B 模型先用 TP=1：4 个 student rollout 副本、4 个同模型 teacher 评分副本。
# 同步训练中 teacher 在 actor 更新和验证阶段空闲；8 卡分配不代表持续满载。
STUDENT_GPUS=${STUDENT_GPUS:-4}
TEACHER_GPUS=${TEACHER_GPUS:-4}
ROLLOUT_TP=${ROLLOUT_TP:-1}
TEACHER_TP=${TEACHER_TP:-1}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
ROLLOUT_TOKEN_BUDGET=${ROLLOUT_TOKEN_BUDGET:-8192}
TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.4}
TEACHER_MAX_NUM_SEQS=${TEACHER_MAX_NUM_SEQS:-64}
TEACHER_TOKEN_BUDGET=${TEACHER_TOKEN_BUDGET:-8192}

# ── 3. 训练与验证预算 ────────────────────────────────────────────────
# batch 是全局 prompt 数；n=1、mini-batch=batch、ppo_epochs=1，每批更新一次。
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SEED=${SEED:-42}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-7168}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}

# 验证允许更长的回答；VAL_N 是每题采样次数，VAL_BATCH_SIZE 是提交窗口题数。
MAX_VAL_RESPONSE_LENGTH=${MAX_VAL_RESPONSE_LENGTH:-31744}
VAL_N=${VAL_N:-16}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-16}
TEST_FREQ=${TEST_FREQ:-20}
SAVE_FREQ=${SAVE_FREQ:-50}

# ── 4. 启动前检查与派生值 ────────────────────────────────────────────
for name in STUDENT_GPUS TEACHER_GPUS ROLLOUT_TP TEACHER_TP TRAIN_BATCH_SIZE \
    MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH MAX_VAL_RESPONSE_LENGTH VAL_N VAL_BATCH_SIZE PPO_MAX_TOKEN_LEN_PER_GPU; do
    if [[ ! ${!name} =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer" >&2
        exit 2
    fi
done
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

# Student 同时容纳训练与验证；teacher 只评分训练轨迹。额外预留一个 token。
STUDENT_CONTEXT=$((MAX_PROMPT_LENGTH + (MAX_RESPONSE_LENGTH > MAX_VAL_RESPONSE_LENGTH ? MAX_RESPONSE_LENGTH : MAX_VAL_RESPONSE_LENGTH) + 1))
TEACHER_CONTEXT=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))
EXPERIMENT_NAME=${EXPERIMENT_NAME:-math-full-sampled-$(date +%Y%m%d-%H%M%S)-$$}
RUN_DIR=${RUN_DIR:-$REPO_ROOT/outputs/$EXPERIMENT_NAME}

# ── 5. 按职责组织原生 Hydra 参数 ──────────────────────────────────────

# 训练用完整 DAPO；验证按 data_source 分别记录三个数学数据集的结果。
DATA_ARGS=(
    "data.train_files=['$TRAIN_FILE']"
    "data.val_files=['$MATH_FILE','$AMC_FILE','$AIME_FILE']"
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
)

# FSDP2 在 student 池内分片；动态 token batching 与 offload 控制训练显存。
ACTOR_ARGS=(
    "actor_rollout_ref.model.path='$STUDENT_MODEL'"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU"
    "actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE"
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    "actor_rollout_ref.actor.optim.lr=$LEARNING_RATE"
    actor_rollout_ref.actor.optim.betas=[0.9,0.999]
    actor_rollout_ref.actor.optim.weight_decay=0.01
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0.0
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

# Student 用 vLLM 生成训练回答；与 actor 共用 4 张卡，分阶段切换。
ROLLOUT_ARGS=(
    actor_rollout_ref.rollout.name=vllm
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP"
    "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION"
    "actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS"
    "actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_TOKEN_BUDGET"
    "actor_rollout_ref.rollout.max_model_len=$STUDENT_CONTEXT"
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.calculate_log_probs=True
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
    "reward.custom_reward_function.path='$SCRIPT_DIR/reward.py'"
    reward.custom_reward_function.name=reward_func
)

# 固定 teacher 权重，用 vLLM 评分 student 已生成的 token；多个副本共享同一模型。
DISTILLATION_ARGS=(
    distillation.enabled=True
    distillation.nnodes=1
    "distillation.n_gpus_per_node=$TEACHER_GPUS"
    "distillation.teacher_models.teacher_model.model_path='$TEACHER_MODEL'"
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.temperature=1.0
    "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=$TEACHER_TP"
    "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=$TEACHER_GPU_MEMORY_UTILIZATION"
    "distillation.teacher_models.teacher_model.inference.max_model_len=$TEACHER_CONTEXT"
    "distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=$TEACHER_TOKEN_BUDGET"
    "distillation.teacher_models.teacher_model.inference.max_num_seqs=$TEACHER_MAX_NUM_SEQS"
    distillation.distillation_loss.loss_mode=k1
    distillation.distillation_loss.use_policy_gradient=True
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.loss_max_clamp=null
    distillation.distillation_loss.log_prob_min_clamp=null
)

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
    'trainer.logger=[console]'
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
# 已激活原生环境（开发机）：VERL_USE_UV=0，PYTHON_BIN 默认为 python3。
# uv 模式：driver 与 Ray worker 使用相同 extras，包含数学 verifier 依赖。
LAUNCH=("${PYTHON_BIN:-python3}")
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
    LAUNCH=(uv run --frozen --all-packages --extra fsdp --extra vllm --extra math python3)
    RAY=('ray_kwargs.ray_init.runtime_env.py_executable="uv run --frozen --all-packages --extra fsdp --extra vllm --extra math"')
fi
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
for path in "$TRAIN_FILE" "$MATH_FILE" "$AMC_FILE" "$AIME_FILE"; do
    if [[ ! -f $path ]]; then
        echo "Missing dataset: $path" >&2
        exit 2
    fi
done
if [[ -e $RUN_DIR ]]; then
    echo "Use a new RUN_DIR; already exists: $RUN_DIR" >&2
    exit 2
fi
# 同时记录源码状态、完整命令和解析配置，便于回溯每次实验。
mkdir -p "$RUN_DIR"
git rev-parse HEAD > "$RUN_DIR/source_commit.txt"
git diff --binary HEAD > "$RUN_DIR/source_changes.patch"
printf '%q ' "${CMD[@]}" "${ARGS[@]}" "${RAY[@]}" "$@" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
"${CMD[@]}" --cfg job --resolve "${ARGS[@]}" "${RAY[@]}" "$@" > "$RUN_DIR/resolved_config.yaml"
exec "${CMD[@]}" "${ARGS[@]}" "${RAY[@]}" "$@"
