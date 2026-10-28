# Sampled token 与 full-vocab 的公平性能对照

## 背景与 motivation

控制相同目标和工作量后，才比较估计器与reuse的实际执行成本。 综合研究问题见[计算成本与执行效率](../README.md)。

记录日期：2026-10-26。状态：实验方案已登记，完整训练的统一入口可用；固定前缀双分支回放工具待实现，尚未启动 GPU 测量。本实验比较执行成本，不计入正式数学质量实验的 run 数，也不新增 E 编号。

## 实验问题

在去掉 sampled 的非必要 anchor、任务 advantage 和任务 PPO 开销后，比较 sampled token 与 full-vocab 的 teacher 评分、actor 更新、完整 outer 耗时及显存。主要指标是速度，显存用于解释容量与分批差异。

已有 E14/E15 比较的是 precompute Full-KL 与 legacy vanilla sampled，包含不同学习率、轨迹和源码状态。其观察及边界见[已有对照](../full-vocab/training.md#full-kl-与-sampled)，本页不复制旧结果表。

## 主对照矩阵

| 组别 | Objective | Estimator | eta / kappa | 每批 reuse | Anchor |
| --- | --- | --- | --- | ---: | --- |
| S1 | mdpo | sampled | 1 / 0 | 1 | 无 |
| F1 | mdpo | full_vocab | 1 / 0 | 1 | 无 |
| S16 | mdpo | sampled | 1 / 0 | 16 | 无 |
| F16 | mdpo | full_vocab | 1 / 0 | 16 | 无 |

S 使用 current residual、生成时行为概率作分母的未截断 token-IS；F 使用原教师 reverse KL endpoint。固定前缀、正确行为概率、完整支持及相同 p/T 数值定义下，S 的动作期望梯度对应 F；有限样本梯度和 loss 不要求逐次相等。[数学条件](../../architecture/full-kl-walkthrough.md#73-与-sampled-current-is-的精确联系)。

两侧均为纯蒸馏，无任务奖励、ref/critic、任务 advantage、PPO clipping、额外 ratio guard 或 anchor forward。S 必须保留生成时 log-prob；F 的 loss 不需要它。这项采集成本属于实际端到端成本，另外拆出 rollout 阶段，不能把它误归到 actor。

`termination_token_ids=[]` 在两侧显式固定。终止动作合并会改变 Full-KL 的输出空间，且 sampled MDPO 当前不支持对应配置，因此不继承 E41 的终止动作设置。Kappa 在 eta=1 时不改变目标，显式设 0 用于减少配置歧义。

## 两层测量与工具选择

| 层次 | 要回答的问题 | 工具决定 | 当前状态 |
| --- | --- | --- | --- |
| 固定前缀 | 相同 token/mask、模型初态下，teacher→actor 与 actor 的成本差多少 | 需补双 estimator 回放工具，复用现有 worker/engine/loss | 待实现；不提供虚假的可运行命令 |
| 自然生成训练 | 两条实际生产路径每个 outer 的成本与工作量如何变化 | 直接使用 [run_math_opd_ablation_fsdp2.sh](../../../../run_math_opd_ablation_fsdp2.sh) | 已支持；命令见下文 |

完整训练不另写 launcher、不复制 Hydra 配置和 loss。使用 `custom --method mdpo --estimator sampled/full_vocab --eta 1 --kappa 0`，其余参数统一覆盖。不要使用 `--method sampled --policy-loss vanilla` 或只切到 legacy `token_is` 来代表已经精简的 sampled 主线；后者仍会进入 Trainer 的旧准备阶段。

### 固定前缀工具的缺口

现有 [pipeline.py](../../../../benchmarks/profile/pipeline.py)可复用 worker 初始化、同步计时、显存归约和输出结构，但不能直接完成本实验：它仅支持 full、从 input/output 文本重新分词、没有 behavior log-prob，并逐次调用 `train_batch`。直接重复调用 16 次会重复 load/offload，不等于生产 `train_mini_batch` 的一次驻留内 16 次 reuse。

现有 [replay_fixed_prefix.py](../../../../research/precision/replay_fixed_prefix.py)针对 MDPO 数值审计和多个精度分支，也不能作为 S/F 性能工具直接复用。普通 rollout JSONL 只保存解码文本与分数，不能恢复原 token/behavior；不能用重新分词和事后 student log-prob 冒充实际行为概率。

待实现工具放在 `examples/mirroropd/benchmarks/` 的对应专题中，职责限制如下，不建立新的训练框架：

1. 在统一生成入口取得一批真实轨迹时，导出原始 prompt/response token IDs、position IDs、loss/response mask、每个动作对应的 processed behavior log-prob、样本身份和实际生成配置。记录模型、tokenizer、代码版本；保留终止 token，不 decode→encode。
2. 固定相同样本顺序、DP 分片、micro-batch 布局和有效 token 数。两侧从相同 student/optimizer 初态开始；主计时使用 lr=0 保持模型参数固定，optimizer 仍真实执行。非零 LR 的更新健康检查另做，不混入速度样本。
3. 调用生产 teacher 评分语义：S 获取 sampled log-prob，F 获取选中 hidden 并使用冻结 head。Teacher 每轮只准备一次，reuse 中复用；不能两侧都无代价地读取预缓存标签后将其称为完整管线计时。
4. 调用现有 `train_mini_batch`，按 reuse 执行一次驻留周期；禁用无关诊断，保留计时/计数。分别测含 teacher 的管线、actor 调用和内部更新，保留两侧 payload 的真实传输成本。Direct worker 测量应明确不含 Ray/TQ/vLLM。
5. 每个进程预热至少 3 个 actor 调用，再测 9 个调用；S–F–F–S 交错，各 reuse 独立执行。进程级结果单列，不能把 inner updates 当独立重复。
6. 数值核对采用小词表枚举/独立参考验证期望梯度；真实前缀记录两种投影路径的 teacher/current action log-prob 差异，分别报告 BF16 输入、归一化和 VJP 精度。随机单动作梯度不要求等于 full 梯度。

冻结的行为策略只生成一次输入；除非作为另一个明确命名的实验，不将事后 q 概率替换为 beta，也不为方便回放而删掉 IS。细化 schema 与实现前先核对已有 TQ 导出接口，以免增加重复 payload 管道。

### 自然生成的范围

该层固定模型、prompt 顺序、seed、配置和运行资源，但不能保证相同生成轨迹。每组计划 13 个 outer，前 3 个单列预热，4–13 为 10 个正常统计样本；样本仍有预热趋势时延长全部组，不只删去某组慢点。S1/F1 各 13 次更新，S16/F16 各 208 次更新。

计时主运行不启用详细 profiler、验证、checkpoint 或额外生成文本 dump。标准 verifier 流程两侧保留；它不参与 loss，但可能影响 rollout pipeline 等待。需要 kernel/NVML 归因时，另做独立采集窗口并标明采样开销。

## 共同配置

| 项目 | 固定值 |
| --- | --- |
| 本机资源 | 同一组 4 张 H800；只使用已分配的 GPU 0–3，shared-sync、FSDP2、TP=1 |
| 模型 | DeepSeek-R1-Distill-Qwen-1.5B student / JustRL-DeepSeek-1.5B teacher |
| 数据 | DAPO，global prompt batch=mini-batch=64，n=1，seed=42 |
| 长度 | Prompt cap4096、response cap7168；保留自然停止 |
| 采样 | T=1、top-p=1、top-k=-1、do_sample=True、repetition_penalty=1 |
| 数值 | Actor FP32 参数、BF16 mixed precision、FP32 归约；teacher BF16 |
| Optimizer | AdamW，LR=1e-6（自然生成层）、betas=(0.9,0.999)、weight decay=0.01、clip_grad=1 |
| 调度 | Constant LR、warmup=0；参数与 optimizer offload 开启；activation checkpointing 开启 |
| 分批 | Actor/teacher token budget32768；主对照均关闭 gradient accumulation no-sync |
| Full 实现 | precompute_fused，token chunk128、vocab chunk8192，hidden async write/prefetch开启 |
| 诊断 | Entropy、mirror KL diagnostics、分位数关闭；保留生产默认的基本更新记录 |
| 日志 | Console+file，两侧相同；运行目录不复用，不恢复旧 checkpoint |

这是“同预算下的当前实现”主对照。Full 的必要 objective 位置统计与 sampled 基本观测不完全相同，要记录其差别，不能称为零诊断纯 kernel 测试。后续分块/融合 sampled head 优化作为单独 A/B，不悄悄并入本矩阵。也可另做两侧各自调优后的容量/吞吐比较，但必须与主表分开。

## 完整训练启动命令

以下在仓库根目录执行。训练仍由统一入口负责；Bash 函数只将四组实验共用参数写在一处。`trial` 区分独立进程；实际运行前分配 GPU、满足共享空闲门槛并固定同一源码快照。

```bash
export PYTHON_BIN=/root/.local/share/mirroropd/venvs/profile-cu128/bin/python
export MODEL_ROOT="$PWD/../model"
export DATA_ROOT="$PWD/../MD-OPD/data"

run_perf_case() {
    local estimator=$1 reuse=$2 trial=$3
    case "$estimator" in sampled|full_vocab) ;; *) return 2 ;; esac
    case "$reuse" in 1|16) ;; *) return 2 ;; esac
    local run_name="perf-${estimator}-r${reuse}-${trial}"
    local run_dir="$PWD/outputs/sampled-full-performance/$run_name"
    STUDENT_GPUS=4 TEACHER_GPUS=0 DISTILLATION_EXECUTION_MODE=shared_sync \
    STUDENT_MODEL="$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-1.5B" \
    TEACHER_MODEL="$MODEL_ROOT/JustRL-DeepSeek-1.5B" DOMAIN=math \
    TRAIN_BATCH_SIZE=64 PPO_EPOCHS="$reuse" TOTAL_TRAINING_STEPS=13 \
    TOTAL_EPOCHS=1 LEARNING_RATE=1e-6 SEED=42 TEST_FREQ=-1 SAVE_FREQ=-1 \
    MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=7168 MAX_VAL_RESPONSE_LENGTH=31744 \
    PPO_MAX_TOKEN_LEN_PER_GPU=32768 TEACHER_FSDP_TOKEN_BUDGET=32768 \
    TEACHER_FSDP_DTYPE=bfloat16 GRADIENT_CHECKPOINTING=True \
    PARAM_OFFLOAD=True OPTIMIZER_OFFLOAD=True \
    ROLLOUT_TP=1 ROLLOUT_MAX_NUM_SEQS=128 ROLLOUT_TOKEN_BUDGET=8192 \
    ROLLOUT_GPU_MEMORY_UTILIZATION=0.65 ROLLOUT_ENFORCE_EAGER=False \
    FULL_KL_BACKEND=precompute_fused FULL_KL_TOKEN_CHUNK=128 FULL_KL_VOCAB_CHUNK=8192 \
    FULL_KL_ASYNC_WRITE=True FULL_KL_PREFETCH=True FULL_KL_TERMINATION_TOKEN_IDS='[]' \
    MIRROR_DIAGNOSTICS=False EXPERIMENT_NAME="$run_name" RUN_DIR="$run_dir" \
    VERL_FILE_LOGGER_PATH="$run_dir/metrics.jsonl" VERL_USE_UV=0 \
    bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh custom \
      --method mdpo --estimator "$estimator" --eta 1 --kappa 0 \
      data.train_max_samples=-1 \
      actor_rollout_ref.actor.fsdp_config.seed=42 \
      actor_rollout_ref.actor.data_loader_seed=42 \
      actor_rollout_ref.rollout.seed=42 \
      actor_rollout_ref.rollout.do_sample=True \
      +actor_rollout_ref.rollout.repetition_penalty=1.0 \
      actor_rollout_ref.rollout.logprobs_mode=processed_logprobs \
      actor_rollout_ref.actor.calculate_entropy=False \
      actor_rollout_ref.actor.fsdp_config.model_dtype=float32 \
      '+actor_rollout_ref.actor.fsdp_config.mixed_precision={param_dtype:bfloat16,reduce_dtype:float32}' \
      actor_rollout_ref.actor.fsdp_config.use_no_sync_for_gradient_accumulation=False \
      actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
      actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
      trainer.metrics_quantile_interval=0 \
      'trainer.logger=[console,file]' \
      trainer.val_before_train=False trainer.resume_mode=disable \
      global_profiler.tool=null global_profiler.steps=null \
      "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH='$run_dir/metrics.jsonl'"
}

# 只打印四组命令，不创建训练目录或启动GPU任务。
DRY_RUN=1 run_perf_case sampled 1 preview
DRY_RUN=1 run_perf_case full_vocab 1 preview
DRY_RUN=1 run_perf_case sampled 16 preview
DRY_RUN=1 run_perf_case full_vocab 16 preview
```

`CONFIG_ONLY=1 run_perf_case sampled 1 config` 可进一步导出 Hydra 解析配置；它也不启动训练。若同时设置 DRY_RUN，DRY_RUN 优先。实际执行时清除这两个开关，指定获分配的 `CUDA_VISIBLE_DEVICES=0,1,2,3`，使用新的 trial 名称。统一入口会拒绝已有 RUN_DIR，并记录完整命令、配置、源码 commit 与工作区 patch。

已完成四组命令的 Bash 语法与 dry-run 检查，以及两种 estimator 在 reuse=16 下的 Hydra 解析、蒸馏参数校验和 shared-sync 资源校验；两侧 spec 均不需要 anchor。解析后配置仅在 estimator、对应 PG 模式、rollout log-prob 采集及实验输出路径上不同。检查未加载模型、未启动 Ray 训练；运行时模型/数据兼容性仍由短测确认。

首次执行先做两侧一致的短长度功能检查，确认数据字段、teacher 评分和 optimizer 更新后，再按正式长度跑速度窗口；短长度结果不放进性能主表。自然生成层每个 reuse 至少按 S–F–F–S 执行两次独立进程/方法，保留各次工作量，不把 13 步均值当作独立重复。

## 结果与验收

测量边界、显存定义和统计协议直接沿用[统一测量方法](../full-vocab/methodology.md)。结果按以下字段组织，尚无数值时留空，不以旧 E14/E15 填充。

| 维度 | 必须记录 |
| --- | --- |
| 身份 | Estimator、reuse、trial、实际源码/配置、GPU/依赖、初态与输入来源 |
| 工作量 | Prompt/response/input/有效loss token、长度分布、每 rank micro-batch 数 |
| 速度 | Teacher 含搬运、anchor/old-log-prob/任务 advantage 是否执行、actor、每次更新、权重同步、完整 outer |
| 显存 | Actor 窗口 allocated/reserved 峰值；同窗口 NVML 采样峰值；整卡峰值另列 |
| 健康 | Expected/attempted/applied updates、非有限值、skip、退出码；全局 token 分母 |
| 统计 | 预热/采集窗口、每个进程结果、配对顺序；token/s 是否包含 reuse |

主路径必须确认没有 anchor forward、旧 log-prob 评分和任务 advantage；不能只因日志字段缺失就记为零成本，应结合 resolved spec、调用路径及需要时的 trace。Sampled 必须保留真实 behavior score，full 应关闭不需要的 rollout log-prob 采集。两侧输出 head 与 teacher 数值路径的精度差异同时记录。

若一侧 OOM，先记录共同预算下的容量结果，再以两侧相同的新预算重跑整个对照；不能只缩小一侧分批后仍称同配置比较。单次 sampled 梯度与 full 不同属于估计器差异，不以“参数 delta 逐元素相同”作验收条件。

## 实施顺序与状态

| 工作 | 状态 |
| --- | --- |
| 方案、四组自然生成命令、统一启动器覆盖核对 | 已准备 |
| 原始 token/behavior 导出与固定前缀双分支工具 | 待实现 |
| 两侧短程功能检查 | 未执行 |
| 固定输入 S/F 配对测量，reuse=1/16 | 未执行 |
| 自然生成完整周期测量与独立 profile | 未执行 |
| 对照分析与优化后 sampled 复测 | 未执行 |

计划与运行状态在本页维护。真正完成测量后将小型结果和分析放入本目录，公共性能结论仅链接本实验，不在两个目录复制同一份数据表。
