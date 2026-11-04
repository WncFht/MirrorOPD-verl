# 生成重复检测与提前停止

自 2026-11-05 起，公共 MirrorOPD launcher 的训练 rollout 与验证，以及统一评估器的 `GenerationProtocol`，默认启用 vLLM 精确重复检测：**16–128 token 的片段连续完全重复 8 次，就结束该回答。** 公共 launcher 覆盖 math、code、IF 入口；统一评估协议覆盖其配置的数据集。初始、周期、末次及独立验证是否执行仍由原有训练配置决定。训练和验证保留独立关闭开关；正在运行的进程、历史配置和原始结果不变。

在 1,280 条历史短程回答的离线回放中，这组参数截断 51 条，减少 10.16% 的输出 token，产生 1 条判对→判错、1 条判错→判对。它是优先降低已观察误停损失的选择，不是无损保证；这组离线回放不能估计新训练轨迹上的误停率。四卡训练接入已通过单轮验收；启停检测的配对 GPU 加速和独立样本上的风险上界尚未测量。

## 配置位置与训练调用路径

默认值定义在[公共 launcher](../../../launchers/_run_single_teacher_fsdp2.sh)，不在独立验证脚本里单独设置。正常训练的调用路径是：

```text
run_math_opd_ablation_fsdp2.sh
  → launchers/math_experiment.py
  → launchers/_run_single_teacher_fsdp2.sh
  → rollout.repetition_detection（训练）／rollout.val_kwargs.repetition_detection（验证）
  → agent loop 按 validate 标记独立选取
  → vLLM SamplingParams
```

| 层次             | 文件与职责                                                                                                                                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 训练入口         | [run_math_opd_ablation_fsdp2.sh](../../../run_math_opd_ablation_fsdp2.sh) 经 [math_experiment.py](../../../launchers/math_experiment.py) 选择实验预设，再进入公共 launcher                                    |
| 默认值与参数传递 | [_run_single_teacher_fsdp2.sh](../../../launchers/_run_single_teacher_fsdp2.sh) 在验证预算附近定义环境变量，分别写入训练 `rollout.repetition_detection` 与验证 `val_kwargs.repetition_detection`              |
| 周期性验证       | [trainer_base.py](../../../../../verl/trainer/ppo/v1/trainer_base.py) 在验证时机调用 `_validate()`，为验证 batch 设置 `validate=True`                                                                         |
| 训练／验证采样   | [agent_loop_tq.py](../../../../../verl/trainer/ppo/v1/agent_loop_tq.py) 按 `validate` 分别选取训练或验证重复参数；[legacy agent loop](../../../../../verl/experimental/agent_loop/agent_loop.py) 采用相同约定 |
| 引擎执行         | [vllm_async_server.py](../../../../../verl/workers/rollout/vllm_rollout/vllm_async_server.py) 将字典转换成原生 `RepetitionDetectionParams`，交给 vLLM 停止逻辑                                                |

从当前源码重新启动 E55/E56/E57 等预设时，训练和周期 valid 都默认使用该规则，不需要额外调用独立验证脚本。这不描述这些编号的历史运行；复现历史自然终止协议时，应显式关闭训练和验证检测。E81/E82 与阶段链预设已去掉内置关闭覆盖。

本地 rollout-policy 通过 [runtime.py](../../../../../verl/experimental/rollout_policy/runtime.py) 将训练参数写入实际采样计划，[identity.py](../../../../../verl/experimental/rollout_policy/identity.py) 将规则纳入恢复校验。统一评估器通过 [GenerationProtocol](../../../../../verl/trainer/evaluation/config.py) 声明、校验和传递参数；它有独立采样协议，不读取 `val_kwargs`。三条统计路径的原因计数与比例共用 [finish_reason.py](../../../../../verl/utils/metric/finish_reason.py)。

## 默认参数与生效范围

| 环境变量                       | 公共入口默认值 | 含义                          |
| ------------------------------ | -------------- | ----------------------------- |
| `TRAIN_REPETITION_DETECTION`   | `true`         | 启用训练重复停止；与验证独立  |
| `TRAIN_REPETITION_MIN_PATTERN` | `16`           | 训练候选片段最小长度          |
| `TRAIN_REPETITION_MAX_PATTERN` | `128`          | 训练候选片段最大长度          |
| `TRAIN_REPETITION_MIN_COUNT`   | `8`            | 训练连续完全重复次数          |
| `VAL_REPETITION_DETECTION`     | `true`         | 启用验证重复停止              |
| `VAL_REPETITION_MIN_PATTERN`   | `16`           | 候选重复片段的最小 token 长度 |
| `VAL_REPETITION_MAX_PATTERN`   | `128`          | 候选重复片段的最大 token 长度 |
| `VAL_REPETITION_MIN_COUNT`     | `8`            | 完全相同片段的连续出现次数    |

同时开启训练和验证时的最终配置：

```yaml
actor_rollout_ref:
  rollout:
    repetition_detection:
      min_pattern_size: 16
      max_pattern_size: 128
      min_count: 8
    val_kwargs:
      repetition_detection:
        min_pattern_size: 16
        max_pattern_size: 128
        min_count: 8
```

| 使用方式                                                      | 默认是否生效                                      |
| ------------------------------------------------------------- | ------------------------------------------------- |
| 公共数学训练入口的 valid，包括初始、周期和末次验证            | 是，前提是该次验证按训练配置执行                  |
| 独立数学 student/teacher 验证入口                             | 是，复用相同 launcher                             |
| 训练数据的 rollout                                            | 是；`TRAIN_REPETITION_DETECTION=false` 可独立关闭 |
| 公共 launcher 的 code、IF 领域                                | 是                                                |
| 直接运行 `python -m verl.trainer.main_ppo`，绕过公共 launcher | 否，通用 YAML 默认 `null`；需要显式配置上面的字段 |
| 统一评估器的 `GenerationProtocol`                             | 是；使用协议字段 `repetition_detection=null` 关闭 |
| 研究脚本直接调用 vLLM                                         | 不自动继承，需要查看该脚本实际参数                |
| 原生 vLLM 并发扫描中的 verl 复核                              | 显式关闭，以匹配其原生测速停止协议                |

该接入支持 vLLM 单轮 agent，包括本地 rollout-policy 的 single、trajectory mixture 与 vLLM token fusion 请求。HF reference fusion 暂不支持该检测，开启时明确报错；如需该后端须显式关闭训练检测。其他推理后端或多轮配置同样明确报错。参数必须为正整数，`min_count >= 2`，最小周期不能大于最大周期。已有运行不会热加载这些文件：默认值适用于从当前源码重新启动的进程，显式环境变量或末尾 Hydra override 优先。

## 启动、关闭与覆盖

从仓库根目录执行。下面使用 `DRY_RUN=1` 查看命令，不启动 GPU；实际训练时去掉它，并按已有约定配置设备和模型路径。

```bash
# 正常训练：训练与 valid 自动使用 16–128/8。
DRY_RUN=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E56

# 显式开启训练重复停止；valid 保持默认开启。
DRY_RUN=1 TRAIN_REPETITION_DETECTION=true \
  bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E61

# 独立验证：使用相同默认值。
DRY_RUN=1 bash examples/mirroropd/run_math_validation_fsdp2.sh student

# 不含启发式停止的对照。
DRY_RUN=1 TRAIN_REPETITION_DETECTION=false VAL_REPETITION_DETECTION=false \
  bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E56

# 参数研究：显式扩大到 256，保持其他参数。
DRY_RUN=1 VAL_REPETITION_MAX_PATTERN=256 \
  bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E56
```

统一评估器将 `repetition_detection` 放入 `GenerationProtocol` 的采样元数据、协议哈希与 artifact 身份。MiMo、阶段链和 Qwen prefix launcher 将 `VAL_REPETITION_*` 同步写入各生成协议；直接配置统一评估器时使用 `trainer.evaluation.protocols.<name>.repetition_detection`。显式 `null` 保留自然终止；切换规则后须生成新的 bootstrap，旧 artifact 的身份校验不允许跨协议复用。

训练可用 `TRAIN_REPETITION_DETECTION=false` 或末尾覆盖 `actor_rollout_ref.rollout.repetition_detection=null` 关闭。普通验证也可以在命令末尾覆盖 `actor_rollout_ref.rollout.val_kwargs.repetition_detection=null` 来关闭；统一评估器须覆盖相应 `trainer.evaluation.protocols.<name>.repetition_detection`，不会读取普通验证字段。检查最终命令或保存的配置时，应看到三个参数 `16/128/8`；仅设置 `VAL_REPETITION_MAX_PATTERN` 不会覆盖显式的 `VAL_REPETITION_DETECTION=false`。

## 实际停止行为

检测只检查输出 token，不包含 prompt。设候选片段长度为 p，默认要求末尾的 8p 个 token 能分成八个完全相同、紧邻的片段；p 遍历 16 到 128。它不识别语义相似，也不要求片段边界是自然语言句子。[^vllm-utils]

- 21-token 算式连续重复八次可以触发。
- 一个 token 连续出现八次不会触发；连续出现 128 次可以按 16-token 块触发。
- 每轮修改编号、数字或措辞的推导可能漏检；完整周期超过 128 的循环也可能漏检。
- EOS、显式停止 token 和长度上限的原生检查优先；重复停止不追加或伪造 EOS。

终止后保留已经生成的前缀、token 长度和 verifier 分数，不丢弃该回答、不改变样本分母、不自动续跑。训练沿用真实前缀及其 loss mask；重复停止按正常完成处理，不触发失败补题。重复惩罚保持原值，不通过修改 logits 阻止某个 token。也不使用 `min_tokens` 来延后检测，因为它会同时影响正常停止。

## 指标、风险与验证证据

训练使用 `rollout/finish_reason/`，普通验证使用 `eval/<dataset>/generation/finish_reason/`，统一 accuracy 使用 `eval/accuracy/<dataset>/finish_reason/`。三条路径共用同一统计函数，记录 `response_count`、各原因的 `*_count` 与 `*_fraction`，以及 `truncated_count/fraction`。训练分母为完整 single-turn 提交的实际回答数；普通验证沿用预算槽位分母，未返回槽位也计入 `unknown`，可结合 `materialized_response_count` 核对返回覆盖率；统一 accuracy 要求完整样本覆盖。已有回答缺失或无法识别的停止原因同样计入 `unknown`。原有统一 accuracy 的 `length_finish_fraction` 保留原定义。主要字段为：

- `length_fraction`：后端以 `length` 结束的比例，包含后端实际长度／上下文预算限制，不用返回长度猜测。
- `repetition_fraction`：以精确重复检测结束的比例。
- `truncated_fraction`：`length_fraction + repetition_fraction`，表示这两类已观察到的截断。
- `stop_fraction`：后端报告 `stop` 的比例；不混入只报告 `completed` 的未知具体原因。
- `unknown_fraction`：停止原因未报告或无法识别的比例；它不计入正常停止。完整训练提交即使全部缺少原因，也输出 `unknown_fraction=1`；不完整提交仍省略整批原因统计。

单轮训练还记录 `rollout/repetition_detection/enabled_fraction`：实际请求中开启检测的比例；任一回答缺少该观测时省略该字段，不能把缺失解读为关闭。规则参数由 resolved config 和统一评估协议元数据保留。

开启检测后，原先可能触顶的回答会被更早截断，`length_fraction` 因而可能下降。应同时查看 `repetition_fraction`、`truncated_fraction`、`stop_fraction`、未知比例与 accuracy，不能单独据 cap hit 下降判断停止能力改善。`truncated_fraction` 也不估计关闭检测时的触顶率：被重复截断的回答原本可能继续循环，也可能恢复。`response_length/padded_width_fraction` 只比较本批 padding 宽度，不是长度上限触发率。

逐条验证记录的 `finish_reason` 为 `repetition`；后端输出另有 `backend_stop_reason=repetition_detected`。检测器截断造成的长度下降不能当作模型学会停止。训练阶段开启后，后续 teacher 评分与 reuse 训练都基于截断后的同一前缀，因此它会改变训练数据分布。不同模型或 κ 的对照应使用同一评测规则，并保留关闭检测的对照。

选定参数的判分损失为 1/1,280，即这批回答的 0.078 个百分点；同时有一条评分增加，不能抵消那条损失。正常推导误报、真正循环后的恢复以及 verifier 提取位置变化需要分别审计。正式统计沿用统一 verifier，不用标准答案决定在线停止，也不把出现 boxed 答案当作无风险停止证明。检测准确不等于停止安全。

已有 CPU 检查覆盖实验启动后的配置解析、TQ/legacy 在训练和验证之间隔离参数、vLLM 参数转换、原始 token 保留和独立停止指标。验证测试见[launcher 配置](../../../../../tests/examples/test_mirroropd_math_launcher_on_cpu.py)与[agent loop 采样](../../../../../tests/experimental/agent_loop/test_validation_logprobs_on_cpu.py)。这些检查确认调用路径，不替代启停检测的配对 GPU 加速测量。新增训练接入的 CPU 检查为 13 项采样路由与 10 项 launcher 检查，全部通过。

四卡 GPU 验收中，256 题×4 条回答共 1,024 条，21 条以 `repetition` 停止，7 条触顶，996 条原生停止；随后完成 32 次有限梯度的有效更新。独立强制重复对照在第 128 个 token 触发停止，保留真实 token 且没有追加 EOS。完整设置、并发扫描和限制见[Qwen n=4 批量训练验收](../performance/full-vocab/training.md#qwen-n4-批量训练验收)。这批训练回答没有关闭检测的续跑对照，不能给出训练误停率或净 accuracy 收益。

## 实验与调研入口

- [重复停止参数回放与误停分析](../performance/inference/2026-10-30-重复停止参数回放与误停分析.md)：样本来源、全部精确匹配参数、完整步骤与 MD-OPD 压缩规则、人工复核、恢复案例、AUC 边界和复算命令。
- [验证提前停止方案与实现调研](../performance/inference/2026-10-30-验证提前停止方案与实现调研.md)：PUMA、LYNX、DEER、Gemini CLI 等公开实现的实际状态、检测频率与容量估算、尚未接入的方案。

逐条数据、原始来源说明和参考源码快照保留在同目录 `../performance/inference/data/` 下，不随默认参数改变而重写。

### 参考文献

[^vllm-utils]: vLLM contributors. Scheduler repetition detection and stopping conditions. [Scheduler utilities](https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/utils/).
