# 训练循环与配置入口

本文说明当前 shared-sync OPD 主链的调度。数学目标由 [sampled](sampled-opd.md) 和 [full-vocab](full-vocab.md) 定义，调用级张量布局见[数据与归一化](data-and-loss.md)。

## 配置入口

领域入口调用 [公共 launcher](../../launchers/_run_single_teacher_fsdp2.sh)，合并环境变量、领域参数和命令行覆盖，再进入 [main_ppo.py](../../../../verl/trainer/main_ppo.py)。运行参数和实验预设以[训练配方](../训练配方.md)为准。

[DistillationLossSettings](../../../../verl/workers/config/distillation_spec.py) 只描述目标别名和教师数据需求。`get_distillation_loss_settings` 可以在不导入张量 kernel 的情况下查询 sampled、top-k、hidden 元数据。[DistillationConfig](../../../../verl/workers/config/distillation.py) 校验资源和组合；[PPOTrainer._setup](../../../../verl/trainer/ppo/v1/trainer_base.py) 完成训练级约束和 worker 配置。目标计算注册位于 [register_distillation_loss](../../../../verl/trainer/distillation/losses.py)，静态描述和计算函数的名称必须对应。

Student 的生成协议由其模板、tokenizer、停止设置和最终 rollout 请求共同决定。Teacher 读取原始生成 token，不通过自己的模板重新编码。EOS、截断和特殊 token 的实验依据见[对齐诊断](../failure-reproduction/2026-10-24-OPD实现验收与对齐诊断.md#1-运行时协议与代码入口)。

## 从采样到更新

[PPOTrainer.fit](../../../../verl/trainer/ppo/v1/trainer_base.py) 管理 outer step。其 [PPOTrainer._step_once](../../../../verl/trainer/ppo/v1/trainer_base.py) 按以下顺序处理一次局部训练批次：

1. 调用 [ReplayBuffer.sample](../../../../verl/trainer/ppo/v1/replay_buffer.py) 获取可训练样本；同步模式的 `on_sample_end` 休眠 rollout 副本。
2. 在需要时完成 colocated reward，平衡 DP batch，然后调用共享教师评分。
3. Sampled 模式计算 old-log-prob 和 advantage；Full-vocab 跳过这两步。通用 Trainer 的 ref/critic 分支仍存在，但 shared-sync 配置不启用它们。
4. 调用 actor 更新，收集逐次更新记录和外层统计。

Verifier 可以继续用于评测与诊断；`use_task_rewards=False` 控制任务奖励不进入蒸馏训练目标，不能以是否出现 reward 字段判断目标是否混合奖励。

## 从一次 actor 调用到实际更新

[ActorRolloutRefWorker.update_actor](../../../../verl/workers/engine_workers.py) 调用 [TrainingWorker.train_mini_batch](../../../../verl/workers/engine_workers.py)。Worker 按 `mini_batch_size` 和 `epochs` 构造迭代器，每个 mini-batch 调用一次 `train_batch`。Engine 再按 token 预算拆成 micro-batch，逐个 forward/backward，在该次更新内累积梯度。

| 粒度 | 含义 |
| --- | --- |
| Outer step | Trainer 一轮调度、按需 checkpoint、同步、验证和清理 |
| Mini-batch update | 一次 optimizer 更新尝试 |
| Micro-batch | 为显存和 DP 工作量拆出的前反向片段；不各自推进 optimizer |
| `ppo_epochs` | 对该批数据的迭代次数；复用期间 student 更新、教师标签保持冻结 |

`TrainingWorker.train_mini_batch` 在最后一个 mini-batch 设置 `update_lr_scheduler=True`。因此 scheduler 调用次数与 optimizer 尝试次数独立，不能按 `ppo_epochs` 简单推断 scheduler 步数。

## 收尾、replay 与恢复

`fit` 在训练后按需保存 checkpoint，调用 [PPOTrainerSync.on_step_end](../../../../verl/trainer/ppo/v1/trainer_sync.py) 同步新 student 权重，再按频率验证、输出日志并清理本轮 TQ。教师参数不进入 rollout 权重同步。

ReplayBuffer 管理候选样本及可采样状态，`ppo_epochs` 管理已经取出的一批数据内部复用。两者不是同一个缓存生命周期。shared-sync 当前限制 `parameter_sync_step=1`，不能把异步 Trainer 的 stale-data 能力直接归到这条路径。重启时 inflight prompt 的重发和计数恢复由 Trainer 的恢复流程负责，详见[进度与 checkpoint](progress-checkpoint.md)。

## 验证入口

调用链测试见 [test_pipeline_metrics_on_cpu.py](../../../../tests/trainer/ppo/v1/test_pipeline_metrics_on_cpu.py)、[test_replay_buffer_on_cpu.py](../../../../tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py) 和 [test_reissue_inflight_on_cpu.py](../../../../tests/trainer/ppo/v1/test_reissue_inflight_on_cpu.py)。真实四卡运行的配置、退出结果和更新日志见[验收报告](../refactoring/2026-10-25-主分支四卡验收.md)。
