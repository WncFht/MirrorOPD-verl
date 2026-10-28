# 更新事实、进度与 checkpoint

Mirror 共用本文的更新计数、scheduler 与 outer checkpoint。Anchor 是 actor 调用级临时状态，不进入 checkpoint；恢复后的新调用重新建立 q。已验证两卡 engine 的后续更新一致，V1/Ray 进程冷启动恢复仍列在[待验证范围](../numerics/mirror-objectives/实现方案.md#remaining-validation)，不据此承诺 inner-update 中途恢复。

本页负责更新事件和持久状态的含义。空间维度的指标归约由[指标系统](metrics.md)负责，完整训练时序见[训练循环](training-loop.md)。

## 四种独立事件

| 事件 | 所属模块 | 含义 |
| --- | --- | --- |
| Outer step | Trainer | 一轮外层调度 |
| Update attempt | TrainingWorker | 一次 optimizer 更新尝试 |
| Applied update | Engine | optimizer 实际执行 |
| Scheduler call | Engine / worker 调用边界 | scheduler 实际推进 |

Engine 通过 `optimizer_step_applied` 提供 `True/False/None`。`None` 表示未知，不用梯度范数或 loss 是否有限推断。Worker 在 DP 汇集时用 [execution_consensus](../../../../../verl/workers/update_records.py) 要求各 rank 事实一致；数量不完整、非法类型或 rank 间冲突会报错。

[UpdateResult](../../../../../verl/workers/update_records.py) 将执行事实与 metrics 分开。`take_engine_execution` 和 `consume_legacy_status` 仅在边界兼容旧的 metric-carried 字段；新路径直接传独立事实。

## 进度和日志坐标

[OptimizerProgress](../../../../../verl/trainer/ppo/v1/progress.py) 保存 attempts 与 applied 两个非负计数，要求已知 applied 不大于 attempts。`advance` 传播未知值：某次实际执行状态未知后，绝对成功次数也未知，不用后续观测反推。

[update_log_events](../../../../../verl/trainer/ppo/v1/progress.py) 为每条更新记录附加 `step/outer`、`step/update` 和已知时的 `step/optimizer`。Critic 使用独立坐标。旧 checkpoint 缺少绝对计数时，日志仍可继续，但不伪造绝对 update 轴。

例如一次 actor 调用有四次尝试，其中第三次跳过，则 attempts 增加 4、applied 增加 3。Scheduler 是否推进由实际调用位置决定，不能从 4 或 3 推算。当前 mini-batch 迭代器只在末次更新调用后请求推进 scheduler；日志分别记录用于本次更新的 LR 和推进后的下一轮 LR。

在这条例子中，如果最后一次 `train_batch` 正常返回，scheduler 调用一次。即使最后一次也因为非有限梯度跳过，worker 仍按末次迭代标记调用 scheduler；如果计算抛异常而没有到达调用点，则不能记作已推进。`ppo_epochs` 增加的是 optimizer 尝试次数，学习率调度并不自动改为同样的频率。

## 完成进度与云端状态

`step/outer=k` 出现在 update 记录中，只说明这次更新属于第 k 轮。根据 [PPOTrainer.fit](../../../../../verl/trainer/ppo/v1/trainer_base.py) 的顺序，actor 更新之后还有按需 checkpoint、权重同步、验证、TQ 清理及 outer 日志。因此，最后一条 update 已到本轮预期次数，也不代表整轮收尾已经完成。

核对进度时分别报告最后的 update attempt/applied 计数、最后完整 outer 汇总和最后验证坐标。例如每轮 16 次更新，第 k 轮已经记录 16 次更新而 outer 汇总仍停在 k−1 时，应保留这一差别，不能补造已完成 outer。只有完整 outer、无跳步且 batch=mini-batch，并满足完整轨迹和无额外填充等条件时，才能用 outer×reuse 校验成功更新数。

云端 summary 各字段可能来自不同记录：最后的训练 loss、上一轮 rollout 长度和更早的验证分数不一定共享同一坐标。离线日志的上传/同步时间也可能晚于事件时间。判断实时训练是否运行、成功退出或为何失败，需要结合训练端进程、退出结果和异常日志；W&B 的 `running/failed/finished` 或某个有限 loss 不能独立证明这些事实。按同一 history 事件及其坐标比较指标，缺失尾段明确标为未同步或未记录。

## 保存和恢复的分工

| 状态 | 所属路径 |
| --- | --- |
| Student 参数、optimizer、scheduler | Engine checkpoint 保存和加载 |
| Outer step、actor/critic 尝试与成功计数 | Trainer 的 `trainer_state.json` |
| Teacher | 从配置的固定模型重新初始化；不并入 student checkpoint |
| Mirror anchor、q/T hidden/head 缓存 | 调用级或 outer 级临时状态，不写入 student checkpoint；恢复后按新调用重建 |
| Rollout 权重 | 恢复 student 后经 checkpoint manager 同步 |
| 未完成 prompt | Trainer 的 inflight 重发流程 |

[save_optimizer_progress](../../../../../verl/trainer/ppo/v1/progress.py) 写临时文件后原子替换 `trainer_state.json`，schema version 为 1。[load_optimizer_progress](../../../../../verl/trainer/ppo/v1/progress.py) 校验 schema、计数和 checkpoint 的 global step；文件缺失时返回未知计数并警告，损坏或不一致时拒绝加载。

Scheduler 状态由 engine 保存与恢复，Trainer 不根据 optimizer 计数重建 scheduler。Student 的冻结 teacher head 缓存也不属于 checkpoint 参数。

`trainer.save_freq/test_freq` 都按 outer step 判断。只有对应频率大于 0 才启用周期执行，并在正常到达最后一个训练 step 时执行一次；负值不会因为到达最后 step 而自动保存或验证。`val_before_train` 独立控制初始验证。[E44–E47](../../settings/训练设置.md#mirror-comparison) 的正式默认值是 256 outer、每 32 outer 验证、每 128 outer 保存，不做初始验证。预计保存点不是保存成功的证据，仍需检查对应日志和文件。

## 恢复保证与验证

[test_progress_on_cpu.py](../../../../../tests/trainer/ppo/v1/test_progress_on_cpu.py) 覆盖计数、未知传播和文件校验；[test_update_records_on_cpu.py](../../../../../tests/workers/test_update_records_on_cpu.py) 覆盖执行事实一致性；[test_full_vocab_fsdp2.py](../../../../../tests/special_distributed/test_full_vocab_fsdp2.py) 保存参数、完整 Adam 和 scheduler，再恢复并对同一 batch 执行下一次更新，逐 rank 精确比较状态。

这种下一次更新等价验证不意味着随机生成、数据排队或任意故障点的端到端训练能够逐 token 重放。原 Full-KL 验收见[四卡报告](../../archive/engineering/2026-10-25-主分支四卡验收.md)；sampled/full MDPO 和 full BPO 的恢复测试由 [test_mirror_fsdp2.py](../../../../../tests/special_distributed/test_mirror_fsdp2.py) 及[mirror 验收结果](../numerics/mirror-objectives/实现方案.md#validation)维护。
