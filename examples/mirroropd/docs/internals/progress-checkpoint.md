# 更新事实、进度与 checkpoint

本页负责更新事件和持久状态的含义。空间维度的指标归约由[指标系统](metrics.md)负责，完整训练时序见[训练循环](training-loop.md)。

## 四种独立事件

| 事件 | 所属模块 | 含义 |
| --- | --- | --- |
| Outer step | Trainer | 一轮外层调度 |
| Update attempt | TrainingWorker | 一次 optimizer 更新尝试 |
| Applied update | Engine | optimizer 实际执行 |
| Scheduler call | Engine / worker 调用边界 | scheduler 实际推进 |

Engine 通过 `optimizer_step_applied` 提供 `True/False/None`。`None` 表示未知，不用梯度范数或 loss 是否有限推断。Worker 在 DP 汇集时用 [execution_consensus](../../../../verl/workers/update_records.py) 要求各 rank 事实一致；数量不完整、非法类型或 rank 间冲突会报错。

[UpdateResult](../../../../verl/workers/update_records.py) 将执行事实与 metrics 分开。`take_engine_execution` 和 `consume_legacy_status` 仅在边界兼容旧的 metric-carried 字段；新路径直接传独立事实。

## 进度和日志坐标

[OptimizerProgress](../../../../verl/trainer/ppo/v1/progress.py) 保存 attempts 与 applied 两个非负计数，要求已知 applied 不大于 attempts。`advance` 传播未知值：某次实际执行状态未知后，绝对成功次数也未知，不用后续观测反推。

[update_log_events](../../../../verl/trainer/ppo/v1/progress.py) 为每条更新记录附加 `step/outer`、`step/update` 和已知时的 `step/optimizer`。Critic 使用独立坐标。旧 checkpoint 缺少绝对计数时，日志仍可继续，但不伪造绝对 update 轴。

例如一次 actor 调用有四次尝试，其中第三次跳过，则 attempts 增加 4、applied 增加 3。Scheduler 是否推进由实际调用位置决定，不能从 4 或 3 推算。当前 mini-batch 迭代器只在末次更新调用后请求推进 scheduler；日志分别记录用于本次更新的 LR 和推进后的下一轮 LR。

## 保存和恢复的分工

| 状态 | 所属路径 |
| --- | --- |
| Student 参数、optimizer、scheduler | Engine checkpoint 保存和加载 |
| Outer step、actor/critic 尝试与成功计数 | Trainer 的 `trainer_state.json` |
| Teacher | 从配置的固定模型重新初始化；不并入 student checkpoint |
| Rollout 权重 | 恢复 student 后经 checkpoint manager 同步 |
| 未完成 prompt | Trainer 的 inflight 重发流程 |

[save_optimizer_progress](../../../../verl/trainer/ppo/v1/progress.py) 写临时文件后原子替换 `trainer_state.json`，schema version 为 1。[load_optimizer_progress](../../../../verl/trainer/ppo/v1/progress.py) 校验 schema、计数和 checkpoint 的 global step；文件缺失时返回未知计数并警告，损坏或不一致时拒绝加载。

Scheduler 状态由 engine 保存与恢复，Trainer 不根据 optimizer 计数重建 scheduler。Student 的冻结 teacher head 缓存也不属于 checkpoint 参数。

## 恢复保证与验证

[test_progress_on_cpu.py](../../../../tests/trainer/ppo/v1/test_progress_on_cpu.py) 覆盖计数、未知传播和文件校验；[test_update_records_on_cpu.py](../../../../tests/workers/test_update_records_on_cpu.py) 覆盖执行事实一致性；[test_full_vocab_fsdp2.py](../../../../tests/special_distributed/test_full_vocab_fsdp2.py) 保存参数、完整 Adam 和 scheduler，再恢复并对同一 batch 执行下一次更新，逐 rank 精确比较状态。

这种下一次更新等价验证不意味着随机生成、数据排队或任意故障点的端到端训练能够逐 token 重放。真实 1.5B 短训练和小模型恢复测试的覆盖范围分别见[验收报告](../refactoring/2026-10-25-主分支四卡验收.md)。
