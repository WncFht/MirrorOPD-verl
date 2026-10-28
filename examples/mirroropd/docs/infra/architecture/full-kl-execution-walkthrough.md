# Full-vocab OPD、MDPO、BPO：执行源码阅读路线

本文是从配置到参数更新的行级阅读路线。执行机制分别由训练循环、教师、数据、full-vocab 接入、指标与恢复页面维护；数学推导见[目标与梯度](full-kl-walkthrough.md)，数值实现见[算子走读](full-kl-kernel-walkthrough.md)。源码版本与引用约定统一见[主篇说明](full-kl-walkthrough.md)。

## 阅读顺序

先看[整轮控制流](代码走读.md#整轮控制流)与[actor 内部时序](代码走读.md#actor-内部时序)，再按下表打开代码。每行只列出该处应核对的对象，详细机制沿右列进入。

| 顺序 | 源码与阅读区间 | 核对对象 | 机制正文 |
| --- | --- | --- | --- |
| 1 | [_distillation_args.sh](../../../launchers/_distillation_args.sh#L23)，23–52 行 | 方法名如何形成 objective、estimator、eta、backend | [配置入口](training-loop.md#配置入口) |
| 2 | [resolve_distillation_spec](../../../../../verl/workers/config/distillation_spec.py#L79)，79–101 行；[DistillationLossSettings](../../../../../verl/workers/config/distillation_spec.py#L30)，41–49 行 | payload_kind、anchor_payload、teacher_only_kl | [统一目标分派](full-vocab.md#统一目标分派) |
| 3 | [DistillationLossConfig](../../../../../verl/workers/config/distillation.py#L125)，125–154 行；[DistillationConfig.validate_shared_sync](../../../../../verl/workers/config/distillation.py#L434)，434–490 行 | 目标组合与资源约束分别校验 | [接口与运行约束](full-vocab.md#接口与运行约束) |
| 4 | [ActorRolloutRefWorker.init_model](../../../../../verl/workers/engine_workers.py#L755)，755–780 行 | loss callback、adapter、独立 teacher head | [初始化与 loss 注册](training-loop.md#初始化与-loss-注册) |
| 5 | [PPOTrainer._step_once](../../../../../verl/trainer/ppo/v1/trainer_base.py#L684)，684–736 行 | 固定本轮样本；teacher 在 actor 前完成；mirror 跳过旧 advantage | [从采样到更新](training-loop.md#从采样到更新) |
| 6 | [PPOTrainer._make_shared_teacher_config](../../../../../verl/trainer/ppo/v1/trainer_base.py#L488)，488–508 行；[SharedTeacherScorer.score](../../../../../verl/trainer/distillation/shared_teacher.py#L25)，25–64 行 | 相同 token-to-ID、独立 metadata、等待与卸载 | [教师评分](teacher.md#评分顺序)、[输入协议](teacher.md#输入与-sampled-输出协议) |
| 7 | [prediction_positions](../../../../../verl/workers/utils/padding.py#L34)，34–51 行 | response target 到预测 hidden 的 shift、局部与 packed 坐标 | [预测坐标](data-and-loss.md#两种布局与预测坐标) |
| 8 | [FSDPEngineWithLMHead._full_vocab_output](../../../../../verl/workers/engine/fsdp/transformer_impl.py#L1236)，1236–1263 行；[HiddenBatchIO.publish](../../../../../verl/trainer/distillation/hidden_io.py#L90)，90–113 行 | 选中 hidden、CPU payload、按样本 key 发布 | [教师有效预测行](full-vocab.md#教师只发布有效预测行) |
| 9 | [ActorRolloutRefWorker.update_actor](../../../../../verl/workers/engine_workers.py#L847)，847–866 行；[anchor_scope](../../../../../verl/trainer/distillation/anchor.py#L40)，40–97 行 | q 的捕获覆盖整批，scope 包住全部 reuse | [Anchor 准备与驻留](teacher.md#anchor-准备与模型驻留) |
| 10 | [FrozenAnchorHead](../../../../../verl/trainer/distillation/anchor.py#L14)，14–27 行；[FSDPEngineWithLMHead._full_vocab_output](../../../../../verl/workers/engine/fsdp/transformer_impl.py#L1239)，1239–1248、1273–1288 行 | 独立 head storage、位置与 version 匹配 | [Anchor 生命周期](teacher.md#anchor-准备与模型驻留) |
| 11 | [HiddenBatchIO.__iter__](../../../../../verl/trainer/distillation/hidden_io.py#L70)，70–88 行 | shuffle 后按当前 key 读取 T/q，退出 micro-batch 后移除注入字段 | [按 actor micro-batch 读取](full-vocab.md#按-actor-micro-batch-读取) |
| 12 | [full_vocab_forward](../../../../../verl/models/transformers/full_vocab.py#L29)，29–37 行；[full_vocab_head_forward](../../../../../verl/models/transformers/full_vocab.py#L21)，21–26 行 | root/head FSDP hooks 与选行后的投影 | [Head adapter](full-vocab.md#head-adapter-与-fsdp-hooks) |
| 13 | [full_vocab_objective_result](../../../../../verl/trainer/distillation/full_vocab.py#L359)，359–402 行 | endpoint 或一般 mirror；raw sum 与 detached rows | [Dispatcher](full-kl-kernel-walkthrough.md#1-dispatcher-的逐段阅读) |
| 14 | [FSDPEngine.forward_backward_batch](../../../../../verl/workers/engine/fsdp/transformer_impl.py#L745)，750–761 行；[LossContext.normalize_token_sum](../../../../../verl/workers/utils/loss_context.py#L35)，35–38 行 | mini-batch 全局分母在 micro-batch 切分前产生 | [mask 与分母](data-and-loss.md#三种-mask-与分母) |
| 15 | [distillation_loss](../../../../../verl/trainer/distillation/losses.py#L234)，234–269 行；[distillation_ppo_loss](../../../../../verl/trainer/distillation/losses.py#L188)，188–208 行 | eta 只除一次；纯蒸馏不加入任务 PPO | [归一化](data-and-loss.md#三种-mask-与分母)、[两套 advantage](training-loop.md#两套-advantage-的消费者) |
| 16 | [BaseEngine.train_batch](../../../../../verl/workers/engine/base.py#L113)，128–139 行；[FSDPEngine.optimizer_step](../../../../../verl/workers/engine/fsdp/transformer_impl.py#L833)，833–874 行 | zero_grad、累计 backward、gradient clipping 与 applied | [实际更新](training-loop.md#从一次-actor-调用到实际更新) |
| 17 | [TrainingWorker.train_mini_batch](../../../../../verl/workers/engine_workers.py#L331)，331–383 行；[TrainingWorker.train_batch](../../../../../verl/workers/engine_workers.py#L479)，479–484 行 | mini-batch/epochs 的次数；scheduler 的末次标记 | [进度事件](progress-checkpoint.md#四种独立事件) |
| 18 | [FSDPEngineWithLMHead._full_vocab_output](../../../../../verl/workers/engine/fsdp/transformer_impl.py#L1308)，1308–1326 行；[distillation_loss](../../../../../verl/trainer/distillation/losses.py#L234)，234–269 行 | objective、诊断、位置桶的独立出口 | [指标矩阵](metrics.md#当前目标的指标矩阵)、[位置统计](metrics.md#相对位置统计) |
| 19 | [anchor_scope](../../../../../verl/trainer/distillation/anchor.py#L91)，91–97 行；[ActorRolloutRefWorker.update_actor](../../../../../verl/workers/engine_workers.py#L853)，853–856 行 | q holder 与 teacher head GPU 引用的各自清理 | [资源所有权](teacher.md#资源所有权)、[冻结 head](full-vocab.md#冻结-head-与-io-生命周期) |
| 20 | [PPOTrainer.fit](../../../../../verl/trainer/ppo/v1/trainer_base.py#L514) | 更新记录之后的 checkpoint、同步、验证与 TQ 清理 | [outer 完成边界](progress-checkpoint.md#完成进度与云端状态) |

核对时保持一次具体 mini-batch 的 sample keys、有效位置和 anchor version；形状相同不能证明输入属于同一前缀。

## 手工走读记录

每轮选一个有效 response token，记录以下值，再在复用后的更新中检查哪些发生变化：

- 配置：objective、eta、kappa、teacher_only_kl、anchor_payload。
- 身份：sample key、response target 下标、prediction position、anchor version。
- 数值：当前 hs/ws、冻结 ht/wt 与可选 hq/wq、raw row、全局 N、最终上游缩放。
- 执行：本次 mini-batch 索引、optimizer applied、scheduler 是否推进、日志所属 update。

这份记录用于把[公式中的符号](full-kl-walkthrough.md#1-三种方法共用的计算对象)与实际张量对应起来；各字段的完整定义维护在[数据合同](data-and-loss.md#字段来源布局与消费者)。
