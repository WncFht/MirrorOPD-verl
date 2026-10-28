# 相对 verl 上游的改动地图

本页按训练职责说明 MirrorOPD 增加了什么、沿用了什么。核对范围为固定上游基线 `263653e749306287a5032140cce3d7833c1959d0`（标签 `verl-base-2026-10-19`）至当前工作区；基础重构验收基点为 `cfd6ee38c`，本文还包含后续 mirror 实现与统一实验入口。更新日期为 2026-10-26。基线由仓库初始化时的 [UPSTREAM.md](../../../../../UPSTREAM.md) 固定；不要用以后移动的 `upstream/main` 代替它。后续维护应随实现更新本页。

## 基线已经具备的训练骨架

基线已有 V1 Trainer、同步/异步调度、TransferQueue（TQ）、ReplayBuffer、AgentLoop、vLLM rollout、TrainingWorker、FSDP2 engine，以及独立教师服务和蒸馏 loss 注册。Sampled `k1 + policy gradient` 与调用 vanilla PPO 的路径也已存在。因此，理解本仓时应从 V1 的原有调用链进入，再看共享教师、显式 sampled 变体和 Full-KL 如何接入。

| 层次            | 基线能力                                              | 本仓当前增量                                                                                            | 继续走读                                                                                  |
| --------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| 配方与运行      | Hydra + `main_ppo`，通用训练配置                      | 统一 math 实验规划器及 E44–E47 完整对照、code/IF 单教师入口、原生环境、模型/数据准备和离线传输          | [训练循环](training-loop.md#配置入口)、[训练配方](../../settings/训练设置.md)             |
| Rollout 与评测  | AgentLoop、Continuous Token、vLLM、reward loop        | 显式选择 DeepSeek 协议；独立验证长度；额外 stop token；验证并发、seed、KV 容量和阶段时间观测            | [训练循环](training-loop.md#从数据集到-tq)、[实验入口](../../experiments/README.md)       |
| 教师执行        | 独立 teacher manager/server，rollout 后请求评分       | `shared_sync`：同池 FSDP2 teacher，评分后卸载；sampled 标签或 hidden payload                            | [教师生命周期](teacher.md)                                                                |
| Sampled loss    | sampled KL estimator、负 KL 作为 PG 信号、vanilla PPO | `reinforce/token_is/token_ppo` × `current/rollout`，严格配置约束、mask 与行为概率协议                   | [Sampled OPD](sampled-opd.md)                                                             |
| Full-vocab loss | 没有本仓这条 hidden/head 路径                         | 统一 MDPO/BPO 与 Full-KL endpoint、head adapter、冻结 q/T、分块 VJP、融合 target/诊断及额外教师输出支持 | [Full-vocab](full-vocab.md)                                                               |
| Mirror 目标     | 没有本仓这套 weighted target 和调用级 q 协议          | sampled/full MDPO、full BPO、κ/η resolver、anchor scope；保留 1/eta                                     | [Sampled MDPO](sampled-opd.md#sampled-mdpo)、[统一 full 目标](full-vocab.md#统一目标分派) |
| 数据与归一化    | TQ bridge、packed/nested 数据、全局 loss 聚合         | 保留样本身份；区分预测布局和 response 布局；`LossContext`；拒绝筛选后的分母契约                         | [数据与归一化](data-and-loss.md)                                                          |
| 更新与日志      | mini/micro-batch、optimizer、scheduler、checkpoint    | 执行事实独立传递；逐 update 记录；可合并 Observation；位置分桶；绝对更新计数恢复                        | [指标](metrics.md)、[进度](progress-checkpoint.md)                                        |
| Verifier 与交付 | 通用 reward 接口、追踪与保存                          | 四领域 verifier、离线日志分析、性能探针、验收和文档检查                                                 | [MirrorOPD 入口](../../../README.md)                                                      |

`trainer_sync.py` 和 ReplayBuffer 的调度骨架继续复用上游。新增教师阶段主要插入 `PPOTrainer._step_once`，新增 loss 主要从 `ActorRolloutRefWorker.init_model` 注册到 TrainingWorker。Full-KL 进一步在 FSDP engine 的 forward 中分流；它不需要另写一套 Trainer 或 optimizer。

## 训练主链的源码落点

| 改动                         | 负责的源码                                                                                                                                                                                                                         | 读代码时核对的边界                                                                       |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| 目标声明与参数校验           | [distillation_spec.py](../../../../../verl/workers/config/distillation_spec.py)、[distillation.py](../../../../../verl/workers/config/distillation.py)、[validate_sampled_token_distillation](../../../../../verl/utils/config.py) | 静态 payload 元数据、资源约束和数学模式约束分别生效                                      |
| 共享教师编排                 | [PPOTrainer._setup](../../../../../verl/trainer/ppo/v1/trainer_base.py)、[SharedTeacherScorer.score](../../../../../verl/trainer/distillation/shared_teacher.py)                                                                   | teacher 参数与 student 参数独立；actor 开始前 teacher 已卸载                             |
| Sampled 分支                 | [distillation_ppo_loss](../../../../../verl/trainer/distillation/losses.py)、[_sampled_token_loss](../../../../../verl/trainer/distillation/losses.py)                                                                             | behavior/anchor 分母、advantage detach、task reward 是否使用                             |
| Weighted target / MDPO / BPO | [targets.py](../../../../../verl/trainer/distillation/targets.py)、[objectives.py](../../../../../verl/trainer/distillation/objectives.py)、[anchor.py](../../../../../verl/trainer/distillation/anchor.py)                        | 共用目标构造、sampled primitive、full VJP 与可选 anchor scope；Full-KL 是 eta=1 endpoint |
| Full-vocab 分支              | [FSDPEngineWithLMHead._full_vocab_output](../../../../../verl/workers/engine/fsdp/transformer_impl.py)、[full_vocab_objective_result](../../../../../verl/trainer/distillation/full_vocab.py)                                      | endpoint 委托原 KL API；一般 MDPO/BPO 共用 precompute，q/T 无梯度                        |
| Hidden 运输                  | [HiddenBatchIO](../../../../../verl/trainer/distillation/hidden_io.py)、[transferqueue_utils.py](../../../../../verl/utils/transferqueue_utils.py)                                                                                 | key 随 shuffle 移动；payload 按 actor micro-batch 读取                                   |
| 参数更新                     | [TrainingWorker.train_mini_batch](../../../../../verl/workers/engine_workers.py)、[BaseEngine.train_batch](../../../../../verl/workers/engine/base.py)                                                                             | reuse 次数、micro-batch 累积、optimizer 与 scheduler 的不同频率                          |
| 观测与恢复                   | [update_records.py](../../../../../verl/workers/update_records.py)、[observations.py](../../../../../verl/utils/metric/observations.py)、[progress.py](../../../../../verl/trainer/ppo/v1/progress.py)                             | 缺测、未知、跳过与已执行不能混为一谈                                                     |

## 查阅历史变更的方法

下列命令只读取本地 Git 历史。在仓库根目录运行，可把当前源码与最初 clone 的基线直接比较：

```bash
git diff --stat verl-base-2026-10-19..HEAD -- verl examples/mirroropd tests
git diff verl-base-2026-10-19..HEAD -- verl/trainer/distillation/losses.py
git diff verl-base-2026-10-19..HEAD -- verl/trainer/ppo/v1/trainer_base.py verl/workers/engine/fsdp/transformer_impl.py
git log --reverse --oneline verl-base-2026-10-19..HEAD -- verl/trainer/distillation verl/workers/config/distillation.py
```

共享教师从 `5188828d2` 引入，显式 sampled 变体从 `d40950fc6` 引入，Full-KL 从 `a3989329e` 引入。后续 hidden IO、算子、输出支持、位置诊断及职责拆分已改变代码组织，历史提交只能用于解释来源；当前接口以本库和当前源码为准。

Sampled/full MDPO 与 full BPO 已接入实际 Trainer、worker、loss 和统一入口，普通 Full-KL 保留为 MDPO eta=1 endpoint。算法实现从 `3ce657605` 引入，target/诊断融合优化为 `749000bf2`，统一入口及 E44–E47 为 `6900d29b4`。Part、Mean、current-J、sampled BPO 仍未实现；理论文章不扩大生产支持范围。功能、GPU/短训练验收与长期训练效果分别按[实现证据](../numerics/mirror-objectives/实现方案.md#validation)核对。
