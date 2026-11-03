# 运行接口迁移指南

适用范围：从 `402f6353` 及此前接口迁移到重构分支的当前接口。当前架构见[总览](README.md)，变更依据和分组验收见[重构报告](../../archive/engineering/2026-11-01-运行合同审计与重构方案.md)。指南只描述新运行和开发调用者的升级；旧实验及旧 checkpoint 恢复使用对应冻结代码，原始数据不自动转换。

## Full-vocab 结果生产者

自定义数值后端或研究工具返回 `FullVocabObjectiveResult` 时，必须传入 `objective_rows`（无行观测时为 None）及 `local_objective_sum`（未加权、未折扣、不含 future score 的 detached local 总量）。Scalar-only 后端没有行值时显式传入自己的 local 总量；空 rank 提供标量零。`FullVocabOutput.from_result` 仅接受这个类型，不再接受旧三字段 duck-typed 对象或从训练 surrogate 回退猜测 local 指标。活动精度研究工具已使用显式字段；冻结源码仍使用各自版本的合同。字段语义见[指标合同](metrics.md#当前目标的指标矩阵)。

## 配置与数值后端

| 原表达                             | 当前表达                                | 注意事项                                            |
| ---------------------------------- | --------------------------------------- | --------------------------------------------------- |
| `loss_mode=full_reverse_kl`        | `loss_mode=mirror`，MDPO/full_vocab/η=1 | 数学函数 `full_reverse_kl` 保留，退役的是运行配置名 |
| 蒸馏注册名 `kl`／`low_var_kl`      | `k1`／`k3`                              | 不改变通用 PPO 的 `kl_penalty` 字段含义             |
| 传入或序列化 `loss_settings`       | 删除该输入字段                          | 它是私有派生状态的只读属性，mirror 参数只读         |
| `fused_candidate`                  | `precompute_fused`                      | 旧执行模块和别名已经删除，离线 reader 仍识别旧标签  |
| CPU／FP64 请求 fused，依赖自动回退 | 显式选择 `precompute` 或 `dense`        | fused 要求 CUDA FP16/BF16/FP32，空 CPU shard 也拒绝 |

Full-RKL 的配置增量如下，合并到已有 shared-sync 配方后使用；它不是完整训练 YAML：

```yaml
distillation:
  distillation_loss:
    loss_mode: mirror
    mirror:
      objective: mdpo
      estimator: full_vocab
      kappa: 0
      eta: 1
      diagnostics: false
    use_policy_gradient: false
    use_task_rewards: false
    policy_loss_mode: vanilla
    advantage_source: current
    loss_max_clamp: null
    log_prob_min_clamp: null
    full_vocab:
      backend: precompute_fused
```

不更改历史 E 编号的科学配置。E59–E62 是对应 E55–E58 的融合实现预设，模型、数据和预算的对应关系见[训练设置](../performance/full-vocab/2026-11-01-EOS融合实现与训练验收.md#presets-e59e62eos-补偿融合对照)。配置覆盖后的真实设置仍以 resolved YAML 为准。

## 评估入口

所有组件只从 `trainer.evaluation` 构造。选择 `_target_` 后，其余字段由对应组件的严格 options 校验，未知字段直接报错。

| 退役字段                                                  | 迁移动作                                       |
| --------------------------------------------------------- | ---------------------------------------------- |
| `trainer.readout_callback_class`                          | 移到 `trainer.evaluation._target_`             |
| `rkl_readout`／`source_history_readout`                   | 字段移到 evaluation；使用 SourceHistoryReadout |
| `hf_reload_readout`／`rkl_probe`／`scientific_evaluation` | 字段移到 evaluation；选择原功能的显式组件      |
| Source/history 的 `every`                                 | 展开为 `source_every` 和 `history_every`       |
| Source/history 平铺 `audit_state`                         | 移到唯一的 `score.audit_state`                 |

退役字段应从 trainer 配置中移除，即使值是 null 也会被拒绝。Source/history 和公共 suite 自己拥有评估调度，启用时关闭 `trainer.test_freq`。具体目标、配置增量及限制只在[评估包 README](../../../../../verl/trainer/evaluation/README.md)维护。

公共 suite 保留 bootstrap 初始化；source/history 在训练内建立初始银行，不因入口统一而额外生成公共 benchmark 或 teacher accuracy。Probe、HF reload 和科学扩展各自声明协议和能力。

## 银行与生成产物

新银行绑定完整 checkpoint 内容与有效 engine_contract。权重、嵌套模板、EOS、支持集或 sampler 合同改变时，应创建新产物；相同内容迁移目录可以复用。不能给旧银行补新 hash 或新字段来宣称其满足当前合同。

冻结生成的 `--resume-generation` 仅在身份、协议、seed 与采样数量完全匹配时补缺失样本，不恢复训练中的模型或 Adam。评估使用 current learner 的已验收单模型采样器；公共 suite 可通过独立评估引擎服务非 current 的训练来源。Source/history 的 val-loader 路径仍要求默认 current 训练策略。银行复用、生成恢复和训练 checkpoint 恢复是不同操作，详见评估包说明。

## 内部调用者和扩展

| 原调用方式                                                 | 当前合同                                                             |
| ---------------------------------------------------------- | -------------------------------------------------------------------- |
| `hidden.prediction_positions` 转导出                       | 从 `verl.workers.utils.padding` 导入                                 |
| config 中的 `global_batch_info`、`bind_legacy_config()`    | 显式传 `LossContext`；保留原全局总体                                 |
| 多套 `full_kl_sum/mirror_objective_sum/termination_kl_sum` | `model_output["full_vocab"]` 返回 `FullVocabOutput`                  |
| metrics 中的 `_optimizer_step_applied`                     | 独立字段 `optimizer_step_applied`；未知填 None，缺字段报错           |
| `configure_full_vocab` 不传目标                            | 必须显式传解析后的 hidden-payload spec                               |
| 元数据名称的 string/list 自动转换                          | 非空、无重复的 tuple；payload_kind 是唯一数据类型来源                |
| 回调只实现零散方法，由 Trainer 探测                        | 继承 ReadoutCallback，使用 `__init__(*, config, options)` 并声明能力 |

FullVocabOutput 使用可被 FSDP pytree 识别的 NamedTuple，不能改回普通 dataclass 而遗漏梯度 hook。Policy 的拒绝采样 mask 不得重新定义归一化总体；GMPO 仍是序列级目标，不改成 token 平均。

回调声明 `handles_validation`、`requires_full_batch`、`owns_final_weight_sync`。休眠与恢复交给 RolloutService，避免绕过驻留版本和 sampler 安装；不要通过“evaluation 字段存在”推断这些行为。

## 验证迁移

使用[共享运行说明](../training/runtime.md)中的原生环境与资产设置，先做配置预览，核对 canonical 目标、显式 backend、唯一评估入口和科学预算。以下命令从仓库根执行，只解析配置，不启动训练：

```bash
CONFIG_ONLY=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E59
CONFIG_ONLY=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E62
```

修改内部调用者后，应验证全局分母、空 rank、loss/VJP、更新执行事实和恢复；修改评估组件时，还要验证银行身份、任务频率、失败时不发布完成标记及 sampler 恢复。已有通过范围见重构报告；source/history 最新生命周期仍没有新增一轮完整 vLLM accuracy 验收，不能把配置通过当作完整实验通过。
