# 原生训练的评估合同

本包复用 V1／FSDP2／vLLM 的原生生成与评分，负责评估调度、Parquet 产物、CPU 判分及串行 RKL 读出。训练聚合由 actor 的 `loss_agg_mode` 指定，支持范围见 [Full-vocab 聚合](../../../examples/mirroropd/docs/infra/architecture/full-vocab.md#聚合与未来信用)；评估始终先题内 token 均值再按题平均，history 再按批等权，训练聚合与 future 配置不改变评估口径。当前架构见[总览](../../../examples/mirroropd/docs/infra/architecture/README.md)，旧配置按[迁移指南](../../../examples/mirroropd/docs/infra/architecture/runtime-migration.md)升级。

## 组件与构造

所有组件都从 `trainer.evaluation` 构造：`_target_` 指定类，其余字段是组件的严格 options。组件使用 `__init__(*, config, options)`，未知字段报错；禁用评估时使用 null。构造入口不自动转换旧命名空间。

| 组件                                                                                        | 用途                                                    | options 所有者      |
| ------------------------------------------------------------------------------------------- | ------------------------------------------------------- | ------------------- |
| [EvaluationCoordinator](coordinator.py)                                                     | 来源/history、公共 bank/current RKL、accuracy、最终导出 | EvaluationConfig    |
| [SourceHistoryReadout](source_history.py)                                                   | 只做来源留出与 history，不增加公共 bootstrap            | SourceHistoryConfig |
| [HFReloadReadout](reload.py)                                                                | 新 val-only trainer 从 HF 导出初始化后的独立验收        | HFReloadOptions     |
| [NativeRKLProbe](../../../examples/mirroropd/benchmarks/profile/native_evaluation.py)       | 原生长前缀／评分窗口的基础设施 probe                    | ProbeOptions        |
| [ScientificReadout](../../../examples/mirroropd/research/studies/prefix_fusion/callback.py) | 来源/融合研究的专用科学读出                             | ScientificOptions   |

Source/history 在训练内建立初始 source bank；公共 suite 使用预生成 bootstrap。两者共用 [native 轨迹接口](native.py)、[原生评分器](scoring.py)和 [ArtifactStore](artifacts.py)，保留各自科学工作量。

## 生命周期与资源所有权

[ReadoutCallback](../ppo/readout_callback.py) 定义 on_readout、on_before_weight_sync、on_update_records 和 validate；没有工作的 hook 使用基类空实现。三个能力分别表达独立的责任：

| 能力                   | 含义                                                 | 当前声明                                                          |
| ---------------------- | ---------------------------------------------------- | ----------------------------------------------------------------- |
| handles_validation     | 接管 Trainer 的验证入口                              | Suite、source/history、probe、HF reload 为 true；科学扩展为 false |
| requires_full_batch    | 要求 n=1、完整训练 mini-batch、full-vocab、无 critic | 公共 suite 为 true                                                |
| owns_final_weight_sync | 接管最后一轮同步，避免 Trainer 重复恢复模型          | 公共 suite 为 true                                                |

Trainer 和 RolloutService 不根据 evaluation 配置是否存在推断这些行为。Source/history 的休眠与恢复只经 RolloutService，恢复时重新安装 sampler 并维护驻留版本。新增组件声明实际能力并校验自己的 options，无需给 Trainer 增加一套配置入口。

评估生成始终使用 current learner 的已验收单模型 sampler。公共 suite 可在训练使用 S0、teacher 或 mixture 时，经独立 current 引擎评估；训练来源不改变评估来源。Source/history 的原生 val-loader 路径仍要求训练使用默认 current 策略。

## 来源留出与 history 配置

下面是合并到既有 shared-sync 配置的增量，不是完整训练配方。`data.val_files`必须只含同一个 source_file；训练 n=1，来源验证 n=1 且 response cap 与训练一致，输出目录必须为空。仍要求 fresh run。

```yaml
trainer:
  val_before_train: true
  resume_mode: disable
  test_freq: -1
  evaluation:
    _target_: verl.trainer.evaluation.source_history.SourceHistoryReadout
    output: runs/example/readout
    source_file: /absolute/path/source-heldout.parquet
    source_count: 128
    history_size: 128
    source_every: 1
    history_every: 1
    seed: 42
    score:
      audit_state: false
```

`source_every`／`history_every`分别声明频率，默认均为 1。初始 source 和最终节点强制读出，history 从第一批开始。频率不控制 history 成员更新，成员随每个已训练批更新。

`initial_bank`可指向已有、完整的 Parquet 银行目录。完整模型内容身份、题目文件 hash、输入和有效 engine_contract 必须匹配；银行在本次运行中绑定到固定 manifest hash。共享银行应位于新输出目录之外。旧 JSON 银行不直接作为新运行输入；历史 JSON 结果由验收工具只读支持，不自动迁移或覆盖。

## 公共 suite 与 bootstrap

准备过程按数据来源执行 [prepare](prepare.py)，生成不可覆盖的题目表；当前四域配方准备 6400 道训练题和 128 道来源留出，具体预算仍由配方声明。公开 benchmark 保留全量题目，RKL 面板单独按域抽样。训练候选先排除公开题目的规范化文本／数字变体家族，并核对原生输入模板的 prompt 预算。TextbookReasoning 没有书籍 ID，因此这里是题目家族划分，不是书籍间划分。

Bootstrap 的各 producer 和训练进程共用同一份 resolved verl YAML：

```bash
python -m verl.trainer.evaluation.bootstrap --config /absolute/run.yaml --producer initial
python -m verl.trainer.evaluation.bootstrap --config /absolute/run.yaml --producer teacher
python -m verl.trainer.evaluation.bootstrap --config /absolute/run.yaml --producer finalize
python -m verl.trainer.main_ppo --config-path /absolute --config-name run
```

这些命令须在已有 GPU 分配与进程清理器内运行。Bootstrap 使用本地 Ray，CPU 数量取该 YAML 的 Ray 配置；两个 producer 分别加载模型，使用同一 student 输入模板。`--tensor-parallel`可独立设置 bootstrap 的 TP；current student 使用训练进程现有的 rollout replicas。

统一流程现在默认关闭来源留出和训练history评估；代码保留，通过下面两个独立开关重新启用：

```yaml
trainer:
  evaluation:
    _target_: verl.trainer.evaluation.coordinator.EvaluationCoordinator
    source_enabled: false
    history_enabled: false
```

关闭覆盖初始化、普通边界和终点：不生成或读取source bank，不建立history抽样表，也不运行这两项RKL。`schedules.source/history`只在相应开关为true时生效，单独填写频率不会绕过开关。训练轨迹仍逐批保存一次，并保留题目不重复检查。没有其他到期评估的outer直接跳过评估，不额外park／恢复模型。

最终模型校验优先使用已启用的source bank；关闭时取已有公共initial-student bank中最多4条有效短轨迹，不补生成source回答。导出审计记录具体轨迹和银行hash；独立HF回读默认沿用这些参考。旧`source_bank`参数保留为`generation_bank`别名，旧导出记录仍可按原方式验证。

重新开启时，将对应开关设为true；source还需要`source_file`与`source`生成协议，并在新的bootstrap中准备其银行。四域launcher默认输出关闭配置，可用`--source-heldout`／`--training-history`显式开启。历史运行与冻结配置保持原样；专用SourceHistoryReadout继续作为显式诊断入口保留。

统一配置包含`_target_/output/source_file/bootstrap/datasets/protocols/score/schedules`。所有组件的 options 与 `_target_` 放在同一 `trainer.evaluation` 中；构造边界只解析一次目标，剩余字段由组件的严格类型配置验证。`datasets`逐 bench 声明 name、domain、path、verifier、n、protocol、panel_ids；`protocols`分别定义 source 和 public 的 T／top-p／top-k、回答 cap、prompt 预算和模板。配置拒绝未知字段，public prompt 超长时报错；不静默删题或截短回答。稳定采样 seed 由 seed／dataset／question／sample_index 决定，sample 0 不随 n 或 checkpoint 改变。

| 任务                           | 默认 outer 边界      | 轨迹来源                                            |
| ------------------------------ | -------------------- | --------------------------------------------------- |
| 来源留出 RKL                   | 默认关闭；启用后0与每轮             | 固定 initial-student 回答                           |
| History RKL                    | 默认关闭；启用后每轮，从1开始      | 真实训练回答，批均衡抽样                            |
| Public bank RKL                | 0／1／2，之后每 5 轮 | 固定 initial-student 与 teacher 回答，各取 sample 0 |
| Public current RKL 与 accuracy | 0／25／50            | 当前 student 回答；0 复用 initial 产物              |
| Teacher accuracy               | 0                    | Bootstrap 全量 teacher 回答                         |

初始化0建立公共固定参考；source参考仅在启用时建立；其余边界按每个 schedule 的`at/every/start/final`并集调度。同一边界只编译一次计划；由组件独占调度，trainer.test_freq 必须关闭。满足同一评分协议的轨迹合并后按 teacher hidden 字节预算分窗口，窗口内 teacher→student 串行；预算扣除仍活跃的训练 payload 和拷贝余量。推理只按 DP 数量 padding，padding 不进入科学样本统计。

## 生成、发布与判分的调度边界

同一模型版本下，所有 benchmark 共用一个有限请求流。`generation_concurrency`限制在途请求数，完成一条即可补下一条，包括下一个 benchmark 的题目。默认路由复用训练路径的 least-inflight 实现；`full_determinism`启用时由路由器按 request ID 分配。每题仍保留自己的模板、response cap、n 和 seed，模型版本检查在每条回答完成后执行。

生成任务独立推进，结果队列容量为一个请求窗口；Parquet 仍由单一消费者按`generation_chunk_size`发布。队列满后产生背压，允许持续慢磁盘限制生产速度。内存上限还包含滚动生成器暂存的已完成结果、在途请求和当前写盘 chunk，至多约`3 × generation_concurrency + generation_chunk_size`条结果；prompt 缓存按各协议的题目数限定。新增指标`generated`表示已有生成结果的样本数（包括恢复时复用的样本），原有`completed`继续表示已发布数，`inflight`表示路由器持有的请求数，包含引擎内部排队请求。

`AccuracyCollector.follow_generation`在一个受控线程中读取已发布且通过 hash 校验的 shard，再交给有界 CPU 进程池。生成产物承担判分积压，GPU 生产者只检查判分线程是否失败。CPU 结果按完成顺序消费，避免最早提交的慢题阻塞其他已完成结果。`finish`要求所跟随的 generation 已完成，核对样本覆盖并绑定其 manifest hash；`close`停止读者后才关闭进程池。

取消或失败会关闭请求补位，并等待已开始的写盘、远端 abort 和所有已提交请求确认结束。无法确认时抛出`UnsafeDrainError`，禁止切换模型。跨 benchmark 的补位仅发生在同一个模型版本内；训练 outer 的完整批、更新及权重切换屏障仍然保留。

Generation manifest 的`metadata.execution`记录调度版本、副本数、窗口、chunk、预检／发布耗时和最后提交至最后完成的时长。执行参数不进入样本身份；改变批处理顺序仍可能改变浮点执行与生成 token，不能将相同 seed 解释为逐 token 一致。实测与参数选择见[验证资源测量](../../../examples/mirroropd/docs/infra/performance/inference/validation-resources.md#native-scheduler)。

## 生成与内容身份

Bootstrap 和 current student 均通过 policy_engine_config 安装公共 sampler 与 TokenizerSupport，屏蔽没有 tokenizer 映射的输出行。有效 engine_contract 记录支持 ID、输入 tokenizer／模板、EOS、sampler 版本和执行精度；样本身份使用 generation-v2。银行与当前引擎合同不一致时，在参数更新前报错。

[模型身份](identity.py)与 rollout 共用完整 checkpoint 文件清单。checkpoint-v2 覆盖权重、配置、tokenizer、生成配置及嵌套 chat_templates。相同内容换目录可复用，同路径替换权重会失效。Source bank 使用版本 2 内容合同并保存 engine_contract，运行 manifest 分别绑定 student 与 teacher 身份；更改有效停止条件也会拒绝复用。

旧版未绑定支持 mask、完整权重或生成合同的银行，不能补字段后作为新运行输入。训练／只读 RKL 的原生输出行与生成支持分别记录；共享支持条件化不能靠裁剪 logits 冒充同一指标。数学关系和模型扩展见[动作空间审查](../../../examples/mirroropd/docs/infra/model-compatibility/action-space.md)。

## 冻结生成的恢复与失败处理

Bootstrap 的`initial`或`teacher`可以使用`--resume-generation /旧产物目录`。只有模型权重、题目、n、输入／采样协议、有效 engine contract 和 seed 完全匹配时，才复制已经发布的原始 token 与 mask；包括空回答和错误答案，只补缺失的 sample_index。新目录记录原 manifest 的 hash 和复用数量，旧目录保持原样。这个入口只处理原始冻结权重，不恢复训练中断后的模型或 Adam。

判分器错误带具体 dataset／question／sample 身份。生成代码在函数题初始化或函数执行中调用`exit()`属于模型运行错误；正确 stdin 程序输出后退出仍按原规则比较。环境启动、缺失依赖、执行器协议等错误仍保留为基础设施错误。

共享 scorer 对未确认结束的远端 RPC 执行 fail-stop：不清理仍可能被写入的 TQ partition，不恢复 rollout，不发布完成标记。已确认结束后的本地校验失败仍进行正常清理；状态审计与必要恢复成功后才发布事件。

## 准确率和模型导出

[accuracy.py](accuracy.py)复用 Math-Verify 和已有 LiveCodeBench 隔离执行器，新增 GPQA 最终选项判分，并固定 Google Research 的 IFEval 严格判分实现。[上游版本与文件 hash](vendor/ifeval/SOURCE.json)随源码保存。依赖见[requirements.txt](requirements.txt)，NLTK 数据目录由`NLTK_DATA`指定。CPU 判分使用有界进程队列；每题每个 sample_index 都必须返回结果。未闭合 thinking 段没有最终答案时计零；评测器异常不会变成错误答案。

Public bank 是 generation 产物的 sample 0 引用，无需再复制一份长 token。Generation、accuracy、每轮 RKL event 分别标记完成，因此准确率可以复用已有回答重算；训练本身仍要求 fresh run。训练轨迹保留在单独 shard，history 只保存引用。新的逐更新 Parquet 表直接记录 trainer 的实际 update events。

最终导出关闭原有周期 save，使用原生 FSDP 保存器暂存 model 与 HF 权重，保留 FP32 master 权重，不保存 Adam。固定输入的 native model-only reload 对齐通过后删除临时 FSDP 分片，只保留最终 HF 权重及导出审计。该检查与从 HF 重新初始化后的前向一致性是两项检查，验收证据见文末。

新图表入口是`examples.mirroropd.research.studies.pipeline.plot_unified_evaluation`，直接读取完成的 Parquet 事件，输出训练 loss、来源／history RKL、四域 bank／current RKL、history 批矩阵、accuracy 和可审计 CSV。旧 0–320/H1024 图表仍使用原 reader。

## 资源与恢复责任

`RolloutService` 显式要求 `ensure_current(version)` 与 `park()`。两个内置服务仅在权重同步和所需 sampler 安装成功后声明驻留版本；同步或休眠失败时清除版本标记。异常仍交给调用方停止执行，清除标记不代表可以安全重试未完成的远程请求。

公共 suite 在 before-sync 读出时暂存事件，由正常权重同步成功后的 `on_readout` 发布完成；其他时机先恢复当前 rollout 再完成事件。发布失败时保留 pending 引用，不标记 completed。最终同步仍由组件的 `owns_final_weight_sync` 声明，不能从 evaluation 配置是否存在推断。Source/history 的轻量协议与资源预算保持独立。

## 产物和完成边界

Source/history 的产物布局如下；公共 suite 的布局见后表。

```text
readout/
  manifest.json
  initial-student-bank/
    manifest.json
    trajectories.parquet
  trajectories/outer-001.parquet
  history/outer-001.parquet
  scores/outer-001/
    manifest.json
    rkl.parquet
```

公共 suite 的运行目录保存以下对象；bootstrap 由独立目录提供并绑定其 manifest 身份。

| 路径 | 内容 |
| --- | --- |
| `events/outer-NNN/manifest.json` | 计划、outer/optimizer 进度、coverage、指标、耗时和 aliases |
| `events/outer-NNN/rkl.parquet` | 按银行选择身份记录的 RKL sum/count 和 measure-id |
| `events/outer-NNN/current/` | 本轮 current student 生成的不可变分片 |
| `events/outer-NNN/accuracy/` | 跟随生成分片的 CPU 判分结果及 verifier 来源 |
| `trajectories/`、`history/` | 训练原始轨迹与轻量历史引用 |
| `final-model/` | 最终模型及导出审计 |

训练轨迹只写一次。History 行只有`id/source_id/outer/shard/row_index/history_batch/history_quota`，按分片和 row group 加载实际 token；driver 只保留轻量历史引用和当前 source 银行。

一个对象只有一个 manifest，记录版本、身份、文件 hash／大小／行数及状态。文件不可覆盖；写完临时 Parquet 后 rename，再发布 manifest。评分事件在前向、审计、TQ 清理和必要的 rollout 恢复均成功后才发布。事件完成表示文件完整且要求的恢复已完成。两种读出共用 `coverage`：记录 expected/observed/valid/invalid/missing 行数、各 bank 的覆盖以及排序后的 `(bank, id)` 选择集合 SHA256。发布前要求预期与实际身份集合完全相同，拒绝 bank 内重复、缺失、额外结果、trajectory 身份或已知 mask token 数改变；同一轨迹可由不同 bank 引用。无效数值和零 token 行保留，`coverage.scientific_complete` 为 false；空 RKL 选择也不声明科学读数完整。此 coverage 仅描述 RKL，accuracy 的样本覆盖仍由其独立产物校验。读出集合的最终完成不代表全部 Trainer 日志或模型导出也已完成。

`ArtifactStore`使用本机文件锁与 POSIX rename，不承诺断电恢复或跨节点事务。它不是 TQ 替代品，不保存 teacher hidden、全词表概率或 optimizer。内容 hash 读取校验、非法路径／重复文件／未知列检查失败时直接报错。

## 指标与历史读取

- `eval/rkl/source_heldout/question_mean`：先题内 token 均值，再按题平均。
- `eval/rkl/source_heldout/absolute_reduction`和`reduction_percent`：同一固定银行相对初值；零基线不输出百分比。
- `eval/rkl/history/equal_batch_question_mean`：批内题等权、批间等权。
- `eval/rkl/history/paired_rows`和`paired_absolute_reduction`：相邻实际读出的共同轨迹，避免换样混淆。
- 各来源记录 rows、valid_rows、invalid_rows 和 tokens。任何无效行使该集合主均值缺省，不能将缺失补零或只平均成功部分。
- `score.audit_state=false`明确记录`not_requested`，不当作参数／Adam 哈希检查通过。

训练 token-mean 和旧 JSON 报告的 token-mean 不更名冒充新口径。`examples/mirroropd/benchmarks/profile/audit_fixed_rkl.py`可读取新 Parquet 及旧 JSON，并标记各自 aggregation。

## 执行限制与验收范围

运行使用同步 V1／FSDP2／vLLM、已验证的 token 映射和原生支持 RKL。Native scorer 要求 teacher 输出支持覆盖 student；显式 common_conditioned 在声明的集合上条件化，并记录被排除的质量和 measure-id。公共 suite 要求 n=1、一个全局 mini-batch、无 critic，并关闭周期保存，只保留最终模型；这不意味着所有 readout 组件都有相同预算或保存约束。

RKL host budget 约束临时 hidden、位置与拷贝余量，并扣除活跃训练 payload，不包含常驻模型和 Adam 的全部 CPU 内存。窗口根据 shared teacher 的实际 forward precision 推导 hidden 字节数；已使用的推理 token budget49152 覆盖 public prompt8192 加 response32768，与训练和生成 cap 分开。

历史四卡集成采用 batch8/reuse2/两轮，完成 4 次更新和三次读出，28/46/46 条逻辑轨迹有效，四 rank 只读状态审计通过；原生重载和独立 HF 初始化的固定前缀 RKL 最大误差为 0。历史 CPU 检查为 495 项，另有 21 项初始化调度回归；基础 source/history 的 batch128/reuse32/两轮 64 次更新是另一批验收。这些记录不相加为独立样本，也不作为当前所有路径的重新验收，详见[历史实施与证据](../../../examples/mirroropd/docs/archive/engineering/evaluation/2026-11-01-分阶段代码走读与验收.md)。

两轮运行合同重构的当前检查见[重构报告](../../../examples/mirroropd/docs/archive/engineering/2026-11-01-运行合同审计与重构方案.md#提交与验收)。Source/history 最新生命周期有原生形状的 CPU 协议检查，未增加完整 vLLM accuracy 或 50 outer 实验；模型扩展和更大规模检查仍各自记录。

## 分布、模型与输入适配

`distillation.distillation_loss.full_vocab.distribution`默认为空，保留 native head RKL。小 teacher→大 student 使用显式条件支持，例如：

```yaml
distillation:
  distillation_loss:
    full_vocab:
      distribution:
        mode: common_conditioned
        common_vocab_size: 151936
```

`common_vocab_size`指共同输出行，不会自动变成 tokenizer 交集。也可用成对的`student_ids`／`teacher_ids`声明等价动作的索引顺序。训练与评估均复用`DistributionSpec`；新评分表是EVAL_RKL schema2，measure-id绑定分布定义与实际forward精度profile，并附`measure_id`、`student_excluded_mass_sum`、`teacher_excluded_mass_sum`及可用的`uncoarsened_rkl_sum`。聚合器拒绝将不同 measure-id 合并或计算相对下降，旧schema1表仍可只读分析。

`score.objective_space=true`选择训练配置中的停止动作分组，默认false保持未合并动作的RKL。这里始终比较student与teacher，不构造MDPO的anchor／target。条件化与EOS组合要求完整保留或排除停止类；优化结果的`objective_rows`和未缩放`loss_sum`一致，旧raw位置指标保留原义。

模型支持Qwen2／Qwen3／Qwen3-MoE和Llama的未量化、无bias线性head。启动Ray前检查真实head清单与shape、tokenizer编码／解码语义、输入ID、上下文及有效停止动作。Llama动态日期模板必须在`data.apply_chat_template_kwargs.date_string`和各评估协议的`template_kwargs.date_string`中固定同一个值。

只有ID重新编号而分词语义等价时，可显式设置`full_vocab.input_id_mapping=token_semantics`，并声明匹配的条件输出支持。Teacher前向在独立模型输入上应用映射，原student轨迹不改；teacher生成请求映射输入／停止ID，返回token映射回student坐标，log-prob保持不变。不同分词边界、额外teacher生成动作无法映回student、softcap／bias／量化head均须另立协议或adapter，不能自动裁切后进入主表。

新权重身份复用rollout的`checkpoint_identity`，以`checkpoint-v2:`标识；新运行只接受当前内容合同，不将旧身份自动升级。生成ID只绑定sampling与输入，`answer_channel`改动不再改变生成身份；判分产物另存verifier协议和身份，可对同一回答独立重判。旧生成记录不改写身份或原manifest。

`native.py`统一轨迹/TQ转换和history抽样，source/history 与公共 suite 共用同一组原语；`NativeEvaluationExecution`集中对Trainer私有评分／平衡接口的调用。评估事件另记录实际distribution、teacher transfer／forward、trajectory publish和student readout RPC耗时，原有细粒度NVTX范围继续用于GPU profile。

重判沿用同一冻结题目／参考答案版本，并校验实际输入tokenizer身份。修改判分代码或答案通道可复用回答；修改题目／参考答案属于数据版本变更，不自动跨版本复用。缺少输入身份的旧生成产物使用原冻结评估器，不以新元数据冒充已验收输入。

SourceHistoryReadout 保留 native、identity、未合并读出的原始协议；条件化、输入映射或 objective-space 读出使用 EvaluationCoordinator，不能把不同测量空间写入同一 source/history 指标。训练中的条件化指标使用 conditional_rkl 命名，不复用 native full_kl 的名称。

图目录附带数值 CSV 与 `plot-inputs.json`，可交给同一绘图入口独立重绘，无需模型或原始 rollout；输出目录拒绝覆盖另一 run 的图表。


## Optional source and history diagnostics

`source_enabled` and `history_enabled` explicitly opt in to these readouts; both default to false. A source file or a schedule alone does not enable them. Disabled diagnostics perform no model scoring; training trajectories and duplicate-question checks remain active. Source-disabled bootstrap needs only public initial/teacher generations and their accuracy artifacts. Final export verifies against the source bank when enabled, otherwise against the public initial bank.

Qwen E63–E68 explicitly preserve both diagnostics. E78–E80 enable the common source holdout and disable history. Existing raw evaluation configurations that need these diagnostics must declare the flags; no implicit schema conversion is performed.
