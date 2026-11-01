# 原生训练的评估合同

本包复用 V1／FSDP2／vLLM 的原生生成与评分，负责评估调度、Parquet 产物、CPU 判分及串行 RKL 读出。训练 loss 保持 token-mean；评估先题内 token 均值再按题平均，history 再按批等权。当前架构见[总览](../../../examples/mirroropd/docs/infra/architecture/README.md)，旧配置按[迁移指南](../../../examples/mirroropd/docs/infra/architecture/runtime-migration.md)升级。

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
| requires_full_batch    | 要求 n=1、完整训练 mini-batch、token-mean、无 critic | 公共 suite 为 true                                                |
| owns_final_weight_sync | 接管最后一轮同步，避免 Trainer 重复恢复模型          | 公共 suite 为 true                                                |

Trainer 和 RolloutService 不根据 evaluation 配置是否存在推断这些行为。Source/history 的休眠与恢复只经 RolloutService，恢复时重新安装 sampler 并维护驻留版本。新增组件声明实际能力并校验自己的 options，无需给 Trainer 增加一套配置入口。

Source/history 与公共 suite 均要求 current single-model 的已验收 sampler。其他科学生成协议使用对应显式组件，不伪装成 initial-student source bank。

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

统一配置包含`_target_/output/source_file/bootstrap/datasets/protocols/score/schedules`。所有组件的 options 与 `_target_` 放在同一 `trainer.evaluation` 中；构造边界只解析一次目标，剩余字段由组件的严格类型配置验证。`datasets`逐 bench 声明 name、domain、path、verifier、n、protocol、panel_ids；`protocols`分别定义 source 和 public 的 T／top-p／top-k、回答 cap、prompt 预算和模板。配置拒绝未知字段，public prompt 超长时报错；不静默删题或截短回答。稳定采样 seed 由 seed／dataset／question／sample_index 决定，sample 0 不随 n 或 checkpoint 改变。

| 任务                           | 默认 outer 边界      | 轨迹来源                                            |
| ------------------------------ | -------------------- | --------------------------------------------------- |
| 来源留出 RKL                   | 0 与每轮             | 固定 initial-student 回答                           |
| History RKL                    | 每轮，从 1 开始      | 真实训练回答，批均衡抽样                            |
| Public bank RKL                | 0／1／2，之后每 5 轮 | 固定 initial-student 与 teacher 回答，各取 sample 0 |
| Public current RKL 与 accuracy | 0／25／50            | 当前 student 回答；0 复用 initial 产物              |
| Teacher accuracy               | 0                    | Bootstrap 全量 teacher 回答                         |

初始化 0 必须建立所有固定参考；其余边界按每个 schedule 的`at/every/start/final`并集调度。同一边界只编译一次计划；由组件独占调度，trainer.test_freq 必须关闭。满足同一评分协议的轨迹合并后按 teacher hidden 字节预算分窗口，窗口内 teacher→student 串行；预算扣除仍活跃的训练 payload 和拷贝余量。推理只按 DP 数量 padding，padding 不进入科学样本统计。

## 生成与内容身份

Bootstrap 和 current student 均通过 policy_engine_config 安装公共 sampler 与 TokenizerSupport，屏蔽没有 tokenizer 映射的输出行。有效 engine_contract 记录支持 ID、输入 tokenizer／模板、EOS、sampler 版本和执行精度；样本身份使用 generation-v2。银行与当前引擎合同不一致时，在参数更新前报错。

[模型身份](identity.py)与 rollout 共用完整 checkpoint 文件清单。checkpoint-v2 覆盖权重、配置、tokenizer、生成配置及嵌套 chat_templates。相同内容换目录可复用，同路径替换权重会失效。Source bank 使用版本 2 内容合同并保存 engine_contract，运行 manifest 分别绑定 student 与 teacher 身份；更改有效停止条件也会拒绝复用。

旧版未绑定支持 mask、完整权重或生成合同的银行，不能补字段后作为新运行输入。训练／只读 RKL 的原生输出行与生成支持分别记录；共享支持条件化不能靠裁剪 logits 冒充同一指标。数学关系和模型扩展见[动作空间审查](../../../examples/mirroropd/docs/infra/architecture/evaluation/动作空间与模型兼容.md)。

## 冻结生成的恢复与失败处理

Bootstrap 的`initial`或`teacher`可以使用`--resume-generation /旧产物目录`。只有模型权重、题目、n、输入／采样协议、有效 engine contract 和 seed 完全匹配时，才复制已经发布的原始 token 与 mask；包括空回答和错误答案，只补缺失的 sample_index。新目录记录原 manifest 的 hash 和复用数量，旧目录保持原样。这个入口只处理原始冻结权重，不恢复训练中断后的模型或 Adam。

判分器错误带具体 dataset／question／sample 身份。生成代码在函数题初始化或函数执行中调用`exit()`属于模型运行错误；正确 stdin 程序输出后退出仍按原规则比较。环境启动、缺失依赖、执行器协议等错误仍保留为基础设施错误。

共享 scorer 对未确认结束的远端 RPC 执行 fail-stop：不清理仍可能被写入的 TQ partition，不恢复 rollout，不发布完成标记。已确认结束后的本地校验失败仍进行正常清理；状态审计与必要恢复成功后才发布事件。

## 准确率和模型导出

[accuracy.py](accuracy.py)复用 Math-Verify 和已有 LiveCodeBench 隔离执行器，新增 GPQA 最终选项判分，并固定 Google Research 的 IFEval 严格判分实现。[上游版本与文件 hash](vendor/ifeval/SOURCE.json)随源码保存。依赖见[requirements.txt](requirements.txt)，NLTK 数据目录由`NLTK_DATA`指定。CPU 判分使用有界进程队列；每题每个 sample_index 都必须返回结果。未闭合 thinking 段没有最终答案时计零；评测器异常不会变成错误答案。

Public bank 是 generation 产物的 sample 0 引用，无需再复制一份长 token。Generation、accuracy、每轮 RKL event 分别标记完成，因此准确率可以复用已有回答重算；训练本身仍要求 fresh run。训练轨迹保留在单独 shard，history 只保存引用。新的逐更新 Parquet 表直接记录 trainer 的实际 update events。

最终导出关闭原有周期 save，使用原生 FSDP 保存器暂存 model 与 HF 权重，保留 FP32 master 权重，不保存 Adam。固定输入的 native model-only reload 对齐通过后删除临时 FSDP 分片，只保留最终 HF 权重及导出审计。该检查与从 HF 重新初始化后的前向一致性是两项检查，验收证据见文末。

新图表入口是`examples.mirroropd.research.studies.pipeline.plot_unified_evaluation`，直接读取完成的 Parquet 事件，输出训练 loss、来源／history RKL、四域 bank／current RKL、history 批矩阵、accuracy 和可审计 CSV。旧 0–320/H1024 图表仍使用原 reader。

## 产物和完成边界

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

训练轨迹只写一次。History 行只有`id/source_id/outer/shard/row_index/history_batch/history_quota`，按分片和 row group 加载实际 token；driver 只保留轻量历史引用和当前 source 银行。

一个对象只有一个 manifest，记录版本、身份、文件 hash／大小／行数及状态。文件不可覆盖；写完临时 Parquet 后 rename，再发布 manifest。评分事件在前向、审计、TQ 清理和必要的 rollout 恢复均成功后才发布。事件完成表示文件完整；`coverage.scientific_complete`另表示是否有完整有效科学读数。读出集合的最终完成不代表全部 Trainer 日志或模型导出也已完成。

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

运行使用同步 V1／FSDP2／vLLM、已验证的 token 映射和原生支持 RKL。Scorer 要求 teacher 输出支持覆盖 student；其他概率空间需显式实现与验证。公共 suite 要求 n=1、一个全局 mini-batch、token-mean、无 critic，并关闭周期保存，只保留最终模型；这不意味着所有 readout 组件都有相同预算或保存约束。

RKL host budget 约束临时 hidden、位置与拷贝余量，并扣除活跃训练 payload，不包含常驻模型和 Adam 的全部 CPU 内存。当前资源配方按 BF16 hidden 估算；已使用的推理 token budget49152 覆盖 public prompt8192 加 response32768，与训练和生成 cap 分开。

历史四卡集成采用 batch8/reuse2/两轮，完成 4 次更新和三次读出，28/46/46 条逻辑轨迹有效，四 rank 只读状态审计通过；原生重载和独立 HF 初始化的固定前缀 RKL 最大误差为 0。历史 CPU 检查为 495 项，另有 21 项初始化调度回归；基础 source/history 的 batch128/reuse32/两轮 64 次更新是另一批验收。这些记录不相加为独立样本，也不作为当前所有路径的重新验收，详见[历史实施与证据](../../../examples/mirroropd/docs/infra/architecture/evaluation/代码走读与实施计划.md)。

两轮运行合同重构的当前检查见[重构报告](../../../examples/mirroropd/docs/infra/architecture/2026-11-01-运行合同审计与重构方案.md#提交与验收)。Source/history 最新生命周期有原生形状的 CPU 协议检查，未增加完整 vLLM accuracy 或 50 outer 实验；模型扩展和更大规模检查仍各自记录。
