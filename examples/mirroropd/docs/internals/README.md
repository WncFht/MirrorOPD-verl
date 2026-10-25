# 代码走读文档库

这里维护当前工作区的单教师 OPD、weighted teacher、MDPO/BPO 源码走读，覆盖统一实验入口、rollout、评分、loss、分布式反传、日志与恢复。历史重构基点与上游差异见[改动地图](upstream-changes.md)，当前验证范围见[实现说明](../mirror-descent/实现方案.md#validation)。

阅读目标是能从一份训练配置追到产生梯度的具体表达式，并解释一批数据复用时每个字段何时产生、何时变化。先读[架构总览](../architecture.md)与[训练循环](training-loop.md)，再沿数据、教师和两条目标路径深入。公式推导、指标字典、运行命令和历史证据通过链接引用。

## 训练路径的共同部分与分流

所有方法先由 student 生成轨迹，并在固定前缀上训练；reuse 期间不重新生成 token。Shared-sync 轮流使用同一 GPU 池执行 rollout、teacher 和 actor。配置先解析为目标与数据需求，再决定准备阶段。

| 方法 | Teacher payload | Anchor | 实际训练目标 |
| --- | --- | --- | --- |
| Legacy sampled OPD | 动作 log-prob | vanilla 的 ratio 分母；显式模式用于旧诊断 | current/rollout 信号 × vanilla/token PPO/token IS/NoIS |
| Sampled MDPO | 动作 log-prob | eta<1 时在 actor 调用内准备 scalar；eta=1 跳过 | weighted target 的未截断 IS 梯度估计 |
| 普通 Full-KL | 有效 hidden/head | 无 | MDPO eta=1，$\mathrm{KL}(p\Vert T)$ |
| Full MDPO | 有效 hidden/head | eta<1 时准备 hidden/head | 到 weighted target 的 scaled reverse KL，减冻结常数 |
| Full BPO | 有效 hidden/head | 始终准备 hidden/head，包括 eta=1 | q 加权中心化 log-residual 方差 |

Teacher/anchor 在一次 actor 调用内冻结，当前 student 每次更新重算。三种 mirror 设置共用 anchor scope、target 构造、LossContext 与 optimizer；full 设置共用投影和 precompute 累计。显式旧 sampled 与 MDPO 共用输入对齐和 surrogate primitive，但各自的信号、clipping 和 ratio guard 保持独立。

“On-policy”描述轨迹生成来源。Current sampled IS 与 Full-KL 的联系要求固定前缀、正确提案概率和支持条件；旧 guard 触发后不再与未截断 MDPO 等价。公式见[统一目标定义](../mirror-descent/实现方案.md#objectives)。

## 阅读顺序与问题索引

| 顺序 | 文档 | 解决的问题 |
| --- | --- | --- |
| 0 | [相对上游的改动](upstream-changes.md) | verl 基线已有哪些机制，本仓在什么位置新增或改变行为 |
| 1 | [训练循环](training-loop.md) | 一批数据如何进入训练，reuse、micro-batch 与权重同步怎样组织 |
| 2 | [教师生命周期](teacher.md) | shared 与 standalone 的职责，教师资源和评分发布顺序 |
| 3 | [数据与归一化](data-and-loss.md) | TQ 身份、预测位置、response 布局、三种 mask 和全局分母 |
| 4 | [Sampled OPD](sampled-opd.md) | behavior、anchor、current 的区别，current/frozen 信号与 IS/PPO |
| 5 | [Full-vocab](full-vocab.md) | hidden IO、冻结 head、算子与 VJP、词表支持和精度 |
| 6 | [指标系统](metrics.md) | Observation 跨卡归约、位置分桶、逐 update 日志 |
| 7 | [进度与 checkpoint](progress-checkpoint.md) | 更新是否执行、计数未知、scheduler 和恢复状态 |

预算和字段定义分别由[训练配方](../训练配方.md)及[指标字典](../experiments/metrics/指标字典.md)维护。

按图走读可依次查看：[训练分流](training-loop.md#从采样到更新) → [mini/micro-batch 更新时序](training-loop.md#从一次-actor-调用到实际更新) → [sampled 梯度连接](sampled-opd.md#detach-的作用)或 [Full-KL 数据与反传](full-vocab.md#从教师-hidden-到-student-head) → [mask 与全局归一化](data-and-loss.md#三种-mask-与分母)。教师上下卡和指标出口另有[教师时序图](teacher.md#评分顺序)与[指标归约图](metrics.md#从局部观测到日志)。

## 按实际调用顺序打开源码

| 路线 | 阅读顺序 |
| --- | --- |
| 公共主线 | launcher → `main_ppo` → `PPOTrainer._setup/fit/_step_once` → `ActorRolloutRefWorker.update_actor` → `TrainingWorker.train_mini_batch` → `BaseEngine.train_batch` |
| Sampled 教师 | `SharedTeacherScorer.score` → teacher `infer_batch` → FSDP 下一 token log-prob → TQ `teacher_logprobs` |
| Sampled loss | student `forward_step` → `distillation_ppo_loss` → `distillation_loss` → `_sampled_token_loss` 或 vanilla estimator + PPO → `LossContext` |
| Full-KL 教师 | teacher `forward_step` → `_full_vocab_output` → `full_vocab_forward` → 选 hidden → `HiddenBatchIO.publish` |
| Full-vocab actor | `HiddenBatchIO.__iter__` → student `_full_vocab_output` → `full_vocab_head_forward` → `full_vocab_objective_result` → endpoint KL 或 MDPO/BPO rows → `LossContext.normalize_token_sum / eta` |
| Anchor scope | `update_actor` → `anchor_scope` → 可选整批 scalar/hidden 评分 → 原 train_mini_batch → finally 清理 |
| 更新出口 | `backward` → `optimizer_step` → `UpdateResult/finalize_update` → `OptimizerProgress/update_log_events` → checkpoint、rollout 同步 |

这些符号的可点击源码入口分别集中在对应章节，避免在索引中重复维护完整链接列表。需要快速定位问题时，可直接按下面的证据查找：

| 需要解释的现象 | 首先检查 |
| --- | --- |
| `GRPO` 配置存在，任务得分却不影响 OPD | [两套 advantage 的消费者](training-loop.md#两套-advantage-的消费者) |
| 多次 reuse 后 current 与 rollout 信号异号 | [信号与 surrogate](sampled-opd.md#信号与-surrogate) |
| loss 数值不能跨 sampled/Full-KL 比较 | [sampled 观测](sampled-opd.md#观测与扩展)、[指标阶段](metrics.md#按训练阶段解释日志) |
| mask、EOS 或不等长序列出现错位 | [预测坐标例子](data-and-loss.md#两种布局与预测坐标) |
| 改 GPU 数或 micro-batch 后梯度尺度变化 | [全局分母推导](data-and-loss.md#三种-mask-与分母) |
| Full-KL 的 cached/precompute 显存与耗时不同 | [backend 分工](full-vocab.md#backend-的实际工作分配) |
| teacher head 行数更大，但 tokenizer 相同 | [教师多出的输出行](full-vocab.md#教师多出的输出行) |
| optimizer、scheduler 和 outer step 计数不相等 | [更新预算](training-loop.md#从一次-actor-调用到实际更新)、[进度](progress-checkpoint.md) |

## 维护约定

修改接口、控制流、字段或默认行为时，在同一提交更新所属走读；若行为没有变化，也应核对源码引用是否仍指向实际实现。走读使用稳定文件名，Git 记录版本；实验报告和验收记录保留日期及当时设置。

每篇说明模块职责、关键源码入口、输入输出、执行顺序、必须保持的约定及验证入口。跨模块过程归架构总览，模块细节归对应走读。源码链接的标签使用实际 `Class.method` 或函数名，不依赖易漂移的行号；指向整个文件时使用文件名。

历史文章可以保留原内容，但必须明确其时间范围，并将当前读者导向这里。旧路径和章节锚点保留，避免破坏已有引用。研究中的建议、未实现方案和冻结测量不作为当前接口定义。

## 自动检查

在仓库根目录执行：

```bash
make docs-check
```

检查器仅依赖 Python 标准库，检查本库、架构总览和主导航的本地链接及 Markdown 锚点；对以 Python 符号命名的源码链接检查 AST 定义，对这些页面的 Python 示例检查语法。它不导入训练框架、不读取模型、不启动 GPU，也不能证明文字描述与代码语义一致。语义变化仍需作者走读和相关测试。

检查已接入 pre-commit；现有训练测试负责实际行为。历史实验目录不纳入当前文档的符号一致性要求。

## 内容来源与边界

| 原资料 | 当前说明的归属 |
| --- | --- |
| Full-KL 全流程走读 | training-loop、teacher、data-and-loss、full-vocab |
| 共享八卡运行与架构 | 架构总览、teacher；运行命令仍留 infra |
| KL 计算与公共位置指标分层 | full-vocab 与 metrics |
| 架构重构实施与验收 | data-and-loss、metrics、progress-checkpoint；验收留 refactoring |
| 优势与 IS 实现方案 | sampled-opd；方案和旧预算仍留 opd-objective |
| OPD 实现验收与对齐诊断 | teacher 的输入协议、full-vocab 的输出支持；实验仍留 failure-reproduction |
| 精度误差模型与诊断接口走读 | full-vocab 的生产精度说明；误差推导和探针仍留 research |

网站发布若以后引入，应直接读取这些正文，不复制另一套 Markdown。现阶段以仓库内链接和源码为阅读入口。
