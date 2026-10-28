# OPD 目标与梯度估计

当前实现接口统一见[Sampled OPD](../../infra/architecture/sampled-opd.md)和[进度与 checkpoint](../../infra/architecture/progress-checkpoint.md)；本目录保留理论审计和实施记录。

统一记号见[符号约定](../../theory/符号约定.md)，与 `MirrorOPD/theory/core` 对齐。

固定前缀的数学定义和证明集中在[理论目录](../../theory/README.md)。本目录维护论文实测、源码审计、实验开关和历史验算；已迁出的数学章节集中到各旧文末尾的跳转索引，原锚点继续可用。

| 内容                                                        | 入口                                                                                                  |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| OPD 四种目标、full-vocab、镜像下降与控制变量                | [理论阅读入口](../../theory/README.md)                                                                |
| 论文证据、实现刷新时机、实验建议                            | [固定前缀原审计](2026-10-21-固定前缀下的优势估计与重要性采样.md)                                      |
| 模式合同、更新计数、checkpoint 与历史验证                   | [优势与 IS 消融及更新计数实现方案](../../infra/training/2026-10-21-优势与IS消融及更新计数实现方案.md) |
| Sampled-token、KL baseline、TA-OPD、ResOPD 的源码与实验比较 | [学习信号与梯度估计器审计](2026-10-21-OPD学习信号与梯度估计器对照.md)                                 |
| E1–E15 编号与学习/评测预算                                  | [训练配方](../../experiments/learning-signal/sampled-controls/README.md)                              |
| Full-vocab 算子性能、payload 与工作区容量                   | [算子性能与容量](../../infra/performance/full-vocab/operators.md#容量预算)                            |
| Reverse-KL 实验排序、top-k 支持集与 One-Shot 代码审计       | [实验排序与支持集](2026-10-21-Reverse-KL实验排序与Top-k支持集.md)                                     |
| Full-vocab 工程、参考资料与实现计划                         | [Full-vocab 目录](../../infra/numerics/full-vocab/README.md)                                          |
| CPU 梯度验算                                                | [verify_fixed_prefix.py](../../../research/objectives/verify_fixed_prefix.py)                         |
| 固定版本上的验算结果                                        | [gradient-audit.json](data/gradient-audit.json)                                                       |

系统工程分别由 [指标与验收](../../settings/指标口径.md)和[同步共享八卡方案](../../infra/training/2026-10-22-共享八卡运行与架构.md)负责。改变 GPU 调度、改变 advantage、取消 PPO clipping、切换 IS 分母是不同实验因素。
