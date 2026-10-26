# MirrorOPD 文档导航

正式训练实验统一从[实验目录](experiments/README.md)进入：查看已经运行的实验、实际 setting、待跑项、指标定义和结果分析。理论、训练操作、性能测量与实现验收分别维护。

| 阅读目的 | 入口 | 内容 |
| --- | --- | --- |
| 理解当前系统及模块边界 | [架构总览](architecture.md) | 执行顺序、职责和扩展边界 |
| 沿源码理解训练实现 | [代码走读文档库](internals/README.md) | 教师、布局、目标、指标、更新和恢复 |
| 盘点实验与后续缺口 | [实验目录](experiments/README.md) | E 系列、历史 Math/Coding、运行与分析进度、待跑项 |
| 查看实际 run 与结果 | [运行记录](experiments/运行记录.md) | 实际配置、状态、更新数、最后验证坐标与分数 |
| 查指标定义与比较口径 | [实验指标](experiments/metrics/README.md) | policy-v3、历史字段、验证预算、KL 与位置统计 |
| 复算分析与更新实验记录 | [复现与维护](experiments/复现与维护.md) | 独立日志仓库、快照、运行表、统计与静态绘图 |
| 理解 OPD、full-vocab 与 Mirror Descent | [理论结论与阅读顺序](theory/README.md) | 目标、计算图、四种更新、控制变量和证明 |
| 理解 η、κ 的目标与实现 | [MirrorOPD 形状与插值](mirror-descent/README.md) | 固定 state、动作 IS、current/frozen、clipping |
| 核对数学符号 | [符号约定](符号约定.md) | 教师、学生、锚点、行为策略与工程数量 |
| 准备模型与数据 | [资产准备](资产准备.md) | 路径、导入与检查 |
| 启动训练与选择当前预设 | [训练配方](训练配方.md) | 领域入口、公共长度、资源和参数；实验矩阵链接到各系列 |
| 部署共享资源池及调整吞吐 | [基础设施](infra/README.md) | 运行架构、性能实测、MFU、资源分析 |
| 核对指标实现与验收 | [指标工程](internals/metrics.md) | 实现、迁移、CPU/GPU 验收证据 |
| 阅读架构重构与验收 | [架构重构](refactoring/README.md) | 实施边界、测试与覆盖限制 |
| 使用 full-vocab 训练 | [Full-vocab 工程](full-vocab/README.md) | 调用链、算子、数据通路与实现验收 |
| 查算法接口与源码审计 | [OPD 实现与审计](opd-objective/README.md) | 支持集、优势估计器、配置合同 |
| 查相关研究与数值诊断 | [研究与方法](research/README.md) | 陈旧数据文献与 KL 精度机制 |
| 查失败复现的工程准备 | [对齐诊断与实现](failure-reproduction/README.md) | EOS、tokenizer、词表和短测；正式方案在实验目录 |
| 配置任务判分 | [判分器与依赖](判分器与依赖.md)、[LCB 隔离](lcb/README.md) | 依赖、执行限制与验收 |
| 查旧队列与历史工程记录 | [历史记录](archive/README.md) | 原版本配置、实现背景和测量证据 |

## 维护边界

实验系列 README 维护设计、实际运行概况与待办；逐 run 配置与验证记录由实验快照生成。详细分析明确所用冻结数据，不用当前默认值重写历史 setting。新结果写入新快照，并在系列入口更新状态。

实验设计、分析正文、展示图和专用脚本维护在 `experiments/`；完整运行数据及报告快照统一存放于 `wandb-logs`。共享导出器和工程分析工具仍在 `scripts/`。指标定义只有实验目录一个入口，当前实现见 `internals/metrics.md`，旧指标工程材料归 `archive/metric-implementation/`。

当前代码说明由[走读文档库](internals/README.md)维护；修改接口或行为时，同一提交更新所属页面，并运行 `make docs-check`。日期命名的实现和验收记录保留历史口径。
