# 代码走读文档库

这里维护当前源码的职责、调用链和运行契约。先读[架构总览](../architecture.md)，再沿下表进入具体模块。每个主题只有一个当前详细入口；公式推导、指标字典、运行命令和历史证据通过链接引用。

## 阅读顺序与问题索引

| 顺序 | 文档 | 解决的问题 |
| --- | --- | --- |
| 1 | [训练循环](training-loop.md) | 一批数据如何进入训练，reuse、micro-batch 与权重同步怎样组织 |
| 2 | [教师生命周期](teacher.md) | shared 与 standalone 的职责，教师资源和评分发布顺序 |
| 3 | [数据与归一化](data-and-loss.md) | TQ 身份、预测位置、response 布局、三种 mask 和全局分母 |
| 4 | [Sampled OPD](sampled-opd.md) | behavior、anchor、current 的区别，current/frozen 信号与 IS/PPO |
| 5 | [Full-vocab](full-vocab.md) | hidden IO、冻结 head、算子与 VJP、词表支持和精度 |
| 6 | [指标系统](metrics.md) | Observation 跨卡归约、位置分桶、逐 update 日志 |
| 7 | [进度与 checkpoint](progress-checkpoint.md) | 更新是否执行、计数未知、scheduler 和恢复状态 |

预算和字段定义分别由[训练配方](../训练配方.md)及[指标字典](../metrics/2026-10-22-指标字典.md)维护。

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
