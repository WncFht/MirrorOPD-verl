# 训练指标

当前统计实现与更新记录出口见[指标系统走读](../internals/metrics.md)；字段定义仍以本页指向的指标字典为准。

Last updated: 2026-10-23。

当前实现为 **policy-v3**：恢复条件功能的观测，按所有支持的采样预算输出 eval，补齐 token/sequence pair 和实际 IS/RS 统计。固定 64 字段允许列表已移除。E3 与 token IS + token_k1 RS 均已完成四卡实验和云端回读，详见实现记录。

| 文档 | 内容 |
| --- | --- |
| [公共位置指标与归约](../internals/metrics.md) | 任意逐token标量的相对位置十分桶、caller指定namespace、Full-KL接入 |
| [当前指标字典](2026-10-22-指标字典.md) | 每个字段家族的定义、时刻、分母、横轴、条件和无效值 |
| [完整实现与验收](2026-10-22-完整指标实现与验收.md) | 代码范围、CPU/GPU 证据和启动脚本 |
| [E15 完整训练分析](../research/2026-10-23-E15训练动态与指标分析.md) | 200 次更新、2377 个数值指标、八组图；只分析 krkmq098 |
| [E15 实测字段目录](../data/e15-krkmq098-20261023/metric-catalog.csv) | 实际横轴、覆盖次数、常量标记、范围和窗口统计 |
| [原字段至当前映射](data/metrics-v3-20261022/old-to-current.csv) | 493 个旧字段逐项对照 |
| [E3 实测字段](data/metrics-v3-20261022/observed-fields.csv) | 900 个实际标量字段、横轴和出现次数 |
| [IS/RS 实测字段](data/metrics-v3-20261022/is-rs-observed-fields.csv) | 289 个实际标量字段、横轴和出现次数 |
| [需求完成状态](data/metrics-v3-20261022/requirements-status.csv) | 对照最初审计清单，不把未适用或暂缓项算作已测量 |
| [条件生产端迁移](data/metrics-v3-20261022/conditional-to-current.csv) | critic、top-k、RS、异步等旧运行快照未覆盖的能力 |

基础指标随实际功能启用。训练分位数使用 `trainer.metrics_quantile_interval=8`；eval 重采样使用 `trainer.eval_metrics_resamples=256`，局部 seed 复用 data.seed（None 使用 42）。完整记录与默认 Console/面板显示分开，不靠裁剪出口控制屏幕数量。

## 历史记录

第一版字典、迁移审计、设计报告、旧性能合同及 Correction helper 由[历史指标索引](../archive/README.md#指标系统历史)集中导航。它们保留当时的字段与验收范围；当前配置和定义以本页主表为准。
