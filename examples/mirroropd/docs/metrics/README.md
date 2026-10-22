# 训练指标

Last updated: 2026-10-22。

当前实现为 **policy-v3**：恢复条件功能的观测，按所有支持的采样预算输出 eval，补齐 token/sequence pair 和实际 IS/RS 统计。固定 64 字段允许列表已移除。E3 与 token IS + token_k1 RS 均已完成四卡实验和云端回读，详见实现记录。

| 文档 | 内容 |
| --- | --- |
| [当前指标字典](2026-10-22-指标字典.md) | 每个字段家族的定义、时刻、分母、横轴、条件和无效值 |
| [完整实现与验收](2026-10-22-完整指标实现与验收.md) | 代码范围、CPU/GPU 证据和启动脚本 |
| [原字段至当前映射](data/metrics-v3-20261022/old-to-current.csv) | 493 个旧字段逐项对照 |
| [E3 实测字段](data/metrics-v3-20261022/observed-fields.csv) | 900 个实际标量字段、横轴和出现次数 |
| [IS/RS 实测字段](data/metrics-v3-20261022/is-rs-observed-fields.csv) | 289 个实际标量字段、横轴和出现次数 |
| [需求完成状态](data/metrics-v3-20261022/requirements-status.csv) | 对照最初审计清单，不把未适用或暂缓项算作已测量 |
| [条件生产端迁移](data/metrics-v3-20261022/conditional-to-current.csv) | critic、top-k、RS、异步等旧运行快照未覆盖的能力 |
| [第一版迁移审计](2026-10-22-指标迁移审计与补齐清单.md) | 保留当时的 493 字段盘点与问题来源 |
| [第一版字典](2026-10-22-第一版指标字典.md) | 历史 64 字段口径，不再是当前字段上限 |
| [设计与调研](2026-10-22-训练指标系统设计与调研.md) | 框架参考与第一版取舍，历史记录 |
| [历史性能口径](2026-10-22-历史性能口径与验收.md) | timer、MFU、内存与旧 GPU 实验边界 |
| [旧 Correction helper](2026-10-22-旧Correction辅助字段.md) | 原公式与当前全局 raw/applied 统计的区别 |

基础指标随实际功能启用。训练分位数使用 `trainer.metrics_quantile_interval=8`；eval 重采样使用 `trainer.eval_metrics_resamples=256`，局部 seed 复用 data.seed（None 使用 42）。完整记录与默认 Console/面板显示分开，不靠裁剪出口控制屏幕数量。
