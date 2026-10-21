# 训练指标

Last updated: 2026-10-22。

本目录统一维护当前指标定义、统计归约、日志时间轴与验证边界。新 V1 运行只输出新字段，不双写旧 actor/critic/rollout_corr/val-core/val-aux 名字；历史数据原样保留。

| 文档 | 内容 |
| --- | --- |
| [当前指标字典](2026-10-22-指标字典.md) | 数学 E 系列 64 个可能字段逐项解释、模式差异、单位和采集时刻 |
| [实现与验证](2026-10-22-实现与验证.md) | 实际代码链路、唯一新配置、CPU/Gloo/W&B 测试和未验收范围 |
| [设计与调研](2026-10-22-训练指标系统设计与调研.md) | 八类指标、概率 pair、上游 PR 与架构取舍 |
| [历史性能口径与验收](2026-10-22-历史性能口径与验收.md) | 原 infra 指标说明中的计时、显存及历史 GPU 证据 |
| [旧 Correction 辅助字段](2026-10-22-旧Correction辅助字段.md) | 旧 helper 返回值与新 raw pair 诊断的区别 |
| [历史字段处置](data/metrics-system-20261021/metric-migration-20261022.csv) | 493 个旧字段的替换、重新测量与删除决定 |

唯一新增用户配置为 `trainer.metrics_quantile_interval=8`：0 关闭 p99，1 每次更新，N>1 在首次及每 N 次更新尝试记录。其他指标按已有输入/算法自动启用。框架保留优化、策略、优势、修正、蒸馏、采样、评估、性能八类语义目录，使用多个斜杠表示角色与 pair。

原始调研数据见 [source snapshot](data/metrics-system-20261021/research-snapshot.json)，实际测试汇总见 [verification](data/verification-20261022.json)。
