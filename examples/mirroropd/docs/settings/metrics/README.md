# 实验指标与比较口径

指标解释集中在这里，正式实验使用同一套横轴、分母和验证预算语义。当前实现见[指标系统走读](../../infra/architecture/metrics.md)；旧设计、迁移和验收材料归入[指标工程档案](../../archive/README.md#指标系统历史)。

| 文档 | 用途 |
| --- | --- |
| [当前指标字典](../指标口径.md) | policy-v3 的字段、采集时刻、分母、条件、横轴和无效值 |
| [评测预算、KL 精度与位置分桶](2026-10-24-评测预算语义与KL精度及位置分桶.md) | mean、prefix_mean、pass/majority；条件 KL 数值与相对位置十分桶 |
| [第一版字典](../../archive/metrics/2026-10-22-第一版指标字典.md) | 解释早期日志的冻结口径 |
| [旧 Correction 辅助字段](../../archive/metrics/2026-10-22-旧Correction辅助字段.md) | 保留历史日志中的 ratio/ESS 等辅助统计含义 |

## 比较时的共同约定

- `step/outer` 是生成批次，`step/update` 是更新观测，`step/optimizer` 是实际优化器计数；W&B `_step` 只是事件序号。比较时注明采用的轴。
- 把最后更新位置与最后验证位置分别登记；稀疏 summary 中保留的验证值可能来自更早 checkpoint。
- 报告 mean accuracy、pass@k、majority@k 时同时说明题目数、每题回答数、采样温度、长度与判分器。三个指标不能互换。
- 批内 KL 在对应 optimizer 更新前观测。首末差不包含最后一次更新之后的额外 forward；新批起点与旧批最低值分别分析。
- 题目 bootstrap 表示题目维度的不确定性；同一训练的多个 checkpoint 不是多个 seed。
- 缺少字段保留缺失，不补零。历史 run 没有位置分桶时，不能由总体 KL 反推出前缀或后缀 KL。

具体指标覆盖和结果进入各系列分析。日志中的参数和 metric definitions 是解释该次运行的依据，当前字典不回写历史记录。
