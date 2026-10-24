# 师生参考与批内收敛数据

对应[分析报告](../../research/2026-10-24-Full-KL收敛与师生参考评测.md)。本目录是派生分析，不包含新模型评测。

- `references.json`：读取已有 One-Shot student `6uorrk2k` / teacher `u87buult` 的 2026-10-20 API 快照、`valid-parameters-20261022` 的本仓原始 student 验证汇总，以及 `five-run-comparison-20261024` 的冻结分析。保留来源路径和采样配置，明确缺失长度/熵、旧评测管线差异及 E17 边界。
- `mathematical-checks.json`：有限概率分布枚举；验证序列 KL 链式法则、条件 KL 下降但序列 KL 上升的反例，以及等熵但分布不同的反例。不是语言模型的实测结果。

历史参考只叠加 mean accuracy，不将旧 bootstrap best/majority 估计器冒充新 pass/majority；不把历史参考作为五个 run 的 step 0。批内统计首末均为更新前测量，覆盖 E16 的 3 次、E17 的 15 次已执行更新。

复现入口见[分析复现](../../分析复现.md)。图表在 `figures/teacher-convergence-20261024`；沿用原五组比较的[单文件离线 HTML](../five-run-comparison-20261024/dashboard.html)，其中数据和绘图库全部内嵌。图审计没有错误，仅提示三个数据集面板采用独立纵轴；五条实验曲线加两条灰色历史参考线已人工检查。
