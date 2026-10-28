# Polaris研究与实现数据

按用户要求，Polaris系列的报告证据和复算输入集中在本目录，不再写入`wandb-logs/analysis`。报告与图在上一级，分析脚本在`../scripts/`，生产Forward KL说明在[实现入口](../../../../infra/numerics/full-vocab/forward-kl.md)。

| 目录 | 内容 |
| --- | --- |
| `polaris-grcuotmi-20261026/` | E48最初58 outer冻结快照及初始师生probe |
| `polaris-complete-20261027/` | E48完整256 outer/4096更新与同预算E20 |
| `state-transfer-20261027/` | 固定state迁移、teacher来源、温度和梯度几何 |
| `intersection-study-20261027/` | 扩大固定面板、fresh rollout、交集采样与任务评估 |
| `forward-kl-study-20261027/` | KL方向文献、12组短对照、teacher基座/后训练增量诊断 |
| `forward-kl-implementation-20261027/` | 生产FKL数值检查、分布式验收、真实logits检查及head性能数据 |

已有数值和原始事件保持原样；`source.json`和历史配置中记录的原路径、提交、时间属于来源证据，不因迁移改写。README与命令中的可执行路径已更新。

大型logits、teacher hidden缓存和本机checkpoint位于源码仓库的`outputs/polaris-transfer-cache/`，三个子目录为`polaris-probe-20261026`、`state-transfer-20261027`、`intersection-study-20261027`。这些文件用于重跑模型实验，不需要随报告阅读而加载。

原始训练`.wandb`仍是日志来源；通用解码器继续在独立日志仓库维护。需要新冻结快照时，将项目export的`--out`指向本目录的新子目录，避免覆盖旧分析边界。迁移没有修改模型资产、正式训练记录或云端run。
