# 迁移机制证据索引

本目录保存报告随附的统计、图形和历史冻结证据。原始运行日志、完整逐题输出、大权重与teacher hidden仍保存在仓库同级`wandb-logs/`，不因文档重构移动或覆盖。

| 目录 | 内容 |
| --- | --- |
| `state-transfer-20261027/` | 固定state迁移、teacher来源、温度与梯度几何的历史证据 |
| `intersection-study-20261027/` | 固定面板、fresh rollout、交集采样与任务评估的历史证据 |
| `overview-20261029/` | 当前模型清单、逐领域风险、整体held-out排序、每 student／输入模式一张图及绘图审计 |

原始历史配置与`source.json`保留原路径、版本和时间。当前汇总由[overview.py](../../../../../research/studies/fixed_transfer/overview.py)从完成回执与逐题评分复算；`heldout-ranking.csv`和`figure-order.json`记录当前排序及标签。Polaris生产实验另见[跨来源Polaris研究线](../../cross-origin/polaris/README.md)，不与这里的小型固定状态谱系统计合并。

[返回研究入口](../README.md) · [数据与指标定义](../protocol.md)

当前总览按 R1-1.5B、R1-7B、Qwen3-1.7B thinking／non-thinking、Llama-3B off 分线；另将 2507 原生模板对照单独成图。续训按模式条件去重，只读取完成回执对应的终点；`inventory.json` 分别记录权重配对数和模式条件数。
