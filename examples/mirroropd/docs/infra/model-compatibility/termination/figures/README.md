# 图表文件

PNG 用于直接阅读，SVG 用于放大、排版和编辑。全部图由同一份本地数据与[绘图脚本](../../../../../research/studies/termination/render_figures.py)生成，说明与结论见[实验报告](../README.md)。

| 图                         | PNG                           | SVG                             |
| -------------------------- | ----------------------------- | ------------------------------- |
| 平均长度与后 5 步长度分布  | [预览](length.png)            | [矢量图](length.svg)            |
| 长循环与重复 8-gram        | [预览](repetition.png)        | [矢量图](repetition.svg)        |
| 原始 KL 与实际优化 loss    | [预览](kl-loss.png)           | [矢量图](kl-loss.svg)           |
| 同批次 16 次观测的 KL 变化 | [预览](reuse-kl.png)          | [矢量图](reuse-kl.svg)          |
| 共享色标的位置 KL 热图     | [预览](full-kl-position.png)  | [矢量图](full-kl-position.svg)  |
| 前后窗口的位置 KL 剖面     | [预览](position-profiles.png) | [矢量图](position-profiles.svg) |
| 触顶比例与固定前缀停止概率 | [预览](termination.png)       | [矢量图](termination.svg)       |
| 梯度范数与训练题目得分     | [预览](training-health.png)   | [矢量图](training-health.svg)   |
| 循环前缀的条件预测         | [预览](loop-prefix.png)       | [矢量图](loop-prefix.svg)       |

[完整图形审计](audit.json)包含九组图的检查结果。位置热图共用线性色标；概率和 KL 的对数轴均在图中注明。所有曲线连接实际观测点，无平滑或缺失值补齐。
