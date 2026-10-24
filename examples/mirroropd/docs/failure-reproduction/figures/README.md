# 原图来源与处理

获取日期：2026-10-24。本目录保存支持失败设置的原论文图，供[实验方案](../2026-10-24-Sampled-token与Full-vocab失败复现方案.md)引用。图中文字、坐标、曲线和数据均来自作者；没有重新拟合数据或绘制替代曲线。

## Rethinking OPD，arXiv:2604.13016v2

作者：Yaxuan Li等。图源来自该版本的arXiv HTML／PDF；页面声明CC BY 4.0。本文保留SVG原件，并提供PNG显示版，转换或截取方式如下。[^rethinking]

| 原文图号 | 本地原件 | 本地显示版 | 来源与处理 |
| --- | --- | --- | --- |
| 图2，初始对齐与教师选择 | [SVG](rethinking-fig02-initial-alignment.svg) | [PNG](rethinking-fig02-initial-alignment.png) | `3_1_metrics.svg`，按比例渲染为宽2000像素PNG[^r2] |
| 图8，离线冷启动 | [SVG](rethinking-fig08-cold-start.svg) | [PNG](rethinking-fig08-cold-start.png) | `4_2_combined.svg`，按比例渲染为宽2000像素PNG[^r8] |
| 图5，反向蒸馏 | [SVG](rethinking-fig05-reverse.svg) | [PNG](rethinking-fig05-reverse.png) | `JustRL_val.svg`，按比例渲染为宽2000像素PNG[^r5] |
| 图6，成功／失败教师对照 | [SVG](rethinking-fig06-teacher-pairs.svg) | [PNG](rethinking-fig06-teacher-pairs.png) | `5_1_combined.svg`，按比例渲染为宽2000像素PNG[^r6] |
| 图11，长度汇总与教师续写收益 | [SVG](rethinking-fig11-length-summary.svg) | [PNG](rethinking-fig11-length-summary.png) | PNG取自PDF第14页图区域；SVG原件另保留[^r11] |
| 图12，长窗口动态 | [SVG](rethinking-fig12-length-dynamics.svg) | [PNG](rethinking-fig12-length-dynamics.png) | `6_1_resp_length.svg`，按比例渲染为宽2000像素PNG[^r12] |

图11的PNG从PDF页面坐标矩形 `(62,84,534,234)` 截取，单位为PDF point，以约2000像素宽渲染；保留两个子图及坐标轴，原英文caption在主文中以中文说明替代。使用PDF显示版是为了正确保留柱状图的裁剪边界。

## TIDE，arXiv:2608.09836v1

作者：Zichao Yu等。arXiv页面声明arXiv perpetual non-exclusive license；本目录按研究对照需要引用图1、图2，不把该声明写成CC开放许可。图的权利归原作者／权利人。[^tide]

| 原文图号 | 本地原件 | 本地显示版 | 来源与处理 |
| --- | --- | --- | --- |
| 图1，准确率与长度 | [SVG](tide-fig01-accuracy-length.svg) | [PNG](tide-fig01-accuracy-length.png) | PNG取自PDF第1页图区域，SVG原件另保留[^t1] |
| 图2，重复前缀上的师生分布 | [原PNG](tide-fig02-repetition.png) | 同原PNG | 直接下载作者的 `repetition_distribution_heatmap.png`，未转换[^t2] |

图1的PNG从PDF第1页坐标矩形 `(107,504,505,642)` 截取，单位为PDF point，以约2000像素宽渲染。这样保留原图的填充和裁剪效果；未裁掉子图、图例或数据。SVG转换与PDF截取使用PyMuPDF，显示版均已经人工查看。

方案中的TIDE数字表摘录自原文表1与表3，标明了原始Avg@8、九基准macro及严格boxed判分口径。它们不是本项目三个valid上的新实验结果。

### 参考来源

[^rethinking]: Rethinking OPD v2。[HTML](https://arxiv.org/html/2604.13016v2)、[PDF](https://arxiv.org/pdf/2604.13016v2)、[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。
[^r5]: 图5的arXiv原始SVG。[原文件](https://arxiv.org/html/2604.13016v2/JustRL_val.svg)
[^r6]: 图6的arXiv原始SVG。[原文件](https://arxiv.org/html/2604.13016v2/5_1_combined.svg)
[^r11]: 图11的arXiv原始SVG。[原文件](https://arxiv.org/html/2604.13016v2/6_1_response_summary.svg)
[^r12]: 图12的arXiv原始SVG。[原文件](https://arxiv.org/html/2604.13016v2/6_1_resp_length.svg)
[^tide]: TIDE v1。[HTML](https://arxiv.org/html/2608.09836v1)、[PDF](https://arxiv.org/pdf/2608.09836v1)、[arXiv license说明](https://arxiv.org/licenses/nonexclusive-distrib/1.0/)。
[^t1]: 图1的arXiv原始SVG。[原文件](https://arxiv.org/html/2608.09836v1/teaser.svg)
[^t2]: 图2的arXiv原始PNG。[原文件](https://arxiv.org/html/2608.09836v1/repetition_distribution_heatmap.png)

[^r2]: 图2的arXiv原始SVG。[原文件](https://arxiv.org/html/2604.13016v2/3_1_metrics.svg)
[^r8]: 图8的arXiv原始SVG。[原文件](https://arxiv.org/html/2604.13016v2/4_2_combined.svg)
