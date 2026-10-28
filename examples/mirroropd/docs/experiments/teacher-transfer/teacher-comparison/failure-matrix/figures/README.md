# 原图来源与处理

获取日期：2026-10-24。本目录保存支持失败设置的原论文图，供[实验方案](../2026-10-25-Sampled-token与Full-vocab失败复现方案.md)引用。图中文字、坐标、曲线和数据均来自作者；没有重新拟合数据或绘制替代曲线。

## Rethinking OPD, arXiv:2604.13016v2

作者：Yaxuan Li 等。图源来自该版本的 arXiv HTML／PDF；页面声明 CC BY 4.0。本文保留 SVG 原件，并提供 PNG 显示版，转换或截取方式如下。[^rethinking]

| 原文图号                      | 本地原件                                      | 本地显示版                                    | 来源与处理                                                |
| ----------------------------- | --------------------------------------------- | --------------------------------------------- | --------------------------------------------------------- |
| 图 2，初始对齐与教师选择      | [SVG](rethinking-fig02-initial-alignment.svg) | [PNG](rethinking-fig02-initial-alignment.png) | `3_1_metrics.svg`，按比例渲染为宽 2000 像素 PNG[^r2]      |
| 图 8，离线冷启动              | [SVG](rethinking-fig08-cold-start.svg)        | [PNG](rethinking-fig08-cold-start.png)        | `4_2_combined.svg`，按比例渲染为宽 2000 像素 PNG[^r8]     |
| 图 5，反向蒸馏                | [SVG](rethinking-fig05-reverse.svg)           | [PNG](rethinking-fig05-reverse.png)           | `JustRL_val.svg`，按比例渲染为宽 2000 像素 PNG[^r5]       |
| 图 6，成功／失败教师对照      | [SVG](rethinking-fig06-teacher-pairs.svg)     | [PNG](rethinking-fig06-teacher-pairs.png)     | `5_1_combined.svg`，按比例渲染为宽 2000 像素 PNG[^r6]     |
| 图 11，长度汇总与教师续写收益 | [SVG](rethinking-fig11-length-summary.svg)    | [PNG](rethinking-fig11-length-summary.png)    | PNG 取自 PDF 第 14 页图区域；SVG 原件另保留[^r11]         |
| 图 12，长窗口动态             | [SVG](rethinking-fig12-length-dynamics.svg)   | [PNG](rethinking-fig12-length-dynamics.png)   | `6_1_resp_length.svg`，按比例渲染为宽 2000 像素 PNG[^r12] |

图 11 的 PNG 从 PDF 页面坐标矩形 `(62,84,534,234)` 截取，单位为 PDF point，以约 2000 像素宽渲染；保留两个子图及坐标轴，原英文 caption 在主文中以中文说明替代。使用 PDF 显示版是为了正确保留柱状图的裁剪边界。

## TIDE, arXiv:2608.09836v1

作者：Zichao Yu 等。arXiv 页面声明 arXiv perpetual non-exclusive license；本目录按研究对照需要引用图 1、图 2，不把该声明写成 CC 开放许可。图的权利归原作者／权利人。[^tide]

| 原文图号                   | 本地原件                              | 本地显示版                            | 来源与处理                                                        |
| -------------------------- | ------------------------------------- | ------------------------------------- | ----------------------------------------------------------------- |
| 图 1，准确率与长度         | [SVG](tide-fig01-accuracy-length.svg) | [PNG](tide-fig01-accuracy-length.png) | PNG 取自 PDF 第 1 页图区域，SVG 原件另保留[^t1]                   |
| 图 2，重复前缀上的师生分布 | [原 PNG](tide-fig02-repetition.png)   | 同原 PNG                              | 直接下载作者的 `repetition_distribution_heatmap.png`，未转换[^t2] |

图 1 的 PNG 从 PDF 第 1 页坐标矩形 `(107,504,505,642)` 截取，单位为 PDF point，以约 2000 像素宽渲染。这样保留原图的填充和裁剪效果；未裁掉子图、图例或数据。SVG 转换与 PDF 截取使用 PyMuPDF，显示版均已经人工查看。

方案中的 TIDE 数字表摘录自原文表 1 与表 3，标明了原始 Avg@8、九基准 macro 及严格 boxed 判分口径。它们不是本项目三个 valid 上的新实验结果。

### 参考来源

[^rethinking]: Rethinking OPD v2。[HTML](https://arxiv.org/html/2604.13016v2)、[PDF](https://arxiv.org/pdf/2604.13016v2)、[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。

[^r5]: 图 5 的 arXiv 原始 SVG。[原文件](https://arxiv.org/html/2604.13016v2/JustRL_val.svg)

[^r6]: 图 6 的 arXiv 原始 SVG。[原文件](https://arxiv.org/html/2604.13016v2/5_1_combined.svg)

[^r11]: 图 11 的 arXiv 原始 SVG。[原文件](https://arxiv.org/html/2604.13016v2/6_1_response_summary.svg)

[^r12]: 图 12 的 arXiv 原始 SVG。[原文件](https://arxiv.org/html/2604.13016v2/6_1_resp_length.svg)

[^tide]: TIDE v1。[HTML](https://arxiv.org/html/2608.09836v1)、[PDF](https://arxiv.org/pdf/2608.09836v1)、[arXiv license 说明](https://arxiv.org/licenses/nonexclusive-distrib/1.0/)。

[^t1]: 图 1 的 arXiv 原始 SVG。[原文件](https://arxiv.org/html/2608.09836v1/teaser.svg)

[^t2]: 图 2 的 arXiv 原始 PNG。[原文件](https://arxiv.org/html/2608.09836v1/repetition_distribution_heatmap.png)

[^r2]: 图 2 的 arXiv 原始 SVG。[原文件](https://arxiv.org/html/2604.13016v2/3_1_metrics.svg)

[^r8]: 图 8 的 arXiv 原始 SVG。[原文件](https://arxiv.org/html/2604.13016v2/4_2_combined.svg)
