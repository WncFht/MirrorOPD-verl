# 2609.32722v1 原图与来源核验

对应[单篇精读笔记](../../2026-10-31-同家族OPD缩放规律精读与实验启发.md)。完整原文采用 arXiv v1；PDF／TeX 的来源、仓库外路径与 SHA256 见 [provenance.json](provenance.json)。图像来自作者 PDF 的局部渲染，图中数据、坐标、文字与图例未改写。TeX 是 link-get 的汇总输出，图号和页码以 PDF 为准。

## 编号与选图

| PDF 图号 | PDF 页码 | HTML 锚点／编号 | 阅读用途 |
| --- | --- | --- | --- |
| 2 | 4 | S4.F2／图 2 | 早期动态与后期回落 |
| 4 | 5 | S4.F4／图 4 | 25 格尺寸网格 |
| 5 | 8 | S5.F5／图 5 | teacher 分数与教学价值 |
| 6 | 8 | S5.F8／图 8 | 全样本拟合图与留出验证区别 |
| 7 | 8 | S6.F9／图 9 | Delta 与 Vanilla |
| 8 | 9 | S6.F10／图 10 | 冷启动与纯 OPD |
| 9 | 9 | S6.F11／图 11 | 逐级蒸馏与直接迁移 |

HTML 转换把部分子图／子表单独编号，不能将其编号直接用于 PDF。裁剪坐标、DPI、像素尺寸和图片摘要见[图像 manifest](../../figures/scaling-opd-2609.32722v1/manifest.json)。七张图均已目检，未截断坐标轴或图例。

## 复现

从仓库根目录运行以下命令，PDF 参数指向相同版本原文。脚本要求摘要一致，版本或排版改变时应先重新验收裁剪范围。

~~~bash
uv run --no-project --with pymupdf python examples/mirroropd/docs/literature/teacher-transfer/sources/scaling-opd-2609.32722v1/extract_figures.py --pdf ../wandb-logs/MirrorOPD-verl/literature/scaling-opd-2609.32722v1/paper.pdf --output examples/mirroropd/docs/literature/teacher-transfer/figures/scaling-opd-2609.32722v1
~~~

PNG 是用于正文读图的精选摘录，可以随笔记版本管理；原始 PDF／TeX 位于仓库外。无需下载模型或调用 GPU。原始论文作者为 Yuntai Bao 等，来源链接保存在 manifest 与笔记脚注中。

## 公式代入例

[compute_predictions.py](compute_predictions.py)只将论文公布的近似系数代入三个真实配对的输入，不重新拟合；[prediction-example.json](prediction-example.json)区分计算值与原文观测值。系数经过四舍五入，计算结果不保证逐位重现作者精确预测。
