# 作者公开曲线的后期核查

本目录保存作者公开的小型 JSON 副本、来源版本和处理脚本。完整报告见[Vanilla OPD 后期拟合与任务成绩](../../../2026-10-31-VanillaOPD后期拟合与任务成绩核查.md)。

- author-trajectories.json：25 条 Vanilla、17 条 Delta 的绘图数据。
- author-peaks.json：teacher 分数、SFT 汇总与峰值表。
- author-law.json：公式系数与配对预测。
- author-manifest.json：公开权重导出的选择与记录分数，不包含模型权重。
- provenance.json：固定 GitHub／Hub 版本及每份副本的 SHA256。

## 处理范围

analyze.py 只处理 25 条 Vanilla 曲线，提取初始／峰值／最后记录、最后 50 项均值和对 S0 的位移。缺失 teacher KL 保持空白。最后记录的数组下标与真实 optimizer step 分开命名。

原始网页、文件清单和下载文件位于仓库同级 wandb-logs/MirrorOPD-verl/literature/scaling-opd-2609.32722v1/author-data。完整路径与哈希也保存于报告的 sources.json。

## 从仓库根目录复算

~~~bash
export FIGURE_VIZ_MPL=/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/dotfiles/.agents/skills/figure-viz/assets/mpl
/root/.local/share/mirroropd/venvs/plotting/bin/python examples/mirroropd/docs/literature/teacher-transfer/sources/scaling-opd-2609.32722v1/late-training/analyze.py --raw ../wandb-logs/MirrorOPD-verl/literature/scaling-opd-2609.32722v1/author-data --output examples/mirroropd/docs/literature/teacher-transfer/data/scaling-opd-late-training-20261031 --plot
~~~

原图提取：

~~~bash
uv run --no-project --with pymupdf python examples/mirroropd/docs/literature/teacher-transfer/sources/scaling-opd-2609.32722v1/late-training/extract_proxy_figure.py --pdf ../wandb-logs/MirrorOPD-verl/literature/scaling-opd-2609.32722v1/paper.pdf --output examples/mirroropd/docs/literature/teacher-transfer/data/scaling-opd-late-training-20261031
~~~

这些步骤只使用本地文件和 CPU。图 10 保留原始 proxy 曲线，未作数值数字化；其图示结果与网页曲线的末值分别解释。两张重绘图已运行 figure-viz 审计并目检，面板自动取值范围的提示保留于审计文件。
