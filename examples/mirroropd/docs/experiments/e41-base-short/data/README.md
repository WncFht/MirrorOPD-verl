# 数据说明与复现

本目录包含两组实验的完整训练回答和逐更新指标，以及生成图表所需的诊断数据。所有绘图输入均在此目录内；重算不需要模型权重或GPU。

## 实验标识

| 数据目录 | 图文名称 | 优化目标 |
| --- | --- | --- |
| `baseline/` | 原始词表 KL | 原始完整词表 reverse KL |
| `aligned/` | 终止动作对齐 | 将151643与151645合为一个停止动作后的 reverse KL |

两组均为Qwen3-1.7B-Base学生、Qwen3-4B教师，四张H800、20个outer、batch64、reuse16、320次真实更新。科学配置见[configuration.json](configuration.json)。两组的原始 `distill/full_kl`、`full_kl_position` 保持同一定义；对齐组优化目标单独记录为 `distill/termination_equiv_kl`。

## 压缩原始数据

每个实验目录包含：

| 文件 | 内容 |
| --- | --- |
| `metrics.jsonl.gz` | 原始逐更新和逐outer指标，保留数值精度 |
| `responses.jsonl.gz` | 20批共1280条完整回答，包括prompt、输出文本、参考答案、verifier得分、原始response token ID和长度 |
| `gpu-samples.jsonl.gz` | 每秒采集的四卡显存和GPU忙碌率，覆盖初始化、训练及退出 |
| `prefix-results.json` | 固定前缀上的师生分布、结束概率、原始KL、对齐KL及局部logit梯度 |
| `analysis-reference.json` | 归档的独立分析结果，用于核对重算结果 |
| `execution-summary.json` | 进程退出码和作业墙钟，不作为纯生成测速 |

JSONL按ASCII换行分隔。生成文本可以含Unicode行分隔符，应使用文件逐行读取，不要对解压后的整段文本调用会拆分Unicode行分隔符的`splitlines()`。

## 可直接使用的统计表

| 文件 | 行数 | 统计单位 |
| --- | ---: | --- |
| [outer-metrics.csv](outer-metrics.csv) | 40 | 一组实验的一个outer step |
| [update-metrics.csv](update-metrics.csv) | 640 | 一次optimizer更新前的观测 |
| [full-kl-position.csv](full-kl-position.csv) | 6400 | 一次观测的一个相对位置桶 |
| [position-profiles.csv](position-profiles.csv) | 80 | 一组实验、一个窗口、第1或16次观测、一个位置桶 |
| [window-metrics.csv](window-metrics.csv) | 4 | 一组实验的前5步或后5步，共320条新回答 |
| [reuse-profiles.csv](reuse-profiles.csv) | 64 | 一组实验、一个窗口、一个复用内观测位置 |
| [response-metrics.csv](response-metrics.csv) | 2560 | 一条真实训练回答 |
| [fixed-prefix-stop.csv](fixed-prefix-stop.csv) | 40 | 一个模型在一个固定结束前缀上的停止概率 |
| [loop-prefix.csv](loop-prefix.csv) | 10 | 一组学生模型在一个指定循环次数下的10样本中位数 |

CSV和JSON保留原始或未舍入的派生数值，图文使用千token、百分比和适当小数位。长度单位为token，时间字段以秒为单位，KL以nats/token为单位；字段名中的`pct`是百分数，`repeat8`是0–1比例。

## 重复、长度与位置口径

长度直接来自保存的原始response token，不对解码文本重新分词。触顶表示长度达到7168个token。长循环要求周期为1–128个token、连续片段至少128个token且至少重复4次。重复8-gram比例是每条回答内部重复窗口的比例，先逐回答计算，再对回答取平均。

同一outer中的16次loss观测复用同一批前缀，且都在各自更新之前计算。位置桶按实际回答长度划分；聚合必须先累加sum和token count，再相除。位置图的空桶表示缺测，不补成零。前后窗口的位置剖面按token加权，复用曲线若对批次取均值则按批次等权，两种汇总不混用。

`optim/actor/loss`是实际优化目标；原始词表KL另行保留。改变目标后不能将两组优化loss的绝对大小直接当作生成质量比较。verifier得分来自训练题目，没有独立验证结果。

## 固定前缀诊断

[prefix-cases.json.gz](prefix-cases.json.gz)保存26个诊断样本及精确prompt/response token、138个被观察的前缀位置；两组使用同一份病例。每个实验的`prefix-results.json`包含初始学生、训练后学生相对教师的276条分布比较记录。

结束概率图选用其中10个早期自然结束前缀，循环前缀图选用10个后期循环样本。这些是有目的的机制诊断样本，不用于估计所有输入上的发生率。初始化模型与教师的重复记录按相同前缀读取，不能当成额外独立样本。

[first-batch-reference.json](first-batch-reference.json)记录原始模型独立重算首批训练KL的结果；[stop-gradient-identity.json](stop-gradient-identity.json)记录停止方向梯度恒等式的数值误差；[protocol.json](protocol.json)说明精度和样本选择。诊断公式和解释见[机制说明](../2026-10-26-终止动作与循环退化诊断.md)。

## 样本与代码

[sample-pairs.json](sample-pairs.json)保存两条同题样例：两组verifier均给分、两组均不给分各一条。选择规则和原文片段见[同题样本对照](../2026-10-26-同题样本对照.md)。完整回答仍在原始压缩数据中。

[实现差异](../code/termination-equivalence.patch)使用相对代码路径，保留实验中三个运行文件的改动。它是实验性目标变更，不会由绘图脚本自动应用。对应的[数值测试](../code/test_termination_kl.py)需要PyTorch和已应用该补丁的代码副本，可通过`E41_PATCHED_SOURCE_ROOT`指定该副本；绘图与数据重算不需要这些训练依赖。

## 重算与审计

在实验文档根目录运行：

```bash
python scripts/build_data.py
python scripts/render_figures.py
```

`build_data.py`仅依赖NumPy和标准库；`render_figures.py`另需Matplotlib。前者从压缩原始数据重新统计所有回答和位置桶，并核对每批64条回答、全部320次真实更新、有限数值、位置桶重建KL、无valid及20批prompt集合一致，结果写入[audit.json](audit.json)。

绘图使用[style.json](style.json)中的figure-viz配色和字体设置，离线可独立重绘。在figure-viz审计器可导入时执行完整几何与数据审计；否则执行基础几何检查并明确标明范围。已归档的图形审计见[figures/audit.json](../figures/audit.json)。
