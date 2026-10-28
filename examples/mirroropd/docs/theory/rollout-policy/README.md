# Rollout 策略设计研究

当前入口：[Rollout 策略的目标设计与求解](2026-10-28-Rollout策略的目标设计与求解.md)。更新日期：2026-10-28。

主文围绕一个问题展开：已知 student 和 teacher 的完整下一 token 分布，如何定义并求解一个用于采集训练轨迹的分布。它将目标聚合、divergence、局部最优条件和工程实现分开，统一使用 student 指数权重 $\alpha$，并区分数学证明、CPU 数值核验和待验证的学习效果。

| 阅读目的 | 主文位置 |
| --- | --- |
| 理解变量、约束、Pareto 取舍及 sum / minimax / 平滑目标 | 第 1–2 节 |
| 比较两种 KL 方向、Hellinger 等中心 | 第 3–4 节 |
| 推导 Chernoff 系数、唯一性、停止条件 | 第 5 节 |
| 判断求解 minimax 相比固定中点是否值得 | 第 6 节 |
| 核对公开代码、EOS、温度与 GPU 求解边界 | 第 7 节 |
| 区分局部与序列目标、解释二阶矩的适用范围 | 第 8 节 |
| 查相关论文的实际结论和实验建议 | 第 9–10 节 |

## 证据与复算

- [中心与公开实现核验结果](../../literature/rollout-policy/data/numerical-audit.json)，对应[脚本](../../../research/rollout_policy/audit_centers.py)。
- [聚合目标恒等式核验结果](../../literature/rollout-policy/data/aggregation-audit.json)，对应[脚本](../../../research/rollout_policy/audit_aggregation.py)。
- [作者 Bernoulli Java 示例输出](../../literature/rollout-policy/data/java-audit.txt)，源码与启动器在 `scripts/`。
- `data/` 保存检索记录与数值证据，`scripts/` 保存复算入口。第三方源码与 PDF 下载缓存保留在原研究目录；主文已引用原始来源。中心核验脚本通过 uv 安装固定 commit 的 pyBregMan，无需依赖该缓存。上述核验均为 CPU 合成分布，不代表 LLM 训练或 GPU 性能结果。

在本目录运行：

```bash
python -m examples.mirroropd.research.rollout_policy.audit_centers --out /tmp/centers-recheck.json
python -m examples.mirroropd.research.rollout_policy.audit_aggregation --out /tmp/aggregation-recheck.json
```

## 完整检索记录

| 报告 | 单轮去重条数 | 来源诊断 |
| --- | ---: | --- |
| [第一轮](../../literature/rollout-policy/data/2026-10-28-论文检索原始报告.md) | 76 | [stderr](../../literature/rollout-policy/data/search.stderr.txt) |
| [定向补充](../../literature/rollout-policy/data/2026-10-28-定向补充检索原始报告.md) | 59 | [stderr](../../literature/rollout-policy/data/targeted-search.stderr.txt) |
| [聚合目标补充](../../literature/rollout-policy/data/2026-10-28-聚合目标补充检索原始报告.md) | 58 | [stderr](../../literature/rollout-policy/data/aggregation-search.stderr.txt) |

报告之间没有跨轮去重。原始报告保留全部命中，包括无关结果；正文的文献评价以已核对来源为准。

## 历史与维护

两份早期讨论已归档，原文件名保留为跳转入口。后续关于定义、推荐顺序和实现范围的修改统一进入主文，避免平行文档出现不同结论。

- [分布中心与 Chernoff 系数讨论](../../archive/literature/rollout-policy/2026-10-28-Rollout分布中心与Chernoff系数调研.md)
- [Sum 与 minimax 聚合讨论](../../archive/literature/rollout-policy/2026-10-28-Sum与Minimax聚合目标选择.md)
- [重构清单与历史文件摘要](../../literature/rollout-policy/data/document-refactor.json)

[迁移清单](../../literature/rollout-policy/data/document-migration.json)记录来源和文件映射；首次复算需要联网安装脚本声明的依赖。历史版本中的命令保留原记录，当前复算方式以上述入口为准。
