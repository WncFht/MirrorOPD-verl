# Rollout 策略设计研究

当前入口：[Rollout 来源、最优融合与等效训练权重](objectives.md)。更新日期：2026-11-06。

主文从来源分布和 learner loss 出发，逐步推导统一的等效 advantage，再比较不同最优化目标得到的来源。全文用 $\eta$ 表示 teacher 权重；initial、teacher、局部 fusion 和 IW-OPD 在同一张表中比较，并区分精确 IS、前缀 surrogate 与实验权重。

| 阅读目的                                  | 主文位置                                   |
| ----------------------------------------- | ------------------------------------------ |
| 明确来源、学习目标与符号                  | [第 1 节](objectives.md#notation)          |
| 逐步推导 fresh rollout 上的等效 advantage | [第 2 节](objectives.md#change-of-measure) |
| 求解 sum、minimax、预算和不同距离的中心   | [第 3 节](objectives.md#centers)           |
| 比较各来源的权重及局部 / 序列差异         | [第 4 节](objectives.md#source-weights)    |
| 判断单条 rollout 能算什么、何时需要多条   | [第 5 节](objectives.md#computability)     |
| 求解系数、控制误差并比较 sum / minimax    | [第 6 节](objectives.md#solver)            |
| 核对执行条件、公开实现和数值证据          | [第 7 节](objectives.md#evidence)          |

## 证据与复算

- [中心与公开实现核验结果](../../literature/rollout-policy/data/numerical-audit.json)，对应[脚本](../../../research/checks/rollout_policy/audit_centers.py)。
- [聚合目标恒等式核验结果](../../literature/rollout-policy/data/aggregation-audit.json)，对应[脚本](../../../research/checks/rollout_policy/audit_aggregation.py)。
- [作者 Bernoulli Java 示例输出](../../literature/rollout-policy/data/java-audit.txt)，源码与启动器在 `scripts/`。
- [来源权重与等效 advantage 核验脚本](../../../research/checks/rollout_policy/audit_source_weights.py)：精确枚举前缀分布、局部梯度、重复更新动作比和序列中心，并检查单样本与自归一化 IS，范围见[主文第 7 节](objectives.md#evidence)。
- `data/` 保存检索记录与数值证据，`scripts/` 保存复算入口。第三方源码与 PDF 下载缓存保留在原研究目录；主文已引用原始来源。中心核验脚本通过 uv 安装固定 commit 的 pyBregMan，无需依赖该缓存。上述核验均为 CPU 合成分布，不代表 LLM 训练或 GPU 性能结果。

从仓库根目录运行：

```bash
python -m examples.mirroropd.research.checks.rollout_policy.audit_centers --out /tmp/centers-recheck.json
python -m examples.mirroropd.research.checks.rollout_policy.audit_aggregation --out /tmp/aggregation-recheck.json
python -m examples.mirroropd.research.checks.rollout_policy.audit_source_weights --out runs/rollout-policy-is-20261106/verification.json
```

## 完整检索记录

| 报告                                                                                        | 单轮去重条数 | 来源诊断                                                                     |
| ------------------------------------------------------------------------------------------- | -----------: | ---------------------------------------------------------------------------- |
| [第一轮](../../literature/rollout-policy/data/2026-10-28-论文检索原始报告.md)               |           76 | [stderr](../../literature/rollout-policy/data/search.stderr.txt)             |
| [定向补充](../../literature/rollout-policy/data/2026-10-28-定向补充检索原始报告.md)         |           59 | [stderr](../../literature/rollout-policy/data/targeted-search.stderr.txt)    |
| [聚合目标补充](../../literature/rollout-policy/data/2026-10-28-聚合目标补充检索原始报告.md) |           58 | [stderr](../../literature/rollout-policy/data/aggregation-search.stderr.txt) |

报告之间没有跨轮去重。原始报告保留全部命中，包括无关结果；正文的文献评价以已核对来源为准。

## 历史与维护

两份早期讨论已归档，原文件名保留为跳转入口。后续关于定义、推荐顺序和实现范围的修改统一进入主文，避免平行文档出现不同结论。

- [分布中心与 Chernoff 系数讨论](../../archive/literature/rollout-policy/2026-10-28-Rollout分布中心与Chernoff系数调研.md)
- [Sum 与 minimax 聚合讨论](../../archive/literature/rollout-policy/2026-10-28-Sum与Minimax聚合目标选择.md)
- [重构清单与历史文件摘要](../../literature/rollout-policy/data/document-refactor.json)

[迁移清单](../../literature/rollout-policy/data/document-migration.json)记录来源和文件映射；首次复算需要联网安装脚本声明的依赖。历史版本中的命令保留原记录，当前复算方式以上述入口为准。
