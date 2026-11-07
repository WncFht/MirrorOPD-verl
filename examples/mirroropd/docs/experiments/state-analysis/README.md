# State 分析

本专题分析不同 rollout policy 实际访问的前缀状态、这些状态相对教师参考分布的位置，以及访问差异是否对应 OPD 学习信号差异。核心对象是 $s_t=(x,y_{<t})$ 所诱导的状态分布；表示、分层方式和统计单位要与比较问题匹配。精确前缀适合查重和轨迹审计，不作为大规模语义分布的主统计。

## 当前实现与研究入口

[One-Shot OPD State Coverage 复现设置](2026-11-06-One-Shot-State-Coverage复现.md)记录论文的 teacher hidden 提取、PCA、K-means、reference/held-out 切分与累计 coverage。实现位于 [`research/state_analysis`](../../../research/state_analysis/)，是独立工具包，不属于单项 study。当前已完成 CPU 合成测试与 trajectory loader 验收；真实 teacher 提取和论文数值仍待完整数据与 GPU。

专题扩展按回答的问题组织，维持一份完整的方法与设置正文：

| 分析方向              | 主要回答                                          | 当前状态                                                                                   |
| --------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| 共享表示上的状态覆盖  | policy 是否触及 full-data rollout 的主要状态簇？  | One-Shot coverage 实现已就绪，待真实数据运行                                               |
| MMD / 簇占用差异      | 两种 rollout policy 的状态分布整体相差多少？      | 方法讨论见[文献调研](../../literature/opd/2026-11-06-One-Shot-OPD的State分析与相关技术.md) |
| Held-out 近邻支持诊断 | 哪些 policy 更常进入教师参考样本难以支持的区域？  | 待实现；论文 10-NN 分析已整理在文献调研                                                    |
| 固定状态上的学习信号  | 访问差异是否对应 teacher-student gap 或 KL 差异？ | 与 state coverage 分开报告，方案待接入                                                     |

## 比较原则

对照 policy 应使用同一 prompt 集与目标 prompt 权重、同一生成预算和停止规则、同一冻结 teacher 表示、同一 PCA/K-means 参考空间。优先按 prompt 与预先定义的生成深度分层，再汇总分布差异；重采样单位是完整 rollout，若需要跨题泛化结论则按 prompt 分割。使用的表示分布是前缀分布经固定表示映射后的推前分布，不应解释为原始前缀分布等价。

coverage 衡量命中多少参考簇，不衡量访问频率、cluster 内距离或监督价值；MMD 是整体分布摘要，近邻距离诊断参考支持尾部，固定状态 KL/gap 解释局部学习信号。几种结果分别回答不同问题，不压成一个未经校准的 state diversity 分数。表示、聚类粒度、生成深度与长度权重都要作为设置记录；研究报告区分已测结果、解释和待验证假设。

### 参考文献

方法依据见[One-Shot OPD State 分析与相关技术](../../literature/opd/2026-11-06-One-Shot-OPD的State分析与相关技术.md)，包括原论文 state coverage、MMD、近邻支持和分布比较文献。论文原始来源为 arXiv:2609.04172。
