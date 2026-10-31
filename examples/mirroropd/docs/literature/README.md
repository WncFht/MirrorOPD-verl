# 外部研究与来源

这里维护论文、公开源码审阅及模型卡原文，明确版本与核验范围；不将外部结果计入本项目已完成实验。

- `learning-signal/`：估计器、支持集与陈旧数据研究。
- `teacher-transfer/`、`transfer-generalization/`：师生谱系、能力获取及 SFT/RL 相关研究。
- `optimization/`：数值精度与优化文献。
- `rollout-policy/`：生成分布设计的原始检索与数值参考。
- `infra/`：公开工程实现及后端审阅。

Full-RKL 的归一化见 [OPD 消融与 verl 平均方式对照](learning-signal/2026-10-31-Full-RKL的平均方式与MOPD及verl实现对照.md)：区分词表求和、token 平均、回答平均，以及多卡与 micro-batch 的全局分母；比较 OPD 归一化消融与 DAPO、Dr. GRPO 的相关证据。

师生迁移研究可从[蒸馏边界的论文对照与预算依据](teacher-transfer/2026-10-28-蒸馏边界实验的论文对照与预算依据.md)进入：已有配对实验、早期分化、step 含义，以及完整原始检索记录。

选择后续评测任务见[OPD 评测领域与泛化迁移基准调研](teacher-transfer/2026-10-30-OPD评测领域与泛化迁移基准调研.md)：14 篇核验样本的逐篇编码、领域覆盖统计、泛化轴与 benchmark 候选。

Benchmark 采样次数、解码参数、回答预算与初始化银行生成见[四域正确率评测与 RKL 银行生成协议调研](teacher-transfer/2026-10-31-四域正确率评测与RKL银行生成协议调研.md)：比较逐 bench 的 n，区分 accuracy 全部产物与固定 RKL 子集，并记录已选择的公共 32K 与常规 n 的证据及成本。

深入研究迁移机制见[泛化迁移的 Research Question 综述](transfer-generalization/2026-10-31-泛化迁移研究问题与证据综述.md)：围绕 9 个问题比较 SFT/RL 泛化条件、新能力、知识使用、跨域迁移、教师可学习性、状态来源、初始化、多教师整合与证据判别；逐篇设置及新增追问证据分别归入支撑页面。

相对策略变化专题见[Delta OPD 的分类、研究问题与证据](teacher-transfer/2026-10-31-Delta-OPD论文分类与研究问题综述.md)：15 篇主线覆盖 weak-to-strong、strong-to-weak、同尺寸、跨家族、多教师与自身增量，另区分条件对比、表示/梯度方法及解码前驱。

项目自己的数学推导归[theory](../theory/README.md)，实验计划与结果归[experiments](../experiments/README.md)，当前接口归[infra](../infra/README.md)。第三方来源快照保持原文，不因文档重构改写内容。

训练与评估的 rollout／指标选择见[OPD 逐篇调研](teacher-transfer/2026-10-31-OPD的Rollout来源与分布评估指标调研.md)：12 篇论文与 1 篇技术文章，区分固定前缀拟合、当前行为、教师覆盖及任务能力，并链接待评审协议。

同家族尺寸规律的单篇精读见[2609.32722 图解笔记](teacher-transfer/2026-10-31-同家族OPD缩放规律精读与实验启发.md)：七张论文原图、三个 RQ、策略位移与拟合残差的区别，以及对我们实验的启发。

Gold score 的术语和研究沿革见[gold/proxy 与奖励过优化](teacher-transfer/2026-10-31-Gold-score的来源与奖励过优化研究脉络.md)，相关平方根 KL 命题及反例见[数学推导](../theory/2026-10-31-正确率与序列及Token-KL的关系.md)。

理解 predicted、拟合验证与缩放规律的研究发展，见[缩放律拟合的历史、方法与蒸馏应用](teacher-transfer/2026-10-31-缩放律拟合的历史方法与蒸馏应用.md)：13 篇主线工作，以及如何区分全样本拟合、留尺寸验证和真正外推。

2609.32722 的公开曲线与 checkpoint 数据核查见[Vanilla OPD 后期拟合与任务成绩](teacher-transfer/2026-10-31-VanillaOPD后期拟合与任务成绩核查.md)：25 组峰值／末值、对 S0 的 KL、训练 proxy 与 teacher-validation KL 的可用性。
