# 理论定义、推导与验算

本目录维护数学对象及其适用条件，工程接口和实验状态分别由 infra 与研究正文维护。固定前缀理论把采样得到的状态及其权重视为常量；完整词表积分消除条件动作抽样误差，不自动给出完整序列分布的梯度。

| 阅读顺序与问题                                     | 正文                                                                                          |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| 教师、anchor、当前学生与行为策略                   | [统一符号](符号约定.md)                                                                       |
| Current/frozen × IS/NoIS究竟优化什么，怎样backward | [固定前缀目标与计算图](fixed-prefix.md)，包含标量目标、detach、支持集和有限缓存               |
| KL 正则、MD 与 proximal 的关系，κ/η怎样构造目标    | [镜像下降与 weighted teacher](mirror-descent/objectives.md)，保留普通对数与一般形状的完整推导 |
| Baseline、top-k 和头尾估计怎样改变梯度估计         | [估计器与控制变量](estimators.md)                                                             |
| Rollout 混合分布怎样定义及求解                     | [分布中心](rollout-policy/objectives.md)；[验算与来源](rollout-policy/README.md)              |
| 正确率、序列 KL 与 token KL 怎样连接               | [任务概率与 KL](accuracy-and-kl.md)，区分固定银行和真实访问分布                               |
| 中心化与教师可学习性的解释需要什么条件             | [可学习性](teacher-learnability.md)，数学身份、局部模型与待验证假说分别说明                   |
| 数值误差怎样影响 KL 观测                           | [误差模型](numerical-effects/error-model.md)，推导与原验算范围明确                            |

各正文保留所用论文版本、验算脚本及输入。CPU 小词表恒等式不是 GPU 或长程任务验收；当前代码映射从[架构](../infra/architecture/README.md)进入，研究结论从[五个 RQ](../experiments/README.md)进入。
