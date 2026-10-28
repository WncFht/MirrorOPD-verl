# Sum、minimax 与约束式 rollout 目标

> 历史讨论归档。当前建议与完整叙述见[统一主文](../../../theory/rollout-policy/2026-10-28-Rollout策略的目标设计与求解.md)。正文保留归档前内容，仅调整相对链接；其中运行命令以原研究目录为工作目录。

日期：2026-10-28，Asia/Shanghai。本文补充[上一份分布中心调研](2026-10-28-Rollout分布中心与Chernoff系数调研.md)，聚焦聚合目标，而非重复比较 divergence。结论的变化是：现有证据不足以把 minimax 默认排在等权 sum 前面。对当前没有额外任务效用信号的对称采样研究，建议等权 sum 作为默认参考，minimax 作为限制最差侧偏离的候选；是否采用它由实际轨迹差异和学习效果决定。

固定前缀，记 $a(Q)=D(Q,\mu)$、$b(Q)=D(Q,\pi)$。这里先使用同一种、同尺度的距离；若另设归一化或两侧权重，须明示新的偏好。

| 聚合目标 | 含义 | 适用条件 |
| --- | --- | --- |
| $S(Q)=(a+b)/2$ | 最小化双方平均偏离 | 两侧同等重要，允许一边近一些、另一边远一些 |
| $M(Q)=\max(a,b)$ | 最小化最差侧偏离 | 任何一侧明显脱离都不可接受 |
| $w a+(1-w)b$ | 已知两侧相对权重 | 权重来自明确任务偏好或独立数据 |
| $\min b(Q)$，s.t. $a(Q)\le\varepsilon$ | 在不超过 student 偏离预算下尽量接近 teacher | 明确以 student 为锚点，属于不对称目标 |
| 根据独立任务数据优化采样权重 | 直接优化任务效用或学习改善的代理 | 需要额外反馈，并控制选型过拟合 |

**minimax 的精确额外要求。** 恒等式

$$
\max(a,b)=\frac{a+b}{2}+\frac{|a-b|}{2}
$$

说明 minimax 在平均偏离之外加入了差距惩罚。它不会同时使所有距离都下降；通常要牺牲较近的一侧来改善较远的一侧。这个惩罚的合理性来自“最差侧是瓶颈”的偏好，不能只由“算法应对称”推出，因为等权 sum 也对称。

另一个等价解释是

$$
\max(a,b)=\max_{w\in[0,1]}[wa+(1-w)b].
$$

sum 固定权重，minimax 对权重作最坏情形选择。这与分布鲁棒优化中的不确定权重思想相关，但此处的不确定性集合只有两个分布损失，不等于文献中的一般经验分布鲁棒学习问题。[^dro2016]

**KL 下两者属于同一几何融合族。** 采用 $a=\mathrm{KL}(Q\Vert\mu),b=\mathrm{KL}(Q\Vert\pi)$，定义

$$
Q_w=\frac{\mu^w\pi^{1-w}}{Z(w)},\quad A(w)=\log Z(w).
$$

固定 $w$ 的加权和目标满足

$$
w\mathrm{KL}(Q\Vert\mu)+(1-w)\mathrm{KL}(Q\Vert\pi)
=\mathrm{KL}(Q\Vert Q_w)-A(w).
$$

因此等权 sum 给出 $Q_{1/2}$；minimax 则选择 Chernoff 权重 $w^*$，使两侧 KL 相等。后者的经典几何与数值算法见 Nielsen。[^chernoff2022]

**数值例子。** $\mu=(0.8,0.19,0.01),\pi=(0.1,0.2,0.7)$，所有值以 nats 计。

| 规则 | Student 权重 | Student KL | Teacher KL | 平均 KL | 最大 KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sum | 0.5000 | 0.3789 | 0.7756 | 0.5772 | 0.7756 |
| 中间鲁棒程度 $\rho=0.1$ | 0.4500 | 0.5110 | 0.6563 | 0.5837 | 0.6563 |
| Minimax | 0.4236 | 0.5929 | 0.5929 | 0.5929 | 0.5929 |

这里 minimax 把最差侧从 0.7756 降到 0.5929，但平均距离从 0.5772 增到 0.5929。不能把不同目标函数下的最优值当作无条件的算法优劣。

**无需训练即可计算的两项诊断。** 令 $Q_S=Q_{1/2}$，$Q_M=Q_{w^*}$，

$$
B=-\log\sum_v\sqrt{\mu_v\pi_v},\qquad
C=\max_w[-A(w)].
$$

$B$ 是最优平均 KL，$C$ 是最优最大 KL。由上面的分解，$S(Q)=\mathrm{KL}(Q\Vert Q_S)+B$。在 $Q_M$ 上两侧 KL 都等于 $C$，所以

$$
\boxed{\mathrm{KL}(Q_M\Vert Q_S)=C-B.}
$$

该式把“换成 minimax 之后分布改变多少”与两个最优值的差直接联系起来。若 $C-B$ 很小，二者单步分布在 KL 意义上很近；长序列仍可能累积差异。另令 $\Delta=|a(Q_S)-b(Q_S)|$，由于 $C\ge B$，

$$
0\le M(Q_S)-C\le\Delta/2.
$$

在等权中点上计算两侧 KL，即可得到 minimax 最多能改善多少的上界。一般非负距离还满足 $M(Q_S)\le2M(Q_M)$：$M(Q_S)\le2S(Q_S)\le2S(Q_M)\le2M(Q_M)$。这个 2-近似并不保证两者在下游训练中表现接近。

上述恒等式是本报告的直接推导。100 对 7 类随机分布的 CPU FP64 核验中，$\mathrm{KL}(Q_M\Vert Q_S)=C-B$ 的最大误差为约 $2.98\times10^{-16}$；未运行 LLM 训练。[脚本](../../../../research/rollout_policy/audit_aggregation.py)、[结果](../../../literature/rollout-policy/data/aggregation-audit.json)。

**两分布 Hellinger 下不需要选择 sum 或 max。** 采用 $H^2(p,q)=1-\sum_v\sqrt{p_vq_v}$，两者的共同最优解为

$$
Q_H(v)\propto(\sqrt{\mu(v)}+\sqrt{\pi(v)})^2.
$$

平方根空间中，等权平均中心到两端等距，所以也达到 minimax 下界。证明见上一份报告。此结论专指两个分布、同尺度等权目标；多分布或不等权目标不直接适用。平均 Hellinger 中心公式也出现在 behavior-policy 设计文献中。[^barycenter2025]

**局部与全轨迹的区别。** 设每步 $a_t=\mathrm{KL}(Q(\cdot\mid s_t)\Vert\mu(\cdot\mid s_t))$，$b_t$ 类似。链式法则给出

$$
\max\{\mathrm{KL}(P_Q\Vert P_\mu),\mathrm{KL}(P_Q\Vert P_\pi)\}
=\max\left\{\mathbb E_Q\sum_t a_t,\mathbb E_Q\sum_t b_t\right\}
\le\mathbb E_Q\sum_t\max\{a_t,b_t\}.
$$

逐 token minimax 在每一步都允许最差侧改变，比全轨迹选一个最差侧更保守。例如两步局部距离分别为 $(10,0)$ 和 $(0,10)$，逐步 max 的和为 20，两侧累计距离的 max 为 10。该数字例子只说明聚合次序；单步最优策略对未来状态的影响还需另行处理。Sum 的线性聚合可与求和、期望交换，但逐步贪心 sum 同样不自动成为全轨迹最优控制策略。

**中间聚合是可选研究方向，不建议首轮增加调参。** 定义

$$
F_\rho(Q)=\frac{a+b}{2}+\frac\rho2|a-b|,\qquad0\le\rho\le1.
$$

它等于 $\max_{w\in[(1-\rho)/2,(1+\rho)/2]}[wa+(1-w)b]$。KL 的固定权重最优值为 $-A(w)$，其最大值位于 Chernoff 权重。因此

$$
w_\rho=\operatorname{clip}\left(w^*,\frac{1-\rho}{2},\frac{1+\rho}{2}\right),\qquad Q_\rho=Q_{w_\rho}.
$$

$\rho=0$ 是 sum，$\rho=1$ 是 minimax。这只是有限权重不确定性集合的直接推导，并非声称新的基础算法。它用于说明“担心最差侧”可以连续表达；首轮先比较端点更容易解释。

不要把 $w$ 和 $Q$ 一起最小化固定加权和来自动学习权重：$\min_{Q,w}[w a+(1-w)b]$ 可以取 $w=1,Q=\mu$，目标为零，退化到单一来源。Minimax 中对权重是最大化；若按任务效用学习权重，需要独立定义效用与数据。

**文献证据与实际结果。**

| 工作 | 已核对的结果 | 对我们的含义 |
| --- | --- | --- |
| Nielsen 2022，Chernoff information | 给出 minimax 中心、等距条件及计算 | 证明几何目标最优，不证明 student 最终性能优于 sum。[^chernoff2022] |
| Akopyan et al. 2025，JS 与 minmax | 对 endpoint-first KL，$cR\le JS\le R$，$c=e\ln2/2\approx0.94208$ | 两个最优信息半径相近；不是所有 divergence 的结论，也不是平均中心最大 KL 与最优最大 KL 的 1.061 倍保证。[^js2025] |
| Weissmann et al. 2025，Barycenter Policy Design | §5 在 100 臂、1000 个 softmax policy 上比较平均 KL/χ² 中心、聚类和 MC；同总样本预算下约 10 个簇最好，图 2 平均 1000 次运行 | 主要收益来自结构化分组；该实验未比较 max 中心与 mean 中心，不支持“minimax 实证更好”。[^barycenter2025] |
| Son et al.，RMOD，2025 / v2 2026 | 对多个 reward 最差值进行鲁棒解码；HH 的 GPT-4o 最坏侧胜率为 59.1%，CD-Uniform 为 57.6%，同为 $B=16,K=16$ | 是最差目标与评测一致时的实证支持；两个师生 KL 不是已校准的 reward，不能直接转移效果。[^rmod] |
| Heskes 1997，Selecting Weighting Factors in Logarithmic Opinion Pools | 通过 KL error 的分解讨论 log-pool 权重选择与二次近似；正文主要为理论和任务形式示例 | 说明权重可以与预测误差挂钩，不只由源间距离决定；没有当前 rollout minimax 对照。[^heskes] |
| Yao et al.，Using stacking to average Bayesian predictive distributions，2017 / Bayesian Analysis 2018 | 用留一预测评分优化组合，在模拟和真实数据上比较 BMA、pseudo-BMA 等方法 | 如果具备外部效用反馈，按效用选择权重比纯几何准则更直接；这是预测组合证据，不是蒸馏采样保证。[^stacking] |
| TRB 2026 | student 偏离预算内最接近 teacher，退火回 student；已有两模型对实验 | 支持研究有锚点的约束式目标，但没有证明其胜过本报告的对称 sum/minimax。[^trb] |

关于 JS 结果的进一步区分：令 $m=(\mu+\pi)/2$，反向箭头的平均目标满足 $[\mathrm{KL}(\mu\Vert Q)+\mathrm{KL}(\pi\Vert Q)]/2=JS(\mu,\pi)+\mathrm{KL}(m\Vert Q)$。在该方向的 minimax 中心 $Q_F$，两侧 KL 相等为 $R$，故 $\mathrm{KL}(m\Vert Q_F)=R-JS$。可以用论文界控制这个中心距离，但不能把 $R/JS$ 直接解释为把 $m$ 放进 max 目标后的近似比。

**建议与优先级。** 对现阶段对称 full-vocab rollout，先采用无额外聚合参数的等权 sum 作为参考，比较同一个 divergence 下的 minimax；不要同时换 divergence 和聚合。若选 Hellinger，两分布 mean/max 相同，只需一组。若实际约束是 student 的状态偏离不能超过可解释预算，则约束式目标更直接，但须承认学生锚点的不对称性。最差侧偏离与失败率的关联、最终学生验证表现、token 成本和墙钟成本，才是支持增加 minimax 求解的证据。

几何中心只观察 $\mu,\pi$；“对学习最优”还涉及 teacher 是否正确、当前位置梯度、后续可恢复性和模型容量。现有核对文献没有给出当前 full-vocab LLM 设置中 minimax 普遍优于 sum 的结论。

**检索记录。** 三源新增检索得到 58 条去重记录，未与上一轮做跨报告去重。[完整原始报告](../../../literature/rollout-policy/data/2026-10-28-聚合目标补充检索原始报告.md)、[stderr](../../../literature/rollout-policy/data/aggregation-search.stderr.txt)。包括自动检索的全部无关命中，不把排名当成人工评价。来源诊断原文如下；目标论文已有作者全文，不影响核对。

```text
[open_alex] Invalid paper on query 'Tight Bounds Between the Jensen-Shannon Divergence and the Minmax Divergence': paper title must not be empty
```

### 参考文献

[^chernoff2022]: Frank Nielsen. Revisiting Chernoff Information with Likelihood Ratio Exponential Families. Entropy, 2022. [arXiv:2207.03745v4](https://arxiv.org/html/2207.03745v4)。
[^js2025]: Arseniy Akopyan, Herbert Edelsbrunner, Žiga Virk, Hubert Wagner. Tight Bounds Between the Jensen–Shannon Divergence and the Minmax Divergence. Entropy 27:854, 2025，Theorem 1。[作者 PDF](https://pub.ista.ac.at/~edels/Papers/2025-10-MinmaxDivergence.pdf)。
[^barycenter2025]: Simon Weissmann, Till Freihaut, Claire Vernade, Giorgia Ramponi, Leif Döring. Barycenter Policy Design for Multiple Policy Evaluation. 2025，作者公开稿，§5 与附录 B。[作者 PDF](https://tfreihaut.github.io/assets/publications/2025_is/is_paper.pdf)。
[^rmod]: Seongho Son et al. Robust Multi-Objective Controlled Decoding of Large Language Models. 2025，核对 v2（2026-02-16），§6.2、Table 1。[arXiv:2503.08796v2](https://arxiv.org/html/2503.08796v2)。
[^heskes]: Tom Heskes. Selecting Weighting Factors in Logarithmic Opinion Pools. NIPS 1997。[官方页面](https://papers.nips.cc/paper_files/paper/1997/hash/59f51fd6937412b7e56ded1ea2470c25-Abstract.html)、[全文](https://papers.nips.cc/paper/1413-selecting-weighting-factors-in-logarithmic-opinion-pools.pdf)。
[^stacking]: Yuling Yao, Aki Vehtari, Daniel Simpson, Andrew Gelman. Using stacking to average Bayesian predictive distributions. arXiv 2017，Bayesian Analysis 2018，DOI 10.1214/17-BA1091。[arXiv:1704.02030](https://arxiv.org/abs/1704.02030)。本文核对摘要与方法定位，未重跑其预测实验。
[^dro2016]: Hongseok Namkoong, John C. Duchi. Stochastic Gradient Methods for Distributionally Robust Optimization with f-divergences. NIPS 2016。[官方页面](https://papers.neurips.cc/paper_files/paper/2016/hash/4588e674d3f0faf985047d4c3f13ed0d-Abstract.html)。仅用于鲁棒权重的研究背景，不声称该文已研究本文两模型 rollout。
[^trb]: Daniil Plyusov et al. Trust-Region Behavior Blending for On-Policy Distillation. 2026。[arXiv:2605.31159v1](https://arxiv.org/html/2605.31159v1)。
