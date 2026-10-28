# Rollout 分布中心、divergence 选择与 Chernoff 系数实现

> 历史讨论归档。当前建议与完整叙述见[统一主文](../../../theory/rollout-policy/2026-10-28-Rollout策略的目标设计与求解.md)。正文保留归档前内容，仅调整相对链接；其中运行命令以原研究目录为工作目录。

核对日期：2026-10-28，Asia/Shanghai。范围：有限共同动作空间上的两分布中心、相关文献、公开源码及 CPU 数值验证。没有修改 MirrorOPD-verl，没有运行 GPU 解码或模型训练。

**结论。** KL 是有明确轨迹解释的一种选择，不是 full-vocab 或师生对称性唯一推出的选择。建议保留最小化最大 $\mathrm{KL}(Q\Vert\mu),\mathrm{KL}(Q\Vert\pi)$ 的 Chernoff 中心作为理论主线，并把 Hellinger minimax 中心列为首个必要对照。后者有闭式解，能够保留双方支持，不需要求融合系数。暂不增加重要性权重二阶矩约束；若未来目标包含 IS 估计方差，再按实际采样和目标分布选择相应方向的 $\chi^2$。

**已有工作。** 分布的 minimax 中心、不同 divergence 的中心、Chernoff 最优参数和二分算法都已有研究与代码。尚未在此次检索中核实到把精确两模型 Chernoff 中心直接用作 LLM 蒸馏逐 token rollout 的公开工作；这是一项有范围的检索结果，不能作为首创性证明。

**符号与研究问题。** 固定前缀 $s$，$\mu$ 是当前 student，$\pi$ 是冻结 teacher，$Q$ 是实际生成分布。以下所有中心首先是单个前缀上的条件分布中心。局部中心不自动构成整个自回归序列的全局 minimax 中心。内层学习 loss 与生成分布的距离是两个独立选择。

| 问题 | 中心形式 | 解释 |
| --- | --- | --- |
| $\min_Q\max\{\mathrm{KL}(Q\Vert\mu),\mathrm{KL}(Q\Vert\pi)\}$ | $Q\propto\mu^{\alpha^*}\pi^{1-\alpha^*}$ | 用 $Q$ 访问的事件来衡量偏离；偏好共同支持 |
| $\min_Q\max\{\mathrm{KL}(\mu\Vert Q),\mathrm{KL}(\pi\Vert Q)\}$ | $Q=\lambda^*\mu+(1-\lambda^*)\pi$ | 用两端各自的事件来衡量覆盖；$\lambda^*$ 一般不等于 $1/2$ |
| $\min_Q\max\{H^2(Q,\mu),H^2(Q,\pi)\}$ | $Q\propto(\sqrt\mu+\sqrt\pi)^2$ | 对称且有界；两端平方根球面中点 |
| $\min_Q\max\{\mathrm{TV}(Q,\mu),\mathrm{TV}(Q,\pi)\}$ | $(\mu+\pi)/2$ 是最优解之一 | 直接控制事件概率差，但最优中心可能不唯一 |
| $\min_Q\max\{\chi^2(\mu\Vert Q),\chi^2(\pi\Vert Q)\}$ | $Q\propto\sqrt{\lambda^*\mu^2+(1-\lambda^*)\pi^2}$ | 面向从 $Q$ 样本估计两端期望时的权重二阶矩 |

表中不同距离的数值半径不能直接横向比较。KL 两种方向的中心分别来自指数族和混合族；这些区别在 Chernoff 几何与信息半径文献中已有讨论。[^nielsen2022][^minmax2025] $\chi^2$ 的方差用途与多策略采样设计直接相关。[^barycenter2025]

**KL 的适用理由与局限。** 对同一 prompt 分布、同一环境转移、有限长度且 EOS 后吸收的自回归策略，链式法则给出

$$
\mathrm{KL}(P_Q\Vert P_\mu)
=\mathbb E_{\tau\sim Q}\sum_t\mathrm{KL}\bigl(Q(\cdot\mid s_t)\Vert\mu(\cdot\mid s_t)\bigr).
$$

因此在 $Q$ 访问的每个前缀上控制两个 $Q$-first KL，能控制两端对应的轨迹 KL。局部上界 $\varepsilon$ 只给出长度 $T$ 时的 $T\varepsilon$ 上界；逐前缀最小化并不考虑动作对未来状态的影响，所以不能据此声称全轨迹最优。其他距离也能给出轨迹比较，但通常没有这一相同形式的可加链式分解。

full-vocab 消除的是固定前缀上词表求和的 Monte Carlo 近似，不会让不同 divergence 具有相同中心，也不会消除轨迹采样随机性。师生交换对称性可以由双边 max 保证，即使单个距离是有方向的 KL；它本身不能裁决 KL 的方向。

**共同低概率事件的反例。** 下列是精确构造的三事件分布，不是模型实验：

$$
\mu=(0.999,10^{-12},0.001-10^{-12}),\qquad
\pi=(10^{-12},0.999,0.001-10^{-12}).
$$

交换前两个事件等价于交换师生，因此 Chernoff 最优系数是 $1/2$。实际结果如下。

| 分布 | 事件 1 | 事件 2 | 事件 3 |
| --- | ---: | ---: | ---: |
| Student | 0.999 | $10^{-12}$ | 约 0.001 |
| Teacher | $10^{-12}$ | 0.999 | 约 0.001 |
| KL minimax | 0.0009975 | 0.0009975 | 0.9980050 |
| Hellinger minimax | 0.4990010 | 0.4990010 | 0.0019980 |
| 算术中点 | 0.4995 | 0.4995 | 约 0.001 |

KL 中心把双方仅给约 0.1% 概率的共同事件提升到约 99.8%。它仍严格最优，但其共同 KL 半径约为 6.906 nats，并不能称为两边都很 on-policy。这个反例表明：共同支持偏好有可能选择双方都不常走的路径；无法只凭最优性证明解题质量。Hellinger 保留两端覆盖，也因此会访问某一端不支持的事件；它没有同时消除所有失配。

**Hellinger 两分布 minimax 的完整推导。** 采用

$$
H^2(p,q)=\frac12\sum_v(\sqrt{p_v}-\sqrt{q_v})^2
=1-\sum_v\sqrt{p_vq_v}.
$$

设 $a=\sqrt\mu,b=\sqrt\pi,u=\sqrt Q$，则三者都是非负单位向量。最小化最大 Hellinger 平方距离等价于最大化 $\min\{u^Ta,u^Tb\}$。对任意可行 $u$，

$$
\min\{u^Ta,u^Tb\}\le\frac{u^T(a+b)}2\le\frac{\|a+b\|_2}2.
$$

取 $u^*=(a+b)/\|a+b\|_2$ 时，因 $\|a\|=\|b\|=1$，两内积相等，两个不等式同时取等。故全局最优解是

$$
Q_H(v)=\frac{(\sqrt{\mu(v)}+\sqrt{\pi(v)})^2}{2+2B},
\qquad B=\sum_v\sqrt{\mu(v)\pi(v)}.
$$

共同半径为 $1-\sqrt{(1+B)/2}$。这也与概率单纯形上的 Fisher–Rao 测地线中点对应。对两分布，此处平均 Hellinger 平方距离中心也达到 minimax；多于两个分布时不能直接推广为相同结论。Hellinger 平均中心公式已有概率融合与 behavior-policy 文献。[^amari2007][^barycenter2025] 上述两分布 minimax 证明是本报告直接展开的推导。

进一步令 $Q_A=(\mu+\pi)/2$、$Q_G=\sqrt{\mu\pi}/B$，当 $B>0$ 时

$$
Q_H=\frac{Q_A+BQ_G}{1+B}.
$$

重叠很小时 $B\to0$，Hellinger 中心趋于算术中点。它还满足 $Q_H\ge(\mu+\pi)/4$，所以 $\mu/Q_H\le4,\pi/Q_H\le4$。这些是逐 token 的覆盖界；整条轨迹上的比值仍可能随长度相乘，不能把常数 4 直接用于整个序列。

**不同 divergence 的统一表达。** 为避免混淆，将 divergence 阶数写为 $\beta$，融合系数写为 $\lambda$。采用 Candan 的约定：[^candan2020]

$$
D_\beta(p\Vert q)=\frac{1-\sum_v p(v)^\beta q(v)^{1-\beta}}{\beta(1-\beta)},\quad0<\beta<1.
$$

固定拉格朗日权重 $\lambda$，最小化 $\lambda D_\beta(\mu\Vert Q)+(1-\lambda)D_\beta(\pi\Vert Q)$。令 $S_v=\lambda\mu_v^\beta+(1-\lambda)\pi_v^\beta$，对 $Q_v$ 求导并加归一化乘子 $\nu$：

$$
-\frac1\beta S_vQ_v^{-\beta}+\nu=0.
$$

所以

$$
Q_{\beta,\lambda}(v)\propto
\left[\lambda\mu(v)^\beta+(1-\lambda)\pi(v)^\beta\right]^{1/\beta}.
$$

$\beta\to0$ 对应 $\mathrm{KL}(Q\Vert p)$ 和几何融合；$\beta\to1$ 对应 $\mathrm{KL}(p\Vert Q)$ 和算术融合；$\beta=1/2$ 是 Hellinger 的常数倍。对 minimax，还需优化对偶权重 $\lambda$；两端非退化情况下通过相应两侧距离相等确定。Hellinger 两分布情形的最优权重为 $1/2$。文献中 $\alpha$-divergence 的参数约定不统一；它与 Chernoff 的融合参数 $\alpha^*$ 不是同一个量。

**直接相关文献。**

| 工作 | 已核对内容 | 与本问题的关系和限制 |
| --- | --- | --- |
| Amari, Integration of Stochastic Models by Minimizing α-Divergence, Neural Computation 2007 | 用不同 α-divergence 导出不同概率融合均值 | 是平均 divergence 中心；不能直接把所有均值当作 minimax 解。[^amari2007] |
| Nielsen, An Information-Geometric Characterization of Chernoff Information, IEEE SPL 2013 | 指数测地线、KL 等距分割面和 Chernoff 点 | 几何融合及最优参数已有经典解释。[^nielsen2013] |
| Nielsen, Revisiting Chernoff Information with Likelihood Ratio Exponential Families, Entropy 2022 | 似然比指数族、最优参数、等 KL 条件、数值计算 | 与我们两端 $Q$-first KL minimax 直接对应。[^nielsen2022] |
| Candan, Chebyshev Center Computation on Probability Simplex With α-Divergence Measure, IEEE SPL 2020 | 多分布 minimax 中心、α-divergence、广义 Arimoto–Blahut 求解 | 直接回答“换 div 是否已有研究”；论文阶数与融合权重须分开。[^candan2020] |
| Koliander et al., Fusion of Probability Density Functions, 2022 | 概率密度融合的多种构造与解释 | 用于整理 arithmetic/geometric pooling，不能当作 LLM 效果证据。[^fusion2022] |
| Akopyan et al., Tight Bounds Between the Jensen–Shannon Divergence and the Minmax Divergence, Entropy 2025 | endpoint-first KL minimax 与加权 JS 的关系 | 是反向箭头的中心，得到算术族；论文 JS 数值近似结论不能照搬到 Chernoff 半径。[^minmax2025] |
| Weissmann et al., Barycenter Policy Design for Multiple Policy Evaluation, EWRL 2025 / 作者公开稿 | KL、Hellinger、χ² 平均中心作为 behavior policy，附录 B.4/B.5 讨论最大 KL / 最大权重方差 | 与采样设计最接近；理论在随机 bandit 与 IS policy evaluation，不是 full-vocab LLM 蒸馏。早期 arXiv:2503.02735 标题为 Clustered KL-barycenter design for policy evaluation，只聚焦 KL；不能混引版本。[^barycenter2025][^clustered2025] |
| Nielsen & Soen, pyBregMan, arXiv:2408.04175, 2024 | 信息几何库、Chernoff point、categorical/multinomial 支持 | 提供实际 α* 数值实现，见下文源码审计。[^pybregman2024] |

**LLM 附近工作。** TRB 在 student-centered KL 预算内选最接近 teacher 的生成分布，约束是单边的，不能与对称 Chernoff 中心混称。VESPO 的 $Q$ 是旧 behavior 样本重加权的隐式分布；不是直接运行这个 $Q$ 解码。RMOD 在解码时优化最差 reward 组合，而不是师生分布中心。Divergence Decoding 用 JS 阈值选择 specialist 或 generalist，属于不对称路由，没有求 Chernoff 系数。[^trb2026][^vespo2026][^rmod2025][^divdec2026]

新近 Reverse-KL Policy Distillation: Statistical Guarantees and Limits 区分 teacher full-distribution feedback 与 sampled-action feedback，说明 full feedback、状态来源和优化目标是不同层次的设计；该文没有证明当前对称融合策略最优。[^rkl2026]

**公开实现与实测。**

1. pyBregMan 主分支核对时为 `6f375e4d38e8ff798801bf5d34572ed52b1fadbb`。`bregman/dissimilarity/bregman.py:270` 的 `ChernoffInformation.chernoff_point` 通过 Bregman 测地线及两侧 divergence 做二分。实际停止条件是系数区间宽度，不是 docstring 所说的 divergence 差。默认 THETA 坐标在 categorical 指数族中对应几何融合；返回值是 point_1 的指数权重。[^py-code]
2. `MultinomialManifold(k=V,n=1)` 可以表示 categorical 分布。已直接运行原始源码，在 40 对 5 类随机分布上与独立 log-space 求根比较，最大系数差约 $3.16\times10^{-12}$。示例 $\mu=(0.8,0.19,0.01),\pi=(0.1,0.2,0.7)$ 返回 `0.423611045229336`；独立参考为 `0.423611045228528`。
3. 同一 commit 的 `ChernoffInformation.dissimilarity` 不等于我们定义的 Chernoff 半径。它把 `1-alpha_star` 传给 `SkewBurbeaRaoDivergence`，而后者还带 $1/[\alpha(1-\alpha)]$ 因子。示例返回 `2.187512456903406`，标准半径是 `0.592858680184349`。这同时涉及指数方向与缩放；不要把这个标量接口当作标准半径直接使用。系数 helper 在上述范围内通过验证。[^py-code]
4. 作者 Frank Nielsen 的 `P5/CategoricalChernoffInformationPDE/BernoulliExpFam.java` 明确给出 `ChernoffExponent`、`ChernoffExponentCF` 等闭式实现。已编译原始 Java 并运行作者示例 $p=0.1,q=0.7$，得到 $\alpha^*=0.466076329662343$、中心概率 $0.360848806714530$、两侧 KL 都约 $0.244321377719849$。这覆盖 Bernoulli，不是一般大词表向量化求解器。[^java-code]
5. 作者还公开了多元 Gaussian 的数值二分代码。它证明 Chernoff 系数计算早有软件，但 Gaussian 专用实现不能直接作为语言模型 sampler。[^java-gaussian]

上述代码是研究软件。pyBregMan 采用 NumPy/Autograd 和单个点对接口，不是 GPU 批量 full-vocab decoder；源码 categorical normalizer 直接计算 exp，还需要针对大 logits 的 log-sum-exp 稳定化。我们的推理系统仍需自行处理批处理、共同事件空间、EOS、温度、分片归约和双 KV。

**CPU 审计产物。** [独立检查脚本](../../../../research/rollout_policy/audit_centers.py)、[数值结果](../../../literature/rollout-policy/data/numerical-audit.json)、[作者 Java 输出](../../../literature/rollout-policy/data/java-audit.txt)。Hellinger 闭式 minimax 与独立 SLSQP 约束优化在 40 对随机分布上的最大半径差约 $4.39\times10^{-13}$。这只核验数学和参考代码，不提供任务性能判断。上游代码副本保存在 `code/`，保留原内容和许可证。

运行方式：

```bash
uv run --with 'numpy<2' --with scipy --with autograd code/audit_centers.py
```

**实验取舍。** 第一轮固定 learner loss、词表事件空间、prompt 顺序、模型初始化、temperature 和 token 预算，只比较：纯 student、纯 teacher、固定算术/几何中点、KL 两种方向的 minimax、Hellinger minimax。不同 loss 的对照放在下一轮，避免同时改变访问状态和优化方向。记录双方实际距离、共同事件覆盖、停止概率、序列长度、最终学生准确率和总时间。KL minimax 若只在大分歧位置明显改动分布，应分析这些位置是否集中于 EOS 或模板差异。

**可主张的贡献范围。** Chernoff 公式、等距条件、二分法和 divergence-dependent centers 已有 prior art。若继续研究，贡献应落在状态采集策略对训练的因果作用、与 full-vocab divergence 的交互、局部与序列层面的差距、或高效解码实现，而不是重新命名已有 α* 求解。

**检索覆盖与原始记录。** 跨 arXiv、OpenAlex、Crossref 两轮检索，第一轮 76 条去重记录，第二轮 59 条去重记录；两轮之间没有再次合并去重，不能将两数相加当作独立论文数。原始结果包含无关命中，完整保留，不把自动排序当作人工相关性结论。

- [第一轮完整原始报告](../../../literature/rollout-policy/data/2026-10-28-论文检索原始报告.md)
- [第二轮完整原始报告](../../../literature/rollout-policy/data/2026-10-28-定向补充检索原始报告.md)
- [第一轮 stderr](../../../literature/rollout-policy/data/search.stderr.txt)、[第二轮 stderr](../../../literature/rollout-policy/data/targeted-search.stderr.txt)

第一轮来源诊断原文：

```text
[open_alex] Invalid paper on query 'Tight Bounds Between the Jensen-Shannon Divergence and the Minmax Divergence': paper title must not be empty
```

该记录缺标题被跳过，同名目标论文已通过原作者 PDF 核实。第二轮来源无 diagnostic；两个精确 arXiv ID 均成功解析。

### 参考文献

[^amari2007]: Shun-ichi Amari. Integration of Stochastic Models by Minimizing α-Divergence. Neural Computation 19(10):2780–2796, 2007. DOI 10.1162/neco.2007.19.10.2780. [作者机构 PDF](https://bsi-ni.brain.riken.jp/database/file/284/290.pdf)。
[^nielsen2013]: Frank Nielsen. An Information-Geometric Characterization of Chernoff Information. IEEE Signal Processing Letters 20(3):269–272, 2013. DOI 10.1109/LSP.2013.2243726. [作者 PDF](https://www.lix.polytechnique.fr/Labo/Frank.Nielsen/J2013-SPL-ChernoffInformationIG.pdf)。
[^nielsen2022]: Frank Nielsen. Revisiting Chernoff Information with Likelihood Ratio Exponential Families. Entropy 24(10):1400, 2022. [arXiv:2207.03745v4](https://arxiv.org/html/2207.03745v4)。
[^candan2020]: Çağatay Candan. Chebyshev Center Computation on Probability Simplex With α-Divergence Measure. IEEE Signal Processing Letters 27:1515–1519, 2020. DOI 10.1109/LSP.2020.3018661. [作者 PDF](https://users.metu.edu.tr/home103/ccandan/wwwhome/pub_dir/Cheb_Center_IEEE_SPL_2021.pdf)。
[^fusion2022]: Günther Koliander, Yousef El-Laham, Petar M. Djurić, Franz Hlawatsch. Fusion of Probability Density Functions. 2022. [arXiv:2202.11633](https://arxiv.org/abs/2202.11633)。本报告仅用其主题与融合综述定位，不将摘要当作具体定理证明。
[^minmax2025]: Arseniy Akopyan, Herbert Edelsbrunner, Žiga Virk, Hubert Wagner. Tight Bounds Between the Jensen–Shannon Divergence and the Minmax Divergence. Entropy 27:854, 2025. [作者 PDF](https://pub.ista.ac.at/~edels/Papers/2025-10-MinmaxDivergence.pdf)。
[^barycenter2025]: Simon Weissmann, Till Freihaut, Claire Vernade, Giorgia Ramponi, Leif Döring. Barycenter Policy Design for Multiple Policy Evaluation. EWRL 2025 公开稿及作者公开稿。公式 (3)、表 1、附录 B.4/B.5。本文详细分析使用 34 页作者稿。 [作者稿](https://tfreihaut.github.io/assets/publications/2025_is/is_paper.pdf)、[EWRL 稿](https://openreview.net/pdf/a041a53d80087e57b06466c0cd09fa5a4f20c5e4.pdf)。
[^clustered2025]: 同一作者组. Clustered KL-barycenter design for policy evaluation. 2025. [arXiv:2503.02735v1](https://arxiv.org/html/2503.02735v1)。
[^pybregman2024]: Frank Nielsen, Alexander Soen. pyBregMan: A Python library for Bregman Manifolds. 2024. [arXiv:2408.04175](https://arxiv.org/html/2408.04175)。
[^py-code]: Alexander Soen, Frank Nielsen. pyBregMan 源码，commit `6f375e4d38e8ff798801bf5d34572ed52b1fadbb`，核对时为 main HEAD。[ChernoffInformation](https://github.com/alexandersoen/pyBregMan/blob/6f375e4d38e8ff798801bf5d34572ed52b1fadbb/bregman/dissimilarity/bregman.py#L249)、[categorical 支持](https://github.com/alexandersoen/pyBregMan/blob/6f375e4d38e8ff798801bf5d34572ed52b1fadbb/bregman/application/distribution/exponential_family/multinomial/multinomial.py)。
[^java-code]: Frank Nielsen. BernoulliExpFam.java，源码注明 November 2024，commit `e4c782ff4fb52e0c2b9d7285f47e41a1324ef900`。[原始源码](https://github.com/FrankNielsen/FrankNielsen.github.io/blob/e4c782ff4fb52e0c2b9d7285f47e41a1324ef900/P5/CategoricalChernoffInformationPDE/BernoulliExpFam.java)。
[^java-gaussian]: Frank Nielsen. DemoChernoffInformatioMultivariateGaussian.java，同上 commit。[原始源码](https://github.com/FrankNielsen/FrankNielsen.github.io/blob/e4c782ff4fb52e0c2b9d7285f47e41a1324ef900/ChernoffInformation/DemoChernoffInformatioMultivariateGaussian.java)。仅作源码核对，未执行这个 Gaussian demo。
[^trb2026]: Daniil Plyusov et al. Trust-Region Behavior Blending for On-Policy Distillation. 2026. [arXiv:2605.31159v1](https://arxiv.org/html/2605.31159v1)。
[^vespo2026]: Guobin Shen et al. VESPO: Variational Sequence-level Soft Policy Optimization for Off-Policy LLM Training. 2026. [arXiv:2602.10693v3](https://arxiv.org/html/2602.10693v3)。
[^rmod2025]: Seongho Son et al. Robust Multi-Objective Controlled Decoding of Large Language Models. 2025，核对 v2（2026-02-16）。[arXiv:2503.08796v2](https://arxiv.org/html/2503.08796v2)。
[^divdec2026]: Divergence Decoding: Training-Free Capability Fusion. 2026. [arXiv:2607.27248](https://arxiv.org/html/2607.27248)。
[^rkl2026]: Haolin Liu, Chen-Yu Wei, Julian Zimmert. Reverse-KL Policy Distillation: Statistical Guarantees and Limits. 2026，v2（2026-10-25）。[arXiv:2610.30844v2](https://arxiv.org/abs/2610.30844v2)。本报告仅核对摘要中的 feedback 区分，未审计其完整证明。
