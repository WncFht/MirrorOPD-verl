# Prefix IS、状态分布与批内过拟合

完整词表 KL 能精确计算给定 prefix 上的条件目标；它不能保证更新后所有 prefix 的 KL 都下降。本文区分条件梯度、状态访问分布及 prefix importance sampling（IS），实验依据见[逐 inner bank 诊断](../experiments/reuse-budget/inner-bank/README.md)。

相关算法的研究脉络见[Prefix 与 sequence IS 文献调研](../literature/learning-signal/2026-11-16-prefix与sequence-IS调研.md)：token policy-gradient 的累计比率包含当前动作；full-vocab 条件 KL 的状态重加权只包含到达 prefix 的历史动作。

## 无偏的范围

令 $s=(x,y_{<t})$，学生为 $p_\theta$，教师为 $T$，固定状态的条件风险为

$$
f_\theta(s)=D_{\mathrm{KL}}(p_\theta(\cdot\mid s)\Vert T(\cdot\mid s)).
$$

当前诊断使用 full-vocab reverse KL，future mode 为 none。对一个固定状态积分整个词表，消除了当前动作采样的误差；在固定参数 $\theta$、新鲜状态 $s\sim d$ 下，样本梯度对 $\mathbb E_d\nabla f_\theta(s)$ 无偏。反复更新同一有限批后，$\theta$ 已依赖这批数据，不能继续把它当成与当前参数独立的新样本。参见[固定前缀与有限缓存](fixed-prefix.md)。

即使每一步都是精确的总体训练梯度，也不能保证另一个状态分布上的风险下降。令 $g_B$ 为该批梯度，$g_H$ 为 held-out 梯度，AdamW 在当前步的实际方向为 $v$，更新为 $\theta^+=\theta-\eta v$。一阶有

$$
\Delta K_H=-\eta\langle g_H,v\rangle+O(\eta^2\|v\|^2).
$$

训练损失下降只要求相应训练梯度与 $v$ 的内积为正；$g_H$ 与它的内积可能为负。共享参数、有限批的状态覆盖、Adam 动量及曲率均可能参与。这个公式解释了“条件信号正确”与“held-out 反弹”如何同时成立，不能单独识别哪一项是实验中的主因。

“prefix 问题”和“state 问题”在自回归模型中不是两个完全分离的对象：prefix 就是 state。需要进一步区分复用后的访问分布过时、有限状态覆盖不足，以及训练分布与固定 initial/teacher bank 的目标不同。

## 两种不能混用的 prefix IS

若这批轨迹由行为策略 $\beta$ 生成，要将旧策略的 prefix 分布校正到当前学生，用

$$
w_t=\frac{P_\theta(y_{<t}\mid x)}{P_\beta(y_{<t}\mid x)}
=\prod_{i<t}\frac{p_\theta(y_i\mid x,y_{<i})}{\beta(y_i\mid x,y_{<i})}.
$$

完整词表已对当前位置的动作积分，因此 prefix 比率不含 $y_t$。要求 prompt 分布相同、目标支持集包含于行为支持集，并使用实际 rollout 的 temperature、top-p/top-k、token 支持集和停止协议；不是任意模型裸 logits 的比率。随机长度时需明确存活状态及归一化；总 token 均值是期望累计风险除以期望 token 数，有限样本分子分母的比值通常不严格无偏。

若改用 $P_T/P_\beta$，得到的是教师访问分布上的训练信号，它改变了目标。teacher/student 不接近，也不等价于 current/behavior 不接近；后者在每轮新 rollout 时从同一权重开始，但有限精度及引擎协议仍需核查。

对固定目标参数、行为策略下独立采样的轨迹，$\mathbb E_\beta[w_t h(s_t)]=\mathbb E_{P_\theta}[h(s_t)]$。但用这批样本自适应拟合出来的参数再算权重，并不会自动恢复“新鲜独立样本”的无偏保证，也不会产生从未抽到的新状态。

此外，若真正的目标是 $J(\theta)=\mathbb E_{s\sim d_\theta}f_\theta(s)$，则

$$
\nabla J=\mathbb E_{d_\theta}\nabla f_\theta(s)
+\mathbb E_{d_\theta}[f_\theta(s)\nabla\log d_\theta(s)].
$$

把 IS 权重 detach 后，只校正第一项的访问分布，是 semi-gradient；它不是包含 prefix score/future-cost 项的完整序列目标梯度。对权重不 detach 则需要重新核对整套目标、归一化和梯度估计，不能视作无影响的工程开关。

## 方差与有效样本量的界

以下固定 prompt 和 prefix 长度 $h=t-1$，行为与目标的 prefix 分布分别为 $B_h,P_h$，$w=P_h/B_h$。在支持条件成立时，

$$
\mathbb E_Bw=1,\quad
\operatorname{Var}_B(w)=\chi^2(P_h\Vert B_h)
=\exp(D_2(P_h\Vert B_h))-1.
$$

对独立同分布轨迹，经验 $\mathrm{ESS}=(\sum_j w_j)^2/\sum_jw_j^2$ 的渐近比例为

$$
\lim_{N\to\infty}\mathrm{ESS}/N=\exp(-D_2(P_h\Vert B_h)).
$$

这是权重退化指标，不是所有梯度分量方差的完整表达；梯度幅度与权重的相关性也重要。不同长度和同一轨迹内的状态存在依赖，不能把全部 token 当成独立样本。经验 ESS 也可能漏掉未出现的极端尾部。

若在每个可访问历史上，条件 Rényi-2 divergence 一致满足

$$
D_2(p_\theta(\cdot\mid s)\Vert\beta(\cdot\mid s))\le\epsilon,
$$

写 $W_h=\prod_{i=1}^h r_i$，则

$$
\mathbb E_B[W_h^2\mid y_{<h}]
=W_{h-1}^2\sum_a\frac{p_\theta(a\mid s_h)^2}{\beta(a\mid s_h)}
\le W_{h-1}^2e^\epsilon.
$$

逐次取期望得到 $\mathbb E_BW_h^2\le e^{h\epsilon}$，因此 $\operatorname{Var}(W_h)\le e^{h\epsilon}-1$，渐近 ESS 比例至少为 $e^{-h\epsilon}$。这是需要逐状态一致上界的保证，不可用平均 token KL 直接替换 $\epsilon$。若所有 token 的 log-ratio 上界之和为 $C$，则另有 $W_h\le e^C$；同样需要真正的逐路径界。

以 $h=4096$ 为例，要由此保证渐近 ESS 比例至少 50%，必须有 $\epsilon\le\log2/4096\approx1.69\times10^{-4}$；保证 90% 需要 $\epsilon\le2.57\times10^{-5}$。若上界为 0.001，这一保证仅为 1.66%。**保证变弱不等于实测 ESS 就低**，具体仍取决于真实轨迹分布。

## 为什么平均 KL 小还不够

KL 没有给重要性权重二阶矩提供一般上界。取二点分布 $P=(1-a,a)$、$B=(1-b,b)$，$b=e^{-M}$、$a=\delta/M$。当 $M\to\infty$，$D_{\mathrm{KL}}(P\Vert B)\to\delta$，但

$$
\mathbb E_Bw^2=\frac{(1-a)^2}{1-b}+\frac{a^2}{b}\to\infty.
$$

$\delta$ 可以任意小；罕见事件的巨大比率仍让方差任意大。序列 KL 的 chain rule 是条件 KL 的期望和，不会把平均约束升级为上面所需的逐状态二阶矩约束。

对有界标量贡献 $|f|\le M_f$，将权重截断为 $\min(w,c)$ 的偏差满足

$$
|\mathbb E_B[(w-\min(w,c))f]|
\le M_f\mathbb E_B(w-c)_+
\le M_f\mathbb E_Bw^2/c.
$$

若有独立采样，截断贡献的样本均值方差还有粗界 $\operatorname{Var}(\bar Z)\le M_f^2c/N$，因为 $\min(w,c)^2\le cw$。这些界揭示方差与偏差的取舍；神经网络梯度不自动满足给定 $M_f$。Self-normalized IS 有有限样本偏差，短窗口乘积也通常不再是完整 prefix 分布的精确校正。

## 对当前问题的选择

先测试后期减少 reuse、及时生成新状态，并在相同训练 GPU 时间下评价；若希望更多独立信号，再比较增加不同 prompt 数的 rollout batch。增加同一问题回答数与增加不同问题数需要区分。前者增加条件轨迹多样性，后者同时扩大问题覆盖。

Prefix IS 更适合在已经证实 current/behavior 访问漂移明显、且权重尾部可控时作为修正。可先在诊断分支记录 inner1/8/16 的同轨迹行为与当前 sampled-token log-prob，按长度和位置报告 log-weight 分位数、最大权重、ESS、二阶矩及截断质量，并跨独立 outer 汇总。当前实验只保存 KL 汇总，无法实测这些量；不要把它们从 teacher KL 反推出来。即使 IS 权重接近 1，有限批拟合及跨分布梯度冲突仍可能存在。
