# Prefix IS 与 Sequence IS：从经典 RL 到 LLM RL

本文按校正对象区分 prefix、per-decision 与 sequence importance sampling（IS），重点比较长时域 RL 如何在精确性和权重方差之间取舍。对 LLM token policy-gradient 来说，最直接的经典连接是 per-decision IS、Retrace 和 V-trace；近期 LLM 论文进一步提出 GSPO、VESPO、CTPO 与分段 prefix 校正。本文的结论会单独映射到 MirrorOPD 的 full-vocab 条件 KL，不把 sampled-token RL 的公式直接套到蒸馏目标上。

## 先看结论

| 想修正什么                             | 常见权重                                         | 主要代价                               | 代表方法                       |
| -------------------------------------- | ------------------------------------------------ | -------------------------------------- | ------------------------------ |
| 固定状态上的动作期望                   | 当前动作比率 $\pi(a_t\mid s_t)/\mu(a_t\mid s_t)$ | 若状态来自旧策略，仍有状态分布偏差     | 普通 token PPO/GRPO 的局部比率 |
| token policy-gradient 的状态和动作访问 | 累计到当前位置的乘积 $\prod_{j\le t}\rho_j$      | 前缀越长，权重方差越高                 | Per-decision IS；LLM 中的 CTPO |
| 当前策略下的完整序列期望               | 完整轨迹比 $W=\prod_{j=1}^{T}\rho_j$             | 长度方向的乘积方差通常最严重           | 轨迹 IS；VESPO 对 $W$ 做软重塑 |
| 使用多个版本生成的分段轨迹             | 每个行为策略片段各自的 prefix ratio              | 需要记录分段边界、行为版本和片段支持   | SPIRE                          |
| 长期/稳态状态访问分布                  | 状态占用密度比 $d^\pi(s)/d^\mu(s)$               | 需要额外学习密度比或值函数，并要求覆盖 | DICE 系列                      |

在 MirrorOPD 中，full-vocab KL 在给定 prefix 上已经精确积分了当前位置的动作，不需要为该动作再乘 action IS。若想把旧 rollout 前缀上的条件梯度改成当前学生访问分布下的期望，才需要 prefix state ratio；这会重新引入长前缀权重退化。当前更稳妥的主要手段是刷新 rollout 和扩大独立 prompt/state 覆盖，IS 先作为诊断或受控估计器研究。

## 重要文献与阅读顺序

### LLM RL：从 sequence IS 到 prefix IS

| 优先级 | 工作                     | 它校正的对象                                   | 阅读时抓住的结论                                                                                                                  |
| ------ | ------------------------ | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 1      | VESPO (2026)             | 完整回答的标量回报 policy gradient             | 先将序列权重重塑解释成改变隐式提议分布，再用 KL 接近性和二阶矩约束设计平滑权重。适合研究大 policy lag；不保持原始 IS 无偏性。     |
| 2      | CTPO (2026)              | token policy-gradient 的前缀状态与当前位置动作 | 累计到当前 token 的比率与 token 项配套，避免整条 response 比率把无关后缀噪声带到较早 token。实际 clip 后已改变估计器。            |
| 3      | SPIRE (2026)             | 由不同版本策略生成的多个轨迹片段               | 按行为策略切段、分段权重与预算调度有启发；正文 Eq. (2) 与 Appendix A 的无偏证明使用的权重不同，本文给出反例，暂不采纳其无偏保证。 |
| 4      | GSPO (2025)              | 组内 response 的相对更新与 clip                | 用序列几何平均 ratio 作序列级优化信号，提升长序列和 MoE 训练稳定性；$W^{1/T}$ 是长度归一化代理量，不是 sequence IS 密度比。       |
| 5      | Prefix-ratio 系列 (2026) | 按当前位置之前的 token ratio 修正 token 目标   | 直接 prefix weighting 在极长前缀上也会有尾部问题；论文的过滤/裁剪策略与位置分布需要一起读，不能只引用“prefix correction 无偏”。   |

**VESPO 的推导与实用算法要分开看。**[^vespo] 设行为分布为 $\mu$、目标策略为 $\pi$，完整序列比率 $W(\tau)=P_\pi(\tau)/P_\mu(\tau)$。普通轨迹 IS 用 $W$，保留目标测度下的期望，但 $W$ 的二阶矩会随序列漂移增大。VESPO 将 $\phi(W)$ 作用在 sequence-level gradient 上，并令 $Q(\tau)\propto\mu(\tau)\phi(W(\tau))$。其变分推导给出 $\phi(W)=W^\alpha e^{-\lambda W}$ 一类解；实际使用平移形式 $W^{c_1}e^{c_2(1-W)}$，并可对正负优势取不同参数。论文将权重 detach 后作为 REINFORCE 梯度缩放。只要 $\phi(W)\ne W$，期望对应的测度就不再是原始 $P_\pi$；平滑和有界性是以目标偏移换较稳方差。结果支持其在作者设置的高 stale ratio 上更稳定，不能单凭论文证明它适用于 state-conditioned KL 拟合。

VESPO 的提议分布优化具体为

$$
\min_Q\left\{D_{\mathrm{KL}}(Q\Vert\mu)-\alpha\mathbb E_Q\log W\right\},
\qquad\mathbb E_QW\le C,\quad\int Q=1.
$$

对 $\mathcal L=D_{\mathrm{KL}}(Q\Vert\mu)-\alpha E_Q\log W+\lambda(E_QW-C)+\nu(\int Q-1)$ 求变分导数，得到 $\log(Q/\mu)+1-\alpha\log W+\lambda W+\nu=0$，因此 $Q^*\propto\mu W^\alpha e^{-\lambda W}$。实际核 $\phi(W)=W^{c_1}e^{c_2(1-W)}$ 满足 $\phi(1)=1$；对 $c_1,c_2>0$，最大值在 $W=c_1/c_2$，为 $(c_1/c_2)^{c_1}e^{c_2-c_1}$。这是权重有界，不自动意味着网络 score、优势或整条梯度有界。若优势冻结，其更新还可解释为 $E_\mu[F(W)A]$ 的梯度，其中 $F'(W)=\phi(W)/W$；并非 $E_\mu[\phi(W)A]$ 的直接梯度。

**CTPO 与 per-decision IS 的对应关系依赖优势项。**[^ctpo][^precup] 对可按时间分解的奖励 $r_t$，目标策略回报可写成逐奖励项；第 $t$ 项只需累计到产生该奖励所需的 prefix ratio，之后的 suffix ratios 条件期望为 1。token policy-gradient 则在第 $t$ 个 score 前累计到当前动作。若目标是终局 reward，旧策略下的终局 reward 同时依赖旧策略生成的 suffix；仅乘当前位置 prefix ratio 并不能自动把该 suffix 换成目标策略 suffix。严格构造需目标策略的 $Q^\pi(s_t,a_t)$、完整轨迹比率，或有相应边界假设的控制变量/递归估计。CTPO 的公式应与其 token advantage 定义、group-relative baseline、clipping 和采样协议一起核对；“prefix product 比 sequence product 方差小”本身不是无偏证明。

**SPIRE 应作为待修正的分段思路阅读。**[^spire] 该论文以 segment 行为策略记录 partial rollout、replay 和 policy lag，并声称在 prefix-measurable 边界下无偏。正文 Eq. (2) 用各段独立权重乘各段 score，Appendix A 却只证明完整乘积权重的换测度恒等式；两个估计器一般不相等。下文给出反例。分段记录行为策略本身有用，但目前不能借用该文的无偏命题作为算法保证。

### 与 rollout/update 预算最相关：POIS (2018)

_Policy Optimization via Importance Sampling_（Metelli、Papini、Faccio、Restelli，NeurIPS 2018）直接把“何时停止在当前批上优化、何时采新轨迹”作为问题：使用 IS 复用轨迹，以高置信回报下界构造离线 surrogate；策略偏离行为策略后，IS 不确定性惩罚增大，限制继续在旧批上优化。[^pois]

这比只 clip 权重更贴近我们的预算问题：优化收益必须覆盖分布漂移带来的估计不确定性。POIS 的理论基于其具体回报界、轨迹/参数策略与采样条件，不能仅凭训练 rebound 或 batch ESS 就宣称获得同样的泛化保证。对当前 MirrorOPD，可借鉴“复用收益与统计不确定性一起决定 refresh”的设计原则，再单独证明条件 KL 目标和采样依赖下的版本；这比直接移植其 policy-gradient loss 更合理。

理解 POIS 所依据的不确定性原则，可以先看一个固定目标策略下的通用 IS 界（以下为标准 Cantelli 界的说明式，不冒充论文全部算法）。若 $|R|\le R_{\max}$、$N$ 条轨迹独立、$W=P_\pi/P_\mu$ 且二阶矩有限，$\hat J=N^{-1}\sum_iW_iR_i$ 满足

$$
\operatorname{Var}(\hat J)\le\frac{R_{\max}^2}{N}\mathbb E_\mu W^2
=\frac{R_{\max}^2}{N}e^{D_2(P_\pi\Vert P_\mu)}.
$$

Cantelli 不等式给出以至少 $1-\delta$ 的概率成立的下界

$$
J(\pi)\ge\hat J-R_{\max}
\sqrt{\frac{1-\delta}{\delta N}e^{D_2(P_\pi\Vert P_\mu)}}.
$$

同样的旧样本上继续更新，估计回报可能增加，但策略漂移使二阶矩惩罚也增大；补新样本则能降低统计惩罚。这正是比单看训练 loss 更合理的 reuse/fresh 权衡。该式只对固定 $\pi$ 和已知真实二阶矩直接成立：把总体 $D_2$ 换成 64 条轨迹的经验 ESS，或在同批数据上反复择优优化 $\pi$，需要额外的估计/一致界与数据依赖分析。RKL 无普遍有限上界，不能直接把 $R_{\max}$ 换成一个未验证常数。[^pois]

### LLM 之前：长时域 RL 怎样避免整条轨迹连乘

早期强化学习并没有一个普遍有效的“更聪明的完整 sequence ratio”。文献主要利用目标的结构，采用逐决策回报、递归价值估计、截断 trace、控制变量或直接估计 occupancy ratio。

| 方法                                                 | 校正机制                                                       | 保留什么                                                         | 代价与适用范围                                                                                   |
| ---------------------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Per-decision IS (Precup et al., 2000)[^precup]       | 每个时刻的奖励乘至该时刻所需的 prefix product                  | 对按时序分解的回报保持精确期望                                   | prefix 越长仍会增方差；依赖奖励分解，不能把旧策略的终局 reward 当成目标策略未来值。              |
| Doubly robust OPE (Jiang & Li, 2016)[^doubly-robust] | IS 校正加值函数模型控制变量                                    | 模型精确或重要性覆盖充分时可降低方差；偏差误差与 IS 误差可互补   | 需要拟合 value/Q；经典保证是 off-policy evaluation，不自动给深度 actor 的训练梯度保证。          |
| ACER (Wang et al., 2017)[^acer]                      | 截断 action ratio，并加偏差校正项与 trust region               | 在具体 actor-critic 估计器中缓解极端权重                         | 校正项和 trust-region 增加复杂度；截断单独使用会引入偏差。                                       |
| V-trace / IMPALA (Espeholt et al., 2018)[^vtrace]    | 单步 TD ratio 与 trace 系数分别截断                            | 分布式 actor-learner 的有界递归估计和明确的固定点                | 对应的固定点随截断参数改变；属于 critic/TD 结构，不是原始轨迹无偏 IS。                           |
| Retrace($\lambda$) (Munos et al., 2016)[^retrace]    | $c_t=\lambda\min(1,\rho_t)$ 截断多步 trace                     | 对广泛行为策略保持收缩 Bellman 算子，并有效利用近 on-policy 数据 | 依靠 $Q$ 递归与 Bellman 算子；不直接估完整序列回报或任意散度梯度。                               |
| Emphatic weightings (Imani et al., 2018)[^ace]       | 通过 follow-on trace 构造与目标 policy gradient 相容的状态权重 | 给出一般 off-policy policy-gradient theorem                      | 额外迹变量方差可很大，强调状态分布校正不能只看 action ratio。                                    |
| DICE / DualDICE (Nachum et al., 2019)[^dice]         | 直接估计状态或 state-action occupancy density ratio            | 避开逐步 ratio 的长乘积                                          | 变成密度比函数估计与 saddle-point 优化；要求覆盖和函数逼近条件，无法补回行为分布从未覆盖的状态。 |

这些方法中，**最优雅的共同点是不用一个全序列权重硬扛所有误差**。可分解奖励时用 per-decision IS；有可靠 value/Q 时用 doubly robust、V-trace 或 Retrace；平稳占用比可识别时学习 density ratio。代价分别是状态/动作覆盖、value 模型误差、截断目标偏移或额外密度比优化。它们都明确依赖被估计的量，不能把某个方法的稳定性定理转述成“IS 方差已解决”。

## MirrorOPD 的精确映射

### 目标一：本轮固定 rollout prefix 上的 Mirror Descent 拟合

令 $\mathcal S_k$ 为本轮 rollout 产生并固定的 prefix 集，$\hat d_k$ 为其经验测度，单 prefix 的完整词表损失为

$$
f_\theta(s)=\mathrm{KL}\bigl(p_\theta(\cdot\mid s)\Vert T(\cdot\mid s)\bigr).
$$

本轮 MD 子问题是 $L_k(\theta)=\mathbb E_{s\sim\hat d_k}f_\theta(s)$。其定义就是在这批固定状态上拟合目标；若各 prefix 的条件分布与损失都用 full-vocab 精确求和，动作层没有 Monte Carlo 误差。对这个固定经验目标，额外乘 $P_\theta(s)/P_\beta(s)$ 会把目标改成另一状态加权目标。因此它不需要 prefix IS。

这个无偏说法仅指新鲜独立 prefix 下，对固定参数的 minibatch 梯度估计 $\mathbb E_{s\sim d}[\nabla f_\theta(s)]$；重复优化同一批后，参数依赖该批数据，不能继续按独立新样本解释。full-vocab 保证的是条件动作积分精确，不保证 held-out prefix 上的损失下降。

### 目标二：旧 prefix 估计当前 student 访问状态上的条件风险

若目标明确改成 $J_{\mathrm{cond}}(\theta)=\mathbb E_{s\sim d_\theta}f_\theta(s)$，而 prefix 由行为策略 $\beta$ 采得，则有限时域状态比率为

$$
w_t^{\mathrm{state}}=\frac{P_\theta(y_{<t}\mid x)}{P_\beta(y_{<t}\mid x)}
=\prod_{j<t}\frac{p_\theta(y_j\mid x,y_{<j})}{\beta(y_j\mid x,y_{<j})}.
$$

它将旧 prefix 上的**条件风险均值**变换为当前访问测度上的均值，故只校正了 $\mathbb E_{d_\theta}[\nabla f_\theta(s)]$ 这一项。若要优化整个 $J_{\mathrm{cond}}$，访问测度本身依赖 $\theta$，梯度还含

$$
\mathbb E_{s\sim d_\theta}\left[f_\theta(s)\nabla_\theta\log P_\theta(s\mid x)\right],
$$

亦即 prefix score/future-cost 项。stop-gradient 的 state weight 不会自动产生它。若实验的 held-out 指标是固定的 initial/teacher bank，则它也不是 $d_\theta$ 风险；向其加 student prefix IS 会更换评测分布。

### 为什么长 prefix IS 通常不优雅

即使每 token 的 log-ratio 很小，prefix 权重的对数仍为和：$\log w_t=\sum_{j<t}\log\rho_j$。方差受每个位置的均值偏差与协方差共同影响；token 相关、温度或采样截断 mismatch 会使尾部更重。自归一化能限制整体尺度但有限样本有偏；clip 能界定数值但改变测度；位置窗口乘积更稳却不再是完整 prefix ratio；按 ESS 调温能自适应控制退化但仍选择了一个偏置目标。任何方法都必须报告目标权重与实际用的变换权重、原始/截断 ESS、位置分桶、尾部质量、梯度有效样本数。

MirrorOPD 此处的优雅处理顺序是：新鲜 rollout 更新目标状态测度；通过更多独立 prompts/states 扩大覆盖；用 fixed held-out banks 评估跨状态泛化；将 prefix IS 保留为概率诊断或有明确目标定义的离线消融。若要引入 state weighting，先做短 prefix、stop-gradient condition-risk ablation，并独立审计行为分布概率，而不是把 VESPO 的 sequence-level advantage weight 搬过来。

## 长时域问题如何出现

把自回归生成写成 MDP：$s_t=(x,y_{<t})$，$a_t=y_t$，行为策略为 $\mu$，当前策略为 $\pi$，单 token 比率为

$$
\rho_t=\frac{\pi(a_t\mid s_t)}{\mu(a_t\mid s_t)}.
$$

策略梯度中的第 $t$ 项依赖两种随机性：前缀状态 $s_t$ 如何到达，以及当前位置动作 $a_t$ 如何采样。因此从行为轨迹估计当前策略下的 token 项，通常需要累计至当前位置的比率

$$
\rho_{1:t}=\prod_{j=1}^{t}\rho_j.
$$

若对一个固定状态上完整动作分布直接计算条件 KL，动作期望由词表求和完成，当前位置的 $\rho_t$ 不需要通过 Monte Carlo 估计。把该固定状态的梯度从旧状态分布 $d^\mu_t$ 改为目标状态分布 $d^\pi_t$，使用的是到达状态的比率

$$
w^{\mathrm{state}}_t=
\frac{P_\pi(y_{<t}\mid x)}{P_\mu(y_{<t}\mid x)}
=\prod_{j<t}\rho_j.
$$

这就是本文所谓的 prefix IS。它与 token policy-gradient 中包含当前动作的 $\rho_{1:t}$ 相差一个 $\rho_t$，因为两个估计对象不同。

对完整序列标量回报 $R(\tau)$，换测度的精确轨迹比是

$$
W_T=\frac{P_\pi(y_{1:T}\mid x)}{P_\mu(y_{1:T}\mid x)}
=\prod_{t=1}^{T}\rho_t.
$$

在支持条件与固定参数条件成立时，普通 IS 的期望精确，但多个随机比率相乘，尾部很容易支配有限样本。裁剪、截断、自归一化、长度归一化与 VESPO 式重塑都在改变偏差—方差权衡；只有未变换的正确比率保留原来的精确测度恒等式。

## 经典 RL 的几种处理

### Per-decision IS：只带到当前决策

轨迹级估计把完整轨迹权重 $W_T$ 乘在整条序列目标上。Per-decision IS 利用因果结构，将时刻 $t$ 的奖励/梯度项只乘到该项所需的前缀权重。对按时刻分解的回报，形式通常是 $\sum_t \gamma^t (\prod_{j\le t}\rho_j)r_t$，而不是让每个早期奖励都承担整个未来后缀的随机比率。

这不是把 prefix product 截短成近似：在条件正确时，后缀动作比率的条件期望为 1，因此它们对较早的项没有必要，只增加方差。Precup、Sutton 与 Singh 的 eligibility-trace 工作把同一问题扩展到多步值估计；后来的 doubly robust 估计再加入模型/值函数控制变量，在模型误差和 IS 噪声之间取舍。前提是目标量能按时间分解，且优势/价值项定义与策略相符。

### Retrace 与 V-trace：截断 trace 系数换稳定性

Retrace($\lambda$) 用 $c_t=\lambda\min(1,\rho_t)$ 限制多步校正项，并构造收缩的 Bellman 算子。它面向 off-policy 值函数评估/控制，保留适当的固定点性质，同时避免原始长乘积。它不是一个仍然完全无偏的完整轨迹 IS 估计器；安全性来自特定算子、截断方式和固定点分析。

IMPALA 的 V-trace 使用两个截断尺度：$\bar\rho$ 控制单步 TD 残差权重，$\bar c$ 控制 trace 延伸。它容忍 actor/learner 异步与 policy lag；截断会改变估计目标所对应的策略，论文通过相应固定点刻画这种偏差。工程上的优点是逐步更新、数值有界；适用基础则包括 critic、TD 递推和可用的状态转移结构。

ACER[^acer] 也采用截断 IS，并加入偏差校正项和 trust-region 更新，试图避免把所有极端权重简单压扁。它说明“截断即丢弃”并非唯一方案，但校正项本身会增加估计复杂度。

### Occupancy ratio：估计状态密度比，而非乘完整历史

DualDICE[^dice] 等方法直接学习 $d^\pi(s)/d^\mu(s)$ 或相关稳态占用比，绕过逐 token 连乘。这是对 state IS 的另一条路线，而非序列动作 IS。它依赖覆盖、函数逼近、分布平稳性和相应的 saddle-point/投影条件；对确定性 token 前缀树，目标与行为支持不足时，任何 density-ratio learner 也无法补出零覆盖状态。

这些经典方法给出的共同原则是：利用目标的时间结构，尽早消掉不相关后缀；若必须保留长历史，就通过截断、控制变量、信任域或直接占用比估计管理方差。它们通常把偏差放在明确的目标或算子变化中分析，而不是宣称一个启发式权重仍是精确 IS。

### Per-decision IS 的精确公式与终局奖励边界

取固定长度 $H$，相同初始状态分布与相同环境转移，且 $P_\pi\ll P_\mu$。令 $W_t=\prod_{j=1}^t\rho_j$，$r_t$ 在第 $t$ 次动作及其转移后可观测，$G=\sum_{t=1}^H\gamma^{t-1}r_t$。由于未来动作比率在给定历史下的条件期望为 1，

$$
\mathbb E_\mu[W_H r_t]
=\mathbb E_\mu[\mathbb E_\mu(W_H\mid s_1,a_1,\ldots,s_{t+1})r_t]
=\mathbb E_\mu[W_t r_t].
$$

因此两个估计器都有目标期望：

$$
J(\pi)=\mathbb E_\mu[W_H G]
=\mathbb E_\mu\left[\sum_{t=1}^H\gamma^{t-1}W_t r_t\right].
$$

对只在 $H$ 给奖励的回答任务，上式只剩 $\mathbb E_\mu[W_H R]$，没有自然变成 $W_tR$。这解释了为什么经典 per-decision 思想不能直接让终局 RL 变成低方差 prefix-only RL。若有真实 $Q_t^\pi(s_t,a_t)$，策略梯度才可按

$$
\nabla J(\pi_\theta)
=\sum_{t=1}^H\gamma^{t-1}\mathbb E_\mu
\left[W_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)Q_t^\pi(s_t,a_t)\right]
$$

计算；此处 $Q_t^\pi$ 是从 $t$ 开始的折扣回报均值。旧 rollout 的 sampled return 条件期望一般是 $Q_t^\mu$，不能直接替代。值函数方法把 suffix 校正转成学习 $Q^\pi$ 的问题，而非免费丢掉后缀。[^precup]

### Doubly robust：用价值模型解释大部分，只让 IS 校正残差

给定固定的近似模型 $\hat Q_t$，令 $\hat V_t(s)=\mathbb E_{a\sim\pi}\hat Q_t(s,a)$，$\hat V_{H+1}=0$。有限时域 sequential doubly robust 估计器的一种写法为

$$
\widehat J_{\mathrm{DR}}
=\hat V_1(s_1)+\sum_{t=1}^H\gamma^{t-1}W_t
\left[r_t+\gamma\hat V_{t+1}(s_{t+1})-\hat Q_t(s_t,a_t)\right].
$$

在正确 ratio、覆盖、可积及估计器独立性条件下，换测度后相邻价值项望远镜相消，只剩 $J(\pi)$。近似值函数足够好时，随机 TD 残差较小，通常比直接让巨大 $W_H$ 乘完整回报更稳。它仍包含 prefix products，方差不会无条件消失；模型由同一批数据拟合时要单独处理数据依赖，常用独立拟合或 sample splitting。该论文针对 OPE；把它迁移到 actor 训练还要审计策略梯度与 value 依赖。[^doubly-robust]

### ACER：截断后显式补回缺的动作质量

对固定状态和任意可积向量 $g(a)$，$\rho(a)=\pi(a)/\mu(a)$ 有精确恒等式

$$
\mathbb E_\mu[\rho(a)g(a)]
=\mathbb E_\mu[\min(c,\rho(a))g(a)]
+\mathbb E_\pi\left[\left(1-\frac{c}{\rho(a)}\right)_+g(a)\right].
$$

第一项限制已有行为样本的权重；第二项是从目标动作分布计算的偏差校正。ACER 将这类分解与 Retrace、trust region 组合。它比“直接 clip 然后声称无偏”更清楚，但并没有自动校正状态访问分布。MirrorOPD 的 next-token 已用全词表求和，恰好无需在这一层使用 sampled-action IS；剩余难点是 prefix 测度与有限状态覆盖。[^acer]

### V-trace：分别限制单步修正和历史传播

定义 $\bar\rho_t=\min(\bar\rho,\rho_t)$、$c_t=\min(\bar c,\rho_t)$。IMPALA 的截断残差和递归 target 为

$$
\delta_t=\bar\rho_t[r_t+\gamma V(s_{t+1})-V(s_t)],
\qquad
v_s=V(s_s)+\sum_{t=s}^{s+n-1}\gamma^{t-s}
\left(\prod_{i=s}^{t-1}c_i\right)\delta_t.
$$

常取 $\bar c\le1$，历史 trace 的乘积便不再因单个大 ratio 膨胀。该值算子的固定点对应

$$
\pi_{\bar\rho}(a\mid s)
=\frac{\min(\bar\rho\mu(a\mid s),\pi(a\mid s))}
{\sum_b\min(\bar\rho\mu(b\mid s),\pi(b\mid s))}.
$$

$\bar\rho$ 决定目标策略偏移，$\bar c$ 决定多步传播和方差；两种角色分开是其设计亮点。这是 Bellman/value target 的保证，actor 使用的状态分布与近似 critic 仍需另行分析。[^vtrace]

### 状态分布校正：OffPAC、ACE 与 DICE 的分工

OffPAC 给出了早期高效 off-policy actor-critic 实现，但“只修正当前动作”不能概括一般目标的真实 policy gradient。ACE 的 emphatic policy-gradient theorem 及反例表明，状态权重不正确时，某些早期 semi-gradient 方法会收敛到错误解。Emphatic weighting 用 follow-on trace 形成目标所需的权重；其长期乘积仍可能带来方差。[^offpac][^ace]

_Breaking the Curse of Horizon_ (2018) 直接研究 stationary visitation density ratio，而非逐轨迹乘 ratio；DualDICE (2019) 进一步使折扣 state-action occupancy-ratio 估计不要求显式知道行为策略。它们把序列历史问题变成占用分布的函数估计问题，适合有共享、可重复访问状态的环境。[^breaking][^dice]

LLM 的原始状态 $s_t=(x,y_{<t})$ 包含完整历史；在确定性 prefix 树上，每个状态有唯一历史。若不做表示压缩，状态密度比本身就是该历史的 prefix product，换个名字不会消掉二阶矩。采用压缩表征后可能共享统计强度，但要论证该表征对未来奖励/条件风险是否充分，否则引入新的近似偏差。这是 DICE 迁移到我们 full-prefix OPD 的主要障碍。

### Marginalized IS：有限时域直接递推状态边缘分布

_Towards Optimal Off-Policy Evaluation for Reinforcement Learning with Marginalized Importance Sampling_（Xie、Ma、Wang，NeurIPS 2019）研究有限时域、非平稳 episodic MDP，逐步估计目标策略的状态边缘分布，用 $d_t^\pi(s)/d_t^\mu(s)$ 替代每条历史的随机 product。论文在相应覆盖与估计条件下给出随 horizon 多项式变化的 OPE 误差界。[^mis]

它比 stationary DICE 更直接对应“第几个 token”上的 state ratio，但收益依赖状态能够跨历史共享统计。若把每个完整 prefix 都当独特 state，状态数量及覆盖难度随长度膨胀，不能把论文的多项式 horizon 依赖解释成任何 LLM prefix IS 都会低方差。采用表征压缩或分层状态时，需要另行说明 Markov/充分性假设。

### TRPO：固定旧状态测度是有意定义的代理目标

TRPO 使用旧策略状态访问分布下的局部 surrogate，并借助 performance-difference identity、优势函数及策略距离上界控制它与实际回报改善的差异。它没有每次更新都用长 prefix IS 恢复当前状态访问分布；PPO 进一步用 clip/KL 目标作可实现近似。对无限时域折扣任务，TRPO 的经典界使用最大状态下的策略距离和有界优势；实际 sampled average KL 及 PPO clipping 不自动满足该理论条件。[^trpo]

这与 MirrorOPD 的固定 prefix 子问题在结构上相近：先固定 $d_k$ 解局部条件分布问题，再 rollout 更新访问测度。区别是 TRPO 的性能界需要与整体回报对应的优势；仅有逐状态 full-vocab KL 的解析正确性，还不能推出当前平台期 held-out bank 单调下降。借鉴点是明确代理目标、限制偏移并及时 refresh，而非直接搬用 TRPO 的性能定理。

### 二阶矩、ESS 与可证明的 prefix bound

对正确比率、固定 prompt 与位置，设 $P_t$ 和 $M_t$ 为目标和行为 prefix 分布，则

$$
\mathbb E_{M_t}w_t=1,\qquad
\mathbb E_{M_t}w_t^2
=\exp D_2(P_t\Vert M_t)
=1+\chi^2(P_t\Vert M_t).
$$

若每个可达历史都满足 $\mathbb E_{a\sim\mu(\cdot\mid s)}[\rho(a,s)^2]\le1+\epsilon$，利用逐步条件期望递推得

$$
\mathbb E_\mu W_t^2\le(1+\epsilon)^t.
$$

这项假设是逐历史的二阶矩控制，强于训练平均 KL 很小。$D_\mathrm{KL}$ 小通常不能给 $D_2$ 上界；罕见动作上的大 ratio 足以让后者很大。独立同分布轨迹、固定长度、有限二阶矩及大样本下，常用 $\mathrm{ESS}/N\approx1/\mathbb E_\mu W_t^2$。有限批 ESS 是随机诊断量，token 之间也不独立，不能把一条长序列的所有 token 算成许多独立轨迹。对我们的 64-response probe，应优先报告按相同位置的跨轨迹 ESS 和跨批分布。

## LLM RL 的代表论文

### 优先读：VESPO、CTPO 与 GSPO

| 论文                     | IS 层级与核心式                                                  | 偏差/方差处理                                                                                    | 对我们的启发                                                                                                 |
| ------------------------ | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| VESPO，Shen et al., 2026 | 完整序列比 $W=\prod_t\rho_t$；使用 $\phi(W)=W^{c_1}e^{c_2(1-W)}$ | 在序列权重上平滑降权，避免长度归一化；偏离 $\phi(W)=W$ 后优化的是重塑 surrogate                  | 适合 scalar sequence-reward policy gradient 的异策略稳态；不是 full-vocab 条件 KL 的现成修正                 |
| CTPO，Zhang et al., 2026 | 对 token 梯度项用累计比 $\rho_{1:t}=\prod_{j\le t}\rho_j$        | 精确 prefix 校正避免 suffix 权重；位置自适应 clip 根据 log-ratio 随位置增长调节，clip 后重新有偏 | 最接近“每个位置都用足所需 prefix”的 sampled policy-gradient 理论；full-vocab OPD 少一个当前位置 action ratio |
| GSPO，Zheng et al., 2025 | $s(y)=\exp(T^{-1}\sum_t\log\rho_t)=W_T^{1/T}$                    | 整序列 clip 与组级奖励匹配；几何平均稳定数值，但不等于真实序列密度比                             | 证明 sequence-level objective/clip 可实用；不能把长度归一化 ratio 当作无偏 sequence IS                       |

VESPO 从重塑权重诱导的提议分布 $Q\propto\mu\phi(W)$ 出发，考虑 $Q$ 与行为/目标策略的 KL 接近性，并对 $E_Q[W]$ 加矩约束，得到理论核 $\phi(W)=W^\alpha e^{-\lambda W}$。实验中使用归一化平移式 $\phi(W)=W^{c_1}e^{c_2(1-W)}$，并按正负优势分别调参，重塑系数 detach 后作为梯度缩放。重要解释是：变分推导给了权重设计原则，实际训练仍是有意偏离原始无偏 IS 的 sequence-level surrogate；$c_1,c_2$ 的经验设定不等于在线强制满足理论矩约束。文中在 batch 大小固定的前提下扫描 gbs/mbs=4–64，报告数学任务 reward/accuracy、长度、KL、entropy、policy-gradient loss，以及完全异步结果。它处理的是 rollout-age 的策略梯度，不是 prefix 条件散度的状态加权。

CTPO 将 LLM 生成表示为有限时域 MDP，指出 GRPO 当前 token ratio 漏掉 state prefix；完整 sequence ratio 对每个早期 token 梯度还纳入了不相关后缀。它讨论累计到当前动作的比率对 token policy-gradient 的测度变换；严格身份需要目标策略的条件价值/优势，而实际 outcome advantage 另有条件，随后对累计 log-ratio 使用随 $\sqrt t$ 增长的对数裁剪范围。这个根号关系来自其方差/实验刻画，不是任意语言、工具轨迹或强相关 token 下的分布无关保证。组相对 outcome advantage、clip、有限 rollout 和工具终止均须另查无偏条件。

GSPO[^gspo] 的长度归一化有明确实用意义：不让几百/几千个 token 的原始概率乘积直接决定 clip 区间。代价是 $W^{1/T}$ 不再是 $P_\pi(y)/P_\mu(y)$，所以它不是精确序列重要性抽样。论文还报告 MoE 路由扰动与序列级 ratio 的稳定性，但其优势不应单独归因于“精确校正了 off-policy state”。

### Prefix 与分段工作

**MinPRO / Prefix-ratio study (Lei et al., 2026)**[^minpro]针对 prefix 漂移提出 prefix importance ratio 与基于 ratio 的位置处理。arXiv:2601.22718 当前元数据标题为 _Stabilizing Off-policy LLM Optimization with Prefix Importance Ratio_；旧实验综述使用过“A Step Back / MinPRO”称呼，标题和版本需要以引用的原文版本为准。我们已有逐篇核对：完整 prefix ratio 本身未改善其训练，按 ratio 过滤最小 1% token 的实验反而难以学习，作者讨论了尾部位置偏置。这个负结果很重要：正确的分布比率并不保证有限样本加权后梯度更有用；长链的 prefix ESS 和目标 token 所处位置必须一起看。设置、指标及原文定位见[陈旧数据实验综述](2026-10-21-陈旧数据训练的指标与实验对照.md#104-minproprefix-信息与丢弃位置)。

**SPIRE（Zhan & Zhang, EMNLP 2026）：正文估计器与证明不一致。**[^spire] 令第 $s$ 段的比率为 $w_s$。完整序列权重 $W=\prod_s w_s$ 的正确换测度是

$$
\mathbb E_\beta\left[W A\sum_s\nabla\log w_s\right].
$$

正文 Eq. (2) 使用的更新项却是

$$
\mathbb E_\beta\left[A\sum_s w_s\nabla\log w_s\right].
$$

Appendix A 证明的是前者；prefix-measurable boundaries 只允许正确分解行为密度，不能把 $W$ 替换成每段自己的 $w_s$。此外，Eq. (2) 将含梯度的向量写成 scalar loss 也需要解释；这里按作者意图把它理解为直接的梯度估计器来审计。

取两步、固定分段边界、独立动作 $a,b\in\{0,1\}$，行为策略都为 Bernoulli$(0.5)$；目标第一步为 Bernoulli$(p)$、第二步为固定 Bernoulli$(q)$，终局回报 $R=ab$。只对 $p$ 求导，真实目标 $J=pq$，故 $\partial_p J=q$。单独第一段加权时

$$
\mathbb E_\beta[Rw_1\partial_p\log\pi_p(a)]
=\mathbb E_{b\sim\beta}b\cdot\partial_p\mathbb E_{a\sim\pi_p}a
=0.5,
$$

第二段对 $p$ 的 score 为 0；取 $q=0.8$，估计器期望 $0.5$ 与真实梯度 $0.8$ 不同。加回 $w_2$ 后才恢复 $0.8$。固定边界满足该文 stopping-time 假设，故该假设单独不足以保证正文估计器无偏。$A=R$ 已是合法零 baseline 情形，不能靠 group baseline 消去这个一般缺口。

该工作关于“只校正当前段便将指数复杂度变为线性”的主张需要额外状态价值、目标前缀测度或其他结构。本文暂不把它与 Retrace、V-trace、DR 等已有明确算子/恒等式的工作放在同等证据等级。数值枚举反例保存在 [SPIRE 两步反例](sources/prefix-is-20261116/spire-counterexample.json)。

**MinPRO 的实际替代量。** 它用 $m_t=\min_{j<t}\rho_j$，并将 $\prod_{j\le t}\rho_j$ 替换为 $m_t\rho_t$，clip 后 detach，作为 sampled log-prob 梯度的系数。它只保留“历史是否出现过不利 ratio”的信息，不等于原始 prefix 密度比。优点是 running minimum 易算，不存在乘积上溢；代价是新的目标偏差，而且 minimum 的分布仍会随位置变化，不能承诺位置无偏。[^minpro]

Turn-level IS 与 segment methods 将校正作用在多轮 agent 的 turn 或分段行为策略，减少整条对话一个 product 的权重跨度。适用条件通常是边界带有正确的 behavior policy 记录，段内奖励/状态可按段处理。它们解决的是 episode 结构与采样预算，不会自动替代每个 token prefix 的状态比率。

CTPO、MinPRO 与当前已有的 VESPO/GSPO 定量细节、实验设置和失败边界已在[陈旧数据训练综述](2026-10-21-陈旧数据训练的指标与实验对照.md)维护；本文只保留与经典 RL、不同 IS 对象及 MirrorOPD 映射有关的比较。

## 当前 MirrorOPD 的实测证据

本机从 update480 的模型与 AdamW checkpoint 分叉所做的三批测量，报告了同评分器 $p_k/p_0$ 的 4096-token ESS/N 在 inner16 约为 2.33%–5.08%；真实行为分母在 inner0 已有明显长 prefix 集中。三批 initial Math RKL 均在 inner8→16 回升，teacher bank 则有不同走势。完整逐样本证据与限制见[平台 prefix IS 诊断](../../experiments/reuse-budget/inner-bank/prefix-probe.md)。它是 64 条轨迹下的经验 ESS，不是总体方差保证，也不是 IS 加权训练实验；当前支持“不能盲目将长 prefix 权重塞进训练”，不支持“所有状态权正则化都无效”。

## 我们的研究优先级

1. **主要处理：提早 refresh 与增加独立 prompt/state。** 这直接补目标分布覆盖，不依赖极端权重重加权未覆盖区域。比较时按 rollout、actor update 和训练总 GPU 时间分别对齐。
2. **先做概率校准，再解释 IS ESS。** 对固定参数、相同轨迹重复评分，隔离 FSDP/vLLM 数值误差、tokenizer 支持、temperature/top-p、stop 条件和 token 对齐。长 prefix log-ratio 对小的每 token 系统误差很敏感。
3. **如要测校正，先做短 prefix、state-only、detached 权重的离线梯度对照。** 同时画原始/截断 ESS、有效轨迹数、被剪质量、位置分桶和梯度 cosine。短窗口乘积、clipping 或 self-normalization 需明确标为有偏估计。
4. **若目标转向 terminal-reward RL，再按 CTPO/VESPO 的估计对象做独立消融。** CTPO 的 per-token cumulative ratio 与 VESPO 的 full-sequence transformed ratio 是两条不同的算法路线；不能用 MirrorOPD 的条件 full-vocab bank 结果宣称谁更好。

## 参考文献

[^precup]: Doina Precup, Richard S. Sutton, Satinder Singh. Eligibility Traces for Off-Policy Policy Evaluation. _ICML_, 2000. [论文与版本](https://www.cs.utexas.edu/~shivaram/readings/b2hd-PrecupSS2000.html).

[^retrace]: Rémi Munos, Tom Stepleton, Anna Harutyunyan, Marc G. Bellemare. Safe and Efficient Off-Policy Reinforcement Learning. _NeurIPS_, 2016. [论文](https://arxiv.org/abs/1606.02647).

[^vtrace]: Lasse Espeholt et al. IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures. _ICML_, 2018. [PMLR](https://proceedings.mlr.press/v80/espeholt18a.html).

[^acer]: Wang et al. Sample Efficient Actor-Critic with Experience Replay (ACER). _ICLR_, 2017. [arXiv](https://arxiv.org/abs/1611.01224).

[^doubly-robust]: Nan Jiang, Lihong Li. Doubly Robust Off-policy Value Evaluation for Reinforcement Learning. _ICML_, 2016. [PMLR](https://proceedings.mlr.press/v48/jiang16.html).

[^dice]: Ofir Nachum et al. DualDICE: Behavior-Agnostic Estimation of Discounted Stationary Distribution Corrections. _NeurIPS_, 2019. [arXiv](https://arxiv.org/abs/1906.04733).

[^gspo]: Chujie Zheng et al. Group Sequence Policy Optimization. arXiv:2507.18071, 2025. [arXiv HTML](https://arxiv.org/html/2507.18071v2).

[^vespo]: Guobin Shen et al. VESPO: Variational Sequence-level Soft Policy Optimization for Off-Policy LLM Training. arXiv:2602.10693, 2026. [arXiv HTML](https://arxiv.org/html/2602.10693v3).

[^ctpo]: Yuheng Zhang et al. Rethinking Importance Sampling in LLM Policy Optimization: A Cumulative Token Perspective. arXiv:2605.07331, 2026. [arXiv HTML](https://arxiv.org/html/2605.07331v1).

[^minpro]: Shiye Lei, Zhihao Cheng, Dacheng Tao. A Step Back: Prefix Importance Ratio Stabilizes Policy Optimization (v1)；v2 检索元数据题名为 Stabilizing Off-policy LLM Optimization with Prefix Importance Ratio. arXiv:2601.22718, 2026. [arXiv HTML](https://arxiv.org/html/2601.22718v1); 本地实证审阅见[既有综述](2026-10-21-陈旧数据训练的指标与实验对照.md#104-minproprefix-信息与丢弃位置).

[^spire]: Chenlu Zhan, Yufei Zhang. SPIRE: Unifying Partial Rollouts, Replay, and Asynchrony in LLM Reinforcement Learning via Segment-wise Prefix Importance Sampling. _EMNLP_, 2026, pp. 36688–36701. [ACL Anthology](https://aclanthology.org/2026.emnlp-main.1613/).

[^offpac]: Thomas Degris, Martha White, Richard S. Sutton. Off-Policy Actor-Critic. _ICML_, 2012. [arXiv](https://arxiv.org/abs/1205.4839).

[^ace]: Ehsan Imani, Eric Graves, Martha White. An Off-policy Policy Gradient Theorem Using Emphatic Weightings. _NeurIPS_, 2018. [arXiv](https://arxiv.org/abs/1811.09013).

[^breaking]: Qiang Liu, Lihong Li, Ziyang Tang, Dengyong Zhou. Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation. _NeurIPS_, 2018. [arXiv](https://arxiv.org/abs/1810.12429).

[^trpo]: John Schulman et al. Trust Region Policy Optimization. _ICML_, 2015. [PMLR](https://proceedings.mlr.press/v37/schulman15.html).

## 检索与核对记录

完整原始检索（包括无关命中与来源诊断）保留在[初始检索](sources/prefix-is-20261116/initial-search.md)、[扩展检索](sources/prefix-is-20261116/expanded-search.md)和[经典 RL 检索](sources/prefix-is-20261116/classic-search.md)。已读取 VESPO、GSPO、CTPO、MinPRO v1 官方 HTML，以及 Retrace、ACER、DR、ACE、SPIRE 官方 PDF；其他经典工作核对官方 arXiv/PMLR 摘要，文中不把未逐定理审阅的摘要声明视为独立验证。SPIRE 的 Eq. (2)、Theorem 1 与 Appendix A 单独作了反例核验。前文 MinPRO 的版本题名差异来自 v1 HTML 与当前 API 元数据，保留明确版本即可。

[^pois]: Alberto Maria Metelli, Matteo Papini, Francesco Faccio, Marcello Restelli. Policy Optimization via Importance Sampling. _NeurIPS_, 2018. [arXiv](https://arxiv.org/abs/1809.06098).

[^mis]: Tengyang Xie, Yifei Ma, Yu-Xiang Wang. Towards Optimal Off-Policy Evaluation for Reinforcement Learning with Marginalized Importance Sampling. _NeurIPS_, 2019. [arXiv](https://arxiv.org/abs/1906.03393).
