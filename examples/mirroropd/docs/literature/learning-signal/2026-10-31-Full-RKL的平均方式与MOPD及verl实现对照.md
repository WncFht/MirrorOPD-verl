# Full-RKL 的平均方式、OPD 消融与 verl 实现对照

核验日期：2026-10-31；TOP-D 公开实现补充核验于 2026-11-03。本文讨论 student rollout 上的 full-vocabulary reverse KL（全词表反向 KL，简称 full-RKL），比较论文的长度归一化与本机 verl 实现。论文定义、当前实现和建议分别标明；没有修改训练代码、运行 GPU 实验或推断历史实验实际使用了某个默认值。

**OPD 与 MOPD 不因教师数量不同而需要不同的 mean。词表维求和由 KL 定义决定，回答内和 batch 的平均则取决于目标与样本权重。现有消融不足以证明 `seq-mean-token-mean` 是 full-RKL 的统一优选。** 仅在复现 MOPD 原始论文的目标时，其明确的回答平均才要求这个配置。本机 MirrorOPD 的 full-RKL 当前采用全局 `token-mean`，数学上自洽；`full` 规定是否覆盖完整词表，不能决定外层如何平均。

TOP-D 的公开启动配方也显式采用 `seq-mean-token-mean`。[^topd-common] 这是复现其公开配方时需要对齐的配置；该配方聚合 sampled-token surrogate，不能作为 full-RKL 两种平均方式的独立消融，详见第 2.9 节。

## 1. 四种聚合的精确定义

设第 $i$ 条回答的第 $t$ 个预测前缀为 $s_{it}$，student 和 teacher 在该前缀的完整条件分布分别为 $p_{it}$ 和 $q_{it}$。先计算一个位置上的 KL：

$$
k_{it}=D_{\mathrm{KL}}(p_{it}\Vert q_{it})
=\sum_{v\in\mathcal V}p_{it}(v)\bigl[\log p_{it}(v)-\log q_{it}(v)\bigr].
$$

这里对词表是概率加权求和；概率本身已经实现对 $v\sim p_{it}$ 的期望。再除以词表大小会额外缩小 loss 和梯度。对 logits、log-probability 或词表项直接取普通算术平均，都不是上述 KL。

令有效训练掩码为 $m_{it}\in\{0,1\}$，$L_i=\sum_t m_{it}$，$S_i=\sum_t m_{it}k_{it}$，$N=\sum_i L_i$。下表先假定全部 $B$ 条回答非空。

| verl 配置                 | 全局 mini-batch 的标量  | 样本与 token 的权重含义                |
| ------------------------- | ----------------------- | -------------------------------------- |
| `token-mean`              | $\sum_i S_i/N$          | 每个有效位置等权；长回答包含更多位置   |
| `seq-mean-token-mean`     | $\frac1B\sum_i S_i/L_i$ | 每条回答的均值等权                     |
| `seq-mean-token-sum`      | $\frac1B\sum_i S_i$     | 每条回答的累计 KL 等权，不消除长度尺度 |
| `seq-mean-token-sum-norm` | $\frac1{BC}\sum_i S_i$  | 在上一项基础上除以固定尺度 $C$         |

这些名称及运算顺序已核对上游 `agg_loss`。[^verl-agg] 若所有回答等长，前两种相同；变长回答下，通常不同。固定 $C$ 也不等于每条回答各自的 $L_i$。若 `loss_scale_factor` 未设置，当前 `agg_loss` 会用输入 mask 的最后一维作为尺度；跨 micro-batch 使用这个模式时应显式保证尺度一致。

### 1.1 长度加权的影响

设一条回答长 2、每个位置 KL 为 1；另一条长 8、每个位置 KL 为 3：

$$
\mathcal L_{\mathrm{token}}=\frac{2\cdot1+8\cdot3}{2+8}=2.6,
\qquad
\mathcal L_{\mathrm{seq}}=\frac{1+3}{2}=2.
$$

前者两条回答的总权重为 20% 和 80%，后者为 50% 和 50%。这是假想的权重示例，不能单凭长度推断真实梯度大小，真实梯度还取决于每个位置的方向和幅度。

进一步令 $a_i=S_i/L_i$、$\bar L=B^{-1}\sum_iL_i$、$\bar a=B^{-1}\sum_i a_i$，按 $1/B$ 定义经验协方差。直接展开得：

$$
\begin{aligned}
\operatorname{Cov}_B(L,a)
&=\frac1B\sum_i(L_i-\bar L)(a_i-\bar a)\\
&=\frac1B\sum_i L_i a_i-\bar L\bar a,\\
\mathcal L_{\mathrm{token}}-\mathcal L_{\mathrm{seq}}
&=\frac{\operatorname{Cov}_B(L,a)}{\bar L}.
\end{aligned}
$$

因此，两种平均的差异来自长度与每条回答平均 KL 的相关性。训练时改变平均方式，还会改变不同位置梯度的相对权重，通常不能靠一个统一学习率系数抵消。相比之下，在同一个固定 batch 上，`seq-mean-token-sum` 恰好等于 `token-mean` 乘以 $N/B$；这个比例仍可能随训练改变。

### 1.2 多教师的域权重

若域 $d$ 在 batch 中有 $B_d$ 条回答，平均长度为 $\bar L_d$，则 `token-mean` 对该域分配的总系数为

$$
\frac{B_d\bar L_d}{\sum_e B_e\bar L_e},
$$

而 `seq-mean-token-mean` 的域系数为 $B_d/B$。例如两个域回答数相同、长度分别为 1000 和 4000，前者会形成 20%/80% 的 token 权重，后者仍为 50%/50% 的回答权重。这是聚合公式的推导，不是关于哪种配置成绩更好的实验证据。

MOPD 通常按域给回答路由到一位教师。域权重由数据采样与聚合方式共同决定。若想让每个域完全等权，需要另行定义按域平均的目标；仅设置 `seq-mean-token-mean` 不会自动抵消不均衡的域采样数量。

## 2. 论文证据

| 来源                                      | 可以直接核实的位置                      | 时间维和回答维的定义                                                                   | verl 对应方式                       |
| ----------------------------------------- | --------------------------------------- | -------------------------------------------------------------------------------------- | ----------------------------------- |
| MOPD，Ma 等，2026                         | §3.2，Eq. (1)                           | $\mathbb E_{x,y}[\frac1{\lvert y\rvert}\sum_t\sum_v p(v)\log(p(v)/q_d(v))]$            | `seq-mean-token-mean`               |
| MiMo-V2-Flash，2026                       | §4.4，Eq. (7)                           | sampled PG surrogate 内显式使用 $1/\lvert y\rvert$                                     | `seq-mean-token-mean`               |
| GKD，Agarwal 等，ICLR 2024                | Eq. (2)，Algorithm 1 第 10 行           | 先对一条序列的条件散度除以 $L_y$，再按 $1/B$ 聚合样本梯度                              | `seq-mean-token-mean`               |
| Rethinking OPD I，2026                    | §2.2，Eq. (4)                           | full-vocabulary 形式写为 $\mathbb E[\sum_t D_{\mathrm{KL}}]$                           | 按字面是 `seq-mean-token-sum`       |
| One-Shot OPD，2026                        | 附录 Table 3 与 Table 8                 | 单教师 OPD 标为 `token-mean`；MOPD 标为 `seq-mean-token-mean`                          | 两张表分别采用不同方式              |
| TOP-D 公开实现，2026                      | 固定 commit 的共同启动脚本与 `agg_loss` | 对每条回答的有效 token loss 求均值，再对回答求均值；聚合对象为 sampled-token surrogate | `seq-mean-token-mean`，公开配方配置 |
| MiniLLM，2023 预印本／ICLR 2024，后续修订 | §2.2，Length Normalization              | 对未来 reward-to-go 加长度归一化                                                       | 不能据此直接指定外层 loss reduction |

上述论文条目依据原文，TOP-D 条目依据固定版本的公开实现；不能把一项工作的归一化直接外推为所有 OPD 的统一标准。[^mopd][^mimo][^gkd][^rethinking][^oneshot][^minillm][^topd-common][^topd-algos]

### 2.1 MOPD 对 full-RKL 的答案是明确的

MOPD Eq. (1) 已经写出了 full-vocabulary RKL 的基本目标；§3.2.1 和 §3.2.2 随后分别介绍 sampled PG 与 teacher-top-$k$ 实现。三种表达中的时间归一化都保留 $1/\lvert y\rvert$。因此，将词表内的计算改成 full-RKL，没有数学理由顺带把外层回答平均改成全局 token 平均。[^mopd]

MOPD 的 TopK 公式还包含 $-p(v)+q(v)$ 修正，不能直接当成重归一化子词表 KL。覆盖完整词表时，这两个修正项的和为 $-1+1=0$，剩下标准 full-RKL；这里仍然没有 $1/\lvert\mathcal V\rvert$。

MiMo Eq. (7) 中的 $w_t$ 是训练／推理差异的 token 重要性权重，回答外仍保留 $1/\lvert y\rvert$。过滤掉某些 token 后，改用“通过过滤的 token 数”作分母会改变原式权重，不能作为默认等价变形。[^mimo]

### 2.2 One-Shot 论文与公开默认脚本存在可见差异

One-Shot 原文 Table 8 的 MOPD 聚合为 `seq-mean-token-mean`，而公开 `recipe/_common.sh` 第 65 行默认 `LOSS_AGG_MODE=token-mean`，第 197 行将它传给 actor；`recipe/mopd.sh` 自身没有覆盖这个变量。本机副本也符合这个结构。[^oneshot][^oneshot-common][^oneshot-mopd]

这能证明“论文报告的 MOPD 配置”和“当前公开脚本不加覆盖时的默认行为”不同。脚本允许环境变量和命令行覆盖；缺少作者每次运行的 resolved config 时，不能断言论文实验实际跑错了，或用公开默认值改写论文表格。

### 2.3 OPD 与 MOPD 共用同一个聚合问题

将单教师条件分布 $q(v\mid s)$ 换成按样本路由的 $q_{d(x)}(v\mid s)$，得到 MOPD。这个替换没有引入 $1/L_i$ 或 $1/\sum_iL_i$；令路由恒定，又回到单教师 OPD。因此“OPD 用 token mean，MOPD 用回答 mean”不是理论结论。多域长度差异会使权重差异更明显，但问题在单域、单教师的变长回答中已经存在。

论文公式能说明作者定义了哪个目标；它不能独自证明该目标在其他模型与任务上更好。对完整序列分布 KL，链式法则对应时间求和；对长度归一化的条件分布拟合，可以选回答平均；对有效训练位置的经验分布，可以选全局 token 平均。比较它们的任务效果，需要控制变量的消融。

### 2.4 与平均方式有关的实际消融

进一步以 OPD、aggregation、length normalization、token mean 等关键词检索，并核对论文正文与附录。严格目标是寻找：保持 full-vocabulary RKL、师生配对、数据、rollout、更新预算和优化设置相同，只切换 `token-mean` 与 `seq-mean-token-mean`。在已核验的文献中，没有找到满足这一条件的公开消融；下面的相关实验必须按改动对象解释。

| 论文与位置                                          | 实际消融变量                                          | 可核实的结果                                                                    | 与这里的 mean 的关系                                               |
| --------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| TurnOPD，§7.2，Table 6                              | 多轮轨迹内按 token 聚合、按 turn 聚合、渐进混合       | ALFWorld-1.7B 的 Same-Step Avg@4 为 83.0、85.0、85.1                            | 直接消融 KL 聚合权重，但不是跨回答的两种 mean；采用 top-$K$ RKL    |
| MiniLLM，§3.4，显示标题为 Table 4 的表              | 去掉未来 reward-to-go 的长度归一化                    | 验证 Rouge-L：27.4→17.4；Dolly：24.6→14.7                                       | OPD 的长度归一化证据，但修改未来回报，不是 full-RKL 外层 reduction |
| ReNIO，§4.4，Table 4                                | 去掉样本重要性权重的 batch 均值归一化                 | 三个数学 benchmark 的 Avg@12:42.04→39.54                                        | 说明权重尺度有影响，不能判定 token mean 与回答 mean 谁优           |
| DAPO，§3.3，§4.2 Table 1                            | sample-level mean 改为 token-level mean               | 渐进加法消融中 AIME24 avg@32:41→42                                              | 对这两个 reduction 最直接的比较，但任务是 RLVR，不是 OPD           |
| Dr. GRPO，附录 C，Figure 8                          | 分别移除长度归一化、reward 标准差归一化，并比较全移除 | 去掉长度归一化的变体响应更短，奖励与评测表现优于 vanilla GRPO                   | 明确消融长度分母；RLVR，且使用固定尺度，不是纯 OPD 证据            |
| Sampled-Token RKL Analysis／SuRe，附录 C.6，Table 9 | surprise 权重均值归一化、打乱、反转、uplift-only 控制 | MATH-500 avg@4：Vanilla 66.55，mean-normalized SuRe 69.20，exact-shuffled 67.95 | 检查 token 权重分配与尺度；各方法外层仍使用 token-mean             |

上述消融依据原文，数字保留原指标定义。[^turnopd][^minillm][^renio][^dapo][^drgrpo][^sure] MiniLLM HTML 的正文交叉引用仍写 Table 8，而实际包含这四组数值的表标题是 Table 4；定位时以 §3.4 和表内容为准。TurnOPD 的附录 D 明确将基础聚合标为 `seq-mean-token-mean`，其按 turn 平衡的改动不能被改称为在整个 batch 上使用 `token-mean`。

TurnOPD 的 85.1 与 85.0 只差 0.1 个百分点，该表没有提供足以判断这个差异是否显著的置信区间。其较清楚的证据是：在这个多轮 agent 设置中，83.0 的基础聚合与按 turn 调整权重后的结果不同；这不建立任意任务上的通用排序。

DAPO 的 41→42 是前序技巧已经加入后的渐进结果，不是从 vanilla GRPO 的 30 直接提高到 42 的单因素效应。Dr. GRPO 附录单独比较去掉两个归一化项，能够支持长度项有独立作用；把其主要方法的全部收益都归因于长度项仍不成立。

### 2.5 Full-RKL 的采用记录与尚缺少的对照

On-Policy or Off-Policy Learning? A Systematic Study of Distillation Dynamics 使用完整词表条件 KL，并在附录 C.5 明确区分：固定 horizon 的 token 平均仅给序列 KL 乘一个常数；其实际变长实验按每条 completion 的 realized length 归一化，通常不再是序列 KL 的常数倍。该文比较 rollout 来源、KL 方向、学习率、梯度裁剪和 sampled／full KL，但没有将两种外层 mean 作为独立消融。它证明回答平均是已被使用的 full-RKL 设置，不证明它优于全局 token 平均。[^dynamics]

GNDPO 的 global normalization 指跨 batch 对 token advantage 减均值、除标准差；IW-OPD 则用前缀累计差异构造位置权重。这些名字包含 normalization 的工作，也不能作为本问题中两个分母的直接比较。[^gndpo][^iwopd] Rethinking On-Policy Self-Distillation for Thinking Models 所说的部分 length-normalized 分析涉及 deliberation marker 频率，附录 E 则比较 token mask；它也没有给出这里要求的 mean 消融。[^thinking]

据此，文献支持“归一化和相对权重会影响训练”，还不支持“只要是 OPD/full-RKL，就应该统一改成回答平均”。延续当前全局 token 平均是可辩护的基线；若要回答本项目哪种方式任务表现更好，需要在同一 full-RKL 路径上做直接对照。实验还应另行控制全局 loss 尺度，因为 token mean 与固定尺度 token sum 对同一个 batch 只差整体倍数，而回答 mean 还会改变样本间相对权重。

### 2.6 标准 RL 目标下，长度分母有明确的理论答案

若目标是每道题的期望最终奖励，设完整回答包含终止动作，奖励 $R(x,y)$ 不显式依赖待求导参数，在常规可交换求导与期望的条件下：

$$
\begin{aligned}
J(\theta)&=\mathbb E_{x,y\sim\pi_\theta}R(x,y),\\
\nabla J
&=\mathbb E\left[R(x,y)\nabla\log\pi_\theta(y\mid x)\right]\\
&=\mathbb E\left[R(x,y)\sum_{t=1}^{|y|}\nabla\log\pi_\theta(y_t\mid s_t)\right].
\end{aligned}
$$

这里每条回答内必须求和，再对回答／问题平均。batch 外的回答平均体现每个 rollout 的采样权重；内部的 token 求和来自完整回答 log-probability 的导数，并不表示把一次终局奖励重复计算多次。采用合法的状态 baseline 或 return-to-go 可降低方差，但不产生额外的 $1/|y|$。Dr. GRPO 附录 A 的推导遵循这个目标。[^drgrpo]

因此，单看 loss 聚合，忠实于上述期望奖励梯度的参考形式是 `seq-mean-token-sum`，或再除以固定正数 $C$ 的 `seq-mean-token-sum-norm`。固定缩放对应优化 $J/C$，不改变最优策略。组内含自身 reward 的均值 baseline 还带有 $(G-1)/G$ 常数因子，可用 leave-one-out 修正；PPO clipping、reward 标准差归一化、off-policy correction 等是额外问题。固定长度分母只解决聚合引入的偏差，不能使整个实用 PPO 流程自动变成严格无偏。

GRPO 的 $1/L_i$ 对每条回答的 score-gradient 重新加权。对于相同正 advantage，短回答的系数更大；对于相同负 advantage，长回答被更弱地惩罚。这是 Dr. GRPO 指出的“正确回答趋短、错误回答趋长”的优化偏差。其附录 C 的独立消融支持长度项确实影响回答长度与训练效果。[^drgrpo]

baseline 的性质也会被改变。合法的回答前 baseline $b(x)$ 满足

$$
\mathbb E_y[b(x)\nabla\log\pi_\theta(y\mid x)]=0,
$$

但加入随机长度权重后，通常

$$
\mathbb E_y\left[\frac{b(x)}{|y|}\nabla\log\pi_\theta(y\mid x)\right]\ne0.
$$

所以，“无 baseline 时相当于优化长度加权 reward”也不能无条件推广到已有 baseline 的 GRPO。

一个自构造的精确反例：同一题只有短、长两种回答，长度分别为 1、2，二者 reward 都为 1；令长回答概率为 $p=\sigma(\theta)=1/2$。真实 $J=1$，故 $\nabla J=0$。无 baseline 的 REINFORCE 是合法无偏估计，其两个结果的 score-gradient 分别为 $-1/2$、$+1/2$，期望为零。若各自再除以实际长度，期望变成 $-1/8$，即在奖励完全不关心长度时凭空引入缩短偏好。这是本文对聚合的反例，不是论文实验。

### 2.7 DAPO 与固定分母的区别是 batch 随机缩放

令 $g_i$ 表示一条回答的 token-sum 梯度，$\bar L_B=B^{-1}\sum_iL_i$。对完全相同的 batch：

$$
\hat g_{\mathrm{DAPO}}
=\frac{\sum_i g_i}{\sum_iL_i}
=\frac1{\bar L_B}\underbrace{\left(\frac1B\sum_i g_i\right)}_{\hat g_{\mathrm{sum}}}.
$$

DAPO 去掉了样本之间各自的 $1/L_i$ 系数，但仍按当前 batch 平均长度缩放整个梯度。这正是它相对回答 mean 的重要变化。其经验结果支持这种变化改善长 CoT 训练，不能据此声称随机长度分母给出了原始期望奖励的精确无偏梯度。[^dapo]

对于有限 batch，$\bar L_B$ 与 $\hat g_{\mathrm{sum}}$ 可能相关，通常有 $\mathbb E[\hat g_{\mathrm{sum}}/\bar L_B]\ne\mathbb E[\hat g_{\mathrm{sum}}]/\mathbb E[\bar L_B]$。大 batch 下，在适当大数定律条件下，它近似原梯度除以期望长度；单个已采样 batch 的梯度方向则与未归一化版本完全一致。优化器、梯度裁剪和不同 batch 的缩放会影响实际更新。

也不能把 token mean 自动称为“每 token 奖励最大化”：若真正目标为 $\mathbb E[R]/\mathbb E[L]$，商法则还要求包含 $\nabla\mathbb E[L]$ 项，一般 token-mean PG 没有这个项。

### 2.8 后续方差研究的最优性依赖假设

VL Norm: Rethink Loss Aggregation in RLVR（2509.07558，初版名为 $\Delta L$ Normalization）进一步研究长度与梯度噪声，在 §3 把问题定义为：独立梯度估计具有相同均值，方差与各自长度成正比，在给定无偏约束下求最小方差线性组合。该统计模型下得到逆长度权重；论文报告其与 DAPO 结合，在 CountDown-3B 上达到同一 0.866 accuracy 的步数从 1000 降为 375。[^vlnorm]

该最优性不能脱离统计模型使用。真实长度是策略生成的随机量，长度与 reward、score-gradient 可相关；从 $\mathbb E[g_i]=g$ 不能推出 $\mathbb E[x_i(L)g_i]=\mathbb E[x_i(L)]g$。对实际随机长度重加权，要额外核对按长度条件化的均值或相关性。这里是本文对推广范围的判断，不是声称论文已经证明随机长度下的无条件无偏性。

另一个自构造的两轨迹精确枚举：短回答 $L=1,R=0$，长回答 $L=2,R=1$，长回答概率为 $\sigma(\theta)=1/2$，每 batch 独立采两条回答，不用 baseline，固定尺度 $M=1$。真实奖励梯度和普通回答均值估计的期望都是 $1/4$；VL Norm 的 $\alpha=1$ 逆长度加权估计期望为 $5/24$，DAPO 的 token mean 为 $7/48$。枚举表明，不同长度条件下梯度均值不同的实际轨迹分布，不满足“只凭权重和固定就保持无偏”的推广。后两者在这个例子中仍有正确更新方向，但不是同一个无偏估计。

$\lambda$-GRPO 则将 token 偏好设为可学习参数并报告性能提升，研究的是优化与经验表现；它不建立对所有 reward、任务、策略都唯一最优的长度分母。[^lambda-grpo]

所以，固定期望奖励目标后，“忠实于策略梯度的聚合”有明确答案：token sum、回答 mean、可加固定尺度。至于哪种近似有更好的有限预算表现，方差、裁剪和优化器仍会影响结果。这个区分应延续到 OPD：若要求完整序列 reverse-KL 目标，时间维同样由链式法则得到 sum；但直接反传固定前缀 KL 还涉及第 7 节的 semi-gradient 边界。

### 2.9 TOP-D 公开配方的回答平均与回报归一化

固定官方 commit `9a2716cba22cab508825ad3be3f78327bb8139e5` 的共同启动脚本显式设置 `actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean`。同版本 `agg_loss` 先对一条回答的有效 token loss 求均值，再以全局回答数归一化，并补偿数据并行的梯度平均。[^topd-common][^topd-algos] 若第 $i$ 条回答有 $L_i$ 个有效位置，逐位置 surrogate 为 $\ell_{it}$，其非空回答上的全局聚合为

$$
\mathcal L_{\mathrm{TOPD,agg}}
=\frac1B\sum_{i=1}^{B}\frac1{L_i}\sum_t m_{it}\ell_{it}.
$$

这与本机 full-RKL 的全局 token 平均具有不同的回答权重。相同平均逐 token loss 下，一条 16k 回答和一条 2k 回答在 token mean 中的总系数比为 8:1，在上述回答平均中为 1:1；实际梯度还取决于各位置的方向和幅度。TOP-D 这里的 $\ell_{it}$ 是 sampled-token 更新项，不能将它与本机条件 KL 当作同一个逐位置目标。

同版本完整 TOP-D 还先构造“当前即时 reward＋后续有效 token reward 的平均”，再对同一 prompt 的全部回答有效 token 做均值／标准差归一化。[^topd-algos] 未来回报的平均、prompt 组内标准化和外层回答平均分别改变信用分配、信号尺度与回答权重；仅替换外层 aggregate 不等于复现完整 TOP-D。

该公开配置能确定复现时的聚合选择，尚不能将 TOP-D 的任务收益归因于回答平均。此处没有保持 full-RKL、数据、rollout 和更新预算不变的单变量 aggregate 对照，也没有论文每次运行的 resolved config。[版本、来源与哈希核对](../../experiments/learning-signal/data/e55-e57-20261101-2259/topd-comparison.json)及[本地算法与预算对照](../../experiments/learning-signal/weighted-targets.md)保留了核验对象与实验边界。

## 3. 本机 full-RKL 的实际平均方式

核验仓库 HEAD：`fe3005a13464b7a80a36f49ae24c479c34cbbc7e`。下面五个关键源码文件在核验时没有工作树 diff；仓库其他文件存在既有改动。One-Shot 副本 HEAD 为 `6b6b1782cc3f04749dd077de19353f839f6bc048`，MD-OPD 副本 HEAD 为 `eb7deb07908676598a8f315696f7fde5795b9c79`。这些版本标识只说明代码审阅对象，不证明某次历史运行使用了同一版本或配置。

| 层次           | 当前实现                                                                                          | 核验源码                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| 选取预测位置   | 按 `loss_mask` 选择对应的 response 预测前缀                                                       | [padding.py](../../../../../verl/workers/utils/padding.py) 的 `prediction_positions` |
| 词表与位置累加 | full-vocab 内核计算各位置 KL；`PrecomputedKL` 将 `kl.sum()` 累加为 `total`                        | [full_vocab.py](../../../../../verl/trainer/distillation/full_vocab.py)              |
| 配置约束       | full-RKL 分支要求 `config.loss_agg_mode == "token-mean"`，否则报错                                | [losses.py](../../../../../verl/trainer/distillation/losses.py)                      |
| 全局归一化     | `total * dp_size / batch_num_tokens`                                                              | [loss_context.py](../../../../../verl/workers/utils/loss_context.py)                 |
| 分母与梯度累积 | mini-batch 的 `loss_mask.sum()` 经 DP all-reduce 后写入分母，再拆 micro-batch；各次 backward 累加 | [transformer_impl.py](../../../../../verl/workers/engine/fsdp/transformer_impl.py)   |

因此这条 FSDP full-RKL 路径最终实现：

$$
\mathcal L_{\mathrm{current}}
=\frac{\sum_{i,t}m_{it}D_{\mathrm{KL}}(p_{it}\Vert q_{it})}{\sum_{i,t}m_{it}}.
$$

`full_kl_count` 是当前调用选中位置的数量，用于指标统计；反传分母来自全局 `batch_num_tokens`。不要将这两个 count 混用。当前 launcher 也显式设置 `actor_rollout_ref.actor.loss_agg_mode=token-mean`，见 [_run_single_teacher_fsdp2.sh](../../../launchers/_run_single_teacher_fsdp2.sh)。

这条 full-RKL 路径是本机扩展。上游 verl 的 OPD 文档目前列出 teacher-top-$k$ FKL 与 sampled RKL 估计器，不能把本机 `full_reverse_kl` 配置直接当成所有上游版本都支持的接口。上游的通用 `agg_loss` 本身支持这里讨论的多种聚合。[^verl-opd][^verl-agg]

## 4. 多卡与 micro-batch 应怎样归一化

令 DP 大小为 $D$，一个 optimizer mini-batch 被分到 rank $r$ 的 micro-batch $a$。设该片段有效 KL 总和为 $S_{ra}$，整个 mini-batch 的有效 token 数为 $N$。若后端对 DP 梯度取平均，每片反传的 loss 应为

$$
\ell_{ra}=\frac{D}{N}S_{ra}.
$$

累加 micro-batch 并执行 DP 平均后，梯度为

$$
\frac1D\sum_r\sum_a\nabla\ell_{ra}
=\frac1N\sum_{r,a}\nabla S_{ra}.
$$

这正是全局 `token-mean` 的梯度。本机 FSDP 路径采用这个合同：共享全局分母，各片 backward 直接相加，没有再除以 micro-batch 数。若外部训练封装还自动除以梯度累积次数，需要重新配平，不能叠加两套归一化。

对 MOPD 的回答平均，应改为

$$
\ell_{ra}^{\mathrm{seq}}
=\frac{D}{B}\sum_{i\in(r,a)}\frac{S_i}{L_i}.
$$

这里的 $B$ 是一次 optimizer 更新涵盖的全局回答数，应包含 rollout 的重复回答，而不是单卡 micro-batch 数。当前 trainer 会把 `ppo_mini_batch_size` 乘以 `rollout.n` 作为全局 batch 信息，见 [ray_trainer.py](../../../../../verl/trainer/ppo/ray_trainer.py)。若存在完全被 mask 掉的回答，必须明确是保留它作为零贡献样本，还是从 $B$ 中剔除；不能只修改分子而让分母语义不明。

### 4.1 local mean 的平均会改变权重

前面的两条回答若各放一张卡，直接平均两卡的 local token mean 得到 $(1+3)/2=2$；全局 token mean 应为 $2.6$。只有各卡有效 token 数相同，这两种运算才等价。micro-batch 的问题相同：不能先各自除以局部 token 数，再无权平均。

日志同样应保存 sum 与 count 后求比值。对跨 batch 的 token 平均指标应计算 $\sum_j S_j/\sum_j N_j$；对回答平均指标应累计 $\sum_i S_i/L_i$ 与回答数。按优化步等权平均的 loss 是另一种时间统计，应明确命名。

以上 DP 推导适用于梯度按 DP 平均、micro-batch 梯度直接累加的合同。Megatron 的 pipeline/context-parallel 调度可能另有 scaling；本机 `agg_loss` 的 docstring 也提示 Megatron 需要对应调度缩放。不能把 FSDP 的缩放代码无条件复制到其他后端。

## 5. PyTorch 的 mean 与 batchmean

PyTorch `F.kl_div(input, target)` 的方向是 $\mathrm{KL}(\text{target}\Vert\exp(\text{input}))$，`mean` 除以输出元素数，`batchmean` 只除以第一个维度。官方文档也明确指出 `mean` 不等于通常的 KL 数值。[^pytorch]

假定输入形状为 `[B, T, V]` 且没有 padding：

| 写法                                                        | 结果                                   |
| ----------------------------------------------------------- | -------------------------------------- |
| `F.kl_div(log_q, log_p, log_target=True, reduction="mean")` | RKL 总和除以 $BTV$，多除了一次词表大小 |
| 同一输入，`reduction="batchmean"`                           | 总和除以 $B$，是 `seq-mean-token-sum`  |
| 先选出有效位置，变成 `[N, V]`，再用 `batchmean`             | 总和除以 $N$，是 `token-mean`          |
| `reduction="none"` 后词表求和，再显式按序列长度归一化       | 可表达 MOPD 的 `seq-mean-token-mean`   |

若采用 `kl_div` 写 RKL，student 的 log-probability 在 `target` 位置，必须保留它的梯度；冻结的是 teacher。为避免 API 方向混淆，可直接写定义：

```python
import torch.nn.functional as F


def full_rkl_per_token(student_logits, teacher_logits):
    log_p = F.log_softmax(student_logits.float(), dim=-1)
    log_q = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)


def aggregate_single_batch(token_kl, response_mask, mode):
    # 单进程、完整 optimizer batch 的参考式，不包含分布式 scaling。
    mask = response_mask.to(token_kl.dtype)
    lengths = mask.sum(dim=-1)
    sums = (token_kl * mask).sum(dim=-1)
    if mode == "token-mean":
        return sums.sum() / lengths.sum().clamp_min(1)
    if mode == "seq-mean-token-mean":
        valid = lengths > 0
        return (sums / lengths.clamp_min(1)).sum() / valid.sum().clamp_min(1)
    raise ValueError(mode)
```

该参考式假定 student 与 teacher 的词表／token ID 已对齐，logits 有限，并明确把空回答从回答分母中剔除。训练系统可以对全局零有效 token 直接报错；本机 FSDP 正是如此。`clamp_min` 仅使参考式在空输入下定义为零，不代表应该训练空 batch。

通常应统计有效 response 预测位置，排除 prompt、padding 和 EOS 后位置；是否计入 EOS 本身由实际 response mask 决定。工具返回文本、IS rejection 或其他筛选会引入新的掩码语义，应分别说明“训练哪些位置”和“用什么长度归一化”。原始论文简单的 $\lvert y\rvert$ 与复杂 agent mask 的计数不自动相同。

## 6. 将本机 full-RKL 改成 MOPD 平均所需的条件

若只保留 `full_kl_sum=\sum_iS_i`，然后把分母换成 $B$，得到的是 `seq-mean-token-sum`。原来各回答的 $L_i$ 已丢失，无法恢复 $\sum_i S_i/L_i$。

正确的加权方式是为回答 $i$ 的每个有效位置设置 $c_{it}=1/L_i$，先累加 $\sum_{it}m_{it}c_{it}k_{it}$，再乘 $D/B$。对 student logit $z_{itj}$，词表内 KL 的导数可直接推导：

$$
\begin{aligned}
\frac{\partial k_{it}}{\partial z_{itj}}
&=\sum_v p_{it}(v)(\mathbf1_{v=j}-p_{it}(j))
   \left(\log\frac{p_{it}(v)}{q_{it}(v)}+1\right)\\
&=p_{it}(j)\left(\log\frac{p_{it}(j)}{q_{it}(j)}-k_{it}\right).
\end{aligned}
$$

要实现回答平均，这个导数还需乘 $D/(BL_i)$。对当前 `precompute` 路径，这要求在累加 hidden-state 梯度和 LM-head 梯度前应用每行权重。诊断用 `token_kl` 被标记为不可微，事后拿它做回答平均只能改变日志，不能实现加权训练。

这是一项实现建议，尚未修改或验证新增分支。仅把配置改成 `seq-mean-token-mean`，当前 full-RKL 会被显式校验拒绝。

## 7. 目标函数与反传范围

full-RKL 的“精确”指给定一个前缀，词表内条件 KL 的精确计算。它不自动表示对整个 rollout 分布求了完整梯度。对一般目标

$$
J(\theta)=\mathbb E_{y\sim\pi_\theta}F_\theta(y),
$$

完整导数在可交换微分与求和等常规条件下是

$$
\nabla J
=\mathbb E_{y\sim\pi_\theta}
\left[\nabla F_\theta(y)+F_\theta(y)\nabla\log\pi_\theta(y)\right].
$$

将采样出的 rollout 固定后直接反传 full-RKL，只计算第一项。GKD 明确采用不对采样分布反传的训练方式；本机 full-RKL 也应在这个范围内理解。[^gkd] MOPD 的 sampled PG 式用于说明其 surrogate 的归一化，不能仅凭“full-vocabulary”就声称已经涵盖所有未来状态影响。

同理，单个 sampled log-ratio 的数值是给定前缀下 KL 的无偏估计，不代表把这个 log-ratio 直接 `.mean().backward()` 就有正确的蒸馏梯度。teacher 项被冻结时，该写法对 student 的导数只剩 $\nabla\log p_\theta(y_t)$。sampled PG 需要 detached teacher/student gap 作为系数，full-RKL 则直接对所有词表概率求导。变长回答的随机长度权重还可能与采样 token 相关，不能省略条件就声称两个完整训练更新完全等价。

## 8. 配置选择与验证边界

- 复现 MOPD、MiMo 的回答权重定义或 TOP-D 的上述公开配方时，选择 `seq-mean-token-mean`；将本机 full-RKL 改成该方式需使内核支持每个回答的长度权重。这个复现要求不构成一般 OPD 优选该方式的实验证据。
- 延续本机已有 full-RKL 实验的训练定义：保留全局 `token-mean`；与 sampled、MDPO 等方法比较时，统一外层聚合，避免同时改变算法和样本权重。
- 研究完整序列 KL 或累计代价：先定义状态分布和是否对采样分布反传，再决定 `seq-mean-token-sum`；不要把条件 KL 累计量与完整轨迹梯度混为一谈。
- 报告时同时记录 token 平均、回答平均、累计量、长度分布及各域的回答数／token 数，可辨别拟合变化与权重变化。新增统计不代表已有日志能够事后完整恢复这些量。

配套的 [CPU 数值核验](data/full-rkl-mean-20261031/verification.json) 对照解析例子、PyTorch reduction、全局 DP／micro-batch 梯度、回答权重与 logit 梯度；核验脚本为 [verify_mean.py](data/full-rkl-mean-20261031/verify_mean.py)。它验证小张量上的数学与聚合，不替代真实分布式通信、训练性能或任务成绩实验。

文献检索的六篇定向结果完整保存在 [paper-search-focused.md](sources/full-rkl-mean-20261031/paper-search-focused.md)，初始宽检索完整保存在 [paper-search-initial.md](sources/full-rkl-mean-20261031/paper-search-initial.md)。原始检索报告保留工具输出，不将其自动推荐顺序作为本文证据强弱排序。

OPD 归一化消融的 30 篇候选检索完整结果为 [paper-search-ablation.md](sources/full-rkl-mean-20261031/paper-search-ablation.md)，七篇定向论文的元数据核验为 [paper-search-ablation-focused.md](sources/full-rkl-mean-20261031/paper-search-ablation-focused.md)。候选报告包含检索误召回，是否构成相关消融以第 2.4–2.5 节的原文核验为准。

RL 长度聚合的四篇定向检索完整结果为 [paper-search-rl-length.md](sources/full-rkl-mean-20261031/paper-search-rl-length.md)。第 2.6–2.8 节的两个小型反例采用有限轨迹精确枚举，不依赖 Monte Carlo 误差。

### 参考文献

[^mopd]: Ma et al. MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training. arXiv:2606.30406v1, 2026. [§3.2，Eq. (1)–(5)](https://arxiv.org/html/2606.30406v1#S3.SS2)。

[^mimo]: Xiao et al. MiMo-V2-Flash Technical Report. arXiv:2601.02780v2, 2026. [§4.4，Eq. (7)–(8)](https://arxiv.org/html/2601.02780v2#S4.SS4)。

[^gkd]: Agarwal et al. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. ICLR 2024. [Eq. (2),§3.1, Algorithm 1](https://arxiv.org/html/2306.13649v3)。

[^rethinking]: Li et al. Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe. arXiv:2604.13016v2, 2026. [§2.2，Eq. (4)](https://arxiv.org/html/2604.13016v2#S2.SS2)。

[^oneshot]: Fu et al. Rethinking On-Policy Distillation of Large Language Models II: One Training Example. arXiv:2609.04172v1, 2026. [Table 3, Table 8](https://arxiv.org/html/2609.04172v1)。

[^minillm]: Gu et al. MiniLLM: On-Policy Distillation of Large Language Models. arXiv:2306.08543v6, 2026 修订；初版为 2023，ICLR 2024 的题名为 MiniLLM: Knowledge Distillation of Large Language Models. [§2.2，Length Normalization](https://arxiv.org/html/2306.08543v6)。

[^topd-common]: xie-lab-ml. TOP-D 共同启动配方，2026。固定 commit `9a2716cba22cab508825ad3be3f78327bb8139e5`，[显式聚合配置](https://github.com/xie-lab-ml/TOP-D/blob/9a2716cba22cab508825ad3be3f78327bb8139e5/recipe/opd/run_qwen3_1p7b_base_qwen3_teacher_dapo17k_common.sh)，2026-11-03 核验。

[^topd-algos]: xie-lab-ml. TOP-D `core_algos.py`，2026。同一固定 commit，[`agg_loss`、future return 与 prompt 组内标准化](https://github.com/xie-lab-ml/TOP-D/blob/9a2716cba22cab508825ad3be3f78327bb8139e5/verl/verl/trainer/ppo/core_algos.py)，2026-11-03 核验。

[^verl-agg]: verl contributors. `agg_loss`. [上游源码](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/core_algos.py#L1140)，2026-10-31 核验，main 为可变链接。

[^verl-opd]: verl contributors. On-Policy Distillation. [官方文档](https://verl.readthedocs.io/en/latest/algo/opd.html)，2026-10-31 核验，latest 为可变链接。

[^oneshot-common]: Thinking-Space. One-Shot-OPD `recipe/_common.sh`. [公开默认参数](https://github.com/Thinking-Space/One-Shot-OPD/blob/main/recipe/_common.sh#L65)，2026-10-31 核验。

[^oneshot-mopd]: Thinking-Space. One-Shot-OPD `recipe/mopd.sh`. [公开 MOPD launcher](https://github.com/Thinking-Space/One-Shot-OPD/blob/main/recipe/mopd.sh)，2026-10-31 核验。

[^pytorch]: PyTorch contributors. `torch.nn.functional.kl_div`. [官方 API 文档](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.kl_div.html)，2026-10-31 核验。

[^turnopd]: Zhou et al. TurnOPD: Making On-Policy Distillation Turn-Aware for Efficient Long-Horizon Agent Training. arXiv:2607.05804v2, 2026. [§7.2，Table 6，附录 D](https://arxiv.org/html/2607.05804v2)。

[^renio]: Lin et al. ReNIO: Reweighting Negative Trajectory Importance for LLM On-Policy Distillation. arXiv:2606.23104v1, 2026. [§3.4、§4.4，Table 4](https://arxiv.org/html/2606.23104v1)。

[^dapo]: Yu et al. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476v2, 2025. [§3.3、§4.2，Table 1](https://arxiv.org/html/2503.14476v2)。

[^drgrpo]: Liu et al. Understanding R1-Zero-Like Training: A Critical Perspective. arXiv:2503.20783v2, 2025. [§3、附录 C，Figure 8](https://arxiv.org/html/2503.20783v2)。

[^sure]: A Token-Level Analysis of Sampled-Token Reverse-KL On-Policy Distillation. arXiv:2608.25643, 2026. [附录 B.1、C.6，Table 9](https://arxiv.org/html/2608.25643)，2026-10-31 核验。

[^dynamics]: On-Policy or Off-Policy Learning? A Systematic Study of Distillation Dynamics. arXiv:2609.35259, 2026. [附录 C.5、D.3](https://arxiv.org/html/2609.35259)，2026-10-31 核验。

[^gndpo]: Hao et al. Stabilizing On-Policy Distillation for MLLM Reasoning with Global Normalization. arXiv:2606.09091v1, 2026. [§3.2](https://arxiv.org/html/2606.09091v1)。

[^iwopd]: Xie et al. On the Position Bias of On-Policy Distillation. arXiv:2606.22600v3, 2026. [§4–5](https://arxiv.org/html/2606.22600v3)。

[^thinking]: Kaur et al. Rethinking On-Policy Self-Distillation for Thinking Models. arXiv:2607.05184, 2026. [§4.3、附录 E](https://arxiv.org/html/2607.05184)，2026-10-31 核验。

[^vlnorm]: He et al. VL Norm: Rethink Loss Aggregation in RLVR. arXiv:2509.07558v2, 2025. [§2–4，附录 A.4–A.5](https://arxiv.org/html/2509.07558v2)。

[^lambda-grpo]: Wang et al. $\lambda$-GRPO: Unifying the GRPO Frameworks with Learnable Token Preferences. arXiv:2510.06870v2, 2025. [§2–4](https://arxiv.org/html/2510.06870v2)。
