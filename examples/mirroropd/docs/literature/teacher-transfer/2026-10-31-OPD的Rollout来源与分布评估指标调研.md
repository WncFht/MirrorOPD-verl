# OPD 的 rollout 来源与分布评估指标：逐篇调研

项目当前执行设计见[四域单域训练与 RKL 评估协议](../../experiments/teacher-transfer/protocols/online-four-domain.md)。本阶段仅采用 RKL；本文的其他指标与预算比较属于文献证据或早期候选。

整理于 2026-10-31，文献核验截至 2026-10-30。本页围绕三个问题组织证据：训练是否需要持续刷新 student rollout；应在哪种前缀上测 KL；分布拟合与任务能力如何分别评估。核心材料为 12 篇论文和 1 篇研究机构技术文章；后文明确区分论文观察、理论适用范围与本项目建议。配套的[指标与 rollout 协议提案](../../archive/experiments/teacher-transfer/2026-10-31-刷新与读出早期候选.md)给出可执行的评估定义与实验矩阵，状态为待评审，未启动新训练。

## 1. 范围与阅读顺序

arXiv／OpenAlex 检索及精确 ID 查询共返回 66 条去重记录，两个检索来源均无诊断错误，10 个指定 arXiv ID 全部解析成功。[完整原始检索](sources/rollout-metrics-20261030/完整原始检索.md)包含弱相关条目，不能把 66 条都算成已精读论文，也不代表穷尽该领域。正文选取与本项目问题直接相关的材料，并补充核验 DAgger 的会议原文、Teacher Hacking 的正文与会议元数据、Thinking Machines 的官方技术文章。版本、章节和选取理由见[核验清单](sources/rollout-metrics-20261030/reviewed-sources.json)。

建议先读 GKD 建立术语，再对照 Semi-OPD 和受控 distillation dynamics，随后读 Reverse-KL 理论与 Teacher Hacking，最后结合 Rethinking OPD／Every Coin 解释我们的教师谱系。本文不重新列 benchmark 采购式清单；领域选择由[评测领域调研](2026-10-30-OPD评测领域与泛化迁移基准调研.md)维护，已有论文预算与配对数字由[蒸馏边界论文对照](2026-10-28-蒸馏边界实验的论文对照与预算依据.md)维护。

| 材料                              | 本页核对版本／类型               | 与本项目最直接的关系                                    |
| --------------------------------- | -------------------------------- | ------------------------------------------------------- |
| DAgger                            | AISTATS 2011 定稿                | 状态访问分布随策略变化；专家标签与行为策略分开          |
| GKD                               | ICLR 2024；arXiv v3              | rollout 来源与 token divergence 可以独立选择            |
| MiniLLM                           | ICLR 2024 工作；核对后续 v6      | 序列 RKL 的策略梯度、混合采样与曝光偏差诊断             |
| Thinking Machines OPD             | 2025 技术文章，访问时页面        | sampled-token 实现与 current-student rollout 的工程语义 |
| Teacher Hacking                   | ICML 2025；方法细节核对 arXiv v1 | teacher 距离与真实目标可以分离                          |
| Rethinking OPD                    | 2026 预印本 v2                   | overlap、熵及前缀深度诊断                               |
| Every Coin Has Two Sides          | 2026 预印本 v3                   | 同源／异源、同域／跨域的观测分离                        |
| Semi-OPD                          | 2026 预印本 v1                   | 固定初始 student 与在线 student 的直接比较              |
| On-Policy or Off-Policy Learning? | 2026 预印本 v1                   | full-vocab、rollout 来源、KL 方向及学习率的受控比较     |
| Reverse-KL Policy Distillation    | 2026 预印本 v2                   | 历史前缀经验拟合与真实策略风险的差别                    |
| Transfer-Stratified OPD           | 2026 预印本 v1                   | teacher 改进、student 获取量与逐题分层                  |
| Scaling Properties                | 2026 预印本 v1                   | 策略位移、早期迁移与后期任务峰值                        |
| TOP-D                             | 2026 预印本 v2                   | 行为策略、更新后策略与数据复用的区分                    |

论文中的 on-policy 通常指相对于生成当批轨迹的学生；在同批上多次更新后，轨迹对最新参数已经有滞后。teacher 给 student 前缀打分时常使用 teacher forcing；这不表示前缀由 teacher 生成。下面使用 RKL 表示 KL(student∥teacher)，FKL 表示 KL(teacher∥student)，不沿用各论文可能相反的符号排列。

## 2. 基础方法与状态分布

### 2.1 DAgger: A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning

**研究问题与做法。** 序列决策中，学习者动作会改变后续状态。在专家访问的状态上预测正确，并不能保证在学习者自己访问的状态上也正确。DAgger 迭代执行当前策略（早期可混入专家），让专家为访问状态提供动作标签，聚合数据后重新学习。它不是只生成一份专家演示后永久训练。[^dagger]

**评估与结果。** 原文用模仿学习任务及序列标注检验实际运行表现，同时给出依赖无遗憾学习与额外 reduction 假设的性能保证。这里的监督是专家动作／代理损失，并非语言模型 full-vocab RKL。

**对应启发与边界。** 它解释了为什么需要检查 current-student 访问的状态，也提示历史状态聚合可以成为控制组。它没有证明每一步 Adam 都必须重新 rollout，更没有给出我们 reuse16 的最佳刷新频率。不能把 DAgger 的条件性保证移植成任意 LLM 蒸馏目标的保证。

### 2.2 GKD：On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes

**研究问题与做法。** GKD 将训练轨迹来源与比较条件分布的 divergence 分开：轨迹可来自固定输出数据、当前学生，或两者混合；在同一前缀上比较 teacher/student 的完整 token 分布。论文比较 FKL、RKL 和广义 JSD，算法以 student-data fraction 控制来源。采样过程停止梯度，训练采用逐响应长度归一化的 token 损失。[^gkd]

**评估与结果。** 在摘要、翻译、算术推理和指令蒸馏上评估任务结果；优势随任务与 divergence 改变，并非单一 RKL 配方在所有任务最优。方法强调处理学生自生成前缀上的分布不匹配。

**对应启发与边界。** 我们应将“refresh 与否”和“RKL／FKL”作为不同因素。GKD 的 token 条件目标可借鉴，但停止采样梯度的更新不能直接称为完整序列 KL 梯度；它也没有检验本项目同构远源 teacher 在固定银行上是否能充分拟合。

### 2.3 MiniLLM：序列 RKL 与优化估计器

**研究问题与做法。** MiniLLM 从序列级 RKL 出发，采用策略梯度思路，并引入单步分解、teacher-mixed sampling、长度归一化等稳定化设计。它与直接在缓存前缀上反传 full-vocab 条件 KL 的训练循环不同。本页读取 v6，不能把其中全部细节默认为 ICLR 初版的原样内容。[^minillm]

**评估与结果。** 论文报告指令输出质量、自动指标与人工偏好，并分析曝光偏差、校准和生成多样性；曝光偏差诊断比较自由生成状态与参考状态上的误差累积。

**对应启发与边界。** 应同时测“相同前缀上能否匹配”和“自己生成时是否偏离”。方法效果包含采样与优化设计的共同作用，不能把改进全部归给 refresh，或把它的 sampled／policy-gradient 结果直接视作当前 full-vocab 更新的预期。

### 2.4 Thinking Machines：On-Policy Distillation

**研究问题与做法。** 官方技术文章用当前学生生成轨迹，由 teacher 评分实际采样 token，再以 student／teacher log-prob 差构造训练信号。文中为简化采用零折扣，强调即时 token 信号，不使用完整未来回报。[^tml]

**评估与结果。** 案例涉及数学推理与个性化行为恢复，主要报告任务或行为效果，并展示重复使用 prompt、但重新生成多条轨迹的实验。

**对应启发与边界。** “同一道题再生成”与“同一条轨迹反复反传”必须分别记账。该文是实践报告，不是刷新频率的受控普遍结论；即时 sampled-token 更新也不能不经验证就等同于我们的固定状态解析梯度。本文只引用其方法语义，不以案例成绩预测本项目结果。

## 3. 刷新是否更好：直接对照与反例

### 3.1 Semi-OPD：When Do We Need On-Policy Distillation? Distilling on Offline Student Rollouts Is Often Better

**研究问题与做法。** 固定初始 student 生成的离线回答，与每步由更新后 student 生成的在线回答比较。主对照共享 prompt 序列、teacher、优化设置与 sampled-token RKL 目标。初始 top-64 overlap 用来分析配对差异。[^semi]

**评估与结果。** 核心比较覆盖 17 组配对，任务指标为四项数学评测的 avg@32；论文报告 14 组由 Semi-OPD 占优，同时给末步差和归一化曲线面积差。两种差值不能互换。作者还检查长度、熵与 overlap 等动态。

**对应启发与边界。** 固定 S0 应保留为强对照，不能预设 refresh 必胜。但该研究没有证明固定 S0 的 full-vocab 风险一定更低，也没有普遍证明低 overlap 是因果障碍。其部分效率比较使用不同 rollout cap，不能把最大速度比全部归因于冻结轨迹。

### 3.2 On-Policy or Off-Policy Learning? A Systematic Study of Distillation Dynamics

**研究问题与做法。** 将 student／teacher rollout、full-vocab FKL／RKL、两个学习率交叉比较；主实验全参数训练 150 次更新，每配置三个 seed，覆盖 Llama／Qwen 与不同推理任务。[^dynamics]

**评估与结果。** 同时测训练域任务成绩、跨域保持、输出覆盖和参数更新稀疏性。作者报告 KL 方向更明显地影响任务成绩与覆盖，学习率影响遗忘与更新稀疏性；RKL 对 rollout 来源更敏感，FKL 在其条件下较稳健。更难 Countdown 变体上仍观察到 on-policy 收益。

**对应启发与边界。** 这是对我们 full-vocab 设置最直接的设计参照：先独立控制来源与目标。附录 C.5 明确区分序列 KL 的链式恒等式与 semi-gradient；均按实际长度归一化又会改变目标。结论仍受模型、任务、学习率与预算限制，不能据此删除全部 on-policy 分支。

### 3.3 On Teacher Hacking in Language Model Distillation

**研究问题与做法。** 构造 oracle→teacher→student，使研究者能同时测 student 对 teacher 的代理距离和对 oracle 的目标距离；比较离线固定数据与在线生成，并改变数据多样性。会议发表信息按 ICML 2025 PMLR 记录，以下方法说明按 arXiv v1。[^hacking]

**评估与结果。** 论文观察到离线蒸馏中，teacher 距离继续改善而 oracle 距离恶化；在线数据生成及更丰富的数据可以缓解其观察到的现象。它将 teacher hacking 与通常的训练／验证过拟合区分开。

**对应启发与边界。** 若研究目标是忠实复制 teacher，teacher 距离仍是直接指标；若目标是正确性或人类偏好，就需要独立任务测量。不能从该文推出我们冻结 S0 的每条实验都发生了 hacking，也不能把“增加轨迹多样性”和“跟随当前参数刷新”视作同一个处理。

### 3.4 Reverse-KL Policy Distillation: Statistical Guarantees and Limits

**研究问题与做法。** 在有限策略类、有限时域 MDP 中，比较在线 student／离线 teacher 轨迹以及 full-distribution／sampled-action 两种反馈。目标是真实轨迹 RKL；分析依赖明确的矩条件或轨迹似然比条件。[^rk-theory]

**结果。** 论文区分可实现性、近似误差与估计误差，给出不同反馈条件的统计界；还构造反例，说明仅按历史访问状态上的经验损失选择候选策略，可能无法获得好的真实轨迹风险。其修正涉及访问分布、轨迹损失截断和策略空间学习。

**对应启发与边界。** 当前策略的风险应在当前策略自己的轨迹上验证，历史银行的低损失不能替代它。论文的有限类算法、oracle 优化和最坏情形反例，不是 Transformer＋Adam 的收敛定理，也没有证明某个同构 LLM 配对不可蒸馏。理论中的 full feedback 与我们的 full-vocab 监督相关，但“信息更充分”不自动保证实际优化到位。

## 4. 与教师谱系、迁移和指标解释直接相关的工作

### 4.1 Rethinking On-Policy Distillation: Phenomenology, Mechanism, and Recipe

**研究问题与做法。** 比较配对、后训练来源与输入模式，分析 student 访问状态上的 top-k overlap、交集 token advantage、熵差，以及不同前缀深度的监督。默认实验与 top-k 监督有关，另比较 sampled-token 和 full-vocab 方案。[^rethinking]

**评估与结果。** 报告任务成绩与上述动态；成功配对往往伴随高概率 token 更一致。其 teacher-aligned prompt 实验也出现 overlap 数量较低、交集概率质量却更高的情况，说明集合计数并不能完整表达分布接近。

**对应启发与边界。** overlap 与熵适合解释 KL 的变化，不能替代 KL 或作为成功阈值。作者关于局部几何、thinking pattern 的解释不能直接当作容量下界。我们应使用共享输入、完整支持指标和独立状态银行检验这些机制；更多配对数字沿用已有预算调研，不在这里重复。

### 4.2 Every Coin Has Two Sides: On the Dual Nature of Generalization in OPD

**研究问题与做法。** 研究同源／异源师生在训练难度、语言、推理跨度、跨领域及多教师条件下的差异；训练采用 sampled-token 风格 OPD，另跟踪 top-16 overlap。正文核验结合 v3 原文与本地 TeX。[^coin]

**评估与结果。** 作者观察到其同源配对有更广的任务迁移，异源配对更偏向训练分布拟合；多教师中还观察到不同领域之间的干扰。其“整体 policy 对齐”解释由任务迁移与局部 overlap 等证据支持。

**对应启发与边界。** 我们需要同时保留训练与跨领域留出，检查局部拟合能否传播。top-16 overlap 和有限任务成绩不能单独证明完整序列分布相等；同尺寸、同祖先、相同参数化也不是同一分类变量。当前研究正是在补充 full-vocab、多来源 state 的证据。

### 4.3 Transfer-Stratified OPD for RL-Improved Reasoning Teachers

**研究问题与做法。** 对照 teacher 在 GRPO 前后的版本，按逐题 teacher／student 能力差异分层，再结合筛题、FKL／RKL routing、token gating 和 entropy brake。[^ts]

**评估与结果。** 使用任务成绩及作者定义的 transfer-gap 指标，区分 student 尚未解决的题与偶尔能解决但不稳定的题。研究表明 teacher 的额外 RL 收益未必全部转为 student 收益，联合方法能在其实验中改善部分转移。

**对应启发与边界。** 逐题分布和阶段配对有价值；teacher-before／after 两个终点的 student KL 降幅之差，不等于“吸收的 RL 内容”。该方法同时改变多项训练条件，不能将其总收益单独归给 FKL、refresh 或筛题。我们的主拟合矩阵暂不以 acc 筛 teacher。

### 4.4 Scaling Properties of Same-Family On-Policy Distillation

完整的图解、局部几何与局限分析见[单篇精读笔记](2026-10-31-同家族OPD缩放规律精读与实验启发.md)，其中图号统一按 PDF 核验。

**研究问题与做法。** 在 Qwen2.5 同家族、五个规模的 25 个 Vanilla-OPD 配对中研究大小方向与迁移动态，并对早期有用迁移区间建立经验模型。[^scaling]

**评估与结果。** 论文将 held-out gold score 与相对学生初始化的平方根 token KL 联系起来，另分析任务峰值和弱教强结果。完整网格中的曲线主要是单次运行描述，峰值及后期行为不能由早期近似关系直接外推。

**对应启发与边界。** 相对 S0 的 KL 应叫策略位移，不是 teacher 拟合残差。学生可能一边远离 S0，一边接近 teacher；两条 KL 的上升／下降没有矛盾。该研究不能提供本项目跨家族、不同输入协议的普遍幂律预测，或用 teacher 分数单独决定配对价值。

### 4.5 Trust Region Policy Distillation（TOP-D）

**研究问题与做法。** 用 proximal teacher 改写监督信号，区分生成数据的行为策略与更新后的策略，并通过 trust-region 风格目标复用轨迹。其估计器、return 与归一化机制不同于直接 full-vocab 条件 KL。[^topd]

**评估与结果。** 在数学任务上报告准确率、训练稳定性及效率，并提供依赖特定条件的理论分析。它说明刷新频率、旧轨迹复用和目标修正需要共同交代。

**对应启发与边界。** 不能把论文的 global batch／不同 minibatch 数当成本项目同一批轨迹 reuse16，也不能直接把 proximal-teacher 的损失当作对原始 teacher 的拟合指标。即使以后训练采用稳定化目标，评估也应继续面对冻结的原始 teacher。

## 5. 相互冲突的结论如何放在一起

| 看似冲突                                              | 应先核对的区别                                                       | 对我们可以成立的判断                     |
| ----------------------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------- |
| GKD 强调 online，Semi-OPD 强调固定 S0                 | 模型、任务、目标估计器、轨迹覆盖和评价终点不同                       | 两条训练线都应保留；不预设全体配对的赢家 |
| 受控研究中 FKL 稳健，MiniLLM 强调 RKL                 | 条件 semi-gradient 与序列策略梯度、采样和归一化不同                  | rollout、KL 方向与梯度路径分开记录       |
| Teacher Hacking 中在线有帮助，Semi-OPD 中在线有时退步 | oracle 距离和任务准确率不是同一目标；数据多样性与 on-policyness 不同 | 不能只靠“在线／离线”标签解释机制         |
| 同源任务迁移好，同构 SFT 仍可能留有 KL 残差           | 函数可表示性、优化预算、state 覆盖与任务成功率不同                   | 有限预算平台不能直接叫容量下界           |
| 某论文 KL 上升而我们希望 KL 下降                      | 对 S0 的策略位移与对 teacher 的残差混用了参照                        | 每张图必须写明 KL 两端和 state 来源      |

DAgger 的条件性保证与 Reverse-KL 的历史状态反例不构成直接矛盾：两者的目标风险、代理损失、策略类及优化假设不同。不能只看到都使用 student 轨迹，就把一个定理用于另一个训练过程。

以上是对实验设计的综合判断，不是对这些论文做元分析后得到的效应量。不同论文的 step、数据量、生成数、采样参数和目标都不一致；本页不平均它们的收益数字。

## 6. 对本项目的建议

训练侧先保留固定初始学生轨迹，并新增按 rollout batch 刷新的学生轨迹主线；teacher rollout 训练作为后续的覆盖诊断，不与 refresh 同时引入新的 KL 方向。评估侧采用共同固定 S0 银行、每位 teacher 的冻结银行、每个 checkpoint 的 current-student 银行，分别衡量固定状态拟合、教师行为覆盖和学生实际行为。三套银行在独立留出题上建立，训练题／旧批继续单列。

主指标建议为共同固定银行上的 RKL、current-student 银行上的 RKL、teacher 银行上的 FKL，补充交叉方向、JS／TV、熵与概率质量诊断。绝对残差、相对降幅和逐题尾部分开报告；acc 作为低频独立结果检查，不作为 teacher 准入条件。冻结评估银行、动态评估银行、真实序列 KL 与解码加权条件风险的区别，见[完整协议提案](../../archive/experiments/teacher-transfer/2026-10-31-刷新与读出早期候选.md)。

“完全吸收”的数值阈值暂不定义。首先验证测量是否覆盖目标行为、数值参考是否可靠，以及训练／留出是否同步改善，再讨论残差的科学解释。

### 参考文献

[^dagger]: Ross, Gordon, Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. AISTATS 2011. [会议原文](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)。重点：Algorithm 3.1、状态分布与 no-regret 条件。

[^gkd]: Agarwal et al. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. ICLR 2024，arXiv:2306.13649v3. [正文](https://arxiv.org/html/2306.13649v3)。§2–4、Algorithm 1、§5。

[^minillm]: Gu et al. MiniLLM: On-Policy Distillation of Large Language Models. ICLR 2024 工作的后续 arXiv:2306.08543v6，2026-01-31. [核对版本](https://arxiv.org/html/2306.08543v6)。§2、§3.3、附录 B.5。

[^tml]: Kevin Lu and Thinking Machines Lab. On-Policy Distillation. 2025-10-27，访问于 2026-10-30. [官方技术文章](https://thinkingmachines.ai/blog/on-policy-distillation/)。Implementation、Loss function、Pseudocode、Discussion；网页可能继续更新。

[^hacking]: Tiapkin et al. On Teacher Hacking in Language Model Distillation. ICML 2025. [PMLR 会议记录](https://proceedings.mlr.press/v267/tiapkin25a.html)、[核对正文 v1](https://arxiv.org/html/2502.02671v1)。§2–4；正文版本与会议元数据分别标注。

[^rethinking]: Yaxuan Li et al. Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe. arXiv:2604.13016v2, 2026. [正文](https://arxiv.org/html/2604.13016v2)。§2.2–2.3、§4、§5.2、§6。

[^coin]: Zhaoyi Li et al. Every Coin Has Two Sides: On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models. arXiv:2608.16647v3, 2026. [版本入口](https://arxiv.org/abs/2608.16647v3)、[本地原文](../../experiments/teacher-transfer/transfer-mechanism/data/state-transfer-20261027/papers/2608.16647v3.tex)。§2–5、图 6、附录 B。

[^semi]: Zhao et al. When Do We Need On-Policy Distillation? Distilling on Offline Student Rollouts Is Often Better. arXiv:2610.11291v1, 2026. [正文](https://arxiv.org/html/2610.11291v1)。§2–4、图 1/2、表 3、附录 A/C。

[^dynamics]: Piskorz, Berthon, van der Schaar. On-Policy or Off-Policy Learning? A Systematic Study of Distillation Dynamics. arXiv:2609.35259v1, 2026. [正文](https://arxiv.org/html/2609.35259v1)。§3–6、附录 C.5/D。

[^rk-theory]: Liu, Wei, Zimmert. Reverse-KL Policy Distillation: Statistical Guarantees and Limits. arXiv:2610.30844v2, 2026. [正文](https://arxiv.org/html/2610.30844v2)。§2、Proposition 3.1、§3–4。

[^ts]: Transfer-Stratified On-Policy Distillation for RL-Improved Reasoning Teachers. arXiv:2610.05974v1, 2026. [正文](https://arxiv.org/html/2610.05974v1)。§3–5、附录 B/C。

[^scaling]: Bao et al. Scaling Properties of Same-Family On-Policy Distillation. arXiv:2609.32722v1, 2026. [正文](https://arxiv.org/html/2609.32722v1)。§3–6、附录 B/D/E。

[^topd]: Xie et al. Trust Region Policy Distillation. arXiv:2607.04751v2, 2026. [正文](https://arxiv.org/html/2607.04751v2)。§3、Algorithm 1、附录 D。
