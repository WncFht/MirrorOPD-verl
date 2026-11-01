# 四域单域训练与 RKL 评估协议

本页记录 2026-10-31 确定的按批刷新与四域读出基线，并明确后续运行的协议版本。既有[固定状态 0–320](fixed-state.md)独立解释；公共配置不回写历史实验。

训练和验证采用共享 GPU 池的串行阶段，允许当前 outer／评估窗口临时复用 teacher hidden，不预计算长期全题池 latent。分布指标仅 RKL，accuracy 为低频辅助读出。训练 loss 已确定使用 token-mean；报告 RKL 使用题内有效 token 均值，再按题平均。学习率 1e-6、reuse32；仅保存一个最终模型，不保存 optimizer 或周期恢复点。

工程实现与代码缺口集中在[verl 接入方案](../../../archive/engineering/evaluation/2026-10-31-四域接入设计.md)。本页定义研究预算，资源参数不改变题数、n、cap 和指标含义。

## 适用版本与运行证据

| 条件             | 实际采用与范围                                                                                                                     |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 原版线上四域协议 | 下文保留 source/history 启用、公共 cap32768 的定义与工作量；[JustRL 数学分支](../cross-domain-transfer.md)已完成 50outer/1600 更新 |
| 后续阶段对照     | [MiMo、Olmo3 与 Light-R1](../post-training-stages.md)明确关闭 source/history，公共长度按各模型协议执行；不能沿用原版工作量估算     |
| 当前执行入口     | [原生评估](../../../infra/evaluation/README.md)及源码包合同维护开关、身份和失败行为；各 run 的 resolved config 绑定真实选择        |

Source/history是可选读出；以下抽样和频率描述其启用时的语义。关闭不删除真实训练轨迹，不自动生成source银行。公共bank与任务评测的频率以各项实验声明为准。完整50outer结果只覆盖报告中的师生/训练域，不外推为其他分支完成。

## 1. 研究问题与分支关系

研究固定 teacher 能否被规定初始化的 student 拟合，以及训练来源改变后，同来源新题和其他领域上的残差如何变化。

同一 student／模式的各分支从完全相同的 initial student 权重开始，teacher 在整个分支内冻结。改变训练领域时固定师生对和公共评估题；分别训练 Math、Code、Science、IF 四条分支，不把四域混为一个训练流，也不把四域顺序续训成一条链。后续更换 teacher 时另计分支。

Thinking 与 non-thinking 独立成线，分别生成轨迹。相同线内输入模板、token 含义、EOS 和支持集条件须验收；2507 专用模板等不兼容条件单列。本文不新增模型名单，也不把所有既有 teacher 自动加入四域矩阵。

## 2. 训练数据与预算

| 项目                     | 已确认设置                                            |
| ------------------------ | ----------------------------------------------------- |
| Math                     | DAPO-Math-17K，本地清洗候选池                         |
| Code                     | agentica-org/DeepCoder-Preview-Dataset 的训练 configs |
| Science                  | MegaScience/TextbookReasoning，去除Math／CS           |
| IF                       | nvidia/Nemotron-Cascade-2-RL-data 的 IF-RL 子集       |
| 每 outer 新题            | 128 道，跨 outer 无放回                               |
| 每题训练回答             | 1 条                                                  |
| Outer 总数               | 50                                                    |
| Reuse                    | 同批 128 条回答完整复用 32 次                         |
| 每分支独立训练题         | 6,400 道                                              |
| 每分支 optimizer updates | 1,600 次                                              |
| 训练 response cap        | 7,168 token，包含 thinking 及最终回答                 |

每个 outer 开始，由该分支当时的 current student 生成 128 条回答；批内 32 次更新保持题目与回答前缀不变，每次更新重算当前 student 分布。它是按批刷新、批内复用的训练，不能称每次更新均严格 on-policy。

一次 optimizer update 覆盖全局 128 条回答。microbatch 与梯度累积用于实现该全局 batch；不能切成多个独立 optimizer step 后仍称相同 reuse 预算。四个训练域合计每师生对 4 分支、25,600 个训练题实例、6,400 次更新。1600 是计划更新数；另记实际 attempted／applied／skipped，数值异常不能靠未声明的额外更新补齐预算。

训练不按 teacher 准确率、初始 RKL 或训练效果挑题。全词表 teacher 分布是蒸馏信号；数据集原有答案或 CoT 不作为额外训练目标，也不放进 teacher 的特权输入。

## 3. 三类评估集合

| 集合             | 题数／条数                                       | 回答来源                                                              | 回答 cap | 频率                         |
| ---------------- | ------------------------------------------------ | --------------------------------------------------------------------- | -------- | ---------------------------- |
| 当前训练来源留出 | 128 道                                           | 固定 initial student rollout，每题 1 条                               | 7,168    | 0 及每 outer                 |
| 公共 benchmark   | Math／Code／Science／IF 各 128，共 512 道 RKL 题 | 从 initial student 与 teacher 的初始化 accuracy 产物，每题各保留 1 条 | 32,768   | 固定 bank：0/1/2 及每 5outer |
| 动态历史训练集合 | 总量 128 条                                      | 当时实际参与训练的 current student rollout                            | 7,168    | 每 outer，从 1 开始          |

### 3.1 当前训练来源留出

DAPO 分支只设 128 道 DAPO 来源留出；DeepCoder、TextbookReasoning 和 IF-RL 分支同理。它们均不参与该研究的梯度训练。不再设置每分支四域合计 2,048 题的来源留出，不另设锁定确认集。

同一训练来源、student 和模式的各 teacher 分支共享这 128 道题及 initial student 前缀。它们在训练开始前生成一次，后续只重新评分。当前不刷新留出上的 current student rollout，因此曲线表示固定状态上的同来源泛化。

### 3.2 公共 benchmark

| 领域    | 候选题库                                                   | 本轮固定题数 |
| ------- | ---------------------------------------------------------- | ------------ |
| Math    | 既有 MATH500、AMC23、AIME24/25/26 合计 630 题              | 128          |
| Code    | DeepCoder 配套 LCB v5 test279 题，2024-08-01 至 2025-02-01 | 128          |
| Science | GPQA-Diamond198 题                                         | 128          |
| IF      | IFEval541 题                                               | 128          |

四域题目对所有训练域保持一致。模型／模式不同可以有不同轨迹，但题目 ID 与输入协议身份必须记录。initial student 轨迹可在相同 student／模式的 teacher 分支间共享；teacher 轨迹在相同 teacher／模板／模式的训练域分支间共享。

两个轨迹银行合计 1,024 条回答。两者都计算同一方向的 RKL，即 student 对 teacher 的条件分布差距；teacher rollout 只改变前缀来源，不把指标改成 FKL。不同 teacher 的 teacher 轨迹银行不同，跨 teacher 主排序使用共享 initial student 前缀的结果。

公共验证 response cap 为 32,768；训练与本来源留出为 7,168，history 保留真实训练轨迹。同一评估域／模式下，各模型采用相同 cap，与训练领域无关。RKL 使用实际保留的整条回答，并记录有效长度和 cap；前 7,168／完整 16,384 的双时域读出暂缓。

Accuracy 使用各 benchmark 完整题集，共 1648 题，采样次数为 MATH500=1、AMC23=16、各年 AIME=16、LCB=10、GPQA=4、IFEval=1。公共 RKL 使用其中 512 题的自定义面板，每题保留 1 条回答；该抽样不改变 accuracy 的题数与 n。已准备运行的具体题目由各自 manifest 固定，初版草案中的抽样待办已由这些题表落实。

上述题库构成本协议的任务范围；IFBench、HumanEval+、MBPP+、MMLU-Pro 和 BBH 不在本次评测中。

公共银行由初始化 bench 评估产出：initial student 与 teacher 分别按各 benchmark 的 accuracy 采样协议生成回答，再对预先指定的每域 128 题保留固定 sample index 0 的原始 token。Accuracy 使用其全部回答；RKL 不要求对所有 accuracy 样本评分。样本选择不看正确率、长度、RKL 或返回先后。对两个银行的初始 RKL 都使用 initial student→teacher；teacher 生成轨迹不意味着改成 teacher 与自己比较。后续固定银行评分与 current bench 生成各自调度。来源留出 128 题仍单独准备。论文依据、n／cap 候选和实现分工见[正确率与银行生成调研](../../../literature/teacher-transfer/2026-10-31-四域正确率评测与RKL银行生成协议调研.md)。

### 3.3 真实训练轨迹

保存每批全部 128 条实际进入训练的回答 token，不用同题的 initial student 回答替换。每次更新直接记录当前批 RKL loss，不新增 inner0／1／4／8／16／32 的额外评分。loss 是该次更新之前的读数；outer 结束的历史／留出读数对应 32 次更新全部完成后的权重。

## 4. 动态历史抽样

设已经完成 $m$ 个 batch，评估容量 $K=128$，定义：

$$
q_m=\left\lfloor\frac{128}{m}\right\rfloor,\qquad r_m=128-mq_m.
$$

$r_m$ 个 batch 各分配 $q_m+1$ 条，其余各分配 $q_m$ 条。只考虑本协议 $m\le50$，每批至少两条。

| 已训练 batch 数 | 配额                   |
| --------------- | ---------------------- |
| 1               | 128                    |
| 2               | 每批 64                |
| 3               | 两批 43，一批 42       |
| 10              | 八批 13，两批 12       |
| 25              | 三批 6，二十二批 5     |
| 50              | 二十八批 3，二十二批 2 |

每个 batch 和批内每个题目分别获得固定、独立的随机优先级，与 loss 和正确率无关；以稳定 ID 打破并列。batch 优先级最小的 $r_m$ 批获得额外一条，各批保留题目优先级最小的指定条数。相同训练题序的分支共享抽样优先级，实际回答 token 仍归各自分支。

旧批配额只会不变或减少：$q_m$ 不变时 $r_m$ 减少；$q_m$ 下降时，新配额最大值不超过旧配额最小值。因此从旧集合删减、从新批补入即可。退出评估集合不删除完整训练记录。

历史汇总保持 batch 等权；批内对各题的 token 均值等权平均，形式为：

$$
\widehat R_m(\theta)=\frac1m\sum_{b=1}^{m}\frac1{n_b(m)}\sum_{i\in S_b(m)}R_\theta(x_{bi},y_{bi}^{\mathrm{train}}).
$$

所有完整训练 batch 同为 128 题，上式对应对完整历史题目等权平均。此前工程验收用批内 token-mean、再按 batch 等权，保留其历史定义；新协议需切换到题内均值再按题平均。不能让抽 3 条的 batch 比抽 2 条的 batch 多占权重。

每轮集合变化，历史曲线同时受到参数更新、新批加入和成员替换影响，标记为“动态历史训练 RKL”。它不等价于固定测试集学习曲线；固定来源留出提供后者。相邻两轮都保留的轨迹可以单列配对变化，不拿前一轮不同集合的均值作为相对降幅分母。每批终点仅 2–3 条，不做精细的逐批难度排名。

## 5. 唯一分布指标：RKL

令 $p_\theta$ 为 student 条件分布，$q$ 为冻结 teacher 条件分布。对指定前缀 $s$：

$$
r_\theta(s)=\sum_v p_\theta(v\mid s)\log\frac{p_\theta(v\mid s)}{q(v\mid s)}.
$$

使用自然对数，单位为 nat；在共同、已验收的词表语义上计算 full-vocab RKL，不改成 sampled-token 估计或 top-k 近似。原有共同支持条件 RKL 的配对继续单列，输出支持范围与重新归一化定义；不把条件 RKL 与原生全支持 RKL 混入一个总榜。兼容性元数据属于验收记录，不是新增分布指标。

对实际保留的回答 $y_i$，首先保存不依赖平均方式的累计 RKL 和有效长度 $L_i$；生成受各集合已约定的 cap 限制：

$$
C_i=\sum_{t=1}^{L_i}r_\theta(x_i,y_{i,<t}).
$$

**训练与报告采用不同的已确认权重。** 训练保持全局有效 token 均值；固定评估集合对每题先求有效 token 均值，再按题等权平均：

$$
\mathcal L_{\mathrm{train}}=\frac{\sum_i C_i}{\sum_i L_i},\qquad
R_i=\frac{C_i}{L_i},\qquad
\overline R_{\mathrm{eval}}=\frac1N\sum_{i=1}^{N}R_i.
$$

分布式训练必须全局归约 sum/count，不能直接平均不同有效长度的微批均值。公共面板在各域内按题平均后四域等权；history 使用第 4 节的批等权估计。保存每题 sum/count 以支持审计；有效长度为零时均值不定义，报告无效记录及 coverage，不填零、不静默丢弃。

EOS 后的吸收状态贡献零，达到 cap 不补造 EOS；prompt 位置不计入回答 RKL。保存 RKL sum、有效 token、cap 和终止原因，使后续可以在相同原始读出上比较聚合方式。改变训练 loss 归一化仍会改变优化过程，不能靠事后换图等价替代。

固定面板报告初始绝对 RKL、终点绝对 RKL、绝对下降量和相对降幅：

$$
\mathrm{Reduction}=1-\frac{\overline R_{\mathrm{after}}}{\overline R_{\mathrm{initial}}}.
$$

初始风险为零时相对降幅不定义，接近零时以绝对量为主。公共面板先在各域内按题等权平均，再按四域等权汇总；相对降幅由汇总后的初始／终点风险计算，不平均每题百分比。初始与终点使用同一前缀、同一时域及支持口径。

本阶段不计算 FKL、TV、JS 或熵，不为历史绘图兼容而在评分内部额外计算它们。teacher-rollout RKL 仍是固定 teacher 前缀上的条件风险，不据此命名为 teacher 到 student 的序列 KL。

## 6. 生成参数与读出时间

| 用途           | 生成协议                                   | 说明                                                                              |
| -------------- | ------------------------------------------ | --------------------------------------------------------------------------------- |
| 训练／来源留出 | T1、top-p1、top-k=-1、response cap7168     | 训练每 outer 由 current student 刷新；来源留出只生成 initial student bank         |
| 公共 benchmark | T1、top-p0.95、top-k=-1、response cap32768 | 采用 Every Coin 跨域协议作为基线；各 bench 的 n 见第 3 节，模型／模式间不临时改变 |
| RKL 评分       | T1、完整且验收过的词表支持                 | 重新 teacher-force 原始前缀；生成采样截断不应用于评分分布                         |

采样参数的原文比较见[专项调研](../../../literature/teacher-transfer/2026-10-31-四域正确率评测与RKL银行生成协议调研.md)。Template、thinking 模式、EOS／停止规则和 prompt 长度上限随输入验收冻结；不依赖未记录的模型默认值。公共 32K 是回答上限，执行预算还必须容纳 prompt。

| 任务                          | 输入                                      | Outer 读出点                                      |
| ----------------------------- | ----------------------------------------- | ------------------------------------------------- |
| 训练 loss                     | 当前批真实训练轨迹                        | 每次 optimizer update 自然记录，不增加 inner 评分 |
| 来源留出 RKL                  | 本来源 128 题的固定 initial student 轨迹  | 0 及每 outer 结束                                 |
| History RKL                   | 128 条动态历史实际训练轨迹                | 每 outer 结束，从 1 开始                          |
| 公共 initial-student bank RKL | 四域 512 题，各 1 条固定回答              | 0、1、2，以及每 5outer                            |
| 公共 teacher bank RKL         | 同一 512 题，各 1 条固定 teacher 回答     | 0、1、2，以及每 5outer                            |
| 公共 current accuracy＋RKL    | 按 bench 的 n 生成；RKL 选 512 题 sample0 | 0、25、50                                         |
| Teacher accuracy              | 初始化公共生成的完整回答集合              | 只在初始化；同身份结果可复用                      |

这是统一实施基线，不是已跑出的完整曲线。每 5outer 对应 160 次更新，25／50 对应计划 800／1600 次；执行日志另记真实成功更新数。公共 32K 性能验收若表明成本超预算，可在正式矩阵开始前统一修订频率，不按某个 teacher 成绩临时加减节点。

初始化 current 与 initial bank 在权重和生成协议一致时共用同一生成与 RKL 结果；后续二者状态分布不同，分开报告。公共 teacher bank 改变的仅是前缀来源，RKL 两侧仍是 current student 与固定 teacher。

Source heldout 测量同来源未见题；未来训练批次的传播由独立配对诊断测量。History 和 source 的前缀来源不同，解释差值时需同时标明来源。

## 7. 初始化与一轮训练

开始前冻结题序、评估题目 ID、sample0 及输入协议。Initial student 与 teacher 分别完成公共生成，accuracy 消费全部回答，bank 各保留 512 条。Initial student 另生成本来源 128 条 7K 留出。两个公共 bank 和来源 bank 的初始 RKL 都比较 initial student→teacher；teacher 基线不与自己比较。

每个 outer 使用 current student 生成 128 道新题，保存真实 token，teacher 骨干对本批只算一次，student 复用同批前缀更新 32 次。随后更新 history，按第 6 节选择到期任务，在同一更新后权重上完成读出，再进入下一批。公共 current 任务生成一次，同时供 verifier 和选中的 RKL 使用。

长期保存题目／轨迹身份、原始 token 与 mask、实际生成协议、终止原因、逐题 RKL sum/count及训练指标。Teacher hidden 只在本批或有界评分窗口临时存活，读完释放；不保存整批全词表概率。存储、模型切换与最终权重导出的具体合同见[执行设计](../../../archive/engineering/evaluation/2026-11-01-执行与存储设计依据.md)。

相邻 history 读出共同保留的轨迹可从逐题结果计算配对变化，不增加前向。不同来源的 RKL 分别报告；四域汇总也只在同一轨迹来源、支持集和聚合定义内进行。Accuracy 分别报告各 bench 的 avg@n／采样 pass@1／strict-prompt；不把重复回答当独立题，也不据成绩筛选 teacher 或训练终点。

## 8. 原版启用 source/history 的题量与 token 预算

| 项目                        | 每分支预算                              |
| --------------------------- | --------------------------------------- |
| 真实训练回答                | 6400 条；最多 45,875,200 个回答 token   |
| reuse32 优化位置            | 最多 1,468,006,400 个回答位置           |
| 本来源固定留出 bank         | 128 条；最多 917,504 个回答 token       |
| 公共 initial student bank   | 512 条；最多 16,777,216 个回答 token    |
| 公共 teacher bank           | 512 条；最多 16,777,216 个回答 token    |
| 50 次 history＋来源常规评分 | 12,800 条次；最多 91,750,400 个回答位置 |

这些是达到 cap 时的上限，不含 prompt 计算。初始化来源读出、公共 bank／current 评分和 accuracy 生成另计。EOS 提前结束会减少实际位置。

| 节点                           | 待评分逻辑轨迹数 | 公共生成                                         |
| ------------------------------ | ---------------: | ------------------------------------------------ |
| 初始 0                         |        最多 1152 | initial student 与 teacher 各一轮；source 另 128 |
| 普通 outer                     |              256 | 无                                               |
| 加两份固定 bank                |             1280 | 无                                               |
| 同时有 current bench（25／50） |             1792 | current student 一轮，各 bench 沿用 n            |

公共 RKL 面板为 512 题，两个固定 bank 各保留 512 条回答；current RKL 沿用同一面板的 sample0。Accuracy 在 0／25／50 均使用下表完整题集，teacher 初始化也相同。去污染保护清单覆盖全部 1648 道 accuracy 题，而不只覆盖 bank 子集。

| Benchmark        | 全集题数 |   n | 每模型每节点回答数 |
| ---------------- | -------: | --: | -----------------: |
| MATH500          |      500 |   1 |                500 |
| AMC23            |       40 |  16 |                640 |
| AIME24           |       30 |  16 |                480 |
| AIME25           |       30 |  16 |                480 |
| AIME26           |       30 |  16 |                480 |
| LCB v5 既定 test |      279 |  10 |               2790 |
| GPQA-Diamond     |      198 |   4 |                792 |
| IFEval           |      541 |   1 |                541 |
| 合计             |     1648 |   — |               6703 |

冷启动时 initial student 和 teacher 各生成 6703 条，再加来源留出 128 条，共 13534 条；outer25／50 各生成 6703 条。已有同身份产物可复用。Bank 选样不另生成；公共 RKL 轨迹预算仍按上表 1152／1280／1792 等节点计数。

完整四卡两轮 7K 验收中，JustRL→R1-1.5B 的 256 条来源/history 评分约 25–27 秒，含完整状态哈希；不将其外推为 32K 公共 bank 耗时。单卡合成 token 原型及串／并行比较统一见[性能证据](../../../infra/performance/full-vocab/training.md)。

## 9. 数据划分原则

先构造题族并隔离重复，再抽取训练与留出，所有研究分支使用共同保护清单。6,400 训练题之外预留本来源 128 题。公共 benchmark 题及近重复变体从各训练来源排除。原始数据集的 train/test 命名不替代项目内的使用角色。

DAPO 缺少可靠题型／难度元数据，先随机抽样并检查题长覆盖，不把题长命名为难度。DeepCoder 按来源，TextbookReasoning 按学科，IF 按约束类型和约束数量检查分层覆盖。具体比例、题族算法和随机种子尚未冻结；不依据 teacher 答对率、初始 KL 或后续结果挑题。

抽样方案先做 CPU 数据探索再冻结：比较纯随机、按真实来源／学科比例分层、为小类保留配额三种方案；在相同题量下重复多个 seed，报告类别覆盖、分布偏差、prompt 长度分布、题族泄漏和样本集合稳定性。分层若改变目标分布，需要明确配额或估计权重；不依据后续 RKL 下降选方案。公共数学 128 题的子集配额、来源留出 128 题和训练 6400 题分别验收。四域探索尚未完成，暂不选定正式 seed 或名单。

首轮 CPU 探索使用本地数学全集 630 题，对三种 128 题抽样运行 200 个 seed。纯随机时，61/200 次至少一个子集不足 4 题，各年 AIME 最少仅 1 题；按原题库比例分层（MATH500/AMC23/AIME24/25/26 为 102/8/6/6/6）消除了这一配额波动。覆盖候选 64/16/16/16/16 提供更多小子集样本，但改变了数学混合分布；其样本 prompt 字符均值相对全集的平均绝对偏差 40.8 字符，随机为 10.9、比例分层为 11.8。字符长度仅作描述，不代表难度或模型 token 数。当前证据支持避免小子集偶然缺样，不能证明哪种方案的 RKL 估计更优；未冻结配额，也未做模型前向。原始计数与数据哈希见[抽样探针](../transfer-mechanism/data/sampling-decisions-20261031/math-sampling-probe.json)。

训练题选定后随机打乱分成 50 批；历史集合使用第 4 节的批均衡随机规则。所有 teacher 在相同 student／模式／训练来源下共享训练题 ID 和顺序，按批 refresh 后的真实轨迹随参数分叉，不能要求 token 也相同。

## 10. 初版选择记录与当前入口

| 项目                           | 基线／待决边界                                                                                                  |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| 题目清单、题族、分层比例、seed | DAPO 已有 6400＋128 工程清单；四域正式去污染、公共 512 清单及来源划分待验收                                     |
| Accuracy 题量                  | 已确认：各节点全集 1648 题、6703 条回答／模型；bank RKL 才抽取每域 128 题                                       |
| Loss 平均与报告聚合            | 已确认：训练 token-mean；评估题内 token 均值后按题平均；history 再按 batch 等权                                 |
| LR／AdamW／scheduler／clip     | 原版候选为 LR1e-6、reuse32、constant、betas0.9/0.999、wd0.01、clip1；已完成运行的实际值由对应报告及 config 维护 |
| Prompt 长度与输入              | 统计四域渲染后长度，冻结模式、stop、template、tokenizer 及模型 revision；不能静默裁短公共 cap                   |
| 数值与资源                     | 原生 BF16 乘法／FP32 输出路径已有代表组验收；32K、其他模型、共享支持条件与六卡 DP 需独立验收                    |
| Checkpoint 保留                | 已确认：只保存最终 model-only；不保存 optimizer 或中间 checkpoint，不提供同条件中途续训                         |

已确认选择与剩余抽样／资源选择见[决策记录](../../../../.agents/notes/proposed/architecture/2026-10-31-统一验证的任务计划与产物边界.md)。本节保留初版选择背景；当前运行按文首版本表和所属报告解释，不从旧提案拼接另一套默认值。

| 阅读目的                                     | 唯一主入口                                                                                                     |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| 本轮科学设置                                 | 本文                                                                                                           |
| 代码架构、任务封装与模型生命周期             | [verl 接入方案](../../../archive/engineering/evaluation/2026-10-31-四域接入设计.md)                            |
| 逐项代码证据、缺口与实施验收                 | [代码走读](../../../archive/engineering/evaluation/2026-11-01-分阶段代码走读与验收.md)                         |
| 生成 n、采样参数、cap 及 verifier 的论文依据 | [正确率与银行调研](../../../literature/teacher-transfer/2026-10-31-四域正确率评测与RKL银行生成协议调研.md)     |
| 数据来源与超参数比较                         | [训练数据调研](../../../literature/teacher-transfer/2026-10-31-OPD训练数据与超参数及验证规模比较.md)           |
| Rollout 与 RKL 含义                          | [指标调研](../../../literature/teacher-transfer/2026-10-31-OPD的Rollout来源与分布评估指标调研.md)              |
| 早期备选与题池探索                           | [旧提案](../../../archive/experiments/teacher-transfer/2026-10-31-刷新与读出早期候选.md)，不再维护现行执行配置 |
