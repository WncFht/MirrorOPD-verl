# Full-vocab OPD 优化器大规模实验计划

更新日期：2026-10-27。状态：待审阅的实验与开发计划；尚未实现本文新增功能，尚未启动大型训练。范围：全参数 student、冻结 teacher、完整词表 reverse KL。沿用 MirrorOPD 的 FSDP2、shared_sync、单节点训练路径。

## 1. 建议与研究目标

建议做。当前最值得验证的结论是：在独立调好学习率、控制参数组和精度实现后，SGD 或 Muon 是否仍能提高 OPD 的最终任务能力，以及优势是否随轨迹复用次数改变。单看训练 KL 无法回答这个问题。

主实验保留四个候选：AdamW、AdamW beta1=0、无动量 SGD、Muon+ 辅助 AdamW。全部先设 weight decay=0；分别研究 reuse=1 与 reuse=16。每个候选独立搜索学习率，再跑三个训练 seed。使用同一 checkpoint 和梯度的反事实重放解释机制，避免把全部诊断都做成完整训练。

优先补三项能力：实际更新与 BF16 可见更新的采样诊断、固定前缀的分布 probe、正确的 FSDP2 Muon 混合参数组。SGD 和 AdamW beta1=0 已可通过现有配置选择，不需要另写算法。

| 待回答的问题 | 能支持结论的证据 | 单独不足以支持的证据 |
| --- | --- | --- |
| 哪个 optimizer 更适合实际 OPD | 独立 HPO 后，多 seed 的 held-out 任务正确率与计算预算曲线 | 最初几步的训练 KL |
| 是否只是 weight decay 导致差异 | wd=0 排名，以及同一更新状态下分离衰减的重放 | AdamW 默认 wd=.01 与 SGD 默认 wd=0 的比较 |
| 动量是否滞后于刷新后的前缀分布 | beta1 消融、刷新边界附近的历史方向和函数变化、重放干预 | 一个动量夹角或冷启动首步 |
| Adam 的预条件化是否不合适 | 同梯度下控制动量、幅度和精度的更新比较 | AdamW 与 SGD 共用数值相同的 LR |
| Muon 的矩阵几何是否有用 | 辅助组固定、完整逻辑矩阵更新正确、独立 LR 搜索后的结果 | Muon 同时放大 embedding/head LR 后更好 |
| BF16 是否削弱某类有效更新 | 同一 FP32 更新的权重舍入干预，以及后续累积轨迹 | FP32 master 的参数位移非零 |

## 2. 已有证据与当前实现边界

[小实验报告](README.md)已验证条件 reverse KL 的梯度、局部曲率及在线半梯度的区别。它只有 4 条训练 prompt、4 条固定 probe、一个 seed、64-token response 和 8 次 BF16 更新，不能给出任务能力或长期 optimizer 排名。

| 小实验观察 | 对大型实验的要求 |
| --- | --- |
| 全部 wd=0 时，刷新条件下 SGD 的固定 probe KL 下降约 31.05%，AdamW 约 1.37%，Muon 约 7.85% | 优势不必依赖 weight decay；仍需 HPO 与任务评测 |
| 首步 FP32 参数 L2 已对齐，但最终 SGD 对初始 student 的 probe KL 约为 AdamW 的 110 倍 | 参数位移相同不等于函数变化相同；同时记录这两个空间 |
| Adam beta1=0 只带来有限改善 | 动量不是目前全部差距的解释；继续分离二阶矩、步幅和精度 |
| 同一 Adam 更新中，FP32 权重变化比例约 80.70%，BF16 可见变化约 0.763% | 记录 FP32 master 与 BF16 前向可见权重的差别；保留固定 buffers 的舍入控制 |
| FP32 SGD 的一次同批 KL 也出现上升，虽然梯度与更新内积为负 | 方向下降不保证有限步长下降；加入同批 pre/post KL |
| Muon 辅助 AdamW 与矩阵组共用校准后的 LR | 大实验必须把辅助组 LR 单独设置；小实验不构成纯正交化消融 |

代码检查结果如下。链接指向当前工作区，正式实验前应冻结经验证的代码版本与 resolved config；不新增资产 checksum/hash 清单。

| 功能 | 当前状态 | 计划处理 |
| --- | --- | --- |
| SGD / AdamW beta1 配置 | [build_optimizer](../../../../../../../verl/workers/config/optimizer.py)支持类名与 override kwargs | 复用；增加配置展开检查 |
| Full-KL 逐次 loss、相对位置分桶 | [指标说明](../../../../infra/architecture/metrics.md)已有 `distill/full_kl`、`distill/full_kl_position/*` | 保留 sum/count 归约；明确这是更新前的值 |
| 梯度范数和更新执行状态 | [FSDP engine](../../../../../../../verl/workers/engine/fsdp/transformer_impl.py)与 [update_records.py](../../../../../../../verl/workers/update_records.py)已有 | 补 clip factor、每组实际 LR、诊断挂点 |
| 任务 mean、pass@k、majority@k、失败计数 | [指标字典](../../../../settings/指标口径.md)已有 | 直接使用，不新增重复指标 |
| FP32 master 与实际更新、BF16 舍入比例 | 正式训练未形成上述完整诊断 | 新增有采样范围标记的诊断 |
| 固定前缀 teacher KL / step KL / initial-student KL | 普通 Full-KL 不自动提供这三种比较 | 新增独立 probe；不改变训练目标 |
| Muon + auxiliary AdamW | Megatron 有相关配置；当前 FSDP optimizer 直接接收 `module.parameters()` | 新增语义分组与分布式适配，不能只改 optimizer 类名 |
| optimizer 状态反事实重放 | [audit_optimizer_history.py](../../../../../research/precision/audit_optimizer_history.py)已有 FSDP2 基础设施 | 复用状态捕获；把原 MDPO 目标切换并验收为纯 Full-KL |

普通 Full-KL 是 eta=1 的端点，不建立训练 anchor；打开 mirror diagnostics 不会自动得到所需的 step KL。当前 Full-KL 的 `calculate_entropy=False` 约束也不能靠开启 sampled entropy 绕过。完整词表熵应在独立 probe 中计算，或以后复用 KL 核已有中间量新增 detached 统计。

## 3. 固定主实验协议

### 3.1 模型、目标与数值路径

首个模型对使用已有资产：`DeepSeek-R1-Distill-Qwen-1.5B` student / `JustRL-DeepSeek-1.5B` teacher。模型实际有约 17.77 亿可训练参数。teacher 冻结，所有 student 参数参与优化。

| 项目 | 建议固定设置 |
| --- | --- |
| loss | 直接完整词表 $D_{KL}(p_\theta\Vert T)$；所有有效 response 位置、token mean |
| 其他目标 | task reward、entropy bonus、anchor regularization 均关闭；无 sampled PG、top-k loss 或 token IS |
| KL 支持 | 保持生产实现的完整师生输出支持和教师归一化语义；不为实验截断后重归一化 |
| engine | 单节点 FSDP2、shared_sync；rollout TP=1；整个比较固定卡数及并行配置 |
| 参数/精度 | FP32 持久参数和 optimizer states；BF16 前反向，FP32 gradient reduce；记录实际运行时 dtype |
| KL backend | 优先现有 `precompute_fused`，token chunk=128、vocab chunk=8192；与 `precompute` 同权重数值对照通过后固定 |
| batch | train batch=64，PPO mini-batch=64，n=1；一轮含一批轨迹，每 reuse 一次产生一次 optimizer 尝试 |
| 数据复用 | 分别 r=1、r=16；模型与 prompts 的其他设置一致 |
| 训练长度 | prompt cap=4096，response cap=7168；按现有过滤语义处理过长 prompt，记录排除数 |
| 采样 | temperature=1、top_p=1、top_k=-1、repetition penalty=1；使用模型既定 chat template、EOS/stop 配置 |
| LR schedule | constant、warmup=0；独立 HPO；主实验所有参数组 wd=0 |
| 梯度裁剪 | global norm=1，记录实际裁剪因子；不把 `clip_grad=0` 当作关闭裁剪 |
| 其他运行配置 | gradient checkpointing、param/optimizer offload、microbatch/token budget 固定；优先沿用已有可运行配方 |
| 随机性 | 训练 seed=42/43/44，统一数据、rollout、模型 seed 设置；评估使用独立固定 seed |

使用同一个 prompt 调度可形成配对比较；不同 optimizer 产生的 response 自然不同。相同随机 seed 不代表相同在线轨迹，不能声称在线比较固定了训练样本。

Full-KL 的目标每一步基于已生成的前缀，不反传采样过程。r=1 最接近逐批 on-policy 更新，r=16 检验已有 E20/E44 类配方中多次复用的影响。两者都完整枚举词表，但只有前者每次更新前都刷新轨迹。

### 3.2 数据划分与评测

从现有 DAPO 训练源按固定 seed、原始样本 ID 划出 512 条 dev，其余训练。先做可实现的规范化题面去重与 split 交叉检查，记录数量；不能据此宣称消除了教师预训练污染。训练 epoch 上限根据实际剩余样本数设置，确保指定 outer 数真实完成。

dev 的 512 题全部用于 HPO 的任务评估。其中固定 128 题额外建立初始 student 前缀 bank，供低频 KL probe。bank 只生成一次，各候选共享；每个评估 seed 的 bank 与生成参数保存下来。另从 train 取 32 题做 LR/步幅校准，不能拿最终测试集做校准。

正式测试建议沿用项目已有 MATH-500、AMC23、AIME25 资产与 grader，启动前确认实际数据可读、题数和名称。测试集只评估初始模型及事先规定的最终 checkpoint。HPO、checkpoint 选择使用 dev，不能挑测试曲线最高点。

| 用途 | 生成预算与频率 | 主读数 |
| --- | --- | --- |
| HPO dev | 512 题 × n=4，筛选阶段末端评估；response cap=31744 | 按题等权 `acc/mean`，附失败/截断比例 |
| 正式 dev | outer=0/64/128/256，512 题 × n=4，同样长回答上限 | 固定进度的任务曲线；最终 checkpoint 为主结果 |
| 最终测试 | 每题 n=16，cap=31744；各方法相同采样与判分 | `acc/mean` 主指标，pass@1/4/16、majority@16 次指标 |
| 固定前缀 probe | 初始及每 32 outer；128 题，每题固定抽至多 128 个 response 位置 | teacher KL、initial-student KL、熵；保留完整前缀输入 |

probe 抽样只减少诊断位置的 head 计算，不减少训练位置或词表。抽样分层、权重与位置固定，使 token-weighted 和 response-weighted 口径都可恢复；逐条保留计数。不要截短输入序列来冒充在原长前缀上的同一位置测量。

增加每个评测数据集的 response 长度、自然终止率、达到上限比例、答案提取失败率、评分执行失败率。错误答案与执行失败分别计数。初始模型 KL 漂移只是行为变化，不是遗忘测量；若要研究通用能力保持，另立具备有效 grader 的独立任务评测，不用漂移代替能力分数。

## 4. Optimizer 设置与运行矩阵

### 4.1 主候选和学习率搜索

所有候选共同使用 global clip=1、wd=0、constant schedule。Adam 组 epsilon=1e-8；不继承未知历史 optimizer 状态，每条独立训练从相同 student 初始化冷启动。

| 名称 | 设置 | 第一轮 LR 候选 |
| --- | --- | --- |
| AdamW | betas=(.9,.999) | 1e-7、3e-7、1e-6、3e-6、1e-5 |
| Adam0 | AdamW，betas=(0,.999) | 1e-7、3e-7、1e-6、3e-6、1e-5 |
| SGD | momentum=0、nesterov=False | 1e-3、3e-3、1e-2、3e-2、1e-1 |
| Muon+AdamW | 隐藏线性矩阵：momentum=.95、Nesterov、NS=5、`match_rms_adamw`；其他参数：AdamW(.9,.999) | 矩阵 LR：1e-6、3e-6、1e-5、3e-5、1e-4；aux LR 固定 1e-6 |

这些是搜索起点，不是推荐最优值。若某方法最佳点位于网格边缘，所有方法保留相同的额外搜索机会，每个方向先扩展一档；最终报告实际试验次数与总搜索开销。不能把稳定性失败的 trial 从成本账本中删掉。

Muon 的分组按模块语义识别，不能仅用 `ndim==2`：hidden `Linear.weight` 使用 Muon；embedding、lm_head、norm、bias 使用 auxiliary AdamW；共享参数只计一次。完整词表输出头在辅助组。先固定辅助 LR，研究矩阵更新；若 Muon 成为最终候选，补 aux LR={3e-7,1e-6,3e-6}，并给 AdamW/SGD 相同预算的分组 LR 控制。核心阶段的结论应写为“所测试的 Muon 混合配方”，不能外推为 Muon 所有参数组设置。

### 4.2 推荐完整预算

两个 reuse 条件分别搜索，避免把 r=16 的最优 LR 强加给 r=1。HPO 用 seed42，主实验三个 seed 从初始模型独立重训。阶段 B 继续阶段 A 的完整 checkpoint，保留 optimizer、scheduler、数据进度与 RNG。

| 阶段 | 配置 | 数量 | 每配置预算 | 总 outer | 总 optimizer 尝试 |
| --- | --- | ---: | --- | ---: | ---: |
| A：LR 筛选 | 4 方法 × 5 LR × 2 reuse | 40 | 32 outer | 1,280 | 10,880 |
| B：长一些的 HPO 确认 | 每方法/复用条件保留 2 LR，共 16 条续跑 | 16 次续跑 | 32 → 128 outer，新增 96 | 1,536 | 13,056 |
| C：主结果 | 4 方法 × 2 reuse × 3 seeds | 24 | 256 outer | 6,144 | 52,224 |
| 核心合计 | 64 条独立训练，另含 16 次续跑 | — | — | 8,960 | 76,160 |

阶段 A 按 dev mean 选两档 LR，但先检查统计不确定性；差异不清楚时保留更分散的 LR，而不是根据训练 KL 淘汰。阶段 B 按 128 outer 的 dev mean 冻结配置，最终测试仍不可见。若早期曲线交叉明显，扩大确认预算并给所有方法同等机会。主结论使用三个 seed 的结果，HPO 阶段不充当重复种子。

该预算的新生成回答数为 573,440；在每条都达到 7,168 token 的上界下，新 response 约 41.10 亿 token、learner 重复处理 response 约 349.39 亿 token。实际数量以有效 mask 计数，另加 prompt、teacher、评测、probe 与失败重试成本。三个小测试集长回答 n=16 也可能占显著预算，应单列。

这是可分批执行的大实验，不能直接给出可信 GPU 天数。先在正式分配的同型号独占资源上完成代表性训练与评测 profile，再用实际长度分布估算；共享 GPU 空闲显存上的耗时不用于成本结论。

算力受限时，先完成 r=16 的半套：20 条筛选、8 次续跑、12 条主实验，共 4,480 outer、71,680 次尝试。随后补 r=1 的对应半套（4,480 次尝试，但同样有 4,480 outer 的采样/teacher 成本）。只做前半套时，结论限定在 reuse16，不能称为一般 on-policy OPD 排名。

### 4.3 预算比较的解释

同一 reuse 内，四种方法的 outer、prompt 数、更新尝试数一致。响应长度不同会导致实际 token 与时间不同，因此同时画任务分数对新生成 token、learner token、成功更新数、GPU-hours 的曲线。

r=1 与 r=16 的主比较固定生成 prompt 预算，更新数相差 16 倍。另做等更新数的辅助比较：使用主实验 r=1 的 outer=256 checkpoint，与额外保存的 r=16 outer=16 checkpoint，都是 256 次尝试；它们看到的新轨迹量不同。两个比较分别回答数据利用与更新预算问题，不宣称同时控制所有成本。若有跳步，要按实际成功数重新对齐。

## 5. 指标增补与采样频率

### 5.1 必须保留及低成本新增

以下新增名称是拟议接口，实施时与 [metric schema](../../../../../../../verl/utils/metric/schema.py)统一注册，避免产生两套同义字段。

| 指标 | 状态 / 频率 | 定义和用途 |
| --- | --- | --- |
| `distill/full_kl`、相对位置分桶 | 已有，每更新 | 更新前、有效 response token 加权；输出 sum/count |
| `optim/actor/grad_norm_preclip`、update_applied/skipped | 已有，每尝试 | 判断实际执行与数值失败；不从 loss 猜测执行状态 |
| `optim/actor/clip_factor` | 新增，每尝试 | 真实作用于梯度的缩放系数；与全局范数一致 |
| `optim/actor/groups/{group}/lr_used`、wd、parameter_count | 新增 | LR 每更新；静态组信息每 run；记录实际执行值而非仅配置默认值 |
| response/training tokens、response count | 已有量复用，补累计轴 | 区分新生成、teacher 评分、learner 重复处理的工作量；计入失败/重试开销 |
| rollout policy version、reuse age | 新增，每更新 | 记录本批生成时的成功更新编号，以及当前更新前的差；不把 inner index 当真实 policy age |
| termination/cap/extraction failure | 部分统计复用，补缺项 | 每 outer/评测记录；检查是否用提前结束换取低 KL 或成本 |
| dtype、group semantics、软件版本、resolved config | 每 run，恢复后核对 | 区分持久参数、compute 参数、gradient reduce、optimizer state、rollout 权重 dtype |

总 token 数不能用 `outer × batch × cap` 替代。成功 optimizer 数和尝试数分别记录；学习率 scheduler 是否在跳步时前进也要随运行协议冻结。

### 5.2 参数更新与历史状态

令更新前 FP32 参数为 $\theta_t$，实际更新后为 $\theta_{t+1}$，梯度 $g_t$ 指 unscale 后、裁剪前、已按全局 loss 归一化的梯度：

$$
\Delta_t=\theta_{t+1}-\theta_t,\qquad
a_t=g_t^\top\Delta_t,\qquad
c_t=\frac{g_t^\top\Delta_t}{\Vert g_t\Vert\Vert\Delta_t\Vert}.
$$

推荐采样更新前后参数，并记录以下量。每组分别输出 backbone、embedding、head、norm/bias；tied 参数按实际唯一存储归组并注明。

| 拟议指标后缀 | 定义 | 目的 |
| --- | --- | --- |
| `delta_l2`、`delta_rms`、`relative_delta_l2` | 实际 FP32 位移，后者除以同范围更新前参数范数 | 检查真正步幅与各组分配 |
| `grad_dot_delta`、`grad_delta_cos` | 上式的内积与夹角 | 检查一阶下降及历史项的影响 |
| `history_grad_cos` | 更新前 $\cos(m_{t-1},g_t)$ | 观察刷新边界上的历史一致性；零历史时 undefined |
| `direction_grad_cos` | optimizer 实际使用的预条件化/正交化方向与当前梯度的夹角 | 与历史 buffer 夹角分开；Muon 另记 NS 前后方向 |
| `vhat` / `inv_sqrt_vhat` 分布 | Adam 各组中偏差校正二阶矩及实际分母缩放的采样分位数 | 描述坐标缩放；不能解释成 Hessian 的测量 |
| `decay_delta_l2` / `data_delta_l2` | wd 消融时分离衰减与数据项 | 不能用总步幅推断衰减大小 |

默认前 8 次更新密集记录，随后每 32 次尝试记录一次，并覆盖选定 outer 的第一次和最后一次更新。采样固定于逻辑参数坐标，所有候选相同；建议每组分层采样，总计不超过 100 万坐标。输出采样数、总体参数数与抽样概率；全局估计范数使用相应加权和，字段带 `sampled`，不冒称精确范数。小组可以全量观察。

不要每步复制全部 1.5B 参数。诊断快照设置每 rank 内存上限，超过时降采样并记录覆盖率。精确全量更新范数和状态分析放在少量离线 checkpoint 重放中；若以后给 fused optimizer 增加内部统计，须证明没有改变更新。

跨 rank 先对 dot product、平方和及有效计数归约，再算 cosine/RMS。复制的逻辑参数只统计一次，FSDP padding 不进入分母；不能平均 rank-local cosine。分位数来自统一采样池或固定边界直方图，不平均各卡分位数。

### 5.3 BF16 前向可见更新

对实际 compute dtype 的确定性 cast $Q$，在同一采样坐标定义：

$$
\Delta^{\mathrm{vis}}_t=Q(\theta_{t+1})-Q(\theta_t),\qquad
f_{\mathrm{change}}=\frac{\#\{i:Q(\theta_{t+1,i})\ne Q(\theta_{t,i})\}}{N}.
$$

每次参数采样同时记录 FP32 changed fraction、BF16 visible changed fraction、$\Vert\Delta^{vis}\Vert/\Vert\Delta\Vert$、$\Vert\Delta^{vis}-\Delta\Vert/\Vert\Delta\Vert$ 及两者 cosine。差值与范数在 FP32/FP64 中计算。这里是 cast 前后权重的差，不是 `cast(delta)`。

字段必须注明它是从 master 模拟 cast 的诊断；先用 FSDP2 实际 compute 权重与 rollout 同步后权重验证一致性。若训练/rollout 的转换不同，应分别记录。比例分母为零时输出 undefined。

可见变化率低并不证明训练永远无效：FP32 master 可以跨步积累直至越过 BF16 阈值。补充每 32/128 次成功更新的累计可见位移，验证累积效应。舍入误差范数比不能称为“丢失的更新能量比例”。

### 5.4 函数变化与同批下降

在同一组固定前缀 $s$ 上区分四个读数，单位统一为 nats/token：

$$
\begin{aligned}
K_T(t)&=\mathbb E_s D_{KL}(p_t\Vert T),\\
K_{\mathrm{step}}(t)&=\mathbb E_s D_{KL}(p_{t+1}\Vert p_t),\\
K_{\mathrm{init}}(t)&=\mathbb E_s D_{KL}(p_t\Vert p_0),\\
K_{\mathrm{stale}}(t)&=\mathbb E_s D_{KL}(p_t\Vert p_{\mathrm{rollout}}).
\end{aligned}
$$

`K_step` 在少量同批抽样前缀上测，`K_init` 在固定 dev bank 测，`K_stale` 在对应 batch 的同一批前缀测。明确 KL 方向；都以完整词表归一化。历史分布仅作诊断，不能混入训练目标。参考 logits 不常驻为 `B×L×V`，使用有限位置、分块 head 计算和离线重放控制内存。

同批 pre/post KL 使用同一 teacher、输入、mask、位置、归一化、数值 backend 和可控 RNG，得到 $\Delta L=L(\theta_{t+1})-L(\theta_t)$。若精确全梯度/全位置损失来自离线重放，再比较 $\Delta L$ 与 $g^\top\Delta$；前者减后者是有限步长的一阶 Taylor 残差，不直接称为 Hessian 特征值。线上抽样 loss 不能与全 batch 梯度内积拼成严格 Taylor 等式。

训练前/后 probe 建议每 16 outer 选择该 outer 首次和末次更新，每次最多 8 条 response、每条 32 个位置。先 profile 开销。完整 dev bank 每 32 outer 测一次。相对位置分桶继续保留，另在离线分析加绝对位置段和自然终止/截断分组，避免不同长度的相对位置错配。

在同一 probe 计算完整词表 entropy、EOS 集合总概率，可诊断概率集中与停止行为。训练数据上的概率熵与输出正确率是不同量；不能把熵降低自动解释为能力提升。

### 5.5 日志轴、统计和性能

更新前 loss 与更新动作绑定到 `step/update`；pre/post 诊断同一条记录保留 update 编号和成功状态。固定 dev/test 评测绑定对应 `step/optimizer`，同时附 outer、生成/训练 token 累计值和评测使用的权重版本。新 probe 字段需显式登记 axis，避免默认路由导致误读。

所有均值保留 sum/count，空桶、零范数和没有执行的更新输出 undefined/event，不填零。每条回答先取均值再跨回答平均，与全部 token 平均都保留，名称明确区分。位置、长度、梯度等高维分布不按每个样本创建 W&B key；原始行数据写本地日志仓库，在线只写摘要。

轻量诊断目标开销低于 5%，应通过 diagnostics on/off 的配对 profile 实测。较重 probe、离线重放和评测分别计时。记录训练、生成、teacher、同步/offload、评测、诊断的耗时；分阶段 peak memory 需要明确 reset 与同步边界，当前累计 high-water 不能直接当阶段峰值。optimizer 速度排名只在相同独占资源、相同诊断配置下报告。

## 6. 用小量额外实验解释机制

主实验完成前即可从指定 outer=1/16/64/256 的 checkpoint 做离线重放。保存模型、optimizer、scheduler、RNG 和所需 batch；r=16 的 outer=16 checkpoint 也用于等更新数比较。诊断后不得把修改后的状态写回主训练。

### 6.1 同状态、同梯度的反事实更新

从 AdamW checkpoint 捕获一个真实 Full-KL 梯度，对同一权重、batch、teacher 和 buffers 比较原生 Adam、仅去掉衰减、仅删除历史一阶矩、仅替换二阶矩缩放、冷启动 Adam、SGD 方向、Muon 方向。记录每种候选是否保留 step counter、bias correction 和历史状态；直接清零 m 但保留计数，与从头训练 beta1=0 是不同干预。

需要新增或适配的矩阵如下：

| 控制方式 | 设置 | 回答的问题 |
| --- | --- | --- |
| 原始步幅 | 真实 LR 和状态 | 原生更新会做什么 |
| 等参数 L2 | 每个方向用标量缩放到相同 FP32 位移 | 排除全局参数幅度；SGD 一阶优势在此约束下有数学必然性 |
| 等函数变化 | 在校准前缀上匹配 `K_step`，独立 dev 前缀测 teacher KL 改善 | 参数空间排名是否只是函数步幅差异 |
| 小幅度扫描 | 原生步长 × {0,.25,.5,1,2} | 检查有限步长曲率与过冲 |

等函数变化使用预先冻结的目标 KL，例如 {1e-5,1e-4,1e-3} nats/token；在固定有限区间内搜索，记录无法匹配的候选，不能假设大步长 KL 全局单调。它是局部机制诊断，不把每步 line search 偷换进主实验 optimizer。

补充执行当前梯度为零的 $U_t(0)$ 与真实梯度的 $U_t(g_t)$，比较历史项、衰减项以及当前梯度的增量影响。Adam 的分母也随梯度改变，不能简单线性相加解释全部更新。Muon 与 Adam 历史来自不同长期轨迹；在 Adam checkpoint 上构造 Muon 方向必须说明历史如何定义，不能称作原生 Muon 训练状态。

### 6.2 精度干预

对完全相同的 FP32 gradient、optimizer state 和更新，比较：FP32 权重+FP32 arithmetic、BF16 舍入权重+FP32 arithmetic、BF16 权重+BF16 arithmetic。保持 rotary cache 等所有 buffers、teacher 和输入固定，第二项隔离权重舍入，第三项再加入运算精度影响。

先在 1.5B 的离线小批次运行，不要求整套长序列训练改为 FP32。若证据充分，再选 AdamW 与一个赢家做短程 FP32 或部分高精度消融；只提高 head 精度不能当作全模型舍入问题已经消除。随机舍入属于新算法/实现，放在后续扩展，不与原始 optimizer 主结果混合。

### 6.3 权重衰减、动量和裁剪

wd 主实验全部为零，可先判断优势是否不依赖衰减。后续对 AdamW 与主实验赢家做 wd={0,.01,.1}，固定各自已选 LR，并记录累计收缩因子及真实衰减位移。不同 LR 的相同 wd 产生不同 $\prod_t(1-\eta_t\lambda)$，不是相同正则化强度。另做累计收缩量匹配或同 LR 下的反事实重放，才能更干净地归因。

原生 torch SGD 的 `weight_decay` 是把 $\lambda\theta$ 加入梯度；有动量时不同于 AdamW 的解耦衰减。若需要 SGD 衰减组，明确实现独立 decoupled decay 并验收，不能直接把同名配置当成同一种算法。

动量扩展优先 SGD momentum=.9 与 Muon momentum=0，分别重新校准/搜索 LR；原生 SGD momentum buffer 没有 Adam 的同一套归一化与 bias correction，不能用同 LR 声称只改历史平滑。Adam beta1=.9/0 主实验已经覆盖核心一阶矩消融。

裁剪扩展仅在主结果显示裁剪频繁且各方法明显不同时增加。当前 engine 的 max_norm=0 可能清零梯度；应新增明确关闭分支或经验证的无裁剪配置，再比较。相同裁剪因子对 SGD 步长与近似尺度不变的 Adam/Muon 影响不同，应把该事实纳入解释。

### 6.4 转移与谱分析

第二个模型对优先复用已有 `Qwen3-1.7B-Base / Qwen3-8B` 协议，验证 tokenizer、完整输出支持、chat template 和 EOS 的既定行为，再对 AdamW 与主实验赢家独立做小规模 LR 搜索及三个 seed。若先使用现成 `Polaris-7B-Preview` teacher、保持 R1-1.5B student，则明确该组检验 teacher 转移，不能宣称已跨 student 架构泛化。

7–8B student 属于后续规模扩展，需先选定实际可用且协议兼容的更强 teacher，再测显存与计算预算。不要为凑规模把 teacher 强弱与 optimizer 效应同时改变。

谱分析仅针对少数预先选定层的 q/o/up/down 矩阵，在稀疏 checkpoint 上计算 gradient、momentum、单步更新、累计参数差分各自的谱。用 stable rank、奇异值能量分布描述，不能把累计差分低秩直接解释成 Muon 单步正交化好或坏。无需在线全模型 SVD 或 Hessian 特征分解。

## 7. 代码变更清单与验收

本文仅提出下面的改动。实施时先交付最小可运行版本，再增加昂贵诊断，避免实验变成另一套训练框架。

| 优先级 | 位置 / 拟新增模块 | 具体改动 | 验收要求 |
| --- | --- | --- | --- |
| P0 | `launchers/math_experiment.py` 及拟新增 optimizer-study 配置生成器 | 生成候选、LR、reuse、seed 矩阵；拆出 dev/test；dry-run 输出 resolved config 与计划预算 | 所有组合只用 full_kl；步数、batch、wd、参数组、seed 可核对；不启动训练 |
| P0 | `verl/workers/config/optimizer.py` + FSDP `_build_optimizer` | 增加具名模块参数分组、Muon+AdamW wrapper；支持独立 matrix/aux LR | 参数无遗漏/重复；全参数训练；scheduler、offload、checkpoint 和恢复均覆盖两个 optimizer 状态 |
| P0 | 拟新增 `verl/utils/optimizers/muon_fsdp.py` | 对完整逻辑矩阵做 NS，正确处理 DTensor shard/replica；先完成慢但正确的参考路径 | 单卡/2 卡/4 卡同梯度更新与参考一致；不能对各 rank 局部切片分别正交化 |
| P0 | FSDP `optimizer_step` + 拟新增诊断 helper | unscale 后/clip 后/step 后挂点；按预算采样参数与状态，记录实际 step 与舍入量 | diagnostics off/on 参数轨迹一致；跳步无虚假位移；统计归约正确；记录开销 |
| P0 | 拟新增 fixed-prefix probe runner | 复用 Full-KL 计算路径，支持 teacher/init/pre-step/rollout 参考 | 固定输入、mask、完整词表与权重版本；不修改训练状态；不分配全量常驻 logits |
| P0 | `update_records.py`、`metric/schema.py` 与指标文档 | 注册新字段的语义、归约、时间轴和 undefined 行为 | 不平均均值/夹角；attempt 与 applied 坐标清晰；旧指标不重命名破坏已有分析 |
| P1 | `audit_optimizer_history.py` 可复用部分 + 新 Full-KL 重放入口 | 捕获同梯度状态，分离历史/预条件化/衰减/精度；幅度匹配 | 原生重放与真实下一步一致，再解释反事实；修改 buffers 的干预必须显式标记 |
| P1 | 本实验目录分析脚本 | 统一绘制质量—预算曲线、paired seed 差、prompt 聚类区间、分组诊断 | 读取原始日志，不复制通用 W&B 解码器；输出缺失数据与失败 trial |
| P2 | 少量层谱分析、额外精度/动量/衰减实验 | 依据主结果选择扩展 | 主结果与探索结果分开报告 |

Muon 的参考实现可以逐矩阵 all-gather、更新、再分片以验证语义，但这不是默认高效方案。必须 profile 峰值内存、通信和 offload；过高则先优化适配，再纳入效率排名。不能为了使用已有 Megatron Muon 临时切换整个训练后端并把差异全部归因于 optimizer。

需要的测试集中在有实质风险的部分：小矩阵完整更新 oracle、混合参数组/tied weights、FSDP2 跨卡一致性、checkpoint 连续训练等价性、梯度裁剪/跳步、metric sufficient-statistic 归约、probe 不污染 RNG/模型/optimizer 状态。SGD/Adam 配置选择只做 smoke 与 kwargs 检查，不重写 PyTorch 的数学单测。

现有命令可通过 `CONFIG_ONLY=1` 与 `--method full_kl` 审阅展开配置。optimizer 参数可用 `actor_rollout_ref.actor.optim.optimizer=SGD`、`actor_rollout_ref.actor.optim.override_optimizer_config`、`actor_rollout_ref.actor.optim.betas`、`actor_rollout_ref.actor.optim.weight_decay` 表达。新的 hybrid optimizer 与 diagnostics 字段待实现，本文不提供会让人误以为已经可运行的完整 Muon 启动命令。

## 8. 结果判定与执行顺序

任务结果逐数据集报告均值、三个训练 seed 的原始结果与离散程度。按同一 seed 做方法间配对差；评估回答按 prompt 聚类 bootstrap，不能把同题 16 次回答当成 16 道独立题。仅有三个训练 seed 时，不能靠题目 bootstrap 掩盖训练随机性。预先以 1 个百分点作为值得继续投入的实用差异参考，同时报告区间；小数据集区间宽时结论保留不确定性。

只有在独立 HPO、多 seed、最终任务得分和预算曲线上仍有一致收益时，才建议替换默认 optimizer。若只降低固定 probe KL，就报告条件分布拟合收益。若只在 reuse16 有收益，则限定于高复用训练；若只在某个 teacher/model pair 成立，明确模型对边界。完整报告附 HPO 成本和稳定性失败率。

执行顺序：

1. 冻结协议和数据 split；新增配置矩阵，检查现有 Full-KL 数值路径、dtype 与初始化一致性。
2. 完成分组 Muon 与 P0 诊断验收；各候选跑短 smoke，保存/恢复各验证一次。
3. 在正式资源上 profile，给出真实 GPU-hours 预算；开始阶段 A/B，冻结各 reuse 的 LR。
4. 跑阶段 C；dev 按预定进度评估，保存必要机制 checkpoint；最终再运行测试集。
5. 用预定 checkpoint 做反事实重放；有明确机制信号后开展 wd、动量、精度和转移扩展。

原始日志、逐题输出、probe 数据和诊断快照放到同级 `wandb-logs/MirrorOPD-verl/analysis/` 下的独立运行目录；本仓目录保留计划、配置生成器、分析脚本及成品图表。保留实验所需配置与样本 ID，不新增资产哈希清单，不修改既有历史运行结果。
