# Agent Note: Full-vocab 聚合与未来信用实现计划

Status: proposed

状态：聚合与 future 已有 CPU、CUDA、真实 FSDP2 和性能证据；[数值验收](../../../../docs/infra/numerics/full-vocab/validation.md#聚合与未来信用验收2026-11-04)保留混合精度 MDPO 严格参数对比失败，[完整 actor 配对](../../../../docs/infra/performance/full-vocab/training.md)记录测量范围。第 2、5、7 节保留 2026-11-03 的实施基线和当时阶段状态，不代表当前验收缺口；当前配置与支持矩阵见 [Full-vocab 数据流](../../../../docs/infra/architecture/full-vocab.md#聚合与未来信用)。单投影连续分块、统计复用和公共 VJP 已由 [KL 后端统一执行决策](../../implemented/architecture/2026-11-04-KL后端统一执行与VJP重构.md)落实；本文继续拥有数学合同、原设计理由与验收要求，未完成项不因迁移结项。

目标是在现有 Full-vocab 更新链中增加回答级聚合和未来信用分配，支持 sequence KL、TOP-D 风格 future mean 及 discount 对照。实现保持原训练循环、PPO、rollout reuse 和 anchor 生命周期；默认 future 关闭。CPU 验收不启动正式训练，冻结运行配置与源码快照保持原状。

阅读顺序：[范围](#1-范围与默认行为) → [当前代码链](#2-当前实现与修改落点) → [数学合同](#3-未来信用的数学合同) → [聚合](#4-聚合权重与梯度的连接) → [执行设计](#5-计算流水线与模块分工) → [实施与验收](#7-实施阶段与验收)。聚合文献与目标边界由[平均方式调研](../../../../docs/literature/learning-signal/2026-10-31-Full-RKL的平均方式与MOPD及verl实现对照.md)维护，科学消融问题由 [weighted targets](../../../../docs/experiments/learning-signal/weighted-targets.md) 承接。

## 1. 范围与默认行为

| 项目     | 计划                                                                                      | 默认及边界                                                  |
| -------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| 聚合     | 接入 `token-mean`、`seq-mean-token-mean`、`seq-mean-token-sum`、`seq-mean-token-sum-norm` | 保留 `token-mean`；公共 API 已有的 `token-sum` 不因重构丢失 |
| 未来信用 | 当前 Full-vocab 目标直接求导，未来同一目标代价经当前 sampled action 的 score 回传         | 默认 `none`，恢复现有 immediate-only 行为                   |
| 变体     | future sum、future mean、discounted sum、discounted mean、完整 discounted sequence 目标   | 每种语义明确命名；γ 仅用于 discount                         |
| 训练更新 | 沿用 optimizer、anchor、reuse、mini-batch、梯度累积和裁剪                                 | 不引入 TOP-D PPO/clipping 或 prompt 组内标准化              |
| 系数     | local 与 future 的相对系数固定为 1                                                        | 不增加 λ；现有 η 的目标缩放对合成项统一应用一次             |
| 验证     | 参考梯度、分块路径、分布式一致性和性能比较                                                | 工程验收与正式任务消融分开                                  |

实现顺序为：先扩展聚合，再接入可验证的未来信用参考，最后优化生产分块与 fused 路径。这样可以分别识别样本权重变化、梯度变化和 kernel 性能变化。

## 2. 当前实现与修改落点

### 2.1 配置、采样与冻结目标

1. [MirrorConfig 与 DistillationLossSettings](../../../../../../verl/workers/config/distillation_spec.py#L12) 描述目标和 payload。`teacher_only_kl` 把 `mdpo/full_vocab/eta=1` 识别为纯 reverse-KL；`anchor_payload` 决定是否需要冻结 anchor。新增 future 配置应由同一解析边界验证，不能由 trainer 或 kernel 各自猜测方法。
2. [FullVocabConfig](../../../../../../verl/workers/config/distillation.py#L37) 承载 backend、chunk、支持集及终止动作配置；[统一 launcher](../../../../launchers/_run_single_teacher_fsdp2.sh#L283) 当前显式传入 `loss_agg_mode=token-mean`。入口已经通过 [build_plan](../../../../launchers/math_experiment.py#L481) 和 shell 末尾参数支持 Hydra override；优先复用这条透传路径并核对 resolved config，不新增平行的聚合 CLI。
3. [PPOTrainer._step_once](../../../../../../verl/trainer/ppo/v1/trainer_base.py#L721) 编排 rollout、teacher 评分和 actor 更新；[PPOTrainer._compute_shared_teacher_log_probs](../../../../../../verl/trainer/ppo/v1/trainer_base.py#L2016) 对 Full-vocab 选择 hidden payload。Full-vocab future 需要当前 student 的 KL，不能放进普通任务 [PPOTrainer._compute_advantage](../../../../../../verl/trainer/ppo/v1/trainer_base.py#L2176) 后整轮复用。
4. [PPOTrainer._update_actor](../../../../../../verl/trainer/ppo/v1/trainer_base.py#L2273) 将小字段及样本身份交给 worker，设置 `global_batch_size`、mini-batch 和 epochs。Teacher/anchor hidden 由 [HiddenBatchIO](../../../../../../verl/trainer/distillation/hidden_io.py) 按 micro-batch 身份读取，不增加一份 trainer 级 future payload。
5. [ActorRolloutRefWorker.update_actor](../../../../../../verl/workers/engine_workers.py#L907) 用 [anchor_scope](../../../../../../verl/trainer/distillation/anchor.py#L41) 包住整次 `train_mini_batch`，后者遍历 mini-batch/epochs。保持 q 的原冻结边界；每次当前 student forward 都重算 local KL 和 future，不能缓存第一次 inner update 的 future。

这里不新增第三个模型或额外 backbone 评分。固定的是给定前缀上的 teacher/anchor 分布；这些分布随前缀改变，不能当作全序列统一的常量。

### 2.2 Engine、坐标与归一化

[FSDPEngine.forward_backward_batch](../../../../../../verl/workers/engine/fsdp/transformer_impl.py#L768) 先对当前 optimizer mini-batch 的 `loss_mask.sum()` 做 DP all-reduce，再拆 micro-batch。因此现有 `batch_num_tokens` 的作用域是一次更新，而非整次 rollout 或单个 micro-batch。

[FSDPEngineWithLMHead._full_vocab_output](../../../../../../verl/workers/engine/fsdp/transformer_impl.py#L1267) 通过 [prediction_positions](../../../../../../verl/workers/utils/padding.py#L34) 选择预测 hidden，并检查 teacher/anchor 位置与版本。当前已经有构造 future 所需的原始 token 和逐样本位置：

- sampled action 从每条 `input_ids[prediction_position + 1]` 取得，保持与 hidden 行一一对应，不能再 shift 一次。
- 每条位置数组的长度给出扁平 token 行的分段边界。不能跨回答做一个无分段 `cumsum`。
- [response_target_coordinates](../../../../../../verl/workers/utils/padding.py#L54) 给出原始 response 坐标并保留 mask 空洞，可用于 discount 距离和位置指标。

实际计算仍须经过 `head(selected, full_vocab_callback=compute)`。绕开包装 head 直接读取权重会破坏 FSDP2 hooks；返回结果仍须像 [FullVocabOutput](../../../../../../verl/trainer/distillation/output.py#L13) 一样把可微 tensor 暴露在 NamedTuple/pytree 中。

[distillation_loss](../../../../../../verl/trainer/distillation/losses.py#L217) 当前拒绝非 `token-mean`，再调用 [LossContext.normalize_token_sum](../../../../../../verl/workers/utils/loss_context.py#L32)。通用 RL 的 [compute_policy_loss_vanilla](../../../../../../verl/trainer/ppo/core_algos.py#L1291) 先构造可微 token loss，再用 [agg_loss](../../../../../../verl/trainer/ppo/core_algos.py#L1145) 归约；[LossContext.aggregate](../../../../../../verl/workers/utils/loss_context.py#L27) 已经转发到这个公共实现。这是可复用的聚合语义，但 Full-vocab 的梯度生成时机不同。

### 2.3 聚合与 kernel 耦合的实际原因

[precompute_rows](../../../../../../verl/trainer/distillation/vjp.py#L43) 逐 token chunk 调 row kernel，得到行值及 logits 梯度 `dz`，立即完成：

$$
dH_{\rm chunk}=dZ_{\rm chunk}W,\qquad
dW\mathrel{+}=dZ_{\rm chunk}^{\mathsf T}H_{\rm chunk}.
$$

它只保存累计好的 `dH/dW`，将逐 token 行值标成不可微。Backward 只把标量 upstream 乘到保存的梯度上。因此事后对 detached 行值套 `agg_loss` 只能改变数值，无法改变已经合并的逐行梯度。每行聚合权重、未来 score 项都必须在上述 GEMM 之前生效。

[RecomputedKL.backward](../../../../../../verl/trainer/distillation/reverse_vocab.py#L91) 支持逐行 upstream，是加权梯度的现有参考，但会重新投影；不能据此承诺与 `precompute` 相同的成本。[ForwardRecompute](../../../../../../verl/trainer/distillation/forward_vocab.py#L82) 也在 forward 累积梯度，扩展即时项聚合时需要一并核对；reverse sequence-KL 的 future 语义不自动适用于 forward KL 或 BPO。

[precompute_kl_rows](../../../../../../verl/trainer/distillation/rows.py#L19) 和 [_mirror_rows](../../../../../../verl/trainer/distillation/rows.py#L87) 目前只返回行值、`dz` 及诊断，未向外保留 score 修正所需的 student 概率/归一化信息。仅把 chunk 遍历倒过来还不够，需要调整 chunk 内的计算边界。

## 3. 未来信用的数学合同

### 3.1 当前条件 KL 与完整序列梯度

令 $p_\theta$ 为当前 student，$C(\cdot\mid s)$ 为对 student 参数冻结、归一化的参考分布。令 $s_t$ 为生成 $a_t$ 前的状态，$\mathcal F_t$ 为动作采样前的信息，定义

$$
c_t=\log\frac{p_\theta(a_t\mid s_t)}{C(a_t\mid s_t)},\qquad
\psi_t=\nabla_\theta\log p_\theta(a_t\mid s_t),
$$

$$
k_t=\mathbb E[c_t\mid\mathcal F_t]
=D_{\rm KL}(p_\theta(\cdot\mid s_t)\Vert C(\cdot\mid s_t)),\qquad
A_t=\nabla_\theta k_t\ \text{（固定前缀）}.
$$

对 on-policy 的完整回答，包含一致的 EOS 或明确的固定有限时域，并满足支持集及微分交换条件时，序列 reverse-KL 目标为

$$
J=\mathbb E\sum_t c_t=\mathbb E\sum_t k_t.
$$

普通 sampled reward-to-go 梯度与计划中的 Full-vocab 估计分别为

$$
\widehat g_S=\sum_t\psi_t\sum_{u\ge t}c_u,\qquad
\widehat g_F=\sum_t\left[A_t+\psi_t\sum_{u>t}k_u\right].
$$

两者期望相等：当前项利用 $\mathbb E[\psi_tc_t\mid\mathcal F_t]=A_t$；未来项利用 $t<u$ 时 $\psi_t$ 对 $\mathcal F_u$ 可测，得到 $\mathbb E[\psi_tc_u]=\mathbb E[\psi_tk_u]$。当前条件动作期望被精确积分，未来仍沿采到的状态路径估计；没有枚举未来分支，也没有消除状态采样方差。局部条件方差被消除不保证整个梯度估计的总方差必然下降，跨时协方差仍会影响结果。

上述无偏等式只用于相应 on-policy 目标。现有 rollout reuse 在 student 改变后仍是原有的 off-policy 近似；增加 future 不自动修复它。因长度上限截断的尾部没有观测，默认不凭空补 bootstrap，也不把截断标成自然 EOS。

### 3.2 固定 rollout 上的可微 surrogate

用正 KL 作为待最小化的 cost，令 $F_t$ 为只含严格未来的 detached cost，参考实现采用

$$
L_{\rm sur}=\sum_t\left[k_t+\operatorname{sg}(F_t)\log p_\theta(a_t\mid s_t)\right].
$$

对当前 logits 的梯度为

$$
g^Z_t=g^{\rm KL}_t+\operatorname{sg}(F_t)(e_{a_t}-p_t).
$$

这里用加号是因为最小化 cost；若写成 reward，须先取负号。Future 项严格排除当前项，避免和 $A_t$ 重复计算；`F_t` stop-gradient，防止重复穿过未来 KL。无需构造 `[tokens, vocab]` one-hot，sampled action 项通过索引累加即可。该 surrogate 的标量值不是序列 KL，不可沿用 KL 指标名。

### 3.3 统一 suffix 变换与六种模式

用从 0 开始的原始 response token 坐标 $\tau_t$ 定义 discount 距离；动作下标从 1 开始，连续回答有 $\tau_t=t-1$。无效位置不贡献 cost；mask 空洞仍消耗时间距离。对每条回答独立计算

$$
S_t=\sum_{u>t}\gamma^{\tau_u-\tau_t}k_u,\qquad
W_t=\sum_{u>t}\gamma^{\tau_u-\tau_t}.
$$

对相邻有效位置的距离 $\Delta_t=\tau_{t+1}-\tau_t$，可用递推

$$
S_t=\gamma^{\Delta_t}(k_{t+1}+S_{t+1}),\qquad
W_t=\gamma^{\Delta_t}(1+W_{t+1}).
$$

末有效位置的 $S,W$ 均为 0；$W=0$ 时 mean 返回 0。

| `future.mode`         | $F_t$          | local 权重                                    | 含义                             |
| --------------------- | -------------- | --------------------------------------------- | -------------------------------- |
| `none`                | 0              | 1                                             | 当前实现                         |
| `sum`                 | $S_t$，γ=1     | 1                                             | 完整 sequence-KL 的未来信用      |
| `mean`                | $S_t/W_t$，γ=1 | 1                                             | TOP-D 风格 future mean 结构      |
| `discounted_sum`      | $S_t$          | 1                                             | 只衰减未来影响，保留即时项权重   |
| `discounted_mean`     | $S_t/W_t$      | 1                                             | 以折扣权重和作分母的未来加权均值 |
| `discounted_sequence` | $S_t$          | $\gamma^{\tau_t}$，同时乘到 local 与 score 项 | 明确的折扣序列目标               |

表中 response 坐标从 0 开始。最后一行对应

$$
J_\gamma=\mathbb E\sum_t\gamma^{\tau_t}k_t,\qquad
\widehat g_\gamma=\sum_t\gamma^{\tau_t}\left[A_t+\psi_tS_t\right].
$$

仅对 future discount 而保持 local 权重为 1，不等于这个完整折扣目标。Mean 的分母是未来折扣权重和；不另加“除未来 token 数”的第三种归一化。γ=1 时，两种 discounted sum 模式在数学上归约到 `sum`，discounted mean 归约到 `mean`，作为退化测试。

初期单轮纯文本路径通常是连续 mask。带空洞的路径要验证上述时钟选择；若被 mask 的动作也由 student 采样且影响后续状态，省略这些动作的 score 只能解释为 masked surrogate，不能宣称完整序列梯度。

### 3.4 与官方 TOP-D 和现有 MDPO 的边界

官方 TOP-D 的 `_compute_topd_token_returns` 使用 sampled 即时 reward 加严格未来 reward 的均值；共同 launcher 显式设置 `seq-mean-token-mean`。其后还存在 prompt 组内标准化。引用固定 commit 的实现，不把本计划的 Full-vocab mean 称为完整 TOP-D 复现。[^topd-algos][^topd-common]

官方即时 reward 的形式为

$$
r_t^{\rm TOPD}=\log\left[(1-\alpha)+\alpha\frac{T(a_t\mid s_t)}{q(a_t\mid s_t)}\right].
$$

在 $p=q$ 且 $C=(1-\alpha)q+\alpha T$ 时，$c_t=-r_t^{\rm TOPD}$。离开这个更新起点后，当前 $\log(p/C)$ 和固定 anchor 下的 sampled TOP-D reward 不再相同。因此这里复用 future mean 的结构，选择的 future cost 是当前 Full-vocab KL，而非声称逐步等价于官方 sampled update。

现有 [build_log_weights](../../../../../../verl/trainer/distillation/targets.py#L24) 返回未归一化的 b；[full_vocab_rows](../../../../../../verl/trainer/distillation/objectives.py#L49) 的 MDPO 行值为 `target_kl - log_z`。`log_z(s)` 虽对 student 参数冻结，却随状态变化，进入 future 后会改变状态分布梯度，不能当作可随意丢弃的常数。

推荐统一使用实际训练代价行 $j_t$ 构造 future，不增加 `cost_source` 开关。纯 teacher reverse-KL 有 $j_t=k_t$；词表对齐、共同支持集上的 κ=1 MDPO 算术混合目标在理想数学形式下也有 `log_z=0`。一般 κ 的 MDPO 则保持 $j_t=k_t-\log Z(s_t)$，future 中完整保留 normalizer，以延续当前目标。

第 3.2–3.3 节的 suffix 和 surrogate 在一般 MDPO 下将 $k_t$ 换成 $j_t$，local VJP 仍是现有行目标的梯度；$1/\eta$ 继续在最后统一应用。此时 `sum` 对应原始 MDPO 的累计轨迹代价，不能标成归一化 sequence KL；若后续要研究归一化 target KL，应单独定义目标并验证状态项差异。BPO、forward KL、common-conditioned 的 future 语义不在首批接入范围，新增组合先显式拒绝。κ=1 的局部信号现有对照见 [test_mdpo_topd_termination_on_cpu.py](../../../../../../tests/trainer/distillation/test_mdpo_topd_termination_on_cpu.py#L35)。

这里保持目标一致的依据是：$j_t$ 在采样动作前已由状态和当前分布确定，对 $J_j=\mathbb E\sum_tj_t$，因果 score 展开给出

$$
\nabla J_j=\mathbb E\sum_t\left[\left.\nabla j_t\right|_{s_t\,\mathrm{fixed}}+\psi_t\sum_{u>t}j_u\right].
$$

所以 normalizer 的固定前缀导数虽为零，它在未来轨迹代价中的 score 项仍需保留。这个扩展复用同一 suffix 算法，不增加用户调参项。

## 4. 聚合权重与梯度的连接

设回答 $i$ 的有效长度为 $L_i>0$，有效回答数为 $B$，总 token 数为 $N$，token loss 为 $\ell_{it}$，$S_i=\sum_tm_{it}\ell_{it}$。四种主要聚合为

$$
L_{\rm token}=\frac{\sum_iS_i}{N},\qquad
L_{\rm seq\text{-}mean}=\frac1B\sum_i\frac{S_i}{L_i},
$$

$$
L_{\rm seq\text{-}sum}=\frac1B\sum_iS_i,\qquad
L_{\rm seq\text{-}sum\text{-}norm}=\frac1{BC}\sum_iS_i.
$$

$C$ 是跨 micro-batch 和训练更新固定的尺度。公共 `agg_loss` 在未传 `loss_scale_factor` 时回退到 mask 宽度；新 Full-vocab 路径应要求 norm 模式显式传固定值，避免动态 padding 宽度改变训练尺度。

### 4.1 单一权重合同

将每行因子拆为 kernel 前可知的回答权重 $a_i$ 和最终标量 $d$。DP 规模为 $D$，以下因子用于 DP 梯度取平均的执行方式：

| 聚合                         | kernel 前回答权重 $a_i$ | 最终因子 $d$ |
| ---------------------------- | ----------------------- | ------------ |
| `token-mean`                 | 1                       | $D/N$        |
| `seq-mean-token-mean`        | $1/L_i$                 | $D/B$        |
| `seq-mean-token-sum`         | 1                       | $D/B$        |
| `seq-mean-token-sum-norm`    | 1                       | $D/(BC)$     |
| `token-sum`（公共 API 兼容） | 1                       | $D$          |

`LossContext` 负责计算和校验这些因子；数值 kernel 只接受解析好的权重，不接收 Hydra config 或执行通信。公共 `agg_loss` 作为矩阵参考，用一致性测试防止两套语义漂移。新方法不再建立第二个 `full_vocab.loss_agg_mode`。

最终 loss 为 $d\sum_{it}a_i[j_{it}+\operatorname{sg}(F_{it})\log p(a_{it}\mid s_{it})]/\mathrm{divisor}$，其中 $j$ 是第 3.4 节定义的实际训练代价，divisor 沿用现有 η 规则。完整 discounted sequence 模式再按第 3.3 节加入时间权重。全局因子仅应用一次；不得在 kernel 及 loss 层各除一次 token 数。

`B` 必须是当前 optimizer mini-batch 跨 DP 的非 padding、非全 mask 回答数。当前 trainer 传入的是声明的 `global_batch_size`，engine 仅显式重算 token 分母，尚不能仅凭配置保证该值满足新模式。实施时在拆 micro-batch 前解析逻辑回答资格并全局计数；若 padding 身份尚未进入 engine，则在 bridge 保留资格信息。避免修改其它 RL 调用对既有 `global_batch_size` 的依赖；Full-vocab 使用经核对的调用级分母。全局零有效 token/回答报错，局部空 rank 保持合法的零贡献和分布式 collective 顺序。仅在需要回答分母的聚合模式新增相应计数，默认 token-mean 不增加这一 collective。

### 4.2 聚合 surrogate 与精确序列目标

对这四种聚合，回答内的 $a_i$ 相同，故可以乘到 local 和 future score 两项。随机回答长度分母与随机 batch token 分母可能与动作及梯度相关；将它们当固定 rollout 权重应用，是明确的加权 surrogate，不自动保留第 3.1 节的无偏性。完整序列 KL 的基准使用回答 mean、token sum，可附固定 $C$。

以后若引入逐位置代价权重 $w_u$，完整目标的 future 系数应为 $\sum_{u>t}w_uk_u$。不能把未来未加权的和一概乘上当前位置 $w_t$；第 3.3 节完整 discount 的相对/绝对时间因子正是这一规则的特例。聚合层和 suffix 层应保留这个区分，避免为了接口统一而改错目标。

## 5. 计算流水线与模块分工

### 5.1 最小数据合同

以下名称为拟议接口，当前源码尚未提供。

| 拥有者                                         | 输入与输出                                                           | 修改理由                                                                   |
| ---------------------------------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `FullVocabConfig` / resolver                   | `future.mode=none`；discount 模式使用 `future.gamma`                 | 显式校验枚举、有限 γ、$0<\gamma\le1$ 及 backend/目标能力；不支持时提前报错 |
| Engine + padding 工具                          | 构造每回答 offsets、sampled action IDs、原始 response 坐标、有效长度 | 只在已有 hidden 选择处解析布局，kernel 不重建 token 对齐                   |
| `LossContext`                                  | 返回回答行权重、全局标量因子                                         | 统一聚合定义，DP 计数仍在 engine 完成                                      |
| 新 `verl/trainer/distillation/future.py`       | 分段 cost、坐标、γ、模式 → detached suffix 系数                      | 纯张量数学，不认识 TQ、optimizer、teacher checkpoint 或日志命名            |
| `full_vocab.py`                                | 行权重、布局、future spec → local/future 合成 VJP                    | 在 `dH/dW` 合并前完成所有逐行作用                                          |
| `full_vocab_triton.py`                         | 已解析的数值系数及当前块概率统计                                     | 实现行内修正，保持现有稳定求和及 pivot 补偿                                |
| `FullVocabObjectiveResult` / `FullVocabOutput` | 可微训练 numerator 与 detached 诊断分开                              | 不以 surrogate 数值覆盖 KL 统计，保持 FSDP pytree 可见性                   |

只抽取两类可复用数学：归约权重与分段 suffix。暂不建立通用 reward 插件注册表、额外 trainer 阶段或新的缓存服务。Future cost 的合法性由目标解析边界决定；kernel 内不按实验编号分派。

### 5.2 参考路径

小张量 dense reference 保留可微逐行训练代价，生成 detached future，再构造第 3.2 节 surrogate，调用公共 `agg_loss` 验证。它负责验证符号、时间权重、η、布局与梯度；先接受 `[tokens,V]` 小规模显存成本，不作为生产速度结论。

Teacher、anchor 均 detach；当前 student hidden 和 head 都保留梯度。纯 teacher 目标、κ=1 MDPO 和一般 κ 的 MDPO 分别验证；一般 κ 必须核对保留 `log_z` 的 future 与完整轨迹代价梯度，避免只覆盖 teacher-only shortcut。

### 5.3 生产分块候选与实现取舍

CPU 实现采用单投影的反向分块路径，数学工具收敛到 `reduction.py`，执行桥接放在 `trajectory.py`，没有增加 reward 注册表。实际按回答切块，以标量 carry 连接相邻块；doubling scan 的每块复杂度为 $O(n\log n)$。该选择让回答边界与 mask 时间距离可独立检验，但不同回答之间不合并 GEMM，小回答的成本要在 GPU 上测量。既有 fused 行算子先计算 local VJP，score 在 Torch 中合成并重建 pivot；尚未把 score 融入 Triton。以下为设计时的候选流水线，其性能目标仍需 profiler 验收。

优先验证逆序 token chunk 的单次 head 投影路径：

1. Student backbone 仍做一次 teacher-forced forward；按既有预测位置选 hidden，进入包装 head 回调。
2. 按扁平 token 行逆序取 chunk。在当前块投影 student/teacher/所需 anchor，得到实际训练 cost、local VJP 所需统计及 sampled log-prob；MDPO 保留目标 normalizer。
3. 结合当前块右侧的 suffix carry，在每条回答内求严格未来 $S,W$。块可跨回答，边界必须 reset；块内包含部分回答时，保存其 carry 供前一块继续使用。
4. 在释放当前块的概率/normalizer 前，形成 local 聚合权重及 future score 系数，将两项合入 `dZ`，随后执行原有 `grad_hidden/grad_head` GEMM。
5. 保存最终 `dH/dW`，只保留需要的 $O(N)$ 行诊断/系数。Backward 继续接受最终全局标量 upstream，沿用 engine 的梯度累积、裁剪与 optimizer。

第 2–4 步要求重构现有 row-kernel 边界：当前接口生成 `dz` 后已丢弃概率统计。候选实现应将“统计/局部代价”和“应用 local/future 系数”在当前 chunk 内连接，禁止从已有 `dz` 反推概率。工程目标是复用当前块 logits/normalizer，是否做到零额外投影需由 profiler 确认。

普通 `precompute` 先通过梯度参考，再接 `precompute_fused`。Triton 的 [weighted_rows](../../../../../../verl/trainer/distillation/full_vocab_triton.py#L176) 名称指其残差数值算法，不是现成的回答聚合接口。修改 [_weighted_apply / _weighted_finish](../../../../../../verl/trainer/distillation/full_vocab_triton.py#L96) 或 [fused_mirror_logits](../../../../../../verl/trainer/distillation/full_vocab_triton.py#L442) 时，要把新增项纳入正确的 pivot 行补偿，验证 logits 梯度零和，不能在补偿完成后任意拼接破坏稳定性的项。

若逆序融合不便复用既有数值结构，备选是先求 $O(N)$ cost 与 suffix，再重算块 VJP；它更易与 reference 对齐，但增加 head 投影。只有明确的正确性/显存收益或测量证据才采用该备选，不预先承诺所有 backend 都有单投影性能。`recompute` 可利用逐行 upstream 扩展聚合；只返回标量且缺少所需行统计的 Liger 路径，对不支持的权重/future 组合明确拒绝，禁止静默切换 backend。

### 5.4 EOS 与支持集

[termination_equivalent_kl_rows](../../../../../../verl/trainer/distillation/rows.py#L32) 在合并终止动作的空间训练，同时保留原始词表 KL 诊断；mirror 路径也有 coarsen/expand。Future cost 应与实际训练动作空间一致，不能为了日志方便累计 raw-vocabulary KL。合并终止动作在同一单轮终止语义下测试，终止处 future 为 0；若 future score 使用合并动作，按动作映射展开梯度。

首批生产验收包括现有 E57 类路径所需的终止动作等价，但必须与 dense reference 单独核对；验收前不宣称已支持。条件支持集可能改变行为采样分布与 score 的含义，不能只把 KL 的 support 切片代码复制过来。现有不带 future 的支持集能力保持原状，新增 future 的未验收组合显式报错。

### 5.5 计算量与并行性

| 成本          | 预期变化与需验证内容                                                                      |
| ------------- | ----------------------------------------------------------------------------------------- |
| Backbone / TQ | 不增加 backbone 次数，不增加 teacher hidden 运输；每次 inner update 刷新当前 student cost |
| Head 投影     | 逆序候选争取复用当前 chunk；双遍备选会增加投影，必须分别测量                              |
| Suffix        | 一维分段扫描约 $O(N)$；长回答跨块有依赖，不能把所有 chunk 当独立任务                      |
| 词表梯度      | score 修正包含对 $p$ 的逐元素处理；未融合会增加 $O(NV)$ 内存流量，不能称为完全免费        |
| 显存          | 不长期缓存全序列 `[N,V]`；保持当前 chunk 及 $O(N)$ 元数据，实际峰值待测                   |
| 并行          | 保留块内词表计算、不同回答及 DP 并行；扫描可用分段 scan，跨块 carry 有时序约束            |
| 数值          | 逆序改变 `dW` 累加顺序；要求误差与梯度方向验收，不能要求逐 bit 一致                       |

“保持现有快路径”指 `token-mean + future=none` 仍调用原来的 dispatch 和 reduction，不支付 suffix 或通用权重分配成本。新模式按能力选择已验证路径，不把性能假设当接口保证。

## 6. 指标和配置可审计性

[FullVocabOutput.from_result](../../../../../../verl/trainer/distillation/output.py#L21) 当前从 `loss_sum` 生成部分主指标，加入 future 后必须调整来源。保留 `distillation/full_kl` 及其位置统计的原始 KL 语义；MDPO 的原 local objective、target/teacher/anchor KL 和 target normalizer 也不受聚合或 future surrogate 值污染。

新增观测分清：即时训练代价、future 系数分布、future score surrogate、合成训练 surrogate、有效 token/回答分母，以及聚合后的训练尺度。原有 token 诊断仍用自己的计数，不拿 micro-batch 均值再简单平均。η 只按同一目标合同作用一次；日志注明某项是未缩放 cost 还是最终训练量。

Resolved config 记录 `loss_agg_mode`、固定 $C$、future 模式、γ、实际训练动作空间和 backend；验证/任务分数不改定义。检查未来系数在末 token 为零、不同长度的系数分布及 local/future 梯度范数，可帮助解释消融，但不以这些诊断替代任务结果。

## 7. 实施阶段与验收

### 7.1 按依赖拆分代码修改

| 阶段              | 主要文件                                                      | 通过条件                                                 |
| ----------------- | ------------------------------------------------------------- | -------------------------------------------------------- |
| A：聚合合同       | `loss_context.py`、`losses.py`、engine 分母、launcher         | 四种主聚合与公共参考一致；旧 `token-mean` 路径回归通过   |
| B：future 参考    | `distillation.py` / resolver、`reduction.py`、dense reference | 六种模式、边界和有限轨迹枚举梯度通过；未支持组合早期拒绝 |
| C：普通生产路径   | `full_vocab.py`、engine 布局、result/output                   | `precompute` 的 hidden/head 梯度与参考一致；指标不混淆   |
| D：fused 与分布式 | `full_vocab_triton.py`、FSDP2 集成测试                        | EOS、数值补偿、空 rank、DP/micro/chunk 一致性通过        |
| E：性能与实验交接 | 性能基准及对应当前参考文档                                    | 同输入性能比较完成，记录实际成本；正式训练消融另排       |

A–C 的 CPU 验收包括真实 tensor 校验、配置/布局与数值参考、trainer 初始化和 update 编排。数值测试分别覆盖 [suffix 与聚合合同](../../../../../../tests/trainer/distillation/test_future_credit_on_cpu.py)、[独立参考、有限轨迹枚举和分块 VJP](../../../../../../tests/trainer/distillation/test_trajectory_vjp_on_cpu.py)、[配置与训练接入](../../../../../../tests/trainer/distillation/test_full_vocab_reduction_on_cpu.py)。Trainer 入口测试见 [test_trainer_base_on_cpu.py](../../../../../../tests/trainer/ppo/v1/test_trainer_base_on_cpu.py)，覆盖五种聚合和既有 batch 约束；[输出合同测试](../../../../../../tests/trainer/distillation/test_full_vocab_output_on_cpu.py) 覆盖回答权重、η、DP 尺度及 local/surrogate 指标分离，[只读评测测试](../../../../../../tests/trainer/distillation/test_fixed_rkl_readout_on_cpu.py) 验证训练聚合/future 不进入 readout。CPU 回归为 `926 passed, 18 skipped`，命令如下；不适用的非 discounted 模式与非单位 γ 组合显式跳过。D 已增加 CUDA 测试与 FSDP2 参数入口，执行待 GPU 空闲；E 尚未测量。CPU 通过不等于 D/E 完成。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest tests/trainer/distillation/*on_cpu.py tests/trainer/ppo/test_loss_aggregation_on_cpu.py tests/trainer/ppo/v1/test_trainer_base_on_cpu.py -q
```

### 7.2 已有实现与测试复用

| 已有测试                                                                                                                                                                                                                           | 可复用的覆盖与需补充的验收                                                  |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| [test_loss_aggregation_on_cpu.py](../../../../../../tests/trainer/ppo/test_loss_aggregation_on_cpu.py)                                                                                                                             | 已有 `token-sum` 的 DP/micro-batch 性质；新增四种聚合的 Full-vocab 权重对照 |
| [test_loss_context_on_cpu.py](../../../../../../tests/trainer/distillation/test_loss_context_on_cpu.py)                                                                                                                            | 调用级分母、配置不被 loss 修改；补非均匀长度、padding 和实际回答数          |
| [test_full_vocab_on_cpu.py](../../../../../../tests/trainer/distillation/test_full_vocab_on_cpu.py)                                                                                                                                | 参考值与梯度；补每行权重、future 及不同 chunk 边界                          |
| [test_full_vocab_output_on_cpu.py](../../../../../../tests/trainer/distillation/test_full_vocab_output_on_cpu.py)                                                                                                                  | η/指标缩放和 FSDP 可见 tensor；补 local/true KL/future surrogate 分离       |
| [test_mdpo_topd_termination_on_cpu.py](../../../../../../tests/trainer/distillation/test_mdpo_topd_termination_on_cpu.py)                                                                                                          | κ=1 即时信号与完整更新的区别、EOS coarsen 梯度；补 future 组合              |
| [test_backend_contract_on_cpu.py](../../../../../../tests/trainer/distillation/test_backend_contract_on_cpu.py)                                                                                                                    | fused 明确要求 CUDA、无隐式 fallback；补目标/backend/future 能力矩阵        |
| [test_termination_fused_on_gpu.py](../../../../../../tests/trainer/distillation/test_termination_fused_on_gpu.py)、[test_mdpo_termination_on_gpu.py](../../../../../../tests/trainer/distillation/test_mdpo_termination_on_gpu.py) | GPU 数值及终止等价；补 score 修正、pivot 补偿和误差范围                     |
| [test_full_vocab_fsdp2.py](../../../../../../tests/special_distributed/test_full_vocab_fsdp2.py)、[test_mirror_fsdp2.py](../../../../../../tests/special_distributed/test_mirror_fsdp2.py)                                         | 包装 head、backbone/head 同时更新和 anchor 生命周期；补新模式真实通信       |

CPU 数学与接口验收覆盖：

- 小词表、短时域精确枚举轨迹，将解析 $\nabla J$ 与 sampled/Full-vocab future-sum 期望逐项比较；完整 discount 同样枚举，mean 单独与其 surrogate 比较；一般 κ 的 MDPO 额外枚举原始轨迹代价，检验状态相关 normalizer。
- 序列长度 0/1、多回答、mask 空洞、EOS、固定时域及截断尾部；future 不含当前项且绝不跨回答。
- γ=1 退化关系、γ 接近 0 的行为、非法参数及没有未来时的零分母处理。
- 对 student hidden/head 分别检验梯度，teacher/anchor 无梯度；验证 future detach，避免重复求导。
- 非均匀长度、不同 DP 划分、动态 micro-batch、跨序列/跨 token chunk 边界，与一次完整矩阵归约一致。
- 原 `none + token-mean` 结果与梯度不变；其它即时项目标仍保持原接口语义。

GPU/分布式验收等待资源空闲后单独执行。通过 CPU 参考不能替代真实 FSDP hooks、collective、FP16/BF16 和 fused 数值验证。CUDA 新模式入口为 [test_trajectory_on_gpu.py](../../../../../../tests/trainer/distillation/test_trajectory_on_gpu.py)；[Full-vocab FSDP2 集成测试](../../../../../../tests/special_distributed/test_full_vocab_fsdp2.py) 新增 `--loss-agg-mode`、`--loss-scale-factor`、`--future-mode`、`--gamma`，保留重复更新、空 local mask、head/backbone 和 checkpoint 检查。可用 `--backend precompute --loss-agg-mode seq-mean-token-mean --future-mode mean` 检查 TOP-D 风格结构，再以 `--future-mode discounted_sequence --gamma 0.9` 检查完整折扣路径；这两组尚未运行。

### 7.3 性能验收与消融交接

以[现有 Full-vocab 性能基准](../../../../docs/infra/performance/full-vocab/README.md)为入口，固定模型、精度、输入 token、长度分布、backend、chunk、DP 和 micro-batch，比较旧快路径、新聚合、future sum、future mean 及 discount。为了区分遍历顺序与新增计算，先固定相同 token 块仅切换遍历顺序，再比较跨回答/按回答切块，最后比较 future score 与 suffix。无 future 保留顺序路径；future 只在每条回答内逆序分块，backbone causal forward 不变。分别记录 head/kernel、suffix、反向、optimizer 和整步耗时、tokens/s、峰值显存及投影次数；同时检查梯度误差，不以吞吐改进掩盖目标变化。性能比较需要相同 warmup 和重复测量；允许的预算在正式测量前确定，不在未测量时承诺固定开销百分比。

工程验收后，科学实验先固定预算和聚合，只切换 `none/sum/mean`；再独立比较聚合及 discount。大 batch、小 batch、rollout reuse 和未来信号的交互保留为后续研究轴，避免首轮同时改动导致无法归因。具体模型、seed 和训练预算继续写入 [weighted targets](../../../../docs/experiments/learning-signal/weighted-targets.md)，此处不预先分配新的实验编号。

### 参考文献

[^topd-algos]: xie-lab-ml. TOP-D `core_algos.py`. 固定 commit `9a2716cba22cab508825ad3be3f78327bb8139e5`，[即时 reward、future mean 与 prompt 标准化实现](https://github.com/xie-lab-ml/TOP-D/blob/9a2716cba22cab508825ad3be3f78327bb8139e5/verl/verl/trainer/ppo/core_algos.py#L370-L440)。源码核对日期：2026-11-03；本地既有核验产物见 [topd-comparison.json](../../../../docs/experiments/learning-signal/data/e55-e57-20261101-2259/topd-comparison.json)。

[^topd-common]: xie-lab-ml. TOP-D 共同启动脚本。同一固定 commit，[显式 `seq-mean-token-mean` 配置](https://github.com/xie-lab-ml/TOP-D/blob/9a2716cba22cab508825ad3be3f78327bb8139e5/recipe/opd/run_qwen3_1p7b_base_qwen3_teacher_dapo17k_common.sh#L343)。其公开配方与本计划仅借用的 future-mean 结构应分开比较。
