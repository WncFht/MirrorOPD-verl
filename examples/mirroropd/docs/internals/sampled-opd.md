# Sampled-token OPD 目标与梯度路径

Sampled OPD 在生成到的 response token 上构造教师监督。符号体系见[符号约定](../符号约定.md)，固定前缀目标的完整推导见[理论入口](../theory/README.md)。本页负责把数学对象映射到实际字段和分支。

## 四种概率

| 概率 | 字段或来源 | 同一批复用时 |
| --- | --- | --- |
| 行为策略 $\mu$ | `rollout_log_probs`，生成时记录 | 固定 |
| 更新前锚点 $\pi_{old}$ | `old_log_probs`，通常由 actor 在更新前重算 | 固定；bypass 可改变来源 |
| 当前 student $\pi_\theta$ | 每次 forward 的 `model_output['log_probs']` | 随参数改变 |
| Teacher $T$ | `teacher_logprobs` | 冻结 |

Behavior 与 anchor 即使来源模型相同，也不能视为同一字段。推理和训练实现可能有差异；bypass 配置另有明确语义。生成数据的分布和当前模型在固定前缀上的分布也需分别考虑。

## 入口与模式分流

[distillation_ppo_loss](../../../../verl/trainer/distillation/losses.py) 处理蒸馏与通用 policy loss 的组合。[distillation_loss](../../../../verl/trainer/distillation/losses.py) 先处理 Full-KL，再将非 `vanilla` 模式送入 [_sampled_token_loss](../../../../verl/trainer/distillation/losses.py)。共享 sampled 配置要求 `k1 + use_policy_gradient=True`，且不混合任务奖励。

[DistillationLossConfig](../../../../verl/workers/config/distillation.py) 验证模式和参数。`advantage_source` 的 frozen 选项在配置里写作 `rollout`，不是 `frozen`。

## 显式 sampled 分支

在有效 response token 上，定义：

$$
A_{current}=\operatorname{stopgrad}(\log T-\log\pi_\theta),\qquad
A_{rollout}=\log T-\log\mu.
$$

Teacher、behavior 和两种 advantage 均不参与反传。实现先用 mask 将无效位置的概率置零，再进行减法、指数和乘法，避免 padding 中的 NaN 污染 loss 或梯度。

| `policy_loss_mode` | 逐 token surrogate | 梯度来源 |
| --- | --- | --- |
| `reinforce` | $-A\log\pi_\theta$ | 当前 log-prob |
| `token_is` | $-rA$ | 当前概率与 behavior 比值 |
| `token_ppo` | $\max(-rA,-\operatorname{clip}(r,1-\epsilon_l,1+\epsilon_h)A)$ | 未被 clipping 截断的当前比值分支 |

其中实际比值含数值保护：

$$
r=\exp\bigl(\operatorname{clip}(\log\pi_\theta-\log\mu,-20,20)\bigr).
$$

这个 log-ratio 保护与 PPO clipping 是两个不同操作。最终通过 [LossContext.aggregate](../../../../verl/workers/utils/loss_context.py) 使用调用级全局分母聚合。动作 IS 改变固定前缀上的 token 权重，不会自动修正整个前缀的状态分布。

## Vanilla 分支

`vanilla` 调用已有 [compute_policy_loss_vanilla](../../../../verl/trainer/ppo/core_algos.py)。它使用 `old_log_probs` 作比值分母，包含既有 PPO/dual-clip 行为，并可接收外部 rollout correction 权重；名称不表示“没有 clipping”。其 current k1 信号取负后 detach，作为 policy advantage。

因此，`vanilla`、`reinforce`、`token_is`、`token_ppo` 不能只按名称相互替代。特别是 vanilla 的 anchor 分母与显式 sampled 分支的 behavior 分母不同。字段、布局与分母必须一起核对。

## 观测与扩展

训练 surrogate、sampled log-ratio 和完整词表 KL 是不同量。逐 update 指标描述产生该次更新的 forward，不代表更新后对同一 batch 的重测结果。字段定义见[指标字典](../experiments/metrics/指标字典.md)，统计出口见[指标系统](metrics.md)。

新增 sampled 变体应优先复用 `_sampled_token_loss` 的布局、detach 和归一化规则，保持原有模式数值不变。

## 验证入口

[test_sampled_token_loss_on_cpu.py](../../../../tests/trainer/distillation/test_sampled_token_loss_on_cpu.py) 检查目标、梯度、mask 和模式差异；[test_shared_teacher_fsdp2.py](../../../../tests/special_distributed/test_shared_teacher_fsdp2.py) 覆盖 current-IS、rollout-PPO、多次复用和恢复。数学比较不以四次短训练 loss 的升降替代。
