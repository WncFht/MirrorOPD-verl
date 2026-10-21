# 旧 Correction 辅助字段与新输出的区别

Last updated: 2026-10-22。

本页整合原 rollout correction 文档中的指标说明，用于解释历史日志和 `rollout_corr_helper.py` 的内部返回值。当前 V1 对外输出由[指标字典](2026-10-22-指标字典.md)定义；它不会为了兼容而双写下面的旧字段。算法本身的 IS/RS 配置继续在[rollout correction](../../../../docs/algo/rollout_corr.md)维护。

## 输入角色与支持

内部帮助函数比较 old_log_probs 与 rollout_log_probs。Decoupled 模式的 old 是本批入口冻结训练锚点；bypass 路径可能使用当前训练策略。只有对应权重版本一致时，才能把偏移单独解释为训练/推理引擎差异。当前新输出分别使用 anchor_behavior 和 current_behavior，名称不随配置静默改变角色。

## 内部统计口径

| 旧字段族 | 内部含义 | 新输出中的处理 |
| --- | --- | --- |
| `rollout_is_mean/std/min/max` | correction 帮助函数处理后的权重统计，具体路径包含 safety bound、TIS 或置零 | 当前 E 系列没有额外 rollout correction，不输出一套虚构的 applied IS；raw pair 使用明确角色的新字段 |
| `rollout_is_eff_sample_size` | 归一化非负权重的 ESS/N，帮助函数使用处理后的权重 | 新 current_behavior ESS 用原始 token ratio 的 log-space 状态；二者不视为同一个测量 |
| `rollout_is_ratio_fraction_high/low` | 超出配置上下界的原始权重比例，处理层级依 helper 配置 | 当前 raw pair 的 0.5/2 尾部比例固定为描述刻度，不等于 PPO clip |
| `rollout_is_seq_*` | 将权重按 sequence 汇总后的统计 | 当前 token 配方不常驻输出 |
| `rollout_rs_masked_fraction` | RS 在原有效 token 集合上拒绝的比例 | 原 mask 与训练使用的 mask 分开；当前 E 配方无 RS |
| `rollout_rs_seq_masked_fraction` | 至少一个 token 被 RS 拒绝的 sequence 比例 | 不把 token 比例和 sequence 比例互换 |
| `kl` | rollout−old 的 sampled log-prob 差均值 | 有限样本值可为负，不等于精确 KL 变成负值 |
| `k3_kl` | exp(z)−1−z 的 sampled 均值 | 新入口字段为 policy/anchor_behavior/k3；需满足采样与支持条件 |
| `chi2_token` | 带数值保护的 mean(w²)−1 | 有限样本估计可以为负，不能据此断言真实 χ² 负值；新默认集合不保留 |
| `chi2_seq` | sequence product ratio 的二阶量 | 长序列尾部敏感，不作为新默认稳定性面板 |
| `training_ppl`、`rollout_ppl` | per-sequence exp(−mean log-prob)，再对 sequence 平均 | 不等于 token 级语料 PPL；新默认集合不保留冗余衍生量 |
| `log_ppl_diff` | 对 sequence 内平均的 rollout−old log-prob 差再汇总 | 先按 sequence 平均，不能与全局 token 加权 log-ratio 均值直接等同 |
| `log_ppl_abs_diff` | 上述 per-sequence 差的绝对值均值 | 不等于逐 token 绝对差均值 |

旧说明中“mean ratio 接近 1 表示没有偏移”“sample χ² 一定非负”等简化解释不作为本项目判断规则。尾部和均值需要联合观察；数据也必须来自所声明的分母分布。参考公式见[数学说明](../../../../docs/algo/rollout_corr_math.md)。

## 文档职责

- [当前字典](2026-10-22-指标字典.md)：新输出的字段、单位、采集时刻与默认横轴。
- [历史字段处置](data/metrics-system-20261021/metric-migration-20261022.csv)：已观察 493 个旧键的替换/删除决定。
- [历史性能口径](2026-10-22-历史性能口径与验收.md)：timer、显存及历史 GPU 验收的适用边界。

代码依据：[rollout_corr_helper.py](../../../../verl/trainer/ppo/rollout_corr_helper.py)、[policy_metrics.py](../../../../verl/trainer/ppo/policy_metrics.py)。旧函数仍服务其他算法入口，保留内部 API 不代表对外字段兼容。
