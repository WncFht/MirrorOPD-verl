# Sampled OPD 的 full-KL baseline 扩展

本页定义尚未接入的 sampled+full-KL baseline。已经可运行的 direct Full-KL、MDPO/BPO 见[目标与实现](../../architecture/full-kl-walkthrough.md)，两类方法的性能不能混用。

## 目标与接口

固定前缀上的 current baseline 使用无梯度标量 $K=\mathrm{KL}(p\Vert T)$，与动作上的 sampled 信号组成：

$$
A(a)=\operatorname{stopgrad}[\log T(a)-\log p(a)+K].
$$

完整支持、正确行为 IS 与停止梯度是控制变量解释的条件；PPO clipping、ratio guard 和动作筛选需另行分析。不能把当前动作自己的 k1 同时作为 baseline，否则会与负 k1 信号抵消。理论见[条件梯度与控制变量](../../../theory/2026-10-22-条件梯度估计与控制变量.md)。

数值接口需要返回每位置 detached KL，训练接口还需要可微 sampled log-prob；二者共享 student forward。不要为 baseline 再执行一次 student backbone，也不能把只返回可微 full loss sum 的接口直接当成 baseline 接口。

## Current 与 anchor 两种生命周期

| 选择 | 数值来源 | Reuse 中的行为 |
| --- | --- | --- |
| Current | 当前 student hidden/head 与冻结 T | 每次更新重算 KL；可融合 sampled score 与 KL 统计 |
| Anchor | 更新前 q 与 T | 整个 actor 调用固定 scalar，后续不再等于 current KL |

Current 首版可以复用已有 student logits，再分块重建 teacher；必须记录保留完整 student logits 的内存代价。优化版在 head 前接入，联合返回可微 sampled score 与 detached KL，共享投影/归一化。若从 detached hidden 重投影，额外 head 成本需单列。

Anchor 版可在更新前评分中产生 $K(q,T)$，随后仅保留每位置 scalar并释放不再需要的 hidden/head。该方案必须保留 q 的版本、预测位置和样本身份；不能因为首次 p=q 就把其来源标为 current。

## 接入边界

复用[hidden/head 通路](../../architecture/full-vocab.md)与[mask/全局分母](../../architecture/data-and-loss.md)，训练归约由 sampled loss 负责。Baseline 的 kind、current/anchor 来源与目标校验是显式配置，不由 `use_liger`、模型 fused 开关或现有 Full-KL backend 推断。

Teacher 同时需要 sampled score，可与 KL 重建统计共算或以独立小 payload 提供。支持内归一化、top-k baseline、低秩/量化 teacher head 都属于独立目标或近似，不能作为 exact full baseline 的透明替换。

## 验收与资源判断

按独立 dense oracle检查数值与 sampled score 梯度，枚举有限动作验证行为 IS 的期望梯度，确认 baseline detached，核对 current/anchor 的刷新、mask/shift、空 rank、FSDP hooks和恢复边界。然后比较固定输入单步参数变化、短训练以及完整任务质量。

资源公式、value-only 原型与原始数据见[算子性能](../../performance/full-vocab/operators.md#value-only-与近似方案的边界)。Value-only 时间未包含 sampled backward，完整训练收益仍待测。无可运行 baseline 配置前，不把它加入生产性能排名。
