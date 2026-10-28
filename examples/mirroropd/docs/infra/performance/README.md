# 工程性能与资源测量

相同目标与工作量下比较执行成本，区分固定前缀算子测量、自然生成完整批和整轮训练；周期 GPU 观测不替代连续 trace。

- [Full-vocab 性能专题](full-vocab/README.md)。
- [Sampled/full公平性能方案](sampled-full/2026-10-26-Sampled与Full公平性能方案.md)。
- [学生推理参数实测](2026-10-27-学生模型推理性能与参数选择.md)。
- [Qwen 完整验证资源](inference/2026-10-27-并发64验证阶段资源实测.md)与[并发估计](inference/2026-10-27-验证并发的逐Checkpoint估计.md)。
- [训练与验证测量](2026-10-22-训练与验证性能实测.md)。

Reuse 和 response cap 如何改变学习效果与预算，归[预算研究](../../experiments/reuse-budget/README.md)。

[模型验证资源与并发标定](inference/2026-10-28-模型验证资源与并发标定.md)保存历史 A–F 的接口独立测量，按冻结条件解释。
