# 模型兼容性与生成语义

这里维护模板、token-ID、输出支持、终止动作和模型接入诊断。Qwen Base 修复及 E54 异常不作为核心研究分支。

- [终止动作受控修复](termination/2026-10-26-终止动作与循环退化诊断.md)与[同题样本](termination/2026-10-26-同题样本对照.md)。
- [E54 长度诊断](qwen-e54/2026-10-29-E54长度膨胀诊断.md)。
- [输出支持、模板与对齐验收](probes/2026-10-24-OPD实现验收与对齐诊断.md)。
- [Polaris 设置和工程验收](polaris/2026-10-28-Polaris设置与工程验收索引.md)。

修正后的正式任务成绩在[师生配对实验](../../experiments/teacher-transfer/teacher-comparison/README.md)中报告，并引用已明确的工程条件。
