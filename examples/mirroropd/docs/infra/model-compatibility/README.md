# 模型兼容、动作空间与接入验收

本目录维护输入模板、token-ID、输出支持、EOS、head 适配与工程诊断。相同模型家族或可解析配置不代替真实接入验收；科学配对和结果进入[教师吸收研究](../../experiments/teacher-transfer/README.md)。

| 核对任务                               | 正文                                                                                                                                                                                 |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 原生/共同支持、排除质量与 ID 映射      | [动作空间与配对合同](action-space.md)                                                                                                                                                |
| 生成停止与训练动作合并                 | [终止动作配置](termination/2026-10-26-终止动作对齐的配置与实现.md)、[受控诊断](termination/2026-10-26-终止动作与循环退化诊断.md)及[同题样本](termination/2026-10-26-同题样本对照.md) |
| 历史 teacher 输入、词表和短测          | [实现对齐诊断](probes/2026-10-24-OPD实现验收与对齐诊断.md)，保留原版本范围                                                                                                           |
| Polaris 不同 head 宽度与共享输入       | [Polaris 接入](polaris.md)                                                                                                                                                           |
| MiMo 下一 token 路径与 MTP 关闭        | [MiMo 验收](mimo/2026-11-01-MiMo原生下一token接入验收.md)                                                                                                                            |
| Olmo3 滑窗和 Light-R1 共同输入协议     | [阶段链接入](stage-chains/2026-11-01-Olmo3与LightR1阶段链接入验收.md)                                                                                                                |
| Qwen Base 长度膨胀的工程前提与故障证据 | [E54 诊断](qwen-e54/2026-10-29-E54长度膨胀诊断.md)                                                                                                                                   |

κ/η的科学比较与 E55–E57 结果统一见[TOP-D 复现与消融](../../experiments/learning-signal/weighted-targets.md)。模型阶段的公开资料见[模型参考](../../settings/模型与师生配对.md)，SFT/SFT+RL 的科学预算见[阶段对照](../../experiments/teacher-transfer/post-training-stages.md)。
