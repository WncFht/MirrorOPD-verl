# MirrorOPD 维护规则

本层补充仓库根指令。文档组织与研究边界见 [docs/AGENTS.md](docs/AGENTS.md)；工具入口见[维护流程](docs/infra/maintenance/README.md)。

- 文档编写和归位使用 [maintain-docs](.agents/skills/maintain-docs/SKILL.md)，句级审阅使用 [maintain-prose](.agents/skills/maintain-prose/SKILL.md)。仅有长期取舍时使用[决策记录](.agents/skills/maintain-decision-notes/SKILL.md)。
- 来源先改，生成内容随后更新。不把当前默认设置写回旧实验；不自动增加训练、GPU 或联网实验。
- 格式化执行 `make docs-format`；完整文档检查执行 `make docs-check`；维护工具变更执行 `make maintenance-check`。Python 继续使用根仓库的 Ruff 与已有测试。
- 冻结数据、来源模型卡、历史归档和源码快照不参加格式化。自动工具不删文、不按篇幅压缩推导、不判断科学结论。
- Git hook 的维护检查只读暂存快照，不运行格式化、不改 index。不覆盖仓库已有 hooks 或安装全局技能。
- 小实验按用途归位：科学机制在研究线，公式验算在 theory，执行和算子检查在 infra。研究报告按背景、实际 setting、结果、分析组织，实际配置仍在各报告中。
