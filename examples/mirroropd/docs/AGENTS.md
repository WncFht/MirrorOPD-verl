# 文档归属与检查

本层执行 [maintain-docs](../.agents/skills/maintain-docs/SKILL.md) 和 [maintain-prose](../.agents/skills/maintain-prose/SKILL.md)，保留父层规则。

- `experiments/` 的四条线固定为训练信号、reuse 预算、rollout 来源/混合的最终 acc、师生迁移机制。Polaris 固定 state 和跨 batch 实验归迁移线，Qwen 修复及 E54 诊断归 infra。
- `settings/` 只维护公共资料和参数含义；每项实验报告记录自己的实际 setting、预算及证据边界。完全蒸馏教师的定义暂留开放，不新增评价门槛。
- 理论定义归 `theory/`，工程合同和验收归 `infra/`，外部资料归 `literature/`。同一读者问题只维护一份正文；先保留独有证据，再合并重复和修复入链。
- 正文按自然段写，不硬换行。使用数学环境和 Markdown 表格；格式化保护代码块、公式和 HTML 属性，不自动修改科研数值。
- `archive/`、`data/`、`sources/`、原始模型卡及证据源码保持冻结。活动页面进入这些目录的引用仍核对路径，缺少仓外证据单独报告。
- 不把同步排障、凭据、GPU 清理和执行流水写进研究正文。计划、完成、负结果和待验证边界如实区分。
- `make docs-check` 验证格式、结构、引用、源码符号和规则；语义复核仍需阅读正文及证据。检查入口见[维护流程](infra/maintenance/README.md)。
