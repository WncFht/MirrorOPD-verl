# 文档维护流程

本项目采用 agent-maintenance 的文档归位、文字审阅、决策记录、Markdown 解析和暂存快照方法，复用现有 Make、pre-commit、Ruff 与项目测试。来源及许可见[适配说明](../../../tools/maintenance/SOURCES.md)。

## 使用

从仓库根目录运行：

```bash
make docs-setup
make docs-format
make docs-check
make maintenance-check
```

`docs-setup` 为本项目工具安装 npm 锁定依赖及 Python 解析依赖，不修改训练依赖声明。也可在单独的 Python 环境安装后通过 `PYTHON=/路径/python make docs-check` 使用。

`docs-format` 使用 AutoCorrect 整理正文间距，再用 Prettier 统一布局。代码块、公式和 HTML 属性受到保护，嵌入代码不由 Prettier 重排。格式化可能改变标题锚点，修改后需修复入链。运行表由原生成器刷新后再执行格式化，不手工修改其分数。

`docs-check` 是只读检查，依次检查格式、Markdown lint、解析后的链接/图片/HTML、锚点、源码符号、Python 示例、skill 元数据及 Notes。常驻规则硬超限会失败；长研究正文只给审阅提示。缺依赖或空扫描明确失败。

## 范围与证据

扫描 `examples/mirroropd` 的维护 Markdown，包括未提交文件和 `.agents`。排除 archive/history、data/sources、原始模型卡、Notes 封存、依赖目录和符号链接。格式化与检查共享同一选择规则；可用下面的命令查看实际输入：

```bash
python examples/mirroropd/tools/maintenance/pipeline.py list
```

相邻 `wandb-logs`、`MirrorOPD`、`model`/`models` 和 `runs` 可作为明确声明的仓外证据。当前机器没有这些证据时，统计为未核验引用，不能声称内容已验证。`data/` 和 `archive/evidence/` 中未随源码发布的原始记录也单独计数；当前源码、文档、图片与锚点的缺失仍失败。检查不访问外部网站，不启动训练或 GPU，也不验证数学和实验解释的正确性。

冻结 Python 不进入 Ruff、mypy、许可头或 compileall hooks，第三方原文与归档不适用项目名称拼写检查。[Git 属性](../../../../../.gitattributes)保留来源附件、图表和 CSV 的原始空白；维护正文仍接受格式与空白检查。

## 提交与 CI

现有 pre-commit 的 MirrorOPD 条目导出 Git index 快照后检查，未暂存修复不会替代待提交内容。它不运行 formatter、不改写 index、不安装或覆盖 Git hooks。部分暂存时，先审阅格式变化，再按原工作流暂存需要的部分。

CI 复用现有 pre-commit workflow 安装锁定工具并检查提交快照；没有新增发布流程或第二个调度器。仓库已安装的 pre-commit 入口继续生效，远端 required checks 是否启用不由本地配置保证。

## 内容审阅

工具通过后，按[文档技能](../../../.agents/skills/maintain-docs/SKILL.md)核对事实归属，再按[文字技能](../../../.agents/skills/maintain-prose/SKILL.md)清理会话残留和失效计划。保留实际 setting、负结果、必要推导和证据限制。四条研究线及各实验独立 setting 的约定见[实验维护说明](../../experiments/维护说明.md)。
