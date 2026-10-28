# 来源与适配

维护规则、Markdown/HTML 解析、Notes 检查和暂存快照方案来自本机 `agent-maintenance`，基点为 `045f4dcea71628c3f1209355d16e06209b109a9a`。规则采用接入时工作区内容，来源仓库含未提交的维护规则改动，不能将所复制规则称为该提交的逐字副本。许可原文见 [LICENSE.agent-maintenance](LICENSE.agent-maintenance)。

MirrorOPD 保留已有 Make、Ruff、源码符号/示例检查与测试。适配内容包括子项目扫描边界、仓外日志证据、长理论文档、真实暂存快照和受保护 Markdown 内容。没有引入第二套 checks.json、全局技能安装器、自动发布、MPE 或来源项目的历史 Notes。

格式工具由本目录的 npm 声明和锁文件固定：Prettier 3.9.6、Markdownlint CLI2 0.23.3、AutoCorrect Node 2.14.0。Python Markdown 解析器和 YAML 依赖固定在 requirements.txt；来源系统 CLI 的 AutoCorrect 2.16.3 不作为本接入的已验证版本。
