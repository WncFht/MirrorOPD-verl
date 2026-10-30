# Rollout 工作区与分支

本页规定开发、执行和历史证据的归属。核对日期为 2026-10-31；机器快照与本轮操作回执保存在[工作区记录](../../../../../../../runs/rollout-organization-20261031/workspaces.json)。目录均位于同一个工作区根下，表中省略共同前缀 `MirrorOPD-verl`。

## 开发与执行角色

| 目录                  | 分支或固定代码                                      | 角色与修改边界                                                         |
| --------------------- | --------------------------------------------------- | ---------------------------------------------------------------------- |
| 主目录                | `main`                                              | 主入口及其他研究工作；有正在编辑的未提交修改，本轮仅同步明确归属的文档 |
| `-rollout`            | `feat/rollout-composition`                          | 唯一的 rollout 组合开发主线，承接后续统一接口与专项计划                |
| `-fixed-state-engine` | `feat/fixed-state-engine`                           | 独立的固定状态研究/引擎工作，不属于本轮组合架构合并范围                |
| `-vllm-expanded`      | `feat/fusion-expanded-experiments`，固定 `c52e3eb6` | 科学实验执行快照；该分支名不表示可在其上继续开发                       |
| `-vllm-engineering`   | detached，`a34724a3`                                | 工程验收执行快照，保持代码固定                                         |
| `-fusion-release`     | detached，`d56e081d`                                | HF 历史证据快照                                                        |
| `-vllm-acceptance`    | detached，`9666884f`                                | 原生生命周期验收快照和输出                                             |
| `-vllm-fusion`        | detached，`91674e68`                                | 旧前缀实验与性能证据                                                   |
| `-vllm-pm-acceptance` | detached，`36aa8ab9`                                | PM、读出与恢复检查证据                                                 |
| `-vllm-scientific`    | detached，`73b30f06`                                | 科学协议早期执行与失败记录                                             |

新工作只进入 `feat/rollout-composition`。运行中的任务继续读取各自固定 checkout，不能为了“合并目录”改变它们的 HEAD 或源码。固定状态引擎是另一条工作线，不按同名 rollout 关键词自动合并。

## 本轮归并

原 `feat/vllm-probability-fusion` 已更名为 `feat/rollout-composition`，原开发目录 `-vllm-probability` 已迁至 `-rollout`。旧目录名保留为兼容软链，不再是额外 worktree，避免历史命令和引用突然失效。代码基线保持 `e197915c`，本轮改动限于组织与文档。

此前已整合的 R1、原生 vLLM、PM、refit 修复和 main 已提交内容，继续由同一条开发主线承载。历史执行快照保留归档标签；不把冻结实验分支当作第二条开发主线。

未提交的驻留超时实现已连同完整 patch、文件哈希、原基点和已有测试日志入口[封存](../../../../../../../runs/rollout-organization-20261031/parked-residency/manifest.json)。该草稿未完成 GPU 验证，不属于已交付能力。后续相关计划需要时再恢复，避免其与本轮架构整理混在一起。

## 历史目录与数据

五个历史快照目录包含或关联被文档引用的输出、checkpoint 和源码证据。Git 提交和标签保存源码历史，不能替代未提交的运行产物。后续若减少这些目录，应先核对依赖、归档输出、修复引用，再移除 worktree；本轮不删除这些证据。

目录数量不代表仍有同样数量的开发分支。当前只有一条 rollout 组合开发主线，另有一条独立固定状态工作线和一个冻结科学执行分支。

## 文档与代码的交付边界

统一架构文档描述目标接口；当前代码仍使用原 student/teacher 配置。文档同步到 main 不意味着原生 fusion 代码、推荐执行配置或新左右模型接口已经合入 main。后续计划应明确需要修改的用户入口、配置兼容和运行语义，再执行相应代码交付。
