# Rollout 工作区与分支

本页区分开发、执行和历史存档。核对日期为 2026-10-31；操作及验证证据见[清理回执](../../../../../../../runs/worktree-cleanup-20261031/complete.json)。目录均位于同一个工作区根下，表中省略共同前缀 `MirrorOPD-verl`。

## 保留的 Git 工作区

| 目录             | 分支或固定代码             | 用途                                                                                        |
| ---------------- | -------------------------- | ------------------------------------------------------------------------------------------- |
| 主目录           | `main`                     | 公共入口及研究工作；已有未提交修改保留，不由目录清理自动提交或覆盖                          |
| `-rollout`       | `feat/rollout-composition` | 唯一开发主线；已整合原 rollout 分支和 fixed-state-engine 分支的模型资产整理、代码及评审提交 |
| `-vllm-expanded` | detached，`c52e3eb6`       | 正在执行的科学实验快照；源码和版本保持固定，不在这里继续开发                                |

原 `feat/fixed-state-engine` 的已提交内容合入统一开发分支后，其工作树和分支不再单独保留；这不表示固定状态引擎的所有拟议能力已实现。固定状态与动态 rollout 仍按各自研究和接口边界维护。

科学实验原分支 `feat/fusion-expanded-experiments` 的补丁早已整合进开发历史。执行目录保留相同 commit 的 detached HEAD，版本另由归档/执行标签保留，删除分支名不会替换正在运行的代码。

## 历史源码与运行产物

历史源码集中在 `runs/source-archives/20261031/`，每份包含原 commit、恢复标签、逐文件校验和说明。下列目录不再是 Git worktree：

| 原目录后缀            | 原代码     | 保留内容                           |
| --------------------- | ---------- | ---------------------------------- |
| `-fusion-release`     | `d56e081d` | HF 历史源码与原 outputs 链接       |
| `-vllm-acceptance`    | `9666884f` | 原生生命周期输出与源码             |
| `-vllm-fusion`        | `91674e68` | 旧前缀实验、性能证据与源码         |
| `-vllm-pm-acceptance` | `36aa8ab9` | PM、读出、恢复检查产物与源码       |
| `-vllm-scientific`    | `73b30f06` | 科学协议早期执行、失败记录与源码   |
| `-vllm-engineering`   | `a34724a3` | 工程验收输出与源码；占位工具已迁出 |
| `-fixed-state-engine` | `c8d96f0b` | 已合并的资产实现与评审源码         |

原位置的 outputs 保持原路径、文件和 inode，避免已有审计和研究报告因路径迁移失效。源码条目是指向集中存档的兼容链接；原目录中的 `ARCHIVED-WORKSPACE.md` 给出恢复标签。需要重跑或修改历史版本时，从标签建立新的明确工作树，不把兼容目录当作开发目录。

`-vllm-probability` 仍是指向 `-rollout` 的兼容软链，不是额外 worktree。旧路径只是引用兼容层，日常开发只使用 `-rollout`。

## 占位与冻结执行的独立性

GPU 占位守护使用 `runs/gpu-expansion-20261030/idle-occupancy/runtime/resource_guard.py` 的固定工具副本，不再依赖整个工程 checkout。迁移前后核对了文件哈希、进程身份、GPU 锁与科学任务租用记录；占位程序经正常交接重新启动，科学训练进程没有停止。

实验源码、配置和运行产物的身份仍由各自 manifest 记录。归档只整理代码位置和 Git 登记，不把失败结果改成通过，也不删除未通过候选。

## 未完成工作与后续开发

原未提交的驻留超时实现已以[可恢复 patch](../../../../../../../runs/rollout-organization-20261031/parked-residency/manifest.json)封存，包含原基点、文件哈希和已有 CPU 测试证据；它未完成 GPU 验证，不属于已交付能力。

后续接口与开发计划进入 `feat/rollout-composition`。新架构文档定义目标语义，当前 runtime 仍使用旧 student/teacher 配置；主仓库的文档可见性不等于原生后端或新左右模型接口已经发布到 main。
