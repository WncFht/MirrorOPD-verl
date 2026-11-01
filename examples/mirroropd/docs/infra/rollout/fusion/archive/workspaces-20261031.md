# Rollout 工作区与分支

> 历史记录：按原文所述日期、代码版本和实验协议解释；其中“当前”“待完成”不代表最新状态。当前入口见 [Rollout 支持范围](../support.md)。

本页区分开发、执行和历史存档。核对日期为 2026-10-31。新工作树统一放在仓库内 `.worktrees/`，新运行产物和清理归档放在仓库内 `runs/`；历史运行按原 manifest 解释。

统一接口与后续八个架构重构提交已随 `c8b14ffb` 合入 main，同时保留 main 的 RKL 读出、性能对照和研究文档。合并前后代码树一致，相关 CPU 回归 276 项通过；不把代码整合当成新的性能验收。旧开发现场与逐文件比较保存在仓库内 `runs/rollout-maintenance-20261031/`。

## 保留的 Git 工作区

| 目录                                            | 分支或固定代码                 | 用途                                           |
| ----------------------------------------------- | ------------------------------ | ---------------------------------------------- |
| `MirrorOPD-verl`                                | `main`                         | 已整合的公共入口、rollout 架构与 RKL 实现      |
| `MirrorOPD-verl/.worktrees/rollout-maintenance` | `refactor/rollout-maintenance` | 从整理后的 main 建立，先修代码走读确认的问题   |
| 同级 `MirrorOPD-verl-vllm-expanded`             | detached，`c52e3eb6`           | 正在执行的科学实验快照，完成前保持原路径和代码 |

原 `MirrorOPD-verl-rkl-opt-20261031` 包含固定状态 RKL 读出、投影与传输优化、验证布置对照及四卡验收。核心实现已经进入 main；旧工作区差异为格式、较早文档及较少的验收字段。归档后移除其 Git 工作区登记，保留 main 的更新版本。原 `MirrorOPD-verl-rollout` 已合并，现场归档后移除工作区和 `feat/rollout-composition` 分支。

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

原位置的 outputs 保持原路径、文件和 inode，避免已有审计和研究报告因路径迁移失效。源码兼容链接和生成缓存已移除；原目录仅保留 outputs 与存档说明，源码和恢复标签统一位于集中存档。需要重跑或修改历史版本时，从标签建立新的明确工作树，不把兼容目录当作开发目录。

旧开发别名 `-vllm-probability` 已删除；`-fixed-state-engine` 的空目录也已删除。日常开发使用仓库内 `.worktrees/rollout-maintenance`，历史源码直接从集中存档或 Git 标签读取。清理回执见 `runs/rollout-composition-20261031/compatibility-cleanup.json`。

## 占位与冻结执行的独立性

GPU 占位守护使用 `runs/gpu-expansion-20261030/idle-occupancy/runtime/resource_guard.py` 的固定工具副本，不再依赖整个工程 checkout。迁移前后核对了文件哈希、进程身份、GPU 锁与科学任务租用记录；占位程序经正常交接重新启动，科学训练进程没有停止。

实验源码、配置和运行产物的身份仍由各自 manifest 记录。归档只整理代码位置和 Git 登记，不把失败结果改成通过，也不删除未通过候选。

## 未完成工作与后续开发

原驻留超时草稿已以[历史 patch](../../../../../../../../runs/rollout-organization-20261031/parked-residency/manifest.json)封存，包含原基点、文件哈希和已有 CPU 测试证据；它未完成 GPU 验证，且对应代码已被统一接口与生命周期修复替代，不再作为待合并开发项。当前通过范围以[实施计划](2026-10-31-Rollout统一接口实施计划.md)为准。

后续代码走读和处理顺序见[实施计划](2026-10-31-Rollout统一接口实施计划.md)。公共接口保持 model/left/right，内部职责与恢复合同见[统一语义](../semantics.md)。新的优化候选先保留在维护分支，经过对应回归再合入 main；冻结科学实验不跟随 main 切换。
