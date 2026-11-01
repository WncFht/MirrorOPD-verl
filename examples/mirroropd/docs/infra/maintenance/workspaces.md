# 开发与冻结执行工作区

以下是 2026-11-01 文档整理时的 Git 登记快照，不是训练进程或 GPU 占用清单。当前登记请在仓库根目录执行 `git worktree list --porcelain`，未提交改动另用 `git status --short` 核对。新工作树放仓库内 `.worktrees/`，运行产物放 `runs/`。

| 位置（相对仓库）                   | 分支                                  | 提交与锁           |
| ---------------------------------- | ------------------------------------- | ------------------ |
| `仓库根目录`                       | `main`                                | `00e8af30`         |
| `../MirrorOPD-verl-vllm-expanded`  | `detached`                            | `c52e3eb6`         |
| `.worktrees/architecture-refactor` | `refactor/runtime-contracts-20261101` | `d3eb9b39`         |
| `.worktrees/eos-full-vocab`        | `perf/eos-full-vocab-20261101`        | `8f92d9b5`         |
| `.worktrees/kappa1-batch-study`    | `detached`                            | `402f6353`         |
| `.worktrees/qwen-native`           | `detached`                            | `00e8af30`；locked |
| `.worktrees/qwen-prefix-study`     | `feat/qwen-prefix-study`              | `462e3f3a`         |
| `.worktrees/rollout-maintenance`   | `refactor/rollout-maintenance`        | `00e8af30`         |

main 有未提交的模型兼容、原生评估和研究修改；文档中的已提交基线与这些改动分开说明。分支名、detached 状态、目录年龄和相同 HEAD 都不能证明目录可删除。清理前核对未提交成果、独有提交、运行及后处理的源码依赖、输出引用和恢复版本；冻结目录不原地切换代码。

本轮只更新文档，没有合并或删除 worktree、停止作业或释放 GPU。此前清理结果见[2026-10-31 回执](../rollout/fusion/archive/workspaces-20261031.md)。
