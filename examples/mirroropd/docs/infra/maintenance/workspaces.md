# 开发与冻结执行工作区

2026-11-01，运行合同重构与 main 的动作空间、token 映射和评估实现已整合到本地 main，代码提交为 `1d4ef0ee`。合并包含 `3f0b5cf2` 的原生 Qwen 入口适配、`cb67abac` 的文档结构整合，以及原来未提交的相关实现。远端没有推送；冻结实验继续使用各自固定版本。

## 保留的工作区

下表记录合并清理后的目录及保留理由。运行状态会变化，当前登记用 `git worktree list --porcelain` 核对，未提交成果用对应目录的 `git status --short` 核对。

| 位置（相对仓库）                  | 版本或分支                           | 保留理由                                                         |
| --------------------------------- | ------------------------------------ | ---------------------------------------------------------------- |
| `仓库根目录`                      | `main`；代码基线 `1d4ef0ee`          | 后续开发入口；独立研究报告、图表和性能材料仍保留为未提交修改     |
| `../MirrorOPD-verl-vllm-expanded` | detached `c52e3eb6`                  | 冻结的执行版本及历史产物；检查时无运行进程，产物尚未完整迁移归档 |
| `.worktrees/kappa1-batch-study`   | detached `402f6353`                  | 检查时无运行进程，但有 11 项未提交修改，需先审查成果             |
| `.worktrees/qwen-native`          | detached `00e8af30`；locked          | 原生 Qwen 队列仍使用此冻结源码                                   |
| `.worktrees/qwen-prefix-study`    | `feat/qwen-prefix-study`；`462e3f3a` | Qwen 任务仍运行，且有尚未整合的独有提交                          |

新工作树放仓库内 `.worktrees/`，运行产物放 `runs/`。分支名、目录年龄、相同 HEAD 或 GPU 暂时空闲都不能证明目录可删除。冻结目录不原地切换代码。

## 已清理及恢复入口

| 移除的工作树                       | 清理依据                                                             | 恢复入口                                   |
| ---------------------------------- | -------------------------------------------------------------------- | ------------------------------------------ |
| `.worktrees/eos-full-vocab`        | 无未提交修改及运行依赖；完整树与 main 历史提交 `2408c8e8` 相同       | tag `archive/eos-full-vocab-20261101`      |
| `.worktrees/rollout-maintenance`   | `00e8af30` 已被 main 包含；独有 Hydra 输出已复制并核对 hash          | tag `archive/rollout-maintenance-20261101` |
| `.worktrees/architecture-refactor` | 全部提交已被 main 包含；测试进程退出，剩余忽略文件均为缓存或依赖链接 | tag `archive/runtime-contracts-20261101`   |

对应开发分支一并移除，恢复 tag 保留。原始交接文件、worktree 清理回执和运行日志在 [`runs/merge-runtime-20261101`](../../../../../runs/merge-runtime-20261101)。合并前 main 的完整工作区另存 `refs/archive/main-before-runtime-merge-1793514879`；相关实现的原始快照保存在 `refs/archive/model-evaluation-handoff-final-20261101`。两者是恢复快照，不是经过验证的开发提交。

切换前后的 SHA256 核对确认，85 个未纳入整合提交的工作区文件状态保持一致，包括删除状态和未跟踪材料。没有删除这些成果，也没有停止冻结实验。

## 合并验证与适用范围

整合版本保留统一的 `FullVocabOutput` 和可被 FSDP 遍历的具名结果，条件 RKL 指标与 native 指标分开。评估使用当前 checkpoint 身份和显式组件配置；配置模块不再因包级通配导出而加载 loss 实现。

- 蒸馏、配置、训练循环和 rollout 服务 CPU 回归：577 项通过。
- 实验规划与数值路线检查：61 项通过；受影响的启动器检查：22 项通过。
- CUDA 条件分布核：12 项通过。
- 两卡 FSDP：共同支持、token ID 重排、Llama、EOS MDPO、EOS 端点共 5 组通过。前三组及 EOS MDPO 检查 checkpoint 恢复；EOS 两组与 `precompute` 对照，更新后参数最大绝对差为 0。
- 提交 hooks、Ruff、类型检查、配置生成及文档检查通过。文档检查明确列出的仓外历史证据未在本工作树验证。

这些检查验证接口整合、梯度、更新与恢复，不代表重新完成 50 outer 正式实验，也不提供新的整轮性能结论。旧模型配对验收仍按其记录的源码 hash 解释，见[动作空间与模型兼容](../architecture/evaluation/动作空间与模型兼容.md)。此前工作区清理记录见[2026-10-31 回执](../rollout/fusion/archive/workspaces-20261031.md)。
