# 开发与冻结执行工作区

2026-11-01 核对。运行合同与动作空间整合以`1d4ef0ee`为代码基线；已完成的 actor 设置 benchmark 现已通过`e48ed77f`归并到 main。冻结实验仍按各自 source commit 解释，清理不改变运行中的源码，也不自动启动训练。

## 当前工作区与恢复

主工作区用于日常开发。2026-11-18 退役的工作树保留本地恢复 tag，命名为 `archive/20261118/<工作树名>`。独有的未提交补丁和未跟踪源码见[恢复材料](../../../research-evidence/engineering-evidence/recovery/)。

用 `git worktree add --detach .worktrees/<工作树名> archive/20261118/<工作树名>` 检出基线，然后应用对应的 `<工作树名>-working.patch`。有对应的 `<工作树名>-untracked.tar` 时，将它解包到该工作树。

新工作树放仓库内 `.worktrees/`，运行产物放 `runs/`。分支名、目录年龄、相同 HEAD 或 GPU 暂时空闲都不能证明目录可删除。冻结目录不原地切换代码。

## 已清理及恢复入口

| 移除的工作树                       | 清理依据                                                                                                                        | 恢复入口                                   |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| `.worktrees/eos-full-vocab`        | 无未提交修改及运行依赖；完整树与 main 历史提交 `2408c8e8` 相同                                                                  | tag `archive/eos-full-vocab-20261101`      |
| `.worktrees/rollout-maintenance`   | `00e8af30` 已被 main 包含；独有 Hydra 输出已复制并核对 hash                                                                     | tag `archive/rollout-maintenance-20261101` |
| `.worktrees/architecture-refactor` | 全部提交已被 main 包含；测试进程退出，剩余忽略文件均为缓存或依赖链接                                                            | tag `archive/runtime-contracts-20261101`   |
| `.worktrees/kappa1-batch-study`    | 无运行进程；11 项重复停止改动通过反向 patch 检查确认已在 main，原始补丁与文件哈希已归档                                         | tag `archive/kappa1-batch-study-20261101`  |
| `.worktrees/kappa1-main-20261101`  | benchmark 通过正常提交 hooks 后归并到 main；旧测量源码另存补丁及未跟踪文件快照                                                  | tag `archive/kappa1-main-20261101`         |
| `.worktrees/qwen-native`           | 队列为`baseline_ready`，initial／teacher／finalize 全部 exit0，active_gpus 为空、无活进程、未开启自动训练，确认后解除旧锁并移除 | tag `archive/qwen-native-20261101`         |
| `../MirrorOPD-verl-vllm-expanded`  | 无活进程；3 个分支提交的 patch 均已被 main 包含；约 403GiB 产物单独保留                                                         | tag `archive/vllm-expanded-20261101`       |

此前已完成整合的开发分支已移除，恢复 tag 保留。合并前 main 的完整工作区另存 `refs/archive/main-before-runtime-merge-1793514879`；相关实现的原始快照保存在 `refs/archive/model-evaluation-handoff-final-20261101`。两者是恢复快照，不是经过验证的开发提交。

先前运行合同整合时，切换前后 SHA256 核对确认 85 个未纳入整合提交的工作区文件状态保持一致，包括删除状态和未跟踪材料。没有删除这些成果，也没有停止冻结实验。

Expanded 的评测明细直接从[正式证据目录](../../../research-evidence/prefix-expanded/retired-worktrees-20261101/vllm-expanded/outputs/)读取。六个旧 vLLM 同级目录和原路径的符号链接均已删除。

待运行的κ=1／η=0.1 正式计划与配置预览已改为主仓库入口；实际启动前仍需冻结选定的已提交版本。历史运行命令及冻结配置保留原路径和版本，不随清理重写。

## 合并验证与适用范围

Actor benchmark 整合的正常提交 hooks、Ruff、mypy、生成配置及文档检查通过。清理期间未启动新 GPU 实验，性能与数值证据沿用对应报告；原先合同整合的验证范围如下。

整合版本保留统一的 `FullVocabOutput` 和可被 FSDP 遍历的具名结果，条件 RKL 指标与 native 指标分开。评估使用当前 checkpoint 身份和显式组件配置；配置模块不再因包级通配导出而加载 loss 实现。

- 蒸馏、配置、训练循环和 rollout 服务 CPU 回归：577 项通过。
- 实验规划与数值路线检查：61 项通过；受影响的启动器检查：22 项通过。
- CUDA 条件分布核：12 项通过。
- 两卡 FSDP：共同支持、token ID 重排、Llama、EOS MDPO、EOS 端点共 5 组通过。前三组及 EOS MDPO 检查 checkpoint 恢复；EOS 两组与 `precompute` 对照，更新后参数最大绝对差为 0。
- 提交 hooks、Ruff、类型检查、配置生成及文档检查通过。文档检查明确列出的仓外历史证据未在本工作树验证。

这些检查验证接口整合、梯度、更新与恢复，不代表重新完成 50 outer 正式实验，也不提供新的整轮性能结论。旧模型配对验收仍按其记录的源码 hash 解释，见[动作空间与模型兼容](../model-compatibility/action-space.md)。此前工作区清理记录见[2026-10-31 回执](../rollout/fusion/archive/workspaces-20261031.md)。
