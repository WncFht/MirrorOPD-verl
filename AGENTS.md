# MirrorOPD-verl

- 原始实验日志在仓库同级的 `wandb-logs/`，分析优先读取本地日志。
- 新建工作树放在本仓库的 `.worktrees/`，运行日志、临时产物和清理归档放在本仓库的 `runs/`；不在仓库同级另建目录。运行中的冻结实验与既有共享模型、环境保持原位。
- 项目说明见 [examples/mirroropd](examples/mirroropd/README.md)，部署与传输见 [meituan](meituan/README.md)。
- Python 修改后，对本次修改的文件依次执行 `pre-commit run ruff --files <files>`、`pre-commit run ruff-format --files <files>`，直到通过。
