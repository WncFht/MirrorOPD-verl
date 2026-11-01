# MirrorOPD 实验与训练框架

MirrorOPD 在 verl 上提供单教师 sampled/full-vocab 蒸馏、weighted teacher 与 MDPO/BPO、可组合 rollout 及原生评估。研究材料和工程文档从[docs](docs/README.md)进入；仓库根 docs 保留上游 verl 文档。

## 最短使用路径

在仓库根目录用 PATH 中的 Python 查看预设，不启动 GPU 训练：

```bash
bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh --overview
bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh --list-presets
```

配置预览会加载原生训练依赖，需要先按[运行环境](docs/infra/training/runtime.md)设置 `PYTHON_BIN`；默认 `.venv-native/bin/python` 必须实际存在。设置后运行：

```bash
CONFIG_ONLY=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh custom --method full_kl
```

统一 math 入口按预设或 custom 选择模型与预算，`--method`选择 sampled、full_kl、full_forward_kl、mdpo 或 bpo。编号清单由[规划器](launchers/math_experiment.py)生成，不在本页复制易过期的编号范围。方法、覆盖优先级和 coding/IF 入口见[启动指南](docs/infra/training/launch.md)。

实际训练前按[资产准备](docs/infra/training/资产准备.md)配置模型和数据，再按[运行环境与资源](docs/infra/training/runtime.md)选择 Python、已分配 GPU 和输出目录。Hope 部署使用已有`make assets-check`、`make opd-submit`流程；部署、日志传回见[meituan](../../meituan/README.md)。冻结运行保持原代码与原协议。

## 按任务阅读

| 任务                            | 入口                                                                                       |
| ------------------------------- | ------------------------------------------------------------------------------------------ |
| 查研究问题、协议与结果          | [实验研究](docs/experiments/README.md)，五个 RQ；progress 仅保存周报                       |
| 查数学定义与证明                | [理论](docs/theory/README.md)                                                              |
| 查公共参数、模型、数据与指标    | [设置参考](docs/settings/README.md)                                                        |
| 理解实现与扩展                  | [架构](docs/infra/architecture/README.md)、[代码走读](docs/infra/architecture/代码走读.md) |
| 检查数值、optimizer、模型或性能 | [工程实验](docs/infra/README.md)                                                           |
| 复算与生成图表                  | [研究代码](research/README.md)、[实验维护](docs/experiments/维护说明.md)                   |
| 维护文档与提交检查              | [维护流程](docs/infra/maintenance/README.md)                                               |

## 工具与来源

`assets/`准备身份和数据；`launchers/`构造配置与启动；`research/`维护科学分析和验算；`benchmarks/`维护执行性能与接入测量；`smoke/`保存专项短测；`rewards/`提供判分入口；`tools/`维护文档与运行辅助工具。docs 里的 Python 仅为冻结来源，不作为活跃工具。

[完整验证负载的 vLLM sweep](benchmarks/vllm_sweep/README.md)维护原生生成性能的协议、恢复和统计。候选并发、独立服务最快值与原 verl 验收分开解释，不由一次局部 benchmark 自动更改训练参数。
