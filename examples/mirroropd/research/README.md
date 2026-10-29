# MirrorOPD 研究代码

从仓库根目录运行。`studies` 按研究问题组织实验、分析和绘图；`core` 提供公共 IO、指标、统计和证据身份；`backends` 负责 HF、FSDP2 与评测的执行；`runtime` 管理进程和运行源码；`checks` 保存独立数学核验。文档仍位于原来的 `docs` 目录。

## 查找与运行

```bash
# 根据源码发现全部可执行模块，不导入模型或启动任务
python -m examples.mirroropd.research

# 查看固定状态迁移和单遍跨批研究的参数
python -m examples.mirroropd.research fixed_transfer --help
python -m examples.mirroropd.research cross_batch.long_batch_transfer --help

# 重算已有统计，使用明确的数据目录
python -m examples.mirroropd.research cross_batch.analyze_long_batch_transfer \
  --data ../wandb-logs/MirrorOPD-verl/analysis/twenty-batch-pass-20261027

# 从同一次本地日志读取生成运行表和验证表
python -m examples.mirroropd.research production.refresh_tables --logs-repo ../wandb-logs
```

命令名是 `studies` 下的相对模块名；数学核验使用 `checks.objectives.verify_fixed_prefix` 等名称。包内含 `__main__.py` 时直接使用包名，例如 `fixed_transfer`。命令由 Python 源码发现，不维护注册表或 manifest。也可以直接运行完整模块名，例如 `python -m examples.mirroropd.research.studies.cross_batch.long_batch_transfer --help`；分布式入口使用 `torchrun --nproc-per-node=2 --module examples.mirroropd.research.studies.cross_batch.long_batch_transfer ...`。

旧的 `transfer`、`precision`、`analysis`、`common`、`objectives`、`rollout_policy` 顶层包和旧的两段式 CLI 已删除，没有转发模块。活动文档使用新路径。冻结证据中的命令描述当时执行的源码，不是当前入口。

`audit` 表示核验目的，部分核验执行真实模型更新。运行前阅读模块说明和参数；列出命令不会调用实验。纯统计使用标准库或 NumPy；绘图通过 `core.plotting.load_style` 加载 `FIGURE_VIZ_MPL` 指定的 figure-viz `assets/mpl` 目录。

## 研究与报告

| 研究包                                   | 主要用途                                 | 文档入口                                                                                                                          |
| ---------------------------------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `state_sources`                          | S/I 来源、在线交集与固定输入控制         | [跨源泛化总览](../docs/experiments/teacher-transfer/transfer-mechanism/2026-10-27-跨源蒸馏泛化研究总览.md)                        |
| `cross_batch`                            | B/N/L 跨批、覆盖、batch 与单遍训练       | [跨批结果](../docs/experiments/teacher-transfer/transfer-mechanism/cross-batch/2026-10-28-跨批次迁移与单遍训练结果.md)            |
| `capability_filter`                      | 筛选、温度、功能保持与续写控制           | [能力筛选假说](../docs/experiments/teacher-transfer/transfer-mechanism/capability-filter/2026-10-28-OPD能力筛选假说的受控实验.md) |
| `spectral`                               | 梯度谱与教师候选核对                     | [谱与同架构诊断](../docs/experiments/teacher-transfer/transfer-mechanism/spectral/README.md)                                      |
| `polaris_transfer`                       | E48、KL 方向和教师谱系探针               | [Polaris 迁移](../docs/experiments/teacher-transfer/cross-origin/polaris/README.md)                                               |
| `boundary`                               | R1 边界与续训实验的明确协议              | [运行说明](studies/boundary/README.md)                                                                                            |
| `fixed_transfer`                         | 按既有计划组织模型、冻结银行、训练和评测 | [固定状态迁移](studies/fixed_transfer/README.md)                                                                                  |
| `precision`、`polaris`、`rkl_optimizers` | 精度、优化器与反弹机制                   | [优化器与精度](../docs/experiments/teacher-transfer/optimizer-precision/README.md)                                                |
| `termination`                            | 终止动作、E54 长度和目标干预             | [E54 诊断](../docs/infra/model-compatibility/qwen-e54/2026-10-29-E54长度膨胀诊断.md)                                              |
| `qwen_base`、`qwen_teachers`             | Qwen 教师与生产运行分析                  | [教师对比](../docs/experiments/teacher-transfer/teacher-comparison/README.md)                                                     |
| `production`、`production_precision`     | 运行表、预算、生产精度与 KL 轨迹         | [实验维护说明](../docs/experiments/维护说明.md)                                                                                   |
| `pipeline`                               | 已记录流水线、性能与旧运行证据分析       | [性能研究](../docs/infra/performance/README.md)                                                                                   |

## 公共实现与实验差异

IO 统一从 `core.io` 导入，包含严格 JSON、JSONL/gzip、原子替换与 CSV。产物指纹、完整回执和追加记录修复位于 `core.artifacts`；这些是自动生成的运行证据，不是人工登记目录。通用日志坐标和统计位于 `core.logs`，实验 inventory 的读取位于 `core.inventory`；运行与验证 Markdown 只由 `production.refresh_tables` 生成。

`core.metrics` 计算原生支持上的分布距离，调用者决定轨迹和聚合分母。条件共享支持、原生完整支持、概率排序和 logits 排序有不同含义，协议显式保留。`core.statistics` 与 `core.transfer_metrics` 分别实现已有配对统计和 GapRecovery，不把不同实验的置信区间规则混用。

固定状态迁移与边界实验共享 `backends.hf_rkl` 的原生 RKL VJP 和 teacher 概率缓存计算；完整前缀 HF 执行位于 `backends.hf_models`。稀疏来源和在线探针使用 `backends.hf_native` 的原有解码协议。FSDP2 干预使用 `backends.fsdp`，独立 FP64/自动微分参考保留独立计算。评测的请求重试与聚合在 `backends.evaluation`，判分在 `core.math_grading`；采样数通过参数传入。

L 的具体设置放在 `studies/cross_batch/protocol.py`，核对归一化和恢复身份。不同研究的预算、模型和数据选择由研究包负责，不增加统一 Trainer、任意插件或另一门配置语言。扩展已有固定状态实验使用已有计划；只有新数学或执行协议才增加实现。

## 源码、缓存与恢复

固定状态队列将研究代码和实际依赖的 `verl` 源码复制到执行目录，worker 和 guard 都从该目录启动。`MIRROROPD_WORKSPACE` 保持数据、模型与资源控制位于原工作区；模型执行代码的变化参与 teacher 缓存和数值检查身份。未完成训练与评测分别恢复，权重读出不被解释为包含 Adam 的连续恢复状态。

GPU 归属由运行资源配置核对；科学参数不绑定某组物理卡。当前固定状态队列仍使用本机已有的 GPU4–7 分配。所有身份不匹配均报错，重算使用新的输出位置，不覆盖冻结证据。

## 验证

使用具备项目依赖的 Python 环境。纯 CPU 回归包括：

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/examples/research \
  tests/trainer/distillation/test_research_common_on_cpu.py \
  tests/trainer/distillation/test_precision_factors_on_cpu.py \
  tests/trainer/distillation/test_opd_precision_controls_on_cpu.py \
  tests/trainer/distillation/test_mirror_solver_controls_on_cpu.py \
  tests/trainer/distillation/test_optimizer_counterfactual_on_cpu.py
```

研究测试集中放在 `tests/examples/research` 并使用 `_on_cpu.py` 命名，进入现有 CPU CI。验证覆盖源码隔离、无后端命令发现、缓存错配、评测续接、完整参数梯度与 tied-weight 保存。真实模型数值与性能检查仍由对应 GPU 核验入口执行；CPU 通过不代替 GPU 验收。
