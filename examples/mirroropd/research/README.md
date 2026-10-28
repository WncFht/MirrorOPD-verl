# MirrorOPD 研究代码

研究代码统一放在这个 Python 包；文档目录保留报告、图和证据索引。运行命令从 `MirrorOPD-verl` 仓库根目录执行。原 `docs/experiments/polaris-transfer/scripts/` 与 `docs/mirror-descent/scripts/` 的实现已迁走，旧目录没有转发脚本。

## 结构与依赖

```text
research/
  common/       指标、轨迹缓存、JSON、bootstrap、固定批次配置、绘图样式加载
    native.py   HF 模型加载、完整前缀前向与学生/教师/交集解码
  transfer/     state 来源、跨批迁移、覆盖、batch、目标方向与筛选实验
  precision/
    engine.py   FSDP2 worker、独立 FP64 评分、参数快照
    ...         精度、VJP、优化器与 solver 干预及相应统计
  __main__.py   列出和运行实验，发现命令时不导入模型后端
```

依赖方向为“实验 → 公共函数或对应执行后端”。纯 JSON 统计只依赖标准库或 NumPy；列出实验不导入 Torch。HF 和 FSDP2 使用各自的执行实现，独立 FP64/VJP 核验保留自己的参考公式。没有插件注册表、统一 Trainer 基类或第二套任务调度器。

历史实验的区别由具体模块明确保留，例如 S 系列对 head 的 BF16 舍入、早期 KL-direction probe 在共享支持上的距离、I/B/L 的不同归一化。公共 `metrics.py` 只负责给定 state 的原生支持分布距离，不替调用者选择轨迹来源或归一化分母。解码历史 profile 的 top-k 按概率取值，L 的固定评分按 logits 取值，保留各自数值行为；新的指标协议须显式定义 ties 和支持集。

## 统一入口

```bash
# 列出两类可执行模块，不启动训练
python -m examples.mirroropd.research transfer
python -m examples.mirroropd.research precision

# 查看某个实验的参数；原有实验子命令继续保留
python -m examples.mirroropd.research transfer long_batch_transfer --help
python -m examples.mirroropd.research transfer intersection_study --help

# CPU 重算已有 L 统计：输出写到显式指定的数据目录
python -m examples.mirroropd.research transfer analyze_long_batch_transfer \
  --data ../wandb-logs/MirrorOPD-verl/analysis/twenty-batch-pass-20261027
```

也可以直接用完整模块名：`python -m examples.mirroropd.research.transfer.analyze_long_batch_transfer ...`。分布式入口使用 `torchrun --nproc-per-node=2 --module examples.mirroropd.research.transfer.long_batch_transfer ...`；不能再将模块文件路径直接交给 Python。模块名维持原文件 stem，历史参数保留，已有文档中的命令和源码链接同步迁移。

`audit_` 表示核验目的，并不保证只读或只使用 CPU；precision 的 solver/optimizer 干预可能执行真实训练。是否启动模型以模块 docstring 和 `--help` 为准。当前重构仅运行 CPU 核验与历史统计复算，双来源 R 实验尚未启动。

## 常用实验入口

| 系列 | 准备／运行 | 统计／核验／绘图 |
| --- | --- | --- |
| S：稀疏 state 来源 | `state_transfer_probe` | `summarize_state_transfer`、`audit_state_transfer`、`plot_state_transfer` |
| I：在线来源与交集 | `intersection_study` 的各子命令 | `summarize_intersection`、`analyze_intersection_controls`、`audit_intersection_study`、`plot_intersection_study` |
| B：跨批、覆盖与顺序 | `batch_generalization`、`batch_gradient_geometry` | `summarize_batch_generalization`、`analyze_batch_controls`、`audit_batch_generalization`、`plot_batch_transfer_detail` |
| N：batch 与阶段 | `batch_size_stage`、`probe_overlap_mass` | `analyze_batch_size_stage`、`audit_batch_size_stage`、`plot_batch_size_stage` |
| L：40 批单遍 | `prepare_long_batch_transfer`、`long_batch_transfer`、`cross_score_long_transfer` | `analyze_long_batch_transfer`、`audit_long_batch_transfer`、`plot_long_batch_transfer` |
| 筛选假说 | `prepare_filter_study`、`filter_train`、`filter_transfer_proxy`、`filter_synthetic`、`filter_if_eval` | `analyze_filter_study`、`analyze_filter_controls`、`audit_filter_study`、`plot_filter_study`；独立 CPU checker：`grade_filter_if.py` |
| 生产 E48 与 KL 方向 | `analyze_completed_transfer`、`kl_direction_probe` | `analyze_forward_study`、`audit_forward_study`、对应 `plot_*` |
| FSDP2 精度与 solver | `replay_fixed_prefix`、`audit_mirror_solvers`、`factor_mirror_precision` | `summarize_mirror_solvers`、`summarize_precision_localization`、`summarize_opd_transfer`、`plot_precision_report` |
| 优化器历史与实际位移 | `audit_optimizer_history`、`probe_mirror_forward` | `summarize_optimizer_history` |

前七行使用 `transfer`，后两行使用 `precision`。绘图需要设置 `FIGURE_VIZ_MPL` 为 figure-viz 的 `assets/mpl`；该依赖在解析命令参数后才加载，所以 `--help` 不要求安装或配置绘图 skill。`plot_precision_report` 保留原 `--figure-skill` 参数。

筛选假说另有 `filter_calibration` 和 `filter_continue_math` 两个冻结权重诊断入口；完整设置、已完成结果和复算命令见 [OPD 能力筛选假说报告](../docs/experiments/teacher-transfer/transfer-mechanism/capability-filter/2026-10-28-OPD能力筛选假说的受控实验.md)。

## 缓存、恢复与配置

`common/cache.py` 的 `HiddenCache` 使用来源、题目、重复编号、seed、生成 checkpoint 等身份区分路径，并在 payload 中保存完整 prompt/response token IDs 与模型路径、dtype、attention 实现和 hidden 协议。读取时逐项核对，长度相同而 token 不同也会拒绝。模型路径必须代表不可变的权重版本；不能在同一路径覆盖权重后继续复用缓存。

新缓存写入 `validated-hidden-v1/`。历史纯 tensor 缓存无法证明对应哪条完整轨迹，因此保留原文件，首次运行时重新计算新格式；不自动升级或删除旧缓存。没有生成文件摘要或 hash 清单。历史 filter 的内存字典仍限定“一题一来源一轨迹”，重复键会明确报错；需要同题多 rollout 的新实验应直接使用完整轨迹身份。

L 的训练、准备、统计和核验通过 `common/protocol.py` 使用冻结的 `LONG_PASS` 设置。统计和训练入口核对 `protocol.json`，避免把 H 或 batch 改了却仍使用旧分母；新的双来源 R 使用自己的明确协议，不假装是旧 L 的一个无差别重跑。

新保存的 L checkpoint 包含 teacher、模型目录、完整轨迹身份、协议与 world size；恢复时验证一致性、步数及继续训练范围，分别保存和恢复各 rank 的 Torch/CUDA/Python RNG。旧 checkpoint 缺少这些身份字段，改后的训练入口明确拒绝恢复；它们仍可供只读评分。新研究从初始模型重跑，旧权重不被改写或删除。

## 修改与验证约定

增加实验时先复用 `common` 或对应后端中的函数，实验模块负责设置、数据选择和执行顺序。只有相同数学与精度协议的实现才合并；历史特殊行为在模块和报告中写明。独立 oracle 不调用被核验函数来计算期待值。

格式与静态检查沿用仓库 Ruff 配置。公共 CPU 测试覆盖 teacher 额外输出行、原生 RKL 梯度、集合/质量区别、同题多 seed 缓存、前缀与模型错配、原子 JSON、配对统计、checkpoint 身份及评分失败后的内核恢复。运行：

```bash
ruff check examples/mirroropd/research
ruff format --check examples/mirroropd/research
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/trainer/distillation/test_research_common_on_cpu.py \
  tests/trainer/distillation/test_precision_factors_on_cpu.py \
  tests/trainer/distillation/test_opd_precision_controls_on_cpu.py \
  tests/trainer/distillation/test_mirror_solver_controls_on_cpu.py \
  tests/trainer/distillation/test_optimizer_counterfactual_on_cpu.py
```

重构的历史回归在临时数据副本中执行，不覆盖原始证据。三套公共指标在 FP32/FP64 上与重构前逐项完全一致；L 的 99 个文件比较一致，I/B/N 的统计回归分别比较 135/220/173 个证据文件也一致；L 的九张 PNG 与重构前文件内容完全一致。35 项公共接口与既有精度／优化器 CPU 测试通过。后续新增协议的 GPU 数值和性能检查在有卡后单独执行，CPU 验证不替代实际训练验收。

相关研究：[跨源泛化总览](../docs/experiments/teacher-transfer/transfer-mechanism/2026-10-27-跨源蒸馏泛化研究总览.md)、[双来源 R 协议与跨批历史结果](../docs/experiments/teacher-transfer/transfer-mechanism/README.md)、[浮点精度报告](../docs/experiments/teacher-transfer/optimizer-precision/mechanism/2026-10-27-MDPO反弹机制与浮点精度研究报告.md)。

## 文档工具迁移

通用分析由 `analysis` 包维护，CLI 使用 `python -m examples.mirroropd.research.analysis.<模块>`；终止动作、Qwen和共享机制统计使用各自子包。当前双表入口为 `analysis.refresh_tables --logs-repo ../wandb-logs`，不读取人工catalog。数学核验归 `objectives` 与 `rollout_policy`；工程测量归 `examples.mirroropd.benchmarks`。冻结源码仍随证据保留。
