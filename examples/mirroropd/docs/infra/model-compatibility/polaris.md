# Polaris 模型兼容与接入验收

本页只维护输入、词表/head 与历史短程验收。E48/E49 的实际配方、完整结果和研究解释统一见[Polaris 研究](../../experiments/teacher-transfer/cases/polaris.md)。以下短测和 CPU 检查保留 2026-10-26 的环境与范围，不代替后续正式训练。

## 模型与本机路径

Polaris 权重已下载到仓库同级 `model/hub/POLARIS-Project--Polaris-7B-Preview/faea72cf7123681069668e8da45c1c074e378830/`，包含四个 safetensors 分片。新增资产目录条目，支持通过[准备脚本](../../../assets/prepare_models.py)指定`--models Polaris-7B-Preview`。

本机 student/teacher 均为 Qwen2 架构。Tokenizer 实际`get_vocab()`比较与 chat template 检查确认共用 student token 序列的兼容性；head 行数分别 151936/152064，hidden 宽度分别 1536/3584。Full-KL 接口支持不等 hidden 宽度及 teacher 额外输出行，保留 teacher 原生 softmax 归一化；该师生组合已完成下述四卡短训练验收。

本机已用的训练解释器是 `/root/.local/share/mirroropd/venvs/profile-cu128/bin/python`，数学数据在仓库同级 `MD-OPD/data/`；公共默认 `.venv-native/bin/python` 与 `data/` 是其他部署环境的约定，本机执行需要显式覆盖。

## 启动与独立评测

在仓库根目录先设置本机路径。以下只打印配置，不启动 GPU 任务：

```bash
export PYTHON_BIN=/root/.local/share/mirroropd/venvs/profile-cu128/bin/python
export MODEL_ROOT="$PWD/../model"
export DATA_ROOT="$PWD/../MD-OPD/data"
DRY_RUN=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E48
```

原始 student/teacher 的独立评测选择器分别是`E48-valid-student`和`E48-valid-teacher`。正式训练前应在同一验证协议下测量教师，不能直接用论文或模型卡分数作为本实验的教师参考。

## 四卡短训练结果

短训练使用 GPU 0–3，完成 3 个 outer step、48 次真实更新。保留 batch64、reuse16、LR1e-6、训练回答上限 7 Ki token、teacher 评分预算 32 Ki token、precompute/512/8192，以及 student 128/0.65。验证在三个数据集各取前两题，每题一个回答、上限 512 token，在训练前和第 3 步后执行；引擎上下文仍保留完整验证所需容量。短训练不用于准确率比较，也不替代完整验证与 checkpoint 验收。

验收检查全部通过：3 步训练均完成，48 次更新连续且实际应用，没有跳过更新；loss 与裁剪前梯度范数有限，teacher 前向成功，训练前后各 6 条验证输出完整且评分失败为零。Loss 范围为 0.186–0.213。每 0.2 秒采样四卡显存，并按 teacher 参数传入与前向的时间区间统计峰值，teacher 阶段共 131 次采样。显存是整卡采样占用，不能当作精确的单模块张量峰值，也可能遗漏瞬时峰值。

| Outer step | 完整步骤耗时（分） | Teacher 前向（秒） | Actor 更新（分） | 累计实际更新 |
| ---------: | -----------------: | -----------------: | ---------------: | -----------: |
|          1 |               2.35 |                5.1 |             1.68 |           16 |
|          2 |               2.14 |                4.7 |             1.61 |           32 |
|          3 |               2.14 |                4.4 |             1.58 |           48 |

| GPU | Teacher 阶段显存峰值（GiB） | 全程显存峰值（GiB） | 总显存（GiB） |
| --: | --------------------------: | ------------------: | ------------: |
|   0 |                        12.7 |                54.1 |          79.6 |
|   1 |                        12.4 |                54.1 |          79.6 |
|   2 |                        12.4 |                54.1 |          79.6 |
|   3 |                        12.6 |                54.1 |          79.6 |

三个训练批的平均回答长度约为 6.2–6.5 千 token，最长回答均达到 7 Ki token 上限。该负载下 teacher 预算无需降低；3 步结果不能证明长程收敛，也不能保证所有后续长度分布下都不会 OOM。

代码采用独立快照。GPU 占用工具完成四卡启动、重复启动保护、心跳与计算进展、停止释放和重新启动验证后，停止占用并启动训练；GPU4–7 不参与。详细结果见短训练报告（本地证据：`outputs/polaris-smoke-20261026-bsud-quicksilver-hppk-13/result.md`）、[逐项验收](../../../../../outputs/polaris-smoke-20261026-bsud-quicksilver-hppk-13/result.json)和最终解析配置（本地证据：`outputs/polaris-smoke-20261026-bsud-quicksilver-hppk-13/resolved-config.yaml`）。

上述表格仅覆盖四卡短训练。正式 E48 `grcuotmi` 在 8 张 A100 80GB 上正常完成 256/256 outer；正式 run 的指标、边界与诊断见[完整训练与后期平台分析](../../experiments/teacher-transfer/cases/polaris.md#e48)。

## CPU 验证

2026-10-26 已通过 E48 的 Hydra 最终配置解析与配置验证，确认四卡覆盖下仍为 batch64、reuse16、256 outer、LR1e-6、初始验证开启，以及 teacher token budget32768。预设规划测试验证 E48 沿用 E20 预算、teacher 独立评测不继承训练预算；未知编号测试已将原来的 E48 更新为 E49 并通过。模型资产测试 16 项通过，文档链接及 shell 语法检查通过。

实际资产检查确认 Polaris 四个权重分片齐全；师生 `get_vocab()` 的 151665 项 token→ID 映射完全一致，chat template 相同。DAPO 的 17917 行、MATH-500 的 500 行、AMC23 的 40 行、AIME25 的 30 行全部通过数据结构检查。以上均为 CPU 检查，不包含模型前向、梯度更新或显存验收。
