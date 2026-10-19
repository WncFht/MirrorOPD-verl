# MirrorOPD-verl

基于固定版本 verl 开发 MirrorOPD 蒸馏算法与 OPD-State 轨迹来源、状态诊断功能。训练与验证复用 verl 的 V1 trainer、AgentLoop、TransferQueue 和模型引擎。

当前完成仓库初始化与上游版本固定；MirrorOPD / OPD-State 的新增算法、配方和诊断尚未迁入。原有 verl 功能保留，实验的数据集与模型选择参考 One-Shot-OPD 的固定版本。[^oneshot]

## 固定基线

| 项目 | 版本 |
| --- | --- |
| verl 上游 | `verl-project/verl` |
| 上游 commit | `263653e749306287a5032140cce3d7833c1959d0` |
| 上游提交时间 | 2026-10-19 13:08:09（Asia/Shanghai） |
| 基线标签 | `verl-base-2026-10-19` |
| 配方参考 | One-Shot-OPD `6b6b1782cc3f04749dd077de19353f839f6bc048` |

保留上游 Git 历史；`origin` 指向本项目，`upstream` 指向官方 verl。后续修改从该基线继续提交。依赖声明及 `uv.lock` 沿用基线，升级上游或依赖时单独记录。来源与版本更新约定见 [UPSTREAM.md](UPSTREAM.md)。[^verl]

## 模型与数据参考

以下记录 One-Shot-OPD 的默认单教师配方，模型名称对应其脚本中的 checkpoint 目录名；实际路径由运行配置提供。[^recipe]

| 领域 | Student | Teacher | 训练数据 | 验证 / 测试参考 |
| --- | --- | --- | --- | --- |
| 数学 | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | DAPO-Math-17k | MATH-500、AMC23、AIME25 |
| 代码 | DeepSeek-R1-Distill-Qwen-1.5B | DeepCoder-1.5B-Preview | TACO | LiveCodeBench v6 |
| 指令遵循 | DeepSeek-R1-Distill-Qwen-1.5B | UltraData-IF-1.5B | Multi-IF 首轮数据 | Multi-IF 多轮评测 |
| 函数调用 | Qwen2.5-Coder-1.5B-Instruct | Hammer2.1-1.5b | xLAM function-calling-60k | BFCL v3 |

多教师配方另用 `Nemotron-Research-Reasoning-Qwen-1.5B-v2` 作为 code teacher，并按数据的 `ability` 字段路由。One-shot 数学对照使用 DAPO 第 22 题重复 64 行；template 模式使用模板前缀。训练预算、验证预算和判分协议需分别对齐。[^mopd]

数据准备与 schema 参考 One-Shot-OPD 的 `data/prep/` 和 `data/README.md`。本仓初始化不包含模型权重或新增训练数据。One-Shot-OPD 的启动脚本使用 V0 trainer 与自定义 FSDP teacher，后续迁移将使用本仓 V1 接口。[^data]

## 开发入口

| 范围 | 入口 |
| --- | --- |
| 配置与启动 | [main_ppo.py](verl/trainer/main_ppo.py)、[ppo_trainer.yaml](verl/trainer/config/ppo_trainer.yaml) |
| V1 训练、验证与 checkpoint | [trainer_base.py](verl/trainer/ppo/v1/trainer_base.py) |
| 轨迹生成与 TQ 接入 | [agent_loop](verl/experimental/agent_loop/)、[agent_loop_tq.py](verl/trainer/ppo/v1/agent_loop_tq.py) |
| Teacher 评分 | [teacher_loop](verl/experimental/teacher_loop/) |
| 蒸馏目标与配置 | [distillation](verl/trainer/distillation/)、[distillation.py](verl/workers/config/distillation.py) |
| 训练后端 | [engine](verl/workers/engine/) |
| 原有 OPD 示例 | [on_policy_distillation_trainer](examples/on_policy_distillation_trainer/) |

计划逐步加入 MDPO/BPO 与可调 teacher 信号、冻结锚点和分布评分、多来源 rollout 与 fusion、固定轨迹对照，以及完整词表 KL 和状态诊断。每项功能先核对当前 verl 的能力，再以独立修改接入；数学目标、采样策略和计算优化分别验证。

## 安装与上游文档

环境准备使用固定基线的 [安装说明](docs/start/install.rst) 和 [uv.lock](uv.lock)。不同训练与推理后端使用相应依赖组合。原上游介绍完整保留在 [README.verl.md](README.verl.md)，OPD 接口见 [docs/algo/opd.md](docs/algo/opd.md)。当前初始化仅检查文档与版本记录，未运行训练验收。

## 来源与许可

保留 verl 的 [LICENSE](LICENSE)、[Notice.txt](Notice.txt) 与原文件版权声明。后续迁入 MirrorOPD、OPD-State 或其他研究代码时，分别记录来源版本和许可证。

### 参考文献

[^verl]: verl-project. verl 固定基线，2026. [263653e7](https://github.com/verl-project/verl/tree/263653e749306287a5032140cce3d7833c1959d0).
[^oneshot]: Thinking-Space. One-Shot-OPD 固定参考，2026. [6b6b1782](https://github.com/Thinking-Space/One-Shot-OPD/tree/6b6b1782cc3f04749dd077de19353f839f6bc048).
[^recipe]: Thinking-Space. 单教师 OPD 配方. [recipe/opd.sh](https://github.com/Thinking-Space/One-Shot-OPD/blob/6b6b1782cc3f04749dd077de19353f839f6bc048/recipe/opd.sh).
[^mopd]: Thinking-Space. 多教师 OPD 配方. [recipe/mopd.sh](https://github.com/Thinking-Space/One-Shot-OPD/blob/6b6b1782cc3f04749dd077de19353f839f6bc048/recipe/mopd.sh).
[^data]: Thinking-Space. 数据布局与准备. [data/README.md](https://github.com/Thinking-Space/One-Shot-OPD/blob/6b6b1782cc3f04749dd077de19353f839f6bc048/data/README.md).
