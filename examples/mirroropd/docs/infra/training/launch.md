# 实验入口与方法选择

从仓库根目录使用统一 launcher。模型资产与运行环境见[资产准备](资产准备.md)及[运行指南](runtime.md)，公共参数含义见[训练设置](../../settings/训练设置.md)。下列预算是预设说明，实际实验的 resolved config 与原报告保持独立。

<a id="launcher-实验预设与运行记录"></a>

## 实验预设与运行记录

训练实验统一从 [run_math_opd_ablation_fsdp2.sh](../../../run_math_opd_ablation_fsdp2.sh) 进入。E1–E47 提供模型、长度、学习率、reuse、验证和保存预算及默认方法；`--method` 只切换算法，`custom` 使用公共配方而不加载编号预算。

| 方法              | 配置与含义                                                                 | 可用选项                                                                            |
| ----------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `sampled`         | 原 `k1` sampled OPD，保留 current/rollout、vanilla/token PPO/token IS/NoIS | `--advantage current/rollout`、`--policy-loss vanilla/token_is/token_ppo/reinforce` |
| `full_kl`         | 规范化为 `mirror: mdpo + full_vocab + eta=1`，无需 anchor                  | `FULL_KL_*`；eta 固定为 1                                                           |
| `full_forward_kl` | 完整 student 输出支持上的 Forward KL，无 anchor                            | `FULL_KL_*`；不等 head 显式设置`FULL_KL_FORWARD_TEACHER_SUPPORT=student`            |
| `mdpo`            | Weighted teacher + scaled MDPO，保留外部 1/eta                             | `--estimator sampled/full_vocab`、`--kappa`、`--eta`                                |
| `bpo`             | Weighted teacher + q 加权中心化回归                                        | 只支持 full_vocab；`--kappa`、`--eta`                                               |

旧 sampled 的 frozen、PPO、NoIS 不是 MDPO 的 eta=1 特例。Current token-IS 只在旧 ±20 ratio guard 未触发时与 sampled MDPO eta=1 相同。入口统一复用代码，保留这些不同目标和估计器的语义。

```bash
# 原编号默认不变：E15 sampled vanilla，E13 普通 Full-KL。
bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E15
bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E13

# 同一 E17 模型与 16 次 reuse 预算，比较 MDPO/BPO。
bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E17 --method mdpo --kappa 0.5 --eta 0.3
bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E17 --method bpo --kappa 0.5 --eta 0.3

# 使用 E13 的模型和预算，换成 sampled MDPO。
bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E13 --method mdpo --estimator sampled --eta 0.3

# 不加载编号预算；通过环境或 Hydra 明确自己的实验设置。
PPO_EPOCHS=4 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh custom --method full_kl
DRY_RUN=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E28 --method bpo --eta 0.5
```

优先级为预设默认值 < 环境变量 < 方法选项 < 末尾 Hydra overrides。环境方法变量是 `EXPERIMENT_METHOD`，mirror 参数沿用 `MIRROR_ESTIMATOR/KAPPA/ETA/DIAGNOSTICS`；旧 sampled 沿用 `OPD_ADVANTAGE_SOURCE/POLICY_LOSS_MODE/CLIP_EPS`。CLI 的 `--kappa/--eta` 只适用于 MDPO/BPO，sampled 专用选项只适用于 `--method sampled`，冲突组合在启动前报错。

加 `--method` 后不再是原编号算法的原样复现；自动实验名附带方法/形状参数，最终以保存的 config 为准。`-valid-student/-valid-teacher` 仍验证原始模型，不接受方法选项，也不把融合分布 m 当成一个可直接生成的模型。

Math 训练只维护一个公共入口。[math_experiment.py](../../../launchers/math_experiment.py) 用不可变的 ModelPair/Budget/Preset 数据表生成预设并合并参数；算法参数由 [\_distillation_args.sh](../../../launchers/_distillation_args.sh) 构造，资源和数据由公共 launcher 管理。E13–E43 中的编号 full 预设保持 precompute、IO 开启；E44–E47 默认 precompute_fused、chunk=128、IO 开启；`custom --method full_kl` 默认 precompute、IO 关闭。需要其他 backend 时显式设置 FULL_KL_BACKEND。CUDA BF16 的一般 MDPO/BPO 可设置 `FULL_KL_BACKEND=precompute_fused`，启用融合 target 与诊断归约；chunk=128/512 的时间和显存比较见[数值层优化实测](../performance/full-vocab/operators.md#mdpo-bpo-target-与诊断融合)。这里的收益不代表整轮吞吐，也不改变普通 Full-KL endpoint 的运算路径。

入口头部保留分组对齐速查表，分别展示模型/长度对照、sampled 信号和训练预算；`--overview` 显示同一份概览，`--list-presets` 列出逐编号的完整模型信息。注释表由预设数据生成，并通过测试检查同步，配置仍只在 Python 中维护一份。规划器只依赖 PATH 中的 Python 3 标准库；PYTHON_BIN 仍单独指定训练与 Ray 的原生环境。

实验编号、历史实际配置和待跑项分别维护在以下系列；mirror 的目标、支持矩阵与[验证缺口](../numerics/mirror-objectives/reference.md#remaining-validation)集中在实现说明。

| 系列                                  | 入口                                                                          |
| ------------------------------------- | ----------------------------------------------------------------------------- |
| Sampled 信号、IS/PPO 与复用           | [E1–E12](../../experiments/learning-signal/sampled-controls.md)               |
| Full-KL、学习率与批内复用             | [E13–E17](../../experiments/learning-signal/sampled-full.md)                  |
| 4096 次更新的等预算对照与原始师生验证 | [E18–E20](../../experiments/reuse-budget/budget.md#equal-updates)             |
| 模型／长度失败复现与 reuse16          | [实验矩阵](../../experiments/teacher-transfer/failure-reproduction.md#matrix) |

```bash
# 每条命令对应一个独立作业；使用调度器分配的 GPU。
STUDENT_GPUS=8 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E13
CONFIG_ONLY=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E13
```

环境变量、方法选项与末尾 Hydra 参数可覆盖预设，Hydra 参数优先。各系列区分当前默认与历史实际配置：例如 E13–E17 当前默认 250 outer、开启初始验证，已归档主 run 为 200 outer、无初始验证。运行后以实际 config 和成功更新数登记[实验记录](../../experiments/运行记录.md)。

Full-KL 的 `FULL_KL_BACKEND`、token/vocab chunk、async write/prefetch 开关见[当前实现接口](../architecture/full-vocab.md)。同一对照保持 backend 一致；图表横轴与有效预算按[指标口径](../../settings/指标口径.md)解释。资源覆盖与 checkpoint 恢复见[运行与架构](runtime.md#验证执行配置与四卡八卡启动)。

<a id="launcher-polaris-跨源教师-e48"></a>

### Polaris 跨源教师 E48

E48 将 E20 的教师替换为 `Polaris-7B-Preview`，学生保持 `DeepSeek-R1-Distill-Qwen-1.5B`，开启训练前验证。Full-KL（MDPO eta=1）、DAPO、batch=mini-batch=64、n=1、reuse16、256 outer／4096 次更新、LR=1e-6、seed42 均保留；训练 prompt/response cap=4096/7168，验证 cap=31744、n=16。每 32 outer 验证、每 64 outer 保存，保留 4 个 checkpoint；关闭 group filtering 和失败组 refill。

Teacher FSDP 评分预算沿用 32768，未降低。Backend 保留`precompute`、token chunk512、vocab chunk8192，async write/prefetch开启。资源池默认8卡；本机四卡短测通过`STUDENT_GPUS=4`覆盖，实际 GPU 由调用者明确分配。短测配置与结果见[E48 四卡验收](../../experiments/teacher-transfer/cases/polaris.md)。

```bash
# 原始师生独立评测也使用同一验证协议；每条命令是一个独立作业。
bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E48-valid-student
bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E48-valid-teacher
bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E48
```

上述命令还需设置当前机器的`CUDA_VISIBLE_DEVICES`、`STUDENT_GPUS`、`PYTHON_BIN`、`MODEL_ROOT`和`DATA_ROOT`。Teacher 独立生成评测默认并发 32；这不改变训练阶段 FSDP teacher 评分预算 32768。

<a id="launcher-polaris-forward-kl-对照-e49"></a>

### Polaris Forward KL 对照 E49

E49 从 E48 复制模型、batch64、reuse16、256 outer、学习率和完整验证预算，使用`full_forward_kl`与`precompute_fused`。Teacher 显式条件化到 student 输出行，同时记录原生 RKL、FKL、两套位置桶、entropy、JS、TV 和支持集外质量。实现与内存说明见[Forward KL](../numerics/full-vocab/forward-kl.md)，数据与实验入口见[Polaris 系列](../../experiments/teacher-transfer/cases/polaris.md)。

```bash
DRY_RUN=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E49
bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E49
```

和 E48 做同版本对照时，建议显式给两组设置`FULL_KL_BACKEND=precompute_fused FULL_KL_TOKEN_CHUNK=512`。不要把历史 E48 与新代码的全部差异都归因于 KL 方向。

<a id="launcher-full-kl--mdpo--bpo-正式对照-e44e47"></a>

### Full-KL / MDPO / BPO 正式对照 E44–E47

<a id="launcher-mirror-comparison"></a>

这四组已登记为完整预设，可直接用编号启动。预算沿用 E20:256 outer、每批 reuse=16；按默认 batch=mini-batch=64、n=1 计算，每组计划 4096 次 optimizer 更新。保存间隔按用户要求改为 128 outer。

| 编号 | 方法                                 | Estimator  | κ   | η   | 对照目的                            |
| ---- | ------------------------------------ | ---------- | --- | --- | ----------------------------------- |
| E44  | 普通 Full-KL，即 MDPO eta=1 endpoint | full_vocab | —   | 1   | 当前实现下的教师 KL 基线，无 anchor |
| E45  | MDPO                                 | full_vocab | 0   | 0.3 | 几何融合目标及 anchor 约束          |
| E46  | BPO                                  | full_vocab | 0   | 0.3 | 与 E45 相同融合目标下的内层拟合方式 |
| E47  | MDPO                                 | full_vocab | 0.5 | 0.3 | 与 E45 比较 weighted teacher 的形状 |

| 共同设置                         | 默认值                                                                                            |
| -------------------------------- | ------------------------------------------------------------------------------------------------- |
| Student / teacher                | DeepSeek-R1-Distill-Qwen-1.5B / JustRL-DeepSeek-1.5B                                              |
| 训练数据                         | DAPO，全量可用数据；关闭 group filtering 与失败组 refill，与 E20 一致                             |
| Batch / mini-batch / rollout n   | 64 / 64 / 1                                                                                       |
| LR / scheduler / seed            | 1e-6 / constant、无 warmup / 42；数据、actor 和 rollout seed 对齐                                 |
| Outer / reuse / 更新数           | 256 / 16 / 4096                                                                                   |
| 训练 prompt / response cap       | 4096 / 7168                                                                                       |
| 验证                             | MATH-500、AMC23、AIME24、AIME25、AIME26；VAL_N=16，temperature=0.7，top-p=0.9，response cap=31744 |
| 验证时机                         | 沿用 E20，不做初始验证；outer 32、64、96、128、160、192、224、256                                 |
| Checkpoint                       | 每 128 outer 保存；正常完整运行保存 outer 128、256，保留上限 4 份                                 |
| Full-vocab backend / token chunk | precompute_fused / 128                                                                            |
| Hidden IO                        | async write 与 prefetch 均开启                                                                    |
| 诊断                             | E45–E47 开启 mirror diagnostics；E44 沿用 Full-KL endpoint 指标，不额外建立 anchor                |

κ=0 时，E45 的实际训练目标为 $\mathrm{KL}(p\Vert T)+(1/\eta-1)\mathrm{KL}(p\Vert q)$。E46 使用同一个冻结目标 b，但优化 q 加权残差方差；E47 改变融合形状。各目标均保留规定的 $1/\eta$ 缩放，具体定义见[实现方案](../../archive/engineering/2026-11-01-Mirror实现方案与历史验收.md#kernels)。不同 objective 的原始 loss 不直接横向排名，比较任务验证、teacher KL、anchor 漂移和相同更新预算下的实际耗时。

```bash
# 配置检查不启动 Ray 或训练。
CONFIG_ONLY=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E45

# 按实际分配的卡启动；E44/E46/E47 只需替换编号。
CUDA_VISIBLE_DEVICES=0,1,2,3 STUDENT_GPUS=4 \
  bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E45
```

师生资产和原生 Python 路径遵循仓库运行约定；本机旧目录可用 MODEL_ROOT、DATA_ROOT、PYTHON_BIN 显式指定。卡数属于运行资源，不改变默认 global batch 和更新预算。正式长序列运行前仍需验证容量：已有 1.5B smoke 的 response cap 为 64，不能作为 7168 长度下的显存保证。

E44–E47 沿用现有覆盖优先级：预设 < 环境变量 < 方法选项 < Hydra。直接调用编号即得到上表配置；覆盖方法、预算或评测参数后应按衍生设置登记，最终以 resolved_config.yaml 为准。现有命名还没有实现对所有覆盖项的自动衍生标记，不能仅凭 E 编号推断实际配置。E44 与历史 E20 的 backend、chunk、保存间隔不同，需要重新运行作为同版本对照。

配置验收已通过 247 项入口、Hydra 配置和日志识别测试；E1–E43 的 43 组启动计划与改动前一致，E46 的实际 CONFIG_ONLY 解析也通过。验收日志位于 outputs/mirror-screening-presets/acceptance.log，E46 解析配置位于同目录 E46-resolved.yaml。四组正式训练尚未启动；实际 run 产生后再进入运行记录，数值层短测和既有历史结果不计作 E44–E47 完成。

<a id="launcher-coding-与-if-训练入口"></a>

## Coding 与 IF 训练入口

三个领域共用 [\_run_single_teacher_fsdp2.sh](../../../launchers/_run_single_teacher_fsdp2.sh)，默认均为 DeepSeek-R1-Distill-Qwen-1.5B student、共享八卡、student TP=1 和 sampled-token `k1 + PG`。训练 batch 与 loss 保持原设置。coding 与 IF 的模型、训练数据和验证方式参考 One-Shot-OPD 的 `recipe/opd.sh`。[^oneshot]

| 入口                                                  | Teacher                  | 训练数据                           | 训练期间验证                            |
| ----------------------------------------------------- | ------------------------ | ---------------------------------- | --------------------------------------- |
| [math](../../../run_math_opd_ablation_fsdp2.sh)       | `JustRL-DeepSeek-1.5B`   | `math/DAPO/train.parquet`          | MATH-500、AMC23、AIME24、AIME25、AIME26 |
| [coding](../../../run_code_deepseek_r1_1_5b_fsdp2.sh) | `DeepCoder-1.5B-Preview` | `code/TACO/train.parquet`          | `code/LCBv6/test.parquet`               |
| [IF](../../../run_if_deepseek_r1_1_5b_fsdp2.sh)       | `UltraData-IF-1.5B`      | `if/MultiIF/train.parquet`，第一轮 | 关闭；正式 MultiIF 需要独立三轮评测     |

模型默认放在仓库同级 `model/`，数据默认放在同级 `data/`；已有目录可用 `MODEL_ROOT`、`DATA_ROOT` 指定。coding 可用 `CODE_VAL_FILE` 选择另一份单轮验证 parquet。现有 `prepare_models.py`、`prepare_math_data.py` 和 `make assets-check` 仍用于数学资产；coding / IF 的 teacher 和 parquet 需要另行准备。训练启动前会检查选中领域的数据文件。

```bash
bash examples/mirroropd/run_code_deepseek_r1_1_5b_fsdp2.sh

OPENCOMPASS_ROOT=/共享目录/opencompass NLTK_DATA=/共享目录/nltk_data \
  bash examples/mirroropd/run_if_deepseek_r1_1_5b_fsdp2.sh
```

IF 默认设置 `data.val_files=[]`、`TEST_FREQ=-1` 和 `trainer.val_before_train=False`，不会把三轮数据当作单轮验证。V1 支持这种只训练的配置；空验证列表配合启用验证的选项会报错。MultiIF checker 仍会对训练回答计算诊断分数，因此 IF 启动时会检查 `OPENCOMPASS_ROOT` 并加载 checker；该变量和可选的 `NLTK_DATA` 会显式传给 Ray worker。依赖见[判分器与依赖](../evaluation/判分器与依赖.md)。完整三轮生成程序尚未迁入本仓，不能把训练分数当作正式 MultiIF 评测结果。

coding / IF 即使关闭 task reward 对 loss 的贡献，仍会计算 verifier 诊断分数。coding 的 `gen` 还包含 CPU 测试执行成本；LCB runner 的全局等待预算包含新进程启动时间，部署前应使用已知正确的小程序检查判分环境。数学的采样耗时不能直接外推到这两个领域。

Hope 仍通过 `make opd-submit` 提交，worker 直接调用带领域前缀的配方，默认 math。`.hope` 的一次性迁移、DOMAIN 设置与预检边界见[Hope 直连入口](runtime.md#hope-直连入口)。

<a id="launcher-参考文献"></a>

### 参考文献

[^oneshot]: Thinking-Space. One-Shot-OPD 单教师领域配方，2026. [固定来源](https://github.com/Thinking-Space/One-Shot-OPD/blob/6b6b1782cc3f04749dd077de19353f839f6bc048/recipe/opd.sh)。

<a id="budget-check"></a>

<a id="equal-updates-附录-d实际更新预算验收"></a>

## 核对真实更新预算

新增 [verify_math_update_budget.py](../../../tools/verify_math_update_budget.py)，只读现有日志导出，不重新实现 .wandb 解码。它检查最终配置、完整逐更新记录、每个 outer 的完成记录，以及验证节点的成功更新坐标。出现跳步、非有限 KL、缺失末步汇总或缺失最终验证时返回非零状态。

```bash
# 日志先通过既有本地导出流程处理；替换 RUN_ID 和新输出目录。
python3 -m examples.mirroropd.research.studies.pipeline.analyze_run export \
  --run wncfht/MirrorOPD-verl/RUN_ID --out ../wandb-logs/derived/MirrorOPD-verl/RUN_ID-analysis

"${PYTHON_BIN:-.venv-native/bin/python}" examples/mirroropd/tools/verify_math_update_budget.py \
  --config ../wandb-logs/derived/MirrorOPD-verl/RUN_ID-analysis/config.json \
  --history ../wandb-logs/derived/MirrorOPD-verl/RUN_ID-analysis/history.jsonl \
  --expected-updates 4096
```

也接受 `resolved_config.yaml` 和未压缩 `history.jsonl`。导出文件的实际扩展名以导出目录为准。只有成功更新数相等，才把三组标记为等更新预算；验收不证明 checkpoint 文件完整、数据唯一或模型效果收敛。效果比较继续同时看 update、生成 token 和实际耗时，相同 update 数不等于相同算力消耗。

<a id="budget-presets"></a>

<a id="equal-updates-附录-be18e20-预设与运行入口"></a>

## E18–E20 的预算预设

入口仍为 [run_math_opd_ablation_fsdp2.sh](../../../run_math_opd_ablation_fsdp2.sh)。三组默认如下：

| 预设            | PPO epochs | Outer steps | 更新数 | 消费数据记录数 | TEST_FREQ | SAVE_FREQ |
| --------------- | ---------: | ----------: | -----: | -------------: | --------: | --------: |
| E18，高复用     |         64 |          64 |   4096 |           4096 |         8 |        16 |
| E19，中复用     |         32 |         128 |   4096 |           8192 |        16 |        32 |
| E20，更多新轨迹 |         16 |         256 |   4096 |          16384 |        32 |        64 |

频率单位为 outer step。三组都在 update 512、1024、……、4096 验证，在 update 1024、2048、3072、4096 保存；保留四份 checkpoint。默认不做训练前验证，原始 student 的独立验证作为共享参考。

共同条件：`TOTAL_EPOCHS=1`，全局 batch=mini-batch=64，每题一条训练回答，seed=42，constant LR=1e-6、零 warmup、与 E17 相同的 AdamW 和梯度裁剪。Full-KL 使用 `precompute`、async write 和 prefetch；不额外加入 sampled-token IS 或 PPO clipping。动态分组筛选和失败补采关闭，采用同步训练。

数据仍是原始 `math/DAPO/train.parquet`，没有派生去重文件。已有 17917 条记录，当前长度配置过滤后数量不变，最多 279 个完整 batch。E20 的 256 个 batch 消费 16384 条，留有 1533 条余量。三组使用同一文件、同一 data seed、`train_max_samples=-1` 和现有独立随机 sampler，较短实验读取较长实验的题目序列前缀；相同题目后续生成的回答不要求一致。

```bash
# 三条命令分别用于独立作业，不自动申请或争用同一组 GPU。
STUDENT_GPUS=8 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E18
STUDENT_GPUS=8 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E19
STUDENT_GPUS=8 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E20

CONFIG_ONLY=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E20
```

所有预设仍允许环境变量及末尾 Hydra 覆盖，方便小规模 smoke。覆盖 batch、epochs、outer 或数据后，原表的 4096 updates 和单遍容量不再自动成立，应重新计算。没有增加通用 trainer 预算约束，也没有改变数据耗尽后的既有行为；正常预设通过单数据 epoch 和足够的样本容量避免回绕。

<a id="original-model-validation"></a>

<a id="equal-updates-附录-a原始-student--teacher-验证"></a>

## 原始 Student 与 Teacher 的独立验证

新增 [run_math_validation_fsdp2.sh](../../../run_math_validation_fsdp2.sh)，调用既有数学公共脚本和 V1 `_validate()`。`student` 选择 `STUDENT_MODEL`，`teacher` 选择 `TEACHER_MODEL`，然后把所选模型放到 `actor_rollout_ref.model.path` 作为生成模型。默认路径分别为 `MODEL_ROOT/DeepSeek-R1-Distill-Qwen-1.5B` 和 `MODEL_ROOT/JustRL-DeepSeek-1.5B`。

包装脚本设置 `distillation.enabled=False`、`trainer.val_only=True`、`trainer.val_before_train=True`、`trainer.resume_mode=disable`、`trainer.save_freq=-1`；没有蒸馏 teacher worker，也不执行 optimizer 更新。资源变量 `STUDENT_GPUS` 在这里表示被评测模型的共享资源池。沿用现有 trainer 初始化，不新增另一套生成或评分实现。

默认评测预算仍是 MATH-500/AMC23/AIME25 的 500/40/30 题，每题 16 条回答，temperature=0.7、top-p=0.9、回答上限 31744。全量提交、CUDA Graph、TP1、每副本并发 128、显存比例 0.65、调度 token 预算 8192，来自已有公共数学配置，包装脚本不复制这些默认值。

采集现有 mean accuracy、pass@k、majority@k、长度统计、停止原因、截断/答案缺失和完整性指标。逐回答结果沿用 `RUN_DIR/validation/records/0.jsonl`，其他输出及原生环境见[运行与架构](runtime.md)。生成阶段保持不返回训练专用 log-prob；本轮没有新增 hidden/head 分块或概率重放。

从仓库根目录，在各自已分配 GPU 的作业中运行：

```bash
# CUDA_VISIBLE_DEVICES 使用作业实际获分配的设备；STUDENT_GPUS 与数量一致。
STUDENT_GPUS=8 bash examples/mirroropd/run_math_validation_fsdp2.sh student
STUDENT_GPUS=8 bash examples/mirroropd/run_math_validation_fsdp2.sh teacher

# 检查最终命令或解析配置，不启动 Ray/GPU。
DRY_RUN=1 bash examples/mirroropd/run_math_validation_fsdp2.sh teacher
CONFIG_ONLY=1 bash examples/mirroropd/run_math_validation_fsdp2.sh student
```

本机原有模型、数据、Python 不在默认资产位置时，显式设置：

```bash
export MODEL_ROOT=/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/model
export DATA_ROOT=/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/MD-OPD/data
export PYTHON_BIN=/root/.local/share/mirroropd/venvs/profile-cu128/bin/python
export VERL_USE_UV=0
# 仅在 GPU0–3 空闲、已获得共享使用权时运行本机四卡任务。
export CUDA_VISIBLE_DEVICES=0,1,2,3
export STUDENT_GPUS=4
```

自定义权重可以用 `STUDENT_MODEL` / `TEACHER_MODEL`，或末尾 `actor_rollout_ref.model.path=...` 覆盖。末尾 Hydra 参数仍有最高优先级，包括 val-only 等开关；修改后按实际合成配置判断作业行为。两个模型使用独立实验目录，避免覆盖结果。既有初始化依赖仍会读取训练数据，因此纯 valid 入口也要求正常的数据资产。
