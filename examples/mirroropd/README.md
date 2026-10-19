# MirrorOPD 数学训练与 verifier

四个领域的 verifier 从 One-Shot-OPD commit `6b6b1782cc3f04749dd077de19353f839f6bc048` 迁入，数值与任务判分逻辑保留在 `verl/utils/reward_score/one_shot_opd/`。这里提供一个原生 custom reward 入口；训练算法、reward 冻结方式和 IS 尚未修改。[^source]

## 数学资产准备与检查

模型默认放在仓库同级的 `model/`（单数），数据放在同级 `data/`。开发机基目录为 `/home/hadoop-hmart-poistar/dolphinfs_hdd_hadoop-hmart-poistar/fanghaotian/`：

```text
fanghaotian/
├── MirrorOPD-verl/
├── model/
│   ├── DeepSeek-R1-Distill-Qwen-1.5B/
│   └── JustRL-DeepSeek-1.5B/
└── data/
    ├── math_manifest.json
    └── math/
        ├── DAPO/train.parquet
        ├── MATH-500/test.parquet
        ├── AMC23/test.parquet
        └── AIME25/test.parquet
```

| 脚本 | 用途 |
| --- | --- |
| [prepare_models.py](prepare_models.py) | 从已有目录复制模型文件、物化缓存符号链接，或显式下载固定 Hub revision；写 `model_manifest.json` |
| [prepare_math_data.py](prepare_math_data.py) | 导入已有 parquet，或从五个原始文件重建四份数学数据；统一 Arrow schema，写 `math_manifest.json` |
| [check_assets.py](check_assets.py) | 模型仅检查必要文件；逐行检查数据，统计 prompt 长度、重复与训练/验证题面重叠，写 `asset_check.json` |

在已有资产的机器上，可以离线导入旧目录；路径参数填写该机器的实际绝对路径：

```bash
.venv-native/bin/python examples/mirroropd/prepare_models.py \
  --source-root /已有的/models --model-root /目标/model
.venv-native/bin/python examples/mirroropd/prepare_math_data.py \
  --source-root /已有的/MD-OPD/data --data-root /目标/data
```

原文件保留，目标文件按新目录创建。数据输出已存在时默认拒绝覆盖；确认要重建时显式传 `--overwrite`。模型目标已存在时直接运行检查，无需重新导入。复制两个模型约涉及 7.1 GB 权重，只需执行一次。

从原始数据重建时使用 `--raw-dir`；源目录包含 `dapo-math-17k.parquet`、`math500_test.jsonl`、`amc23.parquet`、`aime2025-I.jsonl`、`aime2025-II.jsonl`。转换沿用固定 MD-OPD 的数学规则：DAPO 按源 `extra_info.index` 保留第一次出现，去掉原始 Answer 格式包装；AMC/AIME 与 MATH-500 保留各自题面和答案，AIME 按 I、II 顺序合并。[^math-prep]

```bash
.venv-native/bin/python examples/mirroropd/prepare_math_data.py --raw-dir /原始文件目录

# 仅在能访问外网的机器上显式下载；不会安装新的 Python 包。
.venv-native/bin/python examples/mirroropd/prepare_models.py --download
.venv-native/bin/python examples/mirroropd/prepare_math_data.py --download
```

模型下载固定为 DeepSeek `ad9f0ae0`、JustRL `0637e409`。[^student-model] [^teacher-model] 数据下载使用 ModelScope 的上述原始数据镜像，镜像 revision 为 `master`；manifest 记录来源与数据行数。重复实验应保留原始文件，不把浮动镜像 revision 当作不可变版本。

准备完成后在开发机执行：

```bash
make assets-check
# 只检查数据，不要求模型已到位：
make assets-check ASSET_CHECK_ARGS=--data-only
```

默认模型检查只要求 `config.json`、`tokenizer.json`、`tokenizer_config.json`、`model.safetensors` 存在且非空，配置 JSON 能解析。`generation_config.json` 是可选文件，缺少时不报错；准备脚本在来源中有该文件时会保留它。检查不读取完整权重或逐张量核对，也不计算文件摘要。student tokenizer 仅用于计算数据 prompt 长度；`--data-only` 跳过模型和 tokenizer，不计算 token 长度。

检查要求四份数据分别为 17917、500、40、30 行，五个字段及嵌套类型一致；每行必须有非空 user prompt、正确 `data_source`、`ability=math`、非空字符串答案、`style=rule`、正确 split 和唯一 `extra_info.index`。直接检查实际 parquet 内容，不依赖 manifest。prompt 长度使用 student 的 chat template 和 generation prompt 计算；超过限制的数量会报告，训练端仍按 `filter_overlong_prompts` 处理。这些检查确认数据格式与训练入口兼容，不能判定题目的标准答案是否正确。

重复题面和训练/验证交集只报告，不自动改变配方。2026-10-20 本机重建的 18487 行与既有数据逐行一致；DAPO 有 662 行重复题面，与 MATH-500、AMC23、AIME25 分别有 2、5、0 道完全相同的题面。完整词面匹配不是语义去污染；若要删除重叠，应另建数据版本和对照实验。

同次 CPU 检查确认四份数据的最大 prompt 长度分别为 1493、796、220、258 token，均未超过 4096；没有执行 GPU 训练。

模型和数据需要另行上传一次，不进入 `make archive-pack` 的源码包。数据传输包应包含顶层 `data/`，在开发机解压到上述 `fanghaotian/` 基目录后再运行检查；这类资产包使用普通 `tar -xzf ... -C <基目录>`，不使用会保护运行目录的源码 `archive-unpack`。

## 单教师数学 full-data 配方

[run_deepseek_r1_1_5b_fsdp2.sh](run_deepseek_r1_1_5b_fsdp2.sh) 使用 V1 同步 trainer、FSDP2 student 和独立 vLLM teacher。数学设置参考 One-Shot-OPD 当前可执行配方：DeepSeek-R1-Distill-Qwen-1.5B student、JustRL-DeepSeek-1.5B teacher、完整 DAPO 数据、sampled-token `k1 + PG`、batch 与 mini-batch 均为 64、每批一次更新、AdamW 学习率 1e-6。目前仍采用主线在 loss 中构造的 current-policy advantage；未加入 frozen-reward 分支，不能称为原研究实现的完整等价复现。

默认资源与长度：

| 项目 | 默认值 |
| --- | --- |
| Student 资源池 | 4 张 GPU；FSDP2 数据并行，训练与 rollout 共用 |
| Student rollout | TP=1，共 4 个推理副本 |
| Teacher 资源池 | 4 张 GPU；TP=1，同一个 teacher 的 4 个评分副本 |
| Prompt / 训练 response | 4096 / 7168 token |
| 验证 response | 31744 token |
| Student / teacher 最大 context | 35841 / 11265 token，均预留一个 token |
| 验证集 | MATH-500、AMC23、AIME25；500 + 40 + 30 题 |
| 验证采样 | 每题 16 次，temperature=0.7、top-p=0.9 |
| 验证提交窗口 / 推理并发 | 每窗口 16 题；每个 student 推理副本最多 32 条序列 |
| 训练预算 | 1 epoch；全量 17917 条数据未过滤时约 279 steps |
| 验证 / 保存间隔 | 20 / 50 global steps，并在最后一步触发 |

默认训练 1 epoch。当前 17917 条 DAPO 的 prompt 均未超过 4096，过滤后仍是 17917 条；V1 trainer 按 `floor(17917 / 64) = 279` 计算步数，最后不足一个 batch 的 61 条不参与这个 epoch。每步全局采样 64 道题，每题生成 1 条回答；`ppo_mini_batch_size=64`、`ppo_epochs=1`，每步做一次全局参数更新。64 是整个 student 资源池的 batch；4 个 student 训练 rank 平均各处理 16 条，内部动态 token batching 与梯度累积按实际序列长度安排。

验证发生在 step 20、40、……、260 和最终 step 279，共 14 次；`val_before_train=False`，训练前不额外验证。每次完整覆盖 MATH-500 的 500 题、AMC23 的 40 题、AIME25 的 30 题，合计 570 题，每题采样 16 次，共生成 9120 条回答。`VAL_BATCH_SIZE=16` 是每次向推理端提交 16 道题的窗口，最多展开为 256 条回答请求；实际同时运行的请求数还受每个推理副本 `max_num_seqs=32` 约束。每条验证回答最多 31744 token，验证生成量会显著增加训练总耗时。checkpoint 保存于 step 50、100、150、200、250、279。

TP（Tensor Parallelism，张量并行）把一个模型副本中的权重矩阵及其计算拆到多张 GPU 上协作完成。推理 TP=1 时，一个副本占 1 张卡；TP=2 时，一个副本占 2 张卡。当前 student rollout 的 4 卡、TP=1 对应 4 个相同模型的生成副本；teacher 的 4 卡、TP=1 对应 4 个相同模型的评分副本。若同一个 4 卡资源池改为 TP=2，则只有 2 个推理副本。student 参数更新另用 FSDP2 在 4 张卡上分片训练；推理 TP 与训练的 FSDP2 是两组独立配置。

默认使用全部 8 张卡：4 张用于 student，4 张用于 teacher。两个 1.5B 模型先用 TP=1，通过独立副本并行处理请求；这是面向 80GB GPU 的初始配置，尚未做 GPU 吞吐验收。ResOPD 的公开配方也用 4+4，但其 Qwen3.5-27B teacher 用 TP=4，将一个副本放在四张卡上。[^resopd] 本配方的四个 teacher 副本权重相同，仍是单教师蒸馏。

分配全部 8 张卡不代表每个阶段全部满载：同步训练在 student 生成期间可并发评分已经完成的轨迹，actor 更新期间 teacher 空闲，验证也不调用 teacher。后续依据评分耗时与 rollout 耗时判断四个评分副本的收益。需要缩减资源时可设 `TEACHER_GPUS=1`，恢复 4+1。修改资源时，batch 必须是 student DP 数的整数倍，各推理资源池的 GPU 数必须是其 TP 的整数倍；`STUDENT_GPUS` 与 `TEACHER_GPUS` 是相加的独立资源池，不是同一组卡的两种计数。

脚本将常用环境变量集中在前半部分，按模型与数据、GPU、训练与验证预算分组；原生 Hydra 参数按 data、actor、rollout、validation、distillation、algorithm、trainer 分成数组，最后统一组装。通常改顶部默认值或使用环境变量即可；新增原生配置放入对应数组。

模型默认从仓库相邻的 `model/` 读取，数据默认从相邻 `data/` 读取；可用 `MODEL_ROOT`、`DATA_ROOT` 或各文件变量覆盖。目录、准备与检查入口见前文。数据文件分别是 `math/DAPO/train.parquet`、`math/MATH-500/test.parquet`、`math/AMC23/test.parquet` 和 `math/AIME25/test.parquet`。

```bash
# 仅打印启动命令，不启动 Ray，也不安装依赖。
DRY_RUN=1 bash examples/mirroropd/run_deepseek_r1_1_5b_fsdp2.sh

# 只解析 Hydra 配置；默认使用本仓库的 .venv-native/bin/python。
CONFIG_ONLY=1 bash examples/mirroropd/run_deepseek_r1_1_5b_fsdp2.sh

# 获得整机 8 张卡的分配、激活原生 Python 环境后运行。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VERL_USE_UV=0 \
  bash examples/mirroropd/run_deepseek_r1_1_5b_fsdp2.sh
```

默认使用本仓库 `.venv-native/bin/python` 的绝对路径，同时设置 driver 和 Ray worker 的解释器，无需依赖终端是否激活环境。已有其他独立环境可设置 `PYTHON_BIN=/absolute/path/to/python`；环境需具备当前 verl 的 V1/TransferQueue 和 verifier 依赖。上游锁文件环境只在显式设置 `VERL_USE_UV=1` 时启用。正式启动要求显式设置 `CUDA_VISIBLE_DEVICES`，并检查四个 parquet 文件存在。

`RUN_DIR` 必须是新目录，默认在 `outputs/` 下；保存实际命令、Git commit、相对 HEAD 的工作区 patch、解析配置、checkpoint 和验证输出。CLI 末尾参数仍是原生 Hydra overrides。改变训练步数时还要给足 `TOTAL_EPOCHS`，因为外层训练同时受 epoch 与 total step 限制。

正式运行会将 stdout/stderr 同时显示在终端并保存到 `$RUN_DIR/train.log`，默认路径为 `outputs/<实验名>/train.log`。Python 使用无缓冲输出；训练失败时保留非零退出码。启动时打印日志的完整路径，可用 `tail -f <日志完整路径>` 跟踪。`DRY_RUN=1` 和 `CONFIG_ONLY=1` 不创建日志或实验目录。

通过人工源码包部署、没有 `.git` 的目录同样可以启动；此时 `source_commit.txt` 标记版本不可用，`source_changes.patch` 为空。开发机已有本地 Hope 配置时，可通过 `make opd-submit` 提交单机 8 卡任务；`hope/` 不随源码包同步，完整操作见 [Hope 说明](../../hope/README.md)。

配方默认记录 console 与离线 W&B。SDK 写入根目录固定为本仓库，实际 run 位于 `wandb/offline-run-*/`；Ray 训练进程使用同样的路径和模式。可用 `make sync-log-pack` 打包后人工下载，完整流程见 [美团开发机同步说明](../../meituan/README.md)。

## 独立验证长度

新增 `actor_rollout_ref.rollout.val_kwargs.max_tokens`，默认 `null`，保留原有训练/验证共用 response budget 的行为。非空时，验证请求和最终 response、mask、log-prob、router replay 输出使用同一预算；V1 的 TQ 保存未 padding 的实际输出，旧 DataProto 路径按验证预算 padding。

当前支持 vLLM/SGLang 的 `single_turn_agent`；本配方使用 vLLM。配置或数据行选择其他 agent 时会拒绝独立验证预算，避免多轮 agent 静默沿用训练上限；会覆盖请求长度的其他后端也会被配置检查拒绝。应显式配置足以容纳 prompt 与验证预算的 `rollout.max_model_len`；底层服务仍受模型 context 和显存约束。该设置不改变 teacher 的训练评分长度。

ResOPD 的公开版本 `3bdf019d` 只设置一个 `MAX_RESPONSE_LENGTH`，没有这条独立验证长度配置；本配方不依赖它的启动脚本。[^resopd]

## 使用

从仓库根目录向现有训练命令追加：

```bash
reward.custom_reward_function.path=examples/mirroropd/reward.py \
reward.custom_reward_function.name=reward_func
```

路径配置使用本仓代码，不依赖旁边的 One-Shot-OPD checkout。路由按 `data_source` 精确匹配，未知来源报错；原实现把未知来源交给数学 verifier，容易掩盖数据配置错误。

| 领域 | 主要 data_source | 判分与返回 |
| --- | --- | --- |
| 数学 | `DAPO`、`MATH-500`、`AMC23`、`AIME25` | 先提取 boxed answer；缺少 boxed answer 时对末尾 2000 字符使用 Math-Verify；返回 `score`、`acc`、格式等指标 |
| 代码 | `taco`、`livecodebench`、`humanevalplus` 等 | 保留原代码提取、stdin/function 测试与 supplied-test 执行逻辑 |
| 指令遵循 | `MultiIF` | 使用 OpenCompass MultiIF checker，返回 strict/loose 的 prompt 与 instruction 统计 |
| 函数调用 | `xlam`、`bfcl_simple`、`bfcl_multiple`、`bfcl_parallel`、`bfcl_parallel_multiple`、`bfcl_live_simple` | 保留原调用解析、参数规范化及顺序无关匹配 |

完整来源集合见 [dispatcher](../../verl/utils/reward_score/one_shot_opd/__init__.py)。各领域延迟导入，使用数学或 FC 不会加载 Multi-IF 的依赖。

## 依赖与运行边界

- 数学需要 `math-verify`、`latex2sympy2_extended`、`pylatexenc` 和 `sympy`；保留原 timeout helper，复用当前 verl 的 Math-Verify subprocess 实现。
- 代码执行需要 Linux/POSIX、NumPy 和 pandas。HumanEval+ 的批量 EvalPlus API 另外需要 `evalplus`；单 completion 的 supplied-test 路径不依赖它。进程隔离与资源限制不构成不可信代码的安全沙箱。
- Multi-IF 需要设置 `OPENCOMPASS_ROOT`，指向包含 `opencompass/datasets/MultiIF/` 的 OpenCompass 源码树；需要 `nltk`、`absl-py`，语言检测还需要 `langdetect`，句子判分需要 NLTK 的 `punkt` / `punkt_tab` 数据。缺少 checker 源码会报错，不计为答错。
- FC verifier 对齐原仓库的 xLAM / 五类 BFCL 数据约定，不能代表完整 BFCL 官方评测流程。
- Multi-IF 这里迁移的是逐轮 verifier 和 `score_single_turn()`。完整多轮生成与最后一轮/多语言聚合仍需相应 AgentLoop 或评测调用方；此次没有迁移独立生成程序。

依赖应通过 uv 安装在任务自己的环境中，不修改其他训练任务的共享环境。OpenCompass 验证使用 commit `efb91b2809e91997c5d043d468e4aa986fbb341f` 的四个 checker 文件。[^opencompass]

## 相对来源的接入修改

- 统一入口改为明确路由与延迟加载；不迁入原 math 包中失效的旧 code 路由。
- 原有数学、代码和 FC 判分规则保留；来源版权声明及 LiveCodeBench MIT 许可证随文件保留。
- LCB 使用可在 reward 线程中调用的 spawn worker，关闭 manager 并回收子进程。
- supplied-test 执行使用当前 Python，超时清理自己创建的进程组；资源限制在子解释器执行，避免在线程中使用 `preexec_fn`。
- 去除批量 HumanEval+ 中未消费结果的重复 sanitize，保持按输入顺序输出。

## 检查

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q tests/utils/reward_score/test_one_shot_opd_on_cpu.py
```

设置 `OPENCOMPASS_ROOT` 后会运行真实 Multi-IF checker 测试；未设置时仅跳过该项，其余路由、缺依赖报错、数学、代码执行和 FC 检查仍运行。测试覆盖正确/错误答案、数学回退、reward 线程调用、stdin/function 测试、超时、子进程回收、FC 参数与多调用匹配、Hydra 文件入口。完整 benchmark 与 GPU 训练不属于这些 CPU 测试的覆盖范围。

2026-10-20 的 CPU 验证为 33 passed，包含上述固定 OpenCompass checker；环境为 Python 3.12.15、PyTorch 2.7.1、Math-Verify 0.9.0、latex2sympy2_extended 1.11.0、pylatexenc 2.11、NumPy 1.26.4。另通过 Ruff、Python 语法与文档本地链接检查。该测试使用现有独立环境，未更改仓库依赖锁，也不代表已在锁定的完整 GPU 栈中验收。

独立长度和 launcher 的针对性测试：

```bash
python -m pytest -q tests/experimental/agent_loop/test_validation_response_budget_on_cpu.py \
  tests/examples/test_mirroropd_math_launcher_on_cpu.py
```

2026-10-20 核对本机三个数学验证 parquet 共 570 题，用当前 student tokenizer 计算的最大 prompt 长度分别为 796、220、258 token；均未超过配方的 4096 上限。CPU 测试使用 mocked generation 和 TQ 写入来验证长度传递；真实 GPU 训练尚未执行。

### 参考文献

[^source]: Thinking-Space. One-Shot-OPD task verifiers. [固定来源](https://github.com/Thinking-Space/One-Shot-OPD/tree/6b6b1782cc3f04749dd077de19353f839f6bc048/verl/verl/utils/reward_score).
[^opencompass]: OpenCompass. MultiIF instruction checkers. [固定来源](https://github.com/open-compass/opencompass/tree/efb91b2809e91997c5d043d468e4aa986fbb341f/opencompass/datasets/MultiIF).
[^resopd]: InternLM. ResOPD training launcher. [固定来源](https://github.com/InternLM/ResOPD/blob/3bdf019daf9818419fe38dbc6516e41ca5a70dcb/examples/resopd/run_resopd.sh).
[^math-prep]: MD-OPD contributors. 数学数据准备，commit `cef14fa0b552b40033f8b7296ce3853d10bf479b`. [固定来源](https://github.com/WncFht/MD-OPD/tree/cef14fa0b552b40033f8b7296ce3853d10bf479b/data/prep)。本仓独立实现相同转换，不在运行时导入相邻 checkout。
[^student-model]: DeepSeek. DeepSeek-R1-Distill-Qwen-1.5B. [固定快照](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/tree/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562)。
[^teacher-model]: hbx. JustRL-DeepSeek-1.5B. [固定快照](https://huggingface.co/hbx/JustRL-DeepSeek-1.5B/tree/0637e4096c789c67f9eecbe8355e0bdeddede1c2)。
