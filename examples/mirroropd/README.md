# MirrorOPD 数学训练与 verifier

四个领域的 verifier 从 One-Shot-OPD commit `6b6b1782cc3f04749dd077de19353f839f6bc048` 迁入，数值与任务判分逻辑保留在 `verl/utils/reward_score/one_shot_opd/`。这里提供一个原生 custom reward 入口；训练算法、reward 冻结方式和 IS 尚未修改。[^source]

## 单教师数学 full-data 配方

[run_deepseek_r1_1_5b_fsdp2.sh](run_deepseek_r1_1_5b_fsdp2.sh) 使用 V1 同步 trainer、FSDP2 student 和独立 vLLM teacher。数学设置参考 One-Shot-OPD 当前可执行配方：DeepSeek-R1-Distill-Qwen-1.5B student、JustRL-DeepSeek-1.5B teacher、完整 DAPO 数据、sampled-token `k1 + PG`、batch 与 mini-batch 均为 64、每批一次更新、AdamW 学习率 1e-6。目前仍采用主线在 loss 中构造的 current-policy advantage；未加入 frozen-reward 分支，不能称为原研究实现的完整等价复现。

默认资源与长度：

| 项目 | 默认值 |
| --- | --- |
| Student 资源池 | 4 张 GPU；FSDP2 数据并行，训练与 rollout 共用 |
| Student rollout | TP=1，共 4 个推理副本 |
| Teacher 资源池 | 1 张 GPU；TP=1，1 个副本 |
| Prompt / 训练 response | 4096 / 7168 token |
| 验证 response | 31744 token |
| Student / teacher 最大 context | 35841 / 11265 token，均预留一个 token |
| 验证集 | MATH-500、AMC23、AIME25；500 + 40 + 30 题 |
| 验证采样 | 每题 16 次，temperature=0.7、top-p=0.9 |
| 验证提交窗口 / 推理并发 | 每窗口 16 题；每个 student 推理副本最多 32 条序列 |
| 训练预算 | 1 epoch；全量 17917 条数据未过滤时约 279 steps |
| 验证 / 保存间隔 | 20 / 50 global steps，并在最后一步触发 |

这是一组面向 1.5B 模型、H800 80GB 的初始配置，尚未做 GPU 吞吐验收。先保留 batch=64，依据 teacher 评分耗时再调整 teacher 副本数。修改资源时，batch 必须是 student DP 数的整数倍，各推理资源池的 GPU 数必须是其 TP 的整数倍；`STUDENT_GPUS` 与 `TEACHER_GPUS` 是相加的独立资源池，不是同一组卡的两种计数。

模型默认从仓库相邻的 `models/` 读取，数据默认从相邻 `MD-OPD/data/` 读取；可用 `MODEL_ROOT`、`DATA_ROOT` 或各文件变量覆盖。数据文件分别是 `math/DAPO/train.parquet`、`math/MATH-500/test.parquet`、`math/AMC23/test.parquet` 和 `math/AIME25/test.parquet`。

```bash
# 仅打印启动命令，不启动 Ray，也不安装依赖。
DRY_RUN=1 bash examples/mirroropd/run_deepseek_r1_1_5b_fsdp2.sh

# 只解析 Hydra 配置；默认通过锁文件的 fsdp + vllm + math extras 运行。
CONFIG_ONLY=1 bash examples/mirroropd/run_deepseek_r1_1_5b_fsdp2.sh

# 获得 GPU 分配后设置 CUDA_VISIBLE_DEVICES，至少覆盖默认的 5 张卡。
# 脚本不会选择物理 GPU，也不会清理其他训练进程。
bash examples/mirroropd/run_deepseek_r1_1_5b_fsdp2.sh
```

默认 uv 启动同步配置 driver 与 Ray worker 的 Python。已有独立环境可设置 `VERL_USE_UV=0 PYTHON_BIN=/absolute/path/to/python`；环境需具备当前 verl 的 V1/TransferQueue 和 verifier 依赖。正式启动要求显式设置 `CUDA_VISIBLE_DEVICES`，并检查四个 parquet 文件存在。

`RUN_DIR` 必须是新目录，默认在 `outputs/` 下；保存实际命令、Git commit、相对 HEAD 的工作区 patch、解析配置、checkpoint 和验证输出。CLI 末尾参数仍是原生 Hydra overrides。改变训练步数时还要给足 `TOTAL_EPOCHS`，因为外层训练同时受 epoch 与 total step 限制。

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
