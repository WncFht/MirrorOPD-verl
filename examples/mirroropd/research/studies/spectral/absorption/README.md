# OPD 谱吸收实验

本包运行同源 base / teacher / OPD checkpoint 的谱干预与梯度诊断，统一 Qwen、R1、Llama、MiMo 的执行入口。`base` 是对应配对的 initial student。科学解释见[谱实验报告](../../../../docs/experiments/teacher-transfer/absorption/README.md)。

从仓库根目录执行。运行目录保存输入、源码快照、缓存与结果；可维护实现位于本包。旧实验继续使用原快照，新实验通过显式 `--run-root` 读取配置。

## 走读顺序

| 源码                                             | 负责什么                                   | 主要输入与输出                              |
| ------------------------------------------------ | ------------------------------------------ | ------------------------------------------- |
| `protocol.py`                                    | 路径、停止信号、tensor 原子落盘            | `RunContext`，显式 run root                 |
| `banks.py`                                       | 生成固定状态、teacher 参考、原生全词表风险 | token bank → 逐题/逐 token RKL、FKL、JS、熵 |
| `directions.py`                                  | 精确 SVD、方向缓存、参数干预               | checkpoint 三元组 → 矩阵方向、编辑回执      |
| `gradients.py`                                   | 精确 RKL 反向、按题/层/模块的投影与梯度和  | bank + 方向 → 局部导数、向量一致性          |
| `journal.py`                                     | 按状态组恢复                               | 身份一致的完整组 → 跳过已完成计算           |
| `plan.py`                                        | 分片预检和输入准备                         | manifest + prompts → 冻结源码和 task ledger |
| `queue.py`                                       | 原子 claim、依赖、独立 attempt、调度       | 显式任务 → broker/guard worker              |
| `monitor.py`                                     | CPU 巡检与完成后分析                       | 结果标记 → 覆盖表、统计结果                 |
| `analyze*.py`、`plot.py`                         | 配对统计、层级聚合、论文图                 | 不可变结果 → JSON/CSV/图                    |
| `benchmark.py`、`benchmark_model.py`、`infra.py` | 真实方向与真实反向的性能/一致性验收        | FP64 CPU 参考 → 误差、耗时、显存            |

通用投影位于 `research/backends/direction_projection.py`；资源适配位于 `research/runtime/broker.py`。两者不包含 teacher 标签或科学参数。矩阵数学复用上级 `spectral/decomposition.py`，checkpoint 流式读取复用 `spectral/weights.py`，模型执行复用 `teacher_difference/runtime.py`。

## 实际测量协议

- `manifest.json` 包含 `jobs`，每项显式给出 `job_id`、`base`、`teacher_path`、`endpoint`、`gradient_stages`、`family`、`run_id`；其余研究元数据原样保留供统计使用。`prompts.json` 每项含 `row_id`、`domain`、`split`、`messages`。
- 两个状态来源是 initial student 与 teacher 的 greedy 输出，各题 cap 128。Qwen 通过已有 `run_id` 的 nonthinking 标识控制模板；批量生成采用左填充、batch 4。它们是新生成的诊断状态。
- 模型采用 FP32 master、BF16 backbone、FP32 head；teacher 参考与被评分模型使用相同执行。保存 full-vocabulary RKL、FKL、JS、两侧熵。主值是每题求和除固定 H128，另存实际 token 均值与逐 token 数组。
- teacher-self floor 和无编辑导出重载的最大逐题 RKL 误差均要求不超过 `1e-6`。
- 对全部 attention/MLP 二维权重分别做精确 FP64 SVD。Learned 是 `endpoint - base`，missing 是 `teacher - endpoint`；off/null 来自 base thin-SVD frame。方阵 null 显式归零。
- 25 个干预条件包含 learned top 1%/5% 的原幅度与恢复范数、missing off/null 的三个 beta、各 band 的三 seed Gaussian/随机框架，以及 off 的前三分层段。加 base、endpoint、teacher-self、reload 共 29 个评分文件。
- 梯度阶段按每个状态来源 × split × domain 的前两题、首尾位置采样，常见为 48 sites。反向计算所有选中矩阵；保留逐位置投影与每组完整梯度和。方向单位导数是 `<g,d>/||d||`，向量一致性是 `||sum g_i|| / sum ||g_i||`。组合层与固定层分别聚合。

## 单任务与 CPU 分析

```bash
python -m examples.mirroropd.research spectral.absorption --help
python -m examples.mirroropd.research spectral.absorption \
  --run-root "$RUN" --job 8 --attempt "$ATTEMPT" --phase gradients \
  --projection cuda-stream
python -m examples.mirroropd.research spectral.absorption.analyze_interventions \
  --run-root "$RUN" --output-dir "$ANALYSIS"
python -m examples.mirroropd.research spectral.absorption.analyze_gradients \
  --run-root "$RUN" --output "$GRADIENT_JSON"
```

GPU 命令由现有 broker/resource guard 启动。`cpu-reference` 是默认投影实现，`cuda-stream` 是已提供数值验收入口的优化实现。绘图前设置 `FIGURE_VIZ_MPL`，然后调用 `spectral.absorption.plot --run-root ... --output-dir ...`。

## 资源与恢复

新队列先由 `spectral.absorption.plan` 读取明确的 manifest/prompts，预检实际权重分片，复制 research 与依赖源码。它支持 `--input-run` 复用旧 bank、参考、方向和完整梯度阶段；每项写入新 attempt，旧数据通过链接只读使用。输出目录已存在就报错，避免覆盖失败证据。

随后执行 `spectral.absorption.queue run --run-root ... --repo ... --resource ... --gpus ...`。`--resource` 指向本机已运行的 broker；只有该 broker 确认处于 `occupying` 的卡才会申请。每卡经过 reserve、guard、attach、启动门再执行。任务 claim 由文件锁保护；同一新队列的 supervisor 锁防止重复启动。旧队列迁移必须先在它自己的队列锁下移交尚未启动的任务。`--stop-path` 可使接续队列同时遵守前驱 STOP。

梯度完整组先保存 tensor，再发布含 SHA256 的组 JSON。恢复要求模型、bank、矩阵记录和投影 backend 身份一致，并核验 tensor 大小与 SHA256；不完整组重算，已完成组复用。遇到错误保留 attempt 与错误回执。完成后 broker 核验 owner 退出、锁释放和 CUDA context 消失，自动恢复 `gpu-occupy`；新队列还记录最终恢复检查。

## 性能验收

GPU 投影缓存三个 FP32 方向的锁页内存，每次只传一个矩阵，在 GPU 转 FP64 并完成点积与范数，仅标量回传 CPU。方向范数跨位置复用；完整梯度仍以 FP32 回传 CPU，按原有顺序累积组梯度和。主机内存成本约为三个选中参数集合，显存临时成本为一个矩阵的 FP64 计算。

`benchmark` 使用真实方向与固定随机梯度；`benchmark_model` 使用真实模型一次 RKL backward 的全层梯度，包含完整梯度 D2H。验收比较范数相对误差及点积误差除以 `||g|| ||d||`，阈值 `1e-10`；后者能处理接近零的抵消点积。分别记录冷缓存和预热缓存时间，不能把投影速度比直接写成整轮实验速度比。

CPU 回归：

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/examples/research/test_direction_projection_on_cpu.py
```

已验收的真实梯度基准（同一 backward，196 个矩阵；单位秒）：

| 模型           | CPU 投影与 D2H | GPU 冷缓存 | GPU 预热缓存 | 最大归一化误差 |
| -------------- | -------------: | ---------: | -----------: | -------------: |
| Qwen SFT       |          39.34 |      14.62 |         2.05 |       3.35e-12 |
| Llama CoReward |          80.33 |      34.24 |         4.96 |       6.34e-12 |

完整测量与占用恢复回执见[基准数据](../../../../docs/infra/performance/data/spectral-projection-20261110/summary.json)。这些值包含梯度回传，不包含组 tensor 写盘、模型加载或整轮采样。首次读取未缓存的方向文件还受共享文件系统吞吐影响；生产任务另外保存逐位置 `timings.json`。

持续 CPU 巡检使用 `spectral.absorption.monitor --run-root ...`。接续运行会等新旧队列均退出后汇总，统计数据始终保留逐题配对；未知或缺分片的模型保留缺项，不能计入完成数。停止信号保留在所属 run 内，长期接口和优化实现在本包维护。

## 性能记录与中断处理

每个梯度恢复会话独立写 `performance-<session>.json`，记录完整会话 wall time、sites/min、模型加载、反向、方向读取/传输/投影、梯度 D2H、CPU 累积、组范数、tensor 写盘与组校验提交。反向阶段使用 CUDA 同步边界；投影阶段回传标量、D2H 阶段回传完整梯度，均在返回时完成。未落入这些区段的开销另报 `unattributed_seconds`，总吞吐包含组间落盘成本。恢复后的 sites/min 只计本次会话新增位置。

任务 claim 记录 worker 的 PID/start ticks，子进程启动后补充其身份，并每 10 秒更新心跳。Supervisor 启动时核对遗留 `running`：存在活跃 worker/child 时拒绝接管；两者确认退出后标为 `interrupted` 并保留 attempt，不自动重试；缺少身份的旧任务保留待核验。独立检查使用：

```bash
python -m examples.mirroropd.research spectral.absorption.queue recover --run-root "$RUN"
```

该命令持有 supervisor 锁，活动 supervisor 存在时无法执行。重试通过新计划与新 attempt 显式安排。旧版不完整组缺 SHA256 时需要先核验迁移，不能直接当作新 journal 恢复；原有完整阶段仍作为冻结证据复用。

资源适配器在 guard 启动或 attach 失败时清理本次 guard，并通过 broker 按 owner 释放尚未 attach 的预约。Attach 期间收到 STOP 不开启执行门；已 attach 的进程退出后由 broker 自动恢复占用。清理错误和占用核验均独立留回执，占用未恢复时不会返回成功。

故障回归包含 PID 重用、活跃子进程、外部 STOP、guard 启动失败、attach 失败、组文件等大小损坏、子进程超时回收，以及微型模型实际反向后的组恢复：

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/examples/research/test_absorption_runtime_on_cpu.py \
  tests/examples/research/test_direction_projection_on_cpu.py
```
