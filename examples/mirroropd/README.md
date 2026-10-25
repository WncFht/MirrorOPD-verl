# sampled-token OPD 领域配方

这里维护 math、coding 与 instruction following（IF）的单教师 sampled-token OPD 配方，以及数学、代码、IF 和函数调用 verifier。当前 math 基线使用 DeepSeek-R1-Distill-Qwen-1.5B student 与 JustRL-DeepSeek-1.5B teacher，采用 verl V1 同步训练和 `k1 + policy gradient` 蒸馏。

## 脚本目录

顶层保留领域训练、数学消融、独立验证和 full-vocab 入口。辅助脚本按用途组织：

| 目录 | 内容 |
| --- | --- |
| `assets/` | `check_assets.py`、`prepare_models.py`、`prepare_math_data.py` |
| `launchers/` | 公共 `_run_single_teacher_fsdp2.sh`，由顶层入口调用 |
| `benchmarks/` | Decode、并发、KV 容量和限时吞吐测量 |
| `smoke/` | Sampled、full-vocab 和指标链路的短测入口 |
| `tools/` | Full-vocab 作业队列、更新预算和指标验收工具 |
| `rewards/` | `reward.py` 自定义判分路由 |
| `docs/` | 文档、冻结实验记录和对应研究探针 |

辅助命令使用新路径，例如 `python examples/mirroropd/assets/prepare_models.py --help`、`bash examples/mirroropd/smoke/smoke_full_reverse_kl.sh`。Makefile、当前启动脚本和测试已同步迁移；历史实验记录保留运行时的路径，重跑历史版本时使用记录中的源码版本。

### 完整验证负载的 vLLM sweep

[`benchmarks/vllm_sweep/`](benchmarks/vllm_sweep/) 管理八个原始模型的推理参数扫描：`prepare_data.py` 复用现有 prompt builder 导出每模型570题×16请求，核对token回编码；`plan.json` 保存基线及候选；`run_sweep.py` 调用原生 `vllm bench sweep serve`，按四卡DP4/TP1排队执行。`worker.py` 保留当前verl对词表外token的屏蔽，原始模型文件不变。正常EOS、温度0.7、top-p0.9和31744回答上限用于全部正式测量；必须显式传 `--custom-output-len -1`，否则原生CLI的256默认值会覆盖JSONL预算。

```bash
PYTHONPATH=. "$PYTHON_BIN" examples/mirroropd/benchmarks/vllm_sweep/run_sweep.py run \
  --model-root "$MODEL_ROOT" --data-root "$DATA_ROOT" \
  --output outputs/vllm-sweep/20261025-full
```

使用 `preflight --models qwen17` 做4请求、64输出token的启动验收，该结果不会参与正式排名；`dry-run` 只打印原生命令。正式执行先覆盖八模型的基线和候选1；耗时或TPS改善超过3%时继续候选2，再测较优配置的16384 token预算、重复测量和原verl完整valid。原生benchmark的整批生成耗时用于排名（不含模型加载、编译和判分），原verl复核另行保存验证指标；同时保留输出token数及TPS；自然生成长度变化不等同于纯调度收益。

同一 `--output` 目录可恢复队列：已完成且样本完整的作业跳过，失败结果不会被选为推荐值。输出包含 `resolved_plan.json`、逐模型 `data/`、`runs/` 下的原生结果与日志、`state.json`、`summary.csv` 和 `selected_params.json`；推荐值在全部原valid复核完成前标为 provisional。模型支持较大并发的结论以实测为准，运行入口不自动改写训练参数。执行仅占用GPU0–3；如发现其他进程占用则停止，不终止外部进程。

## 文档入口

MirrorOPD 自有文档、调研与展示图位于 `examples/mirroropd/docs/`；正式实验数据与报告快照存放在独立 `wandb-logs` 仓库。仓库根 `docs/` 保留上游文档。指标定义统一维护在 `docs/experiments/metrics/`，实现走读位于 `docs/internals/metrics.md`。

| 文档 | 内容 |
| --- | --- |
| [架构总览](docs/architecture.md) | 当前模块职责、跨模块执行顺序和扩展边界 |
| [代码走读](docs/internals/README.md) | 训练、教师、目标、指标和 checkpoint 的当前实现；`make docs-check` 检查引用 |
| [资产准备](docs/资产准备.md) | 模型、数学数据目录、导入与 CPU 检查 |
| [训练配方](docs/训练配方.md) | Math / coding / IF 入口、GPU 资源池、训练与验证预算、Hope 启动 |
| [OPD 失败复现方案](docs/failure-reproduction/README.md) | DAPO 与三 valid 下的模型／长度设置、sampled→full 对照、原论文图表与排除设置附录 |
| [判分器与依赖](docs/判分器与依赖.md) | 四领域 reward 路由、依赖、迁入来源和检查范围 |
| [LCB 判分隔离调研](docs/lcb/README.md) | 内存故障来源、受控 CPU 实验、轻量执行器与资源隔离方案 |
| [训练基础设施](docs/infra/README.md) | 运行与调度、实测报告、batch/rollout 调优、指标/MFU、模型扩展和上游调研的统一入口 |
| [OPD 目标与消融](docs/opd-objective/README.md) | 固定 prefix 推导、优势/IS 语义及更新计数的待实施方案 |
| [分析复现](docs/experiments/复现与维护.md) | 本地 W&B 归档解码、项目快照、统计与图表；云端回读仅作显式核对或补救 |
| [实验指标](docs/experiments/metrics/README.md) | 当前字段字典、更新横轴与比较口径；实现见指标系统走读 |

## 启动入口

开发机准备好环境和资产后，仍由 Hope 的统一入口启动：

```bash
make assets-check
make opd-submit
```

调用关系为 `make opd-submit` → Hope `worker.script` → `run_${DOMAIN}_deepseek_r1_1_5b_fsdp2.sh` → `launchers/_run_single_teacher_fsdp2.sh`，`DOMAIN` 默认 `math`，也可为 `code` 或 `if`。具体领域脚本如下；旧的无领域前缀入口已删除。

- [Math](run_math_deepseek_r1_1_5b_fsdp2.sh)
- [Coding](run_code_deepseek_r1_1_5b_fsdp2.sh)
- [IF](run_if_deepseek_r1_1_5b_fsdp2.sh)

只查看命令时可在仓库根目录执行：

```bash
DRY_RUN=1 bash examples/mirroropd/run_math_deepseek_r1_1_5b_fsdp2.sh
```

数学消融统一使用 [run_math_opd_ablation_fsdp2.sh](run_math_opd_ablation_fsdp2.sh)。短窗基础版为 E15（sampled）／E13（full），其他模型／长度基础版为 E23–E32；对应reuse16版本为 E33–E43及已有E17。E21/E22分别兼容映射到E15/E13。当前模型对照均跑250个outer steps，每100步及最终步保存；基础版每批更新1次，reuse版16次。`E15-valid-student`／`E15-valid-teacher` 可独立验证原始模型。每次调用运行一个选项，GPU数量由 `STUDENT_GPUS` 与 `CUDA_VISIBLE_DEVICES` 明确指定；完整编号、模型和预算见[实验方案](docs/experiments/failure-reproduction/2026-10-25-Sampled-token与Full-vocab失败复现方案.md#failure-launcher)。

```bash
# 只检查 E28 的命令，不启动 GPU 作业。
CUDA_VISIBLE_DEVICES=0,1,2,3 STUDENT_GPUS=4 DRY_RUN=1 \
  bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E28
```

`hope/` 由各机器本地维护，不进入源码包。Worker 直接调用随源码更新的领域脚本，入口迁移见[决策记录](docs/infra/2026-10-22-共享八卡运行与架构.md#hope-直连入口)。部署、日志传回和默认 W&B 同步流程见[开发机同步说明](../../meituan/README.md)。
