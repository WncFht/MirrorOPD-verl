# sampled-token OPD 领域配方

这里维护 math、coding 与 instruction following（IF）的单教师 sampled-token OPD 配方，以及数学、代码、IF 和函数调用 verifier。当前 math 基线使用 DeepSeek-R1-Distill-Qwen-1.5B student 与 JustRL-DeepSeek-1.5B teacher，采用 verl V1 同步训练和 `k1 + policy gradient` 蒸馏。

## 文档入口

MirrorOPD 自有文档、调研和配套图片、数据统一放在 `examples/mirroropd/docs/`；仓库根 `docs/` 保留上游文档。训练指标集中维护在 `docs/metrics/`。

| 文档 | 内容 |
| --- | --- |
| [资产准备](docs/资产准备.md) | 模型、数学数据目录、导入与 CPU 检查 |
| [训练配方](docs/训练配方.md) | Math / coding / IF 入口、GPU 资源池、训练与验证预算、Hope 启动 |
| [判分器与依赖](docs/判分器与依赖.md) | 四领域 reward 路由、依赖、迁入来源和检查范围 |
| [LCB 判分隔离调研](docs/lcb/README.md) | 内存故障来源、受控 CPU 实验、轻量执行器与资源隔离方案 |
| [训练基础设施](docs/infra/README.md) | 运行与调度、实测报告、batch/rollout 调优、指标/MFU、模型扩展和上游调研的统一入口 |
| [OPD 目标与消融](docs/opd-objective/README.md) | 固定 prefix 推导、优势/IS 语义及更新计数的待实施方案 |
| [分析复现](docs/分析复现.md) | W&B 只读导出、历史日志解析、统计和图表重建 |
| [训练指标](docs/metrics/README.md) | 当前字段字典、统计归约、更新横轴、调研及测试边界 |

## 启动入口

开发机准备好环境和资产后，仍由 Hope 的统一入口启动：

```bash
make assets-check
make opd-submit
```

调用关系为 `make opd-submit` → Hope `worker.script` → `run_${DOMAIN}_deepseek_r1_1_5b_fsdp2.sh` → `_run_single_teacher_fsdp2.sh`，`DOMAIN` 默认 `math`，也可为 `code` 或 `if`。具体领域脚本如下；旧的无领域前缀入口已删除。

- [Math](run_math_deepseek_r1_1_5b_fsdp2.sh)
- [Coding](run_code_deepseek_r1_1_5b_fsdp2.sh)
- [IF](run_if_deepseek_r1_1_5b_fsdp2.sh)

只查看命令时可在仓库根目录执行：

```bash
DRY_RUN=1 bash examples/mirroropd/run_math_deepseek_r1_1_5b_fsdp2.sh
```

`hope/` 由各机器本地维护，不进入源码包。Worker 直接调用随源码更新的领域脚本，入口迁移见[决策记录](docs/infra/2026-10-21-共享八卡运行与架构.md#hope-直连入口)。部署、日志传回和默认 W&B 同步流程见[开发机同步说明](../../meituan/README.md)，Hope 设置见 [Hope 文档](../../hope/README.md)。
