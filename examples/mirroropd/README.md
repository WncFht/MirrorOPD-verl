# sampled-token OPD 领域配方

这里维护 math、coding 与 instruction following（IF）的单教师 sampled-token OPD 配方，以及数学、代码、IF 和函数调用 verifier。当前 math 基线使用 DeepSeek-R1-Distill-Qwen-1.5B student 与 JustRL-DeepSeek-1.5B teacher，采用 verl V1 同步训练和 `k1 + policy gradient` 蒸馏。

## 文档入口

| 文档 | 内容 |
| --- | --- |
| [资产准备](docs/资产准备.md) | 模型、数学数据目录、导入与 CPU 检查 |
| [训练配方](docs/训练配方.md) | Math / coding / IF 入口、GPU 资源池、训练与验证预算、Hope 启动 |
| [判分器与依赖](docs/判分器与依赖.md) | 四领域 reward 路由、依赖、迁入来源和检查范围 |
| [MFU 计算](docs/2026-10-20-MFU计算.md) | Qwen2 架构逐项参数量、FLOPs、长度平方项、计时分母与数值复核 |
| [数学基线 b38hw8v0](docs/2026-10-20-sampled-token-OPD数学基线-b38hw8v0.md) | 背景、算法与实际设置、38 步、step 20 与 student / teacher 参考、指标公式与图 |
| [显存与速度优化](docs/2026-10-20-显存与速度优化.md) | 固定验证设置下的生成效率、整卡利用率、共享 8 卡 FSDP teacher 方案 |
| [OPD 上游方案](docs/2026-10-20-OPD上游优化与GPU调度调研.md) | Issue/PR 合并状态、teacher 共卡历史、动态借卡、原生 4+3+1 候选 |
| [流水线组件审计](docs/2026-10-20-OPD流水线组件与阶段审计.md) | 阶段对齐、逐卡使用、36 个验证窗口长尾、CPU 与 profiling 缺口 |
| [分析复现](docs/分析复现.md) | W&B 只读导出、历史日志解析、统计和图表重建 |

## 启动入口

开发机准备好环境和资产后，仍由 Hope 的统一入口启动：

```bash
make assets-check
make opd-submit
```

调用关系为 `make opd-submit` → `hope/mirroropd/run.sh` → `run_${DOMAIN}_deepseek_r1_1_5b_fsdp2.sh` → `_run_single_teacher_fsdp2.sh`，`DOMAIN` 默认 `math`，也可为 `code` 或 `if`。具体领域脚本如下；旧的无领域前缀入口已删除。

- [Math](run_math_deepseek_r1_1_5b_fsdp2.sh)
- [Coding](run_code_deepseek_r1_1_5b_fsdp2.sh)
- [IF](run_if_deepseek_r1_1_5b_fsdp2.sh)

只查看命令时可在仓库根目录执行：

```bash
DRY_RUN=1 bash examples/mirroropd/run_math_deepseek_r1_1_5b_fsdp2.sh
```

`hope/` 由各机器本地维护，不进入源码包。部署、日志传回和默认 W&B 同步流程见[开发机同步说明](../../meituan/README.md)，Hope 设置见 [Hope 文档](../../hope/README.md)。
