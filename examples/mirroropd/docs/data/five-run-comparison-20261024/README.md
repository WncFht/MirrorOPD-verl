# 五组 Math 实验的冻结分析数据

报告：[五组 Math 实验对比分析](../../research/2026-10-24-五组Math实验对比分析.md)。本地生成的[离线 HTML](dashboard.html)包含全部指标和绘图库，可单文件下载；该生成文件不进入源码提交，跨机需单独复制或按下列命令重建。

| 标签 | Run | 本地解码状态 | 完整 outer | 成功更新 | 最后验证 outer |
| --- | --- | --- | ---: | ---: | ---: |
| E13 | jt2r9h88 | finished / exit 0 | 200 | 200 | 200 |
| E14 | aq474oay | finished / exit 0 | 200 | 200 | 200 |
| E15 | krkmq098 | finished / exit 0 | 200 | 200 | 200 |
| E16 | ujy2qx5d | finished / exit 0 | 200 | 800 | 200 |
| E17 | cix5i2x2 | partial_snapshot / 无 exit | 177 | 2848 | 160 |

原始日志来自同级 `wandb-logs` 仓库的 `35b1610`，解码器版本 2，W&B SDK 0.28.1。项目导出器没有请求云端 history；`source.json` 保存每个 run 的本地路径、解码状态、记录数、时间和 metadata 来源。Raw `.wandb` 留在日志仓库，不复制进源码仓库。

每个 run 子目录保存已脱敏 config、summary、硬件、软件版本、来源，以及无损压缩的 history、metric definitions 和完整解码的周期性 system 记录。`.wandb` 的 environment 记录用于恢复缺少附件的硬件信息，多个 writer 不混合。E17 的不完整尾部留在原始日志；这里只包含成功解码的完整记录。初次缺少环境的导出保留于 `outputs/five-run-comparison-20261024/raw`，补全环境后的新导出为 `raw-v2`，两者 history 字节核对一致。

## 派生文件与口径

- `analysis.json`：配置、覆盖、专题使用的验证字段、训练工作量、计时、梯度裁剪、批内 KL 与资源统计；其余验证字段完整保留在曲线与指标目录中。
- `comparison.json.gz`：全部数值指标的原始轴曲线、坐标映射和分析结果，供绘图及 HTML 使用。
- `metric-catalog.csv`：每个 run/metric 的轴、观测数、首尾、极值、均值、中位数与常量标记。`first20_mean/last20_mean` 指该指标前/后最多 20 个观测，不是统一的 20 个 outer step；验证不足 20 点时是全部观测。
- `metric-coverage.csv`：2378 个指标名称的跨 run 覆盖，缺失写为 `missing`。
- `config-differences.json`：全部 13 个不同配置字段，保留实验路径等非算法差异。
- `provenance.json`：来源提交、原始/导出记录关系与源码版本可追溯性检查。

横轴从原始 metric definitions 读取，不用当前代码猜测。`normalize` 拒绝同一指标同一轴上的冲突观测。Epoch 内的多条更新不聚合成一个 outer loss；E17 第 178 批的更新保留，但不伪造第 178 步的 outer timer、评分或验证。

费用账只加 outer 与 validation，checkpoint、rollout_batch、validation_rollout 保持嵌套关系。普通步统计去掉首步、验证步和 checkpoint 步。资源阶段边界根据 host 计时近似重建，分别保留不裁边和裁边 10 秒的结果；GPU 样本约 15 秒一次，不代表连续 trace。

## 复现

统计、绘图、离线 HTML 与测试的唯一命令入口见[分析复现](../../分析复现.md#五组-math-实验比较)。统计不需模型/GPU/凭据；绘图和 HTML 使用已有分析依赖。

更新原始数据时使用新的快照目录，不覆盖本目录：先按[分析复现](../../分析复现.md)用 `analyze_run.py export --run ... --out ...` 导出各 run，再向比较器提供 `--snapshot-from /新导出父目录` 与新的 `--data`。无需复制或另写一套 W&B 二进制解码器。

图表审计已运行且无 error。不同数据集采用独立 y 轴以看清各自变化，误差条端帽/同色双柱会被通用系列计数器重复计数；对应 warning 保留在每张图的 audit 文件中，并经人工看图确认。图中的 CI 是题目 bootstrap，平滑曲线是训练的 20-step trailing mean，均不是跨训练 seed 的误差带。
