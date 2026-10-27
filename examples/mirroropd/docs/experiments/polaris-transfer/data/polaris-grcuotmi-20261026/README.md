# E48 grcuotmi 冻结分析证据

日期：2026-10-26。正文位于源码仓库 `examples/mirroropd/docs/experiments/polaris-transfer/2026-10-26-E48异源教师收敛诊断.md`。

`run/` 是显式本地 export 的 E48 原始快照：987 个 history 事件，928 次更新，58 个完整 outer，验证 0/32，1816 个系统记录；尾部 partial_snapshot。`cloud-audit.json` 是只读 API 身份、状态和边界核对，未上传或修改云端；state=failed 不提供停止原因。日志仓库基准提交为 `b7ba279bef7943ef0d9145f0aea3cff7354161d0`，E48 记录的代码提交为 `71a969c207213ea9013ad57b0dc9f0bd94c2fe0f`。

`E20/`、`E17/`、`E18/`、`E19/`、`E35/` 保存各对照的本地 history/config/metric definitions/run metadata。已核对这些解码缓存的原始文件大小及 mtime 与现存原始日志相同。比较只使用正文注明的预算边界；快照包含更晚历史不代表全部用于同预算对照。

| 产物 | 内容 |
| --- | --- |
| `statistics.json` | 数据边界、批首/末及窗口统计、验证序列；窗口均值不能当作跨 seed 方差 |
| `*-metrics.csv` | 各有限数值指标的次数、首末、范围、均值与覆盖；缺失不补零 |
| `config-differences.json` | E48 与各对照的逐字段配置差异 |
| `system-summary.json` | E48 全程周期性 GPU 观测，各设备均值和最大值；含验证和切换 |
| `probe/` | 六条历史初始 student 回答、304 个选定前缀、三个初始模型的 CPU BF16 backbone / FP32 head 比较、温度反事实 |
| `terminal-probe/` | 上述四条短/中完整回答后的 EOS 预测；不含训练后 checkpoint |
| `kernel-checks.json` | run 提交的五种 Full-KL backend 与 CPU dense reference 比较，以及明确标为示意的两状态梯度例子 |
| `kernel-pytest.log` | 当前工作区 Full-KL 与位置归约的 46 项检查结果 |
| `verification.json` | 独立数值与横轴核验 |

Probe 的输入保存在 cases.json；长回答最多前向至 response 7168 token。位置是按预先确定的若干绝对位置块抽取，不是总体随机 token 样本，不外推为训练集 KL、模型能力或显著性。温度反事实仅改变 teacher 概率，不执行参数优化。原始 logits 可由配套脚本重建，缓存在 `outputs/polaris-transfer-cache/polaris-probe-20261026/`，不属于归档证据。

复算脚本与命令见正文。脚本对存在的冻结训练快照不重新拉取云端；需要覆盖新边界时选新目录。原始 `.wandb` 保持在 `MirrorOPD-verl/wandb/`。没有新增训练任务、正式 checkpoint 或开发机文件传输。
