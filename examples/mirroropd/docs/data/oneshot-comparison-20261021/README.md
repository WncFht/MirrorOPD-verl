# One-Shot历史运行只读快照

结果与统计边界见[八卡历史运行对照](../../archive/2026-10-23-历史八卡与One-Shot运行对照.md#47-one-shot历史训练与当前路径)。数据抓取日期为2026-10-21，各run的精确UTC时间在`source.json`；只读API导出，没有登录切换、上传或启动训练。

`catalog.json`保留21个run的名称/状态；`selection.json`记录选择理由。`mdxt04lm`、`mzwm9q04`、`5aub1k4h`分别为full-data直接优势、full-data折扣优势、one-shot直接优势。重新读取的本仓`rcxtujbi`仍只有12步，无完整验证。

每个run目录保存脱敏配置、硬件、主要包版本、来源、完整训练history的gzip JSONL及系统sampled stream。系统API即使请求100000个样本也不保证全量；缺测值不补零。未保存完整控制台文本、主机身份或GPU UUID。

`comparison.json`同时保留step2–12和所有非首步/非保存步统计。旧One-Shot的teacher timer嵌套在adv中，不重复相加；短teacher/actor阶段不发布稀疏GPU采样的精确利用率。`validation-queue-analysis.json`复用[本机验证采样](../validation-optimization-20261021/README.md)，比例按有效采样轮数统计，不当作连续时间积分。

从仓库根目录复算，不需要GPU或网络：

```bash
python3 examples/mirroropd/docs/scripts/compare_oneshot_runs.py
python3 examples/mirroropd/docs/scripts/analyze_validation_queue.py
```

脚本会覆盖本目录对应派生JSON，不修改原始API快照。原始未压缩导出与探索过程仍保留在`outputs/oneshot-comparison-20261021/`。外部run链接只在实测报告维护。
