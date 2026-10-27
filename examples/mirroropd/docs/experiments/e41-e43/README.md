# Qwen3-4B 教师的 E41 / E43 后续实验

两组实验使用与已完成 E39 相同的 Full-KL 实现、batch 64、reuse16、LR1e-6、seed42、8 张 H800 共享同步执行。预算根据 E39 后期平台现象缩短为 100 outer；只对 E41/E43 更新预设，不改写历史结果。

| 设置 | E41 | E43 |
| --- | --- | --- |
| Student | Qwen3-1.7B-Base | Qwen3-1.7B |
| Teacher | Qwen3-4B | Qwen3-4B-Instruct-2507 |
| Outer / 参数更新预算 | 100 / 1600 | 100 / 1600 |
| 训练 prompt / response cap | 1024 / 7168 | 4096 / 7168 |
| 初始验证 | 关闭 | 开启 |
| 验证频率 | 每 10 outer | 每 10 outer |
| 验证规模 | 570 题×16，回答上限 31744 | 570 题×16，回答上限 31744 |
| 验证采样 | 温度 0.7、top-p 0.9、top-k -1 | 温度 0.7、top-p 0.9、top-k -1 |
| 推理资源 | 每卡并发 64，显存 0.80，TP1，Graph | 每卡并发 64，显存 0.80，TP1，Graph |
| 终止动作 KL | 合并 151643 与 151645 | 原生终止协议，未启用终止动作合并 |
| 保存 | outer 50、100 | outer 50、100 |
| 当前状态 | 运行中，首批更新已通过验收 | E41 结束后自动启动 |

两组均使用 non-thinking 提示。E41 和 E43 的 student 与 teacher 均不同，这不是只改变一个模型大小的严格控制变量对照。E43 的 prompt 预算沿用该模型原预设；两者的实际数学验证题目集合和采样预算相同。

运行时固定已完成 E39 的源码快照（a73dcbc0f），叠加此次 Python 启动预设调整。W&B 在线同步；逐批训练指标、完整验证回答、GPU 采样和 vLLM 指标独立保存。vLLM 统计开启，并记录每个引擎的 KV 使用率、running/waiting 请求数与抢占计数；这些观测为尽力采集，采集异常不会中断训练，缺失数据不会补为零。

执行顺序为 E41→E43，每组使用全部 8 卡。第一组退出后先清理本组 Ray 进程，再启动第二组；即使某组训练以非零码退出，也保留失败状态并继续下一组。整个任务正常结束、异常退出或监督进程意外终止后，独立守护程序清理本任务残留进程并调用 gpu_occupy.sh 占用 GPU 4–7，验证就绪后才标记恢复完成。

通用入口：

```bash
bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E41
bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E43
```

资源、模型目录、Python 环境与日志仍按公共启动约定设置；显式环境变量和末尾 Hydra 参数可覆盖默认值。以上是实验设置与执行约定，不代表结果已经产生。

E41 在线运行：[W&B 页面](https://wandb.ai/wncfht/MirrorOPD-verl/runs/53dd553d)。E43 的运行标识已预分配为`eccc09a1`，实际启动后才产生云端数据。本次自动衔接与占用恢复由独立后台监督程序执行，公共入口负责单组训练。
