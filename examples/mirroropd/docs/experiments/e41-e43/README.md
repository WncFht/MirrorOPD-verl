# Qwen3 小模型的教师对照实验

本组比较 Qwen3-4B 与 Qwen3-30B-A3B-Instruct-2507 对小模型的蒸馏表现。使用已完成 E39 的 Full-KL 实现、batch 64、reuse16、学习率 1e-6、seed 42，每组独占 8 张 H800。按 E41 → E54 → E43 顺序执行。

| 设置 | E41 | E54 | E43 |
| --- | --- | --- | --- |
| Student | Qwen3-1.7B-Base | Qwen3-1.7B-Base | Qwen3-1.7B |
| Teacher | Qwen3-4B | Qwen3-30B-A3B-Instruct-2507 | Qwen3-4B-Instruct-2507 |
| Outer / 参数更新预算 | 100 / 1,600 | 100 / 1,600 | 100 / 1,600 |
| 训练 prompt / response cap | 1,024 / 7,168 | 1,024 / 7,168 | 4,096 / 7,168 |
| 初始验证 | 关闭 | 关闭 | 开启 |
| 验证 / 保存间隔 | 10 / 50 outer | 10 / 50 outer | 10 / 50 outer |
| 本次验证题目数 | 570，原三集 | 630，五集 | 630，五集 |
| 每题采样 / 回答上限 | 16 / 31.7k token | 16 / 31.7k token | 16 / 31.7k token |
| 每卡推理并发 / 显存预算 | 64 / 0.80 | 64 / 0.80 | 64 / 0.80 |
| 终止动作 KL | 合并 151643、151645 | 合并 151643、151645 | 原生终止协议 |
| 当前状态 | 运行中 | 等待 E41；启动前执行八卡 MoE 验收 | 等待 E54 |

三组都使用 non-thinking 提示和相同验证采样：温度 0.7、top-p 0.9、top-k -1。训练响应上限为 7,168，验证响应上限精确值为 31,744。教师使用共享八卡 FSDP2 评分，推理并发 64 指 student 的生成引擎，不是给 teacher 启动 64 路 vLLM。

E41 启动时只有 MATH-500、AMC23、AIME25；此次扩充不改变运行中的题目集合。E54、E43 加入 AIME24、AIME26。以后重新通过公共入口启动 E41，也默认使用五集。跨实验比较应分别报告各数据集；与旧 E39/E41 比较总体分数时，只聚合共同的三集。数据来源、训练题面重合和指标口径见[数学验证协议](../../数学验证协议.md)。

E54 与 E41 保持同一 Base student、损失和优化预算。30B-A3B 教师是 MoE，虽然每 token 激活约 3.3B 参数，仍须存放全部约 30.5B 参数；八卡 FSDP2 分片加载。代码只扩展 Full-KL 输出适配器和冻结输出头对 `qwen3_moe` 的支持，保留原有损失计算。CPU 微型模型已核对 logits、损失和梯度；八卡验收将在 E41 释放资源后执行，实际大模型显存和完整训练稳定性仍以 E54 运行为准。

E43 同时改变 student 和 teacher，不能用 E41/E43 的差异单独归因于教师。三个运行都使用独立源码副本：E41 保持原快照；E54/E43 在同一 E39 运行时基础上加入评测路由、MoE 支持及启动预设，不受共享工作区后续编辑影响。

W&B 在线同步，保留训练指标、完整验证回答、GPU 采样及 vLLM 的 KV 使用率、运行/等待请求数、抢占计数。遥测异常不会中断训练，也不会把缺失值补成零。

| 实验 | W&B |
| --- | --- |
| E41 | [53dd553d](https://wandb.ai/wncfht/MirrorOPD-verl/runs/53dd553d) |
| E54 | [885f73ac](https://wandb.ai/wncfht/MirrorOPD-verl/runs/885f73ac)，启动后产生记录 |
| E43 | [eccc09a1](https://wandb.ai/wncfht/MirrorOPD-verl/runs/eccc09a1)，启动后产生记录 |

单组入口：

```bash
bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E41
bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E54
bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E43
```

本机已配置顺序执行，不要重复运行上述命令。每组退出后先清理本组进程，再启动下一组；训练失败保留失败状态后继续。全部结束、异常退出或监督程序意外终止时，独立守护程序调用 `gpu_occupy.sh` 恢复占用物理 GPU 4–7，并核实实际就绪。模型和数据未准备好时不会启动后续实验。
