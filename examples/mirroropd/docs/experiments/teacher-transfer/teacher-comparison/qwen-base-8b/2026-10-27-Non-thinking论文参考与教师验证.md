# Non-thinking 论文参考与 Qwen3-8B 完整验证

2026-10-27 核对当前运行配置与真实 tokenizer 渲染。`data.apply_chat_template_kwargs.enable_thinking=False`，Qwen3-1.7B-Base 与 Qwen3-8B 的实际提示均以空的 `<think>\n\n</think>\n\n` 块结束，属于 non-thinking 提示协议。Base checkpoint 本身没有经过 Qwen3 后训练的双模式能力训练；这里的 non-thinking 指提示设置。Teacher 在蒸馏时对相同 student token 前缀评分，而独立 teacher 验证是让 Qwen3-8B 自己生成。Non-thinking 仍允许回答中出现数学推导，不等于短回答或没有推理。

与本实验最接近的是 TIDE 论文：模型对正是 Qwen3-1.7B-Base student 与 Qwen3-8B teacher，附录 D.4 明确关闭 thinking。原文表 1 报告 teacher 的 MATH-500、AMC23、AIME25 平均正确率分别为 83.3%、68.8%、23.3%。其评测为每题 8 次、温度 0.7、top-p 0.95、最大回答 31,744 token；我们的完整验证为每题 16 次、top-p 0.9，因此这些分数只作为论文参考。[^tide]

Qwen3 官方技术报告表 18 单独评测 Qwen3-8B non-thinking：MATH-500 为 87.4%，AIME25 为 20.9%。报告使用的 non-thinking 解码设置包括温度 0.7、top-p 0.8、top-k 20、presence penalty 1.5，并为 AIME 使用 38,912 token 上限；不能拿这些结果替代当前验证协议的实测。[^qwen3]

Semi-OPD 论文表 3 的 pair (e) 也测过 Qwen3-1.7B ← Qwen3-8B non-thinking，但 student 是后训练的 Qwen3-1.7B，并非 Base。该文主要使用 AIME24/25/26 与 HMMT Feb26、avg@32、温度与 top-p 均为 1.0，适合研究蒸馏动态，不宜直接比较我们的分数。[^semi]

完整检索输出原样保存于[论文检索报告](../../../../../../../../wandb-logs/MirrorOPD-verl/analysis/e39-qwen3base-validation-20261027-step140/2026-10-27-paper-search-original.md)，含三个条目与来源统计；三个 arXiv 标识均成功解析。

完整 teacher 验证入口是 [run_teacher_validation.sh](../../../../../research/analysis/qwen_base/run_teacher_validation.sh)。从仓库根目录运行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash examples/mirroropd/research/analysis/qwen_base/run_teacher_validation.sh
```

默认 4 张可见 GPU，脚本按 `CUDA_VISIBLE_DEVICES` 自动计算卡数；使用 8 卡时改为 `0,1,2,3,4,5,6,7`。资源起点是每卡并发 16、显存比例 0.8、TP1、Graph。模型固定 Qwen3-8B；完整数据为 MATH-500 500 题、AMC23 40 题、AIME25 30 题，每题 16 次，共 9,120 条。温度 0.7、top-p 0.9、top-k -1、最大回答 31,744 token，thinking 关闭。只验证，不更新参数。并发和显存比例是资源设置，不是论文成绩复现协议。

脚本优先使用仓库 `.venv-native/bin/python`，其次使用本机已经验证的 `/root/.local/share/mirroropd/venvs/profile-cu128/bin/python`；其他机器可通过 `PYTHON_BIN` 指向已有 verl 环境。模型、数据路径分别支持 `MODEL_ROOT`、`DATA_ROOT`。已完成 Bash 语法和 CONFIG_ONLY 解析检查，确认 teacher 模型、val-only、non-thinking、采样参数、数据文件与 GPU 数均正确；没有在本机增加 GPU 验证任务。

W&B 默认在线上传到 `wncfht/MirrorOPD-verl`，使用工作区 `.secrets/netrc`；设置 `WANDB_MODE=offline` 可保留离线记录。运行目录会在终端打印，集中包含 `train.log`、`resolved_config.yaml`、`metrics.jsonl` 和 `validation/records/0.jsonl`。测完保留这些文件及 W&B 链接，即可分析整体与分数据集的准确率、长度、截断、样本分布和 teacher–student 差距。

### 参考文献

[^tide]: Zichao Yu 等。Mismatch Matters: On-Policy Distillation Beyond Token Agreement. arXiv:2608.09836v1, 2026. [表 1 与附录 D.4](https://arxiv.org/html/2608.09836v1).
[^qwen3]: Qwen Team. Qwen3 Technical Report. arXiv:2505.09388v1, 2025. [第 4.3 节与表 18](https://arxiv.org/html/2505.09388v1).
[^semi]: When Do We Need On-Policy Distillation? Distilling on Offline Student Rollouts Is Often Better. arXiv:2610.11291, 2026. [表 3 与附录 A](https://arxiv.org/html/2610.11291).
