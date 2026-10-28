# E48 完整日志与 KL 方向研究，2026-10-27

正文：源码仓库 `examples/mirroropd/docs/experiments/teacher-transfer/cross-origin/polaris/2026-10-27-E48后期平台与Forward-KL及知识迁移.md`。

## 证据边界

`run/`由本地 W&B 通用解码器经项目 export 适配生成：4353 条 history、4096 次更新、256 个完整 outer，exit=0；8018 条周期性系统记录。`cloud-audit.json`只读核对云端 finished 和相同边界。`E20/`是同预算同源参考的冻结 history/config/run metadata。旧 58-outer 报告和数据没有覆盖。

短实验使用 `../state-transfer-20261027/` 的 24 题、72 条冻结轨迹和 3376 个 state。每组仅四道训练数学题、128 个前段评分位置，其余固定前缀用作对照。两 fold 训练题互不重叠，held-out 数学 8 题、代码 4 题、IF4 题。训练前向保留完整冻结参考输入，只在前段选择 loss，前后评估也保持输入不变。

初始 student：两 teacher×两 KL 方向×两 fold，共 8 组；本机 Polaris/student 蒸馏 64 步 checkpoint：两 KL 方向×两 fold，共 4 组。共 12 组、192 次完整全参数更新。Warm64 checkpoint 不是原 E48 后期 checkpoint，其训练 prompt 与本轮 24 题没有标准化文本交集；只用于受控的方向切换实验。

所有组使用相同 fresh GPU AdamW、LR=1e-6、clip=1、16 步。记录第 1/8/16 步实际参数位移。全部参数更新；FP32 master、BF16 backbone、FP32 head。没有改写原模型目录或现有 checkpoint，没有启动正式长期训练。

Teacher 明确在 student 输出行上条件化并重新归一化。因为 Polaris 比 student 多 128 个输出行，严格 native FKL 对零扩展 student 为无穷；这里不隐瞒支持集变换。静态数学 student-state 上额外 teacher 质量最大约 1.35e-8。

## 文件

| 文件 | 内容 |
| --- | --- |
| `run-statistics.json` | 完整 E48 的窗口、位置、更新健康、任务与生成结果 |
| `gradient-diagnostics.json` | 每 state 的 RKL/FKL/JS/TV、精确 logit 梯度、teacher 质量覆盖与罕见动作指标 |
| `gradient-summary.json`、`gradient-tail-summary.json` | 分组均值、梯度尾部和拼接梯度 cosine |
| `gradient-check.json` | 解析 logit 梯度与 FP64 autograd 比较 |
| `initial-*-fold*/` | 两方向、两 teacher、两 fold 的 16 步训练及固定 state 评估 |
| `warm64-*-fold*/` | 同一已有 checkpoint 的两方向对照 |
| `direction-summary.json` | 每组在相同输入和相同指标上的相对变化 |
| `lineage-state-metrics.json`、`lineage-summary.json` | DS-distill-7B 基座评分及 logit 梯度分解；不是 KL 或能力占比 |
| `verification.json` | 连续更新、相同 step-0、固定 state、有限数值、非零位移和支持集语义核验 |
| `papers/` | 直接取得并核对的论文 LaTeX 原文 |
| `paper-search.md` | 聚合检索工具的原样完整报告，不修改结果排序或删除噪声命中 |
| `search-diagnostics.log` | 聚合检索的原样 stdout/stderr，包含 API 429 细节 |

聚合检索的 arXiv API 被限流，Crossref 关键词结果主要不相关，因此这些命中没有作为正文论据。随后使用网页检索定位并直接获取 arXiv 原文，核对模型、目标、设置与结果。正文引用的是这些一手来源。API 错误原文见上述完整文件，包含：

```text
[id-lookup] arxiv resolution failed: 429 Client Error: Unknown Error for url: https://export.arxiv.org/api/query?id_list=2610.09639%2C2609.28145%2C2610.05974%2C2609.35259%2C2609.38666&max_results=5
```

## 复算

从源码仓库根目录执行。GPU 只用于模型短训练；统计脚本可以直接读冻结结果。新实验使用新 OUT，不覆盖已有证据。Warm64 复算还需要下列本机 checkpoint，报告阅读无需传输该模型。

```bash
DIRECTION_PY=/root/.local/share/mirroropd/venvs/profile-cu128/bin/python
DIRECTION_OUT=examples/mirroropd/docs/experiments/teacher-transfer/cross-origin/polaris/data/forward-kl-study-20261027
STATE_DATA=examples/mirroropd/docs/experiments/teacher-transfer/transfer-mechanism/data/state-transfer-20261027
STATE_CACHE=outputs/polaris-transfer-cache/state-transfer-20261027

CUDA_VISIBLE_DEVICES='' "$DIRECTION_PY" -m examples.mirroropd.research.transfer.kl_direction_probe diagnostic \
  --models ../model --data "$STATE_DATA" --cache "$STATE_CACHE" --out "$DIRECTION_OUT"

# 示例：同一个初始student，Polaris监督，FKL，训练fold0。
# teacher=justrl/polaris；direction=reverse/forward；fold=0/1。
CUDA_VISIBLE_DEVICES=0 "$DIRECTION_PY" -m examples.mirroropd.research.transfer.kl_direction_probe train \
  --models ../model --data "$STATE_DATA" --cache "$STATE_CACHE" --out "$DIRECTION_OUT" \
  --teacher polaris --direction forward --fold 0 --steps 16

# Warm64另加：
# --tag warm64 --checkpoint outputs/polaris-transfer-cache/intersection-study-20261027/checkpoints/polaris-student-k16

python3 -m examples.mirroropd.research.transfer.analyze_forward_study --data "$DIRECTION_OUT"
CUDA_VISIBLE_DEVICES='' "$DIRECTION_PY" -m examples.mirroropd.research.transfer.audit_forward_study --data "$DIRECTION_OUT"
CUDA_VISIBLE_DEVICES=0 "$DIRECTION_PY" -m examples.mirroropd.research.transfer.probe_teacher_lineage \
  --models ../model --data "$STATE_DATA" --cache "$STATE_CACHE" --out "$DIRECTION_OUT"
FIGURE_VIZ_MPL=../dotfiles/.agents/skills/figure-viz/assets/mpl \
  "$DIRECTION_PY" -m examples.mirroropd.research.transfer.plot_forward_study \
  --data "$DIRECTION_OUT" --out examples/mirroropd/docs/experiments/teacher-transfer/cross-origin/polaris/figures/forward-kl-study-20261027
```

短训练脚本检查至少 48GiB 物理余量，allocator 限 42GiB；实际峰值约 33.6GiB。共享 GPU 余量不足时会等待，不中止其他任务。基座 probe 另用 24GiB allocator 上限，只做前向与 CPU 统计。这里是固定 state 的短实验入口，不能仅增加 steps 就当作完整 online OPD。生产 FKL 还需要显式接入目标、teacher 支持归一化、位置及双向指标。
