# 完整权重诊断与随机对照

本页说明已有 `spectral weights` 入口。跨 teacher、跨训练阶段的纯谱统计优先使用 [scan](scan.md)；需要 diag/off/null 吸收与随机对照时使用本入口。两者复用数学和 checkpoint 组件，结果协议独立。

```bash
python -m examples.mirroropd.research.studies.spectral weights \
  --base "$BASE" --teacher "$TEACHER" --opd "$OPD" \
  --output runs/spectral-analysis/weights
```

`weights` 逐矩阵读取，不加载完整模型。默认仅 attention/MLP 二维权重；`--scope` 为正则，可先限制一层。默认 FP64 CPU 算术，可选 `--device cuda:0 --arithmetic float32`。先提升存储值再相减，来源 dtype 和选中 tensor 内容 SHA256 写入结果；BF16 导出提升精度不会恢复 FP32 master。

输出含 teacher、OPD 和 missing 更新的全谱秩指标、ISO 标量、base frame 的 diag/off/null 三方能量与吸收，以及同形等范数 Gaussian、真实谱随机框架基线。每矩阵独立 JSON，`protocol.json` 记录模型身份和范围。base SVD 在该矩阵的比较中复用。全谱计算成本高，先检查一层的重构误差和运行时间，再扩展范围。

## 复用、恢复与执行边界

不同 OPD checkpoint 使用同一组 base/teacher 时，指定共享 `--cache` 保存 teacher 与随机对照的标量统计；命中后省去对应的全谱分解与随机框架生成。键包含原始 tensor 内容 SHA256、tensor 名、算术精度、设备、求解器、随机种子和 PyTorch 版本，独立于 OPD checkpoint。大矩阵与完整奇异框架保持瞬时驻留，避免将缓存扩展成大量方向文件。base frame 仍需每个 checkpoint 重算；缓存收益与矩阵尺寸有关。

```bash
python -m examples.mirroropd.research.studies.spectral weights \
  --base "$BASE" --teacher "$TEACHER" --opd "$OPD" \
  --device cuda:0 --driver gesvd --cache runs/spectral-analysis/references \
  --output runs/spectral-analysis/weights
```

`--driver gesvd` 与 `--driver gesvdj` 选择 CUDA 精确求解器；默认沿用 PyTorch。CPU 保持默认，不传 `--driver`。求解器应先在目标卡和真实矩阵上测量耗时与重构误差。预计算的真实奇异值在 teacher 更新和随机框架对照之间复用；numerical rank 仍来自完整奇异值。

同一命令追加 `--resume` 可恢复版本化权重输出。恢复前核对协议与来源身份，每个跳过的矩阵再次核对原始内容 SHA256；改变输入、范围、精度或求解器时应选择新目录。旧版无 `identity.json` 的目录拒绝恢复。每矩阵结果原子落盘，`progress.json` 给出完成数量和最近矩阵耗时，全部完成后发布 `complete.json`。同目录仅允许一个 writer；共享参考项被其他 writer 占用时立即报错，待其释放后恢复命令。

实现保留三层关系：`extract` 产出可复用的不可覆盖数组；纯数学函数接收矩阵和可选预计算谱；`weights`、`residuals` 等适配器负责内容校验、分块读取和结果发布。指纹和 JSON 写入复用 `core.artifacts`、`core.io`，逐矩阵锁与恢复复用 `teacher_difference.CheckpointStore`，模型加载和 native KL 继续调用既有后端。GPU 的分配与释放由外部 broker/resource guard 管理，分析模块不另建调度器。

相关入口见[模块说明](README.md)，性能与实验预算见 [ISO 计算预算](../../../docs/experiments/teacher-transfer/absorption/附录-定义与复算.md)。
