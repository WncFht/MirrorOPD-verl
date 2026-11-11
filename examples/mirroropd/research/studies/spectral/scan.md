# 全层与训练阶段的分组谱扫描

`scan.py` 读取已有配对 manifest，输出每个 base 矩阵下 teacher、student endpoint 和可选中间阶段的精确谱统计。该入口不加载语言模型、不前向、不保存参数方向。完整分量吸收和随机对照使用 [weights](weights.md)，实验范围与时间见 [ISO 计算预算](../../../docs/experiments/teacher-transfer/absorption/附录-定义与复算.md)。

## 输入、范围与命令

在仓库根目录执行。manifest 的每个 job 提供 `job_id`、`base`、`teacher_path`、`endpoint`、`teacher`；启用 `--stages` 时另外读取 `gradient_stages` 的名称与路径。checkpoint 是本地完整 safetensors 导出。默认选取所有层 attention/MLP 二维权重；七投影模型对应 q/k/v/o、gate/up/down。选中矩阵逐一校验 shape，额外且未选中的 MTP 参数不影响比较。谱系身份由 manifest 决定，同 shape 不构成同源证据。

```bash
python -m examples.mirroropd.research.studies.spectral.scan \
  --manifest runs/teacher-spectrum-expansion-20261110-v3/manifest.json \
  --output runs/spectral-scan-new \
  --device cuda:0 --driver gesvd --transpose-wide \
  --updates --stages --cache runs/spectral-scan-cache \
  --threads 2 --stop-paths runs/spectral-scan-new/STOP
```

上例是 worker 内的分析命令；预约 GPU 时通过 `research.runtime.broker.run_leased` 启动。`--device cuda:0` 表示 guard 映射后的逻辑设备。CPU 使用 `--device cpu` 并省略 `--driver`；算术固定 FP64。`--job-ids` 可限制配对，`--pattern` 可限制层和矩阵。报告面板沿用 `absorption.cohort.report_eligible`；显式请求污染条件或不存在的 ID 会报错。

## 指标与计算路径

每个 base 矩阵执行一次 thin SVD，供同一组所有目标复用。每个不同 target 计算完整权重奇异值、相对 base 的谱距离 δΣ、更新归一化距离 ρΣ、base frame 对角更新能量对应的 κ。数学定义沿用 `decomposition.weight_summary`。对角能量通过矩阵乘法和逐项收缩求得，省去完整 diag/off/null 张量重构。

记某一层某个投影的 base 权重为 $W_0\in\mathbb R^{m\times n}$，目标权重为 $W$，$D=W-W_0$，$r=\min(m,n)$。所有相减和范数都在这一矩阵的相同坐标上计算，不拼成整个模型的大矩阵。Base 的 thin-SVD 为 $W_0=U_0\Sigma_0V_0^\top$。定义：

$$
s=\Vert\sigma(W)-\sigma(W_0)\Vert_2,\qquad
\delta_\Sigma=\frac{s}{\Vert W_0\Vert_F},\qquad
\rho_\Sigma=\frac{s}{\Vert D\Vert_F},
$$

$$
c_i=u_{0,i}^\top Dv_{0,i},\qquad
\kappa_\Sigma=\frac{mn}{r}\frac{\sum_{i=1}^{r}c_i^2}{\Vert D\Vert_F^2}.
$$

δΣ 表示完整权重谱相对 base 尺度的移动；ρΣ 用更新大小归一化该移动；κ 比较更新在 base 对角谱方向上的能量和各向同性更新的尺度。它们描述不同方面，结果同时保留未归一化的 $s$ 与 $\Vert D\Vert_F^2$。更新谱的 rank-16 能量比例来自 **$D$ 的 SVD**：前 16 个奇异值平方和除以全部平方和；完整权重 `singular_values` 来自 **$W$ 的 SVD**，两个对象分别保存。

`--transpose-wide` 将宽矩阵转置成瘦高矩阵后调用相同精确 solver，再还原 base 的 U/V。完整谱保持不变；收益按形状实测。这里只改变算子布局，未改成随机或截断 SVD。

| 选项        | 额外计算                                                | 结果                                     |
| ----------- | ------------------------------------------------------- | ---------------------------------------- |
| 默认        | 每个 target 完整权重的 `svdvals`                        | `singular_values`、δΣ/ρΣ/κ、绝对更新能量 |
| `--updates` | target−base 及每个配对 teacher−student 的精确 `svdvals` | 更新与 missing 的全谱秩、能量统计        |
| `--stages`  | 读取 manifest 中可取得的阶段；重复路径去重              | 各阶段 target 统计和配对记录             |

零更新时 ρΣ 和 κ 保存为 `null`。初始阶段的完整权重奇异值复用 base；启用更新谱后，零更新谱仍按当前统一路径求解。完整权重的奇异值数组可供后续计算谱分位点；更新和 missing 谱同时保存完整奇异值数组与 `spectrum_summary`，后续统计不同 rank、谱分位点或熵无需再次分解；不保存所有方向。

## 缓存、恢复与结果

同一个 base 矩阵的全部目标在单个 worker 中处理，避免跨阶段重复 base SVD。共享缓存只保存标量和奇异值，缓存键包含 tensor 名、base/target 原始内容 SHA256 和 dtype、FP64 算术、逻辑设备、solver、转置选项、更新谱开关及 PyTorch 版本。不同路径但相同内容可复用；暖缓存仍需读取、哈希和 base 分解，missing 谱当前逐配对重算。

缓存项从查找到计算和发布期间持有 `CheckpointStore` 锁。遇到其他 writer 的同项锁时立即失败；待其结束后恢复。阶段尽量在同一个组内扫描，减少跨 worker 争用。

八卡执行时，各 worker 使用相同 manifest 和 `--shards 8`，分别传入 `--shard-index 0` 至 `7`，并使用独立输出目录。每个 base 的排序矩阵按序号模 8 分配，同一矩阵的全部阶段归属同一个 worker。分片配置纳入恢复协议；同一共享缓存允许不同矩阵并发发布。资源预约继续调用原 broker，不另建分布式调度服务。

当前扫描协议与缓存指标版本为 2，追加保存更新及 missing 的全部奇异值；版本 1 的基准摘要不作为版本 2 的缓存命中。每个 base 矩阵原子保存一份 JSON，包含 `inputs`、`targets`、`pairs`、`svd_calls` 和分项计时；`protocol.json` 记录完整配置和 checkpoint 身份。`complete.json` 仅在扫描全部结束后发布。失败保留目录，输入未改时用原命令追加 `--resume`；恢复会重新校验原始 tensor 哈希。改变来源、范围、solver 或选项时使用新输出目录。

`performance.json` 汇总初始化、墙钟和各阶段耗时。GPU 计时前后同步；单矩阵 `phases_seconds` 在自身结果写盘前截取，因此不含本条 `result_publish`，全局汇总包含此项。墙钟还包括缓存锁、进度发布与 Python 调度等开销，不能只把 SVD 项称作总时间。STOP 在矩阵边界和每个 target 前检查；正在执行的单次 solver 由 guard 统一保护。

## 数值与性能验证

`tests/research/test_spectral_scan.py` 覆盖方阵、宽矩阵、瘦高矩阵与原数学实现的一致性，以及阶段复用、暖缓存、恢复、内容变更拒绝、STOP 和污染条件排除。限量真实矩阵基准入口为：

```bash
python -m examples.mirroropd.research.studies.spectral.scan_benchmark \
  --manifest runs/spectral-scan-infra-20261111-v1/manifest.json \
  --job-id 1 --output runs/spectral-scan-benchmark-new \
  --stop-paths runs/spectral-scan-benchmark-new/STOP
```

基准依次运行原布局冷扫描、转置宽矩阵冷扫描、追加更新谱冷扫描及暖缓存扫描；使用第 0 层 q/k/up/down 四种形状。k 矩阵与 CPU FP64 原实现比较 δΣ/ρΣ/κ，四矩阵比较布局前后指标，相对误差要求 ≤10⁻⁶。另测单个 k 的原完整诊断；两者输出范围不同，耗时比不能称为全功能等价加速。真实数据与阶段预算由 [scan_costs.py](scan_costs.py) 复算。
