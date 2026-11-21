# Rollout 状态几何

复用已保存的回答，由每个 pair 的固定 teacher 编码 prefix，比较训练阶段与 initial／teacher 回答的状态分布。主协议使用全部训练 student rollout 建立固定参考空间，构造接近 One-Shot OPD 的 full-data reference。[^paper] 模块没有生成回答的入口。运行产物放在仓库 `runs/`；[交互报告](../../../docs/experiments/teacher-transfer/checkpoint-dynamics/latent-state.html)和 PNG／PDF 位于所属实验专题，旧的[六来源 pooled 首版](../../../docs/experiments/teacher-transfer/checkpoint-dynamics/latent-state-pooled.html)保留作方法对照。

## 参考与查询

状态为生成响应 token 前的 prefix，签名是固定 teacher 最终层、final norm 后的 hidden vector。每条回答取最多八个相对位置，索引为 `prompt_length - 1 + response_action`。`action=0` 是共享 prompt 末尾状态，仅展示，排除在拟合和指标之外，因此长回答最多贡献七个指标状态。

每个 pair 使用 40 outer × 64 = 2,560 条已保存训练回答。按完整 prompt token 序列的哈希作约 60/40 reference／test 划分，同 prompt 跨 outer 保持同侧。PCA50 只在训练 reference 状态拟合一次；K=50／100／200 共用 PCA，分别拟合 K-means。训练 test 和 initial／teacher 各 128 条 Math 回答全部只变换和分配。bank 保存的原始哈希 split 不参与筛选，报告将其标为 query。参考不按训练阶段重新平衡，也不加入 teacher rollout。

各 pair 有独立 teacher 和参考空间，簇编号与绝对距离不能跨 pair 对齐。六个展示来源为 initial、四个训练阶段（outer 1–10／11–20／21–30／31–40）、teacher；另汇总四阶段全部训练 test 作为指标基线。初始／教师 bank 同题，训练阶段之间题目不同，分布变化同时包含 prompt、回答长度和策略变化。训练 rollout 在该 outer 更新前生成，outer40 的回答不等同最终 step640 的新采样；producer 计数保留原值，不换算为 optimizer updates。

| 指标                           | 定义与用途                                                                                                    |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| JS → 训练测试集                | 查询来源与训练 test 的簇占用分布之 Jensen–Shannon divergence，单位 bits；同时保存全量值和等回答预算重采样结果 |
| Reference coverage             | 等回答数下访问过的训练参考簇比例；预算取各查询组最小可用回答数，上限 128，重复 30 次                          |
| Reference 10NN cosine distance | 查询状态到训练 reference 的十近邻平均 cosine 距离，使用原始 hidden 空间                                       |
| 超过训练 test p95              | 超过训练 test 邻域距离第 95 百分位的状态比例；描述性阈值                                                      |
| 簇占用矩阵                     | 各来源全部查询状态的计数与行归一化占比；不采用 coverage 的子集抽样                                            |
| t-SNE                          | 对所有展示点的 PCA 坐标拟合的探索性地图；二维位置和线长不进入上述指标                                         |

JS 每次先抽取查询回答，再从训练 test 中排除这些回答 ID，抽取等量目标回答。Training test 基线采用同样的不相交子集规则。预算上限 128，并受最小查询组和训练 test 半数限制；各 K 使用同一规则和种子，重复 30 次。± 为子集抽样标准差，不是总体置信区间；同回答的七个状态有相关性。

覆盖更多参考簇不代表任务得分更高。K-means 对参考之外的状态也总会分配最近簇，需同时看原空间距离。添加新查询来源不需要重拟合 PCA／K-means；增加训练参考数据或改变取点协议则要另建 plan。t-SNE 使用全部展示点，新增来源会改变地图，不能据此断言原几何改变。与论文的差异包括训练条件、整题哈希划分、最多七个指标状态和 128 条查询预算；不能直接比较论文数值。

## 全局教师对照

首页的全局趋势由 `overview.teacher_overview` 读取已有 K50／100／200 标签计算，不重拟合空间。四个训练阶段只取几何 test，Initial／Teacher 使用全部 query 候选。每次各抽 64 条回答，重复 30 次；JS 目标是随机选取的 Teacher 子集，Teacher 基线与目标是两个不相交的 64 条子集，Initial 与目标按相同 prompt 配对。曲线误差棒为子集抽样标准差，不是置信区间。Coverage 使用同一批查询回答，Teacher 水平线也采用 64 条预算；与原有上限 128 条的训练 test 指标不可混用。

首页热图展示全部簇，每行固定一个簇、列为 Initial 和四个训练阶段，颜色表示全部查询状态占比减 Teacher 占比，单位百分点；同一 pair／K 共用对称平方根色阶以显示稀有簇的差异。可以按 Teacher 质量、末段差异或簇编号排序，悬停查看数值，点击固定并跳转簇明细。热图用全部测试状态，曲线用等回答预算，二者用途不同。不同阶段题目不同，曲线是访问分布的描述性比较。

## 网站访问

从仓库根目录启动只读网站，默认监听 `127.0.0.1:8766`，转发 8766 端口后打开 `http://localhost:8766/`：

```bash
python -m examples.mirroropd.research.studies.latent_state serve --port 8766
```

网站独立加载页面、交互组件和 `/api/report` 数据，JSON 支持 gzip 传输；不需要下载或手动打开 HTML。启动时读取已有报告快照，不重新提取状态或计算指标。`--report /path/to/latent-state.html` 可切换报告，`--host` 可指定监听地址。数据更新后重启服务，前台按 Ctrl-C 停止。离线 HTML 入口继续可用；两种方式共用组件与数据。

## 数据与执行

`evaluation/trajectories/outer-NNN.parquet` 保存 prompt／response token、mask、outer 和 producer 元数据，未保存本模块所需 hidden vectors。`prepare` 选择完整的 40-outer 训练源，默认保留每 outer 全部 64 条，再加入已有 initial／teacher 各 128 条 Math 回答，每 pair 共 2,816 条。`--per-outer` 可显式缩减预算，新 plan 会记录该设置。

```bash
python -m examples.mirroropd.research.studies.latent_state prepare \
  --catalog examples/mirroropd/research-evidence/latent-state/state-latent-20261115/rollout-inventory-e94-e123.json \
  --output runs/my-training-reference --pairs E100 E102 E109
python -m examples.mirroropd.research.studies.latent_state extract \
  --plan runs/my-training-reference/plan.json --pair E100
python -m examples.mirroropd.research.studies.latent_state analyze \
  --plan runs/my-training-reference/plan.json --pair E100
python -m examples.mirroropd.research.studies.latent_state report \
  --plan runs/my-training-reference/plan.json \
  --output examples/mirroropd/docs/experiments/teacher-transfer/checkpoint-dynamics/latent-state.html
```

`extract` 和 `analyze` 需对计划中的每个 pair 分别完成，随后生成报告。准备和提取使用与原 Parquet 兼容的 PyArrow，本机为 `/root/.local/share/mirroropd/venvs/profile-cu128/bin/python`，含 Torch／Transformers。CPU 分析需要 NumPy／scikit-learn，报告另需 matplotlib；本机 `/usr/bin/python` 满足这些依赖。报告沿用 `research.core.plotting`，设置 `FIGURE_VIZ_MPL` 为 figure-viz skill 的 `assets/mpl` 目录。命令不自动安装依赖；GPU 提取须在资源租约内执行。

`queue` 通过现有 broker 等待空闲占位卡，每个 pair 租用一张卡，默认在前四卡中用三张并行。提取结束归还卡，再顺序执行 CPU 分析；所有 pair 就绪后才生成报告。`--workers 1` 可串行，`--extract-only` 仅做提取。不会重分配现有模型租约，默认等待上限 24 小时。

```bash
/path/to/gpu-python -m examples.mirroropd.research.studies.latent_state.queue \
  --plan runs/my-training-reference/plan.json \
  --broker /path/to/existing-idle-occupancy --gpus 0 1 2 3 --workers 3 \
  --analysis-python /usr/bin/python --output /path/to/report.html
```

队列持有独占锁并写 `queue-status.json`。在 plan 同目录创建 `STOP` 可取消等待或当前受保护子进程。失败保留分片与日志，重跑同一命令可复用已验证缓存；准备阶段拒绝覆盖原 plan。缓存绑定输入内容哈希、teacher 配置与权重文件元数据、选择规则、提取代码和执行设置，并校验 NPZ 内容哈希。权重使用文件大小与 mtime 标识，没有逐字节哈希数 GB 权重，共享模型目录需保持只读。身份变化会拒绝复用，应另建 plan；旧 pooled 结果可继续阅读，其缓存不自动迁移到新 schema。

## 代码与扩展

`data.normalize` 是唯一 token／mask 适配，拒绝 mask 空洞；`artifacts` 统一身份和原子写入。`extract` 按长度排序，在 padded-token budget65536、最大 batch8 内组批；decoder-only SDPA 前向不产生 logits 或 KV cache，只搬运选定状态。每个 pair 加载一次 teacher，首次校验完整序列与截断因果 prefix 的签名一致性。缓存附带有效 tokens/s、前向耗时、峰值显存和当前位置之前最多 64 token 的文本片段，片段仅帮助解释轨迹，不代表对 hidden vector 的语义解码。

`analysis` 与独立的 `geometry` 函数在 CPU 上拟合和计算指标，保存 PCA、簇中心、参考索引、来源身份和依赖版本。`report_data` 验证配置与签名身份，读取全部 41 个原有 RKL 评估节点，整理实测执行成本；`report` 只重绘，不重拟合或抽样。新增指标不必修改 GPU 提取，新增报告不必修改几何拟合。`history` 支持逐批库存查看与 JSONL 导出。

```bash
python -m examples.mirroropd.research.studies.latent_state.history \
  --evaluation /path/to/training/E100/evaluation
```

固定同题的最终 checkpoint 对照需要另提供已保存回答，不能用训练末轮回答替代。旧 `latent_state.checkpoint` 原型已由 prepare／extract／analyze／report 入口替代。

## 可复用交互组件

三个组件使用原生浏览器 API，接受数据和回调，不依赖实验 ID、框架或网络。报告把脚本、样式和数据内嵌为单文件 HTML。

| 组件                                    | 输入与行为                                                                                                                                                                                    |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [LatentScatter](viz/scatter.js)         | 接受 `{points, groups, tooltip, onSelect}`；共用尺度的六图、来源／位置着色、reference／test／query 筛选、叠加、缩放平移和 PNG 导出；悬停详情占据图外固定区域                                  |
| [TrajectoryExplorer](viz/trajectory.js) | 接受 `{points, groups, texts, onStep}`；按来源／题目查找回答，编号箭头、滑块、逐步播放、每步簇标签和 prefix 文本；切换同题来源时保留 prompt                                                   |
| [ClusterOccupancy](viz/occupancy.js)    | 接受 `{points, groups, onSelect, onPoint}`，`update(k, metrics)` 更换 K；每行一个簇，按 Initial／四阶段／Teacher 列出占比或相对 Teacher 的百分点差；排序、20 行分页、选簇条形图和示例回答选择 |

地图点字段为 `{x,y,group,trajectory,order,position,metadata}`；`metadata.clusters` 同时保存 K50／100／200 的真实标签。`position` 为 0–1 的相对回答位置。轨迹文本独立存储，避免在八个状态上重复内嵌题目。控制器用回调联动簇高亮、选中回答和当前状态，地图筛选不改变统计指标。地图旁默认选择 K200，可切换 K50／100；K 改变真实簇标签与指标，不改变 t-SNE 坐标。地图默认两列大图，支持单来源放大、点大小调节和突出选中轨迹。响应位置使用蓝→绿→橙的色相渐变及图例，其他来源仅以低透明背景显示。

```javascript
const view = new LatentScatter(root, {
  points, groups,
  tooltip: point => JSON.stringify(point.metadata),
  onSelect: point => explorer.select(point.trajectory, point.order)
});
view.update({mode: "position", k: "100", cluster: 4});
view.destroy();
```

散点组件由 `outputs/opd-latent-1109v2.html` 的 canvas、分组高亮与命中检测整理而来。滚轮缩放、拖动平移、双击或 0 重置；+/- 缩放、Esc 清除地图选择。各 panel 共用缩放并保持 x／y 等比例，DPR 与容器 resize 自动重绘。`destroy()` 取消动画帧／播放定时器，移除组件 DOM 和观察器；容器内容归组件所有。报告默认进入全局趋势，簇明细、地图、单条回答与输出匹配另设视图；设置、执行成本和辅助指标统一折叠。另提供 pair 标签、Math／Code／IF／三域等权 RKL、固定 bank 切换、线性／对数／剩余比例读数和 SVG／JSON 导出。

## 来源与验证

上游状态几何实现固定于提交 `2308464413d1cf0cd1ad50d740858b4be1e7f3df`，七个冻结源码及哈希见 [sources/provenance.json](sources/provenance.json)，保存为 `.py.txt`。[^source] CPU 回归覆盖状态对齐、左 padding、整题划分、训练参考隔离、JS 两侧回答互斥、kNN 数值、缓存篡改拒绝与 producer 身份保留。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m pytest \
  tests/examples/research/test_latent_state_on_cpu.py -q
```

### 参考文献

[^paper]: One-Shot OPD. Appendix B.1. arXiv 2026. [2609.04172v1](https://arxiv.org/html/2609.04172v1#A2.SS1).

[^source]: explcre. Latent state map. GitHub 2026. [固定版本源码](https://github.com/explcre/opd-state-literature-review/tree/2308464413d1cf0cd1ad50d740858b4be1e7f3df/experiments/2026-10-30-opd-diagnostics/oneshot/latent_map).
