# 本地训练指标监控

对照当前和历史训练的 held-out RKL、相对初始变化、每批首次／末次更新前 loss 与梯度范数。页面每 15 秒刷新，曲线无平滑；历史运行按选择加载，日志留在本机，无需 W&B 联网。

## 启动与查看

在仓库根目录、具备 Python 3.10+ 的环境中启动：

```bash
python -m examples.mirroropd.tools.metrics_dashboard \
  --root examples/mirroropd/research-evidence/reuse-and-optimizers/sgd-e102-platform-20261114-retry1 \
  --host 127.0.0.1 --port 8765
```

浏览器打开 `http://127.0.0.1:8765`。远程查看通过 SSH 转发该端口；可直连的内网可指定 `--host 0.0.0.0`。服务只提供固定静态资源、运行目录和已登记的指标，不提供任意文件读取、写入或训练控制。

左侧按编号、模型、优化器搜索并勾选运行，也可筛选师生配对。默认显示当前两组和 catalog 中标记 `default` 的历史运行；“E102 对照”选择同配对，“当前两组”恢复当前 SGD。“清空”卸载所有曲线。选择与指标设置保存在浏览器 localStorage。

上方选择题集、原始 RKL／相对降幅／初始归一化、outer／optimizer step、线性／对数轴。图表支持悬浮联动读数、图例隐藏、滚轮和滑块缩放、重置及图片导出。自动刷新保留缩放区间；暂停刷新后可手动读取。没有指标的运行不补点，页面显示缺失提示；刷新失败保留上一轮数据。

## 输入与读取

`--root/plan.json` 的 `lanes` 每项包含 `study`、`experiment`、`lr`、`devices`，计划包含 `outer` 和 `reuse`。示例读取已归档的 E124/E125。相对 study 路径以 `--root` 为起点。每组读取 `training-sgd/<experiment>/metrics.jsonl` 和 `status.json`。这个当前运行入口沿用 E102 SGD 对照的身份；其他当前协议需要显式扩展元数据，不能仅替换计划后假定同一配对。

可选 `--catalog` 为 JSON，顶层 `runs` 数组逐项登记历史运行，字段为：

| 字段                                | 含义                                                                    |
| ----------------------------------- | ----------------------------------------------------------------------- |
| `id` / `name`                       | 唯一运行 ID / 显示名；与当前 ID 不重复                                  |
| `pair` / `student` / `teacher`      | 配对编号和模型身份                                                      |
| `optimizer` / `lr` / `weight_decay` | 实际配置，不从实验名字推断                                              |
| `budget` / `reuse`                  | outer 预算和每批更新次数                                                |
| `group` / `default` / `note`        | 列表分组、是否默认选择和比较限制                                        |
| `metrics` / `status`                | 日志和状态 JSON 路径；`status` 可省略                                   |
| `completion`                        | 可选单运行完成凭据；存在时覆盖共享队列后续失败状态                      |
| `panel_scope`                       | 开发集专用历史运行设为 `development`；其 math domain 只在开发集视图展示 |

`/api/runs` 只返回运行元数据和日志更新时间，不解析全部历史指标。`/api/metrics?id=E124&id=E125` 仅读取选中的运行，空选择返回空曲线；未知 ID 拒绝。标量字段过滤掉无关大块诊断，按文件修改时间与大小缓存，追加后重新解析。此缓存适合当前数十条、每条数百事件的日志，不是百万点时序数据库。

JSONL 中追加尚未结束的末行等待下轮重读；其他解析失败计数并提示。非有限值不进入图，原始日志保留。日志更新时间包括 rollout、评分和 checkpoint，不能仅凭时间判定卡死。

## 指标解释

Held-out 按题平均完整回答的 token RKL；训练 loss 按 token 聚合，两者不直接相减。训练实线读取第 1 次更新前，虚线读取最后一次 reuse 更新前；缺少这些日志的运行不补曲线。开发集不代替正式 held-out。

新日志优先读取 `dataset/math`，避免混入同属 math domain 的开发集；旧日志没有该 namespace 时读取 domain/math。不同运行独立采样的固定 bank 应分别参考初始值。历史 AdamW WD=0.01、当前 SGD WD=0，比较包含衰减差异。跨 pair 的绝对 RKL 不代表统一难度；平台需由后段多个点判断，RKL 不代替准确率。

## 界面参考与依赖

运行选择、图例、横轴切换与缩放参考 W&B line plots 的交互组织[^wandb]。图表采用本地固定版本 Apache ECharts 5.6.0，运行时不依赖 CDN；上游许可证和 NOTICE 随 `vendor/` 保留。没有引入前端构建链或独立数据库。

### 参考文献

[^wandb]: Weights & Biases. Line plots overview. [W&B 文档](https://docs.wandb.ai/models/app/features/panels/line-plot).
