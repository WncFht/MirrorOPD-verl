# 固定状态吸收轨迹的交互探索

打开交互页面（本地文件 `data/explorer.html`）。页面包含 47 个 run、891 个检查点和五个评分题池，数据及绘图库全部内嵌，下载或复制单个 HTML 即可离线查看。若 IDE 只显示源码或链接不能执行脚本，用系统浏览器打开同一文件；支持从 `file://` 直接访问，也支持普通静态网站托管。

## 操作

选择 student／输入模式，再选择视图与指标。训练池→留出轨迹展示 Math／Code／IF／Avg 四个面板；时间视图增加训练池，共五个面板；终点散点视图在更新上限内为每条 run 选择最后一个有效快照。

尺寸使用点形编码：圆点为同尺寸、菱形为跨尺寸。颜色标识 teacher，所有 teacher 的数据连线均为实线。原始配对图的灰色点状线表示横纵轴数值相等；η 诊断的参考值是 1，Δ 诊断的参考值是 0。侧栏可以只看同尺寸、只看跨尺寸，或混合查看；阶段筛选独立区分追加 SFT、RL 未另加 SFT、Base／OPD／其他对照。

| 操作           | 行为                                                                                               |
| -------------- | -------------------------------------------------------------------------------------------------- |
| 勾选 teacher   | 显示或隐藏对应曲线                                                                                 |
| “仅看”         | 隔离一个 teacher，便于看细节                                                                       |
| 搜索 teacher   | 缩短侧栏列表；不自动修改曲线选择                                                                   |
| 更新上限滑块   | 保留该步及之前的实际快照；64-step run 不外推                                                       |
| 悬停点         | 显示 teacher、尺寸、step、初始／当前 RKL、下降比例、吸收倍数和评分题数                             |
| 滚轮或框选     | 放大感兴趣区域；双击恢复范围                                                                       |
| 图表右上角相机 | 导出当前面板的 PNG                                                                                 |
| 导出当前 CSV   | 导出当前筛选、步数上限下五个题池的读数；终点视图仅导出各 run 最后有效快照                          |
| 复制地址       | 地址中的片段保存 student、视图、指标、尺寸、阶段、teacher 显隐和更新上限；重新打开同一 HTML 可恢复 |

没有符合条件的 run 时页面显示空状态；可调整筛选或点击重置。两个原生 2507 模板没有同尺寸 teacher。所有快照来自同一执行实例的单遍前缀，连续快照有时间依赖，不作独立 seed 重复。

## 曲线量化视图

“视图”新增终点吸收量、累计迁移效率、早期—后期迁移效率、累计吸收差 Δ_D 随 step 四项；定义、静态图和复算数据见[曲线量化诊断](absorption-diagnostics.md)。新视图固定使用各自指标，指标菜单禁用，其他筛选与步数上限继续生效。末窗需要至少 128 步，训练改善量不超过 0.01 dex 时不显示比值；展开图卡可查看缺失原因与原始增量。新视图导出的 CSV 包含派生指标和缺失状态，64 步 run 保留真实预算。

## 数据口径

指标和题池定义沿用[全教师轨迹](student-trajectories.md)。对数吸收量使用线性坐标呈现 $\log_{10}(K_0/K_t)$；RKL 残差使用对数坐标。所有面板统一 fixed-H 分母。部分早期 run 的训练池评分为 128 题，其余为 320 题，悬停和 CSV 均保留真实题数。

“RL，未另加 SFT”限定在谱系中的比较起点之后，仍可继承 R1 蒸馏 SFT 或官方后训练历史；它不等同于整个训练史没有 SFT。同尺寸阶段分类与尺寸关系分别筛选，不合并为“同源”标签。

## 构建与验证

页面使用 Plotly.js 和浏览器原生 HTML／CSS／JavaScript。47 个 run 的数据规模可以全部装入静态页面，不需要后端、账号或数据服务。Plotly 自带悬停、缩放和图片导出；筛选与教师选择由页面处理。导出图及论文用静态图仍由[既有 matplotlib 图集](student-trajectories.md)提供，交互页主要用于探索。

生成器读取已验证的[配对清单](figures/student-paths/inventory.csv)与[逐快照 CSV](figures/student-paths/snapshots.csv)，核对每个 run 的五个题池与 16 步网格，内嵌数据和 Plotly.js；数据哈希与页面哈希保存在[构建回执](figures/student-paths/explorer-provenance.json)。修改 CSV 后重新运行构建器，页面不会自动读取外部 CSV。

在仓库根目录构建：

```bash
uv run --no-project --with plotly==7.1.0 python -m examples.mirroropd.research.studies.fixed_transfer.build_explorer
```

页面源码拆成[结构模板](../../../../research/studies/fixed_transfer/explorer.html)、[交互逻辑](../../../../research/studies/fixed_transfer/explorer.js)和[样式](../../../../research/studies/fixed_transfer/explorer.css)，[构建器](../../../../research/studies/fixed_transfer/build_explorer.py)将三者及 Plotly、数据打包成一个离线 HTML。[指标与视图配置](../../../../research/studies/fixed_transfer/explorer_schema.py)集中定义名称、符号、单位、视图行为和 teacher 配色顺序；[量化函数](../../../../research/studies/fixed_transfer/quantify_absorption.py)提供共享的数据加载、计算和比值状态。静态诊断图与 HTML 使用相同的 teacher 排序和颜色，筛选不会重新分配颜色或 teacher 编号。回执同时记录模板、样式、脚本和指标源码的哈希。

面板顶部按当前视图显示定义、公式和读法，侧栏将原始轨迹与量化诊断分组。统一符号为 $K$（RKL）、$A$（对数吸收量）、$\Delta$（累计吸收差）、$\eta$（迁移效率）；$T$ 为训练评分池、$D$ 为当前留出面板、$e$ 为上限内实际终点。原始轨迹切换到诊断再切回时保留原始指标，诊断视图显示自己的固定指标。

CSV 保留已有存储字段名，避免改变已有复算脚本；对应关系见[符号与导出字段](absorption-diagnostics.md#符号与导出字段)。累计、早窗、末窗比值分别保存状态，防止把预算不足、训练反弹或微小进展混成同一个缺失原因。浏览器验证覆盖离线打开、七组 student、尺寸／阶段筛选、单 teacher 隔离、时间与终点视图、step0／64 截点、CSV 与原值对齐、状态恢复；不发起外部网络请求。

## 两个参考网站可确认的技术

2026-11-07 检查公开 HTML、脚本路径与实际 JavaScript 资源，得到以下前端证据。公开资源可以识别浏览器端框架与页面呈现方式，无法据此确定私有后端与数据库。

| 网站                                  | 可确认的部分                          | 证据与限制                                                                                                                                                                                   |
| ------------------------------------- | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DeepSWE[^deepswe]                     | React、Vite、Radix 组件；Vercel 服务  | 页面加载 `/assets/index-DYZDg9jy.js`，资源内有 `__vite__mapDeps`、React runtime、`react-dom` 和 Radix slot 标识；响应包含 Vercel 标识。页面包含 SVG 与图表交互节点，尚不足以确认具体图表库。 |
| Cognition FrontierCode[^frontiercode] | React、Next.js；Vercel 服务；SVG 图表 | HTML 使用 `/_next/static/immutable/chunks/` 与 Turbopack 资源，脚本包含 React 和 `next/dist`；响应包含 Vercel 标识。图表脚本包含 SVG 元素与坐标、标签相关代码，未确认具体第三方绘图库。      |

对当前实验，更有用的是这些页面的交互方式：挑选模型、悬停精确读数、控制图表密度。若之后需要公开站点、多篇研究报告、共享布局或持续更新数据，可以再用 React／Next.js 组织网站；当前独立 HTML 已覆盖本地读图与筛选需求。

### 参考文献

[^deepswe]: Datacurve. DeepSWE. [deepswe.datacurve.ai](https://deepswe.datacurve.ai/)。

[^frontiercode]: Cognition. FrontierCode Leaderboard. [cognition.com/frontiercode](https://cognition.com/frontiercode)。
