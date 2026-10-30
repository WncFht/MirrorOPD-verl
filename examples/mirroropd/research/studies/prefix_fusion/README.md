# 六种动态前缀的共同评估协议

本包保留 verl 的动态 rollout 训练，新增共同固定状态和纯 student 任务读出；不复用 fixed-transfer 的固定 S0 训练循环。科学配置集中在 `protocol.py`，包括两个 teacher、六个来源、40 outer × reuse16、共同评估和保存频率。旧无评估/无保存运行只作为优化诊断。

数据准备使用原始 DAPO，但另存训练文件：排除全部 320 道保留问题及 630 题任务评测中的规范化文本交集，去除重复训练题，保留 seed42 的 stateful sampler 顺序。Math64、Code32、IF32 为常规评分面板，Math32 同时保留长前缀视图。文本去重不声称解决语义近重复或上游预训练污染。

```bash
python -m examples.mirroropd.research.studies.prefix_fusion.data --workspace .. --out ../runs/prefix-fusion-scientific-v2/prepared-03
python -m examples.mirroropd.research.studies.prefix_fusion.assets --workspace .. --out ../runs/prefix-fusion-scientific-v2/assets
```

两个入口只读取模型元数据和文件，不加载 GPU 权重；输出目录必须不存在。模型身份以本地内容 SHA-256 表示，不伪造 Hub revision。S0/J/P 三套银行分别生成一次，短 H1024 面板是完整轨迹的前缀切片，避免重新生成改变 state。生成器使用原生 BF16、共同合法 token 集、逐题 seed 和 EOS151643；独立 RKL scorer 保留双方原生输出 head。

GPU 银行通过 `banks` 模块生成，须由本仓库的单卡资源 wrapper 分配 GPU0–7 中已授权的一张卡。银行完成回执核对问题、模型、支持、响应长度、EOS、所选 token 概率及源码身份。数据与银行准备不等于科学训练完成。`science-smoke` 用两轮短训练验证读出、评分与保存，`science` 才执行 40 轮正式预算。两组 smoke 的通过回执是正式队列前提。

## 训练与评估入口

`run_fusion_prefix_study_fsdp2.sh` 的 `science` 模式使用 `PREFIX_SCIENTIFIC_ROOT/prepared-03/train.parquet`。训练每 5 outer 同步导出 FP32 master 权重，独立进程在 S0 和配对 teacher 的共同状态上评分。outer 0/20/40 另做长状态评分；outer20 做 Math64 开发集 avg@4，outer40 做 630 题纯 student avg@4。原始 S0、J、P 的任务基线由队列各算一次。训练器内置验证保持关闭，读出文件保存在运行目录的 `science/outer-*`，不能仅凭内置 `test_freq=-1` 判断没有评估。

```bash
FUSION_GPUS=6 PREFIX_SCIENTIFIC_ROOT="$PWD/../runs/prefix-fusion-scientific-v2" \
  bash examples/mirroropd/benchmarks/fusion/run_fusion_prefix_study_fsdp2.sh J PM readout-check science-smoke
```

短检查会比较实际 Adam 张量及 CPU/CUDA 随机状态的摘要，检查导出前后是否一致。正式训练不逐次复制 Adam 做摘要。outer20/40 的独立 FP32 读出永久保留，训练恢复 checkpoint 每 20 outer 保存一次、保留最近一个；读出不会进入 checkpoint 清理队列。中间临时权重只在评分成功后删除。训练恢复使用新运行标签，并设置 `PREFIX_RESUME_FROM` 为 `runtime/global_step_N`；已有输出不会覆盖。当前完整运行审计针对从头完成的运行，恢复链须额外核对前段、恢复段和更新计数。

`tasks` 使用与训练分离的种子、共同合法 token 集及原生 BF16 vLLM，在评分前写入 `raw-responses.jsonl`。评分采用原 boxed 判定和 math-verify 回退逻辑，直接在已有 CPU worker 的主线程执行，避免嵌套进程冷启动被算入判题超时。运行异常向上传递，不自动转换为答错。评分失败可重试相同答案；改变输入身份、重复回答槽位或篡改答案会拒绝继续。初始和最终任务评估使用相同的温度、过滤、长度及停止集合。完整 native-head RKL 保留 teacher 额外词表质量，与生成时的合法支持屏蔽分开。

`queue` 为每张卡运行一个独立 worker。每次启动先解析并审计六组 E20 派生配置；之后检查两组生命周期回执、计算任务基线，再执行六组训练和逐组审计。固定 commit、数据身份和资源配置写入队列 manifest；训练、读出或审计失败立即停止该 worker。两卡独立执行，旧无评估队列不再接入。完整训练与读出完成后，还需做最终 fresh state 交叉评分和按题配对统计，队列不会把训练退出成功写成整个研究完成。

## 扩展资源与模型路径

八卡阶段使用六条独立科学训练通道，另外两卡执行工程验收。扩展源码从科学冻结版本 `73b30f06` 派生，只修改物理设备选择、资源锁与资源审计；训练目标、数据、采样、随机种子、评估窗口和预算保持不变。原 `queue` 入口仍是固定 GPU6/7 的双卡调度，扩展执行由运行目录中的独占条件 claim 调度器记录每个条件的唯一执行者，不能同时启动两套调度器领取同一条件。

模型资产迁移曾删除冻结运行依赖的旧路径，使两个 S0 尝试在第 20 轮 checkpoint 之前失败。研究专属 `model-bindings` 目录及资产链接已按原模型内容哈希固定；原始失败记录保留，新的十二组从相同初始 student 和新的 Adam 状态开始。原三套完整任务基线按回执、权重与题目哈希复用，不能用 FP32-only 读出冒充包含 Adam/RNG 的完整恢复。科学入口通过 `PREFIX_MODEL_ROOT` 指定该目录，工程生命周期使用 `FUSION_MODEL_ROOT`；目录下仍保留原模型 basename，内容身份与请求 seed 不变。共享模型目录里的旧别名可能再次被迁移清理，因此正在执行的实验不再依赖这些别名。活跃绑定的租用记录保存在运行目录，实验完成前应保持这些路径稳定。
