# 六种动态前缀的共同评估协议

本包保留 verl 的动态 rollout 训练，新增共同固定状态和纯 student 任务读出；不复用 fixed-transfer 的固定 S0 训练循环。科学配置集中在 `protocol.py`，包括两个 teacher、六个来源、40 outer × reuse16、共同评估和保存频率。旧无评估/无保存运行只作为优化诊断。

数据准备使用原始 DAPO，但另存训练文件：排除全部 320 道保留问题及 630 题任务评测中的规范化文本交集，去除重复训练题，保留 seed42 的 stateful sampler 顺序。Math64、Code32、IF32 为常规评分面板，Math32 同时保留长前缀视图。文本去重不声称解决语义近重复或上游预训练污染。

```bash
python -m examples.mirroropd.research.studies.prefix_fusion.data --workspace .. --out ../runs/prefix-fusion-scientific-v2/prepared-03
python -m examples.mirroropd.research.studies.prefix_fusion.assets --workspace .. --out ../runs/prefix-fusion-scientific-v2/assets
```

两个入口只读取模型元数据和文件，不加载 GPU 权重；输出目录必须不存在。模型身份以本地内容 SHA-256 表示，不伪造 Hub revision。S0/J/P 三套银行分别生成一次，短 H1024 面板是完整轨迹的前缀切片，避免重新生成改变 state。生成器使用原生 BF16、共同合法 token 集、逐题 seed 和 EOS151643；独立 RKL scorer 保留双方原生输出 head。

GPU 银行通过 `banks` 模块生成，须由本仓库的单卡资源 wrapper 分配 GPU6 或 GPU7。银行完成回执核对问题、模型、支持、响应长度、EOS、所选 token 概率及源码身份。数据与银行准备不等于科学训练完成；训练读出与任务评估接线仍在实施，未通过对应验收前不启动新版十二组。
