# E39/E41/E43 冻结绘图输入

本目录由 teacher-transfer 专题维护，供 E39、E41、E43 的正式绘图读取。周报引用专题图。指标来自既有测评，本目录没有新增训练或测评。

- `manifest.json` 登记三个 run 的配置路径与 SHA-256、初始参考归属，E39 七份绘图输入，以及 E41/E43 history、解压后的系统采样、file logger、vLLM 采样和退出记录的哈希。配置和完整训练日志仍保存在相邻 `wandb-logs/`；路径相对于该仓库根目录。
- `e43-teacher-metrics.json`、`e43-teacher-complete.json`、`e43-teacher-protocol.json` 是已有 teacher 全集测评的原样冻结副本。原位置和字节哈希保存在 manifest。它们共同证明模型身份、630 题 × 4 回答、解码设置及完成状态；重画不读取原 `runs/` 目录。
- E43 teacher 的温度、top-p、response cap、thinking 和统计口径与 student 核对；采样数和 seed 各自保留。Teacher avg@4、student avg@16 都是平均正确率，不是 pass@k。
- E39 只有历史匹配 8 题子集参考，保留独立子集图及其冻结输入哈希，不生成 teacher 全集水平线；该子集摘要没有 E43 这种完整的生成回执。E41 全集 teacher 参考缺失；初始点引用经完整配置核对的 E39 outer0。

读取器从实际训练配置提取模型、thinking、Full-RKL 条件、reuse、batch、LR、训练 cap 与评测解码字段，并检查配置哈希。E39/E41 的停止动作合并 IDs 与 E43 原生目标分别保留在输出协议中。E43 teacher 必须同时通过字节哈希、完成回执指纹、协议指纹、指标哈希、题数和每题采样数检查，才会用于绘图。

同目录的原始副本保持冻结。新的测评结果使用新的证据包；修正当前读取或显示逻辑时修改绘图源码，并更新专题正文和生成产物。当前生成器、输出路径与复算命令统一见[专题复算入口](../../failure-reproduction.md#qwen-figure-reproduction)。
