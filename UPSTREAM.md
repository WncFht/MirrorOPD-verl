# 上游版本与来源

## verl 基线

本仓从官方 verl commit `263653e749306287a5032140cce3d7833c1959d0` 开始开发，保留完整可达 Git 历史。基线标签为 `verl-base-2026-10-19`。[^verl]

- 上游提交时间：2026-10-19 13:08:09（Asia/Shanghai）。
- 上游提交标题：`[veomni,ci] feat: add uv-based veomni training backend and cover it in GPU CI (#8274)`。
- `origin`：`https://github.com/WncFht/MirrorOPD-verl.git`。
- `upstream`：`https://github.com/verl-project/verl.git`。

初始化提交仅整理 README 与来源记录。`README.verl.md` 与基线的 `README.md` 内容相同；训练源码、依赖声明、锁文件、许可证和 recipe submodule 引用沿用基线。

## 算法与配方来源

MirrorOPD 与 OPD-State 是拟迁移的算法和实验实现来源，初始化时尚未引入其代码。实际迁移时应记录各自的来源 commit、修改文件与验证结果。

模型和训练数据配方参考 One-Shot-OPD commit `6b6b1782cc3f04749dd077de19353f839f6bc048`。它的嵌入式 verl 和 V0/FSDP teacher 路径不是本仓的框架基线；本仓使用上述官方 verl 的 V1 训练入口。[^oneshot]

## 版本更新

基线标签保持指向原 commit，不随开发移动。后续上游同步或依赖升级使用独立提交，记录新的来源版本与兼容性验证；已有实验保留运行时实际 commit、解析配置、模型与数据身份。

查看当前开发相对固定基线的改动：

```bash
git diff verl-base-2026-10-19..HEAD -- verl examples tests
git log --oneline verl-base-2026-10-19..HEAD
```

### 参考文献

[^verl]: verl-project. verl. [固定 commit](https://github.com/verl-project/verl/commit/263653e749306287a5032140cce3d7833c1959d0).
[^oneshot]: Thinking-Space. One-Shot-OPD. [固定 commit](https://github.com/Thinking-Space/One-Shot-OPD/commit/6b6b1782cc3f04749dd077de19353f839f6bc048).
