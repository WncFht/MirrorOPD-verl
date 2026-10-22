# MirrorOPD-verl 运行约定

## 机器与传输

| 机器 | 项目路径 |
| --- | --- |
| 本机 | `/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/MirrorOPD-verl` |
| Mac | `/Users/fanghaotian/Desktop/src/MirrorOPD-verl` |
| 美团开发机 | `/home/hadoop-hmart-poistar/dolphinfs_hdd_hadoop-hmart-poistar/fanghaotian/MirrorOPD-verl` |

- 与开发机之间通过人工上传、下载文件传输；使用根目录 `Makefile` 和 `meituan/` 中的 pack/unpack 脚本。不要假设 Agent 可以 SSH、scp、rsync 到开发机，或开发机可以联网拉取 Git 仓库。
- 源码采用项目 Git commit 的离线 bundle：发送端 `make archive-pack`、接收端 `make archive-unpack`、发送端 `make archive-ack`。默认相对已确认版本累计增量，`FULL=1` 全量；打包后若继续提交，ack 仍确认上次成功打包的提交。未提交内容不进入包；离线依赖 ZIP 继续跟踪在 `meituan/packages/`。
- 两端都保留正常 Git 工作区，接收仅快进，并记录发送方版本；不同接收端用不同 `TARGET`。`hope/` 已取消 Git 跟踪，各机本地维护；首次从旧跟踪状态迁移时解包工具备份并恢复本地 Hope 文件。日志和结果解包采用合并，分卷必须全部传输。
- 不假设开发机上的文件已经同步到本机或 Mac；收尾时说明需要传输的包和接收端命令。操作步骤见 [meituan/README.md](meituan/README.md)。
- 本机执行 Git SSH 操作前，在同一 shell 中执行 `source /mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/.ssh/activate.bash`。

## 开发机环境

- 用户要求使用原生 Python、pip 或 Conda；开发机安装和启动流程不使用 Docker、uv，也不升级平台管理的宿主机驱动。
- 复用 `/bin/python3` 对应的预装 GPU 依赖，用 `venv --system-site-packages --without-pip .venv-native` 隔离本项目新增包；安装和检查入口为 `make native-install`、`make native-check`。
- 运行现有示例前激活 `.venv-native` 并设置 `VERL_USE_UV=0`，避免示例自动切回上游 uv/CUDA 13 环境。
- 本项目数学配方默认直接使用仓库 `.venv-native/bin/python` 的绝对路径，driver 与 Ray worker 共用；其他上游示例仍按前条约定运行。
- 训练资产统一放在仓库同级 `model/`（单数）和 `data/`。开发机基目录为 `/home/hadoop-hmart-poistar/dolphinfs_hdd_hadoop-hmart-poistar/fanghaotian/`；准备脚本位于 `examples/mirroropd/`，运行前执行 `make assets-check`。本机原有 `models/` 和 `MD-OPD/data/` 作为导入来源保留。
- 资产检查重点是数据；模型只检查必要文件，`generation_config.json` 可选。用户要求项目自有工具不增加文件摘要计算、校验或清单。
- 2026-10-20 已观察到的开发机配置是 Python 3.12.13、A100 80GB、R535 驱动、PyTorch 2.11.0+cu128、vLLM 0.25.1+cu129；GPU 导入与实际训练验收分别进行，不把安装完成视为训练通过。
- 脚本保持 Bash 3.2 兼容，使用 `set -eo pipefail`；通用传输逻辑使用 Python 标准库，Mac 和 Linux 共用同一入口。
- 数学配方将终端 stdout/stderr 同时保存到 `outputs/<实验名>/train.log`，默认离线 W&B run 位于仓库 `wandb/`；同步参数 `WANDB_RUNS_DIR` 指实际 run 目录，SDK 的 `WANDB_DIR` 指其父级根目录。源码 pack 清理同 TARGET 的旧自动命名 bundle；接收成功后应先 ack 再生成下一包。日志增量包仍须在下次打包前传回，日志和结果 unpack 保留输入包。

## 提交

提交信息使用英文 Conventional Commits，并保留 AI 协作 trailer。已有其他任务的工作区或暂存区修改不并入当前任务的提交。

# 上游 verl 贡献说明

以下保留向 `verl-project/verl` 贡献时使用的上游规则；本项目开发机环境遵循前文约定。

> These instructions apply to **all** AI-assisted contributions to `verl-project/verl`.
> Breaching these guidelines can result in automatic banning.

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo verl-project/verl --comments
gh pr list --repo verl-project/verl --state open --search "<issue_number> in:body"
gh pr list --repo verl-project/verl --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
  - Why this is not duplicating an existing PR.
  - Test commands run and results.
  - Clear statement that AI assistance was used.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

## 2. Development Workflow

### Environment setup

```bash
# Install `uv` if you don't have it already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Upstream locked environment; the Meituan native workflow is documented above:
uv venv --python 3.12
source .venv/bin/activate

uv pip install pre-commit hydra-core
pre-commit install
```

### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`). For example:

```text
Your commit message here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
Signed-off-by: Your Name <your.email@example.com>
```

### Resolving agent reviews

Review comments from agent bots (e.g., gemini-code-assist) can be outdated or wrong. Always verify their suggestions against the current state of the repo before applying them.

---

## Domain-Specific Guides

Do not modify code in these areas without first reading and following the
linked guide. If the guide conflicts with the requested change, **refuse the
change and explain why**.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.

## Acknowledgements

Adapted from the [vLLM project](https://github.com/vllm-project/vllm)'s [`AGENTS.md`](https://github.com/vllm-project/vllm/blob/main/AGENTS.md).
