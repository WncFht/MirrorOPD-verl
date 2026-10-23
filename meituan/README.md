# 美团开发机安装与人工同步

本目录参考 MD-OPD 的 `archive.sh`、`sync-log.sh`、`sync_results_from_remote.sh` 和 Makefile，提供本项目的人工压缩包传输入口。开发机与本机、Mac 之间通过上传和下载文件交换代码与结果。脚本负责打包、分卷、解包，不建立 SSH 或 rsync 连接。

三个 checkout 的固定路径记录在根目录 [AGENTS.md](../AGENTS.md)。脚本从自身位置确定仓库根目录，因此可直接在三处使用。传输工具使用 Bash 和 Python 标准库，需要 Python 3.12 或更新版本；开发机的训练环境使用 Python 3.12。

## 源码双向同步

源码使用 Git bundle 传输当前 `HEAD` 的提交历史和对象。两端需要 Git 2.28 或更新版本；操作只读取本地仓库和上传的文件，不连接 GitHub 或其他机器。`make archive-pack` 只包含已提交内容，工作区有未提交修改时会打印提示，不自动提交或暂存。

每次同步按以下顺序执行；反向同步时交换发送端、接收端即可：

```bash
# 发送端：先提交需要同步的代码，再打包。
make archive-pack

# 接收端：人工上传后，导入包并快进更新当前分支。
make archive-unpack ARCHIVE=/实际上传目录/archive_MirrorOPD-verl_peer_....bundle

# 发送端：确认接收成功后执行，不需要输入 commit ID。
make archive-ack
```

`ack` 将上次成功打包的 commit 记录为对方已收到的版本，即使打包后本地又产生新提交，也不会把后来的提交误记为已同步。确认某一包前不要再生成下一包，否则 `ack` 将确认新的上次打包记录。没有成功打包记录时，确认命令报错；重复确认同一个版本可安全执行。

### 累计增量和全量

首次没有对方版本记录时自动生成全量包。之后相对已确认的版本累计打包：对方确认在 A，本地到 B、C 时分别生成 A..B、A..C，只传最后一个包就包含尚未确认的全部提交。打包本身不推进已确认版本。没有新提交时不生成包，也不改上次打包记录。

```bash
make archive-pack FULL=1
```

全量 bundle 包含当前提交可达的历史，可以初始化一个没有 Git 仓库的接收端；它不代表强制覆盖分叉历史。增量包缺少前置提交时会报错，需要补传前置包或改传全量包。Git 历史中的旧 Hope 配置、已删除文件仍可能存在于全量包中，当前工作区只部署目标提交的文件。

默认输出位于 `transfer_snap/archive_MirrorOPD-verl_peer_*.bundle`；超过 10 MiB 时输出 `.bundle.part.000`、`.001` 等分卷。全部分卷放在同一目录，解包时指定 `.part.000`。`CHUNK_SIZE` 可调整分卷字节数。可通过 `ARCHIVE=自定义名称.bundle` 指定新包名；已有包名不能覆盖。`ARCHIVE_SOURCES` 不再支持，bundle 传输完整的提交范围。

成功打包后，只清理同目录、同 `TARGET` 的旧自动命名 bundle 和分卷；自定义名称、其他接收端的包、旧 tar 包、日志和结果包均保留。成功接收后删除本次输入包及其分卷，设置 `KEEP_ARCHIVE_AFTER_UNPACK=1` 可保留。接收失败保留输入文件。未指定 `ARCHIVE` 时，依次在 `~/Downloads`、当前目录和 `transfer_snap/` 查找最新 bundle；有多个来源时应显式指定文件。

### 两端版本记录

成功 `unpack` 后，接收端记录发送方至少拥有包中的目标提交，可直接作为下一次反向增量的基线。`ack` 则让原发送端知道接收端已经收到。读取旧包或重复确认不会将已知版本倒退。状态保存在本地 Git 的 `refs/archive/peers/<TARGET>/packed` 和 `confirmed` 引用中，不进入源码提交，也不随 bundle 覆盖另一台机器的状态。

默认 `TARGET=peer` 适用于固定的两台机器。如果一个 checkout 向多台机器传包，用不同名称分别记录：

```bash
# Mac 给开发机打包、确认。
make archive-pack TARGET=dev
make archive-ack TARGET=dev

# 开发机收到 Mac 的包；这里 TARGET 表示发送方在开发机上的本地名称。
make archive-unpack TARGET=mac ARCHIVE=/实际上传目录/代码包.bundle
```

接收端只允许快进当前分支，不自动合并分叉，也不覆盖未提交的源码修改或新跟踪路径处已有的本地文件。收到旧版本时保留较新的当前提交。已有 checkout 的分支名称不变；首次初始化创建 `main`。子模块只传输提交引用，不联网更新子模块：`recipe/` 若需要使用，应另行交付其仓库和工作区。

### 首次安装与旧 tar 工作区迁移

全量打包同时生成独立脚本 `transfer_snap/archive_bootstrap.py`。接收端还没有新版命令时，人工上传该脚本和全量 bundle（或全部分卷），直接运行：

```bash
python3 /实际上传目录/archive_bootstrap.py unpack \
  /实际上传目录/archive_MirrorOPD-verl_peer_....bundle \
  /目标项目绝对路径
```

脚本不依赖仓库中的其他 Python 模块。没有 `.git` 的目标目录会初始化为正常 Git 工作区；旧源码里被覆盖的同名文件会备份到项目同级 `.项目名-before-bundle-时间戳/`。本地 `hope/`、虚拟环境、日志、结果及其他未被目标提交占用的路径保留。旧源码中不再被新提交跟踪的额外文件也保留，可检查备份和 `git status` 后人工整理。新安装不自动配置 GitHub remote、作者或邮箱；需要在开发机提交时，按该机器的用户身份设置 Git。

首次迁移必须使用这个新版脚本，不能用旧 tar 解包器或 `tar -xzf` 处理 bundle。迁移成功后使用常规的 `make archive-unpack`，并在发送端执行 `make archive-ack`。若目标本身是已有 Git 工作区，新脚本按正常快进流程处理。

`hope/` 已加入 `.gitignore` 并取消跟踪，实际运行配置由每台机器维护。旧 Git 工作区收到取消跟踪的提交时，解包工具先将当地 Hope 配置临时保存到 `.git/archive-hope-backup`，再完成 Git 更新并恢复这些文件；不会用发送端的配置替换接收端配置。Hope 下存在已暂存修改时，先提交或取消暂存。若迁移进程被强制终止并遗留备份，脚本会停止后续接收，要求先恢复该备份。使用普通 `git pull` 或手动 checkout 跨越这个删除提交时不会获得脚本的保护，应先自行备份。

旧 `.tar.gz` 包如仍需使用，可调用保留的旧入口；此操作不更新 Git 提交或 bundle 确认记录，已有 Git 工作区中会表现为工作区修改：

```bash
python3 meituan/transfer.py code unpack /实际上传目录/旧包.tar.gz /目标项目目录
```

W&B 和结果归档继续使用下文原有命令，不参与 Git bundle 同步。

## 原生 Python 与离线依赖

仓库包含 `packages/mirroropd-native-deps-20261020.zip`，内容是 PyPI 官方发布的 TransferQueue 0.1.11、liger-kernel 0.8.4 wheel、来源说明，以及环境检查脚本。两个 wheel 都是 `py3-none-any`，无需编译 CUDA 扩展。[^transferqueue] [^liger]

将源码包上传并解压到开发机项目目录后执行：

```bash
make native-install
make native-check
```

`native-install` 使用 `/bin/python3` 创建 `.venv-native`，继承预装的 GPU 依赖；将 ZIP 解压到被 Git 忽略的 `offline-wheels/`；按固定版本离线安装本项目源码及两个缺失包。它不会联网解析依赖。pytest 等开发机额外工具需要预先安装；这两个 wheel 不构成完整 GPU 环境。

`native-check` 检查解释器、源码路径、包版本、vLLM 0.25.1 的 `_C_stable_libtorch` 扩展和其他导入，并报告 GPU 是否可见。它不启动训练，GPU kernel 和完整训练还需单独验收。

日常运行保持同一个解释器：

```bash
source .venv-native/bin/activate
export VERL_USE_UV=0
python -m verl.trainer.main_ppo --help
```

本项目数学配方已默认选择 `<仓库绝对路径>/.venv-native/bin/python`，并将同一个路径传给 Ray worker。可先运行 `DRY_RUN=1 bash examples/mirroropd/run_math_deepseek_r1_1_5b_fsdp2.sh` 检查命令，再用 `CONFIG_ONLY=1` 解析配置。需要其他环境时显式设置 `PYTHON_BIN=/绝对路径/bin/python`；不要把它指向只包含系统预装包的 `/bin/python3`。

## 提交 8 卡任务

开发机安装、数据检查完成，且本地已有 `hope/mirroropd/mirroropd.hope` 后，在仓库根目录执行 `make opd-submit`。它提交接收端保留的 Hope 配置，默认由集群 worker 使用现有 `.venv-native` 执行单机共享 8 卡训练：student rollout、冻结 teacher 评分、actor 更新顺序共池执行，student 推理 TP=1。旧 4+4 可通过 `DISTILLATION_EXECUTION_MODE=standalone` 选择。`hope/` 不随源码同步，调度配置由开发机本地维护；worker 直接执行随源码更新的领域脚本。旧入口的一次性迁移见[Hope 直连入口](../examples/mirroropd/docs/infra/2026-10-22-共享八卡运行与架构.md#hope-直连入口)。

## 模型与数据布局

模型和数据统一放在源码仓库相邻的 `model/`（单数）与 `data/`，供开发机和训练任务共享。开发机基目录为 `/home/hadoop-hmart-poistar/dolphinfs_hdd_hadoop-hmart-poistar/fanghaotian/`。本机原有 `models/`、`MD-OPD/data/` 保留为导入来源；新默认目录不再依赖相邻 MD-OPD checkout。

```text
fanghaotian/
├── MirrorOPD-verl/
│   ├── .venv-native/
│   ├── outputs/
│   └── wandb/
├── model/
│   ├── DeepSeek-R1-Distill-Qwen-1.5B/
│   └── JustRL-DeepSeek-1.5B/
└── data/
    ├── math_manifest.json
    └── math/
        ├── DAPO/train.parquet
        ├── MATH-500/test.parquet
        ├── AMC23/test.parquet
        └── AIME25/test.parquet
```

当前两个模型使用 Hugging Face 格式目录，包含配置、tokenizer 与 `model.safetensors`；`generation_config.json` 可选，chat template 可以包含在 tokenizer 配置中或单独保存。`examples/mirroropd/prepare_models.py` 可复制已有目录并物化 HF 缓存符号链接，或在可联网机器显式下载固定版本。数据使用已预处理的 parquet；`prepare_math_data.py` 支持从旧 parquet 导入，或从原始文件重建。

`MODEL_ROOT`、`DATA_ROOT` 只指定目录；student、teacher 和四个 parquet 文件名由配方拼接。默认会自动定位仓库相邻的 `model/`、`data/`。需要不同文件时分别覆盖 `STUDENT_MODEL`、`TEACHER_MODEL`、`TRAIN_FILE`、`MATH_FILE`、`AMC_FILE`、`AIME_FILE`。

模型和数据准备后执行 `make assets-check`；模型仅检查必要文件，重点检查四份数据的字段、行数、空答案、重复题、题面重叠和 prompt 长度。只查数据可用 `make assets-check ASSET_CHECK_ARGS=--data-only`，会跳过模型与 token 长度计算。检查报告写到 `data/asset_check.json`。准备方式及检查范围见 [数学资产说明](../examples/mirroropd/docs/资产准备.md)。这些检查在 CPU 完成，不加载模型 tensor，不启动 GPU 训练。

`make archive-pack` 只负责源码；上述模型和数据需要另行准备并人工传输一次。已有文件无需随每次代码同步重传。

## 日志下载

在开发机执行：

```bash
make sync-log-pack
```

日志新包完整生成后，自动清理同目录中 `log_MirrorOPD-verl*.tar.gz` 及其旧分卷；打包失败或没有新增、修改文件时保留旧包。自定义名称的历史包不参与按前缀清理。日志仍按增量打包，因此每个包应在下一次打包前传回接收端。原始 `wandb/` 日志保留。

它扫描 `wandb/offline-run-*`，对有新增或修改文件的 run 打包完整目录，以便接收端合并。`.wandb` 旁文件一并纳入。人工下载生成的 `transfer_snap/log_MirrorOPD-verl_*.tar.gz` 或全部分卷后，在本机或 Mac 执行：

```bash
make sync-log-unpack LOG_ARCHIVE=/实际下载目录/log_MirrorOPD-verl_....tar.gz
```

日志接收使用独立的 `WncFht/wandb-logs` Git 仓库。Mac 上的仓库位于 `~/src/wandb-logs`；本机可将它 clone 到源码仓库同级。默认接收目录按以下顺序确定：

1. 显式设置的 `WANDB_RUNS_DIR`，直接作为实际 run 目录。
2. 显式设置的 `WANDB_LOGS_REPO` 下的 `MirrorOPD-verl/wandb/`。
3. 如果源码仓库同级存在 `wandb-logs/.git`，使用该仓库的 `MirrorOPD-verl/wandb/`。
4. 否则使用 `~/src/wandb-logs/MirrorOPD-verl/wandb/`。

解包默认依次执行：合并到日志仓库、上传 W&B、Git commit、Git push。接收端需提前安装并登录 `wandb`，日志仓库需配置 Git 提交身份、远端及当前分支的 upstream。每个 run 先复制到临时目录，再执行 `wandb sync`；同步产生的 config、summary 改写和同步状态留在临时目录，日志仓库保留收到的原始文件。临时副本会移除 `.wandb.synced` 和 `.wandb.syncstate`，以便重新上传增量包中的 run 快照；这可能重传已有记录，run ID 保持不变。

Mac 接收并推送日志：

```bash
make sync-log-unpack LOG_ARCHIVE=/实际下载目录/log_MirrorOPD-verl_....tar.gz
```

W&B 上传只处理本次包内的 run；Git 自动提交覆盖整个接收目录，默认是日志仓库中的 `MirrorOPD-verl/wandb/`。这个目录里以前漏提交的新增文件、修改和删除都会一起提交；`.gitignore` 排除的未跟踪文件仍不提交，其他项目或目录的暂存修改保留。没有文件变化时跳过 commit，但仍执行 push，以支持上次提交成功、推送失败后的重试。脚本使用普通 `git push`，不自动拉取、变基或强制推送；遇到远端分歧时先处理 Git 分歧，再重跑。W&B 上传失败时停止后续提交和推送；Git 推送失败时保留本地提交。两种失败都保留已解包日志及输入包。

`WANDB_SYNC=0` 只关闭 W&B 上传；`WANDB_GIT_SYNC=0` 关闭自动 Git 提交和推送。仅解包时同时设置两项：

```bash
make sync-log-unpack LOG_ARCHIVE=/实际下载目录/log_MirrorOPD-verl_....tar.gz \
  WANDB_SYNC=0 WANDB_GIT_SYNC=0
```

直接调用 `meituan/sync-log.sh unpack` 的默认行为相同，对应覆盖参数为 `MIRROROPD_WANDB_DIR`、`MIRROROPD_WANDB_LOGS_REPO`、`MIRROROPD_WANDB_SYNC` 和 `MIRROROPD_WANDB_GIT_SYNC`。上传过程要求整个 run 的附属文件可读；未打包且只存在于开发机的 artifact 外部引用无法通过临时复制恢复。

Mac 和本机的接收、分析流程直接使用独立日志仓库，不在源码仓库维护 `wandb` 目录或兼容软链接。Mac 如果执行过旧的软链接迁移命令，可以在源码仓库执行 `test ! -L wandb || unlink wandb`，只移除链接，目标日志不受影响。真实目录中的旧 run 应先迁入日志仓库，再移除空目录。已启动的旧 `wandb sync` 进程仍可能写原目录，应等其退出后再迁移；上传临时副本不会隔离其他进程对原目录的写入。

本地解码工具位于 `wandb-logs/tools/wandb_local.py`，不依赖训练仓库或 W&B 云端。首次在日志仓库执行 `make setup`，之后用 `make refresh` 拉取 Git 更新并解码；只分析现有文件用 `make decode`，按 run 导出 CSV 用 `make csv RUN=<run-id>`。原始日志位于 `MirrorOPD-verl/wandb/`，生成的 history、系统监控、配置和 summary 位于被 Git 忽略的 `derived/MirrorOPD-verl/<run目录>/`。上传 W&B 和本地解码是独立路径；本地解码不调用 login、sync 或 Public API。

MirrorOPD 的统计和绘图使用 `examples/mirroropd/docs/scripts/analyze_run.py export` 从本地日志准备项目快照；它委托独立仓库解码，不另维护解析器。旧云端回读改为显式 `export-cloud`，仅用于核对或补救。具体使用步骤和历史数据边界见[分析复现](../examples/mirroropd/docs/分析复现.md)。

打包端仍默认扫描源码仓库的 `wandb/`；`WANDB_LOGS_REPO` 只影响接收端。显式设置 `WANDB_RUNS_DIR=自定义目录` 可以覆盖打包源或接收位置。这个参数表示实际 run 目录；SDK 的 `WANDB_DIR` 表示其父级写入根目录，两个参数分开设置。

本项目的数学配方默认启用 `console` 和 `wandb`，设置 `WANDB_MODE=offline`、`WANDB_DIR=<仓库根目录>`，并将这些环境变量显式传给 Ray 训练进程。W&B 实际记录位于 `<仓库>/wandb/offline-run-*/`，与同步脚本的默认扫描目录一致。运行输出和 checkpoint 仍保存在 `outputs/<实验名>/`。

完整终端 stdout/stderr 同时保存到 `outputs/<实验名>/train.log`，用 `tail -f outputs/<实验名>/train.log` 查看。这个文本日志通过 `make sync-results-pack` 随实验结果打包；`make sync-log-pack` 负责 W&B 离线 run。

## 结果下载

在开发机执行：

```bash
make sync-results-pack RESULTS_REMOTE_DIR=outputs
```

默认扫描 `outputs/`，跳过 `checkpoints/` 子目录及大于 100 MiB 的单个文件，只打包新增或修改的结果。文件变化按大小与纳秒修改时间判断。人工下载包或全部分卷后，在本机或 Mac 执行：

```bash
make sync-results-unpack RESULTS_ARCHIVE=/实际下载目录/MirrorOPD_results_....tar.gz
```

默认合并到 `results/`；用 `RESULTS_LOCAL_DIR=自定义目录` 指定接收位置。日志和结果同步都不传播删除操作；同名文件由收到的包覆盖，因此按生成顺序解包。

增量状态保存在打包端 `transfer_snap/logs_state.json` 和 `results_state.json`，只在成功生成包后推进。需要重发时可再次传输保留的包；需要重新全量打包时删除相应状态文件。

## 参数与入口

| Make 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `PYTHON` | `python3` | 运行传输工具与 ZIP 解包 |
| `SYSTEM_PYTHON` | `/bin/python3` | 创建开发机原生虚拟环境 |
| `SNAP_DIR` | `transfer_snap` | 压缩包与增量状态目录 |
| `CHUNK_SIZE` | `10485760` | 分卷大小，单位为字节 |
| `MAX_FILE_MB` | `100` | 结果文件大小上限，单位为 MiB |
| `WANDB_RUNS_DIR` | 留空，按操作选择 | 覆盖实际 run 目录；打包默认 `wandb`，解包默认独立日志仓库 |
| `WANDB_LOGS_REPO` | 同级 Git 仓库 `../wandb-logs`，不存在则 `~/src/wandb-logs` | 接收端日志仓库，run 合并到其中的 `MirrorOPD-verl/wandb/` |
| `WANDB_SYNC` | `1` | 解包后从临时副本上传 W&B；设为 `0` 跳过上传 |
| `WANDB_GIT_SYNC` | `1` | 上传完成后提交并推送该项目接收目录的全部日志变化；设为 `0` 跳过 Git 操作 |
| `KEEP_ARCHIVE_AFTER_UNPACK` | `0` | 设为 `1` 保留刚解包的源码输入包；更旧源码包仍清理 |
| `MODEL_ROOT`、`DATA_ROOT` | `../model`、`../data` 的绝对路径 | 数学资产检查目录 |
| `LOG_ACTION`、`ACTION` | `unpack` | `make sync-log`、`make sync-results` 的操作 |

三个 shell 脚本保留 `pack`、`unpack` 命令行入口，调用同一套 `transfer.py` 逻辑。直接调用 shell 时用 `MIRROROPD_*` 环境变量设置参数；通常使用 Makefile 即可。

## 验证

源码包内容可用 `tar -tzf <源码包>` 查看，应包含 `Makefile`、`AGENTS.md`、`meituan/`、`verl/` 等源码，包含离线 ZIP，排除虚拟环境、模型和训练输出。分卷需要全部上传后再解包。

开发机先运行 `make native-check`，确认本仓库源码路径、两个补充依赖和 vLLM 扩展导入正确；运行 `make assets-check` 检查模型及四份数据。再单独测试 GPU 张量运算；上述检查不替代完整训练验收。

不加载模型即可验证离线 W&B 是否按约定落盘：

```bash
make wandb-check
make sync-log-pack
```

预期产生 `wandb/offline-run-*/run-*.wandb` 和日志包。人工传回 Mac 后运行 `make sync-log-unpack LOG_ARCHIVE=<下载包>`，确认 `~/src/wandb-logs/MirrorOPD-verl/wandb/` 出现对应 run，W&B 上传和 Git 推送均成功。显式指定了接收路径或存在同级日志仓库时，检查实际接收目录。仅验证本地解包时添加 `WANDB_SYNC=0 WANDB_GIT_SYNC=0`，无需安装、登录 W&B。

### 参考文献

[^transferqueue]: TransferQueue contributors. TransferQueue 0.1.11 release files. PyPI, 2026. [发布页](https://pypi.org/project/TransferQueue/0.1.11/)。wheel 来源保存在 ZIP 内的 `sources.json`。
[^liger]: Liger Kernel contributors. liger-kernel 0.8.4 release files. PyPI, 2026. [发布页](https://pypi.org/project/liger-kernel/0.8.4/)。wheel 来源保存在 ZIP 内的 `sources.json`。
