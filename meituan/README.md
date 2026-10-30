# 美团开发机安装与人工同步

本目录参考 MD-OPD 的 `archive.sh`、`sync-log.sh`、`sync_results_from_remote.sh` 和 Makefile，提供本项目的人工压缩包传输入口。开发机与本机、Mac 之间通过上传和下载文件交换代码与结果。脚本负责打包、分卷、解包，不建立 SSH 或 rsync 连接。

三个 checkout 的固定路径记录在根目录 [AGENTS.md](../AGENTS.md)。脚本从自身位置确定仓库根目录，因此可直接在三处使用。传输工具使用 Bash 和 Python 标准库，需要 Python 3.12 或更新版本；开发机的训练环境使用 Python 3.12。

## 源码双向同步

2026-10-27 已清理项目历史中的原始实验明细。旧 checkout 首次迁移须按[源码历史清理与迁移](docs/2026-10-27-源码历史清理与迁移.md)处理；不要将旧分支合并回清理后的分支。源码文件限额为 2 MiB，实验数据摘要限额为 1 MiB；原始日志和 rollout 明细只保留在本机。`archive-pack` 会检查整个可达历史，阻止这类内容重新进入 bundle。

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

成功打包后，只清理同目录、同 `TARGET`、生成时间更早的自动命名 bundle 和分卷，保留未来时间戳的包。成功接收后删除本次输入包及其分卷，并在输入目录、接收项目根目录和接收端 `SNAP_DIR` 中清理旧自动命名源码包。支持 `.tar.gz`、`.tgz`、`.zip` 及其数字分卷，兼容有无微秒的文件名时间戳。

旧源码包的文件名时间须早于收到的 bundle；自动命名 bundle 按生成时间判断，上传修改时间不影响清理。自定义 bundle 则要求旧包的文件名时间和修改时间都早于输入包修改时间。Git 跟踪文件、符号链接、自定义名称、同时间或更新的包、其他 bundle、日志和结果包保留。`KEEP_ARCHIVE_AFTER_UNPACK=1` 保留输入包及历史源码包；更新失败也全部保留。终端和诊断记录列出删除结果、保留原因及扫描目录。`archive_bootstrap.py` 是安装工具，保留供恢复使用。未指定 `ARCHIVE` 时，依次在 `~/Downloads`、当前目录和 `transfer_snap/` 查找最新 bundle；有多个来源时应显式指定文件。

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

独立 `archive_bootstrap.py` 必须同时指定包和目标目录；省略参数会在读取包、写文件或清理分卷前退出。相对目标目录按运行命令时的当前目录解释，通常建议使用绝对路径。仓库内的 `make archive-unpack` 仍默认更新当前项目。执行完应对照输出中的 `Unpacked into` 路径，并在该目录运行 `git --no-pager log -1 --oneline`，确认查看和运行的是实际更新的 checkout。

旧 bootstrap 若被复制到项目根目录，再以 `python3 archive_bootstrap.py unpack` 无参数执行，会把项目父目录当成默认目标。它可能在父目录安装了新 checkout，而原项目子目录仍停留在旧提交。遇到这种情况，先分别检查两个目录的 Git 根路径与 HEAD；若父目录确实包含所需新版本，可以用 `git clone --no-local --branch main /已确认的新版本目录 /新的项目目录` 在本机恢复一个独立 checkout，无需重新上传已经消费的分卷。旧项目、父目录的模型和环境文件先保留，恢复正确 checkout 后再迁移本地配置。

脚本不依赖仓库中的其他 Python 模块。没有 `.git` 的目标目录会初始化为正常 Git 工作区；旧源码里被覆盖的同名文件会备份到项目同级 `.项目名-before-bundle-时间戳/`。本地 `hope/`、虚拟环境、日志、结果及其他未被目标提交占用的路径保留。旧源码中不再被新提交跟踪的额外文件也保留，可检查备份和 `git status` 后人工整理。新安装不自动配置 GitHub remote、作者或邮箱；需要在开发机提交时，按该机器的用户身份设置 Git。

首次迁移必须使用这个新版脚本，不能用旧 tar 解包器或 `tar -xzf` 处理 bundle。迁移成功后使用常规的 `make archive-unpack`，并在发送端执行 `make archive-ack`。若目标本身是已有 Git 工作区，新脚本按正常快进流程处理。

升级解包器本身时，本次 `make archive-unpack` 运行的是启动时加载的旧代码，新清理规则通常要到下次解包才生效。如果需要升级当次就使用新规则，在已更新的发送端执行 `make archive-pack FULL=1`，同时传输新生成的 bootstrap，并按上面的 Python 命令接收。

`hope/` 已加入 `.gitignore` 并取消跟踪，实际运行配置由每台机器维护。旧 Git 工作区收到取消跟踪的提交时，解包工具先将当地 Hope 配置临时保存到 `.git/archive-hope-backup`，再完成 Git 更新并恢复这些文件；不会用发送端的配置替换接收端配置。Hope 下存在已暂存修改时，先提交或取消暂存。若迁移进程被强制终止并遗留备份，脚本会停止后续接收，要求先恢复该备份。使用普通 `git pull` 或手动 checkout 跨越这个删除提交时不会获得脚本的保护，应先自行备份。

旧 `.tar.gz` 包如仍需使用，可调用保留的旧入口；此操作不更新 Git 提交或 bundle 确认记录，已有 Git 工作区中会表现为工作区修改：

```bash
python3 meituan/transfer.py code unpack /实际上传目录/旧包.tar.gz /目标项目目录
```

W&B 和结果归档继续使用下文原有命令，不参与 Git bundle 同步。

## 状态与故障定位

在源码仓库查看本机状态：

```bash
make sync-status
make archive-status TARGET=peer
make sync-log-status
python3 meituan/sync_status.py --json
```

这些命令只读取本地文件和 Git 状态，不打包、不上传、不推送，也不联网查询远程。代码部分显示当前 HEAD、上次打包提交、已确认的对方提交、相对基线的 ahead/behind 和未提交文件。日志部分显示打包源、实际接收目录、run 数量、打包状态文件、日志仓库的本地提交状态，以及最近八次相关操作。没有记录的历史状态显示 unknown；本地已知的 peer/upstream 位置可能落后于远程实际状态。

`archive-pack`、`archive-unpack`、`archive-ack` 和日志、结果 pack/unpack 都会在启动时打印诊断日志路径。默认每次生成独立的 `transfer_snap/operations/<时间戳>_<操作>_<唯一编号>.log` 与同名 `.json`。文本日志保留命令输出和异常栈；JSON 报告记录入口脚本、机器、进程、Python 版本、输入输出路径、相关提交、各阶段状态及耗时。输入包的自动清理不会删除这些诊断记录；记录不自动过期，可在排障结束后手动归档。

接收源码到其他 `DEST` 时，日志默认放在接收目录的 `transfer_snap/operations/`；设置 `SNAP_DIR` 时使用对应目录。所有入口均可通过 `OPERATION_DIR=/绝对诊断目录` 覆盖，查看状态时使用同一参数。独立 `archive_bootstrap.py` 已内嵌日志模块，仍只需上传一个 Python 文件。

```bash
make sync-log-unpack LOG_ARCHIVE=/下载目录/日志包.tar.gz OPERATION_DIR=/tmp/mirroropd-sync
make sync-log-status OPERATION_DIR=/tmp/mirroropd-sync
# 实时查看启动消息中给出的具体日志文件。
tail -f /tmp/mirroropd-sync/具体操作.log
```

运行时显示阶段开始、成功或失败、分卷路径与大小、W&B run 名及进度、Git 提交和推送结果；长时间操作每 15 秒显示等待提示。未提交源码导致接收停止时，终端显示前 30 行文件状态，完整列表及恢复提示保存在日志。源码仍按原有快进规则处理，不因为增加诊断而自动覆盖或提交本地变化。

交互终端默认使用颜色：成功为绿色，错误为红色，不完整快照和警告为黄色。`COLOR=always` 强制启用，`COLOR=never` 或 `NO_COLOR=1` 关闭；管道输出默认不带颜色。实现只依赖标准库，独立 bootstrap 同样可用。文本日志与 JSON 报告不含颜色控制码；完整命令和详细上下文写入日志，终端保留阶段进度和结果。

日志接收报告记录 `logs.merge`、`logs.git_commit`、`logs.enqueue`、`logs.git_fetch`、`logs.git_merge` 和 `logs.git_push`；上传报告记录 `wandb.preflight` 与 `wandb.batch`，逐 run 结果持久保存在上传队列中。批次中部分 run 成功时，成功任务不会随失败任务一起重传。Git 推送和云端上传可以分别重试，重跑产生新报告，不覆盖旧记录。

接收默认 `WANDB_SYNC=0`，云端阶段显示 skipped/queued，不能解读为已经上传。整体 success 表示该次命令已正常处理可执行任务；训练中的快照仍可能等待更新，须结合队列汇总判断数据完整性。上传器联合 CLI 结果与 SDK 完成标记记录快照结果；状态命令不实时查询云端，也不充当日志传输回执；`logs_state.json` 仍然只表示已打包，不代表送达。原有“日志包必须在下一次打包前传回”的规则不变。

实际失败记录 failed，Ctrl-C 记录 interrupted。上传仅剩不完整快照等待新数据时，默认记录 success、退出码为 0，并显示黄色 WAITING 提示；队列仍保留 incomplete 状态。`STRICT=1` 时这种情况记录 partial、退出码为 2，Make 会显示 `Error 2`。真正的上传或 Git 故障仍返回失败；成功快照不会重传。机器掉电、进程被强制终止或磁盘无法继续写入时，报告可能停留在 running；状态页显示最后更新时间，不能仅凭 running 判断进程仍活着。日志不保存完整环境变量，并遮蔽已知密钥环境值、常见 token/password 字段和 URL 密码；诊断文件仍包含本机路径和工具输出，默认文件权限为仅当前用户可读写。

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

本项目数学配方已默认选择 `<仓库绝对路径>/.venv-native/bin/python`，并将同一个路径传给 Ray worker。可先运行 `DRY_RUN=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh custom` 检查命令，再用 `CONFIG_ONLY=1` 解析配置。需要其他环境时显式设置 `PYTHON_BIN=/绝对路径/bin/python`；不要把它指向只包含系统预装包的 `/bin/python3`。

## 提交 8 卡任务

开发机安装、数据检查完成，且本地已有 `hope/mirroropd/mirroropd.hope` 后，在仓库根目录执行 `make opd-submit`。它提交接收端保留的 Hope 配置，默认由集群 worker 使用现有 `.venv-native` 执行单机共享 8 卡训练：student rollout、冻结 teacher 评分、actor 更新顺序共池执行，student 推理 TP=1。旧 4+4 可通过 `DISTILLATION_EXECUTION_MODE=standalone` 选择。`hope/` 不随源码同步，调度配置由开发机本地维护；worker 直接执行随源码更新的领域脚本。旧入口的一次性迁移见[Hope 直连入口](../examples/mirroropd/docs/infra/training/2026-10-22-共享八卡运行与架构.md#hope-直连入口)。

## 模型与数据布局

模型和数据统一放在源码仓库相邻的 `model/`（单数）与 `data/`，供开发机和训练任务共享。开发机基目录为 `/home/hadoop-hmart-poistar/dolphinfs_hdd_hadoop-hmart-poistar/fanghaotian/`。本机模型已迁移到 `model/hub/` 和 `model/checkpoints/`，旧 `models/` 及旧模型位置不提供兼容链接。数据目录不随模型迁移改变。

```text
fanghaotian/
├── MirrorOPD-verl/
│   ├── .venv-native/
│   ├── outputs/
│   └── wandb/
├── model/
│   ├── hub/<组织--仓库>/<revision>/
│   ├── checkpoints/<实验相对路径>/
│   └── registry.json
└── data/
    ├── math_manifest.json
    └── math/
        ├── DAPO/train.parquet
        ├── MATH-500/test.parquet
        ├── AMC23/test.parquet
        └── AIME25/test.parquet
```

原始模型按 repo、不可变 revision 与可选 subfolder 保存为实体 Hugging Face 格式目录，包含配置、tokenizer 与权重；`generation_config.json` 可选，chat template 可以包含在 tokenizer 配置中或单独保存。`examples/mirroropd/assets/prepare_models.py` 可复制已有目录并物化 HF 缓存符号链接，或在可联网机器显式下载固定版本。数据使用已预处理的 parquet；`prepare_math_data.py` 支持从旧 parquet 导入，或从原始文件重建。

`MODEL_ROOT`、`DATA_ROOT` 指定资产根目录；student、teacher 通过统一注册表解析到 `model/hub/` 下的固定版本，四个 parquet 文件名按领域配置选择。默认会自动定位仓库相邻的 `model/`、`data/`；worktree 使用显式的 `MIRROROPD_WORKSPACE` 指向共享工作区。需要不同文件时分别覆盖 `STUDENT_MODEL`、`TEACHER_MODEL`、`TRAIN_FILE`、`MATH_FILE`、`AMC_FILE`、`AIME_FILE`。

模型和数据准备后执行 `make assets-check`；模型按统一注册表选择固定版本并检查必要文件，重点检查四份数据的字段、行数、空答案、重复题、题面重叠和 prompt 长度。只查数据可用 `make assets-check ASSET_CHECK_ARGS=--data-only`，会跳过模型与 token 长度计算。检查报告写到 `data/asset_check.json`。准备方式及检查范围见 [数学资产说明](../examples/mirroropd/docs/infra/training/资产准备.md)。这些检查在 CPU 完成，不加载模型 tensor，不启动 GPU 训练。

正式领域脚本将权重与完整恢复 bundle 默认写到 `model/checkpoints/production/<run-relative>/checkpoints`；`RUN_DIR` 继续保存日志、配置和验证输出。`CHECKPOINT_DIR` 可显式覆盖恢复目录，新运行拒绝覆盖非空的 checkpoint 目录；`DRY_RUN` 和 `CONFIG_ONLY` 不创建这些目录。

`make archive-pack` 只负责源码；上述模型和数据需要另行准备并人工传输一次。已有文件无需随每次代码同步重传。

## 日志接收与云端上传

开发机仍执行 `make sync-log-pack`，扫描源码仓库的 `wandb/offline-run-*`，将发生变化的 run 整个目录打包。超过 10 MiB 分卷，人工传回全部分卷。打包端仍按大小、修改时间记录 `logs_state.json`，成功后清理旧自动命名日志包；每个包必须在下一次打包前传回。这个打包状态不代表云端已经收到数据。

Mac 或分析机接收到独立的 `wandb-logs` Git 仓库：

```bash
make sync-log-unpack LOG_ARCHIVE=/下载目录/log_MirrorOPD-verl_....tar.gz
```

默认先合并原始文件、创建本地 Git 提交并登记持久上传队列，再获取远程 upstream、快进或合并、登记远程带回的 run，最后推送 Git。先保存本地提交和队列，因此远程故障不会丢失刚收到的日志。云端上传已从默认接收路径分离，W&B 故障不会挡住本地提交。Git push 失败时，本地提交和队列保留，执行 `make sync-log-push` 重试推送；也可以先运行云端上传命令。输入日志包始终保留。

`sync-log-upload` 默认自动完成 fetch → 快进或 merge → 登记新快照 → push → W&B 上传。其他机器新推送的 run 会自动入队，无需先手动 pull 或 enqueue。`sync-log-enqueue` 先提交本地日志再执行同一 Git 同步流程；`sync-log-push` 也执行获取、合并、入队和推送，但不上传 W&B。upload 和 push 不自动提交工作区修改；本地未提交日志需要先 enqueue 或 unpack。

获取和推送都使用当前分支配置的 upstream remote/branch；不会默认为 origin/main，也不会使用另一个 pushDefault 目标。历史分叉时允许无冲突的普通 merge，保留双方提交，不 rebase、不 force push。需要合并而工作区存在未提交或未跟踪文件时会停止 Git 同步，不自动 stash；其他目录的暂存修改不会被夹带进提交。已有 merge/rebase 等操作也须先处理，工具不会接管。

自动合并冲突时记录冲突文件并执行 merge --abort，恢复合并前的干净工作区，双方提交保留；终端提示可用于手动重做合并的提交 ID。同一个 `.wandb` 文件在两边分别修改时，需判断应保留哪份原始数据，不自动选 ours/theirs。若远程在 fetch 与 push 之间再次前进，推送可能仍被拒绝，重新运行命令即可再获取、合并；不会无限循环重试。

上传入口的 Git 获取、合并或推送失败不阻止已有队列上传，但命令最终仍报告 Git 失败。默认每次 fetch/push 时限为 120 秒，兼容参数 `GIT_PUSH_TIMEOUT` 同时控制两者。诊断记录保存获取目标、合并前后提交、冲突路径、Git 输出及恢复提示。

接收目录仍按以下优先级选择：显式 `WANDB_RUNS_DIR`；`WANDB_LOGS_REPO/MirrorOPD-verl/wandb/`；源码仓库同级的 `wandb-logs/MirrorOPD-verl/wandb/`；最后回退到 `~/src/wandb-logs/MirrorOPD-verl/wandb/`。接收端日志仓库需要配置提交身份及 Git upstream。自动提交只包含该项目的日志路径，其他目录的暂存修改保留。

### 首次安装上传环境

在实际上传机器上执行一次：

```bash
make sync-log-setup
```

命令使用 Python venv 和 pip 创建 `.venv-wandb-sync`，固定安装本机验证过的 `wandb==0.31.0`；需要联网，Python 至少 3.10。它不升级训练环境或独立日志仓库的解码环境。新版 SDK 已移除旧的内部 DataStore 接口，因此不要直接替换本地解码器所用的 SDK。

上传入口优先使用此独立环境，也可以通过 `WANDB_BIN=/绝对路径/wandb` 指定上传 CLI。启动时记录执行路径、版本和引擎；拒绝低于 0.31.0 的 SDK，要求 `wandb beta sync` 所需参数可用，不自动退回旧 Python 同步器。认证默认复用已有 `.netrc` 或 `WANDB_API_KEY`。`WANDB_NETRC=/绝对路径/netrc` 可显式指定凭据文件，并覆盖环境中的 API key；日志只记录凭据来源，不输出密钥。上传前检查与实际上传使用同一凭据和显式目标。默认目标为 `WANDB_ENTITY=wncfht WANDB_PROJECT=MirrorOPD-verl`；其他账户应明确设置这两个参数。

### 队列、重试和状态

```bash
make sync-log-upload                  # 处理整个接收历史中已入队的待传任务
make sync-log-status                  # 查看本地队列及已有核验结果，不联网
make sync-log-audit RUN=krkmq098       # 只读核对云端 history，包括重复、缺失与冲突
make sync-log-upload RUN=krkmq098     # 只处理指定 run
make sync-log-upload RUN=krkmq098 STRICT=1  # 要求所选快照全部同步完成
make sync-log-upload RETRY=1          # 排障后忽略冷却与尝试次数限制，仍只尝试一轮
make sync-log-enqueue                 # 将迁移前已收到的日志提交、登记入队并推送 Git
make sync-log-push                    # 获取、合并、入队并推送日志仓库，不上传 W&B
```

每次接收会检查该项目日志目录中所有已提交 offline run，因此此前收到但没有上传的 run 也能入队。队列以 run 路径、Git tree ID 和云端目标区分快照。Git 推送传递原始日志，`.git/mirroropd-upload/queue.json`、锁和 SDK 标记仍只属于本机；另一台机器的上传结果通过云端核验识别，不能从本地队列或 `.synced` 推断。

每个待处理任务先从已提交快照读取 run 身份、退出记录及 history。未完成快照直接等待，不联网；完整快照再在明确指定的云端项目中查询同 ID 的 run。检查使用独立上传环境的 Python，按稀疏字段组合分组读取云端历史，比较本地每个 `_step` 的实际字段和值，浮点容差为 `1e-8`；只忽略 SDK 运行耗时 `_runtime`。云端额外字段允许保留。还会核对服务端 history 行数与扫描结果；新版 SDK 的视图可能隐藏重复物理行，因此计数不一致也会阻止上传。只检查最后 step、summary、云端 finished 状态或文件大小都不足以确认覆盖。

| 证据                                                   | 行为与队列结果                                                                  |
| ------------------------------------------------------ | ------------------------------------------------------------------------------- |
| 云端覆盖本地全部 history，且没有重复或值冲突           | 不调用 sync，记录 `covered`                                                     |
| 云端已有同 ID，但计数不一致、重复、缺行或值冲突        | 记录 failed 并保留重复/缺失数量及冲突位置；阻止整份重传，`RETRY=1` 也不绕过检查 |
| 可访问的项目中不存在该 run，且本地有完整记录和退出记录 | 调用 core sync；SDK 完成标记与上传后云端 history 核验都通过，才记录 succeeded   |
| 本地没有退出记录，或尾部记录未写完                     | 记录 incomplete，等待更新的源快照，不启动云端上传                               |
| 认证、项目权限、DNS 或查询超时                         | 不把错误当作 run 不存在；停止当前队列轮次，未处理任务保留 pending               |

`covered` 只证明训练/验证指标已存在，不能补证早期上传的 system、console、附件完整性。它可以用于接管其他机器或旧版工具上传的历史。已经存在重复行的云端 run 不会被自动删除或重建；分析优先使用原始本地日志。已有部分云端历史的 run 需要独立修复，本入口暂不实现历史裁剪或增量补写。新的未完成训练可先在本地解码分析，完成后再上传；这是为避免新快照整份追加而采用的默认策略。

队列位于独立日志仓库的 `.git/mirroropd-upload/queue.json`；`cache/` 保存不可变 Git 快照的上传副本，`attempts/` 保存 SDK 尝试目录，`preflight-*.json` 保存云端比对结果。SDK 改写的 config、summary 和标记留在副本，原始日志不变。任务持有 Git ref 防止 GC 回收待传对象。不要删除队列或缓存来重试；删除本机状态不能撤回已上传的历史。

新快照会替代尚未开始或失败的旧任务；已有 succeeded/covered 快照在本机直接跳过。若需要核查云端后来是否被删除或改动，使用 `sync-log-audit`。audit 不上传、不修改任务结果、不 fetch/push，但会创建本地核验报告和需要的快照缓存。一次只应在一台机器上执行 upload；本机锁不能协调两台机器同时检查“run 不存在”后又同时创建它的竞争。

### 超时与重试

默认并发两个不同 run，每批总时限 900 秒，连续 120 秒没有新的 SDK 输出就中止；相同进度行反复打印不重置无进展计时。无新输出只是可观察的停滞信号，不等于能证明网络没有传输。确有长时间静默的上传可提高 `WANDB_UPLOAD_IDLE_TIMEOUT`。云端检查单独受 `WANDB_UPLOAD_CHECK_TIMEOUT=120` 约束。

```bash
make sync-log-upload RUN=7rwc1ioz WANDB_UPLOAD_JOBS=1
make sync-log-upload RUN=7rwc1ioz RETRY=1 WANDB_UPLOAD_IDLE_TIMEOUT=180
```

SDK 超时或失败后停止后续批次，避免把同一网络故障重复等待多个 900 秒。诊断保留最后输出、尝试目录和 SDK 日志路径；可识别时列出 DNS、TLS、代理、认证、限流或 timeout 类别，没有匹配到类别也不代表网络正常。排查先选一个 run、并发 1，对照 `wandb.preflight` 与 `wandb.batch` 的耗时。Git fetch/push 正常只能证明 Git 路径可用，不能证明 W&B API、文件流或对象上传路径正常。

失败任务默认最多三次尝试，并有从 60 秒开始的指数冷却。`RETRY=1` 忽略冷却和尝试次数限制，但仍先核验云端；中断后已经传完的 history 可以被接管，缺行的旧 run 会明确停止，不再盲目追加。上传器的本机 worker 锁由 SDK 子进程继承；正常中断或超时会终止进程组。

### 不完整快照与完成含义

日志尾部未写完、没有退出记录时，默认等待后续快照，期间仍可本地分析。退出码非零的 run 也可以是完整日志；日志传输完成与训练成功分开判断。不会裁剪原始日志、补造退出记录或对静态归档使用 `--live`。

只剩 incomplete 时命令显示 WAITING 并正常退出；`STRICT=1` 要求选中任务没有等待项。严格模式和 `RETRY=1` 都不会补回缺失的源数据。若 SDK 在上传前检查后仍报告 incomplete，保留原有完成标记检查与等待逻辑。SDK 退出码 0 不能单独证明成功，上传后还须核对云端行和值。

### 兼容参数和本地分析

`WANDB_SYNC=1` 可在接收、提交、入队和推送后立即处理队列，适合仍想一键执行的情况。只复制文件、不提交、不入队时，使用 `WANDB_SYNC=0 WANDB_GIT_SYNC=0`；随后可用 `make sync-log-enqueue` 登记。持久上传依赖已提交快照，因此不支持 `WANDB_SYNC=1 WANDB_GIT_SYNC=0`。

单独运行 `sync-log-enqueue` 或 `sync-log-upload` 时，`WANDB_GIT_SYNC=0` 关闭自动 fetch/merge/push：enqueue 仍需创建本地提交才能登记快照，upload 只处理已有队列，不发现远程的新日志。显式 `sync-log-push` 始终执行 Git 同步。

run ID 保持原始身份，entity/project 由 `WANDB_ENTITY`、`WANDB_PROJECT` 明确设置并在入队时固定。直接调用 Python 时使用对应的 `MIRROROPD_UPLOAD_ENTITY`、`MIRROROPD_UPLOAD_PROJECT`。旧队列空目标优先采用原 run 身份，缺失时采用当前显式设置，最终解析目标写入核验报告并传给 SDK；不会使用凭据账户的默认 entity。

本地解码工具仍位于独立仓库的 `wandb-logs/tools/wandb_local.py`。在该仓库使用 `make setup`、`make decode`、`make csv RUN=<run-id>`，无需等待云端上传。原始数据位于 `MirrorOPD-verl/wandb/`，派生数据位于 `derived/MirrorOPD-verl/`。MirrorOPD 的分析入口继续委托这个解码器，见[分析复现](../examples/mirroropd/docs/experiments/维护说明.md)。

终端 stdout/stderr 保存到 `outputs/<实验名>/train.log`，通过 `sync-results-pack` 传回；`sync-log-pack` 负责 W&B 离线 run。重复上传与 Mac 超时的证据、当前验证及限制见 [重复上传与超时分析](docs/2026-10-29-W&B重复上传与超时分析.md)；早期新 run 上传性能见 [2026-10-23 验证记录](docs/2026-10-24-W&B上传重构与本机验证.md)，其旧版本队列行为不作为当前重试依据。

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

| Make 参数                       | 默认值                                                     | 用途                                                                                                    |
| ------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `PYTHON`                        | `python3`                                                  | 运行传输工具与 ZIP 解包                                                                                 |
| `SYSTEM_PYTHON`                 | `/bin/python3`                                             | 创建开发机原生虚拟环境                                                                                  |
| `SNAP_DIR`                      | `transfer_snap`                                            | 压缩包与增量状态目录                                                                                    |
| `OPERATION_DIR`                 | `SNAP_DIR/operations`                                      | 每次操作的文本日志与 JSON 报告目录；状态命令使用相同参数                                                |
| `COLOR`                         | `auto`                                                     | 终端颜色：`auto`、`always`、`never`；`NO_COLOR` 优先关闭                                                |
| `CHUNK_SIZE`                    | `10485760`                                                 | 分卷大小，单位为字节                                                                                    |
| `MAX_FILE_MB`                   | `100`                                                      | 结果文件大小上限，单位为 MiB                                                                            |
| `WANDB_RUNS_DIR`                | 留空，按操作选择                                           | 覆盖实际 run 目录；打包默认 `wandb`，解包默认独立日志仓库                                               |
| `WANDB_LOGS_REPO`               | 同级 Git 仓库 `../wandb-logs`，不存在则 `~/src/wandb-logs` | 接收端日志仓库，run 合并到其中的 `MirrorOPD-verl/wandb/`                                                |
| `WANDB_SYNC`                    | `0`                                                        | 接收默认只入队；设为 `1` 在接收完成后立即处理上传队列                                                   |
| `WANDB_GIT_SYNC`                | `1`                                                        | 接收后提交、入队并推送日志；接收设为 `0` 只合并文件，独立 enqueue/upload 设为 `0` 关闭 fetch/merge/push |
| `GIT_PUSH_TIMEOUT`              | `120`                                                      | 日志仓库每次 Git fetch/push 时限，秒                                                                    |
| `WANDB_BIN`                     | 优先 `.venv-wandb-sync/bin/wandb`                          | 独立上传 CLI 路径                                                                                       |
| `WANDB_UPLOAD_JOBS`             | `2`                                                        | 同时上传的不同 run 数，范围 1–8                                                                         |
| `WANDB_UPLOAD_TIMEOUT`          | `900`                                                      | 每批上传总时限，秒                                                                                      |
| `WANDB_UPLOAD_IDLE_TIMEOUT`     | `120`                                                      | SDK 无新输出的最长等待，秒                                                                              |
| `WANDB_UPLOAD_CHECK_TIMEOUT`    | `120`                                                      | 每个快照云端核验总时限，秒                                                                              |
| `WANDB_ENTITY`、`WANDB_PROJECT` | `wncfht`、`MirrorOPD-verl`                                 | 入队时固定的显式云端目标                                                                                |
| `WANDB_NETRC`                   | 空                                                         | 指定 netrc 文件，覆盖环境 API key                                                                       |
| `RUN`、`RETRY`                  | 空、`0`                                                    | 上传 run 选择；显式覆盖冷却/尝试次数限制                                                                |
| `STRICT`                        | `0`                                                        | 设为 `1` 时，所选快照仍有 incomplete 则退出 2；独立上传及 `WANDB_SYNC=1` 接收均生效                     |
| `KEEP_ARCHIVE_AFTER_UNPACK`     | `0`                                                        | 设为 `1` 保留刚解包的源码输入包；日志和结果输入包始终保留                                               |
| `MODEL_ROOT`、`DATA_ROOT`       | `../model`、`../data` 的绝对路径                           | 数学资产检查目录                                                                                        |
| `LOG_ACTION`、`ACTION`          | `unpack`                                                   | `make sync-log`、`make sync-results` 的操作                                                             |

源码 shell 入口调用 `code_archive.py`；日志、结果入口调用 `transfer.py`。日志的提交与持久上传队列由 `log_sync.py` 维护；各入口共用 `sync_observe.py` 保存运行诊断，`sync_status.py` 读取状态。直接调用 shell 或独立 bootstrap 时用 `MIRROROPD_*` 环境变量设置参数，诊断目录对应 `MIRROROPD_OPERATION_DIR`；通常使用 Makefile 即可。

## 验证

完整源码 bundle 可用 `git bundle list-heads <源码包.bundle>` 查看目标提交；源码与依赖 ZIP 以该提交的 Git 跟踪内容为准。日志和结果 tar 包可用 `tar -tzf <日志或结果包>` 查看。分卷需要全部传输后再解包。

开发机先运行 `make native-check`，确认本仓库源码路径、两个补充依赖和 vLLM 扩展导入正确；运行 `make assets-check` 检查模型及四份数据。再单独测试 GPU 张量运算；上述检查不替代完整训练验收。

不加载模型即可验证离线 W&B 是否按约定落盘：

```bash
make wandb-check
make sync-log-pack
```

预期产生 `wandb/offline-run-*/run-*.wandb` 和日志包。人工传回 Mac 后运行 `make sync-log-unpack LOG_ARCHIVE=<下载包>`，确认 `~/src/wandb-logs/MirrorOPD-verl/wandb/` 出现对应 run，Git 提交、推送及队列登记成功；再执行 `make sync-log-upload` 检查云端结果。显式指定了接收路径或存在同级日志仓库时，检查实际接收目录。仅验证本地解包时添加 `WANDB_SYNC=0 WANDB_GIT_SYNC=0`，无需安装、登录 W&B。

### 参考文献

[^transferqueue]: TransferQueue contributors. TransferQueue 0.1.11 release files. PyPI, 2026. [发布页](https://pypi.org/project/TransferQueue/0.1.11/)。wheel 来源保存在 ZIP 内的 `sources.json`。

[^liger]: Liger Kernel contributors. liger-kernel 0.8.4 release files. PyPI, 2026. [发布页](https://pypi.org/project/liger-kernel/0.8.4/)。wheel 来源保存在 ZIP 内的 `sources.json`。
