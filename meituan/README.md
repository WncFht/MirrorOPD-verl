# 美团开发机安装与人工同步

本目录参考 MD-OPD 的 `archive.sh`、`sync-log.sh`、`sync_results_from_remote.sh` 和 Makefile，提供本项目的人工压缩包传输入口。开发机与本机、Mac 之间通过上传和下载文件交换代码与结果。脚本负责打包、分卷、解包，不建立 SSH 或 rsync 连接。

三个 checkout 的固定路径记录在根目录 [AGENTS.md](../AGENTS.md)。脚本从自身位置确定仓库根目录，因此可直接在三处使用。传输工具使用 Bash 和 Python 标准库，需要 Python 3.12 或更新版本；开发机的训练环境使用 Python 3.12。

## 源码上传

在本机或 Mac 的仓库根目录执行：

```bash
make archive-pack
```

输出位于 `transfer_snap/archive_MirrorOPD-verl_*.tar.gz`。超过 10 MiB 时只输出 `.tar.gz.part.000`、`.001` 等分卷。人工上传全部输出文件到开发机。第一次开发机尚未收到本目录时，先用 `tar -xzf <源码包> -C <项目目录>` 解包；分卷先用 `cat <完整包名>.part.* > <完整包名>` 合并。

收到 Makefile 和 `meituan/` 后，在开发机仓库根目录执行：

```bash
make archive-unpack ARCHIVE=/实际上传目录/archive_MirrorOPD-verl_....tar.gz
# 分卷：把全部分卷放在同一目录，只指定其中的 .part.000。
make archive-unpack ARCHIVE=/实际上传目录/archive_MirrorOPD-verl_....tar.gz.part.000
```

源码包使用当前工作区中的文件，包含尚未提交的修改和 `meituan/packages/` 下的离线 ZIP。有 Git 时读取已跟踪文件和未被忽略的新增文件；无 Git 时扫描源码目录。默认排除 `.git`、虚拟环境、缓存、`model/`、`models/`、`data/`、`outputs/`、`results/`、`wandb/`、checkpoint 和已有传输包。

源码解包沿用 MD-OPD 的替换语义：先完整解包到临时目录，再替换归档涉及的顶层文件或目录，例如 `verl/`、`examples/`、`meituan/`。这些目录中未进入新包的旧文件也会移除；运行目录不参与替换。解包前应保存接收端在这些源码目录里的修改。

可用 `ARCHIVE_SOURCES="Makefile meituan"` 选择仓库顶层路径，用 `ARCHIVE=自定义名称.tar.gz` 设置包名，用 `DEST=/目标项目目录` 设置源码解包位置。源码包完整生成后，自动清理同目录中 `archive_MirrorOPD-verl*.tar.gz` 及其旧分卷；固定同名包也可重新生成，旧文件在新包完整写入后替换。日志包、结果包和依赖 ZIP 不参与这项清理。

源码成功解包后，清理输入目录中更旧的源码包，并删除刚使用的包或全部分卷；比当前输入更新的源码包保留。解包失败时保留输入包。需要保留刚解包的输入文件时执行：

```bash
make archive-unpack ARCHIVE=/实际上传目录/archive_MirrorOPD-verl_....tar.gz KEEP_ARCHIVE_AFTER_UNPACK=1
```

未指定解包输入时，依次在 `~/Downloads`、当前目录、`transfer_snap/` 中查找匹配的最新包。上述清理仅针对源码 archive；日志和结果增量包仍保留，便于按顺序人工传输。

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

本项目数学配方已默认选择 `<仓库绝对路径>/.venv-native/bin/python`，并将同一个路径传给 Ray worker。可先运行 `DRY_RUN=1 bash examples/mirroropd/run_deepseek_r1_1_5b_fsdp2.sh` 检查命令，再用 `CONFIG_ONLY=1` 解析配置。需要其他环境时显式设置 `PYTHON_BIN=/绝对路径/bin/python`；不要把它指向只包含系统预装包的 `/bin/python3`。

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

模型和数据准备后执行 `make assets-check`；模型仅检查必要文件，重点检查四份数据的字段、行数、空答案、重复题、题面重叠和 prompt 长度。只查数据可用 `make assets-check ASSET_CHECK_ARGS=--data-only`，会跳过模型与 token 长度计算。检查报告写到 `data/asset_check.json`。准备方式及检查范围见 [数学资产说明](../examples/mirroropd/README.md)。这些检查在 CPU 完成，不加载模型 tensor，不启动 GPU 训练。

`make archive-pack` 只负责源码；上述模型和数据需要另行准备并人工传输一次。已有文件无需随每次代码同步重传。

## 日志下载

在开发机执行：

```bash
make sync-log-pack
```

它扫描 `wandb/offline-run-*`，对有新增或修改文件的 run 打包完整目录，以便接收端合并。`.wandb` 旁文件一并纳入。人工下载生成的 `transfer_snap/log_MirrorOPD-verl_*.tar.gz` 或全部分卷后，在本机或 Mac 执行：

```bash
make sync-log-unpack LOG_ARCHIVE=/实际下载目录/log_MirrorOPD-verl_....tar.gz
```

默认只合并到本地 `wandb/`。需要上传到 W&B 时，显式添加 `WANDB_SYNC=1`；接收端应已安装并登录 `wandb`。可用 `WANDB_RUNS_DIR=自定义目录` 改变打包源或接收位置。这个参数表示实际 run 目录；SDK 的 `WANDB_DIR` 表示其父级写入根目录，两个参数分开设置。

本项目的数学配方默认启用 `console` 和 `wandb`，设置 `WANDB_MODE=offline`、`WANDB_DIR=<仓库根目录>`，并将这些环境变量显式传给 Ray 训练进程。W&B 实际记录位于 `<仓库>/wandb/offline-run-*/`，与同步脚本的默认扫描目录一致。运行输出和 checkpoint 仍保存在 `outputs/<实验名>/`。

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
| `WANDB_RUNS_DIR` | `wandb` | 实际 W&B run 目录，用于日志打包和合并 |
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

预期产生 `wandb/offline-run-*/run-*.wandb` 和日志包。人工传回 Mac 后运行 `make sync-log-unpack LOG_ARCHIVE=<下载包>`，确认本地出现对应 run；仅在准备上传到 W&B 服务时添加 `WANDB_SYNC=1`。

### 参考文献

[^transferqueue]: TransferQueue contributors. TransferQueue 0.1.11 release files. PyPI, 2026. [发布页](https://pypi.org/project/TransferQueue/0.1.11/)。wheel 来源保存在 ZIP 内的 `sources.json`。
[^liger]: Liger Kernel contributors. liger-kernel 0.8.4 release files. PyPI, 2026. [发布页](https://pypi.org/project/liger-kernel/0.8.4/)。wheel 来源保存在 ZIP 内的 `sources.json`。
