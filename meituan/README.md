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

源码包使用当前工作区中的文件，包含尚未提交的修改和 `meituan/packages/` 下的离线 ZIP。有 Git 时读取已跟踪文件和未被忽略的新增文件；无 Git 时扫描源码目录。默认排除 `.git`、虚拟环境、缓存、`models/`、`data/`、`outputs/`、`results/`、`wandb/`、checkpoint 和已有传输包。

源码解包沿用 MD-OPD 的替换语义：先完整解包到临时目录，再替换归档涉及的顶层文件或目录，例如 `verl/`、`examples/`、`meituan/`。这些目录中未进入新包的旧文件也会移除；运行目录不参与替换。解包前应保存接收端在这些源码目录里的修改。

可用 `ARCHIVE_SOURCES="Makefile meituan"` 选择仓库顶层路径，用 `ARCHIVE=自定义名称.tar.gz` 设置包名，用 `DEST=/目标项目目录` 设置源码解包位置。已有同名包会报错；打包和解包均保留源包，便于重复人工传输。未指定解包输入时，依次在 `~/Downloads`、当前目录、`transfer_snap/` 中查找匹配的最新包。

## 原生 Python 与离线依赖

仓库包含 `packages/mirroropd-native-deps-20261020.zip`，内容是 PyPI 官方发布的 TransferQueue 0.1.11、liger-kernel 0.8.4 wheel、来源与 SHA-256 清单，以及环境检查脚本。两个 wheel 都是 `py3-none-any`，无需编译 CUDA 扩展。[^transferqueue] [^liger]

将源码包上传并解压到开发机项目目录后执行：

```bash
make native-install
make native-check
```

`native-install` 使用 `/bin/python3` 创建 `.venv-native`，继承预装的 GPU 依赖；将 ZIP 解压到被 Git 忽略的 `offline-wheels/`；离线安装本项目源码及两个缺失包，并通过 pip 校验固定哈希。它不会联网解析依赖。pytest 等开发机额外工具需要预先安装；这两个 wheel 不构成完整 GPU 环境。

`native-check` 检查解释器、源码路径、包版本、vLLM 0.25.1 的 `_C_stable_libtorch` 扩展和其他导入，并报告 GPU 是否可见。它不启动训练，GPU kernel 和完整训练还需单独验收。

日常运行保持同一个解释器：

```bash
source .venv-native/bin/activate
export VERL_USE_UV=0
python -m verl.trainer.main_ppo --help
```

## 日志下载

在开发机执行：

```bash
make sync-log-pack
```

它扫描 `wandb/offline-run-*`，对有新增或修改文件的 run 打包完整目录，以便接收端合并。`.wandb` 旁文件一并纳入。人工下载生成的 `transfer_snap/log_MirrorOPD-verl_*.tar.gz` 或全部分卷后，在本机或 Mac 执行：

```bash
make sync-log-unpack LOG_ARCHIVE=/实际下载目录/log_MirrorOPD-verl_....tar.gz
```

默认只合并到本地 `wandb/`。需要上传到 W&B 时，显式添加 `WANDB_SYNC=1`；接收端应已安装并登录 `wandb`。可用 `WANDB_DIR=自定义目录` 改变打包源或接收位置。

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
| `LOG_ACTION`、`ACTION` | `unpack` | `make sync-log`、`make sync-results` 的操作 |

三个 shell 脚本保留 `pack`、`unpack` 命令行入口，调用同一套 `transfer.py` 逻辑。直接调用 shell 时用 `MIRROROPD_*` 环境变量设置参数；通常使用 Makefile 即可。

### 参考文献

[^transferqueue]: TransferQueue contributors. TransferQueue 0.1.11 release files. PyPI, 2026. [发布页](https://pypi.org/project/TransferQueue/0.1.11/)。wheel 来源与哈希保存在 ZIP 内的 `sources.json`、`SHA256SUMS`。
[^liger]: Liger Kernel contributors. liger-kernel 0.8.4 release files. PyPI, 2026. [发布页](https://pypi.org/project/liger-kernel/0.8.4/)。wheel 来源与哈希保存在 ZIP 内的 `sources.json`、`SHA256SUMS`。
