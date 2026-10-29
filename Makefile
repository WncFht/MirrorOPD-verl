.DEFAULT_GOAL := help

PYTHON ?= python3
SYSTEM_PYTHON ?= /bin/python3
NATIVE_PYTHON := .venv-native/bin/python
NATIVE_PACKAGE := meituan/packages/mirroropd-native-deps-20261020.zip
ARCHIVE ?=
ARCHIVE_SOURCES ?=
FULL ?= 0
TARGET ?= peer
DEST ?= .
ACTION ?= unpack
LOG_ACTION ?= $(ACTION)
LOG_ARCHIVE ?=
RESULTS_ARCHIVE ?=
WANDB_RUNS_DIR ?=
WANDB_LOGS_REPO ?=
WANDB_SYNC ?= 0
WANDB_GIT_SYNC ?= 1
WANDB_BIN ?=
WANDB_UPLOAD_JOBS ?= 2
WANDB_UPLOAD_TIMEOUT ?= 900
WANDB_UPLOAD_IDLE_TIMEOUT ?= 120
WANDB_UPLOAD_CHECK_TIMEOUT ?= 120
WANDB_ENTITY ?= wncfht
WANDB_PROJECT ?= MirrorOPD-verl
WANDB_NETRC ?=
GIT_PUSH_TIMEOUT ?= 120
RUN ?=
RETRY ?= 0
STRICT ?= 0
KEEP_ARCHIVE_AFTER_UNPACK ?= 0
RESULTS_REMOTE_DIR ?= outputs
RESULTS_LOCAL_DIR ?= results
SNAP_DIR ?= transfer_snap
OPERATION_DIR ?=
COLOR ?= auto
CHUNK_SIZE ?= 10485760
MAX_FILE_MB ?= 100
MODEL_ROOT ?= $(abspath ../model)
DATA_ROOT ?= $(abspath ../data)
ASSET_CHECK_ARGS ?=
HOPE_DIR ?= hope/mirroropd
HOPE_FILE ?= mirroropd.hope

export MIRROROPD_PYTHON = $(PYTHON)
export MIRROROPD_SNAP_DIR = $(SNAP_DIR)
export MIRROROPD_OPERATION_DIR = $(OPERATION_DIR)
export MIRROROPD_COLOR = $(COLOR)
export MIRROROPD_CHUNK_SIZE = $(CHUNK_SIZE)
export MIRROROPD_ARCHIVE_FULL = $(FULL)
export MIRROROPD_ARCHIVE_TARGET = $(TARGET)
export MIRROROPD_WANDB_DIR = $(WANDB_RUNS_DIR)
export MIRROROPD_WANDB_LOGS_REPO = $(WANDB_LOGS_REPO)
export MIRROROPD_WANDB_SYNC = $(WANDB_SYNC)
export MIRROROPD_WANDB_GIT_SYNC = $(WANDB_GIT_SYNC)
export MIRROROPD_WANDB_BIN = $(WANDB_BIN)
export MIRROROPD_UPLOAD_JOBS = $(WANDB_UPLOAD_JOBS)
export MIRROROPD_UPLOAD_TIMEOUT = $(WANDB_UPLOAD_TIMEOUT)
export MIRROROPD_UPLOAD_IDLE_TIMEOUT = $(WANDB_UPLOAD_IDLE_TIMEOUT)
export MIRROROPD_UPLOAD_CHECK_TIMEOUT = $(WANDB_UPLOAD_CHECK_TIMEOUT)
export MIRROROPD_UPLOAD_ENTITY = $(WANDB_ENTITY)
export MIRROROPD_UPLOAD_PROJECT = $(WANDB_PROJECT)
export MIRROROPD_UPLOAD_NETRC = $(WANDB_NETRC)
export MIRROROPD_GIT_PUSH_TIMEOUT = $(GIT_PUSH_TIMEOUT)
export MIRROROPD_UPLOAD_RUN = $(RUN)
export MIRROROPD_UPLOAD_RETRY = $(RETRY)
export MIRROROPD_UPLOAD_STRICT = $(STRICT)
export KEEP_ARCHIVE_AFTER_UNPACK
export MIRROROPD_RESULTS_SOURCE = $(RESULTS_REMOTE_DIR)
export MIRROROPD_RESULTS_DEST = $(RESULTS_LOCAL_DIR)
export MIRROROPD_MAX_FILE_MB = $(MAX_FILE_MB)

.PHONY: help archive archive-pack archive-unpack archive-ack sync-log sync-log-pack sync-log-unpack
.PHONY: sync-results sync-results-pack sync-results-unpack native-env native-unpack native-install native-check wandb-check assets-check
.PHONY: opd-submit
.PHONY: docs-check docs-format docs-setup maintenance-check
.PHONY: sync-status archive-status sync-log-status
.PHONY: sync-log-upload sync-log-enqueue sync-log-push sync-log-setup sync-log-audit

help:
	@echo '文档维护：make docs-setup / docs-format / docs-check / maintenance-check（无需 GPU）'
	@echo '同步状态：make sync-status（只读）；archive-status / sync-log-status 分别查看代码、日志'
	@echo '诊断记录：默认 transfer_snap/operations/*.log 和 *.json；OPERATION_DIR 可指定目录'
	@echo '终端颜色：COLOR=auto|always|never；NO_COLOR 可关闭；日志与 JSON 始终为纯文本'
	@echo '源码打包：make archive-pack [FULL=1] [TARGET=peer] [ARCHIVE=...]（Git 提交，默认累计增量）'
	@echo '源码接收：make archive-unpack [ARCHIVE=包.bundle或.part.000] [DEST=.] [TARGET=peer]'
	@echo '同步确认：接收成功后，在发送端执行 make archive-ack [TARGET=peer]，确认上次打包的提交'
	@echo '日志接收：make sync-log-unpack [LOG_ARCHIVE=...]（提交、登记持久队列并推送 Git，默认不等待 W&B）'
	@echo '云端上传：make sync-log-upload [RUN=run-id] [RETRY=1]；首次 make sync-log-setup 安装独立上传环境'
	@echo '上传前核对云端 history；已有完整指标直接跳过，已有缺测/冲突阻止整份重传；中途快照等待结束记录'
	@echo '目标：WANDB_ENTITY=wncfht WANDB_PROJECT=MirrorOPD-verl；WANDB_NETRC 可显式覆盖环境 API key'
	@echo '超时：WANDB_UPLOAD_IDLE_TIMEOUT=120（无新输出）、WANDB_UPLOAD_TIMEOUT=900（总时限）'
	@echo '云端只读核验：make sync-log-audit [RUN=run-id]（不上传、不修改队列结果、不同步 Git）'
	@echo '训练中不完整快照默认等待更新并正常退出；STRICT=1 要求所选快照全部同步完成'
	@echo '日志 Git 自动同步：获取 upstream、快进或无冲突合并、新 run 入队、推送；WANDB_GIT_SYNC=0 可关闭'
	@echo '旧日志提交入队：make sync-log-enqueue；仅同步 Git 并入队：make sync-log-push [GIT_PUSH_TIMEOUT=120]'
	@echo '仅解包：make sync-log-unpack WANDB_SYNC=0 WANDB_GIT_SYNC=0（不提交、不入队）'
	@echo '结果下载：make sync-results-pack [RESULTS_REMOTE_DIR=outputs]；传回后 make sync-results-unpack [RESULTS_ARCHIVE=...]'
	@echo '离线环境：make native-install；make native-check（开发机，系统 Python 3.12）'
	@echo 'W&B 验证：make wandb-check（只写一条离线日志，不加载模型）'
	@echo '训练资产：make assets-check [MODEL_ROOT=../model] [DATA_ROOT=../data] [ASSET_CHECK_ARGS=--data-only]'
	@echo '8 卡训练：make opd-submit（开发机，提交 hope/mirroropd/mirroropd.hope）'
	@echo '压缩包默认放在 transfer_snap/，超过 10 MiB 分卷；上传下载由人工完成。'
	@echo '日志打包成功后清理旧日志包及分卷；增量包请在下次打包前传回。'
	@echo '源码打包成功后清理同 TARGET 的旧自动命名 bundle；解包后删除输入包，KEEP_ARCHIVE_AFTER_UNPACK=1 可保留。'

docs-check:
	@$(PYTHON) examples/mirroropd/tools/maintenance/pipeline.py check

docs-format:
	@$(PYTHON) examples/mirroropd/tools/maintenance/pipeline.py format

docs-setup:
	@npm --prefix examples/mirroropd/tools/maintenance ci --ignore-scripts
	@$(PYTHON) -m pip install -r examples/mirroropd/tools/maintenance/requirements.txt

maintenance-check: docs-check
	@$(PYTHON) examples/mirroropd/tools/maintenance/pipeline.py test

archive archive-pack:
	@bash meituan/archive.sh pack "$(ARCHIVE)" $(ARCHIVE_SOURCES)

archive-unpack:
	@bash meituan/archive.sh unpack "$(ARCHIVE)" "$(DEST)"

archive-ack:
	@bash meituan/archive.sh ack

sync-status:
	@$(PYTHON) meituan/sync_status.py

archive-status:
	@$(PYTHON) meituan/sync_status.py --section code

sync-log-status:
	@$(PYTHON) meituan/sync_status.py --section logs

sync-log:
	@bash meituan/sync-log.sh "$(LOG_ACTION)" "$(LOG_ARCHIVE)"

sync-log-pack:
	@bash meituan/sync-log.sh pack "$(LOG_ARCHIVE)"

sync-log-unpack:
	@bash meituan/sync-log.sh unpack "$(LOG_ARCHIVE)"

sync-log-upload:
	@$(PYTHON) meituan/log_sync.py upload

sync-log-audit:
	@$(PYTHON) meituan/log_sync.py audit

sync-log-enqueue:
	@$(PYTHON) meituan/log_sync.py enqueue

sync-log-push:
	@$(PYTHON) meituan/log_sync.py push

sync-log-setup:
	$(PYTHON) -m venv .venv-wandb-sync
	.venv-wandb-sync/bin/python -m pip install -r meituan/requirements-wandb-sync.txt

sync-results:
	@bash meituan/sync_results_from_remote.sh "$(ACTION)" "$(RESULTS_ARCHIVE)"

sync-results-pack:
	@bash meituan/sync_results_from_remote.sh pack "$(RESULTS_ARCHIVE)"

sync-results-unpack:
	@bash meituan/sync_results_from_remote.sh unpack "$(RESULTS_ARCHIVE)"

native-env:
	$(SYSTEM_PYTHON) -m venv --system-site-packages --without-pip .venv-native

native-unpack:
	$(PYTHON) -m zipfile -e "$(NATIVE_PACKAGE)" .

native-install: native-env native-unpack
	$(NATIVE_PYTHON) -m pip install --no-index --no-deps --no-build-isolation -e .
	$(NATIVE_PYTHON) -m pip install --no-index --no-deps --find-links=offline-wheels -r meituan/requirements-native.txt

native-check: native-unpack
	$(NATIVE_PYTHON) offline-wheels/check_native_env.py

wandb-check:
	$(NATIVE_PYTHON) meituan/check_wandb.py

assets-check:
	$(NATIVE_PYTHON) examples/mirroropd/assets/check_assets.py --model-root "$(MODEL_ROOT)" --data-root "$(DATA_ROOT)" $(ASSET_CHECK_ARGS)

opd-submit:
	@test -f "$(HOPE_DIR)/$(HOPE_FILE)" || { echo "Hope file not found: $(HOPE_DIR)/$(HOPE_FILE)" >&2; exit 1; }
	@command -v hope >/dev/null 2>&1 || { echo "hope CLI not found; submit from the developer machine" >&2; exit 1; }
	@cd "$(HOPE_DIR)" && hope run "$(HOPE_FILE)"
