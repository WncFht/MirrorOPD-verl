.DEFAULT_GOAL := help

PYTHON ?= python3
SYSTEM_PYTHON ?= /bin/python3
NATIVE_PYTHON := .venv-native/bin/python
NATIVE_PACKAGE := meituan/packages/mirroropd-native-deps-20261020.zip
ARCHIVE ?=
ARCHIVE_SOURCES ?=
DEST ?= .
ACTION ?= unpack
LOG_ACTION ?= $(ACTION)
LOG_ARCHIVE ?=
RESULTS_ARCHIVE ?=
WANDB_RUNS_DIR ?= wandb
WANDB_SYNC ?= 0
KEEP_ARCHIVE_AFTER_UNPACK ?= 0
RESULTS_REMOTE_DIR ?= outputs
RESULTS_LOCAL_DIR ?= results
SNAP_DIR ?= transfer_snap
CHUNK_SIZE ?= 10485760
MAX_FILE_MB ?= 100
MODEL_ROOT ?= $(abspath ../model)
DATA_ROOT ?= $(abspath ../data)
ASSET_CHECK_ARGS ?=

export MIRROROPD_PYTHON = $(PYTHON)
export MIRROROPD_SNAP_DIR = $(SNAP_DIR)
export MIRROROPD_CHUNK_SIZE = $(CHUNK_SIZE)
export MIRROROPD_WANDB_DIR = $(WANDB_RUNS_DIR)
export MIRROROPD_WANDB_SYNC = $(WANDB_SYNC)
export KEEP_ARCHIVE_AFTER_UNPACK
export MIRROROPD_RESULTS_SOURCE = $(RESULTS_REMOTE_DIR)
export MIRROROPD_RESULTS_DEST = $(RESULTS_LOCAL_DIR)
export MIRROROPD_MAX_FILE_MB = $(MAX_FILE_MB)

.PHONY: help archive archive-pack archive-unpack sync-log sync-log-pack sync-log-unpack
.PHONY: sync-results sync-results-pack sync-results-unpack native-env native-unpack native-install native-check wandb-check assets-check

help:
	@echo '源码上传：make archive-pack [ARCHIVE=...] [ARCHIVE_SOURCES="Makefile meituan verl"]'
	@echo '源码解包：make archive-unpack [ARCHIVE=压缩包或.part.000] [DEST=.]'
	@echo '日志下载：make sync-log-pack；传回后 make sync-log-unpack [LOG_ARCHIVE=...] [WANDB_SYNC=1]'
	@echo '结果下载：make sync-results-pack [RESULTS_REMOTE_DIR=outputs]；传回后 make sync-results-unpack [RESULTS_ARCHIVE=...]'
	@echo '离线环境：make native-install；make native-check（开发机，系统 Python 3.12）'
	@echo 'W&B 验证：make wandb-check（只写一条离线日志，不加载模型）'
	@echo '训练资产：make assets-check [MODEL_ROOT=../model] [DATA_ROOT=../data] [ASSET_CHECK_ARGS=--data-only]'
	@echo '压缩包默认放在 transfer_snap/，超过 10 MiB 分卷；上传下载由人工完成。'
	@echo '源码打包成功后清理旧源码包；解包成功后删除输入包，可设 KEEP_ARCHIVE_AFTER_UNPACK=1 保留。'

archive archive-pack:
	@bash meituan/archive.sh pack "$(ARCHIVE)" $(ARCHIVE_SOURCES)

archive-unpack:
	@bash meituan/archive.sh unpack "$(ARCHIVE)" "$(DEST)"

sync-log:
	@bash meituan/sync-log.sh "$(LOG_ACTION)" "$(LOG_ARCHIVE)"

sync-log-pack:
	@bash meituan/sync-log.sh pack "$(LOG_ARCHIVE)"

sync-log-unpack:
	@bash meituan/sync-log.sh unpack "$(LOG_ARCHIVE)"

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
	$(NATIVE_PYTHON) examples/mirroropd/check_assets.py --model-root "$(MODEL_ROOT)" --data-root "$(DATA_ROOT)" $(ASSET_CHECK_ARGS)
