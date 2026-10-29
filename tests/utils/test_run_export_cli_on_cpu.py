# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Prevent implicit historical run selection without making network requests."""

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "examples/mirroropd/research/studies/pipeline/analyze_run.py"


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location("run_export_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_requires_run_before_any_side_effect(cli, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "export", "--out", str(tmp_path / "snapshot")])
    monkeypatch.setattr(cli, "export", lambda args: pytest.fail("Exporter must not run without an explicit run"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "--run" in capsys.readouterr().err
    assert not (tmp_path / "snapshot").exists()


def test_export_passes_explicit_identity(cli, monkeypatch, tmp_path):
    run = "wncfht/MirrorOPD-verl/krkmq098"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "export", "--run", run, "--out", str(tmp_path / "snapshot")])
    observed = []
    monkeypatch.setattr(cli, "export", lambda args: observed.append(args.run))
    cli.main()
    assert observed == [run]


def test_cloud_export_is_explicit(cli, monkeypatch, tmp_path):
    run = "wncfht/MirrorOPD-verl/krkmq098"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "export-cloud", "--run", run, "--out", str(tmp_path / "cloud")])
    observed = []
    monkeypatch.setattr(cli, "export_cloud", lambda args: observed.append(args.run))
    monkeypatch.setattr(cli, "export", lambda args: pytest.fail("Cloud command must be explicit"))
    cli.main()
    assert observed == [run]


def test_local_export_never_falls_back_to_cloud(cli, monkeypatch, tmp_path):
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "export",
            "--run",
            "team/project/missing",
            "--logs-repo",
            str(tmp_path / "logs"),
            "--out",
            str(out),
        ],
    )
    monkeypatch.setattr(cli, "export_cloud", lambda args: pytest.fail("No implicit network fallback"))
    with pytest.raises(SystemExit, match="no cloud fallback"):
        cli.main()
    assert not out.exists()


def test_environment_updates_keep_initial_fields_and_reject_mixed_writers(cli, tmp_path):
    result = cli.environment_records_metadata(
        [
            {"writer_id": "driver", "python": "3.12", "git": {"commit": "abc"}, "host": "private"},
            {"writer_id": "driver", "gpu_type": "NVIDIA A800", "gpu_count": 8, "cuda_version": "12.8"},
        ]
    )
    cli.write_environment(tmp_path, result, "torch==2.11.0\n")
    saved = json.loads((tmp_path / "hardware.json").read_text())
    assert saved["python"] == "3.12" and saved["gpu_count"] == 8
    assert saved["gpu"] == "NVIDIA A800" and saved["cuda"] == "12.8"
    assert "host" not in saved and "writer_id" not in saved
    with pytest.raises(ValueError, match="Multiple environment writers"):
        cli.environment_records_metadata([{"writer_id": "a"}, {"writer_id": "b"}])


@pytest.mark.parametrize("status", ["finished", "snapshot", "partial_snapshot"])
def test_local_export_preserves_history_and_marks_recorded_system_data(cli, monkeypatch, tmp_path, status):
    logs = tmp_path / "logs"
    raw = logs / "project/wandb/offline-run-test"
    raw.mkdir(parents=True)
    (raw / "run-test.wandb").write_bytes(b"fixture")
    for name in (".venv/bin/python", "tools/wandb_local.py"):
        file = logs / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    decoded = logs / "derived/project/offline-run-test"
    decoded.mkdir(parents=True)
    metadata = {
        "run": {"run_id": "test", "project": "project"},
        "status": status,
        "exit_code": 0 if status == "finished" else None,
        "history_rows": 2,
        "step_range": [0, 1],
    }
    (decoded / "run.json").write_text(json.dumps(metadata))
    raw_history = b'{"_step":0,"loss":0.0}\n{"_step":1,"accuracy":1.0}\n'
    (decoded / "history.jsonl").write_bytes(raw_history)
    (decoded / "config.json").write_text('{"api_key":"private", "lr":0.1}')
    (decoded / "summary.json").write_text('{"accuracy":1.0}')
    (decoded / "metric-definitions.jsonl").write_text("")
    with gzip.open(decoded / "system.jsonl.gz", "wt") as stream:
        stream.write(json.dumps({"timestamp": "2026-10-22T15:11:26.674956562Z", "metrics": {"gpu.0.gpu": 0}}) + "\n")
    calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(cli, "export_cloud", lambda args: pytest.fail("No cloud read"))
    monkeypatch.setattr(cli, "summarize", lambda *args: pytest.fail("Do not run legacy analysis on policy-v3 exports"))
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "export", "--run", "team/project/test", "--logs-repo", str(logs), "--out", str(out)]
    )
    cli.main()
    assert len(calls) == 1
    assert calls[0][0][0][0] == str(logs / ".venv/bin/python")
    assert (out / "history.jsonl").read_bytes() == raw_history
    assert json.loads((out / "config.json").read_text()) == {"api_key": "[REDACTED]", "lr": 0.1}
    source = json.loads((out / "source.json").read_text())
    assert source["decoder_status"] == status
    assert source["online_history_retrieved"] is False
    assert source["system_api_sampled"] is False
    assert source["hardware_metadata_available"] is False
    system = cli.read_rows(out / "system.jsonl.gz")
    assert system[0]["_timestamp_ns"] == 1792681886674956562
    assert system[0]["system.gpu.0.gpu"] == 0
    assert not (out / "system-sampled.jsonl.gz").exists()
    with pytest.raises(FileExistsError):
        cli.main()


@pytest.mark.parametrize("run", [None, "wncfht/MirrorOPD-verl/krkmq098"])
def test_imported_log_does_not_inherit_another_run(cli, monkeypatch, tmp_path, run):
    log = tmp_path / "train.log"
    log.write_text("step:1 - timing_s/step:2 - response_length/mean:10\n")
    out = tmp_path / "snapshot"
    argv = [str(SCRIPT), "import-log", "--log", str(log), "--out", str(out)]
    if run:
        argv += ["--run", run]
    monkeypatch.setattr(sys, "argv", argv)
    # Identity capture belongs to import_log; legacy numerical analysis has separate tests.
    monkeypatch.setattr(cli, "summarize", lambda *args: None)
    cli.main()
    assert json.loads((out / "source.json").read_text())["run"] == run
