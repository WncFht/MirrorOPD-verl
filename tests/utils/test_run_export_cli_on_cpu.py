# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Prevent implicit historical run selection without making network requests."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "examples/mirroropd/docs/scripts/analyze_run.py"


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
