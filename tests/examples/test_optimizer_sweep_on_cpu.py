# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""An optimizer probe must preserve bank identity while isolating all output paths."""

import json

import yaml

from examples.mirroropd.research.core.io import dump
from examples.mirroropd.tools import absorption_queue, optimizer_sweep
from verl.trainer.evaluation.artifacts import file_sha256


def test_prepare_preserves_bank_and_separates_runtime_paths(tmp_path, monkeypatch):
    origin = tmp_path / "reference"
    origin.mkdir()
    (origin / "questions").mkdir()
    (origin / "questions/train.parquet").write_bytes(b"same training data")
    bootstrap = tmp_path / "bank"
    bootstrap.mkdir()
    (bootstrap / "manifest.json").write_text("{}")
    source = origin / "source"
    output = origin / "training-sgd/E124"
    path = origin / "E124.yaml"
    config = dict(
        data=dict(train_files=[str(origin / "questions/train.parquet")]),
        actor_rollout_ref=dict(actor=dict(optim=dict(lr=0.01))),
        trainer=dict(
            evaluation=dict(bootstrap=str(bootstrap), datasets=[dict(path="unchanged heldout")]),
            default_local_dir=str(output / "runtime"),
        ),
        ray_kwargs=dict(
            ray_init=dict(
                runtime_env=dict(
                    env_vars=dict(PYTHONPATH=str(source), VERL_FILE_LOGGER_PATH=str(output / "metrics.jsonl"))
                )
            )
        ),
    )
    path.write_text(yaml.safe_dump(config))
    dump(
        origin / "prepared.json",
        dict(
            experiments=dict(E124=str(path)),
            teacher_configs=dict(E124=str(path)),
            configs={str(path): file_sha256(path)},
            source=str(source),
            environment=dict(PYTHONPATH=str(source), WANDB_MODE="offline"),
            models=[],
            settings=dict(outer=40, reuse=16),
        ),
    )

    def freeze(repo, target):
        target.mkdir()
        return target

    monkeypatch.setattr(optimizer_sweep, "freeze", freeze)
    monkeypatch.setattr(absorption_queue, "preflight", lambda path: None)
    checks = []
    monkeypatch.setattr(absorption_queue, "verify_reused_bootstrap", lambda config, digest: checks.append(config))
    root = tmp_path / "sweep"
    optimizer_sweep.prepare(tmp_path, root, origin)
    plan = json.loads((root / "plan.json").read_text())
    assert len(plan["lanes"]) == 4
    assert len(checks) == 9
    for lane in plan["lanes"]:
        from pathlib import Path

        manifest = json.loads((Path(lane["study"]) / "prepared.json").read_text())
        config = yaml.safe_load(Path(manifest["experiments"][lane["experiment"]]).read_text())
        evaluation = config["trainer"]["evaluation"]
        assert evaluation["bootstrap"] == str(bootstrap)
        assert evaluation["datasets"] == [dict(path="unchanged heldout")]
        assert manifest["environment"]["PYTHONPATH"] == str(root / "source")
        env = config["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"]
        assert env["PYTHONPATH"] == str(root / "source")
        log_path = Path(lane["study"]) / "training-muon" / lane["experiment"] / "metrics.jsonl"
        assert env["VERL_FILE_LOGGER_PATH"] == str(log_path)
        assert config["trainer"]["logger"] == ["console", "file", "wandb"]
        assert config["actor_rollout_ref"]["actor"]["optim"]["lr"] == lane["lr"]
        assert config["actor_rollout_ref"]["actor"]["optim"]["override_optimizer_config"]["aux_lr"] == 1e-6
