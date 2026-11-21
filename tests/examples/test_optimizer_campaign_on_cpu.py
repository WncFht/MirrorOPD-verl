# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Matched trials must share evaluation inputs and select only complete development windows."""

import json
from pathlib import Path

import pytest

from examples.mirroropd.tools.optimizer_campaign import configure, metrics_summary, report


def reference():
    return dict(
        actor_rollout_ref=dict(actor=dict(optim={}, checkpoint={})),
        trainer=dict(evaluation=dict(datasets=[dict(name="math", path="/original/heldout.parquet")])),
        ray_kwargs=dict(ray_init=dict(runtime_env=dict(env_vars={}))),
    )


def test_matched_configs_preserve_bank_identity_and_checkpoint_state():
    kwargs = dict(
        name="E130",
        study=Path("/new"),
        source=Path("/source"),
        original_source="/old/source",
        origin=Path("/original"),
        bootstrap=Path("/shared/bank"),
        lr=1e-6,
        development=dict(name="development_math", path="/dev/math.parquet"),
    )
    for kind, expected in [("adamw", "AdamW"), ("muon", "MuonAdamW")]:
        config = configure(reference(), kind=kind, **kwargs)
        actor = config["actor_rollout_ref"]["actor"]
        assert actor["optim"]["optimizer"] == expected
        assert actor["optim"]["weight_decay"] == 0
        assert actor["checkpoint"]["load_contents"] == ["model", "optimizer", "extra"]
        evaluation = config["trainer"]["evaluation"]
        assert evaluation["datasets"][0]["path"] == "/original/heldout.parquet"
        assert evaluation["bootstrap"] == "/shared/bank"
        assert evaluation["export_outers"] == [20, 40]
        assert config["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"]["WANDB_MODE"] == "offline"


def write_curve(path, *, dev, heldout, valid=32, outer=40):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in range(outer + 1):
        data = {"step/outer": step, "step/optimizer": step * 16}
        for bank in ("initial", "teacher"):
            for dataset, end, count in [("math", heldout, 128), ("development_math", dev, valid)]:
                prefix = f"eval/rkl/public_{bank}/dataset/{dataset}/"
                data.update(
                    {
                        prefix + "question_mean": 1 if step == 0 else end,
                        prefix + "valid_rows": count,
                        prefix + "invalid_rows": 0,
                    }
                )
        rows.append(json.dumps(dict(data=data)))
    path.write_text("\n".join(rows) + '\n{"data":')


def test_partial_append_and_incomplete_development_are_not_final(tmp_path):
    path = tmp_path / "metrics.jsonl"
    write_curve(path, dev=0.2, heldout=0.1, outer=39)
    result = metrics_summary(path)
    assert result["updates"] == 624
    assert not result["banks"]["initial/development_math"]["final_window"]
    write_curve(path, dev=0.2, heldout=0.1, valid=1)
    result = metrics_summary(path)
    assert not result["banks"]["initial/development_math"]["coverage_complete"]


def test_selection_uses_development_not_heldout_or_single_best_point(tmp_path):
    jobs = []
    for index, (dev, heldout, valid) in enumerate([(0.3, 0.01, 32), (0.2, 0.5, 32), (0.1, 0.001, 1)]):
        study = tmp_path / str(index)
        write_curve(study / "training-adamw" / str(index) / "metrics.jsonl", dev=dev, heldout=heldout, valid=valid)
        (study / "status.json").write_text(json.dumps(dict(state="complete")))
        jobs.append(
            dict(experiment=str(index), pair="E102", optimizer="adamw", study=str(study), lr=(index + 1) * 1e-6)
        )
    (tmp_path / "plan.json").write_text(json.dumps(dict(jobs=jobs)))
    result = report(tmp_path)
    assert result["development_selection"]["E102/adamw"]["experiment"] == "1"
    assert result["development_selection"]["E102/adamw"]["score"] == pytest.approx(0.2)


def test_retry_rejects_possible_updates_before_creating_attempt(tmp_path):
    import pytest

    from examples.mirroropd.tools.optimizer_campaign import retry

    origin = tmp_path / "original"
    study = origin / "e130"
    (study / "logs").mkdir(parents=True)
    (origin / "controller-status.json").write_text(json.dumps(dict(state="failed")))
    (origin / "plan.json").write_text(
        json.dumps(dict(jobs=[dict(experiment="E130", optimizer="adamw", study=str(study))]))
    )
    (study / "status.json").write_text(json.dumps(dict(state="stopped", stages=[dict(label="E130-train")])))
    (study / "logs/E130-train.log").write_text("create worker group dict_keys\n")
    (study / "training-adamw").mkdir()
    with pytest.raises(ValueError, match="verified full checkpoint"):
        retry(tmp_path, tmp_path / "retry", origin, {"E130": "E152"})
    assert not (tmp_path / "retry").exists()


def test_outer_zero_is_valid_during_concurrent_report(tmp_path):
    path = tmp_path / "metrics.jsonl"
    write_curve(path, dev=0.2, heldout=0.1, outer=0)
    result = metrics_summary(path)
    assert result["banks"]["initial/development_math"]["outer"] == 0
    assert result["banks"]["initial/development_math"]["tail_mean"] == 1
    assert not result["banks"]["initial/development_math"]["final_window"]
