# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Frozen teacher references preserve metric, decoding protocol and completeness."""

import copy
import hashlib
import json

import pytest

from examples.mirroropd.research.core.artifacts import fingerprint
from examples.mirroropd.research.studies.qwen_teachers.figure_inputs import (
    DEFAULT_MANIFEST,
    checked_json,
    metric_label,
    teacher_reference,
)


@pytest.fixture
def evidence():
    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    return manifest["teacher_evidence"]["e43-teacher"]


@pytest.fixture
def student():
    return {
        "teacher": "Qwen3-4B-Instruct-2507",
        "evaluation": {
            "metric": "mean_correctness",
            "samples": 16,
            "temperature": 0.7,
            "top_p": 0.9,
            "response_cap": 31744,
            "enable_thinking": False,
            "seed": 42,
        },
    }


def test_full_teacher_reference_keeps_independent_sampling(evidence, student):
    result = teacher_reference(evidence, DEFAULT_MANIFEST.parent, student)
    assert result["evaluation"]["samples"] == 4 and result["evaluation"]["seed"] == 72000000
    scores = {r["task"]: r["accuracy"] for r in result["rows"]}
    assert scores == pytest.approx(
        {"math500": 0.947, "amc23": 0.95625, "aime24": 19 / 30, "aime25": 5 / 12, "aime26": 29 / 60}
    )
    assert metric_label(student["evaluation"]) == "avg@16"


@pytest.mark.parametrize(
    "key,value", [("response_cap", 16384), ("temperature", 1), ("enable_thinking", True), ("metric", "pass@4")]
)
def test_rejects_incompatible_evaluation(evidence, student, key, value):
    student["evaluation"][key] = value
    with pytest.raises(ValueError, match="evaluation protocol mismatch"):
        teacher_reference(evidence, DEFAULT_MANIFEST.parent, student)


def test_rejects_wrong_teacher(evidence, student):
    student["teacher"] = "Qwen3-4B"
    with pytest.raises(ValueError, match="Teacher model"):
        teacher_reference(evidence, DEFAULT_MANIFEST.parent, student)


def test_changed_frozen_bytes_fail_before_reading(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"response_cap":31744}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text('{"response_cap":16384}')
    with pytest.raises(ValueError, match="hash mismatch"):
        checked_json(path, digest)


def test_partial_teacher_benchmark_fails_even_with_consistent_hashes(tmp_path, evidence, student):
    modified = copy.deepcopy(evidence)
    records = {k: json.loads((DEFAULT_MANIFEST.parent / v["path"]).read_text()) for k, v in evidence.items()}
    records["metrics"]["eval/math500/observed_response_count"] -= 1
    for key in ["metrics", "protocol", "complete"]:
        if key == "complete":
            records[key]["metrics_sha256"] = modified["metrics"]["sha256"]
        path = tmp_path / modified[key]["path"]
        path.write_text(json.dumps(records[key]))
        modified[key]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Incomplete teacher benchmark: math500"):
        teacher_reference(modified, tmp_path, student)


def test_receipt_cannot_be_reused_for_another_protocol(tmp_path, evidence, student):
    modified = copy.deepcopy(evidence)
    for key, item in modified.items():
        record = json.loads((DEFAULT_MANIFEST.parent / item["path"]).read_text())
        if key == "protocol":
            record["chat_template_kwargs"]["enable_thinking"] = True
        path = tmp_path / item["path"]
        path.write_text(json.dumps(record))
        item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="completion receipt does not match its protocol"):
        teacher_reference(modified, tmp_path, student)


def test_receipt_fingerprint_is_checked(tmp_path, evidence, student):
    modified = copy.deepcopy(evidence)
    for key, item in modified.items():
        record = json.loads((DEFAULT_MANIFEST.parent / item["path"]).read_text())
        if key == "complete":
            record["inputs"]["profile"]["samples"] = 16
            assert record["fingerprint"] != fingerprint(record["inputs"])
        path = tmp_path / item["path"]
        path.write_text(json.dumps(record))
        item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="completion receipt does not match its protocol"):
        teacher_reference(modified, tmp_path, student)


def test_config_extraction_preserves_distinct_training_and_eval_settings():
    from examples.mirroropd.research.studies.qwen_teachers.figure_inputs import config_protocol, setting_label

    config = {
        "actor_rollout_ref": {
            "actor": {"ppo_epochs": 8, "optim": {"lr": 3e-6}},
            "model": {"path": "/models/student"},
            "rollout": {
                "n": 1,
                "temperature": 1,
                "top_p": 1,
                "response_length": 4096,
                "seed": 43,
                "val_kwargs": {"n": 32, "temperature": 0.6, "top_p": 0.95, "top_k": -1, "max_tokens": 16384},
            },
        },
        "data": {
            "apply_chat_template_kwargs": {"enable_thinking": True},
            "train_batch_size": 32,
            "val_files": ["/data/AIME25/test.parquet"],
        },
        "trainer": {"val_before_train": False},
        "distillation": {
            "teacher_models": {"teacher_model": {"model_path": "/models/teacher"}},
            "distillation_loss": {
                "loss_mode": "mirror",
                "use_task_rewards": False,
                "mirror": {"objective": "mdpo", "estimator": "full_vocab", "kappa": 0, "eta": 1},
                "full_vocab": {"termination_token_ids": [151643, 151645]},
            },
        },
    }
    result = config_protocol(config)
    assert result["training"]["response_cap"] == 4096
    assert result["evaluation"]["response_cap"] == 16384
    assert result["training"]["termination_token_ids"] == [151643, 151645]
    assert metric_label(result["evaluation"]) == "avg@32"
    label = setting_label({"experiment": "fixture", **result})
    assert "Thinking ON" in label and "reuse8, batch32, LR 3e-6" in label
    config["distillation"]["distillation_loss"]["mirror"]["kappa"] = 1
    with pytest.raises(ValueError, match="require Full-RKL"):
        config_protocol(config)


def test_telemetry_hash_ignores_gzip_container_timestamp(tmp_path):
    import gzip

    from examples.mirroropd.research.studies.qwen_teachers.figure_inputs import check_source

    raw = b'{"time":1,"gpu":80}\n'
    path = tmp_path / "system.jsonl.gz"
    entry = {"hash_scope": "uncompressed_bytes", "sha256": hashlib.sha256(raw).hexdigest()}
    for timestamp in [1, 2]:
        path.write_bytes(gzip.compress(raw, mtime=timestamp))
        check_source(path, entry)
    path.write_bytes(gzip.compress(raw.replace(b"80", b"90"), mtime=2))
    with pytest.raises(ValueError, match="hash mismatch"):
        check_source(path, entry)


def test_percentage_display_uses_half_up_without_changing_raw_scores():
    from examples.mirroropd.research.studies.qwen_teachers.figure_inputs import percent_label

    assert percent_label(0.95625) == "95.63"
    assert percent_label(0.28125) == "28.13"
