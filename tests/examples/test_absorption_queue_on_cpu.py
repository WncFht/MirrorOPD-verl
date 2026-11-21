# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual queue stage wiring without CUDA, Ray or subprocesses."""

import json
import os
import sys
from pathlib import Path

from examples.mirroropd.tools import absorption_queue as queue


def test_prepare_selection_excludes_completed_pairs_and_rejects_duplicate_ids():
    import pytest

    rows = queue.selected_pairs([f"E{n}" for n in range(105, 124)])
    assert [row[0] for row in rows] == list(range(105, 124))
    assert len(queue.selected_pairs()) == 30
    for names in ([], ["E105", "E105"], ["E174"]):
        with pytest.raises(ValueError, match="unique registered"):
            queue.selected_pairs(names)


def test_queue_defers_runtime_interpolation_and_uses_legacy_guard(monkeypatch, tmp_path):
    from verl.trainer.evaluation import artifacts

    study = tmp_path / "study"
    source = tmp_path / "source"
    study.mkdir()
    source.mkdir()
    config = study / "E94.yaml"
    config.write_text(
        "actor_rollout_ref:\n  model: {path: initial}\n  rollout: {}\n"
        "trainer:\n  evaluation: {protocols: {}, datasets: []}\n"
        f"ray_kwargs:\n  ray_init:\n    runtime_env:\n      py_executable: {sys.executable}\n"
        "runtime: ${oc.env:ABSORPTION_UNSET_RUNTIME_TEST}\n"
    )
    monkeypatch.delenv("ABSORPTION_UNSET_RUNTIME_TEST", raising=False)
    broker = tmp_path / "broker.py"
    broker.write_text(
        "def registry(): return {'gpus': {'0': {'desired': 'released', 'owner': None}}}\n"
        "def reserve(*args, **kwargs): pass\n"
        "def attach(*args, **kwargs): pass\n"
    )
    manifest = dict(
        repo=str(tmp_path),
        source=str(source),
        devices=[0],
        code={},
        environment={},
        configs={str(config): artifacts.file_sha256(config)},
        experiments={"E94": str(config)},
        teacher_configs={"E94": str(config)},
    )
    (study / "prepared.json").write_text(json.dumps(manifest))
    calls = []

    class Process:
        pid = os.getpid()

        def __init__(self, args, **kwargs):
            calls.append(args)
            assert args[0] == sys.executable
            assert args[args.index("--") + 1] == sys.executable
            assert "--source" not in args
            assert args[args.index("--") + 2] == "-c"
            assert args[args.index("--") + 4] == str(source)
            assert kwargs["env"]["VERL_ROLLOUT_IPC_NAMESPACE"]

        def poll(self):
            return 0

        def wait(self, **kwargs):
            return 0

    class Store:
        def __init__(self, root):
            self.root = Path(root)

        def manifest(self, **kwargs):
            outer = self.root.name
            metadata = dict(coverage={"scientific_complete": True}, metrics={}, native_reload_rkl="passed")
            if outer.startswith("outer-"):
                metadata["optimizer_steps"] = int(outer.removeprefix("outer-")) * 16
            return {"metadata": metadata}

    monkeypatch.setattr(queue.subprocess, "Popen", Process)
    monkeypatch.setattr(queue, "sys", type("ControllerPython", (), {"executable": "/different/controller/python"}))
    monkeypatch.setattr(artifacts, "ArtifactStore", Store)
    queue.run(study, broker)
    assert len(calls) == 4
    assert json.loads((study / "exit.json").read_text())["returncode"] == 0


def test_llama_uses_default_text_protocol_for_banks_and_training(tmp_path):
    from examples.mirroropd.launchers.absorption import options

    (tmp_path / "questions").mkdir()
    (tmp_path / "questions/datasets.json").write_text("[]")
    overrides = options({"ABSORPTION_ROOT": str(tmp_path)}, "E103", tmp_path)
    assert "data.continuous_token.model_family=default" in overrides
    evaluation = next(x for x in overrides if x.startswith("++trainer.evaluation="))
    assert 'model_family:"default"' in evaluation


def test_partition_accepts_historical_extended_and_partial_manifests():
    import pytest

    for numbers in (range(94, 112), range(94, 124), range(112, 124), [103, 105, 112, 119]):
        front, back = queue.partition_numbers([f"E{n}" for n in numbers])
        assert set(front).isdisjoint(back)
        assert set(front) | set(back) == set(numbers)
    front, back = queue.partition_numbers([f"E{n}" for n in range(94, 124)])
    assert len(front) == len(back) == 15
    assert 103 in front and 104 in back
    with pytest.raises(ValueError, match="unregistered"):
        queue.partition_numbers(["E174"])


def test_cpu_admission_rejects_domain_overlap_and_prefix_mismatch():
    import pytest

    from examples.mirroropd.research.studies.absorption_dynamics.acceptance import check_questions

    class Tokenizer:
        def apply_chat_template(self, prompt, **kwargs):
            assert kwargs["tokenize"] is False
            return prompt[0]["content"]

        def encode(self, text, **kwargs):
            return list(text.encode())

    class Contract:
        def validate_prefix(self, ids):
            assert ids

    student = Tokenizer()
    tables = {
        name: [dict(prompt=[dict(role="user", content=content)])]
        for name, content in [("train", "question a"), ("math", "question b")]
    }
    summary = check_questions(student, student, Contract(), tables, {})
    assert summary["math"]["rows"] == 1
    tables["math"] = tables["train"]
    with pytest.raises(ValueError, match="overlap"):
        check_questions(student, student, Contract(), tables, {})
    tables["math"] = [dict(prompt=[dict(role="user", content="x" * 4097)])]
    with pytest.raises(ValueError, match="Overlong"):
        check_questions(student, student, Contract(), tables, {})

    class OtherTokenizer(Tokenizer):
        def encode(self, text, **kwargs):
            return [42]

    with pytest.raises(ValueError, match="Shared-prefix"):
        check_questions(student, OtherTokenizer(), Contract(), tables, {})


def test_optimizer_branches_require_calibration_and_preserve_historical_output(tmp_path):
    import pytest

    from examples.mirroropd.launchers.absorption import options

    (tmp_path / "questions").mkdir()
    (tmp_path / "questions/datasets.json").write_text("[]")
    for kind in ("sgd", "muon"):
        with pytest.raises(ValueError, match="calibrated"):
            options({"ABSORPTION_ROOT": str(tmp_path), "ABSORPTION_OPTIMIZER": kind}, "E94", tmp_path)
    for number in range(94, 115):
        env = {"ABSORPTION_ROOT": str(tmp_path), "ABSORPTION_OPTIMIZER": "sgd", "ABSORPTION_OPTIMIZER_LR": "0.01"}
        overrides = options(env, f"E{number}", tmp_path)
        assert env["RUN_DIR"] == str(tmp_path / "training-sgd" / f"E{number}")
        assert "actor_rollout_ref.actor.optim.optimizer=SGD" in overrides
        assert env["LEARNING_RATE"] == "0.01"
        assert "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]" in overrides
    env = {
        "ABSORPTION_ROOT": str(tmp_path),
        "ABSORPTION_OPTIMIZER": "muon",
        "ABSORPTION_OPTIMIZER_LR": "1e-6",
        "ABSORPTION_AUX_LR": "1e-6",
    }
    overrides = options(env, "E99", tmp_path)
    assert "actor_rollout_ref.actor.optim.optimizer=MuonAdamW" in overrides
    assert env["RUN_DIR"] == str(tmp_path / "training-muon/E99")


def test_failed_reservation_is_released_before_process_attach(tmp_path):
    import pytest

    study = tmp_path / "study"
    study.mkdir()
    (study / "prepared.json").write_text(
        json.dumps(
            dict(
                repo=str(tmp_path),
                source=str(tmp_path),
                devices=[0],
                code={},
                environment={},
                configs={},
                experiments={},
                teacher_configs={},
            )
        )
    )
    marker = tmp_path / "released"
    broker = tmp_path / "broker.py"
    broker.write_text(
        "from pathlib import Path\n"
        "entry = {'desired': 'released', 'owner': None}\n"
        "def registry(): return {'gpus': {'0': entry}}\n"
        "def reserve(gpu, owner):\n"
        "    entry.update(desired='reserved', owner=owner)\n"
        "    raise RuntimeError('guardian identity mismatch')\n"
        "def release(gpu, owner, reason):\n"
        "    assert entry['owner'] == owner\n"
        f"    Path({str(marker)!r}).write_text(reason)\n"
    )
    with pytest.raises(RuntimeError, match="guardian identity"):
        queue.run(study, broker)
    assert "before attaching" in marker.read_text()
    assert json.loads((study / "status.json").read_text())["allocation_cleanup_errors"] == []


def test_numbered_sgd_lr_presets(tmp_path):
    from examples.mirroropd.launchers.absorption import options

    (tmp_path / "questions").mkdir()
    (tmp_path / "questions/datasets.json").write_text("[]")
    for name, lr in (("E124", "0.01"), ("E125", "0.03")):
        env = {"ABSORPTION_ROOT": str(tmp_path)}
        overrides = options(env, name, tmp_path)
        assert "actor_rollout_ref.actor.optim.optimizer=SGD" in overrides
        assert env["LEARNING_RATE"] == lr
        assert env["RUN_DIR"] == str(tmp_path / "training-sgd" / name)
