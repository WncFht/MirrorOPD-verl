# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual queue stage wiring without CUDA, Ray or subprocesses."""

import json
import os
from pathlib import Path

from examples.mirroropd.research.studies.absorption_dynamics import queue


def test_prepare_selection_excludes_completed_pairs_and_rejects_duplicate_ids():
    import pytest

    rows = queue.selected_pairs([f"E{n}" for n in range(105, 124)])
    assert [row[0] for row in rows] == list(range(105, 124))
    assert len(queue.selected_pairs()) == 30
    for names in ([], ["E105", "E105"], ["E124"]):
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
        queue.partition_numbers(["E124"])


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
