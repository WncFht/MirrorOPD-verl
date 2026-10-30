# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Retuning cannot silently change a saved Adam trajectory's execution configuration."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from examples.mirroropd.research.core.artifacts import file_hash, fingerprint
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.studies.fixed_transfer import training, validation
from examples.mirroropd.research.studies.fixed_transfer.spec import Execution


def test_resume_inherits_validated_execution_and_preserves_retuning(tmp_path, monkeypatch):
    inputs = {"versions": ("a", "b")}
    key = fingerprint(inputs)
    origin = tmp_path / "old/arms/a"
    source = tmp_path / "old/numerical" / key
    execution = asdict(Execution(microbatch=4))
    signature = lambda study, arm, selected: {"science": "same", "execution": asdict(selected)}
    monkeypatch.setattr(training, "resume_signature", signature)
    monkeypatch.setattr(validation, "profile_inputs", lambda *a: inputs)
    monkeypatch.setattr(validation, "profile_directory", lambda study, _: study.directory / "numerical" / key)
    dump(
        source / "complete.json",
        dict(inputs=inputs, fingerprint=key, execution=execution, teacher_head_reserve_gib=0.25),
    )
    original_hash = file_hash(source / "complete.json")
    dump(origin / "protocol.json", dict(inputs={"numerical_profile": key}, execution=execution))
    dump(origin / "resume/complete.json", dict(signature=fingerprint(signature(None, None, Execution(**execution)))))
    model = tmp_path / "model"
    dump(model / "config.json", dict(vocab_size=3, hidden_size=2))
    study = SimpleNamespace(
        directory=tmp_path / "new",
        continuation={"resume_from": {"a": str(origin)}},
        arms=[SimpleNamespace(name="a", teacher="teacher")],
        models={"teacher": SimpleNamespace(directory=model)},
    )
    destination = study.directory / "numerical" / key
    retuned = dict(inputs=inputs, fingerprint=key, execution=asdict(Execution(microbatch=2)))
    dump(destination / "complete.json", retuned)
    assert validation.inherit_resume_profile(study, "teacher")
    assert read(destination / "complete.json")["execution"]["microbatch"] == 4
    note = read(destination / "inheritance.json")
    assert read(note["archived_retuning"] + "/complete.json")["execution"]["microbatch"] == 2
    assert file_hash(source / "complete.json") == original_hash
    assert validation.inherit_resume_profile(study, "teacher")
    dump(destination / "complete.json", retuned)
    dump(study.directory / "arms/a/training.json", [{"step": 321}])
    with pytest.raises(ValueError, match="after scientific updates"):
        validation.inherit_resume_profile(study, "teacher")
