# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Expanded legacy banks reuse old states without mutating historical heads or receipts."""

from types import SimpleNamespace

import torch

from examples.mirroropd.research.core.artifacts import file_hash
from examples.mirroropd.research.core.io import dump
from examples.mirroropd.research.studies.boundary import fast_engine as engine


def test_expansion_reuses_history_and_generates_only_missing_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "OLD_CACHE", tmp_path / "old")
    monkeypatch.setattr(engine, "CACHE", tmp_path / "previous")
    monkeypatch.setattr(engine, "MODELS", {"teacher": ("org/model", "sha")})
    monkeypatch.setattr(engine, "model_path", lambda _: tmp_path / "model")
    source = tmp_path / "old/teachers/teacher/r1"
    source.mkdir(parents=True)
    rows = [dict(id=i, prompt_ids=[1], response_ids=[2], source="frozen") for i in [0, 1]]
    torch.save(torch.ones(3, 2), source / "head.pt")
    torch.save(dict(rows[0], teacher=("org/model", "sha"), hidden=torch.ones(1, 2)), source / "0000.pt")
    dump(source / "complete.json", dict(teacher=["org/model", "sha"], rows=1))
    before = {p.name: file_hash(p) for p in source.iterdir()}
    generated = []
    monkeypatch.setattr(
        engine, "load", lambda *a, **k: SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones(3, 2)))
    )

    def hidden(model, row):
        generated.append(row["id"])
        return torch.full((1, 2), 2.0)

    monkeypatch.setattr(engine, "hidden", hidden)
    original_load = torch.load
    monkeypatch.setattr(torch, "load", lambda path, **kw: original_load(path, **dict(kw, map_location="cpu")))
    head, states = engine.teacher_cache("teacher", rows, cache_root=tmp_path / "expanded")
    assert generated == [1] and len(states) == 2 and head.shape == (3, 2)
    assert before == {p.name: file_hash(p) for p in source.iterdir()}
    assert (tmp_path / "expanded/teachers/teacher/r1/head.pt").is_symlink()
    engine.teacher_cache("teacher", rows, cache_root=tmp_path / "expanded")
    assert generated == [1]
