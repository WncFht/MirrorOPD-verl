# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the One-Shot state-coverage primitives."""

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from . import oneshot_state_coverage as pipeline
from .core import (
    assign_clusters,
    coverage,
    deterministic_rollout_split,
    fit_kmeans,
    fit_pca,
    relative_positions,
    transform_pca,
)
from .oneshot_state_coverage import fit, score


def test_relative_positions_are_uniform_and_unique():
    assert relative_positions(list(range(10)), 4) == [0, 3, 6, 9]
    assert relative_positions([2, 4], 8) == [2, 4]
    assert relative_positions([], 8) == []


def test_pca_round_trip_dimension_and_variance():
    values = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    pca = fit_pca(values, 1)
    transformed = transform_pca(values, pca)
    assert transformed.shape == (4, 1)
    assert np.isclose(float(pca["total_explained_variance_ratio"]), 1.0)


def test_kmeans_finds_two_separated_clusters_and_coverage():
    values = np.asarray([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]], dtype=np.float32)
    centers, info = fit_kmeans(values, 2, seed=3)
    labels = assign_clusters(values, centers)
    assert info["occupied_clusters"] == 2
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert coverage(labels[:2], 2) == (1, 0.5)
    assert coverage(labels, 2) == (2, 1.0)


def test_rollout_split_keeps_rollouts_whole_and_is_deterministic():
    records = [
        {"source": "full", "trajectory_id": f"full::{i}", "outer": group}
        for group in (1, 2)
        for i in range(group * 10, group * 10 + 5)
    ]
    reference_a, heldout_a = deterministic_rollout_split(records, source="full", group_field="outer", seed=17)
    reference_b, heldout_b = deterministic_rollout_split(records, source="full", group_field="outer", seed=17)
    assert (reference_a, heldout_a) == (reference_b, heldout_b)
    assert not reference_a & heldout_a
    assert reference_a | heldout_a == {row["trajectory_id"] for row in records}
    assert len(reference_a) == 6
    assert len(heldout_a) == 4


def test_invalid_kmeans_and_split_fail_loudly():
    with pytest.raises(ValueError, match="K must"):
        fit_kmeans(np.ones((2, 2)), 3)
    with pytest.raises(ValueError, match="fewer than two"):
        deterministic_rollout_split(
            [{"source": "full", "trajectory_id": "full::one", "outer": 1}],
            source="full",
            group_field="outer",
            seed=1,
        )


def _write_extraction(directory: Path, source: str):
    directory.mkdir()
    vectors = []
    records = []
    rng = np.random.default_rng(4 if source == "full" else 5)
    for step in (1, 2):
        for rollout in range(5):
            trajectory_id = f"{source}::step-{step}-rollout-{rollout}"
            center = np.asarray([0.0, 0.0, 0.0, 0.0]) if rollout % 2 == 0 else np.asarray([8.0, 8.0, 0.0, 0.0])
            for state_index in range(4):
                vectors.append(center + rng.normal(0.0, 0.05, size=4))
                records.append(
                    {
                        "trajectory_id": trajectory_id,
                        "source": source,
                        "step": step,
                        "state_index": state_index,
                    }
                )
    np.save(directory / "vectors.npy", np.asarray(vectors, dtype=np.float32), allow_pickle=False)
    with (directory / "states.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")
    (directory / "metadata.json").write_text(
        json.dumps({"teacher_model": "teacher", "model_config_sha256": "same"}) + "\n"
    )


def test_fit_and_score_share_codebook_and_hold_out_reference_rollouts(tmp_path):
    full = tmp_path / "full"
    policy = tmp_path / "policy"
    _write_extraction(full, "full")
    _write_extraction(policy, "policy")
    fitted = tmp_path / "fit"
    fit(
        Namespace(
            extractions=[full, policy],
            reference_source="full",
            split_field="step",
            seed=17,
            reference_fraction=0.6,
            pca_dim=2,
            clusters=2,
            max_iter=50,
            tol=1e-4,
            output=fitted,
        )
    )
    fit_metadata = json.loads((fitted / "fit.json").read_text())
    split = json.loads((fitted / "rollout_split.json").read_text())
    assert fit_metadata["reference_rollout_count"] == 6
    assert fit_metadata["heldout_rollout_count"] == 4
    assert not set(split["reference_rollouts"]) & set(split["heldout_rollouts"])

    results = tmp_path / "results"
    score(
        Namespace(
            extractions=[full, policy],
            fit=fitted,
            step_field="step",
            rollouts_per_step=2,
            repeats=3,
            seed=29,
            max_step=2,
            distance_chunk_size=16,
            allow_fewer=False,
            output=results,
        )
    )
    with (results / "coverage-summary.csv").open(encoding="utf-8") as stream:
        lines = stream.readlines()
    assert len(lines) == 5
    assert "coverage_mean" in lines[0]
    assert (results / "coverage-repeats.csv").is_file()


def test_extract_uses_prefix_hidden_positions_and_response_mask(tmp_path, monkeypatch):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(
                pad_token_id=0,
                eos_token_id=31,
                to_dict=lambda: {"_name_or_path": "dummy", "hidden_size": 3},
            )

        def forward(self, input_ids, **kwargs):
            hidden = input_ids.float().unsqueeze(-1).expand(*input_ids.shape, 3)
            return SimpleNamespace(hidden_states=(hidden,))

    rows = [
        {
            "id": "a",
            "source_id": "prompt-a",
            "step": 1,
            "prompt_ids": [1, 2, 3],
            "response_ids": [4, 5, 6, 7],
            "response_mask": [True, False, True, True],
        },
        {
            "id": "b",
            "source_id": "prompt-b",
            "step": 1,
            "prompt_ids": [1, 2],
            "response_ids": [8, 9, 10],
            "response_mask": [True, True, True],
        },
    ]
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "extracted"
    dummy = DummyModel()
    monkeypatch.setattr(pipeline, "_load_hf_model", lambda args: (torch, dummy, "cpu"))
    pipeline.extract(
        Namespace(
            inputs=[source],
            output=output,
            model="dummy",
            source="teacher",
            positions=2,
            batch_size=2,
            device="cpu",
            dtype="float32",
            allow_download=False,
            trust_remote_code=False,
        )
    )
    vectors = np.load(output / "vectors.npy", allow_pickle=False)
    with (output / "states.jsonl").open(encoding="utf-8") as stream:
        states = [json.loads(line) for line in stream]
    assert vectors[:, 0].tolist() == [3.0, 6.0, 2.0, 9.0]
    assert [row["response_position"] for row in states] == [0, 3, 0, 2]
    assert all(row["source"] == "teacher" for row in states)
