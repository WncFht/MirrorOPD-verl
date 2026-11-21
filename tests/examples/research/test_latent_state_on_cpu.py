# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU regression for state alignment, shared references and saved trajectory identity."""

import numpy as np
import pytest

from examples.mirroropd.research.studies.latent_state.data import normalize
from examples.mirroropd.research.studies.latent_state.geometry import coverage_draws, state_positions
from examples.mirroropd.research.studies.latent_state.history import load_history


def test_prefix_positions_are_before_actions_and_unique():
    assert state_positions(5, 1) == [4]
    assert state_positions(5, 3) == [4, 5, 6]
    assert state_positions(5, 100)[-1] == 103
    with pytest.raises(ValueError):
        state_positions(0, 3)


def test_draws_share_prompt_sample_for_equal_pools():
    prompt = np.tile(np.arange(10), 2)
    labels = np.tile(np.arange(10) % 3, 2)
    groups = {name: (np.arange(20) // 10 == i, list(range(10))) for i, name in enumerate(("a", "b"))}
    result = coverage_draws(labels, prompt, groups, 3, n_prompts=3, n_draws=10)
    assert result["a"]["values"] == result["b"]["values"]


def test_adapter_preserves_tokens_masks_and_whole_prompt_holdout():
    row = dict(id="q", source_id="q", prompt_ids=[7], response_ids=[8, 9, 0], response_mask=[True, True, False])
    a = normalize(row, dict(group="initial"))
    b = normalize(dict(row, response_ids=[9, 8, 0]), dict(group="teacher"))
    assert a["tokens"] == [7, 8]
    assert a["split"] == b["split"]
    assert a["prompt_key"] == b["prompt_key"]
    assert a["response_tokens"] == 2
    row["response_mask"] = [True, False, True]
    with pytest.raises(ValueError, match="Noncontiguous"):
        normalize(row, dict(group="initial"))


def test_history_keeps_original_policy_timing_and_masks(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder = tmp_path / "trajectories"
    folder.mkdir()
    original = dict(
        id="40:q",
        source_id="q",
        outer=40,
        min_global_steps=39,
        max_global_steps=39,
        prompt_ids=[1, 2],
        response_ids=[3, 0],
        response_mask=[True, False],
    )
    pq.write_table(pa.Table.from_pylist([original]), folder / "outer-040.parquet")
    rows, shards = load_history(tmp_path)
    assert rows == [original]
    assert shards[0]["rows"] == 1
    assert len(shards[0]["sha256"]) == 64


def test_atomic_cache_rejects_content_changes(tmp_path):
    from examples.mirroropd.research.studies.latent_state.artifacts import cached, publish

    path = tmp_path / "states.npz"
    assert not cached(path, "a")
    publish(path, "a", dict(features=np.ones((3, 4))))
    assert cached(path, "a")
    with pytest.raises(ValueError, match="identity"):
        cached(path, "b")
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="content"):
        cached(path, "a")


def test_cosine_knn_matches_sklearn():
    from sklearn.neighbors import NearestNeighbors

    from examples.mirroropd.research.studies.latent_state.geometry import cosine_knn

    rng = np.random.default_rng(3)
    reference, query = rng.normal(size=(40, 13)), rng.normal(size=(17, 13))
    expected = NearestNeighbors(n_neighbors=10, metric="cosine").fit(reference).kneighbors(query)[0].mean(axis=1)
    np.testing.assert_allclose(cosine_knn(query, reference, chunk=4), expected, atol=1e-7)


def test_batched_left_padding_matches_individual_causal_prefix():
    import torch
    from transformers import LlamaConfig, LlamaModel

    from examples.mirroropd.research.studies.latent_state.extract import signatures

    torch.manual_seed(42)
    model = LlamaModel(
        LlamaConfig(
            vocab_size=30,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            attention_dropout=0,
        )
    ).eval()
    rows = [dict(tokens=[1, 2, 3, 4, 5], positions=[1, 3]), dict(tokens=[3, 4], positions=[0, 1])]
    actual = signatures(model, rows, "cpu", 0)
    expected = np.concatenate([signatures(model, [row], "cpu", 0) for row in rows])
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    prefix = dict(tokens=[1, 2, 3, 4], positions=[3])
    np.testing.assert_allclose(actual[1], signatures(model, [prefix], "cpu", 0)[0], atol=1e-5)


def test_analysis_fits_training_only_reference_and_scores_query_banks(tmp_path, monkeypatch):
    from examples.mirroropd.research.core.io import dump, read
    from examples.mirroropd.research.studies.latent_state import analysis
    from examples.mirroropd.research.studies.latent_state.data import GROUPS

    rng = np.random.default_rng(1)
    # Bank states must never enter fitting; bank queries use every saved response.
    records = [
        dict(
            group=g,
            prompt=str(p),
            trajectory=f"{g}:{p}",
            split="reference" if p < 12 else "test",
            action=a,
            rank=a,
            outer=0,
            response_tokens=3,
        )
        for g in GROUPS
        for p in range(20 + GROUPS.index(g))
        for a in range(3)
    ]
    data = {k: np.array([r[k] for r in records]) for k in records[0]}
    data["features"] = rng.normal(size=(len(records), 9)).astype(np.float32)
    plan = tmp_path / "plan.json"
    dump(plan, dict(groups=GROUPS, seed=42))
    monkeypatch.setattr(analysis, "load_signatures", lambda *_: (data, ["fixture"]))
    result = analysis.analyze(plan, "test", ks=(3, 4), components=5)
    with np.load(tmp_path / "analysis/test/geometry.npz") as fitted:
        indices = fitted["reference_indices"]
        assert np.all(data["split"][indices] == "reference")
        assert np.all(data["action"][indices] > 0)
        assert not np.isin(data["group"][indices], ["initial", "teacher"]).any()
        assert len(indices) == 4 * 12 * 2
        np.testing.assert_allclose(fitted["pca_mean"], data["features"][indices].mean(axis=0), atol=1e-6)
    for g in GROUPS:
        assert result["metrics"]["4"][g]["states"] == np.sum(
            (data["group"] == g) & ((data["split"] == "test") | (g in ("initial", "teacher"))) & (data["action"] > 0)
        )
    assert result["metrics"]["4"]["training_test"]["js_to_training"] == 0
    assert len(read(tmp_path / "analysis/test/result.json")["map"]["points"]) == len(records)


def test_balanced_js_target_excludes_query_trajectories():
    from examples.mirroropd.research.studies.latent_state.analysis import balanced_js

    groups = ["initial", "outer01_10", "teacher"]
    data = dict(
        group=np.repeat(groups, 10),
        trajectory=np.array([f"{g}:{p}" for g in groups for p in range(10)]),
    )
    # Every state has its own cluster: disjoint target/query must have JS=1.
    labels = np.arange(30)
    budget, result = balanced_js(labels, data, np.ones(30, dtype=bool), groups, 30, 42)
    assert budget == 5
    for row in result.values():
        np.testing.assert_allclose(row["values"], np.ones(30))


def test_reference_masks_are_invariant_to_bank_state_values():
    from examples.mirroropd.research.studies.latent_state.analysis import reference_masks

    data = dict(
        group=np.array(["initial", "outer01_10", "outer01_10", "teacher"]),
        split=np.array(["reference", "reference", "test", "reference"]),
        action=np.ones(4),
    )
    ref, test, training = reference_masks(data)
    np.testing.assert_array_equal(ref, [1])
    np.testing.assert_array_equal(test, [True, False, True, True])
    np.testing.assert_array_equal(training, [False, True, True, False])


def test_report_reads_all_existing_rkl_nodes_and_keeps_domains(tmp_path):
    from examples.mirroropd.research.core.io import dump
    from examples.mirroropd.research.studies.latent_state.report_data import risk_rows

    evaluation = tmp_path / "evaluation"
    for outer in (0, 1, 40):
        metrics = {}
        for bank in ("public_initial", "public_teacher"):
            for domain in ("math", "code", "if", "all"):
                prefix = f"eval/rkl/{bank}/{domain}/"
                metrics.update(
                    {
                        prefix + "question_mean": 1 / (outer + 1),
                        prefix + "reduction_percent": 100 * outer / (outer + 1),
                        prefix + "rows": 128,
                        prefix + "tokens": 1000,
                    }
                )
        dump(
            evaluation / f"events/outer-{outer:03d}/manifest.json",
            dict(metadata=dict(outer=outer, optimizer_steps=outer * 16, metrics=metrics)),
        )
    plan = dict(pairs={"E100": dict(evaluation=str(evaluation), risks=[])})
    rows, sources = risk_rows(plan)
    assert len(rows) == 3 * 2 * 4
    assert len(sources) == 3
    assert {row["steps"] for row in rows} == {0, 16, 640}
    assert {row["domain"] for row in rows} == {"math", "code", "if", "all"}
    assert next(row for row in rows if row["steps"] == 640)["rkl"] == 1 / 41


def test_teacher_overview_matches_prompts_and_disjoint_calibration():
    from examples.mirroropd.research.studies.latent_state.overview import teacher_overview

    groups = ["initial", "outer01_10", "outer11_20", "outer21_30", "outer31_40", "teacher"]
    points = [
        dict(
            group=g,
            trajectory=f"{g}:{i}",
            metadata=dict(
                action=1,
                split="query" if g in ("initial", "teacher") else "test",
                prompt=str(i),
                clusters={str(k): i for k in (50, 100, 200)},
            ),
        )
        for g in groups
        for i in range(10)
    ]
    result = teacher_overview(points, groups, draws=8)
    assert result["budget"] == 5
    for k in ("50", "100", "200"):
        assert result["metrics"][k]["initial"]["js"]["values"] == [0] * 8
        np.testing.assert_allclose(result["metrics"][k]["teacher"]["js"]["values"], np.ones(8))
        assert result["metrics"][k]["teacher"]["coverage"]["mean"] == 5 / int(k)
        assert sum(result["metrics"][k]["initial"]["occupancy"]) == pytest.approx(1)
    reference_points = [dict(p, metadata=dict(p["metadata"], split="reference")) for p in points]
    assert teacher_overview(points + reference_points, groups, draws=8) == result
