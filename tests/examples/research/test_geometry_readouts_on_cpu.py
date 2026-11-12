# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Keep derived geometry coverage and spectral CDFs reproducible."""

import gzip
import json

import pytest

from examples.mirroropd.research.studies.absorption_dynamics.geometry_readouts import collect, compact


def test_spectral_readout_preserves_energy_and_raw_input():
    raw = dict(singular_values=[3, 2, 1], angles_deg=[10, 20], median_angle_deg=15, energy=14)
    result = compact(raw)
    assert result["spectrum_cdf"]["energy_fraction"] == pytest.approx([9 / 14, 13 / 14, 1])
    assert result["median_angle_deg"] == 15
    assert "singular_values" not in result and "angles_deg" not in result
    assert raw["singular_values"] == [3, 2, 1]


def test_coverage_requires_all_seven_projections_and_exact_task_identity(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    tasks = [
        dict(condition="E94", tensor=f"model.layers.0.{module}.{proj}_proj.weight", points=[[16, "saved"]])
        for module, projections in (("self_attn", ("q", "k", "v", "o")), ("mlp", ("gate", "up", "down")))
        for proj in projections
    ]
    (source / "protocol.json").write_text(
        json.dumps(dict(tasks=tasks, pairs={"E94": dict(matrix_count=7, risks=[], teacher_parameter_comparison=True)}))
    )
    for i, task in enumerate(tasks):
        record = dict(task=task, stages=[dict(step=16, singular_values=[3, 2, 1])])
        (source / f"matrix-{i:05}.json.gz").write_bytes(gzip.compress(json.dumps(record).encode()))
        result = collect(source, tmp_path / "data")
        pair = result["pairs"]["E94"]
        assert pair["complete"] == (i == 6)
        assert pair["complete_layers"] == ([0] if i == 6 else [])
    record["task"]["points"] = [[16, "changed-checkpoint"]]
    (source / "matrix-00006.json.gz").write_bytes(gzip.compress(json.dumps(record).encode()))
    with pytest.raises(ValueError, match="identity differs"):
        collect(source, tmp_path / "invalid")


def test_angle_export_preserves_exact_ecdf_and_quantile_endpoints():
    raw = dict(angles_deg=[0, 10, 20, 90], mean_angle_deg=30)
    result = compact(raw)
    angles = result["angle_distribution"]
    assert angles["count"] == 4
    assert angles["quantiles_deg"][0] == 0
    assert angles["quantiles_deg"][-1] == 90
    assert angles["ecdf_fraction"][angles["ecdf_degrees"].index(10)] == 0.5
    assert result["mean_angle_deg"] == 30
    assert raw["angles_deg"] == [0, 10, 20, 90]


def test_multiple_sources_merge_disjoint_pairs_and_reject_duplicates(tmp_path):
    sources = [tmp_path / name for name in ("back4", "front4")]
    for directory, name in zip(sources, ("E94", "E103"), strict=True):
        directory.mkdir()
        task = dict(condition=name, tensor="model.layers.0.self_attn.k_proj.weight", points=[[16, "saved"]])
        (directory / "protocol.json").write_text(
            json.dumps(
                dict(tasks=[task], pairs={name: dict(matrix_count=7, risks=[], teacher_parameter_comparison=True)})
            )
        )
        record = dict(task=task, stages=[dict(step=16, singular_values=[3, 2, 1])])
        (directory / "matrix-00000.json.gz").write_bytes(gzip.compress(json.dumps(record).encode()))
    result = collect(sources[0], tmp_path / "data", sources[1:])
    assert set(result["pairs"]) == {"E94", "E103"}
    assert all(len(p["matrices"]) == 1 for p in result["pairs"].values())
    assert not any(p["complete"] for p in result["pairs"].values())
    with pytest.raises(ValueError, match="unique"):
        collect(sources[0], tmp_path / "bad", sources[:1])
    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    (duplicate / "protocol.json").write_bytes((sources[0] / "protocol.json").read_bytes())
    with pytest.raises(ValueError, match="disjoint"):
        collect(sources[0], tmp_path / "bad", [duplicate])
