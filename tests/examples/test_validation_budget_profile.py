# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from examples.mirroropd.launchers.prepare_validation_budget import ROLLOUTS, prepare


def test_math4_rest16_prepares_per_benchmark_rollout_counts_without_mutating_sources(tmp_path):
    inputs = []
    for name in ROLLOUTS:
        path = tmp_path / "source" / name / "test.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.table({"data_source": [name, name], "prompt": [[], []]}), path)
        inputs.append(path)

    output = tmp_path / "run" / "validation-budget" / "math4-rest16"
    manifest = prepare(output, inputs)

    assert {name: entry["rollout_n"] for name, entry in manifest["datasets"].items()} == ROLLOUTS
    for name in ROLLOUTS:
        source = pq.read_table(tmp_path / "source" / name / "test.parquet")
        prepared = pq.read_table(output / name / "test.parquet")
        assert "__rollout_n__" not in source.column_names
        assert prepared.column("__rollout_n__").to_pylist() == [ROLLOUTS[name]] * 2
        assert prepared.column("data_source").to_pylist() == [name, name]


def test_math4_rest16_rejects_unknown_or_duplicate_benchmarks(tmp_path):
    path = tmp_path / "test.parquet"
    pq.write_table(pa.table({"data_source": ["unknown"]}), path)
    with pytest.raises(ValueError, match="Unexpected or duplicate"):
        prepare(tmp_path / "output", [path] * len(ROLLOUTS))
