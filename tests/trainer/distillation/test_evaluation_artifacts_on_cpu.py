# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Publication, selective loading and integrity of real compressed Parquet artifacts."""

import json
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
import pytest

from verl.trainer.evaluation.artifacts import HISTORY, TRAJECTORY, ArtifactStore


def rows():
    return [
        dict(
            id=f"1:{i}",
            source_id=str(i),
            outer=1,
            prompt_ids=[151644, i],
            response_ids=list(range(i + 1)),
            response_mask=[1] * i + [0],
            finish_reason="length",
        )
        for i in range(8)
    ]


def test_roundtrip_and_lightweight_history_preserve_identity_order(tmp_path):
    store = ArtifactStore.create(tmp_path, kind="run", metadata={"model": "revision-a"})
    original = rows()
    store.write_rows("training/one.parquet", iter(original), schema=TRAJECTORY, batch_rows=2)
    assert pq.ParquetFile(tmp_path / "training/one.parquet").num_row_groups == 4
    selected = store.read_rows("training/one.parquet", indices=[7, 1, 7], columns=["id", "response_ids"])
    assert [r["id"] for r in selected] == ["1:7", "1:1", "1:7"]
    assert selected[0]["response_ids"] == original[7]["response_ids"]
    references = [
        dict(
            id=f"1:{i}",
            source_id=str(i),
            outer=1,
            shard="training/one.parquet",
            row_index=i,
            history_batch=1,
            history_quota=2,
        )
        for i in (7, 1)
    ]
    store.write_rows("history/one.parquet", references, schema=HISTORY)
    assert "response_ids" not in pq.read_schema(tmp_path / "history/one.parquet").names
    recovered = store.resolve(store.read_rows("history/one.parquet"))
    assert [r["response_ids"] for r in recovered] == [original[i]["response_ids"] for i in (7, 1)]
    assert recovered[0]["response_mask"] == [True] * 7 + [False]
    assert all(r["bank"] == "history" for r in recovered)
    assert "prompt_ids" not in json.dumps(store.manifest())
    references[0]["id"] = "wrong"
    with pytest.raises(ValueError, match="reference"):
        store.resolve(references)


def test_publication_and_hash_validation(tmp_path):
    store = ArtifactStore.create(tmp_path, kind="bank", metadata={})
    with pytest.raises(ValueError, match="Incomplete"):
        store.manifest(complete=True)
    store.write_rows("bank.parquet", rows(), schema=TRAJECTORY)
    with pytest.raises(FileExistsError):
        store.write_rows("bank.parquet", rows(), schema=TRAJECTORY)
    store.finish()
    assert store.manifest(complete=True)["status"] == "complete"
    with pytest.raises(FileExistsError):
        store.write_rows("other.parquet", rows(), schema=TRAJECTORY)
    with pytest.raises(FileExistsError):
        ArtifactStore.create(tmp_path, kind="bank", metadata={})
    path = tmp_path / "bank.parquet"
    raw = path.read_bytes()
    path.write_bytes(raw[:20] + bytes([raw[20] ^ 1]) + raw[21:])
    with pytest.raises(ValueError, match="content changed"):
        store.read_rows("bank.parquet")


def test_partial_write_is_not_published_and_paths_are_confined(tmp_path):
    store = ArtifactStore.create(tmp_path, kind="run", metadata={})

    def broken():
        yield rows()[0]
        raise RuntimeError("producer failed")

    with pytest.raises(RuntimeError, match="producer failed"):
        store.write_rows("partial.parquet", broken(), schema=TRAJECTORY, batch_rows=1)
    assert store.manifest()["files"] == {}
    assert not list(tmp_path.glob("*.parquet"))
    assert not list(tmp_path.glob("*.tmp"))
    for name in ("../escape.parquet", "/tmp/escape.parquet"):
        with pytest.raises(ValueError, match="inside"):
            store.write_rows(name, rows(), schema=TRAJECTORY)
    with pytest.raises(ValueError, match="Unknown columns"):
        store.write_rows("bad.parquet", [rows()[0] | {"teacher_hidden": [1]}], schema=TRAJECTORY)
    with pytest.raises(ValueError, match="Invalid trajectory"):
        store.write_rows("bad.parquet", [rows()[0] | {"response_mask": [2]}], schema=TRAJECTORY)
    with pytest.raises(ValueError, match="int32"):
        store.write_rows("bad.parquet", [rows()[0] | {"response_ids": [1.5]}], schema=TRAJECTORY)
    with pytest.raises(ValueError, match="Duplicate"):
        store.write_rows("bad.parquet", [rows()[0], rows()[0]], schema=TRAJECTORY)


def test_missing_and_invalid_references_do_not_read_another_row(tmp_path):
    store = ArtifactStore.create(tmp_path, kind="run", metadata={})
    store.write_rows("one.parquet", rows(), schema=TRAJECTORY)
    for indices in ([-1], [8], [True]):
        with pytest.raises(ValueError, match="row index"):
            store.read_rows("one.parquet", indices=indices)
    with pytest.raises(KeyError):
        store.read_rows("unpublished.parquet")


def test_independent_publishers_do_not_lose_manifest_entries(tmp_path):
    store = ArtifactStore.create(tmp_path, kind="run", metadata={})

    def publish(index):
        return ArtifactStore(tmp_path).write_rows(f"batch-{index}.parquet", rows(), schema=TRAJECTORY)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(publish, range(4)))
    assert len(store.manifest()["files"]) == 4
    assert all(result["rows"] == 8 for result in results)
    store.finish()
    assert store.manifest(complete=True)["status"] == "complete"
