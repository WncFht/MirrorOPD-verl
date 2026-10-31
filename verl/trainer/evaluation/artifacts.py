# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Single-node Parquet artifacts; one manifest publishes immutable table files.

This is a local library, not a transport, database, or training checkpoint.
File locks serialize publishers. Atomic renames provide publication visibility,
not a promise of power-loss durability on arbitrary mounted filesystems.
"""

import fcntl
import hashlib
import itertools
import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from numbers import Integral
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _schema(fields):
    return pa.schema(fields, metadata={b"verl.evaluation.schema_version": b"1"})


TRAJECTORY = _schema(
    [
        ("id", pa.string()),
        ("source_id", pa.string()),
        ("outer", pa.int64()),
        ("prompt_ids", pa.list_(pa.int32())),
        ("response_ids", pa.list_(pa.int32())),
        ("response_mask", pa.list_(pa.bool_())),
        ("rollout_key", pa.string()),
        ("min_global_steps", pa.int64()),
        ("max_global_steps", pa.int64()),
        ("data_source", pa.string()),
        ("finish_reason", pa.string()),
        ("bank", pa.string()),
    ]
)
HISTORY = _schema(
    [
        ("id", pa.string()),
        ("source_id", pa.string()),
        ("outer", pa.int64()),
        ("shard", pa.string()),
        ("row_index", pa.int64()),
        ("history_batch", pa.int64()),
        ("history_quota", pa.int64()),
    ]
)
RKL = _schema(
    [
        ("id", pa.string()),
        ("source_id", pa.string()),
        ("bank", pa.string()),
        ("history_batch", pa.int64()),
        ("rkl_sum", pa.float64()),
        ("tokens", pa.int64()),
    ]
)

GENERATION = _schema(
    list(TRAJECTORY)
    + [
        ("dataset", pa.string()),
        ("domain", pa.string()),
        ("sample_index", pa.int32()),
        ("protocol_id", pa.string()),
        ("seed", pa.int64()),
        ("producer", pa.string()),
        ("prompt_tokens", pa.int32()),
        ("response_tokens", pa.int32()),
        ("valid_tokens", pa.int32()),
        ("generation_seconds", pa.float64()),
        ("thinking_open", pa.bool_()),
    ]
)
EVAL_RKL = _schema(
    list(RKL)
    + [
        ("dataset", pa.string()),
        ("domain", pa.string()),
        ("sample_index", pa.int32()),
        ("trajectory_id", pa.string()),
    ]
)
ACCURACY = _schema(
    [
        ("id", pa.string()),
        ("source_id", pa.string()),
        ("dataset", pa.string()),
        ("domain", pa.string()),
        ("sample_index", pa.int32()),
        ("score", pa.float64()),
        ("status", pa.string()),
        ("answer", pa.string()),
        ("seconds", pa.float64()),
        ("finish_reason", pa.string()),
        ("response_tokens", pa.int32()),
        ("instructions_correct", pa.int32()),
        ("instructions_total", pa.int32()),
    ]
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ArtifactStore:
    """Publish bounded Parquet batches and resolve lightweight trajectory references.

    Files are append-only and relative to root. Readers only use files listed in
    the manifest, including while a run is active. Reusable banks and score events
    additionally require a complete manifest. No token arrays belong in metadata.
    """

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest_path = self.root / "manifest.json"

    @contextmanager
    def _lock(self):
        with (self.root / ".writer.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    @classmethod
    def create(cls, root, *, kind, metadata):
        store = cls(root)
        store.root.mkdir(parents=True, exist_ok=True)
        with store._lock():
            if any(p.name != ".writer.lock" for p in store.root.iterdir()):
                raise FileExistsError(f"Artifact directory is not empty: {store.root}")
            _atomic_json(
                store.manifest_path,
                {
                    "schema_version": 1,
                    "kind": kind,
                    "artifact_id": uuid.uuid4().hex,
                    "status": "active",
                    "metadata": metadata,
                    "files": {},
                },
            )
        return store

    def manifest(self, *, complete=False):
        value = json.loads(self.manifest_path.read_text())
        if value.get("schema_version") != 1:
            raise ValueError("Unsupported evaluation artifact schema")
        if complete and value["status"] != "complete":
            raise ValueError(f"Incomplete artifact: {self.root}")
        return value

    def _path(self, name):
        relative = Path(name)
        path = (self.root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(self.root):
            raise ValueError(f"Artifact path must stay inside its root: {name}")
        return path

    def write_rows(self, name, rows, *, schema, batch_rows=32):
        """Stream rows to one immutable shard, then publish its hash and row count."""
        if batch_rows < 1 or Path(name).suffix != ".parquet":
            raise ValueError("Positive write batch size and a Parquet filename required")
        destination = self._path(name)
        with self._lock():
            manifest = self.manifest()
            if manifest["status"] != "active" or name in manifest["files"] or destination.exists():
                raise FileExistsError(f"Cannot overwrite an artifact table: {name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            count = 0
            seen_ids = set()
            try:
                iterator = iter(rows)
                with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
                    while batch := list(itertools.islice(iterator, batch_rows)):
                        normalized = []
                        for original in batch:
                            row = dict(original)
                            if set(row) - set(schema.names):
                                raise ValueError(f"Unknown columns: {set(row) - set(schema.names)}")
                            if schema.equals(TRAJECTORY) or schema.equals(GENERATION):
                                mask = row.get("response_mask", [])
                                if (
                                    not row.get("id")
                                    or not row.get("source_id")
                                    or not row.get("prompt_ids")
                                    or (not row.get("response_ids") and schema.equals(TRAJECTORY))
                                    or len(mask) != len(row["response_ids"])
                                    or any(x not in (0, 1, False, True) for x in mask)
                                ):
                                    raise ValueError("Invalid trajectory identity, tokens or response mask")
                                if row["id"] in seen_ids:
                                    raise ValueError("Duplicate trajectory ID")
                                seen_ids.add(row["id"])
                                if any(
                                    not isinstance(token, Integral) or isinstance(token, bool) or not 0 <= token < 2**31
                                    for field in ("prompt_ids", "response_ids")
                                    for token in row[field]
                                ):
                                    raise ValueError("Raw token IDs must be nonnegative int32 values")
                                row["response_mask"] = [bool(x) for x in mask]
                            normalized.append(row)
                        writer.write_table(pa.Table.from_pylist(normalized, schema=schema), row_group_size=batch_rows)
                        count += len(batch)
                entry = {"rows": count, "bytes": temporary.stat().st_size, "sha256": file_sha256(temporary)}
                os.replace(temporary, destination)
                manifest["files"][name] = entry
                _atomic_json(self.manifest_path, manifest)
                return entry
            finally:
                temporary.unlink(missing_ok=True)

    def finish(self, *, metadata=None):
        """Publish completion only after the caller's scoring/audit/cleanup succeeded."""
        with self._lock():
            manifest = self.manifest()
            if manifest["status"] != "active":
                raise ValueError("Artifact already completed")
            for name in manifest["files"]:
                self._verified_path(name, manifest)
            manifest["metadata"].update(metadata or {})
            manifest["status"] = "complete"
            _atomic_json(self.manifest_path, manifest)

    def _verified_path(self, name, manifest=None):
        entry = (manifest or self.manifest())["files"][name]
        path = self._path(name)
        if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Artifact content changed: {name}")
        return path

    def read_rows(self, name, *, indices=None, columns=None):
        """Read selected row groups, preserving requested order without loading a whole bank."""
        manifest = self.manifest()
        parquet = pq.ParquetFile(self._verified_path(name, manifest))
        size = manifest["files"][name]["rows"]
        if parquet.metadata.num_rows != size:
            raise ValueError("Parquet row count differs from manifest")
        if indices is None:
            return parquet.read(columns=columns).to_pylist()
        indices = list(indices)
        if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= size for i in indices):
            raise ValueError("Invalid artifact row index")
        wanted, selected, offset = set(indices), {}, 0
        for group in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group).num_rows
            local = [i for i in wanted if offset <= i < offset + count]
            if local:
                table = parquet.read_row_group(group, columns=columns)
                values = table.take(pa.array([i - offset for i in local], type=pa.int64())).to_pylist()
                selected.update(zip(local, values, strict=True))
            offset += count
        return [selected[i] for i in indices]

    def iter_rows(self, name, *, columns=None, batch_rows=32):
        """Stream bounded decoded batches rather than a complete generation table."""
        if batch_rows < 1:
            raise ValueError("Positive read batch size required")
        file = pq.ParquetFile(self._verified_path(name))
        for batch in file.iter_batches(batch_size=batch_rows, columns=columns):
            yield from batch.to_pylist()

    def write_shards(self, prefix, rows, *, schema, max_rows=32, max_bytes=8 * 1024**2):
        """Bound application-side buffering by rows and estimated raw token bytes."""
        if max_rows < 1 or max_bytes < 1:
            raise ValueError("Positive shard limits required")
        chunk, size, names = [], 0, []

        def flush():
            name = f"{prefix}/part-{len(names):05d}.parquet"
            self.write_rows(name, chunk, schema=schema, batch_rows=min(max_rows, 32))
            names.append(name)

        for row in rows:
            cost = 4 * (len(row.get("prompt_ids", ())) + len(row.get("response_ids", ())))
            cost += len(row.get("response_mask", ())) + 1024
            if cost > max_bytes:
                raise ValueError("A complete trajectory exceeds the storage shard budget")
            if chunk and (len(chunk) >= max_rows or size + cost > max_bytes):
                flush()
                chunk, size = [], 0
            chunk.append(row)
            size += cost
        if chunk:
            flush()
        return names

    def resolve(self, references):
        """Resolve history references in shard groups and verify logical identity."""
        groups, result = defaultdict(list), [None] * len(references)
        for position, ref in enumerate(references):
            groups[ref["shard"]].append((position, ref))
        for shard, items in groups.items():
            rows = self.read_rows(shard, indices=[ref["row_index"] for _, ref in items])
            for (position, ref), row in zip(items, rows, strict=True):
                if row["id"] != ref["id"] or row["source_id"] != ref["source_id"]:
                    raise ValueError("History reference does not match its trajectory")
                result[position] = row | {
                    "bank": "history",
                    "history_batch": ref["history_batch"],
                    "history_quota": ref["history_quota"],
                }
        return result
