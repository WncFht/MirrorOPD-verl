# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Recover published frozen-model samples; never resume an optimizer or resample a saved answer."""

from .artifacts import GENERATION, ArtifactStore, file_sha256, identity
from .generation_contract import sample_identity


def replay_published(destination, previous, specifications, protocols, *, on_batch=None, progress=None):
    source = ArtifactStore(previous)
    saved = source.manifest()
    expected = destination.manifest()["metadata"]
    if saved["kind"] != "generation" or expected["outer"] != 0 or expected["expected_version"] is not None:
        raise ValueError("Only frozen bootstrap generation can reuse published samples")
    fields = (
        "producer",
        "model_identity",
        "outer",
        "protocols",
        "expected_samples",
        "expected_version",
        "question_set_sha256",
        "engine_contract_id",
        "engine_contract",
        "sample_identity_schema",
    )

    def comparable(meta, key):
        value = meta.get(key)
        if key == "protocols" and value is not None:
            return {
                name: {k: v for k, v in protocol.items() if k != "answer_channel"} for name, protocol in value.items()
            }
        return value

    if any(identity(comparable(saved["metadata"], k)) != identity(comparable(expected, k)) for k in fields):
        raise ValueError("Resume model, questions, sampling counts or input protocol changed")
    source_hash = file_sha256(source.manifest_path)
    questions = {(s.name, q["id"]): (q, s, protocols[s.protocol]) for s, rows in specifications for q in rows}
    seen, ids, shard = set(), set(), 0
    for name in sorted(saved["files"]):
        if not name.startswith("generations/"):
            raise ValueError("Unknown frozen generation table")
        rows = source.read_rows(name)
        outputs = []
        for row in rows:
            q, spec, protocol = questions[row["dataset"], row["source_id"]]
            key = row["dataset"], row["source_id"], row["sample_index"]
            if key in seen or row["id"] in ids or not 0 <= row["sample_index"] < spec.n:
                raise ValueError("Duplicate or out-of-range frozen sample")
            if row["protocol_id"] != protocol.protocol_id or row["seed"] != protocol.sample_seed(*key):
                raise ValueError("Frozen sample seed/protocol differs")
            expected_id = sample_identity(
                expected["model_identity"],
                0,
                spec.name,
                q["id"],
                row["sample_index"],
                protocol.protocol_id,
                expected["engine_contract_id"],
            )
            if row["id"] != expected_id or row["outer"] != 0:
                raise ValueError("Frozen sample identity does not match the original model and request")
            seen.add(key)
            ids.add(row["id"])
            outputs.append((row, q, spec, protocol))
        destination.write_rows(f"generations/part-{shard:05d}.parquet", rows, schema=GENERATION)
        shard += 1
        if on_batch is not None:
            on_batch(outputs)
        if progress is not None:
            progress({"phase": "generation_reuse", "completed": len(seen), "expected": expected["expected_samples"]})
    if file_sha256(source.manifest_path) != source_hash:
        raise ValueError("Source generation was modified during recovery")
    return (
        seen,
        shard,
        {"resumed_from": str(source.root), "resumed_manifest_sha256": source_hash, "reused_samples": len(seen)},
    )
