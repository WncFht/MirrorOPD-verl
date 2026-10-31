# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Small trajectory indices shared by bank selection, history and RKL windows."""

from collections import defaultdict

from .artifacts import ArtifactStore

INDEX_COLUMNS = [
    "id",
    "source_id",
    "dataset",
    "domain",
    "sample_index",
    "protocol_id",
    "prompt_tokens",
    "response_tokens",
    "valid_tokens",
    "producer",
    "outer",
]


def trajectory_index(store, *, bank, prefixes=("generations/", "trajectories/")):
    manifest = store.manifest(complete=True)
    result = []
    for name in sorted(manifest["files"]):
        if not name.startswith(prefixes):
            continue
        for index, row in enumerate(store.iter_rows(name, columns=INDEX_COLUMNS)):
            result.append(
                row
                | {
                    "artifact": str(store.root),
                    "shard": name,
                    "row_index": index,
                    "bank": bank,
                    "trajectory_id": row["id"],
                    "id": f"{bank}:{row['id']}",
                }
            )
    if not result or len({row["id"] for row in result}) != len(result):
        raise ValueError("Empty or duplicate trajectory index")
    return result


def load_trajectories(references):
    """Read only row groups belonging to this scoring window, in requested order."""
    groups = defaultdict(list)
    output = [None] * len(references)
    for position, ref in enumerate(references):
        groups[ref["artifact"], ref["shard"]].append((position, ref))
    for (artifact, shard), items in groups.items():
        rows = ArtifactStore(artifact).read_rows(shard, indices=[ref["row_index"] for _, ref in items])
        for (position, ref), row in zip(items, rows, strict=True):
            if row["id"] != ref.get("trajectory_id", ref["id"]) or row["source_id"] != ref["source_id"]:
                raise ValueError("Trajectory reference identity mismatch")
            output[position] = row | {
                key: ref[key]
                for key in (
                    "id",
                    "bank",
                    "trajectory_id",
                    "history_batch",
                    "history_quota",
                )
                if key in ref
            }
    return output


def select_panel(references, specs):
    expected = {(d.name, qid) for d in specs for qid in d.panel_ids}
    selected = [r for r in references if r["sample_index"] == 0 and (r["dataset"], r["source_id"]) in expected]
    if {(r["dataset"], r["source_id"]) for r in selected} != expected or len(selected) != len(expected):
        raise ValueError("Generation artifact does not cover every selected panel question/sample0")
    return selected
