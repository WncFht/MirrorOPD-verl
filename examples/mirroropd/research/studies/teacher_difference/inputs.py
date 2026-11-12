# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Load and validate frozen state records without changing their sampling."""

import json
from pathlib import Path

from examples.mirroropd.research.core.artifacts import fingerprint

REQUIRED = ("prompt_ids", "response_ids")


def _read(path: Path):
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    value = json.loads(path.read_text())
    if isinstance(value, dict):
        for key in ("records", "rows", "data"):
            if isinstance(value.get(key), list):
                return value[key]
        raise ValueError(f"{path} contains an object, not a record list")
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list")
    return value


def load_records(path: str | Path):
    """Read one JSON/JSONL file, or all JSON files in a directory in name order."""
    path = Path(path)
    if path.is_file():
        return _read(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted((*path.glob("*.json"), *path.glob("*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No JSON/JSONL records under {path}")
    rows = []
    for file in files:
        rows.extend(_read(file))
    return rows


def normalize_records(rows, *, cap: int | None = None, bank_id: str = "default"):
    """Copy records, truncate only after EOS, and add stable diagnostic metadata."""
    if cap is not None and cap < 1:
        raise ValueError("cap must be positive")
    out = []
    seen = set()
    row_ids = set()
    for index, source in enumerate(rows):
        row = dict(source)
        for key in REQUIRED:
            if key not in row or not isinstance(row[key], list):
                raise ValueError(f"record {index} is missing list field {key}")
        prompt_ids = [int(x) for x in row["prompt_ids"]]
        response_ids = [int(x) for x in row["response_ids"]]
        if not prompt_ids:
            raise ValueError(f"record {index} has an empty prompt")
        eos = row.get("eos_token_ids", row.get("eos_token_id"))
        eos_ids = set() if eos is None else {int(x) for x in (eos if isinstance(eos, list) else [eos])}
        stop = next((i for i, token in enumerate(response_ids) if token in eos_ids), None)
        if stop is not None:
            response_ids = response_ids[: stop + 1]
        if cap is not None:
            response_ids = response_ids[:cap]
        if not response_ids:
            raise ValueError(f"record {index} has an empty response after truncation")
        key = tuple(prompt_ids)
        if key in seen:
            raise ValueError(f"duplicate prompt_ids in bank {bank_id}: record {index}")
        seen.add(key)
        row.update(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            row_id=str(row.get("row_id", row.get("id", index))),
            bank_id=bank_id,
            token_count=len(response_ids),
        )
        if row["row_id"] in row_ids:
            raise ValueError(f"duplicate row_id in bank {bank_id}: {row['row_id']}")
        row_ids.add(row["row_id"])
        out.append(row)
    return out


def validate_records(rows, *, expected=None):
    if not rows:
        raise ValueError("empty record bank")
    ids = [row["row_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("row_id is not unique")
    if expected is not None and len(rows) != expected:
        raise ValueError(f"expected {expected} records, found {len(rows)}")
    for row in rows:
        if row["token_count"] != len(row["response_ids"]):
            raise ValueError(f"token_count mismatch for {row['row_id']}")
        if any(not isinstance(token, int) for token in row["prompt_ids"] + row["response_ids"]):
            raise TypeError(f"non-integer token in {row['row_id']}")
    return {"records": len(rows), "response_tokens": sum(row["token_count"] for row in rows)}


def source_fingerprint(rows):
    return fingerprint(rows)


def validate_source_policy(policy):
    if policy not in ("training_batches", "independent"):
        raise ValueError(f"Unknown source policy: {policy!r}")
    return policy


def future_source_groups(task, rows):
    """Only observed training-batch provenance supports a future-batch claim.

    An independent diagnostic source has no training exposure clock. It is
    included in all-train summaries, never relabelled as a future batch.
    """
    if validate_source_policy(task.get("source_policy", "training_batches")) == "independent":
        return set()
    groups = {}
    for row in rows:
        if row["split"] != "train":
            continue
        key = row.get("diagnostic_group", row.get("group"))
        if key is None or "batch_index" not in row:
            raise ValueError("training_batches requires group and batch_index; use independent for a new source panel")
        unseen = row["batch_index"] >= task["stage"] // 16
        groups[key] = groups.get(key, True) and unseen
    return {key for key, unseen in groups.items() if unseen}


def validate_source_exposure(task, rows):
    if validate_source_policy(task.get("source_policy", "training_batches")) == "independent":
        return
    future = future_source_groups(task, rows)
    if any(row.get("diagnostic_group", row.get("group")) not in future for row in rows):
        raise ValueError("Source must be unseen at checkpoint")
