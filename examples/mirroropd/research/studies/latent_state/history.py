# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Stream saved training rollouts without inference; preserve producer counters verbatim."""

import argparse
import json
from pathlib import Path

from examples.mirroropd.research.core.io import dump

from .artifacts import digest
from .data import normalize


def iter_history(evaluation):
    import pyarrow.parquet as pq

    paths = sorted((Path(evaluation) / "trajectories").glob("outer-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No training trajectory shards in {evaluation}")
    seen = set()
    for path in paths:
        parquet = pq.ParquetFile(path)
        shard = dict(path=str(path.resolve()), rows=parquet.metadata.num_rows, sha256=digest(path))
        values = []
        for batch in parquet.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                if row["id"] in seen:
                    raise ValueError(f"Duplicate training trajectory ID: {row['id']}")
                seen.add(row["id"])
                normalize(row, dict(group="history"))
                values.append(row)
            yield values, shard
            values = []


def load_history(evaluation):
    """Compatibility convenience for small callers; CLI inventory streams one batch at a time."""
    rows, shards = [], []
    for values, shard in iter_history(evaluation):
        rows.extend(values)
        if not shards or shards[-1]["path"] != shard["path"]:
            shards.append(shard)
    return rows, shards


def inventory(evaluation, export=None):
    shards, outers, prompts = [], set(), set()
    count = tokens = 0
    fields = None
    for rows, shard in iter_history(evaluation):
        if not shards or shards[-1]["path"] != shard["path"]:
            shards.append(shard)
        fields = fields or list(rows[0])
        count += len(rows)
        for row in rows:
            outers.add(row["outer"])
            prompts.add(row["source_id"])
            tokens += sum(row["response_mask"])
            if export:
                export.write(json.dumps(row) + "\n")
    return dict(
        evaluation=str(Path(evaluation).resolve()),
        shards=shards,
        rows=count,
        outers=sorted(outers),
        source_prompts=len(prompts),
        valid_response_tokens=tokens,
        fields=fields,
        hidden_vectors_present=False,
        timing="Current-student rows precede that outer's updates; producer counters are not optimizer updates.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        with (args.output / "rollouts.jsonl").open("x") as stream:
            result = inventory(args.evaluation, export=stream)
        dump(args.output / "inventory.json", result)
    else:
        result = inventory(args.evaluation)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
