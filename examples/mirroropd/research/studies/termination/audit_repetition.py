# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Describe repetition in archived responses without labeling every repeat a loop."""

import argparse
import json
from pathlib import Path

from .replay_repetition import first_repetitions


def repeat_fraction(ids, n):
    windows = len(ids) - n + 1
    if windows <= 0:
        return 0.0
    return 1 - len({tuple(ids[i : i + n]) for i in range(windows)}) / windows


def summarize(rows):
    tokens = sum(r["length"] for r in rows)
    result = {"responses": len(rows), "tokens": tokens}
    for field in ("repeat8", "repeat32", "tail2048_repeat8", "tail2048_repeat32"):
        n = 8 if field.endswith("repeat8") else 32
        weights = [max(0, (min(r["length"], 2048) if field.startswith("tail") else r["length"]) - n + 1) for r in rows]
        result[field] = {
            "response_mean": sum(r[field] for r in rows) / len(rows),
            "pooled_window_fraction": sum(r[field] * w for r, w in zip(rows, weights, strict=True)) / sum(weights),
            "thresholds": {
                str(t): {
                    "responses": sum(r[field] >= t for r in rows),
                    "tokens_in_these_responses": sum(r["length"] for r in rows if r[field] >= t),
                }
                for t in (0.5, 0.8, 0.9, 0.95)
            },
        }
    for name in ("block-8", "block512-8", "block1024-8", "block2048-8"):
        result[name] = {
            "hits": sum(r["detections"][name][1] is not None for r in rows),
            "tokens_saved": sum(r["length"] - r["detections"][name][0] for r in rows),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    previous = [json.loads(line) for line in args.previous.read_text().splitlines()]
    files, rows = {}, []
    for i, old in enumerate(previous):
        source = old["source"]
        if source not in files:
            files[source] = {r["prompt_id"]: r for r in map(json.loads, (args.runs / source).read_text().splitlines())}
        raw = files[source][old["prompt_id"]]
        ids = raw["response_ids"]
        assert len(ids) == old["length"]
        repeated = repeat_fraction(ids, 8)
        assert abs(repeated - old["repeat8"]) < 1e-12
        row = {key: old[key] for key in ("source", "prompt_id", "length", "acc", "finish_reason", "repeat8")}
        row.update(
            repeat32=repeat_fraction(ids, 32),
            tail2048_repeat8=repeat_fraction(ids[-2048:], 8),
            tail2048_repeat32=repeat_fraction(ids[-2048:], 32),
        )
        row["detections"] = {
            name: (old["profiles"][name]["length"], old["profiles"][name]["period"])
            for name in ("block-8", "block512-8")
        }
        row["detections"].update(first_repetitions(ids, {"block1024-8": (16, 1024, 8), "block2048-8": (16, 2048, 8)}))
        rows.append(row)
        if (i + 1) % 128 == 0:
            print(f"Audited {i + 1}/{len(previous)}", flush=True)
    assert len(rows) == 1280
    summary = {
        "all": summarize(rows),
        "cap": summarize([r for r in rows if r["finish_reason"] == "length"]),
        "natural_stop": summarize([r for r in rows if r["finish_reason"] == "stop"]),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
