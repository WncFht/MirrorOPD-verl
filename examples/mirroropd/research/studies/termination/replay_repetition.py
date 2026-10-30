# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU replay of vLLM's exact contiguous-token repetition stop, without generation.

Find the earliest stop for every profile, regrade the retained prefix, and retain
sample-level evidence. Token savings are counterfactual on recorded trajectories,
not measured wall-time speedups. EOS and the response cap precede repetition.
"""

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROFILES = {
    "short-8": (1, 32, 8),
    "block64-8": (16, 64, 8),
    "block128-8": (16, 128, 8),
    "block-4": (16, 128, 4),
    "block256-4": (16, 256, 4),
    "block-6": (16, 256, 6),
    "block-8": (16, 256, 8),
    "block-16": (16, 256, 16),
    "block512-8": (16, 512, 8),
}


def benchmark_detector():
    """Measure native CPU checks on synthetic non-repeating tokens, not decode time."""
    import random
    import time

    from vllm.sampling_params import RepetitionDetectionParams
    from vllm.v1.core.sched.utils import check_sequence_repetition
    from vllm.v1.utils import ConstantList

    rng = random.Random(42)
    ids = ConstantList([rng.randrange(151936) for _ in range(8192)])
    results = {}
    for upper in (32, 64, 128, 256, 512):
        params = RepetitionDetectionParams(min_pattern_size=16, max_pattern_size=upper, min_count=8)
        trials = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(2000):
                check_sequence_repetition(ids, params)
            trials.append((time.perf_counter() - start) / 2000 * 1e6)
        results[str(upper)] = {"us_per_check_median": sorted(trials)[2], "trials_us": trials}
    return {
        "method": "native check_sequence_repetition; ConstantList of 8192 uniform random token IDs; "
        "seed42; 5 x 2000 calls; no GPU; synthetic negative case, not measured decode overhead",
        "results": results,
    }


def first_repetitions(ids, profiles=PROFILES):
    """Exact first suffix match, vectorized over positions for offline replay only.

    For period p, c identical adjacent blocks require (c-1)*p consecutive
    equalities ids[i] == ids[i-p]. The online engine retains its native detector.
    """
    tokens = np.asarray(ids, dtype=np.int64)
    # Existing cap/EOS at the final position wins over repetition.
    result = {name: (len(ids), None) for name in profiles}
    for period in range(1, max(v[1] for v in profiles.values()) + 1):
        if period >= len(tokens):
            break
        equal = tokens[period:] == tokens[:-period]
        edges = np.diff(np.concatenate(([False], equal, [False])).astype(np.int8))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        lengths = ends - starts
        for name, (minimum, maximum, count) in profiles.items():
            if not minimum <= period <= maximum:
                continue
            needed = period * (count - 1)
            eligible = np.flatnonzero(lengths >= needed)
            if len(eligible):
                stop = int(period + starts[eligible[0]] + needed)
                if stop < result[name][0]:
                    result[name] = (stop, period)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    from examples.mirroropd.research.core.math_grading import grade, init_grader

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    sources = sorted(
        p
        for run in ("e54-topd-mdpo-20261029", "e54-mdpo-sweep-20261029")
        for p in (args.runs / run / "evaluation").glob("*/scored.jsonl")
    )
    assert len(sources) == 20, f"Expected 20 evaluation files, got {len(sources)}"
    rows, jobs, job_indices = [], [], {}
    for path in sources:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            ids = row["response_ids"]
            assert len(ids) == row["response_length"] and row["finish_reason"] in {"stop", "length"}
            assert not set(ids[:-1]) & {151643, 151645}
            replay = first_repetitions(ids)
            entry = {
                "source": str(path.relative_to(args.runs)),
                "prompt_id": row["prompt_id"],
                "data_source": row["data_source"],
                "length": len(ids),
                "acc": bool(row["acc"]),
                "finish_reason": row["finish_reason"],
                "repeat8": row["repeat8"],
                "profiles": {},
            }
            if any(period is not None for _, period in replay.values()):
                entry["baseline_grade_index"] = len(jobs)
                jobs.append({key: row[key] for key in ("output", "response_ids", "gts", "data_source")})
            for name, (stop, period) in replay.items():
                record = {"length": stop, "period": period, "acc": bool(row["acc"])}
                if period is not None:
                    key = (len(rows), stop)
                    if key not in job_indices:
                        job_indices[key] = len(jobs)
                        prefix = tokenizer.decode(ids[:stop], skip_special_tokens=False)
                        jobs.append(
                            {
                                "output": prefix,
                                "response_ids": ids[:stop],
                                "gts": row["gts"],
                                "data_source": row["data_source"],
                            }
                        )
                    record["grade_index"] = job_indices[key]
                    record["pattern"] = tokenizer.decode(ids[stop - period : stop], skip_special_tokens=False)
                    record["prefix_tail"] = tokenizer.decode(ids[max(0, stop - 256) : stop], skip_special_tokens=False)
                    record["original_tail"] = row["output"][-1200:]
                entry["profiles"][name] = record
            rows.append(entry)
    assert len(rows) == 1280
    print(f"Replayed {len(rows)} responses; grading {len(jobs)} prefixes and original responses", flush=True)
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn"), initializer=init_grader
    ) as pool:
        scores = list(pool.map(grade, jobs))
    for row in rows:
        if "baseline_grade_index" in row:
            row["regraded_original_acc"] = bool(scores[row.pop("baseline_grade_index")]["acc"])
        for record in row["profiles"].values():
            if "grade_index" in record:
                record["acc"] = bool(scores[record.pop("grade_index")]["acc"])
    summary = {
        "responses": len(rows),
        "original_correct": sum(r["acc"] for r in rows),
        "original_tokens": sum(r["length"] for r in rows),
        "profiles": {},
    }
    summary["baseline_regrade_disagreements"] = sum(r["acc"] != r.get("regraded_original_acc", r["acc"]) for r in rows)
    for name, params in PROFILES.items():
        hit = [r for r in rows if r["profiles"][name]["period"] is not None]
        summary["profiles"][name] = {
            "min_pattern_size": params[0],
            "max_pattern_size": params[1],
            "min_count": params[2],
            "hits": len(hit),
            "hit_original_stop": sum(r["finish_reason"] == "stop" for r in hit),
            "hit_original_cap": sum(r["finish_reason"] == "length" for r in hit),
            "correct_lost": sum(r["acc"] and not r["profiles"][name]["acc"] for r in hit),
            "correct_gained": sum(not r["acc"] and r["profiles"][name]["acc"] for r in hit),
            "original_correct_hit": sum(r["acc"] for r in hit),
            "tokens_saved": sum(r["length"] - r["profiles"][name]["length"] for r in rows),
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
