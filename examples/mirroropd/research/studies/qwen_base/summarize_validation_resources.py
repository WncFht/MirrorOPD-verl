# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0


def main():
    import argparse
    import json
    import statistics
    import time
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Summarize recorded validation resource samples")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    r = args.run
    metrics = [json.loads(line)["data"] for line in (r / "metrics.jsonl").read_text().splitlines()]
    vals = [d for d in metrics if "eval/math500/acc/mean" in d]
    timers = [
        json.loads(line)
        for line in (r / "timeline/events-1745308.jsonl").read_text().splitlines()
        if '"name": "timer/testing"' in line
    ]
    timers = [d for d in timers if d["status"] == "ok"][: len(vals)]
    samples = [json.loads(line) for line in (r / "gpu-samples.jsonl").read_text().splitlines()]
    result = []
    for d, t in zip(vals, timers, strict=False):
        end = t["wall_ns"] / 1e9
        start = end - (t["end_ns"] - t["start_ns"]) / 1e9
        data = []
        for s in samples:
            if start <= s["time"] <= end and s.get("returncode") == 0:
                g = []
                for line in s["gpus"]:
                    try:
                        index, u, m = [float(x.strip()) for x in line.split(",")]
                        g.append({"index": int(index), "util": u, "gib": m / 1024})
                    except ValueError:
                        pass
                if len(g) == 8:
                    data.append(g)
        flat = [v for g in data for v in g]
        if not flat:
            continue
        out = {
            "outer": d["step/outer"],
            "start": start,
            "end": end,
            "minutes": (end - start) / 60,
            "samples": len(data),
            "mean_gpu_util_pct": statistics.mean(v["util"] for v in flat),
            "gpu_util_pct_by_gpu": [statistics.mean(g[i]["util"] for g in data) for i in range(8)],
            "low_util_fraction": sum(v["util"] < 50 for v in flat) / len(flat),
            "mean_mem_gib": statistics.mean(v["gib"] for v in flat),
            "min_mem_gib": min(v["gib"] for v in flat),
            "max_mem_gib": max(v["gib"] for v in flat),
            "mem_gib_peak_by_gpu": [max(g[i]["gib"] for g in data) for i in range(8)],
            "output_tokens": sum(
                d[f"eval/{n}/generation/response_length/mean"] * d[f"eval/{n}/response_count"]
                for n in ["math500", "amc23", "aime25"]
            ),
        }
        out["output_tokens_per_gpu_second"] = out["output_tokens"] / (end - start) / 8
        result.append(out)
        print(
            out["outer"],
            round(out["minutes"], 1),
            out["samples"],
            round(out["mean_gpu_util_pct"], 1),
            round(out["mean_mem_gib"], 2),
            round(out["max_mem_gib"], 2),
            round(100 * out["low_util_fraction"], 1),
            round(out["output_tokens_per_gpu_second"]),
        )
    args.out.write_text(
        json.dumps(
            {
                "observed_at": time.time(),
                ("sampling"): (
                    "Approximately every 30 seconds; sample/card-weighted means; "
                    "sampled memory peaks; full validation windows including tail"
                ),
                "rows": result,
            },
            indent=2,
        )
        + "\n"
    )
    print("LAST", json.dumps(result[-1]))


if __name__ == "__main__":
    main()
