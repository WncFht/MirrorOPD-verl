"""Paired top-rank task readout and generated-state risk, preserving source hashes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from analyze_generation import summary

ROOT = Path(__file__).resolve().parent


def load(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk-version", default="1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = []
    top = {}
    for label, prior in [("rl", "generation-readout-rl-v1"), ("sft", "generation-readout-v2")]:
        path = ROOT / prior / "endpoint/endpoint/records.jsonl"
        sources.append(path)
        rows = load(path)
        base = {r["prompt_id"]: r for r in rows}
        top[label] = {"endpoint": summary(rows, base)}
        for rank in ["top1", "top5"]:
            path = ROOT / "generation-readout-top-v1" / f"{label}-{rank}" / f"{label}-{rank}" / "records.jsonl"
            sources.append(path)
            rows = load(path)
            assert {r["prompt_id"] for r in rows} == set(base)
            top[label][rank] = summary(rows, base)
    risk = []
    root = ROOT / f"rollout-risk-v{args.risk_version}"
    plan = json.loads((root / "inputs.json").read_text())
    for i, task in enumerate(plan["lanes"]["endpoint"]):
        bankpath = Path(task["records"])
        sources.append(bankpath)
        bank = {str(r["row_id"]): r for r in json.loads(bankpath.read_text())}
        conditions = {}
        for lane in ["endpoint", "patch-off-b1", "random-frames-seed0", "teacher"]:
            path = root / f"{lane}-{i:02d}.json"
            sources.append(path)
            values = json.loads(path.read_text())["rows"]
            conditions[lane] = {str(r["row_id"]): r["loss"] for r in values}
        keys = set(bank)
        assert all(set(v) == keys for v in conditions.values())
        groups = {"all": keys}
        for key in keys:
            row = bank[key]
            for field in ["domain", "full_correct", "full_finish"]:
                if field in row:
                    groups.setdefault(f"{field}={row[field]}", set()).add(key)
        report = {}
        for name, keys in groups.items():
            ids = sorted(keys)
            e = np.array([conditions["endpoint"][k] for k in ids])
            p = np.array([conditions["patch-off-b1"][k] for k in ids])
            delta = p - e
            rng = np.random.default_rng(2026111008)
            boot = rng.integers(0, len(ids), (5000, len(ids)))
            report[name] = dict(
                n=len(ids),
                means={c: float(np.mean([v[k] for k in ids])) for c, v in conditions.items()},
                patch_minus_endpoint=float(delta.mean()),
                paired_bootstrap_95=np.quantile(delta[boot].mean(1), [0.025, 0.975]).tolist(),
                rows=[
                    dict(
                        row_id=k,
                        endpoint=conditions["endpoint"][k],
                        patch=conditions["patch-off-b1"][k],
                        random=conditions["random-frames-seed0"][k],
                        teacher_self=conditions["teacher"][k],
                    )
                    for k in ids
                ],
            )
        risk.append(dict(pair=task["pair"], origin=task["origin"], window=task["window"], groups=report))
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    output.write_text(
        json.dumps(
            dict(
                top=top,
                native_rollout_risk=risk,
                sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            ),
            indent=2,
        )
    )
    print(json.dumps(top, indent=2))


if __name__ == "__main__":
    main()
