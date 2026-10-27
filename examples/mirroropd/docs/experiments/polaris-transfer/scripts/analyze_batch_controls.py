#!/usr/bin/env python3
"""Paired fixed/fresh contrasts and old-batch retention checks."""

import argparse
import collections
import json
import random
import statistics as st
from pathlib import Path


def read(p):
    return json.loads(p.read_text())


def dump(p, x):
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False) + "\n")


def ci(xs):
    rng = random.Random(42)
    values = sorted(st.mean(rng.choices(xs, k=len(xs))) for _ in range(2000))
    return [values[49], values[1949]]


def index(rows):
    return {r["id"]: r for r in rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    a = p.parse_args()
    root = a.data
    pairs = [
        ("polaris-roundrobin", "polaris-blocked", 128),
        ("polaris-mixed", "polaris-blocked", 128),
        ("polaris-rr-lr3", "polaris-roundrobin", 128),
        ("polaris-rich-lr1", "polaris-roundrobin", 128),
        ("polaris-rich-lr3", "polaris-rr-lr3", 128),
        ("polaris-rich-lr3", "polaris-rich-lr1", 128),
        ("warm-forward", "warm-reverse", 32),
        ("warm-js", "warm-reverse", 32),
    ]
    comparisons = []
    for left, right, step in pairs:
        l = index(read(root / left / f"fixed-{step}.json"))
        r = index(read(root / right / f"fixed-{step}.json"))
        assert l.keys() == r.keys()
        for domain in ["math", "code", "if"]:
            ids = [i for i in l if l[i]["split"] == "heldout" and l[i]["domain"] == domain]
            for metric in ["rkl", "fkl_shared", "js", "tv"]:
                delta = [(l[i]["stats"]["all"][metric] - r[i]["stats"]["all"][metric]) / 1024 for i in ids]
                comparisons.append(
                    dict(
                        left=left,
                        right=right,
                        step=step,
                        domain=domain,
                        measure="fixed",
                        metric=metric,
                        mean_difference=st.mean(delta),
                        paired_prompt_ci=ci(delta),
                        prompts=len(ids),
                    )
                )
        lf = read(root / left / f"fresh-{step}.json")
        rf = read(root / right / f"fresh-{step}.json")
        li = {(r["id"], r["seed"]): r for r in lf}
        ri = {(r["id"], r["seed"]): r for r in rf}
        assert li.keys() == ri.keys()
        grouped = collections.defaultdict(list)
        lengths = collections.defaultdict(list)
        for key in li:
            grouped[key[0]].append(li[key]["stats"]["all"]["rkl"] - ri[key]["stats"]["all"]["rkl"])
            lengths[key[0]].append(len(li[key]["response_ids"]) - len(ri[key]["response_ids"]))
        delta = [st.mean(v) for v in grouped.values()]
        comparisons.append(
            dict(
                left=left,
                right=right,
                step=step,
                domain="math",
                measure="fresh",
                metric="sequence_rkl",
                mean_difference=st.mean(delta),
                paired_prompt_ci=ci(delta),
                prompts=len(delta),
                length_difference=st.mean(st.mean(v) for v in lengths.values()),
            )
        )
    dump(root / "paired-controls.json", comparisons)
    retention = []
    for teacher in ["justrl", "polaris"]:
        prev = read(root / f"baseline-{teacher}/fixed.json")
        seen = set()
        rows = []
        for step in range(16, 129, 16):
            current = read(root / f"{teacher}-blocked/fixed-{step}.json")
            trained = "ABCD"[(step // 16 - 1) % 4]
            for group in sorted(seen - {trained}):
                old = st.mean(r["stats"]["all"]["rkl"] / 1024 for r in prev if r["group"] == group)
                new = st.mean(r["stats"]["all"]["rkl"] / 1024 for r in current if r["group"] == group)
                rows.append(
                    dict(step=step, trained=trained, old_group=group, before=old, after=new, increase=new - old)
                )
            prev = current
            seen.add(trained)
        retention.append(
            dict(teacher=teacher, checks=len(rows), increases=sum(r["increase"] > 1e-8 for r in rows), records=rows)
        )
    dump(root / "retention-checks.json", retention)
    broad = []
    initial = index(read(root / "rich-train-baseline.json"))
    for name in ["polaris-rich-lr1", "polaris-rich-lr3"]:
        rows = index(read(root / name / "train-pool-128.json"))
        assert rows.keys() == initial.keys()
        broad.append(
            dict(
                arm=name,
                prompts=512,
                before=st.mean(r["stats"]["all"]["rkl"] / 1024 for r in initial.values()),
                after=st.mean(r["stats"]["all"]["rkl"] / 1024 for r in rows.values()),
            )
        )
    dump(root / "rich-pool-fit.json", broad)
    coverage = []
    for n, file in [(448, "coverage-gradient.json"), (64, "coverage-gradient-64.json")]:
        r = read(root / file)[0]
        g = r["gram"]
        alpha = 64 / (64 + n)
        beta = n / (64 + n)
        v = alpha**2 * g[0][0] + beta**2 * g[1][1] + 2 * alpha * beta * g[0][1]
        combined = (alpha * g[0][2] + beta * g[1][2]) / (v * g[2][2]) ** 0.5
        coverage.append(
            dict(
                new_prompts=n,
                seen_held_cosine=r["cosine"][0][2],
                new_held_cosine=r["cosine"][1][2],
                combined_held_cosine=combined,
                groups=r["groups"],
                losses=r["losses"],
            )
        )
    dump(root / "coverage-gradient-summary.json", coverage)
    positions, overlap = [], []
    for teacher in ['justrl', 'polaris']:
        before = read(root / f'baseline-{teacher}/fixed.json')
        after = read(root / f'{teacher}-blocked/fixed-128.json')
        for band in ['early', 'middle', 'late']:
            values = []
            for rows in [before, after]:
                rs = [r for r in rows if r['group'] in ['E', 'F', 'G', 'H']]
                tokens = sum(r['stats'][band]['tokens'] for r in rs)
                values.append({k: sum(r['stats'][band][k] for r in rs) / tokens for k in ['rkl', 'hp', 'hq']})
            positions.append(dict(teacher=teacher, band=band, before=values[0], after=values[1]))
        for stage, file in [('initial', f'baseline-{teacher}/fresh.json'), ('128', f'{teacher}-blocked/fresh-128.json')]:
            rows = read(root / file)
            tokens = sum(r['stats']['all']['alive'] for r in rows)
            keys = ['overlap16', 'jaccard16', 'common_p16', 'common_q16'] + [f'category16_{i}' for i in range(4)]
            overlap.append(dict(teacher=teacher, stage=stage, tokens=tokens,
                                **{k: sum(r['stats']['all'][k] for r in rows) / tokens for k in keys}))
    dump(root / 'position-summary.json', positions)
    dump(root / 'fresh-overlap-summary.json', overlap)
    for r in comparisons:
        if r["domain"] == "math" and r["metric"] in ["rkl", "sequence_rkl"]:
            print(r)
    print("retention", [(r["teacher"], r["checks"], r["increases"]) for r in retention])
    print("coverage", coverage)


if __name__ == "__main__":
    main()
