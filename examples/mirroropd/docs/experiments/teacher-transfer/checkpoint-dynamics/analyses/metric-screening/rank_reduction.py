"""Recompute relative-reduction correlations and rank all existing measurements.

Each observation is a pair. Reduction is 1-final/initial; positive correlation
means larger feature accompanies more reduction. Rank by absolute association,
retain sign, and provide separate signed ranks. Undefined results remain blank.
"""

import csv
import hashlib
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent))
import pair_identity as identity  # noqa: E402


def correlation(rows, within=False):
    if len(rows) < 3:
        return None
    x, y = (rankdata([r[i] for r in rows]) for i in (1, 2))
    if within:
        for family in {r[3] for r in rows}:
            mask = np.array([r[3] == family for r in rows])
            x[mask] -= x[mask].mean()
            y[mask] -= y[mask].mean()
    x -= x.mean()
    y -= y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / den) if den > 1e-14 else None


def table(rows, columns):
    return (
        "<table><tr>"
        + "".join(f"<th>{html.escape(k)}</th>" for k in columns)
        + "</tr>"
        + "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(r.get(k, '')))}</td>" for k in columns) + "</tr>" for r in rows
        )
        + "</table>"
    )


def main():
    data = json.loads((OUT / "screen-data.json").read_text())
    source = list(csv.DictReader((OUT / "correlations.csv").open()))
    records = []
    checked = 0
    for old in source:
        if not old["outcome"].endswith("/remaining"):
            continue
        scope, f, cohort = (old[k] for k in ["scope", "feature", "cohort"])
        rows = []
        for name, p in data["pairs"].items():
            value = p["scopes"][scope].get(f)
            if value is None or (cohort == "same_coordinate" and p["cross_size"]):
                continue
            if cohort in identity.FAMILIES and p["family"] != cohort:
                continue
            bank, domain, _ = old["outcome"].split("/")
            initial = p["scopes"]["all"][f"baseline_{bank}_{domain}"]
            final = p["outcomes"][f"{bank}/{domain}/absolute"]
            rows.append((name, value, 1 - final / initial, p["student_group"]))
        rho = correlation(rows, cohort == "within_student")
        if old["rho"]:
            assert rho is not None and abs(rho + float(old["rho"])) < 1e-10
            checked += 1
        else:
            assert rho is None
        group = data["features"][f]["group"]
        pretraining = (
            (f.startswith("teacher_") and f != "teacher_spectrum_remaining")
            or f.startswith("baseline_")
            or f == "base_energy"
        )
        record = dict(
            scope=scope,
            feature=f,
            bank=bank,
            domain=domain,
            cohort=cohort,
            n=len(rows),
            rho=rho,
            abs_rho=abs(rho) if rho is not None else None,
            rank_abs=None,
            rank_positive=None,
            group=group,
            available_before_opd=pretraining,
            loo_min=-float(old["loo_max"]) if old["loo_max"] else None,
            loo_max=-float(old["loo_min"]) if old["loo_min"] else None,
            pairs=",".join(r[0] for r in rows),
        )
        records.append(record)
    groups = defaultdict(list)
    for r in records:
        groups[r["scope"], r["bank"], r["domain"], r["cohort"]].append(r)
    for rows in groups.values():
        valid = [r for r in rows if r["rho"] is not None]
        for r, rank in zip(valid, rankdata([-round(r["abs_rho"], 12) for r in valid], method="min"), strict=True):
            r["rank_abs"] = int(rank)
        for r, rank in zip(valid, rankdata([-round(r["rho"], 12) for r in valid], method="min"), strict=True):
            r["rank_positive"] = int(rank)
    ordered = sorted(
        records, key=lambda r: (r["scope"], r["cohort"], r["bank"], r["domain"], r["rank_abs"] or 100000, r["feature"])
    )
    for name, subset in [
        ("relative-reduction-ranked.csv", ordered),
        ("relative-reduction-all-model-ranked.csv", [r for r in ordered if r["scope"] == "all"]),
        (
            "relative-reduction-pre-opd-ranked.csv",
            [r for r in ordered if r["scope"] == "all" and r["available_before_opd"]],
        ),
    ]:
        with (OUT / name).open("w") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(ordered[0]))
            writer.writeheader()
            writer.writerows(subset)
    amplitude = []
    for bank in ["initial", "teacher"]:
        for domain in ["math", "code", "if", "all"]:
            a, b = [
                next(
                    r
                    for r in records
                    if r["scope"] == "all"
                    and r["cohort"] == "all"
                    and r["bank"] == bank
                    and r["domain"] == domain
                    and r["feature"] == f
                )
                for f in ["teacher_global_amplitude", "teacher_amplitude"]
            ]
            amplitude.append(
                dict(
                    bank=bank,
                    domain=domain,
                    global_rho=round(a["rho"], 6),
                    median_rho=round(b["rho"], 6),
                    difference=round(a["rho"] - b["rho"], 6),
                )
            )
    sections = [table(amplitude, ["bank", "domain", "global_rho", "median_rho", "difference"])]
    columns = ["rank_abs", "feature", "rho", "n", "group", "available_before_opd"]
    for cohort in dict.fromkeys(r["cohort"] for r in source):
        for bank in ["initial", "teacher"]:
            for domain in ["math", "code", "if", "all"]:
                rows = [
                    dict(r, rho=round(r["rho"], 4))
                    for r in ordered
                    if r["scope"] == "all"
                    and r["cohort"] == cohort
                    and r["bank"] == bank
                    and r["domain"] == domain
                    and r["rho"] is not None
                ]
                title = f"{cohort} / {bank} / {domain}: {len(rows)} defined metrics"
                sections.append(f"<details><summary>{title}</summary>{table(rows, columns)}</details>")
    (OUT / "relative-reduction-ranked.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>相对下降相关排序</title>'
        "<style>body{font:15px system-ui;margin:30px}td,th{padding:5px 12px;border-bottom:1px solid #ddd;"
        "text-align:left}summary{padding:12px;background:#eee;margin-top:8px;cursor:pointer}</style>"
        "<h1>全部指标 × RKL 相对下降比例</h1><p>目标=1−终点/初始；正相关表示指标大时下降比例大。"
        "rank_abs 按相关绝对值降序，同值共享名次；保留 rho 符号。每点一个 pair。"
        "全模型314字段全部保留，含常量及无定义项；下表仅展示有定义项。"
        "teacher参数字段排除E102；跨尺寸不等于跨家族。预训练可得仅是可测标签，不是预测验证。"
        "完整模块/深度结果见CSV。</p>" + "\n".join(sections)
    )
    (OUT / "reduction-ranking-provenance.json").write_text(
        json.dumps(
            dict(
                source_sha256=hashlib.sha256((OUT / "screen-data.json").read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                records=len(records),
                direct_correlations_checked=checked,
                amplitude_comparison=amplitude,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(dict(records=len(records), checked=checked, amplitude=amplitude), indent=2))


if __name__ == "__main__":
    main()
