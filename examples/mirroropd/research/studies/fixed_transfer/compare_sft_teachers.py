# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare archived online OPD curves for four SFT teachers, without training.

Semantic ledger: Qwen RAW/CCS0/CCS1 share a Base and OPD protocol; MiMo E73
is a historical, unmatched comparison. Columns distinguish prefix generators;
rows distinguish math from the equal-domain macro average. Every point is an
observed checkpoint, joined only within its run. RKL always means student||teacher.
Residuals divide risks on the same bank by that bank's step-zero risk. Curves
represent one training seed each; no training-seed confidence bands are implied.
Geometry ledger: four stable series colors, two-by-two panels with common axis
limits, a global legend above the panels and a neutral caption below them.
"""

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from examples.mirroropd.research.studies.fixed_transfer.mimo_absorption import build_rows

ROOT = Path(__file__).resolve().parents[5]
DOCS = ROOT / "examples/mirroropd/docs/experiments/teacher-transfer"
QWEN = DOCS / "data/teacher-construction-20261108/opd-raw256-v1"
DATA = DOCS / "data/teacher-construction-20261108/sft-comparison-v1"
FIGURES = DOCS / "figures/teacher-construction-20261108/sft-comparison-v1"
ARMS = ("RAW", "CCS-rho0", "CCS-rho1", "MiMo-SFT")
LABELS = ("Qwen RAW", "Qwen CCS0", "Qwen CCS1", "MiMo SFT (E73)")
DOMAINS = ("math", "code", "science", "if")
BANKS = ("Base-bank", "teacher-bank")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_rows():
    receipt = json.loads((QWEN / "archive-receipt.json").read_text())
    sources = []

    def checked(name):
        path = QWEN / name
        if digest(path) != receipt["artifacts"][name]:
            raise ValueError(f"Frozen Qwen input changed: {name}")
        sources.append(path)
        return path

    path = checked("risk-summary.csv")
    with path.open() as stream:
        summary = list(csv.DictReader(stream))
    rows = []
    for arm in ARMS[:-1]:
        for bank, filename in zip(BANKS, ("heldout.jsonl", "heldout-teacher-bank.jsonl"), strict=True):
            raw = checked(f"inputs/opd/{arm}/{filename}")
            groups = defaultdict(list)
            identities = defaultdict(set)
            for line in raw.read_text().splitlines():
                item = json.loads(line)
                key = item["step"], item["domain"]
                if item["id"] in identities[key] or item["tokens"] <= 0:
                    raise ValueError("Duplicate question or empty response")
                identities[key].add(item["id"])
                groups[key].append(item["rkl"])
            if set(groups) != {(s, d) for s in (0, 16, 64, 128, 256) for d in DOMAINS}:
                raise ValueError("Unexpected Qwen held-out grid")
            for domain in DOMAINS:
                if any(identities[s, domain] != identities[0, domain] for s in (16, 64, 128, 256)):
                    raise ValueError("Question identities changed across checkpoints")
            means = {key: math.fsum(values) / len(values) for key, values in groups.items()}
            if any(len(values) != 128 for values in groups.values()):
                raise ValueError("Expected 128 paired questions per domain")
            for source in summary:
                if (source["arm"], source["bank"]) != (arm, bank):
                    continue
                step, domain = int(source["step"]), source["domain"]
                expected = math.fsum(means[step, d] for d in DOMAINS) / 4 if domain == "macro4" else means[step, domain]
                if not math.isclose(expected, float(source["rkl"]), rel_tol=1e-12, abs_tol=1e-15):
                    raise ValueError("Qwen question-level recomputation differs from summary")
                rows.append(
                    dict(
                        arm=arm,
                        bank=bank,
                        domain=domain,
                        step=step,
                        n=int(source["n"]),
                        rkl=expected,
                        initial_rkl=float(source["initial_rkl"]),
                    )
                )
    mimo, mimo_sources = build_rows()
    sources.extend(mimo_sources)
    for source in mimo:
        if source["run"] != "E73":
            continue
        rows.append(
            dict(
                arm="MiMo-SFT",
                bank={"public_initial": "Base-bank", "public_teacher": "teacher-bank"}[source["bank"]],
                domain="macro4" if source["domain"] == "all" else source["domain"],
                step=source["step"],
                n=source["questions"],
                rkl=source["rkl"],
                initial_rkl=source["initial_rkl"],
            )
        )
    keys = {(r["arm"], r["bank"], r["domain"], r["step"]) for r in rows}
    if len(keys) != len(rows) or len(rows) != 220:
        raise ValueError("Incomplete or duplicate comparison")
    lookup = {(r["arm"], r["bank"], r["domain"], r["step"]): r for r in rows}
    for row in rows:
        initial = lookup[row["arm"], row["bank"], row["domain"], 0]["rkl"]
        if not math.isclose(initial, row["initial_rkl"], rel_tol=1e-12):
            raise ValueError("Wrong residual denominator")
        if row["domain"] == "macro4":
            average = math.fsum(lookup[row["arm"], row["bank"], d, row["step"]]["rkl"] for d in DOMAINS) / 4
            if not math.isclose(average, row["rkl"], rel_tol=1e-8):
                raise ValueError("Macro must average absolute domain risks before taking the ratio")
        row["residual_percent"] = 100 * row["rkl"] / initial
        row["decrease_percent"] = 100 - row["residual_percent"]
    return rows, sources


def render(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from examples.mirroropd.research.core.plotting import load_style, save_figure

    fv = load_style()
    fv.apply_rc("paper", fontsize=9)
    colors = fv.series_colors("paper", 4)
    for metric, ylabel, limits in (
        ("residual_percent", "Residual RKL (%)", (0.5, 120)),
        ("rkl", "RKL (nat/token)", (0.0004, 4)),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
        for panel, (domain, title) in enumerate((("math", "Math"), ("macro4", "Four-domain macro"))):
            for col, bank in enumerate(BANKS):
                ax = axes[panel, col]
                for arm, label, color, marker in zip(ARMS, LABELS, colors, ("o", "s", "^", "D"), strict=True):
                    values = sorted(
                        (r for r in rows if (r["arm"], r["bank"], r["domain"]) == (arm, bank, domain)),
                        key=lambda r: r["step"],
                    )
                    ax.plot(
                        [r["step"] for r in values],
                        [r[metric] for r in values],
                        color=color,
                        marker=marker,
                        markersize=3.5,
                        linewidth=1.4,
                        label=label,
                    )
                prefix = "Base prefixes" if col == 0 else "Own-teacher prefixes"
                ax.set_title(f"{title} / {prefix}", fontsize=9)
                ax.set_xlabel("OPD optimizer updates")
                ax.set_ylabel(ylabel)
                ax.set_yscale("log")
                ax.set_ylim(*limits)
                ax.set_xlim(-8, 328)
                ax.set_xticks((0, 80, 160, 240, 320))
                ax.grid(alpha=0.18)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.99), frameon=False)
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.86, wspace=0.35, hspace=0.55)
        save_figure(
            fig,
            FIGURES / metric,
            "Observed held-out checkpoints; one seed/run. Qwen: H256, reuse2. MiMo: long banks, reuse16.",
            caption_size=8,
        )


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows, sources = load_rows()
    write_csv(DATA / "curves.csv", rows)
    endpoints = [r for r in rows if r["step"] == (320 if r["arm"] == "MiMo-SFT" else 256)]
    write_csv(DATA / "endpoints.csv", endpoints)
    settings = {
        "Qwen": {
            "arms": list(ARMS[:-1]),
            "student": "Qwen3-1.7B-Base",
            "teacher_updates": 128,
            "opd_updates": 256,
            "batch": 16,
            "reuse": 2,
            "fresh_rollout_batches": 128,
            "training_response_cap": 1024,
            "bank_response_cap": 256,
        },
        "MiMo": {
            "arm": "MiMo-SFT",
            "run": "E73",
            "student": "MiMo-7B-Base",
            "teacher": "MiMo-7B-SFT public checkpoint",
            "opd_updates": 320,
            "batch": 128,
            "reuse": 16,
            "fresh_rollout_batches": 20,
            "training_response_cap": 7168,
            "bank_response_cap": {"math": 24576, "code": 24576, "science": 24576, "if": 8192},
        },
        "comparison_scope": "Qwen arms share an OPD protocol; MiMo is unmatched historical context",
    }
    (DATA / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    render(rows)
    provenance = {
        "comparison": "Three controlled Qwen SFT arms and one unmatched historical MiMo SFT run",
        "normalization": "question token mean, domain mean, then equal-domain macro; ratio after aggregation",
        "uncertainty": "one training seed per curve; no seed uncertainty bands",
        "validation": "Qwen recomputed from question-level JSONL; source hashes, step grids and macro means checked",
        "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in sources},
        "generator": {Path(__file__).relative_to(ROOT).as_posix(): digest(Path(__file__))},
        "outputs": {
            p.relative_to(ROOT).as_posix(): digest(p)
            for directory in (DATA, FIGURES)
            for p in sorted(directory.iterdir())
            if p.name != "provenance.json"
        },
    }
    (DATA / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"curve_rows": len(rows), "endpoint_rows": len(endpoints), "data": str(DATA)}))


if __name__ == "__main__":
    main()
