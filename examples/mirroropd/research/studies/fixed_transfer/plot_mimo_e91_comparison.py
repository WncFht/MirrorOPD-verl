# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Render an auditable endpoint and absorption comparison for MiMo E71-E73/E91.

The E71-E73 arms start from MiMo-7B-Base; E91 starts from MiMo-7B-SFT.
RKL is therefore plotted as within-arm relative reduction, while accuracy is
shown as the final student endpoint with E91's initial SFT checkpoint marked.
Semantic ledger: colors identify runs; bars are question-mean avg@n accuracy;
diamonds are E91's initial SFT scores; lines are each run's teacher-bank RKL
reduction relative to its own step zero. Banks use independent trajectories.
Geometry ledger: RKL x is optimizer updates (0-800), y is percent reduction;
accuracy x is eight categorical benchmarks, y is accuracy (0-100 percent).
"""

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
DOCS = ROOT / "examples/mirroropd/docs/experiments/teacher-transfer"
DATA = DOCS / "data/mimo-stage-20261104-final"
E91 = DOCS / "figures/e91-live-20261108"
INITIAL = DOCS / "figures/e91-initial-20261107/initial-readout.csv"
OUTPUT = DOCS / "figures/mimo-stage-20261104-final/e91-comparison"
RUN = ROOT / "runs/e91-mimo-rlzero-sft-20261107"
ARMS = ("E71", "E72", "E73", "E91")
LABELS = {
    "E71": "E71 SFT+RL → Base student",
    "E72": "E72 direct RL → Base student",
    "E73": "E73 SFT → Base student",
    "E91": "E91 direct RL → SFT student",
}
DATASETS = ("MATH-500", "AMC23", "AIME24", "AIME25", "AIME26", "LCBv5", "GPQA-Diamond", "IFEval")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def verified_accuracy(directory, sources):
    import pyarrow.parquet as pq

    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "complete"
    sources.append(path)
    groups = defaultdict(lambda: defaultdict(list))
    keys = set()
    for name, entry in manifest["files"].items():
        if not name.startswith("scores/"):
            continue
        part = directory / name
        assert hashlib.sha256(part.read_bytes()).hexdigest() == entry["sha256"]
        sources.append(part)
        for row in pq.read_table(part, columns=["dataset", "source_id", "sample_index", "status", "score"]).to_pylist():
            key = row["dataset"], row["source_id"], row["sample_index"]
            assert key not in keys and row["status"] in {"ok", "missing_final_answer"}
            assert math.isfinite(row["score"])
            keys.add(key)
            groups[row["dataset"]][row["source_id"]].append(row["score"])
    metrics = manifest["metadata"]["metrics"]
    values = {}
    for dataset in DATASETS:
        questions = groups[dataset]
        prefix = f"eval/accuracy/{dataset}"
        assert len(questions) == metrics[prefix + "/questions"]
        assert sum(map(len, questions.values())) == metrics[prefix + "/samples"]
        expected_n = metrics[prefix + "/samples"] / len(questions)
        assert all(len(scores) == expected_n for scores in questions.values())
        values[dataset] = math.fsum(math.fsum(scores) / len(scores) for scores in questions.values()) / len(questions)
        assert math.isclose(values[dataset], metrics[prefix + "/avg_at_n"], abs_tol=1e-12)
    return values


def verify_e91_rkl(sources):
    import pyarrow.parquet as pq

    audit = []
    for snapshot in read_csv(E91 / "evaluation.csv"):
        outer = int(snapshot["outer"])
        event = RUN / f"formal/E91/evaluation/events/outer-{outer:03d}"
        manifest_path = event / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["status"] == "complete"
        path = event / "rkl.parquet"
        sources.extend([manifest_path, path])
        entry = manifest["files"]["rkl.parquet"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        groups = defaultdict(list)
        keys = set()
        for row in pq.read_table(path).to_pylist():
            if row["bank"] != "public_teacher":
                continue
            key = row["domain"], row["source_id"], row["sample_index"]
            assert key not in keys and row["tokens"] > 0
            keys.add(key)
            value = row["rkl_sum"] / row["tokens"]
            assert math.isfinite(value) and value >= 0
            groups[row["domain"]].append(value)
        assert set(groups) == {"math", "code", "science", "if"}
        assert all(len(values) == 128 for values in groups.values())
        by_domain = {domain: math.fsum(values) / len(values) for domain, values in groups.items()}
        average = math.fsum(by_domain.values()) / 4
        assert math.isclose(average, float(snapshot["teacher_rkl"]), abs_tol=1e-12)
        audit.append(dict(outer=outer, optimizer_updates=int(snapshot["step"]), all=average, **by_domain))
    return audit


def load_rkl():
    rows = []
    for arm in ("E71", "E72", "E73"):
        source = DATA / arm / "fixed-bank-rkl.csv"
        values = [row for row in read_csv(source) if row["bank"] == "public_teacher" and row["domain"] == "all"]
        initial = float(values[0]["rkl"])
        for row in values:
            rows.append(
                dict(
                    arm=arm,
                    outer=int(row["outer"]),
                    optimizer_updates=int(row["optimizer_updates"]),
                    initial_rkl=initial,
                    rkl=float(row["rkl"]),
                    reduction_percent=100 * (1 - float(row["rkl"]) / initial),
                )
            )
    e91_rows = read_csv(E91 / "evaluation.csv")
    e91_teacher_initial = float(e91_rows[0]["teacher_rkl"])
    for row in e91_rows:
        value = float(row["teacher_rkl"])
        rows.append(
            dict(
                arm="E91",
                outer=int(row["outer"]),
                optimizer_updates=int(row["step"]),
                initial_rkl=e91_teacher_initial,
                rkl=value,
                reduction_percent=100 * (1 - value / e91_teacher_initial),
            )
        )
    return rows


def load_accuracy(sources):
    rows = [row for row in read_csv(DATA / "accuracy.csv") if row["role"] == "student"]
    for row in rows:
        row["accuracy"] = float(row["accuracy"])
    initial = {row["benchmark"]: float(row["accuracy"]) for row in read_csv(INITIAL)}
    checked_initial = verified_accuracy(RUN / "bootstrap/initial-accuracy", sources)
    assert all(math.isclose(initial[d], checked_initial[d], abs_tol=1e-12) for d in DATASETS)
    e91 = verified_accuracy(RUN / "formal/E91/evaluation/events/outer-020/accuracy", sources)
    values = {arm: {} for arm in ARMS}
    for row in rows:
        values[row["arm"]][row["dataset"]] = row["accuracy"]
    values["E91"] = e91
    return values, initial


def render():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from examples.mirroropd.research.core.plotting import load_style

    fv = load_style()
    fv.apply_rc("paper", fontsize=10)
    colors = dict(zip(ARMS, [fv.HUES[c] for c in ("vteal", "vorange", "vviolet", "vcoral")], strict=True))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = []
    e91_rkl_audit = verify_e91_rkl(sources)
    rkl = load_rkl()
    accuracy, initial = load_accuracy(sources)

    def save_figure(fig, path, caption):
        fig.text(0.5, 0.025, caption, ha="center", fontsize=8, color=fv.NEUTRALS["baseline"])
        fig.canvas.draw()
        problems = fv.audit(fig, print_slot=fig.get_figwidth())
        lines = fv.format_audit_lines(problems)
        path.with_suffix(".audit.txt").write_text(
            "\n".join(lines) + "\n" if lines else "PASS: fv.audit(fig) returned no issues.\n"
        )
        assert not any(fv.problem_severity(p) == "error" for p in problems), lines
        for extension in ("png", "pdf", "svg"):
            fig.savefig(path.with_suffix("." + extension), dpi=180, facecolor=fv.NEUTRALS["paper"])
        plt.close(fig)

    with (OUTPUT / "rkl-comparison.csv").open("w", newline="") as stream:
        fields = list(rkl[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rkl)
    with (OUTPUT / "accuracy-comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["arm", "dataset", "accuracy", "source"])
        for arm in ARMS:
            for dataset in DATASETS:
                writer.writerow([arm, dataset, accuracy[arm][dataset], "final_student"])
        for dataset in DATASETS:
            writer.writerow(["E91", dataset, initial[dataset], "initial_sft"])

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for arm in ARMS:
        chosen = [row for row in rkl if row["arm"] == arm]
        ax.plot(
            [row["optimizer_updates"] for row in chosen],
            [row["reduction_percent"] for row in chosen],
            marker="o",
            ms=4,
            lw=1.6,
            color=colors[arm],
            label=LABELS[arm],
        )
    ax.set(xlabel="Optimizer updates", ylabel="Teacher-bank RKL reduction (%)", ylim=(-10, 110))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.subplots_adjust(bottom=0.22)
    save_figure(
        fig,
        OUTPUT / "rkl-absorption-comparison",
        "Within-arm reduction on each arm's teacher-prefix bank; E91 starts from MiMo-7B-SFT.",
    )

    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    x = np.arange(len(DATASETS))
    width = 0.18
    for index, arm in enumerate(ARMS):
        ax.bar(
            x + (index - 1.5) * width,
            [100 * accuracy[arm][dataset] for dataset in DATASETS],
            width,
            color=colors[arm],
            label=LABELS[arm],
        )
    ax.plot(
        x + 1.5 * width,
        [100 * initial[dataset] for dataset in DATASETS],
        color=fv.NEUTRALS["baseline"],
        marker="D",
        ls="none",
        label="E91 initial SFT",
    )
    ax.set(
        xticks=x,
        xticklabels=["MATH500", "AMC23", "AIME24", "AIME25", "AIME26", "LCBv5", "GPQA", "IFEval"],
        xlabel="Benchmark",
        ylabel="Question-mean avg@n (%)",
        ylim=(0, 100),
    )
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.subplots_adjust(bottom=0.24, right=0.72)
    save_figure(
        fig,
        OUTPUT / "accuracy-comparison",
        "Final student accuracy; E91 begins from an SFT checkpoint, while E71-E73 begin from Base.",
    )

    source_paths = [DATA / arm / "fixed-bank-rkl.csv" for arm in ("E71", "E72", "E73")]
    source_paths += [E91 / "evaluation.csv", DATA / "accuracy.csv", INITIAL, *sources]
    (OUTPUT / "source-audit.json").write_text(
        json.dumps(
            dict(e91_accuracy_recomputed=True, e91_initial_recomputed=True, e91_teacher_bank=e91_rkl_audit), indent=2
        )
        + "\n"
    )
    provenance = {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "arms": ARMS,
        "labels": LABELS,
        "rkl_source": "teacher-prefix fixed-bank RKL for E71-E73; E91 evaluation.csv teacher_rkl",
        "accuracy_source": "final student accuracy for E71-E73; E91 outer-020 accuracy manifest",
        "sources": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths
        },
        "generator": str(Path(__file__).relative_to(ROOT)),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifacts": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in OUTPUT.iterdir()
            if path.is_file() and path.name != "provenance.json"
        },
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    render()
