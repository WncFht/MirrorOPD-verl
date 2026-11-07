# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit and plot the completed E91 initial SFT readout without GPU work.

Semantic ledger: one bar is one benchmark's initial-checkpoint avg@n, mean
response length, or observed engine finish category. Scores average samples
within each question, then questions; no training benefit is inferred.
Geometry ledger: shared benchmark rows; accuracy and stop shares span 0--100%,
length uses thousands of generated tokens. Every horizontal bar starts at zero.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from omegaconf import OmegaConf

from examples.mirroropd.research.studies.weighted_targets.plot_e81_validation import ROOT, style

RUN = ROOT / "runs/e91-mimo-rlzero-sft-20261107"
OUT = ROOT / "examples/mirroropd/docs/experiments/teacher-transfer/figures/e91-initial-20261107"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(run):
    """Recompute every score and stop count from immutable scored records."""
    bootstrap = run / "bootstrap"
    accuracy = bootstrap / "initial-accuracy"
    manifest_path = accuracy / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    generation_path = bootstrap / "initial/manifest.json"
    generation = json.loads(generation_path.read_text())
    assert manifest["status"] == generation["status"] == "complete"
    assert manifest["metadata"]["generation_sha256"] == digest(generation_path)
    config_path = run / "configs/E91.yaml"
    config = OmegaConf.load(config_path)
    sources = [manifest_path, generation_path, config_path, Path(__file__)]
    groups = defaultdict(list)
    for name, expected in manifest["files"].items():
        if not name.startswith("scores/"):
            continue
        path = accuracy / name
        assert digest(path) == expected["sha256"], path
        sources.append(path)
        for row in pq.read_table(path).to_pylist():
            assert row["status"] in {"ok", "missing_final_answer"}
            assert math.isfinite(row["score"]) and 0 <= row["score"] <= 1
            groups[row["dataset"]].append(row)
    rows = []
    metadata = manifest["metadata"]["metrics"]
    for spec in config.trainer.evaluation.datasets:
        values = groups.pop(spec.name)
        questions = pq.read_table(spec.path, columns=["id"]).column("id").to_pylist()
        expected = {(qid, i) for qid in questions for i in range(spec.n)}
        observed = [(r["source_id"], r["sample_index"]) for r in values]
        assert len(set(observed)) == len(observed) and set(observed) == expected
        per_question = defaultdict(list)
        for row in values:
            per_question[row["source_id"]].append(row["score"])
        finish = Counter(r["finish_reason"] for r in values)
        assert set(finish) <= {"stop", "length", "repetition"}
        prefix = f"eval/accuracy/{spec.name}/"
        accuracy_value = statistics.mean(statistics.mean(v) for v in per_question.values())
        length = statistics.mean(r["response_tokens"] for r in values)
        assert math.isclose(accuracy_value, metadata[prefix + "avg_at_n"], abs_tol=1e-12)
        assert math.isclose(length, metadata[prefix + "response_tokens_mean"], abs_tol=1e-9)
        for reason in ("stop", "length", "repetition"):
            assert finish[reason] == metadata[prefix + f"finish_reason/{reason}_count"]
        rows.append(
            dict(
                benchmark=spec.name,
                domain=spec.domain,
                questions=len(questions),
                n=spec.n,
                responses=len(values),
                accuracy=accuracy_value,
                mean_length=length,
                **{f"{reason}_count": finish[reason] for reason in ("stop", "length", "repetition")},
                **{f"{reason}_rate": finish[reason] / len(values) for reason in ("stop", "length", "repetition")},
            )
        )
        sources.append(Path(spec.path))
    assert not groups
    assert sum(r["responses"] for r in rows) == generation["metadata"]["observed_samples"] == 6703
    return rows, sources, generation


def plot(rows, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fv = style()
    theme = fv.apply_rc("paper")
    colors = fv.series_colors("paper", 4)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.4), sharey=True)
    y = np.arange(len(rows))
    axes[0].barh(y, [100 * r["accuracy"] for r in rows], color=colors[0])
    axes[1].barh(y, [r["mean_length"] / 1000 for r in rows], color=fv.NEUTRALS["baseline"])
    left = np.zeros(len(rows))
    for reason, color, label in (
        ("stop", colors[0], "EOS / stop"),
        ("length", colors[1], "Length cap"),
        ("repetition", colors[2], "Repetition"),
    ):
        values = np.array([100 * r[f"{reason}_rate"] for r in rows])
        axes[2].barh(y, values, left=left, color=color, label=label)
        left += values
    for ax, title, label in zip(
        axes,
        ("Initial SFT accuracy", "Generated response length", "Observed finish reasons"),
        ("avg@n (%)", "Mean response length (k tokens)", "Responses (%)"),
        strict=True,
    ):
        ax.set(title=title, xlabel=label)
        ax.set_yticks(y, [r["benchmark"] for r in rows])
        fv.apply_frame(ax, theme.frame)
        fv.apply_grid(ax, theme.grid)
    axes[0].set_ylabel("Benchmark")
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 100)
    axes[1].set_xlim(0, 16)
    axes[2].set_xlim(0, 100)
    fig.legend(*axes[2].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.53, 0.065), ncol=3)
    fig.text(
        0.53,
        0.02,
        "MiMo-7B-SFT before E91 updates | 1,648 questions / 6,703 responses | task-specific n; no pass@n",
        ha="center",
        fontsize="small",
        color=fv.NEUTRALS["baseline"],
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.canvas.draw()
    problems = fv.audit(fig, print_slot=fig.get_figwidth())
    (out / "initial-readout-audit.json").write_text(json.dumps(problems, indent=2) + "\n")
    if any(fv.problem_severity(p) == "error" for p in problems):
        raise RuntimeError(problems)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(out / f"initial-readout.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    rows, sources, generation = summarize(args.run)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "initial-readout.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = generation["metadata"]
    provenance = dict(
        snapshot_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        sources=[dict(path=str(p.relative_to(ROOT)), sha256=digest(p)) for p in sources],
        model_identity=metadata["model_identity"],
        protocols=metadata["protocols"],
        execution=metadata["execution"],
        limitation="Initial SFT baseline only; teacher bootstrap and all E91 optimization/readouts remain separate.",
    )
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    audit = dict(
        questions=sum(r["questions"] for r in rows),
        responses=sum(r["responses"] for r in rows),
        checked_unique_question_samples=True,
        checked_scored_shard_hashes=True,
        checked_manifest_metrics=True,
        optimizer_updates=0,
    )
    (args.out / "source-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    plot(rows, args.out)
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
