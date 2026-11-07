#!/usr/bin/env python3
"""Build an auditable inventory and early-position comparison for position RKL.

The exact p00--10% comparison only includes records whose source stores the
native ten relative-response deciles. Three-band ``positions.csv`` outputs
from the fixed-state appendix are inventoried separately and are not silently
treated as p00--10% measurements.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[6]
WORKSPACE = ROOT.parent
STYLE = ROOT.parent / "dotfiles/.agents/skills/figure-viz/assets/mpl/fvstyle.py"
spec = importlib.util.spec_from_file_location("fvstyle", STYLE)
fv = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fv
spec.loader.exec_module(fv)

DOCS = ROOT / "examples/mirroropd/docs/experiments/teacher-transfer"
OUT = DOCS / "figures/position-kl-inventory-20261107"
DATA = DOCS / "data/position-kl-inventory-20261107"
E49_HISTORY = WORKSPACE / "wandb-logs/derived/MirrorOPD-verl/offline-run-20261027_175635-zyru8wdf/history.jsonl"
BINS = [f"p{i:02d}_{i + 10:02d}" for i in range(0, 100, 10)]
LABELS = [f"{i}–{i + 10}%" for i in range(0, 100, 10)]
AUDITS: dict[str, list[str]] = {}


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def native_from_csv(path: Path, experiment: str, student: str, teacher: str, objective: str, scope: str) -> dict:
    rows = csv_rows(path)
    first = [r for r in rows if r.get("update") == "1" and r.get("bucket", r.get("bin")) == "p00_10"]
    if not first:
        raise ValueError(f"no native p00_10 row in {path}")
    first_row = first[0]
    profile = {r["bucket"]: float(r["rkl_mean"]) for r in rows if r.get("update") == "1" and r.get("bucket") in BINS}
    return {
        "experiment": experiment,
        "student": student,
        "teacher": teacher,
        "objective": objective,
        "run_scope": scope,
        "updates": max(int(r["update"]) for r in rows),
        "outer": max(int(r["outer"]) for r in rows),
        "reuse": max(int(r["update_in_outer"]) for r in rows),
        "source_path": str(path.relative_to(WORKSPACE)),
        "position_namespace": "native relative response deciles",
        "has_full_rkl_position": "yes",
        "first_update": "1",
        "initial_p00_10": float(first_row["rkl_mean"]),
        "initial_full_kl": "",
        "initial_profile_available": "yes" if len(profile) == 10 else "no",
        "notes": "Raw teacher RKL diagnostic; values are pre-update observations.",
        "_profile": profile,
    }


def native_from_termination(path: Path, arm: str, objective: str) -> dict:
    rows = csv_rows(path)
    rr = [r for r in rows if r["arm"] == arm]
    first = [r for r in rr if r["outer"] == "1" and r["update"] == "1" and r["bin"] == "0"][0]
    profile = {
        f"p{int(r['bin']) * 10:02d}_{(int(r['bin']) + 1) * 10:02d}": float(r["mean"])
        for r in rr
        if r["outer"] == "1" and r["update"] == "1"
    }
    return {
        "experiment": f"termination-{arm}",
        "student": "Qwen3-1.7B-Base",
        "teacher": "Qwen3-4B",
        "objective": objective,
        "run_scope": "20-outer controlled diagnostic",
        "updates": max(int(r["update"]) for r in rr),
        "outer": max(int(r["outer"]) for r in rr),
        "reuse": max(int(r["reuse_index"]) for r in rr),
        "source_path": str(path.relative_to(WORKSPACE)),
        "position_namespace": "native relative response deciles",
        "has_full_rkl_position": "yes",
        "first_update": "1",
        "initial_p00_10": float(first["mean"]),
        "initial_full_kl": "",
        "initial_profile_available": "yes" if len(profile) == 10 else "no",
        "notes": "Raw full-vocabulary RKL position; aligned arm optimizes termination_equiv_kl.",
        "_profile": profile,
    }


def history_initial(path: Path, experiment: str, student: str, teacher: str, objective: str, scope: str) -> dict:
    rows = []
    for line in path.read_text().splitlines():
        item = json.loads(line)
        row = item.get("data", item)
        if "distill/full_kl_position/p00_10/mean" in row:
            rows.append(row)
    rows.sort(key=lambda r: int(r.get("step/update", 0)))
    if not rows:
        raise ValueError(f"no native position rows in {path}")
    row = rows[0]
    profile = {
        bucket: float(row[f"distill/full_kl_position/{bucket}/mean"])
        for bucket in BINS
        if f"distill/full_kl_position/{bucket}/mean" in row
    }
    return {
        "experiment": experiment,
        "student": student,
        "teacher": teacher,
        "objective": objective,
        "run_scope": scope,
        "updates": max(int(r.get("step/update", 0)) for r in rows),
        "outer": max(int(r.get("step/outer", 0)) for r in rows),
        "reuse": "",
        "source_path": str(path.relative_to(WORKSPACE)),
        "position_namespace": "native relative response deciles",
        "has_full_rkl_position": "yes",
        "first_update": str(row.get("step/update", 1)),
        "initial_p00_10": float(row["distill/full_kl_position/p00_10/mean"]),
        "initial_full_kl": float(row.get("distill/full_kl", "nan")),
        "initial_profile_available": "yes" if len(profile) == 10 else "no",
        "notes": "Raw teacher RKL position diagnostic; endpoint is pre-update.",
        "_profile": profile,
    }


def appendix_inventory() -> list[dict[str, str | float]]:
    base = DOCS / "transfer-mechanism/appendix/data/run-figures"
    rows: list[dict[str, str | float]] = []
    for path in sorted(base.glob("*/*/positions.csv")):
        if path.parent.parent.name != "main":
            continue
        fields = csv_rows(path)
        first = fields[0] if fields else {}
        rows.append(
            {
                "experiment": path.parent.name,
                "student": "see run README",
                "teacher": "see run README",
                "objective": "teacher RKL diagnostic",
                "run_scope": "fixed-state appendix, 0–320 updates",
                "updates": max((int(r["step"]) for r in fields), default=""),
                "outer": "",
                "reuse": "",
                "source_path": str(path.relative_to(WORKSPACE)),
                "position_namespace": "three relative bands (early/middle/late)",
                "has_full_rkl_position": "yes (three bands)",
                "first_update": first.get("step", ""),
                "initial_p00_10": "",
                "initial_full_kl": "",
                "initial_profile_available": "no (not deciles)",
                "notes": "Inventoried separately; early band must not be relabeled p00–10%.",
            }
        )
    return rows


def finish(fig: plt.Figure, stem: str, caption: str) -> None:
    fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=8, color=fv.NEUTRALS["baseline"])
    fig.canvas.draw()
    problems = fv.audit(fig, print_slot=fig.get_figwidth())
    lines = fv.format_audit_lines(problems)
    AUDITS[stem] = lines
    (OUT / f"{stem}.audit.txt").write_text(
        "\n".join(lines) + "\n" if lines else "PASS: fv.audit(fig) returned no issues.\n"
    )
    if any(fv.problem_severity(p) == "error" for p in problems):
        raise RuntimeError(f"figure audit failed: {stem}: {lines}")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=180, facecolor=fv.NEUTRALS["paper"])
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = [k for k in rows[0] if not k.startswith("_")]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in keys} for row in rows)


def main() -> None:
    fv.apply_rc("paper", fontsize=10)
    assert json.loads(E49_HISTORY.with_name("run.json").read_text())["run"]["run_id"] == "zyru8wdf"
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    # The source CSVs contain multiple runs; select each run explicitly.
    mimo_csv = DOCS / "data/mimo-position-kl-20261106/position-kl.csv"
    for run, teacher, scope in [
        ("E71", "MiMo-7B-RL", "formal stage run"),
        ("E72", "MiMo-7B-RL-Zero", "formal stage run"),
        ("E73", "MiMo-7B-SFT", "formal stage run"),
    ]:
        source = csv_rows(mimo_csv)
        source = [r for r in source if r["run"] == run]
        first = next(r for r in source if r["update"] == "1" and r["bucket"] == "p00_10")
        rows.append(
            {
                "experiment": run,
                "student": "MiMo-7B-Base",
                "teacher": teacher,
                "objective": "full-vocab reverse KL",
                "run_scope": scope,
                "updates": max(int(r["update"]) for r in source),
                "outer": max(int(r["outer"]) for r in source),
                "reuse": max(int(r["update_in_outer"]) for r in source),
                "source_path": str(mimo_csv.relative_to(WORKSPACE)),
                "position_namespace": "native relative response deciles",
                "has_full_rkl_position": "yes",
                "first_update": "1",
                "initial_p00_10": float(first["rkl_mean"]),
                "initial_full_kl": "",
                "initial_profile_available": "yes",
                "notes": "Raw teacher RKL diagnostic; pre-update observation.",
                "_profile": {r["bucket"]: float(r["rkl_mean"]) for r in source if r["update"] == "1"},
            }
        )
    polaris_csv = DOCS / "data/polaris-position-kl-20261106/position-kl.csv"
    source = [r for r in csv_rows(polaris_csv) if r["run"] == "E48"]
    first = next(r for r in source if r["update"] == "1" and r["bucket"] == "p00_10")
    rows.append(
        {
            "experiment": "E48",
            "student": "DeepSeek-R1-Distill-Qwen-1.5B",
            "teacher": "Polaris-7B-Preview",
            "objective": "full-vocab reverse KL",
            "run_scope": "formal 256-outer run",
            "updates": max(int(r["update"]) for r in source),
            "outer": max(int(r["outer"]) for r in source),
            "reuse": max(int(r["update_in_outer"]) for r in source),
            "source_path": str(polaris_csv.relative_to(WORKSPACE)),
            "position_namespace": "native relative response deciles",
            "has_full_rkl_position": "yes",
            "first_update": "1",
            "initial_p00_10": float(first["rkl_mean"]),
            "initial_full_kl": "",
            "initial_profile_available": "yes",
            "notes": "Raw teacher RKL diagnostic; pre-update observation.",
            "_profile": {r["bucket"]: float(r["rkl_mean"]) for r in source if r["update"] == "1"},
        }
    )
    rows.extend(
        [
            history_initial(
                E49_HISTORY,
                "E49",
                "DeepSeek-R1-Distill-Qwen-1.5B",
                "Polaris-7B-Preview",
                "conditional FKL (raw RKL diagnostic)",
                "formal 256-outer run",
            ),
            history_initial(
                DOCS.parent.parent / "archive/experiments/polaris/data/polaris-grcuotmi-20261026/E35/history.jsonl",
                "E35",
                "DeepSeek-R1-Distill-Qwen-1.5B",
                "JustRL-DeepSeek-1.5B",
                "full-vocab reverse KL",
                "formal 250-outer run",
            ),
            history_initial(
                WORKSPACE / "wandb-logs/MirrorOPD-verl/analysis/e39-qwen3base-final-20261027/history.jsonl",
                "E39",
                "Qwen3-1.7B-Base",
                "Qwen3-8B",
                "termination-equivalent objective (raw RKL diagnostic)",
                "formal 200-outer run",
            ),
            history_initial(
                WORKSPACE / "wandb-logs/MirrorOPD-verl/analysis/qwen-e41-e43-final-20261028/E41/metrics.jsonl",
                "E41",
                "Qwen3-1.7B-Base",
                "Qwen3-4B",
                "termination-equivalent objective (raw RKL diagnostic)",
                "formal 100-outer run",
            ),
            history_initial(
                WORKSPACE / "wandb-logs/MirrorOPD-verl/analysis/qwen-e41-e43-final-20261028/E43/metrics.jsonl",
                "E43",
                "Qwen3-1.7B Non-thinking",
                "Qwen3-4B-Instruct-2507",
                "full-vocab reverse KL",
                "formal 100-outer run",
            ),
        ]
    )
    termination = DOCS.parent.parent / "infra/model-compatibility/termination/data/full-kl-position.csv"
    rows.extend(
        [
            native_from_termination(termination, "baseline", "raw full-vocab reverse KL"),
            native_from_termination(termination, "aligned", "termination-equivalent objective (raw RKL diagnostic)"),
        ]
    )
    appendix = appendix_inventory()
    write_csv(DATA / "inventory.csv", rows + appendix)
    initial_rows = []
    for row in rows:
        initial_rows.append(
            {
                k: row.get(k, "")
                for k in [
                    "experiment",
                    "student",
                    "teacher",
                    "objective",
                    "run_scope",
                    "initial_p00_10",
                    "initial_full_kl",
                    "source_path",
                ]
            }
        )
    write_csv(DATA / "initial-position-kl.csv", initial_rows)
    profile_rows = []
    for row in rows:
        for bucket in BINS:
            if bucket in row.get("_profile", {}):
                profile_rows.append(
                    {"experiment": row["experiment"], "bucket": bucket, "rkl_mean": row["_profile"][bucket]}
                )
    write_csv(DATA / "initial-position-profile.csv", profile_rows)

    formal = [r for r in rows if r.get("initial_profile_available") == "yes"]
    order = [
        "E39",
        "E41",
        "E43",
        "termination-baseline",
        "termination-aligned",
        "E35",
        "E48",
        "E49",
        "E71",
        "E73",
        "E72",
    ]
    formal.sort(key=lambda r: order.index(r["experiment"]) if r["experiment"] in order else len(order))
    groups = {
        "mimo": (["E71", "E73", "E72"], "MiMo initial p00–10% position RKL", "MiMo-7B-RL / SFT / RL-Zero"),
        "polaris": (
            ["E48", "E49"],
            "Polaris initial p00–10% position RKL",
            "Polaris-7B-Preview; E49 retains raw RKL diagnostics",
        ),
        "qwen": (
            ["E39", "E41", "E43", "termination-baseline", "termination-aligned"],
            "Qwen/termination initial p00–10% position RKL",
            "Formal runs and controlled termination comparison",
        ),
        "same-origin": (["E35"], "Same-origin reference initial p00–10% position RKL", "JustRL-DeepSeek-1.5B teacher"),
    }
    colors = {
        "mimo": fv.HUES["vteal"],
        "polaris": fv.HUES["vorange"],
        "qwen": fv.HUES["vviolet"],
        "same-origin": fv.HUES["vgreen"],
    }
    by_id = {r["experiment"]: r for r in formal}
    short_labels = {"termination-baseline": "term-B", "termination-aligned": "term-A"}
    for stem, (ids, graph_title, subtitle) in groups.items():
        subset = [by_id[identifier] for identifier in ids if identifier in by_id]
        xs = np.arange(len(subset))
        vals = np.array([float(r["initial_p00_10"]) for r in subset])
        fig, ax = plt.subplots(figsize=(8.5 if len(subset) > 1 else 5.5, 5.5))
        fig.subplots_adjust(bottom=0.28, top=0.78)
        for x, value, r in zip(xs, vals, subset, strict=True):
            marker = "o"
            ax.scatter(
                x,
                value,
                s=60,
                marker=marker,
                color=colors[stem],
                edgecolor=fv.NEUTRALS["vink"],
                linewidth=0.6,
                zorder=3,
            )
            ax.text(x, value * 1.12, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
        ax.set_yscale("log")
        ax.set_ylabel("Initial p00–10% position RKL (nats, log scale)")
        ax.set_xlabel("Experiment (first pre-update observation)")
        ax.set_xticks(xs, [short_labels.get(r["experiment"], r["experiment"]) for r in subset], rotation=30, ha="right")
        ax.set_ylim(vals.min() / 2.5, vals.max() * 2.5)
        ax.grid(axis="y", which="major", alpha=0.2)
        fig.suptitle(graph_title + "\n" + subtitle, y=0.95, fontsize=11)
        finish(
            fig, f"initial-p00-10-{stem}", "Native ten-decile teacher RKL; each family is plotted in its own figure."
        )

        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        fig.subplots_adjust(bottom=0.24, top=0.78, right=0.78)
        for line_index, r in enumerate(subset):
            profile = r["_profile"]
            if len(profile) != 10:
                continue
            line_styles = [("-", "o"), ("--", "s"), (":", "^"), ("-.", "D")]
            linestyle, marker = line_styles[line_index % len(line_styles)]
            ax.plot(
                range(10),
                [profile[b] for b in BINS],
                linestyle=linestyle,
                marker=marker,
                ms=3,
                lw=1.1,
                color=colors[stem],
                alpha=0.78,
                label=r["experiment"],
            )
        ax.set_yscale("log")
        ax.set_xticks(range(10), LABELS, rotation=30, ha="right")
        ax.set_xlabel("Relative response position")
        ax.set_ylabel("Position-bin RKL (nats, log scale)")
        ax.grid(True, which="major", alpha=0.2)
        if len(subset) > 1:
            ax.legend(frameon=False, ncol=1, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.suptitle(graph_title.replace("p00–10%", "position profile") + "\n" + subtitle, y=0.95, fontsize=11)
        finish(
            fig,
            f"initial-position-profile-{stem}",
            "Each curve is one run's first pre-update native ten-decile profile; no cross-family scale is implied.",
        )

    manifest = {
        "sources": sorted({row["source_path"] for row in rows + appendix}),
        "source_sha256": {
            path: source_hash(WORKSPACE / path) for path in sorted({row["source_path"] for row in rows + appendix})
        },
        "native_comparison_scope": "formal and controlled runs with exact p00_10 native decile records",
        "excluded_from_p00_10": "fixed-state appendix three-band positions and MDPO objective_position",
        "renderer_sha256": source_hash(Path(__file__)),
        "counts": {"native_rows": len(rows), "three_band_rows": len(appendix)},
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "figure-audit.json").write_text(json.dumps(AUDITS, indent=2) + "\n")
    print(f"Wrote {len(rows)} native inventory rows and {len(appendix)} three-band rows.")


if __name__ == "__main__":
    main()
