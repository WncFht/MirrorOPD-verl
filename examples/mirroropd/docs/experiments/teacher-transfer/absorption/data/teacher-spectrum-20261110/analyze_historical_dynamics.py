"""Recompute frozen historical KL dynamics without loading models.

Semantic ledger: each curve is one original branch, fixed prompt panel, and
prefix bank. Historical fixed-H1024 and construction token-mean risks remain
separate. Windows are predefined checkpoint differences, paired by question.
Geometry ledger: x is actual optimizer updates; y is absolute RKL in nats per
budget position (history) or per valid token (construction). No branch stitching.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
BASE = REPO / "examples/mirroropd/docs/experiments/teacher-transfer"
HISTORY = REPO.parent / "wandb-logs/MirrorOPD-verl/analysis"
sys.path.insert(0, str(REPO.parent / "dotfiles/.agents/skills/figure-viz/assets/mpl"))
import fvstyle as fv  # noqa: E402

OUT = ROOT / "historical-dynamics-20261110"
PANELS = {
    "train_pool": list("ABCDEFGHIJKLMNOPQRST"),
    "held_math": [f"M{i:02d}" for i in range(1, 5)],
    "held_all": [*[f"M{i:02d}" for i in range(1, 5)], "Code01", "Code02", "IF01", "IF02"],
}
BRANCHES = {
    "GRPO-KL thinking": ("qwen3-teacher-lineage-20261029", "Q3-grpo-kl"),
    "Samsung SFT thinking": ("qwen3-teacher-lineage-20261029", "Q3-sft-gsm8k"),
    "Thinking-Space SFT thinking": ("qwen3-teacher-lineage-20261029", "Q3-sft-deepmath"),
    "JustRL fixed-S0": ("r1-samesize-deterministic320-20261030", "CONT-J"),
    "Samsung SFT nonthinking": ("qwen3-nonthinking-origin-controls1280-20261030", "NT-C-sft-gsm8k"),
    "Thinking-Space SFT nonthinking": ("qwen3-mode-nonthinking-wave3-20261029", "NT-SFT"),
}
WINDOWS = [(0, 64), (64, 320), (256, 320), (320, 640), (640, 1280), (1216, 1280)]
SOURCES = {}


def read(path):
    payload = path.read_bytes()
    SOURCES[str(path.resolve())] = hashlib.sha256(payload).hexdigest()
    return json.loads(payload)


def rows_csv(path):
    SOURCES[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return list(csv.DictReader(path.open()))


def write_csv(name, rows):
    with (OUT / name).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def historical():
    curves, windows, receipts, paired = [], [], {}, {}
    rng = np.random.default_rng(20261110)
    for label, (study, arm) in BRANCHES.items():
        directory = HISTORY / study / "arms" / arm
        receipts[label] = read(directory / "complete.json")
        snapshots = {}
        identity = None
        for path in sorted(directory.glob("fixed-*.json"), key=lambda p: int(p.stem.split("-")[-1])):
            step = int(path.stem.split("-")[-1])
            records = read(path)
            current = {(str(r["id"]), r["group"], r["domain"], r["split"], r["length"]) for r in records}
            if identity is None:
                identity = current
            assert current == identity, (label, step, "changed question identity")
            snapshots[step] = {}
            for panel, groups in PANELS.items():
                selected = [r for r in records if r["group"] in groups]
                assert selected
                values = {str(r["id"]): r["stats"]["all"]["rkl"] / 1024 for r in selected}
                assert len(values) == len(selected)
                snapshots[step][panel] = values
                value = float(np.mean(list(values.values())))
                initial = float(np.mean(list(snapshots[0][panel].values())))
                curves.append(
                    dict(
                        branch=label,
                        step=step,
                        panel=panel,
                        n=len(values),
                        rkl=value,
                        initial_rkl=initial,
                        remaining=value / initial,
                    )
                )
        paired[label] = snapshots
        for start, stop in WINDOWS:
            if start not in snapshots or stop not in snapshots:
                continue
            for panel in PANELS:
                first, last = snapshots[start][panel], snapshots[stop][panel]
                ids = sorted(first)
                assert ids == sorted(last)
                a, b = np.array([first[i] for i in ids]), np.array([last[i] for i in ids])
                delta = b - a
                samples = delta[rng.integers(len(ids), size=(2000, len(ids)))].mean(axis=1)
                low, high = np.quantile(samples, [0.025, 0.975])
                windows.append(
                    dict(
                        branch=label,
                        panel=panel,
                        start=start,
                        stop=stop,
                        n=len(ids),
                        initial=float(a.mean()),
                        final=float(b.mean()),
                        delta=float(delta.mean()),
                        delta_ci_low=float(low),
                        delta_ci_high=float(high),
                        decrease_percent=float(100 * (1 - b.mean() / a.mean())),
                        delta_per_64=float(delta.mean() * 64 / (stop - start)),
                    )
                )
    return curves, windows, receipts, paired


def save(fig, name, caption):
    fig.get_layout_engine().set(rect=(0, 0.09, 1, 0.88))
    fig.text(0.5, 0.015, caption, ha="center", fontsize=8, color=fv.NEUTRALS["baseline"])
    fig.canvas.draw()
    audit = [str(x) for x in fv.audit(fig, print_slot=11)]
    (OUT / f"{name}.audit.json").write_text(json.dumps(audit, indent=2))
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    print(name, audit)
    plt.close(fig)


def plot(curves, sft):
    fv.apply_rc("paper", fontsize=10)
    colors = fv.series_colors("paper", 4)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    groups = [list(BRANCHES)[:4], list(BRANCHES)[4:]]
    for ax, labels, title in zip(
        axes, groups, ["Original thinking / R1 branches", "Long nonthinking SFT branches"], strict=True
    ):
        for color, label in zip(colors if len(labels) == 4 else colors[1:3], labels, strict=True):
            rows = [r for r in curves if r["branch"] == label and r["panel"] == "held_all"]
            ax.plot([r["step"] for r in rows], [r["rkl"] for r in rows], color=color, label=label, linewidth=1.8)
        ax.set(xlabel="Optimizer updates", ylabel="Held-out RKL (nats / H position)", yscale="log", title=title)
        ax.set_ylim(5e-4, 1)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.38), frameon=False, fontsize=8, ncol=2)
    save(fig, "absolute-kl-dynamics", "Fixed S0 prefixes; H=1024; Math64 + Code32 + IF32; distinct original branches")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained", sharey=True)
    for ax, bank in zip(axes, ["Base-bank", "teacher-bank"], strict=True):
        for color, arm in zip(colors, ["RAW", "CCS-rho0", "CCS-rho1", "MiMo-SFT"], strict=True):
            rows = [r for r in sft if r["bank"] == bank and r["arm"] == arm and r["domain"] == "macro4"]
            if not rows:
                continue
            ax.plot(
                [int(r["step"]) for r in rows],
                [float(r["rkl"]) for r in rows],
                "o-",
                markersize=3,
                color=color,
                label=arm,
            )
        ax.set(xlabel="Optimizer updates", ylabel="Macro4 RKL (nats / valid token)", yscale="log", title=bank)
        ax.set_ylim(4e-4, 3)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False, fontsize=8)
    save(
        fig, "sft-prefix-dynamics", "Qwen H<=256; MiMo native long banks; different training budgets and prefix lengths"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained", sharey=True)
    for ax, branch in zip(axes, list(BRANCHES)[4:], strict=True):
        for color, panel, label in zip(
            colors, ["train_pool", "held_all"], ["Fixed training pool", "Held-out"], strict=False
        ):
            rows = [r for r in curves if r["branch"] == branch and r["panel"] == panel]
            ax.plot([r["step"] for r in rows], [r["rkl"] for r in rows], color=color, label=label)
        ax.set(xlabel="Optimizer updates", ylabel="RKL (nats / H position)", yscale="log", title=branch)
        ax.set_ylim(5e-4, 0.3)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), frameon=False, ncol=2)
    save(fig, "long-sft-train-heldout", "Fixed S0 prefixes; training pool320; held-out Math64 + Code32 + IF32; H=1024")


def main():
    OUT.mkdir(exist_ok=True)
    curves, windows, receipts, paired = historical()
    prefix_checks = []
    for earlier, later, arm in [
        ("qwen3-nonthinking-origin-controls320-20261030", BRANCHES["Samsung SFT nonthinking"][0], "NT-C-sft-gsm8k"),
        ("qwen3-mode-nonthinking-stage320-20261029", BRANCHES["Thinking-Space SFT nonthinking"][0], "NT-SFT"),
    ]:
        for step in range(0, 321, 16):
            paths = [HISTORY / study / "arms" / arm / f"fixed-{step:03d}.json" for study in [earlier, later]]
            digests = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
            assert digests[0] == digests[1], (arm, step, "resume history changed")
            prefix_checks.append(
                dict(arm=arm, step=step, sha256=digests[0], earlier=str(paths[0]), later=str(paths[1]))
            )
    sft = rows_csv(BASE / "data/teacher-construction-20261108/sft-comparison-v1/curves.csv")
    # Preserve evidence from a distinct on-policy run, not a held-out curve.
    online = [
        r for r in rows_csv(BASE / "data/case-position-kl-20261107/r1-1.5b-endpoints.csv") if r["experiment"] == "E35"
    ]
    online_windows = []
    for start, stop in [(1, 25), (101, 125), (201, 225), (226, 250)]:
        for endpoint in ["first", "last"]:
            values = [
                float(r["full_kl"]) for r in online if start <= int(r["outer"]) <= stop and r["endpoint"] == endpoint
            ]
            online_windows.append(
                dict(outer_start=start, outer_stop=stop, endpoint=endpoint, n=len(values), rkl=float(np.mean(values)))
            )
    write_csv("historical-curves.csv", curves)
    write_csv("historical-windows.csv", windows)
    (OUT / "paired-question-risks.json").write_text(json.dumps(paired))
    (OUT / "analysis.json").write_text(
        json.dumps(
            dict(
                protocol="Historical fixed-H1024 question mean; standard Math64/Code32/IF32; train pool320",
                uncertainty=(
                    "2000 paired question bootstraps; one training seed; windows predefined; no multiplicity correction"
                ),
                windows=windows,
                resume_prefix_checks=prefix_checks,
                construction_curves=sft,
                justrl_online_windows=online_windows,
                receipts=receipts,
                sources=SOURCES,
            ),
            indent=2,
        )
    )
    plot(curves, sft)
    SOURCES[str(Path(__file__))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / "provenance.json").write_text(json.dumps(SOURCES, indent=2))
    print("curves", len(curves), "windows", len(windows))


if __name__ == "__main__":
    main()
