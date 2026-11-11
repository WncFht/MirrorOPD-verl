"""Summarize parameter spectra and compare three Qwen teacher updates on CPU.

Ledger: an observation is a pair/layer/role; no layer is a teacher replicate.
CDF lines average three k_proj matrices at equal rank fractions; colors identify
teacher pairs and are fixed across teacher/student panels. Similarity separates
normalized singular values, actual update cosine, and rank16 projector overlap.
"""

import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))
from examples.mirroropd.research.studies.spectral.weights import TensorCheckpoint  # noqa: E402

sys.path.insert(0, str(ROOT.parents[2] / "dotfiles/.agents/skills/figure-viz/assets/mpl"))
import fvstyle as fv  # noqa: E402

OUT = ROOT / "spectrum-screen-v1"
SOURCES = {}
PAIRS = ["qwen-grpo", "qwen-sft", "qwen-thinking-space-sft", "r1-justrl"]
LABELS = ["Qwen GRPO", "Samsung SFT", "Thinking-Space SFT", "R1 JustRL"]


def read(path):
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def summarize(singular):
    s = np.array(singular)
    energy = s**2
    p = energy / energy.sum()
    return dict(frobenius=float(np.sqrt(energy.sum())), stable_rank=float(1 / p[0]),
                effective_rank=float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())),
                top16=float(p[:16].sum()))


def main():
    torch.set_num_threads(2)
    assert (OUT / "complete.json").exists()
    rows, curves = [], {}
    for pair in PAIRS:
        curves[pair] = dict(teacher=[], student=[])
        for layer in [0, 13, 27]:
            new = pair == "qwen-thinking-space-sft"
            path = OUT / f"thinking-space-l{layer:02d}.json" if new else ROOT / f"temporal-svd/{pair}-l{layer:02d}.json"
            data = read(path)
            stages = {x["step"]: x for x in data["stages"]}
            base_s = np.array(data["base_singular_values"])
            base_norm = np.linalg.norm(base_s)
            teacher_s = np.array(data["teacher_singular_values"])
            delta_s = np.array(data["teacher_update_singular_values"])
            curves[pair]["teacher"].append(delta_s)
            curves[pair]["student"].append(np.array(stages[320]["update_singular_values"]))
            for role, step, singular in [
                ("base-weight", 0, base_s), ("teacher-weight", 0, teacher_s),
                ("teacher-update", 0, delta_s),
                *[("student-weight", step, x["weight_singular_values"]) for step, x in stages.items()],
                *[("student-update", step, x["update_singular_values"]) for step, x in stages.items()],
            ]:
                row = dict(pair=pair, layer=layer, tensor=data["tensor"], role=role, step=step,
                           **summarize(singular), norm_over_base=summarize(singular)["frobenius"] / base_norm,
                           kappa_spec=None, rho_sigma=None)
                if role in ["teacher-update", "student-update"]:
                    if new:
                        facts = data["teacher"] if role == "teacher-update" else stages[step]["opd"]
                    else:
                        actual_step = step if step else 320
                        directory = ROOT / f"jobs/{pair}.weights-s{actual_step}-l{layer:02d}/attempt-01/weights"
                        found = [read(p) for p in sorted(directory.glob("[0-9]*.json"))]
                        fact = next(x for x in found if x["tensor"] == data["tensor"])
                        facts = fact["teacher" if role == "teacher-update" else "opd"]
                    row.update(kappa_spec=facts["kappa_spec"], rho_sigma=facts["rho_sigma"])
                rows.append(row)
            for step, stage in stages.items():
                if new:
                    facts = stage["missing"]
                else:
                    directory = ROOT / f"jobs/{pair}.weights-s{step}-l{layer:02d}/attempt-01/weights"
                    facts = next(read(p)["missing"] for p in directory.glob("[0-9]*.json")
                                 if json.loads(p.read_text())["tensor"] == data["tensor"])
                rows.append(dict(pair=pair, layer=layer, tensor=data["tensor"], role="missing-update", step=step,
                                 frobenius=facts["frobenius"], stable_rank=facts["stable_rank"], effective_rank=None,
                                 top16=facts["top_energy"]["16"], norm_over_base=facts["frobenius"] / base_norm,
                                 kappa_spec=None, rho_sigma=None))
    with (OUT / "fingerprints.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for pair in PAIRS:
        for role, step in sorted({(r["role"], r["step"]) for r in rows if r["pair"] == pair}):
            selected = [r for r in rows if (r["pair"], r["role"], r["step"]) == (pair, role, step)]
            summaries.append(dict(pair=pair, role=role, step=step, matrices=len(selected), **{
                name: float(np.median([r[name] for r in selected])) if selected[0][name] is not None else None
                for name in ["norm_over_base", "stable_rank", "effective_rank", "top16", "kappa_spec", "rho_sigma"]
            }))
    # Direct comparison only within the common Qwen coordinate system.
    manifest = read(ROOT / "manifest.json")
    configs = {x["label"]: x for x in manifest["configs"]}
    protocol = read(OUT / "protocol.json")
    base = TensorCheckpoint(configs["qwen-sft"]["base"])
    paths = {p: configs[p]["teacher"] for p in ["qwen-grpo", "qwen-sft"]}
    paths[PAIRS[2]] = protocol["checkpoints"]["teacher"]["path"]
    teachers = {p: TensorCheckpoint(path) for p, path in paths.items()}
    similarity = []
    for layer in [0, 13, 27]:
        key = f"model.layers.{layer}.self_attn.k_proj.weight"
        b = base.get(key).double()
        directions = {p: c.get(key).double() - b for p, c in teachers.items()}
        frames = {p: torch.linalg.svd(delta, full_matrices=False) for p, delta in directions.items()}
        for p, q in itertools.combinations(paths, 2):
            u, s, v = frames[p]
            uq, sq, vq = frames[q]
            cosine = float((directions[p] * directions[q]).sum() / (directions[p].norm() * directions[q].norm()))
            similarity.append(dict(layer=layer, first=p, second=q, update_cosine=cosine,
                                   normalized_spectrum_distance=float((s / s.norm() - sq / sq.norm()).norm()),
                                   left_overlap16=float((u[:, :16].T @ uq[:, :16]).square().sum() / 16),
                                   right_overlap16=float((v[:16] @ vq[:16].T).square().sum() / 16)))
    result = dict(scope="k_proj layers0/13/27; four teacher pairs; descriptive screen, no fitted classifier",
                  summaries=summaries, same_base_teacher_similarity=similarity,
                  missing_coverage=["RAW/CCS and MiMo parameter spectra", "Thinking-Space full-layer spectra",
                                    "long-stage spectral dynamics", "independent teacher-level predictor validation"],
                  sources=SOURCES)
    (OUT / "screen-summary.json").write_text(json.dumps(result, indent=2))
    fv.apply_rc("paper", fontsize=10)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained", sharey=True)
    grid = np.geomspace(1 / 256, 1, 100)
    colors = fv.series_colors("paper", 4)
    for ax, role, title in zip(axes, ["teacher", "student"], ["Teacher update T-B", "Student update S320-B"], strict=True):
        for color, pair, label in zip(colors, PAIRS, LABELS, strict=True):
            values = []
            for s in curves[pair][role]:
                cdf = np.cumsum(s**2) / np.sum(s**2)
                values.append(np.interp(grid, np.arange(1, len(s) + 1) / len(s), cdf))
            ax.plot(grid, np.mean(values, axis=0), color=color, label=label)
        ax.set(xscale="log", xlabel="Retained rank / min(matrix shape)", ylabel="Cumulative squared singular-value fraction",
               title=title, ylim=(0, 1.02), xlim=(grid[0], 1))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=2, frameon=False, fontsize=8)
    fig.get_layout_engine().set(rect=(0, 0.09, 1, 0.88))
    fig.text(0.5, 0.015, "Exact FP64 SVD; mean of k_proj layers 0/13/27; four pairs, not four method classes",
             ha="center", fontsize=8, color=fv.NEUTRALS["baseline"])
    fig.canvas.draw()
    audit = [str(x) for x in fv.audit(fig, print_slot=11)]
    (OUT / "spectrum-cdf.audit.json").write_text(json.dumps(audit, indent=2))
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"spectrum-cdf.{extension}", dpi=180, bbox_inches="tight")
    print(json.dumps(summaries, indent=2))
    print("figure audit", audit)
    SOURCES[str(Path(__file__))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / "summary-provenance.json").write_text(json.dumps(SOURCES, indent=2))


if __name__ == "__main__":
    main()
