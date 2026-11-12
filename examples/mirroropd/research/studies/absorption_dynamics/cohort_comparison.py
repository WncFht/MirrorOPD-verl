# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit teacher origins and compare absorption across immutable matrix records.

Ledger: each scatter point is one pair, teacher panels show pair-level summaries.
Color encodes the teacher's added SFT/RL stage; cross-size is a separate category.
Angles describe cumulative-update right frames, not complete-weight frames.
Partial coverage is retained explicitly and matched layers support comparisons.
"""

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from examples.mirroropd.research.core.plotting import save_figure

from ..spectral.update_dynamics import atomic_json
from .geometry_plotting import grid, setup
from .geometry_probes import relation

ORIGIN = {f"E{i}": "SFT" if i in (96, 101, 104) else "Cross-size" if i == 102 else "RL" for i in range(94, 105)}


def shape_distance(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a / a.sum() - b / b.sum()))


def features(record):
    base = np.asarray(record["initial_singular_values"])
    base_energy = float(np.dot(base, base))
    stages = record["stages"]
    first, final = stages[0], stages[-1]
    q = min(record["shape"])
    values = dict(
        base_energy=base_energy,
        student_energy=final["energy"],
        student_amplitude=math.sqrt(final["energy"] / base_energy),
        student_stable_rank=final["stable_rank"],
        student_rank90_fraction=final["rank90"] / q,
        student_weight_sd=shape_distance(base, final["weight"]["singular_values"]),
        student_update_sd=shape_distance(first["singular_values"], final["singular_values"]),
        student_weight_scale=float(np.dot(base, final["weight"]["singular_values"]) / base_energy),
    )
    for key in ("delta_sigma", "rho_sigma", "kappa_spec"):
        values[f"student_{key}"] = final["weight"][key]
    for k in (16, 32, 64, 128, 256):
        for label, stage in (("early", first), ("final", final)):
            values[f"{label}_energy{k}"] = next(
                x["captured_energy"] for x in stage["ranks"] if x["requested_rank"] == k
            )
        values[f"energy{k}_change"] = values[f"final_energy{k}"] - values[f"early_energy{k}"]
        for anchor in (16, 64, 192, 320, 480):
            pair = relation(record, anchor, 640, k)
            values[f"overlap{k}_{anchor}_final"] = pair["adjusted_overlap"]
            values[f"angle{k}_{anchor}_final"] = pair["median_angle_deg"]
            values[f"anglep90_{k}_{anchor}_final"] = pair["p90_angle_deg"]
            values[f"capture{k}_{anchor}_final"] = pair["a_captures_b"]
            values[f"capture_optimal{k}_{anchor}_final"] = pair["a_captures_b"] / values[f"final_energy{k}"]
        for anchor in (16, 320, 480):
            increment = record["increments"][-1]
            capture = next(x for x in increment["captures"] if x["anchor"] == anchor and x["requested_rank"] == k)
            values[f"late_increment{k}_anchor{anchor}"] = capture["relative_to_optimal"]
        wf = next(f for f in final["weight"]["frames"] if f["requested_rank"] == k)
        values[f"weight_frame_angle{k}"] = wf["median_angle_deg"]
    teacher = record.get("teacher")
    if teacher and teacher["energy"]:
        tw = teacher["weight"]
        values.update(
            teacher_energy=teacher["energy"],
            teacher_amplitude=math.sqrt(teacher["energy"] / base_energy),
            teacher_stable_rank=teacher["stable_rank"],
            teacher_rank90_fraction=teacher["rank90"] / q,
            teacher_weight_sd=shape_distance(base, tw["singular_values"]),
            teacher_weight_scale=float(np.dot(base, tw["singular_values"]) / base_energy),
        )
        for key in ("delta_sigma", "rho_sigma", "kappa_spec"):
            values[f"teacher_{key}"] = tw[key]
        for k in (16, 32, 64, 128, 256):
            values[f"teacher_energy{k}"] = next(
                x["captured_energy"] for x in teacher["ranks"] if x["requested_rank"] == k
            )
        for key in ("norm_ratio", "cosine", "residual_energy_ratio"):
            values[f"parameter_{key}"] = final["teacher_comparison"][key]
        for k in (16, 64, 256):
            f = next(x for x in final["teacher_comparison"]["frames"] if x["requested_rank"] == k)
            values[f"teacher_update_overlap{k}"] = f["adjusted_overlap"]
    return values


def summarize(rows):
    result = {}
    for key in sorted({k for r in rows for k in r["values"]}):
        vals = [r["values"][key] for r in rows if key in r["values"]]
        result[key] = dict(
            median=float(np.median(vals)),
            p25=float(np.quantile(vals, 0.25)),
            p75=float(np.quantile(vals, 0.75)),
            count=len(vals),
        )
    for label in ("teacher", "student"):
        if rows and all(f"{label}_energy" in r["values"] for r in rows):
            result[f"{label}_global_amplitude"] = math.sqrt(
                sum(r["values"][f"{label}_energy"] for r in rows) / sum(r["values"]["base_energy"] for r in rows)
            )
    return result


def collect(sources, output):
    pairs, accepted = {}, {}
    for source in sources:
        plan = json.loads((source / "protocol.json").read_text())
        for name, meta in plan["pairs"].items():
            if name in pairs:
                raise ValueError("Pair identities must be disjoint across sources")
            pairs[name] = dict(origin=ORIGIN[name], expected=meta["matrix_count"], matrices=[], risks=meta["risks"])
        accepted.update({(t["condition"], t["tensor"]): t for t in plan["tasks"]})
    for source in sources:
        for path in sorted(source.glob("matrix-*.json.gz")):
            payload = path.read_bytes()
            r = json.loads(gzip.decompress(payload))
            task = r["task"]
            if task != accepted[task["condition"], task["tensor"]]:
                raise ValueError("Matrix identity differs from accepted scientific plan")
            if [x["step"] for x in r["stages"]] != [x[0] for x in task["points"]]:
                raise ValueError("Incomplete checkpoint trajectory")
            pairs[task["condition"]]["matrices"].append(
                dict(
                    tensor=task["tensor"],
                    layer=int(task["tensor"].split(".")[2]),
                    module=task["tensor"].split(".")[-2],
                    source_sha256=hashlib.sha256(payload).hexdigest(),
                    values=features(r),
                )
            )
    for pair in pairs.values():
        counts = defaultdict(int)
        for m in pair["matrices"]:
            counts[m["layer"]] += 1
        layers = [layer for layer, count in counts.items() if count == 7]
        pair["complete_layers"] = sorted(layers)
        pair["summary"] = summarize([m for m in pair["matrices"] if m["layer"] in layers])
        pair["measured"] = len(pair["matrices"])
    common = sorted(set.intersection(*(set(p["complete_layers"]) for p in pairs.values())))
    for pair in pairs.values():
        pair["matched_summary"] = summarize([m for m in pair["matrices"] if m["layer"] in common])
        pair["modules"] = {
            module: summarize(
                [m for m in pair["matrices"] if m["module"] == module and m["layer"] in pair["complete_layers"]]
            )
            for module in sorted({m["module"] for m in pair["matrices"]})
        }
    data = dict(
        schema="teacher-absorption-cohorts-v1",
        common_layers=common,
        pairs=dict(sorted(pairs.items(), key=lambda x: int(x[0][1:]))),
    )
    data["descriptive_correlations"] = screen(pairs)
    atomic_json(output, data)
    return data


def screen(pairs):
    # Pair, not matrix, is the statistical unit; these are descriptive screens.
    correlations = []
    complete = [p for p in pairs.values() if p["measured"] == p["expected"]]
    for feature in sorted(set.union(*(set(p["summary"]) for p in complete))):
        for bank in ("public_initial", "public_teacher"):
            for domain in ("math", "all"):
                for relative in (False, True):
                    x, y = [], []
                    key = f"eval/rkl/{bank}/{domain}/question_mean"
                    for p in complete:
                        if feature not in p["summary"]:
                            continue
                        v = p["summary"][feature]
                        x.append(v["median"] if isinstance(v, dict) else v)
                        y.append(p["risks"][-1]["metrics"][key] / (p["risks"][0]["metrics"][key] if relative else 1))
                    if len(x) >= 3 and np.ptp(x) and np.ptp(y):
                        correlations.append(
                            dict(
                                feature=feature,
                                bank=bank,
                                domain=domain,
                                relative=relative,
                                n=len(x),
                                spearman=float(spearmanr(x, y).statistic),
                            )
                        )
    return correlations


def place_labels(fig, ax, points):
    """Place direct point labels inside axes using nonoverlapping display boxes."""
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = []
    locations = [ax.transData.transform((x, y)) for _, x, y in points]
    for name, x, y in points:
        placed = False
        for dy in (8, -12, 20, -24, 32, -36, 44, -48):
            for dx in (5, -5, 26, -26, 45, -45):
                text = ax.annotate(
                    name,
                    (x, y),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha="left" if dx > 0 else "right",
                    va="center",
                    fontsize=8,
                )
                box = text.get_window_extent(renderer).expanded(1.08, 1.2)
                boundary = ax.get_window_extent(renderer)
                if (
                    boundary.contains(box.x0, box.y0)
                    and boundary.contains(box.x1, box.y1)
                    and not any(box.overlaps(other) for other in occupied)
                    and not any(box.overlaps(Bbox.from_bounds(px - 8, py - 8, 16, 16)) for px, py in locations)
                ):
                    occupied.append(box)
                    placed = True
                    break
                text.remove()
            if placed:
                break
        if not placed:
            raise ValueError(f"Cannot place pair label {name}; enlarge figure")


def scatter_origins(ax, points, colors):
    """Use one artist per scientific category, retaining per-pair direct labels."""
    for origin, color in colors.items():
        subset = [(x, y) for x, y, group in points if group == origin]
        if subset:
            x, y = zip(*subset, strict=True)
            ax.scatter(x, y, color=color, marker="s" if origin == "SFT" else "o", s=40)


def render(data, destination):
    plt, fv = setup()
    palette = fv.series_colors("paper", 3)
    colors = dict(zip(("RL", "SFT", "Cross-size"), palette, strict=True))
    names = list(data["pairs"])
    destination.mkdir(parents=True, exist_ok=True)
    fig, axes = grid(2, 2, width=12, height=8)
    fields = [
        ("teacher_global_amplitude", "Teacher update / initial weight (%)", True),
        ("teacher_energy16", "Teacher top16 update energy (%)", False),
        ("teacher_rho_sigma", "Teacher rho_sigma", False),
        ("teacher_kappa_spec", "Teacher kappa_spec", False),
    ]
    for ax, (key, label, scalar) in zip(axes.flat, fields, strict=True):
        scatter_points = []
        for i, name in enumerate(names):
            p = data["pairs"][name]
            value = p["summary"].get(key)
            if value is None:
                continue
            value = value if scalar else value["median"]
            if key in ("teacher_global_amplitude", "teacher_energy16"):
                value *= 100
            scatter_points.append((i, value, p["origin"]))
            ax.text(i, value * 1.12, f"{value:.2g}", ha="center", va="bottom", fontsize=8)
        scatter_origins(ax, scatter_points, colors)
        ax.set_xticks(
            range(len(names)),
            [n + ("*" if data["pairs"][n]["measured"] < data["pairs"][n]["expected"] else "") for n in names],
            rotation=45,
        )
        ax.set_ylabel(label)
        ax.set_yscale("log")
        ax.margins(y=0.3)
    fig.suptitle("Teacher changes relative to initial student: SFT versus RL")
    save_figure(
        fig,
        destination / "teacher-origins",
        "Teal=RL; orange=SFT; third color=cross-size. "
        "* partial full-layer coverage; matrix medians except global amplitude.",
    )
    fig, axes = grid(2, 3, width=13, height=8)
    fields = [
        ("overlap64_16_final", "Step16/final adjusted overlap k64"),
        ("overlap64_320_final", "Step320/final adjusted overlap k64"),
        ("late_increment64_anchor320", "Late increment capture / optimal k64"),
        ("student_rank90_fraction", "OPD rank90 / min(shape)"),
        ("student_weight_sd", "Full-weight spectral shape drift"),
        ("parameter_residual_energy_ratio", "Parameter residual energy ratio"),
    ]
    label_sets = []
    for ax, (key, label) in zip(axes.flat, fields, strict=True):
        points = []
        scatter_points = []
        for name, p in data["pairs"].items():
            # Exact same layer/module coordinates for every pair, including partial E104.
            v = p["matched_summary"].get(key)
            if v is None:
                continue
            y = p["risks"][-1]["metrics"]["eval/rkl/public_initial/math/question_mean"]
            scatter_points.append((v["median"], y, p["origin"]))
            points.append((name, v["median"], y))
        scatter_origins(ax, scatter_points, colors)
        ax.set_xlabel(label)
        ax.set_ylabel("Final Math RKL: initial bank")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 0.6)
        ax.margins(x=0.2, y=0.25)
        if key == "student_weight_sd":
            ax.set_xscale("log")
        label_sets.append((ax, points))
    fig.suptitle("Absorption versus geometry: identical layer/module coverage")
    for ax, points in label_sets:
        place_labels(fig, ax, points)
    save_figure(
        fig,
        destination / "absorption-features",
        f"Pair-level medians over {len(data['common_layers'])} common layers; "
        "7 projections each. Color=teacher origin; labels=pair IDs.",
    )
    fig, axes = grid(1, 2, width=12, height=4.8)
    label_sets = []
    for ax, bank in zip(axes.flat, ("public_initial", "public_teacher"), strict=True):
        points = []
        scatter_points = []
        for i, name in enumerate(names):
            p = data["pairs"][name]
            metrics = p["risks"][-1]["metrics"]
            scatter_points.append(
                (
                    metrics[f"eval/rkl/{bank}/math/question_mean"],
                    metrics[f"eval/rkl/{bank}/all/question_mean"],
                    p["origin"],
                )
            )
            points.append(
                (name, metrics[f"eval/rkl/{bank}/math/question_mean"], metrics[f"eval/rkl/{bank}/all/question_mean"])
            )
        scatter_origins(ax, scatter_points, colors)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Final Math RKL")
        ax.set_ylabel("Final equal-domain RKL")
        ax.set_title(bank.replace("public_", ""))
        ax.set_xlim(5e-5, 1.5)
        ax.set_ylim(1e-4, 5)
        label_sets.append((ax, points))
    fig.suptitle("Math absorption and broad-domain absorption differ")
    for ax, points in label_sets:
        place_labels(fig, ax, points)
    save_figure(
        fig,
        destination / "math-versus-all",
        "Each point is one completed 640-update training; both axes are absolute RKL; all is equal-domain mean.",
    )
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source", type=Path, nargs="+")
    inputs.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    if args.source:
        if args.output is None:
            parser.error("--source requires --output")
        data = collect(args.source, args.output)
    else:
        data = json.loads(args.data.read_text())
        if args.output:
            data["descriptive_correlations"] = screen(data["pairs"])
            atomic_json(args.output, data)
    if args.figures:
        render(data, args.figures)


if __name__ == "__main__":
    main()
