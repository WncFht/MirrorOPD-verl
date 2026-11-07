# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Plot frozen position objectives and first-update profiles; --refresh reads logs.

Semantic ledger: heatmap = token-mean MDPO objective/eta in one response
decile, before an optimizer operation. Every heatmap shares one color scale
across its first/last panels. Profile lines compare FIRST updates of the
initial and last observed outers; never substitute the last reuse observation.
Teacher RKL and MDPO objective stay in separate figures and tables. The ten
bins use each response's own valid length, not absolute token positions.
E78--E80 and fixed-state three-band diagnostics are outside this comparison.
Geometry ledger: actual outer IDs, ten deciles, no smoothing/interpolation.
Default rendering reads frozen CSV; only --refresh reads changing histories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from examples.mirroropd.research.studies.weighted_targets.metrics import trusted_metrics
from examples.mirroropd.research.studies.weighted_targets.plot_e81_validation import ROOT, style

DOCS = ROOT / "examples/mirroropd/docs/experiments"
LEARNING = DOCS / "learning-signal"
TRANSFER = DOCS / "teacher-transfer"
OUT = LEARNING / "figures/mdpo-position-20261107"
PROFILES = TRANSFER / "figures/first-update-profiles-20261107"
BINS = [f"p{i:02d}_{i + 10:02d}" for i in range(0, 100, 10)]
LABELS = [f"{i}-{i + 10}%" for i in range(0, 100, 10)]
MDPO = "distill/mdpo/objective_position"
RKL = "distill/full_kl_position"
RUNS = {
    "E55": ROOT.parent / "wandb-logs/derived/MirrorOPD-verl/offline-run-20261030_094536-j5uaal3j/history.jsonl",
    "E56": ROOT.parent / "wandb-logs/derived/MirrorOPD-verl/offline-run-20261029_234048-zsnhqzts/history.jsonl",
    "E57": ROOT.parent / "wandb-logs/derived/MirrorOPD-verl/offline-run-20261030_015214-6j5knkmk/history.jsonl",
    "E81": ROOT / "runs/topd-large-batch-20261105/formal/E81/metrics.jsonl",
    "E82": ROOT / "runs/topd-large-batch-20261105/formal/E82/metrics.jsonl",
    "E83": ROOT / "runs/topd-learning-signal-20261106/formal/E83/metrics.jsonl",
}
TEACHERS = {
    **dict.fromkeys(RUNS, "Qwen3-30B-A3B-Instruct-2507"),
    "E71": "MiMo-7B-RL",
    "E72": "MiMo-7B-RL-Zero",
    "E73": "MiMo-7B-SFT",
    "E35": "JustRL-DeepSeek-1.5B",
    "E48": "Polaris-7B-Preview",
    "E49": "Polaris-7B-Preview (conditional FKL training)",
    "E39": "Qwen3-8B",
    "E41": "Qwen3-4B",
    "E43": "Qwen3-4B-Instruct-2507",
    "termination-baseline": "Qwen3-4B (native RKL training)",
    "termination-aligned": "Qwen3-4B (EOS-aligned training)",
}
FV = None


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def stamp(path, raw=None):
    raw = path.read_bytes() if raw is None else raw
    return {"path": str(path.relative_to(ROOT.parent)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def vector(row, prefix):
    values = np.array([float(row[f"{prefix}/{b}/mean"]) for b in BINS])
    assert np.isfinite(values).all() and (values > 0).all()
    return values


def freeze():
    sources, audits, endpoints, profiles = [], {}, [], []
    for experiment, path in RUNS.items():
        original = path.read_bytes()
        # Ignore a trailing, not-yet-complete append, never a complete corrupt record.
        original = original[: original.rfind(b"\n") + 1]
        clean, exclusion = trusted_metrics(path, original)
        sources.append(stamp(path, original))
        if exclusion is not None:
            sources.append(stamp(path.with_name("metrics-exclusions.json"), exclusion))
        rows = [json.loads(line) for line in clean.splitlines()]
        rows = [r.get("data", r) for r in rows]
        rows = [r for r in rows if f"{MDPO}/p00_10/mean" in r]
        assert [int(r["step/update"]) for r in rows] == list(range(1, len(rows) + 1)), experiment
        groups = defaultdict(list)
        errors, position_errors, normalizers = [], [], []
        for row in rows:
            u, outer = int(row["step/update"]), int(row["step/outer"])
            assert outer == (u - 1) // 16 + 1
            groups[outer].append(row)
            target, log_z = float(row["distill/target_kl"]), float(row["distill/target_log_z"])
            errors.append(abs(float(row["distill/mdpo/objective"]) - (target - log_z) / 0.1))
            normalizers.append(abs(log_z))
            sums = sum(float(row[f"{MDPO}/{b}/sum"]) for b in BINS)
            count = sum(float(row[f"{MDPO}/{b}/token_count"]) for b in BINS)
            position_errors.append(abs(sums / count - float(row["distill/mdpo/objective"])))
            for b in BINS:
                assert float(row[f"{MDPO}/{b}/token_count"]) > 0
                assert np.isclose(
                    float(row[f"{MDPO}/{b}/sum"]) / float(row[f"{MDPO}/{b}/token_count"]),
                    float(row[f"{MDPO}/{b}/mean"]),
                    atol=1e-8,
                )
        assert max(errors) < 1e-6 and max(position_errors) < 1e-6
        # Keep a last partially logged outer visible as such, without pretending
        # that its last observation is the sixteenth update.
        for outer, group in sorted(groups.items()):
            for endpoint, row in (("first", group[0]), ("last", group[-1])):
                endpoints.append(
                    {
                        "experiment": experiment,
                        "outer": outer,
                        "endpoint": endpoint,
                        "update": int(row["step/update"]),
                        "reuse": (int(row["step/update"]) - 1) % 16 + 1,
                        "observations_in_outer": len(group),
                        **dict(zip(BINS, vector(row, MDPO), strict=True)),
                    }
                )
        for which, outer in (("initial", min(groups)), ("latest", max(groups))):
            row = groups[outer][0]
            assert (int(row["step/update"]) - 1) % 16 == 0
            profiles.append(
                {
                    "experiment": experiment,
                    "metric": "MDPO objective",
                    "which": which,
                    "outer": outer,
                    "update": int(row["step/update"]),
                    **dict(zip(BINS, vector(row, MDPO), strict=True)),
                }
            )
        audits[experiment] = {
            "updates": len(rows),
            "last_observed_outer": max(groups),
            "last_outer_observations": len(groups[max(groups)]),
            "max_objective_identity_error": max(errors),
            "max_weighted_position_identity_error": max(position_errors),
            "max_abs_log_z": max(normalizers),
            "native_teacher_position_present": any(f"{RKL}/p00_10/mean" in r for r in rows),
        }
    # Reuse the already audited source adapters, without invoking their renderer.
    path = TRANSFER / "figures/plot_case_position_kl.py"
    spec = importlib.util.spec_from_file_location("case_position_sources", path)
    cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases)
    qwen, e35, paths = cases.collect()
    groups = qwen | e35
    for experiment, reuse, count in (("E71", 32, 25), ("E72", 16, 20), ("E73", 16, 20)):
        path = ROOT / f"runs/mimo-stage-transfer-20261101/formal/{experiment}/metrics.jsonl"
        paths.append(path)
        groups[experiment] = cases.from_history(path, True, reuse, count)
    for experiment, path in (
        ("E48", TRANSFER / "cross-origin/polaris/data/polaris-complete-20261027/run/history.jsonl"),
        ("E49", cases.E49_HISTORY),
    ):
        paths.append(path)
        groups[experiment] = cases.from_history(path, False, 16, 256)
    paths.append(DOCS.parent / "archive/experiments/polaris/data/polaris-grcuotmi-20261026/E35/history.jsonl")
    term_path = DOCS.parent / "infra/model-compatibility/termination/data/full-kl-position.csv"
    paths.extend([term_path, Path(cases.__file__)])
    term_rows = read_csv(term_path)
    for arm in ("baseline", "aligned"):
        gg = defaultdict(list)
        for outer in sorted({int(r["outer"]) for r in term_rows if r["arm"] == arm}):
            selected = [
                r for r in term_rows if r["arm"] == arm and int(r["outer"]) == outer and r["reuse_index"] == "1"
            ]
            assert len(selected) == 10
            means = {BINS[int(r["bin"])]: float(r["mean"]) for r in selected}
            gg[outer].append(cases.position_row(outer, 1, means))
        groups[f"termination-{arm}"] = gg
    for experiment, gg in groups.items():
        for which, outer in (("initial", min(gg)), ("latest", max(gg))):
            row = gg[outer][0]
            profiles.append(
                {
                    "experiment": experiment,
                    "metric": "Teacher RKL",
                    "which": which,
                    "outer": outer,
                    "update": int(row["step/update"]),
                    **dict(zip(BINS, vector(row, RKL), strict=True)),
                }
            )
    sources.extend(stamp(path) for path in paths)
    sources.extend(
        stamp(ROOT / f"runs/topd-large-batch-20261105/source/verl/trainer/distillation/{name}.py")
        for name in ("output", "objectives")
    )
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "endpoints.csv", endpoints)
    write_csv(OUT / "profiles.csv", profiles)
    manifest = {
        "captured_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "sources": sources,
        "mdpo_audit": audits,
        "selection": "initial outer first update versus last observed outer FIRST update, both pre-operation",
        "excluded": ["E78", "E79", "E80", "fixed-state three-band diagnostics"],
    }
    (OUT / "provenance.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def save(fig, directory, stem, caption):
    directory.mkdir(parents=True, exist_ok=True)
    fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=9, color=FV.NEUTRALS["baseline"])
    fig.canvas.draw()
    # Ten deciles deliberately reuse four hues with distinct styles/markers.
    # The palette-length warning counts series, ignoring these extra channels.
    ignore = ["palette-overflow"] if stem.endswith("-lines") else []
    problems = FV.audit(fig, print_slot=fig.get_figwidth(), ignore=ignore)
    lines = FV.format_audit_lines(problems)
    (directory / f"{stem}.audit.txt").write_text("\n".join(lines) + "\n" if lines else "PASS: fv.audit(fig)\n")
    if any(FV.problem_severity(p) == "error" for p in problems):
        raise RuntimeError(f"Figure audit failed: {stem}: {lines}")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(directory / f"{stem}.{extension}", dpi=160, facecolor=FV.NEUTRALS["paper"])
    plt.close(fig)


def heatmaps(experiment, rows):
    outers = sorted({int(r["outer"]) for r in rows})
    lookup = {(int(r["outer"]), r["endpoint"]): r for r in rows}
    for name, xs in (("all", outers), ("early", outers[:5]), ("late", outers[-5:])):
        mats = [np.array([[float(lookup[o, endpoint][b]) for o in xs] for b in BINS]) for endpoint in ("first", "last")]
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.subplots_adjust(left=0.08, right=0.87, bottom=0.20, top=0.81, wspace=0.27)
        for ax, mat, endpoint in zip(axes, mats, ("First", "Last observed"), strict=True):
            image = ax.imshow(
                mat,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap=FV.ramp_cmap("seq"),
                vmin=0,
                vmax=max(float(m.max()) for m in mats),
                extent=(xs[0] - 0.5, xs[-1] + 0.5, -0.5, 9.5),
            )
            ticks = xs if len(xs) <= 8 else sorted({round(v) for v in np.linspace(xs[0], xs[-1], 7)})
            ax.set_xticks(ticks)
            ax.set_yticks(range(10), LABELS)
            ax.set_xlabel("Outer")
            ax.set_title(f"{endpoint} update observation", pad=12)
        axes[0].set_ylabel("Relative response position")
        cb = fig.colorbar(image, ax=list(axes), pad=0.03, shrink=0.85)
        cb.set_label("MDPO objective (token mean; shared scale)")
        fig.suptitle(
            f"{experiment} | Teacher: {TEACHERS[experiment]}\nMDPO position objective | {name} outer window", y=0.97
        )
        save(
            fig,
            OUT,
            f"{experiment.lower()}-heatmap-{name}",
            ("First/last are pre-operation observations; one shared linear color scale within this figure."),
        )


def training_lines(experiment, rows):
    outers = sorted({int(r["outer"]) for r in rows})
    lookup = {(int(r["outer"]), r["endpoint"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.27, top=0.80, wspace=0.20)
    hues = [FV.HUES[name] for name in ("vteal", "vorange", "vviolet", "vblue")]
    handles = []
    values = [float(r[b]) for r in rows for b in BINS]
    for column, (ax, endpoint) in enumerate(zip(axes, ("first", "last"), strict=True)):
        for index, b in enumerate(BINS):
            color, ls, marker = hues[index % 4], ("-", "--", ":")[index // 4], ("o", "s", "^")[index // 4]
            ax.plot(
                outers,
                [float(lookup[o, endpoint][b]) for o in outers],
                color=color,
                ls=ls,
                marker=marker,
                markersize=3,
                markevery=max(1, len(outers) // 12),
                lw=1.4,
            )
            if column == 0:
                handles.append(Line2D([], [], color=color, ls=ls, marker=marker, label=LABELS[index]))
        ax.set_yscale("log")
        ax.set_ylim(min(values) * 0.8, max(values) * 1.25)
        ax.set_xlabel("Outer")
        ax.set_title("First update observation" if endpoint == "first" else "Last observed update observation", pad=12)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("MDPO position objective (log scale)")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.085), ncol=5, frameon=False)
    fig.suptitle(f"{experiment} | Teacher: {TEACHERS[experiment]}\nOne curve per relative-position decile", y=0.97)
    save(
        fig,
        OUT,
        f"{experiment.lower()}-lines",
        ("Each outer uses its own rollout batch; no smoothing; first and last panels share a log scale."),
    )


def profile_figures(rows):
    for experiment in dict.fromkeys(r["experiment"] for r in rows):
        pair = [
            next(r for r in rows if r["experiment"] == experiment and r["which"] == which)
            for which in ("initial", "latest")
        ]
        fig, ax = plt.subplots(figsize=(12, 6.5))
        fig.subplots_adjust(left=0.11, right=0.96, bottom=0.28, top=0.79)
        for row, color, marker, ls in zip(pair, ("vteal", "vorange"), ("o", "s"), ("-", "--"), strict=True):
            ax.plot(
                range(10),
                [float(row[b]) for b in BINS],
                color=FV.HUES[color],
                marker=marker,
                ls=ls,
                lw=2,
                label=f"{row['which'].capitalize()} outer {row['outer']} | first update {row['update']}",
            )
        ax.set_yscale("log")
        ax.set_xticks(range(10), LABELS, rotation=25, ha="right")
        ax.set_xlabel("Relative response position (deciles)")
        ax.set_ylabel(f"{pair[0]['metric']} (token mean; log scale)")
        ax.grid(alpha=0.2)
        fig.legend(
            *ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.065), ncol=1, frameon=False
        )
        fig.suptitle(
            f"{experiment} | Teacher: {TEACHERS[experiment]}\nInitial vs latest outer: FIRST update in both", y=0.97
        )
        save(
            fig,
            PROFILES,
            f"{experiment.lower()}-first-profiles",
            (
                "Both observations precede the first optimizer"
                " operation on their own rollout batch; no end"
                "point averaging."
            ),
        )


def profile_summary(profiles):
    result = []
    for row in profiles:
        values = np.array([float(row[b]) for b in BINS])
        early, late = float(values[:3].mean()), float(values[-3:].mean())
        result.append(
            {
                **{k: row[k] for k in ("experiment", "metric", "which", "outer", "update")},
                "p00_10": float(values[0]),
                "p90_100": float(values[-1]),
                "early_0_30": early,
                "late_70_100": late,
                "early_late_ratio": early / late,
                "peak_position": LABELS[int(values.argmax())],
            }
        )
    write_csv(OUT / "profile-summary.csv", result)
    return result


def table(rows):
    lines = [
        "| 实验 | 观测 | outer / update | 0–10% | 90–100% | 前30% / 后30% | 最大位置桶 |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        which = "初始 first" if r["which"] == "initial" else "末 outer first"
        lines.append(
            f"| {r['experiment']} | {which} | {r['outer']} / {r['update']} | "
            f"{r['p00_10']:.6g} | {r['p90_100']:.6g} | {r['early_late_ratio']:.3f} | {r['peak_position']} |"
        )
    return "\n".join(lines)


def documents(endpoints, profiles, summary, manifest):
    timestamp = manifest["captured_at"]
    mdpo = [
        "# MDPO position objective：E55–E57、E81–E83",
        "",
        f"数据快照：{timestamp}。每组独立绘图，展示 first / last update 随 outer 的位置热力图、"
        "首尾五个 outer 放大，以及每个位置桶一条训练曲线。初始与末 outer 的 first-update 横向位置曲线见"
        "[位置形状汇总](../teacher-transfer/cases/first-update-position-profiles.md)。",
        "",
        "## 指标与实际 setting",
        "",
        (
            "位置日志为 `distill/mdpo/objective_position/p00_10"
            "/{mean,sum,token_count}` 至 `p90_100`。每个桶对有效 t"
            "oken 取均值，十个桶按各自回答的有效长度划分。诊断始终按 token 平均；E83 的"
            "实际训练 loss 使用回答等权聚合，因此不能把这个诊断当成 E83 的 actor lo"
            "ss。"
        ),
        "",
        (
            "令当前 student 为 $p$、本轮冻结 anchor 为 $q$、teacher 为"
            " $T$，未归一化 target 为 $u$，$Z=\\sum_a u(a)$，归一化 ta"
            "rget 为 $m=u/Z$。实际记录的是"
        ),
        "",
        "$$\nJ_{\\mathrm{pos}}=\\frac{D_{\\mathrm{KL}}(p\\Vert m)-\\log Z}{\\eta}.\n$$",
        "",
        (
            "E57、E81、E82、E83 使用 κ=1、η=0.1，此时 $m=0.9q+0.1T$"
            "，$Z=1$，所以图中数值就是 target RKL 的 10 倍；乘以 0.1 即可得到"
            " target RKL，位置形状不变。E55（κ=0）与 E56（κ=0.5）的 $\\lo"
            "g Z$ 随前缀变化，图中同时包含 target RKL 与归一化项，现有 positio"
            "n 日志不能把两者分离。六组均未记录独立的 teacher position RKL。"
        ),
        "",
        (
            "[冻结运行的 objective 实现](../../../../../runs/topd"
            "-large-batch-20261105/source/verl/trainer/dis"
            "tillation/objectives.py)与[输出缩放、分桶实现](../../.."
            "/../../runs/topd-large-batch-20261105/source/"
            "verl/trainer/distillation/output.py)说明上述关系。每次"
            "有效更新均核对全局 `objective=(target_kl-target_log_z)"
            "/0.1`，并按桶内 token 数加权复算 position 均值；误差见[来源核验]("
            "figures/mdpo-position-20261107/provenance.jso"
            "n)。"
        ),
        "",
        "| 实验 | κ / η | batch / reuse | 聚合 | 日志覆盖 outer / update |",
        "| --- | --- | --- | --- | --- |",
    ]
    for e, audit in manifest["mdpo_audit"].items():
        kappa = {"E55": "0", "E56": "0.5"}.get(e, "1")
        mdpo.append(
            f"| {e} | {kappa} / 0.1 | {64 if e in ('E55', 'E56', 'E57') else 256} / 16 | "
            f"{'seq-mean-token-mean' if e == 'E83' else 'token-mean'} | "
            f"{audit['last_observed_outer']} / {audit['updates']} |"
        )
    mdpo.extend(
        [
            "",
            (
                "模型均为 Qwen3-1.7B-Base ← Qwen3-30B-A3B-Instruct"
                "-2507，LR=1e-6，future=none。其余历史配置、执行精度与停止协议差异分"
                "别见[E55–E57 主报告](weighted-targets.md)和[当前实验](w"
                "eighted-targets-active.md)。E82 应用已有 SHA256 排除"
                "清单，隔离重复实例写入；仅取有效连续 update。E84–E90 在该快照未找到正式 p"
                "osition 指标，不补造曲线。"
            ),
            "",
            (
                "first / last 是同一 outer 中第 1 / 第 16 次 optimize"
                "r 操作前的观测，两点间为 15 次更新。若最后一个 outer 尚未写满 16 次，la"
                "st 只表示最后已记录的观测，图表数据保留实际 reuse；覆盖表不表示运行已经结束。每张"
                "热力图全部面板共用一个线性色阶，early / late 放大窗口各自重新定标；训练线图 "
                "first / last 共用对数纵轴。跨 outer 的 batch、前缀和 targe"
                "t 都会变化。"
            ),
            "",
        ]
    )
    for e in RUNS:
        rr = [r for r in summary if r["experiment"] == e]
        mdpo.extend(
            [
                f"## {e}",
                "",
                table(rr),
                "",
                f"![{e} 全程热力图](figures/mdpo-position-20261107/{e.lower()}-heatmap-all.png)",
                "",
                f"![{e} 前五个 outer 放大](figures/mdpo-position-20261107/{e.lower()}-heatmap-early.png)",
                "",
                f"![{e} 后五个 outer 放大](figures/mdpo-position-20261107/{e.lower()}-heatmap-late.png)",
                "",
                f"![{e} 每个 position 一条训练曲线](figures/mdpo-position-20261107/{e.lower()}-lines.png)",
                "",
            ]
        )
    mdpo.extend(
        [
            "## 数据与复画",
            "",
            (
                "[逐 outer 首末观测](figures/mdpo-position-20261107"
                "/endpoints.csv)、[各实验初始和末 outer first-update 原"
                "值](figures/mdpo-position-20261107/profiles.cs"
                "v)、[形状数值表](figures/mdpo-position-20261107/pro"
                "file-summary.csv)与[来源哈希和核验](figures/mdpo-posi"
                "tion-20261107/provenance.json)使用同一冻结快照。图提供 PN"
                "G、PDF、SVG 和 `.audit.txt`。"
            ),
            "",
            (
                "[生成器](../../../research/studies/weighted_targ"
                "ets/plot_position_profiles.py)默认读取冻结 CSV，同时复画"
                "本页和位置形状汇总；`--refresh` 才重新读取运行日志。"
            ),
            "",
            "```bash",
            (
                "uv run --no-project --with matplotlib --with "
                "numpy python -m examples.mirroropd.research.s"
                "tudies.weighted_targets.plot_position_profile"
                "s"
            ),
            "```",
            "",
        ]
    )
    (LEARNING / "mdpo-position-kl.md").write_text("\n".join(mdpo))
    combined = [
        "# Position KL：初始与末 outer 的 first-update 位置曲线",
        "",
        f"数据快照：{timestamp}。每张图只包含一个实验的两条曲线：第一个 outer 的 first update，"
        "以及最后已记录 outer 的 first update。所有值都在各自首次 optimizer 操作前测量，"
        "末 outer 的曲线没有使用该批第 16 次观测。",
        "",
        "## 比较口径",
        "",
        (
            "横轴是相对回答位置的十等分桶，不是前十个 token；纵轴使用对数尺度，同时保留高低顺序和"
            "倍数关系。每个桶是有效 token 均值。表中“前30% / 后30%”为前三个桶的算术均"
            "值除以后三个桶的算术均值：大于 1 表示前段较高，小于 1 表示后段较高。这个形状摘要不等"
            "于完整回答的 token 加权 KL。最大位置桶用于识别中段峰值，不假定曲线单调。"
        ),
        "",
        (
            "各 outer 使用自己的新 rollout，末 outer first update 已"
            "包含此前所有 outer 的训练影响，尚未拟合当前 batch。两端的题目、前缀和长度分布"
            "不同；这些曲线描述当时访问状态上的形状，不是固定 state 的前后测量。不同 teach"
            "er 与 MDPO 的 target 不共用一个比较标尺，因此按实验分图、按指标分表。"
        ),
        "",
        (
            "覆盖此前 MiMo、R1-1.5B、Qwen 的九组正式实验、六组 MDPO，以及两组终止"
            "动作诊断。E78–E80 不纳入。固定 state 附录只有三个位置带，且无对应 oute"
            "r/first-update 结构，保留在[清单](../appendix/positio"
            "n-kl-inventory.md)，不混入十等分图。"
        ),
        "",
    ]
    latest = {r["experiment"]: r for r in summary if r["which"] == "latest"}
    ratios = [latest[e]["early_late_ratio"] for e in RUNS]
    peak_count = sum(latest[e]["peak_position"] == "0-10%" for e in RUNS)
    combined.extend(
        [
            "## 位置形状概览",
            "",
            f"MDPO 六组中有 {peak_count} 组在末 outer 的最大桶为 0–10%；末 outer 的前30% / 后30%"
            f"为 {min(ratios):.2f}–{max(ratios):.2f}。"
            "MiMo E71/E73 的末 outer 比值为 1.54/1.80，前段残差更突出；E72 为 1.06，形状较平。",
            "",
            "Polaris E48 的末 outer 比值为 0.81，最大桶在 70–80%；E49 为 1.03，前后接近。"
            "Qwen E39/E41 的末 outer 最高桶在 90–100%，但前三桶均值仍略高于后三桶均值，"
            "对应末端上翘而非全程递增。E35 的末 outer 比值为 0.61，最大桶在 50–60%。",
            "",
        ]
    )
    for heading, experiments in (
        ("MiMo：原始 teacher RKL", ["E71", "E72", "E73"]),
        ("R1-1.5B：原始 teacher RKL", ["E48", "E49", "E35"]),
        ("Qwen：原始 teacher RKL", ["E39", "E41", "E43"]),
        ("MDPO：target 相关 objective", list(RUNS)),
        ("终止动作对照：原始 teacher RKL", ["termination-baseline", "termination-aligned"]),
    ):
        combined.extend([f"## {heading}", ""])
        if heading.startswith("MDPO"):
            combined.extend(
                [
                    (
                        "κ=1 的 E57、E81–E83 为 target RKL 的 10 倍；E55/E56"
                        " 还包含归一化项。完整定义与训练热力图见[MDPO 专题](../../learning-"
                        "signal/mdpo-position-kl.md)。"
                    ),
                    "",
                ]
            )
        if heading.startswith("R1"):
            combined.extend(["E49 优化 conditional FKL，此处绘制日志中独立记录的原始 teacher RKL。", ""])
        combined.extend([table([r for e in experiments for r in summary if r["experiment"] == e]), ""])
        for e in experiments:
            combined.extend(
                [
                    f"### {e}：{TEACHERS[e]}",
                    "",
                    f"![{e} first-update 位置曲线]"
                    f"(../figures/first-update-profiles-20261107/{e.lower()}-first-profiles.png)",
                    "",
                ]
            )
    combined.extend(
        [
            "## 数据与复画",
            "",
            (
                "[完整十桶原值](../../learning-signal/figures/mdpo-p"
                "osition-20261107/profiles.csv)、[摘要表](../../le"
                "arning-signal/figures/mdpo-position-20261107/"
                "profile-summary.csv)和[来源核验](../../learning-si"
                "gnal/figures/mdpo-position-20261107/provenanc"
                "e.json)使用同一快照；复画命令见[MDPO 专题](../../learning-s"
                "ignal/mdpo-position-kl.md#数据与复画)。"
            ),
            "",
        ]
    )
    (TRANSFER / "cases/first-update-position-profiles.md").write_text("\n".join(combined))


def main():
    global FV
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Read current logs and freeze a new audited snapshot.")
    args = parser.parse_args()
    if args.refresh:
        freeze()
    FV = style()
    FV.apply_rc("paper", fontsize=11)
    endpoints, profiles = read_csv(OUT / "endpoints.csv"), read_csv(OUT / "profiles.csv")
    manifest = json.loads((OUT / "provenance.json").read_text())
    summary = profile_summary(profiles)
    for experiment in RUNS:
        rows = [r for r in endpoints if r["experiment"] == experiment]
        heatmaps(experiment, rows)
        training_lines(experiment, rows)
    profile_figures(profiles)
    documents(endpoints, profiles, summary, manifest)
    (OUT / "renderer.json").write_text(
        json.dumps({"source": stamp(Path(__file__)), "figures": 24 + len(profiles) // 2}, indent=2) + "\n"
    )
    print(f"Rendered 24 MDPO training figures + {len(profiles) // 2} independent first-update profile figures.")


if __name__ == "__main__":
    main()
