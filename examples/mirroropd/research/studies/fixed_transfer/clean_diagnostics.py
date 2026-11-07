# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only first-pass diagnostics; never train or stitch resumed trajectories.

Semantic ledger: each row measures one run/checkpoint/panel/position band.
hp/hq are student/teacher Shannon entropies in nats. overlap16 is |P16∩Q16|/16;
common_p16/common_q16 are probability mass on that intersection. Missing fields
remain absent. Fixed-H risk and observed-token averages are separate columns.
Geometry ledger: x is optimizer updates, categorical positions have equal cell
width (not equal token span), y is an explicitly labelled measured quantity.
Colors identify early/middle/late or student/teacher consistently within a plot.
Training loss segments stop at batch boundaries. Only the first A→T pass is read.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from examples.mirroropd.research.core.artifacts import file_hash
from examples.mirroropd.research.core.io import dump, read, write_csv, write_text
from examples.mirroropd.research.core.paths import REPO
from examples.mirroropd.research.studies.fixed_transfer.figure_data import GROUPS, load_run, registry
from examples.mirroropd.research.studies.fixed_transfer.figure_render import Renderer

BASE = REPO / "examples/mirroropd/docs/experiments/teacher-transfer"
DEFAULT = BASE / "transfer-mechanism/appendix/data/clean-diagnostics"
BANDS = ["early", "middle", "late"]
WIDTHS = dict(all=1024, early=256, middle=256, late=512)
METRICS = ["rkl", "hp", "hq", "js", "tv", "overlap16", "common_p16", "common_q16"]
PANEL_LABELS = {
    "Training scored": "Fixed training pool",
    "All held-out": "All held-out",
    "Math": "Math bank",
    "Code": "Code bank",
    "IF": "IF bank",
}


def aggregate(data):
    """Recompute every saved snapshot; assert first-pass and fixed-state coverage."""
    spec, reuse = data["spec"], data["reuse"]
    expected = list(range(0, spec["budget"] + 1, reuse))
    if list(data["snapshots"]) != expected:
        raise ValueError(f"Incomplete checkpoint grid: {spec['run_id']}")
    if sorted(r["step"] for r in data["training"]) != list(range(1, spec["budget"] + 1)):
        raise ValueError(f"Incomplete first-pass training log: {spec['run_id']}")
    if Path(spec["initial_source"]).parent != Path(spec["final_source"]).parent:
        raise ValueError("Mixed execution instances")
    data["panels"].pop("Old A-D", None)
    data["panels"] = {p: data["panels"][p] for p in PANEL_LABELS if p in data["panels"]}
    out, initial = [], {}
    entropy_drift = 0.0
    for step, rows in data["snapshots"].items():
        for panel, groups in data["panels"].items():
            selected = [r for r in rows if r["group"] in groups]
            for band in ["all", *BANDS]:
                stats = [r["stats"].get(band) for r in selected]
                if all(s is None for s in stats):
                    continue
                if any(s is None for s in stats):
                    raise ValueError(f"Partial position-band recording: {panel}/{band}")
                tokens = sum(s["tokens"] for s in stats)
                active = [s for s in stats if s["tokens"] > 0]
                for metric in METRICS:
                    present = [metric in s for s in active]
                    if any(present) and not all(present):
                        raise ValueError(f"Partial metric coverage: {panel}/{band}/{metric}")
                    if not all(present) or not tokens:
                        continue
                    total = sum(s[metric] for s in active)
                    fixed = total / (len(selected) * data["horizon"])
                    observed = total / tokens
                    identity = (panel, band, metric)
                    if step == 0:
                        initial[identity] = (fixed, observed, tokens)
                    f0, o0, n0 = initial[identity]
                    if tokens != n0:
                        raise ValueError("Frozen-state token coverage changed")
                    if metric == "hq":
                        entropy_drift = max(entropy_drift, abs(observed - o0))
                    out.append(
                        dict(
                            run_id=spec["run_id"],
                            student=spec["student"],
                            teacher=spec["teacher"],
                            step=step,
                            panel=panel,
                            band=band,
                            metric=metric,
                            prompts=len(selected),
                            tokens=tokens,
                            coverage=tokens / (len(selected) * WIDTHS[band]),
                            sum=total,
                            fixed_h=fixed,
                            observed=observed,
                            initial_fixed_h=f0,
                            initial_observed=o0,
                            decrease_fixed_h=f0 - fixed,
                            decrease_observed=o0 - observed,
                            remaining=observed / o0 if o0 else None,
                        )
                    )
    if entropy_drift > 1e-6:
        raise ValueError(f"Frozen teacher entropy drift: {entropy_drift}")
    return out, entropy_drift


class CleanRenderer(Renderer):
    def __init__(self, data, output, rows):
        super().__init__(data, output)
        self.rows = rows
        self.panels = list(data["panels"])
        self.band_colors = dict(zip(BANDS, self.colors[:3], strict=True))

    def values(self, panel, metric, band="all", column="observed"):
        return [
            (r["step"], r[column])
            for r in self.rows
            if r["panel"] == panel and r["metric"] == metric and r["band"] == band and r[column] is not None
        ]

    def axes(self, title):
        fig, axes = self.figure(title, len(self.panels), 3)
        for ax, panel in zip(axes, self.panels, strict=True):
            ax.set(title=PANEL_LABELS[panel], xlabel="Optimizer updates")
        return fig, axes

    def finish(self, fig, key, title, caption):
        self.save(
            fig,
            key,
            title,
            caption + "\nFirst pass only; no repeated-batch round. Same size does not establish direct ancestry.",
        )

    def legend(self, fig, ax):
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1 - 0.85 / fig.get_figheight()),
            ncol=min(4, len(labels)),
        )

    def risk(self):
        for key, column, ylabel, log in [
            ("rkl-absolute", "fixed_h", "RKL / (prompts x 1024)", False),
            ("rkl-log", "fixed_h", "RKL / (prompts x 1024), log", True),
            ("rkl-decrease", "decrease_fixed_h", "Initial RKL - current RKL", False),
            ("rkl-remaining", "remaining", "RKL / initial RKL", False),
        ]:
            fig, axes = self.axes(key.replace("-", " "))
            for ax, panel in zip(axes, self.panels, strict=True):
                points = self.values(panel, "rkl", column=column)
                ax.plot(*zip(*points, strict=True), color=self.panel_colors[panel], marker=".", markersize=3)
                ax.set_ylabel(ylabel)
                if log:
                    if any(y <= 0 for _, y in points):
                        raise ValueError("Nonpositive RKL on log axis")
                    ax.set_yscale("log")
                elif column != "decrease_fixed_h":
                    ax.set_ylim(bottom=0)
                else:
                    ax.axhline(0, color=self.ink, linewidth=0.5)
            self.finish(
                fig,
                key,
                key,
                "Fixed state identities and denominators; training pool includes future training batches.",
            )

    def position_curves(self, metric, key, gap=False):
        if not self.values(self.panels[0], metric, "early"):
            return
        title = "Position entropy gap: student - teacher" if gap else f"Position {metric.upper()} by updates"
        fig, axes = self.axes(title)
        for ax, panel in zip(axes, self.panels, strict=True):
            for band in BANDS:
                points = self.values(panel, metric, band)
                if gap:
                    other = dict(self.values(panel, "hq", band))
                    points = [(x, y - other[x]) for x, y in points]
                ax.plot(*zip(*points, strict=True), color=self.band_colors[band], label=band)
            ax.set_ylabel("Student - teacher entropy (nats)" if gap else f"{metric.upper()} / observed token")
            if not gap:
                ax.set_ylim(bottom=0)
            else:
                ax.axhline(0, color=self.ink, linewidth=0.5)
        self.legend(fig, axes[0])
        self.finish(
            fig,
            key,
            title,
            "early: 0-255; middle: 256-511; late: 512-1023. Denominator: observed tokens within each band.",
        )

    def heatmap(self, relative=False):
        if not self.values(self.panels[0], "rkl", "early"):
            return
        title = "Position residual / initial" if relative else "Absolute position RKL"
        fig, axes = self.figure(title, len(self.panels), 2)
        for ax, panel in zip(axes, self.panels, strict=True):
            ax.set(title=PANEL_LABELS[panel], xlabel="Optimizer updates")
        column = "remaining" if relative else "observed"
        arrays = [np.array([[v for _, v in self.values(p, "rkl", b, column)] for b in BANDS]) for p in self.panels]
        vmax = max(float(a.max()) for a in arrays)
        for ax, panel, values in zip(axes, self.panels, arrays, strict=True):
            steps = [x for x, _ in self.values(panel, "rkl")]
            edges = np.array([*steps, steps[-1] + self.d["reuse"]]) - self.d["reuse"] / 2
            im = ax.pcolormesh(
                edges, np.arange(4) - 0.5, values, cmap=self.fv.ramp_cmap("seq"), vmin=0, vmax=vmax, shading="flat"
            )
            ax.set(yticks=range(3), yticklabels=BANDS, ylabel="Position band")
            ax.invert_yaxis()
            bar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.04)
            bar.set_label("RKL / initial" if relative else "RKL / token")
        self.finish(
            fig,
            "position-relative-heatmap" if relative else "position-absolute-heatmap",
            title,
            "Common color scale across banks; equally spaced categories represent unequal token spans. "
            "No interpolation.",
        )

    def entropy(self):
        fig, axes = self.axes("Student and frozen-teacher entropy")
        for ax, panel in zip(axes, self.panels, strict=True):
            for metric, label, color, style in [
                ("hp", "Student", self.colors[0], "-"),
                ("hq", "Teacher", self.colors[1], "--"),
            ]:
                ax.plot(*zip(*self.values(panel, metric), strict=True), label=label, color=color, linestyle=style)
            ax.set(ylabel="Entropy / observed token (nats)", ylim=(0, None))
        self.legend(fig, axes[0])
        self.finish(
            fig,
            "entropy",
            "Student and teacher entropy",
            "Frozen S0 states; teacher entropy is checked for checkpoint invariance. "
            "Entropy matching alone is not distribution matching.",
        )
        self.position_curves("hp", "position-student-entropy")
        self.position_curves("hp", "position-entropy-gap", gap=True)

    def position_profiles(self, metric, name):
        if not self.values(self.panels[0], metric, "early"):
            return
        fig, axes = self.axes(f"{name}: position profiles at saved checkpoints")
        steps = sorted({0, self.d["reuse"], min(64, self.d["spec"]["budget"]), self.d["spec"]["budget"]})
        for ax, panel in zip(axes, self.panels, strict=True):
            for step, color in zip(steps, self.colors, strict=False):
                values = [dict(self.values(panel, metric, band))[step] for band in BANDS]
                ax.plot(range(3), values, color=color, marker="o", markersize=3, label=f"update {step}")
            ax.set(
                xticks=range(3),
                xticklabels=BANDS,
                xlabel="Absolute position band",
                ylabel=f"{name} / observed token",
                ylim=(0, None),
            )
        self.legend(fig, axes[0])
        self.finish(
            fig,
            f"position-profile-{metric}",
            f"{name} position profiles",
            "Profiles at saved updates; early=0-255, middle=256-511, late=512-1023. "
            "Equal categorical spacing does not imply equal token spans.",
        )

    def training(self):
        rows = self.d["training"]
        fig, axes = self.figure("First-pass training loss and within-batch progress", 3, 3)
        for group in GROUPS:
            chosen = [r for r in rows if r["group"] == group]
            if not chosen:
                continue
            for ax, metric in zip(axes[:2], ["loss", "grad_norm"], strict=True):
                ax.plot([r["step"] for r in chosen], [r[metric] for r in chosen], color=self.colors[0], linewidth=0.9)
            axes[2].plot(
                [r["step"] for r in chosen],
                [r["loss"] - chosen[0]["loss"] for r in chosen],
                color=self.colors[2],
                linewidth=0.9,
            )
        for ax, ylabel in zip(
            axes,
            ["Pre-update current-batch loss", "Gradient norm (before clipping)", "Loss - first loss of current batch"],
            strict=True,
        ):
            ax.set(xlabel="Logged training update (pre-update)", ylabel=ylabel)
        self.finish(
            fig,
            "training-loss",
            "Training loss",
            "Batch16 x reuse16; segments stop at switches. "
            "A logged loss at step t scores theta_(t-1), not the fixed theta_t snapshot.",
        )

    def distances(self):
        fig, axes = self.figure("Distribution distances on fixed panels", 2, 2)
        for ax, metric in zip(axes, ["js", "tv"], strict=True):
            for panel in self.panels:
                if panel == "All held-out":
                    continue
                ax.plot(
                    *zip(*self.values(panel, metric), strict=True),
                    label=PANEL_LABELS[panel],
                    color=self.panel_colors[panel],
                )
            ax.set(xlabel="Optimizer updates", ylabel=f"{metric.upper()} / observed token", ylim=(0, None))
        self.legend(fig, axes[0])
        self.finish(
            fig,
            "js-tv",
            "JS and TV",
            "Observed-token averages on the same fixed states; permanent banks are separate from training.",
        )

    def overlap(self):
        if not self.values(self.panels[0], "overlap16"):
            return
        self.position_curves("overlap16", "position-top16-overlap")
        fig, axes = self.axes("Top-16 shared probability mass")
        for ax, panel in zip(axes, self.panels, strict=True):
            for metric, label, color in [
                ("common_p16", "Student mass", self.colors[0]),
                ("common_q16", "Teacher mass", self.colors[1]),
                ("overlap16", "Intersection / 16", self.colors[2]),
            ]:
                ax.plot(*zip(*self.values(panel, metric), strict=True), color=color, label=label)
            ax.set(ylabel="Observed-token mean", ylim=(0, 1))
        self.legend(fig, axes[0])
        self.finish(
            fig,
            "top16-mass",
            "Top-16 overlap and intersection mass",
            "P16 and Q16 rank logits; shared mass sums probabilities on P16 intersect Q16. "
            "Count overlap is not probability mass.",
        )

    def question_distribution(self):
        fig, axes = self.figure("Paired held-out prompt residuals", 2, 2)
        for panel in ["Math", "Code", "IF"]:
            groups = self.d["panels"][panel]
            chosen = [r for r in self.d["questions"] if r["group"] in groups]
            color = self.panel_colors[panel]
            axes[0].scatter(
                [r["initial"] for r in chosen], [r["final"] for r in chosen], s=12, color=color, alpha=0.65, label=panel
            )
            values = np.sort([r["decrease"] for r in chosen])
            axes[1].step(values, np.arange(1, len(values) + 1) / len(values), where="post", color=color, label=panel)
        bound = max(max(r["initial"], r["final"]) for r in self.d["questions"] if r["group"] not in GROUPS)
        axes[0].plot([0, bound], [0, bound], color=self.ink, linestyle="--", linewidth=0.7)
        axes[0].set(
            xlabel="Initial prompt RKL / H1024",
            ylabel="Final prompt RKL / H1024",
            xlim=(0, bound * 1.03),
            ylim=(0, bound * 1.03),
        )
        axes[1].set(xlabel="Initial - final prompt RKL / H1024", ylabel="Empirical cumulative fraction", ylim=(0, 1.03))
        axes[1].axvline(0, color=self.ink, linewidth=0.5)
        self.legend(fig, axes[0])
        self.finish(
            fig,
            "prompt-distribution",
            "Prompt distribution",
            "Each point is one held-out prompt. ECDF describes prompts, not variation across training seeds.",
        )

    def render_clean(self):
        self.risk()
        self.position_curves("rkl", "position-rkl")
        self.position_profiles("rkl", "RKL")
        self.heatmap()
        self.heatmap(relative=True)
        self.entropy()
        self.position_profiles("hp", "Student entropy")
        self.training()
        self.distances()
        self.question_distribution()
        self.overlap()
        return self.entries


def build(spec, output, force=False):
    data = load_run(spec)
    rows, drift = aggregate(data)
    target = output / spec["run_id"]
    code = {
        name: file_hash(Path(__file__).with_name(name))
        for name in ["clean_diagnostics.py", "figure_data.py", "figure_render.py", "clean_comparisons.py"]
    }
    receipt = target / "manifest.json"
    if receipt.exists() and not force:
        cached = read(receipt)
        if (
            cached["code"] == code
            and cached["sources"] == data["sources"]
            and all((target / p).exists() and file_hash(target / p) == h for p, h in cached["artifacts"].items())
        ):
            return cached
    target.mkdir(parents=True, exist_ok=True)
    receipt.unlink(missing_ok=True)
    write_csv(target / "metrics.csv", rows)
    write_csv(target / "training.csv", data["training"])
    write_csv(target / "questions.csv", data["questions"])
    dump(target / "sources.json", data["sources"])
    renderer = CleanRenderer(data, target, rows)
    figures = renderer.render_clean()
    topk = any(r["metric"] == "overlap16" for r in rows)
    result = dict(
        spec=spec,
        code=code,
        sources=data["sources"],
        figures=figures,
        teacher_label=renderer.teacher[0],
        student_label=renderer.student[0],
        same_size=renderer.teacher[1] == renderer.student[1],
        top16_available=topk,
        teacher_entropy_max_drift=drift,
        panels=data["panels"],
        snapshots=list(data["snapshots"]),
        training_steps=len(data["training"]),
    )
    lines = [
        f"# {renderer.title}",
        "",
        f"单遍 0–{spec['budget']} 更新；student 输入：{renderer.student[2]}。"
        f"teacher：{renderer.teacher[2]}。支持口径：`{spec['support']}`。",
        "",
        f"同尺寸：{result['same_size']}。top-16 overlap：{'已记录' if topk else '未记录，不补零'}。"
        "同尺寸不等于同初始化谱系，跨尺寸条件仅作上下文。",
        "",
        "固定训练池包含尚未访问的训练题；Math／Code／IF 是永久留出 bank。没有旧 A–D 专用面板。",
        "",
        "early=0–255，middle=256–511，late=512–1023。位置指标除以实际 token；"
        "全局 fixed-H RKL 除以题数×1024。覆盖量与两种分母均保留在 CSV。",
        "",
        "[全部指标 CSV](metrics.csv) · [训练 CSV](training.csv) · [逐题 CSV](questions.csv)"
        " · [来源](sources.json) · [返回图集](../README.md)",
        "",
    ]
    for f in figures:
        lines += [
            f"## {f['title']}",
            "",
            f"![{f['title']}]({f['key']}.png)",
            "",
            f"[PDF]({f['key']}.pdf) · [审计]({f['key']}.audit.json)",
            "",
        ]
    write_text(target / "README.md", "\n".join(lines))
    result["artifacts"] = {p.name: file_hash(p) for p in target.iterdir() if p.is_file() and p.name != "manifest.json"}
    dump(receipt, result)
    return result


def index(results, output, excluded):
    lines = [
        "# 单遍固定状态诊断图集",
        "",
        f"{len(results)} 个输入条件，{sum(len(r['figures']) for r in results)} 张逐 run 图；"
        "每张都有 PNG、PDF、CSV 与 fv.audit 审计。",
        "",
        "来源为固定状态主清单；仅保留原执行实例 0–320 内完整的 A→T 单遍前缀。"
        "每批 reuse16 是原协议；不使用后续跨轮重访，不混接执行实例，不生成旧 A–D 面板。Polaris 条件排除。",
        "",
        "同尺寸条件先列；跨尺寸仅作上下文，不能用于解释同源 RL／SFT 的因果差异。"
        "不同输入模式分组；64／128／320 终点必须按共同预算比较。固定训练池与永久留出 bank 分开。",
        "",
        "[跨 teacher 比较](comparisons/README.md) · [清单 CSV](runs.csv) · [排除清单](excluded.json)",
        "",
    ]
    table = []
    for student in sorted({r["spec"]["student"] for r in results}):
        lines += [
            f"## {student}",
            "",
            "| Teacher | 更新 | 尺寸 | top-16 | 完整图谱 |",
            "| --- | ---: | --- | --- | --- |",
        ]
        for r in sorted(
            [r for r in results if r["spec"]["student"] == student],
            key=lambda x: (not x["same_size"], x["spec"]["teacher"]),
        ):
            s = r["spec"]
            size = "同尺寸（需另核谱系）" if r["same_size"] else "跨尺寸上下文"
            lines.append(
                f"| {r['teacher_label']} | {s['budget']} | {size} | "
                f"{'有' if r['top16_available'] else '未记录'} | [图谱]({s['run_id']}/README.md) |"
            )
            table.append(
                dict(
                    run_id=s["run_id"],
                    student=student,
                    teacher=s["teacher"],
                    budget=s["budget"],
                    same_size=r["same_size"],
                    top16=r["top16_available"],
                    figures=len(r["figures"]),
                    entropy_drift=r["teacher_entropy_max_drift"],
                    source=s["final_source"],
                )
            )
        lines.append("")
    write_csv(output / "runs.csv", table)
    dump(output / "excluded.json", excluded)
    write_text(output / "README.md", "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--run", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    specs = registry(BASE / "transfer-mechanism/data/overview-20261029")
    selected, excluded = [], []
    for spec in specs:
        if spec["scope"] != "main" or "polaris" in spec["teacher"]:
            excluded.append(
                dict(
                    run_id=spec["run_id"],
                    reason="supplement" if spec["scope"] != "main" else "Polaris excluded from current comparison",
                )
            )
        elif not args.run or args.run in [spec["run_id"], spec["teacher"], spec["arm"]]:
            selected.append(spec)
    if not selected:
        raise ValueError("No selected runs")
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        tasks = {pool.submit(build, s, args.output, args.force): s["run_id"] for s in selected}
        for task in as_completed(tasks):
            result = task.result()
            results.append(result)
            print(
                f"{len(results)}/{len(selected)}: {result['spec']['run_id']} ({len(result['figures'])} figures)",
                flush=True,
            )
    from examples.mirroropd.research.studies.fixed_transfer.clean_comparisons import comparisons

    comparisons(results, args.output)
    index(results, args.output, excluded)
    print(f"Complete: {args.output / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
