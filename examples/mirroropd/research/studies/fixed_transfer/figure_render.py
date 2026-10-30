# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Run figure ledger: lines are measured updates; cells are batch/block risks.

Colors identify panel domains consistently. Solid/dashed curves distinguish
same/different model sizes. Position risks use observed tokens, other risks H.
Missing observations are blank, never interpolated into the exported tables.
"""

import textwrap

import numpy as np

from examples.mirroropd.research.core.io import dump
from examples.mirroropd.research.core.plotting import load_style
from examples.mirroropd.research.studies.fixed_transfer.figure_data import GROUPS, mean
from examples.mirroropd.research.studies.fixed_transfer.overview import MODEL_LABELS


class Renderer:
    def __init__(self, data, output):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.plt, self.d, self.out = plt, data, output
        self.fv = load_style()
        self.fv.apply_rc("paper", fontsize=10)
        self.colors = [self.fv.HUES[k] for k in ["vblue", "vorange", "vteal", "vviolet"]]
        self.ink = self.fv.NEUTRALS["vink"]
        self.panel_colors = {
            "Training scored": self.colors[0],
            "Old A-D": self.fv.NEUTRALS["baseline"],
            "Math": self.colors[2],
            "Code": self.colors[3],
            "IF": self.colors[1],
            "All held-out": self.ink,
        }
        spec = data["spec"]
        teacher = (
            "coreward-i-llama" if spec["family"] == "Llama" and spec["teacher"] == "coreward-i" else spec["teacher"]
        )
        name, size, description = MODEL_LABELS[teacher]
        if "nonthinking" in spec["student"]:
            description = description.replace("thinking input", "non-thinking input")
        if "native_template_" in spec["support"]:
            description = "specialist native template; isolated control"
        self.teacher = (name, size, description)
        self.student = MODEL_LABELS[spec["student"]]
        self.style = "-" if self.teacher[1] == self.student[1] else "--"
        self.title = f"{self.teacher[0]} -> {self.student[0]}"
        self.entries = []

    def figure(self, title, n=1, cols=2):
        cols = min(n, cols)
        rows = (n + cols - 1) // cols
        fig, axes = self.plt.subplots(rows, cols, figsize=(14 if cols > 1 else 11, 3.5 * rows + 3.0), squeeze=False)
        fig.subplots_adjust(
            left=0.10,
            right=0.96,
            top=1 - 1.85 / fig.get_figheight(),
            bottom=1.45 / fig.get_figheight(),
            hspace=0.55,
            wspace=0.35,
        )
        fig.suptitle("\n".join(textwrap.wrap(self.title, 110)) + "\n" + title, fontsize=12, y=0.98)
        for ax in axes.flat:
            ax.spines[["top", "right"]].set_visible(False)
        for ax in list(axes.flat)[n:]:
            ax.set_visible(False)
        return fig, list(axes.flat)[:n]

    def save(self, fig, key, title, caption):
        fig.text(
            0.5,
            0.025,
            "\n".join(textwrap.fill(line, 125) for line in caption.splitlines())
            + "\n"
            + self.student[2]
            + "; "
            + self.d["spec"]["support"],
            ha="center",
            fontsize=8,
            color=self.fv.NEUTRALS["baseline"],
        )
        fig.canvas.draw()
        issues = self.fv.audit(fig, print_slot=fig.get_figwidth())
        dump(self.out / f"{key}.audit.json", [str(p) for p in issues])
        if any(self.fv.problem_severity(p) == "error" for p in issues):
            raise RuntimeError(f"{key}: {issues}")
        for ext in ["png", "pdf"]:
            fig.savefig(self.out / f"{key}.{ext}", dpi=140)
        self.plt.close(fig)
        self.entries.append(dict(key=key, title=title, warnings=len(issues)))

    def curves(self, metric="rkl", relative=False):
        panels = list(self.d["panels"])
        key = "rkl-reduction" if relative else f"{metric}-curves"
        title = "RKL reduction from initial (%)" if relative else f"Absolute {metric.upper()} on fixed panels"
        fig, axes = self.figure(title, len(panels), 3)
        for i, (panel, ax) in enumerate(zip(panels, axes, strict=True)):
            selected = [r for r in self.d["curves"] if r["panel"] == panel and r["metric"] == metric]
            chosen = [(r["step"], r["reduction"] if relative else r["value"]) for r in selected]
            chosen = [(x, y) for x, y in chosen if y is not None]
            if chosen:
                ax.plot(
                    *zip(*chosen, strict=True),
                    color=self.panel_colors[panel],
                    linestyle=self.style,
                    marker=".",
                    markersize=3,
                )
            ax.set(
                title=f"{panel} (n={16 * len(self.d['panels'][panel])})",
                xlabel="Optimizer updates",
                ylabel="Reduction (%)" if relative else metric.upper(),
            )
            ax.axhline(0, color=self.ink, linewidth=0.4)
        self.save(
            fig,
            key,
            title,
            "Denominator: prompts x H1024. Training scored includes future batches; old A-D is fully "
            "visited at step 64.\nSolid: same size; dashed: different size. All held-out pools "
            "Math/Code/IF by prompt count.",
        )

    def training(self):
        rows = self.d["training"]
        metrics = [
            (k, title)
            for k, title in [
                ("loss", "Current batch RKL (pre-update)"),
                ("grad_norm", "Gradient norm before clipping"),
                ("parameter_step_norm", "Parameter displacement norm"),
            ]
            if any(k in r for r in rows)
        ]
        if not metrics:
            return
        fig, axes = self.figure("Per-update training diagnostics", len(metrics))
        for ax, (metric, title) in zip(axes, metrics, strict=True):
            # Separate segments: batch switches are not a global loss trajectory.
            steps, values = [], []
            for group in GROUPS:
                selected = [r for r in rows if r["group"] == group and metric in r]
                if selected:
                    steps.extend([r["step"] for r in selected] + [np.nan])
                    values.extend([r[metric] for r in selected] + [np.nan])
            ax.plot(steps, values, color=self.colors[0], linestyle=self.style)
            for step in range(self.d["reuse"], self.d["spec"]["budget"], self.d["reuse"]):
                ax.axvline(step + 0.5, color=self.fv.NEUTRALS["baseline"], alpha=0.2, linewidth=0.5)
            ax.set(title=title, xlabel="Optimizer update (loss measured before update)", ylabel=metric)
        self.save(
            fig,
            "training",
            "逐步训练诊断",
            "Each disconnected segment uses a different batch. Gradient norm is pre-clipping; no smoothing.",
        )

    def matrix(self, effects=True):
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

        available = {r["group"] for r in self.d["snapshots"][0]}
        targets = {g: [g] for g in GROUPS if g in available} | {
            k: v for k, v in self.d["panels"].items() if k in ["Math", "Code", "IF", "All held-out"]
        }
        if effects:
            steps = sorted({r["step"] for r in self.d["effects"]})
            values = {(r["target"], r["step"]): r["decrease"] for r in self.d["effects"]}
            matrix = np.array([[values.get((g, s), np.nan) for s in steps] for g in targets])
            bound = max(float(np.nanmax(np.abs(matrix))), 1e-12)
            cmap = LinearSegmentedColormap.from_list(
                "effect", [self.colors[1], self.fv.NEUTRALS["paper"], self.colors[0]]
            )
            options = dict(cmap=cmap, norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound))
            title, key, label = "Effect of each training block", "batch-effects", "RKL before - after (blue improves)"
        else:
            steps = list(self.d["snapshots"])
            matrix = np.array(
                [
                    [mean(self.d["snapshots"][s], groups, self.d["horizon"]) for s in steps]
                    for groups in targets.values()
                ]
            )
            options = dict(
                cmap=LinearSegmentedColormap.from_list("risk", [self.fv.NEUTRALS["paper"], self.colors[0]]), vmin=0
            )
            title, key, label = "Absolute risk across batches and updates", "batch-risks", "Absolute RKL"
        if not steps:
            return
        fig, (ax,) = self.figure(title)
        fig.set_size_inches(12, max(9, len(targets) * 0.29 + 3))
        fig.subplots_adjust(left=0.18, right=0.88, bottom=0.20, top=0.85)
        im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", **options)
        ax.set_xticks(
            range(len(steps)),
            [GROUPS[s // self.d["reuse"] - 1] if effects else str(s) for s in steps],
            rotation=0 if effects else 60,
        )
        ax.set_yticks(range(len(targets)), list(targets))
        ax.set(
            xlabel="Batch optimized" if effects else "Optimizer updates",
            ylabel="Evaluated fixed batch / held-out panel",
        )
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.035).set_label(label)
        self.save(
            fig,
            key,
            title,
            "Rows A-T: training-pool batches; bottom rows: fixed held-out. All held-out overlaps the "
            "domain rows.\nAbsolute scale without clipping; blank cells are unavailable. One "
            "frozen-state run, no significance test.",
        )

    def forgetting(self):
        rows = self.d["forgetting"]
        if not rows:
            return
        fig, axes = self.figure("Learning and subsequent forgetting by batch", 2)
        x = np.arange(len(rows))
        for j, field in enumerate(["initial", "own", "final"]):
            axes[0].plot(
                x, [r[field] for r in rows], color=self.colors[j], marker=".", label=field, linestyle=self.style
            )
        axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.23), ncol=3, frameon=False)
        axes[0].set(ylabel="Absolute RKL", xlabel="Training batch")
        axes[1].bar(x, [r["forgetting"] for r in rows], color=self.colors[1])
        axes[1].axhline(0, color=self.ink, linewidth=0.6)
        axes[1].set(ylabel="Final - immediately after own block", xlabel="Training batch")
        for ax in axes:
            ax.set_xticks(x, [r["group"] for r in rows])
        self.save(
            fig,
            "forgetting",
            "批次拟合与遗忘",
            "Own: snapshot immediately after this batch's updates. Positive right-hand bars mean "
            "forgetting.\nOnly visited batches with both readouts are included; no best-checkpoint "
            "selection.",
        )

    def questions(self):
        panels = [k for k in ["Training scored", "Math", "Code", "IF"] if k in self.d["panels"]]
        fig, axes = self.figure("Per-question initial and final RKL", len(panels))
        for i, (panel, ax) in enumerate(zip(panels, axes, strict=True)):
            rows = [r for r in self.d["questions"] if r["group"] in self.d["panels"][panel]]
            bound = max(max(r["initial"], r["final"]) for r in rows) * 1.05
            ax.plot([0, bound], [0, bound], color=self.ink, linewidth=0.7, linestyle=":")
            ax.scatter(
                [r["initial"] for r in rows],
                [r["final"] for r in rows],
                color=self.panel_colors[panel],
                s=10,
                alpha=0.65,
            )
            ax.set(
                title=panel,
                xlabel="Initial RKL / H",
                ylabel="Final RKL / H",
                xlim=(-0.03 * bound, bound),
                ylim=(-0.03 * bound, bound),
            )
        self.save(
            fig,
            "question-scatter",
            "逐题初始与最终风险",
            "Each dot is one frozen prompt/response. Below diagonal: improvement. Panel axis ranges differ.",
        )
        fig, axes = self.figure("Distribution of per-question RKL changes", len(panels))
        for i, (panel, ax) in enumerate(zip(panels, axes, strict=True)):
            values = np.sort([r["decrease"] for r in self.d["questions"] if r["group"] in self.d["panels"][panel]])
            ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", color=self.panel_colors[panel])
            ax.axvline(0, color=self.ink, linewidth=0.6)
            ax.set(
                title=f"{panel}: {100 * (values > 0).mean():.1f}% improved",
                xlabel="Initial - final RKL / H",
                ylabel="Empirical cumulative fraction",
            )
        self.save(
            fig,
            "question-ecdf",
            "逐题变化分布",
            "Negative values worsen; positive values improve. Fractions describe prompts, not "
            "uncertainty across training seeds.",
        )

    def positions(self):
        panels = [k for k in ["Training scored", "Math", "Code", "IF"] if k in self.d["panels"]]
        if not self.d["positions"]:
            return
        fig, axes = self.figure("RKL by response position (observed-token mean)", len(panels))
        for panel, ax in zip(panels, axes, strict=True):
            for index, step in enumerate([0, self.d["spec"]["budget"]]):
                rows = [r for r in self.d["positions"] if r["panel"] == panel and r["step"] == step]
                ax.plot(
                    [r["band"] for r in rows],
                    [r["rkl"] for r in rows],
                    color=self.colors[index],
                    marker="o",
                    label=f"step {step}",
                    linestyle=self.style,
                )
            ax.set(
                title=panel,
                xlabel="Position: early 0-255 / middle 256-511 / late 512-1023",
                ylabel="RKL / observed token",
            )
        axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.24), ncol=2, frameon=False)
        self.save(
            fig,
            "positions",
            "回答位置残差",
            "Position means use observed token counts (exported in positions.csv), unlike fixed-H "
            "objective plots.\nShort responses contribute only to positions they reach; no per-token "
            "interpolation.",
        )

    def auxiliary(self):
        metrics = ["js", "tv", "hp", "hq"]
        fig, axes = self.figure("Distribution diagnostics on train and held-out", len(metrics))
        for metric, ax in zip(metrics, axes, strict=True):
            for index, panel in enumerate(["Training scored", "All held-out"]):
                rows = [r for r in self.d["curves"] if r["metric"] == metric and r["panel"] == panel]
                ax.plot(
                    [r["step"] for r in rows],
                    [r["value"] for r in rows],
                    color=self.colors[index],
                    label=panel,
                    linestyle=self.style,
                )
            ax.set(
                title={"hp": "Student entropy", "hq": "Teacher entropy"}.get(metric, metric.upper()),
                xlabel="Optimizer updates",
                ylabel=f"{metric} / budget position",
            )
        axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.24), ncol=2, frameon=False)
        self.save(
            fig,
            "auxiliary",
            "JS、TV 与熵",
            "All metrics use prompts x H1024. Frozen teacher entropy should remain constant on fixed states.",
        )

    def runtime(self):
        metrics = [
            k
            for k in ["update_seconds", "backward_seconds", "optimizer_seconds", "cache_seconds", "peak_gib"]
            if any(k in r for r in self.d["training"])
        ]
        if not metrics:
            return
        fig, axes = self.figure("Recorded runtime diagnostics", len(metrics))
        for metric, ax in zip(metrics, axes, strict=True):
            rows = [r for r in self.d["training"] if metric in r]
            values = [max(r[metric]) if isinstance(r[metric], list) else r[metric] for r in rows]
            ax.plot([r["step"] for r in rows], values, color=self.colors[0])
            ax.set(
                title=metric,
                xlabel="Optimizer updates",
                ylabel="GiB (max device)" if metric == "peak_gib" else "Seconds",
            )
        self.save(
            fig,
            "runtime",
            "执行耗时与显存记录",
            "Historical logging definitions differ by implementation. Peak memory may be cumulative; "
            "these are not GPU-utilization measurements.",
        )

    def transfer(self):
        fig, axes = self.figure("Transfer to current, old and unvisited batches", 2)
        for i, relation in enumerate(["current", "seen", "not_yet_seen"]):
            selected = [r for r in self.d["effects"] if r["relation"] == relation]
            steps = sorted({r["step"] for r in selected})
            values = [np.mean([r["decrease"] for r in selected if r["step"] == step]) for step in steps]
            axes[0].plot(steps, values, color=self.colors[i], label=relation, linestyle=self.style)
        for i, panel in enumerate(["Math", "Code", "IF"]):
            rows = [r for r in self.d["effects"] if r["target"] == panel]
            axes[1].plot(
                [r["step"] for r in rows],
                [r["decrease"] for r in rows],
                color=self.panel_colors[panel],
                label=panel,
                linestyle=self.style,
            )
        for ax in axes:
            ax.axhline(0, color=self.ink, linewidth=0.5)
            ax.set(xlabel="End of training block (updates)", ylabel="Mean RKL before - after")
            ax.legend(loc="upper left", bbox_to_anchor=(0, 1.22), ncol=3, frameon=False)
        self.save(
            fig,
            "transfer",
            "当前批、旧批、未见批与留出的净变化",
            "Positive improves. Old/unvisited sets change across blocks; only scored batches contribute.\n"
            "Effects include optimizer history. These are descriptive interval changes, not causal coefficients.",
        )

    def support(self):
        metrics = [
            m
            for m in ["student_tail_mass", "teacher_tail_mass", "extra_teacher_mass"]
            if any(r["metric"] == m and r["value"] > 0 for r in self.d["curves"])
        ]
        if not metrics:
            return
        fig, axes = self.figure("Probability mass outside the compared support", len(metrics))
        for metric, ax in zip(metrics, axes, strict=True):
            for i, panel in enumerate(["Training scored", "All held-out"]):
                rows = [r for r in self.d["curves"] if r["metric"] == metric and r["panel"] == panel]
                ax.plot([r["step"] for r in rows], [r["value"] for r in rows], color=self.colors[i], label=panel)
            ax.set(title=metric, xlabel="Optimizer updates", ylabel="Mass / budget position")
        axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.22), ncol=2, frameon=False)
        self.save(
            fig,
            "support",
            "共享支持之外的概率质量",
            "Mass is recorded separately from conditional RKL. Division by prompts x H includes early-stop padding.",
        )

    def render(self):
        self.training()
        self.curves()
        self.curves(relative=True)
        self.matrix(effects=False)
        self.matrix()
        self.forgetting()
        self.transfer()
        self.support()
        self.questions()
        self.positions()
        self.auxiliary()
        self.runtime()
        return self.entries
