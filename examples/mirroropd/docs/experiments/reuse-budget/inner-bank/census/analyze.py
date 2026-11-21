"""Census of within-outer fixed-bank RKL reversals, one sequential training run.

Ledger: one row/heatmap row = one complete16update outer, with17bank states.
Positive risk changes are worse. Distinguish any upward step from giving back
an earlier gain, and both from net whole-outer deterioration. Best inner is a
retrospective diagnostic, never a causal optimal reuse policy.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = next(p for p in ROOT.parents if (p / "pyproject.toml").exists())
SOURCE = (
    REPO
    / "examples/mirroropd/research-evidence/reuse-and-optimizers/fixed16-inner-bank-20261116/training"
    / "reuse-fixed16-shuffle-innerbank-s42/metrics.jsonl"
)
SNAP = ROOT / "metrics-snapshot.jsonl"
if not SNAP.exists():
    raise FileNotFoundError("The frozen analysis snapshot is required; do not substitute live metrics")
raw = SNAP.read_bytes()
records = []
for line in raw.splitlines():
    try:
        records.append(json.loads(line).get("data", {}))
    except json.JSONDecodeError:
        continue
U = {int(d["step/optimizer"]): d for d in records if d.get("optim/actor/update_applied") == 1}
B = {int(d["step/optimizer"]): d for d in records if "eval/rkl/public_initial/dataset/math/question_mean" in d}
ends = [
    e
    for e in range(16, max(B) + 1, 16)
    if all(u in B for u in range(e - 16, e + 1)) and all(u in U for u in range(e - 15, e + 1))
]


def stage(start):
    return "early <200" if start < 200 else "plateau 208–399" if start < 400 else "plateau >=400"


def census(end, bank="public_initial", domain="math"):
    start = end - 16
    key = f"eval/rkl/{bank}/dataset/{domain}/question_mean"
    k = np.array([B[u][key] for u in range(start, end + 1)])
    delta = np.diff(k)
    best = int(np.argmin(k))
    peak_gain = k[0] - min(k)
    giveback = k[-1] - min(k)
    loss = np.array([U[u]["optim/actor/loss"] for u in range(start + 1, end + 1)])
    r = dict(
        outer=end // 16,
        start=start,
        end=end,
        stage=stage(start),
        bank=bank,
        domain=domain,
        risk_start=k[0],
        risk_end=k[-1],
        any_up=bool(np.any(delta > 1e-10)),
        up_steps=int(sum(delta > 1e-10)),
        up_fraction=float(np.mean(delta > 1e-10)),
        up_total=float(np.maximum(delta, 0).sum()),
        down_total=float(np.maximum(-delta, 0).sum()),
        best_inner=best,
        best_risk=float(min(k)),
        peak_gain=float(peak_gain),
        giveback=float(giveback),
        net_gain=float(k[0] - k[-1]),
        giveback_fraction=float(giveback / peak_gain) if peak_gain > 1e-4 else None,
        first8_gain=float(k[0] - k[8]),
        last8_gain=float(k[8] - k[-1]),
        loss_start=float(loss[0]),
        loss_descent=float(loss[0] - loss[-1]),
        loss_rebound=float(loss[0] - U[start]["optim/actor/loss"]) if start in U else None,
        bank_path=k.tolist(),
        loss_path=loss.tolist(),
    )
    for eps in [1e-4, 5e-4]:
        suffix = "1e-4" if eps == 1e-4 else "5e-4"
        # Terminal risk exceeds an earlier best, after some initial improvement.
        r["giveback_gt_" + suffix] = bool(best > 0 and peak_gain > eps and giveback > eps)
        r["up_steps_gt_" + suffix] = int(sum(delta > eps))
        r["earliest_near_best_" + suffix] = int(np.flatnonzero(k <= min(k) + eps)[0])
    return r


rows = [census(e) for e in ends]
allrows = [
    census(e, bank, domain)
    for bank in ["public_initial", "public_teacher"]
    for domain in ["math", "code", "if"]
    for e in ends
]
fields = [k for k in rows[0] if not k.endswith("_path")]
with (ROOT / "outer-census.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows([{k: r[k] for k in fields} for r in rows])


def summarize(rs):
    n = len(rs)
    eligible = [r for r in rs if r["giveback_fraction"] is not None]
    out = dict(
        n=n,
        any_up=sum(r["any_up"] for r in rs),
        monotone=sum(not r["any_up"] for r in rs),
        total_up_steps=sum(r["up_steps"] for r in rs),
        total_steps=16 * n,
        rise_then_finish_worse_than_best=sum(r["best_inner"] > 0 and r["giveback"] > 1e-10 for r in rs),
        giveback_gt_1e4=sum(r["giveback_gt_1e-4"] for r in rs),
        giveback_gt_5e4=sum(r["giveback_gt_5e-4"] for r in rs),
        net_worse=sum(r["net_gain"] < -1e-10 for r in rs),
        last8_worse=sum(r["last8_gain"] < -1e-10 for r in rs),
        mean_first8_gain=float(np.mean([r["first8_gain"] for r in rs])),
        mean_last8_gain=float(np.mean([r["last8_gain"] for r in rs])),
        mean_net_gain=float(np.mean([r["net_gain"] for r in rs])),
        mean_giveback=float(np.mean([r["giveback"] for r in rs])),
        median_giveback=float(np.median([r["giveback"] for r in rs])),
        median_giveback_fraction=float(np.median([r["giveback_fraction"] for r in eligible])) if eligible else None,
        giveback_fraction_eligible=len(eligible),
        best_inner_hist={str(i): sum(r["best_inner"] == i for r in rs) for i in range(17)},
        median_best_inner=float(np.median([r["best_inner"] for r in rs])),
        median_earliest_within_1e4=float(np.median([r["earliest_near_best_1e-4"] for r in rs])),
        mean_loss_descent=float(np.mean([r["loss_descent"] for r in rs])),
    )
    out["risk_curve_mean_change"] = [
        float(np.mean([r["bank_path"][i] - r["risk_start"] for r in rs])) for i in range(17)
    ]
    return out


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def rank(x):
    _, ids, n = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(n) - (n - 1) / 2)[ids]


def residual(x, c):
    c = np.column_stack([np.ones(len(x)), c])
    return x - c @ np.linalg.lstsq(c, x, rcond=None)[0]


groups = {"all": rows, "after200": [r for r in rows if r["start"] >= 200]}
for label in dict.fromkeys(r["stage"] for r in rows):
    groups[label] = [r for r in rows if r["stage"] == label]
summary = dict(
    source=str(SOURCE),
    sha256=hashlib.sha256(raw).hexdigest(),
    max_update=max(U),
    max_bank=max(B),
    complete_outers=len(rows),
    included_updates=ends[-1],
    groups={k: summarize(rs) for k, rs in groups.items()},
    other_banks={},
    relations={},
    rows=rows,
    caveats=[
        "One run; correlated batches, not independent seeds",
        "Upticks are sensitive;1e-4/5e-4 are absolute sensitivity thresholds, not estimated noise floors",
        "Best-inner/giveback selected retrospectively; not a deployed policy",
    ],
)
for bank in ["public_initial", "public_teacher"]:
    for domain in ["math", "code", "if"]:
        selected = [r for r in allrows if r["bank"] == bank and r["domain"] == domain]
        summary["other_banks"][bank + "/" + domain] = {
            "all": summarize(selected),
            "after200": summarize([r for r in selected if r["start"] >= 200]),
        }
for label in ["all", "after200", "plateau >=400"]:
    rs = [r for r in groups[label] if r["loss_rebound"] is not None]
    c = np.array([[r["start"], r["risk_start"]] for r in rs])
    for xkey in ["loss_start", "loss_rebound", "loss_descent"]:
        for ykey in ["giveback", "last8_gain", "net_gain"]:
            x = np.array([r[xkey] for r in rs])
            y = np.array([r[ykey] for r in rs])
            summary["relations"][label + ":" + xkey + ":" + ykey] = dict(
                n=len(rs),
                pearson=corr(x, y),
                spearman=corr(rank(x), rank(y)),
                partial_stage_risk=corr(residual(x, c), residual(y, c)),
            )
late = groups["after200"]
robustness = {}
rng = np.random.default_rng(42)
for xkey, ykey in [
    ("loss_descent", "giveback"),
    ("loss_descent", "last8_gain"),
    ("loss_descent", "net_gain"),
    ("loss_rebound", "giveback"),
]:
    x = np.array([r[xkey] for r in late])
    y = np.array([r[ykey] for r in late])
    n = len(x)
    estimates = []
    for _ in range(4000):
        ids = np.concatenate([(a + np.arange(3)) % n for a in rng.integers(0, n, size=(n + 2) // 3)])[:n]
        estimates.append(corr(x[ids], y[ids]))
    loo = [corr(np.delete(x, j), np.delete(y, j)) for j in range(n)]
    robustness[xkey + ":" + ykey] = dict(
        pearson=corr(x, y),
        circular3outer_bootstrap95=np.quantile(estimates, [0.025, 0.975]).tolist(),
        leave_one_outer_out_range=[min(loo), max(loo)],
    )
summary["robustness_after200"] = robustness
for label, group in groups.items():
    for suffix in ["1e-4", "5e-4"]:
        summary["groups"][label]["up_steps_gt_" + suffix] = sum(r["up_steps_gt_" + suffix] for r in group)
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
sys.path.insert(0, str(REPO.parent / "dotfiles/.agents/skills/figure-viz/assets/mpl"))
import fvstyle as fv  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

fv.apply_rc("paper")
plt.rcParams.update(
    {"font.size": 12, "axes.labelsize": 12, "axes.titlesize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10}
)
colors = fv.series_colors("paper", 4)


def save(fig, name):
    fig.canvas.draw()
    audit = fv.audit(fig, ignore=["facet-ylim-mismatch", "legend-coverage", "print-fit"])
    (ROOT / f"{name}-audit.json").write_text(json.dumps(audit, default=str, indent=2) + "\n")
    for ext in ["png", "pdf"]:
        fig.savefig(ROOT / f"{name}.{ext}", dpi=180)
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(10, 8), layout="constrained", gridspec_kw={"width_ratios": [1.6, 1]})
mat = np.array([100 * (np.array(r["bank_path"]) / r["risk_start"] - 1) for r in rows])
vmax = np.percentile(np.abs(mat), 95)
im = axes[0].imshow(
    mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", origin="upper", extent=(-0.5, 16.5, len(rows) - 0.5, -0.5)
)
axes[0].scatter(
    [r["best_inner"] for r in rows],
    np.arange(len(rows)),
    s=12,
    c=fv.NEUTRALS["ink"] if "ink" in fv.NEUTRALS else colors[3],
    marker="x",
    linewidths=0.8,
)
axes[0].set_xlabel("Inner updates completed")
axes[0].set_ylabel("Outer (start update)")
ticks = list(range(0, len(rows), 3))
axes[0].set_yticks(ticks, [f"{rows[i]['outer']} ({rows[i]['start']})" for i in ticks])
axes[0].set_xticks([0, 4, 8, 12, 16])
axes[0].set_title("Relative Math RKL change; × = minimum")
fig.colorbar(im, ax=axes[0], label="Change from outer start (%)", shrink=0.7, pad=0.02)
axes[1].plot([r["giveback"] for r in rows], np.arange(len(rows)), color=colors[2], marker="o", markersize=3)
axes[1].set_ylim(len(rows) - 0.5, -0.5)
axes[1].set_yticks([])
axes[1].set_ylabel("Outer order (same as left)")
axes[1].set_xlabel("End RKL minus within-outer minimum")
axes[1].set_title("Terminal giveback (absolute RKL)")
for ax in axes:
    for i in range(1, len(rows)):
        if rows[i]["stage"] != rows[i - 1]["stage"]:
            ax.axhline(i - 0.5, color=colors[3], linestyle="--", linewidth=0.9)
fig.get_layout_engine().set(rect=(0.015, 0.07, 0.97, 0.86))
fig.text(
    0.5,
    0.015,
    "Fixed16 shuffle; each row is one outer; blue = lower risk, red = higher; boundaries at start200/400",
    ha="center",
    fontsize=10,
    color=fv.NEUTRALS["baseline"],
)
save(fig, "all-outer-heatmap")

fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), layout="constrained")
labels = list(dict.fromkeys(r["stage"] for r in rows))
for color, label in zip(colors, labels, strict=False):
    rs = groups[label]
    paths = np.array([np.array(r["bank_path"]) - r["risk_start"] for r in rs])
    label2 = f"{label} (n={len(rs)})"
    axes[0].plot(range(17), paths.mean(axis=0), color=color, label=label2, marker="o", markersize=2)
    axes[1].step(
        range(17),
        np.cumsum([sum(r["best_inner"] == i for r in rs) for i in range(17)]) / len(rs),
        where="post",
        color=color,
        label=label2,
    )
    axes[2].plot(
        [r["start"] for r in rs], [r["last8_gain"] for r in rs], color=color, marker="o", markersize=3, label=label2
    )
axes[0].set_xlabel("Inner updates completed")
axes[0].set_ylabel("Mean Math RKL change")
axes[1].set_xlabel("Best inner (retrospective)")
axes[1].set_ylabel("Fraction with minimum at or before x")
axes[1].set_ylim(0, 1.05)
axes[2].set_xlabel("Outer start update")
axes[2].set_ylabel("Last8 Math RKL drop (positive = better)")
axes[2].axhline(0, color=fv.NEUTRALS["baseline"], linestyle=":", linewidth=0.8)
for ax in axes[:2]:
    ax.set_xticks([0, 4, 8, 12, 16])
handles, leglabels = axes[0].get_legend_handles_labels()
fig.legend(handles, leglabels, ncol=3, loc="upper center", frameon=False, fontsize=10)
fig.get_layout_engine().set(rect=(0.015, 0.13, 0.97, 0.70))
fig.text(
    0.5,
    0.02,
    "Early stage shown separately; minimum is selected in hindsight; mean gain is not a causal reuse-policy estimate",
    ha="center",
    fontsize=10,
    color=fv.NEUTRALS["baseline"],
)
save(fig, "stage-distributions")

fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), layout="constrained")
rs = groups["after200"]
lastaxes = axes.ravel()
for ax, xkey, ykey in zip(
    lastaxes,
    ["loss_rebound", "loss_descent", "loss_start", "loss_rebound"],
    ["giveback", "giveback", "net_gain", "last8_gain"],
    strict=True,
):
    x = np.array([r[xkey] for r in rs])
    y = np.array([r[ykey] for r in rs])
    sc = ax.scatter(x, y, c=[r["start"] for r in rs], cmap="viridis", s=32)
    xx = np.array([x.min(), x.max()])
    ax.plot(xx, np.polyval(np.polyfit(x, y, 1), xx), color=colors[2], linewidth=1)
    ax.set_xlabel(
        {
            "loss_rebound": "Cross-batch loss rebound",
            "loss_descent": "Within-batch loss drop (15 updates)",
            "loss_start": "New-batch starting loss",
        }[xkey]
    )
    ax.set_ylabel(
        {
            "giveback": "Terminal RKL giveback",
            "net_gain": "Whole-outer Math RKL drop",
            "last8_gain": "Last8 Math RKL drop",
        }[ykey]
    )
    ax.set_title(f"After200 only: r={corr(x, y):+.2f}, n={len(rs)}")
fig.colorbar(sc, ax=axes, label="Outer start update", shrink=0.7, pad=0.02)
fig.get_layout_engine().set(rect=(0.015, 0.08, 0.97, 0.85))
fig.text(
    0.5,
    0.015,
    "Positive giveback = deterioration from best; positive drop = improvement; correlated samples from one run",
    ha="center",
    fontsize=10,
    color=fv.NEUTRALS["baseline"],
)
save(fig, "loss-versus-reversal")
print(json.dumps({k: v for k, v in summary.items() if k not in ["rows", "other_banks", "relations"]}, indent=2))
