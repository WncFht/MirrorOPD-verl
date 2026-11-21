"""Late-stage local rebound analysis, restricted to weights >=200.

Ledger: one outer observation is a complete 16-update batch starting >=200.
L(u) is training loss measured before update u+1; K(u) is fixed Math risk
at the same weights. Local bank rebound = K(end)-min(K within outer).
Residual plots remove outer intercepts and common inner-position means.
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
ds = [json.loads(line).get("data", {}) for line in raw.splitlines()]
U = {int(d["step/optimizer"]): d for d in ds if d.get("optim/actor/update_applied") == 1}
L = {u - 1: d["optim/actor/loss"] for u, d in U.items()}
K = {
    int(d["step/optimizer"]): d["eval/rkl/public_initial/dataset/math/question_mean"]
    for d in ds
    if "eval/rkl/public_initial/dataset/math/question_mean" in d
}
rows = []
for a in range(208, max(K) - 15, 16):
    if not all(u in K for u in range(a, a + 17)) or not all(u in L for u in range(a, a + 16)):
        continue
    k = np.array([K[a + i] for i in range(17)])
    losses = np.array([L[a + i] for i in range(16)])
    rows.append(
        dict(
            start=a,
            outer=a // 16 + 1,
            rebound=L[a] - L[a - 1],
            loss_start=losses[0],
            loss_end=losses[-1],
            loss_descent=losses[0] - losses[-1],
            bank_start=k[0],
            bank_end=k[-1],
            first4_gain=k[0] - k[4],
            first8_gain=k[0] - k[8],
            last8_rise=k[-1] - k[8],
            total_gain=k[0] - k[-1],
            rebound_from_min=k[-1] - min(k),
            best_inner=int(np.argmin(k)),
            tokens=U[a + 1]["distill/valid_tokens"],
        )
    )
with (ROOT / "outer-data.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def rank(x):
    _, inv, n = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(n) - (n - 1) / 2)[inv]


def resid(x, c):
    c = np.column_stack([np.ones(len(x)), c])
    return x - c @ np.linalg.lstsq(c, x, rcond=None)[0]


def vec(key):
    return np.array([r[key] for r in rows])


stage = np.column_stack([vec("start"), vec("bank_start")])
summary = dict(
    min_weight=200,
    first_complete_outer_start=208,
    max_successful_update=max(U),
    last_complete_outer_end=rows[-1]["start"] + 16,
    n=len(rows),
    source=str(SOURCE),
    sha256=hashlib.sha256(raw).hexdigest(),
    stats={},
    cases=[r for r in rows if r["start"] in [352, 368, 384, 400, 416]],
)
for xkey in ["rebound", "loss_start", "loss_end", "loss_descent"]:
    for ykey in ["first4_gain", "first8_gain", "last8_rise", "rebound_from_min", "total_gain"]:
        x, y = vec(xkey), vec(ykey)
        summary["stats"][xkey + ":" + ykey] = dict(
            pearson=corr(x, y),
            spearman=corr(rank(x), rank(y)),
            partial_stage_bank=corr(resid(x, stage), resid(y, stage)),
            partial_stage_bank_tokens=corr(
                resid(x, np.column_stack([stage, np.log(vec("tokens"))])),
                resid(y, np.column_stack([stage, np.log(vec("tokens"))])),
            ),
        )
starts = [r["start"] for r in rows]
x = np.array([L[a + i] for a in starts for i in range(16)])
y = np.array([K[a + i] for a in starts for i in range(16)])
out = np.repeat(np.arange(len(rows)), 16)
inner = np.tile(np.arange(16), len(rows))
co = np.column_stack([(out == j) * 1 for j in range(1, len(rows))])
ci = np.column_stack([(inner == j) * 1 for j in range(1, 16)])
rx = resid(x, np.column_stack([co, ci]))
ry = resid(y, np.column_stack([co, ci]))
# Resample entire outers; refit both effects in each circular block sample.
xm = x.reshape(-1, 16)
ym = y.reshape(-1, 16)


def twoway(a):
    return a - a.mean(axis=1, keepdims=True) - a.mean(axis=0, keepdims=True) + a.mean()


rng = np.random.default_rng(42)
bs = []
for _ in range(3000):
    ix = np.concatenate(
        [(a + np.arange(3)) % len(rows) for a in rng.integers(0, len(rows), size=(len(rows) + 2) // 3)]
    )[: len(rows)]
    bs.append(corr(twoway(xm[ix]).ravel(), twoway(ym[ix]).ravel()))
summary["levels"] = dict(
    raw=corr(x, y),
    outer_centered=corr(resid(x, co), resid(y, co)),
    outer_inner_adjusted=corr(rx, ry),
    block_bootstrap95=np.quantile(bs, [0.025, 0.975]).tolist(),
)
summary["half_correlations"] = {}
for lo, hi in [(0, 8), (8, 16)]:
    a, b = xm[:, lo:hi], ym[:, lo:hi]
    ra, rb = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
    summary["half_correlations"][f"{lo}-{hi - 1}"] = dict(
        outer_centered=corr(ra.ravel(), rb.ravel()),
        negative_outers=sum(corr(aa, bb) < 0 for aa, bb in zip(a, b, strict=True)),
    )
summary["phase_risk_change"] = [float(np.mean([K[a + i] - K[a] for a in starts])) for i in range(17)]
summary["phase_means"] = [dict(inner=i, loss=float(xm[:, i].mean()), risk=float(ym[:, i].mean())) for i in range(16)]
summary["late8"] = dict(
    n=len(rows),
    worsening=int(sum(vec("last8_rise") > 0)),
    mean_first8_gain=float(vec("first8_gain").mean()),
    mean_last8_rise=float(vec("last8_rise").mean()),
)
# Raw same-weight state correlations and changes in the highlighted region.
for a in [368, 384]:
    k = np.array([K[a + i] for i in range(17)])
    summary.setdefault("window_readouts", {})[str(a)] = [
        dict(inner=i, weight=a + i, loss=L.get(a + i), risk=K[a + i], risk_change=K[a + i] - K[a]) for i in range(17)
    ]
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
np.savetxt(
    ROOT / "residual-points.csv",
    np.column_stack([out, inner, rx, ry]),
    delimiter=",",
    header="outer_index,inner,loss_residual,risk_residual",
    comments="",
)
sys.path.insert(0, str(REPO.parent / "dotfiles/.agents/skills/figure-viz/assets/mpl"))
import fvstyle as fv  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

fv.apply_rc("paper")
plt.rcParams.update(
    {"font.size": 12, "axes.labelsize": 12, "axes.titlesize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11}
)
colors = fv.series_colors("paper", 4)


def save(fig, name):
    fig.canvas.draw()
    audit = fv.audit(fig, ignore=["facet-ylim-mismatch", "legend-coverage", "print-fit"])
    (ROOT / f"{name}-audit.json").write_text(json.dumps(audit, default=str, indent=2) + "\n")
    for ext in ["png", "pdf"]:
        fig.savefig(ROOT / f"{name}.{ext}", dpi=180)
    plt.close(fig)


fig, axes = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True, layout="constrained")
weights = np.arange(352, 417)
axes[0].plot(weights, [L[u] for u in weights], color=colors[0], marker=".", markersize=3)
axes[1].plot(weights, [K[u] for u in weights], color=colors[1], marker=".", markersize=3)
for ax in axes:
    for boundary in [352, 368, 384, 400, 416]:
        ax.axvline(boundary, color=fv.NEUTRALS["baseline"], linestyle=":", linewidth=0.8)
axes[0].set_ylabel("Training loss at weights u")
axes[1].set_ylabel("Initial Math bank RKL at u")
axes[1].set_xlabel("Successful updates / weight index u")
axes[1].set_xticks([352, 360, 368, 376, 384, 392, 400, 408, 416])
fig.get_layout_engine().set(rect=(0.015, 0.08, 0.97, 0.87))
fig.text(
    0.5,
    0.015,
    "Dotted lines = new batch; loss is before update u+1; bank is measured at the same weights",
    ha="center",
    fontsize=10,
    color=fv.NEUTRALS["baseline"],
)
save(fig, "368-384-aligned")

fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
for color, a in zip(colors[:2], [368, 384], strict=True):
    axes[0, 0].plot(range(16), [L[a] - L[a + i] for i in range(16)], color=color, label=f"Start {a}")
    axes[0, 1].plot(range(17), [K[a + i] - K[a] for i in range(17)], color=color, label=f"Start {a}")
axes[0, 0].set_ylabel("Loss decrease from outer start")
axes[0, 0].set_xlabel("Inner updates completed")
axes[0, 1].set_ylabel("Math RKL change (positive = worse)")
axes[0, 1].set_xlabel("Inner updates completed")
axes[0, 1].axhline(0, color=fv.NEUTRALS["baseline"], linestyle=":", linewidth=0.8)
for ax in axes[0]:
    ax.legend(frameon=False, fontsize=10)
for ax, xkey, ykey, label in [
    (axes[1, 0], "rebound", "first8_gain", "Cross-outer: rebound vs first8 gain"),
    (axes[1, 1], "loss_start", "first8_gain", "Cross-outer: start loss vs first8 gain"),
]:
    xx, yy = vec(xkey), vec(ykey)
    sc = ax.scatter(xx, yy, c=vec("start"), cmap="viridis", s=30)
    line = np.array([xx.min(), xx.max()])
    ax.plot(line, np.polyval(np.polyfit(xx, yy, 1), line), color=colors[2], linewidth=1)
    ax.set_title(f"{label}\nr={corr(xx, yy):+.2f}, n={len(rows)}", fontsize=11)
    ax.set_xlabel("Loss rebound" if xkey == "rebound" else "Training loss at new batch start")
    ax.set_ylabel("First8 Math RKL drop (positive = better)")
fig.colorbar(sc, ax=axes[1], label="Outer start update", shrink=0.75, pad=0.02)
fig.get_layout_engine().set(rect=(0.015, 0.07, 0.97, 0.86))
fig.text(
    0.5,
    0.015,
    "Only full outers starting after update200; same weights; local bank rise differs from whole-outer gain",
    ha="center",
    fontsize=10,
    color=fv.NEUTRALS["baseline"],
)
save(fig, "local-rebound-and-signal")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
axes[0].plot(range(17), summary["phase_risk_change"], color=colors[1], marker="o", markersize=3)
axes[0].set_xlabel("Inner updates completed")
axes[0].set_ylabel("Mean Math RKL change from outer start")
axes[1].scatter(rx, ry, c=inner, cmap="viridis", s=12, alpha=0.65)
line = np.array([rx.min(), rx.max()])
axes[1].plot(line, np.polyval(np.polyfit(rx, ry, 1), line), color=colors[2], linewidth=1)
axes[1].set_xlabel("Loss residual (outer + inner removed)")
axes[1].set_ylabel("Math RKL residual (same weights)")
axes[1].set_title(f"Late-stage adjusted level correlation r={corr(rx, ry):+.2f}", fontsize=11)
fig.get_layout_engine().set(rect=(0.015, 0.12, 0.97, 0.78))
fig.text(
    0.5,
    0.02,
    "One sequential run; residual anticorrelation describes states, not a causal effect of increasing loss",
    ha="center",
    fontsize=10,
    color=fv.NEUTRALS["baseline"],
)
save(fig, "plateau-residuals")
print(json.dumps({k: v for k, v in summary.items() if k not in ["window_readouts", "phase_means", "cases"]}, indent=2))
