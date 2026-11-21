"""Audit saved observables; plot a conditional-D2 bound, not measured IS ESS.

Ledger: curves = assumed uniform one-token D2 bounds, x = prefix tokens,
y = lower bound on asymptotic weight ESS fraction under those assumptions.
Measured inputs are frozen at update560; no model/GPU inference is performed.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = next(p for p in ROOT.parents if (p / "pyproject.toml").exists())
SNAP = ROOT / "census/metrics-snapshot.jsonl"
raw = SNAP.read_bytes()
records = [json.loads(line)["data"] for line in raw.splitlines()]
updates = [d for d in records if d.get("optim/actor/update_applied") == 1]
lengths = []
for d in updates:
    u = int(d["step/optimizer"])
    if u >= 209 and u % 16 == 1:
        count = sum(
            v for k, v in d.items() if k.startswith("distill/teacher_rkl_position/") and k.endswith("/token_count")
        )
        lengths.append({"update": u, "batch_mean_valid_tokens": count / 64})
with (ROOT / "prefix-lengths.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(lengths[0]))
    writer.writeheader()
    writer.writerows(lengths)
values = np.array([d["batch_mean_valid_tokens"] for d in lengths])
observed_banks = [
    {
        "update": int(d["step/optimizer"]),
        "initial_math": d["eval/rkl/public_initial/dataset/math/question_mean"],
        "teacher_math": d["eval/rkl/public_teacher/dataset/math/question_mean"],
    }
    for d in records
    if "eval/rkl/public_initial/dataset/math/question_mean" in d and d["step/optimizer"] in [0, 208, 320, 480, 560]
]
summary = {
    "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
    "measured_is_variance": None,
    "measured_current_behavior_kl": None,
    "reason": (
        "Saved metrics contain teacher KL and length bins; bank parquet contains KL sums; "
        "generation parquet contains token IDs but neither behavior nor current sampled-token log-probabilities."
    ),
    "training_length_estimator": (
        "Sum ten teacher_rkl_position token_count bins / batch64, first inner of each outer starting>=208; "
        "loss-mask valid tokens, not necessarily all generated tokens."
    ),
    "batches": len(lengths),
    "mean_valid_tokens": float(values.mean()),
    "min_batch_mean": float(values.min()),
    "max_batch_mean": float(values.max()),
    "bank_conditional_teacher_rkl": observed_banks,
    "uniform_conditional_d2_required_for_ess50_at4096": float(np.log(2) / 4096),
    "uniform_conditional_d2_required_for_ess90_at4096": float(-np.log(0.9) / 4096),
}
(ROOT / "prefix-diagnostics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
sys.path.insert(0, str(REPO.parent / "dotfiles/.agents/skills/figure-viz/assets/mpl"))
import fvstyle as fv  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

fv.apply_rc("paper")
fig, ax = plt.subplots(figsize=(7.2, 3.9), layout="constrained")
h = np.linspace(0, 7168, 300)
for eps, color in zip([0.00001, 0.0001, 0.001], fv.series_colors("paper", 3), strict=True):
    ax.plot(h, np.exp(-h * eps), color=color, label=f"Uniform token D2 bound = {eps:g}")
ax.set(xlabel="Prefix length (tokens)", ylabel="Guaranteed asymptotic ESS fraction", ylim=(0, 1.04))
ax.set_title("Conditional bound sensitivity; NOT measured IS variance", fontsize=11)
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
ax.grid(alpha=0.15)
fig.canvas.draw()
audit = fv.audit(fig)
(ROOT / "prefix-bound-audit.json").write_text(json.dumps(audit, default=str, indent=2) + "\n")
for ext in ["png", "pdf"]:
    fig.savefig(ROOT / f"prefix-bound.{ext}", dpi=180)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("audit", audit)
