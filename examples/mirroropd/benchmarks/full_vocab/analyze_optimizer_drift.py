# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Replay AdamW and compare saved tiny MDPO head calls with a CPU FP64 oracle.

Use test_mirror_fsdp2.py --diagnostic-dir first. This does not alter the strict
backend parameter gate; local FP32 updates are checked against an independent
FP64 optimizer formula. Diagnostic snapshots are trusted local torch files.
"""

import argparse
import json
from pathlib import Path

import torch

from verl.trainer.distillation.full_vocab import _mirror_rows
from verl.trainer.distillation.termination import TerminationActionMap
from verl.workers.config.distillation_spec import DistillationLossSettings


def adam(snapshot, name):
    """Predict one local AdamW update in FP64 from the recorded state."""
    i = snapshot["parameter_names"].index(name)
    o = snapshot["optimizer_before"]
    group = next(g for g in o["param_groups"] if i in g["params"])
    state = o["state"].get(i, {})
    p = snapshot["parameters_before"][name].double()
    g = snapshot["gradients_after_clip"][name].double()
    b1, b2 = group["betas"]
    t = float(state.get("step", 0)) + 1
    m = state.get("exp_avg", torch.zeros_like(g)).double() * b1 + (1 - b1) * g
    v = state.get("exp_avg_sq", torch.zeros_like(g)).double() * b2 + (1 - b2) * g.square()
    return p * (1 - group["lr"] * group["weight_decay"]) - group["lr"] * (m / (1 - b1**t)) / (
        torch.sqrt(v / (1 - b2**t)) + group["eps"]
    )


def main():
    """Analyze the two-rank, three-update weighted-MDPO diagnostic fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--termination-ids", type=int, nargs="+", default=[1, 4])
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = args.diagnostic_dir
    summaries = []
    strict_ab_passed = True
    spec = DistillationLossSettings(
        names=("mirror",), payload_kind="hidden", objective="mdpo", eta=args.eta, kappa=args.kappa, diagnostics=True
    )

    for rank in range(2):
        x = torch.load(root / f"rank{rank}.pt", weights_only=False)
        assert len(x) == 6
        strict_ab_passed &= all(
            torch.allclose(x[2]["parameters_after"][n], x[5]["parameters_after"][n], atol=1e-6, rtol=1e-5)
            for n in x[2]["parameters_after"]
        )
        for i in range(3):
            a, b = x[i], x[i + 3]
            row = {"rank": rank, "step": i, "parameters": [], "head": []}
            for name in a["parameters_before"]:
                pa, pb = a["parameters_after"][name], b["parameters_after"][name]
                delta = float((pa - pb).abs().max())
                gd = float((a["gradients_before_clip"][name] - b["gradients_before_clip"][name]).abs().max())
                oa, ob = adam(a, name), adam(b, name)
                residual = max(float((oa - pa.double()).abs().max()), float((ob - pb.double()).abs().max()))
                torch.testing.assert_close(oa, pa.double(), atol=2e-8, rtol=2e-6)
                torch.testing.assert_close(ob, pb.double(), atol=2e-8, rtol=2e-6)
                row["parameters"].append(
                    {
                        "name": name,
                        "before_max_abs": float(
                            (a["parameters_before"][name] - b["parameters_before"][name]).abs().max()
                        ),
                        "gradient_before_clip_max_abs": gd,
                        "gradient_after_clip_max_abs": float(
                            (a["gradients_after_clip"][name] - b["gradients_after_clip"][name]).abs().max()
                        ),
                        "after_max_abs": delta,
                        "fp64_adam_predicted_difference": float((oa - ob).abs().max()),
                        "fp64_adam_max_residual": residual,
                    }
                )
            for p, q in zip(a["head_calls"], b["head_calls"], strict=True):
                assert all(torch.equal(t, u) for t, u in zip(p["inputs"], q["inputs"], strict=True))
                z = [t.double() for t in p["inputs"]]
                actions = TerminationActionMap.create(len(z[1]), args.termination_ids, "cpu")
                out = _mirror_rows(*z, spec, actions=actions)
                row["head"].append(
                    {
                        "input_equal": True,
                        "dz_backend_difference": float((p["output"][1] - q["output"][1]).abs().max()),
                        "fused_dz_fp64_error": float((p["output"][1] - out[1]).abs().max()),
                        "precompute_dz_fp64_error": float((q["output"][1] - out[1]).abs().max()),
                    }
                )
            summaries.append(row)
    args.output.write_text(
        json.dumps(
            {
                "diagnostic_dir": str(root),
                "strict_ab_passed": strict_ab_passed,
                "optimizer_formula": (
                    "FP64 AdamW with recorded before-state and post-clip gradient; "
                    "checked atol=2e-8 rtol=2e-6 against individual FP32 updates, not the backend gate"
                ),
                "steps": summaries,
            },
            indent=2,
        )
        + "\n"
    )
    for r in summaries:
        p = max(r["parameters"], key=lambda z: z["after_max_abs"])
        print(r["rank"], r["step"], p)


if __name__ == "__main__":
    main()
