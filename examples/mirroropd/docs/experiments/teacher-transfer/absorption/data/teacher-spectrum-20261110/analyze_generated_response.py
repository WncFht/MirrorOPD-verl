"""Group new generated-prefix directional responses by origin, domain and layer."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def summarize(rows, name):
    ds = [r["directions"][name] for r in rows]
    g = np.array([d["reverse_kl_unit_derivative"] * d["parameter_norm"] for d in ds])
    return dict(
        n=len(rows),
        questions=len({r["row_id"] for r in rows}),
        median_energy_over_gaussian=float(
            np.median(
                [
                    r["directions"][name]["unit_response_energy"] / r["directions"]["gaussian"]["unit_response_energy"]
                    for r in rows
                ]
            )
        ),
        actual_displacement_mean_slope=float(g.mean()),
        net_retention=float(abs(g.sum()) / abs(g).sum()) if abs(g).sum() else None,
        improving_fraction=float(np.mean(g < 0)),
        mean_request_cosine=float(np.mean([d["request_cosine"] for d in ds])),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = ROOT / "response-probes-v10"
    report = dict(conditions={}, stage_ratios={}, sha256={})
    allrows = {}
    for label in ["sft-base", "sft-early", "sft-endpoint", "rl-endpoint"]:
        path = root / f"{label}.json"
        rows = json.loads(path.read_text())["rows"]
        assert max(r["repeat_max_logprob_error"] for r in rows) <= 1e-5
        allrows[label] = rows
        report["sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        groups = {"all": rows}
        for row in rows:
            key = f"{row['origin']}/{row['domain']}/layer{row['layer']}"
            groups.setdefault(key, []).append(row)
        report["conditions"][label] = {
            group: {
                name: summarize(values, name)
                for name in [
                    "final-learned",
                    "final-diag",
                    "final-off",
                    "final-null",
                    "random_frames-0",
                    "random_frames-1",
                    "random_frames-2",
                ]
            }
            for group, values in groups.items()
        }
    base = {(r["row_id"], r["position"], r["layer"]): r for r in allrows["sft-base"]}
    for stage in ["early", "endpoint"]:
        values = {}
        rows = allrows[f"sft-{stage}"]
        assert {(r["row_id"], r["position"], r["layer"]) for r in rows} == set(base)
        for name in ["final-learned", "final-off", "final-null"]:
            ratios = []
            for row in rows:
                b = base[(row["row_id"], row["position"], row["layer"])]["directions"][name]
                d = row["directions"][name]
                assert abs(d["parameter_norm"] / b["parameter_norm"] - 1) < 1e-6
                ratios.append(d["unit_response_energy"] / b["unit_response_energy"])
            values[name] = float(np.median(ratios))
        report["stage_ratios"][f"{stage}_over_base"] = values
    args.output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            dict(
                conditions={k: v["all"] for k, v in report["conditions"].items()}, stage_ratios=report["stage_ratios"]
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
