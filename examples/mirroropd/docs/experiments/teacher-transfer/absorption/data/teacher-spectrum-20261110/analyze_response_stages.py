"""Compare identical unit parameter directions under early and endpoint Jacobians."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def identity(row):
    return row["row_id"], row["position"], row["layer"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--early-version", type=int, default=7)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = dict(pairs={}, inputs={})
    for label in ["qwen-grpo", "qwen-sft", "r1-justrl"]:
        paths = [ROOT / f"response-probes-v{v}/{label}-full.json" for v in [5, 6, args.early_version]]
        data = [json.loads(p.read_text()) for p in paths]
        bykey = [{identity(r): r for r in d["rows"]} for d in data]
        assert set(bykey[0]) == set(bykey[1]) == set(bykey[2])
        assert all(r["repeat_max_logprob_error"] == 0 for d in data for r in d["rows"])
        pair = dict(n=len(bykey[0]), early_stage=data[2]["stage"], directions={})
        for direction in ["learned", "diag", "off", "null"]:
            entries = []
            for key, final in bykey[0].items():
                early = bykey[2][key]
                a = early["directions"][f"final-{direction}" if direction != "learned" else "final-learned"]
                b = final["directions"][direction]
                # Identical source weights give exactly the same normalized direction.
                assert np.isclose(a["parameter_norm"], b["parameter_norm"], rtol=1e-7)
                entries.append(
                    dict(
                        row_id=key[0],
                        position=key[1],
                        layer=key[2],
                        domain=final["domain"],
                        split=final["split"],
                        endpoint_over_early_energy=b["unit_response_energy"] / a["unit_response_energy"],
                        early_energy_over_gaussian=a["unit_response_energy"]
                        / early["directions"]["gaussian"]["unit_response_energy"],
                        final_energy_over_gaussian=b["unit_response_energy"]
                        / final["directions"]["gaussian"]["unit_response_energy"],
                    )
                )
            pair["directions"][direction] = dict(
                endpoint_over_early_energy_median=float(np.median([x["endpoint_over_early_energy"] for x in entries])),
                early_energy_over_gaussian_median=float(np.median([x["early_energy_over_gaussian"] for x in entries])),
                final_energy_over_gaussian_median=float(np.median([x["final_energy_over_gaussian"] for x in entries])),
                rows=entries,
            )
        report["pairs"][label] = pair
        report["inputs"] |= {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: {n: {a: b for a, b in v.items() if a != "rows"} for n, v in p["directions"].items()}
                for k, p in report["pairs"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
