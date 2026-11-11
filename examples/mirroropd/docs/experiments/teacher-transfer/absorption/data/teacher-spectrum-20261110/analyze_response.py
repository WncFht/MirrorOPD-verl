"""Summarize live-prefix response panels with exact KL slopes and cancellation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = dict(pairs={}, inputs={})
    for path in sorted(args.directory.glob("*-full.json")):
        data = json.loads(path.read_text())
        rows = data["rows"]
        report["inputs"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        pair = dict(
            n=len(rows),
            questions=len(set(r["row_id"] for r in rows)),
            max_repeat_error=max(r["repeat_max_logprob_error"] for r in rows),
            directions={},
        )
        for direction in rows[0]["directions"]:
            groups = {"all": rows}
            groups |= {
                f"{split}/{domain}": [r for r in rows if r["split"] == split and r["domain"] == domain]
                for split in ["train", "heldout"]
                for domain in ["math", "code", "if"]
            }
            groups |= {
                f"{split}/{domain}/layer{layer}": [
                    r for r in rows if r["split"] == split and r["domain"] == domain and r["layer"] == layer
                ]
                for split in ["train", "heldout"]
                for domain in ["math", "code", "if"]
                for layer in [0, 13, 27]
            }
            values = {}
            for name, selected in groups.items():
                if not selected:
                    continue
                slopes = np.array([r["directions"][direction]["reverse_kl_unit_derivative"] for r in selected])
                values[name] = dict(
                    n=len(selected),
                    mean_slope=float(slopes.mean()),
                    negative_slopes=int((slopes < 0).sum()),
                    cancellation=float(abs(slopes.sum()) / np.abs(slopes).sum()) if np.abs(slopes).sum() else 0,
                    median_energy_ratio=float(
                        np.median(
                            [
                                r["directions"][direction]["unit_response_energy"]
                                / r["directions"]["gaussian"]["unit_response_energy"]
                                for r in selected
                            ]
                        )
                    ),
                )
            pair["directions"][direction] = values
        report["pairs"][data["label"]] = pair
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v["directions"]["off"] for k, v in report["pairs"].items()}, indent=2))


if __name__ == "__main__":
    main()
