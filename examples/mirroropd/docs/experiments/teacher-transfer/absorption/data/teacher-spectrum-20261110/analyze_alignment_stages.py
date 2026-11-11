"""Same-direction local teacher-KL slopes at base, early, and endpoint, grouped by layer."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    result = {}
    sources = []
    for label in ["qwen-grpo", "qwen-sft", "r1-justrl"]:
        stages = {}
        identities = {}
        for stage, version in [("base", 9), ("early", 7), ("endpoint", 5)]:
            path = ROOT / f"response-probes-v{version}/{label}-full.json"
            sources.append(path)
            rows = json.loads(path.read_text())["rows"]
            entries = {}
            for direction in ["learned", "off", "null"]:
                name = direction if stage == "endpoint" else f"final-{direction}"
                groups = {}
                for row in rows:
                    d = row["directions"][name]
                    key = (str(row["row_id"]), row["position"], row["layer"])
                    if stage == "base":
                        identities[(direction, key)] = d["parameter_norm"]
                    else:
                        assert abs(d["parameter_norm"] / identities[(direction, key)] - 1) < 1e-6
                    group = f"{row['split']}/{row['domain']}/layer{row['layer']}"
                    groups.setdefault(group, []).append(d)
                summaries = {}
                for group, values in groups.items():
                    # Multiplication by the norm gives a slope for the actual fixed parameter displacement.
                    slopes = np.array([v["reverse_kl_unit_derivative"] * v["parameter_norm"] for v in values])
                    summaries[group] = dict(
                        n=len(values),
                        mean_actual_displacement_slope=float(slopes.mean()),
                        mean_unit_request_cosine=float(np.mean([v["request_cosine"] for v in values])),
                        improving_fraction=float(np.mean(slopes < 0)),
                        net_retention=float(abs(slopes.sum()) / abs(slopes).sum()),
                    )
                entries[direction] = summaries
            stages[stage] = entries
        result[label] = stages
    output = ROOT / "response-alignment-stages-008.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(
        json.dumps(
            dict(
                pairs=result,
                sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            ),
            indent=2,
        )
    )
    for label, stages in result.items():
        print(label)
        for stage, directions in stages.items():
            print(
                stage,
                {
                    name: {
                        g: round(v["mean_actual_displacement_slope"], 7)
                        for g, v in groups.items()
                        if g.endswith("layer0")
                    }
                    for name, groups in directions.items()
                },
            )


if __name__ == "__main__":
    main()
