"""Full-layer and matched live-prefix response summaries, tick 6 immutable outputs."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def save(name, value):
    path = ROOT / name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2))


def main():
    report = dict(weights={}, responses={}, provenance={})
    for label in ["qwen-grpo", "qwen-sft", "r1-justrl"]:
        stages = {}
        for stage in ["80", "320"] if label == "r1-justrl" else ["64", "320"]:
            paths = sorted((ROOT / "results").glob(f"{label}.weights-s{stage}-l*/weights/[0-9]*.json"))
            rows = [json.loads(p.read_text()) for p in paths]
            rows = [r for r in rows if "tensor" in r]
            # These are all seven attention/MLP matrices, all 28 layers.
            assert len(rows) == 196, (label, stage, len(rows))
            total_teacher = sum(r["teacher"]["frobenius"] ** 2 for r in rows)
            total_opd = sum(r["opd"]["frobenius"] ** 2 for r in rows)
            total_missing = sum(r["missing"]["frobenius"] ** 2 for r in rows)
            stages[stage] = dict(
                tensors=len(rows),
                layers=sorted(set(int(re.search(r"layers\.(\d+)", r["tensor"])[1]) for r in rows)),
                learned_to_teacher_norm=float(np.sqrt(total_opd / total_teacher)),
                parameter_missing_energy_ratio=total_missing / total_teacher,
                parameter_cosine=(total_teacher + total_opd - total_missing) / (2 * np.sqrt(total_teacher * total_opd)),
                roles={
                    role: {
                        key: float(np.median([r[role][key] for r in rows]))
                        for key in ["stable_rank", "rho_sigma", "kappa_spec"]
                    }
                    | dict(top16_median=float(np.median([r[role]["top_energy"]["16"] for r in rows])))
                    for role in ["teacher", "opd"]
                },
                bands={
                    band: {
                        key: sum(
                            r["base_frame_bands"][band]["teacher_energy"] * r["base_frame_bands"][band][key]
                            for r in rows
                        )
                        / sum(r["base_frame_bands"][band]["teacher_energy"] for r in rows)
                        for key in ["absorption", "missing"]
                    }
                    for band in ["diag", "off", "null"]
                },
                tensor_rows=[dict(tensor=r["tensor"], teacher=r["teacher"], opd=r["opd"]) for r in rows],
            )
        report["weights"][label] = stages
        source = ROOT / f"response-probes-v2/{label}-full.json"
        report["provenance"][str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        rows = json.loads(source.read_text())["rows"]
        report["responses"][label] = dict(
            n=len(rows),
            max_repeat_error=max(r["repeat_max_logprob_error"] for r in rows),
            max_cache_error=max(r["cache_max_logprob_error"] for r in rows),
            directions={
                direction: dict(
                    energy_over_gaussian_median=float(
                        np.median(
                            [
                                r["directions"][direction]["unit_response_energy"]
                                / r["directions"]["gaussian"]["unit_response_energy"]
                                for r in rows
                            ]
                        )
                    ),
                    cosine_median=float(np.median([r["directions"][direction]["request_cosine"] for r in rows])),
                    positive_alignment=sum(r["directions"][direction]["request_inner"] > 0 for r in rows),
                    by_layer={
                        str(layer): dict(
                            energy_over_gaussian_median=float(
                                np.median(
                                    [
                                        r["directions"][direction]["unit_response_energy"]
                                        / r["directions"]["gaussian"]["unit_response_energy"]
                                        for r in rows
                                        if r["layer"] == layer
                                    ]
                                )
                            ),
                            cosine_median=float(
                                np.median(
                                    [r["directions"][direction]["request_cosine"] for r in rows if r["layer"] == layer]
                                )
                            ),
                        )
                        for layer in [0, 13, 27]
                    },
                )
                for direction in ["learned", "diag", "off", "null", "gaussian"]
            },
            rows=rows,
        )
    save("spectral-analysis-006.json", report)
    print(
        json.dumps(
            {
                k: {
                    label: {
                        s: {a: b for a, b in v.items() if a not in ["tensor_rows", "layers", "bands"]}
                        for s, v in stages.items()
                    }
                    for label, stages in report[k].items()
                }
                for k in ["weights"]
            },
            indent=2,
        )
    )
    for label, values in report["responses"].items():
        print(
            label, {d: {k: v for k, v in vals.items() if k != "by_layer"} for d, vals in values["directions"].items()}
        )


if __name__ == "__main__":
    main()
