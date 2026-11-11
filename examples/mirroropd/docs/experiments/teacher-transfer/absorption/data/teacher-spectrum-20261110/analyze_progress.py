"""Reproducible, question-paired progress summary; never selects on heldout."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
from examples.mirroropd.research.core.io import dump  # noqa: E402


def bands(rows):
    result = {}
    for name in rows[0]["bands"]:
        values = [r["bands"][name] for r in rows]
        total = sum(v["teacher_energy"] for v in values)
        if total <= 0:
            continue
        result[name] = dict(
            teacher_energy=total,
            absorption=sum(v["teacher_energy"] * v["absorption"] for v in values) / total,
            missing=sum(v["teacher_energy"] * v["missing"] for v in values) / total,
        )
    for value in result.values():
        value["energy_fraction"] = value["teacher_energy"] / result["all"]["teacher_energy"]
    return result


def main():
    plan = json.loads((ROOT / "manifest.json").read_text())
    state = json.loads((ROOT / "status.json").read_text())
    prior = REPO / "runs/teacher-absorption-spectrum-20261109"
    report = dict(
        checked_at=time.time(),
        allocation=json.loads((ROOT / "allocation.json").read_text()),
        counts=state["counts"],
        pairs={},
        interventions={},
        progress={},
    )
    for command in ("score", "extract", "head", "residuals", "representations", "weights", "reconstruct"):
        jobs = [j for j in plan["jobs"] if j["commands"][0][2] == command]
        complete = [state["jobs"][j["id"]] for j in jobs if state["jobs"][j["id"]]["phase"] == "complete"]
        duration = [(s["finished_at"] - s["started_at"]) / 60 for s in complete]
        report["progress"][command] = dict(
            complete=len(complete), total=len(jobs), median_minutes=float(np.median(duration)) if duration else None
        )
    generator = np.random.default_rng(20261110)
    for config in plan["configs"]:
        label = config["label"]
        records = json.loads(Path(config["records"]).read_text())
        domains = {str(r.get("row_id", r.get("id", i))): r["domain"] for i, r in enumerate(records)}
        risks = {}
        for role in ("base", *[f"s{s}" for s in config["stages"]]):
            path = ROOT / "results" / f"{label}.risk-{role}" / "risk.json"
            if path.exists():
                risks[role] = json.loads(path.read_text())["rows"]
        pair = dict(native_risk={}, output={}, head={}, weight_scope_21={})
        for split in ("train", "heldout"):
            for domain in ("all", "math", "code", "if"):
                selected = [
                    r["row_id"]
                    for r in risks["base"]
                    if r["split"] == split and (domain == "all" or domains[r["row_id"]] == domain)
                ]
                if not selected:
                    continue
                base_lookup = {r["row_id"]: r["loss"] for r in risks["base"]}
                b = np.array([base_lookup[i] for i in selected])
                for role, rows in risks.items():
                    lookup = {r["row_id"]: r["loss"] for r in rows}
                    values = np.array([lookup[i] for i in selected])
                    indices = generator.integers(0, len(b), size=(2000, len(b)))
                    ratio = values[indices].mean(1) / b[indices].mean(1)
                    pair["native_risk"][f"{split}/{domain}/{role}"] = dict(
                        n=len(b),
                        base=float(b.mean()),
                        loss=float(values.mean()),
                        remaining=float(values.mean() / b.mean()),
                        paired_question_bootstrap_95=np.quantile(ratio, [0.025, 0.975]).tolist(),
                    )
        full = prior / label / "full"
        for step in ("320",):
            data = json.loads((full / f"output-s{step}" / "scores.json").read_text())
            pair["output"][f"fisher/r64/s{step}"] = dict(
                calibration_capture=1 - data["diagnostics"]["relative_frobenius_error"] ** 2,
                heldout=bands(data["evaluation"]),
                random_heldout=bands(data["random_evaluation"]),
            )
            for metric in ("fisher", "logit", "probability"):
                for path in (ROOT / "results").glob(f"{label}.output-s{step}-{metric}-*/output/scores.json"):
                    data = json.loads(path.read_text())
                    pair["output"][path.parent.parent.name] = dict(
                        diagnostics=data["diagnostics"], heldout=bands(data["evaluation"])
                    )
        if config["same_coordinates"]:
            weight_rows = [json.loads(p.read_text()) for p in (full / "weights-s320").glob("[0-9]*.json")]
            for role in ("teacher", "opd"):
                pair["weight_scope_21"][role] = {
                    key: float(np.median([r[role][key] for r in weight_rows]))
                    for key in ("kappa_spec", "stable_rank", "rho_sigma", "delta_sigma")
                }
            for path in (ROOT / "results").glob(f"{label}.head-*/head.json"):
                data = json.loads(path.read_text())
                pair["head"][path.parent.name] = data["rows"]
            original = {r["row_id"]: r["loss"] for r in risks["s320"]}
            base = {r["row_id"]: r["loss"] for r in risks["base"]}
            for path in (ROOT / "results").glob(f"{label}.intervention-*/risk.json"):
                rows = json.loads(path.read_text())["rows"]
                report["interventions"][path.parent.name] = {}
                for split in ("train", "heldout"):
                    selected = [r for r in rows if r["split"] == split]
                    b = np.mean([base[r["row_id"]] for r in selected])
                    o = np.mean([original[r["row_id"]] for r in selected])
                    loss = np.mean([r["loss"] for r in selected])
                    report["interventions"][path.parent.name][split] = dict(
                        loss=float(loss), remaining=float(loss / b), gain_retained=float((b - loss) / (b - o))
                    )
        report["pairs"][label] = pair
    dump(ROOT / "progress-analysis.json", report)
    print(json.dumps(dict(counts=report["counts"], progress=report["progress"]), ensure_ascii=False))
    for label, pair in report["pairs"].items():
        print(label, "heldout", pair["native_risk"]["heldout/all/s320"], "weights", pair["weight_scope_21"])
        print("domains", {d: pair["native_risk"].get(f"heldout/{d}/s320") for d in ("math", "code", "if")})


if __name__ == "__main__":
    main()
