"""Same six-matrix screen for controlled SFT teachers and MiMo stage pairs.

These protocol strata remain separate from H1024 explorer curves. Three Qwen
teachers share a controlled Base/128-step SFT recipe. MiMo uses its original
long banks and supports Base/SFT/RL/Zero lineage comparisons.
"""

import csv
import hashlib
import json
from pathlib import Path

import torch
from measure_expanded_spectrum import OUT, ROOT, Measurements, dump

REPO = ROOT.parents[1]
WORK = REPO.parent
DOC = REPO / "examples/mirroropd/docs/experiments/teacher-transfer"


def main():
    torch.set_num_threads(2)
    assert (OUT / "measurement-complete.json").exists()
    m = Measurements()
    protocol_path = ROOT.parent / "teacher-construction-20261108/llm-v1/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    construction = protocol_path.parent
    conditions = []
    for arm in ["RAW", "CCS-rho0", "CCS-rho1"]:
        teacher = construction / "teachers" / arm / "step0128"
        student = construction / "opd" / arm / "step0256"
        conditions.append(
            dict(
                run_id="controlled-" + arm,
                base=protocol["base"],
                teacher_path=str(teacher),
                checkpoints={"256": str(student)} if student.exists() else {},
                protocol="controlled Qwen SFT128 / OPD256 / question-token-domain macro",
            )
        )
    registry = json.loads((REPO / "examples/mirroropd/assets/model_registry.json").read_text())["models"]
    paths = {
        x["repo"].split("/")[-1]: str((WORK / x["path"]).resolve()) for x in registry if "XiaomiMiMo/" in x["repo"]
    }
    for label, b, t in [
        ("MiMo-SFT", "MiMo-7B-Base", "MiMo-7B-SFT"),
        ("MiMo-RL", "MiMo-7B-Base", "MiMo-7B-RL"),
        ("MiMo-RL-from-SFT", "MiMo-7B-SFT", "MiMo-7B-RL"),
        ("MiMo-Zero-from-SFT", "MiMo-7B-SFT", "MiMo-7B-RL-Zero"),
        ("MiMo-Zero", "MiMo-7B-Base", "MiMo-7B-RL-Zero"),
    ]:
        conditions.append(
            dict(
                run_id=label,
                base=paths[b],
                teacher_path=paths[t],
                checkpoints={},
                protocol="MiMo historical long-bank question-token-domain macro",
            )
        )
    summaries = []
    for r in conditions:
        ident, compatible = m.pair(r)
        destination = OUT / "ancillary-runs" / r["run_id"]
        destination.parent.mkdir(exist_ok=True)
        (OUT / "pairs" / ident).rename(destination)
        matrices = [json.loads(p.read_text()) for p in destination.glob("*.json")]
        assert compatible and len(matrices) == 6
        keys = ["norm_over_base", "stable_rank", "top1pct", "top5pct", "kappa_spec", "rho_sigma"]
        import statistics

        facts = {key: statistics.median(p["teacher_delta"][key] for p in matrices) for key in keys}
        summaries.append(dict(**r, features=facts))
    qcurves = DOC / "data/teacher-construction-20261108/sft-comparison-v1/curves.csv"
    qdata = list(csv.DictReader(qcurves.open()))
    for row in summaries:
        if row["run_id"].startswith("controlled-"):
            arm = row["run_id"].replace("controlled-", "")
            row["curves"] = [x for x in qdata if x["arm"] == arm]
        elif row["run_id"] in ["MiMo-SFT", "MiMo-RL", "MiMo-Zero"]:
            arm = {"MiMo-SFT": "E73", "MiMo-RL": "E71", "MiMo-Zero": "E72"}[row["run_id"]]
            path = DOC / f"data/mimo-stage-20261104-final/{arm}/fixed-bank-rkl.csv"
            row["curves"] = list(csv.DictReader(path.open()))
            row["curve_source"] = str(path)
            row["curve_source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    dump(
        OUT / "ancillary-summary.json",
        dict(
            conditions=summaries,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            curves_sha256=hashlib.sha256(qcurves.read_bytes()).hexdigest(),
        ),
    )
    print("Eight ancillary pairs complete; protocols separate", flush=True)


if __name__ == "__main__":
    main()
