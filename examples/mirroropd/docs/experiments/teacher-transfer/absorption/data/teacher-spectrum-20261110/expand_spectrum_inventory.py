"""Resolve explorer curves to immutable model assets and exact-budget students."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
WORK = REPO.parent
OUT = ROOT / "spectrum-expanded-v4"
DOC = REPO / "examples/mirroropd/docs/experiments/teacher-transfer"
PLANS = REPO / "examples/mirroropd/research/studies/fixed_transfer/plans"
SOURCES = {}


def read(path):
    path = Path(path)
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def absolute(path):
    p = Path(path)
    return p if p.is_absolute() else WORK / p


def main():
    OUT.mkdir(exist_ok=False)
    provenance = read(DOC / "fixed-state-results/figures/student-paths/provenance.json")
    registry_data = read(REPO / "examples/mirroropd/assets/model_registry.json")
    registry = registry_data["models"]
    locations = registry_data["locations"]
    plans = [(p, read(p)) for p in PLANS.glob("*.json")]
    rows = []
    for item in provenance["sources"]:
        s = item["spec"]
        source = Path(s["initial_source"]).parent
        protocol = read(source / "protocol.json")
        complete = read(source / "complete.json")
        study = source.parent.parent.name
        student_key = "q3-1.7" if s["student"].startswith("q3") else "student"
        if s["student"] in ["r1", "r7"]:
            assets = []
            for alias in [s["student"], s["teacher"]]:
                matches = [m for m in registry if alias in m.get("aliases", [])]
                assert len(matches) == 1, (alias, matches)
                assets.append(matches[0])
            evidence = "model_registry aliases + original protocol"
        else:
            candidates = [
                (p, d)
                for p, d in plans
                if isinstance(d.get("models"), dict) and student_key in d["models"] and s["teacher"] in d["models"]
            ]
            exact = [(p, d) for p, d in candidates if d.get("name") == study]
            if exact:
                candidates = exact
            identities = {
                (str(d["models"][student_key]["path"]), str(d["models"][s["teacher"]]["path"])) for p, d in candidates
            }
            assert len(identities) == 1, (s["run_id"], identities)
            p, d = candidates[0]
            assets = [d["models"][student_key], d["models"][s["teacher"]]]
            evidence = str(p)
        base, teacher = assets
        base_path, teacher_path = absolute(base["path"]), absolute(teacher["path"])
        bconfig, tconfig = read(base_path / "config.json"), read(teacher_path / "config.json")
        fields = [
            "model_type",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
        ]
        shape_compatible = all(bconfig.get(k) == tconfig.get(k) for k in fields)
        if s["student"] in ["r1", "r7"]:
            direct = (
                s["student"] == "r1"
                and s["teacher"] in ["justrl", "minimath", "nemotron", "brorl", "deepscaler", "deepcoder"]
            ) or (s["student"] == "r7" and s["teacher"] in ["light", "skywork", "polaris"])
            lineage = "published_direct_parent" if direct else "cross_size"
        else:
            parent = teacher.get("parent")
            lineage = (
                "published_direct_parent"
                if parent == base.get("repo")
                else ("cross_size" if not shape_compatible else "shared_family_branch_or_control")
            )
        budget = int(s["budget"])
        checkpoints = {}
        plan_study = d.get("name", study) if s["student"] not in ["r1", "r7"] else study
        # The exact study/arm directory is authoritative, not a later continuation.
        arm = s["arm"]
        roots = [
            WORK / "model/checkpoints/teacher-studies/checkpoints" / study / arm,
            WORK / "wandb-logs/derived/MirrorOPD-verl" / study / "models" / arm,
            WORK / "wandb-logs/derived/MirrorOPD-verl" / study / "checkpoints" / arm,
            WORK / "model/checkpoints" / study / "models" / arm,
            WORK / "model/checkpoints" / study / "checkpoints" / arm,
            WORK / "model/checkpoints/teacher-studies/checkpoints" / plan_study / arm,
        ]
        for step in sorted({64, budget}):
            if step > budget:
                continue
            candidates = []
            for root in roots:
                candidates.extend([root / f"step{step:03d}", root / f"step{step:04d}"])
            for field in ["weights", "final_weights"]:
                if complete.get(field):
                    candidates.append(absolute(complete[field]))
                if protocol.get(field):
                    candidates.append(absolute(protocol[field]))
            mapped = []
            for path in candidates:
                for old, new in sorted(locations.items(), key=lambda x: -len(x[0])):
                    try:
                        suffix = path.relative_to(WORK / old)
                    except ValueError:
                        continue
                    mapped.append(WORK / new / suffix)
                    break
            verified = []
            expected = Path(s["initial_source"]).parent / f"fixed-{step:03d}.json"
            for path in (WORK / "model/checkpoints/teacher-studies/checkpoints").glob(f"*/{arm}/step{step:03d}"):
                candidate_study = path.parents[1].name
                aliases = [candidate_study, candidate_study.replace("qwen3-lineage-", "qwen3-teacher-lineage-")]
                for alias in aliases:
                    score = (
                        WORK / "wandb-logs/MirrorOPD-verl/analysis" / alias / "arms" / arm / f"fixed-{step:03d}.json"
                    )
                    if expected.exists() and score.exists() and score.read_bytes() == expected.read_bytes():
                        verified.append(path)
                        break
            # Same arm plus a byte-identical frozen snapshot links continuation readouts.
            for path in candidates + mapped + verified:
                if path.name not in [f"step{step:03d}", f"step{step:04d}"]:
                    continue
                if (path / "config.json").exists() and (
                    (path / "model.safetensors").exists() or (path / "model.safetensors.index.json").exists()
                ):
                    checkpoints[str(step)] = str(path.resolve())
                    break
        row = dict(
            run_id=s["run_id"],
            student=s["student"],
            teacher=s["teacher"],
            study=study,
            arm=arm,
            budget=budget,
            base=str(base_path.resolve()),
            teacher_path=str(teacher_path.resolve()),
            base_repo=base.get("repo"),
            teacher_repo=teacher.get("repo"),
            teacher_revision=teacher.get("revision"),
            parent=teacher.get("parent"),
            method=teacher.get("method"),
            lineage=lineage,
            architecture_compatible=shape_compatible,
            checkpoints=checkpoints,
            missing_student_steps=[
                step for step in sorted({64, budget}) if step <= budget and str(step) not in checkpoints
            ],
            source_protocol=str(source / "protocol.json"),
            model_evidence=evidence,
            initial_source=s["initial_source"],
            final_source=s["final_source"],
        )
        rows.append(row)
    ids = [r["run_id"] for r in rows]
    assert len(ids) == len(set(ids))
    html = (DOC / "fixed-state-results/explorer.html").read_text()
    a = html.index("const DATA = ") + len("const DATA = ")
    data, _ = json.JSONDecoder().raw_decode(html[a:])
    assert set(ids) == {r["run_id"] for r in data["runs"]}
    for row in rows:
        curve = next(r for r in data["runs"] if r["run_id"] == row["run_id"])
        assert row["budget"] == curve["budget"]
        row.update(label=curve["label"], added_stage=curve["added_stage"])
    (OUT / "inventory.json").write_text(json.dumps(dict(runs=rows, sources=SOURCES), indent=2))
    (OUT / "explorer-data.json").write_text(json.dumps(data))
    with (OUT / "coverage.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        "runs",
        len(rows),
        "teachers",
        len({r["teacher_path"] for r in rows}),
        "base-teacher pairs",
        len({(r["base"], r["teacher_path"]) for r in rows}),
    )
    print(
        "compatible",
        sum(r["architecture_compatible"] for r in rows),
        "students",
        sum(len(r["checkpoints"]) for r in rows),
    )
    for r in rows:
        print(r["run_id"], r["lineage"], r["teacher_repo"], list(r["checkpoints"]), r["missing_student_steps"])


if __name__ == "__main__":
    main()
