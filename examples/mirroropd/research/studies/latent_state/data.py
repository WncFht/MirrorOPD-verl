# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One trajectory adapter for immutable training history and fixed generation banks."""

from pathlib import Path

from examples.mirroropd.research.core.io import dump, read

from .artifacts import digest, fingerprint, model_identity
from .geometry import state_positions

GROUPS = ["initial", "outer01_10", "outer11_20", "outer21_30", "outer31_40", "teacher"]


def select_indices(rows, count, seed):
    """Deterministically thin existing responses; never generate new tokens."""
    return sorted(
        sorted(range(len(rows)), key=lambda i: fingerprint([seed, rows[i]["source_id"], rows[i]["sample_index"]]))[
            :count
        ]
    )


def read_selected(task):
    """Read only row groups containing selected rows, retaining requested order."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(task["source"])
    wanted, found, start = set(task["indices"]), {}, 0
    for group in range(parquet.num_row_groups):
        end = start + parquet.metadata.row_group(group).num_rows
        take = wanted.intersection(range(start, end))
        if take:
            table = parquet.read_row_group(group)
            for index in sorted(take):
                found[index] = table.slice(index - start, 1).to_pylist()[0]
        start = end
    if len(found) != len(wanted):
        raise ValueError("Selected trajectory index is outside its shard")
    return [found[i] for i in task["indices"]]


def normalize(row, task, seed=42):
    """Apply only the valid response prefix; retain original trajectory and outer identity."""
    mask = list(map(bool, row["response_mask"]))
    n = sum(mask)
    if not row["prompt_ids"] or not n or len(mask) != len(row["response_ids"]):
        raise ValueError(f"Malformed trajectory: {row['id']}")
    if mask != [True] * n + [False] * (len(mask) - n):
        raise ValueError(f"Noncontiguous response mask: {row['id']}")
    prompt, response = list(row["prompt_ids"]), list(row["response_ids"][:n])
    prompt_key = fingerprint(prompt)
    return dict(
        trajectory_id=f"{task['group']}:{row['id']}",
        source_id=str(row["source_id"]),
        group=task["group"],
        outer=int(row.get("outer", 0)),
        prompt_key=prompt_key,
        # Same prompt stays on one side across all sources, preventing shared-prefix leakage.
        split="reference" if int(fingerprint([seed, prompt_key])[:8], 16) % 5 < 3 else "test",
        tokens=prompt + response[:-1],
        positions=state_positions(len(prompt), n),
        prompt_tokens=len(prompt),
        response_tokens=n,
    )


def prepare(catalog, output, *, pairs, per_outer=64, seed=42):
    """Create a frozen source plan from completed existing evaluations, without model inference."""
    import pyarrow.parquet as pq

    if per_outer < 1 or len(set(pairs)) != len(pairs):
        raise ValueError("Select positive per-outer count and unique pair IDs")
    output = Path(output)
    if (output / "plan.json").exists():
        raise FileExistsError(output / "plan.json")
    index = read(catalog)["pairs"]
    result = {}
    for pair in pairs:
        candidates = [r for r in index[pair]["training"] if r["files"] == 40]
        if len(candidates) != 1:
            raise ValueError(f"Require exactly one completed 40-outer trajectory source: {pair}")
        evaluation = Path(candidates[0]["evaluation"])
        manifest = read(evaluation / "manifest.json")
        if manifest["status"] != "complete":
            raise ValueError(f"Training evaluation is not complete: {pair}")
        cfg = manifest["metadata"]["configuration"]
        bootstrap = Path(cfg["bootstrap"])
        bootstrap_meta = read(bootstrap / "manifest.json")["metadata"]
        # Authoritative model path is recorded by the existing training configuration.
        import yaml

        config_path = evaluation.parents[2] / "configs" / f"{pair}.yaml"
        config = yaml.safe_load(config_path.read_text())
        teacher = config["distillation"]["teacher_models"]["teacher_model"]["model_path"]
        tasks = []

        def add(path, indices, group, tasks=tasks):
            if not indices:
                return
            record = dict(source=str(path.resolve()), source_sha256=digest(path), indices=indices, group=group)
            record["id"] = fingerprint(record)[:20]
            tasks.append(record)

        paths = sorted((evaluation / "trajectories").glob("outer-*.parquet"))
        if [int(p.stem.split("-")[-1]) for p in paths] != list(range(1, 41)):
            raise ValueError(f"Require exactly outer001 through outer040: {evaluation}")
        for path in paths:
            outer = int(path.stem.split("-")[-1])
            table = pq.ParquetFile(path).read(columns=["source_id", "sample_index"])
            if table.num_rows < per_outer:
                raise ValueError(f"Not enough saved responses in {path}")
            group = GROUPS[1 + (outer - 1) // 10]
            add(path, select_indices(table.to_pylist(), per_outer, seed), group)
        for name, bank in (("initial", "public_initial"), ("teacher", "public_teacher")):
            directory = Path(cfg["fixed_banks"].get(bank, bootstrap / name))
            child = read(directory / "manifest.json")
            if child["status"] != "complete":
                raise ValueError(f"Incomplete fixed bank: {directory}")
            for shard in sorted(child["files"]):
                if not shard.startswith("generations/") or not shard.endswith(".parquet"):
                    continue
                path = directory / shard
                rows = pq.ParquetFile(path).read(columns=["domain", "sample_index"]).to_pylist()
                indices = [i for i, row in enumerate(rows) if row["domain"] == "math" and row["sample_index"] == 0]
                for start in range(0, len(indices), 32):
                    add(path, indices[start : start + 32], name)
        selected = [normalize(row, task, seed) for task in tasks for row in read_selected(task)]
        if len({r["trajectory_id"] for r in selected}) != len(selected):
            raise ValueError(f"Duplicate selected trajectory identity: {pair}")
        banks = {
            g: {r["source_id"]: r["prompt_key"] for r in selected if r["group"] == g} for g in ("initial", "teacher")
        }
        if banks["initial"] != banks["teacher"] or len(banks["initial"]) != 128:
            raise ValueError(f"Require 128 matching initial/teacher Math prompts: {pair}")
        summary = dict(
            trajectories=len(selected),
            max_prefix_tokens=max(len(r["tokens"]) for r in selected),
            group_splits={
                g: {
                    side: sum(r["group"] == g and r["split"] == side for r in selected)
                    for side in ("reference", "test")
                }
                for g in GROUPS
            },
        )
        result[pair] = dict(
            summary=summary,
            teacher=model_identity(teacher),
            config=str(config_path.resolve()),
            config_sha256=digest(config_path),
            evaluation=str(evaluation),
            evaluation_manifest_sha256=digest(evaluation / "manifest.json"),
            bootstrap_manifest_sha256=digest(bootstrap / "manifest.json"),
            bootstrap_identity=bootstrap_meta.get("identity"),
            tasks=tasks,
            groups=GROUPS,
            risks=[
                dict(outer=o, metadata=read(evaluation / f"events/outer-{o:03d}/manifest.json")["metadata"])
                for o in (0, 10, 20, 30, 40)
            ],
        )
    plan = dict(
        schema="training-reference-state-v3",
        pairs=result,
        seed=seed,
        per_outer=per_outer,
        positions=8,
        reference_fraction=0.6,
        groups=GROUPS,
        sampling="Existing responses only; deterministic thinning within every outer; no new generation",
        fitting=(
            "Training student states only; whole-prompt 60/40 reference/test split; all bank trajectories query only"
        ),
    )
    dump(output / "plan.json", plan)
    return plan
