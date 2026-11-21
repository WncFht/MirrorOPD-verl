# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit retained optimizer snapshots and reuse exact checkpoint spectrum/frame measurements."""

import argparse
import itertools
from pathlib import Path

from examples.mirroropd.research.core.io import dump, read


def prepare(catalog, output, *, experiments=None, outers=(20, 40)):
    import yaml

    from examples.mirroropd.research.studies.absorption_dynamics.spectral import RANKS, SCOPE
    from examples.mirroropd.research.studies.spectral.weights import TensorCheckpoint

    selected = set(experiments or ("E124", "E125", "E126", "E127", "E128", "E129", "E152", "E153", "E156", "E157"))
    if experiments is not None and (not experiments or len(experiments) != len(selected)):
        raise ValueError("Select nonempty unique experiment IDs")
    if not outers or list(outers) != sorted(set(outers)) or any(not 1 <= o <= 40 for o in outers):
        raise ValueError("Select unique increasing outer steps in [1, 40]")
    runs, inventory, tasks, pairs = [], [], [], {}
    for item in read(catalog)["runs"]:
        if item["id"] not in selected:
            continue
        if item["id"] in pairs:
            raise ValueError(f"Duplicate catalog experiment: {item['id']}")
        metrics = Path(item["metrics"])
        study = metrics.parents[2]
        name = item["id"]
        receipt = study / f"{name}-complete.json"
        if not receipt.exists():
            if experiments is None:
                continue
            raise ValueError(f"Missing completion receipt: {name}")
        if read(receipt) != dict(outer=40, updates=640, bank_cap=7168, accuracy=False):
            raise ValueError(f"Unexpected training budget: {name}")
        config = yaml.safe_load((study / "configs" / f"{name}.yaml").read_text())
        evaluation = metrics.parent / "evaluation"
        base = TensorCheckpoint(config["actor_rollout_ref"]["model"]["path"])
        tensors = base.selected(SCOPE)
        layers = read(base.path / "config.json")["num_hidden_layers"]
        expected = {
            f"model.layers.{layer}.{module}.{projection}_proj.weight"
            for layer in range(layers)
            for module, projections in (("self_attn", ("q", "k", "v", "o")), ("mlp", ("gate", "up", "down")))
            for projection in projections
        }
        if set(tensors) != expected:
            raise ValueError(f"Incomplete projection mapping: {name}")
        risks = []
        for outer in range(41):
            event = read(evaluation / f"events/outer-{outer:03d}/manifest.json")
            meta = event["metadata"]
            if (
                event["status"] != "complete"
                or not meta["coverage"]["scientific_complete"]
                or meta["optimizer_steps"] != outer * 16
            ):
                raise ValueError(f"Incomplete RKL event: {name}/{outer}")
            risks.append(
                dict(
                    outer=outer,
                    optimizer_steps=outer * 16,
                    metrics={k: v for k, v in meta["metrics"].items() if k.startswith("eval/rkl/")},
                )
            )
        points, audits = [], []
        for outer in outers:
            folder = evaluation / ("final-model" if outer == 40 else f"snapshots/outer-{outer:03d}")
            audit = read(folder / "export-audit/manifest.json")
            meta = audit["metadata"]
            if (
                audit["status"] != "complete"
                or meta["native_reload_rkl"] != "passed"
                or meta["outer"] != outer
                or meta["optimizer_steps"] != outer * 16
            ):
                raise ValueError(f"Unverified export: {name}/{outer}")
            path = folder / "huggingface"
            if any(not (path / file).is_file() for file in meta["hf_files"]):
                raise ValueError(f"Missing export weights: {name}/{outer}")
            base.compatible(TensorCheckpoint(path))
            points.append([outer * 16, str(path)])
            audits.append(audit)
        runtime = metrics.parent / "runtime"
        retained = []
        for step in sorted(runtime.glob("global_step_*")):
            actor = step / "actor"
            retained.append(
                dict(
                    outer=int(step.name.split("_")[-1]),
                    actor_present=actor.is_dir(),
                    optimizer_shards=len(list(actor.glob("optim*.pt"))),
                    model_shards=len(list(actor.glob("model*.pt"))),
                    data_present=(step / "data.pt").is_file(),
                )
            )
        inventory.append(dict(experiment=name, runtime=retained, model_exports=[int(step / 16) for step, _ in points]))
        run = dict(
            experiment=name,
            pair=item["pair"],
            optimizer=item["optimizer"],
            lr=item["lr"],
            base=str(base.path),
            points=points,
            tensors=tensors,
        )
        runs.append(run)
        pairs[name] = dict(
            base=str(base.path),
            teacher=config["distillation"]["teacher_models"]["teacher_model"]["model_path"],
            matrix_count=len(tensors),
            teacher_parameter_comparison=False,
            risks=risks,
            export_audits=audits,
            optimizer=item["optimizer"],
            lr=item["lr"],
            pair=item["pair"],
        )
        tasks.extend(
            dict(condition=name, base=str(base.path), tensor=tensor, points=points, ranks=list(RANKS), threads=2)
            for tensor in run["tensors"]
        )
    if experiments is not None and set(pairs) != selected:
        raise ValueError(f"Missing catalog experiments: {sorted(selected - set(pairs))}")
    result = dict(
        schema="absorption-geometry-plan-v1",
        ranks=list(RANKS),
        outers=list(outers),
        pairs=pairs,
        frame_side="right",
        arithmetic="float64",
        units="optimizer_steps",
        runs=runs,
        inventory=inventory,
        tasks=tasks,
        teacher_parameter_comparison=False,
    )
    dump(output, result)
    return result


def pilot(manifest, output):
    """CPU exact SVD on three k projections; compare both left and right subspaces across runs."""
    import torch

    from examples.mirroropd.research.studies.spectral.right_dynamics import angle_metrics, spectrum
    from examples.mirroropd.research.studies.spectral.weights import TensorCheckpoint, tensor_digest

    torch.set_num_threads(2)
    plan = read(manifest)
    records, comparisons = [], []
    cache = {}
    for run in plan["runs"]:
        if run["pair"] != "E102":
            continue
        base = TensorCheckpoint(run["base"])
        layers = read(base.path / "config.json")["num_hidden_layers"]
        target = TensorCheckpoint(run["points"][-1][1])
        for layer in (0, layers // 2, layers - 1):
            key = f"model.layers.{layer}.self_attn.k_proj.weight"
            initial, final = base.get(key), target.get(key)
            delta = final.double() - initial.double()
            u, s, vh = torch.linalg.svd(delta, full_matrices=False)
            energy = float(delta.square().sum())
            if energy <= 0:
                raise ValueError("Zero delta has no identifiable singular frame")
            ranks = [rank for rank in (16, 32, 64) if rank < min(delta.shape)]
            record = dict(
                experiment=run["experiment"],
                optimizer=run["optimizer"],
                lr=run["lr"],
                tensor=key,
                shape=list(delta.shape),
                singular_values=s.tolist(),
                base_sha256=tensor_digest(initial),
                target_sha256=tensor_digest(final),
                **spectrum(s, energy, ranks),
            )
            records.append(record)
            cache[(run["experiment"], key)] = (u[:, : max(ranks)], vh[: max(ranks)].T, delta, ranks)
    for a, b in itertools.combinations(records, 2):
        if a["tensor"] != b["tensor"]:
            continue
        if a["base_sha256"] != b["base_sha256"]:
            raise ValueError("Cross-run frames require identical initial weights")
        ua, va, da, ranks = cache[(a["experiment"], a["tensor"])]
        ub, vb, db, _ = cache[(b["experiment"], b["tensor"])]
        comparisons.append(
            dict(
                a=a["experiment"],
                b=b["experiment"],
                tensor=a["tensor"],
                delta_cosine=float((da * db).sum() / (da.norm() * db.norm())),
                frames=[
                    dict(rank=k, left=angle_metrics(ua[:, :k], ub[:, :k]), right=angle_metrics(va[:, :k], vb[:, :k]))
                    for k in ranks
                ],
            )
        )
    dump(
        output,
        dict(
            scope="three k projections, final cumulative delta; exploratory, not full-model evidence",
            records=records,
            comparisons=comparisons,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "pilot"))
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+")
    parser.add_argument("--outers", nargs="+", type=int, default=[20, 40])
    args = parser.parse_args()
    if args.action == "prepare":
        if args.catalog is None:
            parser.error("prepare requires --catalog")
        prepare(args.catalog, args.output, experiments=args.experiments, outers=args.outers)
    else:
        if args.manifest is None:
            parser.error("pilot requires --manifest")
        pilot(args.manifest, args.output)


if __name__ == "__main__":
    main()
