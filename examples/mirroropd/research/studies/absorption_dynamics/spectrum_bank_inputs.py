# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Bind spectrum interventions to the exact completed training banks and checkpoints."""

import gzip
import json
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf

from verl.trainer.evaluation.artifacts import ArtifactStore, file_sha256, identity
from verl.trainer.evaluation.trajectory import load_trajectories, trajectory_index

from ..spectral.update_dynamics import atomic_json


def prepare(geometry_manifests, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    pairs = {}
    for path in geometry_manifests:
        plan = json.loads(Path(path).read_text())
        for name, meta in plan["pairs"].items():
            if name in pairs:
                raise ValueError("Duplicate pair identity")
            task = next(t for t in plan["tasks"] if t["condition"] == name)
            root = next(p for p in Path(task["points"][0][1]).parents if p.name == "evaluation")
            store = ArtifactStore(root)
            configuration = store.manifest()["metadata"]["configuration"]
            bootstrap = Path(configuration["bootstrap"])
            bound = ArtifactStore(bootstrap).manifest(complete=True)
            points = []
            for step, checkpoint in task["points"]:
                event = ArtifactStore(root / f"events/outer-{step // 16:03d}")
                event.manifest(complete=True)
                expected = event.read_rows("rkl.parquet")
                if len(expected) != 512 or any(r["tokens"] <= 0 for r in expected):
                    raise ValueError("Require 512 valid fixed dual-bank training scores")
                points.append(dict(step=step, checkpoint=checkpoint, training_rows=expected))
            expected = {r["id"]: r for r in points[0]["training_rows"]}
            rows, evidence = [], {}
            for bank, child in (("public_initial", "initial"), ("public_teacher", "teacher")):
                directory = Path(configuration["fixed_banks"].get(bank, bootstrap / child))
                bank_store = ArtifactStore(directory)
                manifest = bank_store.manifest(complete=True)
                if directory == bootstrap / child and bound["metadata"]["children_sha256"][child] != file_sha256(
                    bank_store.manifest_path
                ):
                    raise ValueError("Bootstrap child manifest changed")
                for filename, entry in manifest["files"].items():
                    if file_sha256(directory / filename) != entry["sha256"]:
                        raise ValueError("Fixed bank payload changed")
                refs = [r for r in trajectory_index(bank_store, bank=bank) if r["id"] in expected]
                rows.extend(load_trajectories(refs))
                evidence[bank] = dict(manifest_sha256=file_sha256(bank_store.manifest_path), directory=str(directory))
            validate_rows(rows, points)
            # Distribution and forward precision remain those of this training pair.
            configs = list(root.parents[2].glob(f"configs/{name}.yaml"))
            if len(configs) != 1:
                raise ValueError("Cannot locate authoritative pair configuration")
            cfg = OmegaConf.to_container(OmegaConf.load(configs[0]), resolve=False)
            full = cfg["distillation"]["distillation_loss"]["full_vocab"]
            if full.get("input_id_mapping", "identity") != "identity":
                raise ValueError("Permuted teacher inputs require an explicit adapter")
            for section in (cfg["actor_rollout_ref"]["actor"]["fsdp_config"], cfg["distillation"]["shared_teacher"]):
                if section.get("forward_precision") is not None or section.get("mixed_precision", {}).get(
                    "param_dtype", "bf16"
                ) not in (
                    "bf16",
                    "bfloat16",
                ):
                    raise ValueError("This standalone adapter requires the recorded native BF16 forward")
            data = dict(
                schema="spectrum-bank-pair-v1",
                pair=name,
                base=meta["base"],
                teacher=meta["teacher"],
                rows=rows,
                points=points,
                banks=evidence,
                distribution=full.get("distribution", {}),
                input_mapping="identity",
                cap=7168,
                expected_rows=512,
                execution="HF native BF16 weights/hidden; FP32-output head; mainline stable RKL; FA2",
                scope="all transformer q/k/v/o/gate/up/down; other parameters unchanged",
            )
            target = destination / f"{name}.json.gz"
            target.write_bytes(gzip.compress(json.dumps(data, allow_nan=False).encode(), mtime=0))
            pairs[name] = dict(
                input=str(target.resolve()),
                sha256=file_sha256(target),
                rows=512,
                tokens=sum(r["valid_tokens"] for r in rows),
                points=[p["step"] for p in points],
            )
    result = dict(schema="spectrum-bank-plan-v1", pairs=pairs)
    atomic_json(destination / "plan.json", result)
    return result


def validate_rows(rows, points):
    actual = {r["id"]: r for r in rows}
    if len(actual) != len(rows) or len(rows) != 512:
        raise ValueError("Dual-bank coverage must contain 512 distinct trajectories")
    expected_counts = {
        (b, d): n for b in ("public_initial", "public_teacher") for d, n in (("math", 128), ("code", 64), ("if", 64))
    }
    if Counter((r["bank"], r["domain"]) for r in rows) != expected_counts:
        raise ValueError("Unexpected bank/domain coverage")
    for row in rows:
        mask = row["response_mask"]
        if not row["prompt_ids"] or not row["response_ids"] or len(mask) != len(row["response_ids"]):
            raise ValueError("Malformed fixed trajectory")
        if len(mask) > 7168 or sum(mask) != row["valid_tokens"] or not sum(mask):
            raise ValueError("Fixed response mask/cap changed")
    for point in points:
        expected = {r["id"]: r for r in point["training_rows"]}
        if expected.keys() != actual.keys() or any(
            expected[k]["tokens"] != sum(r["response_mask"]) for k, r in actual.items()
        ):
            raise ValueError("Training score identity/mask changed between checkpoints")
    return identity(rows)
