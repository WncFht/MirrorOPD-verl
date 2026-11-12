# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepare audited, read-only geometry tasks for completed formal OPD pairs."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from ..spectral.trajectory import run
from ..spectral.update_dynamics import atomic_json
from ..spectral.weights import TensorCheckpoint

OUTERS = (1, 2, 3, 4, 8, 12, 20, 30, 40)
RANKS = (16, 32, 64, 128, 256)
SCOPE = r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"


def read(path):
    return json.loads(Path(path).read_text())


def prepare(root, pairs, teacher_geometry=()):
    """Require explicit same-coordinate teacher selection, never infer ancestry from width."""
    root = Path(root).resolve()
    if not pairs or len(pairs) != len(set(pairs)) or not set(teacher_geometry).issubset(pairs):
        raise ValueError("Select unique pairs; teacher-geometry IDs must be selected pairs")
    lanes = {}
    for lane in sorted(root.glob("absorption-*4")):
        if not (lane / "prepared.json").exists():
            continue
        prepared = read(lane / "prepared.json")
        for pair, config in prepared["experiments"].items():
            if pair not in pairs or not (lane / f"{pair}-complete.json").exists():
                continue
            if pair in lanes:
                raise ValueError("Pair appears in two lanes")
            lanes[pair] = lane, config
    result = dict(
        schema="absorption-geometry-plan-v1",
        units="optimizer_steps",
        ranks=list(RANKS),
        outers=list(OUTERS),
        frame_side="right",
        arithmetic="float64",
        pairs={},
        tasks=[],
    )
    for pair in pairs:
        if pair not in lanes:
            raise ValueError(f"No completed attempt for pair: {pair}")
        lane, config = lanes[pair]
        status = read(lane / "status.json")
        if not any(
            s["label"] == f"{pair}-train" and s["state"] == "complete" and s["returncode"] == 0
            for s in status["stages"]
        ):
            raise ValueError(f"Pair training is not complete: {pair}")
        receipt = read(lane / f"{pair}-complete.json")
        if receipt != dict(outer=40, updates=640, bank_cap=7168, accuracy=False):
            raise ValueError(f"Pair budget acceptance missing: {pair}")
        cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=False)
        base_path = cfg["actor_rollout_ref"]["model"]["path"]
        # Teacher execution config is authoritative, not a model-name heuristic.
        teacher_path = cfg["distillation"]["teacher_models"]["teacher_model"]["model_path"]
        if any(not isinstance(path, str) or "${" in path for path in (base_path, teacher_path)):
            raise ValueError("Checkpoint paths must be explicit; analysis does not resolve training environments")
        base = TensorCheckpoint(base_path)
        tensors = base.selected(SCOPE)
        layers = read(Path(base_path) / "config.json")["num_hidden_layers"]
        expected = {
            f"model.layers.{layer}.{module}.{projection}_proj.weight"
            for layer in range(layers)
            for module, projections in (("self_attn", ("q", "k", "v", "o")), ("mlp", ("gate", "up", "down")))
            for projection in projections
        }
        if set(tensors) != expected:
            raise ValueError(f"Architecture requires explicit projection mapping: {pair}")
        evroot = lane / "training" / pair / "evaluation"
        risks = []
        for outer in range(41):
            event = read(evroot / f"events/outer-{outer:03d}/manifest.json")
            m = event["metadata"]
            if (
                event["status"] != "complete"
                or not m["coverage"]["scientific_complete"]
                or m["optimizer_steps"] != outer * 16
            ):
                raise ValueError(f"Incomplete RKL event: {pair}/{outer}")
            risks.append(
                dict(
                    outer=outer,
                    optimizer_steps=outer * 16,
                    metrics={k: v for k, v in m["metrics"].items() if k.startswith("eval/rkl/")},
                )
            )
        points, audits = [], []
        for outer in OUTERS:
            location = evroot / ("final-model" if outer == 40 else f"snapshots/outer-{outer:03d}")
            audit = read(location / "export-audit/manifest.json")
            meta = audit["metadata"]
            if (
                audit["status"] != "complete"
                or meta["native_reload_rkl"] != "passed"
                or meta["outer"] != outer
                or meta["optimizer_steps"] != outer * 16
            ):
                raise ValueError(f"Snapshot gate failed: {pair}/{outer}")
            path = location / "huggingface"
            if any(not (path / name).is_file() for name in meta["hf_files"]):
                raise ValueError(f"Snapshot file missing: {pair}/{outer}")
            base.compatible(TensorCheckpoint(path))
            points.append([outer * 16, str(path)])
            audits.append(audit)
        if pair in teacher_geometry:
            base.compatible(TensorCheckpoint(teacher_path))
        result["pairs"][pair] = dict(
            base=base_path,
            teacher=teacher_path,
            matrix_count=len(tensors),
            teacher_parameter_comparison=pair in teacher_geometry,
            risks=risks,
            export_audits=audits,
        )
        result["tasks"].extend(
            dict(
                condition=pair,
                base=base_path,
                tensor=tensor,
                points=points,
                ranks=list(RANKS),
                threads=2,
                **({"teacher_path": teacher_path} if pair in teacher_geometry else {}),
            )
            for tensor in tensors
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("prepare")
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--pairs", nargs="+", required=True)
    build.add_argument("--teacher-geometry", nargs="*", default=[])
    build.add_argument("--manifest", type=Path, required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--devices", nargs="+", default=["cpu"])
    execute.add_argument("--workers", type=int, default=1)
    execute.add_argument("--driver", choices=["gesvd", "gesvdj"])
    execute.add_argument("--resume-from", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.action == "prepare":
        if args.manifest.exists():
            raise FileExistsError(args.manifest)
        atomic_json(args.manifest, prepare(args.root, args.pairs, args.teacher_geometry))
    else:
        manifest = read(args.manifest)
        if manifest["schema"] != "absorption-geometry-plan-v1":
            raise ValueError("Unknown geometry protocol")
        run(manifest, args.output, args.devices, args.workers, driver=args.driver, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
