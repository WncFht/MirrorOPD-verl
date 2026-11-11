# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Compile declarative checkpoint series and inspect resumable measurement status.

A series binds a trained branch to checkpoints, targets, and frozen banks.
Compilation resolves all local paths and validates inputs before GPU work.
Execution remains the single stage worker; no model-specific execution forks.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from .inputs import load_records, normalize_records, source_fingerprint, validate_records
from .runtime import checkpoint_identity, write_json


def compile_plan(plan, *, base, gpus):
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("Use a nonempty list of distinct physical GPU indices")

    def resolve(value):
        path = Path(value)
        return str((base / path).resolve()) if not path.is_absolute() else str(path.resolve())

    models = {key: resolve(value) for key, value in plan["models"].items()}
    identities = {key: checkpoint_identity(path) for key, path in models.items()}
    defaults = dict(cap=1024, group_size=8, checkpointing=True) | plan.get("defaults", {})
    banks = {}
    for name, bank in plan["banks"].items():
        bank = dict(defaults) | bank
        bank["records"] = resolve(bank.pop("path"))
        if bank["cap"] < 1 or bank["group_size"] < 1:
            raise ValueError("cap and group_size must be positive")
        rows = normalize_records(load_records(bank["records"]), cap=bank["cap"], bank_id=name)
        bank["audit"] = validate_records(rows)
        bank["fingerprint"] = source_fingerprint(rows)
        banks[name] = bank
    tasks, ids = [], set()
    for i, series in enumerate(plan["series"]):
        lane = series.get("lane", gpus[i % len(gpus)])
        if lane not in gpus:
            raise ValueError(f"Series lane {lane} is outside the declared GPU allocation")
        for checkpoint in series["checkpoints"]:
            student = checkpoint["student"]
            step = checkpoint["step"]
            if not isinstance(step, int) or step < 0:
                raise ValueError("Checkpoint step must be a nonnegative integer")
            for target in series["targets"]:
                for bank_name in series["banks"]:
                    bank = banks[bank_name]
                    name = f"{series['id']}-{target}-{bank_name}-s{step:04d}"
                    if len(name) > 180:
                        digest = hashlib.sha256(name.encode()).hexdigest()[:16]
                        name = f"{name[:140]}-{digest}"
                    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in ids:
                        raise ValueError(f"Invalid or duplicate task id: {name}")
                    ids.add(name)
                    tasks.append(
                        dict(
                            id=name,
                            lane=lane,
                            model=series["model"],
                            branch=series.get("branch", series["id"]),
                            label=series.get("label", series.get("branch", series["id"])),
                            target=target,
                            bank=bank_name,
                            stage=step,
                            student=models[student],
                            teacher=models[target],
                            records=bank["records"],
                            cap=bank["cap"],
                            group_size=bank["group_size"],
                            checkpointing=bank["checkpointing"],
                            input_identity=dict(
                                student=identities[student], teacher=identities[target], records=bank["fingerprint"]
                            ),
                        )
                    )
    if not tasks:
        raise ValueError("Plan contains no measurement tasks")
    return dict(schema=3, tasks=tasks, banks=banks, gpus=gpus, execution=plan.get("execution", {}), source_plan=plan)


def status(manifest, root):
    rows = []
    for task in manifest["tasks"]:
        directory = root / "results" / task["id"]
        result = directory / "result.json"
        state = "pending"
        if result.exists():
            payload = json.loads(result.read_text())
            signature = payload.get("signature", {})
            state = "complete" if payload.get("finished") and signature.get("task") == task else "stale"
        elif (directory / "input.json").exists():
            state = "partial"
        rows.append(
            dict(
                task=task["id"],
                lane=task["lane"],
                state=state,
                completed_groups=len(list(directory.glob("group-*.json"))),
            )
        )
    return dict(completed=sum(r["state"] == "complete" for r in rows), planned=len(rows), tasks=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compile_parser = commands.add_parser("compile")
    compile_parser.add_argument("--plan", type=Path, required=True)
    compile_parser.add_argument("--gpus", type=int, nargs="+", required=True)
    compile_parser.add_argument("--out", type=Path, required=True)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compile":
        plan = json.loads(args.plan.read_text())
        manifest = compile_plan(plan, base=args.plan.resolve().parent, gpus=args.gpus)
        write_json(args.out, manifest)
        print(json.dumps(dict(tasks=len(manifest["tasks"]), manifest=str(args.out))))
    else:
        print(json.dumps(status(json.loads(args.manifest.read_text()), args.manifest.parent), indent=2))


if __name__ == "__main__":
    main()
