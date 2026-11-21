# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Local-only Trackio view of explicitly registered, append-only training logs.

Requires trackio==0.45.1 in a separate environment. The source JSONL files remain
unchanged. Trackio's synchronous bulk storage API supplies transactional writes
and deterministic log IDs, so replay after a crash cannot duplicate observations.
"""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path

from examples.mirroropd.research.core.io import dump, read

VERSION = "mirroropd-trackio-v1"
PROJECT = "mirroropd"
OVERVIEW = "reuse-ablation"
HISTORY = "history"


def registry(legacy_root, catalog, campaigns):
    """Reuse admitted metadata, never recursively discover frozen run copies."""
    runs = list(read(catalog)["runs"])
    plan = read(legacy_root / "plan.json")
    for lane in plan["lanes"]:
        study, name = Path(lane["study"]), lane["experiment"]
        runs.append(
            dict(
                id=name,
                name=name,
                pair="E102",
                optimizer="SGD",
                lr=lane["lr"],
                weight_decay=0,
                budget=plan["outer"],
                reuse=plan.get("reuse", 16),
                student="Qwen3-1.7B",
                teacher="Qwen3-4B-Instruct-2507",
                group="SGD E102",
                panel_scope="heldout",
                metrics=str(study / "training-sgd" / name / "metrics.jsonl"),
                status=str(study / "status.json"),
            )
        )
    for campaign in campaigns:
        plan = read(campaign / "plan.json")
        for run in plan["runs"]:
            runs.append(
                dict(
                    id=run["name"],
                    name=run["arm"],
                    pair="Qwen3-1.7B <- Qwen3-4B-Instruct-2507",
                    student=plan["pair"]["student"],
                    teacher=plan["pair"]["teacher"],
                    optimizer=plan["optimizer"]["name"],
                    lr=plan["optimizer"]["lr"],
                    weight_decay=plan["optimizer"]["weight_decay"],
                    seed=run["seed"],
                    reuse=run["reuse"],
                    adaptive=run["adaptive"],
                    budget_updates=plan["updates"],
                    group="Reuse ablation",
                    panel_scope="heldout",
                    metrics=str(campaign / "training" / run["name"] / "metrics.jsonl"),
                    status=str(campaign / "status.json"),
                    config=run["config"],
                    overview=True,
                )
            )
    paths, ids = set(), set()
    for run in runs:
        path = str(Path(run["metrics"]).resolve())
        if path in paths or run["id"] in ids:
            raise ValueError(f"Duplicate registered run identity: {run['id']}")
        paths.add(path)
        ids.add(run["id"])
        run["metrics"] = path
        run["run_id"] = hashlib.sha256(f"{VERSION}:{path}".encode()).hexdigest()[:24]
    return runs


def numeric(values):
    return {k: v for k, v in values.items() if type(v) in (int, float) and math.isfinite(v)}


def observations(record, run):
    """Raw scalar view plus a small, update-aligned view of the reuse study."""
    values = numeric(record["data"])
    update = values.get("step/optimizer")
    if type(update) not in (int, float) or update < 0 or int(update) != update:
        raise ValueError("Missing or invalid successful optimizer-update coordinate")
    update = int(update)
    is_loss = "optim/actor/loss" in values
    applied = values.get("optim/actor/update_applied")
    if is_loss and applied not in (None, 1):
        raise ValueError("Skipped optimizer update cannot be treated as a training observation")
    # Native optimizer-loss rows are observations before the logged update.
    x = update - 1 if is_loss else update
    if x < 0:
        raise ValueError("Pre-update training loss has no preceding successful-update coordinate")
    result = [(PROJECT, x, values)]
    selected = {}
    if is_loss:
        selected["train/loss"] = values["optim/actor/loss"]
        for key, value in values.items():
            if key.startswith(("optim/actor/", "distill/")) and "/position/" not in key and "_position/" not in key:
                selected[key] = value
    # Keep the overview usable; detailed position bins remain in PROJECT.
    for key, value in values.items():
        if key.startswith(("perf/timing/", "perf/throughput/")) or key in {
            "rollout/response_length/mean",
            "rollout/response_tokens",
            "rollout/response_count",
            "rollout/data_epoch",
        }:
            selected[key] = value
    post_update = {}
    for bank in ("public_initial", "public_teacher"):
        for domain in ("math", "code", "if"):
            key = f"eval/rkl/{bank}/dataset/{domain}/question_mean"
            if key in values:
                post_update[f"heldout/{domain}/{bank}"] = values[key]
            for statistic in ("absolute_reduction", "reduction_percent"):
                key = f"eval/rkl/{bank}/dataset/{domain}/{statistic}"
                if key in values:
                    post_update[f"heldout/{domain}/{bank}/{statistic}"] = values[key]
    for raw, alias in {
        "reuse/gap_over_drop": "controller/S",
        "reuse/used": "controller/reuse",
        "reuse/next": "controller/next_reuse",
    }.items():
        if raw in values:
            post_update[alias] = values[raw]
    for key, value in values.items():
        if key.startswith("reuse/") and key not in {"reuse/gap_over_drop", "reuse/used", "reuse/next"}:
            post_update[key] = value
    if is_loss and post_update:
        raise ValueError("Mixed pre-update loss and post-update measurements require separate coordinates")
    selected.update(post_update)
    if selected:
        result.append((HISTORY, x, selected))
        if run.get("overview"):
            result.append((OVERVIEW, x, selected))
        if run.get("comparison_project"):
            result.append((run["comparison_project"], x, selected))
    return result


def poll_run(run, previous, storage):
    path = Path(run["metrics"])
    if not path.exists():
        if previous.get("offset", 0):
            raise ValueError("Previously imported source disappeared")
        return previous, dict(state="pending", rows=0)
    stat = path.stat()
    # A replacement must be audited, even if size and mtime happen to match.
    stamp = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]
    projects = (PROJECT, HISTORY, OVERVIEW) if run.get("overview") else (PROJECT, HISTORY)
    if run.get("comparison_project"):
        projects += (run["comparison_project"],)
    counts_match = all(
        project in previous.get("counts", {})
        and storage.get_log_count(project, run_id=run["run_id"]) == previous["counts"][project]
        for project in projects
    )
    payload = path.read_bytes()
    offset = previous.get("offset", 0)
    if len(payload) < offset or (offset and hashlib.sha256(payload[:offset]).hexdigest() != previous.get("sha256")):
        raise ValueError("Imported source prefix was truncated or modified; register a new attempt")
    if previous.get("stamp") == stamp and counts_match:
        return previous, dict(state="following", rows=previous["rows"], unchanged=True)
    # A deleted database can be rebuilt from source, without trusting a cursor.
    if not counts_match:
        offset = 0
        previous = {}
    end = payload.rfind(b"\n") + 1
    batches = {}
    rows = previous.get("rows", 0)
    for line in payload[offset:end].splitlines(keepends=True):
        start = offset
        offset += len(line)
        try:
            record = json.loads(line)
            points = observations(record, run)
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Malformed source record at byte {start}: {error}") from error
        log_id = hashlib.sha256(f"{VERSION}:{run['run_id']}:{start}:".encode() + line).hexdigest()
        for project, step, metrics in points:
            batch = batches.setdefault(project, dict(metrics=[], steps=[], ids=[]))
            batch["metrics"].append(metrics)
            batch["steps"].append(step)
            batch["ids"].append(log_id)
        rows += 1
    config = {k: v for k, v in run.items() if k not in ("run_id", "overview")}
    config.update(
        imported_by=VERSION,
        axis="successful optimizer updates; training loss before update; evaluation and controller after update",
        raw_source=run["metrics"],
    )
    for project, batch in batches.items():
        storage.bulk_log(
            project=project,
            run=(
                run["name"]
                if project == OVERVIEW
                else run.get("comparison_name", run["id"])
                if project == run.get("comparison_project")
                else run["id"]
            ),
            run_id=run["run_id"],
            metrics_list=batch["metrics"],
            steps=batch["steps"],
            log_ids=batch["ids"],
            config=config,
        )
    # Save the cursor only after both projects have committed. Deterministic
    # log_id uniqueness makes replay safe across the two independent commits.
    state = dict(
        stamp=stamp,
        offset=end,
        rows=rows,
        sha256=hashlib.sha256(payload[:end]).hexdigest(),
        counts={project: storage.get_log_count(project, run_id=run["run_id"]) for project in projects},
    )
    return state, dict(state="following", rows=rows, partial_tail_bytes=len(payload) - end)


def synchronize(runs, root, storage):
    state_path = root / "cursors.json"
    cursors = read(state_path) if state_path.exists() else {}
    status = {}
    for run in runs:
        try:
            cursor, report = poll_run(run, cursors.get(run["id"], {}), storage)
            cursors[run["id"]] = cursor
            status[run["id"]] = report
        except Exception as error:
            status[run["id"]] = dict(state="error", error=repr(error))
    dump(state_path, cursors)
    dump(root / "import-status.json", dict(updated_at=time.time(), runs=status))
    return status


def configure_local(data):
    # Must happen before importing Trackio. Never inherit a configured remote
    # destination or implicitly synchronize to a Space or dataset.
    for name in ("TRACKIO_SPACE_ID", "TRACKIO_SERVER_URL", "TRACKIO_DATASET_ID", "TRACKIO_BUCKET_ID"):
        os.environ.pop(name, None)
    os.environ.update(
        TRACKIO_DIR=str(data), HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", GRADIO_ANALYTICS_ENABLED="False"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Service state directory")
    parser.add_argument("--data", type=Path, required=True, help="Local disk for SQLite; avoid network filesystems")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    args.root.mkdir(parents=True, exist_ok=True)
    configure_local(args.data.resolve())
    if importlib.metadata.version("trackio") != "0.45.1":
        raise RuntimeError("This adapter is validated against trackio==0.45.1")
    import trackio
    from trackio.sqlite_storage import SQLiteStorage

    with (args.root / "service.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runs = read(args.registry)["runs"]
        status = synchronize(runs, args.root, SQLiteStorage)
        errors = {name: row for name, row in status.items() if row["state"] == "error"}
        if errors:
            raise RuntimeError(f"Initial import failed; inspect import-status.json: {list(errors)}")
        if args.once:
            return
        # Local UI only: no share tunnel, account, remote backend or GPU logging.
        trackio.show(
            project=OVERVIEW,
            host=args.host,
            server_port=args.port,
            share=False,
            open_browser=False,
            block_thread=False,
            mcp_server=False,
        )
        dump(
            args.root / "service.json",
            dict(pid=os.getpid(), port=args.port, host=args.host, data=str(args.data.resolve()), version="0.45.1"),
        )
        while True:
            time.sleep(args.interval)
            synchronize(read(args.registry)["runs"], args.root, SQLiteStorage)


if __name__ == "__main__":
    main()
