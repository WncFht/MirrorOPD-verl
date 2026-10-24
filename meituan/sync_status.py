# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read local synchronization state without packing, uploading, fetching or changing Git refs."""

import argparse
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    from meituan import code_archive, log_sync, sync_observe, transfer
except ModuleNotFoundError:
    import code_archive
    import log_sync
    import sync_observe
    import transfer


def query(repo, *args):
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "--no-pager", "-c", "core.quotepath=false", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def repository_status(repo):
    return {
        "path": str(repo),
        "head": query(repo, "log", "-1", "--format=%H %s"),
        "branch": query(repo, "symbolic-ref", "--quiet", "--short", "HEAD"),
        "working_tree": query(repo, "status", "--short", "--untracked-files=normal"),
        "branch_status": query(repo, "status", "-sb", "--untracked-files=no"),
    }


def code_status(root, target):
    packed, confirmed = code_archive.refs(target)
    result = repository_status(root)
    result.update(
        target=target,
        packed=query(root, "rev-parse", "--verify", packed),
        confirmed=query(root, "rev-parse", "--verify", confirmed),
        relative_to_peer=query(root, "rev-list", "--left-right", "--count", f"{confirmed}...HEAD"),
    )
    return result


def logs_status(snap):
    state_path = snap / "logs_state.json"
    state = {"path": str(state_path), "meaning": "last packed files, NOT delivery or cloud acknowledgement"}
    if state_path.is_file():
        try:
            files = json.loads(state_path.read_text())
            if not isinstance(files, dict) or not all(isinstance(name, str) for name in files):
                raise ValueError("Expected a filename-to-state mapping in logs_state.json")
            state.update(
                files=len(files),
                runs=len({name.split("/")[0] for name in files}),
                updated_at=datetime.fromtimestamp(state_path.stat().st_mtime, timezone.utc).isoformat(),
            )
        except (OSError, ValueError, TypeError) as error:
            state["error"] = str(error)
    source = transfer.wandb_runs_dir("pack")
    destination = transfer.wandb_runs_dir("unpack")
    toplevel = query(destination, "rev-parse", "--show-toplevel") if destination.exists() else None
    queue = {"status": "not_initialized", "jobs": [], "counts": {}}
    if toplevel:
        gitdir = query(Path(toplevel), "rev-parse", "--absolute-git-dir")
        if gitdir:
            path = Path(gitdir) / "mirroropd-upload/queue.json"
            queue["path"] = str(path)
            if path.exists():
                try:
                    data = log_sync.load(path.parent)
                    queue.update(
                        status="available", jobs=data["jobs"], counts=dict(Counter(j["status"] for j in data["jobs"]))
                    )
                except (OSError, ValueError, KeyError, TypeError) as error:
                    queue.update(status="unreadable", error=str(error))
    return {
        "source": str(source),
        "source_runs": sum(1 for path in source.glob("offline-run-*") if path.is_dir()),
        "destination": str(destination),
        "destination_runs": sum(1 for path in destination.glob("offline-run-*") if path.is_dir()),
        "packing_state": state,
        "git": repository_status(Path(toplevel)) if toplevel else None,
        "upload_queue": queue,
        "cloud_status": "unknown: only recorded CLI upload outcomes are available; no live cloud query",
    }


def reports(directory, section, limit=8):
    result = []
    for path in sorted(directory.glob("*.json"), reverse=True):
        try:
            report = json.loads(path.read_text())
            if report.get("status") not in ("running", "success", "failed", "interrupted", "partial"):
                raise ValueError("Missing or unknown operation status")
            if not isinstance(report.get("stages"), list) or any(
                not isinstance(step, dict) or "name" not in step or "status" not in step for step in report["stages"]
            ):
                raise ValueError("Missing or invalid operation stages")
            action = report.get("action", "")
            if section == "code" and not action.startswith("archive-"):
                continue
            if section == "logs" and not action.startswith("logs-"):
                continue
            report["report"] = str(path)
            result.append(report)
        except (OSError, ValueError, TypeError, AttributeError) as error:
            result.append({"report": str(path), "status": "unreadable", "error": str(error)})
        if len(result) >= limit:
            break
    return result


def display(state):
    sync_observe.console_print(
        "Local synchronization status (no network queries; peer and upstream versions may be stale)"
    )
    if "code" in state:
        code = state["code"]
        sync_observe.console_print(f"\nCode: {code['path']} | target={code['target']}")
        for key in ("head", "packed", "confirmed"):
            sync_observe.console_print(f"  {key}: {code[key] or 'unknown / no record'}")
        if code["relative_to_peer"]:
            behind, ahead = code["relative_to_peer"].split()
            sync_observe.console_print(f"  Relative to known peer: behind={behind}, ahead={ahead}")
        working = "unavailable" if code["working_tree"] is None else (code["working_tree"] or "clean")
        sync_observe.console_print("  Working tree: " + working)
    if "logs" in state:
        logs = state["logs"]
        sync_observe.console_print(f"\nW&B source: {logs['source']} ({logs['source_runs']} run directories)")
        sync_observe.console_print(
            f"W&B destination: {logs['destination']} ({logs['destination_runs']} run directories)"
        )
        sync_observe.console_print("  Packing state: " + json.dumps(logs["packing_state"], ensure_ascii=False))
        sync_observe.console_print("  Cloud: " + logs["cloud_status"])
        queue = logs["upload_queue"]
        sync_observe.console_print(f"  Upload queue: {queue['status']} {queue.get('path', '')}")
        sync_observe.console_print("  Tasks: " + json.dumps(queue["counts"]))
        if queue["counts"].get("incomplete"):
            sync_observe.console_print(
                f"  WAITING {queue['counts']['incomplete']} snapshot(s) for newer source data; "
                "incomplete does not by itself mean an upload failure or prove training is still running"
            )
        for job in queue["jobs"]:
            if job["status"] not in ("succeeded", "superseded"):
                sync_observe.console_print(
                    f"    {job['run']}: {job['status']}, attempts={job['attempts']}, tree={job['tree'][:12]}"
                )
                if job.get("error"):
                    sync_observe.console_print(f"      {job['error']}")
        if logs["git"]:
            sync_observe.console_print(f"  Log Git HEAD: {logs['git']['head'] or 'unknown'}")
            sync_observe.console_print(f"  Log Git state: {logs['git']['branch_status'] or 'unknown'}")
            sync_observe.console_print(f"  Log Git changes: {logs['git']['working_tree'] or 'clean'}")
    sync_observe.console_print(f"\nRecent operations: {state['operation_dir']}")
    if not state["operations"]:
        sync_observe.console_print(
            "  No recorded operations. Earlier successes/failures cannot be reconstructed from these reports."
        )
    for report in state["operations"]:
        sync_observe.console_print(f"  {report.get('started_at', '?')} {report.get('action', '?')}: {report['status']}")
        if report["status"] == "running":
            sync_observe.console_print(
                f"    Running or interrupted without final report; last heartbeat {report.get('updated_at', '?')}"
            )
        details = report.get("details", {})
        if details.get("archive"):
            sync_observe.console_print(f"    Archive: {details['archive']}")
        summary = details.get("upload_summary", {})
        if summary.get("waiting_for_snapshot"):
            sync_observe.console_print(
                f"    WAITING {summary['waiting_for_snapshot']} snapshot(s) for newer source data"
            )
        for step in report.get("stages", []):
            if step["name"] in (
                "logs.merge",
                "logs.enqueue",
                "wandb.upload",
                "wandb.batch",
                "logs.git_commit",
                "logs.git_push",
            ):
                sync_observe.console_print(
                    f"    {step['name']}: {step['status']} {step.get('details', {}).get('run', '')}"
                )
        if report.get("action") == "logs-unpack" and report["status"] != "success":
            seen = {step["name"] for step in report.get("stages", [])}
            if "logs.git_push" not in seen:
                sync_observe.console_print("    Git push: not reached in this invocation (or disabled)")
        if report.get("failed_stage"):
            sync_observe.console_print(f"    Failed stage: {report['failed_stage']}")
        if report.get("error"):
            sync_observe.console_print(
                f"    {'Waiting' if report['status'] == 'partial' else 'Error'}: {report['error']}"
            )
        sync_observe.console_print(f"    Log: {report.get('log', '?')}")
        sync_observe.console_print(f"    Report: {report['report']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("all", "code", "logs"), default="all")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    snap = transfer.repo_path(os.environ.get("MIRROROPD_SNAP_DIR", "transfer_snap"))
    directory = Path(os.environ.get("MIRROROPD_OPERATION_DIR") or snap / "operations").expanduser().resolve()
    state = {"operation_dir": str(directory), "operations": reports(directory, args.section)}
    if args.section != "logs":
        state["code"] = code_status(root, os.environ.get("MIRROROPD_ARCHIVE_TARGET", "peer"))
    if args.section != "code":
        state["logs"] = logs_status(snap)
    state = sync_observe.scrub(state)
    if args.json:
        print(json.dumps(state, ensure_ascii=False, indent=2))
    else:
        display(state)


if __name__ == "__main__":
    main()
