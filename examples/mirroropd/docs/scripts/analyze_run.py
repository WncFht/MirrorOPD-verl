#!/usr/bin/env python3
"""Prepare analysis snapshots from the local log archive by default.

The explicit export-cloud command retains read-only API access for audits or
runs absent from the archive. No command logs in, uploads, or starts training.
"""

import argparse
import base64
import gzip
import json
import math
import netrc
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median


def write_json(path, obj, allow_nan=False):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=allow_nan) + "\n")


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(s) for s in stream if s.strip()]


def write_rows(path, rows, allow_nan=False):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=allow_nan) + "\n")


def scrub(obj, nonfinite_to_none=True):
    if isinstance(obj, dict):
        return {
            k: "[REDACTED]"
            if re.search(r"api.?key|password|secret|credential|access.?token", k, re.I)
            else scrub(v, nonfinite_to_none)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [scrub(v, nonfinite_to_none) for v in obj]
    if nonfinite_to_none and isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def import_log(args):
    rows = {}
    for line in args.log.read_text().splitlines():
        match = re.search(r"\bstep:(\d+) - ", line)
        if not match:
            continue
        row = {"training/global_step": int(match[1])}
        for field in line[match.end() :].split(" - "):
            key, sep, value = field.partition(":")
            if not sep:
                continue
            value = re.sub(r"^np\.(?:float\d*|int\d*)\((.*)\)$", r"\1", value.strip())
            try:
                row[key] = float(value)
            except ValueError:
                continue
        rows[int(match[1])] = row
    if not rows:
        raise SystemExit("No step:N metric rows found")
    args.out.mkdir(parents=True, exist_ok=False)
    write_rows(args.out / "history.jsonl", [rows[k] for k in sorted(rows)])
    write_json(
        args.out / "source.json",
        {
            "kind": "user-provided-console-log",
            "run": args.run,
            "coverage": [min(rows), max(rows)],
            "rows": len(rows),
            "online_history_retrieved": False,
            "system_metrics_available": False,
            "note": "Historical excerpt; latest run state, actual run config and validation results are unavailable.",
        },
    )
    summarize(args.out, args.gpus, args.batch)


def export(args):
    """Delegate decoding to wandb-logs, then adapt its output for project analysis."""
    if args.out.exists():
        raise FileExistsError(f"Snapshot already exists: {args.out}")
    parts = args.run.split("/")
    if len(parts) != 3 or any(not part or part in (".", "..") for part in parts):
        raise SystemExit("--run must be entity/project/run_id")
    entity, project, run_id = parts
    sibling = Path(__file__).resolve().parents[4].parent / "wandb-logs"
    logs_repo = args.logs_repo or os.environ.get("WANDB_LOGS_REPO")
    logs_repo = (
        Path(logs_repo).expanduser()
        if logs_repo
        else (sibling if (sibling / ".git").exists() else Path.home() / "src/wandb-logs")
    )
    logs_repo = logs_repo.resolve()
    candidates = sorted((logs_repo / project / "wandb").glob(f"*/run-{run_id}.wandb"))
    candidates = [p for p in candidates if not p.parent.is_symlink() and not p.is_symlink()]
    if len(candidates) != 1:
        raise SystemExit(f"Expected one local run, found {len(candidates)} in {logs_repo}; no cloud fallback")
    raw_run = candidates[0].parent
    python = logs_repo / ".venv/bin/python"
    decoder = logs_repo / "tools/wandb_local.py"
    if not python.is_file() or not decoder.is_file():
        raise SystemExit(f"Initialize the local reader first: make -C {logs_repo} setup")
    subprocess.run([str(python), str(decoder), "decode", "--project", project, "--run", run_id], check=True)
    decoded = logs_repo / "derived" / project / raw_run.name
    record = json.loads((decoded / "run.json").read_text())
    identity = record["run"]
    if identity.get("run_id") != run_id or identity.get("project") != project:
        raise ValueError("Decoded run identity does not match --run")
    if identity.get("entity") and identity["entity"] != entity:
        raise ValueError("Decoded entity does not match --run")
    if record["status"] not in ("finished", "snapshot", "partial_snapshot"):
        raise ValueError(f"Cannot analyze decoder status {record['status']}")
    if not record["history_rows"]:
        raise ValueError("Local run has no decoded history records")
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(decoded / "history.jsonl", args.out / "history.jsonl")
    for name in ("config.json", "summary.json"):
        write_json(args.out / name, scrub(json.loads((decoded / name).read_text()), False), allow_nan=True)
    shutil.copyfile(decoded / "metric-definitions.jsonl", args.out / "metric-definitions.jsonl")
    system = []
    for row in read_rows(decoded / "system.jsonl.gz"):
        ns = timestamp_ns(row["timestamp"])
        system.append(
            {
                "_timestamp": ns / 1e9,
                "_timestamp_ns": ns,
                **{"system." + key: value for key, value in row["metrics"].items()},
            }
        )
    write_rows(args.out / "system.jsonl.gz", system, allow_nan=True)
    metadata_file = raw_run / "files/wandb-metadata.json"
    requirements_file = raw_run / "files/requirements.txt"
    metadata = json.loads(metadata_file.read_text()) if metadata_file.is_file() else {}
    environment_source = "wandb-metadata.json" if metadata else None
    environment_file = decoded / "environment.jsonl"
    if not metadata and environment_file.is_file():
        metadata = environment_records_metadata(read_rows(environment_file))
        environment_source = "local .wandb environment records" if metadata else None
    requirements = requirements_file.read_text() if requirements_file.is_file() else ""
    write_environment(args.out, metadata, requirements)
    write_json(
        args.out / "source.json",
        {
            "kind": "wandb-offline",
            "run": args.run,
            "identity_source": "explicit --run, checked against recorded project and run_id",
            "entity_recorded": identity.get("entity"),
            "raw_run": str(raw_run),
            "decoded_run": str(decoded),
            "decoder_status": record["status"],
            "exit_code": record["exit_code"],
            "rows": record["history_rows"],
            "step_range": record["step_range"],
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "history_method": "local .wandb records; sparse rows preserved without sampling",
            "online_history_retrieved": False,
            "system_metrics_available": bool(system),
            "system_rows": len(system),
            "system_api_sampled": False,
            "system_method": "all decoded stats records; timestamps retain integer nanoseconds",
            "hardware_metadata_available": bool(metadata),
            "environment_metadata_source": environment_source,
            "requirements_available": requirements_file.is_file(),
            "note": (
                "No exit or an unfinished tail does not prove a complete run; "
                "telemetry is periodic, not a continuous trace."
            ),
        },
    )
    print(
        f"Local snapshot: {args.out}; status={record['status']}, history={record['history_rows']}, system={len(system)}"
    )


def timestamp_ns(value):
    """Convert a protobuf UTC timestamp without dropping its nanosecond remainder."""
    if not value.endswith("Z"):
        raise ValueError("Expected a UTC protobuf timestamp")
    seconds, _, fractional = value[:-1].partition(".")
    epoch = int(datetime.strptime(seconds, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
    return epoch * 10**9 + int(fractional.ljust(9, "0"))


def environment_records_metadata(rows):
    """Combine partial records for one writer; never conflate different machines."""
    writers = {r.get("writer_id") for r in rows if r.get("writer_id")}
    if len(writers) > 1:
        raise ValueError("Multiple environment writers require explicit hardware attribution")
    merged = {}
    for row in rows:
        merged.update(row)
    for source, target in (("gpu_type", "gpu"), ("cuda_version", "cuda")):
        if source in merged:
            merged[target] = merged.pop(source)
    return merged


def write_environment(out, metadata, requirements):
    """Use the run's recorded environment only; empty inputs remain visibly absent."""
    selected = {
        k: metadata[k] for k in ("gpu", "gpu_count", "gpu_nvidia", "cuda", "python", "os", "git") if k in metadata
    }
    if "gpu_nvidia" in selected:
        selected["gpu_nvidia"] = [
            {k: v for k, v in gpu.items() if k in ("name", "memory_total", "cuda_cores", "architecture")}
            for gpu in selected["gpu_nvidia"]
        ]
    if "git" in selected:
        selected["git"] = {"commit": selected["git"].get("commit")}
    write_json(out / "hardware.json", scrub(selected))
    names = {"torch", "vllm", "transformers", "transfer-queue", "liger-kernel", "flash-attn", "triton", "wandb", "ray"}
    versions = [s for s in requirements.splitlines() if s.split("==")[0].lower().replace("_", "-") in names]
    (out / "package-versions.txt").write_text("\n".join(versions) + "\n")


def export_cloud(args):
    """Explicit cloud audit/recovery entry; never invoked as an automatic fallback."""
    import wandb

    kwargs = {"overrides": {"base_url": args.base_url}, "timeout": 30}
    if args.api_key_file:
        kwargs["api_key"] = args.api_key_file.read_text().strip()
    if args.netrc_file:
        from urllib.parse import urlparse

        credentials = netrc.netrc(args.netrc_file)
        auth = credentials.authenticators(urlparse(args.base_url).hostname) or credentials.authenticators(
            "api.wandb.ai"
        )
        if not auth:
            raise SystemExit("No matching W&B host in netrc file")
        kwargs["api_key"] = auth[2]
    api = wandb.Api(**kwargs)
    run = api.run(args.run)
    # Freeze a boundary; a running run can grow during this read-only snapshot.
    last_step = run.lastHistoryStep
    rows = list(run.scan_history(page_size=500, max_step=last_step + 1))
    if not rows:
        raise SystemExit("Run is accessible but no history rows were returned")
    args.out.mkdir(parents=True, exist_ok=False)
    write_rows(args.out / "history.jsonl", scrub(rows))
    write_json(args.out / "config.json", scrub(dict(run.config)))
    # summary_metrics is the JSON dict; summary wraps nested values as SDK objects.
    write_json(args.out / "summary.json", scrub(run.summary_metrics))
    source = {
        "kind": "wandb-api",
        "run": args.run,
        "state_at_fetch": run.state,
        "created_at": run.created_at,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "history_boundary_step": last_step,
        "rows": len(rows),
        "history_method": "scan_history(keys=None)",
        "online_history_retrieved": True,
    }
    try:
        system = run.history(stream="system", samples=args.system_samples, pandas=False)
        write_rows(args.out / "system-sampled.jsonl.gz", scrub(system))
        source["system_metrics_available"] = bool(system)
        source["system_method"] = f"history(stream=system,samples={args.system_samples}); completeness NOT guaranteed"
    except Exception as exc:
        source["system_metrics_available"] = False
        source["system_error_type"] = type(exc).__name__
    write_json(args.out / "source.json", source)
    try:
        metadata = json.loads(download_file(args.base_url, args.run, "wandb-metadata.json", api.api_key))
        selected = {
            k: metadata[k] for k in ("gpu", "gpu_count", "gpu_nvidia", "cuda", "python", "os", "git") if k in metadata
        }
        if "gpu_nvidia" in selected:
            selected["gpu_nvidia"] = [
                {k: v for k, v in gpu.items() if k in ("name", "memory_total", "cuda_cores", "architecture")}
                for gpu in selected["gpu_nvidia"]
            ]
        if "git" in selected:
            selected["git"] = {"commit": selected["git"].get("commit")}
        write_json(args.out / "hardware.json", scrub(selected))
    except Exception as exc:
        print("Hardware metadata unavailable:", type(exc).__name__)
    try:
        requirements = download_file(args.base_url, args.run, "requirements.txt", api.api_key).decode()
        names = {
            "torch",
            "vllm",
            "transformers",
            "transfer-queue",
            "liger-kernel",
            "flash-attn",
            "triton",
            "wandb",
            "ray",
        }
        versions = [s for s in requirements.splitlines() if s.split("==")[0].lower().replace("_", "-") in names]
        (args.out / "package-versions.txt").write_text("\n".join(versions) + "\n")
    except Exception as exc:
        print("Package metadata unavailable:", type(exc).__name__)
    print(f"Cloud snapshot: {args.out}; system history is API sampled")


def download_file(base_url, run_path, name, api_key):
    """Read a run file with the same explicit credential as the Public API."""
    from urllib.parse import quote
    from urllib.request import HTTPRedirectHandler, Request, build_opener

    class FileRedirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
            if redirected is not None:
                redirected.remove_header("Authorization")
            return redirected

    url = base_url.rstrip("/") + "/files/" + quote(run_path, safe="/") + "/" + quote(name)
    authorization = base64.b64encode(("api:" + api_key).encode()).decode()
    request = Request(url, headers={"Authorization": "Basic " + authorization, "User-Agent": "W&B Public Client"})
    with build_opener(FileRedirect()).open(request, timeout=30) as response:
        return response.read()


def is_number(value):
    """Accept finite observations; absent/null sparse history is not a measured zero."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def summarize(directory, gpus, batch):
    rows = read_rows(directory / "history.jsonl")
    train = [r for r in rows if is_number(r.get("timing_s/step")) and is_number(r.get("training/global_step"))]
    steady = [r for r in train if r["training/global_step"] > 1]
    if not steady:
        steady = train
    result = {"training_rows": len(train), "normalization_gpus": gpus, "prompts_per_step": batch}
    if train:
        result["last_observed_global_step"] = max(r["training/global_step"] for r in train)
        result["observed_training_seconds"] = sum(r["timing_s/step"] for r in train)
        result["observed_testing_seconds"] = sum(
            r["timing_s/testing"] for r in rows if is_number(r.get("timing_s/testing"))
        )
        total = result["observed_training_seconds"] + result["observed_testing_seconds"]
        result["testing_fraction_of_timed_work"] = result["observed_testing_seconds"] / total
    if steady:
        result["steady_steps"] = [r["training/global_step"] for r in steady]
        result["steady"] = {}
        for k in sorted({k for r in steady for k in r if not k.startswith(("val-", "_"))}):
            vals = [r[k] for r in steady if is_number(r.get(k))]
            if vals:
                result["steady"][k] = {"mean": mean(vals), "median": median(vals), "min": min(vals), "max": max(vals)}
        step = mean(r["timing_s/step"] for r in steady)
        # Only disjoint step components belong in this sum. rollout_batch overlaps gen;
        # testing is outside step. Optional checkpoint time is amortized across steps.
        timing_keys = [
            "gen",
            "teacher_forward",
            "teacher_transfer",
            "old_log_prob",
            "adv",
            "update_actor",
            "save_checkpoint",
            "update_weights",
        ]
        result["stage_seconds"] = {
            k: sum(r["timing_s/" + k] for r in steady if is_number(r.get("timing_s/" + k))) / len(steady)
            for k in timing_keys
            if any(is_number(r.get("timing_s/" + k)) for r in steady)
        }
        result["stage_seconds"]["other"] = step - sum(result["stage_seconds"].values())
        result["stage_percent"] = {k: v / step * 100 for k, v in result["stage_seconds"].items()}
        result["aggregate_valid_tokens_per_second"] = sum(r.get("perf/total_num_tokens", 0) for r in steady) / sum(
            r["timing_s/step"] for r in steady
        )
        result["normalized_valid_tokens_per_second"] = result["aggregate_valid_tokens_per_second"] / gpus
        result["prompts_per_minute"] = batch / step * 60
    for key in ["response_length/clip_ratio", "critic/score/mean"]:
        vals = [r[key] for r in train if is_number(r.get(key))]
        if vals:
            result["pooled_" + key] = mean(vals)  # equal global batch size
    write_json(directory / "analysis.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "steady"}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("import-log", "export", "export-cloud", "summarize"):
        p = sub.add_parser(command)
        p.add_argument(
            "--out", type=Path, required=True, help="new snapshot directory, or existing directory for summarize"
        )
        p.add_argument("--gpus", type=int, default=8)
        p.add_argument("--batch", type=int, default=64)
        if command != "summarize":
            p.add_argument(
                "--run",
                required=command in ("export", "export-cloud"),
                help="explicit entity/project/run for export; omit for console logs with unknown run identity",
            )
        if command == "import-log":
            p.add_argument("--log", type=Path, required=True)
        if command == "export":
            p.add_argument("--logs-repo", type=Path, help="local wandb-logs checkout (otherwise auto-detected)")
        if command == "export-cloud":
            auth = p.add_mutually_exclusive_group()
            auth.add_argument("--api-key-file", type=Path)
            auth.add_argument("--netrc-file", type=Path)
            p.add_argument("--base-url", default="https://api.forge.coreweave.com")
            p.add_argument("--system-samples", type=int, default=100000)
    args = parser.parse_args()
    if args.gpus <= 0 or args.batch <= 0:
        parser.error("gpus and batch must be positive")
    if args.command == "summarize":
        summarize(args.out, args.gpus, args.batch)
    elif args.command == "import-log":
        import_log(args)
    elif args.command == "export-cloud":
        export_cloud(args)
    else:
        export(args)


if __name__ == "__main__":
    main()
