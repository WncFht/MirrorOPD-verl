#!/usr/bin/env python3
"""Export read-only W&B snapshots or extract numeric metrics from a console log.

No login/sync/upload occurs. Training history uses scan_history without a keys
intersection; system history is explicitly a sampled SDK export. Credentials
come from the SDK's usual environment/netrc or --api-key-file, never output.
"""

import argparse
import base64
import gzip
import json
import math
import netrc
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(s) for s in stream if s.strip()]


def write_rows(path, rows):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def scrub(obj):
    if isinstance(obj, dict):
        return {
            k: "[REDACTED]" if re.search(r"api.?key|password|secret|credential|access.?token", k, re.I) else scrub(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
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
    write_json(args.out / "summary.json", scrub(dict(run.summary)))
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
    summarize(args.out, args.gpus, args.batch)


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


def summarize(directory, gpus, batch):
    rows = read_rows(directory / "history.jsonl")
    train = [r for r in rows if "timing_s/step" in r and "training/global_step" in r]
    steady = [r for r in train if r["training/global_step"] > 1]
    if not steady:
        steady = train
    result = {"training_rows": len(train), "normalization_gpus": gpus, "prompts_per_step": batch}
    if train:
        result["last_observed_global_step"] = max(r["training/global_step"] for r in train)
        result["observed_training_seconds"] = sum(r["timing_s/step"] for r in train)
        result["observed_testing_seconds"] = sum(r.get("timing_s/testing", 0) for r in rows)
        total = result["observed_training_seconds"] + result["observed_testing_seconds"]
        result["testing_fraction_of_timed_work"] = result["observed_testing_seconds"] / total
    if steady:
        result["steady_steps"] = [r["training/global_step"] for r in steady]
        result["steady"] = {}
        for k in sorted({k for r in steady for k in r if not k.startswith(("val-", "_"))}):
            vals = [r[k] for r in steady if isinstance(r.get(k), (int, float)) and math.isfinite(r[k])]
            if vals:
                result["steady"][k] = {"mean": mean(vals), "median": median(vals), "min": min(vals), "max": max(vals)}
        step = mean(r["timing_s/step"] for r in steady)
        timing_keys = ["gen", "old_log_prob", "adv", "update_actor", "update_weights"]
        result["stage_seconds"] = {k: mean(r.get("timing_s/" + k, 0) for r in steady) for k in timing_keys}
        result["stage_seconds"]["other"] = step - sum(result["stage_seconds"].values())
        result["stage_percent"] = {k: v / step * 100 for k, v in result["stage_seconds"].items()}
        result["aggregate_valid_tokens_per_second"] = sum(r.get("perf/total_num_tokens", 0) for r in steady) / sum(
            r["timing_s/step"] for r in steady
        )
        result["normalized_valid_tokens_per_second"] = result["aggregate_valid_tokens_per_second"] / gpus
        result["prompts_per_minute"] = batch / step * 60
    for key in ["response_length/clip_ratio", "critic/score/mean"]:
        vals = [r[key] for r in train if key in r]
        if vals:
            result["pooled_" + key] = mean(vals)  # equal global batch size
    write_json(directory / "analysis.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "steady"}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("import-log", "export", "summarize"):
        p = sub.add_parser(command)
        p.add_argument(
            "--out", type=Path, required=True, help="new snapshot directory, or existing directory for summarize"
        )
        p.add_argument("--gpus", type=int, default=8)
        p.add_argument("--batch", type=int, default=64)
        if command != "summarize":
            p.add_argument("--run", default="wncfht/MirrorOPD-verl/b38hw8v0")
        if command == "import-log":
            p.add_argument("--log", type=Path, required=True)
        if command == "export":
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
    else:
        export(args)


if __name__ == "__main__":
    main()
