# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only live and historical training curves, loaded only when selected."""

import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


def read_records(path):
    records, malformed = [], 0
    if not path.exists():
        return records, malformed
    lines = path.read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except ValueError:
            if index != len(lines) - 1 or line.endswith("\n"):
                malformed += 1
            continue
        values = {
            k: v
            for k, v in row.get("data", {}).items()
            if isinstance(v, int | float)
            and math.isfinite(v)
            and (
                k.startswith(("step/", "optim/actor/", "rollout/data_reuse/"))
                or (k.startswith("eval/rkl/") and k.endswith(("/question_mean", "/invalid_rows", "/valid_rows")))
            )
        }
        records.append(dict(step=row.get("step"), data=values))
    return records, malformed


class Dashboard:
    def __init__(self, root, catalog=None):
        self.root, self.catalog = root, catalog
        self.cache = {}
        self.lock = threading.Lock()

    def runs(self):
        plan = json.loads((self.root / "plan.json").read_text())
        runs = []
        for lane in plan["lanes"]:
            study, name = self.root / lane["study"], lane["experiment"]
            runs.append(
                dict(
                    id=name,
                    name=name,
                    pair="E102",
                    optimizer="SGD",
                    lr=lane["lr"],
                    weight_decay=0,
                    budget=plan["outer"],
                    group="当前 SGD 对照",
                    default=True,
                    student="Qwen3-1.7B",
                    teacher="Qwen3-4B-Instruct-2507",
                    metrics=str(study / "training-sgd" / name / "metrics.jsonl"),
                    status=str(study / "status.json"),
                    note="正式训练；含独立开发集",
                    **{"reuse": plan.get("reuse", 16)},
                )
            )
        if self.catalog:
            runs.extend(json.loads(self.catalog.read_text())["runs"])
        ids = [r["id"] for r in runs]
        if len(set(ids)) != len(ids):
            raise ValueError("Dashboard run IDs must be unique")
        return runs

    def catalog_data(self):
        runs = []
        for run in self.runs():
            path = Path(run["metrics"])
            status = {}
            if run.get("status") and Path(run["status"]).exists():
                status = json.loads(Path(run["status"]).read_text())
            if run.get("completion") and Path(run["completion"]).is_file():
                status = {"state": "complete"}
            runs.append(
                dict(
                    run,
                    state=status.get("state", "archived"),
                    error=status.get("error"),
                    available=path.is_file(),
                    age_seconds=time.time() - path.stat().st_mtime if path.exists() else None,
                )
            )
        return dict(updated_at=time.time(), runs=runs)

    def metrics(self, ids):
        selected = [run for run in self.runs() if run["id"] in ids]
        if set(ids) - {r["id"] for r in selected}:
            raise ValueError("Unknown run ID")
        result = []
        with self.lock:
            for run in selected:
                path = Path(run["metrics"])
                stat = path.stat() if path.exists() else None
                stamp = (stat.st_mtime_ns, stat.st_size) if stat else None
                cached = self.cache.get(str(path))
                if not cached or cached[0] != stamp:
                    records, malformed = read_records(path)
                    cached = (stamp, records, malformed)
                    self.cache[str(path)] = cached
                result.append(dict(id=run["id"], records=cached[1], malformed_records=cached[2]))
        return dict(updated_at=time.time(), runs=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--catalog", type=Path, help="Explicit historical run metadata; never scans source snapshots")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    assets = Path(__file__).parent
    dashboard = Dashboard(args.root.resolve(), args.catalog)
    dashboard.catalog_data()
    static = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        "/style.css": ("style.css", "text/css; charset=utf-8"),
        "/echarts.min.js": ("vendor/echarts.min.js", "text/javascript; charset=utf-8"),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            try:
                if url.path in static:
                    filename, content_type = static[url.path]
                    body = (assets / filename).read_bytes()
                elif url.path == "/api/runs":
                    body = json.dumps(dashboard.catalog_data(), allow_nan=False).encode()
                    content_type = "application/json"
                elif url.path == "/api/metrics":
                    ids = parse_qs(url.query).get("id", [])
                    body = json.dumps(dashboard.metrics(ids), allow_nan=False).encode()
                    content_type = "application/json"
                else:
                    self.send_error(404)
                    return
            except (OSError, ValueError, KeyError) as error:
                self.send_error(503, str(error))
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Dashboard listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
