# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Serve the state explorer with separate assets and a compressed read-only data endpoint."""

import gzip
import json
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_REPORT = (
    Path(__file__).resolve().parents[3] / "docs/experiments/teacher-transfer/checkpoint-dynamics/latent-state.html"
)


def serve(*, report=DEFAULT_REPORT, host="127.0.0.1", port=8766):
    """Load an existing report once; serve no model inference or repository listings."""
    report = Path(report).resolve(strict=True)
    assets = Path(__file__).parent / "viz"
    match = re.search(r'<script type="application/json" id="report-data">(.*?)</script>', report.read_text(), re.DOTALL)
    if match is None:
        raise ValueError(f"Report has no embedded state data: {report}")
    data = match.group(1).encode()
    json.loads(data)
    compressed = gzip.compress(data, compresslevel=6, mtime=0)
    styles = "\n".join(
        f'<link rel="stylesheet" href="assets/{name}">' for name in ("report.css", "scatter.css", "overview.css")
    )
    scripts = "\n".join(
        f'<script defer src="assets/{name}"></script>'
        for name in ("scatter.js", "trajectory.js", "occupancy.js", "overview.js", "report.js")
    )
    shell = (assets / "report.html").read_text().replace("<!--STYLE-->", styles)
    shell = shell.replace("<!--DATA-->", "").replace("<!--SCRIPT-->", scripts).encode()
    asset_names = {
        "overview.css",
        "report.css",
        "scatter.css",
        "scatter.js",
        "trajectory.js",
        "occupancy.js",
        "overview.js",
        "report.js",
    }

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            self.respond(False)

        def do_HEAD(self):
            self.respond(True)

        def respond(self, head):
            path = urlsplit(self.path).path
            encoding = None
            if path in ("/", "/index.html"):
                body, kind = shell, "text/html; charset=utf-8"
            elif path == "/api/report":
                body, kind = data, "application/json; charset=utf-8"
                if "gzip" in self.headers.get("Accept-Encoding", ""):
                    body, encoding = compressed, "gzip"
            elif path == "/healthz":
                body, kind = b'{"status":"ok"}', "application/json"
            elif path.startswith("/assets/") and path.removeprefix("/assets/") in asset_names:
                file = assets / path.removeprefix("/assets/")
                body, kind = file.read_bytes(), self.guess_type(str(file))
            elif path in ("/README.md", "/latent-state-pooled.html") or (
                path.startswith("/figures/") and Path(path).name.startswith("latent-state-")
            ):
                # Only the owning topic's explicit report documents and figures are public.
                file = (report.parent / path.lstrip("/")).resolve()
                if not file.is_relative_to(report.parent) or not file.is_file():
                    self.send_error(404)
                    return
                body, kind = file.read_bytes(), self.guess_type(str(file))
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            if path == "/api/report":
                self.send_header("Vary", "Accept-Encoding")
            if encoding:
                self.send_header("Content-Encoding", encoding)
            self.end_headers()
            if not head:
                self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), partial(Handler, directory=str(report.parent)))
    print(f"State explorer: http://{host}:{server.server_port}/", flush=True)
    print(f"Read-only report: {report}; JSON {len(data):,} bytes / gzip {len(compressed):,} bytes", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
