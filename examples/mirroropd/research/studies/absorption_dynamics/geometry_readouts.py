# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU readout of immutable geometry records; preserve coverage and pairing."""

import argparse
import gzip
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from examples.mirroropd.launchers.absorption import PAIRS
from examples.mirroropd.research.core.io import dump

from .geometry_figures import render
from .geometry_probes import angle_distribution, derive
from .sigma_variation import measure as measure_sigma


def compact(value):
    """Keep scalar probes and energy CDFs, without duplicating frames or raw spectra."""
    if isinstance(value, list):
        return [compact(x) for x in value]
    if not isinstance(value, dict):
        return value
    result = {k: compact(v) for k, v in value.items() if k not in ("singular_values", "angles_deg")}
    if "angles_deg" in value:
        result["angle_distribution"] = angle_distribution(value["angles_deg"])
    if "singular_values" in value:
        energy = np.square(value["singular_values"])
        ranks = np.unique(np.rint(np.linspace(1, len(energy), 65)).astype(int))
        result["spectrum_cdf"] = dict(
            ranks=ranks.tolist(), energy_fraction=(np.cumsum(energy)[ranks - 1] / energy.sum()).tolist()
        )
    return result


def collect(source, destination, extra_sources=()):
    """Export a self-contained scientific readout; partial coverage stays explicit."""
    sources = (source, *extra_sources)
    if len({Path(p).resolve() for p in sources}) != len(sources):
        raise ValueError("Geometry sources must be unique")
    protocol = dict(tasks=[], pairs={})
    for directory in sources:
        part = json.loads((directory / "protocol.json").read_text())
        if protocol["pairs"].keys() & part["pairs"].keys():
            raise ValueError("Geometry sources must have disjoint pair identities")
        protocol["pairs"].update(part["pairs"])
        protocol["tasks"].extend(part["tasks"])
    plan = {(t["condition"], t["tensor"]): t for t in protocol["tasks"]}
    labels = {f"E{r[0]}": dict(student=r[1], teacher=r[2]) for r in PAIRS}
    result = dict(
        schema="absorption-geometry-readout-v3",
        captured_at=time.time(),
        pairs={},
        failed_tasks=sum(len(list(p.glob("error-*.json"))) for p in sources),
        measurement_exit={
            str(p): json.loads((p / "exit.json").read_text()) if (p / "exit.json").exists() else None for p in sources
        },
    )
    for name, meta in protocol["pairs"].items():
        result["pairs"][name] = dict(
            **labels[name],
            expected=meta["matrix_count"],
            matrices=[],
            risks=meta["risks"],
            sources={},
            teacher_parameter_comparison=meta["teacher_parameter_comparison"],
        )
    for path in sorted(path for directory in sources for path in directory.glob("matrix-*.json.gz")):
        raw = path.read_bytes()
        record = json.loads(gzip.decompress(raw))
        task = record.pop("task")
        name, tensor = task["condition"], task["tensor"]
        if task != plan[name, tensor]:
            raise ValueError("Output tensor identity differs from accepted plan")
        if [s["step"] for s in record["stages"]] != [p[0] for p in task["points"]]:
            raise ValueError("Missing checkpoint stage")
        matrix = compact(record)
        if "initial_singular_values" in record:
            matrix["sigma_variation"] = measure_sigma(record)
            matrix["spectral_shape"] = derive(matrix["sigma_variation"])
        matrix.update(tensor=tensor, layer=int(tensor.split(".")[2]), module=tensor.split(".")[-2])
        pair = result["pairs"][name]
        if any(m["tensor"] == tensor for m in pair["matrices"]):
            raise ValueError("Duplicate tensor output")
        pair["matrices"].append(matrix)
        pair["sources"][path.name] = hashlib.sha256(raw).hexdigest()
    for pair in result["pairs"].values():
        pair["complete"] = len(pair["matrices"]) == pair["expected"]
        counts = defaultdict(int)
        for m in pair["matrices"]:
            counts[m["layer"]] += 1
        pair["complete_layers"] = sorted(k for k, n in counts.items() if n == 7)
    destination.mkdir(parents=True, exist_ok=True)
    data = gzip.compress(json.dumps(result, allow_nan=False, separators=(",", ":")).encode(), compresslevel=3, mtime=0)
    tmp = destination / "readout.json.tmp"
    tmp.write_bytes(data)
    tmp.replace(destination / "readout.json.gz")
    dump(
        destination / "coverage.json",
        {
            n: dict(
                complete=p["complete"],
                measured=len(p["matrices"]),
                expected=p["expected"],
                complete_layers=p["complete_layers"],
            )
            for n, p in result["pairs"].items()
        },
    )
    return result


def publish(data, figures, report=None):
    """One publication path for live collection and exported-data replotting."""
    from .frame_figures import render as render_frames
    from .overview_figures import render as render_overview
    from .sigma_figures import render as render_sigma
    from .singular_figures import render as render_singular

    figures.mkdir(parents=True, exist_ok=True)
    # Completed pairs are immutable: avoid replotting them every publication tick.
    code_digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        code_digest.update(path.read_bytes())
    cache_path = figures / "publication.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    states = {}
    selected = {}
    for name, pair in data["pairs"].items():
        fingerprint = hashlib.sha256(json.dumps(pair["sources"], sort_keys=True).encode()).hexdigest()
        states[name] = code_digest.hexdigest() + fingerprint
        if cache.get("pairs", {}).get(name) != states[name]:
            selected[name] = pair
    changed = dict(data, pairs=selected)
    render(changed, figures)
    render_sigma(changed, figures)
    render_frames(changed, figures)
    render_singular(changed, figures)
    render_overview(data, figures)
    dump(cache_path, dict(schema="geometry-publication-v1", at=time.time(), pairs=states))
    if report:
        from .geometry_report import write_report

        write_report(data, report, figures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source", type=Path)
    inputs.add_argument("--readout", type=Path)
    parser.add_argument(
        "--extra-source", type=Path, action="append", default=[], help="Additional disjoint pair outputs"
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=60, help="CPU publication polling interval in seconds")
    parser.add_argument("--report", type=Path, help="Independent report directory, updated with each readout")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.readout:
        if args.extra_source:
            parser.error("--extra-source requires --source")
        if args.watch:
            parser.error("--watch requires --source")
        data = json.loads(gzip.decompress(args.readout.read_bytes()))
        publish(data, args.figures, args.report)
        return
    if args.data is None:
        parser.error("--source requires --data")
    sources = (args.source, *args.extra_source)
    last_count = -1
    last_exit = None
    while True:
        count = sum(len(list(p.glob("matrix-*.json.gz"))) for p in sources)
        exit_state = tuple((p / "exit.json").read_bytes() if (p / "exit.json").exists() else None for p in sources)
        if count != last_count or exit_state != last_exit:
            data = collect(args.source, args.data, args.extra_source)
            publish(data, args.figures, args.report)
            last_count = count
            last_exit = exit_state
        if not args.watch or all(e is not None for e in exit_state):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
