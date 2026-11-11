# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Aggregate completed exact gradients without conflating layers and states."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def partition(key):
    return int(key.split(".")[2])


def aggregate(rows, vector_rows, keys):
    by_site = defaultdict(list)
    for row in rows:
        if row["key"] in keys:
            by_site[row["row_id"], row["position"]].append(row)
    denominator = sum(math.sqrt(sum(r["gradient_norm"] ** 2 for r in site)) for site in by_site.values())
    numerator = math.sqrt(sum(r["sum_gradient_norm"] ** 2 for r in vector_rows if r["key"] in keys))
    bands = {}
    for band in ["learned", "off", "null"]:
        effects = [sum(r["directions"][band]["dot"] for r in site) for site in by_site.values()]
        norm2 = sum(r["directions"][band]["norm"] ** 2 for r in next(iter(by_site.values())))
        signed = sum(effects)
        absolute = sum(abs(x) for x in effects)
        bands[band] = dict(
            summed_derivative=signed,
            sum_abs_site_derivative=absolute,
            state_scalar_coherence=abs(signed) / absolute if absolute else None,
            negative_site_fraction=sum(x < 0 for x in effects) / len(effects),
            direction_norm=math.sqrt(norm2),
            descent_direction_cosine=-signed / (numerator * math.sqrt(norm2)) if numerator * norm2 else None,
        )
    return dict(
        sites=len(by_site),
        matrices=len(keys),
        summed_gradient_norm=numerator,
        sum_site_gradient_norm=denominator,
        vector_coherence=numerator / denominator if denominator else None,
        directions=bands,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run_root / "manifest.json").read_text())
    summaries, coverage, sources = [], [], []
    for job in manifest["jobs"]:
        folder = args.run_root / "jobs" / f"{job['job_id']:03d}" / "attempt-001/gradients"
        for stage in job["gradient_stages"]:
            path = folder / stage
            complete = (path / "complete.json").exists()
            coverage.append(dict(job_id=job["job_id"], stage=stage, complete=complete))
            if not complete:
                continue
            rawfile = path / "per-site-projections.json"
            vectorfile = path / "vector-coherence.json"
            for source in [rawfile, vectorfile, path / "complete.json"]:
                sources.append(dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
            raw = json.loads(rawfile.read_text())
            vectors = json.loads(vectorfile.read_text())
            groups = defaultdict(list)
            for row in raw:
                groups[row["origin"], row["split"], row["domain"]].append(row)
            for (origin, split, domain), rows in groups.items():
                group = f"{origin}:{split}:{domain}"
                vector_rows = [r for r in vectors if r["group"] == group]
                keys = {r["key"] for r in rows}
                if len(vector_rows) != len(keys):
                    raise ValueError("Missing or duplicate matrix gradient summaries")
                identity = dict(
                    job_id=job["job_id"], run_id=job["run_id"], stage=stage, origin=origin, split=split, domain=domain
                )
                buckets = [("all", keys)]
                for layer in sorted({partition(k) for k in keys}):
                    buckets.append((f"layer-{layer}", {k for k in keys if partition(k) == layer}))
                for module in sorted({k.split(".")[-2] for k in keys}):
                    buckets.append((module, {k for k in keys if k.split(".")[-2] == module}))
                for scope, subset in buckets:
                    result = aggregate(rows, vector_rows, subset)
                    if result["vector_coherence"] is not None and result["vector_coherence"] > 1.00001:
                        raise ValueError("Triangle inequality violated by gradient identity")
                    summaries.append(identity | dict(scope=scope) | result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                schema=1,
                coverage=coverage,
                sources=sources,
                rows=summaries,
                definitions=dict(
                    vector_coherence="norm(sum_i gradient_i)/sum_i norm(gradient_i), disjoint matrices concatenated",
                    state_scalar_coherence="abs(sum_i <g_i,d>)/sum_i abs(<g_i,d>), fixed combined direction",
                    descent_direction_cosine="cos(-sum_i gradient_i, fixed direction)",
                    state_source="newly generated initial-student/teacher prefixes; not historical training rollout",
                ),
            ),
            indent=2,
        )
    )
    print(json.dumps(dict(complete_stages=sum(x["complete"] for x in coverage), rows=len(summaries))))


if __name__ == "__main__":
    main()
