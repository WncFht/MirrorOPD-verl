#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Embed equal-update statistics, every metric series and Plotly in offline HTML."""

# ruff: noqa: E402 -- Use the same palette as the audited static figures.
import argparse
import gzip
import json
import sys
from pathlib import Path

from plotly.offline import get_plotlyjs

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import fvlocate

fvlocate.ensure()
import fvstyle as fv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.data / "comparison.json.gz", "rt") as stream:
        bundle = json.load(stream)
    colors = dict(zip(bundle["analysis"]["runs"], fv.series_colors("paper", 3), strict=True))
    template = Path(__file__).with_name("equal_budget.html").read_text()
    document = template.replace("/*__PLOTLY__*/", get_plotlyjs())
    document = document.replace(
        "/*__DATA__*/", json.dumps(bundle, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    )
    document = document.replace("/*__COLORS__*/", json.dumps(colors))
    args.output.write_text(document)
    print(args.output, args.output.stat().st_size)


if __name__ == "__main__":
    main()
