#!/usr/bin/env python3
"""Embed all measurements and Plotly in one standalone offline HTML report."""

import argparse
import gzip
import json
from pathlib import Path

from plotly.offline import get_plotlyjs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--references", type=Path, help="Optional explicitly labeled historical reference data")
    args = parser.parse_args()
    with gzip.open(args.data / "comparison.json.gz", "rt") as stream:
        bundle = json.load(stream)
    bundle["config_differences"] = json.loads((args.data / "config-differences.json").read_text())
    if args.references:
        bundle["references"] = json.loads(args.references.read_text())
    template = Path(__file__).with_name("policy_comparison.html").read_text()
    payload = json.dumps(bundle, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    document = template.replace("/*__PLOTLY__*/", get_plotlyjs()).replace("/*__DATA__*/", payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document)
    print(args.output, args.output.stat().st_size)


if __name__ == "__main__":
    main()
