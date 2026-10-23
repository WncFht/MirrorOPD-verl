# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Render a self-contained, network-free dashboard from collected measurements."""

import argparse
import json
from pathlib import Path

from plotly.offline import get_plotlyjs


def render(data, destination):
    template = Path(__file__).with_name("dashboard.html").read_text()
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    document = template.replace("/*__PLOTLY__*/", get_plotlyjs()).replace("/*__DATA__*/", payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(document)
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", type=Path, help="Embed reviewed measurements in the same offline HTML")
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    if args.review:
        data["review"] = json.loads(args.review.read_text())
    render(data, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
