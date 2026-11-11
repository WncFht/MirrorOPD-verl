# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CLI for freezing and auditing input record banks before GPU diagnostics."""

import argparse
import json
from pathlib import Path

from .inputs import load_records, normalize_records, source_fingerprint, validate_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--bank-id", required=True)
    parser.add_argument("--cap", type=int)
    parser.add_argument("--expected", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = sum((load_records(path) for path in args.records), [])
    rows = normalize_records(raw, cap=args.cap, bank_id=args.bank_id)
    counts = validate_records(rows, expected=args.expected)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "bank_id": args.bank_id,
                "source": [str(p) for p in args.records],
                "fingerprint": source_fingerprint(rows),
                **counts,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"finished": True, **counts}, indent=2))


if __name__ == "__main__":
    main()
