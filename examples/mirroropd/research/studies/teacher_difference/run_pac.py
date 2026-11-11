# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Run P/A/C on one frozen JSON/JSONL bank with local HF checkpoints."""

import argparse
import json
from pathlib import Path

import torch

from .inputs import load_records, normalize_records, source_fingerprint, validate_records
from .metrics import pac_summary
from .pac import compute_batch
from .runtime import load_model, teacher_cache, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--bank-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    raw = load_records(args.records)
    rows = normalize_records(raw, cap=args.cap, bank_id=args.bank_id)
    if args.limit is not None:
        rows = rows[: args.limit]
    audit = validate_records(rows)
    cached = teacher_cache(args.out.parent / "cache", args.teacher, rows, device=device)
    student = load_model(args.student, checkpointing=device.type == "cuda", device=device)
    result = compute_batch(
        student,
        cached["head"].to(device),
        cached["hidden"],
        rows,
        cap=args.cap,
        retain_samples=False,
    )
    summary = pac_summary(
        pressure=result["pressure"],
        gradient_vectors=result["gradient_vectors"],
        gradient_norm_sq=result["gradient_norm_sq"],
    )
    output = {
        "finished": True,
        "student": str(args.student),
        "teacher": str(args.teacher),
        "bank_id": args.bank_id,
        "source": str(args.records),
        "source_fingerprint": source_fingerprint(rows),
        "audit": audit,
        "summary": summary,
        "records": [
            {
                "row_id": row["row_id"],
                "pressure": float(result["pressure"][i]),
                "gradient_norm_sq": float(result["gradient_norm_sq"][i]),
                "loss": result["losses"][i],
                "domain": row.get("domain"),
                "source": row.get("source"),
            }
            for i, row in enumerate(rows)
        ],
        "loss_mean": sum(result["losses"]) / len(result["losses"]),
        "parameter_count": result["parameter_count"],
        "dtype": {"student": "float32", "teacher": "bfloat16" if device.type == "cuda" else "float32"},
    }
    write_json(args.out, output)
    print(json.dumps(output, indent=3))


if __name__ == "__main__":
    main()
