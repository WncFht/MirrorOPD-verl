# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU admission of registered absorption pairs; never reserve GPUs or launch training."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from examples.mirroropd.assets.paths import available_model
from examples.mirroropd.launchers.absorption import PAIRS, STOP_TOKEN_IDS, template_kwargs


def messages(row):
    value = row["prompt"]
    return json.loads(value) if isinstance(value, str) else value


def check_questions(student, teacher, contract, tables, template):
    """Reject overlap, over-budget prefixes and changes to the shared token sequence."""
    summary = {}
    normalized = {}
    for name, rows in tables.items():
        lengths = []
        normalized[name] = set()
        for row in rows:
            prompt = messages(row)
            normalized[name].add(" ".join(" ".join(m["content"].split()).casefold() for m in prompt))
            text = student.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, **template)
            ids = student.encode(text, add_special_tokens=False)
            if teacher.encode(text, add_special_tokens=False) != ids:
                raise ValueError(f"Shared-prefix encoding differs: {name}/{row.get('id')}")
            if len(ids) > 4096:
                raise ValueError(f"Overlong prompt: {name}/{row.get('id')}: {len(ids)}")
            contract.validate_prefix(ids)
            lengths.append(len(ids))
        if not lengths:
            raise ValueError(f"Empty question table: {name}")
        summary[name] = dict(rows=len(rows), max_prompt_tokens=max(lengths), shared_prefix_encoding_mismatches=0)
    if any(normalized["train"] & values for name, values in normalized.items() if name != "train"):
        raise ValueError("Training and heldout prompt overlap")
    return summary


def audit_pair(row, tables):
    from transformers import AutoTokenizer

    from verl.trainer.distillation.compatibility import inspect_pair

    number, student, teacher, family, concurrency, _ = row
    source, target = map(available_model, (student, teacher))
    stops = STOP_TOKEN_IDS.get(number, ())
    contract = inspect_pair(str(source), str(target), stop_token_ids=stops, trust_remote_code=family == "qwen25")
    if min(contract.student_context, contract.teacher_context) < 4096 + 7168:
        raise ValueError("Pair cannot support the declared prompt and response budgets")
    tokenizers = [
        AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=family == "qwen25")
        for path in (source, target)
    ]
    if not set(tokenizers[1].get_vocab()) <= set(tokenizers[0].get_vocab()):
        raise ValueError("Teacher generation support is not closed under student input")
    return dict(
        experiment=f"E{number}",
        student=student,
        teacher=teacher,
        status="cpu_accepted",
        training_domain="DAPO Math",
        concurrency=concurrency,
        student_asset=str(source),
        teacher_asset=str(target),
        contract=contract.metadata(),
        student_eos=tokenizers[0].eos_token_id,
        teacher_eos=tokenizers[1].eos_token_id,
        stop_token_ids=list(stops),
        teacher_generation_closed=True,
        prompt_budgets=check_questions(*tokenizers, contract, tables, template_kwargs(row)),
    )


def main():
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", choices=[f"E{r[0]}" for r in PAIRS], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Acceptance receipts are immutable; select a new output")
    if len(args.experiments) != len(set(args.experiments)):
        parser.error("Experiment IDs must be unique")
    tables = {
        name: pq.read_table(args.questions / f"{name}.parquet").to_pylist() for name in ("train", "math", "code", "if")
    }
    if len(tables["train"]) < 2560:
        raise ValueError("40 outer require at least 2560 fresh DAPO prompts")
    tables["train"] = tables["train"][:2560]
    if any(len(tables[name]) != count for name, count in (("math", 128), ("code", 64), ("if", 64))):
        raise ValueError("Heldout panels must contain Math128, Code64, IF64 questions")
    results = []
    for row in PAIRS:
        if f"E{row[0]}" not in args.experiments:
            continue
        try:
            result = audit_pair(row, tables)
        except Exception as error:
            result = dict(experiment=f"E{row[0]}", status="blocked", error=repr(error))
        results.append(result)
        print(result["experiment"], result["status"], result.get("error", ""), flush=True)
    from verl.trainer.evaluation.artifacts import file_sha256

    receipt = dict(
        schema=1,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        training_launched=False,
        gpu_validation=False,
        pairs=results,
        question_files={name: file_sha256(args.questions / f"{name}.parquet") for name in tables},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if any(r["status"] != "cpu_accepted" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
