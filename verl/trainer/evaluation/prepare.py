# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepare frozen question tables from explicit pinned local dataset inputs."""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import ArtifactStore, file_sha256
from .selection import fingerprints, source_split, stratified_sample

QUESTION = pa.schema(
    [
        ("id", pa.string()),
        ("dataset", pa.string()),
        ("domain", pa.string()),
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("ground_truth", pa.string()),
        ("stratum", pa.string()),
        ("family", pa.string()),
    ]
)


def question(dataset, domain, raw, *, ground_truth="", prompt=None, stratum=None):
    fingerprint, family = fingerprints(raw)
    return dict(
        id=fingerprint,
        dataset=dataset,
        domain=domain,
        family=family,
        stratum=stratum or dataset,
        prompt=prompt or [{"role": "user", "content": raw}],
        ground_truth=ground_truth if isinstance(ground_truth, str) else json.dumps(ground_truth),
    )


def iter_input(path):
    path = Path(path)
    if path.suffix == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            yield from batch.to_pylist()
    else:
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def adapt_input(kind, name, path):
    for index, row in enumerate(iter_input(path)):
        if kind == "math":
            prompt = row["prompt"]
            raw = "\n".join(m["content"] for m in prompt if m["role"] == "user")
            # All local math sources share this known response-format suffix.
            for suffix in (
                "Let's think step by step and output the final answer within \\boxed{}.",
                "Please reason step by step, and put your final answer within \\boxed{}.",
            ):
                raw = raw.removesuffix(suffix).strip()
            yield question(name, "math", raw, prompt=prompt, ground_truth=row["reward_model"]["ground_truth"])
        elif kind == "code":
            text = row["problem"]
            starter = row.get("starter_code") or ""
            prompt = text + ("\n\nUse this starter code:\n```python\n" + starter + "\n```" if starter else "")
            prompt += "\n\nReturn your solution in a Python code block."
            truth = {"tests_ref": {"path": str(Path(path).resolve()), "row": index}}
            yield question(name, "code", text, prompt=[{"role": "user", "content": prompt}], ground_truth=truth)
        elif kind == "gpqa":
            choices = dict(zip(row["choices"]["label"], row["choices"]["text"], strict=True))
            if set(choices) != set("ABCD") or row["answer"] not in choices:
                raise ValueError("Malformed GPQA choices")
            prompt = row["problem"] + "\n" + "\n".join(f"{k}. {choices[k]}" for k in "ABCD")
            prompt += "\n\nGive your final answer as a single letter A, B, C, or D."
            yield question(
                name,
                "science",
                row["problem"],
                ground_truth=row["answer"],
                prompt=[{"role": "user", "content": prompt}],
            )
        elif kind == "ifeval":
            truth = {k: row[k] for k in ("key", "instruction_id_list", "prompt", "kwargs")}
            yield question(
                name,
                "if",
                row["prompt"],
                ground_truth=truth,
                stratum=f"{len(row['instruction_id_list'])}:{row['instruction_id_list'][0].split(':')[0]}",
            )
        elif kind == "textbook":
            if row["subject"].lower() not in {"math", "cs"}:
                yield question(name, "science", row["question"], stratum=row["subject"])
        elif kind == "if_train":
            # Source training uses RKL only; its constraints are not the public IFEval label set.
            yield question(name, "if", row["prompt"], stratum=str(len(row["instruction_id_list"])))
        else:
            raise ValueError(f"Unsupported raw input adapter: {kind}")


def prepare(
    inputs,
    destination,
    *,
    train_count=6400,
    panel_count=128,
    seed=42,
    prompt_tokens=None,
    source_prompt_limit=2048,
    public_prompt_limit=8192,
):
    """inputs maps public/training to explicit name/kind/path/n records."""
    store = ArtifactStore.create(
        destination,
        kind="question_sets",
        metadata={
            "inputs": inputs,
            "seed": seed,
            "selection": "proportional_largest_remainder_hash_priority",
            "decontamination": "normalized_text_numeric_families; no semantic-paraphrase claim",
        },
    )
    public, datasets = [], []
    domains = {}
    for entry in inputs["public"]:
        rows = [
            r
            for path in entry.get("paths", [entry.get("path")])
            for r in adapt_input(entry["kind"], entry["name"], path)
        ]
        if len(rows) != entry["count"]:
            raise ValueError(f"Public benchmark {entry['name']} has unexpected coverage")
        if prompt_tokens is not None and any(prompt_tokens(r["prompt"]) > public_prompt_limit for r in rows):
            raise ValueError(f"Public benchmark {entry['name']} exceeds prompt budget; change budget, not coverage")
        public.extend(rows)
        domains.setdefault(rows[0]["domain"], []).extend(rows)
        name = f"public/{entry['name']}.parquet"
        store.write_rows(name, rows, schema=QUESTION)
        datasets.append(
            dict(
                name=entry["name"],
                domain=rows[0]["domain"],
                path=str(store.root / name),
                verifier=entry["kind"] if entry["kind"] != "code" else "lcb",
                n=entry["n"],
                protocol="public",
            )
        )
    panels = {
        domain: stratified_sample(rows, panel_count, seed, namespace=f"public-{domain}")
        for domain, rows in domains.items()
    }
    for entry in datasets:
        entry["panel_ids"] = [r["id"] for r in panels[entry["domain"]] if r["dataset"] == entry["name"]]
    reports, paths = {}, {}
    for domain, entries in inputs["training"].items():
        candidates = (
            r
            for entry in entries
            for path in entry.get("paths", [entry.get("path")])
            for r in adapt_input(entry["kind"], entry["name"], path)
        )
        eligibility = {"overlong": 0}

        def eligible(candidates, eligibility=eligibility):
            for row in candidates:
                if prompt_tokens is not None and prompt_tokens(row["prompt"]) > source_prompt_limit:
                    eligibility["overlong"] += 1
                else:
                    yield row

        train, heldout, report = source_split(
            eligible(candidates), public, train_count=train_count, heldout_count=panel_count, seed=seed
        )
        report.update(eligibility)
        reports[domain] = report
        store.write_rows(f"source/{domain}.parquet", heldout, schema=QUESTION)
        train_rows = [
            dict(
                prompt=r["prompt"],
                data_source=r["dataset"],
                ability=domain,
                reward_model={"style": "rule", "ground_truth": r["ground_truth"]},
                extra_info={"index": r["id"], "data_source": r["dataset"], "split": "train"},
            )
            for r in train
        ]
        schema = pa.Table.from_pylist(train_rows).schema
        store.write_rows(f"training/{domain}.parquet", train_rows, schema=schema)
        paths[domain] = {
            "train": str(store.root / f"training/{domain}.parquet"),
            "source": str(store.root / f"source/{domain}.parquet"),
        }
    store.finish(
        metadata={
            "datasets": datasets,
            "paths": paths,
            "split_reports": reports,
            "input_sha256": {
                p: file_sha256(p)
                for e in inputs["public"] + [e for entries in inputs["training"].values() for e in entries]
                for p in e.get("paths", [e.get("path")])
            },
        }
    )
    return store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-count", type=int, default=6400)
    parser.add_argument("--panel-count", type=int, default=128)
    parser.add_argument("--model")
    parser.add_argument("--model-family", default="auto")
    args = parser.parse_args()
    prompt_tokens = None
    if args.model:
        from transformers import AutoConfig, AutoTokenizer

        from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

        builder = create_continuous_token_builder(
            AutoTokenizer.from_pretrained(args.model),
            model_family=args.model_family,
            hf_model_type=AutoConfig.from_pretrained(args.model).model_type,
        )
        prompt_tokens = lambda messages: len(builder.build_initial_tokens(messages))
    prepare(
        json.loads(Path(args.inputs).read_text()),
        args.output,
        train_count=args.train_count,
        panel_count=args.panel_count,
        prompt_tokens=prompt_tokens,
    )


if __name__ == "__main__":
    main()
