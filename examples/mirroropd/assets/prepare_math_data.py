# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Import prepared math parquet or rebuild the math datasets from upstream raw files.

Conversion rules follow MD-OPD cef14fa0b552b40033f8b7296ce3853d10bf479b,
data/prep/prepare_dapo_math500.py and prepare_amc23_aime25.py (Apache-2.0).
"""

import argparse
import json
import math
import shutil
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from check_assets import DATA_ROOT, DATASETS, SCHEMA, check_data, write_json

RAW_FILES = {
    "dapo-math-17k.parquet": ("BytedTsinghua-SIA/DAPO-Math-17k", "data/dapo-math-17k.parquet"),
    "math500_test.jsonl": ("HuggingFaceH4/MATH-500", "test.jsonl"),
    "amc23.parquet": ("knoveleng/AMC-23", "data/train-00000-of-00001.parquet"),
    "aime2025-I.jsonl": ("opencompass/AIME2025", "aime2025-I.jsonl"),
    "aime2025-II.jsonl": ("opencompass/AIME2025", "aime2025-II.jsonl"),
}
# Pin benchmark inputs so future preparations retain the same questions and answers.
AIME_FILES = {
    "AIME24": (
        "aime2024.parquet",
        "Maxwell-Jia/AIME_2024",
        "8d88b2876a82a080e2f172cc9b25d0d9d2cb4792",
        "aime_2024_problems.parquet",
        "Problem",
        "Answer",
        "ID",
    ),
    "AIME26": (
        "aime2026.parquet",
        "MathArena/aime_2026",
        "d61f486805306e6e6b5913504c07763a99e1b112",
        "data/train-00000-of-00001.parquet",
        "problem",
        "answer",
        "problem_idx",
    ),
}

DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be of "
    "the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
DAPO_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def make_row(content, answer, name, index):
    """Build the same explicit math schema used by the reference recipe."""
    if answer is None or isinstance(answer, float) and not math.isfinite(answer):
        raise ValueError(f"{name} raw row {index}: missing or nonfinite answer")
    return {
        "prompt": [{"content": content.strip(), "role": "user"}],
        "data_source": name,
        "ability": "math",
        "reward_model": {"ground_truth": str(answer), "style": "rule"},
        "extra_info": {
            "difficulty": "",
            "index": f"{name}-{index}",
            "n_tests": 0,
            "source": name,
            "split": DATASETS[name]["split"],
        },
    }


def download_raw(raw_dir, names=None):
    """Download only on explicit request; publish each complete raw file atomically."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    names = set(DATASETS if names is None else names)
    legacy = RAW_FILES if names - AIME_FILES.keys() else {}
    for filename, (repo, member) in legacy.items():
        destination = raw_dir / filename
        if destination.is_file() and destination.stat().st_size:
            continue
        query = urllib.parse.urlencode({"Revision": "master", "FilePath": member})
        url = f"https://www.modelscope.cn/api/v1/datasets/{repo}/repo?{query}"
        print(f"Downloading {url}", flush=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        urllib.request.urlretrieve(url, partial)
        partial.replace(destination)

    for name in names & AIME_FILES.keys():
        filename, repo, revision, member, *_ = AIME_FILES[name]
        destination = raw_dir / filename
        if destination.is_file() and destination.stat().st_size:
            continue
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{member}"
        print(f"Downloading {url}", flush=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        urllib.request.urlretrieve(url, partial)
        partial.replace(destination)


def read_jsonl(path):
    """Read a raw JSONL benchmark in its source order."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_legacy_rows(raw_dir):
    """Keep first DAPO index occurrences, preserving benchmark content and ordering."""
    rows, seen = [], set()
    raw = pq.ParquetFile(raw_dir / "dapo-math-17k.parquet")
    print(f"Streaming {raw.metadata.num_rows} raw DAPO rows; retaining first occurrences by source index", flush=True)
    for batch in raw.iter_batches(batch_size=4096, columns=["prompt", "extra_info", "reward_model"]):
        for item in batch.to_pylist():
            identifier = str(item["extra_info"]["index"])
            if identifier in seen:
                continue
            seen.add(identifier)
            text = item["prompt"][0]["content"].removeprefix(DAPO_PREFIX).removesuffix(DAPO_SUFFIX)
            rows.append(make_row(text, item["reward_model"]["ground_truth"], "DAPO", len(rows)))
    yield "DAPO", rows
    rows = [
        make_row(item["problem"], item["answer"], "MATH-500", index)
        for index, item in enumerate(read_jsonl(raw_dir / "math500_test.jsonl"))
    ]
    yield "MATH-500", rows
    rows = [
        make_row(item["problem"], str(item["answer"]).strip(), "AMC23", index)
        for index, item in enumerate(pq.read_table(raw_dir / "amc23.parquet").to_pylist())
    ]
    yield "AMC23", rows
    raw_aime = []
    for part in ("I", "II"):
        items = read_jsonl(raw_dir / f"aime2025-{part}.jsonl")
        if len(items) != 15:
            raise ValueError(f"AIME2025-{part}: expected 15 questions, got {len(items)}")
        raw_aime.extend(items)
    yield (
        "AIME25",
        [
            make_row(item["question"], str(item["answer"]).strip(), "AIME25", index)
            for index, item in enumerate(raw_aime)
        ],
    )


def build_rows(raw_dir, names=None):
    """Keep benchmark membership; never include worked solutions in prompts."""
    names = set(DATASETS if names is None else names)
    if names - AIME_FILES.keys():
        for name, rows in build_legacy_rows(raw_dir):
            if name in names:
                yield name, rows
    for name, spec in AIME_FILES.items():
        if name not in names:
            continue
        filename, _, _, _, problem_key, answer_key, id_key = spec
        items = pq.read_table(raw_dir / filename).to_pylist()
        ids = [str(item[id_key]) for item in items]
        if len(items) != 30 or len(set(ids)) != 30:
            raise ValueError(f"{name}: expected 30 unique source problem IDs")
        if name == "AIME24":
            expected = {f"2024-{part}-{i}" for part in ("I", "II") for i in range(1, 16)}
        else:
            expected = {str(i) for i in range(1, 31)}
        if set(ids) != expected:
            raise ValueError(f"{name}: source IDs do not cover the complete competition")
        rows = []
        for item, identifier in zip(items, ids, strict=True):
            answer = str(item[answer_key]).strip()
            if not answer.isascii() or not answer.isdecimal() or not 0 <= int(answer) <= 999:
                raise ValueError(f"{name} {identifier}: invalid AIME integer answer {answer!r}")
            rows.append(make_row(item[problem_key], str(int(answer)), name, identifier))
        yield name, rows


def main():
    """Stage and validate selected files before updating the prepared dataset directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-root", type=Path, help="existing prepared root containing math/")
    source.add_argument("--raw-dir", type=Path, help="pre-downloaded raw files, no network")
    source.add_argument("--download", action="store_true", help="download missing raw files from upstream repositories")
    parser.add_argument("--overwrite", action="store_true", help="replace existing prepared output files")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    args = parser.parse_args()
    selected = {name: DATASETS[name] for name in args.datasets}
    if args.source_root is not None:
        args.source_root = args.source_root.expanduser().resolve()
    root = args.data_root.expanduser().resolve()
    for spec in selected.values():
        if (root / spec["path"]).exists() and not args.overwrite:
            raise FileExistsError(f"{root / spec['path']} exists; check it or explicitly use --overwrite")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"mode": "import" if args.source_root else "rebuild", "data": {}, "raw": {}}
    raw_dir = args.raw_dir.expanduser().resolve() if args.raw_dir else root / ".cache/raw-math"
    if args.download:
        download_raw(raw_dir, selected)
    if not args.source_root:
        legacy = RAW_FILES if set(selected) - AIME_FILES.keys() else {}
        for filename, (repo, member) in legacy.items():
            manifest["raw"][filename] = {
                "repo": repo,
                "member": member,
                "mirror_revision": "master",
            }
        for name in set(selected) & AIME_FILES.keys():
            filename, repo, revision, member, *_ = AIME_FILES[name]
            manifest["raw"][filename] = {"repo": repo, "revision": revision, "member": member}
    with tempfile.TemporaryDirectory(prefix=".prepare-math-", dir=root) as temp:
        staged = Path(temp)
        if args.source_root:
            manifest["source_root"] = str(args.source_root.expanduser().resolve())
            for spec in selected.values():
                destination = staged / spec["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(args.source_root / spec["path"], destination)
        else:
            for name, rows in build_rows(raw_dir, selected):
                destination = staged / DATASETS[name]["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), destination)
        for name, spec in selected.items():
            report, _ = check_data(staged / spec["path"], name)
            report["path"] = spec["path"]
            manifest["data"][name] = report
        for spec in selected.values():
            destination = root / spec["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            (staged / spec["path"]).replace(destination)
    write_json(root / "math_manifest.json", manifest)
    print(f"Prepared {len(selected)} math datasets: {root}")


if __name__ == "__main__":
    main()
