# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Source-bound preparation, conservative near-duplicate splits and frozen sampling tables."""

import collections
import hashlib
import importlib.metadata
import json
import math
import random
import re
import unicodedata
from pathlib import Path

from .objectives import ARMS, calibrate_alpha, probabilities, selection_table, weight_diagnostics


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def records(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def text_key(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def near_components(texts, threshold=0.9):
    """Exact Jaccard threshold join on ordered 4-word shingles, with prefix filtering.

    Exact text and source-ID grouping is additional. This operational definition
    catches near copies, not arbitrary mathematical paraphrases.
    """
    parent = list(range(len(texts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    exact, sets = {}, []
    for i, text in enumerate(texts):
        normalized = text_key(text)
        if normalized in exact:
            union(i, exact[normalized])
        else:
            exact[normalized] = i
        words = re.findall(r"\S+", normalized.casefold())
        sets.append(set(zip(*(words[j:] for j in range(4)), strict=False)) or {tuple(words)})
    frequency = collections.Counter(s for signatures in sets for s in signatures)
    inverted = collections.defaultdict(list)
    for i in sorted(range(len(sets)), key=lambda k: len(sets[k])):
        a = sets[i]
        ordered = sorted(a, key=lambda s: (frequency[s], s))
        prefix = ordered[: len(a) - math.ceil(threshold * len(a)) + 1]
        candidates = {j for s in prefix for j in inverted[s] if len(sets[j]) >= threshold * len(a)}
        for j in candidates:
            b = sets[j]
            if find(i) != find(j) and len(a & b) >= threshold * len(a | b):
                union(i, j)
        for s in prefix:
            inverted[s].append(i)
    return [find(i) for i in range(len(texts))]


def code_inventory():
    here = Path(__file__).resolve()
    repo = here.parents[6]
    paths = list(here.parent.glob("*.py")) + [
        repo / "examples/mirroropd/research/backends/hf_models.py",
        repo / "examples/mirroropd/research/studies/fixed_transfer/teacher_construction_llm.py",
        repo / "examples/mirroropd/research/core/math_grading.py",
        repo / "verl/utils/reward_score/one_shot_opd/math.py",
    ]
    paths += list((repo / "verl/utils/reward_score/one_shot_opd/ttrl_math").rglob("*.py"))
    paths += [
        repo / "verl/utils/reward_score/math_verify.py",
        repo / "verl/utils/reward_score/math_dapo.py",
        repo / "verl/trainer/evaluation/accuracy.py",
    ]
    return {str(p): sha(p) for p in sorted(paths)}


def environment():
    return {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "vllm", "flash-attn", "numpy", "math-verify", "pyarrow")
    }


def model_inventory(path):
    return {
        str(p.resolve()): dict(
            size=p.stat().st_size,
            mtime_ns=p.stat().st_mtime_ns,
            sha256=sha(p) if p.stat().st_size < 20_000_000 else None,
        )
        for p in sorted(Path(path).iterdir())
        if p.is_file() and p.suffix in (".json", ".safetensors", ".txt", ".jinja")
    }


def contract(root):
    cfg = read(root / "protocol.json")
    if code_inventory() != cfg["code"]:
        raise ValueError("Study source changed after preparation; use a fresh run or audited migration")
    if environment() != cfg["environment"]:
        raise ValueError("Frozen Python dependency versions changed")
    for name in ("base", "expert"):
        if model_inventory(cfg[name]) != cfg["models"][name]:
            raise ValueError(f"Frozen {name} files changed")
    for name, expected in cfg["prepared"].items():
        if sha(root / name) != expected:
            raise ValueError(f"Prepared question population changed: {name}")
    return cfg


def prepare(root, plan):
    import pyarrow.parquet as pq
    from transformers import AutoConfig, AutoTokenizer

    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Preparation requires a fresh directory")
    cfg = read(plan)
    for name in ("base", "expert", "source", "benchmark_dir", "ood_dir"):
        cfg[name] = str(Path(cfg[name]).resolve())
    cfg["wait_for"] = [str(Path(p).resolve()) for p in cfg.get("wait_for", [])]
    tokenizers = {name: AutoTokenizer.from_pretrained(cfg[name], local_files_only=True) for name in ("base", "expert")}
    if tokenizers["base"].get_vocab() != tokenizers["expert"].get_vocab():
        raise ValueError("Body token ID mappings differ")
    support = set(tokenizers["base"].get_vocab().values())
    if support != set(range(max(support) + 1)):
        raise ValueError("Noncontiguous tokenizer support requires an explicit support mask")
    cfg["support_size"] = len(support)
    for name in tokenizers:
        model_config = AutoConfig.from_pretrained(cfg[name], local_files_only=True)
        if model_config.max_position_embeddings < cfg["max_length"]:
            raise ValueError("Configured context exceeds actual model context")
    source = pq.read_table(cfg["source"]).to_pylist()
    references = []
    benchmark_paths = sorted(Path(cfg["benchmark_dir"]).glob("*.parquet"))
    if not benchmark_paths:
        raise ValueError("Benchmark exclusion set is empty")
    for path in benchmark_paths:
        references.extend(pq.read_table(path).to_pylist())

    def question(row):
        return "\n\n".join(m["content"] for m in row["prompt"] if m["role"] == "user")

    texts = [question(v) for v in references + source]
    components = near_components(texts, cfg["near_duplicate_jaccard"])
    forbidden = set(components[: len(references)])
    by_group, seen_source, excluded = {}, {}, collections.Counter()
    for offset, row in enumerate(source, len(references)):
        group = components[offset]
        source_id = str(row.get("data_source", "")) + ":" + str(row.get("extra_info", {}).get("index", offset))
        # Repeated source IDs never enter multiple populations, including conflicting texts.
        if source_id in seen_source:
            excluded["source_id_duplicate"] += 1
            continue
        seen_source[source_id] = group
        if group in forbidden or group in by_group:
            excluded["benchmark_or_near_duplicate"] += 1
            continue
        text = texts[offset]
        prompt = tokenizers["base"].encode(text + "\n\nAnswer:", add_special_tokens=False)
        messages = [
            dict(role="user", content=text + "\n\nGive a complete solution and put the final answer in \\boxed{}.")
        ]
        expert = tokenizers["expert"].apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False
        )
        if max(len(prompt), len(expert)) > cfg["max_prompt"]:
            excluded["prompt_length"] += 1
            continue
        by_group[group] = dict(
            id=fingerprint(text_key(text)),
            source_id=source_id,
            cluster=group,
            domain="math",
            text=text,
            prompt_ids=prompt,
            expert_prompt_ids=expert,
            ground_truth=row["reward_model"]["ground_truth"],
        )
    values = list(by_group.values())
    random.Random(cfg["split_seed"]).shuffle(values)
    splits, first = {}, 0
    # Reserve holdouts before the complete candidate-generation train reservoir.
    for split in ("calibration", "dev", "test", "opd"):
        count = cfg["counts"][split]
        splits[split] = values[first : first + count]
        first += count
    splits["train"] = values[first:]
    if len(splits["train"]) < cfg["counts"]["train"]:
        raise ValueError(f"Only {len(splits['train'])} train clusters remain after exclusions")
    root.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        jsonl(root / "prepared" / f"{split}.jsonl", (r | dict(split=split) for r in rows))
    # OOD diagnostics are question-disjoint from all public benchmarks and each other.
    ood_paths = []
    reserved_texts = [question(v) for v in references] + [v["text"] for v in values]
    for domain in ("code", "science", "if"):
        path = Path(cfg["ood_dir"]) / f"{domain}.parquet"
        ood_paths.append(path)
        rows = pq.read_table(path).to_pylist()
        combined = reserved_texts + [question(v) for v in rows]
        clusters = near_components(combined, cfg["near_duplicate_jaccard"])
        blocked, seen, selected = set(clusters[: len(reserved_texts)]), set(), []
        random.Random(cfg["split_seed"]).shuffle(rows)
        # Recompute after permutation so the row-to-cluster mapping stays exact.
        mapping = {text_key(t): c for t, c in zip(combined, clusters, strict=True)}
        for row in rows:
            text = question(row)
            group = mapping[text_key(text)]
            prompt = tokenizers["base"].encode(text + "\n\nAnswer:", add_special_tokens=False)
            if group in blocked or group in seen or len(prompt) > cfg["max_prompt"]:
                continue
            seen.add(group)
            selected.append(
                dict(
                    id=fingerprint(text_key(text)),
                    domain=domain,
                    prompt_ids=prompt,
                    ground_truth=row.get("ground_truth", row.get("reward_model", {}).get("ground_truth")),
                )
            )
        if len(selected) < 256:
            raise ValueError(f"Fewer than 256 benchmark-disjoint {domain} prompts")
        jsonl(root / "prepared" / f"ood-{domain}.jsonl", selected[:256])
        selected_keys = {r["id"] for r in selected[:256]}
        reserved_texts.extend(question(r) for r in rows if fingerprint(text_key(question(r))) in selected_keys)
    cfg["models"] = {name: model_inventory(cfg[name]) for name in tokenizers}
    cfg["code"] = code_inventory()
    cfg["environment"] = environment()
    cfg["sources"] = {str(p): sha(p) for p in [Path(cfg["source"]), *benchmark_paths, *ood_paths]}
    cfg["prepared"] = {str(p.relative_to(root)): sha(p) for p in sorted((root / "prepared").glob("*.jsonl"))}
    write(root / "protocol.json", cfg)
    write(
        root / "preparation.json",
        dict(
            source_rows=len(source),
            accepted_clusters=len(values),
            excluded=dict(excluded),
            splits={k: len(v) for k, v in splits.items()},
            near_duplicate_definition=(
                "NFKC exact + 4-word shingle Jaccard >= threshold; mathematical paraphrases not guaranteed"
            ),
        ),
    )


def frozen_pools(root, split):
    cfg = contract(root)
    expected = list(records(root / "prepared" / f"{split}.jsonl"))
    groups = []
    for q in expected:
        path = root / "scored" / split / f"{q['id']}.json"
        if not path.exists():
            if split == "train" and not (root / "generated" / split / f"{q['id']}.json").exists():
                continue
            raise ValueError(f"Missing scoring receipt for {q['id']}")
        group = read(path)
        if len(group["candidates"]) >= cfg["min_correct"]:
            groups.append(group)
        if split == "train" and len(groups) == cfg["counts"]["train"]:
            break
    required = (
        cfg["counts"][split] if split == "train" else math.ceil(cfg["counts"][split] * cfg["min_calibration_coverage"])
    )
    if len(groups) < required:
        raise ValueError(f"Insufficient valid {split} groups: {len(groups)}; no holdout recycling")
    return groups


def freeze(root):
    cfg = contract(root)
    if (root / "frozen.json").exists():
        raise FileExistsError("Sampling tables are immutable")
    for split in ("calibration", "train"):
        if not read(root / f"quality-{split}.json")["passed"]:
            raise ValueError(f"{split} candidate quality has not passed")
    calibration = frozen_pools(root, "calibration")
    groups = frozen_pools(root, "train")
    alpha, diagnostics = calibrate_alpha([g["candidates"] for g in calibration])
    jsonl(root / "pool.jsonl", groups)
    cumulative = {epoch: [] for epoch in (1, 2, 4)}
    lookup = {g["id"]: {c["candidate_id"]: len(c["content_ids"]) + 1 for c in g["candidates"]} for g in groups}
    for seed in cfg["seeds"]:
        draws = list(selection_table(groups, seed, alpha, epochs=cfg["epochs"]))
        jsonl(root / f"selection-{seed}.jsonl", draws)
        totals = {arm: 0 for arm in ARMS}
        for offset, draw in enumerate(draws, 1):
            for arm in ARMS:
                totals[arm] += lookup[draw["question_id"]][draw["choices"][arm]]
            if offset % len(groups) == 0 and draw["epoch"] in cumulative:
                cumulative[draw["epoch"]].extend(totals.values())
    jsonl(
        root / "weights.jsonl",
        (
            dict(
                id=g["id"],
                arm=arm,
                probabilities=(p := probabilities(g["candidates"], arm, alpha)),
                **weight_diagnostics(p),
            )
            for g in groups
            for arm in ARMS
        ),
    )
    paths = [root / "pool.jsonl", root / "weights.jsonl", *root.glob("selection-*.jsonl")]
    write(
        root / "frozen.json",
        dict(
            alpha=alpha,
            calibration=diagnostics,
            train_questions=len(groups),
            files={p.name: sha(p) for p in paths},
            protocol_sha256=sha(root / "protocol.json"),
            token_readout_budgets=sorted(set(min(totals) for totals in cumulative.values())),
        ),
    )
