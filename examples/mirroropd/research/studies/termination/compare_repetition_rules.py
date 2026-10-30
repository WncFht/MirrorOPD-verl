# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU comparison of exact completed-step reuse and MD-OPD compression windows.

Online savings start after the evidence is observable, never at the beginning
of a flagged window. AUC against semantic loop labels requires a labeled corpus;
the existing exploratory review is reported separately from repetition proxies.
"""

import argparse
import bisect
import codecs
import importlib.util
import json
import multiprocessing
import re
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HEADINGS = re.compile(r"(?m)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?(?:Step[ \t]+\d+|步骤[ \t]*\d+)[^\n]*\n")
NUMBER = re.compile(r"^(\s*(?:#{1,6}\s*)?(?:\*\*)?(?:Step\s+|步骤\s*))\d+")
ZLIB_PROFILES = {
    "zlib1024-t10": (1024, 10),
    "zlib1024-t8": (1024, 8),
    "zlib1024-t6": (1024, 6),
    "zlib1024-t4": (1024, 4),
    "zlib4096-t10": (4096, 10),
    "zlib4096-t6": (4096, 6),
    "zlib8192-t10": (8192, 10),
}


def byte_decoder():
    """Inverse of the GPT/Qwen byte-level alphabet; no tokenizer re-encoding."""
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    alphabet = {chr(b): b for b in visible}
    missing = [b for b in range(256) if b not in visible]
    alphabet.update({chr(256 + i): b for i, b in enumerate(missing)})
    return alphabet


def character_token_ends(ids, tokenizer, alphabet):
    """Map observed Unicode character positions to actual sampled-token prefixes."""
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    special = set(tokenizer.all_special_ids)
    chunks, ends, length = [], [], 0
    for token, token_id in zip(tokenizer.convert_ids_to_tokens(ids), ids, strict=True):
        chunk = "" if token_id in special else decoder.decode(bytes(alphabet[c] for c in token))
        chunks.append(chunk)
        length += len(chunk)
        ends.append(length)
    # vLLM incremental detokenization does not expose an unfinished UTF-8
    # character at a length cap. Do not synthesize a replacement character.
    return "".join(chunks), ends


def completed_blocks(text):
    """Yield (start, end, observed_end); the final block only completes at EOS."""
    headings = list(HEADINGS.finditer(text))
    # Start with paragraphs, and switch only when the first complete Step
    # heading is observed. Future headings must not change earlier decisions.
    paragraph_end = headings[0].start() if headings else len(text)
    start = 0
    for boundary in re.finditer(r"\n[ \t]*\n+", text[:paragraph_end]):
        yield start, boundary.start(), boundary.end()
        start = boundary.end()
    if headings and start < paragraph_end:
        yield start, paragraph_end, headings[0].end()
    for current, following in zip(headings, headings[1:], strict=False):
        yield current.start(), following.start(), following.end()


def step_stops(text, token_ends):
    """Replay fixed 32-block history; short/new blocks reset consecutive repeats."""
    profiles = {f"step-n{n}-t{tokens}": (n, tokens, True) for n in (3, 4, 6) for tokens in (256, 512, 1024)}
    profiles["step-raw-n3-t512"] = (3, 512, False)
    result = {name: len(token_ends) for name in profiles}
    blocks = list(completed_blocks(text))
    for normalize in (False, True):
        history = deque(maxlen=32)
        count = repeated_tokens = 0
        for start, end, observed in blocks:
            block = text[start:end]
            if normalize:
                block = NUMBER.sub(r"\g<1>#", block)
            key = " ".join(block.split())
            size = max(0, bisect.bisect_left(token_ends, end) + 1 - bisect.bisect_right(token_ends, start))
            if size >= 64 and key in history:
                count += 1
                repeated_tokens += size
            else:
                count = repeated_tokens = 0
            for name, (minimum_count, minimum_tokens, normalized) in profiles.items():
                if normalized == normalize and count >= minimum_count and repeated_tokens >= minimum_tokens:
                    result[name] = min(result[name], bisect.bisect_left(token_ends, observed) + 1)
            history.append(key)
    return result


def review_auc(rows, labels):
    """Pairwise AUC on the explicitly reviewed subset, not population performance."""
    result = {}
    for window in ("1024", "4096", "8192"):
        positive = [r["scores"][window] for r in rows if labels.get((r["source"], r["prompt_id"])) == 1]
        negative = [r["scores"][window] for r in rows if labels.get((r["source"], r["prompt_id"])) == 0]
        result[window] = sum(float(p > n) + 0.5 * float(p == n) for p in positive for n in negative) / (
            len(positive) * len(negative)
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--mdopd", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    from examples.mirroropd.research.core.math_grading import grade, init_grader

    module_path = args.mdopd / "verl/verl/trainer/ppo/repetition_metrics.py"
    spec = importlib.util.spec_from_file_location("reference_repetition", module_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    alphabet = byte_decoder()
    previous = list(map(json.loads, args.previous.read_text().splitlines()))
    files, rows, jobs, timings = {}, [], [], []
    for i, old in enumerate(previous):
        source = old["source"]
        if source not in files:
            files[source] = {x["prompt_id"]: x for x in map(json.loads, (args.runs / source).read_text().splitlines())}
        raw = files[source][old["prompt_id"]]
        ids = raw["response_ids"]
        text, ends = character_token_ends(ids, tokenizer, alphabet)
        if text != raw["output"] and text.rstrip("\ufffd") == raw["output"]:
            # vLLM also withholds trailing replacement glyphs until the next
            # token resolves them; none of these final glyphs can justify a stop.
            text = raw["output"]
            ends = [min(end, len(text)) for end in ends]
        assert text == raw["output"], (source, raw["prompt_id"], "sampled token decoding mismatch")
        stops = step_stops(text, ends)
        scores, offline_flags = {}, {}
        for window in (1024, 4096, 8192):
            begin = time.perf_counter()
            ratios = reference.window_ratios(text, window_size=window, stride=256)
            timings.append({"window": window, "calls": len(ratios), "seconds": time.perf_counter() - begin})
            # The reference additionally checks an end-aligned tail window. It
            # only becomes available at EOS, so it cannot save any generation.
            starts = reference._window_starts(len(text), window, 256)
            scores[str(window)] = float(max(ratios))
            for name, (w, threshold) in ZLIB_PROFILES.items():
                if w != window:
                    continue
                offline_flags[name] = bool(max(ratios) >= threshold)
                candidates = [
                    s + window
                    for s, score in zip(starts, ratios, strict=True)
                    if score >= threshold and s % 256 == 0 and s + window < len(text)
                ]
                stop = bisect.bisect_left(ends, min(candidates)) + 1 if candidates else len(ids)
                stops[name] = stop
        ref = reference.analyze_response(text)
        assert scores["1024"] == ref["max_window_ratio"]
        assert offline_flags["zlib1024-t10"] == bool(ref["repetition"])
        row = {k: old[k] for k in ("source", "prompt_id", "length", "acc", "finish_reason", "repeat8")}
        row.update(scores=scores, offline_flags=offline_flags, profiles={})
        grade_indices = {}
        for name, stop in stops.items():
            rec = {"length": stop, "acc": row["acc"]}
            if stop < len(ids):
                if stop not in grade_indices:
                    grade_indices[stop] = len(jobs)
                    jobs.append(
                        {
                            "response_ids": ids[:stop],
                            "output": tokenizer.decode(ids[:stop], skip_special_tokens=True),
                            "data_source": raw["data_source"],
                            "gts": raw["gts"],
                        }
                    )
                rec["grade_index"] = grade_indices[stop]
            row["profiles"][name] = rec
        rows.append(row)
        if (i + 1) % 128 == 0:
            print(f"Replayed {i + 1}/{len(previous)}", flush=True)
    print(f"Grading {len(jobs)} distinct retained prefixes", flush=True)
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn"), initializer=init_grader
    ) as pool:
        grades = list(pool.map(grade, jobs))
    for row in rows:
        for rec in row["profiles"].values():
            if "grade_index" in rec:
                rec["acc"] = bool(grades[rec.pop("grade_index")]["acc"])
    review = json.loads(args.review.read_text())["cases"]
    labels = {
        (r["source"], r["prompt_id"]): int(r["judgment"] in {"loop", "late_loop"})
        for r in review
        if r["judgment"] in {"loop", "late_loop", "normal_progress"}
    }
    summary = {
        "responses": len(rows),
        "original_correct": sum(r["acc"] for r in rows),
        "original_tokens": sum(r["length"] for r in rows),
        "profiles": {},
        "review_positive": sum(labels.values()),
        "review_negative": len(labels) - sum(labels.values()),
        "auc_status": (
            "exploratory selected-case ranking only: 9 positives, 1 negative; "
            "selection bias and repeated questions; NOT a population AUC estimate"
        ),
        "review_auc": review_auc(rows, labels),
    }
    for name in rows[0]["profiles"]:
        hit = [r for r in rows if r["profiles"][name]["length"] < r["length"]]
        summary["profiles"][name] = {
            "stopped": len(hit),
            "tokens_saved": sum(r["length"] - r["profiles"][name]["length"] for r in hit),
            "correct_lost": sum(r["acc"] and not r["profiles"][name]["acc"] for r in hit),
            "correct_gained": sum(not r["acc"] and r["profiles"][name]["acc"] for r in hit),
            "cap_stopped": sum(r["finish_reason"] == "length" for r in hit),
            "high_repeat8_stopped": sum(r["repeat8"] >= 0.8 for r in hit),
            "review_loop_stopped": sum(labels.get((r["source"], r["prompt_id"])) == 1 for r in hit),
            "review_normal_stopped": sum(labels.get((r["source"], r["prompt_id"])) == 0 for r in hit),
        }
        if name in ZLIB_PROFILES:
            summary["profiles"][name]["offline_flagged"] = sum(r["offline_flags"][name] for r in rows)
    summary["compression_timing"] = {
        str(w): {
            "window_calls": sum(r["calls"] for r in timings if r["window"] == w),
            "seconds": sum(r["seconds"] for r in timings if r["window"] == w),
        }
        for w in (1024, 4096, 8192)
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
