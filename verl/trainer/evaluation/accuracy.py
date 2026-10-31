# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU grading of the exact native generation samples, without regeneration."""

import itertools
import json
import math
import multiprocessing
import os
import random
import re
import signal
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .artifacts import ACCURACY, ArtifactStore, file_sha256, identity


def final_answer(text, *, thinking_open, channel):
    if channel == "whole_response":
        return text.strip()
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    if thinking_open or "<think>" in text:
        return ""
    return text.strip()


def gpqa_letter(text):
    """Require an explicit final answer; never fish a choice out of the reasoning."""
    boxed = re.findall(r"\\boxed\{\s*([A-D])\s*\}", text)
    if boxed:
        return boxed[-1]
    match = re.search(r"(?:^|\n)\s*(?:(?:The\s+)?(?:final\s+)?answer\s*(?:is|:)\s*)?\(?([A-D])\)?[.\s]*$", text, re.I)
    return match.group(1).upper() if match else None


def referenced_tests(reference):
    """Read one raw test row, leaving large test payloads out of prepared questions."""
    path = Path(reference["path"]).resolve()
    stat = path.stat()
    return _referenced_tests(
        str(path), reference["row"], reference.get("column", "tests"), stat.st_size, stat.st_mtime_ns
    )


@lru_cache(maxsize=2)
def _referenced_tests(path, index, column, size_bytes, modified_ns):
    # At most two question payloads per verifier process. Fingerprints invalidate
    # the cache if a raw file changes; callers treat these test records as read-only.
    parquet = pq.ParquetFile(path)
    if type(index) is not int or not 0 <= index < parquet.metadata.num_rows:
        raise ValueError("Invalid test-case row reference")
    for group in range(parquet.num_row_groups):
        size = parquet.metadata.row_group(group).num_rows
        if index < size:
            # Only the requested column is materialized, never the entire dataset.
            columns = [column]
            if "metadata" in parquet.schema_arrow.names:
                columns.append("metadata")
            batches = parquet.iter_batches(batch_size=1, row_groups=[group], columns=columns)
            value = next(itertools.islice(batches, index, index + 1)).to_pylist()[0]
            tests = value[column]
            tests = json.loads(tests) if isinstance(tests, str) else tests
            if isinstance(tests, list) and tests and tests[0].get("testtype") == "functional":
                function = (value.get("metadata") or {}).get("func_name")
                if not function:
                    raise ValueError("Functional code task is missing its declared function name")
                tests = [dict(test, metadata={**(test.get("metadata") or {}), "func_name": function}) for test in tests]
            return tests
        index -= size
    raise AssertionError("Unreachable test-case row")


def verify_sample(row, question, verifier, answer):
    """Bound verifier wall time in its worker process, preserving failure semantics."""

    def expired(signum, frame):
        raise TimeoutError("CPU verifier exceeded 240 seconds; this is not an incorrect answer")

    handler = signal.signal(signal.SIGALRM, expired)
    previous = signal.setitimer(signal.ITIMER_REAL, 240)
    try:
        return _verify_sample(row, question, verifier, answer)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
        signal.signal(signal.SIGALRM, handler)


def _verify_sample(row, question, verifier, answer):
    """Runs in a dedicated CPU process. Infrastructure exceptions stay exceptions."""
    started = time.monotonic()
    seed = row["seed"] % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    truth = question.get("ground_truth")
    if isinstance(truth, str) and verifier in {"ifeval", "lcb"}:
        truth = json.loads(truth)
    details = {}
    if verifier == "math":
        from verl.utils.reward_score.math_verify import _verify_in_subprocess

        score = _verify_in_subprocess("\\boxed{" + str(truth) + "}", answer) if answer else 0.0
    elif verifier == "gpqa":
        if truth not in "ABCD" or len(truth) != 1:
            raise ValueError("GPQA gold must be the frozen permuted choice letter")
        score = float(gpqa_letter(answer) == truth)
    elif verifier == "ifeval":
        from langdetect import DetectorFactory

        from .vendor.ifeval import evaluation_lib

        DetectorFactory.seed = seed
        inp = evaluation_lib.InputExample(**truth)
        result = evaluation_lib.test_instruction_following_strict(inp, {inp.prompt: answer})
        score = float(result.follow_all_instructions)
        details = {
            "instructions_correct": sum(result.follow_instruction_list),
            "instructions_total": len(result.follow_instruction_list),
        }
    elif verifier == "lcb":
        from verl.utils.reward_score.one_shot_opd.code_eval_v2 import extract_code_from_model, lcb_check_correctness

        tests = referenced_tests(truth["tests_ref"]) if "tests_ref" in truth else truth["tests"]
        if not isinstance(tests, list) or not tests or any(not {"input", "output"} <= t.keys() for t in tests):
            raise ValueError("Malformed or missing LiveCodeBench tests")
        code = extract_code_from_model(answer)
        score = (
            float(
                lcb_check_correctness(
                    tests, code, timeout=6, context={"sample_id": row["id"], "dataset": row["dataset"]}
                )[0]
            )
            if code
            else 0.0
        )
    else:
        raise ValueError(f"Unknown verifier {verifier}")
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid accuracy score")
    return {
        key: row[key]
        for key in ("id", "source_id", "dataset", "domain", "sample_index", "finish_reason", "response_tokens")
    } | {
        "score": score,
        "status": "ok" if answer else "missing_final_answer",
        "answer": answer,
        "seconds": time.monotonic() - started,
        **details,
    }


def summarize_accuracy(rows, expected):
    """Each question has exactly n samples; report avg@n, never best-of-n."""
    rows = list(rows)
    observed = [(r["dataset"], r["source_id"], r["sample_index"]) for r in rows]
    if len(set(observed)) != len(observed) or set(observed) != set(expected):
        raise ValueError("Incomplete or duplicate accuracy coverage")
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] not in {"ok", "missing_final_answer"} or not math.isfinite(row["score"]):
            raise ValueError("Unresolved verifier failure cannot become an incorrect answer")
        grouped[row["dataset"]].append(row)
    metrics = {}
    for dataset, values in grouped.items():
        questions = defaultdict(list)
        for row in values:
            questions[row["source_id"]].append(row["score"])
        prefix = f"eval/accuracy/{dataset}"
        metrics.update(
            {
                f"{prefix}/avg_at_n": math.fsum(math.fsum(v) / len(v) for v in questions.values()) / len(questions),
                f"{prefix}/questions": len(questions),
                f"{prefix}/samples": len(values),
                f"{prefix}/response_tokens_mean": math.fsum(r["response_tokens"] for r in values) / len(values),
                f"{prefix}/length_finish_fraction": sum(r["finish_reason"] == "length" for r in values) / len(values),
                f"{prefix}/missing_final_fraction": sum(r["status"] == "missing_final_answer" for r in values)
                / len(values),
            }
        )
        totals = sum(r.get("instructions_total") or 0 for r in values)
        if totals:
            metrics[f"{prefix}/instruction_strict"] = sum(r.get("instructions_correct") or 0 for r in values) / totals
    return metrics


def gap_recovery(current, initial, teacher):
    """Preserve the requested ratio, including negative gaps; zero gaps are undefined."""
    result = {}
    for key, after in current.items():
        if not key.endswith("/avg_at_n"):
            continue
        before, target = initial[key], teacher[key]
        gap = target - before
        prefix = key.removesuffix("/avg_at_n")
        result[f"{prefix}/teacher_gap"] = gap
        result[f"{prefix}/teacher_gap_positive"] = int(gap > 0)
        result[f"{prefix}/gap_recovery_defined"] = int(gap != 0)
        if gap:
            result[f"{prefix}/gap_recovery_percent"] = 100 * (after - before) / gap
    return result


class AccuracyCollector:
    """Bounded process queue; immutable scores remain reusable without GPU regeneration."""

    def __init__(
        self, destination, specifications, tokenizer, *, workers=8, pending_limit=128, timeout=300, input_mib=256
    ):
        specifications = list(specifications)
        self.question_hash = identity([(s.name, s.n, s.protocol, questions) for s, questions in specifications])
        self.store = ArtifactStore.create(
            destination,
            kind="accuracy",
            metadata={
                "aggregation": "question_mean_of_sample_mean",
                "question_set_sha256": self.question_hash,
                "verifier_sha256": file_sha256(__file__),
                "code_input_mib": input_mib,
                "pending_limit": pending_limit,
                "worker_count": workers,
                "ifeval_source_sha256": file_sha256(Path(__file__).parent / "vendor/ifeval/SOURCE.json"),
            },
        )
        self.expected = {
            (spec.name, q["id"], i) for spec, questions in specifications for q in questions for i in range(spec.n)
        }
        self.tokenizer, self.timeout = tokenizer, timeout
        self.pending_limit = pending_limit
        if type(input_mib) is not int or input_mib <= 0:
            raise ValueError("Positive code test payload budget required")
        self.pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_verifier,
            initargs=(input_mib,),
        )
        self.pending, self.rows = deque(), []
        self.chunk = []
        self.shard = 0

    def _flush(self):
        if self.chunk:
            self.store.write_rows(f"scores/part-{self.shard:05d}.parquet", self.chunk, schema=ACCURACY)
            self.shard += 1
            self.chunk = []

    def _consume(self):
        future, sample = self.pending.popleft()
        try:
            result = future.result(timeout=self.timeout)
        except Exception as exc:
            raise RuntimeError(f"Accuracy verifier failed for {sample}") from exc
        self.chunk.append(result)
        # Only scalar evidence stays in RAM after a score has been queued for publication.
        self.rows.append({k: v for k, v in result.items() if k != "answer"})
        if len(self.chunk) >= 32:
            self._flush()

    def submit(self, outputs):
        for row, question, spec, protocol in outputs:
            while len(self.pending) >= self.pending_limit:
                self._consume()
            text = self.tokenizer.decode(row["response_ids"], skip_special_tokens=False)
            answer = final_answer(text, thinking_open=row["thinking_open"], channel=protocol.answer_channel)
            # Remove only terminal control tokens, preserving thinking delimiters above.
            for token in (self.tokenizer.eos_token, self.tokenizer.pad_token):
                if token:
                    answer = answer.removesuffix(token).rstrip()
            compact = {
                k: row[k]
                for k in (
                    "id",
                    "source_id",
                    "dataset",
                    "domain",
                    "sample_index",
                    "seed",
                    "finish_reason",
                    "response_tokens",
                )
            }
            self.pending.append(
                (
                    self.pool.submit(verify_sample, compact, question, spec.verifier, answer),
                    {k: compact[k] for k in ("id", "source_id", "dataset", "sample_index")},
                )
            )

    def finish(self, generation=None):
        try:
            while self.pending:
                self._consume()
            metrics = summarize_accuracy(self.rows, self.expected)
            self._flush()
            binding = {}
            if generation is not None:
                manifest = generation.manifest(complete=True)
                if manifest["metadata"].get("question_set_sha256") != self.question_hash:
                    raise ValueError("Accuracy generation has different questions or sampling counts")
                binding = {
                    "generation_manifest": str(generation.manifest_path),
                    "generation_sha256": file_sha256(generation.manifest_path),
                }
            self.store.finish(metadata={"metrics": metrics, **binding})
            return metrics
        finally:
            self.pool.shutdown(wait=True, cancel_futures=True)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


def _initialize_verifier(input_mib):
    # Only verifier child processes inherit this execution budget; model requests do not.
    os.environ["MIRROROPD_LCB_INPUT_MIB"] = str(input_mib)
