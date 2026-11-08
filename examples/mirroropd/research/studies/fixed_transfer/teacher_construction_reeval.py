# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Resume multi-sample, long-budget capability readouts of existing checkpoints.

This study-specific scheduler reuses native GenerationProtocol, checkpoint
identity, TokenizerSupport, verify_sample and summarize_accuracy. It does not
implement a trainer, KL kernel, sampler or verifier. Original greedy outputs
and frozen training files remain independent. Math also records the original
canonical grader, to expose grader changes separately from sampling changes.
"""

import argparse
import concurrent.futures as futures
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

from examples.mirroropd.research.core.artifacts import records
from examples.mirroropd.research.core.io import dump, read
from verl.trainer.evaluation.accuracy import summarize_accuracy
from verl.trainer.evaluation.artifacts import file_sha256, identity
from verl.trainer.evaluation.config import GenerationProtocol

MODULE = "examples.mirroropd.research.studies.fixed_transfer.teacher_construction_reeval"
SPECS = {
    "MATH-500": ("math", "math", 4),
    "AMC23": ("math", "math", 16),
    "AIME24": ("math", "math", 16),
    "AIME25": ("math", "math", 16),
    "AIME26": ("math", "math", 16),
    "GPQA-Diamond": ("science", "gpqa", 4),
    "IFEval": ("if", "ifeval", 1),
    "LCBv5": ("code", "lcb", 10),
}


def protocol(domain):
    return GenerationProtocol(
        temperature=1,
        top_p=0.95,
        max_new_tokens=8192 if domain == "if" else 24576,
        prompt_limit=8192,
        stop_token_ids=(151643,),
        answer_channel="after_thinking",
        seed=42,
    )


def slots(questions, selected):
    return {
        (q["dataset"], q["id"], i) for q in questions if q["dataset"] in selected for i in range(SPECS[q["dataset"]][2])
    }


def slot(row):
    return row["dataset"], row["source_id"], row["sample_index"]


def validate_rows(rows, questions, selected):
    expected = slots(questions, selected)
    seen = set()
    lookup = {(q["dataset"], q["id"]): q for q in questions}
    for row in rows:
        key = slot(row)
        q = lookup.get(key[:2])
        if key in seen or key not in expected or q is None:
            raise ValueError("Duplicate or foreign response slot")
        p = protocol(q["domain"])
        if row["seed"] != p.sample_seed(*key) or row["protocol_id"] != p.protocol_id:
            raise ValueError("Response sampling identity changed")
        if row["response_tokens"] != len(row["response_ids"]):
            raise ValueError("Response length mismatch")
        if not 0 <= row["response_tokens"] <= p.max_new_tokens:
            raise ValueError("Response cap exceeded")
        seen.add(key)
    return seen, expected


def prepare(root, original):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from examples.mirroropd.research.studies.fixed_transfer.teacher_construction_llm import bare_prompt
    from verl.experimental.rollout_policy.identity import checkpoint_files, checkpoint_identity

    if (root / "plan.json").exists():
        raise FileExistsError("Use the existing plan or a fresh output directory")
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[5]
    source = repo / "runs/unified-evaluation-20261031/questions-v2/public"
    base = Path(read(original / "protocol.json")["base"])
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    old_math = records(original / "prepared/benchmark.jsonl")
    questions = []
    for name, (domain, _, _) in SPECS.items():
        if domain == "math":
            values = [dict(q, dataset=name) for q in old_math if q["benchmark"] == name]
        else:
            values = pq.read_table(source / f"{name}.parquet").to_pylist()
        for q in values:
            ids = q.get("prompt_ids")
            if ids is None:
                ids = tokenizer.encode(bare_prompt(q["prompt"]), add_special_tokens=False)
            if not ids or len(ids) > protocol(domain).prompt_limit:
                raise ValueError("Question exceeds the declared prompt budget")
            questions.append(
                dict(id=q["id"], dataset=name, domain=domain, prompt_ids=ids, ground_truth=q["ground_truth"])
            )
    dump(root / "questions.json", questions)
    states = {"base": base}
    for arm in ("RAW", "CCS-rho0", "CCS-rho1"):
        states[arm + "-teacher128"] = original / "teachers" / arm / "step0128"
        states[arm + "-student256"] = original / "opd" / arm / "step0256"
    states.update(
        {
            "RAW-student64": original / "opd/RAW/step0064",
            "RAW-student128": original / "opd/RAW/step0128",
            "ERT-teacher16": original / "teachers/ERT-verifier/step0016",
        }
    )
    bound = {}
    for label, path in states.items():
        if label != "base" and not (path / "checkpoint-complete.json").exists():
            raise FileNotFoundError(path)
        print("BIND", label, flush=True)
        bound[label] = dict(
            path=str(path.resolve()),
            identity=checkpoint_identity(path),
            files={
                str(f.relative_to(path.resolve())): [f.stat().st_size, f.stat().st_mtime_ns]
                for f in checkpoint_files(path)
            },
        )
    first = ["base", "RAW-teacher128", "RAW-student256"]
    order = first + [s for s in states if s not in first]
    jobs = [dict(label=s, phase="math", datasets=list(SPECS)[:5]) for s in order]
    jobs += [dict(label=s, phase="ood", datasets=list(SPECS)[5:]) for s in list(states)[:7]]
    sources = [
        Path(__file__),
        repo / "verl/trainer/evaluation/accuracy.py",
        repo / "verl/trainer/evaluation/config.py",
        repo / "examples/mirroropd/research/core/math_grading.py",
        original / "prepared/benchmark.jsonl",
        *source.glob("*.parquet"),
    ]
    dump(
        root / "plan.json",
        dict(
            schema=1,
            created=time.time(),
            original=str(original),
            questions_sha256=file_sha256(root / "questions.json"),
            models=bound,
            jobs=jobs,
            protocols={d: protocol(d).metadata() for d in ("math", "science", "code", "if")},
            samples={k: v[2] for k, v in SPECS.items()},
            math500_primary="sample0 avg@1 per four-domain protocol; avg@4 supplementary",
            execution=dict(gpus=[0, 5, 6], max_num_seqs=128, cpu_graders_per_gpu=6),
            sources={str(p): file_sha256(p) for p in sources},
        ),
    )
    for p in sources[:4]:
        target = root / "source" / p.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    print("PREPARED", len(questions), "questions", len(jobs), "jobs", flush=True)


def answer_text(row):
    from verl.trainer.evaluation.accuracy import final_answer

    answer = final_answer(row["text"], thinking_open=False, channel="after_thinking")
    for token in row.get("terminal_tokens", ()):
        if token:
            answer = answer.removesuffix(token).rstrip()
    return answer


def grade_one(row, question):
    from examples.mirroropd.research.core.math_grading import grade
    from verl.trainer.evaluation.accuracy import verify_sample

    answer = answer_text(row)
    value = verify_sample(row, question, SPECS[row["dataset"]][1], answer)
    if row["domain"] == "math":
        extra = grade(
            dict(
                data_source=row["dataset"],
                output=row["text"],
                gts=question["ground_truth"],
                response_ids=row["response_ids"],
            )
        )
        value["canonical_score"] = float(extra["acc"])
    return value


def summarize(dest, questions, selected):
    raw = records(dest / "responses.jsonl", repair=True)
    observed, expected = validate_rows(raw, questions, selected)
    scored = records(dest / "scores.jsonl", repair=True)
    if observed != expected or len(scored) != len(expected):
        return False
    metrics = summarize_accuracy(scored, expected)
    for name in selected:
        rows = [s for s in scored if s["dataset"] == name]
        if SPECS[name][0] == "math":
            metrics[f"canonical/{name}/avg_at_n"] = sum(r["canonical_score"] for r in rows) / len(rows)
        if name == "MATH-500":
            first = [r for r in rows if r["sample_index"] == 0]
            metrics["four_domain_primary/MATH-500/avg_at_1"] = sum(r["score"] for r in first) / len(first)
    dump(dest / "metrics.json", metrics)
    dump(
        dest / "complete.json",
        dict(responses=len(raw), scores=len(scored), expected=len(expected), completed=time.time()),
    )
    return True


def worker(root, index):
    import multiprocessing

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RepetitionDetectionParams, RequestOutputKind

    from examples.mirroropd.research.core.math_grading import init_grader
    from verl.experimental.rollout_policy.identity import checkpoint_files

    plan = read(root / "plan.json")
    job = plan["jobs"][index]
    label, selected = job["label"], job["datasets"]
    model = plan["models"][label]
    dest = root / "results" / label / job["phase"]
    dest.mkdir(parents=True, exist_ok=True)
    if file_sha256(root / "questions.json") != plan["questions_sha256"]:
        raise ValueError("Question identities changed")
    questions = [q for q in read(root / "questions.json") if q["dataset"] in selected]
    inventory = {
        str(f.relative_to(Path(model["path"]))): [f.stat().st_size, f.stat().st_mtime_ns]
        for f in checkpoint_files(model["path"])
    }
    if inventory != model["files"]:
        raise ValueError("Checkpoint files changed after content binding")
    inputs = dict(
        model=model["identity"],
        questions=plan["questions_sha256"],
        datasets=selected,
        protocols=plan["protocols"],
        samples=plan["samples"],
    )
    if (dest / "inputs.json").exists() and read(dest / "inputs.json") != inputs:
        raise ValueError("Partial outputs belong to different inputs")
    dump(dest / "inputs.json", inputs)
    if summarize(dest, questions, selected):
        return
    raw = records(dest / "responses.jsonl", repair=True)
    done, expected = validate_rows(raw, questions, selected)
    scored = records(dest / "scores.jsonl", repair=True)
    score_done = {slot(r) for r in scored}
    if len(score_done) != len(scored) or not score_done <= done:
        raise ValueError("Invalid grading coverage")
    lookup = {(q["dataset"], q["id"]): q for q in questions}
    pending = {}
    started = time.time()
    pool = futures.ProcessPoolExecutor(
        max_workers=6, mp_context=multiprocessing.get_context("spawn"), initializer=init_grader
    )
    with (dest / "responses.jsonl").open("a") as responses, (dest / "scores.jsonl").open("a") as scores:

        def drain(block=False):
            ready = (
                futures.wait(pending, return_when=futures.FIRST_COMPLETED).done
                if block and pending
                else [f for f in pending if f.done()]
            )
            for f in ready:
                row = f.result()
                scores.write(json.dumps(row, ensure_ascii=False) + "\n")
                scores.flush()
                score_done.add(slot(row))
                del pending[f]

        def submit(row):
            pending[pool.submit(grade_one, row, lookup[row["dataset"], row["source_id"]])] = slot(row)
            if len(pending) >= 192:
                drain(True)

        for row in raw:
            if slot(row) not in score_done:
                submit(row)
        tasks = sorted(expected - done)
        random.Random(42).shuffle(tasks)
        if tasks:
            tokenizer = AutoTokenizer.from_pretrained(model["path"], local_files_only=True)
            llm = LLM(
                model=model["path"],
                dtype="bfloat16",
                max_model_len=32768,
                max_num_seqs=128,
                max_num_batched_tokens=8192,
                gpu_memory_utilization=0.88,
                enforce_eager=True,
                enable_prefix_caching=True,
                seed=42,
                generation_config="vllm",
                logits_processors=["verl.workers.rollout.vllm_rollout.tokenizer_support:TokenizerSupport"],
            )
            engine = llm.llm_engine
            iterator = iter(tasks)
            inflight = {}

            def admit():
                while len(inflight) < 256:
                    key = next(iterator, None)
                    if key is None:
                        return
                    q = lookup[key[:2]]
                    p = protocol(q["domain"])
                    params = p.sampling(*key)
                    params["repetition_detection"] = RepetitionDetectionParams(**params["repetition_detection"])
                    params.update(output_kind=RequestOutputKind.FINAL_ONLY, detokenize=False)
                    rid = identity(key)
                    engine.add_request(rid, {"prompt_token_ids": q["prompt_ids"]}, SamplingParams(**params))
                    inflight[rid] = key

            admit()
            while inflight:
                for output in engine.step():
                    if not output.finished:
                        continue
                    key = inflight.pop(output.request_id)
                    q = lookup[key[:2]]
                    prediction = output.outputs[0]
                    ids = list(prediction.token_ids)
                    p = protocol(q["domain"])
                    row = dict(
                        id=identity([model["identity"], key]),
                        source_id=q["id"],
                        dataset=q["dataset"],
                        domain=q["domain"],
                        sample_index=key[2],
                        seed=p.sample_seed(*key),
                        protocol_id=p.protocol_id,
                        response_ids=ids,
                        response_tokens=len(ids),
                        finish_reason=prediction.finish_reason,
                        stop_reason=prediction.stop_reason,
                        text=tokenizer.decode(ids, skip_special_tokens=False),
                        terminal_tokens=[tokenizer.eos_token, tokenizer.pad_token],
                    )
                    responses.write(json.dumps(row, ensure_ascii=False) + "\n")
                    responses.flush()
                    done.add(key)
                    submit(row)
                drain()
                admit()
                dump(
                    dest / "progress.json",
                    dict(
                        generated=len(done),
                        graded=len(score_done),
                        expected=len(expected),
                        elapsed=time.time() - started,
                        gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    ),
                )
        while pending:
            drain(True)
    pool.shutdown(wait=True)
    if not summarize(dest, questions, selected):
        raise ValueError("Evaluation did not complete every declared slot")
    print("COMPLETE", label, job["phase"], len(expected), flush=True)


def run(root):
    plan = read(root / "plan.json")
    jobs = iter(enumerate(plan["jobs"]))
    lock = threading.Lock()
    results = {}

    def lane(gpu):
        while True:
            with lock:
                item = next(jobs, None)
            if item is None:
                return
            index, job = item
            path = root / "results" / job["label"] / job["phase"]
            if (path / "complete.json").exists():
                continue
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(gpu),
                VLLM_PLUGINS="",
                VLLM_WORKER_MULTIPROC_METHOD="spawn",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                TOKENIZERS_PARALLELISM="false",
            )
            log = root / f"job-{index:02d}.log"
            with log.open("a") as stream:
                child = subprocess.Popen(
                    [sys.executable, "-u", "-m", MODULE, "worker", "--root", str(root), "--index", str(index)],
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                with lock:
                    results[str(index)] = dict(**job, gpu=gpu, pid=child.pid, state="running")
                    dump(root / "queue.json", results)
                code = child.wait()
            with lock:
                results[str(index)].update(state="complete" if code == 0 else "failed", returncode=code)
                dump(root / "queue.json", results)
            if code:
                raise RuntimeError(f"Evaluation job {index} failed; inspect {log}")

    with futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in futures.as_completed([pool.submit(lane, gpu) for gpu in (0, 5, 6)]):
            result.result()
    dump(root / "complete.json", dict(jobs=len(plan["jobs"]), completed=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "worker"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--original", type=Path, default=Path("runs/teacher-construction-20261108/llm-v1"))
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == "prepare":
        prepare(root, args.original.resolve())
    elif args.stage == "worker":
        worker(root, args.index)
    else:
        run(root)


if __name__ == "__main__":
    main()
