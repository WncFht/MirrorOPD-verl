# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Frozen Qwen termination probes; no optimizer, rollout service, or training edits.

Compare Base/4B/30B on identical response prefixes and two prompt variants.
Archived cases retain exact token IDs. Text-log cases are explicitly re-encoded,
so they are controlled probes rather than replay of the original sampler state.
"""

import argparse
import gc
import gzip
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.mirroropd.research.core.io import dump

STOP = [151643, 151645]
MODELS = {
    "initial_student": "Qwen3-1.7B-Base",
    "teacher_4b": "Qwen3-4B",
    "teacher_30b": "Qwen3-30B-A3B-Instruct-2507",
}


def text_case(tok, row, category, case_id, positions=None):
    logged = row["input"]
    boundary = "\nassistant\n"
    assert logged.startswith("user\n") and boundary in logged
    user = logged[5 : logged.rindex(boundary)]
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert tok.decode(prompt, skip_special_tokens=True) == logged
    response = tok.encode(row["output"], add_special_tokens=False)
    return dict(
        id=case_id,
        category=category,
        token_source="reencoded_text_log",
        prompt_ids=prompt,
        response_ids=response,
        positions=positions or [len(response)],
        score=row.get("score"),
        output_end=row["output"][-700:],
    )


def build_cases(args, tok):
    with gzip.open(args.archived_cases, "rt") as stream:
        archived = json.load(stream)
    cases = []
    for i, row in enumerate(archived):
        if row["category"] != "early_terminal":
            continue
        assert row["response_ids"][-1] in STOP
        cases.append(
            dict(
                id=f"archived-stop-{i}",
                category="archived_terminal",
                token_source="original_ids",
                prompt_ids=row["prompt_ids"],
                response_ids=row["response_ids"][:-1],
                positions=[len(row["response_ids"]) - 1],
                score=row["score"],
                output_end=row["output_ending"],
            )
        )
    with (args.e54 / "rollouts/1.jsonl").open() as stream:
        initial = [json.loads(line) for line in stream]
    for i, row in enumerate(sorted(initial, key=lambda r: r["input"])):
        ids = tok.encode(row["output"], add_special_tokens=False)
        if not 32 <= len(ids) <= 2048:
            continue
        cases.append(text_case(tok, row, "e54_initial_short_end", f"e54-first-{i}"))
    selected = 0
    with (args.e41 / "rollouts/10.jsonl").open() as stream:
        successful = sorted([json.loads(line) for line in stream], key=lambda r: r["input"])
    for i, row in enumerate(successful):
        ids = tok.encode(row["output"], add_special_tokens=False)
        if row["score"] != 1 or not 64 <= len(ids) <= 2048:
            continue
        cases.append(text_case(tok, row, "e41_correct_end", f"e41-correct-{i}"))
        selected += 1
        if selected == 10:
            break
    for i, (question, answer) in enumerate(
        [("What is 1+1?", "2"), ("What is 7*8?", "56"), ("What is 100-37?", "63"), ("Solve x+3=5.", "2")]
    ):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        response = tok.encode(f"The answer is \\boxed{{{answer}}}.", add_special_tokens=False)
        cases.append(
            dict(
                id=f"synthetic-{i}",
                category="synthetic_correct",
                token_source="synthetic",
                prompt_ids=prompt,
                response_ids=response,
                positions=[len(response)],
                score=1,
            )
        )
    with (args.e54 / "rollouts/29.jsonl").open() as stream:
        later = [json.loads(line) for line in stream]
    candidates = []
    for row in later:
        lines = [s.strip() for s in row["output"].split("\n") if len(s.strip()) >= 20]
        repetition = max((lines.count(line) for line in set(lines)), default=0)
        if repetition >= 4:
            candidates.append((repetition, row))
    for i, (_, row) in enumerate(sorted(candidates, key=lambda p: -p[0])[:6]):
        case = text_case(tok, row, "e54_late_loop", f"e54-loop-{i}")
        n = min(len(case["response_ids"]), 7168)
        case["positions"] = sorted({min(x, n) for x in [0, 128, 512, 1024, 2048, 4096, 6144, n]})
        case["response_ids"] = case["response_ids"][:n]
        cases.append(case)
    dump(args.out / "cases.json", cases)
    return cases


def score_model(args, name, tok, cases):
    start = time.time()
    print("loading", name, flush=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.models / MODELS[name],
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2",
            local_files_only=True,
        )
        .eval()
        .requires_grad_(False)
    )
    head = model.get_output_embeddings().weight.detach().float()
    suffix = tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    records, logits = [], []
    for i, case in enumerate(cases):
        assert case["prompt_ids"][-len(suffix) :] == suffix
        for variant in ["training_prompt", "without_empty_think"]:
            prompt = case["prompt_ids"] if variant == "training_prompt" else case["prompt_ids"][: -len(suffix)]
            positions = case["positions"]
            tokens = prompt + case["response_ids"][: max(positions)]
            indexes = [len(prompt) - 1 + j for j in positions]
            with torch.inference_mode():
                inputs = torch.tensor([tokens], device="cuda:0")
                hidden = model.model(input_ids=inputs, use_cache=False).last_hidden_state[0, indexes]
                z = torch.nn.functional.linear(hidden.float(), head)
                lp = z.log_softmax(-1)
                prob = lp.exp()
                for j, position in enumerate(positions):
                    values, ids = prob[j].topk(5)
                    target = case["response_ids"][position] if position < len(case["response_ids"]) else None
                    records.append(
                        dict(
                            model=name,
                            case_id=case["id"],
                            category=case["category"],
                            variant=variant,
                            position=position,
                            p_stop=prob[j, STOP].sum().item(),
                            p_eot=prob[j, STOP[0]].item(),
                            p_im_end=prob[j, STOP[1]].item(),
                            next_observed_token=target,
                            p_next_observed=prob[j, target].item() if target is not None else None,
                            top_ids=ids.tolist(),
                            top_text=[tok.decode([v]) for v in ids.tolist()],
                            top_probs=values.tolist(),
                        )
                    )
                    logits.append(lp[j].cpu())
                del inputs, hidden, z, lp, prob
        if i % 10 == 0:
            print(name, i + 1, "/", len(cases), "elapsed", round(time.time() - start, 1), flush=True)
    torch.save(torch.stack(logits), args.out / f"{name}-logprobs.pt")
    dump(args.out / f"{name}.json", records)
    info = {"elapsed_seconds": time.time() - start, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
    dump(args.out / f"{name}-runtime.json", info)
    del model, head, logits
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("done", name, info, flush=True)


def compare(args):
    student = (
        torch.load(args.out / "initial_student-logprobs.pt", map_location="cpu", weights_only=True)
        .double()
        .log_softmax(-1)
    )
    records = json.loads((args.out / "initial_student.json").read_text())
    ps = student.exp()
    sp = ps[:, STOP].sum(-1)
    result = []
    for teacher in ["teacher_4b", "teacher_30b"]:
        teacher_rows = json.loads((args.out / f"{teacher}.json").read_text())
        lt = (
            torch.load(args.out / f"{teacher}-logprobs.pt", map_location="cpu", weights_only=True)
            .double()
            .log_softmax(-1)
        )
        qt = lt[:, STOP].exp().sum(-1)
        gap = student - lt
        gap[:, STOP] = (sp.log() - qt.log())[:, None]
        kl = (ps * gap).sum(-1)
        # Derivative of coarsened RKL when both student stop logits shift by +u.
        # Positive derivative means gradient descent suppresses stopping.
        stop_grad = sp * (sp.log() - qt.log() - kl)
        for i, (p, q) in enumerate(zip(records, teacher_rows, strict=True)):
            for key in ["case_id", "variant", "position"]:
                assert p[key] == q[key]
            result.append(
                dict(
                    case_id=p["case_id"],
                    category=p["category"],
                    variant=p["variant"],
                    position=p["position"],
                    teacher=teacher,
                    student_p_stop=p["p_stop"],
                    teacher_p_stop=q["p_stop"],
                    coarse_kl=kl[i].item(),
                    stop_logit_direction=stop_grad[i].item(),
                )
            )
    dump(args.out / "paired-stop-gradients.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--e41", type=Path, required=True)
    parser.add_argument("--e54", type=Path, required=True)
    parser.add_argument("--archived-cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(args.models / MODELS["initial_student"], local_files_only=True)
    for model_name in MODELS.values():
        other = AutoTokenizer.from_pretrained(args.models / model_name, local_files_only=True)
        assert tok.get_vocab() == other.get_vocab(), model_name
    cases = build_cases(args, tok)
    dump(
        args.out / "protocol.json",
        {
            "models": MODELS,
            "device": "cuda:0",
            "backbone_dtype": "bf16",
            "head_dtype": "fp32",
            "case_count": len(cases),
            "stop_ids": STOP,
            "prompt_variants": ["training_prompt", "without_empty_think"],
            "scope": "Frozen counterfactual probes; no trained E54 checkpoint available; no optimizer or new training.",
        },
    )
    for name in MODELS:
        score_model(args, name, tok, cases)
    compare(args)


if __name__ == "__main__":
    main()
