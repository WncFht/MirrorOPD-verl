# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Real-model teacher-construction study with disjoint prompt populations.

Run stages in separate processes: vLLM owns generation memory, HF owns training.
All paths and scientific budgets are frozen in a run's protocol.json.
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def jsonlines(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def question_key(messages):
    content = "\n".join(message["content"] for message in messages if message["role"] == "user")
    return hashlib.sha256(" ".join(content.split()).encode()).hexdigest()


def bare_prompt(messages):
    return "\n\n".join(message["content"] for message in messages) + "\n\nAnswer:"


def prepare(args):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    root = args.root
    if (root / "protocol.json").exists():
        raise FileExistsError("Use a fresh run directory; protocol identities are immutable")
    repo = Path(__file__).resolve().parents[5]
    base = (repo.parent / "model/Qwen3-1.7B-Base").resolve()
    expert = (repo.parent / "model/Qwen3-30B-A3B-Instruct-2507").resolve()
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    expert_tokenizer = AutoTokenizer.from_pretrained(expert, local_files_only=True)
    rng = random.Random(20261108)
    source = repo / "runs/unified-evaluation-20261031/questions-v2"
    sources = {}
    excluded = []
    seen = set()

    def collect(path, domain):
        sources[str(path)] = digest(path)
        result = []
        values = pq.read_table(path).to_pylist()
        rng.shuffle(values)
        for value in values:
            messages = value["prompt"]
            key = question_key(messages)
            if key in seen:
                continue
            ids = tokenizer.encode(bare_prompt(messages), add_special_tokens=False)
            if len(ids) > 2048:
                excluded.append(key)
                continue
            seen.add(key)
            result.append(
                dict(
                    id=key,
                    domain=domain,
                    messages=messages,
                    prompt_ids=ids,
                    ground_truth=value["reward_model"]["ground_truth"]
                    if "reward_model" in value
                    else value["ground_truth"],
                    source=value.get("data_source", value.get("dataset")),
                    extra_info=value.get("extra_info", {}),
                )
            )
        return result

    # Benchmark question texts are reserved before any training population is selected.
    benchmarks = []
    for name in ["MATH-500", "AMC23", "AIME24", "AIME25", "AIME26"]:
        values = collect(source / "public" / f"{name}.parquet", "math-benchmark")
        for value in values:
            value["benchmark"] = name
        benchmarks.extend(values)
    math = collect(source / "training/math.parquet", "math")
    if len(math) < 4352:
        raise ValueError("Insufficient distinct, benchmark-disjoint math prompts")
    sets = dict(
        teacher=math[:2048], opd=math[2048:4096], dev=math[4096:4224], evaluation=math[4224:4352], benchmark=benchmarks
    )
    for domain in ["code", "science", "if"]:
        values = collect(source / "training" / f"{domain}.parquet", domain)
        if len(values) < 128:
            raise ValueError(f"Insufficient held-out {domain} population")
        sets["evaluation"].extend(values[:128])
    for name, values in sets.items():
        jsonlines(root / "prepared" / f"{name}.jsonl", values)
    protocol = dict(
        experiment="teacher-construction-llm-20261108-v1",
        seed=20261108,
        base=str(base),
        expert=str(expert),
        source_files=sources,
        tokenizer_vocab_sha256=hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
        expert_vocab_equal=tokenizer.get_vocab() == expert_tokenizer.get_vocab(),
        populations={
            name: dict(count=len(values), sha256=digest(root / "prepared" / f"{name}.jsonl"))
            for name, values in sets.items()
        },
        exclusions=dict(
            prompt_tokens_over_2048=len(excluded),
            rule="Reserve benchmark exact normalized user text first; global text de-duplication",
        ),
        generation=dict(
            candidates=8,
            cap=1024,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            prompt_limit=2048,
            stop_ids=[151643],
            expert_stop_ids=[151643, 151645],
            expert_cap=4096,
            expert_temperature=0.7,
            expert_top_p=0.8,
            expert_top_k=20,
        ),
        sft=dict(
            updates=128,
            batch=32,
            learning_rate=3e-6,
            weight_decay=0.0,
            normalization="Sum response CE / (32 * 1024), same constant across lengths; one w/K candidate; 2 epochs",
        ),
        ert=dict(
            updates=128,
            batch=16,
            candidates=8,
            learning_rate=1e-6,
            beta=0.02,
            reward="repository DAPO answer verifier in [0,1]; not learned expert reward",
            reuse=1,
            normalization="sequence REINFORCE, LOO reward-minus-sequence-KL; no PPO clipping",
        ),
        opd=dict(
            updates=256,
            batch=16,
            reuse=2,
            learning_rate=1e-6,
            weight_decay=0.01,
            normalization="sum full-vocabulary RKL / (16 * 1024)",
            rollout="fresh current HF actor each 2 updates, temperature 1, top_p 1",
        ),
        evaluation=dict(
            bank="Both fixed Base and own-teacher trajectories, first 256 response tokens",
            per_domain=128,
            scoring=(
                "per-question actual-token mean, then domain mean; both KL directions, "
                "chi-square and centered log-ratio variance"
            ),
            checkpoints=[0, 16, 64, 128, 256],
            benchmark="630 math questions, greedy, cap 2048; not full four-domain capability suite",
        ),
        arms=["RAW", "CCS-rho0", "CCS-rho1", "ERT-verifier"],
        numerical_contract="FP32 master, head, gradients and Adam moments; BF16 backbone autocast; TF32 off",
        limitations=[
            "single seed",
            "RAW cap4096 versus Base candidate cap1024; unmatched lengths and training-token budgets",
            "K8 not K16",
            "verifier reward isolates centering without expert semantic grader",
            "no claim of causal predictor validation from four arms",
            "vLLM BF16 sampling versus FP32-master HF scoring; rho0 controls finite-sample and numerical drift",
        ],
    )
    write(root / "protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False), flush=True)


def reward(text, truth):
    from verl.utils.reward_score.math_dapo import compute_score

    result = compute_score(text, str(truth))
    return float(result["acc"])


def generate(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    protocol = read(args.root / "protocol.json")
    values = rows(args.root / "prepared" / f"{args.population}.jsonl")[args.shard :: args.shards]
    expert = args.kind == "expert"
    model_path = protocol["expert" if expert else "base"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if expert:
        for value in values:
            # Native expert template never enters a Base training prompt.
            value["generation_prompt_ids"] = tokenizer.apply_chat_template(
                value["messages"],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
                enable_thinking=False,
            )
    output = args.root / "generated" / f"{args.population}-{args.kind}-{args.shard}.jsonl"
    existing = rows(output) if output.exists() else []
    completed = {value["id"] for value in existing}
    values = [value for value in values if value["id"] not in completed]
    if not values:
        return
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=6144 if expert else 4096,
        gpu_memory_utilization=0.88,
        max_num_seqs=64,
        seed=protocol["seed"] + args.shard,
        enforce_eager=True,
        enable_prefix_caching=True,
        trust_remote_code=False,
    )
    count = protocol["generation"]["candidates"] if args.population == "teacher" and not expert else 1
    cap = (
        256
        if args.population in {"evaluation", "dev"}
        else protocol["generation"].get("expert_cap", 4096)
        if expert
        else protocol["generation"]["cap"]
    )
    params = SamplingParams(
        n=count,
        temperature=0.7 if expert else 1.0,
        top_p=0.8 if expert else 1.0,
        top_k=20 if expert else -1,
        max_tokens=cap,
        seed=protocol["seed"] + args.shard,
        stop_token_ids=[151643, 151645] if expert else [151643],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with output.open("a") as stream:
        for first in range(0, len(values), 32):
            batch = values[first : first + 32]
            predictions = llm.generate(
                [{"prompt_token_ids": value.get("generation_prompt_ids", value["prompt_ids"])} for value in batch],
                params,
                use_tqdm=False,
            )
            for value, prediction in zip(batch, predictions, strict=True):
                value["candidates"] = []
                for result in prediction.outputs:
                    # Text exchange across expert and Base vocabularies; never compare their logits.
                    ids = list(result.token_ids)
                    value["candidates"].append(
                        dict(
                            text=result.text,
                            generated_ids=ids,
                            finish_reason=result.finish_reason,
                            tokens=len(ids),
                            reward=reward(result.text, value["ground_truth"])
                            if value["domain"].startswith("math")
                            else None,
                        )
                    )
                stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                json.dumps(
                    dict(
                        population=args.population,
                        kind=args.kind,
                        shard=args.shard,
                        completed=len(existing) + first + len(batch),
                        total=len(existing) + len(values),
                        elapsed=time.monotonic() - started,
                    )
                ),
                flush=True,
            )
    write(output.with_suffix(".complete.json"), dict(rows=len(rows(output)), sha256=digest(output)))


def numerical_setup(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)


def load_model(path, trainable=False):
    from types import SimpleNamespace

    from examples.mirroropd.research.backends import hf_models

    model = hf_models.load(SimpleNamespace(directory=Path(path)), trainable=True)
    return model.requires_grad_(trainable)


def train_mode(model):
    from types import SimpleNamespace

    from examples.mirroropd.research.backends import hf_models

    hf_models.training_mode(model, SimpleNamespace(activation_checkpointing=True))


def append_log(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def checkpoint(model, tokenizer, optimizer, path, step, extra_rng=None):
    import numpy as np
    import torch

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    # Every exported training state is independently resumable; writes never overwrite endpoints.
    torch.save(
        dict(
            optimizer=optimizer.state_dict(),
            step=step,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(),
            python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),
            sampler_rng=extra_rng,
        ),
        path / "training-state.pt",
    )
    write(path / "checkpoint-complete.json", dict(step=step, parameter_dtype="float32"))


def generated(root, population, kind):
    paths = sorted((root / "generated").glob(f"{population}-{kind}-*.jsonl"))
    values = [value for path in paths for value in rows(path)]
    expected = rows(root / "prepared" / f"{population}.jsonl")
    indexed = {value["id"]: value for value in values}
    if len(values) != len(indexed) or set(indexed) != {value["id"] for value in expected}:
        raise ValueError(f"Incomplete or duplicated {population}/{kind} generation")
    return [indexed[value["id"]] for value in expected]


def response_row(value, candidate, tokenizer, *, expert=False):
    ids = tokenizer.encode(candidate["text"], add_special_tokens=False) if expert else candidate["generated_ids"]
    ids = list(ids)
    if candidate["finish_reason"] == "stop" and (not ids or ids[-1] != 151643):
        ids.append(151643)
    if not ids:
        ids = [151643]
    return dict(
        id=value["id"],
        domain=value["domain"],
        prompt_ids=value["prompt_ids"],
        response_ids=ids[:4096],
        ground_truth=value["ground_truth"],
    )


def ce_backward(model, batch, coefficients, denominator):
    """Chunked exact CE VJP using the repository's response-aligned backbone path."""
    import torch
    import torch.nn.functional as functional

    from examples.mirroropd.research.backends.hf_models import hidden

    weight = model.get_output_embeddings().weight
    if weight.grad is None:
        weight.grad = torch.zeros_like(weight)
    total = 0.0
    for value, coefficient in zip(batch, coefficients, strict=True):
        states = hidden(model, [value], pad_id=151643)
        detached = states.detach().float()
        targets = torch.tensor(value["response_ids"], device=weight.device)
        dh = torch.empty_like(detached)
        for first in range(0, len(states), 64):
            stop = min(first + 64, len(states))
            with torch.no_grad():
                lp = functional.linear(detached[first:stop], weight).log_softmax(-1)
                selected = lp.gather(1, targets[first:stop, None]).squeeze(1)
                total -= float(selected.sum()) * coefficient / denominator
                dz = lp.exp()
                dz.scatter_add_(1, targets[first:stop, None], -torch.ones((stop - first, 1), device=weight.device))
                dz.mul_(coefficient / denominator)
                weight.grad.addmm_(dz.T, detached[first:stop])
                dh[first:stop] = dz @ weight
        states.backward(dh.to(states.dtype))
    return total


def sft(args):
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from examples.mirroropd.research.studies.fixed_transfer.teacher_construction_toy import pearson_weights

    protocol = read(args.root / "protocol.json")
    numerical_setup(protocol["seed"])
    tokenizer = AutoTokenizer.from_pretrained(protocol["base"], local_files_only=True)
    values = generated(args.root, "teacher", "expert" if args.arm == "RAW" else "base")
    radius = 0.0 if args.arm == "CCS-rho0" else 1.0
    rng = np.random.default_rng(protocol["seed"])
    probabilities = []
    audits = []
    for value in values:
        scores = [candidate["reward"] for candidate in value["candidates"]]
        weights = np.ones(len(scores)) if args.arm == "RAW" else pearson_weights(scores, radius=radius)
        probabilities.append(weights / len(weights))
        audits.append(
            dict(id=value["id"], rewards=scores, weights=weights.tolist(), pearson=float(np.mean((weights - 1) ** 2)))
        )
    out = args.root / "teachers" / args.arm
    if (out / "metrics.jsonl").exists():
        raise FileExistsError("Existing training trajectory; resume requires its optimizer/RNG state")
    out.mkdir(parents=True, exist_ok=True)
    jsonlines(out / "candidate-weights.jsonl", audits)
    model = load_model(protocol["base"], True)
    train_mode(model)
    config = protocol["sft"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.0, foreach=False)
    start = time.monotonic()
    presentations = 0
    for step in range(1, config["updates"] + 1):
        indices = [(step - 1) * config["batch"] + i for i in range(config["batch"])]
        batch = []
        for index in indices:
            index %= len(values)
            candidate = values[index]["candidates"][int(rng.choice(len(probabilities[index]), p=probabilities[index]))]
            batch.append(response_row(values[index], candidate, tokenizer, expert=args.arm == "RAW"))
        optimizer.zero_grad(set_to_none=True)
        loss = ce_backward(model, batch, [1.0] * len(batch), len(batch) * 1024)
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not np.isfinite(loss) or not np.isfinite(norm):
            raise FloatingPointError("Nonfinite SFT update")
        optimizer.step()
        presentations += sum(len(value["response_ids"]) for value in batch)
        append_log(
            out / "metrics.jsonl",
            dict(
                step=step,
                loss=loss,
                gradient_norm=norm,
                presented_tokens=presentations,
                elapsed=time.monotonic() - start,
            ),
        )
        if step in [16, 64, 128]:
            checkpoint(model, tokenizer, optimizer, out / f"step{step:04d}", step, rng.bit_generator.state)
    write(
        out / "complete.json",
        dict(
            updates=config["updates"],
            presented_tokens=presentations,
            all_fail_groups=sum(max(audit["rewards"]) == 0 for audit in audits),
        ),
    )


def sample_actor(model, values, cap=1024, count=1):
    import torch
    import torch.nn.functional as functional

    model.eval()
    result = []

    # Force the FP32 vocabulary projection while retaining BF16 decoder execution.
    def fp32_head(module, arguments, output):
        with torch.autocast("cuda", enabled=False):
            return functional.linear(arguments[0].float(), module.weight)

    hook = model.get_output_embeddings().register_forward_hook(fp32_head)
    try:
        expanded = [value for value in values for _ in range(count)]
        for first in range(0, len(expanded), 8):
            batch = expanded[first : first + 8]
            width = max(len(value["prompt_ids"]) for value in batch)
            ids = torch.tensor(
                [[151643] * (width - len(value["prompt_ids"])) + value["prompt_ids"] for value in batch], device="cuda"
            )
            mask = torch.tensor(
                [[0] * (width - len(value["prompt_ids"])) + [1] * len(value["prompt_ids"]) for value in batch],
                device="cuda",
            )
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_new_tokens=cap,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    eos_token_id=151643,
                    pad_token_id=151643,
                    use_cache=True,
                )
            for value, output in zip(batch, prediction, strict=True):
                response = output[width:].tolist()
                if 151643 in response:
                    response = response[: response.index(151643) + 1]
                result.append(
                    dict(
                        id=value["id"],
                        domain=value["domain"],
                        prompt_ids=value["prompt_ids"],
                        response_ids=response,
                        ground_truth=value["ground_truth"],
                    )
                )
    finally:
        hook.remove()
    train_mode(model)
    return result


def sequence_logprob(model, value):
    import torch
    import torch.nn.functional as functional

    from examples.mirroropd.research.backends.hf_models import hidden

    states = hidden(model, [value], pad_id=151643).float()
    weight = model.get_output_embeddings().weight.float()
    target = torch.tensor(value["response_ids"], device="cuda")
    total = torch.zeros((), device="cuda", dtype=torch.float64)
    for first in range(0, len(states), 64):
        lp = functional.linear(states[first : first + 64], weight).log_softmax(-1)
        total += lp.gather(1, target[first : first + 64, None]).sum(dtype=torch.float64)
    return float(total)


def ert(args):
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    protocol = read(args.root / "protocol.json")
    numerical_setup(protocol["seed"])
    tokenizer = AutoTokenizer.from_pretrained(protocol["base"], local_files_only=True)
    values = rows(args.root / "prepared/teacher.jsonl")
    out = args.root / "teachers" / args.arm
    if (out / "metrics.jsonl").exists():
        raise FileExistsError("Existing ERT trajectory")
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(protocol["base"], True)
    reference = load_model(protocol["base"])
    train_mode(model)
    config = protocol["ert"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.0, foreach=False)
    start = time.monotonic()
    for step in range(1, config["updates"] + 1):
        prompts = values[(step - 1) * config["batch"] : step * config["batch"]]
        batch = sample_actor(model, prompts, count=config["candidates"])
        rewards = np.array(
            [
                reward(tokenizer.decode(value["response_ids"], skip_special_tokens=True), value["ground_truth"])
                for value in batch
            ]
        )
        with torch.no_grad():
            gaps = np.array([sequence_logprob(model, value) - sequence_logprob(reference, value) for value in batch])
        utility = (rewards - config["beta"] * gaps).reshape(len(prompts), config["candidates"])
        advantages = utility - (utility.sum(axis=1, keepdims=True) - utility) / (config["candidates"] - 1)
        jsonlines(
            out / f"rollout-{step:04d}.jsonl",
            [
                dict(**value, reward=float(r), logratio=float(g), advantage=float(a))
                for value, r, g, a in zip(batch, rewards, gaps, advantages.ravel(), strict=True)
            ],
        )
        optimizer.zero_grad(set_to_none=True)
        loss = ce_backward(model, batch, advantages.ravel().tolist(), len(batch))
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not np.isfinite(loss) or not np.isfinite(norm):
            raise FloatingPointError("Nonfinite ERT update")
        optimizer.step()
        append_log(
            out / "metrics.jsonl",
            dict(
                step=step,
                loss=loss,
                gradient_norm=norm,
                reward=float(rewards.mean()),
                sequence_kl_sample_mean=float(gaps.mean()),
                nonzero_advantage_fraction=float(np.mean(advantages != 0)),
                sampled_tokens=sum(len(value["response_ids"]) for value in batch),
                elapsed=time.monotonic() - start,
            ),
        )
        if step in [16, 64, 128]:
            checkpoint(model, tokenizer, optimizer, out / f"step{step:04d}", step)
    write(out / "complete.json", dict(updates=config["updates"], reward="DAPO verifier", beta=config["beta"]))


def evaluate_bank(model, teacher, bank, path, step):
    """Complete native vocabulary; each question receives equal domain weight."""
    import torch
    import torch.nn.functional as functional

    from examples.mirroropd.research.backends.hf_models import hidden

    model.eval()
    with torch.no_grad():
        wp = model.get_output_embeddings().weight.float()
        wq = teacher.get_output_embeddings().weight.float()
        for value in bank:
            hp = hidden(model, [value], pad_id=151643).float()
            hq = hidden(teacher, [value], pad_id=151643).float()
            sums = dict(rkl=0.0, fkl=0.0, chi2=0.0, centered_logratio_variance=0.0)
            for first in range(0, len(hp), 64):
                lp = functional.linear(hp[first : first + 64], wp).log_softmax(-1)
                lq = functional.linear(hq[first : first + 64], wq).log_softmax(-1)
                p, q = lp.exp(), lq.exp()
                delta = lq - lp
                centered = delta - (p * delta).sum(-1, keepdim=True)
                sums["rkl"] += float((p * (-delta)).sum(dtype=torch.float64))
                sums["fkl"] += float((q * delta).sum(dtype=torch.float64))
                # FP64 tail accumulation; no clipped or top-k density ratio.
                sums["chi2"] += float(torch.expm1(torch.logsumexp((2 * lq - lp).double(), dim=-1)).sum())
                sums["centered_logratio_variance"] += float((p * centered.square()).sum(dtype=torch.float64))
            append_log(
                path,
                dict(
                    step=step,
                    id=value["id"],
                    domain=value["domain"],
                    tokens=len(hp),
                    **{key: total / len(hp) for key, total in sums.items()},
                ),
            )
    train_mode(model)


def opd(args):
    from types import SimpleNamespace

    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from examples.mirroropd.research.backends import hf_models, hf_rkl

    protocol = read(args.root / "protocol.json")
    numerical_setup(protocol["seed"])
    tokenizer = AutoTokenizer.from_pretrained(protocol["base"], local_files_only=True)
    teacher_path = args.root / "teachers" / args.arm / "step0128"
    if not (teacher_path / "checkpoint-complete.json").exists():
        raise FileNotFoundError("Teacher endpoint incomplete")
    out = args.root / "opd" / args.arm
    if (out / "metrics.jsonl").exists():
        raise FileExistsError("Existing OPD trajectory")
    out.mkdir(parents=True, exist_ok=True)
    teacher = load_model(teacher_path)
    model = load_model(protocol["base"], True)
    train_mode(model)
    config = protocol["opd"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"], foreach=False
    )
    prompts = rows(args.root / "prepared/opd.jsonl")
    bank = [
        response_row(value, value["candidates"][0], tokenizer) for value in generated(args.root, "evaluation", "base")
    ]
    for value in bank:
        value["response_ids"] = value["response_ids"][:256]
    dev_bank = [response_row(value, value["candidates"][0], tokenizer) for value in generated(args.root, "dev", "base")]
    teacher_bank = sample_actor(teacher, rows(args.root / "prepared/evaluation.jsonl"), cap=256)
    jsonlines(out / "teacher-bank.jsonl", teacher_bank)
    # Identity gate scores the teacher against itself before the train trajectory.
    evaluate_bank(teacher, teacher, bank[:4], out / "teacher-copy.jsonl", 0)
    identity = rows(out / "teacher-copy.jsonl")
    if max(abs(value["rkl"]) for value in identity) > 1e-7:
        raise ArithmeticError("Teacher-copy KL gate failed")
    evaluate_bank(model, teacher, bank, out / "heldout.jsonl", 0)
    evaluate_bank(model, teacher, teacher_bank, out / "heldout-teacher-bank.jsonl", 0)
    evaluate_bank(model, teacher, dev_bank, out / "development.jsonl", 0)
    start = time.monotonic()
    training = SimpleNamespace(denominator=config["batch"] * 1024)
    execution = SimpleNamespace(microbatch=1, token_chunk=64)
    head = teacher.get_output_embeddings().weight.detach().float()
    for outer in range(config["updates"] // config["reuse"]):
        selected = prompts[outer * config["batch"] : (outer + 1) * config["batch"]]
        batch = sample_actor(model, selected)
        for index, value in enumerate(batch):
            value["question_id"] = value["id"]
            value["id"] = index
        jsonlines(out / f"rollout-{outer:04d}.jsonl", batch)
        with torch.no_grad():
            states = {value["id"]: hf_models.hidden(teacher, [value], pad_id=151643).detach() for value in batch}
            qcache = hf_rkl.log_probabilities(head, states, batch, chunk=64)
        for reuse in range(config["reuse"]):
            step = outer * config["reuse"] + reuse + 1
            optimizer.zero_grad(set_to_none=True)
            loss = hf_rkl.backward(model, qcache, batch, training, execution, 151643)
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            if not np.isfinite(loss) or not np.isfinite(norm):
                raise FloatingPointError("Nonfinite OPD update")
            optimizer.step()
            append_log(
                out / "metrics.jsonl",
                dict(
                    step=step,
                    outer=outer,
                    reuse=reuse,
                    loss=loss,
                    gradient_norm=norm,
                    sampled_tokens=sum(len(value["response_ids"]) for value in batch),
                    elapsed=time.monotonic() - start,
                ),
            )
        del qcache, states
        if step in [4, 16, 64, 128, 256]:
            evaluate_bank(model, teacher, dev_bank, out / "development.jsonl", step)
        if step in [16, 64, 128, 256]:
            evaluate_bank(model, teacher, bank, out / "heldout.jsonl", step)
            evaluate_bank(model, teacher, teacher_bank, out / "heldout-teacher-bank.jsonl", step)
            checkpoint(model, tokenizer, optimizer, out / f"step{step:04d}", step)
    write(out / "complete.json", dict(updates=config["updates"], teacher=str(teacher_path)))


def benchmark(args):
    import concurrent.futures
    import multiprocessing
    from collections import defaultdict

    from vllm import LLM, SamplingParams

    from examples.mirroropd.research.core.math_grading import grade, init_grader

    protocol = read(args.root / "protocol.json")
    model = (
        protocol["base"]
        if args.label == "base"
        else str(
            args.root
            / ("teachers" if args.label == "teacher" else "opd")
            / args.arm
            / ("step0128" if args.label == "teacher" else "step0256")
        )
    )
    out = args.root / "benchmarks" / ("base" if args.label == "base" else f"{args.arm}-{args.label}")
    if (out / "complete.json").exists():
        return
    values = rows(args.root / "prepared/benchmark.jsonl")
    llm = LLM(
        model=model,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.88,
        max_num_seqs=64,
        enforce_eager=True,
        seed=protocol["seed"],
    )
    params = SamplingParams(temperature=0.0, max_tokens=2048, stop_token_ids=[151643])
    scored = []
    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn"), initializer=init_grader
    )
    for first in range(0, len(values), 32):
        batch = values[first : first + 32]
        outputs = llm.generate([{"prompt_token_ids": value["prompt_ids"]} for value in batch], params, use_tqdm=False)
        grading = []
        metadata = []
        for value, output in zip(batch, outputs, strict=True):
            prediction = output.outputs[0]
            metadata.append(
                dict(
                    id=value["id"],
                    benchmark=value["benchmark"],
                    text=prediction.text,
                    tokens=len(prediction.token_ids),
                    finish_reason=prediction.finish_reason,
                    dapo_verifier_reward=reward(prediction.text, value["ground_truth"]),
                )
            )
            grading.append(
                dict(
                    data_source=value["benchmark"],
                    output=prediction.text,
                    gts=value["ground_truth"],
                    response_ids=list(prediction.token_ids),
                )
            )
        for value, result in zip(metadata, pool.map(grade, grading), strict=True):
            scored.append(dict(**value, reward=float(result["acc"]), canonical_grade=result))
        jsonlines(out / "responses.jsonl", scored)
        print(json.dumps(dict(label=args.label, arm=args.arm, completed=len(scored))), flush=True)
    pool.shutdown(wait=True)
    grouped = defaultdict(list)
    for value in scored:
        grouped[value["benchmark"]].append(value["reward"])
    write(
        out / "complete.json",
        dict(
            model=model,
            greedy=True,
            response_cap=2048,
            metrics={key: dict(n=len(value), accuracy=sum(value) / len(value)) for key, value in grouped.items()},
        ),
    )


def gpu_gate(args):
    """Exercise real 1.7B forward/backward, sampling and full-vocab updates."""
    from types import SimpleNamespace

    import torch
    from transformers import AutoTokenizer

    from examples.mirroropd.research.backends import hf_models, hf_rkl

    protocol = read(args.root / "protocol.json")
    numerical_setup(protocol["seed"])
    tokenizer = AutoTokenizer.from_pretrained(protocol["base"], local_files_only=True)
    model = load_model(protocol["base"], True)
    reference = load_model(protocol["base"])
    train_mode(model)
    prompts = rows(args.root / "prepared/teacher.jsonl")[:2]
    batch = sample_actor(model, prompts, cap=32)
    for index, value in enumerate(batch):
        value["id"] = index
    with torch.no_grad():
        gaps = [sequence_logprob(model, value) - sequence_logprob(reference, value) for value in batch]
    if max(map(abs, gaps)) > 1e-4:
        raise ArithmeticError(f"Initial actor/reference numerical mismatch: {gaps}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.0, foreach=False)
    parameter = next(model.parameters())
    before = parameter.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    ce = ce_backward(model, batch, [1.0, -0.5], len(batch) * 32)
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    change = float((parameter - before).abs().max())
    del before
    if not (change > 0 and norm > 0):
        raise ArithmeticError("FP32 parameter update gate failed")
    with torch.no_grad():
        head = reference.get_output_embeddings().weight.detach()
        states = {value["id"]: hf_models.hidden(reference, [value], pad_id=151643) for value in batch}
        qcache = hf_rkl.log_probabilities(head, states, batch, chunk=32)
    optimizer.zero_grad(set_to_none=True)
    rkl = hf_rkl.backward(
        model, qcache, batch, SimpleNamespace(denominator=64), SimpleNamespace(microbatch=1, token_chunk=32), 151643
    )
    rkl_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    if not (rkl > 0 and rkl_norm > 0):
        raise ArithmeticError("Full-vocabulary RKL update gate failed")
    checkpoint(model, tokenizer, optimizer, args.root / "gate-checkpoint", 2)
    evidence = dict(
        status="passed",
        ce=ce,
        ce_gradient_norm=norm,
        rkl=rkl,
        rkl_gradient_norm=rkl_norm,
        max_parameter_change=change,
        initial_sequence_logratio=gaps,
        parameter_dtype=str(parameter.dtype),
        moment_dtype=str(optimizer.state[parameter]["exp_avg"].dtype),
        max_memory_allocated=torch.cuda.max_memory_allocated(),
        source_sha256=digest(__file__),
    )
    write(args.root / "gpu-gate.json", evidence)
    print(json.dumps(evidence), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "generate", "sft", "ert", "opd", "benchmark", "gpu-gate"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--population", default="teacher")
    parser.add_argument("--kind", choices=["base", "expert"], default="base")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--arm", default="RAW")
    parser.add_argument("--label", default="teacher", choices=["base", "teacher", "student"])
    args = parser.parse_args()
    args.root = args.root.resolve()
    if (args.root / "source").is_dir():
        import sys

        sys.path.insert(0, str(args.root / "source"))
    if args.stage in {"sft", "ert"}:
        import fcntl

        # One real-model gate owns its GPU; peers have not loaded models yet.
        with (args.root / "gpu-gate.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            if not (args.root / "gpu-gate.json").exists():
                gpu_gate(args)
            if read(args.root / "gpu-gate.json")["status"] != "passed":
                raise RuntimeError("Real-model numerical gate failed")
    {
        "prepare": prepare,
        "generate": generate,
        "sft": sft,
        "ert": ert,
        "opd": opd,
        "benchmark": benchmark,
        "gpu-gate": gpu_gate,
    }[args.stage](args)
    if args.stage == "opd":
        import gc
        import subprocess
        import sys

        import torch

        gc.collect()
        torch.cuda.empty_cache()
        for label in (["base"] if args.arm == "RAW" else []) + ["teacher", "student"]:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    __file__,
                    "benchmark",
                    "--root",
                    str(args.root),
                    "--arm",
                    args.arm,
                    "--label",
                    label,
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
