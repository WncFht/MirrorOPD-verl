# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Full-parameter LLM stage probe on a one-token arithmetic decision task.

This isolates conditional fitting. It does not measure long-rollout coverage.
RL integrates a known binary token reward exactly over the native vocabulary.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-steps", type=int, default=8)
    parser.add_argument("--student-steps", type=int, default=48)
    parser.add_argument("--teacher-step-norm", type=float, default=0.0002)
    parser.add_argument("--student-lr", type=float, default=0.01)
    parser.add_argument("--student-max-step-norm", type=float, default=0.0002)
    parser.add_argument("--rl-kl-coefficient", type=float, default=0.0001)
    parser.add_argument("--memory-fraction", type=float, default=0.22)
    parser.add_argument("--teacher-cache", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "metrics.jsonl").exists():
        raise FileExistsError("Use a fresh output directory to preserve earlier evidence")
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    pairs = [(13, 24), (18, 31), (22, 45), (29, 36), (34, 42), (41, 26), (46, 33), (53, 25)]
    pairs += [(14, 23), (19, 32), (23, 46), (28, 37), (35, 43), (42, 27), (47, 34), (52, 26)]
    prompts = [f"What is the last digit of {a} + {b}? Answer with one digit.\nAnswer:" for a, b in pairs]
    labels = [tok.encode(str((a + b) % 10), add_special_tokens=False) for a, b in pairs]
    assert all(len(label) == 1 for label in labels), labels
    labels = torch.tensor([label[0] for label in labels], device="cuda")
    inputs = tok(prompts, padding=True, return_tensors="pt").to("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
    ).to("cuda")
    model.eval()
    model.config.use_cache = False
    params = list(model.parameters())
    assert sum(p.numel() for p in params) >= 1_000_000_000
    base = {key: val.detach().cpu().clone() for key, val in model.state_dict().items()}
    started = time.monotonic()
    records = []

    def emit(record):
        record["elapsed_seconds"] = time.monotonic() - started
        records.append(record)
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    def log_probs(indices):
        batch = {key: val[indices] for key, val in inputs.items()}
        position_ids = batch["attention_mask"].long().cumsum(-1) - 1
        position_ids.masked_fill_(batch["attention_mask"] == 0, 0)
        hidden = model.model(**batch, position_ids=position_ids, use_cache=False).last_hidden_state[:, -1]
        return model.lm_head(hidden).log_softmax(-1)

    @torch.no_grad()
    def cache():
        return torch.cat([log_probs(slice(i, i + 2)).cpu() for i in range(0, 16, 2)]).to("cuda")

    q0_log = cache()
    torch.save(q0_log.cpu(), args.output / "base-logprobs.pt")

    def loss_for(indices, kind, reference):
        lp = log_probs(indices)
        if kind == "sft":
            return -lp.gather(-1, labels[indices, None]).mean()
        kl = (lp.exp() * (lp - reference[indices])).sum(-1).mean()
        if kind == "rl":
            reward = lp.gather(-1, labels[indices, None]).exp().mean()
            return -reward + args.rl_kl_coefficient * kl
        return kl

    def train_loss(kind, reference, backward=False):
        total = 0.0
        for start in range(0, 8, 2):
            loss = loss_for(slice(start, start + 2), kind, reference) / 4
            if backward:
                loss.backward()
            total += loss.detach().item()
        return total

    @torch.no_grad()
    def eval_record(phase, step, reference):
        lp = cache()
        risk = (lp.exp() * (lp - reference)).sum(-1)
        prob = lp.gather(-1, labels[:, None]).exp().squeeze(-1)
        greedy = (lp.argmax(-1) == labels).float()
        emit(
            dict(
                phase=phase,
                step=step,
                train_kl=risk[:8].mean().item(),
                held_kl=risk[8:].mean().item(),
                train_reward=prob[:8].mean().item(),
                held_reward=prob[8:].mean().item(),
                held_greedy=greedy[8:].mean().item(),
                peak_memory_bytes=torch.cuda.max_memory_allocated(),
            )
        )

    def grad_norm():
        return torch.stack([p.grad.norm().square() for p in params if p.grad is not None]).sum().sqrt().item()

    @torch.no_grad()
    def move(scale):
        for p in params:
            if p.grad is not None:
                p.add_(p.grad, alpha=scale)

    def teach(kind, reference):
        for step in range(args.teacher_steps):
            model.zero_grad(set_to_none=True)
            before = train_loss(kind, reference, backward=True)
            norm = grad_norm()
            rate = args.teacher_step_norm / max(norm, 1e-12)
            # Backtracking keeps the prescribed small norm an upper bound.
            for attempt in range(12):
                move(-rate)
                with torch.no_grad():
                    after = train_loss(kind, reference)
                if after <= before:
                    break
                move(rate)
                rate /= 2
            else:
                raise RuntimeError(f"Teacher line search failed: {kind}")
            emit(dict(phase=f"teacher_{kind}", step=step + 1, loss=after, grad_norm=norm, step_size=rate))
        model.zero_grad(set_to_none=True)

    manifest = dict(
        model=str(args.model.resolve()),
        parameter_count=sum(p.numel() for p in params),
        vocab_size=model.config.vocab_size,
        dtype="float32",
        tf32=False,
        seed=42,
        train_size=8,
        held_size=8,
        pairs=pairs,
        prompts=prompts,
        labels=labels.cpu().tolist(),
        teacher_steps=args.teacher_steps if args.teacher_cache is None else None,
        student_steps=args.student_steps,
        teacher_max_step_norm=args.teacher_step_norm if args.teacher_cache is None else None,
        student_initial_lr=args.student_lr,
        student_max_step_norm=args.student_max_step_norm,
        rl_kl_coefficient=args.rl_kl_coefficient if args.teacher_cache is None else None,
        rl_reference="stage initialization: base for RL; SFT for SFT+RL",
        optimizer="full-batch SGD with deterministic loss backtracking; no momentum/decay",
        scope="H=1 exact full-vocabulary conditional objective; not a long-rollout experiment",
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    if args.teacher_cache is not None:
        parent = json.loads((args.teacher_cache.parent / "manifest.json").read_text())
        for key in ("model", "prompts", "labels", "vocab_size", "parameter_count"):
            assert parent[key] == manifest[key], key
        manifest["teacher_source"] = dict(
            path=str(args.teacher_cache.resolve()),
            sha256=hashlib.sha256(args.teacher_cache.read_bytes()).hexdigest(),
            manifest=parent,
        )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    eval_record("base", 0, q0_log)
    if args.teacher_cache is None:
        teach("sft", q0_log)
        sft = {key: val.detach().cpu().clone() for key, val in model.state_dict().items()}
        targets = {"sft": cache()}
        eval_record("sft_teacher", args.teacher_steps, q0_log)
        teach("rl", targets["sft"])
        targets["sft_rl"] = cache()
        eval_record("sft_rl_teacher", args.teacher_steps * 2, q0_log)
        model.load_state_dict(base)
        teach("rl", q0_log)
        targets["rl"] = cache()
        eval_record("rl_teacher", args.teacher_steps, q0_log)
        # Cache equality is an executable zero-loss witness for the SFT checkpoint.
        model.load_state_dict(sft)
        eval_record("sft_copy_control", 0, targets["sft"])
        del sft
    else:
        targets = torch.load(args.teacher_cache, map_location="cuda", weights_only=True)
        assert set(targets) == {"sft", "rl", "sft_rl"}
        assert all(value.shape == q0_log.shape for value in targets.values())
    torch.save({key: val.cpu() for key, val in targets.items()}, args.output / "teacher-logprobs.pt")
    for name in ("sft", "rl", "sft_rl"):
        model.load_state_dict(base)
        reference = targets[name]
        eval_record(f"opd_{name}", 0, reference)
        rate = args.student_lr
        for step in range(args.student_steps):
            model.zero_grad(set_to_none=True)
            before = train_loss("opd", reference, backward=True)
            norm = grad_norm()
            rate = min(rate, args.student_max_step_norm / max(norm, 1e-12))
            for attempt in range(16):
                move(-rate)
                with torch.no_grad():
                    after = train_loss("opd", reference)
                if after <= before + 1e-8:
                    break
                move(rate)
                rate /= 2
            else:
                raise RuntimeError(f"OPD line search failed: {name}")
            emit(dict(phase=f"update_{name}", step=step + 1, loss=after, grad_norm=norm, step_size=rate))
            if (step + 1) % 8 == 0:
                eval_record(f"opd_{name}", step + 1, reference)
        model.zero_grad(set_to_none=True)
    (args.output / "complete.json").write_text(json.dumps(dict(completed=True, records=len(records)), indent=2))


if __name__ == "__main__":
    main()
