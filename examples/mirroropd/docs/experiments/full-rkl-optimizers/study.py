#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Small full-parameter, full-vocabulary RKL optimizer experiment.

Reuses MirrorOPD's production full_reverse_kl endpoint and its real 1.5B assets.
BF16 model/gradients; persistent FP32 CPU master weights and optimizer states.
Muon uses native torch.optim.Muon per matrix on the GPU, offloading its state
between calls; embeddings/head/norms use native CPU AdamW. No production edits.
"""

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.trainer.distillation.full_vocab import full_reverse_kl, project
from verl.trainer.distillation.numerics import kl_rows


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def norm(xs):
    return math.sqrt(sum(float(x.square().sum()) for x in xs if x is not None))


class CPUOptim:
    def __init__(self, names, masters, arm, lr, device):
        self.names, self.params, self.arm, self.lr, self.device = names, masters, arm, lr, device
        self.matrix = [p.ndim == 2 and ".layers." in n for n, p in zip(names, masters, strict=False)]
        self.muon_states = {}
        selected = [p for p, flag in zip(masters, self.matrix, strict=False) if arm != "muon" or not flag]
        if arm == "sgd":
            self.opt = torch.optim.SGD(selected, lr=lr, momentum=0, weight_decay=0, foreach=True)
        else:
            self.opt = torch.optim.AdamW(
                selected, lr=lr, betas=(0.0 if arm == "adam0" else 0.9, 0.999), eps=1e-8, weight_decay=0, foreach=True
            )

    def set_lr(self, lr):
        self.lr = lr
        for group in self.opt.param_groups:
            group["lr"] = lr

    def momentum_cosine(self, grads):
        dot = m2 = g2 = 0.0
        for i, (p, g) in enumerate(zip(self.params, grads, strict=False)):
            if g is None:
                continue
            state = self.muon_states.get(i, {}) if self.arm == "muon" and self.matrix[i] else self.opt.state.get(p, {})
            m = state.get("momentum_buffer", state.get("exp_avg"))
            if m is not None:
                dot += float((m * g).sum())
                m2 += float(m.square().sum())
                g2 += float(g.square().sum())
        return dot / math.sqrt(m2 * g2) if m2 * g2 else None

    @torch.no_grad()
    def step(self, grads):
        for p, g in zip(self.params, grads, strict=False):
            p.grad = g
        self.opt.step()
        if self.arm == "muon":
            for i, (p, g, matrix) in enumerate(zip(self.params, grads, self.matrix, strict=False)):
                if not matrix or g is None:
                    continue
                work = torch.nn.Parameter(p.to(self.device))
                work.grad = g.to(self.device)
                opt = torch.optim.Muon(
                    [work], lr=self.lr, momentum=0.95, nesterov=True, weight_decay=0, adjust_lr_fn="match_rms_adamw"
                )
                opt.state[work] = {k: v.to(self.device) for k, v in self.muon_states.get(i, {}).items()}
                opt.step()
                p.copy_(work.cpu())
                self.muon_states[i] = {k: v.cpu() for k, v in opt.state[work].items()}
                del opt, work
        self.opt.zero_grad(set_to_none=True)
        for p in self.params:
            p.grad = None


class Study:
    def __init__(self, args):
        self.a = args
        self.started = time.time()
        self.device = torch.device(args.device)
        torch.cuda.set_device(self.device)
        total = torch.cuda.get_device_properties(self.device).total_memory
        torch.cuda.set_per_process_memory_fraction(args.memory_gib * 2**30 / total, self.device)
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.tok = AutoTokenizer.from_pretrained(args.student, local_files_only=True)
        self.prompts = self.prepare_prompts()
        self.forward_dtype = torch.float32 if args.forward_dtype == "float32" else torch.bfloat16
        self.student = (
            AutoModelForCausalLM.from_pretrained(
                args.student, dtype=self.forward_dtype, attn_implementation="sdpa", local_files_only=True
            )
            .eval()
            .to(self.device)
        )
        self.student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # eval() disables dropout, not gradients. Checkpointing is enabled during
        # the explicit train() forward below (Qwen consults self.training).
        self.teacher = (
            AutoModelForCausalLM.from_pretrained(
                args.teacher, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
            )
            .eval()
            .requires_grad_(False)
        )
        self.teacher_head = None
        self.names, params = zip(*self.student.named_parameters(), strict=False)
        self.gpuparams = list(params)
        self.base = [p.detach().cpu().float() for p in params]
        self.numel = sum(p.numel() for p in params)
        save(
            args.out / "protocol.json",
            {
                **vars_serializable(args),
                "torch": torch.__version__,
                "python": sys.executable,
                "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                "parameters": self.numel,
                "trainable_parameters": sum(p.numel() for p in params if p.requires_grad),
                "student_vocab": len(self.student.lm_head.weight),
                "teacher_vocab": len(self.teacher.lm_head.weight),
                "model_dtype": args.forward_dtype,
                "master_and_states": "CPU float32",
                "weight_decay": 0,
                "loss": "production full_reverse_kl, token mean, all generated response positions",
                "calibration": "same initial batch; match first FP32 parameter-step L2 to cold AdamW lr=1e-6",
                "evaluation": "fixed independent initial-student prefixes; conditional KL, not task accuracy",
                "source_endpoints": [
                    "verl.trainer.distillation.full_vocab.full_reverse_kl",
                    "verl.trainer.distillation.numerics.kl_rows",
                ],
                "limitations": [
                    "short truncated responses",
                    "small prompt set",
                    "shared GPU timings",
                    "no task accuracy ranking",
                ],
            },
        )

    def prepare_prompts(self):
        path = self.a.out / "prompts.json"
        if path.exists():
            return json.loads(path.read_text())
        rows = pq.read_table(self.a.data, columns=["prompt", "extra_info"]).to_pylist()
        random.Random(20261027).shuffle(rows)
        result = []
        for row in rows:
            text = self.tok.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
            ids = self.tok.encode(text, add_special_tokens=False)
            if not 24 <= len(ids) <= self.a.prompt_cap:
                continue
            result.append(
                {
                    "id": len(result),
                    "prompt_ids": ids,
                    "prompt": row["prompt"],
                    "source_index": row["extra_info"].get("index"),
                    "split": "train" if len(result) < 4 else "probe",
                }
            )
            if len(result) == 8:
                break
        assert len(result) == 8
        save(path, result)
        return result

    @torch.no_grad()
    def reset(self, masters=None):
        for gp, cp in zip(self.gpuparams, self.base if masters is None else masters, strict=False):
            gp.copy_(cp.to(self.device, dtype=gp.dtype))
        self.student.zero_grad(set_to_none=True)
        self.student.eval()

    @torch.no_grad()
    def rollout(self, prompts, seed):
        self.student.eval()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        result = []
        for prompt in prompts:
            ids = torch.tensor([prompt["prompt_ids"]], device=self.device)
            out = self.student.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=self.a.cap,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                use_cache=True,
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.eos_token_id,
            )
            response = out[0, ids.shape[1] :].tolist()
            result.append(
                {**prompt, "response_ids": response, "text": self.tok.decode(response), "sampling_seed": seed}
            )
            print(
                "rollout",
                prompt["id"],
                "tokens",
                len(response),
                "elapsed",
                round(time.time() - self.started, 1),
                flush=True,
            )
        return result

    def hidden(self, model, rec):
        tokens = rec["prompt_ids"] + rec["response_ids"][:-1]
        ids = torch.tensor([tokens], device=self.device)
        h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
        start = len(rec["prompt_ids"]) - 1
        return h[start : start + len(rec["response_ids"])].contiguous()

    @torch.no_grad()
    def teacher_cache(self, records):
        self.teacher_head = None
        self.teacher.to(self.device)
        result = [self.hidden(self.teacher, r).cpu() for r in records]
        self.teacher_head = self.teacher.lm_head.weight.detach()
        self.teacher.to("cpu")
        torch.cuda.empty_cache()
        return result

    def loss_and_grad(self, records, ht):
        total = sum(len(r["response_ids"]) for r in records)
        grads = [None] * len(self.gpuparams)
        lossvalue = 0.0
        self.student.train()  # model config has dropout=0; activates checkpointing
        for rec, teacher_hidden in zip(records, ht, strict=False):
            self.student.zero_grad(set_to_none=True)
            hs = self.hidden(self.student, rec)
            loss = (
                full_reverse_kl(
                    hs,
                    self.student.lm_head.weight,
                    teacher_hidden.to(self.device),
                    self.teacher_head,
                    backend="precompute",
                    token_chunk=32,
                    vocab_chunk=8192,
                )
                / total
            )
            loss.backward()
            lossvalue += float(loss.detach())
            for i, p in enumerate(self.gpuparams):
                if p.grad is not None:
                    g = p.grad.detach().cpu().float()
                    if grads[i] is None:
                        grads[i] = g
                    else:
                        grads[i].add_(g)
            del loss, hs
        self.student.zero_grad(set_to_none=True)
        self.student.eval()
        magnitude = norm(grads)
        if not math.isfinite(lossvalue) or not math.isfinite(magnitude):
            raise FloatingPointError((lossvalue, magnitude))
        # Same global norm clipping as the common recipe, accumulated in FP32.
        scale = min(1.0, 1.0 / (magnitude + 1e-6))
        if scale < 1:
            for g in grads:
                if g is not None:
                    g.mul_(scale)
        return lossvalue, grads, magnitude

    @torch.no_grad()
    def measure(self, records, ht, initial_logits=None):
        rows = []
        total = 0
        for i, (r, th) in enumerate(zip(records, ht, strict=False)):
            hs = self.hidden(self.student, r)
            z = project(hs, self.student.lm_head.weight)
            t = project(th.to(self.device), self.teacher_head)
            values, _ = kl_rows(z, t)
            lp = z.log_softmax(-1)
            row = {
                "id": r["id"],
                "count": len(z),
                "teacher_kl": float(values.mean()),
                "entropy": float(-(lp.exp() * lp).sum(-1).mean()),
            }
            if initial_logits is not None:
                row["initial_student_kl"] = float(kl_rows(z, initial_logits[i].to(self.device))[0].mean())
            rows.append(row)
            total += len(z)
        keys = [k for k in rows[0] if k not in ("id", "count")]
        return {k: sum(r[k] * r["count"] for r in rows) / total for k in keys} | {"tokens": total, "per_prompt": rows}

    @torch.no_grad()
    def initial_logits(self, records):
        return [project(self.hidden(self.student, r), self.student.lm_head.weight).cpu() for r in records]

    def calibration(self, records, ht, seed):
        self.reset()
        value, grads, gn = self.loss_and_grad(records, ht)
        learning_rates = {}
        norms = {}
        target = None
        for arm in self.a.arms:
            masters = [torch.nn.Parameter(p.clone()) for p in self.base]
            opt = CPUOptim(self.names, masters, arm, 1.0, self.device)
            opt.step(grads)
            unit_norm = norm([p.detach() - b for p, b in zip(masters, self.base, strict=False)])
            norms[arm] = unit_norm
            if arm == "adamw":
                target = unit_norm * self.a.adam_lr
            del opt, masters
            gc.collect()
        assert target is not None
        for arm, unit in norms.items():
            learning_rates[arm] = target / unit
        save(
            self.a.out / f"calibration-{seed}.json",
            {
                "initial_kl": value,
                "grad_norm": gn,
                "reference_step_l2": target,
                "reference_step_rms": target / math.sqrt(self.numel),
                "unit_step_l2": norms,
                "learning_rates": learning_rates,
            },
        )
        del grads
        gc.collect()
        return learning_rates

    def run_seed(self, seed):
        self.reset()
        path = self.a.out / f"initial-rollouts-{seed}.json"
        if path.exists():
            initial = json.loads(path.read_text())
        else:
            initial = self.rollout(self.prompts, seed)
            save(path, initial)
        train = [r for r in initial if r["split"] == "train"]
        probes = [r for r in initial if r["split"] == "probe"]
        probe_ht = self.teacher_cache(probes)
        train_ht = self.teacher_cache(train)
        probe_logits = self.initial_logits(probes)
        rates = self.calibration(train[:2], train_ht[:2], seed)
        self.reset()
        initial_metrics = self.measure(probes, probe_ht, probe_logits)
        save(self.a.out / f"probe-initial-{seed}.json", initial_metrics)
        for mode in self.a.modes:
            for arm in self.a.arms:
                dest = self.a.out / f"{mode}-{arm}-seed{seed}"
                if (dest / "complete.json").exists():
                    print("skip completed", dest.name, flush=True)
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                masters = [torch.nn.Parameter(p.clone()) for p in self.base]
                opt = CPUOptim(self.names, masters, arm, rates[arm], self.device)
                self.reset(masters)
                history = []
                probes_history = [{"step": 0, **initial_metrics}]
                trajectory = []
                arm_start = time.time()
                for outer in range(self.a.outers):
                    ids = [2 * (outer % 2), 2 * (outer % 2) + 1]
                    if mode == "online" and outer > 0:
                        records = self.rollout([self.prompts[i] for i in ids], seed + 1000 + outer)
                        th = self.teacher_cache(records)
                    else:
                        records = [train[i] for i in ids]
                        th = [train_ht[i] for i in ids]
                    trajectory.append({"outer": outer, "records": records})
                    save(dest / "rollouts.json", trajectory)
                    for inner in range(self.a.reuse):
                        value, grads, gn = self.loss_and_grad(records, th)
                        cos = opt.momentum_cosine(grads)
                        before = [p.detach().clone() for p in masters]
                        opt.step(grads)
                        step2 = dot = 0.0
                        for p, b, g in zip(masters, before, grads, strict=False):
                            delta = p.detach() - b
                            step2 += float(delta.square().sum())
                            if g is not None:
                                dot += float((g * delta).sum())
                        self.reset(masters)
                        after = self.measure(records, th)
                        step = outer * self.a.reuse + inner + 1
                        row = {
                            "step": step,
                            "outer": outer,
                            "inner": inner,
                            "pre_kl": value,
                            "post_kl": after["teacher_kl"],
                            "grad_norm": gn,
                            "momentum_cosine": cos,
                            "step_l2": math.sqrt(step2),
                            "gradient_dot_step": dot,
                            "step_rms": math.sqrt(step2 / self.numel),
                            "seconds": time.time() - arm_start,
                            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                            "response_tokens": sum(len(r["response_ids"]) for r in records),
                        }
                        history.append(row)
                        save(dest / "history.json", history)
                        print(dest.name, json.dumps(row), flush=True)
                        del before, grads
                        gc.collect()
                    if (outer + 1) % self.a.probe_every == 0 or outer == self.a.outers - 1:
                        probes_history.append({"step": step, **self.measure(probes, probe_ht, probe_logits)})
                        save(dest / "probe.json", probes_history)
                save(
                    dest / "complete.json",
                    {
                        "steps": len(history),
                        "lr": rates[arm],
                        "arm": arm,
                        "seed": seed,
                        "mode": mode,
                        "final_probe": probes_history[-1],
                        "elapsed_seconds": time.time() - arm_start,
                    },
                )
                del opt, masters
                gc.collect()
                torch.cuda.empty_cache()


def vars_serializable(args):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--student", type=Path, required=True)
    a.add_argument("--teacher", type=Path, required=True)
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--device", default="cuda:0")
    a.add_argument("--memory-gib", type=float, default=12.0)
    a.add_argument("--forward-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    a.add_argument("--cap", type=int, default=64)
    a.add_argument("--prompt-cap", type=int, default=512)
    a.add_argument("--outers", type=int, default=4)
    a.add_argument("--reuse", type=int, default=2)
    a.add_argument("--probe-every", type=int, default=2)
    a.add_argument("--adam-lr", type=float, default=1e-6)
    a.add_argument(
        "--arms", nargs="+", choices=["adamw", "adam0", "sgd", "muon"], default=["adamw", "adam0", "sgd", "muon"]
    )
    a.add_argument("--modes", nargs="+", choices=["frozen", "online"], default=["frozen", "online"])
    a.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = a.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    if "adamw" not in args.arms:
        a.error("AdamW reference is required for calibration")
    args.out.mkdir(parents=True, exist_ok=True)
    study = Study(args)
    for seed in args.seeds:
        study.run_seed(seed)
