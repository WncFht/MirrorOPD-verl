#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Crossed optimizer / student precision on fixed, independent prefix states.

Teacher logits, token IDs, buffers, FP32 masters and FP32 optimizer states are
held fixed across precision arms. TF32 and reduced-precision reductions are
disabled. Both precisions use SDPA math to avoid a kernel-family confound.
Calibration uses four reserved prompts, never the held-out evaluation panel.
"""

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from verl.trainer.distillation.full_vocab import project
from verl.trainer.distillation.numerics import kl_rows


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


class KL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, q):
        value, g = kl_rows(z, q)
        ctx.save_for_backward(g)
        return value.sum()

    @staticmethod
    def backward(ctx, g):
        return ctx.saved_tensors[0] * g, None


def load_model(path, dtype=torch.float32):
    return (
        AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa", local_files_only=True)
        .cuda()
        .eval()
    )


def forward(model, rec, positions=None):
    positions = rec["positions"] if positions is None else positions
    # Keep the entire fixed reference input even when scoring only early states.
    tokens = rec["prompt_ids"] + rec["response_ids"][:-1]
    h = model.model(input_ids=torch.tensor([tokens], device="cuda"), use_cache=False).last_hidden_state[0]
    idx = torch.tensor([len(rec["prompt_ids"]) - 1 + p for p in positions], device="cuda")
    return project(h[idx], model.lm_head.weight)


def prepare(a):
    raw = read(a.previous / "polaris-student-k16/rollout-1.json")
    refs = read(a.previous / "reference-student.json")
    teacherrefs = read(a.previous / "reference-polaris.json")
    records = []
    for r in raw[:28]:
        role = "calibration" if r["id"] < 4 else "train"
        records.append(
            dict(
                uid=f"s{r['id']}",
                id=r["id"],
                domain="math",
                role=role,
                source="student",
                prompt_ids=r["prompt_ids"],
                response_ids=r["response_ids"],
                positions=list(range(min(256, len(r["response_ids"])))),
            )
        )
    for r in refs:
        if 176 <= r["id"] < 188:
            role = "warm_train"
            positions = list(range(min(256, len(r["response_ids"]))))
        elif 128 <= r["id"] < 176 or 192 <= r["id"] < 200 or 208 <= r["id"] < 216:
            role = "test"
            positions = list(range(0, len(r["response_ids"]), 16))
        else:
            continue
        records.append(
            dict(
                uid=f"s{r['id']}",
                id=r["id"],
                domain=r["domain"],
                role=role,
                source="student",
                prompt_ids=r["prompt_ids"],
                response_ids=r["response_ids"],
                positions=positions,
            )
        )
    for r in teacherrefs:
        if 128 <= r["id"] < 144:
            records.append(
                dict(
                    uid=f"p{r['id']}",
                    id=r["id"],
                    domain="math",
                    role="test",
                    source="polaris",
                    prompt_ids=r["prompt_ids"],
                    response_ids=r["response_ids"],
                    positions=list(range(0, len(r["response_ids"]), 16)),
                )
            )
    assert len({r["uid"] for r in records}) == len(records) == 120
    dump(a.out / "records.json", records)
    dump(
        a.out / "protocol.json",
        dict(
            teacher="Polaris-7B-Preview",
            student="DeepSeek-R1-Distill-Qwen-1.5B",
            target_precision="fixed BF16 teacher backbone and weights, FP32 head projection",
            prefix_source=str(a.previous),
            calibration_prompts=4,
            training_folds=[list(range(4, 16)), list(range(16, 28))],
            heldout={"math_student": 48, "math_teacher": 16, "code_student": 8, "if_student": 8},
            training_positions="all first 256 actions; entire original reference prefix retained as input",
            evaluation_positions="every 16th action across original <=1024-token trace",
            masters="float32",
            optimizer_states="float32",
            buffers="original float32; never dtype-cast",
            attention="SDPA math for all arms",
            tf32=False,
            weight_decay_main=0,
            clip=1,
            schedule="two batches of six fixed prompts; 16 updates each; no trajectory regeneration",
            calibration="first master FP32 function drift matched to cold FP32 AdamW lr1e-6 on calibration prompts",
            caveat="fixed-state mechanism experiment; neither full E48 checkpoint replay nor task-accuracy test",
        ),
    )
    model = load_model(a.models / "Polaris-7B-Preview", torch.bfloat16).requires_grad_(False)
    (a.cache / "teacher").mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for i, r in enumerate(records):
            path = a.cache / "teacher" / f"{r['uid']}.pt"
            if not path.exists():
                torch.save(forward(model, r).cpu(), path)
            if i % 8 == 0:
                print("teacher-cache", i, len(records), flush=True)


class Study:
    def __init__(self, a):
        self.a = a
        self.records = read(a.out / "records.json")
        self.cal = [r for r in self.records if r["role"] == "calibration"]
        self.model = load_model(a.initial if a.initial else a.models / "DeepSeek-R1-Distill-Qwen-1.5B")
        if a.round_initial:
            with torch.no_grad():
                for parameter in self.model.parameters():
                    parameter.copy_(parameter.bfloat16().float())
        assert self.model.config.attention_dropout == 0
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.params = list(self.model.parameters())
        self.base = [p.detach().cpu().clone() for p in self.params]
        self.masters = [torch.nn.Parameter(p.detach().cpu().clone()) for p in self.params]
        self.buffers = {n: b.detach().clone() for n, b in self.model.named_buffers()}
        self.targets = {
            r["uid"]: torch.load(a.cache / "teacher" / f"{r['uid']}.pt", weights_only=True) for r in self.records
        }
        self.basecache = a.cache / "baseline" / a.init_tag
        self.basecache.mkdir(parents=True, exist_ok=True)
        self.grad_accum = [None] * len(self.params)
        if a.accum_fp32:

            def make_hook(i):
                def hook(p):
                    if self.grad_accum[i] is None:
                        self.grad_accum[i] = p.grad.detach().float()
                    else:
                        self.grad_accum[i].add_(p.grad)
                    p.grad = None

                return hook

            self.grad_hooks = [p.register_post_accumulate_grad_hook(make_hook(i)) for i, p in enumerate(self.params)]

    @torch.no_grad()
    def work(self, view):
        dtype = torch.bfloat16 if view == "bf16" else torch.float32
        for p, m in zip(self.params, self.masters, strict=False):
            value = m.detach()
            if view == "rounded_fp32":
                value = value.bfloat16().float()
            if p.dtype == dtype:
                p.copy_(value)
            else:
                p.data = value.to(device="cuda", dtype=dtype, copy=True)
            p.grad = None
        for name, b in self.model.named_buffers():
            assert torch.equal(b, self.buffers[name]), name

    @torch.no_grad()
    def reset(self):
        for m, b in zip(self.masters, self.base, strict=False):
            m.copy_(b)
        self.work("fp32")

    def gradient(self, records, precision):
        self.work(precision)
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        self.grad_accum = [None] * len(self.params)
        total = sum(len(r["positions"]) for r in records)
        value = 0.0
        for r in records:
            z = forward(self.model, r)
            q = self.targets[r["uid"]].cuda()
            loss = KL.apply(z, q) / total
            loss.backward()
            value += float(loss.detach())
            del z, q, loss
        for i, (m, p) in enumerate(zip(self.masters, self.params, strict=False)):
            g = self.grad_accum[i] if self.a.accum_fp32 else p.grad
            m.grad = g.detach().cpu().float() if g is not None else None
        self.grad_accum = [None] * len(self.params)
        norm = torch.nn.utils.clip_grad_norm_(self.masters, 1.0)
        assert torch.isfinite(norm) and math.isfinite(value)
        self.model.zero_grad(set_to_none=True)
        return value, float(norm)

    @torch.no_grad()
    def measure(self, records, view, step, old=True):
        self.work(view)
        self.model.eval()
        rows = []
        for r in records:
            z = forward(self.model, r)
            q = self.targets[r["uid"]].cuda()
            kl, _ = kl_rows(z, q)
            lp = z.log_softmax(-1)
            lq = q.log_softmax(-1)
            ps = lp.exp()
            qt = lq.exp()
            v = z.shape[-1]
            lm = torch.logaddexp(lp, lq[:, :v]) - math.log(2)
            tv = 0.5 * ((ps - qt[:, :v]).abs().sum(-1) + qt[:, v:].sum(-1))
            js = 0.5 * (
                (ps * (lp - lm)).sum(-1) + (qt[:, :v] * (lq[:, :v] - lm)).sum(-1) + qt[:, v:].sum(-1) * math.log(2)
            )
            drift = None
            if old:
                original = torch.load(self.basecache / f"{r['uid']}.pt", weights_only=True).cuda()
                drift = kl_rows(z, original)[0]
                del original
            for region, indices in [
                ("all", list(range(len(r["positions"])))),
                ("early", [i for i, p in enumerate(r["positions"]) if p < 256]),
                ("late", [i for i, p in enumerate(r["positions"]) if p >= 512]),
            ]:
                if not indices:
                    continue
                rows.append(
                    dict(
                        uid=r["uid"],
                        id=r["id"],
                        domain=r["domain"],
                        source=r["source"],
                        role=r["role"],
                        view=view,
                        step=step,
                        region=region,
                        n=len(indices),
                        rkl=float(kl[indices].mean()),
                        tv=float(tv[indices].mean()),
                        js=float(js[indices].mean()),
                        drift=float(drift[indices].mean()) if drift is not None else None,
                    )
                )
            del z, q, kl, ps, qt, lp, lq, lm, tv, js, drift
        return rows

    @torch.no_grad()
    def calibration_drift(self):
        self.work("fp32")
        self.model.eval()
        values = []
        for r in self.cal:
            z = forward(self.model, r)
            initial = torch.load(self.basecache / f"{r['uid']}.pt", weights_only=True).cuda()
            values.append(float(kl_rows(z, initial)[0].mean()))
            del z, initial
        return sum(values) / len(values)

    def baseline(self):
        self.reset()
        with torch.inference_mode():
            for i, r in enumerate(self.records):
                path = self.basecache / f"{r['uid']}.pt"
                if not path.exists():
                    torch.save(forward(self.model, r).cpu(), path)
                if i % 16 == 0:
                    print("baseline-logits", self.a.init_tag, i, flush=True)
        rows = []
        for view in ["fp32", "bf16", "rounded_fp32"]:
            rows += self.measure(self.records, view, 0)
        dump(self.a.out / f"baseline-{self.a.init_tag}.json", rows)
        self.reset()
        value, gn = self.gradient(self.cal, "fp32")
        with torch.no_grad():
            for m, b in zip(self.masters, self.base, strict=False):
                if m.grad is not None:
                    m.copy_(b - 1e-6 * m.grad / (m.grad.abs() + 1e-8))
        target = self.calibration_drift()
        dump(
            self.a.out / f"calibration-reference-{self.a.init_tag}.json",
            dict(
                target_drift=target,
                reference_lr=1e-6,
                reference_optimizer="cold AdamW, wd0, FP32",
                calibration_loss=value,
                grad_norm=gn,
            ),
        )
        print("calibration reference", self.a.init_tag, target, flush=True)

    def calibrate(self):
        family = "sgd" if self.a.optimizer == "sgd" else "adam"
        # Warm continuation reuses the initial-run LR instead of tuning on its
        # already-fitted calibration prompts, which could inflate SGD's LR.
        calibration_tag = "initial" if self.a.init_tag.startswith("warm") else self.a.init_tag
        path = self.a.out / (
            f"lr-calibration-{calibration_tag}-{family}-{self.a.precision}"
            f"{'-accum32' if self.a.accum_fp32 else ''}.json"
        )
        if path.exists():
            record = read(path)
            self.reset()
            if self.a.init_tag.startswith("warm"):
                record["details"]["warm_reuses_initial_lr"] = True
            return record["lr"], record["details"]
        if self.a.init_tag.startswith("warm"):
            raise ValueError("Run initial calibration before warm continuation")
        self.reset()
        _, gn = self.gradient(self.cal, self.a.precision)
        directions = []
        for m in self.masters:
            g = m.grad
            directions.append((-g if self.a.optimizer == "sgd" else -g / (g.abs() + 1e-8)).cpu())
            m.grad = None
        target = read(self.a.out / f"calibration-reference-{self.a.init_tag}.json")["target_drift"]
        lo, hi = (1e-8, 1.0) if self.a.optimizer == "sgd" else (1e-9, 1e-4)
        trials = []
        for _ in range(12):
            lr = math.sqrt(lo * hi)
            with torch.no_grad():
                for m, b, d in zip(self.masters, self.base, directions, strict=False):
                    m.copy_(b + lr * d)
            drift = self.calibration_drift()
            trials.append(dict(lr=lr, drift=drift))
            if abs(drift / target - 1) < 0.015:
                break
            if drift > target:
                hi = lr
            else:
                lo = lr
        best = min(trials, key=lambda r: abs(math.log(max(r["drift"], 1e-30) / target)))
        self.reset()
        del directions
        gc.collect()
        details = dict(target=target, selected=best, trials=trials, grad_norm=gn)
        dump(path, dict(lr=best["lr"], details=details))
        return best["lr"], details

    def train(self):
        a = self.a
        name = f"{a.init_tag}-{a.optimizer}-{a.precision}-fold{a.fold}"
        if a.fixed_lr is not None:
            name += "-fixedlr"
        if a.accum_fp32:
            name += "-accum32"
        dest = a.out / name
        dest.mkdir(exist_ok=False)
        with torch.no_grad():
            self.work("fp32")
            self.model.eval()
            z = forward(self.model, self.cal[0])
            old = torch.load(self.basecache / f"{self.cal[0]['uid']}.pt", weights_only=True).cuda()
            assert float((z - old).abs().max()) < 1e-4, "Initial weights no longer match frozen baseline"
            del z, old
        lr, calibration = (a.fixed_lr, None) if a.fixed_lr is not None else self.calibrate()
        if a.init_tag == "initial":
            ids = list(range(4 + a.fold * 12, 16 + a.fold * 12))
            role = "train"
        else:
            ids = list(range(176, 188))
            role = "warm_train"
        train = [r for r in self.records if r["role"] == role and r["id"] in ids]
        assert len(train) == 12
        evaluate = train + [r for r in self.records if r["role"] == "test"]
        monitor = train + [
            r
            for r in self.records
            if r["role"] == "test"
            and r["source"] == "student"
            and (128 <= r["id"] < 144 or 192 <= r["id"] < 196 or 208 <= r["id"] < 212)
        ]
        if a.optimizer == "sgd":
            opt = torch.optim.SGD(self.masters, lr=lr, momentum=0, weight_decay=a.wd, foreach=True)
        else:
            opt = torch.optim.AdamW(
                self.masters,
                lr=lr,
                betas=(0.0 if a.optimizer == "adam0" else 0.9, 0.999),
                eps=1e-8,
                weight_decay=a.wd,
                foreach=True,
            )
        dump(
            dest / "protocol.json",
            dict(
                optimizer=a.optimizer,
                precision=a.precision,
                initial=a.init_tag,
                train_ids=ids,
                steps=a.steps,
                lr=lr,
                weight_decay=a.wd,
                calibration=calibration,
                seen_train_ids=ids if a.steps > 16 else ids[:6],
                accum_fp32=a.accum_fp32,
                round_initial=a.round_initial,
                initial_path=str((a.initial if a.initial else a.models / "DeepSeek-R1-Distill-Qwen-1.5B").resolve()),
                parameters=sum(p.numel() for p in self.params),
                teacher_target="fixed common cached logits",
            ),
        )
        rows = [r for r in read(a.out / f"baseline-{a.init_tag}.json") if r["uid"] in {x["uid"] for x in evaluate}]
        dump(dest / "evaluation.json", rows)
        history = []
        start = time.time()
        for step in range(1, a.steps + 1):
            batch = train[:6] if step <= 16 else train[6:]
            loss, gn = self.gradient(batch, a.precision)
            previous = [m.detach().cpu().clone() for m in self.masters]
            opt.step()
            opt.zero_grad(set_to_none=True)
            delta2 = err2 = 0.0
            visible = 0
            with torch.no_grad():
                for m, b in zip(self.masters, previous, strict=False):
                    old = b
                    delta = m - old
                    rounded = m.bfloat16().float() - old.bfloat16().float()
                    delta2 += float(delta.square().sum())
                    err2 += float((rounded - delta).square().sum())
                    visible += int(torch.count_nonzero(rounded))
            del previous, old, delta, rounded
            history.append(
                dict(
                    step=step,
                    batch="A" if step <= 16 else "B",
                    pre_update_kl=loss,
                    grad_norm=gn,
                    step_l2=delta2**0.5,
                    bf16_visible_fraction=visible / sum(p.numel() for p in self.params),
                    bf16_step_error_ratio=(err2 / delta2) ** 0.5 if delta2 else 0,
                    seconds=time.time() - start,
                    peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                )
            )
            dump(dest / "training.json", history)
            print(name, history[-1], flush=True)
            if step in [1, 8, 16, 17, a.steps]:
                panel = evaluate if step == a.steps else monitor
                for view in ["fp32", "bf16"] + (["rounded_fp32"] if step == a.steps else []):
                    rows += self.measure(panel, view, step)
                dump(dest / "evaluation.json", rows)
        dump(dest / "complete.json", dict(steps=a.steps, finished=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "baseline", "train"])
    for key in ["models", "out", "cache"]:
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--previous", type=Path)
    p.add_argument("--initial", type=Path)
    p.add_argument("--init-tag", default="initial")
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--optimizer", choices=["adamw", "adam0", "sgd"], default="adamw")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--fixed-lr", type=float)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--accum-fp32", action="store_true")
    p.add_argument("--round-initial", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    a.cache.mkdir(parents=True, exist_ok=True)
    if a.initial is not None and not a.initial.exists():
        relocated = (
            Path(__file__).resolve().parents[5]
            / "outputs/polaris-transfer-cache/intersection-study-20261027/checkpoints"
            / a.initial.name
        )
        if relocated.exists():
            a.initial = relocated
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.cuda.set_per_process_memory_fraction(26 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    if a.stage == "prepare":
        prepare(a)
    else:
        study = Study(a)
        if a.stage == "baseline":
            study.baseline()
        else:
            study.train()


if __name__ == "__main__":
    main()
