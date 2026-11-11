# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Controlled online OPD: weight AdamW, ISO frame AdamW, spectrum-only AdamW."""

import argparse
import hashlib
import random
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from examples.mirroropd.research.core.io import dump, read
from verl.trainer.distillation.reverse_vocab import full_reverse_kl_result

from ...teacher_difference.runtime import hidden_states, load_model
from ..absorption.banks import score
from ..absorption.protocol import RunContext
from ..scan import thin_svd
from ..weights import DEFAULT_SCOPE, TensorCheckpoint
from .optimizers import WeightAdam
from .retraction import retract, spectrum_gradient


class FactorAdam:
    """Separate AdamW states on ambient frame variables, followed by polar retraction."""

    def __init__(self, parameters, lr, mode, retraction="svd"):
        self.retraction = retraction
        self.retraction_stats = dict(calls=0, newton=0, fallback=0)
        self.mode, self.entries = mode, []
        self.offload = sum(p.numel() for _, p in parameters) > 2_200_000_000
        for name, parameter in parameters:
            u, s, vh = thin_svd(
                parameter.detach().double(), driver="gesvd" if parameter.is_cuda else None, transpose_wide=True
            )
            u, v, s = u.float(), vh.T.contiguous().float(), s.float()
            if self.offload:
                u, v, s = u.cpu(), v.cpu(), s.cpu()
            if mode == "frames":
                u, v = torch.nn.Parameter(u), torch.nn.Parameter(v)
                optimizer = torch.optim.AdamW([u, v], lr=lr, weight_decay=0)
            else:
                s = torch.nn.Parameter(s)
                optimizer = torch.optim.AdamW([s], lr=lr, weight_decay=0)
            self.entries.append((name, parameter, u, s, v, optimizer))

    def _retract(self, matrix):
        result, info = retract(matrix, method=self.retraction, return_info=True)
        self.retraction_stats["calls"] += 1
        self.retraction_stats["newton"] += info["used"] == "newton"
        self.retraction_stats["fallback"] += info["fallback"] is not None
        return result

    @torch.no_grad()
    def step(self):
        maximum = 0.0
        for _, parameter, u, s, v, optimizer in self.entries:
            g = parameter.grad
            if g is None:
                raise ValueError("Selected parameter has no loss gradient")
            if self.offload:
                ug, vg, sg = u.to(g.device), v.to(g.device), s.to(g.device)
                if self.mode == "frames":
                    u.grad = ((g @ vg) * sg).cpu()
                    v.grad = ((g.T @ ug) * sg).cpu()
                else:
                    s.grad = spectrum_gradient(g, ug, vg).cpu()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if self.mode == "frames":
                    ug = self._retract(u.to(device=g.device, dtype=torch.float64)).float()
                    vg = self._retract(v.to(device=g.device, dtype=torch.float64)).float()
                    u.copy_(ug.cpu())
                    v.copy_(vg.cpu())
                    maximum = max(
                        maximum,
                        float((ug.square().sum(0) - 1).abs().max()),
                        float((vg.square().sum(0) - 1).abs().max()),
                    )
                else:
                    s.clamp_(min=0)
                    sg = s.to(g.device)
                parameter.copy_((ug * sg) @ vg.T)
                continue
            if self.mode == "frames":
                u.grad = (g @ v) * s
                v.grad = (g.T @ u) * s
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                u.copy_(self._retract(u.double()).float())
                v.copy_(self._retract(v.double()).float())
                # A cheap per-column check each step; full orthogonality on checkpoint sites below.
                maximum = max(
                    maximum, float((u.square().sum(0) - 1).abs().max()), float((v.square().sum(0) - 1).abs().max())
                )
            else:
                s.grad = spectrum_gradient(g, u, v)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                s.clamp_(min=0)
            parameter.copy_((u * s) @ v.T)
        return maximum

    @torch.no_grad()
    def gate(self):
        rows = []
        for name, parameter, u, s, v, _ in self.entries:
            if ".k_proj." not in name or not any(f".layers.{i}." in name for i in (0, 13, 27)):
                continue
            u, s, v = [x.to(device=parameter.device, dtype=torch.float64) for x in (u, s, v)]
            identity = torch.eye(u.shape[1], device=u.device, dtype=torch.float64)
            orth = max(
                float((u.double().T @ u.double() - identity).abs().max()),
                float((v.double().T @ v.double() - identity).abs().max()),
            )
            actual = torch.linalg.svdvals(parameter.double(), driver="gesvd" if parameter.is_cuda else None)
            relative = float((actual - s.detach().double().sort(descending=True).values).norm() / s.norm())
            if max(orth, relative) > 1e-5:
                raise ValueError(f"Factor numerical gate failed: {name}, {orth}, {relative}")
            rows.append(dict(tensor=name, orthogonality_max=orth, spectrum_relative_error=relative))
        return rows


def load_training_prompts(path, banks):
    """Require explicit DAPO inputs, disjoint from diagnostic heldout questions."""
    path = Path(path)
    payload = read(path)
    if payload.get("source_family") != "dapo" or not payload.get("source"):
        raise ValueError("Training inputs must declare source_family=dapo and an auditable source")
    rows = payload["rows"]
    forbidden = {tuple(r["prompt_ids"]) for bank in banks.values() for r in bank if r["split"] == "heldout"}
    seen = set()
    for row in rows:
        if row["row_id"] in seen or not row["prompt_ids"] or tuple(row["prompt_ids"]) in forbidden:
            raise ValueError("Duplicate/empty training prompt or overlap with diagnostic heldout")
        seen.add(row["row_id"])
        if row.get("domain") != "math" or not isinstance(row.get("eos_token_id"), int):
            raise ValueError("DAPO rows require domain=math and an integer eos_token_id")
    if not rows:
        raise ValueError("Empty DAPO training set")
    return rows, dict(
        path=str(path.absolute()),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source=payload["source"],
        source_family="dapo",
        count=len(rows),
    )


def online_loss(hs, ws, ht, wt, *, backend="precompute", token_chunk=128, vocab_chunk=8192):
    """Reuse native full-vocabulary RKL; preserve the pilot's fixed-H128 reduction."""
    return (
        full_reverse_kl_result(
            hs, ws, ht, wt, backend=backend, token_chunk=token_chunk, vocab_chunk=vocab_chunk
        ).loss_sum
        / 128
    )


def run(
    manifest,
    job_id,
    output,
    stop,
    mode,
    seed,
    train_prompts,
    steps=64,
    lr=1e-5,
    smoke=False,
    kl_backend="precompute",
    token_chunk=128,
    vocab_chunk=8192,
    retraction="svd",
    teacher_residency="gpu",
    weight_state_device="gpu",
    export_update="delta",
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    context = RunContext(output, stop_paths=(Path(stop),))
    job = next(j for j in read(manifest)["jobs"] if j["job_id"] == job_id)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    evidence = Path(job["evidence_attempt"])
    banks = read(evidence / "banks.json")
    train, train_source = load_training_prompts(train_prompts, banks)
    evaluation = {
        origin: [
            r
            for domain in sorted({v["domain"] for v in rows})
            for r in [v for v in rows if v["split"] == "heldout" and v["domain"] == domain][:2]
        ]
        for origin, rows in banks.items()
    }
    reference, head = torch.load(evidence / "reference.pt", weights_only=True, mmap=True)
    head = head.cuda()
    teacher = (
        load_model(job["teacher_path"], trainable=True, device="cpu" if teacher_residency == "cpu-stage" else "cuda")
        .eval()
        .requires_grad_(False)
    )
    teacher_head = teacher.lm_head.weight.detach().to("cuda")
    model = load_model(job["base"], trainable=True, checkpointing=True, device="cuda")
    model.eval().requires_grad_(False)
    checkpoint = TensorCheckpoint(job["base"])
    keys = checkpoint.selected(DEFAULT_SCOPE)
    if smoke:
        keys = [k for k in keys if ".layers.13.self_attn.k_proj." in k]
        steps = 2
    parameters = [(key, model.get_parameter(key)) for key in keys]
    for _, parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = (
        (
            WeightAdam(parameters, lr)
            if weight_state_device == "cpu"
            else torch.optim.AdamW([p for _, p in parameters], lr=lr, weight_decay=0)
        )
        if mode == "weights"
        else FactorAdam(parameters, lr, mode, retraction=retraction)
    )
    dump(
        output / "protocol.json",
        dict(
            job=job,
            mode=mode,
            seed=seed,
            steps=steps,
            lr=lr,
            smoke=smoke,
            objective="online student-sampled full-vocabulary reverse KL; realized token sum / fixed H128",
            training_data=train_source,
            kl_execution=dict(backend=kl_backend, token_chunk=token_chunk, vocab_chunk=vocab_chunk),
            retraction=retraction,
            teacher_residency=teacher_residency,
            weight_state_device=weight_state_device,
            export_update=export_update,
            arithmetic="FP32 model/factors/Adam states, BF16 autocast backbone, FP64 polar retraction",
            parameters=keys,
            factor_state_offload=getattr(optimizer, "offload", False),
            evaluation_selection="fixed heldout reporting only; no model/step/beta selection",
        ),
    )
    curve = []
    started = time.time()
    for step in range(steps + 1):
        context.check_stop()
        if step in (0, 1, 4, 8, 16, 32, 64) or step == steps:
            model.eval()
            score(
                context,
                model,
                banks if step == steps and not smoke else evaluation,
                reference,
                head,
                output / "scores" / f"step-{step:04d}.json",
                dict(mode=mode, seed=seed, step=step),
            )
            if mode != "weights":
                dump(output / f"factor-gate-{step:04d}.json", optimizer.gate())
        if step == steps:
            break
        row = rng.choice(train)
        model.eval()
        ids = torch.tensor([row["prompt_ids"]], device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            answer = (
                model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    max_new_tokens=128,
                    eos_token_id=row["eos_token_id"],
                    pad_token_id=row["eos_token_id"],
                    use_cache=True,
                )[0, len(row["prompt_ids"]) :]
                .cpu()
                .tolist()
            )
        sampled = dict(row, response_ids=answer)
        model.zero_grad(set_to_none=True)
        model.train()
        if teacher_residency == "cpu-stage":
            teacher.get_decoder().to("cuda")
            with torch.no_grad():
                ht = hidden_states(teacher, sampled["prompt_ids"] + answer[:-1])[len(sampled["prompt_ids"]) - 1 :]
            teacher.get_decoder().to("cpu")
        hs = hidden_states(model, sampled["prompt_ids"] + answer[:-1])[len(sampled["prompt_ids"]) - 1 :]
        if teacher_residency == "gpu":
            with torch.no_grad():
                ht = hidden_states(teacher, sampled["prompt_ids"] + answer[:-1])[len(sampled["prompt_ids"]) - 1 :]
        loss = online_loss(
            hs,
            model.lm_head.weight,
            ht,
            teacher_head,
            backend=kl_backend,
            token_chunk=token_chunk,
            vocab_chunk=vocab_chunk,
        )
        loss.backward()
        before = time.time()
        maximum = optimizer.step()
        optimizer_seconds = time.time() - before
        record = dict(
            step=step + 1,
            train_rkl=float(loss.detach()),
            row_id=row["row_id"],
            domain=row["domain"],
            response_ids=answer,
            optimizer_seconds=optimizer_seconds,
            column_norm_error=maximum,
            retraction_stats=dict(getattr(optimizer, "retraction_stats", {})),
            wall_seconds=time.time() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        curve.append(record)
        dump(output / "learning-curve.json", curve)
        del hs, ht, loss
    if not smoke and export_update == "delta":
        delta = {
            key: (model.get_parameter(key).detach().cpu() - checkpoint.get(key).float()).contiguous() for key in keys
        }
        save_file(delta, str(output / "final-update.safetensors"))
    dump(
        output / "complete.json",
        dict(job_id=job_id, mode=mode, seed=seed, steps=steps, smoke=smoke, wall_seconds=time.time() - started),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output", "stop"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--job-id", type=int, required=True)
    p.add_argument("--mode", choices=["weights", "frames", "spectrum"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument(
        "--train-prompts", required=True, help="Explicit DAPO prompt manifest; diagnostic fallback is forbidden"
    )
    p.add_argument(
        "--kl-backend",
        choices=["dense", "precompute", "precompute_fused", "streaming", "cached", "recompute"],
        default="precompute",
    )
    p.add_argument("--token-chunk", type=int, default=128)
    p.add_argument("--vocab-chunk", type=int, default=8192)
    p.add_argument("--export-update", choices=["delta", "none"], default="delta")
    p.add_argument("--teacher-residency", choices=["gpu", "cpu-stage"], default="gpu")
    p.add_argument("--weight-state-device", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--retraction", choices=["svd", "newton"], default="svd")
    p.add_argument("--smoke", action="store_true")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
