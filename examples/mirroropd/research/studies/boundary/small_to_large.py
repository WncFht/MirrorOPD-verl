# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""R1 small teachers into R1-7B: shared-support KL, original frozen states and two-GPU training."""

import argparse
import gc
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from examples.mirroropd.research.backends.hf_rkl import head_vjp
from examples.mirroropd.research.core import training_state
from examples.mirroropd.research.core.artifacts import file_hash, fingerprint
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.core.metrics import native_metrics
from examples.mirroropd.research.core.paths import REPO
from examples.mirroropd.research.runtime.processes import identity
from examples.mirroropd.research.runtime.resources import Resource
from examples.mirroropd.research.runtime.source import environment, freeze
from examples.mirroropd.research.studies.boundary import native
from examples.mirroropd.research.studies.boundary.common import CACHE as OLD_CACHE
from examples.mirroropd.research.studies.boundary.common import GROUPS, MODELS, WORK, model_path
from examples.mirroropd.research.studies.fixed_transfer.runner import queue

OUT = WORK / "wandb-logs/MirrorOPD-verl/analysis/teacher-boundary-small-to-large-20261029"
CACHE = WORK / "wandb-logs/derived/MirrorOPD-verl/teacher-boundary-small-to-large-20261029"
MODULE = "examples.mirroropd.research.studies.boundary.small_to_large"
SUPPORT = 151936
ARMS = {"TB3-R7": "r1", "TB3-J7": "justrl"}
BASE_OUT = OUT
BASE_CACHE = CACHE
UPDATES = 64
TRAIN_GROUPS = list("ABCD")
SCORE_GROUPS = GROUPS
SAVE_RESUME = False
RUN_AUXILIARY = True
FA_DETERMINISTIC = False
REPLAY_REFERENCE_POLICY = "strict"
REPEATABILITY_EVIDENCE = None
RESUME_FROM = {}


def configure(plan):
    """Keep historical defaults; new budgets have isolated outputs and fixed pool coverage."""
    global OUT, CACHE, UPDATES, TRAIN_GROUPS, SCORE_GROUPS, SAVE_RESUME, RUN_AUXILIARY
    global FA_DETERMINISTIC, REPLAY_REFERENCE_POLICY, REPEATABILITY_EVIDENCE, RESUME_FROM
    value = read(plan)
    counts = {a["updates"] for a in value["arms"]}
    devices = value["gpus"]
    if len(counts) != 1 or len(devices) != 2 or len(set(devices)) != 2 or not set(devices) <= set(range(6)):
        raise ValueError("R1 adapter requires one budget and two distinct physical GPUs within 0-5")
    if {a["name"]: a["teacher"] for a in value["arms"]} != ARMS:
        raise ValueError("R1 adapter requires the two pinned small-teacher arms")
    if any(tuple(value["models"][k]) != MODELS[k] for k in ["r1", "r7", "justrl"]):
        raise ValueError("R1 model revisions differ from the audited adapter")
    expected_optimizer = dict(name="AdamW", lr=1e-6, betas=[0.9, 0.999], eps=1e-8, weight_decay=0.01, clip=1)
    if value["training"]["optimizer"] != expected_optimizer or value["training"]["seed"] != 42:
        raise ValueError("Optimizer settings differ from the audited adapter")
    updates = counts.pop()
    groups = value["training"]["train_groups"]
    if updates <= 0 or updates % 16 or groups != list("ABCDEFGHIJKLMNOPQRST")[: len(groups)]:
        raise ValueError("Use complete reuse blocks and a contiguous fixed A–T pool")
    if not groups or len(groups) > 20 or value["support"]["size"] != SUPPORT:
        raise ValueError("Invalid fixed pool or shared support")
    if value["training"]["batch_size"] != 16 or value["training"]["reuse"] != 16:
        raise ValueError("The audited adapter uses batch16 and reuse16")
    if value["training"]["horizon"] != 1024:
        raise ValueError("The audited adapter uses H1024")
    output = Path(value["output"])
    cache = Path(value.get("cache", str(BASE_CACHE)))
    if updates != 64 and (output == BASE_OUT or cache == BASE_CACHE):
        raise ValueError("Long-budget replay must use new output and checkpoint directories")
    deterministic = value["training"].get("flash_attention_deterministic", False)
    policy = value.get("replay_reference_policy", "strict")
    if not isinstance(deterministic, bool) or policy not in {"strict", "diagnostic"}:
        raise ValueError("Invalid repeatability execution policy")
    proof_hash = None
    if policy == "diagnostic":
        if not deterministic:
            raise ValueError("Diagnostic reference requires an explicitly deterministic new replica")
        proof_path = Path(value["repeatability_probe"])
        proof = read(proof_path)
        assert proof["student"] == list(MODELS["r7"]) and proof["teacher"] == list(MODELS["r1"])
        assert proof["scientific_updates"] == 0 and proof["support"] == SUPPORT
        controls = {r["deterministic"]: r for r in proof["results"]}
        assert controls[False]["unequal_gradient_tensors"] > 0
        assert controls[True]["unequal_gradient_tensors"] == 0 and controls[True]["relative_l2"] == 0
        assert controls[True]["first_loss"] == controls[True]["second_loss"]
        proof_hash = file_hash(proof_path)
    FA_DETERMINISTIC, REPLAY_REFERENCE_POLICY, REPEATABILITY_EVIDENCE = deterministic, policy, proof_hash
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1" if deterministic else "0"
    UPDATES, TRAIN_GROUPS = updates, groups
    SCORE_GROUPS = value["scoring"]["groups"]
    if not set(groups) <= set(SCORE_GROUPS):
        raise ValueError("Scoring must include the entire fixed training pool")
    OUT, CACHE = output, cache
    SAVE_RESUME = value["storage"]["optimizer_export"]
    RUN_AUXILIARY = value.get("run_auxiliary", True)
    RESUME_FROM = value.get("resume_from", {})
    if RESUME_FROM and (set(RESUME_FROM) != set(ARMS) or not SAVE_RESUME or not deterministic):
        raise ValueError("Continuation requires both deterministic arms and retained Adam state")
    for source in RESUME_FROM.values():
        if Path(source).resolve().parent.parent == OUT.resolve():
            raise ValueError("Continuation must use a separate output directory")


def training_group(step):
    return TRAIN_GROUPS[((step - 1) // 16) % len(TRAIN_GROUPS)]


def resume_signature(arm):
    return dict(
        teacher=list(MODELS[ARMS[arm]]),
        student=list(MODELS["r7"]),
        support=SUPPORT,
        bank=fingerprint(rows()),
        train_groups=TRAIN_GROUPS,
        optimizer=dict(lr=1e-6, betas=[0.9, 0.999], eps=1e-8, weight_decay=0.01, clip=1),
        precision="FP32 parameters/gradients/Adam/head; BF16 backbone; TF32 off",
        batch_size=16,
        reuse=16,
        horizon=1024,
        parallel_layers=[14, 14],
        flash_attention_deterministic=FA_DETERMINISTIC,
    )


def rows(source="r7"):
    return [r for r in native.bank(source) if r["group"] in SCORE_GROUPS]


def target_directory(teacher, source):
    old = OLD_CACHE / "teachers" / teacher / source
    for directory in [old, BASE_CACHE / "teachers" / teacher / source]:
        if (directory / "complete.json").exists():
            return directory
    return CACHE / "teachers" / teacher / source


def cache_teacher(teacher, source):
    directory = target_directory(teacher, source)
    if (directory / "complete.json").exists() and all((directory / f"{r['id']:04d}.pt").exists() for r in rows(source)):
        return
    directory.mkdir(parents=True, exist_ok=True)
    model = native.load(teacher)
    torch.save(model.lm_head.weight.detach().cpu(), directory / "head.pt")
    with torch.no_grad():
        for row in rows(source):
            path = directory / f"{row['id']:04d}.pt"
            if not path.exists():
                torch.save(
                    dict(
                        prompt_ids=row["prompt_ids"],
                        response_ids=row["response_ids"],
                        teacher=MODELS[teacher],
                        source=row["source"],
                        hidden=native.hidden(model, row).cpu(),
                    ),
                    path,
                )
    dump(directory / "complete.json", dict(teacher=MODELS[teacher], student=MODELS[source], rows=len(rows(source))))
    print("CACHE_COMPLETE", teacher, source, flush=True)


def target(teacher, source, device):
    directory = target_directory(teacher, source)
    assert read(directory / "complete.json")["teacher"] == list(MODELS[teacher])
    states = {}
    for row in rows(source):
        value = torch.load(directory / f"{row['id']:04d}.pt", map_location="cpu", weights_only=True)
        assert tuple(value["teacher"]) == MODELS[teacher]
        assert all(value[k] == row[k] for k in ["prompt_ids", "response_ids", "source"])
        states[row["id"]] = value["hidden"]
    head = torch.load(directory / "head.pt", map_location=device, weights_only=True).float()
    assert head.shape[0] >= SUPPORT
    return head, states


def train_mode(model):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()


def backward(model, head, states, batch, *, reference=False, chunk=128):
    weight = model.lm_head.weight
    if not reference and weight.grad is None:
        weight.grad = torch.zeros_like(weight)
    total = 0.0
    for row in batch:
        hp = native.hidden(model, row).to(weight.device)
        leaf = hp.detach().float().requires_grad_(reference)
        hq = states[row["id"]].to(weight.device).float()
        dh = None if reference else torch.empty_like(leaf)
        for start in range(0, len(hp), chunk):
            stop = min(start + chunk, len(hp))
            with torch.no_grad():
                lq = F.linear(hq[start:stop], head[:SUPPORT]).log_softmax(-1)
            if reference:
                lp = F.linear(leaf[start:stop], weight[:SUPPORT]).log_softmax(-1)
                loss = (lp.exp() * (lp - lq)).sum() / (16 * 1024)
                total += float(loss.detach())
                loss.backward()
            else:
                loss, dz = head_vjp(leaf[start:stop], weight[:SUPPORT], lq, 16 * 1024)
                with torch.no_grad():
                    weight.grad[:SUPPORT].addmm_(dz.T, leaf[start:stop])
                    dh[start:stop] = dz @ weight[:SUPPORT]
                total += float(loss)
        hp.backward((leaf.grad if reference else dh).to(hp.dtype))
    return total


@torch.no_grad()
def score(model, head, states, panel, step):
    result = []
    device = model.lm_head.weight.device
    for i, row in enumerate(panel):
        hp = native.hidden(model, row).to(device).float()
        hq = states[row["id"]].to(device).float()
        sums = {}
        for start in range(0, len(hp), 128):
            sp = F.linear(hp[start : start + 128], model.lm_head.weight)
            tq = F.linear(hq[start : start + 128], head)
            values = native_metrics(sp[:, :SUPPORT], tq[:, :SUPPORT])
            values["student_tail_mass"] = sp.softmax(-1)[:, SUPPORT:].sum(-1)
            values["teacher_tail_mass"] = tq.softmax(-1)[:, SUPPORT:].sum(-1)
            for key, value in values.items():
                sums[key] = sums.get(key, 0.0) + float(value.sum(dtype=torch.float64))
        result.append(
            dict(
                id=row["id"],
                group=row["group"],
                split=row["split"],
                domain=row["domain"],
                source=row["source"],
                step=step,
                length=len(hp),
                stats={"all": dict(tokens=len(hp), **sums)},
            )
        )
        if (i + 1) % 64 == 0:
            print("SCORE", step, i + 1, len(panel), flush=True)
    return result


def grad_compare(model, saved):
    aa = bb = ab = dd = 0.0
    for name, p in model.named_parameters():
        assert p.grad is not None and name in saved
        before, after = saved[name].reshape(-1), p.grad.reshape(-1)
        for start in range(0, p.numel(), 2**20):
            a = before[start : start + 2**20].to(p.device).double()
            b = after[start : start + 2**20].double()
            aa += float(a.square().sum())
            bb += float(b.square().sum())
            ab += float((a * b).sum())
            dd += float((a - b).square().sum())
    return dict(relative_l2=(dd / max(aa, 1e-30)) ** 0.5, cosine=ab / (aa * bb) ** 0.5)


def optimizer(model):
    return torch.optim.AdamW(
        model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False
    )


def validate():
    model = native.load("r7", student=True, parallel=True)
    train_mode(model)
    panel = rows()
    batches = [[r for r in panel if r["group"] == g] for g in TRAIN_GROUPS]
    batch = max(batches, key=lambda b: max(len(r["prompt_ids"]) + len(r["response_ids"]) for r in b))
    probes = []
    tiny = dict(batch[0], response_ids=batch[0]["response_ids"][:8])
    with torch.no_grad():
        self_states = {tiny["id"]: native.hidden(model, tiny).to(model.lm_head.weight.device).detach()}
    self_loss = backward(model, model.lm_head.weight.detach(), self_states, [tiny])
    self_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), foreach=False))
    assert abs(self_loss) < 1e-8 and self_norm < 1e-5
    dump(OUT / "numerical/self-r7.json", dict(loss=self_loss, grad_norm=self_norm, support=SUPPORT))
    model.zero_grad(set_to_none=True)
    for teacher in ARMS.values():
        head, states = target(teacher, "r7", model.lm_head.weight.device)
        model.zero_grad(set_to_none=True)
        ref = backward(model, head, states, batch, reference=True)
        saved = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        candidate = backward(model, head, states, batch)
        comparison = grad_compare(model, saved)
        assert abs(ref - candidate) < 1e-5 and comparison["relative_l2"] < 0.01 and comparison["cosine"] > 0.9999
        assert torch.count_nonzero(model.lm_head.weight.grad[SUPPORT:]) == 0
        probes.append(dict(teacher=teacher, reference_loss=ref, candidate_loss=candidate, **comparison))
        dump(OUT / "numerical/progress.json", dict(probes=probes))
        del saved, head, states
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    head, states = target("r1", "r7", model.lm_head.weight.device)
    opt = optimizer(model)
    for gpu in range(2):
        torch.cuda.reset_peak_memory_stats(gpu)
    losses = []
    before = model.model.norm.weight.detach().clone()
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        loss = backward(model, head, states, batch)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
        assert math.isfinite(loss) and bool(torch.isfinite(norm))
        opt.step()
        losses.append(loss)
    assert {int(v["step"]) for v in opt.state.values()} == {2}
    displacement = float((model.model.norm.weight.detach() - before).abs().max())
    assert displacement > 0
    dump(
        OUT / "numerical/complete.json",
        dict(
            probes=probes,
            disposable_adam_losses=losses,
            norm_parameter_max_change=displacement,
            peak_gib=[torch.cuda.max_memory_allocated(i) / 2**30 for i in range(2)],
            scientific_updates=0,
            support=SUPPORT,
            denominator=16384,
            batch_ids=[r["id"] for r in batch],
            bank_identity=fingerprint(panel),
            flash_attention_deterministic=FA_DETERMINISTIC,
            gpus=os.environ["CUDA_VISIBLE_DEVICES"],
        ),
    )
    print("VALIDATION_COMPLETE", probes, flush=True)


def replay_probe():
    """Repeat identical S0 backward passes; no parameter or optimizer updates."""
    random.seed(42)
    torch.manual_seed(42)
    model = native.load("r7", student=True, parallel=True)
    train_mode(model)
    head, states = target("r1", "r7", model.lm_head.weight.device)
    batch = [r for r in rows() if r["group"] == "A"]
    prior = os.environ.get("FLASH_ATTENTION_DETERMINISTIC")
    results = []
    try:
        for deterministic in [False, True]:
            os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1" if deterministic else "0"
            model.zero_grad(set_to_none=True)
            first_loss = backward(model, head, states, batch)
            first_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), foreach=False))
            saved = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()}
            model.zero_grad(set_to_none=True)
            second_loss = backward(model, head, states, batch)
            second_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), foreach=False))
            comparison = grad_compare(model, saved)
            unequal = sum(not torch.equal(saved[n], p.grad.detach().cpu()) for n, p in model.named_parameters())
            results.append(
                dict(
                    deterministic=deterministic,
                    first_loss=first_loss,
                    second_loss=second_loss,
                    first_norm=first_norm,
                    second_norm=second_norm,
                    unequal_gradient_tensors=unequal,
                    **comparison,
                )
            )
            dump(OUT / "numerical/replay-repeatability-progress.json", dict(results=results, scientific_updates=0))
            del saved
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        if prior is None:
            os.environ.pop("FLASH_ATTENTION_DETERMINISTIC", None)
        else:
            os.environ["FLASH_ATTENTION_DETERMINISTIC"] = prior
    dump(
        OUT / "numerical/replay-repeatability.json",
        dict(
            results=results,
            scientific_updates=0,
            student=MODELS["r7"],
            teacher=MODELS["r1"],
            support=SUPPORT,
            batch_ids=[r["id"] for r in batch],
            interpretation="Fixed S0/data; only the FlashAttention deterministic-backward flag changes",
        ),
    )


def resume_origin(arm, *, require_probe=True):
    source = RESUME_FROM.get(arm)
    if not source:
        return None, None
    root = Path(source)
    marker = read(root / "resume/complete.json")
    if marker["signature"] != fingerprint(resume_signature(arm)) or not 0 < marker["step"] < UPDATES:
        raise ValueError("R1 continuation science identity or update budget changed")
    if read(root / "complete.json")["updates"] != marker["step"]:
        raise ValueError("R1 continuation source is not a completed stage")
    weights = Path(marker["weights"])
    saved = read(weights / "complete.json")
    if (
        fingerprint(dict(weights=saved["files"], config=file_hash(weights / "config.json")))
        != marker["weight_identity"]
    ):
        raise ValueError("R1 continuation weight manifest changed")
    for entry in saved["files"]:
        path = weights / entry["name"]
        if path.stat().st_size != entry["bytes"] or file_hash(path) != entry["sha256"]:
            raise ValueError("R1 continuation weight content changed")
    if require_probe:
        gate = read(root / "resume/gpu-verified.json")
        if not all(gate[k] == marker[k] for k in ["step", "signature", "weight_identity"]):
            raise ValueError("R1 restore probe belongs to another checkpoint")
        if not gate.get("exact_loaded_values") or not gate.get("rng_exact") or gate.get("scientific_updates") != 0:
            raise ValueError("R1 continuation requires exact GPU restore acceptance")
    return root, marker


def resume_probe(arm):
    from examples.mirroropd.research.core.restore_audit import verify_loaded

    root, marker = resume_origin(arm, require_probe=False)
    if root is None:
        raise ValueError("A continuation source is required for the restore probe")
    model = native.load("r7", student=True, parallel=True, path=marker["weights"])
    train_mode(model)
    opt = optimizer(model)
    step = training_state.restore(
        root / "resume", model, opt, signature=resume_signature(arm), weight_identity=marker["weight_identity"]
    )
    parameters, slots = verify_loaded(root / "resume", marker["weights"], model, opt)
    head, states = target(ARMS[arm], "r7", model.lm_head.weight.device)
    batch = [r for r in rows() if r["group"] == training_group(step + 1)]
    tracked = model.model.norm.weight
    before = tracked.detach().clone()
    opt.zero_grad(set_to_none=True)
    loss = backward(model, head, states, batch)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
    assert math.isfinite(loss) and bool(torch.isfinite(norm))
    opt.step()
    assert {int(v["step"]) for v in opt.state.values()} == {step + 1}
    displacement = float((tracked.detach() - before).abs().max())
    assert displacement > 0
    dump(
        root / "resume/gpu-verified.json",
        dict(
            inputs={key: marker[key] for key in ["step", "signature", "weight_identity"]},
            step=step,
            signature=marker["signature"],
            weight_identity=marker["weight_identity"],
            exact_loaded_values=True,
            rng_exact=True,
            parameters_checked=parameters,
            optimizer_tensors_checked=slots,
            disposable_next_step=step + 1,
            scientific_updates=0,
            loss=loss,
            norm_parameter_max_change=displacement,
        ),
    )


def train(arm):
    assert (OUT / "numerical/complete.json").exists()
    if UPDATES != 64:
        gate = read(OUT / "numerical/complete.json")
        assert gate["bank_identity"] == fingerprint(rows())
        assert gate.get("flash_attention_deterministic", False) == FA_DETERMINISTIC
    dest = OUT / "arms" / arm
    if (dest / "complete.json").exists():
        saved = read(dest / "complete.json")
        if UPDATES != 64 and saved.get("inputs") != dict(signature=resume_signature(arm), updates=UPDATES):
            raise ValueError("Completed R1 replay identity differs")
        return
    if (dest / "training.json").exists():
        raise RuntimeError("Partial science run exists; explicit replay required")
    random.seed(42)
    torch.manual_seed(42)
    origin, restored = resume_origin(arm)
    model = native.load("r7", student=True, parallel=True, path=restored["weights"] if restored else None)
    train_mode(model)
    head, states = target(ARMS[arm], "r7", model.lm_head.weight.device)
    panel = rows()
    opt = optimizer(model)
    start_step, history = 0, []
    if restored:
        start_step = training_state.restore(
            origin / "resume", model, opt, signature=resume_signature(arm), weight_identity=restored["weight_identity"]
        )
        history = read(origin / "training.json")
        if len(history) != start_step or [r["step"] for r in history] != list(range(1, start_step + 1)):
            raise ValueError("R1 restored history differs from Adam step")
        dest.mkdir(parents=True, exist_ok=True)
        for source in origin.glob("fixed-*.json"):
            shutil.copy2(source, dest / source.name)
        shutil.copy2(origin / "initial-replay-acceptance.json", dest / "initial-replay-acceptance.json")
    dump(
        dest / "protocol.json",
        dict(
            arm=arm,
            teacher=MODELS[ARMS[arm]],
            student=MODELS["r7"],
            support=SUPPORT,
            objective="conditional shared-support RKL",
            full_head_rows=model.lm_head.weight.shape[0],
            optimizer_export=SAVE_RESUME,
            updates=UPDATES,
            flash_attention_deterministic=FA_DETERMINISTIC,
            replay_reference_policy=REPLAY_REFERENCE_POLICY,
            repeatability_evidence=REPEATABILITY_EVIDENCE,
            train_groups=TRAIN_GROUPS,
            signature=resume_signature(arm),
            initial_state="original R1-7B" if origin is None else str(origin),
            resume_from_step=start_step,
            gpus=os.environ["CUDA_VISIBLE_DEVICES"],
        ),
    )
    if origin is None:
        initial_scores = score(model, head, states, panel, 0)
        dump(dest / "fixed-000.json", initial_scores)
    if OUT != BASE_OUT and origin is None:
        baseline = {r["id"]: r for r in read(BASE_OUT / "arms" / arm / "fixed-000.json")}
        errors = {
            metric: max(
                abs(r["stats"]["all"][metric] - baseline[r["id"]]["stats"]["all"][metric]) / 1024
                for r in initial_scores
                if r["id"] in baseline
            )
            for metric in ["rkl", "js", "hp", "hq"]
        }
        dump(dest / "initial-replay-acceptance.json", dict(prompts=len(baseline), max_errors=errors))
        assert max(errors.values()) < 1e-8, "Initial distributions changed before any optimizer update"
    started = time.time()
    for step in range(start_step + 1, UPDATES + 1):
        group = training_group(step)
        batch = [r for r in panel if r["group"] == group]
        opt.zero_grad(set_to_none=True)
        begin = time.time()
        loss = backward(model, head, states, batch)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
        assert math.isfinite(loss) and bool(torch.isfinite(norm))
        opt.step()
        for gpu in range(2):
            torch.cuda.synchronize(gpu)
        record = dict(
            step=step,
            group=group,
            loss=loss,
            grad_norm=float(norm),
            update_seconds=time.time() - begin,
            actual_tokens=sum(len(r["response_ids"]) for r in batch),
            elapsed=time.time() - started,
        )
        history.append(record)
        dump(dest / "training.json", history)
        print("UPDATE", arm, record, flush=True)
        opt.zero_grad(set_to_none=True)
        if step % 16 == 0:
            scored = score(model, head, states, panel, step)
            dump(dest / f"fixed-{step:03d}.json", scored)
            if step == 64 and OUT != BASE_OUT:
                old_history = read(BASE_OUT / "arms" / arm / "training.json")
                loss_error = max(abs(a["loss"] - b["loss"]) for a, b in zip(history, old_history, strict=True))
                old_scores = {r["id"]: r for r in read(BASE_OUT / "arms" / arm / "fixed-064.json")}
                risk_error = max(
                    abs(r["stats"]["all"]["rkl"] - old_scores[r["id"]]["stats"]["all"]["rkl"]) / 1024
                    for r in scored
                    if r["id"] in old_scores
                )
                dump(
                    dest / "replay-064.json",
                    dict(
                        loss_max_error=loss_error,
                        risk_max_error=risk_error,
                        reference_policy=REPLAY_REFERENCE_POLICY,
                        flash_attention_deterministic=FA_DETERMINISTIC,
                        exact_old_trajectory_claim=REPLAY_REFERENCE_POLICY == "strict",
                    ),
                )
                if REPLAY_REFERENCE_POLICY == "strict":
                    assert loss_error < 1e-5 and risk_error < 1e-5, "Old 64-step trajectory mismatch"
    final = CACHE / "checkpoints" / arm / f"step{UPDATES:03d}"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final, max_shard_size="4GB", safe_serialization=True)
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(model_path("r7"), local_files_only=True).save_pretrained(final)
    files = [
        dict(name=f.name, bytes=f.stat().st_size, sha256=file_hash(f)) for f in sorted(final.glob("*.safetensors"))
    ]
    weight_identity = fingerprint(dict(weights=files, config=file_hash(final / "config.json")))
    dump(
        final / "complete.json",
        dict(
            step=UPDATES,
            files=files,
            optimizer_saved=SAVE_RESUME,
            support=SUPPORT,
            weight_identity=weight_identity,
        ),
    )
    if SAVE_RESUME:
        training_state.save(
            dest / "resume",
            model,
            opt,
            step=UPDATES,
            signature=resume_signature(arm),
            weights=final,
            weight_identity=weight_identity,
        )
    dump(
        dest / "complete.json",
        dict(
            updates=UPDATES,
            elapsed=time.time() - started,
            weights=str(final),
            support=SUPPORT,
            signature=resume_signature(arm),
            replay_from_initial=origin is None,
            resume_from=str(origin) if origin else None,
            new_updates=UPDATES - start_step,
            inputs=dict(signature=resume_signature(arm), updates=UPDATES),
        ),
    )


def controls():
    # Existing model checkpoints are rescored, never retrained or overwritten.
    for student, teacher, arm, source in [
        ("r7", "polaris", "TB2-P7", "r7"),
        ("r1", "justrl", "TB1-J", "r7"),
        ("r1", "r7", "TB1-R", "r1"),
    ]:
        cache_teacher(teacher, source)
        for step in [0, 64]:
            dest = OUT / "controls" / f"{arm}-{source}-{step:03d}.json"
            if dest.exists():
                continue
            path = None if step == 0 else OLD_CACHE / "checkpoints" / arm / "step064"
            model = native.load(student, student=True, path=path)
            head, states = target(teacher, source, model.lm_head.weight.device)
            dump(dest, score(model, head, states, rows(source), step))
            del model, head, states
            gc.collect()
            torch.cuda.empty_cache()
    dump(OUT / "controls/complete.json", dict(support=SUPPORT))


def evaluate(arm):
    import socket

    from examples.mirroropd.research.studies.boundary import round2_eval

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    round2_eval.OUT = OUT
    sys.argv = [sys.argv[0], arm, "--port", str(port)]
    round2_eval.main()


def run(plan, *, wait_resource=False, probe_only=False, eval_only=False, resume_check_only=False):
    resource = Resource()
    assert resource.config["gpus"] == read(plan)["gpus"]
    directory = OUT / "control" / f"{time.time_ns()}-run"
    directory.mkdir(parents=True)
    freeze(REPO, directory / "source")
    dump(directory / "resource-wait.json", dict(identity=identity(os.getpid()), waiting=wait_resource))
    lease = resource.acquire(wait=wait_resource)
    tag = "small-to-large-" + directory.name
    dump(directory / "owner.json", dict(identity=identity(os.getpid()), tag=tag))
    dump(OUT / "control/latest.json", dict(directory=str(directory), owner=identity(os.getpid()), tag=tag))
    (directory / "plan.json").write_text(Path(plan).read_text())
    plan = directory / "plan.json"
    specs = {
        a: dict(
            path=str(CACHE / "checkpoints" / a / f"step{UPDATES:03d}"),
            template="shared",
            response_cap=31744,
            model_length=32768,
        )
        for a in ARMS
    }
    dump(OUT / "eval-specs.json", specs)
    all_jobs = []

    def job(name, args, marker, width=1, dependencies=()):
        all_jobs.append(
            dict(
                name=name,
                args=args,
                module=MODULE,
                marker=str(marker),
                width=width,
                dependencies=list(dependencies),
                retries=0,
            )
        )

    for t in ARMS.values():
        job("cache-" + t, ["cache", "--teacher", t], target_directory(t, "r7") / "complete.json")
        all_jobs[-1]["required_files"] = [str(target_directory(t, "r7") / f"{r['id']:04d}.pt") for r in rows()]
    job("validate", ["validate"], OUT / "numerical/complete.json", 2, ["cache-r1", "cache-justrl"])
    job("train-R", ["train", "--arm", "TB3-R7"], OUT / "arms/TB3-R7/complete.json", 2, ["validate"])
    job("train-J", ["train", "--arm", "TB3-J7"], OUT / "arms/TB3-J7/complete.json", 2, ["train-R"])
    for arm, source in RESUME_FROM.items():
        marker = read(Path(source) / "resume/complete.json")
        name = "resume-" + arm
        job(name, ["resume-probe", "--arm", arm], Path(source) / "resume/gpu-verified.json", 2, ["cache-" + ARMS[arm]])
        all_jobs[-1]["expected_inputs"] = {key: marker[key] for key in ["step", "signature", "weight_identity"]}
        training_job = next(j for j in all_jobs if j["name"] == ("train-R" if arm == "TB3-R7" else "train-J"))
        training_job["dependencies"].append(name)
    if UPDATES != 64:
        for entry in all_jobs:
            if entry["name"] in {"train-R", "train-J"}:
                arm = "TB3-R7" if entry["name"] == "train-R" else "TB3-J7"
                entry["expected_inputs"] = dict(signature=resume_signature(arm), updates=UPDATES)
    if RUN_AUXILIARY:
        job("controls", ["controls"], OUT / "controls/complete.json", 1, ["validate"])
        for a, t in [("TB3-R7", "train-R"), ("TB3-J7", "train-J")]:
            job("eval-" + a, ["evaluate", "--arm", a], OUT / "evaluations" / a / "complete.json", 1, [t])
    if eval_only:
        all_jobs = [entry for entry in all_jobs if entry["name"].startswith("eval-")]
        for entry in all_jobs:
            entry["dependencies"] = []
    if resume_check_only:
        if not RESUME_FROM:
            raise ValueError("Restore-only scheduling requires continuation sources")
        all_jobs = [entry for entry in all_jobs if entry["name"].startswith(("cache-", "resume-"))]
    if probe_only:
        all_jobs = [entry for entry in all_jobs if entry["name"].startswith("cache-")]
        job(
            "replay-probe",
            ["replay-probe"],
            OUT / "numerical/replay-repeatability.json",
            2,
            ["cache-r1", "cache-justrl"],
        )
    try:
        with (directory / "guard.log").open("w") as log:
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "examples.mirroropd.research.studies.fixed_transfer",
                    "guard",
                    "--directory",
                    str(directory),
                ],
                cwd=directory / "source",
                env=environment(directory / "source", WORK),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.time() + 30
        while not (directory / "guard-ready.json").exists():
            if time.time() > deadline:
                raise TimeoutError("guard startup")
            time.sleep(0.1)

        def stop(sig, frame):
            raise SystemExit(128 + sig)

        for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
            signal.signal(sig, stop)
        resource.prepare()
        queue(
            SimpleNamespace(gpus=tuple(resource.config["gpus"])),
            plan,
            "small-to-large",
            directory,
            tag,
            custom_jobs=all_jobs,
        )
    except BaseException as e:
        dump(directory / "error.json", dict(error=repr(e)))
        raise
    finally:
        for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
            signal.signal(sig, signal.SIG_IGN)
        resource.recover(directory, tag)
        lease.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "action", choices=["run", "cache", "validate", "train", "controls", "evaluate", "replay-probe", "resume-probe"]
    )
    p.add_argument(
        "--plan", type=Path, default=REPO / "examples/mirroropd/research/studies/boundary/plans/small-to-large.json"
    )
    p.add_argument("--wait-resource", action="store_true")
    p.add_argument("--probe-only", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--resume-check-only", action="store_true")
    p.add_argument("--teacher", choices=["r1", "justrl"])
    p.add_argument("--arm", choices=ARMS)
    args = p.parse_args()
    if sum([args.eval_only, args.probe_only, args.resume_check_only]) > 1:
        p.error("Choose one isolated execution phase")
    configure(args.plan)
    if args.action == "run":
        run(
            args.plan,
            wait_resource=args.wait_resource,
            probe_only=args.probe_only,
            eval_only=args.eval_only,
            resume_check_only=args.resume_check_only,
        )
        return
    visible = [int(g) for g in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    assert set(visible) <= set(range(6))
    assert len(visible) == (2 if args.action in ["train", "validate", "replay-probe", "resume-probe"] else 1)
    native.setup()
    if args.action == "cache":
        cache_teacher(args.teacher, "r7")
    elif args.action == "validate":
        validate()
    elif args.action == "replay-probe":
        replay_probe()
    elif args.action == "resume-probe":
        resume_probe(args.arm)
    elif args.action == "train":
        train(args.arm)
    elif args.action == "controls":
        controls()
    else:
        evaluate(args.arm)


if __name__ == "__main__":
    main()
