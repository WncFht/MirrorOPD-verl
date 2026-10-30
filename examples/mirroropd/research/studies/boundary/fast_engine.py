# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact native-support RKL VJP and batch-resident frozen teacher probabilities."""

import gc
from pathlib import Path
from types import SimpleNamespace

import torch

from examples.mirroropd.research.backends import hf_rkl
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.core.metrics import fixed_metrics
from examples.mirroropd.research.studies.boundary.native import bank, hidden, load
from examples.mirroropd.research.studies.boundary.round2_common import CACHE, GROUPS, MODELS, OLD_CACHE, model_path


def panel():
    return [r for r in bank("r1") if r["group"] in GROUPS]


def training_mode(model):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()


def teacher_cache(teacher, rows, *, cache_root=None):
    old = OLD_CACHE / "teachers" / teacher / "r1"
    directory = old if (old / "complete.json").exists() else CACHE / "teachers" / teacher / "r1"
    if cache_root is not None:
        source = directory
        directory = Path(cache_root) / "teachers" / teacher / "r1"
        if directory.resolve() == source.resolve():
            raise ValueError("Expanded caches must not overwrite historical teacher artifacts")
        directory.mkdir(parents=True, exist_ok=True)
        if (source / "complete.json").exists():
            assert read(source / "complete.json")["teacher"] == list(MODELS[teacher])
            for name in ["head.pt", *(f"{r['id']:04d}.pt" for r in rows)]:
                if (source / name).exists() and not (directory / name).exists():
                    (directory / name).symlink_to((source / name).resolve())
    if not (directory / "complete.json").exists() or any(not (directory / f"{r['id']:04d}.pt").exists() for r in rows):
        directory.mkdir(parents=True, exist_ok=True)
        model = load(teacher, path=model_path(teacher))
        if not (directory / "head.pt").exists():
            hf_rkl.save_tensor(directory / "head.pt", model.lm_head.weight.detach().cpu())
        with torch.no_grad():
            for i, row in enumerate(rows):
                path = directory / f"{row['id']:04d}.pt"
                if not path.exists():
                    hf_rkl.save_tensor(
                        path,
                        dict(
                            prompt_ids=row["prompt_ids"],
                            response_ids=row["response_ids"],
                            teacher=MODELS[teacher],
                            source=row["source"],
                            hidden=hidden(model, row).cpu(),
                        ),
                    )
                if i % 32 == 0:
                    print("CACHE", teacher, i, len(rows), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        dump(directory / "complete.json", dict(teacher=MODELS[teacher], rows=len(rows)))
    assert read(directory / "complete.json")["teacher"] == list(MODELS[teacher])
    states = {}
    for row in rows:
        payload = torch.load(directory / f"{row['id']:04d}.pt", map_location="cpu", weights_only=True)
        assert tuple(payload["teacher"]) == MODELS[teacher]
        assert payload["prompt_ids"] == row["prompt_ids"]
        assert payload["response_ids"] == row["response_ids"]
        assert payload["source"] == row["source"]
        states[row["id"]] = payload["hidden"]
    head = torch.load(directory / "head.pt", map_location="cuda", weights_only=True).float()
    return head, states


@torch.no_grad()
def score(model, head, states, rows, step, chunk=256):
    results = []
    bands_spec = [("all", 0, 1024), ("early", 0, 256), ("middle", 256, 512), ("late", 512, 1024)]
    for index, row in enumerate(rows, 1):
        hp = hidden(model, row).float()
        hq = states[row["id"]].to(device=head.device, dtype=torch.float32)
        sums = None
        for start in range(0, len(hp), chunk):
            sp = torch.nn.functional.linear(hp[start : start + chunk], model.lm_head.weight)
            tq = torch.nn.functional.linear(hq[start : start + chunk], head)
            values = fixed_metrics(sp, tq)
            values["extra_teacher_mass"] = tq.softmax(-1)[:, sp.shape[-1] :].sum(-1)
            if sums is None:
                keys = list(values)
                sums = torch.zeros((4, len(keys)), dtype=torch.float64, device=head.device)
            for band, (_, lo, hi) in enumerate(bands_spec):
                left, right = max(lo - start, 0), min(hi - start, len(sp))
                if right > left:
                    sums[band] += torch.stack([values[k][left:right].sum(dtype=torch.float64) for k in keys])
        bands = {
            name: dict(tokens=max(0, min(len(hp), hi) - lo), **dict(zip(keys, vector, strict=True)))
            for (name, lo, hi), vector in zip(bands_spec, sums.tolist(), strict=True)
        }
        results.append(
            dict(
                id=row["id"],
                group=row["group"],
                domain=row["domain"],
                split=row["split"],
                step=step,
                length=len(hp),
                stats=bands,
                source=row["source"],
            )
        )
        if index % 64 == 0:
            print("SCORE", step, index, len(rows), flush=True)
    return results


def backward(model, qcache, rows, chunk=256, microbatch=1):
    """The boundary protocol fixes batch16/H1024; execution shares the native backend."""
    return hf_rkl.backward(
        model,
        qcache,
        rows,
        SimpleNamespace(denominator=16 * 1024),
        SimpleNamespace(microbatch=microbatch, token_chunk=chunk),
        pad_id=151643,
    )
