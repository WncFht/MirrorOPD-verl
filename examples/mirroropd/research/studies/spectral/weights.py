# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only, tensor-streamed HF checkpoint analysis and offline interventions."""

import hashlib
import json
import re
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from examples.mirroropd.research.core.artifacts import fingerprint
from examples.mirroropd.research.core.io import dump
from examples.mirroropd.research.studies.teacher_difference.checkpoints import CheckpointStore
from examples.mirroropd.research.studies.teacher_difference.runtime import checkpoint_identity

from .decomposition import (
    energy,
    frame_components,
    random_controls,
    ratio,
    spectrum_summary,
    top_fraction,
    weight_summary,
)

DEFAULT_SCOPE = r"model\.layers\.\d+\.(self_attn|mlp)\..*\.weight$"


class TensorCheckpoint:
    """Load one safetensors tensor at a time; never instantiate a language model."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        receipt = self.path / "spectral-intervention.json"
        if receipt.exists() and not json.loads(receipt.read_text()).get("complete"):
            raise ValueError(f"Incomplete spectral export: {self.path}")
        self.identity = checkpoint_identity(self.path)
        self.tied = json.loads((self.path / "config.json").read_text()).get("tie_word_embeddings", False)
        index = self.path / "model.safetensors.index.json"
        if index.exists():
            self.files = json.loads(index.read_text())["weight_map"]
        else:
            with safe_open(self.path / "model.safetensors", framework="pt", device="cpu") as stream:
                self.files = dict.fromkeys(stream.keys(), "model.safetensors")
        self.shapes = {}
        for filename in sorted(set(self.files.values())):
            with safe_open(self.path / filename, framework="pt", device="cpu") as stream:
                for key in stream.keys():
                    if self.files.get(key) == filename:
                        self.shapes[key] = tuple(stream.get_slice(key).get_shape())
        if set(self.shapes) != set(self.files):
            raise ValueError(f"Index contains missing tensors: {self.path}")

    def get(self, key):
        if key == "lm_head.weight" and key not in self.files and self.tied:
            key = "model.embed_tokens.weight"
        with safe_open(self.path / self.files[key], framework="pt", device="cpu") as stream:
            return stream.get_tensor(key)

    def compatible(self, other):
        def coordinates(checkpoint):
            return {
                k: v
                for k, v in checkpoint.shapes.items()
                if not (checkpoint.tied and k == "lm_head.weight" and "model.embed_tokens.weight" in checkpoint.shapes)
            }

        if coordinates(self) != coordinates(other) or self.tied != other.tied:
            raise ValueError(
                "Weight operations require matching tensor coordinates (allowing an omitted tied head alias)"
            )

    def selected(self, pattern):
        keys = [k for k, shape in self.shapes.items() if re.search(pattern, k) and len(shape) == 2]
        if not keys:
            raise ValueError(f"No matrices match scope {pattern!r}")
        return sorted(keys)


def tensor_digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def analyze(
    base,
    teacher,
    opd,
    output,
    *,
    pattern=DEFAULT_SCOPE,
    device="cpu",
    dtype=torch.float64,
    seed=0,
    cache=None,
    resume=False,
    driver=None,
):
    """Exact per-matrix geometry, with reusable small teacher summaries and atomic recovery."""
    checkpoints = [TensorCheckpoint(p) for p in (base, teacher, opd)]
    for checkpoint in checkpoints[1:]:
        checkpoints[0].compatible(checkpoint)
    protocol = dict(
        version=2,
        checkpoints=[c.identity for c in checkpoints],
        scope=pattern,
        device=str(device),
        dtype=str(dtype),
        seed=seed,
        driver=driver,
        torch_version=str(torch.__version__),
        subtraction="promote stored tensors before subtraction; not inferred FP32 masters",
    )
    output = Path(output)
    if resume and output.exists() and not (output / "identity.json").exists():
        raise ValueError("Unversioned output cannot be resumed; use a new output directory")
    output.mkdir(parents=True, exist_ok=resume)
    with CheckpointStore(output, protocol) as results:
        dump(output / "protocol.json", protocol)
        keys = checkpoints[0].selected(pattern)
        for index, key in enumerate(keys):
            started = time.monotonic()
            raw = [c.get(key) for c in checkpoints]
            identity = [dict(dtype=str(x.dtype), sha256=tensor_digest(x)) for x in raw]
            saved = results.json(f"{index:05d}")
            if saved is not None:
                if saved["tensor"] != key or saved["inputs"] != identity:
                    raise ValueError(f"Changed tensor content in resumed output: {key}")
                continue
            b, t, o = [x.to(device=device, dtype=dtype) for x in raw]
            dt, do = t - b, o - b
            frame = torch.linalg.svd(b, full_matrices=False, driver=driver)
            teacher_parts = frame_components(b, dt, frame=frame)
            learned_parts = frame_components(b, do, frame=frame)
            bands = {}
            for name, requested in teacher_parts.items():
                learned = learned_parts[name]
                bands[name] = dict(
                    teacher_energy=float(energy(requested)),
                    learned_energy=float(energy(learned)),
                    absorption=ratio((requested.double() * learned.double()).sum(), energy(requested)),
                    missing=ratio(energy(requested - learned), energy(requested)),
                )
            # Only small scalar summaries persist; large singular frames remain transient.
            reference_identity = dict(
                version=2,
                tensor=key,
                inputs=identity[:2],
                dtype=str(dtype),
                device=str(device),
                driver=driver,
                seed=seed + index,
                torch_version=str(torch.__version__),
            )
            store = (
                CheckpointStore(Path(cache) / fingerprint(reference_identity), reference_identity)
                if cache is not None
                else nullcontext(None)
            )
            with store as reference:
                facts = reference.json("reference") if reference is not None else None
                reused = facts is not None
                if facts is None:
                    singular = torch.linalg.svdvals(dt, driver=driver)
                    controls = random_controls(dt, seed=seed + index, singular_values=singular)
                    facts = dict(
                        teacher=weight_summary(b, t, frame=frame, singular_values=singular, driver=driver),
                        controls={
                            name: weight_summary(
                                b,
                                b + x,
                                frame=frame,
                                driver=driver,
                                singular_values=singular if name == "random_frames" else None,
                            )
                            for name, x in controls.items()
                        },
                    )
                    if reference is not None:
                        reference.save_json("reference", facts)
            result = dict(
                tensor=key,
                inputs=identity,
                base_svd_relative_error=ratio(energy(b - (frame[0] * frame[1]) @ frame[2]).sqrt(), energy(b).sqrt()),
                **facts,
                opd=weight_summary(b, o, frame=frame, driver=driver),
                missing=spectrum_summary(t - o, driver=driver),
                base_frame_bands=bands,
                reference_cache_hit=reused,
                wall_seconds=time.monotonic() - started,
            )
            results.save_json(f"{index:05d}", result)
            dump(
                output / "progress.json",
                dict(
                    completed=index + 1,
                    total=len(keys),
                    tensor=key,
                    last_matrix_seconds=result["wall_seconds"],
                    updated_at=time.time(),
                ),
            )
        results.save_json("complete", dict(tensors=len(keys), completed_at=time.time()))


def reconstruct(
    base,
    target,
    output,
    *,
    mode,
    pattern=DEFAULT_SCOPE,
    fraction=0.01,
    restore_norm=False,
    beta=1.0,
    anchor=None,
    device="cpu",
    dtype=torch.float64,
    untouched="target",
    control="none",
    seed=0,
):
    """Write a new HF export; source checkpoints stay untouched.

    top: base + top(target-base). delete-X: target - beta*X(target-base).
    patch-X: target + beta*X(anchor-target), expressed in the base frame.
    Outside scope stays at target, unless untouched='base' is selected by caller.
    """
    if mode not in ("top", "delete-diag", "delete-off", "delete-null", "patch-diag", "patch-off", "patch-null"):
        raise ValueError(f"Unknown reconstruction mode: {mode}")
    if mode.startswith("patch") and anchor is None:
        raise ValueError("patch requires an anchor teacher checkpoint")
    if control not in ("none", "gaussian", "random_frames"):
        raise ValueError("Unknown intervention control")
    if untouched not in ("target", "base"):
        raise ValueError("untouched must be target or base")
    b, t = TensorCheckpoint(base), TensorCheckpoint(target)
    b.compatible(t)
    a = TensorCheckpoint(anchor) if anchor else None
    if a:
        b.compatible(a)
    for checkpoint in (b, t, a):
        if (
            checkpoint is not None
            and checkpoint.tied
            and {"lm_head.weight", "model.embed_tokens.weight"} <= checkpoint.files.keys()
        ):
            if not torch.equal(checkpoint.get("lm_head.weight"), checkpoint.get("model.embed_tokens.weight")):
                raise ValueError("Tied head and embedding contain different stored values")
    keys = set(b.selected(pattern))
    # A tied head can have an omitted alias in safetensors. Keep both endpoints
    # untouched rather than publishing an export whose reload silently changes it.
    if any("embed_tokens" in k or "lm_head" in k for k in keys):
        raise ValueError("Embedding/head edits require an architecture-specific tied-weight policy")
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    config = json.loads((t.path / "config.json").read_text())
    config["torch_dtype"] = "float32"  # Preserve small edits when a consumer requests dtype=auto.
    dump(
        destination / "spectral-intervention.json",
        dict(
            base=b.identity,
            target=t.identity,
            anchor=a.identity if a else None,
            mode=mode,
            scope=pattern,
            fraction=fraction,
            restore_norm=restore_norm,
            beta=beta,
            untouched=untouched,
            control=control,
            seed=seed,
            arithmetic=str(dtype),
            edited_storage="FP32",
            complete=False,
        ),
    )
    index, total, hashes = {}, 0, {}
    for number, key in enumerate(sorted(t.files)):
        raw = t.get(key) if untouched == "target" or key in keys else b.get(key)
        if key in keys:
            original = b.get(key)
            hashes[key] = dict(base=tensor_digest(original), target=tensor_digest(raw))
            w0, wt = original.to(device=device, dtype=dtype), raw.to(device=device, dtype=dtype)
            if mode == "top":
                value = w0 + top_fraction(wt - w0, fraction, restore_norm=restore_norm)
            elif mode.startswith("delete"):
                value = wt - beta * frame_components(w0, wt - w0)[mode.split("-")[1]]
            else:
                teacher = a.get(key)
                hashes[key]["anchor"] = tensor_digest(teacher)
                value = (
                    wt + beta * frame_components(w0, teacher.to(device=device, dtype=dtype) - wt)[mode.split("-")[1]]
                )
            if control != "none":
                reference = w0 if mode == "top" else wt
                value = reference + random_controls(value - reference, seed=seed + number)[control]
            raw = value.float().cpu().contiguous()
        filename = f"model-{number + 1:05d}-of-{len(t.files):05d}.safetensors"
        save_file({key: raw.contiguous()}, destination / filename, metadata={"format": "pt"})
        index[key] = filename
        total += raw.numel() * raw.element_size()
    for path in t.path.iterdir():
        if path.is_file() and (
            path.name
            in (
                "config.json",
                "generation_config.json",
                "merges.txt",
                "vocab.json",
                "added_tokens.json",
                "special_tokens_map.json",
                "chat_template.jinja",
            )
            or path.name.startswith("tokenizer")
        ):
            shutil.copy2(path, destination / path.name)
    dump(destination / "config.json", config)
    dump(destination / "model.safetensors.index.json", dict(metadata=dict(total_size=total), weight_map=index))
    receipt = json.loads((destination / "spectral-intervention.json").read_text())
    dump(destination / "spectral-intervention.json", dict(receipt, tensors=hashes, complete=True))
