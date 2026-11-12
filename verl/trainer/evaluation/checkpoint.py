# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One final model export and native model-only reload parity, without optimizer files."""

import math
import uuid
from pathlib import Path

import pyarrow as pa
from safetensors import safe_open

from .artifacts import ArtifactStore, file_sha256
from .scoring import score_native_rows
from .trajectory import load_trajectories


def final_model_export(
    trainer, references, *, hidden_size, budget, destination, verify=True, live_payload_bytes=0, max_ckpt_to_keep=1
):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Final export destination must be new")
    published = destination
    destination = destination.with_name(f".{destination.name}.staging-{uuid.uuid4().hex}")
    # Same exact rows and batching on both sides; no regenerated answers.
    selected = sorted((r for r in references if r["valid_tokens"] > 0), key=lambda r: (r["valid_tokens"], r["id"]))[:4]
    if verify and not selected:
        raise ValueError("Model reload verification requires a nonempty fixed reference bank")
    kwargs = dict(
        hidden_size=hidden_size, budget=budget, live_payload_bytes=live_payload_bytes, load_window=load_trajectories
    )
    before = score_native_rows(trainer, selected, **kwargs)[0] if verify else []
    trainer.actor_rollout_wg.save_checkpoint(
        str(destination), None, trainer.global_steps, max_ckpt_to_keep=max_ckpt_to_keep
    )
    if list(destination.glob("optim*")) or list(destination.glob("extra_state*")):
        raise RuntimeError("Final export unexpectedly saved optimizer or runtime state")
    tensor_rows = []
    hf = destination / "huggingface"
    for path in sorted(hf.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as stream:
            for name in stream.keys():
                value = stream.get_slice(name)
                tensor_rows.append(
                    {"name": name, "file": path.name, "dtype": value.get_dtype(), "shape": value.get_shape()}
                )
    if not tensor_rows or len({r["name"] for r in tensor_rows}) != len(tensor_rows):
        raise ValueError("HF export has missing or duplicate tensor identities")
    if trainer.config.actor_rollout_ref.actor.checkpoint.get("preserve_hf_master_weights", False):
        if any(row["dtype"] != "F32" for row in tensor_rows):
            raise ValueError("Master-weight export contains non-FP32 tensors")
    max_error = None
    if verify:
        trainer.actor_rollout_wg.load_checkpoint(str(destination), None, del_local_after_load=False)
        after = score_native_rows(trainer, selected, **kwargs)[0]
        old = {r["id"]: r for r in before}
        errors = []
        for row in after:
            reference = old.pop(row["id"])
            if reference["tokens"] != row["tokens"] or not row["tokens"]:
                raise ValueError("Reload parity changed the scored positions")
            left, right = reference["rkl_sum"] / row["tokens"], row["rkl_sum"] / row["tokens"]
            if not math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-7):
                raise RuntimeError("Final native model reload changed fixed-input RKL")
            errors.append(abs(left - right))
        if old:
            raise ValueError("Reload parity lost trajectories")
        max_error = max(errors)
    audit = ArtifactStore.create(
        destination / "export-audit",
        kind="final_model_export",
        metadata={
            "native_reload_rkl": "passed" if verify else "not_requested",
            "max_question_rkl_error": max_error,
            "hf_reload_forward": "not_tested",
            "measure_id": before[0].get("measure_id") if before else None,
            "reference_bank": selected[0]["bank"] if selected else None,
            "reference_ids": [r["id"] for r in selected],
            "reference_generation_bank": selected[0]["artifact"] if selected else None,
            "reference_generation_sha256": file_sha256(ArtifactStore(selected[0]["artifact"]).manifest_path)
            if selected
            else None,
            "outer": trainer.global_steps,
            "optimizer_steps": trainer.actor_optimizer_step,
            "hf_files": {p.name: file_sha256(p) for p in sorted(hf.iterdir()) if p.is_file()},
        },
    )
    audit.write_rows(
        "tensors.parquet",
        tensor_rows,
        schema=pa.schema(
            [
                ("name", pa.string()),
                ("file", pa.string()),
                ("dtype", pa.string()),
                ("shape", pa.list_(pa.int64())),
            ]
        ),
    )
    audit.finish()
    # Keep only the portable final weights after successful native reload validation.
    if verify:
        for path in destination.glob("model_world_size_*_rank_*.pt"):
            path.unlink()
    destination.rename(published)
    return {"eval/checkpoint/native_reload_verified": int(verify), "eval/checkpoint/hf_tensors": len(tensor_rows)}
