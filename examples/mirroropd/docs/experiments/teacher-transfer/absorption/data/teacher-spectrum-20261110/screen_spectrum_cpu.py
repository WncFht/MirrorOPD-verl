"""CPU spectrum screen: one additional SFT teacher, three prespecified matrices.

The measured objects are full weights B/T/S, updates T-B/S-B, and residual T-S.
Rows are matrices, not independent teacher replications. Rank16 is a component
count. Source tensors are promoted to float64 before subtraction. No GPU use.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))
from examples.mirroropd.research.studies.spectral.decomposition import (  # noqa: E402
    spectrum_summary,
    weight_summary,
)
from examples.mirroropd.research.studies.spectral.weights import TensorCheckpoint, tensor_digest  # noqa: E402

OUT = ROOT / "spectrum-screen-v1"


def dump(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    torch.set_num_threads(2)
    if OUT.exists():
        raise FileExistsError(f"Preserve existing evidence: {OUT}")
    OUT.mkdir()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    config = next(x for x in manifest["configs"] if x["label"] == "qwen-sft")
    catalog = json.loads((ROOT.parent / "teacher-difference-stages-20261109/qwen-catalog.json").read_text())
    checkpoints = {x["step"]: x for x in catalog["available"] if x["arm"] == "Q3-sft-deepmath"}
    base = TensorCheckpoint(config["base"])
    teacher = TensorCheckpoint(checkpoints[320]["teacher"])
    students = {step: TensorCheckpoint(checkpoints[step]["checkpoint"]) for step in [64, 320]}
    for other in [teacher, *students.values()]:
        base.compatible(other)
    protocol = dict(
        label="qwen-thinking-space-sft",
        device="cpu",
        arithmetic="float64",
        threads=2,
        layers=[0, 13, 27],
        module="self_attn.k_proj",
        steps=[64, 320],
        checkpoints=dict(
            base=base.identity, teacher=teacher.identity, students={str(k): v.identity for k, v in students.items()}
        ),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        implementation={
            name: hashlib.sha256(
                (ROOT / "source/examples/mirroropd/research/studies/spectral" / name).read_bytes()
            ).hexdigest()
            for name in ["weights.py", "decomposition.py"]
        },
        semantics="stored tensors promoted before subtraction; exact SVD; three matrices only",
    )
    dump(OUT / "protocol.json", protocol)
    started = time.monotonic()
    for layer in protocol["layers"]:
        key = f"model.layers.{layer}.self_attn.k_proj.weight"
        raw_b, raw_t = base.get(key), teacher.get(key)
        b, t = raw_b.double(), raw_t.double()
        frame = torch.linalg.svd(b, full_matrices=False)
        ut, st, vt = torch.linalg.svd(t - b, full_matrices=False)
        full_t = torch.linalg.svdvals(t)
        row = dict(
            label=protocol["label"],
            tensor=key,
            shape=list(b.shape),
            base_dtype=str(raw_b.dtype),
            teacher_dtype=str(raw_t.dtype),
            base_sha256=tensor_digest(raw_b),
            teacher_sha256=tensor_digest(raw_t),
            base_singular_values=frame[1].tolist(),
            teacher_singular_values=full_t.tolist(),
            teacher_update_singular_values=st.tolist(),
            teacher=weight_summary(b, t, frame=frame, singular_values=st),
            stages=[],
        )
        for step, checkpoint in students.items():
            raw_s = checkpoint.get(key)
            s = raw_s.double()
            us, ss, vs = torch.linalg.svd(s - b, full_matrices=False)
            sm = torch.linalg.svdvals(t - s)
            norm_t, norm_s = (t - b).norm(), (s - b).norm()
            stage = dict(
                step=step,
                dtype=str(raw_s.dtype),
                sha256=tensor_digest(raw_s),
                weight_singular_values=torch.linalg.svdvals(s).tolist(),
                update_singular_values=ss.tolist(),
                missing_singular_values=sm.tolist(),
                opd=weight_summary(b, s, frame=frame, singular_values=ss),
                missing=spectrum_summary(t - s, singular_values=sm),
                update_cosine=float(((t - b) * (s - b)).sum() / (norm_t * norm_s)),
                subspace16=dict(
                    left_overlap=float((ut[:, :16].T @ us[:, :16]).square().sum() / 16),
                    right_overlap=float((vt[:16] @ vs[:16].T).square().sum() / 16),
                ),
            )
            assert abs(sum(x * x for x in ss.tolist()) / float(norm_s.square()) - 1) < 1e-8
            row["stages"].append(stage)
        assert float((b - (frame[0] * frame[1]) @ frame[2]).norm() / b.norm()) < 1e-8
        dump(OUT / f"thinking-space-l{layer:02d}.json", row)
        dump(OUT / "progress.json", dict(last_layer=layer, elapsed=time.monotonic() - started))
        print(layer, round(time.monotonic() - started, 2), flush=True)
    dump(OUT / "complete.json", dict(matrices=3, student_observations=6, elapsed=time.monotonic() - started))


if __name__ == "__main__":
    main()
