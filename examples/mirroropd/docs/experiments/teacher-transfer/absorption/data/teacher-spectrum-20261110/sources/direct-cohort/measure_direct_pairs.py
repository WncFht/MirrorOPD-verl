"""CPU all-layer update geometry and seeded middle-layer spectral sketches.

All attention/MLP weight matrices use exact FP64 norms and inner products.
Five middle-layer q/o/MLP matrices use two seeded rank16 randomized SVDs
(oversample8, two power iterations). Captured energy is a lower bound on
optimal top16 energy, not an exact full spectrum. Student endpoints and modes
retain their identities. No model forward pass or GPU is used.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from measure_expanded_spectrum import TensorCheckpoint, tensor_digest

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "direct-pairs-v1"


def sketch(d, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    r = min(24, *d.shape)
    omega = torch.randn(d.shape[1], r, generator=generator, dtype=d.dtype)
    q = torch.linalg.qr(d @ omega, mode="reduced")[0]
    for _ in range(2):
        z = torch.linalg.qr(d.T @ q, mode="reduced")[0]
        q = torch.linalg.qr(d @ z, mode="reduced")[0]
    small = q.T @ d
    u, s, v = torch.linalg.svd(small, full_matrices=False)
    u = q @ u[:, :16]
    s = s[:16]
    v = v[:16]
    total = float(d.square().sum())
    captured = float(s.square().sum())
    assert -1e-10 <= captured / total <= 1 + 1e-8
    return dict(
        seed=seed, rank=len(s), captured_energy=captured, energy_fraction=captured / total, singular_values=s.tolist()
    ), (u, s, v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    pairs = [p for p in json.loads((OUT / "inventory.json").read_text())["pairs"] if p["family"] == args.family]
    start = time.monotonic()
    count = 0
    stagecount = 0
    for p in pairs:
        bck = TensorCheckpoint(p["base"])
        tck = TensorCheckpoint(p["teacher_path"])
        bck.compatible(tck)
        students = {}
        for c in p["conditions"]:
            for step, path in c["checkpoints"].items():
                students[c["run_id"], int(step), path] = TensorCheckpoint(path)
        for sc in students.values():
            bck.compatible(sc)
        n = json.loads((Path(p["base"]) / "config.json").read_text())["num_hidden_layers"]
        middle = (n - 1) // 2
        keys = bck.selected(r"model\.layers\.\d+\.(self_attn|mlp)\..*\.weight$")
        if args.smoke:
            keys = [k for k in keys if k == f"model.layers.{middle}.mlp.up_proj.weight"]
        for ki, key in enumerate(keys):
            folder = OUT / ("smoke" if args.smoke else "matrices") / p["pair_id"]
            dest = folder / (key + ".json")
            if dest.exists():
                assert json.loads(dest.read_text())["source_sha256"] == source_sha
                continue
            then = time.monotonic()
            braw = bck.get(key)
            traw = tck.get(key)
            b = braw.double()
            d = traw.double() - b
            eb = float(b.square().sum())
            ed = float(d.square().sum())
            numel = d.numel()
            sample = d.flatten()[:: max(numel // 65536, 1)]
            second = float(sample.square().mean())
            fourth = float(sample.square().square().mean())
            row = dict(
                pair_id=p["pair_id"],
                teacher=p["teacher"],
                family=p["family"],
                tensor=key,
                shape=list(d.shape),
                elements=numel,
                base_energy=eb,
                delta_energy=ed,
                weight_dot_delta=float((b * d).sum()),
                delta_L1=float(d.abs().sum()),
                sample_kurtosis=fourth / (second * second),
                sample_mean=float(sample.mean()),
                sample_second_moment=second,
                base_sha256=tensor_digest(braw),
                teacher_sha256=tensor_digest(traw),
                stages=[],
                sketches=[],
                source_sha256=source_sha,
            )
            do_sketch = key.startswith(f"model.layers.{middle}.") and any(
                m in key for m in ["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            frames = []
            if do_sketch:
                for seed in [101, 202]:
                    spec, frame = sketch(d, seed)
                    row["sketches"].append(spec)
                    frames.append(frame)
            for (run_id, step, path), sc in students.items():
                sraw = sc.get(key)
                ds = sraw.double() - b
                es = float(ds.square().sum())
                dot = float((d * ds).sum())
                missing = float((d - ds).square().sum())
                assert abs(missing - (ed + es - 2 * dot)) < 1e-8 * max(missing, 1)
                stage = dict(
                    run_id=run_id,
                    step=step,
                    path=path,
                    student_sha256=tensor_digest(sraw),
                    student_energy=es,
                    dot=dot,
                    missing_energy=missing,
                    sketches=[],
                )
                for (u, s, v), sk in zip(frames, row["sketches"], strict=True):
                    diag = (u.T @ ds @ v.T).diagonal()
                    energy = float(s.square().sum())
                    gain = float((diag * s).sum()) / energy
                    stage["sketches"].append(
                        dict(
                            seed=sk["seed"],
                            amplitude_gain=gain,
                            matched_residual=float((diag - s).square().sum()) / energy,
                            matched_energy=float(diag.square().sum()),
                        )
                    )
                row["stages"].append(stage)
                stagecount += 1
            folder.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(row, indent=2))
            count += 1
            if do_sketch or ki % 28 == 0:
                print(
                    args.family,
                    p["teacher"],
                    key,
                    "seconds",
                    round(time.monotonic() - then, 2),
                    "elapsed",
                    round(time.monotonic() - start, 1),
                    flush=True,
                )
        if args.smoke:
            break
    (OUT / (args.family + ("-smoke" if args.smoke else "-complete") + ".json")).write_text(
        json.dumps(
            dict(
                family=args.family,
                pairs=len(pairs),
                matrices_this_attempt=count,
                student_matrices_this_attempt=stagecount,
                elapsed=time.monotonic() - start,
                source_sha256=source_sha,
                device="CPU FP64, threads2",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
