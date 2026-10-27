# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Current-code paired operator benchmark; shared GPU, not an end-to-end claim."""

import gc
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

from verl.trainer.distillation.full_vocab import full_reverse_kl_result


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(0.24)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(__file__).parent
    info = dict(
        scope="head projection + KL + hidden/head VJP only; shared H800; no backbone, optimizer, TQ, rollout",
        torch=torch.__version__,
        device=torch.cuda.get_device_name(),
        N=4096,
        V=151936,
        d=1536,
        chunk=512,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    hs = torch.randn(4096, 1536, device="cuda", dtype=torch.bfloat16) * 0.1
    ht = torch.randn_like(hs) * 0.1
    ws = torch.randn(151936, 1536, device="cuda", dtype=torch.bfloat16) * 0.1
    wt = torch.randn_like(ws) * 0.1
    hs.requires_grad_()
    ws.requires_grad_()
    names = ["dense", "recompute", "precompute", "precompute_fused"]

    def run(name):
        result = full_reverse_kl_result(hs, ws, ht, wt, backend=name, token_chunk=512, vocab_chunk=8192)
        grad = torch.autograd.grad(result.loss_sum, (hs, ws))
        return result, grad

    for _ in range(2):
        for name in names:
            run(name)
    torch.cuda.synchronize()
    timings = {name: [] for name in names}
    for iteration in range(9):
        for name in names if iteration % 2 == 0 else names[::-1]:
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = run(name)
            torch.cuda.synchronize()
            timings[name].append((time.perf_counter() - start) * 1000)
            del result
    output = []
    for name in names:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        result = run(name)
        torch.cuda.synchronize()
        output.append(
            dict(
                backend=name,
                wall_ms=timings[name],
                median_ms=statistics.median(timings[name]),
                allocated_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                loss=float(result[0].loss_sum),
            )
        )
        del result
    info["rows"] = output
    (root / "benchmark.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
