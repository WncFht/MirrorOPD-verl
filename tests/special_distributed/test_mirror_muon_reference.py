# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Four-rank CPU/GPU oracle for full-matrix Muon, including uneven row shards."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/mirroropd/docs/mirror-descent/scripts"))
from solver_optimizers import FullMatrixMuonAdamW


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.device == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl" if args.device == "cuda" else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    mesh = init_device_mesh(args.device, (world,))
    torch.manual_seed(7)
    cases = []
    for shape in [(13, 7), (7, 13)]:
        full_matrix = torch.randn(*shape, device=args.device)
        full_bias = torch.randn(7, device=args.device)
        matrix = torch.nn.Parameter(distribute_tensor(full_matrix.clone(), mesh, [Shard(0)]))
        bias = torch.nn.Parameter(distribute_tensor(full_bias.clone(), mesh, [Shard(0)]))
        ref_matrix, ref_bias = [torch.nn.Parameter(x.clone()) for x in (full_matrix, full_bias)]
        optimizer = FullMatrixMuonAdamW([("model.layers.0.weight", matrix), ("model.norm.weight", bias)], lr=0.002)
        muon = torch.optim.Muon([ref_matrix], lr=0.002, weight_decay=0, adjust_lr_fn="match_rms_adamw")
        adam = torch.optim.AdamW([ref_bias], lr=0.002, weight_decay=0, foreach=False)
        errors = []
        for step in range(4):
            gm, gb = torch.randn_like(full_matrix), torch.randn_like(full_bias)
            matrix.grad, bias.grad = distribute_tensor(gm, mesh, [Shard(0)]), distribute_tensor(gb, mesh, [Shard(0)])
            ref_matrix.grad, ref_bias.grad = gm, gb
            optimizer.step()
            muon.step()
            adam.step()
            got_matrix, got_bias = matrix.full_tensor(), bias.full_tensor()
            torch.testing.assert_close(got_matrix, ref_matrix, atol=0, rtol=0)
            torch.testing.assert_close(got_bias, ref_bias, atol=0, rtol=0)
            errors.append(
                [
                    float((got_matrix - ref_matrix).abs().max().detach()),
                    float((got_bias - ref_bias).abs().max().detach()),
                ]
            )
            if step == 1:
                state = copy.deepcopy(optimizer.state_dict())
                optimizer = FullMatrixMuonAdamW(
                    [("model.layers.0.weight", matrix), ("model.norm.weight", bias)], lr=0.1
                )
                optimizer.load_state_dict(state)
        cases.append({"shape": shape, "errors": errors, "partition": optimizer.partition})
    if rank == 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "device": args.device,
                    "world_size": world,
                    "cases": cases,
                    "resume": True,
                },
                indent=2,
            )
            + "\n"
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
