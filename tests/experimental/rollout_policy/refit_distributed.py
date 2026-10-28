# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Four-rank production named-refit RPC body, including one-rank receive failure."""

import argparse
import asyncio
import inspect
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from transformers import Qwen3Config, Qwen3ForCausalLM

from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.rollout.hf_fusion.adapter import ServerAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3":
        raise ValueError("Use only physical GPUs 0–3")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    records = []
    try:
        for fail in [True, False]:
            torch.manual_seed(9)
            config = Qwen3Config(
                vocab_size=17,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                tie_word_embeddings=True,
            )
            target = object.__new__(ServerAdapter)
            target.settings = {"current": True}
            target.student = Qwen3ForCausalLM(config).eval().requires_grad_(False)
            target.teacher = Qwen3ForCausalLM(config).eval().requires_grad_(False)
            teacher_before = {key: value.clone() for key, value in target.teacher.state_dict().items()}
            target.generating, target.valid, target.loaded_version = False, True, (0, 0)
            state = list(target.student.named_parameters())
            visited = []

            def stream(state=state, visited=visited, fail=fail):
                for index, (name, parameter) in enumerate(state):
                    value = torch.full_like(parameter, 0.125, device=rank)
                    dist.all_reduce(value)
                    value.div_(dist.get_world_size())
                    visited.append(name)
                    yield "injected_bad_name" if fail and rank == 1 and index == 0 else name, value

            engine = SimpleNamespace(
                get_per_tensor_param=lambda **kwargs: (stream(), None), is_param_offload_enabled=False
            )
            worker = SimpleNamespace(
                config=SimpleNamespace(rollout=SimpleNamespace(checkpoint_engine=SimpleNamespace(backend="naive"))),
                actor=SimpleNamespace(engine=engine),
                fusion_rollout=target,
                layered_summon=False,
            )
            error = None
            try:
                asyncio.run(
                    inspect.unwrap(ActorRolloutRefWorker.update_weights)(
                        worker, target="fusion_student", version=(3, 16)
                    )
                )
            except RuntimeError as exc:
                error = str(exc)
            assert bool(error) == fail
            assert len(visited) == len(state), "Every rank must exhaust all collective-bearing tensors"
            assert target.valid == (not fail)
            assert target.loaded_version == (None if fail else (3, 16))
            if not fail:
                assert all(torch.equal(value, torch.full_like(value, 0.125)) for value in target.student.parameters())
            for name, value in target.teacher.state_dict().items():
                torch.testing.assert_close(value, teacher_before[name], rtol=0, atol=0)
            records.append(
                {
                    "rank": rank,
                    "injected_failure": fail,
                    "tensors_consumed": len(visited),
                    "target_valid": target.valid,
                    "loaded_version": target.loaded_version,
                    "teacher_unchanged": True,
                    "error": error,
                }
            )
            dist.barrier()
        output = Path(args.out)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"rank-{rank}.json").write_text(json.dumps(records, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
