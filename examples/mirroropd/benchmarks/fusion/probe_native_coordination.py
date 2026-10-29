# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Run the production refit coordinator with two actor ranks and native IPC receivers.

Small Qwen3 checkpoints isolate collective-bearing export and one-rank failures.
Real dense/MoE model capacity and FSDP training are accepted by separate lifecycles.
"""

import argparse
import asyncio
import inspect
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist


class LocalRPC:
    def __init__(self, llm):
        self.llm = llm
        self.pool = ThreadPoolExecutor(max_workers=1)

    def remote(self, method, **kwargs):
        # Ray starts a remote receiver immediately; start this local stand-in
        # before the IPC sender enters its blocking socket handshake as well.
        return asyncio.wrap_future(self.pool.submit(self.llm.collective_rpc, method, timeout=180, **kwargs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tp", type=int, choices=[1, 2], required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6,7":
        raise ValueError("This two-actor probe requires physical GPUs 6,7")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    from transformers import AutoConfig, Qwen3ForCausalLM
    from vllm import LLM

    from verl.workers.engine_workers import ActorRolloutRefWorker
    from verl.workers.rollout.vllm_fusion.adapter import ServerAdapter
    from verl.workers.rollout.vllm_fusion.compat import fusion_hf_overrides

    root = args.out.resolve()
    if rank == 0:
        root.mkdir(parents=True, exist_ok=False)
        config = AutoConfig.from_pretrained(args.tokenizer, local_files_only=True)
        config.hidden_size, config.intermediate_size, config.num_hidden_layers = 128, 256, 1
        config.num_attention_heads = config.num_key_value_heads = 2
        config.head_dim, config.max_position_embeddings = 64, 1024
        for name in ["student", "teacher"]:
            path = root / name
            torch.manual_seed(42 if name == "student" else 43)
            Qwen3ForCausalLM(config).to(torch.bfloat16).save_pretrained(path)
            for filename in [
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json",
            ]:
                source = args.tokenizer.resolve() / filename
                if source.exists():
                    (path / filename).symlink_to(source)
    dist.barrier()
    student, teacher = root / "student", root / "teacher"
    replica, tp_rank = rank // args.tp, rank % args.tp
    job = f"coordination-{os.environ['MASTER_PORT']}"
    os.environ["VERL_RAY_JOB_ID"] = job
    os.environ["VERL_REPLICA_RANK"] = str(4000 + replica)
    llm = None
    if tp_rank == 0:
        if args.tp == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(6 + rank)
        llm = LLM(
            model=str(student),
            hf_overrides=fusion_hf_overrides(str(teacher)),
            tensor_parallel_size=args.tp,
            dtype="bfloat16",
            enforce_eager=True,
            max_model_len=128,
            max_num_seqs=1,
            max_num_batched_tokens=128,
            kv_cache_memory_bytes=128 * 1024**2,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            attention_config={"backend": "TRITON_ATTN"},
            worker_extension_cls="verl.workers.rollout.vllm_fusion.worker_extension.FusionWorkerExtension",
        )
        llm.collective_rpc("fusion_initialize", args=(True,))
        before = llm.collective_rpc("fusion_weight_digests")
    torch.cuda.set_device(rank)
    dist.barrier()

    target = object.__new__(ServerAdapter)
    target.settings = {"backend": "vllm_fusion", "current": True, "deadline_seconds": 180}
    target.config = SimpleNamespace(checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=16))
    target.leader, target.tp_rank = tp_rank == 0, tp_rank
    target.loaded_version = target.pending_version = None
    target.server = SimpleNamespace(collective_rpc=LocalRPC(llm)) if llm else None
    target.address = f"ipc:///tmp/rl-colocate-zmq-{job}-replica-{4000 + replica}-rank-{tp_rank}.sock"
    from safetensors import safe_open

    records = []
    for fail in [False, True]:
        visited = []

        def stream(visited=visited, fail=fail):
            for path in sorted(student.glob("*.safetensors")):
                with safe_open(path, framework="pt", device="cpu") as handle:
                    for index, name in enumerate(handle.keys()):
                        value = handle.get_tensor(name).to(f"cuda:{rank}")
                        # Collective-bearing stream: a receiver must drain after local errors.
                        dist.all_reduce(value)
                        value.div_(dist.get_world_size())
                        if name == "model.norm.weight":
                            value.mul_(1.125)
                        visited.append(name)
                        yield "injected_bad_name" if fail and rank == 1 and index == 0 else name, value

        worker = SimpleNamespace(
            config=SimpleNamespace(rollout=SimpleNamespace(checkpoint_engine=SimpleNamespace(backend="naive"))),
            actor=SimpleNamespace(
                engine=SimpleNamespace(
                    get_per_tensor_param=lambda **kwargs: (stream(), None), is_param_offload_enabled=False
                )
            ),
            fusion_rollout=target,
            layered_summon=False,
        )
        error = None
        version = (2, 2) if fail else (1, 1)
        try:
            asyncio.run(
                inspect.unwrap(ActorRolloutRefWorker.update_weights)(worker, target="fusion_student", version=version)
            )
        except RuntimeError as exc:
            error = str(exc)
        if bool(error) != fail or target.loaded_version != (None if fail else version):
            raise RuntimeError(f"Actor rank {rank} violated commit protocol: {error}, {target.loaded_version}")
        coverage = [None] * 2
        dist.all_gather_object(coverage, visited)
        if not visited or coverage[0] != coverage[1]:
            raise RuntimeError("Actors did not exhaust the same collective-bearing stream")
        if llm:
            audit = llm.collective_rpc("fusion_audit")
            if any(row["version"] != (None if fail else list(version)) for row in audit):
                raise RuntimeError("Receiver published a partial or stale version")
            after = llm.collective_rpc("fusion_weight_digests")
            if any(a["teacher"] != b["teacher"] for a, b in zip(before, after, strict=True)):
                raise RuntimeError("Actor refit changed teacher weights")
            if not fail and any(a["student"] == b["student"] for a, b in zip(before, after, strict=True)):
                raise RuntimeError("Actor refit did not change student weights")
        records.append(
            {
                "rank": rank,
                "tp": args.tp,
                "replica": replica,
                "tp_rank": tp_rank,
                "injected_failure": fail,
                "tensors_consumed": len(visited),
                "loaded_version": target.loaded_version,
                "error": error,
            }
        )
        dist.barrier()
    (root / f"rank-{rank}.json").write_text(json.dumps(records, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
