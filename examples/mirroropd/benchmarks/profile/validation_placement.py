# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Four-GPU fixed-token validation placement probe, not a Trainer training loop.

FSDP2 backbones use verl load/offload helpers; FP32 heads are replicated for
bounded full-vocabulary scoring. Serial uses real TQ SimpleStorage. Parallel
uses two 2-rank meshes and a bounded NCCL hidden pipeline. Both finish with
four-rank student backbone plus output head resident. No optimizer is loaded.
"""

import argparse
import gc
import json
import os
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import ray
import torch
import torch.distributed as dist
import transfer_queue as tq
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoModel

from verl.trainer.distillation.hidden import FrozenTeacherHead
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.linear import project


def batches(rows, ranks, microbatch, token_budget=0):
    """Pack descending lengths while keeping equal collective counts on all ranks."""
    ordered = sorted(rows, key=lambda r: len(r["prompt_ids"]) + len(r["response_ids"]), reverse=True)
    if not rows or len(rows) % ranks or microbatch < 1:
        raise ValueError("Nonempty rows divisible by ranks and positive microbatch required")
    result = [[] for _ in range(ranks)]
    offset = 0
    while offset < len(ordered):
        width = len(ordered[offset]["prompt_ids"]) + len(ordered[offset]["response_ids"]) - 1
        count = min(microbatch, (len(ordered) - offset) // ranks)
        if token_budget:
            if width > token_budget:
                raise ValueError("Token budget cannot hold the longest complete input")
            count = min(count, token_budget // width)
        for rank in range(ranks):
            result[rank].append(ordered[offset + rank * count : offset + (rank + 1) * count])
        offset += ranks * count
    return result


class Recorder:
    def __init__(self):
        self.events = []
        self.cpu = defaultdict(float)
        self.bytes = defaultdict(int)

    @contextmanager
    def stage(self, name):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        a.record()
        with torch.profiler.record_function(name):
            yield
        b.record()
        self.events.append((name, a, b))
        self.cpu[name] += time.perf_counter() - start

    def finish(self):
        torch.cuda.synchronize()
        gpu = defaultdict(float)
        for name, a, b in self.events:
            gpu[name] += a.elapsed_time(b) / 1000
        return dict(cpu_seconds=dict(self.cpu), cuda_elapsed_seconds=dict(gpu), payload_bytes=dict(self.bytes))


class Backbone:
    def __init__(self, path, mesh, student):
        self.mesh = mesh
        self.model = AutoModel.from_pretrained(
            path,
            local_files_only=True,
            dtype=torch.float32 if student else torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()
        self.model.requires_grad_(False)
        # Cold construction is separate from all placement wall-clock timings.
        self.model.to(torch.cuda.current_device())
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in self.model.layers:
            fully_shard(layer, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        fully_shard(self.model, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        self.offload()

    def load(self):
        load_fsdp_model_to_gpu(self.model)

    def offload(self):
        self.model.reshard()
        offload_fsdp_model_to_cpu(self.model)

    def hidden(self, rows, packed=False):
        sequences = [r["prompt_ids"] + r["response_ids"][:-1] for r in rows]
        width = max(map(len, sequences))
        if packed:
            lengths = [len(s) for s in sequences]
            ids = torch.tensor([[t for seq in sequences for t in seq]], device="cuda")
            positions = torch.tensor([[i for n in lengths for i in range(n)]], device="cuda")
            cumulative = torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = self.model(
                    input_ids=ids,
                    position_ids=positions,
                    use_cache=False,
                    cu_seq_lens_q=cumulative,
                    cu_seq_lens_k=cumulative,
                    max_length_q=width,
                    max_length_k=width,
                ).last_hidden_state[0]
            segments = hidden.split(lengths)
            return [h[len(r["prompt_ids"]) - 1 :].to(torch.bfloat16) for h, r in zip(segments, rows, strict=True)]
        ids = torch.tensor([[0] * (width - len(s)) + s for s in sequences], device="cuda")
        mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in sequences], device="cuda")
        positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.model(
                input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False
            ).last_hidden_state
        return [
            h[i, width - len(s) + len(r["prompt_ids"]) - 1 :].to(torch.bfloat16)
            for i, (s, r) in enumerate(zip(sequences, rows, strict=True))
        ]


def score(hs, ht, sw, tw, chunk, record):
    sums = torch.zeros((), device="cuda", dtype=torch.float64)
    for start in range(0, len(hs), chunk):
        with record.stage("student_head"):
            sp = (
                project(hs[start : start + chunk], sw)
                if sw.dtype == torch.bfloat16
                else torch.nn.functional.linear(hs[start : start + chunk].float(), sw)
            )
        with record.stage("teacher_head"):
            tq_logits = (
                project(ht[start : start + chunk], tw)
                if tw.dtype == torch.bfloat16
                else torch.nn.functional.linear(ht[start : start + chunk].float(), tw)
            )
        with record.stage("log_softmax_pair"):
            lp, lq = sp.log_softmax(-1), tq_logits.log_softmax(-1)
        with record.stage("rkl_reduce"):
            sums += (lp.exp() * (lp - lq)).sum(-1).sum(dtype=torch.float64)
    return sums.item()


def phase_sync():
    torch.cuda.synchronize()
    dist.barrier()


def put_hidden(keys, partition, rows):
    """CPU-only I/O; CUDA events must stay on the main rank thread."""
    start = time.perf_counter()
    tq.kv_batch_put(
        keys=keys,
        partition_id=partition,
        fields=TensorDict(
            {"teacher_hidden": torch.nested.as_nested_tensor(rows, layout=torch.jagged)}, batch_size=[len(keys)]
        ),
    )
    return time.perf_counter() - start


def get_hidden(keys, partition):
    start = time.perf_counter()
    rows = tq.kv_batch_get(keys=keys, partition_id=partition)["teacher_hidden"].unbind()
    return rows, time.perf_counter() - start


def run_once(mode, tag, rows, resources, args, trace=False):
    rank = dist.get_rank()
    s4, t4, local_model, sh, th = resources
    record = Recorder()
    result = []
    score_sw = None
    keys = []
    phase_sync()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    partition = f"placement-{tag}"
    teacher_head_dtype = torch.bfloat16 if args.projection != "strict_fp32" else torch.float32
    if mode == "serial":
        local_batches = batches(rows, 4, args.microbatch, args.token_budget)[rank]
        io_pool = ThreadPoolExecutor(max_workers=1) if args.io_overlap else None
        pending_write = None
        with record.stage("teacher_load_four"):
            t4.load()
            torch.cuda.synchronize()
        for batch in local_batches:
            with record.stage("teacher_backbone"):
                hidden = t4.hidden(batch, packed=args.packed)
            with record.stage("hidden_d2h"):
                cpu_rows = [h.cpu() for h in hidden]
            record.bytes["d2h"] += sum(h.numel() * h.element_size() for h in cpu_rows)
            batch_keys = [str(row["id"]) for row in batch]
            if io_pool:
                if pending_write is not None:
                    with record.stage("tq_write_wait"):
                        record.cpu["tq_write_io"] += pending_write.result()
                pending_write = io_pool.submit(put_hidden, batch_keys, partition, cpu_rows)
            else:
                with record.stage("tq_put"):
                    put_hidden(batch_keys, partition, cpu_rows)
            keys.extend(batch_keys)
            del hidden, cpu_rows
        if pending_write is not None:
            with record.stage("tq_write_drain"):
                record.cpu["tq_write_io"] += pending_write.result()
        with record.stage("teacher_offload_four"):
            t4.offload()
        phase_sync()
        with record.stage("student_load_four"):
            s4.load()
            sw = sh.get(torch.device("cuda", torch.cuda.current_device()), torch.float32)
            score_sw = sw.to(torch.bfloat16) if args.projection == "native_bf16_fp32" else sw
            tw = th.get(torch.device("cuda", torch.cuda.current_device()), teacher_head_dtype)
            torch.cuda.synchronize()
        read_future = (
            io_pool.submit(get_hidden, [str(r["id"]) for r in local_batches[0]], partition) if io_pool else None
        )
        for batch_index, batch in enumerate(local_batches):
            with record.stage("student_backbone"):
                student_h = s4.hidden(batch, packed=args.packed)
            with record.stage("tq_get_wait" if io_pool else "tq_get"):
                cpu_rows, elapsed = (
                    read_future.result() if io_pool else get_hidden([str(r["id"]) for r in batch], partition)
                )
                record.cpu["tq_read_io"] += elapsed
            if io_pool and batch_index + 1 < len(local_batches):
                read_future = io_pool.submit(
                    get_hidden, [str(r["id"]) for r in local_batches[batch_index + 1]], partition
                )
            for row, hp, cpu in zip(batch, student_h, cpu_rows, strict=True):
                with record.stage("hidden_h2d"):
                    hq = cpu.to("cuda", non_blocking=True)
                record.bytes["h2d"] += hq.numel() * hq.element_size()
                result.append(
                    dict(id=row["id"], length=len(hp), rkl_sum=score(hp, hq, score_sw, tw, args.chunk, record))
                )
            del student_h, hp, hq, cpu, cpu_rows
        if io_pool:
            io_pool.shutdown(wait=True)
        with record.stage("cleanup_teacher_payload"):
            tq.kv_clear(keys=keys, partition_id=partition)
            del tw
            th.release()
    else:
        local_rank = rank % 2
        local_batches = batches(rows, 2, args.microbatch, args.token_budget)[local_rank]
        with record.stage("paired_load_two_plus_two"):
            local_model.load()
            if rank >= 2:
                sw = sh.get(torch.device("cuda", torch.cuda.current_device()), torch.float32)
                score_sw = sw.to(torch.bfloat16) if args.projection == "native_bf16_fp32" else sw
                tw = th.get(torch.device("cuda", torch.cuda.current_device()), teacher_head_dtype)
            torch.cuda.synchronize()
        phase_sync()
        if rank < 2:
            pending = deque()
            for batch in local_batches:
                with record.stage("teacher_backbone"):
                    hidden = local_model.hidden(batch, packed=args.packed)
                for h in hidden:
                    # Bound in-flight tensors; Work and tensor lifetimes are coupled.
                    if len(pending) >= args.pipeline_depth:
                        with record.stage("send_backpressure"):
                            work, old = pending.popleft()
                            work.wait()
                        del old
                    value = h.contiguous()
                    with record.stage("hidden_send"):
                        work = dist.isend(value, dst=rank + 2)
                    record.bytes["p2p_send"] += value.numel() * value.element_size()
                    pending.append((work, value))
                del hidden, h
            with record.stage("send_drain"):
                for work, value in pending:
                    work.wait()
            pending.clear()
            del work, value
        else:
            for batch in local_batches:
                with record.stage("student_backbone"):
                    student_h = local_model.hidden(batch, packed=args.packed)
                for row, hp in zip(batch, student_h, strict=True):
                    hq = torch.empty(
                        (len(row["response_ids"]), th.cpu_weight.shape[1]), device="cuda", dtype=torch.bfloat16
                    )
                    with record.stage("hidden_recv_wait"):
                        dist.recv(hq, src=rank - 2)
                    record.bytes["p2p_recv"] += hq.numel() * hq.element_size()
                    result.append(
                        dict(id=row["id"], length=len(hp), rkl_sum=score(hp, hq, score_sw, tw, args.chunk, record))
                    )
                del student_h, hp, hq
        phase_sync()
        with record.stage("paired_offload"):
            local_model.offload()
            if rank >= 2:
                del tw
                th.release()
        phase_sync()
        with record.stage("student_restore_four"):
            s4.load()
            sw = sh.get(torch.device("cuda", torch.cuda.current_device()), torch.float32)
            torch.cuda.synchronize()
    # Both methods end with identical 4-rank student weights and head ready.
    with record.stage("student_ready_probe"):
        probe = dict(rows[rank], response_ids=rows[rank]["response_ids"][:16])
        test = s4.hidden([probe])[0]
        checksum = torch.nn.functional.linear(test[-1:].float(), sw[:16]).cpu().tolist()
        del test
    phase_sync()
    wall = time.perf_counter() - started
    summary = dict(
        rank=rank,
        mode=mode,
        tag=tag,
        wall_seconds=wall,
        profile=trace,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        checksum=checksum,
        **record.finish(),
    )
    summaries = [None] * 4
    dist.all_gather_object(summaries, summary)
    all_rows = [None] * 4
    dist.all_gather_object(all_rows, result)
    if rank == 0:
        flattened = sorted([x for part in all_rows for x in part], key=lambda x: x["id"])
        assert len(flattened) == len(rows) and len({x["id"] for x in flattened}) == len(rows)
        report = dict(
            mode=mode,
            tag=tag,
            rows=len(rows),
            response_tokens=sum(len(r["response_ids"]) for r in rows),
            prompt_tokens=sum(len(r["prompt_ids"]) for r in rows),
            wall_seconds=max(x["wall_seconds"] for x in summaries),
            ranks=summaries,
            results=flattened,
            profile=trace,
            configuration={
                k: getattr(args, k)
                for k in ["microbatch", "token_budget", "io_overlap", "pipeline_depth", "projection", "chunk", "packed"]
            },
        )
        (args.output / f"{tag}.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k not in ("ranks", "results")}), flush=True)
    # Reset to the common CPU-resident starting condition, outside timed window.
    del sw, score_sw
    s4.offload()
    sh.release()
    gc.collect()
    torch.cuda.empty_cache()
    phase_sync()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--token-budget", type=int, default=0)
    parser.add_argument("--io-overlap", action="store_true")
    parser.add_argument("--pipeline-depth", type=int, default=2)
    parser.add_argument(
        "--projection", choices=["strict_fp32", "teacher_bf16_fp32", "native_bf16_fp32"], default="strict_fp32"
    )
    parser.add_argument("--configs", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=["serial", "parallel"], default=["serial", "parallel"])
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace-only", action="store_true")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_num_threads(2)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 4
    ray.init(address=args.ray_address, log_to_driver=False)
    tq.init()
    args.output.mkdir(parents=True, exist_ok=True)
    mesh4 = DeviceMesh("cuda", [0, 1, 2, 3])
    mesh_t = DeviceMesh("cuda", [0, 1])
    mesh_s = DeviceMesh("cuda", [2, 3])
    rank = dist.get_rank()
    rows = json.loads(args.data.read_text())[: args.count]
    if args.smoke:
        rows = [dict(r, prompt_ids=r["prompt_ids"][-64:], response_ids=r["response_ids"][:64]) for r in rows]
    start = time.perf_counter()
    s4 = Backbone(args.student, mesh4, True)
    phase_sync()
    t4 = Backbone(args.teacher, mesh4, False)
    phase_sync()
    local_model = Backbone(args.teacher if rank < 2 else args.student, mesh_t if rank < 2 else mesh_s, rank >= 2)
    sh, th = FrozenTeacherHead(args.student), FrozenTeacherHead(args.teacher)
    phase_sync()
    if rank == 0:
        (args.output / "setup.json").write_text(
            json.dumps(
                dict(
                    arguments=vars(args) | {k: str(v) for k, v in vars(args).items() if isinstance(v, Path)},
                    cold_setup_seconds=time.perf_counter() - start,
                    torch=torch.__version__,
                    device=torch.cuda.get_device_name(),
                    tf32=torch.backends.cuda.matmul.allow_tf32,
                    bf16_reduced_precision_reduction=False,
                    scope=__doc__,
                ),
                indent=2,
            )
        )
    resources = s4, t4, local_model, sh, th
    configurations = json.loads(args.configs.read_text()) if args.configs else [dict(name="default")]
    baseline = {
        k: getattr(args, k)
        for k in [
            "microbatch",
            "token_budget",
            "io_overlap",
            "pipeline_depth",
            "projection",
            "chunk",
            "packed",
            "modes",
        ]
    }
    with torch.no_grad():
        for configuration in configurations:
            unknown = set(configuration) - set(baseline) - {"name"}
            if unknown:
                raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
            name = configuration["name"]
            if not name.replace("-", "").replace("_", "").isalnum():
                raise ValueError("Unsafe configuration name")
            for key, value in baseline.items():
                setattr(args, key, configuration.get(key, value))
            if min(args.microbatch, args.chunk, args.pipeline_depth) < 1 or args.token_budget < 0:
                raise ValueError("Positive batch, chunk, pipeline depth and nonnegative token budget required")
            if not args.modes or set(args.modes) - {"serial", "parallel"}:
                raise ValueError("Expected serial and/or parallel modes")
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            for mode in args.modes:
                run_once(mode, f"{name}-warmup-{mode}", rows[:8], resources, args)
            order = (
                []
                if args.trace_only
                else [m for i in range(args.repeats) for m in (args.modes if i % 2 == 0 else args.modes[::-1])]
            )
            if args.smoke:
                order = args.modes
            for i, mode in enumerate(order):
                run_once(mode, f"{name}-measure-{i:02d}-{mode}", rows, resources, args)
            if args.trace:
                for mode in args.modes:
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=False,
                        profile_memory=False,
                    ) as prof:
                        run_once(
                            mode,
                            f"{name}-profile-{mode}",
                            sorted(rows, key=lambda r: len(r["response_ids"]), reverse=True)[:8],
                            resources,
                            args,
                            True,
                        )
                    prof.export_chrome_trace(str(args.output / f"{name}-{mode}-rank{rank}.trace.json.gz"))
    phase_sync()
    ray.shutdown()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
