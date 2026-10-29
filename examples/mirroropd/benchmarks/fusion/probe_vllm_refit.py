# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise native TP receivers with real IPC, checkpoint tensors and one-rank faults."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def send_checkpoint(args):
    import torch
    from safetensors import safe_open

    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

    torch.cuda.set_device(args.rank)
    names = []
    injected = False

    def weights():
        nonlocal injected
        for path in sorted(Path(args.student).glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    value = handle.get_tensor(name).to(f"cuda:{args.rank}")
                    if name == "model.norm.weight":
                        value = value * 1.125
                    if args.fault and not injected:
                        value = value.flatten()[:1]
                        injected = True
                    names.append(name)
                    yield name, value

    address = f"ipc:///tmp/rl-colocate-zmq-{os.environ['VERL_RAY_JOB_ID']}-replica-9000-rank-{args.rank}.sock"
    sender = BucketedWeightSender(address, bucket_size_mb=128)
    asyncio.run(sender.async_send_weights(weights()))
    Path(args.out).write_text(json.dumps({"rank": args.rank, "sent": names, "injected": injected}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tp", type=int, choices=[1, 2], default=2)
    parser.add_argument("--sender", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--fault", action="store_true")
    args = parser.parse_args()
    if args.sender:
        send_checkpoint(args)
        return
    if len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) != args.tp:
        raise ValueError("Allocate exactly the TP group for this acceptance probe")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["VERL_RAY_JOB_ID"] = f"fusion-refit-{os.getpid()}"
    os.environ["VERL_REPLICA_RANK"] = "9000"

    import torch
    from fingerprint import source_fingerprint
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from verl.workers.rollout.vllm_fusion.compat import fusion_hf_overrides

    (out / "source.json").write_text(json.dumps(source_fingerprint(), indent=2) + "\n")
    (out / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    llm = LLM(
        model=args.student,
        hf_overrides=fusion_hf_overrides(args.teacher),
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=256,
        max_num_seqs=1,
        max_num_batched_tokens=256,
        kv_cache_memory_bytes=512 * 1024**2,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        logprobs_mode="processed_logprobs",
        attention_config={"backend": "TRITON_ATTN"},
        worker_extension_cls="verl.workers.rollout.vllm_fusion.worker_extension.FusionWorkerExtension",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.student, local_files_only=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 17 times 19?"}], add_generation_prompt=True, return_dict=False
    )
    params = SamplingParams(temperature=1, seed=42, max_tokens=1, ignore_eos=True, logprobs=0)
    llm.collective_rpc("fusion_capture", args=(True,))
    llm.collective_rpc("fusion_prepare", args=(0.5,))
    llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)
    llm.collective_rpc("fusion_save_logits", args=(str(out / "before.pt"),))
    before = llm.collective_rpc("fusion_weight_digests")
    llm.collective_rpc("fusion_initialize", args=(True,))

    def transfer(label, version, fault=False):
        children, streams = [], []
        with ThreadPoolExecutor(max_workers=1) as pool:
            receiving = pool.submit(
                llm.collective_rpc,
                "fusion_receive_weights",
                timeout=300,
                kwargs={"version": version, "use_shm": False},
            )
            try:
                for rank in range(args.tp):
                    cmd = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--sender",
                        "--student",
                        args.student,
                        "--rank",
                        str(rank),
                        "--out",
                        str(out / f"{label}-sender-{rank}.json"),
                    ]
                    if fault and rank == args.tp - 1:
                        cmd.append("--fault")
                    stream = (out / f"{label}-sender-{rank}.log").open("x")
                    streams.append(stream)
                    children.append(subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT))
                for child in children:
                    if child.wait(timeout=300) != 0:
                        raise RuntimeError("IPC sender failed; retain logs and terminate this probe")
                errors = receiving.result(timeout=300)
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill()
                        child.wait()
                for stream in streams:
                    stream.close()
        (out / f"{label}-receivers.json").write_text(json.dumps(errors) + "\n")
        sent = [json.loads((out / f"{label}-sender-{rank}.json").read_text()) for rank in range(args.tp)]
        if any(row["sent"] != sent[0]["sent"] for row in sent):
            raise RuntimeError("Ranks did not finish the same complete checkpoint stream")
        return errors, sent

    errors, success_streams = transfer("success", [1, 1])
    if any(errors):
        raise RuntimeError(f"Valid checkpoint refit failed: {errors}")
    llm.collective_rpc("fusion_commit_version", args=([1, 1],))
    audit = llm.collective_rpc("fusion_prepare", args=(0.5, [1, 1]))
    (out / "success-versions.json").write_text(json.dumps(audit, indent=2) + "\n")
    if any(tuple(row["version"]) != (1, 1) for row in audit):
        raise RuntimeError("TP ranks did not publish the same version")
    llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)
    llm.collective_rpc("fusion_save_logits", args=(str(out / "after.pt"),))
    after = llm.collective_rpc("fusion_weight_digests")
    if any(a["teacher"] != b["teacher"] or a["student"] == b["student"] for a, b in zip(before, after, strict=True)):
        raise RuntimeError("Refit changed teacher or failed to change student on a rank")
    left = torch.load(out / "before.pt", map_location="cpu", weights_only=True)
    right = torch.load(out / "after.pt", map_location="cpu", weights_only=True)
    if torch.equal(left["student"], right["student"]) or not torch.equal(left["teacher"], right["teacher"]):
        raise RuntimeError("Refit output evidence disagrees with component isolation")

    def rejects(method, *values):
        try:
            llm.collective_rpc(method, args=values)
        except Exception as exc:
            return str(exc)
        raise RuntimeError(f"{method} incorrectly accepted invalid state")

    stale = rejects("fusion_prepare", 0.5, [9, 9])
    errors, failed_streams = transfer("fault", [2, 2], fault=True)
    if not errors[-1] or any(errors[:-1]):
        raise RuntimeError(f"Expected exactly one malformed rank: {errors}")
    refused = rejects("fusion_prepare", 0.5, [2, 2])
    invalid = llm.collective_rpc("fusion_audit")
    if any(row["version"] is not None for row in invalid):
        raise RuntimeError("A failed receive left a published version")
    report = {
        "passed": True,
        "scope": "Native IPC receiver and TP shard loader; trainer coordination tested separately",
        "tp": args.tp,
        "before": before,
        "after": after,
        "success_versions": audit,
        "success_streams": success_streams,
        "failed_streams": failed_streams,
        "receive_errors": errors,
        "stale_version_rejection": stale,
        "failed_receive_rejection": refused,
        "invalid_versions": invalid,
    }
    (out / "complete.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
