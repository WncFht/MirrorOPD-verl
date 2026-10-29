# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Evaluate arbitrary pinned teachers or saved student endpoints under frozen seeds."""

import argparse
import asyncio
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.studies.boundary.evaluate import evaluate
from examples.mirroropd.research.studies.boundary.round2_common import OLD_OUT, OUT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    spec = read(OUT / "eval-specs.json")[args.name]
    dest = OUT / "evaluations" / args.name
    if (dest / "complete.json").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    prompts = read(
        OUT / "math-native-prompts.json" if spec["template"] == "math_native" else OLD_OUT / "evaluation-prompts.json"
    )
    eos = 151643
    command = [
        str(Path(sys.executable).with_name("vllm")),
        "serve",
        spec["path"],
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        "boundary",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(spec["model_length"]),
        "--max-num-seqs",
        "64",
        "--max-num-batched-tokens",
        "8192",
        "--gpu-memory-utilization",
        "0.88",
        "--enable-prefix-caching",
        "--seed",
        "42",
        "--generation-config",
        "vllm",
    ]
    previous_protocol = dest / "protocol.json"
    if previous_protocol.exists():
        history = dest / "protocol-history"
        history.mkdir(exist_ok=True)
        shutil.copy2(previous_protocol, history / f"{time.time_ns()}.json")
    dump(
        dest / "protocol.json",
        dict(
            spec,
            command=command,
            samples=4,
            temperature=0.7,
            top_p=0.9,
            request_order="seed42 shuffle; each response keeps its own original seed",
            execution_dtype="BF16 deployment readout, matching original baselines",
            connection_policy="force-close; up to3 transport attempts with identical prompt and seed",
            visible_gpus=os.environ.get("CUDA_VISIBLE_DEVICES"),
        ),
    )
    with (dest / "server.log").open("a") as stream:
        server = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        dump(dest / "server-pid.json", dict(pid=server.pid))
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"vLLM exited: {server.returncode}")
                try:
                    with opener.open(f"http://127.0.0.1:{args.port}/health", timeout=3) as response:
                        if response.status == 200:
                            break
                except Exception:
                    time.sleep(2)
            else:
                raise TimeoutError("vLLM startup")
            asyncio.run(
                evaluate(
                    args.name,
                    args.port,
                    spec["template"],
                    dest,
                    prompts=prompts,
                    eos=eos,
                    cap=spec["response_cap"],
                    shuffle=True,
                )
            )
        finally:
            try:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
            except ProcessLookupError:
                pass


if __name__ == "__main__":
    main()
