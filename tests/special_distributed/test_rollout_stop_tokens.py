# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Single-GPU actual server.generate -> AsyncLLM -> CT mask smoke; no Ray cluster.

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python ... --model /path/to/Qwen3-1.7B-Base
"""

import argparse
import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


async def run(args):
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder
    from verl.workers.config import HFModelConfig, RolloutConfig
    from verl.workers.rollout.vllm_rollout import vllm_async_server

    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.model_config = HFModelConfig(path=str(args.model))
    original = RolloutConfig(
        name="vllm", max_model_len=1024, prompt_length=256, response_length=4, tensor_model_parallel_size=1
    )
    server.config = original
    server._validate_configs()
    server.replica_rank = 0
    server.node_rank = 0
    server.global_steps = 0
    server._disaggregation_role = "null"
    server._submission_paused = False
    server._admitting = 0
    server._resume_event = asyncio.Event()
    server._resume_event.set()
    server._rejecting = False
    server.engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=str(args.model),
            dtype="bfloat16",
            tensor_parallel_size=1,
            enforce_eager=True,
            gpu_memory_utilization=0.25,
            max_model_len=1024,
            max_num_seqs=4,
            max_num_batched_tokens=1024,
            enable_prefix_caching=False,
            attention_config={"backend": "TRITON_ATTN"},
        )
    )
    tokenizer = server.model_config.tokenizer
    builder = create_continuous_token_builder(
        tokenizer, model_family="qwen3", chat_template_kwargs={"enable_thinking": False}
    )
    prompt = builder.build_initial_tokens([{"role": "user", "content": "What is 1+1?"}])
    rows = []
    try:
        # Stub only the tracing actor name; all sampling goes through the real server and engine.
        with patch.object(
            vllm_async_server.ray, "get_runtime_context", lambda: SimpleNamespace(get_actor_name=lambda: "stop-smoke")
        ):
            for validate in (False, True):
                for configured in (None, [151645]):
                    server.config = replace(original, stop_token_ids=configured)
                    server._validate_configs()
                    for target, requested in [(151643, []), (151645, []), (17, []), (17, [17])]:
                        max_tokens = 5 if validate else 4
                        params = {
                            "temperature": 0.7 if validate else 1.0,
                            "top_p": 1.0,
                            "top_k": -1,
                            "max_tokens": max_tokens,
                            "logprobs": not validate,
                            "allowed_token_ids": [target],
                            "stop_token_ids": requested,
                        }
                        output = await server.generate(prompt, params, f"stop-{len(rows)}")
                        should_stop = target == 151643 or target in (configured or []) or target in requested
                        assert len(output.token_ids) == (1 if should_stop else max_tokens)
                        assert output.extra_fields["finish_reason"] == ("stop" if should_stop else "length")
                        assert (output.log_probs is None) == validate
                        merged = builder.merge_assistant_tokens(prompt, output.token_ids)
                        mask, lp = builder.align_response_metadata(
                            merged,
                            [],
                            [] if output.log_probs is not None else None,
                            assistant_logprobs=output.log_probs,
                        )
                        assert merged.token_ids == prompt + output.token_ids
                        assert mask == [1] * len(output.token_ids)
                        assert lp == output.log_probs
                        rows.append(
                            {
                                "validate": validate,
                                "configured": configured,
                                "target": target,
                                "requested": requested,
                                "token_ids": output.token_ids,
                                "mask": mask,
                                "finish_reason": output.extra_fields["finish_reason"],
                                "has_logprobs": output.log_probs is not None,
                            }
                        )
    finally:
        server.engine.shutdown()
    result = {"model": str(args.model), "cases": rows, "passed": len(rows)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("NVTX_INJECTION64_PATH", "/nonexistent/verl-disabled-nvtx-injection.so")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
