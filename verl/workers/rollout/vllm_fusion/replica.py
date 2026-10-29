# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Native vLLM server with phase admission and persistent frozen weights."""

import math

import ray

from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica

from .metadata import output_metadata


class FusionServer(vLLMHttpServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fusion_accepting = False
        self.fusion_phase = None
        self.fusion_seen = set()
        self.fusion_expected = 0
        self.fusion_loaded_version = None

    def _get_worker_extension_cls(self):
        return "verl.workers.rollout.vllm_fusion.worker_extension.FusionWorkerExtension"

    def _resolve_sleep_level(self):
        return 1

    async def prepare(self, expected_requests, phase_token, alpha, version):
        self.fusion_accepting = False
        await self.wait_for_requests_to_drain()
        await self.clear_kv_cache()
        audit = await self.engine.collective_rpc("fusion_prepare", args=(alpha, version))
        self.fusion_loaded_version = audit[0]["version"]
        self.fusion_alpha = alpha
        self.fusion_teacher_path = audit[0]["teacher_path"]
        self.fusion_phase = phase_token
        self.fusion_seen = set()
        self.fusion_expected = expected_requests
        await self.resume_generation()
        self.fusion_accepting = True

    async def generate(self, request_id, prompt_ids, sampling_params, **kwargs):
        params = dict(sampling_params)
        if not self.fusion_accepting or params.pop("_fusion_phase_token", None) != self.fusion_phase:
            raise RuntimeError("Fusion request rejected: inactive or stale phase")
        if request_id in self.fusion_seen or len(self.fusion_seen) >= self.fusion_expected:
            raise RuntimeError("Duplicate or excess fusion request")
        cap = params.get("max_tokens", self.config.response_length)
        if cap < 1 or len(prompt_ids) + cap > self.config.max_model_len:
            raise ValueError("Fusion prompt plus declared cap exceeds context; cap will not be silently shortened")
        self.fusion_seen.add(request_id)
        # The fusion API returns the probability of the actual sampled token,
        # even when a full-vocab learner does not consume rollout log-probs.
        params["logprobs"] = True
        output = await super().generate(request_id=request_id, prompt_ids=prompt_ids, sampling_params=params, **kwargs)
        if output.stop_reason == "aborted":
            raise RuntimeError("Fusion request was aborted")
        if (
            output.log_probs is None
            or len(output.log_probs) != len(output.token_ids)
            or any(not math.isfinite(value) for value in output.log_probs)
        ):
            raise RuntimeError("Fusion response is missing finite selected-token behavior probabilities")
        if sampling_params.get("temperature") == 0:
            output.log_probs = [0.0] * len(output.token_ids)
        output.extra_fields.update(
            output_metadata(
                request_id, self.fusion_phase, self.fusion_loaded_version, self.fusion_alpha, self.fusion_teacher_path
            )
        )
        return output

    async def abort_pending(self, reason):
        self.fusion_accepting = False
        await self.abort_all_requests(reject_request=True)
        await self.wait_for_requests_to_drain()

    async def sleep(self):
        self.fusion_accepting = False
        await self.wait_for_requests_to_drain()
        await super().sleep()


class VLLMFusionReplica(vLLMReplica):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(FusionServer)
