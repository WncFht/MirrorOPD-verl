# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only capture of the distribution actually passed to native sampling."""

import torch

from verl.workers.rollout.vllm_fusion.sampling import FusionTopKTopPSampler
from verl.workers.rollout.vllm_fusion.worker_extension import FusionWorkerExtension


class CaptureSampler(FusionTopKTopPSampler):
    def forward(self, logits, generators, k, p):
        tokens, log_probs = super().forward(logits, generators, k, p)
        self.captured_log_probs = log_probs.detach().clone()
        return tokens, log_probs


class MathProbeWorker(FusionWorkerExtension):
    def fusion_prepare(self, alpha, version=None):
        audit = super().fusion_prepare(alpha, version)
        if self.model_runner.sampler is not None:
            self.model_runner.sampler.topk_topp_sampler = CaptureSampler()
        return audit

    def fusion_save_logits(self, path):
        if self.local_rank != 0:
            return
        values = dict(self._fusion_model().component_logits)
        values["processed_log_probs"] = self.model_runner.sampler.topk_topp_sampler.captured_log_probs
        torch.save({name: value.cpu() for name, value in values.items()}, path)
