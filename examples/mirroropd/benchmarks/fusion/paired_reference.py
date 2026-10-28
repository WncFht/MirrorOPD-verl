# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Layout experiment only: two independent GPUs, GPU peer logits transfer, one sampler.

This is not a production rollout backend and has no actor-weight transport. A
retained paired layout must separately pass the trainer/refit lifecycle gates.
"""

from concurrent.futures import ThreadPoolExecutor

import torch

from verl.experimental.rollout_policy.hf_decoder import HFFusionDecoder


class PairedReference(HFFusionDecoder):
    allow_separate_devices = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool = ThreadPoolExecutor(max_workers=2)

    @torch.inference_mode()
    def _thread_forward(self, *args):
        model = args[1]
        with torch.cuda.device(next(model.parameters()).device):
            return self._forward_one(*args)

    def _forward_models(self, models, ids, mask, positions, caches, extras, step):
        futures = [
            self.pool.submit(
                self._thread_forward, name, model, ids, mask, positions, caches.get(name), extras[name], step
            )
            for name, model in models
        ]
        return [future.result() for future in futures]

    def close(self):
        self.pool.shutdown(wait=True)
