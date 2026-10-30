# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Construct a generator and its optional mutable receiver as separate objects."""


class HFWeightTarget:
    def __init__(self, generator):
        self.generator = generator

    async def initialize(self):
        self.generator._ensure_student()

    async def receive(self, weights, version):
        return await self.generator.update_weights(weights, version=version)

    async def commit(self, version):
        self.generator.commit_version(version)


def create_fusion_worker(config, model, settings):
    """Fixed generators deliberately have no actor weight target."""
    if settings.get("backend") == "vllm_fusion":
        from verl.workers.rollout.vllm_fusion.weights import NativeWeightTarget

        return None, NativeWeightTarget(config, settings=settings) if settings["current"] else None
    from verl.workers.config import HFModelConfig
    from verl.workers.rollout.hf_fusion.adapter import ServerAdapter

    if settings.get("student_path"):
        model = HFModelConfig(path=settings["student_path"], trust_remote_code=False)
    generator = ServerAdapter(config, model, settings=settings)
    return generator, HFWeightTarget(generator) if settings["current"] else None
