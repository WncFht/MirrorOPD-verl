# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Worker-local fusion control and diagnostics; no per-token RPC."""

import hashlib
import os
from pathlib import Path

import torch

from verl.utils.device import get_torch_device

from .sampling import FusionTopKTopPSampler


class FusionWorkerExtension:
    def _fusion_model(self):
        getter = getattr(self.model_runner, "get_model", None)
        return getter() if getter else self.model_runner.model

    def fusion_capture(self, enabled):
        """Enable full-logit capture for isolated numerical probes only."""
        model = self._fusion_model()
        model.capture_components = enabled
        model.component_logits = None

    def fusion_save_logits(self, path):
        """Save the last sampling position outside the timed generation path."""
        if self.local_rank != 0:
            return
        values = self._fusion_model().component_logits
        if values is None:
            raise RuntimeError("No fusion logits were captured")
        torch.save({name: value.cpu() for name, value in values.items()}, path)

    def monkey_patch_model(self, vocab_size, banned_token_ids=None):
        # The composite model already masks the complete tokenizer support.
        if banned_token_ids:
            self._fusion_model().legal_tokens[banned_token_ids] = False

    def fusion_initialize(self, current):
        from safetensors import safe_open

        model = self._fusion_model()
        self.fusion_current = current
        self.fusion_version = None if current else ("checkpoint", model.student_path)
        self.fusion_valid = not current
        self.fusion_pending_version = None
        self.fusion_shapes = {}
        for path in sorted(Path(model.student_path).glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    self.fusion_shapes[key] = tuple(handle.get_slice(key).get_shape())
        return self.fusion_audit()

    def fusion_weight_digests(self):
        """Full parameter hashes for acceptance only, outside throughput timing."""
        result = {}
        for component in ("student", "teacher"):
            digest = hashlib.sha256()
            for name, parameter in getattr(self._fusion_model(), component).named_parameters():
                digest.update(name.encode())
                value = parameter.detach().cpu().contiguous().view(torch.uint8).numpy()
                digest.update(memoryview(value).cast("B"))
            result[component] = digest.hexdigest()
        return result

    def fusion_receive_weights(self, version, use_shm=False):
        """Consume every bucket, recording errors without stranding the sender."""
        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        if not self.fusion_current:
            raise ValueError("Frozen fusion student cannot receive actor weights")
        self.fusion_valid = False
        self.fusion_version = self.fusion_pending_version = None
        model = self._fusion_model().student
        expected = self.fusion_shapes
        seen, canonical_seen, loaded, errors = set(), set(), set(), []
        tied = model.config.tie_word_embeddings

        def receive(bucket, is_last):
            for name, tensor in bucket:
                canonical = "model.embed_tokens.weight" if tied and name == "lm_head.weight" else name
                shape = expected.get(canonical)
                if shape is None or tuple(tensor.shape) != shape or not tensor.is_floating_point():
                    errors.append(f"Invalid fusion weight {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
                    continue
                if name in seen:
                    errors.append(f"Duplicate fusion weight: {name}")
                    continue
                if tied and canonical == "model.embed_tokens.weight" and canonical in canonical_seen:
                    before = model.model.embed_tokens.weight.detach().clone()
                    try:
                        # Let the native loader select this rank's vocab shard.
                        loaded.update(model.load_weights([(canonical, tensor)]))
                        if not torch.equal(before, model.model.embed_tokens.weight):
                            errors.append("Conflicting tied fusion embedding/head")
                    except Exception as exc:
                        errors.append(f"Tied fusion alias loader failed: {exc}")
                    seen.add(name)
                    continue
                seen.add(name)
                canonical_seen.add(canonical)
                try:
                    loaded.update(model.load_weights([(canonical, tensor)]))
                except Exception as exc:
                    errors.append(f"Fusion loader failed for {name}: {exc}")

        replica = os.environ.get("VERL_REPLICA_RANK", "0")
        job = os.environ.get("VERL_RAY_JOB_ID", "0")
        address = f"ipc:///tmp/rl-colocate-zmq-{job}-replica-{replica}-rank-{self.local_rank}.sock"
        receiver = BucketedWeightReceiver(zmq_handle=address, device=self.device, use_shm=use_shm)
        receiver.receive_weights(receive)
        missing = set(expected) - canonical_seen
        if tied and "lm_head.weight" in seen:
            missing.discard("model.embed_tokens.weight")
        if missing:
            errors.append(f"Missing fusion weights: {sorted(missing)}")
        missing_parameters = {name for name, _ in model.named_parameters()} - loaded
        if missing_parameters:
            errors.append(f"Native loader omitted parameters: {sorted(missing_parameters)}")
        get_torch_device().synchronize()
        if not errors:
            self.fusion_pending_version = tuple(version)
        return "; ".join(errors) if errors else None

    def fusion_commit_version(self, version):
        if self.fusion_pending_version != tuple(version):
            raise RuntimeError("Cannot commit an incomplete fusion refit")
        self.fusion_version = tuple(version)
        self.fusion_valid = True

    def fusion_prepare(self, alpha, version=None):
        """Prepare sampler and coefficient only while the engine is drained."""
        if hasattr(self, "fusion_valid"):
            if not self.fusion_valid or (self.fusion_current and self.fusion_version != tuple(version)):
                raise RuntimeError("Fusion student version is invalid or stale")
        self._fusion_model().set_alpha(alpha)
        sampler = getattr(self.model_runner, "sampler", None)
        if sampler is not None:
            sampler.topk_topp_sampler = FusionTopKTopPSampler(
                getattr(sampler.topk_topp_sampler, "use_fp64_gumbel", False)
            )
        if not hasattr(self, "fusion_version"):
            self.fusion_version = version
        return self.fusion_audit()

    def fusion_audit(self):
        model = self._fusion_model()
        layers = self.model_runner.vllm_config.compilation_config.static_forward_context
        return {
            "tp_rank": self.local_rank,
            "tensor_parallel_size": self.model_runner.vllm_config.parallel_config.tensor_parallel_size,
            "alpha": model.alpha,
            "version": getattr(self, "fusion_version", None),
            "student_path": model.student_path,
            "teacher_path": model.teacher_path,
            "attention_layers": sorted(layers),
            "student_parameters": sum(p.numel() for p in model.student.parameters()),
            "teacher_parameters": sum(p.numel() for p in model.teacher.parameters()),
            "peak_allocated_bytes": get_torch_device().max_memory_allocated(),
            "peak_reserved_bytes": get_torch_device().max_memory_reserved(),
        }
