# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Named in-process inference target, separate from the learner and default vLLM."""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.experimental.rollout_policy.hf_decoder import DecodeRequest, HFFusionDecoder
from verl.experimental.rollout_policy.sampling import SamplingConfig, validate_tokenizers
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.replica import TokenOutput


class ServerAdapter(BaseRollout):
    def __init__(self, config, model_config, device_mesh=None, *, settings=None):
        super().__init__(config, model_config, device_mesh)
        self.settings = settings or dict(config.engine_kwargs.get("hf_fusion", {}))
        self.student = None
        self.teacher = None
        self.student_cpu_parameters = None
        self.teacher_cpu_parameters = None
        self.loaded_version = None
        self.valid = False
        self.generating = False
        self.alpha = None
        self.phase_token = None
        self.legal_ids = ()
        self.metrics = {}
        self.batch_metrics = []
        self.residency_metrics = {}
        self.device = torch.device("cuda", torch.cuda.current_device())

    def _load(self, path):
        tick = time.monotonic()
        model = (
            AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", local_files_only=True
            )
            .eval()
            .requires_grad_(False)
        )
        from verl.experimental.rollout_policy.precision import configure_precision

        configure_precision(model, "stable")
        self.metrics["load_seconds"] = self.metrics.get("load_seconds", 0.0) + time.monotonic() - tick
        return model

    def _ensure_student(self):
        if self.student is None:
            self.student = self._load(self.model_config.path)
            self.student_cpu_parameters = self._cpu_parameters(self.student)
            self.valid = not self.settings["current"]
            if self.valid:
                self.loaded_version = ("checkpoint", self.model_config.path)

    @staticmethod
    def _cpu_parameters(model):
        # detach keeps the original CPU storage alive when Module.to swaps in GPU
        # storage. These immutable inference parameters never need a D2H backup.
        return {name: parameter.detach() for name, parameter in model.named_parameters(remove_duplicate=False)}

    @staticmethod
    def _park_model(model, backup):
        if backup is not None:
            for name, parameter in model.named_parameters(remove_duplicate=False):
                parameter.data = backup[name]
        # Small buffers may have changed during inference; preserve their state.
        model.to("cpu")

    async def prepare(self, alpha, version, phase_token=None):
        if self.generating:
            raise RuntimeError("Cannot switch fusion phase while generating")
        self.alpha = alpha
        self.batch_metrics = []
        self.residency_metrics = {}
        events = []

        def move(model, name):
            tick = time.monotonic()
            begin = end = None
            if self.device.type == "cuda":
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record(torch.cuda.current_stream(self.device))
            model.to(self.device, non_blocking=True)
            if end is not None:
                end.record(torch.cuda.current_stream(self.device))
                events.append((name, begin, end))
            self.residency_metrics[name + "_host_seconds"] = time.monotonic() - tick

        torch.cuda.reset_peak_memory_stats()
        if alpha < 1:
            self._ensure_student()
            if not self.valid or (self.settings["current"] and self.loaded_version != tuple(version)):
                raise RuntimeError("Fusion student has no valid weights for the requested actor version")
            move(self.student, "student_h2d")
        if alpha > 0:
            if self.teacher is None:
                self.teacher = self._load(self.settings["teacher_path"])
                self.teacher_cpu_parameters = self._cpu_parameters(self.teacher)
            move(self.teacher, "teacher_h2d")
        if not self.legal_ids:
            s_tokenizer = self.model_config.tokenizer
            t_tokenizer = AutoTokenizer.from_pretrained(self.settings["teacher_path"], local_files_only=True)
            # Do not skip mapping validation at alpha endpoints or for equal head widths.
            from transformers import AutoConfig

            t_config = AutoConfig.from_pretrained(self.settings["teacher_path"], local_files_only=True)
            self.legal_ids = validate_tokenizers(
                s_tokenizer, t_tokenizer, self.model_config.hf_config.vocab_size, t_config.vocab_size
            )
        torch.cuda.synchronize()
        for name, begin, end in events:
            self.residency_metrics[name + "_device_seconds"] = begin.elapsed_time(end) / 1000
        self.version = tuple(version)
        self.phase_token = phase_token

    async def resume(self, tags=None):
        if self.student is not None:
            self.student.to(self.device, non_blocking=True)

    async def release(self):
        if self.generating:
            raise RuntimeError("Cannot release fusion weights with active KV sessions")
        self.alpha = None
        for model, backup in ((self.student, self.student_cpu_parameters), (self.teacher, self.teacher_cpu_parameters)):
            if model is not None:
                self._park_model(model, backup)
        torch.cuda.empty_cache()

    async def update_weights(self, weights, wire_format="named_tensors", **kwargs):
        if wire_format != "named_tensors" or not self.settings["current"] or self.generating:
            raise ValueError("Invalid fusion refit target or state")
        self.valid = False
        self.loaded_version = None
        self._ensure_student()
        self.valid = False
        # Keep receiver on CPU during collectives: no full GPU actor clone is required.
        self._park_model(self.student, getattr(self, "student_cpu_parameters", None))
        state = self.student.state_dict()
        aliases = {}
        for name, param in self.student.named_parameters(remove_duplicate=False):
            aliases.setdefault(param.data_ptr(), set()).add(name)
        seen = set()
        error = None
        # Every rank must exhaust the stream, even after a local validation/copy error.
        for name, value in weights:
            try:
                if name not in state or name in seen:
                    raise ValueError(f"Unexpected or duplicate fusion weight: {name}")
                target = state[name]
                if target.shape != value.shape or value.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                    raise ValueError(f"Invalid fusion weight shape/dtype: {name}")
                incoming = value.detach().to(device="cpu", dtype=target.dtype)
                tied_names = aliases.get(target.data_ptr(), set())
                if seen & tied_names and not torch.equal(target, incoming):
                    raise ValueError(f"Conflicting tied fusion weights: {name}")
                target.copy_(incoming)
                seen.add(name)
            except Exception as exc:
                error = error or str(exc)
        for names in aliases.values():
            if seen & names:
                seen.update(names)
        missing = set(dict(self.student.named_parameters(remove_duplicate=False))) - seen
        if missing:
            error = error or f"Missing fusion weights: {sorted(missing)[:8]}"
        self.pending_version = kwargs.get("version") if error is None else None
        return error

    def commit_version(self, version):
        if self.pending_version is None or tuple(version) != tuple(self.pending_version):
            raise RuntimeError("Fusion refit version mismatch")
        self.loaded_version = tuple(version)
        self.valid = True

    def generate_batch(self, rows):
        if self.generating or self.alpha is None:
            raise RuntimeError("Fusion decoder is busy or unprepared")
        self.generating = True
        try:
            if self.alpha < 1 and (
                not self.valid or (self.settings["current"] and self.loaded_version != self.version)
            ):
                raise RuntimeError("Fusion student is stale or partially refitted")
            if self.phase_token is not None and any(
                row["sampling_params"].get("_fusion_phase_token") != self.phase_token for row in rows
            ):
                raise RuntimeError("Fusion request belongs to a different collection phase")
            first = rows[0]["sampling_params"]
            params = {key: value for key, value in first.items() if key != "seed"}
            if any(
                {key: value for key, value in row["sampling_params"].items() if key != "seed"} != params for row in rows
            ):
                raise ValueError("A fusion microbatch requires common sampling parameters")
            if params.get("repetition_penalty", 1.0) != 1.0 or params.get("topk_log_probs"):
                raise ValueError("Unsupported fusion sampling transform")
            config = SamplingConfig(
                temperature=params["temperature"],
                top_p=params["top_p"],
                top_k=params["top_k"],
                max_tokens=params.get("max_tokens", self.config.response_length),
                stop_ids=tuple(self.settings["stop_ids"]),
                legal_ids=self.legal_ids,
            )
            decoder = HFFusionDecoder(
                self.student,
                self.teacher,
                pad_id=self.model_config.tokenizer.pad_token_id,
                compact=self.settings.get("compact", True),
                compact_power_of_two=self.settings.get("compact_power_of_two", True),
                preallocate_kv=not self.settings.get("reference", False),
                flash_kv=True,
                flash_kv_splits=1,
                cuda_graph=not self.settings.get("reference", False),
            )
            requests = [
                DecodeRequest(row["request_id"], tuple(row["prompt_ids"]), int(row["sampling_params"]["seed"]))
                for row in rows
            ]
            results = decoder.generate(requests, self.alpha, config, deadline_seconds=self.settings["deadline_seconds"])
            self.metrics.update(decoder.metrics)
            self.batch_metrics.append(dict(decoder.metrics))
            return [
                TokenOutput(
                    token_ids=result.token_ids,
                    log_probs=result.log_probs if params.get("logprobs") else None,
                    stop_reason=result.finish_reason,
                    extra_fields={
                        "fusion_request_id": result.request_id,
                        "fusion_backend": "hf_reference_stable"
                        if self.settings.get("reference", False)
                        else "hf_dual_kv_graph_stable",
                        "finish_reason": result.finish_reason,
                        "global_steps": self.version[0] if self.settings["current"] and self.alpha < 1 else None,
                        "actor_version": self.loaded_version if self.settings["current"] and self.alpha < 1 else None,
                        "student_version": self.loaded_version if self.alpha < 1 else None,
                        "teacher_path": self.settings["teacher_path"],
                        "teacher_weight": self.alpha,
                    },
                )
                for result in results
            ]
        finally:
            self.generating = False
