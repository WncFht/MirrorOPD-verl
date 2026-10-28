# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Dual-KV decoder with optional append-only storage and request-local decode graphs."""

import inspect
import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from .math_context import inference_math
from .sampling import SamplingConfig, fusion_log_probs, sample_tokens


@dataclass(frozen=True)
class DecodeRequest:
    request_id: str
    prompt_ids: tuple[int, ...]
    seed: int


@dataclass
class DecodeResult:
    request_id: str
    token_ids: list[int]
    log_probs: list[float]
    finish_reason: str


class HFFusionDecoder:
    allow_separate_devices = False

    def __init__(
        self,
        student,
        teacher,
        *,
        pad_id,
        compact=False,
        compact_power_of_two=False,
        stop_diagnostics=False,
        native_stop_ids=None,
        allow_cudnn=False,
        profile=False,
        preallocate_kv=False,
        flash_kv=False,
        flash_kv_splits=0,
        cuda_graph=False,
    ):
        self.student = student
        self.teacher = teacher
        self.pad_id = pad_id
        self.compact = compact
        self.compact_power_of_two = compact_power_of_two
        self.stop_diagnostics = stop_diagnostics
        self.native_stop_ids = native_stop_ids
        self.allow_cudnn = allow_cudnn
        self.profile = profile
        self.profile_step = None
        self.preallocate_kv = preallocate_kv
        self.flash_kv = flash_kv
        self.flash_kv_splits = flash_kv_splits
        self.cuda_graph = cuda_graph
        self.graphs = {}
        if cuda_graph and not flash_kv:
            raise ValueError("CUDA decode graphs require Flash KV attention")
        if flash_kv and not preallocate_kv:
            raise ValueError("Flash KV decode requires append-only KV storage")
        self.metrics = {}
        for model in (student, teacher):
            if model is not None:
                model.eval().requires_grad_(False)

    @torch.inference_mode()
    def generate(self, requests, alpha: float, config: SamplingConfig, *, deadline_seconds=3600):
        cuda = any(
            model is not None and next(model.parameters()).device.type == "cuda"
            for model in (self.student, self.teacher)
        )
        self.graphs.clear()
        try:
            with inference_math(cuda=cuda, allow_cudnn=self.allow_cudnn):
                return self._generate(requests, alpha, config, deadline_seconds=deadline_seconds)
        finally:
            self.graphs.clear()

    def _forward_one(self, name, model, ids, mask, positions, cache, extra, step):
        tick = time.monotonic()
        device = next(model.parameters()).device
        label = name + ("_prefill" if step == 0 else "_decode")
        event = self._profile_begin(device)
        if self.cuda_graph and step > 0:
            from .cuda_graph import DecodeGraph

            with torch.profiler.record_function("fusion/" + label) if self.profile else nullcontext():
                if name not in self.graphs:
                    graph = DecodeGraph(model, ids, positions, mask, cache, extra, splits=self.flash_kv_splits)
                    self.graphs[name] = graph
                    output = graph.output
                else:
                    graph = self.graphs[name]
                    previous_capture = graph.capture_seconds
                    output = graph.replay(ids.to(device), positions.to(device), mask)
                    self.metrics["graph_capture_seconds"] += graph.capture_seconds - previous_capture
            return self._finish_forward(name, output, tick, label, event, device)
        attention_context = nullcontext()
        forward_mask = None if self.flash_kv and step > 0 else mask.to(device, non_blocking=True)
        if self.flash_kv and step > 0:
            from .flash_kv import flash_kv_context

            attention_context = flash_kv_context(model, mask, splits=self.flash_kv_splits, cache=cache)
            forward_mask = None
        with (
            torch.profiler.record_function("fusion/" + label) if self.profile else nullcontext(),
            attention_context,
        ):
            output = model(
                input_ids=ids.to(device, non_blocking=True),
                attention_mask=forward_mask,
                position_ids=positions.to(device, non_blocking=True),
                past_key_values=cache,
                use_cache=True,
                **extra,
            )
        return self._finish_forward(name, output, tick, label, event, device)

    def _finish_forward(self, name, output, tick, label, event, device):
        self._profile_end(label, event, device)
        return name, output, time.monotonic() - tick

    def _profile_begin(self, device):
        if self.profile and device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(device))
            return event
        return None

    def _profile_end(self, label, event, device):
        if event is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream(device))
            self.profile_events.append((label, event, end, device))

    def _forward_models(self, models, ids, mask, positions, caches, extras, step):
        return [
            self._forward_one(name, model, ids, mask, positions, caches.get(name), extras[name], step)
            for name, model in models
        ]

    def _select_tokens(self, log_probs, generators, step):
        return sample_tokens(log_probs, generators)

    def _generate(self, requests, alpha: float, config: SamplingConfig, *, deadline_seconds=3600):
        if not requests or len({r.request_id for r in requests}) != len(requests):
            raise ValueError("Decoder needs nonempty unique request identities")
        if any(not r.prompt_ids for r in requests):
            raise ValueError("Empty prompts are not supported")
        if self.stop_diagnostics and (config.temperature != 1 or config.top_p != 1 or config.top_k > 0):
            raise ValueError("Stop-event diagnostics require T=1 without top-k/top-p filtering")
        models = (
            [("student", self.student)]
            if alpha == 0
            else ([("teacher", self.teacher)] if alpha == 1 else [("student", self.student), ("teacher", self.teacher)])
        )
        if any(model is None for _, model in models):
            raise ValueError("Missing active fusion component")
        device = next(models[0][1].parameters()).device
        if not self.allow_separate_devices and any(next(model.parameters()).device != device for _, model in models):
            raise ValueError("The reference decoder requires colocated TP1 models")
        lengths = [len(r.prompt_ids) for r in requests]
        width = max(lengths)
        ids = torch.full((len(requests), width), self.pad_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, request in enumerate(requests):
            ids[i, -lengths[i] :] = torch.tensor(request.prompt_ids, device=device)
            mask[i, -lengths[i] :] = 1
        positions = mask.cumsum(-1).sub(1).clamp_min(0)
        generators = [torch.Generator(device=device).manual_seed(r.seed) for r in requests]
        results = [DecodeResult(r.request_id, [], [], "length") for r in requests]
        active = list(range(len(requests)))
        finished = [False] * len(requests)
        caches = {}
        if self.preallocate_kv:
            from .kv_cache import AppendCache

            caches = {name: AppendCache(model.config, width + config.max_tokens) for name, model in models}
        stop_sums = {}
        stop_rows = 0
        self.profile_events = []
        self.metrics = {
            "student_prefill": 0.0,
            "teacher_prefill": 0.0,
            "student_decode": 0.0,
            "teacher_decode": 0.0,
            "sampling": 0.0,
            "communication": 0.0,
            "kv": 0.0,
            "kv_peak_bytes": 0,
            "graph_capture_seconds": 0.0,
        }
        forward_extras = {
            name: ({"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(model.forward).parameters else {})
            for name, model in models
        }
        legal_mask = None
        start = time.monotonic()
        for step in range(config.max_tokens):
            if time.monotonic() - start > deadline_seconds:
                raise TimeoutError("Fusion decode deadline exceeded; discard this batch")
            logits = {}
            for name, output, elapsed in self._forward_models(
                models, ids, mask, positions, caches, forward_extras, step
            ):
                caches[name] = output.past_key_values
                transfer_tick = time.monotonic()
                event = self._profile_begin(device)
                with torch.profiler.record_function("fusion/logits_transfer") if self.profile else nullcontext():
                    logits[name] = output.logits[:, -1].to(device, non_blocking=True)
                self._profile_end("logits_transfer", event, device)
                self.metrics["communication"] += time.monotonic() - transfer_tick
                self.metrics[name + ("_prefill" if step == 0 else "_decode")] += elapsed
            kv_bytes = sum(
                tensor.numel() * tensor.element_size()
                for cache in caches.values()
                for layer in cache.layers
                for tensor in (layer.keys, layer.values)
                if tensor is not None
            )
            self.metrics["kv_peak_bytes"] = max(self.metrics["kv_peak_bytes"], kv_bytes)
            # Some model heads pad vocabulary columns. Slice only after validating token IDs.
            common_size = min(value.shape[-1] for value in logits.values())
            logits = {name: value[:, :common_size] for name, value in logits.items()}
            tick = time.monotonic()
            if config.legal_ids and legal_mask is None:
                legal_mask = torch.zeros(common_size, dtype=torch.bool, device=device)
                legal_mask[list(config.legal_ids)] = True
            event = self._profile_begin(device)
            with torch.profiler.record_function("fusion/logits") if self.profile else nullcontext():
                lp = fusion_log_probs(
                    logits.get("student"), logits.get("teacher"), alpha, config, legal_mask=legal_mask
                )
            self._profile_end("logits_fusion", event, device)
            if self.stop_diagnostics and config.stop_ids:
                event = self._profile_begin(device)
                alive = torch.tensor([not done for done in finished], device=device)
                stop_rows += int(alive.sum())
                from .sampling import stop_event_probabilities

                if self.native_stop_ids is None:
                    raise ValueError("Stop diagnostics require the two native EOS IDs")
                probabilities = stop_event_probabilities(
                    logits, alpha, config.stop_ids, self.native_stop_ids, legal_mask=legal_mask
                )
                for key, values in probabilities.items():
                    stop_sums[key] = stop_sums.get(key, 0.0) + values[alive].sum().item()
                self._profile_end("stop_diagnostics", event, device)
            event = self._profile_begin(device)
            with torch.profiler.record_function("fusion/sample_and_output_copy") if self.profile else nullcontext():
                tokens, selected = self._select_tokens(lp, [generators[i] for i in active], step)
                token_list, lp_list = tokens.tolist(), selected.tolist()
            self._profile_end("sampling_and_output_copy", event, device)
            self.metrics["sampling"] += time.monotonic() - tick
            for row, original in enumerate(active):
                if not finished[row]:
                    results[original].token_ids.append(token_list[row])
                    results[original].log_probs.append(lp_list[row])
                    if token_list[row] in config.stop_ids:
                        results[original].finish_reason = "eos"
                        finished[row] = True
            if self.profile_step is not None:
                self.profile_step()
            if all(finished) or step + 1 == config.max_tokens:
                break
            tick = time.monotonic()
            event = self._profile_begin(device)
            ids = tokens[:, None]
            positions = mask.sum(-1, keepdim=True)
            mask = torch.cat((mask, torch.ones_like(ids)), dim=-1)
            if self.compact and any(finished):
                keep_list = [i for i, done in enumerate(finished) if not done]
                if self.compact_power_of_two:
                    target = min(len(active), 1 << (len(keep_list) - 1).bit_length())
                    keep_list += [i for i, done in enumerate(finished) if done][: target - len(keep_list)]
                    keep_list.sort()
                if len(keep_list) < len(active):
                    self.graphs.clear()
                    keep = torch.tensor(keep_list, device=device)
                    for cache in caches.values():
                        cache.batch_select_indices(keep.to(cache.layers[0].keys.device))
                    ids, positions, mask = ids[keep], positions[keep], mask[keep]
                    active = [active[i] for i in keep_list]
                    finished = [finished[i] for i in keep_list]
            self._profile_end("kv_rows_and_positions", event, device)
            self.metrics["kv"] += time.monotonic() - tick
        self.metrics["wall_seconds"] = time.monotonic() - start
        self.metrics["response_tokens"] = sum(len(result.token_ids) for result in results)
        # KV is request-local and never reused across refit or validation.
        self.graphs.clear()
        caches.clear()
        if stop_rows:
            self.metrics["stop_mass_rows"] = stop_rows
            self.metrics["stop_mass_means"] = {key: value / stop_rows for key, value in stop_sums.items()}
        if self.profile_events:
            for device in {row[3] for row in self.profile_events}:
                torch.cuda.synchronize(device)
            for label, begin, end, _ in self.profile_events:
                key = "gpu_elapsed_" + label
                self.metrics[key] = self.metrics.get(key, 0.0) + begin.elapsed_time(end) / 1000
            self.profile_events.clear()
        return results
