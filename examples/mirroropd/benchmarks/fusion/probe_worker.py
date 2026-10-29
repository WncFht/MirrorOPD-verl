# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only forced-prefix replay; never registered as a rollout backend."""

import torch


class PrefixProbe:
    """Capture model distributions before forcing a known continuation token."""

    def configure_probe(self, suffix, checkpoints, components=False, requests=1):
        model = self.model_runner.get_model()
        if not hasattr(self, "probe_original_logits"):
            self.probe_original_logits = model.compute_logits
        self.probe_suffix = suffix
        self.probe_checkpoints = set(checkpoints)
        self.probe_steps = {}
        self.probe_requests = requests
        self.probe_batch_shapes = []
        self.probe_schedule = []
        self.probe_saved = {}
        self.probe_output_rank = None
        if components:
            model.capture_components = True

        def compute(hidden):
            logits = self.probe_original_logits(hidden)
            self.probe_output_rank = logits is not None
            if logits is None:
                return None
            ids = self.model_runner.input_batch.req_ids
            if len(ids) != logits.shape[0]:
                raise RuntimeError("Probe cannot align native request IDs and sampled logits")
            self.probe_batch_shapes.append(len(ids))
            self.probe_schedule.append(
                {"request_ids": list(ids), "steps": [self.probe_steps.get(key, 0) for key in ids]}
            )
            forced = torch.full_like(logits, -torch.inf)
            for row, request_id in enumerate(ids):
                step = self.probe_steps.get(request_id, 0)
                if step in self.probe_checkpoints:
                    values = model.component_logits if components else {"raw": logits}
                    self.probe_saved.setdefault(step, {})[request_id] = {
                        name: (tensor if name == "legal" else tensor[row]).detach().cpu()
                        for name, tensor in values.items()
                    }
                if step >= len(self.probe_suffix):
                    raise RuntimeError("Forced replay exceeded the declared suffix")
                forced[row, self.probe_suffix[step]] = 0
                self.probe_steps[request_id] = step + 1
            return forced

        model.compute_logits = compute

    def save_probe(self, path):
        if self.probe_output_rank is False:
            return {"output_rank": False}
        if len(self.probe_steps) != self.probe_requests or any(
            steps != len(self.probe_suffix) for steps in self.probe_steps.values()
        ):
            raise RuntimeError("Forced replay did not finish the declared suffix")
        saved = {}
        for step, requests in self.probe_saved.items():
            if len(requests) != self.probe_requests:
                raise RuntimeError("A replay checkpoint is missing a request")
            ordered = [requests[key] for key in sorted(requests)]
            saved[step] = {
                name: ordered[0][name] if name == "legal" else torch.stack([row[name] for row in ordered])
                for name in ordered[0]
            }
        torch.save(saved, path)
        return {
            "output_rank": True,
            "batch_shapes": self.probe_batch_shapes,
            "schedule": self.probe_schedule,
            "requests": self.probe_requests,
        }
