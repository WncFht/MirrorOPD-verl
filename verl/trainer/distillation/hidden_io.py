# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU/TQ transfers at micro-batch boundaries; no KL or CUDA kernels."""

from concurrent.futures import ThreadPoolExecutor

import torch
from tensordict import TensorDict

from verl.trainer.distillation.profiling import full_kl_range
from verl.utils import tensordict_utils as tu


class HiddenBatchIO:
    def __init__(self, data, batches, *, get=None, put=None):
        self.batches = batches
        # Old direct callers remain supported at this payload-specific boundary.
        self.keys_field = "sample_keys" if "sample_keys" in data else "full_vocab_keys"
        self.active = self.keys_field in data
        self.capture = tu.get_non_tensor_data(data, "full_vocab_capture", default=None)
        if tu.get_non_tensor_data(data, "full_vocab_teacher", default=False):
            self.capture = "teacher"
        if self.capture not in (None, "teacher", "anchor"):
            raise ValueError(f"Unknown hidden capture source: {self.capture}")
        self.writer = self.active and self.capture is not None
        sources = tu.get_non_tensor_data(data, "full_vocab_sources", default=("teacher",))
        if any(source not in ("teacher", "anchor") for source in sources):
            raise ValueError("Unknown frozen hidden source")
        self.fields = [field for source in sources for field in (f"{source}_hidden", f"{source}_positions")]
        if "anchor" in sources:
            self.fields.append("anchor_version")
        self.student = self.active and tu.get_non_tensor_data(data, "full_vocab_train", default=False)
        partition_field = "sample_partition" if self.keys_field == "sample_keys" else "full_vocab_partition"
        self.partition = tu.get_non_tensor_data(data, partition_field, default=None)
        self.async_write = self.writer and tu.get_non_tensor_data(data, "full_vocab_async_write", default=False)
        self.prefetch = self.student and tu.get_non_tensor_data(data, "full_vocab_prefetch", default=False)
        self.pool = None
        self.pending_write = None
        if self.active and (get is None or put is None):
            from verl.utils.transferqueue_utils import tq

            get, put = tq.kv_batch_get, tq.kv_batch_put
        self.get, self.put = get, put

    def __enter__(self):
        if self.async_write or self.prefetch:
            self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="full-kl-hidden")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.pending_write is not None and exc_type is None:
                with full_kl_range("hidden_write_flush"):
                    self.pending_write.result()
        finally:
            if self.pool is not None:
                self.pool.shutdown(wait=True, cancel_futures=True)

    def _read(self, batch):
        with full_kl_range("hidden_read"):
            payload = self.get(
                keys=list(batch[self.keys_field]),
                partition_id=self.partition,
                select_fields=self.fields,
            )
            if len(payload) != len(batch) or any(field not in payload for field in self.fields):
                raise ValueError("Incomplete teacher/anchor hidden payload")
            return payload

    def __iter__(self):
        if not self.student:
            yield from self.batches
            return
        future = self.pool.submit(self._read, self.batches[0]) if self.prefetch and self.batches else None
        for i, batch in enumerate(self.batches):
            with full_kl_range("hidden_read_wait"):
                payload = future.result() if future is not None else self._read(batch)
            if self.prefetch and i + 1 < len(self.batches):
                future = self.pool.submit(self._read, self.batches[i + 1])
            else:
                future = None
            batch.update(payload)
            try:
                yield batch
            finally:
                # Release every injected source even when forward/backward raises.
                for field in self.fields:
                    batch.pop(field, None)

    def publish(self, output, batch):
        if not self.writer:
            return
        model_output = output["model_output"]
        payload = TensorDict(model_output, batch_size=[len(batch)])
        keys = list(batch[self.keys_field])

        def write():
            with full_kl_range("hidden_write"):
                self.put(keys=keys, partition_id=self.partition, fields=payload)

        if self.async_write:
            # At most one pending write owns CPU tensors; backpressure before submitting the next.
            if self.pending_write is not None:
                with full_kl_range("hidden_write_wait"):
                    self.pending_write.result()
            self.pending_write = self.pool.submit(write)
        else:
            write()
        output["model_output"] = {
            f"{self.capture}_payload_ready": torch.nested.nested_tensor(
                [torch.ones(1, dtype=torch.bool) for _ in keys], layout=torch.jagged
            )
        }
