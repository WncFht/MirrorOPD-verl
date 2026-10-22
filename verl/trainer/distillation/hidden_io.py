# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU/TQ transfers at micro-batch boundaries; no KL or CUDA kernels."""

from concurrent.futures import ThreadPoolExecutor

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu


class HiddenBatchIO:
    def __init__(self, data, batches, *, get=None, put=None):
        self.batches = batches
        self.active = "full_vocab_keys" in data
        self.teacher = self.active and tu.get_non_tensor_data(data, "full_vocab_teacher", default=False)
        self.student = self.active and tu.get_non_tensor_data(data, "full_vocab_train", default=False)
        self.partition = tu.get_non_tensor_data(data, "full_vocab_partition", default=None)
        self.async_write = self.teacher and tu.get_non_tensor_data(data, "full_vocab_async_write", default=False)
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
                self.pending_write.result()
        finally:
            if self.pool is not None:
                self.pool.shutdown(wait=True, cancel_futures=True)

    def _read(self, batch):
        with torch.profiler.record_function("full_vocab/hidden_read"):
            payload = self.get(
                keys=list(batch["full_vocab_keys"]),
                partition_id=self.partition,
                select_fields=["teacher_hidden", "teacher_positions"],
            )
            if len(payload) != len(batch):
                raise ValueError("Incomplete teacher hidden payload")
            return payload

    def __iter__(self):
        if not self.student:
            yield from self.batches
            return
        future = self.pool.submit(self._read, self.batches[0]) if self.prefetch and self.batches else None
        for i, batch in enumerate(self.batches):
            payload = future.result() if future is not None else self._read(batch)
            if self.prefetch and i + 1 < len(self.batches):
                future = self.pool.submit(self._read, self.batches[i + 1])
            else:
                future = None
            batch.update(payload)
            yield batch
            # Do not keep completed payloads in the micro-batch list for the whole update.
            batch.pop("teacher_hidden", None)
            batch.pop("teacher_positions", None)

    def publish(self, output, batch):
        if not self.teacher:
            return
        model_output = output["model_output"]
        payload = TensorDict(model_output, batch_size=[len(batch)])
        keys = list(batch["full_vocab_keys"])

        def write():
            with torch.profiler.record_function("full_vocab/hidden_write"):
                self.put(keys=keys, partition_id=self.partition, fields=payload)

        if self.async_write:
            # At most one pending write owns CPU tensors; backpressure before submitting the next.
            if self.pending_write is not None:
                self.pending_write.result()
            self.pending_write = self.pool.submit(write)
        else:
            write()
        output["model_output"] = {
            "teacher_payload_ready": torch.nested.nested_tensor(
                [torch.ones(1, dtype=torch.bool) for _ in keys], layout=torch.jagged
            )
        }
