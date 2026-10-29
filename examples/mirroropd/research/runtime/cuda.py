# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Configure a worker only within its explicit runtime device allocation."""


def setup():
    import os

    import torch

    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != 1 or devices[0] not in os.environ.get("MIRROROPD_ALLOWED_GPUS", "4,5,6,7").split(","):
        raise ValueError("GPU workers require exactly one explicitly assigned physical GPU from the runtime allocation")

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
