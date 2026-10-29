# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Actual FSDP2 teacher/actor construction must retain identical FP32 RoPE."""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from tensordict import TensorDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

from examples.mirroropd.research.backends import fsdp as replay
from verl.utils import tensordict_utils as tu
from verl.utils.distributed import initialize_global_process_group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_per_process_memory_fraction(0.05)
    _, rank, world = initialize_global_process_group(timeout_second=180)
    paths = [tempfile.mkdtemp(prefix="fsdp-buffer-") if rank == 0 else None]
    dist.broadcast_object_list(paths)
    root = Path(paths[0])
    cfg = Qwen2Config(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        pad_token_id=0,
        eos_token_id=1,
    )
    expected = Qwen2RotaryEmbedding(cfg).inv_freq.clone()
    if rank == 0:
        torch.manual_seed(12)
        Qwen2ForCausalLM(cfg).bfloat16().save_pretrained(root)
        vocabulary = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, **{f"t{i}": i for i in range(3, 32)}}
        PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocabulary, unk_token="[UNK]")),
            pad_token="[PAD]",
            eos_token="[EOS]",
            unk_token="[UNK]",
        ).save_pretrained(root)
    dist.barrier()
    actor, teacher = replay.make_worker(root, world), replay.make_worker(root, world, True)
    ids = (torch.arange(64) % 28 + 3).long()
    data = TensorDict(
        {
            "input_ids": replay.nested([ids]),
            "position_ids": replay.nested([torch.arange(64)]),
            "prompts": replay.nested([ids[:16]]),
            "responses": replay.nested([ids[16:]]),
            "response_mask": replay.nested([torch.ones(48, dtype=torch.long)]),
            "loss_mask": replay.nested([torch.ones(48, dtype=torch.long)]),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(data, temperature=1.0, compute_loss=False, full_vocab_teacher=True, global_token_num=None)
    for _ in range(2):
        for worker in (actor, teacher):
            rope = worker.engine.module.model.rotary_emb
            for name in ("inv_freq", "original_inv_freq"):
                value = getattr(rope, name)
                assert value.dtype == torch.float32
                assert torch.equal(value.cpu(), expected)
        a = actor.infer_batch(data)["teacher_hidden"].values()
        t = teacher.infer_batch(data)["teacher_hidden"].values()
        assert torch.equal(a, t)
    if rank == 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "world_size": world,
                    "buffer_dtype": "float32",
                    "teacher_actor_hidden_exact": True,
                    "repeated_offload_roundtrips": 2,
                },
                indent=2,
            )
            + "\n"
        )
    dist.barrier()
    if rank == 0:
        shutil.rmtree(root)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
