# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""torchrun --standalone --nproc-per-node=2 ... --backend recompute.

Actual FSDP2 head hooks, offload, packed positions, uneven tokens, empty local
mask, and repeated updates. Synthetic small Qwen2; no Ray or vLLM.
"""

import argparse
import json
import shutil
import tempfile
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from tensordict import TensorDict
from test_shared_teacher_fsdp2 import assert_on_cpu, make_worker, nested, teacher_policy_loss
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen2Config

from verl.trainer.distillation.full_vocab import full_reverse_kl
from verl.trainer.distillation.hidden import prediction_positions
from verl.utils import tensordict_utils as tu
from verl.utils.distributed import initialize_global_process_group
from verl.workers.config import DistillationConfig, DistillationLossConfig
from verl.workers.config.distillation import FullVocabConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["dense", "recompute", "cached", "streaming", "precompute", "liger"], default="recompute"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    _, rank, world = initialize_global_process_group(timeout_second=180)
    paths = [tempfile.mkdtemp(prefix="full-kl-fsdp-") if rank == 0 else None]
    dist.broadcast_object_list(paths)
    root = Path(paths[0])
    if rank == 0:
        vocab = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, **{f"t{i}": i for i in range(3, 32)}}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
            pad_token="[PAD]",
            eos_token="[EOS]",
            unk_token="[UNK]",
        )
        for i, name in enumerate(("student", "teacher")):
            torch.manual_seed(100 + i)
            model = AutoModelForCausalLM.from_config(
                Qwen2Config(
                    vocab_size=32,
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    max_position_embeddings=64,
                    tie_word_embeddings=False,
                    eos_token_id=1,
                    pad_token_id=0,
                )
            )
            model.save_pretrained(root / name)
            tokenizer.save_pretrained(root / name)
    dist.barrier()
    actor = make_worker(root / "student", world, teacher=False, full_vocab=True)
    teacher = make_worker(root / "teacher", world, teacher=True, full_vocab=True)
    full_config = FullVocabConfig(backend=args.backend, token_chunk_size=2, vocab_chunk_size=7)
    actor.engine.configure_full_vocab(full_config, str(root / "teacher"))
    distill = DistillationConfig(
        distillation_loss=DistillationLossConfig(
            loss_mode="full_reverse_kl",
            use_policy_gradient=False,
            use_task_rewards=False,
            loss_max_clamp=None,
            log_prob_min_clamp=None,
            full_vocab=full_config,
        )
    )
    actor.set_loss_fn(
        partial(teacher_policy_loss, config=SimpleNamespace(loss_agg_mode="token-mean"), distillation_config=distill)
    )
    rows = [torch.tensor([3 + rank, 7, 8, 9, 1]), torch.tensor([4, 10, 11, 1])]
    original = {n: p.detach().to_local().clone() for n, p in actor.engine.module.named_parameters()}
    records = []
    for step in range(3):
        masks = [torch.tensor([1, 0, 1]), torch.tensor([1, 1])]
        if rank == 0 and step == 1:
            masks = [torch.zeros_like(m) for m in masks]
        data = TensorDict(
            {
                "input_ids": nested(rows),
                "position_ids": nested([torch.arange(len(r)) for r in rows]),
                "prompts": nested([r[:2] for r in rows]),
                "responses": nested([r[2:] for r in rows]),
                "response_mask": nested(masks),
                "loss_mask": nested(masks),
            },
            batch_size=[2],
        )
        tu.assign_non_tensor(data, temperature=1.0, compute_loss=False, full_vocab_teacher=True)
        teacher_result = teacher.infer_batch(data)
        assert_on_cpu(teacher)
        assert all(p.grad is None for p in teacher.engine.module.parameters())
        positions, _ = prediction_positions(data["input_ids"], data["loss_mask"])
        for expected, actual in zip(positions, teacher_result["teacher_positions"].unbind(), strict=True):
            torch.testing.assert_close(expected, actual)
        data.update(teacher_result.select("teacher_hidden", "teacher_positions"))
        if step == 0:
            reference = AutoModelForCausalLM.from_pretrained(root / "student", torch_dtype=torch.float32).cuda().eval()
            total = torch.zeros((), device="cuda")
            with torch.no_grad():
                for ids, pos, th in zip(rows, positions, data["teacher_hidden"].unbind(), strict=True):
                    h = reference.model(ids.cuda()[None], use_cache=False).last_hidden_state[0, pos.cuda()]
                    total += full_reverse_kl(
                        h,
                        reference.lm_head.weight,
                        th.cuda().float(),
                        actor.engine.full_vocab_teacher_head.get(h.device, h.dtype),
                        backend="dense",
                    )
            count = torch.tensor(sum(len(p) for p in positions), device="cuda")
            dist.all_reduce(total)
            dist.all_reduce(count)
            expected_loss = (total / count).item()
            del reference
        tu.assign_non_tensor(data, full_vocab_teacher=False, full_vocab_train=True)
        output = actor.train_batch(data)
        assert_on_cpu(actor)
        metrics = tu.get(output, "metrics")
        loss = metrics["loss"]
        assert torch.isfinite(torch.tensor(loss)), metrics
        if step == 0:
            assert abs(loss - expected_loss) < 5e-4, (loss, expected_loss)
        records.append({"step": step, "loss": loss})
    deltas = {
        n: (p.detach().to_local() - original[n]).float().norm().item()
        for n, p in actor.engine.module.named_parameters()
    }
    assert any(v > 0 for n, v in deltas.items() if "lm_head" in n)
    assert any(v > 0 for n, v in deltas.items() if "layers" in n)
    actor.engine.full_vocab_teacher_head.release()
    if args.checkpoint:
        saved = {n: p.detach().to_local().clone() for n, p in actor.engine.module.named_parameters()}
        actor.save_checkpoint(str(root / "checkpoint"), global_step=3)
        with torch.no_grad():
            for p in actor.engine.module.parameters():
                p.to_local().add_(0.5)
        actor.load_checkpoint(str(root / "checkpoint"))
        for n, p in actor.engine.module.named_parameters():
            torch.testing.assert_close(p.detach().to_local(), saved[n], atol=0, rtol=0)
    result = {"backend": args.backend, "world_size": world, "records": records, "parameter_delta_norms_rank0": deltas}
    result["checkpoint_roundtrip"] = args.checkpoint
    if rank == 0:
        print(json.dumps(result), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    dist.barrier()
    if rank == 0:
        shutil.rmtree(root)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
