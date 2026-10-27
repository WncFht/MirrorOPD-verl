# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""torchrun --standalone --nproc-per-node=2 ... --backend recompute.

Actual FSDP2 head hooks, offload, packed positions, uneven tokens, empty local
mask, and repeated updates. Synthetic Qwen2/Qwen3, optional tied parameters and
unequal output supports; no Ray or vLLM.
"""

import argparse
import copy
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
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen2Config, Qwen3Config

from verl.trainer.distillation.full_vocab import full_reverse_kl
from verl.trainer.distillation.hidden import prediction_positions
from verl.utils import tensordict_utils as tu
from verl.utils.distributed import initialize_global_process_group
from verl.utils.metric.observations import Observation
from verl.workers.config import DistillationConfig, DistillationLossConfig
from verl.workers.config.distillation import FullVocabConfig


def local_snapshot(value):
    """Copy full optimizer containers, comparing each rank's actual shard."""
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if hasattr(value, "to_local"):
            value = value.to_local()
        return value.cpu().clone()
    if isinstance(value, dict):
        return {k: local_snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [local_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(local_snapshot(v) for v in value)
    return copy.deepcopy(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["dense", "recompute", "cached", "streaming", "precompute", "precompute_fused", "liger"],
        default="recompute",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--state-output", type=Path)
    parser.add_argument("--model-type", choices=["qwen2", "qwen3"], default="qwen2")
    parser.add_argument("--tied-student", action="store_true")
    parser.add_argument("--tied-teacher", action="store_true")
    parser.add_argument("--teacher-extra-rows", type=int, default=0)
    parser.add_argument("--teacher-hidden-size", type=int, default=32)
    parser.add_argument("--direction", choices=["reverse", "forward"], default="reverse")
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
            hidden_size = args.teacher_hidden_size if name == "teacher" else 32
            model = AutoModelForCausalLM.from_config(
                (Qwen3Config if args.model_type == "qwen3" else Qwen2Config)(
                    vocab_size=32 + (args.teacher_extra_rows if name == "teacher" else 0),
                    hidden_size=hidden_size,
                    intermediate_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=hidden_size // 4,
                    max_position_embeddings=64,
                    tie_word_embeddings=args.tied_student if name == "student" else args.tied_teacher,
                    eos_token_id=1,
                    pad_token_id=0,
                )
            )
            model.save_pretrained(root / name)
            tokenizer.save_pretrained(root / name)
    dist.barrier()
    actor = make_worker(root / "student", world, teacher=False, full_vocab=True)
    teacher = make_worker(root / "teacher", world, teacher=True, full_vocab=True)
    full_config = FullVocabConfig(
        backend=args.backend,
        token_chunk_size=2,
        vocab_chunk_size=7,
        forward_teacher_support="student" if args.teacher_extra_rows else "equal",
    )
    from verl.workers.config.distillation_spec import get_distillation_loss_settings

    actor.engine.configure_full_vocab(
        full_config, str(root / "teacher"), spec=get_distillation_loss_settings(f"full_{args.direction}_kl")
    )
    distill = DistillationConfig(
        distillation_loss=DistillationLossConfig(
            loss_mode=f"full_{args.direction}_kl",
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
                    if args.direction == "forward":
                        from verl.trainer.distillation.forward_vocab import full_forward_kl_result

                        total += full_forward_kl_result(
                            h,
                            reference.lm_head.weight,
                            th.cuda().float(),
                            actor.engine.full_vocab_teacher_head.get(h.device, h.dtype),
                            backend="dense",
                            teacher_support="student",
                        ).loss_sum
                        continue
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
        record = {"step": step, "loss": loss}
        if args.backend != "liger":
            metric = "full_forward_kl" if args.direction == "forward" else "full_kl"
            prefixes = [f"distillation/{metric}_position/p{i * 10:02d}_{(i + 1) * 10:02d}/" for i in range(10)]
            sums = [Observation.merge_partitions(metrics[p + "sum"]) for p in prefixes]
            counts = [Observation.merge_partitions(metrics[p + "token_count"]) for p in prefixes]
            assert sum(counts) == 4 * (world - 1 if step == 1 else world), counts
            multiplier = world - 1 if step == 1 else world
            assert counts == [multiplier * c for c in [0, 1, 1, 0, 0, 0, 0, 1, 1, 0]], counts
            assert abs(sum(sums) / sum(counts) - Observation.merge_partitions(metrics[f"distillation/{metric}"])) < 1e-7
            if args.direction == "forward":
                assert "distillation/full_kl" in metrics
                assert "distillation/teacher_outside_student_mass" in metrics
            record.update(position_kl_sums=sums, position_token_counts=counts)
        records.append(record)
    deltas = {
        n: (p.detach().to_local() - original[n]).float().norm().item()
        for n, p in actor.engine.module.named_parameters()
    }
    if args.state_output:
        args.state_output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "parameters": local_snapshot(dict(actor.engine.module.named_parameters())),
                "optimizer": local_snapshot(actor.engine.optimizer.state_dict()),
            },
            args.state_output / f"rank{rank}.pt",
        )
    head_name = "embed_tokens" if args.tied_student else "lm_head"
    assert any(v > 0 for n, v in deltas.items() if head_name in n)
    if args.tied_student:
        assert actor.engine.module.lm_head.weight is actor.engine.module.model.embed_tokens.weight
    assert any(v > 0 for n, v in deltas.items() if "layers" in n)
    actor.engine.full_vocab_teacher_head.release()
    if args.checkpoint:
        saved = {n: p.detach().to_local().clone() for n, p in actor.engine.module.named_parameters()}
        # Scheduler invocations are an independent clock, not inferred from updates.
        actor.engine.lr_scheduler_step()
        actor.engine.lr_scheduler_step()
        scheduler_state = copy.deepcopy(actor.engine.lr_scheduler.state_dict())
        optimizer_state = local_snapshot(actor.engine.optimizer.state_dict())
        actor.save_checkpoint(str(root / "checkpoint"), global_step=3)
        actor.train_batch(data)
        actor.engine.lr_scheduler_step()
        assert actor.engine.lr_scheduler.state_dict() != scheduler_state
        next_parameters = {n: p.detach().to_local().cpu().clone() for n, p in actor.engine.module.named_parameters()}
        next_optimizer = local_snapshot(actor.engine.optimizer.state_dict())
        next_scheduler = copy.deepcopy(actor.engine.lr_scheduler.state_dict())
        actor.load_checkpoint(str(root / "checkpoint"))
        for n, p in actor.engine.module.named_parameters():
            torch.testing.assert_close(p.detach().to_local(), saved[n], atol=0, rtol=0)
        assert actor.engine.lr_scheduler.state_dict() == scheduler_state
        torch.testing.assert_close(local_snapshot(actor.engine.optimizer.state_dict()), optimizer_state, atol=0, rtol=0)
        actor.train_batch(data)
        actor.engine.lr_scheduler_step()
        for n, p in actor.engine.module.named_parameters():
            torch.testing.assert_close(p.detach().to_local().cpu(), next_parameters[n], atol=0, rtol=0)
        torch.testing.assert_close(local_snapshot(actor.engine.optimizer.state_dict()), next_optimizer, atol=0, rtol=0)
        assert actor.engine.lr_scheduler.state_dict() == next_scheduler
    result = {
        "backend": args.backend,
        "direction": args.direction,
        "world_size": world,
        "records": records,
        "parameter_delta_norms_rank0": deltas,
    }
    result["checkpoint_roundtrip"] = args.checkpoint
    result["scheduler_optimizer_roundtrip"] = args.checkpoint
    result["resumed_next_update_equivalent"] = args.checkpoint
    result.update(
        model_type=args.model_type,
        tied_student=args.tied_student,
        tied_teacher=args.tied_teacher,
        teacher_extra_rows=args.teacher_extra_rows,
        teacher_hidden_size=args.teacher_hidden_size,
    )
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
