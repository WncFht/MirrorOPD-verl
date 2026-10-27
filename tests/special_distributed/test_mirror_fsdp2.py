# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Tiny real FSDP2 mirror training: anchor scopes, reuse, frozen heads and resume.

Run with torchrun on two GPUs; no Ray/vLLM or downloaded models are needed.
"""

import argparse
import inspect
import json
import math
import shutil
import tempfile
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from tensordict import TensorDict
from test_full_vocab_fsdp2 import local_snapshot
from test_shared_teacher_fsdp2 import assert_on_cpu, make_worker, nested, teacher_policy_loss
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen2Config, Qwen3Config

from verl.utils import tensordict_utils as tu
from verl.utils.distributed import initialize_global_process_group
from verl.workers.config import DistillationConfig, DistillationLossConfig
from verl.workers.config.distillation import FullVocabConfig
from verl.workers.engine_workers import ActorRolloutRefWorker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32", "bf16_fp32"])
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--objective", choices=["mdpo", "bpo"], default="mdpo")
    parser.add_argument("--estimator", choices=["full_vocab", "sampled"], default="full_vocab")
    parser.add_argument("--backend", default="precompute_fused")
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--tied", action="store_true")
    parser.add_argument("--model-type", choices=["qwen2", "qwen3"], default="qwen2")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--compare-legacy", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    _, rank, world = initialize_global_process_group(timeout_second=180)
    paths = [tempfile.mkdtemp(prefix="mirror-fsdp-") if rank == 0 else None]
    dist.broadcast_object_list(paths)
    root = Path(paths[0])
    full = args.estimator == "full_vocab"
    if rank == 0:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(
                WordLevel({"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, **{f"t{i}": i for i in range(3, 32)}}, unk_token="[UNK]")
            ),
            pad_token="[PAD]",
            eos_token="[EOS]",
            unk_token="[UNK]",
        )
        for i, name in enumerate(("student", "teacher")):
            torch.manual_seed(100 + i)
            model = AutoModelForCausalLM.from_config(
                (Qwen3Config if args.model_type == "qwen3" else Qwen2Config)(
                    vocab_size=32 + (3 if name == "teacher" else 0),
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=8,
                    max_position_embeddings=64,
                    tie_word_embeddings=args.tied,
                    eos_token_id=1,
                    pad_token_id=0,
                )
            )
            model.save_pretrained(root / name)
            tokenizer.save_pretrained(root / name)
    dist.barrier()
    actor = make_worker(
        root / "student",
        world,
        teacher=False,
        full_vocab=full,
        precision=args.precision,
        gradient_checkpointing=args.gradient_checkpointing,
        token_budget=2 * args.sequence_length,
    )
    teacher = make_worker(root / "teacher", world, teacher=True, full_vocab=full, token_budget=2 * args.sequence_length)
    observed_dtypes = {}

    def record_dtype(name):
        def capture(module, inputs, output):
            observed_dtypes.setdefault(name, set()).add(str(output.dtype))

        return capture

    if args.precision:
        for name, module in actor.engine.module.named_modules():
            if name.endswith(("embed_tokens", "mlp.gate_proj", "model.norm")):
                module.register_forward_hook(record_dtype(name))
    loss_cfg = DistillationLossConfig(
        loss_mode="mirror",
        mirror=dict(objective=args.objective, estimator=args.estimator, eta=args.eta, kappa=0.5, diagnostics=full),
        full_vocab=FullVocabConfig(backend=args.backend, token_chunk_size=2, vocab_chunk_size=7),
        use_policy_gradient=not full,
        policy_loss_mode="vanilla" if full else "token_is",
        use_task_rewards=False,
        loss_max_clamp=None,
        log_prob_min_clamp=None,
    )
    spec = loss_cfg.loss_settings
    teacher_projection_dtypes = set()
    if full:
        actor.engine.configure_full_vocab(loss_cfg.full_vocab, str(root / "teacher"), spec=spec)
        original_get = actor.engine.full_vocab_teacher_head.get

        def checked_teacher_head(device, dtype):
            teacher_projection_dtypes.add(str(dtype))
            return original_get(device, dtype)

        actor.engine.full_vocab_teacher_head.get = checked_teacher_head
    actor.set_loss_fn(
        partial(
            teacher_policy_loss,
            config=SimpleNamespace(loss_agg_mode="token-mean"),
            distillation_config=DistillationConfig(distillation_loss=loss_cfg),
        )
    )
    rows = [
        torch.cat((torch.arange(args.sequence_length - 1).remainder(28) + 3, torch.tensor([1]))),
        torch.cat((torch.arange(args.sequence_length - 2).remainder(28) + 3, torch.tensor([1]))),
    ]
    # Keep both unequal sequences in one packed micro-batch.
    if actor.engine.scaler is not None:
        # Cross growth and resume boundaries quickly; production interval stays 400.
        actor.engine.scaler.set_growth_interval(2)
        actor.engine.scaler._init_scale = 32.0
    all_records = []
    snapshots = []

    def run_outer(version, empty=False):
        masks = [torch.ones(len(r) - 2, dtype=torch.long) for r in rows]
        masks[0][1] = 0
        if empty and rank == 0:
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
        tu.assign_non_tensor(data, temperature=1.0, compute_loss=False, full_vocab_teacher=full)
        labels = teacher.infer_batch(data)
        assert_on_cpu(teacher)
        if full:
            data.update(labels.select("teacher_hidden", "teacher_positions"))
        else:
            data["teacher_logprobs"] = nested([v.unsqueeze(-1) for v in labels["log_probs"].unbind()])
            data["rollout_log_probs"] = nested([torch.full((len(r) - 2,), -math.log(32)) for r in rows])
        tu.assign_non_tensor(
            data,
            full_vocab_teacher=False,
            full_vocab_train=full,
            mirror_anchor_version=(version, 0),
            global_batch_size=2 * world,
            mini_batch_size=2 * world,
            epochs=3,
            seed=42,
            dataloader_kwargs={"shuffle": True},
            collect_metrics=True,
            collect_update_records=True,
            metrics_attempt_start=0,
        )
        original_train = actor.train_mini_batch

        def checked_train(data):
            if spec.anchor_payload == "hidden":
                head = actor.engine.mirror_anchor_head
                frozen = head.cpu_weight.clone()
                snapshots.append(frozen)
            elif spec.anchor_payload == "scalar":
                frozen = data["old_log_probs"].values().clone()
                snapshots.append(frozen)
            else:
                assert getattr(actor.engine, "mirror_anchor_head", None) is None
            out = original_train(data)
            if spec.anchor_payload == "hidden":
                torch.testing.assert_close(head.cpu_weight, frozen, atol=0, rtol=0)
            elif spec.anchor_payload == "scalar":
                torch.testing.assert_close(data["old_log_probs"].values(), frozen, atol=0, rtol=0)
            return out

        actor.train_mini_batch = checked_train
        try:
            worker = SimpleNamespace(actor=actor, mirror_spec=spec)
            out = inspect.unwrap(ActorRolloutRefWorker.update_actor)(worker, data)
        finally:
            actor.train_mini_batch = original_train
        assert tu.get(out, "optimizer_step_attempts") == 3
        assert tu.get(out, "optimizer_steps") == 3
        records = tu.get(out, "update_records")
        assert len(records) == 3
        values = [entry["metrics"]["optim/actor/loss"] for entry in records]
        assert torch.isfinite(torch.as_tensor(values)).all(), values
        if full and spec.anchor_payload:
            assert abs(records[0]["metrics"]["distill/anchor_kl"]) < 1e-7
        if full:
            assert teacher_projection_dtypes == {"torch.bfloat16"}, teacher_projection_dtypes
        assert getattr(actor.engine, "mirror_anchor_head", None) is None
        assert_on_cpu(actor)
        if spec.anchor_payload:
            elapsed = tu.get(out, "metrics")["perf/anchor/prepare_seconds_max"].aggregate()
            assert elapsed >= 0
            durations = [None] * world
            dist.all_gather_object(durations, elapsed)
            assert len(set(durations)) == 1
        all_records.append({"outer": version, "loss": values, "updates": 3})

    initial = local_snapshot(dict(actor.engine.module.named_parameters()))
    if args.compare_legacy:
        assert spec.teacher_only_kl
        actor.save_checkpoint(str(root / "initial"), global_step=0)
    run_outer(1)
    if args.compare_legacy:
        expected_parameters = local_snapshot(dict(actor.engine.module.named_parameters()))
        expected_optimizer = local_snapshot(actor.engine.optimizer.state_dict())
        expected_scheduler = local_snapshot(actor.engine.lr_scheduler.state_dict())
        actor.load_checkpoint(str(root / "initial"))
        legacy = DistillationLossConfig(
            loss_mode="full_reverse_kl",
            full_vocab=loss_cfg.full_vocab,
            use_policy_gradient=False,
            use_task_rewards=False,
            loss_max_clamp=None,
            log_prob_min_clamp=None,
        )
        actor.engine.full_vocab_spec = legacy.loss_settings
        actor.set_loss_fn(
            partial(
                teacher_policy_loss,
                config=SimpleNamespace(loss_agg_mode="token-mean"),
                distillation_config=DistillationConfig(distillation_loss=legacy),
            )
        )
        run_outer(1)
        torch.testing.assert_close(
            local_snapshot(dict(actor.engine.module.named_parameters())), expected_parameters, atol=0, rtol=0
        )
        torch.testing.assert_close(
            local_snapshot(actor.engine.optimizer.state_dict()), expected_optimizer, atol=0, rtol=0
        )
        assert local_snapshot(actor.engine.lr_scheduler.state_dict()) == expected_scheduler
        actor.engine.full_vocab_spec = spec
        actor.set_loss_fn(
            partial(
                teacher_policy_loss,
                config=SimpleNamespace(loss_agg_mode="token-mean"),
                distillation_config=DistillationConfig(distillation_loss=loss_cfg),
            )
        )
    run_outer(2, empty=True)
    after = local_snapshot(dict(actor.engine.module.named_parameters()))
    assert any(not torch.equal(initial[k], after[k]) for k in initial)
    if spec.anchor_payload:
        assert not torch.equal(snapshots[0], snapshots[1]), "next outer must refresh q"
    if args.checkpoint:
        actor.save_checkpoint(str(root / "checkpoint"), global_step=2)
        run_outer(3)
        expected = local_snapshot(dict(actor.engine.module.named_parameters()))
        expected_opt = local_snapshot(actor.engine.optimizer.state_dict())
        expected_sched = local_snapshot(actor.engine.lr_scheduler.state_dict())
        expected_scaler = actor.engine.scaler.state_dict() if actor.engine.scaler else None
        actor.load_checkpoint(str(root / "checkpoint"))
        run_outer(3)
        torch.testing.assert_close(
            local_snapshot(dict(actor.engine.module.named_parameters())), expected, atol=0, rtol=0
        )
        torch.testing.assert_close(local_snapshot(actor.engine.optimizer.state_dict()), expected_opt, atol=0, rtol=0)
        assert local_snapshot(actor.engine.lr_scheduler.state_dict()) == expected_sched
        assert (actor.engine.scaler.state_dict() if actor.engine.scaler else None) == expected_scaler
    if args.precision:
        expected_dtype = {
            "bf16": "torch.bfloat16",
            "fp16": "torch.float16",
            "fp32": "torch.float32",
            "bf16_fp32": "torch.float32",
        }[args.precision]
        assert observed_dtypes and all(v == {expected_dtype} for v in observed_dtypes.values()), observed_dtypes
    result = dict(
        activation_dtypes={k: sorted(v) for k, v in observed_dtypes.items()},
        teacher_projection_dtypes=sorted(teacher_projection_dtypes),
        precision=args.precision,
        sequence_length=args.sequence_length,
        scaler=actor.engine.scaler.state_dict() if actor.engine.scaler else None,
        objective=args.objective,
        model_type=args.model_type,
        estimator=args.estimator,
        eta=args.eta,
        backend=args.backend,
        tied=args.tied,
        checkpoint=args.checkpoint,
        world_size=world,
        teacher_extra_rows=3,
        records=all_records,
        frozen_anchor=True,
        endpoint_legacy_equal=args.compare_legacy,
        actor_wrapper_and_timing=True,
    )
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    dist.barrier()
    if rank == 0:
        shutil.rmtree(root)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
