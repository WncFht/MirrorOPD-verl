# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only production teacher inference audit against frozen HF reference.

Run with PYTHONPATH pointing to the E54 source snapshot to audit that exact
implementation. Only the four front GPUs are used; no actor or optimizer exists.
The second forward restores canonical RoPE buffers in memory, leaving weights
and packed input unchanged. Comparison to unpacked HF includes backend rounding.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tensordict import TensorDict
from transformers import AutoTokenizer
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

from verl.trainer.distillation.full_vocab import project
from verl.trainer.distillation.hidden import FrozenTeacherHead
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.distributed import initialize_global_process_group
from verl.workers.engine_workers import TrainingWorker


def nested(rows):
    return torch.nested.nested_tensor(rows, layout=torch.jagged)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    _, rank, world = initialize_global_process_group(timeout_second=600)
    assert world == 4
    config = OmegaConf.load(args.config)
    config.trainer.n_gpus_per_node = world
    trainer = PPOTrainerSync.__new__(PPOTrainerSync)
    trainer.config = config
    trainer.distillation_config = omega_conf_to_dataclass(config.distillation)
    trainer.tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path, local_files_only=True)
    worker_config = trainer._make_shared_teacher_config()
    worker = TrainingWorker(worker_config)
    worker.reset()
    assert worker.engine.optimizer is None
    cases = json.loads((args.reference / "cases.json").read_text())
    ends = [r for r in cases if r["category"] == "archived_terminal"]
    loops = [r for r in cases if r["category"] == "e54_late_loop"]
    chosen = [ends[rank], loops[rank]]
    ref_rows = json.loads((args.reference / "teacher_30b.json").read_text())
    ref_lp = torch.load(args.reference / "teacher_30b-logprobs.pt", map_location="cpu", weights_only=True)
    lookup = {(r["case_id"], r["variant"], r["position"]): i for i, r in enumerate(ref_rows)}
    tokens, masks, responses, prompts, expected = [], [], [], [], []
    for case in chosen:
        response = case["response_ids"] + [151643]
        positions = case["positions"]
        mask = torch.zeros(len(response), dtype=torch.long)
        mask[positions] = 1
        prompts.append(torch.tensor(case["prompt_ids"]))
        responses.append(torch.tensor(response))
        tokens.append(torch.tensor(case["prompt_ids"] + response))
        masks.append(mask)
        expected.append(torch.tensor([len(case["prompt_ids"]) - 1 + i for i in positions]))
    head = FrozenTeacherHead(worker_config.model_config.path)
    canonical_rope = Qwen3MoeRotaryEmbedding(worker_config.model_config.hf_config, device="cpu")
    before = {}
    for name, module in worker.engine.module.named_modules():
        if hasattr(module, "inv_freq"):
            before[name] = dict(
                dtype=str(module.inv_freq.dtype),
                max_abs_error=(module.inv_freq.cpu().float() - canonical_rope.inv_freq).abs().max().item(),
            )
    assert before
    records = []
    for mode in ["e54_snapshot", "restore_rope_only"]:
        if mode == "restore_rope_only":
            for module in worker.engine.module.modules():
                if hasattr(module, "inv_freq"):
                    device = module.inv_freq.device
                    module.inv_freq = canonical_rope.inv_freq.to(device).clone()
                    module.original_inv_freq = canonical_rope.original_inv_freq.to(device).clone()
        data = TensorDict(
            dict(
                input_ids=nested(tokens),
                position_ids=nested([torch.arange(len(r)) for r in tokens]),
                prompts=nested(prompts),
                responses=nested(responses),
                response_mask=nested(masks),
                loss_mask=nested(masks),
            ),
            batch_size=[2],
        )
        tu.assign_non_tensor(
            data, temperature=1.0, compute_loss=False, full_vocab_teacher=True, calculate_entropy=False
        )
        output = worker.infer_batch(data)
        assert all(p.grad is None for p in worker.engine.module.parameters())
        for case, h, positions, want in zip(
            chosen, output["teacher_hidden"].unbind(), output["teacher_positions"].unbind(), expected, strict=True
        ):
            torch.testing.assert_close(positions.cpu(), want)
            with torch.inference_mode():
                z = project(h.cuda(), head.get("cuda", h.dtype))
                lp = z.double().log_softmax(-1)
                ref = (
                    torch.stack([ref_lp[lookup[(case["id"], "training_prompt", pos)]] for pos in case["positions"]])
                    .cuda()
                    .double()
                    .log_softmax(-1)
                )
                kl = (ref.exp() * (ref - lp)).sum(-1)
                ps = lp[:, [151643, 151645]].exp().sum(-1)
                qs = ref[:, [151643, 151645]].exp().sum(-1)
                for j, pos in enumerate(case["positions"]):
                    records.append(
                        dict(
                            rank=rank,
                            mode=mode,
                            case_id=case["id"],
                            position=pos,
                            kl_hf_to_worker=kl[j].item(),
                            top1_equal=bool(lp[j].argmax() == ref[j].argmax()),
                            worker_stop_probability=ps[j].item(),
                            hf_stop_probability=qs[j].item(),
                            max_logprob_error=(lp[j] - ref[j]).abs().max().item(),
                        )
                    )
        head.release()
        torch.cuda.empty_cache()
    gathered = [None] * world if rank == 0 else None
    dist.gather_object({"rank": rank, "rope_before": before, "records": records}, gathered, dst=0)
    if rank == 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"world_size": world, "scope": __doc__, "ranks": gathered}, indent=2) + "\n")
        print("Saved production teacher audit", args.out, flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
