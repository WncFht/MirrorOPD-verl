# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit new teacher interfaces and freeze the expanded, read-only evaluation plan."""

from transformers import AutoConfig, AutoTokenizer

from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.studies.boundary.round2_common import (
    ARMS,
    NEW_MODELS,
    OLD_OUT,
    OUT,
    eval_specs,
    model_path,
)


def main():
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    reference = AutoTokenizer.from_pretrained(model_path("r1"), local_files_only=True)
    vocab = {v: k for k, v in reference.get_vocab().items()}
    prompts = read(OLD_OUT / "evaluation-prompts.json")
    audits = {}
    for name, (repo, revision) in NEW_MODELS.items():
        tok = AutoTokenizer.from_pretrained(model_path(name), local_files_only=True)
        cfg = AutoConfig.from_pretrained(model_path(name), local_files_only=True)
        other = {v: k for k, v in tok.get_vocab().items()}
        conflicts = [(i, vocab[i], other[i]) for i in vocab.keys() & other.keys() if vocab[i] != other[i]]
        if name != "math15":
            assert not conflicts, (name, conflicts)
        audits[name] = dict(
            repo=repo,
            revision=revision,
            conflicts=conflicts,
            eos_token_id=tok.eos_token_id,
            context=cfg.max_position_embeddings,
            training_authorized=name != "math15",
        )
        if name == "math15":
            native = []
            for row in prompts:
                ids = tok.apply_chat_template(
                    row["messages"], tokenize=True, add_generation_prompt=True, return_dict=False
                )
                assert len(ids) <= 1024
                native.append(dict(row, prompt_ids=ids, input=tok.decode(ids, skip_special_tokens=False)))
            dump(OUT / "math-native-prompts.json", native)
            audits[name]["evaluation_note"] = (
                "Native Qwen chat template; response cap3072 + prompt<=1024, context4096; no training or GapRecovery."
            )
    dump(OUT / "interface-audit.json", audits)
    dump(OUT / "eval-specs.json", eval_specs())
    dump(
        OUT / "protocol.json",
        dict(
            arms=ARMS,
            new_models=NEW_MODELS,
            effective_batch=16,
            reuse=16,
            horizon=1024,
            lr=1e-6,
            betas=[0.9, 0.999],
            eps=1e-8,
            weight_decay=0.01,
            clip=1,
            scope="3 continuations 64->128; 3 new teachers 0->64; Math15 evaluation only",
            checkpoint_policy="final FP32 weights only; no new optimizer snapshots or per-32 model saves",
            evaluation="Five benchmarks, avg@4, T0.7/top_p0.9, per-prompt seeds72000000+100*i+r; shuffled requests",
            shared_response_cap=31744,
            math_native_response_cap=3072,
            old_reference=str(OLD_OUT),
            heldout_panel="primary M01-M04; additional M05-M12 kept separate",
        ),
    )
    print(audits, flush=True)


if __name__ == "__main__":
    main()
