# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded output probes for mode acceptance, independent of trajectory-bank generation."""

import argparse
from pathlib import Path

from examples.mirroropd.research.backends import hf_models
from examples.mirroropd.research.core.artifacts import complete, file_hash, receipt
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.studies.fixed_transfer.spec import Study


def probe_inputs(study, name):
    rows = read(study.directory / "prepared/bank-prompts.json")
    selected = [next(r for r in rows if r["group"] == g) for g in ["A", "M01", "Code01", "IF01"]]
    return dict(
        model=study.models[name].identity,
        student=study.models[study.student].identity,
        prompts=[dict(id=r["id"], group=r["group"], messages=r["messages"]) for r in selected],
        prompt_policy=study.prompt_policy,
        configured_thinking=study.evaluation.thinking,
        template_source=study.models[study.prompt_policy["template_model"]].identity
        if (study.prompt_policy or {}).get("template_model")
        else None,
        stop_ids=list(study.evaluation.stop_ids),
        max_new_tokens=256,
        sampling="greedy; native BF16 backbone; FP32 output projection",
        source=file_hash(__file__),
        backend=file_hash(hf_models.__file__),
    )


def probe(study, name):
    import torch
    from transformers import AutoTokenizer

    inputs = probe_inputs(study, name)
    destination = study.directory / "mode-probes" / name
    if complete(destination, inputs):
        return
    from examples.mirroropd.research.studies.fixed_transfer.data import prompt_tokenizers

    student_tokenizer, _ = prompt_tokenizers(study)
    tokenizer = AutoTokenizer.from_pretrained(study.models[name].directory, local_files_only=True)
    model = hf_models.load(study.models[name])
    modes = (
        [True, False]
        if model.config.model_type == "qwen3" and not (study.prompt_policy or {}).get("template_model")
        else [study.evaluation.thinking]
    )
    head = model.get_output_embeddings().weight.detach().float()

    def project(module, args, output):
        with torch.autocast("cuda", enabled=False):
            return torch.nn.functional.linear(args[0].float(), head)

    hook = model.get_output_embeddings().register_forward_hook(project)
    results = []
    try:
        with torch.inference_mode():
            for thinking in modes:
                for row in inputs["prompts"]:
                    kwargs = dict(
                        tokenize=True,
                        return_dict=False,
                        add_generation_prompt=True,
                        enable_thinking=thinking,
                        **(study.prompt_policy or {}).get("template_kwargs", {}),
                    )
                    shared = student_tokenizer.apply_chat_template(row["messages"], **kwargs)
                    native = (
                        tokenizer.apply_chat_template(row["messages"], **kwargs) if tokenizer.chat_template else None
                    )
                    variants = {"shared_student": shared}
                    if native is not None and native != shared:
                        variants["native_teacher"] = native
                    for prefix, ids in variants.items():
                        output = model.generate(
                            input_ids=torch.tensor([ids], device="cuda"),
                            attention_mask=torch.ones((1, len(ids)), device="cuda", dtype=torch.long),
                            max_new_tokens=256,
                            do_sample=False,
                            use_cache=True,
                            eos_token_id=list(study.evaluation.stop_ids),
                            pad_token_id=study.evaluation.stop_ids[0],
                        )[0, len(ids) :].tolist()
                        results.append(
                            dict(
                                id=row["id"],
                                group=row["group"],
                                thinking=thinking,
                                prefix=prefix,
                                native_prefix_matches=native == shared if native is not None else None,
                                native_template_available=native is not None,
                                prompt_ids=ids,
                                response_ids=output,
                                text=tokenizer.decode(output, skip_special_tokens=False),
                                ended_with_eos=bool(output and output[-1] in study.evaluation.stop_ids),
                            )
                        )
                        dump(destination / "outputs.json", results)
    finally:
        hook.remove()
    receipt(
        destination,
        inputs,
        records=len(results),
        interpretation="output evidence only; no automatic expert classification",
    )


def main():
    from examples.mirroropd.research.runtime.cuda import setup

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    setup()
    probe(Study.load(args.plan), args.model)


if __name__ == "__main__":
    main()
