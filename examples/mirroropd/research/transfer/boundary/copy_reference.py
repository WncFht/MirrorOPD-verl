# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only teacher-weight references for the student numerical execution path."""

import gc

import torch

from ...common.metrics import fixed_metrics
from . import fast_engine as engine
from .native import hidden, load, setup
from .round2_common import MODELS, OUT, dump, model_path


@torch.no_grad()
def main():
    setup()
    directory = OUT / "teacher-copy"
    rows = [r for r in engine.panel() if r["group"] in ["M01", "M02", "M03", "M04"]]
    assert len(rows) == 64
    summary = []
    for name in ["justrl", "nemotron", "minimath", "deepscaler", "brorl", "deepcoder"]:
        head, states = engine.teacher_cache(name, rows)
        native = load(name, path=model_path(name))
        diagonal = []
        for row in rows[:2]:
            h = hidden(native, row).float()
            q = states[row["id"]].to("cuda").float()
            a = torch.nn.functional.linear(h[:8], head)
            b = torch.nn.functional.linear(q[:8], head)
            diagonal.append(
                dict(
                    id=row["id"],
                    hidden_max_error=float((h - q).abs().max()),
                    max_rkl=float(fixed_metrics(a, b)["rkl"].abs().max()),
                )
            )
        assert max(r["max_rkl"] for r in diagonal) < 1e-5
        # Construct a point in the actual R1 student family, retaining its config and buffers.
        model = load("r1", student=True, path=model_path("r1"))
        buffers = {k: v.clone() for k, v in model.named_buffers()}
        keys = [
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
            "rope_theta",
            "rope_scaling",
            "rms_norm_eps",
            "tie_word_embeddings",
            "vocab_size",
            "use_sliding_window",
        ]
        differences = {
            k: dict(student=getattr(model.config, k, None), teacher=getattr(native.config, k, None))
            for k in keys
            if getattr(model.config, k, None) != getattr(native.config, k, None)
        }
        model.load_state_dict(native.state_dict(), strict=True)
        assert all(torch.equal(v, buffers[k]) for k, v in model.named_buffers())
        del native, buffers
        gc.collect()
        torch.cuda.empty_cache()
        scores = engine.score(model, head, states, rows, 0, 512)
        dump(directory / f"{name}-scores.json", scores)
        summary.append(
            dict(
                teacher=name,
                revision=MODELS[name],
                prompts=len(rows),
                native_diagonal=diagonal,
                student_config="original R1-1.5B; buffers preserved; strict parameter copy",
                config_differences=differences,
                rkl=sum(r["stats"]["all"]["rkl"] / 1024 for r in scores) / len(rows),
                js=sum(r["stats"]["all"]["js"] / 1024 for r in scores) / len(rows),
            )
        )
        dump(directory / "summary.json", summary)
        print("TEACHER_COPY", summary[-1], flush=True)
        del model, head, states
        gc.collect()
        torch.cuda.empty_cache()
    dump(
        directory / "complete.json",
        dict(
            teachers=6,
            scored_prompts=384,
            training_updates=0,
            explanation="FP32 student path vs BF16 teacher: teacher-weight reference, not a lower bound",
        ),
    )


if __name__ == "__main__":
    main()
