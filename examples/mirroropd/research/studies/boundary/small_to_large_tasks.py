# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Paired avg@4 task readouts for the original 64-step small-to-large conditions."""

import argparse

import numpy as np

from examples.mirroropd.research.core.io import dump, read, write_csv
from examples.mirroropd.research.core.transfer_metrics import gap_row, summarize_gap
from examples.mirroropd.research.studies.boundary import small_to_large as study
from examples.mirroropd.research.studies.boundary.round2_analyze import answers
from examples.mirroropd.research.studies.boundary.round2_common import DATASETS, OLD_OUT
from examples.mirroropd.research.studies.fixed_transfer.evaluation import validate_records
from examples.mirroropd.research.studies.fixed_transfer.spec import Evaluation


def analyze(partial=False):
    prompts = read(OLD_OUT / "evaluation-prompts.json")
    profile = Evaluation(stop_ids=(151643,))
    output = study.OUT / "analysis/tasks"
    memo = {}

    def load(directory, model=None):
        if directory not in memo:
            vectors, records = answers(directory)
            done, expected = validate_records(records, prompts, profile)
            assert done == expected
            protocol = read(directory / "protocol.json")
            assert protocol["samples"] == 4 and protocol["template"] == "shared"
            command = protocol["command"]
            assert command[command.index("--dtype") + 1] == "bfloat16"
            assert command[command.index("--max-model-len") + 1] == "32768"
            if model is not None:
                assert protocol["model"] == list(study.MODELS[model])
            else:
                assert protocol["temperature"] == 0.7 and protocol["top_p"] == 0.9
                assert protocol["response_cap"] == 31744
            memo[directory] = vectors
        return memo[directory]

    before = load(OLD_OUT / "baseline/r7-shared", "r7")
    rows, missing = [], []
    rng = np.random.default_rng(2026102905)
    for arm, teacher_name in study.ARMS.items():
        directory = study.OUT / "evaluations" / arm
        if not (directory / "complete.json").exists():
            missing.append(arm)
            continue
        assert read(directory / "protocol.json")["path"] == str(study.CACHE / "checkpoints" / arm / "step064")
        after = load(directory)
        teacher = load(OLD_OUT / "baseline" / (teacher_name + "-shared"), teacher_name)
        means_all, boots_all = [], []
        for dataset in DATASETS:
            means, boots = summarize_gap(before[dataset], teacher[dataset], after[dataset], rng)
            means_all.append(means)
            boots_all.append(boots)
            rows.append(
                dict(arm=arm, teacher=teacher_name, student="r7", step=64, dataset=dataset, **gap_row(means, boots))
            )
        rows.append(
            dict(
                arm=arm,
                teacher=teacher_name,
                student="r7",
                step=64,
                dataset="macro5",
                **gap_row(np.mean(means_all, axis=0), np.mean(boots_all, axis=0)),
            )
        )
        weights = list(DATASETS.values())
        rows.append(
            dict(
                arm=arm,
                teacher=teacher_name,
                student="r7",
                step=64,
                dataset="pooled630",
                **gap_row(
                    np.average(means_all, axis=0, weights=weights), np.average(boots_all, axis=0, weights=weights)
                ),
            )
        )
    if rows:
        write_csv(output / "gap-recovery.csv", rows)
    summary = dict(
        scope="original step64 avg@4; not deterministic step320 readouts",
        missing=missing,
        results=rows,
        sampling="temperature .7, top_p .9, cap31744, seed72000000+100*index+sample",
        uncertainty="5000 paired prompt bootstraps; macro5 averages datasets equally",
    )
    dump(output / "summary.json", summary)
    if not partial:
        assert not missing, missing
        dump(
            output / "verification.json",
            dict(
                complete=True,
                scope="original_step64_avg4",
                checkpoint_step=64,
                response_slots_per_model=2520,
                arms=list(study.ARMS),
                response_cap=31744,
                model_length=32768,
            ),
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    result = analyze(args.partial)
    print({"missing": result["missing"], "macro5": [r for r in result["results"] if r["dataset"] == "macro5"]})


if __name__ == "__main__":
    main()
