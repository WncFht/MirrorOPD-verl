# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only accuracy replay from an immutable generation artifact."""

import argparse
import itertools

from omegaconf import OmegaConf

from verl.trainer.ppo.readout_callback import readout_options
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig

from .accuracy import AccuracyCollector
from .artifacts import ArtifactStore, identity
from .config import EvaluationConfig
from .coordinator import EvaluationCoordinator, read_questions
from .generation_contract import input_identity


def regrade(config, generation, output):
    options = EvaluationConfig.parse(readout_options(config, EvaluationCoordinator))
    source = ArtifactStore(generation)
    manifest = source.manifest(complete=True)
    specifications = [(d, read_questions(d)) for d in options.datasets]
    expected_hash = identity([(s.name, s.n, s.protocol, questions) for s, questions in specifications])
    if manifest["metadata"].get("question_set_sha256") != expected_hash:
        raise ValueError("Replay configuration does not match generated questions and n")
    if identity(
        {k: {a: b for a, b in v.items() if a != "answer_channel"} for k, v in manifest["metadata"]["protocols"].items()}
    ) != identity({k: v.sampling_metadata() for k, v in options.protocols.items()}):
        raise ValueError("Replay sampling or input protocol changed")
    model = omega_conf_to_dataclass(config.actor_rollout_ref.model, dataclass_type=HFModelConfig)
    recorded_input = manifest["metadata"].get("engine_contract", {}).get("input_id")
    if recorded_input is None or recorded_input != input_identity(model):
        raise ValueError(
            "Replay tokenizer/input identity is unverified or changed; "
            "use the original frozen evaluator for legacy artifacts"
        )
    tokenizer = model.tokenizer
    questions = {
        (spec.name, q["id"]): (q, spec, options.protocols[spec.protocol]) for spec, rows in specifications for q in rows
    }
    collector = AccuracyCollector(output, specifications, tokenizer, workers=options.verifier_workers)
    try:
        for name in sorted(manifest["files"]):
            if not name.startswith("generations/"):
                continue
            rows = source.iter_rows(name)
            while batch := list(itertools.islice(rows, 32)):
                collector.submit([(row, *questions[row["dataset"], row["source_id"]]) for row in batch])
        result = collector.finish(source)
        collector = None
        return result
    finally:
        if collector is not None:
            collector.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    regrade(OmegaConf.load(args.config), args.generation, args.output)


if __name__ == "__main__":
    main()
