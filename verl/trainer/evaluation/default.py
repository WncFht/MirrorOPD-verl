# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Default validation protocol using the trainer's native generation and rewards."""

import os
import uuid
from collections import defaultdict

import numpy as np
import transfer_queue as tq

from verl.trainer.ppo.readout_callback import ReadoutCallback
from verl.utils import tensordict_utils as tu
from verl.utils.metric.evaluation import evaluation_records
from verl.utils.pipeline_metrics import aggregate_validation_windows


class DefaultEvaluator(ReadoutCallback):
    """Keep the native validation dataset, reward metrics and JSONL dump protocol."""

    handles_validation = True

    def validate(self, trainer):
        if trainer.val_dataloader is None:
            raise ValueError("Validation requires nonempty data.val_files")
        # Lists to collect samples for the table
        sample_uids = []
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        data_sources = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        dump_all_inputs: list[str] = []
        dump_all_outputs: list[str] = []
        dump_all_keys: list[str] = []
        session_to_sample_idx: dict[str, int] = {}
        # To avoid failure sessions
        expected_acc_counts: dict[tuple[str, str], int] = {}
        prompt_metadata = {}
        source_ordinals = defaultdict(int)
        sample_indices = []
        sample_response_lengths = []
        sample_finish_reasons = []
        sample_tool_counts = []
        source_fields = defaultdict(set)
        pipeline_windows = []

        for batch_dict in trainer.val_dataloader:
            # 1. put batch to agent loop manager
            batch_dict["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object
            )
            batch_size = len(batch_dict["uid"])
            batch_data_sources = batch_dict.get("data_source", ["unknown"] * batch_size)
            batch_rollout_ns = batch_dict.get(
                "__rollout_n__", [trainer.config.actor_rollout_ref.rollout.val_kwargs.n] * batch_size
            )
            for row_index, (uid, data_source, rollout_n) in enumerate(
                zip(batch_dict["uid"], batch_data_sources, batch_rollout_ns, strict=True)
            ):
                rollout_n = int(rollout_n)
                key = (str(data_source), str(uid))
                specification = batch_dict.get("reward_model", [{}] * batch_size)[row_index]
                declared = specification.get("metric_fields", [])
                if isinstance(declared, str) or any(not isinstance(field, str) for field in declared):
                    raise ValueError("reward_model.metric_fields must be a collection of score names")
                source_fields[str(data_source)].update(declared)
                expected_acc_counts[key] = rollout_n
                source_ordinals[str(data_source)] += 1
                prompt_metadata[key] = {
                    "prompt_id": f"{data_source}:{source_ordinals[str(data_source)] - 1}",
                    "metric_fields": list(declared),
                    "prompt": batch_dict["raw_prompt"][row_index],
                    "ground_truth": batch_dict.get("reward_model", [{}] * batch_size)[row_index].get("ground_truth"),
                }

            batch = tu.get_tensordict(batch_dict)
            tu.assign_non_tensor_data(batch, "global_steps", trainer.global_steps)
            tu.assign_non_tensor_data(batch, "validate", True)
            # Register each prompt (GRPO group) in TransferQueue as a tag-only status marker.
            # global_steps is required by ReplayBuffer's metadata sync / staleness ordering.
            tags = [
                {"is_prompt": True, "status": "pending", "global_steps": trainer.global_steps}
                for _ in range(len(batch))
            ]
            tq.kv_batch_put(keys=list(batch["uid"]), partition_id="val", tags=tags)
            trainer._stamp_rollout_submission(batch, validate=True)
            trainer.agent_loop_manager.generate_sequences(batch)

            # 2. sample batch from replay buffer: one prompt (GRPO group) per submitted row.
            batch, _ = trainer.replay_buffer.sample(
                global_steps=trainer.global_steps, partition_id="val", batch_size=len(batch)
            )

            # 3. [OPTIONAL] compute reward score with colocated reward model
            if trainer.reward_loop_manager.reward_loop_worker_handles is None:
                trainer.checkpoint_manager.sleep_replicas()
                batch = trainer._compute_reward_colocate(batch)
                trainer.checkpoint_manager.update_weights()

            # 4. collect necessary data for logging
            # For multi-output agent loops, only use the final output per session for metrics.
            # Keys have format {uid}_{session_id}_{index}; keep only the highest index per session.
            session_max: dict[str, tuple[int, int]] = {}  # session_key -> (max_index, position)
            for pos, key in enumerate(batch.keys):
                parts = key.rsplit("_", 2)
                if len(parts) == 3:
                    session_key = f"{parts[0]}_{parts[1]}"
                    index = int(parts[2])
                    if session_key not in session_max or index > session_max[session_key][0]:
                        session_max[session_key] = (index, pos)
                else:
                    session_max[key] = (0, pos)
            sorted_sessions = sorted(session_max.items(), key=lambda x: x[1][1])
            final_indices = [pos for _, (_, pos) in sorted_sessions]
            final_keys = [batch.keys[i] for i in final_indices]
            sample_indices.extend(int(key.rsplit("_", 2)[1]) for key in final_keys)
            base_offset = len(sample_scores)
            session_to_sample_idx.update(
                {session_key: base_offset + j for j, (session_key, _) in enumerate(sorted_sessions)}
            )

            text_data = tq.kv_batch_get(
                keys=batch.keys, partition_id=batch.partition_id, select_fields=["prompts", "responses"]
            )
            response_lengths = text_data["responses"].offsets().diff().tolist()
            sample_response_lengths.extend(response_lengths[i] for i in final_indices)
            text_data["prompts"] = text_data["prompts"].to_padded_tensor(padding=trainer.tokenizer.pad_token_id)
            text_data["responses"] = text_data["responses"].to_padded_tensor(padding=trainer.tokenizer.pad_token_id)
            all_inputs = [trainer.tokenizer.decode(ids, skip_special_tokens=True) for ids in text_data["prompts"]]
            all_outputs = [trainer.tokenizer.decode(ids, skip_special_tokens=True) for ids in text_data["responses"]]

            fields = ["uid", "rm_scores", "num_turns", "reward_model", "data_source", "extra_fields"]
            data = tq.kv_batch_get(keys=final_keys, partition_id=batch.partition_id, select_fields=fields)

            sample_uids.extend(data.pop("uid").tolist())
            sample_outputs.extend(all_outputs[i] for i in final_indices)
            sample_inputs.extend(all_inputs[i] for i in final_indices)
            scores = data["rm_scores"].sum(dim=1).tolist()
            sample_scores.extend(scores)
            sample_turns.extend(data.pop("num_turns").tolist())
            reward_extra_infos_dict["reward"].extend(scores)

            extra_fields_list = data.pop("extra_fields", None)
            sample_tool_counts.extend(
                [
                    entry.get("tool_call_counts") if isinstance(entry, dict) else None
                    for entry in extra_fields_list.tolist()
                ]
                if extra_fields_list is not None
                else [None] * len(final_indices)
            )
            sample_finish_reasons.extend(
                [
                    entry.get("finish_reason") if isinstance(entry, dict) else None
                    for entry in extra_fields_list.tolist()
                ]
                if extra_fields_list is not None
                else [None] * len(final_indices)
            )
            if extra_fields_list is not None:
                n_prior = len(reward_extra_infos_dict["reward"]) - len(extra_fields_list.tolist())
                for extra_field in extra_fields_list.tolist():
                    reward_extra_info = (
                        extra_field.get("reward_extra_info", {}) if isinstance(extra_field, dict) else {}
                    )
                    for key in reward_extra_infos_dict:
                        if key != "reward" and key not in reward_extra_info:
                            reward_extra_infos_dict[key].append(None)
                    for key, value in reward_extra_info.items():
                        if key not in reward_extra_infos_dict:
                            reward_extra_infos_dict[key] = [None] * n_prior
                        reward_extra_infos_dict[key].append(value)
                    n_prior += 1

            reward_model = data.pop("reward_model", None)
            if reward_model is not None:
                sample_gts.extend([item.get("ground_truth", None) for item in reward_model.tolist()])
            else:
                sample_gts.extend([None] * len(final_indices))

            data_source = data.pop("data_source", None)
            if data_source is not None:
                data_sources.extend(data_source.tolist())
            else:
                data_sources.extend(["unknown"] * len(final_indices))

            dump_all_inputs.extend(all_inputs)
            dump_all_outputs.extend(all_outputs)
            dump_all_keys.extend(batch.keys)

            pipeline_windows.append(trainer._collect_pipeline_metrics(batch))

            # 5. cleanup transfer queue
            tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)

        # logger to wandb
        trainer._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump to local dir
        val_data_dir = trainer.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            # Sort according to uid (so that generations in the same rollout are together)
            sort_keys = []
            for key in dump_all_keys:
                parts = key.rsplit("_", 2)
                sort_keys.append((parts[0], int(parts[1]), int(parts[2])) if len(parts) == 3 else (key, 0, 0))
            sorted_indices = sorted(range(len(dump_all_keys)), key=lambda i: sort_keys[i])
            dump_all_inputs = [dump_all_inputs[i] for i in sorted_indices]
            dump_all_outputs = [dump_all_outputs[i] for i in sorted_indices]
            dump_all_keys = [dump_all_keys[i] for i in sorted_indices]

            # For ground truths, scores and reward extra infos, find the values in the
            # lists for the final samples of each session
            dump_all_sessions = [
                f"{parts[0]}_{parts[1]}" if len(parts) == 3 else key
                for key in dump_all_keys
                for parts in [key.rsplit("_", 2)]
            ]
            session_final_indices = [session_to_sample_idx[session] for session in dump_all_sessions]
            trainer._dump_generations(
                inputs=dump_all_inputs,
                outputs=dump_all_outputs,
                gts=[sample_gts[i] for i in session_final_indices],
                scores=[sample_scores[i] for i in session_final_indices],
                reward_extra_infos_dict={
                    k: [v[i] for i in session_final_indices] for k, v in reward_extra_infos_dict.items()
                }
                | {"uid": dump_all_keys},
                dump_path=val_data_dir,
            )

        if val_data_dir:
            reward_extra_infos_dict["finish_reason"] = sample_finish_reasons
            reward_extra_infos_dict["tool_call_counts"] = sample_tool_counts
            records = evaluation_records(
                data_sources,
                sample_uids,
                reward_extra_infos_dict,
                expected_acc_counts,
                sample_indices=sample_indices,
                prompt_metadata=prompt_metadata,
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                ground_truths=sample_gts,
                response_lengths=sample_response_lengths,
                turns=sample_turns,
            )
            record_keys = set().union(*(record.keys() for record in records)) if records else set()
            trainer._dump_generations(
                inputs=[r["input"] for r in records],
                outputs=[r["output"] for r in records],
                gts=[r["gts"] for r in records],
                scores=[r["score"] for r in records],
                reward_extra_infos_dict={
                    k: [r.get(k) for r in records] for k in record_keys - {"input", "output", "gts", "score"}
                },
                dump_path=os.path.join(val_data_dir, "records"),
            )
        reward_extra_infos_dict["__response_length__"] = sample_response_lengths
        reward_extra_infos_dict["__num_turns__"] = sample_turns
        reward_extra_infos_dict["__finish_reason__"] = sample_finish_reasons
        reward_extra_infos_dict["__tool_calls__"] = sample_tool_counts
        val_metrics = trainer._val_metrics_update(
            data_sources,
            sample_uids,
            reward_extra_infos_dict,
            sample_turns,
            expected_acc_counts=expected_acc_counts,
            sample_indices=sample_indices,
            source_fields=source_fields,
            prompt_ids=[
                prompt_metadata[(str(s), str(u))]["prompt_id"] for s, u in zip(data_sources, sample_uids, strict=True)
            ],
        )
        val_metrics.update(aggregate_validation_windows(pipeline_windows))
        return val_metrics
