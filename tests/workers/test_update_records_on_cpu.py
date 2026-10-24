# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch

from verl.utils import tensordict_utils as tu
from verl.utils.metric.observations import Observation
from verl.utils.metric.utils import Metric
from verl.workers.update_records import UpdateResult, execution_consensus, finalize_update, merge_rank_metrics


@pytest.mark.parametrize("applied", [True, False, None])
def test_execution_fact_is_not_a_metric(applied):
    output = tu.get_tensordict({}, {"metrics": {"loss": 1.0}, "optimizer_step_applied": applied})
    result = UpdateResult.from_worker_output(output, 2)
    assert result.applied is applied and result.metrics == {"loss": 1.0}
    assert execution_consensus([applied, applied], 2) is applied


@pytest.mark.parametrize("statuses", [[True, None], [None, False], [True, False], [1, 1], [True]])
def test_rank_execution_disagreement_cannot_be_averaged(statuses):
    with pytest.raises(RuntimeError):
        execution_consensus(statuses, 2)


def test_legacy_loss_contributions_and_raw_observations_keep_distinct_reductions():
    parts = [
        {"loss_contribution": Metric("sum", 2.0), "raw": Observation.mean(torch.tensor([1.0]))},
        {"loss_contribution": Metric("sum", 4.0), "raw": Observation.mean(torch.tensor([3.0, 5.0, 7.0]))},
    ]
    result = UpdateResult(True, merge_rank_metrics(parts))
    outer, record = finalize_update(
        result, role="actor", collect_records=False, collect_metrics=False, update_index=0, minibatches_per_epoch=1
    )
    assert outer["loss_contribution"] == 3.0
    assert outer["raw"] == 4.0
    assert record is None


def test_record_keeps_unknown_fact_and_undefined_observation():
    result = UpdateResult(None, {"distillation/full_kl": Observation.ratio(0.0, 0.0), "loss": float("nan")})
    outer, record = finalize_update(
        result, role="actor", collect_records=True, collect_metrics=True, update_index=3, minibatches_per_epoch=2
    )
    assert not outer
    assert record["applied"] is None
    assert record["metrics"]["optim/actor/execution_known"] == 0
    assert "optim/actor/update_applied" not in record["metrics"]
    assert record["metrics"]["distill/full_kl"] is None
    assert math.isnan(record["metrics"]["optim/actor/loss"])
    assert record["metrics"]["rollout/data_reuse/epoch_index"] == 1
