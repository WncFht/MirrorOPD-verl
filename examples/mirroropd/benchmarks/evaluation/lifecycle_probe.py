# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Native source/history acceptance with real scoring and a local publication fault.

Use this explicit trainer.evaluation._target_ only in a fresh acceptance job.
The first before-sync event is deliberately refused once at publication. The
probe verifies pending retention, then retries publication without rescoring.
It does not simulate an unfinished GPU RPC or change the scoring protocol.
"""

import json

from verl.trainer.evaluation.artifacts import file_sha256
from verl.trainer.evaluation.source_history import SourceHistoryReadout


class SourceLifecycleProbe(SourceHistoryReadout):
    """Assert the existing lifecycle at actual GPU training boundaries."""

    def _record(self, **facts):
        with self.root.with_suffix(".lifecycle.jsonl").open("a") as stream:
            stream.write(json.dumps(facts, sort_keys=True) + "\n")

    def on_before_weight_sync(self, trainer, batch, outer):
        result = super().on_before_weight_sync(trainer, batch, outer)
        event, _ = self.pending_events[outer]
        assert event.manifest()["status"] == "active" and outer not in self.scored
        before = file_sha256(event.manifest_path)
        assert self.validate(trainer) == {}
        assert file_sha256(event.manifest_path) == before
        self._record(outer=outer, pending_before_sync=True, duplicate_validation_unchanged=True)
        return result

    def on_readout(self, trainer, outer):
        if outer in self.pending_events:
            assert trainer.rollout_service.resident_version == outer
            event, _ = self.pending_events[outer]
            if outer == 1:
                finish = event.finish

                def refuse_publication(*args, **kwargs):
                    raise OSError("Injected acceptance publication failure before manifest write")

                event.finish = refuse_publication
                try:
                    try:
                        super().on_readout(trainer, outer)
                    except OSError as error:
                        assert "Injected acceptance" in str(error)
                    else:
                        raise AssertionError("Publication fault did not propagate")
                finally:
                    event.finish = finish
                assert outer in self.pending_events and outer not in self.scored
                assert event.manifest()["status"] == "active"
                self._record(outer=outer, publication_fault_propagated=True, pending_retained=True)
            result = super().on_readout(trainer, outer)
            assert outer not in self.pending_events and outer in self.scored
            assert event.manifest(complete=True)["status"] == "complete"
            before = file_sha256(event.manifest_path)
            assert self.validate(trainer) == {}
            assert file_sha256(event.manifest_path) == before
            self._record(outer=outer, published_after_sync=True, resident_version=outer, no_rescore_on_retry=True)
            return result
        return super().on_readout(trainer, outer)
