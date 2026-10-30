"""Kunlun hotfix: guard Scheduler._update_from_kv_xfer_finished.

Upstream vLLM (0.28.0) asserts ``req_id in self.requests`` while consuming
finished_recving / finished_sending. When a client batch-abort and a late
KV-transfer-finished event land in the same scheduler step, the req has
already been removed from ``self.requests`` and the assert kills EngineCore
(EngineDeadError). We filter out such stale req_ids before delegating to the
original method so the asserts never fire. Normal (present) req_ids are
untouched; stale transfers are reclaimed by the Mooncake timeout GC.
"""

import logging

from vllm.v1.core.sched.scheduler import Scheduler

_logger = logging.getLogger("vllm_kunlun.kvxfer_guard")

_orig = Scheduler._update_from_kv_xfer_finished

if not getattr(_orig, "_kunlun_kvxfer_guard", False):

    def _safe_update_from_kv_xfer_finished(self, kv_connector_output):
        for attr in ("finished_recving", "finished_sending"):
            ids = getattr(kv_connector_output, attr, None)
            if ids:
                stale = [r for r in ids if r not in self.requests]
                if stale:
                    _logger.warning(
                        "[kvxfer_guard] dropping %d stale %s req_ids: %s",
                        len(stale),
                        attr,
                        stale,
                    )
                    setattr(
                        kv_connector_output,
                        attr,
                        type(ids)(r for r in ids if r in self.requests),
                    )
        return _orig(self, kv_connector_output)

    _safe_update_from_kv_xfer_finished._kunlun_kvxfer_guard = True
    Scheduler._update_from_kv_xfer_finished = _safe_update_from_kv_xfer_finished
    _logger.warning("[kvxfer_guard] patched Scheduler._update_from_kv_xfer_finished")
