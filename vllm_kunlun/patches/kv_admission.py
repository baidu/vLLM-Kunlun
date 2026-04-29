"""
KV Cache Admission Control Patch
=================================
Backports two scheduling improvements to vLLM 0.11 (P800):

1. Full-sequence KV admission gate (from vLLM 0.19 scheduler_reserve_full_isl)
   Before admitting a new request from the waiting queue, check whether the
   *full* sequence (prompt + max output tokens) fits in free KV blocks.  If
   not, block admission so the request stays in the waiting queue until there
   is genuine capacity.  This prevents KV cache from filling to ~99%,
   preemption loops, and the associated throughput collapse.

2. Partial-prefill concurrency limit (from vLLM 0.19 max_num_partial_prefills)
   Limit the number of requests that can simultaneously be in chunked-prefill
   state.  When the limit is reached, no new waiting requests are admitted
   until one of the running prefills completes.

Configuration:
    VLLM_MAX_PARTIAL_PREFILLS  (env var, default=1)
        Maximum number of requests that may be in partial (chunked) prefill
        state at the same time.

Detection heuristic for Gate 1:
    The waiting→running path always passes `new_computed_blocks` (a
    KVCacheBlocks instance, possibly empty) as a positional argument.
    The decode path for already-running requests calls allocate_slots with
    `new_computed_blocks=None` (the default).  We use this to distinguish the
    two cases without modifying the scheduler.
"""

import os

_MAX_PARTIAL_PREFILLS = int(os.environ.get("VLLM_MAX_PARTIAL_PREFILLS", "1"))

# Shared mutable state between the two patches (single-threaded engine loop).
# block_new_prefill: set to True by the scheduler patch when too many running
# requests are already in prefill; read by allocate_slots to reject admission.
_state = {"block_new_prefill": False}


def apply():
    """Patch KVCacheManager.allocate_slots with the full-sequence admission
    gate and the partial-prefill concurrency limit gate."""
    import vllm.v1.core.kv_cache_manager as _kv_mod

    _original_allocate_slots = _kv_mod.KVCacheManager.allocate_slots

    def _patched_allocate_slots(
        self,
        request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks=None,  # KVCacheBlocks | None
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ):
        # Both gates apply only on the waiting→running admission path
        # (signalled by new_computed_blocks being non-None).
        if new_computed_blocks is not None:
            # Gate 1: Partial-prefill concurrency limit.
            # Reject admission when the scheduler has flagged that we already
            # have _MAX_PARTIAL_PREFILLS requests in chunked-prefill state.
            if _state["block_new_prefill"]:
                return None

            # Gate 2: Full-sequence KV admission gate.
            # Reject admission if the entire sequence (prompt + max output)
            # would not fit in the currently free KV blocks.
            full_num_tokens = min(
                request.num_prompt_tokens + request.max_tokens,
                self.max_model_len,
            )
            num_blocks_needed = self.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=full_num_tokens,
                new_computed_blocks=new_computed_blocks.blocks,
                num_encoder_tokens=0,
            )
            if num_blocks_needed > self.block_pool.get_num_free_blocks():
                # Not enough space for the full sequence – keep waiting.
                return None

        return _original_allocate_slots(
            self,
            request,
            num_new_tokens,
            num_new_computed_tokens,
            new_computed_blocks,
            num_lookahead_tokens,
            delay_cache_blocks,
            num_encoder_tokens,
        )

    _kv_mod.KVCacheManager.allocate_slots = _patched_allocate_slots


def apply_scheduler():
    """Patch Scheduler.schedule to enforce the partial-prefill concurrency
    limit by updating _state['block_new_prefill'] before each scheduling step.

    A request is considered to be in 'partial prefill' if
        num_computed_tokens < num_prompt_tokens
    i.e. it has not yet finished processing its prompt tokens.

    If the count of such requests meets or exceeds _MAX_PARTIAL_PREFILLS,
    _state['block_new_prefill'] is set to True, causing allocate_slots to
    reject any new admission from the waiting queue for this step.
    """
    import vllm.v1.core.sched.scheduler as _sched_mod

    _original_schedule = _sched_mod.Scheduler.schedule

    def _patched_schedule(self):
        num_partial = sum(
            1 for r in self.running
            if r.num_computed_tokens < r.num_prompt_tokens
        )
        _state["block_new_prefill"] = (num_partial >= _MAX_PARTIAL_PREFILLS)
        return _original_schedule(self)

    _sched_mod.Scheduler.schedule = _patched_schedule
