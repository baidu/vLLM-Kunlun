# SPDX-License-Identifier: Apache-2.0
"""Keep mamba/GDN groups out of EAGLE's "match one extra block, then pop" dance.

Root cause of the MTP + prefix-caching accuracy problem on hybrid Mamba/GDN
models, and of the fact that the naive fix helps one model while breaking
another.

``HybridKVCacheCoordinator.find_longest_cache_hit``
(vllm/v1/core/kv_cache_coordinator.py:547-566) hands each group one extra block
of search budget when that group is an "eagle group"::

    _max_length = curr_hit_length
    if use_eagle:
        # Eagle needs to match one more block and then pop the last.
        _max_length = min(curr_hit_length + spec.block_size, max_cache_hit_length)

``FullAttentionManager`` then pops the extra block internally, so the net hit
length is right. ``MambaManager`` *cannot* pop: its blocks are
``[null, ..., state]``, so popping removes the state block itself. The obvious
workaround -- have ``MambaManager.find_longest_cache_hit`` search one block less
-- only cancels the extension when the extension actually happened. When
``min(...)`` clips against ``max_cache_hit_length`` the coordinator handed out
*less* than a full block, so subtracting a whole block over-corrects and the
group resumes with its recurrent state a full block behind.

How often the clip fires depends on block_size vs the hit-length distribution,
which is why a fixed ``-1`` lands on opposite sides for different models:
Qwen3.5-35B-A3B (block 640, needed the subtraction) vs Qwen3.6-27B (block 512,
broke because of it). Both directions show up as degenerate repetition /
truncation / accuracy loss, only under MTP + prefix caching together.

Fix the premise instead of the correction: mamba groups never take part in the
extend-and-pop protocol. ``eagle_group_ids`` is where they get enrolled --
kv_cache_coordinator.py:58-64 flags *every* group when no group is explicitly
marked::

    self.eagle_group_ids: set[int] = {
        i for i, g in enumerate(kv_cache_config.kv_cache_groups) if g.is_eagle_group
    }
    # Conservatively fall back to flag all groups when no group is flagged.
    if use_eagle and not self.eagle_group_ids:
        self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))

Dropping mamba groups from that set means the coordinator no longer extends
``_max_length`` for them and no longer expects a pop, so there is nothing to
cancel and no model-dependent correction. Full-attention groups are untouched.

Side benefit: the mamba group stops forfeiting one block of hit length, which
removes the "prompts shorter than two blocks never hit at all" dead zone.

Off switch: ``VLLM_KUNLUN_MAMBA_EAGLE_GROUP_FIX=0``.
"""

import os

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.kv_cache_interface import MambaSpec

logger = init_logger(__name__)

_orig_coordinator_init = KVCacheCoordinator.__init__


def _enabled() -> bool:
    return os.getenv("VLLM_KUNLUN_MAMBA_EAGLE_GROUP_FIX", "1") == "1"


def _mamba_group_ids(kv_cache_config) -> set[int]:
    return {
        i
        for i, g in enumerate(kv_cache_config.kv_cache_groups)
        if isinstance(g.kv_cache_spec, MambaSpec)
    }


def _strip_mamba_eagle_groups(coordinator, kv_cache_config) -> set[int]:
    """Drop mamba groups from ``eagle_group_ids``; returns what was dropped."""
    eagle_ids = getattr(coordinator, "eagle_group_ids", None)
    if not eagle_ids:
        return set()
    dropped = set(eagle_ids) & _mamba_group_ids(kv_cache_config)
    if dropped:
        coordinator.eagle_group_ids = set(eagle_ids) - dropped
    return dropped


def _patched_init(self, *args, **kwargs) -> None:
    _orig_coordinator_init(self, *args, **kwargs)
    if not _enabled():
        return
    kv_cache_config = kwargs.get("kv_cache_config") or (args[0] if args else None)
    if kv_cache_config is None:
        return
    dropped = _strip_mamba_eagle_groups(self, kv_cache_config)
    if dropped:
        logger.info(
            "[KunlunPlugin] mamba groups %s excluded from EAGLE last-block drop; "
            "eagle_group_ids=%s",
            sorted(dropped),
            sorted(self.eagle_group_ids),
        )


KVCacheCoordinator.__init__ = _patched_init
