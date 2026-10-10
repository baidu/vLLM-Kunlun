"""Keep vLLM's CUDA-style device selection when data parallelism is enabled.

``vllm/v1/engine/utils.py`` decides whether each DP rank needs its own
``CUDA_VISIBLE_DEVICES``-style isolation with

    needs_device_env_isolation = not (
        current_platform.is_cuda_alike() or current_platform.is_xpu())

``KunlunPlatform._enum`` is ``PlatformEnum.OOT``, so both checks are False and
vLLM shards the device list per DP rank, setting
``parallel_config.assigned_physical_gpu_ids`` to a single id (e.g. ``[7]`` for
DP rank 7).

But ``GpuWorker.init_device`` takes the CUDA path (``device_type == "cuda"``)
and offsets the worker's local rank by ``dp_local_rank * tp_pp_world_size``,
then asserts ``local_rank < len(assigned_physical_gpu_ids)``. With DP8/TP1 that
is ``7 < 1`` and startup dies with

    AssertionError: local_rank 7 is out of bounds for
                    assigned_physical_gpu_ids [7]

On Kunlun every process sees all eight devices and the worker selects its own by
index, exactly like CUDA, so the per-rank sharding is both unnecessary and
wrong. Neutralise it and let ``assigned_physical_gpu_ids`` stay None, which is
what the CUDA path expects.
"""

import logging

logger = logging.getLogger("vllm_kunlun")


def applied(mod) -> bool:
    fn = getattr(mod, "set_assigned_physical_gpu_ids_for_dp_rank", None)
    return fn is None or getattr(fn, "_kunlun_dp_noop", False)


def apply(mod) -> None:
    name = "set_assigned_physical_gpu_ids_for_dp_rank"
    fn = getattr(mod, name, None)
    if fn is None or getattr(fn, "_kunlun_dp_noop", False):
        return

    def set_assigned_physical_gpu_ids_for_dp_rank(
        vllm_config, local_dp_rank, user_assigned_gpu_ids=None
    ):
        # Only honour an explicit --device-ids list; never shard per DP rank.
        vllm_config.parallel_config.assigned_physical_gpu_ids = user_assigned_gpu_ids

    set_assigned_physical_gpu_ids_for_dp_rank._kunlun_dp_noop = True
    setattr(mod, name, set_assigned_physical_gpu_ids_for_dp_rank)
    logger.info(
        "[KunlunPlugin] DP device sharding disabled; workers select devices by "
        "index like CUDA"
    )
