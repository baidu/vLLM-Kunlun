"""Route vLLM's DeepEP all2all managers to Kunlun's ``BufferV2``.

Kunlun's ``deep_ep`` build ships two buffer classes:

* ``Buffer`` -- a reduced API (no ``get_dispatch_layout``, and ``dispatch``
  without ``handle`` / ``num_tokens_per_rank`` / ``is_token_in_rank``) that
  vLLM's ``prepare_finalize/deepep_*.py`` cannot drive.
* ``BufferV2`` -- keeps the upstream DeepEP API, ignores the arguments its
  runtime does not need and remembers ``num_combined_tokens`` for combine.

vLLM imports ``deep_ep`` lazily inside the all2all managers, so patching the
module object after import is not enough; install a wrapper module in
``sys.modules`` whose ``Buffer`` is a ``BufferV2`` subclass.

The subclass also handles two constructor mismatches:

* it drops kwargs this build does not accept (upstream passes
  ``explicitly_destroy``, ``allow_nvlink_for_low_latency_mode``, ...);
* it fills in ``num_experts``. vLLM never passes it, so the buffer would be
  sized for ``BufferV2``'s default of 256 while dispatch runs with the model's
  real expert count, and the notify step then fails with
  ``bkcl_notify_dispatch_standard failed / recv_num_tokens >= 0``.
"""

import inspect
import logging
import sys
from types import SimpleNamespace

logger = logging.getLogger("vllm_kunlun")


def _infer_num_experts():
    """Routed expert count for the running model, or None if unknown."""
    try:
        from vllm.config import get_current_vllm_config

        text_config = get_current_vllm_config().model_config.hf_text_config
    except Exception:
        return None
    for attr in ("num_experts", "n_routed_experts", "num_local_experts"):
        n = getattr(text_config, attr, None)
        if isinstance(n, int) and n > 0:
            return n
    return None


def _make_buffer_cls(real):
    accepted = set(inspect.signature(real.BufferV2.__init__).parameters) - {"self"}

    class KunlunDeepEPBuffer(real.BufferV2):
        """BufferV2 with the argument handling vLLM expects."""

        def __init__(self, *args, **kwargs):
            dropped = [k for k in kwargs if k not in accepted]
            for k in dropped:
                kwargs.pop(k)
            if dropped:
                logger.debug(
                    "[KunlunPlugin] deep_ep BufferV2 ignores kwargs %s", dropped
                )
            if "num_experts" not in kwargs:
                n = _infer_num_experts()
                if n is not None:
                    kwargs["num_experts"] = n
                    logger.info(
                        "[KunlunPlugin] deep_ep buffer num_experts=%d "
                        "(inferred from model config)",
                        n,
                    )
            super().__init__(*args, **kwargs)

        def destroy(self):
            """Upstream passes explicitly_destroy=True and calls this later."""
            parent = getattr(super(), "destroy", None)
            if parent is not None:
                parent()

    return KunlunDeepEPBuffer


def applied(mod) -> bool:
    real = sys.modules.get("deep_ep")
    return real is not None and getattr(real, "_kunlun_buffer_v2", False)


def apply(mod) -> None:
    try:
        import deep_ep
    except Exception as exc:  # deep_ep is an optional dependency
        logger.info(
            "[KunlunPlugin] deep_ep not available (%s); DeepEP backends disabled",
            str(exc).splitlines()[0],
        )
        return
    if getattr(deep_ep, "_kunlun_buffer_v2", False):
        return
    if not hasattr(deep_ep, "BufferV2"):
        logger.warning(
            "[KunlunPlugin] deep_ep has no BufferV2; leaving Buffer as is"
        )
        return

    wrapped = SimpleNamespace(**deep_ep.__dict__)
    wrapped.Buffer = _make_buffer_cls(deep_ep)
    wrapped._kunlun_buffer_v2 = True
    sys.modules["deep_ep"] = wrapped
    logger.info(
        "[KunlunPlugin] deep_ep.Buffer -> BufferV2 (version %s)",
        getattr(getattr(deep_ep, "version", None), "__version__", "unknown"),
    )
