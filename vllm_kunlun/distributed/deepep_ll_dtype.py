"""Route DeepEP low-latency dispatch/combine through bfloat16.

Kunlun's low-latency kernels only accept bfloat16 and require the combine
output to share the payload dtype

    deep_ep.cpp:207 'x.dim() == 2 and x.is_contiguous()
                     and x.scalar_type() == torch::kBFloat16'
    deep_ep.cpp:305 'out->scalar_type() == x.scalar_type()'

so a model whose activations are not bfloat16 cannot reach them. vLLM does not
cast on this path -- the reference kernels take the payload as it comes -- and
the expert stage has its own dtype requirements, so the casts belong on the
communication calls only.

Two properties of the low-latency API decide where the wrappers go:

1. vLLM drives dispatch and combine with ``return_recv_hook=True``, which
   returns while the kernel is still reading its input; only the hook proves
   arrival. Kunlun records tensor lifetimes for the ``async_finish`` path alone
   (``buffer_v2.py:233``), and upstream is safe because every tensor it passes
   is owned by the caller. A tensor created inside a wrapper has no other owner,
   so it must be held until the hook has run -- otherwise the next allocator
   trim unmaps it underneath a running kernel.

2. ``finalize`` discards the return value of the shared ``_finalize`` helper,
   and Kunlun always returns a callable hook rather than None
   (``deep_ep.cpp:311``), so work deferred into a hook is silently dropped on
   the synchronous path. ``finalize`` and ``finalize_async`` are therefore
   wrapped separately instead of patching ``_finalize``.
"""

import logging

import torch

logger = logging.getLogger("vllm_kunlun")

_FLAG = "_kunlun_ll_dtype_patched"


def applied(mod) -> bool:
    cls = getattr(mod, "DeepEPLLPrepareAndFinalize", None)
    return cls is None or getattr(cls, _FLAG, False)


def _as_bf16(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    return x if x.is_contiguous() else x.contiguous()


def _keepalive_hook(hook, keep):
    """Run the arrival hook, then release the tensors it was reading."""

    def run():
        hook()
        keep.clear()

    return run


def _keepalive_hook_copy(hook, keep, output, staging):
    """Arrival hook that also copies the staged payload into the caller's out."""

    def run():
        hook()
        output.copy_(staging)
        keep.clear()

    return run


def apply(mod) -> None:
    cls = getattr(mod, "DeepEPLLPrepareAndFinalize", None)
    if cls is None or getattr(cls, _FLAG, False):
        return

    orig_prepare_async = cls.prepare_async
    orig_finalize = cls.finalize
    orig_finalize_async = cls.finalize_async

    def prepare_async(self, a1, *args, **kwargs):
        casted = _as_bf16(a1)
        hook, receiver = orig_prepare_async(self, casted, *args, **kwargs)
        if casted is a1:
            return hook, receiver
        return _keepalive_hook(hook, [casted]), receiver

    def finalize(self, output, fused_expert_output, *args, **kwargs):
        x = _as_bf16(fused_expert_output)
        if output.dtype == torch.bfloat16:
            return orig_finalize(self, output, x, *args, **kwargs)
        # This path passes do_recv_hook=False, so combine has guaranteed
        # arrival on the current stream by the time it returns.
        staging = torch.empty_like(output, dtype=torch.bfloat16)
        result = orig_finalize(self, staging, x, *args, **kwargs)
        output.copy_(staging)
        return result

    def finalize_async(self, output, fused_expert_output, *args, **kwargs):
        x = _as_bf16(fused_expert_output)
        if output.dtype == torch.bfloat16:
            hook, done = orig_finalize_async(self, output, x, *args, **kwargs)
            return _keepalive_hook(hook, [x]), done
        staging = torch.empty_like(output, dtype=torch.bfloat16)
        hook, done = orig_finalize_async(self, staging, x, *args, **kwargs)
        return _keepalive_hook_copy(hook, [x, staging], output, staging), done

    cls.prepare_async = prepare_async
    cls.finalize = finalize
    cls.finalize_async = finalize_async
    setattr(cls, _FLAG, True)
    logger.info(
        "[KunlunPlugin] DeepEPLLPrepareAndFinalize dispatches and combines in "
        "bfloat16 (payloads held until the arrival hook runs)"
    )
