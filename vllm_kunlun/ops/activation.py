import logging

import torch
from vllm.model_executor.layers.activation import SiluAndMul as _upstream_cls

logger = logging.getLogger("vllm_kunlun")

def _forward_native(self, x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(out, x)
    return out


# Idempotent monkey-patch: safe under fork() and re-import.
if not getattr(_upstream_cls, "_kunlun_silu_and_mul_patched", False):
    _upstream_cls.forward_native = _forward_native
    _upstream_cls._kunlun_silu_and_mul_patched = True
    logger.info(
        "[KunlunPlugin] SiluAndMul patched in vllm_kunlun/ops/activations.py"
    )