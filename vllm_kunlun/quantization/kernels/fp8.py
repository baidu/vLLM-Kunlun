import torch
import torch.nn.functional as F

from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFP8ScaledMMLinearKernel,
    CutlassFp8BlockScaledMMKernel,
)

from vllm_kunlun.ops.fp8 import dequantize_fp8_blocks


_FP8_BLOCK_SIZE = 128


class _KunlunFp8BlockDequantFallback:
    def apply_weights(self, layer, x, bias=None, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = x.data
        params = self._get_layer_params(layer)
        weight = params.weight
        weight_scale = params.weight_scale_inv
        if weight_scale is None:
            weight_scale = params.weight_scale
        assert weight_scale is not None

        # Converge onto the single block-scaled dequant seam
        # (vllm_kunlun.ops.fp8.dequantize_fp8_blocks): it prefers the fused
        # on-device XPU kernel (fp8->bf16 decode + per-block scale in one
        # launch, no PCIe round-trip) and only degrades to the CPU cast when
        # that op is unavailable / the weight is not on-device. Same [N, K]
        # block semantics as the previous inline CPU path; this removes the
        # per-forward device->host->device round-trip that dominated profiles.
        weight_bf16 = dequantize_fp8_blocks(weight, weight_scale).to(x.device)
        if bias is not None:
            bias = bias.to(torch.bfloat16)
        return F.linear(x.to(torch.bfloat16), weight_bf16, bias)


class KunlunFP8ScaledMMLinearKernel(CutlassFP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(cls, compute_capability=None):
        return True, None


class KunlunFp8BlockScaledMMKernel(
    _KunlunFp8BlockDequantFallback, CutlassFp8BlockScaledMMKernel
):
    @classmethod
    def is_supported(cls, compute_capability=None):
        return True, None

    def process_weights_after_loading(self, layer):
        Fp8BlockScaledMMLinearKernel.process_weights_after_loading(self, layer)
