#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
# Author: Liwei, Tang Shiwen
# Email: liwei157@baidu.com, tangshiwen@baidu.com
#
import os
from typing import Optional

import torch
from vllm.model_executor.kernels.linear import (
    CutlassInt8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)
from vllm.platforms import current_platform

_FORCE_TORCH_INT8_LINEAR = os.environ.get(
    "VLLM_KUNLUN_FORCE_TORCH_INT8_LINEAR", "0"
) == "1"


class KunlunScaledMMLinearKernel(CutlassInt8ScaledMMLinearKernel):

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_out_of_tree():
            return False, "requires OOT platform."
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

        w_q_name, w_s_name, i_s_name, i_zp_name, azp_adj_name = self.layer_param_names

        # change scale to max for klx ops
        with torch.no_grad():
            getattr(layer, w_s_name).mul_(127.0)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        w_q, w_s, x_s, x_zp, azp_adj = self._get_layer_params(layer)
        symmetric = azp_adj is None

        if _FORCE_TORCH_INT8_LINEAR:
            _ws = w_s.float().flatten() / 127.0
            w_float = w_q.float() * _ws.unsqueeze(0)
            out = x.float().matmul(w_float).to(x.dtype)
            if bias is not None:
                out = out + bias.to(x.dtype)
            return out

        x_q, x_s, x_zp, static = torch.ops._C.scaled_int8_quant(
            x=x.contiguous(),
            scale=x_s,
            azp=x_zp,
            symmetric=symmetric,
        )

        bias_arg = bias.to(torch.float32).contiguous() if bias is not None else None

        if x_zp is not None:
            azp = None if static else x_zp
            return torch.ops._C.cutlass_scaled_mm_azp(
                a=x_q,
                b=w_q,
                scale_a=x_s,
                scale_b=(w_s / 127.0).transpose(0, 1),
                out_dtype=x.dtype,
                azp_adj=azp_adj,
                azp=azp,
                bias=bias_arg,
            )

        return torch.ops._C.matmul(
            x=x_q,
            w=w_q.contiguous(),
            out_dtype=x.dtype,
            w_trans=False,
            x_pc_max=x_s * 127.0 if static else x_s,
            w_pc_max=w_s,
            bias=bias_arg,
        )

    def apply_prequantized_weights(
        self,
        layer: torch.nn.Module,
        x_q: torch.Tensor,
        x_max: torch.Tensor,
        out_dtype: torch.dtype,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply a dynamic W8A8 linear to an already quantized activation.

        ``x_q`` and ``x_max`` use the same contract as Kunlun ``quant2d``:
        INT8 ``[M, K]`` data and FP32 per-token absolute maxima.  This entry
        point intentionally supports only the dynamic symmetric scheme used by
        MiniMax-M3; static or asymmetric activation quantization needs its own
        zero-point/scale contract and must continue through ``apply_weights``.
        """
        w_q, w_s, x_s, x_zp, azp_adj = self._get_layer_params(layer)
        if (
            self.config.is_static_input_scheme
            or not self.config.input_symmetric
            or x_s is not None
            or x_zp is not None
            or azp_adj is not None
        ):
            raise ValueError(
                "prequantized Kunlun W8A8 input requires dynamic symmetric "
                "activation quantization"
            )
        if x_q.dtype != torch.int8 or x_q.dim() != 2:
            raise ValueError(
                f"x_q must be a 2D INT8 tensor, got {x_q.dtype} {tuple(x_q.shape)}"
            )
        if x_max.dtype != torch.float32 or x_max.numel() != x_q.shape[0]:
            raise ValueError(
                "x_max must contain one FP32 absolute maximum per input row"
            )

        x_q = x_q.contiguous()
        x_max = x_max.reshape(-1, 1).contiguous()
        bias_arg = bias.to(torch.float32).contiguous() if bias is not None else None

        if _FORCE_TORCH_INT8_LINEAR:
            x_float = x_q.float() * (x_max / 127.0)
            w_float = w_q.float() * (w_s.float().flatten() / 127.0).unsqueeze(0)
            out = x_float.matmul(w_float).to(out_dtype)
            if bias is not None:
                out = out + bias.to(out_dtype)
            return out

        return torch.ops._C.matmul(
            x=x_q,
            w=w_q.contiguous(),
            out_dtype=out_dtype,
            w_trans=False,
            x_pc_max=x_max,
            w_pc_max=w_s,
            bias=bias_arg,
        )
