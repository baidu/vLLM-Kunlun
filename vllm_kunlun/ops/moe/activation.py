#
# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
#
# This file is a part of the vllm-kunlun project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Which Kunlun kernel runs a MoE layer's `f(gate) * up`, and running it.

The four pieces here are one contract and only mean anything together: the two
frozensets say which activation names Kunlun can serve and which of those
allocate their own output, `apply_gated_activation` dispatches on the name, and
`uninterleave_moe_w13` is what produces one of those names, by permuting W13 at
load time. Keeping them in one file is the point -- "reorder the weight" and
"read the reordered weight" are two halves of the same contract.

Unlike `moe_router`, which had to leave `_kunlun_ops` because three callers
shared it, this is a topical split: the only caller outside `KunlunOps.fused_moe`
is a layer's `process_weights_after_loading`, which imports `_kunlun_ops`
anyway. Nothing here registers an out-of-tree op, so importing it has no side
effects.
"""

from typing import Optional

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import kunlun_ops
except ImportError as e:
    logger.warning("Import error msg: %s", e.msg)

# vLLM MoEActivation values with a Kunlun kernel for `f(gate) * up`. Anything
# else, including exact-erf `gelu` and `relu2`, has no kernel and is rejected;
# see apply_gated_activation.
SUPPORTED_MOE_ACTIVATIONS = frozenset(
    {
        "silu",
        "swish",
        "gelu_tanh",
        "swigluoai",
        "swigluoai_uninterleave",
        "swiglustep",
    }
)

# Activations whose implementation returns a freshly allocated [.., inter]
# tensor instead of filling an out param. The workspace must not reserve a
# slice for those, or the reservation is paid for and never used.
MOE_ACTIVATIONS_ALLOCATING_OUTPUT = frozenset({"swigluoai"})

# swigluoai's shape parameters, spelled the way kunlun_ops.swiglu_bias wants:
#     out = clamp(gate, max=limit) * sigmoid(alpha * gate)
#           * (clamp(up, -limit, limit) + beta)
# These are upstream SwigluOAIAndMul's defaults (beta=1 is the `up + 1` term),
# used whenever the layer does not carry explicit swiglu_alpha/beta/limit.
_SWIGLUOAI_ALPHA = 1.702
_SWIGLUOAI_BETA = 1.0
_SWIGLUOAI_LIMIT = 7.0


def apply_gated_activation(
    activation: str,
    gate_up_output: torch.Tensor,
    out: Optional[torch.Tensor],
    swiglu_alpha: Optional[float] = None,
    swiglu_beta: Optional[float] = None,
    swiglu_limit: Optional[float] = None,
) -> torch.Tensor:
    """Apply `f(gate) * up` to the [.., 2 * inter] W13 output.

    Returns the tensor holding the result: `out` for the branches that take an
    out param, or a freshly allocated tensor for the ones in
    `MOE_ACTIVATIONS_ALLOCATING_OUTPUT`, which is why `out` is optional.

    The gate/up split is not the same for every branch, and nothing in the
    signature hints at it -- it is a property of each kernel, pinned down by
    test_fused_moe.py's layout probe:

        silu / gelu_tanh / swiglustep  gate = x[.., :inter], up = x[.., inter:]
        swigluoai_uninterleave (idem)  gate = x[.., :inter], up = x[.., inter:]
        swigluoai                      gate = x[.., 0::2],   up = x[.., 1::2]

    So W13's output rows have to be ordered to match the activation, and
    switching the activation of an already-loaded checkpoint is not a no-op.
    That is why `swigluoai` and `swigluoai_uninterleave` are two names for one
    formula: upstream's `swigluoai` is what gpt-oss checkpoints hold, and
    `swigluoai_uninterleave` is the same math over the packed layout. Kunlun has
    a kernel only for the packed one, so `uninterleave_moe_w13` permutes W13 at
    load time and renames the activation. (`kunlun_ops.swiglu`'s `turn` argument
    looks like it would select between the two layouts; measured to make no
    difference.)

    The interleaved `swigluoai` branch is the slow fallback for callers that did
    not go through the load-time permutation: `_C::swigluoai_and_mul` is
    registered in vllm_kunlun/ops/_custom_ops.py as plain PyTorch, so it is ~8
    elementwise launches over stride-2 views plus an allocation, costing a flat
    ~270 us no matter how few rows it gets. `kunlun_ops.swiglu_bias` does the
    same work in one kernel: 28x faster at 8-768 rows, 3.1x at 4096, 1.4x at
    32768 (bf16, inter=704).

    Activations with no implementation at all are rejected rather than emulated
    -- notably exact-erf `gelu` and `relu2`, for which neither `_C` nor
    `kunlun_ops` provides a gated variant.
    """
    alpha = _SWIGLUOAI_ALPHA if swiglu_alpha is None else swiglu_alpha
    beta = _SWIGLUOAI_BETA if swiglu_beta is None else swiglu_beta
    limit = _SWIGLUOAI_LIMIT if swiglu_limit is None else swiglu_limit
    if activation == "swigluoai":
        if beta != _SWIGLUOAI_BETA:
            # The op hardcodes the `up + 1` term. Silently ignoring beta is the
            # bug class this signature exists to avoid, so refuse instead.
            raise NotImplementedError(
                f"swigluoai only implements beta=1.0, got {beta}; use "
                "swigluoai_uninterleave for other values"
            )
        return torch.ops._C.swigluoai_and_mul(gate_up_output, alpha=alpha, limit=limit)
    flat_in = gate_up_output.view(-1, gate_up_output.shape[-1])
    flat_out = out.view(-1, out.shape[-1])
    if activation in ("silu", "swish"):
        torch.ops._C.silu_and_mul(out, gate_up_output)
    elif activation == "gelu_tanh":
        torch.ops._C.gelu_tanh_and_mul(flat_out, flat_in)
    elif activation == "swiglustep":
        torch.ops._C.swiglustep(flat_out, flat_in, limit)
    elif activation == "swigluoai_uninterleave":
        # Third positional arg is an optional bias added to the input; unused
        # here because W13's bias is already folded in by the caller. The
        # docstring only documents `limit` as a cap on the gate half, but it
        # clamps the linear half to [-limit, limit] too, matching upstream
        # SiluAndMulWithClamp (measured: agreeing to 5.5e-3 relative in bf16,
        # against 5.3e+2 for a reference that leaves the linear half raw).
        kunlun_ops.swiglu_bias(flat_in, flat_out, None, alpha, beta, limit)
    else:
        raise ValueError(f"Unsupported gated MoE activation: {activation}")
    return out


def uninterleave_moe_w13(layer: torch.nn.Module, activation: str) -> str:
    """Reorder W13's rows so `swigluoai` can use a Kunlun kernel.

    Returns the activation name `fused_moe` should be called with from then on.
    Only `swigluoai` is touched; every other activation is returned unchanged,
    because it already reads the layout its checkpoint holds.

    `swigluoai` is the odd one out: gpt-oss checkpoints interleave W13's rows as
    `[gate0, up0, gate1, up1, ...]`, and the only Kunlun kernel for that formula
    (`kunlun_ops.swiglu_bias`) reads the packed `[gate0..gateN, up0..upN]`.
    Without the permutation the activation falls back to a pure-PyTorch op that
    costs a flat ~270 us per call; with it, 28x less at 8-768 rows, 3.1x at
    4096, 1.4x at 32768 (bf16, inter=704). Reordering the weight once at load
    time is what makes that trade, so it is done here rather than per forward
    pass, and the renamed activation is what tells `apply_gated_activation`
    which layout to expect. Upstream carries the same pair of names for the same
    reason (MiniMax-M3 ships the packed layout as `swigluoai_uninterleave`).
    """
    if activation != "swigluoai":
        return activation
    w13 = layer.w13_weight.data
    layer.w13_weight = torch.nn.Parameter(
        torch.cat((w13[:, 0::2, :], w13[:, 1::2, :]), dim=1),
        requires_grad=False,
    )
    w13_bias = getattr(layer, "w13_bias", None)
    if w13_bias is not None:
        bias = w13_bias.data
        layer.w13_bias = torch.nn.Parameter(
            torch.cat((bias[:, 0::2], bias[:, 1::2]), dim=1),
            requires_grad=False,
        )
    return "swigluoai_uninterleave"


__all__ = [name for name in globals() if not name.startswith("_")]
