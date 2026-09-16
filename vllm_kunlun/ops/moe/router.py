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

"""Which Kunlun kernel runs a MoE layer's routing stage, and running it.

Kept apart from `_kunlun_ops` so that the three MoE methods that need a router
-- unquantized (`ops/fused_moe/layer.py`), int8 and int4
(`quantization/compressed_tensors/compressed_tensors_moe.py`) -- and
`KunlunOps.fused_moe` itself can all import it without depending on each other.
Nothing here registers an out-of-tree op, so importing it has no side effects.

`make_moe_router` chooses and validates, `run_moe_router` executes; both are
described on their own. The split exists so the choice can be made once at load
time and the rejection of a config Kunlun cannot serve happens there too.
"""

import functools
from typing import Callable, NamedTuple, Optional

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import kunlun_ops
except ImportError as e:
    logger.warning("Import error msg: %s", e.msg)

# Rows of scratch space the block-statistic kernels expect, one per XPU cluster.
MOE_BLOCK_STATISTIC_ROWS = 12


def _moe_scores(router_logits: torch.Tensor, scoring_func: str) -> torch.Tensor:
    """Turn router logits into the routing probabilities top-k selects from."""
    if scoring_func == "softmax":
        # Takes bf16/fp16/fp32 and always returns fp32; measured, its output is
        # bit-identical whether or not the logits were widened first, so there
        # is nothing to gain by casting here.
        return kunlun_ops.softmax(router_logits)
    if scoring_func == "sigmoid":
        # Unlike the kernel above, torch.sigmoid keeps the input dtype, and
        # half-precision scores round distinct logits together: measured, that
        # changes which experts moe_group_topk selects (fp16 and bf16 both, from
        # M=1 up). sigmoid is monotonic, so the fp32 ordering is the faithful
        # one -- hence the widen. Only this branch pays it.
        return torch.sigmoid(router_logits.to(torch.float32))
    # sqrtsoftplus is only reachable through moe_fused_gate_dsv4, which scores
    # internally, so there is no kernel to feed here.
    raise NotImplementedError(
        f"Kunlun MoE cannot score {scoring_func!r} separately from selection."
    )


class MoeRouter(NamedTuple):
    """A chosen routing kernel plus the fixups that kernel does not fuse in.

    Built by `make_moe_router` from a layer's MoE config and then reused for
    every forward pass: nothing in here depends on the batch, so the choice and
    its validation are paid once instead of per step. Calling it runs the
    routing stage and returns `(weights, expert ids, histogram)`.

    The four kernels and what each leaves for us to do:

        kernel         scores       selects   renorm   routed scale   histogram
        softmax_topk   softmax      top-k     fused    separate       optional
        group_topk     `scoring`    grouped   separate separate       required
        sigmoid_group  sigmoid+bias grouped   fused    fused          required
        fused_gate     `scoring`+b  top-k     fused    fused          none

    "required" means the kernel takes the histogram buffer as a mandatory
    argument, so it is allocated even when the caller does not want the result;
    "none" means `gen_block_statistic` has to fill it afterwards. The histogram
    is the per-expert row count `_prepare_moe_tokens` needs, and every kernel
    that emits one was measured bit-identical to `gen_block_statistic`, so it is
    threaded back out rather than recomputed.
    """

    kernel: str
    top_k: int
    scoring_func: str
    renormalize: bool
    n_group: int
    topk_group: int
    routed_scaling_factor: float
    emits_histogram: bool
    needs_histogram: bool
    renorm_after: bool
    scale_after: bool


@functools.lru_cache(maxsize=None)
def make_moe_router(
    top_k: int,
    scoring_func: str = "softmax",
    renormalize: bool = True,
    use_grouped_topk: bool = False,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    has_e_score_correction_bias: bool = False,
    routed_scaling_factor: float = 1.0,
    has_custom_routing: bool = False,
) -> MoeRouter:
    """Pick the routing kernel for a MoE config, rejecting the ones we lack.

    Every argument is fixed once a layer is built, which is the point: call this
    from `process_weights_after_loading` and a config Kunlun cannot serve fails
    at load time instead of somewhere inside the first forward pass. The result
    is cached, so the later per-forward calls are a dict lookup and the
    validation below runs once per distinct config.

    Only booleans are taken for the bias and the custom function -- the kernel
    choice depends on their presence, not their value, and keeping tensors and
    callables out of the cache key keeps them out of the cache's references too.

    Selection order mirrors upstream's `create_fused_moe_router`: custom routing
    first, then a selection bias, then grouped top-k, then plain top-k. Same
    config in, same kernel out.
    """
    if has_custom_routing:
        # Arbitrary Python, so nothing to choose and nothing to validate; it is
        # also the only router that reads hidden_states (gemma4 folds a
        # per-expert scale through it). Upstream's CustomRoutingRouter is
        # likewise never given routed_scaling_factor, so any scaling is the
        # function's own, and it emits no histogram.
        return MoeRouter(
            kernel="custom",
            top_k=top_k,
            scoring_func=scoring_func,
            renormalize=renormalize,
            n_group=0,
            topk_group=0,
            routed_scaling_factor=1.0,
            emits_histogram=False,
            needs_histogram=False,
            renorm_after=False,
            scale_after=False,
        )

    # Upstream only groups when at least one group count exceeds 1, and falls
    # back to plain top-k otherwise, so the same config has to pick the same
    # kernel here.
    grouped = (
        use_grouped_topk
        and num_expert_group is not None
        and topk_group is not None
        and (num_expert_group > 1 or topk_group > 1)
    )
    scale_fused = routed_scaling_factor == 1.0

    if has_e_score_correction_bias and grouped:
        if scoring_func != "sigmoid":
            raise NotImplementedError(
                "Kunlun MoE has no grouped top-k kernel with a selection bias "
                f"for scoring_func={scoring_func!r}."
            )
        if not renormalize:
            raise NotImplementedError(
                "Kunlun's grouped sigmoid routing kernel always renormalizes; "
                "renormalize=False would need the raw group top-k scores."
            )
        return MoeRouter(
            kernel="sigmoid_group",
            top_k=top_k,
            scoring_func=scoring_func,
            renormalize=renormalize,
            n_group=num_expert_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            emits_histogram=True,
            needs_histogram=True,
            renorm_after=False,
            scale_after=False,
        )

    if has_e_score_correction_bias:
        if scoring_func not in ("sigmoid", "sqrtsoftplus"):
            raise NotImplementedError(
                "Kunlun MoE has no ungrouped top-k kernel with a selection "
                f"bias for scoring_func={scoring_func!r}."
            )
        return MoeRouter(
            kernel="fused_gate",
            top_k=top_k,
            scoring_func=scoring_func,
            renormalize=renormalize,
            n_group=0,
            topk_group=0,
            routed_scaling_factor=routed_scaling_factor,
            emits_histogram=False,
            needs_histogram=False,
            renorm_after=False,
            scale_after=False,
        )

    if grouped or scoring_func != "softmax":
        # A single group of everything is how ungrouped sigmoid routing gets
        # served, since moe_group_topk is the only top-k kernel that scores
        # separately from selection. `_moe_scores` rejects the scoring
        # functions it has no kernel for, but do it here so a bad config
        # cannot reach the first forward pass.
        if scoring_func not in ("softmax", "sigmoid"):
            raise NotImplementedError(
                f"Kunlun MoE cannot score {scoring_func!r} separately from "
                "selection, which grouped top-k requires."
            )
        return MoeRouter(
            kernel="group_topk",
            top_k=top_k,
            scoring_func=scoring_func,
            renormalize=renormalize,
            n_group=num_expert_group if grouped else 1,
            topk_group=topk_group if grouped else 1,
            routed_scaling_factor=routed_scaling_factor,
            emits_histogram=True,
            needs_histogram=True,
            renorm_after=renormalize,
            scale_after=not scale_fused,
        )

    return MoeRouter(
        kernel="softmax_topk",
        top_k=top_k,
        scoring_func=scoring_func,
        renormalize=renormalize,
        n_group=0,
        topk_group=0,
        routed_scaling_factor=routed_scaling_factor,
        emits_histogram=True,
        needs_histogram=False,
        renorm_after=False,
        scale_after=not scale_fused,
    )


def make_moe_router_for_layer(layer: torch.nn.Module) -> MoeRouter:
    """`make_moe_router` fed from a FusedMoE layer's attributes.

    The three quant methods all read the same attributes off the layer, and
    their `apply`/`apply_monolithic` are handed the same values as arguments, so
    building from the layer at load time and from the arguments at call time
    agree by construction.
    """
    return make_moe_router(
        top_k=layer.top_k,
        scoring_func=layer.scoring_func,
        renormalize=layer.renormalize,
        use_grouped_topk=layer.use_grouped_topk,
        num_expert_group=layer.num_expert_group,
        topk_group=layer.topk_group,
        has_e_score_correction_bias=layer.e_score_correction_bias is not None,
        routed_scaling_factor=layer.routed_scaling_factor,
        has_custom_routing=layer.custom_routing_function is not None,
    )


def run_moe_router(
    router: MoeRouter,
    router_logits: torch.Tensor,
    hidden_states: Optional[torch.Tensor] = None,
    e_score_correction_bias: Optional[torch.Tensor] = None,
    custom_routing_function: Optional[Callable] = None,
    want_block_statistic: bool = False,
    block_statistic: Optional[torch.Tensor] = None,
):
    """Run the kernel `router` picked, returning (weights, ids, histogram).

    `want_block_statistic` is honoured whatever the kernel does: the ones that
    emit no histogram -- including a custom routing function, which is
    arbitrary Python -- get one from `gen_block_statistic`, measured
    bit-identical to what the others emit. The kernels that had to be handed a
    buffer anyway still return None when it was not asked for, so a caller
    cannot read a result it did not request.

    `router_logits` is taken in the model's own dtype. Measured, every kernel
    agrees bit-for-bit with an fp32 copy of the logits except `fused_gate`,
    which requires fp32 outright, and sigmoid scoring for `group_topk`, where
    half-precision rounding changes the selection; both widen where they need
    it (see the `fused_gate` branch and `_moe_scores`). An unconditional cast
    here would instead be a kernel launch per MoE layer per step (measured ~2.6
    us under graph replay, ~22 us eager, shape-independent) on every path.
    """
    top_k = router.top_k
    num_tokens, num_experts = router_logits.shape
    device = router_logits.device

    # Watch out for the softmax kernels' expert-count ceiling: measured, they
    # accept up to 640+ experts with block_statistic=None but fail at 481 with
    # `optimized_ops::moe_softmax_topk_norm_fusion failed` once a histogram is
    # passed (docs say n <= 512, and n <= 480 with block_statistic). No model
    # sits in 481..512 today, so there is no fallback here; if one appears, drop
    # the histogram for those shapes and let `_prepare_moe_tokens` recompute it
    # with `gen_block_statistic`, which has no such limit.
    needs_block_statistic = want_block_statistic or router.needs_histogram
    if needs_block_statistic:
        if block_statistic is None:
            block_statistic = torch.empty(
                (MOE_BLOCK_STATISTIC_ROWS, num_experts),
                dtype=torch.int32,
                device=device,
            )
    else:
        block_statistic = None

    if router.kernel == "custom":
        # Arbitrary Python, so it does its own scoring, selection,
        # renormalisation and scaling; only the histogram is left to us.
        topk_weights, topk_ids = custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=router.renormalize,
        )
        topk_weights = topk_weights.to(torch.float32)
        topk_ids = topk_ids.to(torch.int32)
    else:
        topk_weights = torch.empty(
            num_tokens, top_k, dtype=torch.float32, device=device
        )
        topk_ids = torch.empty(num_tokens, top_k, dtype=torch.int32, device=device)

        if router.kernel == "softmax_topk":
            # Measured: `_norm` rescales the top-k weights to sum to 1, the
            # plain kernel returns raw softmax probabilities.
            softmax_topk = (
                torch.ops._C.moe_softmax_topk_norm
                if router.renormalize
                else torch.ops._C.moe_softmax_topk
            )
            softmax_topk(
                x=router_logits,
                normed_score=topk_weights,
                topk_index=topk_ids,
                block_statistic=block_statistic,
            )
        elif router.kernel == "group_topk":
            # Ungrouped routing asks for one group holding every expert rather
            # than the documented (0, 0) "degenerates to plain top-k" mode:
            # measured, that mode silently drops higher-scoring experts once
            # num_experts > 256 (wrong for every seed at 264..512, hard error
            # above), while a single group is exact at every size it accepts.
            #
            # This kernel caps out at 512 experts and raises
            # `optimized_ops::moe_group_topk_fusion failed ret = 1` above that.
            # `kunlun_ops.moe_group_topk_fusion_global` takes the same
            # arguments and goes to 2048, so it is the drop-in for a wider
            # model -- but it is not wired up here because no such model exists
            # yet and its results above 512 have never been checked. It is
            # also uniformly slower where both run (measured, top_k=8:
            # +12..+70% at M=8/4096, +140..+460% at M=512), so it belongs
            # behind a `num_experts > 512` fallback rather than replacing this
            # call.
            kunlun_ops.moe_group_topk(
                _moe_scores(router_logits, router.scoring_func),
                router.n_group,
                router.topk_group,
                topk_weights,
                topk_ids,
                block_statistic,
            )
        elif router.kernel == "sigmoid_group":
            torch.ops._C.moe_sigmoid_group_topk_norm(
                x=router_logits,
                topk_index=topk_ids,
                norm_score=topk_weights,
                block_static=block_statistic,
                bias=e_score_correction_bias,
                scale=router.routed_scaling_factor,
                n_group=router.n_group,
                topk_group=router.topk_group,
            )
        else:
            # The only kernel here that will not take the gate's own dtype:
            # measured, moe_fused_gate_dsv4 requires fp32 input. The other
            # three accept bf16/fp16/fp32 and were measured bit-identical
            # either way, so the cast stays in this branch rather than sitting
            # on the common path where it costs a launch per layer per step.
            #
            # Measured: routed_scaling_factor only takes effect when the
            # trailing flag is set, and it lands on the returned weights either
            # way -- the flag gates the scale rather than moving it to the layer
            # output.
            kunlun_ops.moe_fused_gate_dsv4(
                router_logits.to(torch.float32),
                e_score_correction_bias,
                topk_weights,
                topk_ids,
                top_k,
                router.scoring_func,
                0,
                router.renormalize,
                router.routed_scaling_factor,
                router.routed_scaling_factor != 1.0,
            )

    # Whatever the routing stage did not fuse in, as elementwise passes. The
    # custom function has every `_after` flag off: it renormalises and scales
    # itself, and only the histogram is missing.
    if router.renorm_after:
        topk_weights.div_(topk_weights.sum(-1, keepdim=True))
    if router.scale_after:
        topk_weights.mul_(router.routed_scaling_factor)
    if not router.emits_histogram and block_statistic is not None:
        torch.ops._C.gen_block_statistic(topk_ids, block_statistic)
    if not want_block_statistic:
        block_statistic = None
    return topk_weights, topk_ids, block_statistic


__all__ = [name for name in globals() if not name.startswith("_")]
