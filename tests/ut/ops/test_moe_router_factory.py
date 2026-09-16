"""`make_moe_router` on its own: the selection table, the layer adapter, the cache.

`test_route_matrix.py` only reaches the factory through `route()`, so it proves
the chosen kernel computes the right thing but never that a config picks the
kernel it was meant to. This file checks the choice and the `*_after` flags
directly, which is what makes the table in `MoeRouter`'s docstring testable.
No kernel runs here, so it needs no device.
"""

from types import SimpleNamespace

import pytest

from vllm_kunlun.ops.moe.router import make_moe_router, make_moe_router_for_layer

# config kwargs -> (kernel, n_group, topk_group, emits_histogram,
#                   needs_histogram, renorm_after, scale_after)
CASES = [
    ("plain softmax", dict(top_k=4), ("softmax_topk", 0, 0, True, False, False, False)),
    (
        "plain softmax, scaled",
        dict(top_k=4, routed_scaling_factor=2.5),
        ("softmax_topk", 0, 0, True, False, False, True),
    ),
    (
        "plain softmax, no renorm",
        dict(top_k=4, renormalize=False),
        ("softmax_topk", 0, 0, True, False, False, False),
    ),
    # Ungrouped sigmoid has no kernel of its own: one group of every expert is
    # how it gets served, and the renorm the softmax kernel fuses becomes ours.
    (
        "plain sigmoid -> one group",
        dict(top_k=4, scoring_func="sigmoid"),
        ("group_topk", 1, 1, True, True, True, False),
    ),
    (
        "plain sigmoid, no renorm, scaled",
        dict(
            top_k=4,
            scoring_func="sigmoid",
            renormalize=False,
            routed_scaling_factor=2.5,
        ),
        ("group_topk", 1, 1, True, True, False, True),
    ),
    (
        "grouped softmax",
        dict(top_k=4, use_grouped_topk=True, num_expert_group=8, topk_group=4),
        ("group_topk", 8, 4, True, True, True, False),
    ),
    # Upstream falls back to plain top-k unless a group count exceeds 1, so the
    # same config has to pick the same kernel here.
    (
        "grouped(1,1) softmax -> plain",
        dict(top_k=4, use_grouped_topk=True, num_expert_group=1, topk_group=1),
        ("softmax_topk", 0, 0, True, False, False, False),
    ),
    (
        "grouped but no group counts -> plain",
        dict(top_k=4, use_grouped_topk=True),
        ("softmax_topk", 0, 0, True, False, False, False),
    ),
    (
        "bias, ungrouped sigmoid",
        dict(top_k=4, scoring_func="sigmoid", has_e_score_correction_bias=True),
        ("fused_gate", 0, 0, False, False, False, False),
    ),
    # The kernel fuses the scale, so nothing is left for the tail even at 2.5.
    (
        "bias, ungrouped sqrtsoftplus, scaled",
        dict(
            top_k=4,
            scoring_func="sqrtsoftplus",
            has_e_score_correction_bias=True,
            routed_scaling_factor=2.5,
        ),
        ("fused_gate", 0, 0, False, False, False, False),
    ),
    (
        "bias, grouped sigmoid, scaled",
        dict(
            top_k=4,
            scoring_func="sigmoid",
            use_grouped_topk=True,
            num_expert_group=8,
            topk_group=4,
            has_e_score_correction_bias=True,
            routed_scaling_factor=2.5,
        ),
        ("sigmoid_group", 8, 4, True, True, False, False),
    ),
    # Arbitrary Python does its own everything; only the histogram is ours, and
    # any scaling belongs to the function, so the factor is dropped.
    (
        "custom routing wins over everything",
        dict(
            top_k=4,
            scoring_func="sigmoid",
            use_grouped_topk=True,
            num_expert_group=8,
            topk_group=4,
            has_e_score_correction_bias=True,
            routed_scaling_factor=2.5,
            has_custom_routing=True,
        ),
        ("custom", 0, 0, False, False, False, False),
    ),
]


@pytest.mark.parametrize("name, kwargs, want", CASES, ids=[c[0] for c in CASES])
def test_selection_table(name, kwargs, want):
    r = make_moe_router(**kwargs)
    got = (
        r.kernel,
        r.n_group,
        r.topk_group,
        r.emits_histogram,
        r.needs_histogram,
        r.renorm_after,
        r.scale_after,
    )
    assert got == want


def test_custom_routing_drops_routed_scaling_factor():
    r = make_moe_router(top_k=4, routed_scaling_factor=2.5, has_custom_routing=True)
    assert r.routed_scaling_factor == 1.0


def test_lru_cache_returns_the_same_object_for_one_config():
    make_moe_router.cache_clear()
    before = make_moe_router.cache_info()
    a = make_moe_router(
        top_k=6,
        scoring_func="sigmoid",
        num_expert_group=8,
        topk_group=4,
        use_grouped_topk=True,
    )
    b = make_moe_router(
        top_k=6,
        scoring_func="sigmoid",
        num_expert_group=8,
        topk_group=4,
        use_grouped_topk=True,
    )
    after = make_moe_router.cache_info()
    assert a is b
    assert after.hits == before.hits + 1


def stub(**overrides):
    """A RoutedExperts stand-in: only the attributes the adapter reads."""
    attrs = dict(
        top_k=8,
        scoring_func="softmax",
        renormalize=True,
        use_grouped_topk=False,
        num_expert_group=None,
        topk_group=None,
        e_score_correction_bias=None,
        routed_scaling_factor=1.0,
        custom_routing_function=None,
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


# Only presence matters for the bias and the function, so anything truthy will
# do -- and a tensor would be the wrong thing to hand a cache key anyway.
FOR_LAYER_CASES = [
    ("defaults", stub(), "softmax_topk"),
    (
        "bias present",
        stub(scoring_func="sigmoid", e_score_correction_bias=object()),
        "fused_gate",
    ),
    (
        "grouped",
        stub(use_grouped_topk=True, num_expert_group=8, topk_group=4),
        "group_topk",
    ),
    ("custom", stub(custom_routing_function=lambda **kw: None), "custom"),
]


@pytest.mark.parametrize(
    "name, layer, want_kernel",
    FOR_LAYER_CASES,
    ids=[c[0] for c in FOR_LAYER_CASES],
)
def test_make_moe_router_for_layer_reads_the_layer(name, layer, want_kernel):
    r = make_moe_router_for_layer(layer)
    assert r.kernel == want_kernel
    assert r.top_k == layer.top_k


def test_make_moe_router_for_layer_lands_on_the_shared_cache():
    # Two layers of the same model share a config, so the adapter has to land on
    # the cache too rather than rebuilding per layer.
    assert make_moe_router_for_layer(stub()) is make_moe_router_for_layer(stub())
