"""Expert-id mapping and token preprocessing for Kunlun MoE."""

from typing import Optional

import torch

from .workspace import MoeMetadata

MOE_PREPROCESS_THRESHOLD = 768

_EXPERT_MAP_CACHE: dict[tuple[torch.device, int, int, int], torch.Tensor] = {}
_EXPERT_MAP_RETIRED: list[torch.Tensor] = []


def contiguous_expert_map(
    device: torch.device, global_experts: int, local_experts: int, ep_rank: int
) -> torch.Tensor:
    key = (device, global_experts, local_experts, ep_rank)
    result = _EXPERT_MAP_CACHE.get(key)
    if result is None:
        start = ep_rank * local_experts
        result = torch.full((global_experts,), -1, dtype=torch.int32, device=device)
        end = min(start + local_experts, global_experts)
        if start < end:
            result[start:end] = torch.arange(
                end - start, dtype=torch.int32, device=device
            )
        _EXPERT_MAP_CACHE[key] = result
    return result


def map_expert_ids(topk_ids: torch.Tensor, expert_map: torch.Tensor) -> torch.Tensor:
    if expert_map.device != topk_ids.device:
        raise ValueError("expert_map and router outputs must be on the same device")
    return expert_map[topk_ids.to(torch.int64)].to(torch.int32)


def prepare_moe_tokens(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    workspace: torch.Tensor,
    metadata: MoeMetadata,
    block_statistic: Optional[torch.Tensor] = None,
    index_have_neg: bool = False,
):
    """Group rows by the expert IDs passed here, retaining fixed-shape `-1` rows.

    For sorted input, ``block_statistic`` is only reusable when it was produced
    from these same IDs and this local ``num_experts`` value. Mapped EP callers
    therefore leave it unset and let this function generate the local histogram
    exactly once; small input never needs a histogram.
    """
    num_rows = topk_ids.numel()
    if num_rows <= MOE_PREPROCESS_THRESHOLD:
        return torch.ops.xspeedgate_ops.moe_pre_small(
            topk_ids,
            num_experts,
            index_have_neg=index_have_neg,
            sort_mode=True,
            x=hidden_states,
        )

    hidden_size = hidden_states.shape[1]
    moe_expand = workspace[: num_rows * hidden_size].view(num_rows, hidden_size)
    if block_statistic is None:
        block_statistic = metadata.block_statistic
        torch.ops._C.gen_block_statistic(topk_ids, block_statistic)
    torch.ops._C.moe_pre_sorted(
        x=hidden_states,
        topk_index=topk_ids,
        block_statistic=block_statistic,
        moe_expand=moe_expand,
        moe_index=metadata.sorted_tokens_idx,
        expert_m=metadata.expert_m,
        sorted_tokens_num_lod=metadata.sorted_tokens_num_lod,
        index_have_neg=index_have_neg,
    )
    return metadata.sorted_tokens_idx, metadata.sorted_tokens_num_lod, moe_expand
