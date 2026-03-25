"""layer.py"""

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.library import custom_op
from vllm.attention import Attention as VllmAttention
from vllm.attention import AttentionType
from vllm.attention.layer import MultiHeadAttention as VllmMultiHeadAttention
from vllm.config import CacheConfig
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    is_v1_kv_transfer_group,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.platforms import _Backend


class Attention(VllmAttention):
    """Attention"""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        logits_soft_cap: Optional[float] = None,
        per_layer_sliding_window: Optional[int] = None,
        use_mla: bool = False,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        **extra_impl_args,
    ) -> None:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.
        """
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            cache_config=cache_config,
            quant_config=quant_config,
            logits_soft_cap=logits_soft_cap,
            per_layer_sliding_window=per_layer_sliding_window,
            use_mla=use_mla,
            prefix=prefix,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            **extra_impl_args,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: Optional[torch.Size] = None,
    ) -> torch.Tensor:
        """forward"""
        if self.calculate_kv_scales:
            attn_metadata = get_forward_context().attn_metadata
            if attn_metadata.enable_kv_scales_calculation:
                self.calc_kv_scales(query, key, value)
        if self.use_output:
            output_shape = output_shape if output_shape is not None else query.shape
            output = torch.zeros(output_shape, dtype=query.dtype, device=query.device)
            hidden_size = output_shape[-1]
            # We skip reshaping query, key and value tensors for the MLA
            # backend since these tensors have different semantics and are
            # processed differently.
            if not self.use_mla:
                # Reshape the query, key, and value tensors.
                # NOTE(woosuk): We do this outside the custom op to minimize the
                # CPU overheads from the non-CUDA-graph regions.
                query = query.view(-1, self.num_heads, self.head_size)
                output = output.view(-1, self.num_heads, self.head_size)
                if key is not None:
                    key = key.view(-1, self.num_kv_heads, self.head_size)
                if value is not None:
                    value = value.view(-1, self.num_kv_heads, self.head_size)
            if self.use_direct_call:
                forward_context: ForwardContext = get_forward_context()
                attn_metadata = forward_context.attn_metadata
                if isinstance(attn_metadata, dict):
                    attn_metadata = attn_metadata[self.layer_name]
                self_kv_cache = self.kv_cache[forward_context.virtual_engine]
                self.impl.forward(
                    self, query, key, value, self_kv_cache, attn_metadata, output=output
                )
            else:
                torch.ops.vllm.unified_attention_with_output_kunlun(
                    query, key, value, output, self.layer_name
                )
            return output.view(-1, hidden_size)
        else:
            if self.use_direct_call:
                forward_context = get_forward_context()
                attn_metadata = forward_context.attn_metadata
                if isinstance(attn_metadata, dict):
                    attn_metadata = attn_metadata[self.layer_name]
                self_kv_cache = self.kv_cache[forward_context.virtual_engine]
                return self.impl.forward(
                    self, query, key, value, self_kv_cache, attn_metadata
                )
            else:
                return unified_attention(query, key, value, self.layer_name)


# 重写自 vllm.attention.layer 中的 MultiHeadAttention 类
class MultiHeadAttention(VllmMultiHeadAttention):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
    ):
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
        )
        # kunlun只支持flash_attn
        self.attn_backend = _Backend.FLASH_ATTN

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Input shape: batch_size x seq_len x hidden_size"""
        # TODO(Isotr0py): Use existing backend implementations and support FA3
        bsz, q_len, _ = query.size()
        kv_len = key.size(1)

        query = query.view(bsz, q_len, self.num_heads, self.head_size)
        key = key.view(bsz, kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz, kv_len, self.num_kv_heads, self.head_size)

        if (num_repeat := self.num_queries_per_kv) > 1:
            # Handle MQA and GQA
            key = torch.repeat_interleave(key, num_repeat, dim=2)
            value = torch.repeat_interleave(value, num_repeat, dim=2)

        # kunlun只支持flash_attn
        if self.attn_backend == _Backend.FLASH_ATTN:
            from flash_attn import flash_attn_func

            out = flash_attn_func(query, key, value, softmax_scale=self.scale)
        elif self.attn_backend == _Backend.XFORMERS:
            from xformers import ops as xops

            out = xops.memory_efficient_attention_forward(
                query, key, value, scale=self.scale
            )
        elif self.attn_backend == _Backend.TORCH_SDPA:
            query, key, value = (x.transpose(1, 2) for x in (query, key, value))
            out = F.scaled_dot_product_attention(query, key, value, scale=self.scale)
            out = out.transpose(1, 2)
        elif self.attn_backend == _Backend.PALLAS_VLLM_V1:
            query, key, value = (x.transpose(1, 2) for x in (query, key, value))
            from torch_xla.experimental.custom_kernel import flash_attention

            out = flash_attention(query, key, value, sm_scale=self.scale)
            out = out.transpose(1, 2)

        return out.reshape(bsz, q_len, -1)


def wait_for_kv_layer_from_connector(layer_name: str):
    """wait_for_kv_layer_from_connector"""
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    assert isinstance(attn_metadata, dict)
    connector.wait_for_layer_load(layer_name)


def maybe_save_kv_layer_to_connector(
    layer_name: str, kv_cache_layer: List[torch.Tensor]
):
    """maybe_save_kv_layer_to_connector"""
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    assert isinstance(attn_metadata, dict)
    connector.save_kv_layer(layer_name, kv_cache_layer, attn_metadata[layer_name])


@custom_op("vllm::unified_attention_with_output_kunlun", mutates_args=())
def unified_attention_with_output_kunlun(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: Optional[torch.Tensor] = None,
) -> None:
    wait_for_kv_layer_from_connector(layer_name)
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]
    self.impl.forward(self, query, key, value, kv_cache, attn_metadata, output=output)

    maybe_save_kv_layer_to_connector(layer_name, kv_cache)


def _fake_unified_attention_with_output_kunlun(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: Optional[torch.Tensor] = None,
) -> None:
    return None


unified_attention_with_output_kunlun.register_fake(
    _fake_unified_attention_with_output_kunlun
)


def unified_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """unified_attention"""
    wait_for_kv_layer_from_connector(layer_name)

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]
    output = self.impl.forward(self, query, key, value, kv_cache, attn_metadata)

    maybe_save_kv_layer_to_connector(layer_name, kv_cache)
    return output
