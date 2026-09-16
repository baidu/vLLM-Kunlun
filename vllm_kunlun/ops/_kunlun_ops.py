#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
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

"""kunlun custom op entry"""

import cocopod  # noqa
import torch
import xspeedgate_ops  # noqa
from vllm.logger import init_logger

from vllm_kunlun.ops.moe import entry as _moe_entry

logger = init_logger(__name__)

try:
    import kunlun_ops

    logger.info("Load custom ops library success!")
except ImportError as e:
    logger.warning("Import error msg: %s", e.msg)


_per_token_smooth_quant = True


def is_per_token_smooth_quant():
    """is per token smooth quant"""
    return _per_token_smooth_quant


class KunlunOps:
    """KunlunOps"""

    # Attention ops
    @staticmethod
    def paged_attention_v1(
        output,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        context_lens,
        context_lens_cpu,
        is_context,
        block_size,
        max_context_len,
        alibi_slopes,
        kv_cache_dtype,
        k_scale,
        v_scale,
        tp_rank,
        blocksparse_local_blocks,
        blocksparse_vert_stride,
        blocksparse_block_size,
        blocksparse_head_sliding_step,
        alibi_sqrt=False,
    ):
        """PagedAttentionV1"""
        # block_size = value_cache.shape[2]
        kunlun_ops.paged_attention(
            x=query,
            k_cache=key_cache,
            v_cache=value_cache,
            block_tables=block_tables,
            context_lens_cpu=context_lens_cpu,
            context_lens_xpu=context_lens,
            is_context=is_context,
            is_causal=True,
            out=output,
            vo_head_dim=128,
        )

    @staticmethod
    def paged_attention_v2(
        output,
        exp_sums,
        max_logits,
        tmp_output,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        context_lens,
        context_lens_cpu,
        is_context,
        block_size,
        max_context_len,
        alibi_slopes,
        kv_cache_dtype,
        k_scale,
        v_scale,
        tp_rank,
        blocksparse_local_blocks,
        blocksparse_vert_stride,
        blocksparse_block_size,
        blocksparse_head_sliding_step,
        alibi_sqrt=False,
    ):
        """PagedAttentionV2"""
        # block_size = value_cache.shape[2]
        kunlun_ops.paged_attention(
            x=query,
            k_cache=key_cache,
            v_cache=value_cache,
            block_tables=block_tables,
            context_lens_cpu=context_lens_cpu,
            context_lens_xpu=context_lens,
            is_context=is_context,
            is_causal=True,
            out=output,
            vo_head_dim=128,
        )

    # Activation ops
    @staticmethod
    def silu_and_mul(out: torch.Tensor, x: torch.Tensor):
        """silu and mul"""
        kunlun_ops.silu_and_mul(
            x,
            axis=-1,
            turn=True,
            out=out,
        )

    # Activation ops
    @staticmethod
    def quick_gelu(out: torch.Tensor, x: torch.Tensor):
        """quick gelu"""
        kunlun_ops.quick_gelu(
            x,
            out=out,
        )

    # Layernorm
    @staticmethod
    def rms_norm(
        out,
        x,
        weight,
        epsilon,
    ):
        """rms_norm"""
        kunlun_ops.rmsnorm(x, weight.to(torch.float32), epsilon, out=out)

    @staticmethod
    def fused_add_rms_norm(
        x,
        residual,
        weight,
        epsilon,
    ):
        """fused_add_rms_norm"""
        output = torch.empty_like(x)
        kunlun_ops.add_rmsnorm(
            x, residual, weight.to(torch.float32), epsilon, out=output
        )
        fused_input = x + residual
        residual.copy_(fused_input, non_blocking=True)
        x.copy_(output)

    # Rotary embedding
    @staticmethod
    def rotary_embedding(
        positions, query, key, head_size, cos_sin_cache, is_neox_style
    ):
        """
        refactor RotaryEmbedding forward function
        """
        query_x = query.contiguous()
        key_x = key.contiguous()

        torch.ops._C.rotary_embedding(
            positions, query_x, key_x, head_size, cos_sin_cache, is_neox_style
        )

        return query_x, key_x

    # Rotary embedding
    @staticmethod
    def mrotary_embedding(
        positions, mrope_section, query, key, head_size, cos_sin_cache, is_neox_style
    ):
        """
        refactor RotaryEmbedding forward function
        """
        query_x = query.contiguous()
        key_x = key.contiguous()
        assert is_neox_style
        kunlun_ops.mrotary_embedding_neox(
            positions, query_x, key_x, head_size, cos_sin_cache, mrope_section
        )

        query.data = query_x
        key.data = key_x
        return query, key

    @staticmethod
    def swap_blocks(src, dst, block_mapping):
        """swap_blocks"""
        kunlun_ops.swap_blocks(src, dst, block_mapping)

    @staticmethod
    def copy_blocks(key_caches, value_caches, block_mapping):
        """copy_blocks"""
        for i in range(len(key_caches)):
            key_caches[i] = key_caches[i].contiguous()
            value_caches[i] = value_caches[i].contiguous()
        kunlun_ops.copy_blocks(
            key_caches,
            value_caches,
            block_mapping,
        )

    @staticmethod
    def reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
    ):
        """reshape_and_cache"""
        # slot_mapping_cast = slot_mapping.to(torch.int32)
        kunlun_ops.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

    @staticmethod
    def multi_query_kv_attention(
        usual_seq_lod_xpu: torch.Tensor,
        usual_seq_lod_cpu: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kargs,
    ) -> torch.Tensor:
        """
        query: shape = [num_prompt_tokens, num_heads, head_size]
        """
        if query.dim() == 3:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
        output = torch.empty_like(query)

        B, T, Qh, Hd = query.shape
        KVh = key.size(2)
        if KVh != Qh:
            repeat = Qh // KVh
            key = key.repeat_interleave(repeat, dim=2)  # [B, T, Qh, Hd]
            value = value.repeat_interleave(repeat, dim=2)
        kunlun_ops.attention(
            q=query,
            k_cache=key,
            v_cache=value,
            out=output,
            is_causal=True,
            is_prefill=True,
            context_seq_lod_cpu=usual_seq_lod_cpu,
            context_seq_lod_xpu=usual_seq_lod_xpu,
        )
        return output

    @staticmethod
    def quant_fusedresidual_rmsnorm_op(
        x, residual, weight, bias, scale_to_int, eps, dyn_scale: bool, type: int = 1
    ):
        """Quantized fused residual layer normalization"""
        out = torch.empty_like(x, dtype=torch.int8)

        if is_per_token_smooth_quant():
            out_scale = torch.empty(
                x.shape[:-1], device=x.device, dtype=torch.float
            ).unsqueeze(-1)
        else:
            out_scale = torch.empty(12, device=x.device, dtype=torch.float)

        kunlun_ops.quant_fusedresidual_rmsnorm(
            x,
            residual,
            weight,
            bias,
            eps,
            out=out,
            out_scale=out_scale,
            residual_tensor=residual,
        )

        if residual is None:
            return out, out_scale
        return out, out_scale, residual

    @staticmethod
    def quant_rmsnorm_op(
        x, weight, bias, scale_to_int, eps, dyn_scale: bool, type: int = 1
    ):
        """Quantized RMSNorm"""

        out = torch.empty_like(x, dtype=torch.int8)
        if is_per_token_smooth_quant():
            out_scale = torch.empty(
                x.shape[:-1], device=x.device, dtype=torch.float
            ).unsqueeze(-1)
        else:
            out_scale = torch.empty(12, device=x.device, dtype=torch.float)

        kunlun_ops.quant_rmsnorm(x, weight, bias, eps, out=out, out_scale=out_scale)
        return out, out_scale

    @staticmethod
    def smooth_quant_matmul_column_row_kernels(
        input_tensor,
        weight,
        smoother,
        input_scale,
        weight_scale,
        perTokenScaling,
        perChannelScaling,
        otype,
    ):
        """smooth_quant_matmul_column_row_kernels"""
        input_shape = input_tensor.shape
        weight_shape = weight.shape
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.reshape(-1, input_shape[-1])
            out = torch.empty(
                (input_shape[0] * input_shape[1], weight_shape[0]),
                dtype=torch.float16,
                device=weight.device,
            )
            output_bs_shape = [input_shape[0], input_shape[1]]
        elif input_tensor.dim() == 2:
            out = torch.empty(
                (input_shape[0], weight_shape[0]),
                dtype=torch.float16,
                device=weight.device,
            )
            output_bs_shape = [-1]
        kunlun_ops.smooth_quant_matmul_column_row_kernels(
            input_tensor,
            weight,
            smoother,
            input_scale,
            weight_scale,
            perTokenScaling,
            perChannelScaling,
            out=out,
        )

        out = out.view(*output_bs_shape, weight_shape[0])

        return out

    def _dbg(x):
        if torch.is_tensor(x):
            return (type(x), x.device, x.dtype, x.shape, x.is_contiguous())
        return (type(x), x)

    # MoE implementations live in the lightweight ops.moe package. These thin
    # forwarders preserve the established KunlunOps static API without making
    # MoE callers import this large operator facade. Arguments are passed
    # through rather than restated: see ops/moe/entry.py for the signatures.
    @staticmethod
    def fused_moe(*args, **kwargs):
        """Forward to `ops.moe.entry.fused_moe`."""
        return _moe_entry.fused_moe(*args, **kwargs)

    @staticmethod
    def fused_moe_int8(*args, **kwargs):
        """Forward to `ops.moe.entry.fused_moe_int8`."""
        return _moe_entry.fused_moe_int8(*args, **kwargs)

    @staticmethod
    def fused_moe_ep(*args, **kwargs):
        """Forward to `ops.moe.entry.fused_moe_ep`."""
        return _moe_entry.fused_moe_ep(*args, **kwargs)

    @staticmethod
    def fused_multi_head_latent_page_attention(
        hidden_states: torch.Tensor,
        q_lora_rank: int,
        kv_lora_rank: int,
        q_a_proj_w: torch.Tensor,
        q_a_layernorm_w: torch.Tensor,
        q_b_proj_w: torch.Tensor,
        q_proj_w: torch.Tensor,
        kv_a_proj_w: torch.Tensor,
        kv_a_layernorm_w: torch.Tensor,
        kv_b_proj_w: torch.Tensor,
        o_proj_w: torch.Tensor,
        head_num: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        max_context_len: int,
        layernorm_eps: float,
        scale: float,
        is_causal: bool,
        is_context: bool,
        mp_size: int,
        local_rank: int,
        rotary_pos_embedding: torch.Tensor,
        pa_block_tables: torch.Tensor,
        position: torch.Tensor,
        context_lens_cpu: torch.Tensor,
        slot_mapping: torch.Tensor,
        prompt_lods_cpu: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        """mla pa block"""
        output = torch.empty(
            hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device
        )
        kunlun_ops.xft_multi_head_latent_page_attention_block(
            hidden_states,
            q_lora_rank,
            kv_lora_rank,
            q_a_proj_w,
            q_a_layernorm_w,
            q_b_proj_w,
            q_proj_w,
            kv_a_proj_w,
            kv_a_layernorm_w,
            kv_b_proj_w,
            o_proj_w,
            head_num,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            max_context_len,
            layernorm_eps,
            scale,
            is_causal,
            is_context,
            mp_size,
            local_rank,
            rotary_pos_embedding,
            pa_block_tables,
            position,
            None,
            context_lens_cpu,
            slot_mapping,
            None,
            prompt_lods_cpu,
            out=output,
            k_cache=k_cache,
            v_cache=v_cache,
        )
        return output

    def fused_gdn_gating(
        A_log: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        beta: float = 1.0,
        threshold: float = 20.0,
    ) -> torch.Tensor:
        """fused_gdn_gating"""
        output = kunlun_ops.fused_gdn_gating(
            A_log,
            a,
            dt_bias,
        )
        return output

    def fused_recurrent_gated_delta_rule_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        h0_source: torch.Tensor,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool,
        cu_seqlens: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Core operator for Gated DeltaNet in Qwen3-NEXT model, fusing sigmoid_gating and delta_rule_update together.
        1.  Sigmoid Gating: Gates the input, similar to GLU (Gated Linear Unit).
        2.  Delta Rule Update: Performs a parallel state space model (SSM) recurrent update, combined with a local attention mechanism.
        """

        o, final_state = kunlun_ops.fused_recurrent_gated_delta_rule_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            h0_source,
            output_final_state,
            use_qk_l2norm_in_kernel,
            cu_seqlens,
        )
        return (o, final_state)
