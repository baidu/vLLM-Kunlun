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
import torch_xmlir
import torch
import os
from typing import Optional, List, Dict
import vllm.envs as envs
import os
import ctypes
from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import kunlun_ops
    logger.info(f"Load custom ops library success!")
except ImportError as e:
    logger.warning("Import error msg: %s", e.msg)


_per_token_smooth_quant = True

def is_per_token_smooth_quant():
    """ is per token smooth quant """
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
        alibi_sqrt=False
        ):
        """ PagedAttentionV1 """
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
            vo_head_dim=128
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
        alibi_sqrt=False
        ):
        """ PagedAttentionV2 """
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
            vo_head_dim=128
        )


    # Activation ops
    @staticmethod
    def silu_and_mul(out: torch.Tensor,
                    x: torch.Tensor):
        """ silu and mul """
        kunlun_ops.silu_and_mul(
            x,
            axis=-1,
            turn=True,
            out=out,
            )

    # Activation ops
    @staticmethod
    def quick_gelu(out: torch.Tensor,
                    x: torch.Tensor):
        """ quick gelu """
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
        kunlun_ops.rmsnorm(
            x, weight.to(torch.float32), epsilon, out=out
        )

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
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox_style):
        """
        refactor RotaryEmbedding forward function
        """
        query_x = query.contiguous()
        key_x = key.contiguous()

        num_tokens = query_x.shape[0]
        num_heads = query_x.shape[1] // head_size
        num_kv_heads = key_x.shape[1] // head_size

        torch.ops._C.rotary_embedding(
            positions,
            query_x,
            key_x,
            head_size,
            cos_sin_cache,
            is_neox_style)

        query_x = query_x.view(num_tokens, num_heads * head_size)
        key_x = key_x.view(num_tokens, num_kv_heads * head_size)

        return query_x, key_x

    # Rotary embedding
    @staticmethod
    def mrotary_embedding(
        positions,
        mrope_section,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox_style):
        """
        refactor RotaryEmbedding forward function
        """
        query_x = query.contiguous()
        key_x = key.contiguous()
        query_x_dim = query_x.dim()
        assert is_neox_style
        kunlun_ops.mrotary_embedding_neox(
            positions,
            query_x,
            key_x,
            head_size,
            cos_sin_cache,
            mrope_section)

        query.data = query_x
        key.data  = key_x
        return query, key

    @staticmethod
    def swap_blocks(
            src,
            dst,
            block_mapping):
        """ swap_blocks """
        kunlun_ops.swap_blocks(
                src,
                dst,
                block_mapping
            )

    @staticmethod
    def copy_blocks(
        key_caches,
        value_caches,
        block_mapping):
        """ copy_blocks """
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
        """ reshape_and_cache """
        # slot_mapping_cast = slot_mapping.to(torch.int32)
        kunlun_ops.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping
        )

    @staticmethod
    def multi_query_kv_attention(
        usual_seq_lod_xpu: torch.Tensor,
        usual_seq_lod_cpu: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kargs
    ) -> torch.Tensor:
        """
        query: shape = [num_prompt_tokens, num_heads, head_size]
        """
        if query.dim() == 3:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
        output = torch.empty_like(query)
        alibi_slopes = kargs.get("alibi_slopes", None)
        mask = kargs.get("mask", None)
        is_causal = kargs.get("is_causal", True)
        is_lvsl = kargs.get("is_lvsl", True)

        B, T, Qh, Hd = query.shape
        KVh = key.size(2)
        if KVh != Qh:
            repeat = Qh // KVh
            key = key.repeat_interleave(repeat, dim=2)   # [B, T, Qh, Hd]
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
    def quant_fusedresidual_rmsnorm_op(x,
                                    residual,
                                    weight,
                                    bias,
                                    scale_to_int,
                                    eps,
                                    dyn_scale: bool,
                                    type: int = 1):
        """Quantized fused residual layer normalization"""
        out = torch.empty_like(x, dtype=torch.int8)

        if is_per_token_smooth_quant():
            out_scale = torch.empty(x.shape[:-1], device=x.device, dtype=torch.float).unsqueeze(-1)
        else:
            out_scale = torch.empty(12, device=x.device, dtype=torch.float)

        kunlun_ops.quant_fusedresidual_rmsnorm(x, residual, weight, bias, eps,
                                                        out=out, out_scale=out_scale , residual_tensor=residual)

        if residual is None:
            return out, out_scale
        return out, out_scale, residual

    @staticmethod
    def quant_rmsnorm_op(x,
                        weight,
                        bias,
                        scale_to_int,
                        eps,
                        dyn_scale : bool,
                        type: int = 1):
        """Quantized RMSNorm"""

        out = torch.empty_like(x, dtype=torch.int8)
        if is_per_token_smooth_quant():
            out_scale = torch.empty(x.shape[:-1], device=x.device, dtype=torch.float).unsqueeze(-1)
        else:
            out_scale = torch.empty(12, device=x.device, dtype=torch.float)

        kunlun_ops.quant_rmsnorm(x, weight, bias, eps,
                                            out=out, out_scale=out_scale)
        return out, out_scale

    @staticmethod
    def smooth_quant_matmul_column_row_kernels(input_tensor,
                                            weight,
                                            smoother,
                                            input_scale,
                                            weight_scale,
                                            perTokenScaling,
                                            perChannelScaling,
                                            otype):
        """smooth_quant_matmul_column_row_kernels"""
        input_shape = input_tensor.shape
        weight_shape = weight.shape
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.reshape(-1, input_shape[-1])
            out = torch.empty((input_shape[0] * input_shape[1],
                            weight_shape[0]),
                            dtype=torch.float16,
                            device=weight.device)
            output_bs_shape = [input_shape[0], input_shape[1]]
        elif input_tensor.dim() == 2:
            out = torch.empty((input_shape[0], weight_shape[0]),
                            dtype=torch.float16,
                            device=weight.device)
            output_bs_shape = [-1]
        kunlun_ops.smooth_quant_matmul_column_row_kernels(input_tensor,
                                                                    weight, smoother,
                                                                    input_scale,
                                                                    weight_scale,
                                                                    perTokenScaling,
                                                                    perChannelScaling,
                                                                    out=out)

        out = out.view(*output_bs_shape, weight_shape[0])

        return out

    def _dbg(x):
        if torch.is_tensor(x):
            return (type(x), x.device, x.dtype, x.shape, x.is_contiguous())
        return (type(x), x)
    @staticmethod
    def fused_moe(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        ep_rank: int,
        moe_top_k: int,
        renormalize: bool,
        inplace: bool = False,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        w1_bias: Optional[torch.Tensor] = None,
        w2_bias: Optional[torch.Tensor] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """fused_moe"""
        global_num_experts, up_gate_size, _ = w1.shape
        M, N = hidden_states.shape
        hidden_dim = w2.shape[1]
        normed_score = torch.empty(M,
                            moe_top_k,
                            dtype=torch.float32,
                            device=hidden_states.device)
        topk_ids = torch.empty(M,
                        moe_top_k,
                        dtype=torch.int32,
                        device=hidden_states.device)
        num_blocks = 12
        block_statistic = torch.zeros(
            num_blocks, global_num_experts, dtype=torch.int32, device=hidden_states.device
        )
        router_logits = router_logits.to(torch.float)
        if scoring_func == "softmax":
            torch.ops._C.moe_softmax_topk_norm(
                x=router_logits,
                normed_score=normed_score,
                topk_index=topk_ids,
                block_statistic=None,
                stable=True)
        elif scoring_func == "sigmoid":
            torch.ops._C.moe_sigmoid_group_topk_norm(
                    x=router_logits,
                    topk_index=topk_ids,
                    norm_score=normed_score,
                    block_static=block_statistic,
                    bias=e_score_correction_bias,
                    scale=1.0,
                    n_group=num_expert_group,
                    topk_group=topk_group,
                )

        if w1_bias is not None or w2_bias is not None: 
            # Rignt now this branch is for gpt oss
            # TODO (@xyDong23): faster here using moe_fc kernel
            normed_score = normed_score.to(hidden_states.dtype)
            out = torch.zeros(M * moe_top_k, N, dtype=hidden_states.dtype, device=hidden_states.device)
            repeat_x = hidden_states.repeat_interleave(moe_top_k, dim=0)
            topk_ids_flat = topk_ids.flatten()
            for i in range(global_num_experts):
                experts_id = ep_rank * global_num_experts + i
                selected_token = topk_ids_flat == experts_id
                if selected_token.sum():
                    cur_token = repeat_x[selected_token]
                    up_gate = torch.empty(selected_token.sum(), up_gate_size//2, 
                            dtype=cur_token.dtype, device=cur_token.device)
                    groupgemm1 = cur_token@ w1[i].T
                    # Add w13 bias
                    if w1_bias is not None:
                        groupgemm1 = groupgemm1 + w1_bias[i]
                    up_gate = torch.ops._C.swigluoai_and_mul(groupgemm1)
                    groupgemm2 = up_gate @ w2[i].T
                    # Add w2 bias
                    if w2_bias is not None:
                        groupgemm2 = groupgemm2 + w2_bias[i]
                    out[selected_token] = groupgemm2
            ouput = (out.view(M, moe_top_k, N) * normed_score.unsqueeze(2)).sum(dim=1).to(hidden_states.dtype)
            return ouput
        else:
            moe_expand = torch.empty((M * moe_top_k, N), dtype=hidden_states.dtype, device=hidden_states.device) # [M*top_k, N], float
            expert_m = torch.zeros(global_num_experts, dtype=torch.int32, device=hidden_states.device)             # [E]
            sorted_tokens_num_lod = torch.zeros(global_num_experts + 1, dtype=torch.int32, device=hidden_states.device)  # [E+1]
            sorted_tokens_idx = torch.zeros(M * moe_top_k, dtype=torch.int32, device=hidden_states.device)
            
            torch.ops._C.gen_block_statistic(topk_ids,block_statistic)

            torch.ops._C.moe_pre_sorted(
                x=hidden_states,
                topk_index=topk_ids,
                block_statistic=block_statistic,
                moe_expand=moe_expand,
                moe_index=sorted_tokens_idx,
                expert_m=expert_m,
                sorted_tokens_num_lod=sorted_tokens_num_lod)

            y = torch.empty(M,moe_top_k,
                    w1.shape[1],
                    dtype=hidden_states.dtype,
                    device=hidden_states.device)

            moe_expand = moe_expand.view(M * moe_top_k, hidden_dim)

            torch.ops._C.moe_fc(
                x=moe_expand,
                weight=w1,
                sorted_tokens_num_lod=sorted_tokens_num_lod,
                sorted_tokens_idx=sorted_tokens_idx,
                moe_topk=moe_top_k,
                y=y,
            )

            d = y.shape[-1] // 2
            output_shape = (y.shape[:-1] + (d, ))
            out1 = torch.empty(output_shape, dtype=y.dtype, device=y.device)
            torch.ops._C.silu_and_mul(out1, y)
            
            out = torch.empty(M,moe_top_k,
                    w2.shape[1],
                    dtype=hidden_states.dtype,
                    device=hidden_states.device)

            out1 = out1.reshape(-1, out1.shape[-1])

            torch.ops._C.moe_fc(
                x=out1,
                weight=w2,
                sorted_tokens_num_lod=sorted_tokens_num_lod,
                sorted_tokens_idx=sorted_tokens_idx,
                moe_topk=moe_top_k,
                y=out,
            )

            dequant_scale = torch.ones([M, moe_top_k], dtype = torch.float32, device=out.device)
            output = torch.empty([M, N], dtype=hidden_states.dtype, device=hidden_states.device)
            sorted_tokens_idx = sorted_tokens_idx.view(M, moe_top_k)

            torch.ops._C.moe_post(
                x=out,
                moe_index=sorted_tokens_idx,
                normed_scale=normed_score,
                dequant_scale=dequant_scale,
                y=output
            )
            
            return output

    @staticmethod
    def fused_moe_ep(
        hidden_states: torch.Tensor,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        gating_output: torch.Tensor,
        linear_weights: torch.Tensor,
        ep_rank: int,
        top_k: int,
        renormalize: bool,
        inplace: bool = False,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        w1_bias: Optional[torch.Tensor] = None,
        w2_bias: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        x = hidden_states
        batch, hidden_size  = x.shape
        num_local_experts, up_gate_size, _ = w13_weight.shape

        router_logits =  x.to(linear_weights.dtype)@linear_weights.T
        
        topk_weights = torch.empty(batch,
                            top_k,
                            dtype=router_logits.dtype,
                            device=router_logits.device)
        topk_ids = torch.empty(batch,
                            top_k,
                            dtype=torch.int32,
                            device=router_logits.device)
        block_static = torch.empty(0, dtype=torch.int32,device=router_logits.device)
        torch.ops._C.moe_softmax_topk(router_logits, topk_weights, topk_ids, block_static)

        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(1, keepdim=True)

        topk_weights = topk_weights.to(x.dtype)
        out = torch.zeros(batch * top_k, hidden_size, dtype=x.dtype, device=x.device)
        repeat_x = x.repeat_interleave(top_k, dim=0)
        topk_ids_flat = topk_ids.flatten()
        for i in range(num_local_experts):
            experts_id = ep_rank * num_local_experts + i
            selected_token = topk_ids_flat == experts_id
            if selected_token.sum():
                cur_token = repeat_x[selected_token]
                up_gate = torch.empty(selected_token.sum(), up_gate_size//2, 
                        dtype=cur_token.dtype, device=cur_token.device)
                torch.ops._C.silu_and_mul(up_gate, cur_token@ w13_weight[i].T)
                out[selected_token] = up_gate @ w2_weight[i].T
        output = (out.view(batch, top_k, hidden_size) * topk_weights.unsqueeze(2)).sum(dim=1).to(x.dtype)

        return output

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
        output = torch.empty(hidden_states.shape, dtype=hidden_states.dtype,
                            device=hidden_states.device)
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
            cu_seqlens: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
            '''
            Qwen3-NEXT模型中 Gated DeltaNet的核心算子, 将做完sigmoid_gating和delta_rule_update融合在一起
            1.  Sigmoid Gating: 对输入进行门控, 类似于 GLU (Gated Linear Unit)。
            2.  Delta Rule Update: 执行一个并行的状态空间模型(SSM)的递归更新, 同时结合了一个局部的注意力机制。
            '''

            o, final_state  = kunlun_ops.fused_recurrent_gated_delta_rule_fwd(
                q, k, v, g, beta, scale, h0_source, output_final_state, use_qk_l2norm_in_kernel,
                cu_seqlens)
            return (o, final_state)