# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dual-anchor Qwen3.6 DMTP checkpoint and its trained T1 conditioner."""

import torch
from torch import nn
from torch.nn import functional as F

from .qwen3_dflash2 import (
    DFlash2Qwen3Attention,
    DFlash2Qwen3DecoderLayer,
    DFlash2Qwen3ForCausalLM,
    DFlash2Qwen3Model,
)


class T1Conditioner(nn.Module):
    def __init__(self, width, rank, eps):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        self.eps = eps

    def forward(self, embeddings, native_hidden):
        h = native_hidden.float()
        h = h * (h.square().mean(-1, keepdim=True) + self.eps).rsqrt()
        residual = self.up(F.silu(self.down(h.to(self.down.weight.dtype))))
        blocks = embeddings.view(native_hidden.shape[0], 8, -1)
        return torch.cat(
            (blocks[:, :1], blocks[:, 1:2] + residual[:, None], blocks[:, 2:]), dim=1
        ).reshape_as(embeddings)


class DMTPAttention(DFlash2Qwen3Attention):
    # Set only for the eager boundary path. Indices point strictly before A.
    context_pages = None

    def attend(self, q, k, v):
        if self.context_pages is None:
            return super().attend(q, k, v)
        cache_k, cache_v = self.attn.kv_cache.unbind(0)
        width, heads, dim = 8, self.num_kv_heads, self.head_dim
        replicas = self.num_heads // heads
        q = q.reshape(-1, width, heads, replicas, dim)
        k = k.reshape(-1, width, heads, dim)
        v = v.reshape_as(k)
        outputs = []
        for i, (pages, offsets) in enumerate(self.context_pages):
            # Bound materialization to 4096 historical rows and eight local rows.
            # The block K/V stay in registers or temporary tensors; speculative
            # lookahead can extend past the last scheduler-owned physical page.
            keys = torch.cat((cache_k[pages, :, offsets], k[i])).transpose(0, 1)
            values = torch.cat((cache_v[pages, :, offsets], v[i])).transpose(0, 1)
            queries = q[i].transpose(0, 1).reshape(heads, width * replicas, dim)
            scores = (
                torch.bmm(queries.float(), keys.float().transpose(1, 2)) * self.scaling
            )
            output = torch.bmm(scores.softmax(-1), values.float())
            outputs.append(output.reshape(heads, width, replicas, dim).transpose(0, 1))
        return torch.stack(outputs).reshape(-1, self.num_heads * dim).to(q.dtype)


class DMTPDecoderLayer(DFlash2Qwen3DecoderLayer):
    attention_cls = DMTPAttention
    # The draft always runs a full block, even for shorter verifier lengths.
    draft_block_size = 8


class DMTPBackbone(DFlash2Qwen3Model):
    decoder_layer_cls = DMTPDecoderLayer

    def __init__(self, *, vllm_config, start_layer_id=0, prefix=""):
        config = vllm_config.speculative_config.draft_model_config.hf_config
        cfg = config.dflash_config
        # The export predates rope_parameters. The Kunlun DFlash layers read
        # the rotary fraction from that dictionary, not from the top level.
        config.rope_parameters = dict(
            config.rope_parameters or {},
            rope_type="default",
            rope_theta=10000000.0,
            partial_rotary_factor=config.partial_rotary_factor,
        )
        # A last partial verification still drafts one complete bidirectional block.
        config.max_position_embeddings = max(
            config.max_position_embeddings, vllm_config.model_config.max_model_len + 8
        )
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        self.mtp_prefix_conditioner = T1Conditioner(
            config.hidden_size, cfg["mtp_prefix_rank"], cfg["mtp_prefix_eps"]
        )


class DMTPDraft(DFlash2Qwen3ForCausalLM):
    model_cls = DMTPBackbone

    def load_weights(self, weights):
        seen = set()

        def adapt():
            for name, tensor in weights:
                if name == "hidden_norm.weight":
                    # Only the target-context RMSNorm has zero-centered gamma.
                    tensor = tensor.float() + 1.0
                if name.startswith("mtp_prefix_conditioner."):
                    seen.add(name)
                yield name, tensor

        super().load_weights(adapt())
        if seen != {
            "mtp_prefix_conditioner.down.weight",
            "mtp_prefix_conditioner.up.weight",
        }:
            raise ValueError(
                "DMTP checkpoint is missing trained T1 conditioner weights"
            )
