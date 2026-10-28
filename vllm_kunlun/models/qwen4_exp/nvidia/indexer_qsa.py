# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp weight-free QSA indexer."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.platforms import current_platform
from vllm.model_executor.layers.rotary_embedding.mrope import (
    apply_interleaved_rope,
)
from vllm_kunlun.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)

from ..common.qsa_cache import (
    QSACompressedKeyCache,
    QSAForwardMetadata,
    QSAKeyStateCache,
    canonical_qsa_rope_positions,
)
from .ops.qsa_pre_indexer import qsa_pre_indexer


def _qsa_ops():
    """Q38 XPU PORT：优先用 kunlun 实现，拿不到就退回原来的 triton 模块。

    ``ops/qsa.py`` 全是 triton kernel，XPU 上不可用；``ops/qsa_kunlun.py`` 提供
    同名替身（打分/topk/稀疏 attn 走 kunlun_ops，其余用 torch）。
    置 ``Q38_KL_QSA=0`` 可强制回到 triton 路径，用来区分"昆仑算子算错了"
    和"框架接线错了"。
    """

    from .ops import qsa_kunlun

    if qsa_kunlun.available():
        return qsa_kunlun
    from .ops import qsa as qsa_triton

    return qsa_triton


def apply_qsa_rope(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Apply the main attention's exact 1D/MRoPE composition to QSA heads."""

    num_tokens, _, head_dim = tensor.shape
    rotary_dim = rotary_emb.rotary_dim
    cache = rotary_emb._match_cos_sin_cache_dtype(tensor)  # noqa: SLF001
    cos_sin = cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if positions.ndim == 2:
        # Q38 XPU PORT: upstream calls triton_mrope here. On XPU the launch
        # fails with CUDA_ERROR_NOT_SUPPORTED, and the packaged vLLM's
        # triton_mrope has no is_neox_style parameter (passing it is a
        # TypeError). Compose cos/sin per mrope_section instead and reuse the
        # 1D rotary path below; semantics match
        # MRotaryEmbedding.forward_native.
        assert rotary_emb.mrope_section
        section = rotary_emb.mrope_section
        if rotary_emb.mrope_interleaved:
            cos = apply_interleaved_rope(cos, section)
            sin = apply_interleaved_rope(sin, section)
        else:
            cos = torch.cat(
                [m[i] for i, m in enumerate(cos.split(section, dim=-1))], dim=-1
            )
            sin = torch.cat(
                [m[i] for i, m in enumerate(sin.split(section, dim=-1))], dim=-1
            )

    # Q38 XPU PORT: forward_cuda imports vllm.vllm_flash_attn, which raises
    # ImportError without the CUDA FA extensions. forward_native is the
    # equivalent torch implementation.
    apply_rope = rotary_emb.apply_rotary_emb
    rotate = (
        apply_rope.forward_cuda
        if current_platform.is_cuda()
        else apply_rope.forward_native
    )
    rotated = rotate(
        tensor[..., :rotary_dim],
        cos,
        sin,
    )
    return torch.cat((rotated, tensor[..., rotary_dim:]), dim=-1)


def _supports_fused_pre_indexer(
    rotary_emb: nn.Module,
    head_dim: int,
    num_kv_heads: int,
    compress_ratio: int,
) -> bool:
    rotary_dim = int(rotary_emb.rotary_dim)
    mrope_section = getattr(rotary_emb, "mrope_section", None)
    return (
        bool(getattr(rotary_emb, "is_neox_style", False))
        and (
            not mrope_section
            or (
                len(mrope_section) == 3
                and sum(mrope_section) == rotary_dim // 2
                and bool(getattr(rotary_emb, "mrope_interleaved", False))
            )
        )
        and head_dim == 128
        and rotary_dim == 64
        and num_kv_heads == 1
        and compress_ratio > 1
        and compress_ratio & (compress_ratio - 1) == 0
    )


class QSAIndexer(nn.Module):
    """Replicated Q/K projection plus paged, weight-free QSA selection.

    ``prefix`` must be the checkpoint's indexer prefix, normally
    ``model.layers.N.self_attn.indexer``.  Consequently the trainable names are
    ``index_qk_proj``, ``q_layernorm`` and ``k_layernorm`` under that prefix.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        rotary_emb: nn.Module,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if vllm_config.cache_config is None:
            raise ValueError("QSA requires a paged KV cache")
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA currently requires BF16")

        self.layer_id = int(layer_id)
        self.index_n_heads = int(config.indexer_n_heads)
        self.index_kv_heads = int(config.indexer_kv_heads)
        self.index_head_dim = int(config.indexer_head_dim)
        self.token_topk = int(config.indexer_budget)
        self.compress_ratio = int(config.indexer_compress_ratio)
        self.rotary_emb = rotary_emb
        self.qsa_ops = _qsa_ops()
        # Q38 XPU PORT：qsa_pre_indexer 是 triton 的融合前处理，昆仑路线下没有对应
        # 实现，只能走 unfused 参考路径（norm + rope + 压缩池化 + cache 写入分开做）。
        self.use_fused_pre_indexer = self.qsa_ops.__name__.endswith(
            "qsa"
        ) and _supports_fused_pre_indexer(
            rotary_emb,
            self.index_head_dim,
            self.index_kv_heads,
            self.compress_ratio,
        )
        self.prefix = prefix
        # MTP step 0 selects the target-aligned rows; later steps reuse them
        # while continuing to update the QSA side cache.
        self.skip_topk = False

        self.index_qk_proj = ReplicatedLinear(
            int(config.hidden_size),
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.index_qk_proj" if prefix else "index_qk_proj",
        )
        self.q_layernorm = GemmaRMSNorm(
            self.index_head_dim,
            eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        )
        self.k_layernorm = GemmaRMSNorm(
            self.index_head_dim,
            eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        )

        cache_config = vllm_config.cache_config
        cache_prefix = f"{prefix}." if prefix else ""
        self.raw_key_cache = QSAKeyStateCache(
            head_size=self.index_head_dim,
            dtype=torch.bfloat16,
            cache_rope_positions=vllm_config.model_config.uses_mrope,
            prefix=f"{cache_prefix}raw_key_cache",
            cache_config=cache_config,
            compress_ratio=self.compress_ratio,
            vllm_config=vllm_config,
        )
        self.compressed_key_cache = QSACompressedKeyCache(
            head_size=self.index_head_dim,
            dtype=torch.bfloat16,
            compress_ratio=self.compress_ratio,
            prefix=f"{cache_prefix}compressed_key_cache",
            cache_config=cache_config,
            vllm_config=vllm_config,
        )

    @property
    def output_width(self) -> int:
        return self.token_topk + self.compress_ratio - 1

    def _metadata(
        self,
    ) -> tuple[QSAForwardMetadata, QSAForwardMetadata] | None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            return None
        raw = cast(QSAForwardMetadata, metadata[self.raw_key_cache.prefix])
        compressed = cast(
            QSAForwardMetadata, metadata[self.compressed_key_cache.prefix]
        )
        if raw.num_actual_tokens != compressed.num_actual_tokens:
            raise RuntimeError("QSA side-cache metadata token counts disagree")
        if not raw.logical_positions.is_cuda and (
            not torch.equal(raw.logical_positions, compressed.logical_positions)
        ):
            raise RuntimeError("QSA side-cache metadata positions disagree")
        return raw, compressed

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return fixed-width request-relative token indices padded with ``-1``."""

        metadata = self._metadata()
        if metadata is None:
            # Preserve step-0 indices when later MTP steps reuse the buffer.
            if self.skip_topk and out is not None:
                return out
            result = torch.full(
                (hidden_states.shape[0], self.output_width),
                -1,
                dtype=torch.int32,
                device=hidden_states.device,
            )
            if out is not None:
                out.copy_(result)
                return out
            return result

        qsa_compress_groups_with_ratio = self.qsa_ops.qsa_compress_groups_with_ratio
        qsa_select_paged_tokens = self.qsa_ops.qsa_select_paged_tokens
        qsa_store_cache_rows = self.qsa_ops.qsa_store_cache_rows

        raw_metadata, compressed_metadata = metadata
        num_tokens = raw_metadata.num_actual_tokens
        hidden_states = hidden_states[:num_tokens]
        positions = positions[..., :num_tokens]

        # Q/K projection
        projected_qk, _ = self.index_qk_proj(hidden_states)
        projected_q, raw_keys = projected_qk.split(
            (
                self.index_n_heads * self.index_head_dim,
                self.index_kv_heads * self.index_head_dim,
            ),
            dim=-1,
        )
        raw_key_state_cache = self.raw_key_cache
        compressed_key_cache = self.compressed_key_cache.kv_cache

        if self.use_fused_pre_indexer:
            q = projected_q.new_empty(
                num_tokens,
                self.index_n_heads,
                self.index_head_dim,
            )
            qsa_pre_indexer(
                projected_q,
                raw_keys,
                positions,
                self.rotary_emb.cos_sin_cache,
                self.q_layernorm.weight,
                self.k_layernorm.weight,
                self.q_layernorm.variance_epsilon,
                q,
                raw_key_state_cache.kv_cache,
                raw_metadata.slot_mapping,
                raw_metadata.block_table,
                raw_metadata.query_start_loc,
                raw_metadata.logical_positions,
                compressed_key_cache,
                compressed_metadata.slot_mapping,
                compressed_metadata.k_work_metadata,
                compress_ratio=self.compress_ratio,
                mrope_section=getattr(self.rotary_emb, "mrope_section", None),
                rope_pos_offset=(
                    raw_key_state_cache.rope_position_offset
                    if raw_key_state_cache.rope_position_cache is not None
                    else None
                ),
            )
        else:
            # Unfused reference path
            # Q38 XPU PORT：原来这里用 flashinfer.norm.gemma_rmsnorm，XPU 上没有
            # flashinfer；直接调 GemmaRMSNorm 模块本身（语义一致，
            # x * (1 + w) / rms(x)，平台实现由 vllm 自己挑）。
            q = projected_q.reshape(-1, self.index_n_heads, self.index_head_dim)
            q = self.q_layernorm(q.reshape(-1, self.index_head_dim)).reshape_as(q)
            q = apply_qsa_rope(self.rotary_emb, positions, q)

            raw_key_cache = raw_key_state_cache.key_cache
            rope_position_cache = raw_key_state_cache.rope_position_cache
            if rope_position_cache is None:
                position_rows = raw_metadata.logical_positions.view(-1, 1, 1).expand(
                    -1, 1, 3
                )
            else:
                position_rows = canonical_qsa_rope_positions(positions).to(
                    device=raw_key_cache.device
                )
            compress_extra = {}
            if self.qsa_ops.__name__.endswith("qsa_kunlun"):
                # Q38 XPU PORT: key_cache / rope_position_cache 都是最后一维的
                # 切片（非连续），XPU 上 gather 会先整块物化。把整块连续的
                # kv_cache 一起递过去，让 kunlun 实现从它 gather 再切列。
                compress_extra["state_cache_full"] = raw_key_state_cache.kv_cache
                compress_extra["rope_position_offset"] = (
                    raw_key_state_cache.rope_position_offset
                )
            pooled, first_positions = qsa_compress_groups_with_ratio(
                raw_keys.reshape(-1, 1, self.index_head_dim),
                position_rows,
                raw_key_cache,
                raw_metadata.block_table,
                raw_metadata.token_to_req,
                raw_metadata.query_start_loc,
                raw_metadata.logical_positions,
                compressed_metadata.slot_mapping,
                self.compress_ratio,
                rope_position_cache,
                **compress_extra,
            )
            compressed_keys = self.k_layernorm(
                pooled.reshape(-1, self.index_head_dim)
            ).reshape(-1, 1, self.index_head_dim)
            if getattr(self.rotary_emb, "mrope_section", None):
                first_positions = first_positions.transpose(0, 1)
            else:
                first_positions = first_positions[:, 0]
            compressed_keys = apply_qsa_rope(
                self.rotary_emb,
                first_positions,
                compressed_keys,
            )
            qsa_store_cache_rows(
                compressed_key_cache,
                compressed_metadata.slot_mapping,
                compressed_keys,
            )
            if self.qsa_ops.__name__.endswith("qsa_kunlun"):
                # Q38 XPU PORT: 分别往 key_cache / rope_position_cache 写会 scatter
                # 到最后一维的切片（非连续视图），XPU 上要整块物化 cache。拼成整
                # 行一次写进连续的 kv_cache。
                qsa_store_cache_rows(
                    raw_key_state_cache.kv_cache,
                    raw_metadata.slot_mapping,
                    self.qsa_ops.qsa_pack_ring_rows(
                        raw_keys,
                        position_rows if rope_position_cache is not None else None,
                        raw_key_state_cache.head_size,
                        raw_key_state_cache.rope_position_offset,
                    ),
                )
            else:
                qsa_store_cache_rows(
                    raw_key_cache,
                    raw_metadata.slot_mapping,
                    raw_keys,
                )
                if rope_position_cache is not None:
                    qsa_store_cache_rows(
                        rope_position_cache,
                        raw_metadata.slot_mapping,
                        position_rows,
                    )

        if self.skip_topk:
            if out is None:
                raise RuntimeError("QSA top-k reuse requires an output buffer")
            return out

        # Score compressed keys, select blocks, then expand them to token indices.
        # Q38 XPU PORT：昆仑打分算子的 context_lens 必须有 host 副本，metadata 里
        # 现成的 logical_positions_cpu 直接传下去，省掉每层一次 D2H。
        extra = {}
        if getattr(compressed_metadata, "logical_positions_cpu", None) is not None:
            if self.qsa_ops.__name__.endswith("qsa_kunlun"):
                extra["query_positions_cpu"] = compressed_metadata.logical_positions_cpu[
                    :num_tokens
                ]
        return qsa_select_paged_tokens(
            q,
            compressed_key_cache,
            compressed_metadata.block_table,
            compressed_metadata.token_to_req,
            compressed_metadata.logical_positions,
            compressed_metadata.seq_lens,
            self.token_topk,
            self.compress_ratio,
            out,
            **extra,
        )


__all__ = ["QSAIndexer", "apply_qsa_rope"]
