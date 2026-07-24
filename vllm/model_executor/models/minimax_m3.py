# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The MiniMax AI team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the GPT-NeoX library and the OPT implementations
# in this library. It has been modified from its original forms to accommodate
# minor architectural differences compared to GPT-NeoX and OPT used by the Meta
# AI team that trained the model.
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
"""Inference-only MiniMax-M3 model (W8A8 compressed-tensors) with MSA sparse attention."""

import math
import os
from collections.abc import Iterable
from itertools import islice
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig

import kunlun_ops
import xspeedgate_ops  # noqa: F401  (register torch.ops.xspeedgate_ops)
import vllm_kunlun.platforms.envs as kunlun_envs

from vllm.logger import init_logger
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    ModelConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.distributed import (
    divide,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.activation import get_act_and_mul_fn
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    adjust_block_scale_shard,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    BlockQuantScaleParameter,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.linear_attn import MiniMaxText01RMSNormTP
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec
from vllm_kunlun.v1.attention.backends.minimax_m3_indexer import (
    MiniMaxM3KunlunIndexerBackend,
)

from .interfaces import EagleModelMixin, SupportsEagle3, SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


# ``MSA_DEBUG_LOG`` is intentionally implemented with small device-side trace
# buffers instead of reading tensors inside ``forward``.  Decode normally runs
# under a FULL CUDA Graph: Python logging in the model body only executes while
# the graph is traced/captured, whereas writes to these buffers are replayed for
# every real inference step.  ``MiniMaxM3ForCausalLM.compute_logits`` drains the
# buffers after the model graph has completed and emits the host-side log.
_MSA_DEBUG_PATH_UNSET = 0
_MSA_DEBUG_PATH_DENSE = 1
_MSA_DEBUG_PATH_SPARSE = 2

_MSA_DEBUG_STAGE_UNKNOWN = 0
_MSA_DEBUG_STAGE_PREFILL = 1
_MSA_DEBUG_STAGE_DECODE = 2
_MSA_DEBUG_STAGE_MIXED = 3

_MSA_DEBUG_CG_UNKNOWN = 0
_MSA_DEBUG_CG_NONE = 1
_MSA_DEBUG_CG_PIECEWISE = 2
_MSA_DEBUG_CG_FULL = 3

_MSA_DEBUG_REASON_TO_CODE = {
    None: 0,
    "MSA_FORCE_DENSE=1": 1,
    "missing_processed_index_projection": 2,
    "missing_index_projection_input": 3,
    "main_kv_cache_empty": 4,
    "index_cache_empty": 5,
    "unexpected_index_cache_shape": 6,
    "missing_slot_mapping": 7,
    "index_k_cache_write_failed": 8,
    "main_kv_cache_write_failed": 9,
    "block_tables_missing": 10,
    "prefill_query_start_loc_missing": 11,
    "empty_msa_dimensions": 12,
    "msa_block_score_failed": 13,
    "actual_topk_le_zero": 14,
    "msa_sparse_attention_failed": 15,
    "main_attn_metadata_missing": 16,
    "index_attn_metadata_missing": 17,
    "main_block_table_missing": 18,
    "index_block_table_missing": 19,
    "invalid_block_table_shape": 20,
    "block_table_batch_mismatch": 21,
    "msa_topk_transform_failed": 22,
    "invalid_msa_batch_partition": 23,
    "short_prefill_all_blocks": 24,
    "msa_topk_workspace_too_small": 25,
}
_MSA_DEBUG_CODE_TO_REASON = {
    code: reason for reason, code in _MSA_DEBUG_REASON_TO_CODE.items()
}

# Integer fields in ``MiniMaxM3Attention._msa_debug_meta``.  Keeping this as a
# flat tensor makes the trace CUDA-Graph-safe and cheap to update.
_MSA_META_GENERATION = 0
_MSA_META_PATH = 1
_MSA_META_STAGE = 2
_MSA_META_REASON = 3
_MSA_META_CG_MODE = 4
_MSA_META_TOTAL_TOKENS = 5
_MSA_META_STORED_TOKENS = 6
_MSA_META_BATCH_SIZE = 7
_MSA_META_ACTUAL_TOPK = 8
_MSA_META_BLOCK_SIZE = 9
_MSA_META_NUM_TOPK_HEADS = 10
_MSA_META_SIZE = 11


def _msa_debug_enabled() -> bool:
    return os.environ.get("MSA_DEBUG_LOG", "0") == "1"


def _msa_debug_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(value, minimum)


def _msa_debug_stage_code(stage: str) -> int:
    if stage == "prefill":
        return _MSA_DEBUG_STAGE_PREFILL
    if stage == "decode":
        return _MSA_DEBUG_STAGE_DECODE
    if stage == "mixed":
        return _MSA_DEBUG_STAGE_MIXED
    return _MSA_DEBUG_STAGE_UNKNOWN


def _msa_debug_cg_code(cg_mode: str) -> int:
    mode = cg_mode.upper()
    if "PIECEWISE" in mode:
        return _MSA_DEBUG_CG_PIECEWISE
    if "FULL" in mode:
        return _MSA_DEBUG_CG_FULL
    if "NONE" in mode:
        return _MSA_DEBUG_CG_NONE
    return _MSA_DEBUG_CG_UNKNOWN


def _format_msa_layer_ids(layer_ids: Iterable[int]) -> str:
    """Format layer ids as compact inclusive ranges (for example ``0-2,5``)."""
    ids = sorted(set(layer_ids))
    if not ids:
        return "-"
    ranges: list[str] = []
    start = previous = ids[0]
    for layer_id in ids[1:]:
        if layer_id == previous + 1:
            previous = layer_id
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = layer_id
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _qkv_with_inder_enabled() -> bool:
    # Keep the env spelling requested by the launch script.
    return os.environ.get("VLLM_KUNLUN_QKV_WITH_INDER", "1") != "0"


def _kunlun_moe_gate(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Run the mixed BF16-input/FP32-weight MiniMax router projection.

    Despite the current ``kunlun_ops`` docstring describing ``other`` as
    ``[K, N]``, the installed kernel consumes the same contiguous ``[N, K]``
    layout as ``ReplicatedLinear.weight`` and the checkpoint.  Keeping that
    layout avoids both a post-load weight copy and a forward-time transpose.
    """
    output = torch.empty(
        (hidden_states.shape[0], weight.shape[0]),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    torch.ops._C.minimax_m3_moe_gate(hidden_states, weight, output)
    return output


def _kunlun_moe_gate_enabled(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    return (
        not kunlun_envs.VLLM_KUNLUN_DISABLE_ROUTER_MOE_GATE
        and hidden_states.dtype == torch.bfloat16
        and weight.dtype == torch.float32
    )


def _kunlun_dynamic_prequantized_linear_enabled(linear: nn.Module) -> bool:
    scheme = getattr(linear, "scheme", None)
    kernel = getattr(scheme, "kernel", None)
    kernel_config = getattr(kernel, "config", None)
    return (
        callable(getattr(linear, "forward_prequantized", None))
        and callable(getattr(kernel, "apply_prequantized_weights", None))
        and not getattr(kernel_config, "is_static_input_scheme", True)
        and getattr(kernel_config, "input_symmetric", False)
        and getattr(linear, "bias", None) is None
    )


def _kunlun_prequantized_linear_enabled(linear: nn.Module) -> bool:
    return (
        not kunlun_envs.VLLM_KUNLUN_DISABLE_FUSED_NORM_QUANT
        and _kunlun_dynamic_prequantized_linear_enabled(linear)
    )


def _kunlun_fused_shared_expert_enabled(config: PretrainedConfig) -> bool:
    if os.environ.get("VLLM_KUNLUN_DISABLE_FUSED_MOE_GATE", "0") == "1":
        return False
    if os.environ.get("VLLM_KUNLUN_ENABLE_FUSED_SHARED_EXPERT", "0") != "1":
        return False
    return (
        getattr(config, "shared_intermediate_size", None)
        == getattr(config, "intermediate_size", None)
    )


def _ensure_text_config_attrs(config: PretrainedConfig) -> PretrainedConfig:
    """Forward missing attributes from ``text_config`` to top-level config.

    MiniMaxM3VLConfig is a multimodal wrapper that puts all LLM parameters
    (vocab_size, hidden_size, num_hidden_layers, etc.) inside
    ``config.text_config``.  Copy them to the top-level so that the rest of
    the model code can access ``config.vocab_size`` directly.
    """
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        for attr_name in vars(text_config):
            if attr_name.startswith("_"):
                continue
            if not hasattr(config, attr_name):
                setattr(config, attr_name, getattr(text_config, attr_name))
    return config


class MiniMaxM3DenseMLP(nn.Module):
    """Dense SwiGLU MLP with packed gate_up_proj for non-MoE layers."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act = get_act_and_mul_fn("swigluoai")
        self._fused_act_quant = (
            not kunlun_envs.VLLM_KUNLUN_DISABLE_FUSED_DENSE_SWIGLU_QUANT
            and _kunlun_dynamic_prequantized_linear_enabled(self.down_proj)
        )

    def forward(
        self,
        x: torch.Tensor,
        x_q: torch.Tensor | None = None,
        x_max: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for dense MLP with optional pre-quantized input."""
        if x_q is not None and x_max is not None:
            gate_up, _ = self.gate_up_proj.forward_prequantized(
                x_q, x_max, x.dtype
            )
        else:
            gate_up, _ = self.gate_up_proj(x)
        if self._fused_act_quant:
            x_q, x_max = torch.ops._C.moe_swiglu_quant(
                gate_up,
                alpha=self.act.alpha,
                beta=1.0,
                limit=self.act.limit,
            )
            x, _ = self.down_proj.forward_prequantized(
                x_q, x_max, gate_up.dtype
            )
        else:
            x = self.act(gate_up)
            x, _ = self.down_proj(x)
        return x


class MiniMaxM3SharedExpertMLP(nn.Module):
    """Shared expert MLP (unpacked gate/up/down) for MoE layers."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj",
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act = get_act_and_mul_fn("swigluoai")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, _ = self.gate_proj(x)
        up, _ = self.up_proj(x)
        d = gate.shape[-1]
        combined = torch.cat([gate, up], dim=-1)
        x = self.act(combined)
        x, _ = self.down_proj(x)
        return x


class MiniMaxM3MoE(nn.Module):
    """MoE layer with routed experts (FusedMoE) + shared expert MLP."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size > config.num_local_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_local_experts}."
            )

        self.use_routing_bias = getattr(config, "use_routing_bias", False)
        if self.use_routing_bias:
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(config.num_local_experts, dtype=torch.float32)
            )
            self.e_score_correction_bias.weight_loader = (
                MiniMaxM3MoE.ebias_weight_loader
            )
        else:
            self.e_score_correction_bias = None

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.num_fused_shared_experts = (
            1 if _kunlun_fused_shared_expert_enabled(config) else 0
        )
        self.fuse_shared_experts = self.num_fused_shared_experts > 0
        if self.fuse_shared_experts:
            self.shared_experts = None
        else:
            # Keep shared_experts partial output local and let FusedMoE reduce
            # the routed+shared sum once. This avoids a separate TP all_reduce
            # for the shared branch.
            self.shared_experts = MiniMaxM3SharedExpertMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
            )

        self.experts = FusedMoE(
            num_experts=(
                config.num_local_experts + self.num_fused_shared_experts
            ),
            top_k=(
                config.num_experts_per_tok + self.num_fused_shared_experts
            ),
            use_grouped_topk=True,
            num_expert_group=1,
            topk_group=1,
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            renormalize=True,
            swiglu_limit=getattr(config, "swiglu_limit", None),
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scale_to_output=True,
            shared_experts=self.shared_experts,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            router_logits_dtype=torch.float32,
        )
        self.experts.routed_num_experts = config.num_local_experts
        self.experts.num_kunlun_fused_shared_experts = (
            self.num_fused_shared_experts
        )
        self.experts.kunlun_output_routed_scaling_factor = (
            self.routed_scaling_factor
        )
        # Ensure swiglu params are visible to quantized MoE apply_monolithic
        self.experts.swiglu_alpha = getattr(config, "swiglu_alpha", 1.0) or 1.0
        self.experts.swiglu_beta = getattr(config, "swiglu_beta", None) or 1.0

    @staticmethod
    def ebias_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight.to(torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if _kunlun_moe_gate_enabled(hidden_states, self.gate.weight):
            router_logits = _kunlun_moe_gate(hidden_states, self.gate.weight)
        else:
            router_logits, _ = self.gate(hidden_states.to(torch.float32))
        # FusedMoE handles routed scaling and combines shared experts before
        # the final TP all_reduce, reducing MoE communication from 2x to 1x.
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        return final_hidden_states.view(num_tokens, hidden_dim)


class MiniMaxM3QKVParallelLinearWithIndexer(ColumnParallelLinear):
    """MiniMax-M3 QKV projection fused with sparse-index Q/K projection.

    A single column-parallel GEMM emits, per rank::

        [q | k | v | index_q | index_k]

    ``index_q`` shards exactly like the main KV heads, including the replicated
    KV-head path when TP is larger than the KV-head count. ``index_k`` is one
    shared head and is replicated on every rank.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        total_num_index_heads: int,
        index_head_size: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        assert total_num_index_heads == total_num_kv_heads, (
            "MiniMaxM3QKVParallelLinearWithIndexer requires "
            "total_num_index_heads == total_num_kv_heads"
        )
        assert index_head_size == head_size, (
            "MiniMaxM3QKVParallelLinearWithIndexer assumes index_head_size "
            "matches the main attention head_size"
        )

        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.index_head_size = index_head_size
        self.total_num_index_heads = total_num_index_heads

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        self.num_index_heads = self.num_kv_heads

        self.q_size = self.num_heads * self.head_size
        self.kv_size = self.num_kv_heads * self.head_size
        self.index_q_size = self.num_index_heads * self.index_head_size
        self.index_k_size = self.index_head_size

        # ColumnParallelLinear divides each output_sizes entry by tp_size when it
        # creates the local parameter. Count replicated groups x tp_size so each
        # rank materializes [local q | local/replicated k/v | local/replicated
        # index_q | replicated index_k].
        self.output_sizes = [
            self.q_size * tp_size,
            self.kv_size * tp_size,
            self.kv_size * tp_size,
            self.index_q_size * tp_size,
            self.index_k_size * tp_size,
        ]

        ColumnParallelLinear.__init__(
            self,
            input_size=self.hidden_size,
            output_size=sum(self.output_sizes),
            bias=bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def validate_shard_id(self, shard_id: Any) -> None:
        """Validate that the shard_id is one of the supported values."""
        if shard_id in {"q", "k", "v", "index_q", "index_k"}:
            return
        raise ValueError(
            "Shard id for MiniMaxM3QKVParallelLinearWithIndexer must be one "
            "of 'q', 'k', 'v', 'index_q', or 'index_k', got "
            f"{shard_id}."
        )

    def _get_shard_offset_mapping(self, loaded_shard_id: str) -> int | None:
        return {
            "q": 0,
            "k": self.q_size,
            "v": self.q_size + self.kv_size,
            "index_q": self.q_size + 2 * self.kv_size,
            "index_k": self.q_size + 2 * self.kv_size + self.index_q_size,
        }.get(loaded_shard_id)

    def _get_shard_size_mapping(self, loaded_shard_id: str) -> int | None:
        return {
            "q": self.q_size,
            "k": self.kv_size,
            "v": self.kv_size,
            "index_q": self.index_q_size,
            "index_k": self.index_k_size,
        }.get(loaded_shard_id)

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ) -> None:
        """Load QKV+index weight shards with proper TP offset handling."""
        self.validate_shard_id(loaded_shard_id)

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)
        assert shard_offset is not None and shard_size is not None

        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(
                weight_block_size,
                shard_size,
                shard_offset,
            )

        # index_k is fully replicated: num_heads == tp_size makes
        # load_qkv_weight choose shard 0 on every rank. q/k/v/index_q follow the
        # standard QKV loader; q ignores num_heads and uses tp_rank directly,
        # while k/v/index_q use the KV-head replica factor.
        num_heads = self.tp_size if loaded_shard_id == "index_k" else (
            self.num_kv_head_replicas
        )
        param.load_qkv_weight(
            loaded_weight=loaded_weight,
            num_heads=num_heads,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            tp_rank=self.tp_rank,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ) -> None:
        """Load QKV+index weight shards into the parameter with TP awareness."""
        self.validate_shard_id(loaded_shard_id)
        assert loaded_shard_id in ("q", "k", "v", "index_q", "index_k")

        output_dim = getattr(param, "output_dim", None)
        assert output_dim is not None
        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)
        assert shard_offset is not None and shard_size is not None

        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(
                weight_block_size,
                shard_size,
                shard_offset,
            )

        param_data = param.data.narrow(output_dim, shard_offset, shard_size)
        if loaded_shard_id == "q":
            shard_rank = self.tp_rank
        elif loaded_shard_id == "index_k":
            shard_rank = 0
        else:
            shard_rank = self.tp_rank // self.num_kv_head_replicas
        loaded_weight = loaded_weight.narrow(
            output_dim,
            shard_rank * shard_size,
            shard_size,
        )

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class MiniMaxM3IndexQKParallelLinear(ColumnParallelLinear):
    """Tensor-parallel sparse-index Q/K projection.

    A single column-parallel GEMM emits, per rank::

        [index_q | index_k]

    ``index_q`` follows the same replicated KV-head sharding used by the main
    attention K/V heads. ``index_k`` is one shared head and is replicated on every
    rank. This is the fallback used when
    ``VLLM_KUNLUN_QKV_WITH_INDER=0`` disables the 5-way qkv/index projection.
    """

    def __init__(
        self,
        hidden_size: int,
        total_num_index_heads: int,
        index_head_size: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        self.hidden_size = hidden_size
        self.total_num_index_heads = total_num_index_heads
        self.index_head_size = index_head_size

        tp_size = get_tensor_model_parallel_world_size()
        if tp_size >= self.total_num_index_heads:
            self.num_index_heads = 1
            self.num_index_head_replicas = divide(
                tp_size, self.total_num_index_heads)
        else:
            self.num_index_heads = divide(self.total_num_index_heads, tp_size)
            self.num_index_head_replicas = 1

        self.index_q_size = self.num_index_heads * self.index_head_size
        self.index_k_size = self.index_head_size
        self.output_sizes = [
            self.index_q_size * tp_size,
            self.index_k_size * tp_size,
        ]

        ColumnParallelLinear.__init__(
            self,
            input_size=self.hidden_size,
            output_size=sum(self.output_sizes),
            bias=bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def validate_shard_id(self, shard_id: Any) -> None:
        """Validate that the shard_id is one of the supported values."""
        if shard_id in {"index_q", "index_k"}:
            return
        raise ValueError(
            "Shard id for MiniMaxM3IndexQKParallelLinear must be one of "
            f"'index_q' or 'index_k', got {shard_id}."
        )

    def _get_shard_offset_mapping(self, loaded_shard_id: str) -> int | None:
        return {
            "index_q": 0,
            "index_k": self.index_q_size,
        }.get(loaded_shard_id)

    def _get_shard_size_mapping(self, loaded_shard_id: str) -> int | None:
        return {
            "index_q": self.index_q_size,
            "index_k": self.index_k_size,
        }.get(loaded_shard_id)

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ) -> None:
        """Load index QK weight shards into the parameter with TP awareness."""
        self.validate_shard_id(loaded_shard_id)
        assert loaded_shard_id in ("index_q", "index_k")

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)
        assert shard_offset is not None and shard_size is not None

        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(
                weight_block_size,
                shard_size,
                shard_offset,
            )

        num_heads = self.tp_size if loaded_shard_id == "index_k" else (
            self.num_index_head_replicas
        )
        param.load_qkv_weight(
            loaded_weight=loaded_weight,
            num_heads=num_heads,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            tp_rank=self.tp_rank,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ) -> None:
        """Load index QK weight shards into the parameter with TP awareness."""
        self.validate_shard_id(loaded_shard_id)
        assert loaded_shard_id in ("index_q", "index_k")

        output_dim = getattr(param, "output_dim", None)
        assert output_dim is not None
        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)
        assert shard_offset is not None and shard_size is not None

        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(
                weight_block_size,
                shard_size,
                shard_offset,
            )

        param_data = param.data.narrow(output_dim, shard_offset, shard_size)
        if loaded_shard_id == "index_k":
            shard_rank = 0
        else:
            shard_rank = self.tp_rank // self.num_index_head_replicas
        loaded_weight = loaded_weight.narrow(
            output_dim,
            shard_rank * shard_size,
            shard_size,
        )

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class MiniMaxM3MSAIndexer(nn.Module):
    """MSA indexer producing compact Q/K representation for block scoring.

    Uses index heads with per-head RMSNorm and the same partial RoPE as the
    main Q/K branch. By default the projection is fused into
    ``MiniMaxM3QKVParallelLinearWithIndexer``; when
    ``VLLM_KUNLUN_QKV_WITH_INDER=0`` this module owns a separate TP-aware
    ``index_qk_proj`` and then normalizes/applies RoPE.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        sparse_config: dict[str, Any],
        index_rotary_emb: nn.Module,
        num_index_heads: int | None = None,
        quant_config: QuantizationConfig | None = None,
        use_projector: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_index_heads = sparse_config["sparse_num_index_heads"]
        self.index_dim = sparse_config["sparse_index_dim"]
        self.index_rotary_emb = index_rotary_emb

        if num_index_heads is None:
            tp_size = get_tensor_model_parallel_world_size()
            if tp_size >= self.total_num_index_heads:
                self.num_index_heads = 1
            else:
                self.num_index_heads = divide(self.total_num_index_heads, tp_size)
        else:
            self.num_index_heads = num_index_heads
        self.index_q_size = self.num_index_heads * self.index_dim
        self.index_k_size = self.index_dim
        if use_projector:
            self.index_qk_proj = MiniMaxM3IndexQKParallelLinear(
                self.hidden_size,
                self.total_num_index_heads,
                self.index_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.indexer.index_qk_proj",
            )
            self.num_index_heads = self.index_qk_proj.num_index_heads
            self.index_q_size = self.index_qk_proj.index_q_size
            self.index_k_size = self.index_qk_proj.index_k_size
        else:
            self.index_qk_proj = None

        # Per-head RMSNorm weight — checkpoint stores flat [index_dim], NOT
        # [num_index_heads, index_dim].  Broadcasts across all index heads.
        self.index_q_norm = nn.Parameter(torch.zeros(self.index_dim))
        self.index_k_norm = nn.Parameter(torch.zeros(self.index_dim))
        self._use_fused_index_qknorm_rope = (
            os.environ.get("VLLM_KUNLUN_DISABLE_FUSED_INDEX_QKNORM_ROPE", "0")
            != "1"
            and self.index_dim == 128
            and getattr(index_rotary_emb, "rotary_dim", None) == 64
            and getattr(index_rotary_emb, "is_neox_style", True)
        )
        self._fused_index_csc = None

    def forward(
        self,
        positions: torch.Tensor,
        index_q_raw: torch.Tensor,
        index_k_raw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute compact index Q/K for block scoring.

        Note: checkpoint stores ``index_k_proj`` with only 1 index head,
        so ``index_k`` output is ``[seq_len, 1, index_dim]`` while
        ``index_q`` is ``[seq_len, num_index_heads, index_dim]``.

        Args:
            positions: [seq_len] token positions for index-branch RoPE
            index_q_raw: hidden states when this indexer owns ``index_qk_proj``;
                otherwise [seq_len, local_num_index_heads * index_dim], or packed
                [seq_len, local_index_q + index_k] when index_k_raw is None.
            index_k_raw: [seq_len, index_dim], replicated on every rank.

        Returns:
            index_q: [seq_len, num_index_heads, index_dim], normed + RoPE
            index_k: [seq_len, 1, index_dim], normed + RoPE
        """
        if self.index_qk_proj is not None and index_k_raw is None:
            index_qk, _ = self.index_qk_proj(index_q_raw)
            index_qk = index_qk.contiguous()
            index_q_raw, index_k_raw = index_qk.split(
                [self.index_q_size, self.index_k_size],
                dim=-1,
            )
        elif index_k_raw is None:
            index_qk = index_q_raw.contiguous()
            index_q_raw, index_k_raw = index_qk.split(
                [self.index_q_size, self.index_k_size],
                dim=-1,
            )
        else:
            index_qk = torch.cat(
                (index_q_raw, index_k_raw),
                dim=-1,
            ).contiguous()
        seq_len = index_q_raw.shape[0]
        index_q = index_q_raw.view(seq_len, self.num_index_heads, self.index_dim)
        # index_k_proj only produces 1 index head in checkpoint
        index_k = index_k_raw.view(seq_len, 1, self.index_dim)

        if self._use_fused_index_qknorm_rope:
            q_heads = self.num_index_heads
            k_heads = 1
            csc = self._fused_index_csc
            src = self.index_rotary_emb.cos_sin_cache
            if (csc is None or csc.device != index_qk.device
                    or csc.dtype != index_qk.dtype):
                csc = src.to(index_qk.device, dtype=index_qk.dtype)
                if csc.ndim == 4:
                    csc = csc.squeeze(0).squeeze(0)
                csc = csc.contiguous()
                if not torch.compiler.is_compiling():
                    self._fused_index_csc = csc
            weights = [
                self.index_q_norm.detach().to(
                    device=index_qk.device,
                    dtype=index_qk.dtype,
                ).contiguous(),
                self.index_k_norm.detach().to(
                    device=index_qk.device,
                    dtype=index_qk.dtype,
                ).contiguous(),
            ]
            head_offsets = [0, q_heads]
            head_counts = [q_heads, k_heads]
            _T = index_qk.size(0)
            _MAXB = 8192
            if _T <= _MAXB:
                torch.ops.xspeedgate_ops.fused_gemma_qknorm_rope_grouped(
                    index_qk,
                    weights,
                    head_offsets,
                    head_counts,
                    csc,
                    positions,
                    1e-6,
                )
            else:
                for _s in range(0, _T, _MAXB):
                    _e = min(_s + _MAXB, _T)
                    torch.ops.xspeedgate_ops.fused_gemma_qknorm_rope_grouped(
                        index_qk[_s:_e],
                        weights,
                        head_offsets,
                        head_counts,
                        csc,
                        positions[_s:_e],
                        1e-6,
                    )
            index_q, index_k = index_qk.split(
                [q_heads * self.index_dim, k_heads * self.index_dim],
                dim=-1,
            )
            return (
                index_q.reshape(seq_len, q_heads, self.index_dim),
                index_k.reshape(seq_len, k_heads, self.index_dim),
            )

        # Per-head RMSNorm: (1 + weight) * x / rms(x)
        q_var = index_q.float().pow(2).mean(dim=-1, keepdim=True)
        k_var = index_k.float().pow(2).mean(dim=-1, keepdim=True)
        index_q = (
            (index_q.float() * torch.rsqrt(q_var + 1e-6)
             * (1.0 + self.index_q_norm))
        ).to(index_q_raw.dtype)
        index_k = (
            (index_k.float() * torch.rsqrt(k_var + 1e-6)
             * (1.0 + self.index_k_norm))
        ).to(index_k_raw.dtype)

        index_q_shape = index_q.shape
        index_k_shape = index_k.shape
        index_q, index_k = self.index_rotary_emb(
            positions,
            index_q.contiguous().view(seq_len, -1),
            index_k.contiguous().view(seq_len, -1),
        )
        index_q = index_q.reshape(index_q_shape)
        index_k = index_k.reshape(index_k_shape)

        return index_q, index_k


class MiniMaxM3IndexerCache(nn.Module, AttentionLayerBase):
    """Key-only side cache for MiniMax-M3 MSA index keys.

    This mirrors upstream MiniMax-M3: one compact index-key vector is stored per
    token. Kunlun MSA score kernels consume a paged-MQA 4D view of this tensor
    (``[num_blocks, 1, 128, index_dim]``), but the underlying allocation is the
    official key-only layout ``[num_blocks, 128, index_dim]``.
    """

    def __init__(
        self,
        index_dim: int,
        cache_config: CacheConfig | None,
        prefix: str,
        dtype: torch.dtype = torch.bfloat16,
        backend_cls: type[AttentionBackend] = MiniMaxM3KunlunIndexerBackend,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.layer_name = prefix
        self.index_dim = index_dim
        self.num_kv_heads = 1
        self.cache_config = cache_config
        self.dtype = dtype
        self.backend_cls = backend_cls
        self.kv_cache = torch.tensor([])

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        """Return the attention backend class for this index cache."""
        return self.backend_cls

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        """Return the KV cache spec for the MiniMax-M3 index side-cache."""
        block_size = (
            self.cache_config.block_size
            if self.cache_config is not None
            else vllm_config.cache_config.block_size
        )
        if block_size != 128:
            raise ValueError(
                "MiniMax-M3 MSA requires KV/index-cache page size 128, "
                f"got {block_size}."
            )
        return MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.index_dim,
            dtype=self.dtype,
        )


class MiniMaxM3Attention(nn.Module):
    """Multi-head attention with QK-Norm, RoPE, and optional MSA sparse attention.

    M3 uses:
    - ``head_dim=128``, ``num_attention_heads=64``, ``num_key_value_heads=4``
    - ``rotary_dim=64``, ``rope_theta=5000000``
    - ``use_qk_norm=True``, ``qk_norm_type="per_head"``

    When ``use_sparse=True``, the ``_sparse_forward`` path calls
    ``kunlun_ops.msa_block_score`` + ``kunlun_ops.msa_sparse_attention``
    instead of standard dense flash attention.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rotary_dim: int,
        rope_parameters: dict[str, Any] | None = None,
        attn_window_size: int | None = None,
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_sparse: bool = False,
        sparse_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.use_sparse = use_sparse
        self._force_dense = os.environ.get("MSA_FORCE_DENSE", "0") == "1"
        self._qkv_with_inder = _qkv_with_inder_enabled()
        self._msa_short_prefill_dense = (
            not kunlun_envs.VLLM_KUNLUN_DISABLE_MSA_SHORT_PREFILL_DENSE
        )
        if use_sparse and not self._force_dense:
            assert sparse_config is not None
            self.total_num_index_heads = sparse_config["sparse_num_index_heads"]
            self.index_dim = sparse_config["sparse_index_dim"]
            if self._qkv_with_inder:
                self.qkv_proj = MiniMaxM3QKVParallelLinearWithIndexer(
                    hidden_size,
                    self.head_dim,
                    self.total_num_heads,
                    self.total_num_kv_heads,
                    self.total_num_index_heads,
                    self.index_dim,
                    bias=qkv_bias,
                    quant_config=quant_config,
                    prefix=f"{prefix}.qkv_proj",
                )
                self.local_num_index_heads = self.qkv_proj.num_index_heads
                self.index_q_size = self.qkv_proj.index_q_size
                self.index_k_size = self.qkv_proj.index_k_size
            else:
                self.qkv_proj = QKVParallelLinear(
                    hidden_size,
                    self.head_dim,
                    self.total_num_heads,
                    self.total_num_kv_heads,
                    bias=qkv_bias,
                    quant_config=quant_config,
                    prefix=f"{prefix}.qkv_proj",
                )
                if tp_size >= self.total_num_index_heads:
                    self.local_num_index_heads = 1
                else:
                    self.local_num_index_heads = divide(
                        self.total_num_index_heads, tp_size)
                self.index_q_size = self.local_num_index_heads * self.index_dim
                self.index_k_size = self.index_dim
        else:
            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=qkv_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if rope_parameters is None:
            rope_parameters = {}
        if "rope_theta" not in rope_parameters:
            rope_parameters["rope_theta"] = 5000000.0
        if "partial_rotary_factor" not in rope_parameters:
            rope_parameters["partial_rotary_factor"] = rotary_dim / self.head_dim
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            per_layer_sliding_window=attn_window_size,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        # Per-head QK-norm: checkpoint stores [head_dim] weight shared across heads
        self.q_norm_weight = nn.Parameter(torch.ones(self.head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(self.head_dim))
        self.rms_norm_eps = rms_norm_eps
        # Lazily-materialized (1+w) caches for kunlun qkrmsnorm (plain RMSNorm,
        # no +1). Computed on first forward AFTER load_weights so the real
        # checkpoint values are used. Invalidated if weight version changes.
        self._q_w1_cache = None
        self._k_w1_cache = None
        self._q_w1_ver = -1
        self._k_w1_ver = -1

        # === P0: fused GemmaRMSNorm(1+w) + partial NeoX RoPE ===
        # torch.ops.xspeedgate_ops.fused_gemma_qknorm_rope fuses the main-path
        # split -> qkrmsnorm(q) -> qkrmsnorm(k) -> rotary_emb(q,k) (4 launches +
        # 2 temp tensors) into one in-place kernel on the packed qkv tensor.
        # Requires HEAD_DIM=128 & rotary_dim=64 (matches M3). The kernel applies
        # GemmaRMSNorm (1+w) internally, so it must receive the raw checkpoint
        # norm weights. The fallback qkrmsnorm path materializes (1+w) lazily via
        # _qknorm_w1(...), because kunlun_ops.qkrmsnorm is plain RMSNorm.
        # Verified equivalent to the fallback in test/test_fused_qknorm_rope.py
        # (RAW weight, bf16 cache both PASS; max_abs ~0.016, mean_abs ~1.2e-3).
        # Gated by VLLM_KUNLUN_DISABLE_FUSED_QKNORM_ROPE=1 (fallback to the old
        # qkrmsnorm+rotary_emb path).
        self._use_fused_qknorm_rope = (
            os.environ.get("VLLM_KUNLUN_DISABLE_FUSED_QKNORM_ROPE", "0") != "1"
            and self.head_dim == 128
            and rotary_dim == 64
        )
        self._fused_qk_raw = None  # (q_raw, k_raw); built lazily (post-fold)
        self._fused_csc = None     # cos_sin_cache matched to query device/dtype

        # Physical KV-cache page size used by the MSA kernels to interpret

        # block_tables.  This MUST equal the runtime cache_config.block_size
        # (e.g. --block-size 128), NOT CacheConfig.DEFAULT_BLOCK_SIZE (16) —
        # the MSA kernels address the paged cache by this value, so a mismatch
        # corrupts block_tables addressing.  Falls back to the default only if
        # cache_config is unavailable.
        self.kv_cache_block_size = (
            cache_config.block_size
            if cache_config is not None and cache_config.block_size is not None
            else CacheConfig.DEFAULT_BLOCK_SIZE
        )

        # MSA sparse attention setup
        self.attn_prefix = f"{prefix}.attn"
        self._msa_debug_log = _msa_debug_enabled()
        self._msa_debug_max_tokens = _msa_debug_int(
            "MSA_DEBUG_LOG_MAX_TOKENS", 32, minimum=1
        )
        if use_sparse and self._msa_debug_log:
            assert sparse_config is not None
            debug_topk = max(
                int(sparse_config.get("sparse_topk_blocks", 16)), 1
            )
            # These non-persistent buffers move with the model but never enter
            # the checkpoint.  Their fixed, small shapes are safe to mutate
            # from captured graphs.  For long prefills only the latest
            # ``MSA_DEBUG_LOG_MAX_TOKENS`` rows are retained.
            self.register_buffer(
                "_msa_debug_meta",
                torch.zeros(_MSA_META_SIZE, dtype=torch.int64),
                persistent=False,
            )
            self.register_buffer(
                "_msa_debug_positions",
                torch.full(
                    (self._msa_debug_max_tokens,),
                    -1,
                    dtype=torch.int64,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_msa_debug_topk",
                torch.full(
                    (
                        self._msa_debug_max_tokens,
                        self.num_kv_heads,
                        debug_topk,
                    ),
                    -1,
                    dtype=torch.int32,
                ),
                persistent=False,
            )
        else:
            self.register_buffer("_msa_debug_meta", None, persistent=False)
            self.register_buffer("_msa_debug_positions", None, persistent=False)
            self.register_buffer("_msa_debug_topk", None, persistent=False)
        # MSA_FORCE_DENSE=1 makes sparse layers run plain dense attention. In
        # that mode we skip building the indexer (index_q/k projections + norms)
        # AND the index side-cache entirely, so neither the projection weights
        # nor the KV-manager-allocated index cache blocks consume device memory.
        # The env var is captured once at construction time; the runtime
        # _sparse_forward guard (see below) must agree with it.
        if use_sparse:
            assert sparse_config is not None
            self.sparse_config = sparse_config
            # In dense-forced mode skip the memory-heavy indexer and index
            # side-cache. Setting them to None means:
            #   * no index_q/k projection weights are created/loaded;
            #   * MiniMaxM3IndexerCache is never registered in
            #     static_forward_context, so the KV manager allocates no index
            #     cache blocks for this layer.
            if self._force_dense:
                self.indexer = None
                self.index_cache = None
                if self._msa_debug_log:
                    logger.info_once(
                        "[MSA][startup:layer][L%s] configured=sparse "
                        "effective=dense reason=MSA_FORCE_DENSE=1 "
                        "indexer=skipped index_cache=skipped",
                        self.layer_idx,
                    )
            else:
                self.indexer = MiniMaxM3MSAIndexer(
                    config,
                    sparse_config,
                    self.rotary_emb,
                    num_index_heads=self.local_num_index_heads,
                    quant_config=quant_config,
                    use_projector=not self._qkv_with_inder,
                    prefix=prefix,
                )
                self.index_cache = MiniMaxM3IndexerCache(
                    sparse_config["sparse_index_dim"],
                    cache_config,
                    prefix=f"{prefix}.index_cache",
                )

    def _qknorm_w1(self, weight: torch.Tensor, cache_attr: str, ver_attr: str) -> torch.Tensor:
        """Return cached fp32 (1 + weight). Recompute when the underlying
        Parameter storage changed (after load_weights copies real values) or
        device/dtype/shape differ. Uses data_ptr+_version as a cheap key so
        the cache materializes from the REAL checkpoint weight, not init ones."""
        cache = getattr(self, cache_attr)
        wd = weight.detach()
        key = (wd.data_ptr(), wd._version, wd.device, tuple(wd.shape))
        if cache is None or getattr(self, ver_attr) != key:
            cache = (1.0 + wd.float()).contiguous()
            setattr(self, cache_attr, cache)
            setattr(self, ver_attr, key)
        return cache

    def _run_combined_qknorm_rope(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply main q/k and index q/k Gemma norm + RoPE in one launch."""
        if not qkv.is_contiguous():
            qkv = qkv.contiguous()

        csc = self._fused_csc
        src = self.rotary_emb.cos_sin_cache
        if csc is None or csc.device != qkv.device or csc.dtype != qkv.dtype:
            csc = src.to(qkv.device, dtype=qkv.dtype)
            if csc.ndim == 4:
                csc = csc.squeeze(0).squeeze(0)
            csc = csc.contiguous()
            if not torch.compiler.is_compiling():
                self._fused_csc = csc

        def _raw(weight: torch.Tensor) -> torch.Tensor:
            return weight.detach().to(
                device=qkv.device,
                dtype=qkv.dtype,
            ).contiguous()

        weights = [
            _raw(self.q_norm_weight),
            _raw(self.k_norm_weight),
            _raw(self.indexer.index_q_norm),
            _raw(self.indexer.index_k_norm),
        ]
        index_q_offset = self.num_heads + 2 * self.num_kv_heads
        head_offsets = [
            0,
            self.num_heads,
            index_q_offset,
            index_q_offset + self.local_num_index_heads,
        ]
        head_counts = [
            self.num_heads,
            self.num_kv_heads,
            self.local_num_index_heads,
            1,
        ]

        _T = qkv.size(0)
        _MAXB = 8192
        if _T <= _MAXB:
            torch.ops.xspeedgate_ops.fused_gemma_qknorm_rope_grouped(
                qkv,
                weights,
                head_offsets,
                head_counts,
                csc,
                positions,
                self.rms_norm_eps,
            )
        else:
            for _s in range(0, _T, _MAXB):
                _e = min(_s + _MAXB, _T)
                torch.ops.xspeedgate_ops.fused_gemma_qknorm_rope_grouped(
                    qkv[_s:_e],
                    weights,
                    head_offsets,
                    head_counts,
                    csc,
                    positions[_s:_e],
                    self.rms_norm_eps,
                )
        return qkv

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states_q: torch.Tensor | None = None,
        hidden_states_max: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states_q is not None and hidden_states_max is not None:
            qkv, _ = self.qkv_proj.forward_prequantized(
                hidden_states_q, hidden_states_max, hidden_states.dtype
            )
        else:
            qkv, _ = self.qkv_proj(hidden_states)
        index_q_raw = None
        index_k_raw = None
        index_qk_already_processed = False
        sparse_active = self.use_sparse and not self._force_dense
        combined_qknorm_rope = (
            sparse_active
            and self._qkv_with_inder
            and self._use_fused_qknorm_rope
            and self.indexer is not None
            and getattr(self.indexer, "_use_fused_index_qknorm_rope", False)
        )
        if combined_qknorm_rope:
            qkv = self._run_combined_qknorm_rope(qkv, positions)
            q, k, v, index_q_raw, index_k_raw = qkv.split(
                [
                    self.q_size,
                    self.kv_size,
                    self.kv_size,
                    self.index_q_size,
                    self.index_k_size,
                ],
                dim=-1,
            )
            q = q.contiguous()
            # Keep K/V as column-contiguous views of the fused projection.
            # ``reshape_and_cache_flash`` honors their wider row stride, while
            # materializing both here would copy another 256 BF16 values per
            # token.  Q cannot do this: ``msa_sparse_attention`` requires a
            # compact query tensor.
            index_qk_already_processed = True
        else:
            if sparse_active and self._qkv_with_inder:
                qkv, index_q_raw, index_k_raw = qkv.split(
                    [
                        self.q_size + 2 * self.kv_size,
                        self.index_q_size,
                        self.index_k_size,
                    ],
                    dim=-1,
                )
            elif sparse_active:
                index_q_raw = hidden_states
        if not combined_qknorm_rope:
            if self._use_fused_qknorm_rope:
                # === P0: fused QK-norm + RoPE on the packed qkv (in-place) ===
                # The fused kernel applies GemmaRMSNorm (1+w) internally, so it must
                # receive raw checkpoint norm weights. The fallback path below uses
                # _qknorm_w1(...) because kunlun_ops.qkrmsnorm is plain RMSNorm.
                if self._fused_qk_raw is None:
                    q_raw = self.q_norm_weight.detach().to(
                        self.q_norm_weight.dtype).contiguous()
                    k_raw = self.k_norm_weight.detach().to(
                        self.k_norm_weight.dtype).contiguous()
                    self._fused_qk_raw = (q_raw, k_raw)
                q_raw, k_raw = self._fused_qk_raw

                # cos_sin_cache: [max_pos, rotary_dim=64], matched to query dtype/dev.
                csc = self._fused_csc
                src = self.rotary_emb.cos_sin_cache
                if (csc is None or csc.device != qkv.device
                        or csc.dtype != qkv.dtype):
                    csc = src.to(qkv.device, dtype=qkv.dtype)
                    if csc.ndim == 4:
                        csc = csc.squeeze(0).squeeze(0)
                    csc = csc.contiguous()
                    if not torch.compiler.is_compiling():
                        self._fused_csc = csc

                # The op requires a contiguous qkv (in-place write). qkv_proj
                # normally returns contiguous output already.
                if not qkv.is_contiguous():
                    qkv = qkv.contiguous()
                # The kernel caps qkv.size(0) at 8192, so chunk large prefill/profile
                # batches over rows. Row slices of a contiguous [T, C] tensor share
                # storage, so the in-place op writes back into qkv.
                _T = qkv.size(0)
                _MAXB = 8192
                if _T <= _MAXB:
                    torch.ops.xspeedgate_ops.fused_gemma_qknorm_rope(
                        qkv, q_raw, k_raw, csc, positions,
                        self.num_heads, self.num_kv_heads, self.num_kv_heads,
                        self.rms_norm_eps,
                    )
                else:
                    for _s in range(0, _T, _MAXB):
                        _e = min(_s + _MAXB, _T)
                        torch.ops.xspeedgate_ops.fused_gemma_qknorm_rope(
                            qkv[_s:_e], q_raw, k_raw, csc, positions[_s:_e],
                            self.num_heads, self.num_kv_heads, self.num_kv_heads,
                            self.rms_norm_eps,
                        )
                q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size],
                                    dim=-1)
                q = q.contiguous()
                # Sparse cache insertion accepts the projection view's row
                # stride.  Dense attention keeps its original compact-K
                # contract, including the force-dense fallback configuration.
                if not sparse_active:
                    k = k.contiguous()
            else:
                q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

                # QK-norm: Per-head Gemma-style RMSNorm: (1 + weight) * x / rms(x)
                # Kunlun kernel path: kunlun_ops.qkrmsnorm is plain RMSNorm (no +1),
                # so the Gemma (1+w) weight is materialized as (1.0 + norm_weight).
                M = q.shape[0]
                q3 = q.contiguous().view(M, -1, self.head_dim)
                k3 = k.contiguous().view(M, -1, self.head_dim)
                q_out = torch.empty_like(q3)
                k_out = torch.empty_like(k3)
                kunlun_ops.qkrmsnorm(
                    q3, q_out,
                    self._qknorm_w1(self.q_norm_weight, "_q_w1_cache", "_q_w1_ver"),
                    eps=self.rms_norm_eps,
                )
                kunlun_ops.qkrmsnorm(
                    k3, k_out,
                    self._qknorm_w1(self.k_norm_weight, "_k_w1_cache", "_k_w1_ver"),
                    eps=self.rms_norm_eps,
                )
                q = q_out.view(M, -1)
                k = k_out.view(M, -1)

                q, k = self.rotary_emb(positions, q, k)

        if self.use_sparse:
            return self._sparse_forward(positions, q, k, v,
                                        index_q_raw, index_k_raw,
                                        index_qk_already_processed)
        else:
            attn_output = self.attn(q, k, v)
            output, _ = self.o_proj(attn_output)
            return output

    def _get_msa_cache_state(self, ctx: Any, layer_key: str):
        """Return main/index cache tensors and slot mappings for MSA."""
        kv_cache_tensor = self.attn.kv_cache
        # kv_cache: [2, num_pages, num_kv_heads, page_block_size, head_dim]
        # Profile run: KV cache is empty (not yet allocated), fall back to dense.
        if kv_cache_tensor.numel() == 0:
            return None, "main_kv_cache_empty"
        k_cache = kv_cache_tensor[0]
        v_cache = kv_cache_tensor[1]

        index_cache_tensor = self.index_cache.kv_cache
        if index_cache_tensor.numel() == 0:
            return None, "index_cache_empty"
        if index_cache_tensor.dim() != 3:
            return None, "unexpected_index_cache_shape"
        index_k_cache = index_cache_tensor.view(
            index_cache_tensor.shape[0],
            1,
            index_cache_tensor.shape[1],
            index_cache_tensor.shape[2],
        )

        slot = _get_slot_mapping_for_layer(ctx.slot_mapping, layer_key)
        index_slot = _get_slot_mapping_for_layer(
            ctx.slot_mapping, self.index_cache.prefix
        )
        if slot is None or index_slot is None:
            return None, "missing_slot_mapping"

        return (k_cache, v_cache, index_k_cache, slot, index_slot), None

    def _store_msa_index_k_cache(
        self,
        index_k: torch.Tensor,
        index_k_cache: torch.Tensor,
        index_slot: torch.Tensor,
        index_dim: int,
    ) -> str | None:
        index_tokens = min(index_slot.shape[0], index_k.shape[0])
        index_k_w = index_k.contiguous().view(-1, 1, index_dim)

        def _repair_index_cache() -> None:
            _store_msa_paged_cache_reference(
                index_k_w[:index_tokens],
                index_k_cache,
                index_slot[:index_tokens],
            )

        try:
            ret = kunlun_ops.store_paged_kv_cache(
                index_k_w[:index_tokens],
                index_k_cache,
                index_slot[:index_tokens],
            )
            if ret != 0:
                logger.error(
                    "[MSA:L%s] store_paged_kv_cache returned ret=%s; "
                    "fallback to dense",
                    self.layer_idx,
                    ret,
                )
                _repair_index_cache()
                return "index_k_cache_write_failed"
            return None
        except Exception:
            logger.exception(
                "[MSA:L%s] failed to write index KV cache; "
                "repairing touched slots before dense fallback",
                self.layer_idx,
            )
            try:
                _repair_index_cache()
            except Exception as repair_error:
                raise RuntimeError(
                    "MSA index KV cache write and reference repair both failed"
                ) from repair_error
            return "index_k_cache_write_failed"

    def _store_msa_main_kv_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot: torch.Tensor,
    ) -> str | None:
        # The cache op consumes the last two dimensions densely but honors the
        # token-row stride.  Fused projection slices satisfy that contract, so
        # preserve the views instead of copying K and V into compact buffers.
        k_w = k.view(-1, self.num_kv_heads, self.head_dim)
        v_w = v.view(-1, self.num_kv_heads, self.head_dim)
        main_tokens = min(slot.shape[0], k_w.shape[0])

        def _repair_main_cache() -> None:
            _store_msa_paged_cache_reference(
                k_w[:main_tokens],
                k_cache,
                slot[:main_tokens],
                value=v_w[:main_tokens],
                value_cache=v_cache,
            )

        try:
            ret = kunlun_ops.reshape_and_cache_flash(
                k_w[:main_tokens],
                v_w[:main_tokens],
                k_cache,
                v_cache,
                slot[:main_tokens],
                BLHD_LAYOUT=False,
            )
            if ret != 0:
                logger.error(
                    "[MSA:L%s] reshape_and_cache_flash returned ret=%s; "
                    "fallback to dense",
                    self.layer_idx,
                    ret,
                )
                _repair_main_cache()
                return "main_kv_cache_write_failed"
            return None
        except Exception:
            logger.exception(
                "[MSA:L%s] failed to write main KV cache; repairing touched "
                "slots before dense fallback",
                self.layer_idx,
            )
            try:
                _repair_main_cache()
            except Exception as repair_error:
                raise RuntimeError(
                    "MSA main KV cache write and reference repair both failed"
                ) from repair_error
            return "main_kv_cache_write_failed"

    def _record_msa_debug_state(
        self,
        positions: torch.Tensor,
        *,
        path_code: int,
        stage: str,
        reason: str | None,
        cg_mode: str,
        total_tokens: int,
        batch_size: int,
        topk_idx: torch.Tensor | None = None,
        actual_topk: int = 0,
        block_size: int = 0,
    ) -> None:
        """Write one CUDA-Graph-safe MSA trace snapshot.

        The host does not inspect device data here.  In particular, there is
        no ``cpu()``, ``item()`` or logging call in this helper, so FULL graph
        decode remains capturable.  The fixed buffers are read later from
        ``compute_logits``, after capture/replay has completed.
        """
        if not self._msa_debug_log:
            return
        meta = self._msa_debug_meta
        debug_positions = self._msa_debug_positions
        debug_topk = self._msa_debug_topk
        assert meta is not None
        assert debug_positions is not None
        assert debug_topk is not None

        stored_tokens = min(total_tokens, self._msa_debug_max_tokens)
        source_start = total_tokens - stored_tokens

        meta[_MSA_META_GENERATION].add_(1)
        meta[_MSA_META_PATH].fill_(path_code)
        meta[_MSA_META_STAGE].fill_(_msa_debug_stage_code(stage))
        meta[_MSA_META_REASON].fill_(
            _MSA_DEBUG_REASON_TO_CODE.get(reason, -1)
        )
        meta[_MSA_META_CG_MODE].fill_(_msa_debug_cg_code(cg_mode))
        meta[_MSA_META_TOTAL_TOKENS].fill_(total_tokens)
        meta[_MSA_META_STORED_TOKENS].fill_(stored_tokens)
        meta[_MSA_META_BATCH_SIZE].fill_(batch_size)
        meta[_MSA_META_ACTUAL_TOPK].fill_(actual_topk)
        meta[_MSA_META_BLOCK_SIZE].fill_(block_size)

        debug_positions[:stored_tokens].copy_(
            positions.reshape(-1)[source_start:total_tokens].to(torch.int64)
        )
        debug_topk[:stored_tokens].fill_(-1)
        if topk_idx is None:
            meta[_MSA_META_NUM_TOPK_HEADS].zero_()
            return

        num_topk_heads = min(topk_idx.shape[1], debug_topk.shape[1])
        stored_topk = min(actual_topk, debug_topk.shape[2])
        meta[_MSA_META_NUM_TOPK_HEADS].fill_(num_topk_heads)
        debug_topk[
            :stored_tokens, :num_topk_heads, :stored_topk
        ].copy_(
            topk_idx[
                source_start:total_tokens,
                :num_topk_heads,
                :stored_topk,
            ]
        )

    def _get_msa_debug_snapshot(
        self, *, include_blocks: bool = True
    ) -> dict[str, Any] | None:
        """Drain the latest trace buffers into a host-side snapshot."""
        if not self._msa_debug_log or self._msa_debug_meta is None:
            return None

        meta = self._msa_debug_meta.detach().cpu().tolist()
        generation = int(meta[_MSA_META_GENERATION])
        if generation <= 0:
            return None

        path_code = int(meta[_MSA_META_PATH])
        stage_code = int(meta[_MSA_META_STAGE])
        cg_code = int(meta[_MSA_META_CG_MODE])
        total_tokens = max(int(meta[_MSA_META_TOTAL_TOKENS]), 0)
        stored_tokens = max(int(meta[_MSA_META_STORED_TOKENS]), 0)
        stored_tokens = min(stored_tokens, self._msa_debug_max_tokens)
        actual_topk = max(int(meta[_MSA_META_ACTUAL_TOPK]), 0)
        num_topk_heads = max(int(meta[_MSA_META_NUM_TOPK_HEADS]), 0)

        positions: list[int] = []
        topk: list[list[list[int]]] = []
        if stored_tokens and include_blocks:
            assert self._msa_debug_positions is not None
            positions = [
                int(value)
                for value in self._msa_debug_positions[
                    :stored_tokens
                ].detach().cpu().tolist()
            ]
        if (
            stored_tokens
            and include_blocks
            and path_code == _MSA_DEBUG_PATH_SPARSE
        ):
            assert self._msa_debug_topk is not None
            topk = self._msa_debug_topk[
                :stored_tokens,
                :num_topk_heads,
                :actual_topk,
            ].detach().cpu().tolist()

        return {
            "layer": self.layer_idx,
            "generation": generation,
            "path": {
                _MSA_DEBUG_PATH_DENSE: "dense",
                _MSA_DEBUG_PATH_SPARSE: "msa",
            }.get(path_code, "unset"),
            "stage": {
                _MSA_DEBUG_STAGE_PREFILL: "prefill",
                _MSA_DEBUG_STAGE_DECODE: "decode",
                _MSA_DEBUG_STAGE_MIXED: "mixed",
            }.get(stage_code, "unknown"),
            "reason": _MSA_DEBUG_CODE_TO_REASON.get(
                int(meta[_MSA_META_REASON]),
                f"unknown_code_{int(meta[_MSA_META_REASON])}",
            ),
            "cg_mode": {
                _MSA_DEBUG_CG_NONE: "NONE",
                _MSA_DEBUG_CG_PIECEWISE: "PIECEWISE",
                _MSA_DEBUG_CG_FULL: "FULL",
            }.get(cg_code, "UNKNOWN"),
            "total_tokens": total_tokens,
            "stored_tokens": stored_tokens,
            "first_stored_row": max(total_tokens - stored_tokens, 0),
            "batch_size": max(int(meta[_MSA_META_BATCH_SIZE]), 0),
            "actual_topk": actual_topk,
            "block_size": max(int(meta[_MSA_META_BLOCK_SIZE]), 0),
            "positions": positions,
            "topk": topk,
            "truncated": stored_tokens < total_tokens,
        }

    def _sparse_forward(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        index_q_raw: torch.Tensor | None,
        index_k_raw: torch.Tensor | None,
        index_qk_already_processed: bool = False,
    ) -> torch.Tensor:
        """MSA sparse attention forward path.

        Uses ``msa_block_score`` for coarse block-level scoring and
        ``msa_sparse_attention`` for top-k sparse attention computation.
        """
        num_tokens = q.shape[0]
        msa_debug_log = self._msa_debug_log
        stage = "unknown"
        batch_num = 0

        # cudagraph runtime mode for this forward. FULL means this call is being
        # captured into / replayed from a full CUDA graph. cg_mode is attached
        # to every trace line so the log correlates graph mode with
        # msa/dense. Decode always runs the capture-safe path and prefill never
        # runs under FULL (it goes piecewise), so no capture-safety warning is
        # needed here.
        cg_mode = "unknown"
        if msa_debug_log:
            try:
                _fc = get_forward_context()
                cg_mode = str(getattr(_fc, "cudagraph_runtime_mode", "unknown"))
            except Exception:
                cg_mode = "unknown"

        def _log_msa_path(path: str, reason: str | None = None) -> None:
            # Compiled/CUDA-Graph runtime is reported from the replayed trace
            # buffers after forward.  Keep Python logger calls out of the FX
            # graph; this once-only line is only for eager/profile fallbacks.
            if not msa_debug_log or torch.compiler.is_compiling():
                return
            # stage may still be "unknown" for early fallbacks that trigger
            # before prefill/decode is determined (e.g. empty cache during
            # cudagraph capture, or the force-dense short-circuit). Label the
            # force-dense case explicitly; other pre-stage fallbacks as "pre",
            # so capture-time dense fallbacks stay visible instead of dropped.
            if stage != "unknown":
                log_stage = stage
            elif self._force_dense:
                log_stage = "forced_dense"
            else:
                log_stage = "pre"
            # info_once: first-ever occurrence of each distinct combination.
            if reason is None:
                logger.info_once(
                    "[MSA][trace:L%s] cg=%s stage=%s path=%s",
                    self.layer_idx,
                    cg_mode,
                    log_stage,
                    path,
                )
            else:
                logger.info_once(
                    "[MSA][trace:L%s] cg=%s stage=%s path=%s reason=%s",
                    self.layer_idx,
                    cg_mode,
                    log_stage,
                    path,
                    reason,
                )

        def _dense_output(reason: str) -> torch.Tensor:
            _log_msa_path("dense", reason)
            self._record_msa_debug_state(
                positions,
                path_code=_MSA_DEBUG_PATH_DENSE,
                stage=stage,
                reason=reason,
                cg_mode=cg_mode,
                total_tokens=num_tokens,
                batch_size=batch_num,
            )
            attn_output = self.attn(q, k, v)
            output, _ = self.o_proj(attn_output)
            return output

        if self._force_dense:
            # The old force-dense short circuit happened before metadata was
            # inspected, leaving runtime logs at stage=forced_dense.  In debug
            # mode inspect only the cheap metadata fields so every real call is
            # still identified as prefill or decode.  FULL graph is decode by
            # construction and therefore performs no host scalar read.
            if msa_debug_log:
                try:
                    debug_ctx = get_forward_context()
                    debug_meta = debug_ctx.attn_metadata
                    if isinstance(debug_meta, dict):
                        debug_meta = debug_meta.get(self.attn_prefix)
                    elif isinstance(debug_meta, list):
                        debug_meta = debug_meta[0].get(self.attn_prefix)
                    debug_block_tables = _get_meta_field(
                        debug_meta, "block_tables"
                    )
                    if debug_block_tables is None:
                        debug_block_tables = _get_meta_field(
                            debug_meta, "block_table"
                        )
                    if debug_block_tables is None:
                        debug_block_tables = _get_meta_field(
                            debug_meta, "block_table_tensor"
                        )
                    if debug_block_tables is not None:
                        batch_num = debug_block_tables.shape[0]
                    if _msa_debug_cg_code(cg_mode) == _MSA_DEBUG_CG_FULL:
                        stage = "decode"
                    elif batch_num > 0:
                        debug_actual_tokens = _get_meta_field(
                            debug_meta, "num_actual_tokens"
                        )
                        if debug_actual_tokens is None:
                            debug_actual_tokens = num_tokens
                        if isinstance(debug_actual_tokens, torch.Tensor):
                            debug_actual_tokens = int(
                                debug_actual_tokens.item()
                            )
                        else:
                            debug_actual_tokens = int(debug_actual_tokens)
                        stage = (
                            "prefill"
                            if debug_actual_tokens > batch_num
                            else "decode"
                        )
                except Exception:
                    logger.exception(
                        "[MSA][runtime][L%s] failed to classify forced-dense "
                        "forward stage",
                        self.layer_idx,
                    )
            return _dense_output("MSA_FORCE_DENSE=1")

        # Beyond this point the indexer + index cache are required; in
        # force-dense mode they were never built, so the guard above must have
        # returned. index_dim is read from the indexer submodule here.
        index_dim = self.indexer.index_dim

        # --- Step 1: Compute compact index Q/K ---
        if index_qk_already_processed:
            if index_q_raw is None or index_k_raw is None:
                return _dense_output("missing_processed_index_projection")
            index_q = index_q_raw.reshape(
                num_tokens, self.indexer.num_index_heads, index_dim)
            index_k = index_k_raw.reshape(num_tokens, 1, index_dim)
        else:
            if index_q_raw is None:
                return _dense_output("missing_index_projection_input")
            index_q, index_k = self.indexer(positions, index_q_raw, index_k_raw)
        index_q = index_q.contiguous()
        # index_q: [num_tokens, local_num_index_heads, index_dim]

        # --- Step 2: Get KV cache and attention metadata ---
        ctx = get_forward_context()
        layer_key = self.attn_prefix

        all_attn_meta = ctx.attn_metadata
        main_attn_meta = _get_attn_metadata_for_layer(
            all_attn_meta, layer_key
        )
        index_attn_meta = _get_attn_metadata_for_layer(
            all_attn_meta, self.index_cache.prefix
        )

        cache_state, cache_reason = self._get_msa_cache_state(ctx, layer_key)
        if cache_reason is not None:
            return _dense_output(cache_reason)
        if main_attn_meta is None:
            return _dense_output("main_attn_metadata_missing")
        if index_attn_meta is None:
            return _dense_output("index_attn_metadata_missing")
        k_cache, v_cache, index_k_cache, _slot, _index_slot = cache_state
        # ``msa_block_score`` has a V-cache argument for the common score API,
        # but MiniMax-M3's MSA OnlyScore path reads K only. Upstream does the
        # same logical adaptation by passing the key-only cache as a placeholder
        # V when using score-only kernels.
        index_v_cache = index_k_cache

        # --- Step 2b: Populate the independent index K cache ---
        # The sparse path bypasses Attention.forward, so it writes both the
        # independent index K cache and the main K/V cache explicitly.
        cache_reason = self._store_msa_index_k_cache(
            index_k, index_k_cache, _index_slot, index_dim
        )
        if cache_reason is not None:
            return _dense_output(cache_reason)

        # --- Step 3: Extract metadata fields ---
        main_block_tables = _get_block_table_from_metadata(main_attn_meta)
        index_block_tables = _get_block_table_from_metadata(index_attn_meta)

        seq_lens = _get_meta_field(main_attn_meta, 'seq_lens')
        if seq_lens is None:
            seq_lens = _get_meta_field(main_attn_meta, 'seq_lens_tensor')

        query_start_loc = _get_meta_field(main_attn_meta, 'query_start_loc')
        num_actual_tokens = _get_meta_field(
            main_attn_meta, 'num_actual_tokens'
        )

        # Host-side LOD/length copies precomputed once per forward by the Kunlun
        # attention metadata builder (KunlunMetadata) and shared across all 57
        # sparse layers. Reusing them in the prefill branch below avoids the
        # per-layer cumsum + .cpu()/.item() D2H syncs (the decode graph-safe
        # branch never touches these). None-safe: fall back to recompute when a
        # non-Kunlun metadata type reaches here (warmup / profiling).
        kv_lod_cpu_meta = _get_meta_field(main_attn_meta, 'kv_lod_cpu')
        kv_lod_xpu_meta = _get_meta_field(main_attn_meta, 'kv_lod_xpu')
        qsl_host_meta = _get_meta_field(
            main_attn_meta, 'query_start_loc_host'
        )
        max_query_len_meta = _get_meta_field(
            main_attn_meta, 'max_query_len'
        )
        max_kv_len_meta = _get_meta_field(main_attn_meta, 'max_kv_len')
        prefix_lens_cpu_meta = _get_meta_field(
            main_attn_meta, 'msa_prefix_lens_cpu'
        )
        prefix_lens_xpu_meta = _get_meta_field(
            main_attn_meta, 'msa_prefix_lens_xpu'
        )

        # Check decode/prefill sub-metadata (FlashInfer-style)
        if main_block_tables is None:
            decode = getattr(main_attn_meta, 'decode', None)
            if decode is not None:
                main_block_tables = _get_block_table_from_metadata(decode)
                if seq_lens is None:
                    seq_lens = _get_meta_field(decode, 'seq_lens')
        if main_block_tables is None:
            prefill = getattr(main_attn_meta, 'prefill', None)
            if prefill is not None:
                main_block_tables = _get_block_table_from_metadata(prefill)
                if seq_lens is None:
                    seq_lens = _get_meta_field(prefill, 'seq_lens')

        if main_block_tables is None:
            return _dense_output("main_block_table_missing")
        if index_block_tables is None:
            return _dense_output("index_block_table_missing")
        if main_block_tables.dim() != 2 or index_block_tables.dim() != 2:
            return _dense_output("invalid_block_table_shape")
        if main_block_tables.shape[0] != index_block_tables.shape[0]:
            return _dense_output("block_table_batch_mismatch")

        batch_num = main_block_tables.shape[0]
        main_max_blocks_per_seq = main_block_tables.shape[1]
        index_max_blocks_per_seq = index_block_tables.shape[1]

        # --- Step 4: Determine prefill vs decode ---
        # Decode always runs the capture-safe path (xpu-only LOD, fixed shapes,
        # no host-sync) so the three MSA ops can be captured into a FULL CUDA
        # graph. Under a FULL graph this forward is always pure single-token
        # decode (prefill/mixed run piecewise), so we must NOT call .item() on
        # num_actual_tokens (a D2H sync illegal during capture) — treat FULL as
        # decode directly. In eager / piecewise the .item() is fine.
        try:
            from vllm.config import CUDAGraphMode
            _fc_step4 = get_forward_context()
            is_full_cg = (
                getattr(_fc_step4, "cudagraph_runtime_mode", None)
                == CUDAGraphMode.FULL
            )
        except Exception:
            is_full_cg = False

        if is_full_cg:
            # FULL graph => single-token decode by construction.
            is_prefill = False
            is_mixed = False
            prefill_len = -1
        else:
            actual_tokens = (num_actual_tokens if num_actual_tokens is not None
                             else num_tokens)
            if isinstance(actual_tokens, torch.Tensor):
                actual_tokens = int(actual_tokens.item())
            else:
                actual_tokens = int(actual_tokens)

            # KunlunMetadata already contains the scheduler's exact partition
            # after it reorders requests as ``decode -> prefill``.  Prefer that
            # over inferring the whole batch type from ``tokens > requests``:
            # the latter cannot distinguish a mixed batch from pure prefill.
            # The current MSA LOD kernels support the mixed variable-length
            # batch directly, so mixed batches intentionally use the dynamic
            # prefill calling convention without being split into two kernel
            # launches.
            num_decodes_meta = _get_meta_field(
                main_attn_meta, "num_decodes"
            )
            num_prefills_meta = _get_meta_field(
                main_attn_meta, "num_prefills"
            )
            if num_decodes_meta is not None and num_prefills_meta is not None:
                num_decodes = int(num_decodes_meta)
                num_prefills = int(num_prefills_meta)
                if (
                    num_decodes < 0
                    or num_prefills < 0
                    or num_decodes + num_prefills != batch_num
                ):
                    return _dense_output("invalid_msa_batch_partition")

                num_decode_tokens_meta = _get_meta_field(
                    main_attn_meta, "num_decode_tokens"
                )
                num_prefill_tokens_meta = _get_meta_field(
                    main_attn_meta, "num_prefill_tokens"
                )
                if (
                    num_decode_tokens_meta is not None
                    and num_prefill_tokens_meta is not None
                    and (
                        int(num_decode_tokens_meta)
                        + int(num_prefill_tokens_meta)
                        != actual_tokens
                    )
                ):
                    return _dense_output("invalid_msa_batch_partition")

                # ``num_prefills`` follows the dense backend's configurable
                # decode threshold and may classify a uniform speculative
                # qlen>1 batch as decode.  MSA's graph-safe branch is stricter:
                # it requires exactly one query token per request.  Keep such
                # a batch on the dynamic LOD path even when metadata calls it
                # decode.
                is_prefill = num_prefills > 0 or actual_tokens > batch_num
                is_mixed = num_decodes > 0 and num_prefills > 0
            else:
                # Non-Kunlun metadata can still appear in profiling/tests.
                # Preserve the conservative legacy classification for those
                # callers only; production Kunlun metadata always takes the
                # explicit partition path above.
                is_prefill = actual_tokens > batch_num
                is_mixed = False
            prefill_len = actual_tokens if is_prefill else -1
        if is_mixed:
            stage = "mixed"
        else:
            stage = "prefill" if is_prefill else "decode"

        # Decode is always capture-safe. Prefill stays on the dynamic cpu-LOD
        # path because fixed upper-bound prefill shapes can over-allocate on
        # long or highly batched prompts.
        use_graph_safe = not is_prefill

        # --- Step 5: Build LOD tensors ---
        # Decode (capture-safe): xpu-only LOD (cpu = None), no .cpu()/.item(),
        # so the ops take LOD-branch 3 (VectorParam{nullptr,len,xpu_ptr}, no
        # D2H). Prefill uses dynamic cpu+xpu LOD (branch 1).
        device = q.device
        if use_graph_safe:
            # Decode: exactly 1 query token per sequence.
            lod_q_cpu = None
            # query_start_loc is already the device-side [0, 1, ..., B]
            # buffer for single-token decode.  Reuse it across all sparse
            # layers instead of launching torch.arange 57 times per step.
            if (
                query_start_loc is not None
                and query_start_loc.shape[0] == batch_num + 1
            ):
                lod_q_xpu = query_start_loc.to(
                    device=device, dtype=torch.int32
                )
            else:
                lod_q_xpu = torch.arange(
                    batch_num + 1, dtype=torch.int32, device=device
                )
            if seq_lens is not None:
                seq_lens_xpu = seq_lens.to(device=device, dtype=torch.int32)
                # Eager/piecewise metadata is rebuilt for every forward and
                # already owns the exact device KV LOD. Reuse it across all
                # sparse layers. FULL replay instead recomputes from its
                # graph-managed seq_lens input so lengths cannot go stale.
                if (
                    not is_full_cg
                    and kv_lod_xpu_meta is not None
                    and kv_lod_xpu_meta.shape[0] == batch_num + 1
                ):
                    lod_kv_xpu = kv_lod_xpu_meta.to(
                        device=device, dtype=torch.int32
                    )
                else:
                    lod_kv_xpu = torch.cat([
                        lod_q_xpu[:1],
                        seq_lens_xpu.cumsum(0, dtype=torch.int32),
                    ])
            else:
                seq_lens_xpu = None
                lod_kv_xpu = lod_q_xpu
            lod_kv_cpu = None
        else:
            # Prefill: dynamic cpu+xpu LOD from real query boundaries.
            if query_start_loc is None:
                return _dense_output("prefill_query_start_loc_missing")
            # Fast path: reuse the host/device LOD the metadata builder already
            # computed once for this forward (shared by all sparse layers), so
            # no per-layer cumsum or .cpu() D2H. Requires the Kunlun metadata
            # fields; otherwise recompute (fallback preserves old behavior).
            if (qsl_host_meta is not None and kv_lod_cpu_meta is not None
                    and kv_lod_xpu_meta is not None):
                lod_q_cpu = qsl_host_meta.int()
                lod_q_xpu = query_start_loc.int()
                lod_kv_cpu = kv_lod_cpu_meta.int()
                lod_kv_xpu = kv_lod_xpu_meta.int()
                seq_lens_xpu = None
            else:
                lod_q = query_start_loc.int()
                lod_q_cpu = lod_q.cpu()
                lod_q_xpu = lod_q
                if seq_lens is not None:
                    seq_lens_xpu = seq_lens.to(
                        device=q.device, dtype=torch.int32)
                    lod_kv_xpu = torch.cat([
                        lod_q_xpu[:1],
                        seq_lens_xpu.cumsum(0, dtype=torch.int32),
                    ])
                    lod_kv_cpu = lod_kv_xpu.cpu().int()
                else:
                    lod_kv_cpu = lod_q_cpu
                    lod_kv_xpu = lod_q_xpu

        # max_seqlen_q/k: decode graph-safe path uses fixed upper bounds (no
        # host-sync); prefill derives real per-batch maxima from cpu LOD.
        if use_graph_safe:
            q_lens_cpu = None
            kv_lens_cpu = None
            max_seqlen_q = 1
            # Physical KV extent addressable by the block table (constant).
            max_seqlen_k = min(
                main_max_blocks_per_seq,
                index_max_blocks_per_seq,
            ) * self.kv_cache_block_size
        else:
            q_lens_cpu = lod_q_cpu[1:] - lod_q_cpu[:-1]
            kv_lens_cpu = lod_kv_cpu[1:] - lod_kv_cpu[:-1]
            # lod_*_cpu are host tensors (reused from metadata or built via
            # .cpu() above), so these .item() calls are host ops, not D2H syncs.
            # Prefer the builder's precomputed max_query_len when available.
            if (
                max_query_len_meta is not None
                and int(max_query_len_meta) > 0
            ):
                max_seqlen_q = int(max_query_len_meta)
            else:
                # A uniform speculative qlen>1 batch may still be classified
                # as decode by the dense metadata builder, which leaves its
                # prefill-only max_query_len at zero.  MSA intentionally routes
                # that batch through dynamic LOD, so recover the real maximum.
                max_seqlen_q = (
                    int(q_lens_cpu.max().item()) if q_lens_cpu.numel() else 0
                )
            if max_kv_len_meta is not None:
                max_seqlen_k = int(max_kv_len_meta)
            else:
                max_seqlen_k = (
                    int(kv_lens_cpu.max().item())
                    if kv_lens_cpu.numel()
                    else 0
                )

        # --- Step 6: MSA kernel parameters ---
        block_size_k = self.sparse_config.get("sparse_block_size", 128)
        # page_block_size must match the actual KV cache page size (the runtime
        # cache_config.block_size, e.g. --block-size 128), NOT the sparse
        # attention logical block size and NOT CacheConfig.DEFAULT_BLOCK_SIZE
        # (16).  Resolved once at __init__ from cache_config.
        # Main and index caches are separate cache groups with independent block
        # tables. They currently share the same 128-token physical page size,
        # enforced by MiniMaxM3IndexerCache.get_kv_cache_spec, but each kernel
        # must still receive the table and table width of the cache it reads.
        main_page_block_size = self.kv_cache_block_size
        index_page_block_size = self.kv_cache_block_size
        num_index_heads = index_q.shape[1]
        topk_blocks = self.sparse_config.get("sparse_topk_blocks", 16)
        local_blocks = self.sparse_config.get("sparse_local_block", 1)
        score_type_str = self.sparse_config.get("sparse_score_type", "max")
        score_type = 0 if score_type_str == "max" else 1
        # Clamp max_seqlen_k to the physical KV-cache extent addressable by the
        # block table.  The primary value above is derived from real LOD lengths;
        # this clamp only guards malformed metadata.
        _phys_max_k = min(
            main_max_blocks_per_seq * main_page_block_size,
            index_max_blocks_per_seq * index_page_block_size,
        )
        if _phys_max_k > 0:
            max_seqlen_k = min(max_seqlen_k, _phys_max_k)
        num_score_blocks = (
            max_seqlen_k + block_size_k - 1
        ) // block_size_k

        # If every visible block fits in top-k, MSA performs exactly dense
        # causal attention but still pays for block-score, top-k and selected
        # attention.  Prefill is piecewise/eager and has real host lengths, so
        # this branch is shape-stable and safe even for uneven/prefix batches.
        # Keep the index-cache write above: later chunks/decode may cross the
        # sparse threshold and need all previously projected index keys.
        if (
            not use_graph_safe
            and getattr(self, "_msa_short_prefill_dense", False)
            and num_score_blocks <= topk_blocks
        ):
            return _dense_output("short_prefill_all_blocks")

        # The sparse attention kernel reads the paged main cache directly.
        # The all-block dense bypass above delegates this write to Attention.
        cache_reason = self._store_msa_main_kv_cache(
            k, v, k_cache, v_cache, _slot
        )
        if cache_reason is not None:
            return _dense_output(cache_reason)

        # --- Step 7: Allocate tensors for msa_block_score ---
        # Decode (capture-safe): total_q == num_tokens (buffer size, a shape not
        # a value) so no .item() D2H sync, and no dynamic [:total_q_tokens]
        # slice.
        if use_graph_safe:
            total_q_tokens = num_tokens
        else:
            total_q_tokens = int(lod_q_cpu[-1].item())

        # Guard: cudagraph warmup may produce edge-case dimensions
        if total_q_tokens <= 0 or num_score_blocks <= 0:
            return _dense_output("empty_msa_dimensions")

        # Both outputs are write-only for MiniMax-M3's score-only mode.  Valid
        # score blocks are fully overwritten; invalid blocks are masked by LOD
        # in every top-k implementation.  Avoid two large device memsets per
        # sparse layer (dummy_out alone is [tokens, 1, 128] FP32).
        score_numel = total_q_tokens * num_index_heads * num_score_blocks
        dummy_numel = total_q_tokens * num_index_heads * index_dim
        score_workspace = torch.empty(
            score_numel + dummy_numel,
            dtype=torch.float32,
            device=q.device,
        )
        score = score_workspace[:score_numel].view(
            total_q_tokens, num_index_heads, num_score_blocks
        )
        dummy_out = score_workspace[score_numel:].view(
            total_q_tokens, num_index_heads, index_dim
        )

        # --- Step 8: Call msa_block_score ---
        try:
            ret = kunlun_ops.msa_block_score(
                index_q[:total_q_tokens],
                index_k_cache, index_v_cache,
                score, dummy_out,
                index_block_tables,
                lod_q_cpu, lod_q_xpu,
                lod_kv_cpu, lod_kv_xpu,
                batch_num, prefill_len,
                max_seqlen_q, max_seqlen_k,
                num_index_heads,     # head_num (index heads)
                1,                   # head_num_kv (index KV heads = 1)
                index_dim, index_dim,    # head_dim, head_dim_v
                block_size_k, self.scaling,
                index_max_blocks_per_seq, index_page_block_size,
                None,                # sink
                score_type, False,   # use_tfloat32_gemm
            )
            if ret != 0:
                raise RuntimeError(
                    f"msa_block_score failed with ret={ret}"
                )
        except Exception:
            logger.exception(
                "[MSA:L%s] msa_block_score failed; fallback to dense",
                self.layer_idx,
            )
            return _dense_output("msa_block_score_failed")

        # --- Step 9: Select top-k blocks from index branch scores ---
        # Keep one fixed output width for the whole batch. Each query still has
        # its own causal number of visible blocks; the transform right-pads rows
        # with -1 when that number is below actual_topk, and sparse attention
        # ignores those sentinels. In particular, prefill must not use the
        # shortest sequence to truncate every other request in the batch.
        actual_topk = min(topk_blocks, num_score_blocks)
        if actual_topk <= 0:
            return _dense_output("actual_topk_le_zero")

        # Decode (capture-safe) keeps all length metadata on device. Prefill
        # uses CPU+XPU LOD because the current operator has dynamic shapes.
        if use_graph_safe:
            if (
                not is_full_cg
                and prefix_lens_xpu_meta is not None
                and prefix_lens_xpu_meta.shape[0] == batch_num
            ):
                prefix_lens_xpu = prefix_lens_xpu_meta.to(
                    device=device, dtype=torch.int32
                )
            elif seq_lens_xpu is not None:
                # decode: prefix_len = kv_len - 1 (all context minus the new token).
                prefix_lens_xpu = torch.clamp(
                    seq_lens_xpu - 1, min=0
                ).int()
            else:
                prefix_lens_xpu = torch.zeros(
                    batch_num, dtype=torch.int32, device=device
                )
            prefix_lens_cpu = None
        else:
            if (
                prefix_lens_cpu_meta is not None
                and prefix_lens_xpu_meta is not None
                and prefix_lens_cpu_meta.shape[0] == batch_num
                and prefix_lens_xpu_meta.shape[0] == batch_num
            ):
                prefix_lens_cpu = prefix_lens_cpu_meta.to(torch.int32)
                prefix_lens_xpu = prefix_lens_xpu_meta.to(
                    device=q.device, dtype=torch.int32
                )
            else:
                prefix_lens_cpu = torch.clamp(
                    kv_lens_cpu - q_lens_cpu, min=0
                ).int()
                prefix_lens_xpu = prefix_lens_cpu.to(q.device)

        try:
            # ``msa_block_score`` requires an FP32 ``out`` tensor even in
            # MiniMax-M3's score-only mode, where that output is not consumed.
            # Once block_score returns, reuse the beginning of that dead
            # workspace for the INT32 top-k output (both dtypes are 4 bytes).
            # Flatten before slicing so the resulting [T,H,K] view is compact;
            # slicing dummy_out[..., :K] would retain index_dim stride and force
            # an extra contiguous allocation.  index_dim (128) >= top-k (16) is
            # guaranteed by the current checkpoint invariant; retain the guard
            # below so a future configuration cannot alias past the workspace.
            topk_numel = total_q_tokens * num_index_heads * actual_topk
            if dummy_out.numel() < topk_numel:
                return _dense_output("msa_topk_workspace_too_small")
            topk_idx = (
                dummy_out.view(torch.int32)
                .flatten()[:topk_numel]
                .view(total_q_tokens, num_index_heads, actual_topk)
            )
            ret = kunlun_ops.msa_block_score_topk_transform(
                score,
                lod_q_cpu,
                lod_q_xpu,
                prefix_lens_cpu,
                prefix_lens_xpu,
                topk_idx,
                actual_topk,
                block_size_k,
                index_max_blocks_per_seq,
                local_blocks,
            )
            if ret != 0:
                raise RuntimeError(
                    "msa_block_score_topk_transform failed with ret="
                    f"{ret}"
                )
        except Exception:
            logger.exception(
                "[MSA:L%s] msa_block_score_topk_transform failed; "
                "fallback to dense",
                self.layer_idx,
            )
            return _dense_output("msa_topk_transform_failed")

        if topk_idx.shape[1] != self.num_kv_heads:
            if topk_idx.shape[1] == 1:
                topk_idx = topk_idx.expand(
                    -1, self.num_kv_heads, -1
                ).contiguous()
            else:
                topk_idx = topk_idx[:, :self.num_kv_heads, :].contiguous()
        # The transform writes int32 into the compact dummy_out-backed view.
        # It is already contiguous, so no dtype/layout conversion is required.
        # Invalid top-k blocks are written as -1 by
        # msa_block_score_topk_transform and ignored downstream by
        # msa_sparse_attention (per op contract), so no host-side negative-id
        # check/fallback is needed for either prefill or decode.

        # --- Step 10: Prepare full Q for sparse attention ---
        q_full = q.view(num_tokens, self.num_heads, self.head_dim)

        # --- Step 11: Allocate output and call msa_sparse_attention ---
        if use_graph_safe:
            # FULL-graph decode may contain zero-length padding rows that the
            # kernel intentionally leaves untouched.
            attn_output = torch.zeros(
                num_tokens, self.num_heads, self.head_dim,
                dtype=q.dtype, device=q.device,
            )
        else:
            # Every dynamic prefill/mixed query row is real and fully written.
            # Avoid clearing a potentially large [tokens, heads, dim] buffer.
            attn_output = torch.empty(
                num_tokens, self.num_heads, self.head_dim,
                dtype=q.dtype, device=q.device,
            )

        try:
            ret = kunlun_ops.msa_sparse_attention(
                q_full[:total_q_tokens], k_cache, v_cache,
                main_block_tables,
                lod_q_cpu, lod_q_xpu,
                lod_kv_cpu, lod_kv_xpu,
                topk_idx,
                attn_output[:total_q_tokens],
                actual_topk, block_size_k,
                batch_num,
                self.num_heads, self.num_kv_heads,
                self.head_dim, self.head_dim,       # head_dim, head_dim_v
                max_seqlen_q, max_seqlen_k,
                main_max_blocks_per_seq, main_page_block_size,
                self.scaling, prefill_len,
                False, None, None,       # unpadded_lse, softmax_lse, sink
            )
            if ret != 0:
                raise RuntimeError(
                    f"msa_sparse_attention failed with ret={ret}"
                )
            self._record_msa_debug_state(
                positions,
                path_code=_MSA_DEBUG_PATH_SPARSE,
                stage=stage,
                reason=None,
                cg_mode=cg_mode,
                total_tokens=total_q_tokens,
                batch_size=batch_num,
                topk_idx=topk_idx,
                actual_topk=actual_topk,
                block_size=block_size_k,
            )
            _log_msa_path("msa")
        except Exception:
            logger.exception(
                "[MSA:L%s] msa_sparse_attention failed; fallback to dense",
                self.layer_idx,
            )
            return _dense_output("msa_sparse_attention_failed")

        # --- Step 12: Output projection ---
        attn_output = attn_output.view(num_tokens, -1)
        output, _ = self.o_proj(attn_output)
        return output


def _get_meta_field(metadata, field_name: str):
    """Safely get a field from metadata by name."""
    return getattr(metadata, field_name, None)


def _get_attn_metadata_for_layer(
    attn_metadata: Any,
    layer_name: str,
) -> Any | None:
    """Return one cache group's metadata from ForwardContext metadata."""
    if isinstance(attn_metadata, dict):
        return attn_metadata.get(layer_name)
    if isinstance(attn_metadata, list):
        if not attn_metadata or not isinstance(attn_metadata[0], dict):
            return None
        return attn_metadata[0].get(layer_name)
    return None


def _get_block_table_from_metadata(metadata: Any) -> torch.Tensor | None:
    """Return the physical block table owned by one cache group's metadata."""
    for field_name in ("block_tables", "block_table", "block_table_tensor"):
        block_table = _get_meta_field(metadata, field_name)
        if block_table is not None:
            return block_table
    return None


def _store_msa_paged_cache_reference(
    key: torch.Tensor,
    key_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    value: torch.Tensor | None = None,
    value_cache: torch.Tensor | None = None,
) -> None:
    """Reference paged-cache scatter used only after a Kunlun write failure.

    Cache-write failures are different from score/output failures because the
    cache persists into later decode steps.  A one-step dense fallback alone
    would leave a missing or partially written index key behind.  Repair the
    touched physical slots first; if this simple reference path also fails, the
    caller raises instead of allowing a silent cache hole.
    """
    if key_cache.dim() != 4 or key.dim() != 3:
        raise ValueError(
            "MSA reference cache write expects key [T,H,D] and cache "
            "[P,H,B,D]"
        )
    if key.shape[1] != key_cache.shape[1] or key.shape[2] != key_cache.shape[3]:
        raise ValueError("MSA reference key/cache head shape mismatch")
    if (value is None) != (value_cache is None):
        raise ValueError("MSA reference value and value_cache must be paired")
    if value is not None:
        assert value_cache is not None
        if value_cache.shape != key_cache.shape or value.shape != key.shape:
            raise ValueError("MSA reference value/cache shape mismatch")

    num_tokens = min(key.shape[0], slot_mapping.shape[0])
    slots = slot_mapping[:num_tokens].to(
        device=key_cache.device, dtype=torch.int64
    )
    valid_rows = torch.nonzero(slots >= 0, as_tuple=False).flatten()
    valid_slots = slots.index_select(0, valid_rows)
    page_ids = torch.div(
        valid_slots,
        key_cache.shape[2],
        rounding_mode="floor",
    )
    page_offsets = valid_slots.remainder(key_cache.shape[2])
    key_rows = key[:num_tokens].index_select(0, valid_rows)
    key_cache[page_ids, :, page_offsets, :] = key_rows
    if value is not None:
        assert value_cache is not None
        value_rows = value[:num_tokens].index_select(0, valid_rows)
        value_cache[page_ids, :, page_offsets, :] = value_rows


def _describe_attn_metadata(metadata) -> str:
    if metadata is None:
        return "None"
    fields = []
    for name in (
        "block_table",
        "block_tables",
        "block_table_tensor",
        "seq_lens",
        "query_start_loc",
        "num_actual_tokens",
        "decode",
        "prefill",
    ):
        if hasattr(metadata, name):
            value = getattr(metadata, name)
            if isinstance(value, torch.Tensor):
                fields.append(f"{name}:Tensor{tuple(value.shape)}")
            elif value is None:
                fields.append(f"{name}:None")
            else:
                fields.append(f"{name}:{type(value).__name__}")
    return f"{type(metadata).__module__}.{type(metadata).__name__}({', '.join(fields)})"


def _get_slot_mapping_for_layer(
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]],
    layer_name: str,
) -> torch.Tensor | None:
    if isinstance(slot_mapping, dict):
        return slot_mapping.get(layer_name)
    if isinstance(slot_mapping, list):
        if not slot_mapping:
            return None
        return slot_mapping[0].get(layer_name)
    return None


class MiniMaxM3DecoderLayer(nn.Module):
    """Decoder layer with dense MLP or MoE + optional MSA sparse attention."""

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        sparse_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        if hasattr(config, "max_model_len") and isinstance(config.max_model_len, int):
            max_position_embeddings = max(
                config.max_position_embeddings, config.max_model_len
            )
        layer_idx = int(prefix.split(sep=".")[-1])

        self.layer_idx = layer_idx

        # Determine if this layer should use sparse attention
        use_sparse = False
        layer_sparse_config = None
        if sparse_config is not None:
            freq_list = sparse_config.get("sparse_attention_freq", [])
            if freq_list and layer_idx < len(freq_list):
                use_sparse = bool(freq_list[layer_idx])
            if use_sparse:
                layer_sparse_config = sparse_config

        self.self_attn = MiniMaxM3Attention(
            config=config,
            hidden_size=self.hidden_size,
            layer_idx=layer_idx,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rotary_dim=config.rotary_dim,
            rope_parameters=config.rope_parameters,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            use_sparse=use_sparse,
            sparse_config=layer_sparse_config,
        )

        # Determine dense vs MoE via moe_layer_freq
        moe_layer_freq = getattr(config, "moe_layer_freq", [])
        if moe_layer_freq:
            self.is_moe = bool(moe_layer_freq[layer_idx % len(moe_layer_freq)])
        else:
            self.is_moe = True
        self.use_gemma_norm = getattr(config, "use_gemma_norm", False)
        norm_cls = GemmaRMSNorm if self.use_gemma_norm else RMSNorm

        if self.is_moe:
            self.block_sparse_moe = MiniMaxM3MoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.block_sparse_moe = MiniMaxM3DenseMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.dense_intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = norm_cls(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self._fused_input_norm_quant = (
            self.use_gemma_norm
            and callable(
                getattr(self.input_layernorm, "forward_quantized_oot", None)
            )
            and _kunlun_prequantized_linear_enabled(self.self_attn.qkv_proj)
        )
        self._fused_dense_norm_quant = (
            not self.is_moe
            and self.use_gemma_norm
            and callable(
                getattr(self.post_attention_layernorm, "forward_quantized_oot", None)
            )
            and _kunlun_prequantized_linear_enabled(
                self.block_sparse_moe.gate_up_proj
            )
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        # Self Attention
        hidden_states_q = None
        hidden_states_max = None
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        elif self._fused_input_norm_quant:
            (
                hidden_states,
                residual,
                hidden_states_q,
                hidden_states_max,
            ) = self.input_layernorm.forward_quantized_oot(
                hidden_states, residual
            )
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            hidden_states_q=hidden_states_q,
            hidden_states_max=hidden_states_max,
        )

        # Fully Connected (dense MLP or MoE + shared expert)
        if self._fused_dense_norm_quant:
            (
                hidden_states,
                residual,
                hidden_states_q,
                hidden_states_max,
            ) = self.post_attention_layernorm.forward_quantized_oot(
                hidden_states, residual
            )
            hidden_states = self.block_sparse_moe(
                hidden_states, hidden_states_q, hidden_states_max
            )
        else:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states = self.block_sparse_moe(hidden_states)

        return hidden_states, residual


@support_torch_compile
class MiniMaxM3Model(nn.Module, EagleModelMixin):
    """Transformer backbone with embed_tokens, decoder layers, and final norm."""

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # MiniMaxM3VLConfig is multimodal — forward LLM params from text_config
        config = _ensure_text_config_attrs(config)

        self.config = config
        self._msa_debug_log = _msa_debug_enabled()
        self._msa_debug_every = _msa_debug_int(
            "MSA_DEBUG_LOG_EVERY", 1, minimum=0
        )
        self._msa_debug_all_layers = (
            os.environ.get("MSA_DEBUG_LOG_ALL_LAYERS", "0") == "1"
        )
        self._msa_debug_step = 0
        self._msa_debug_last_generations: dict[int, int] = {}
        try:
            self._msa_debug_tp_rank = get_tensor_model_parallel_rank()
        except Exception:
            self._msa_debug_tp_rank = 0

        self.vocab_size = config.vocab_size
        self.use_gemma_norm = getattr(config, "use_gemma_norm", False)
        norm_cls = GemmaRMSNorm if self.use_gemma_norm else RMSNorm

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=None,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Read sparse_attention_config from config
        sparse_config = None
        raw_sparse = getattr(config, "sparse_attention_config", None)
        if raw_sparse is not None and raw_sparse.get("use_sparse_attention", False):
            sparse_config = {
                "use_sparse_attention": raw_sparse.get("use_sparse_attention", True),
                "sparse_block_size": raw_sparse.get("sparse_block_size", 128),
                "sparse_topk_blocks": raw_sparse.get("sparse_topk_blocks", 16),
                "sparse_num_index_heads": raw_sparse.get(
                    "sparse_num_index_heads", 4),
                "sparse_score_type": raw_sparse.get("sparse_score_type", "max"),
                "sparse_attention_freq": raw_sparse.get(
                    "sparse_attention_freq", []),
                "sparse_disable_index_value": raw_sparse.get(
                    "sparse_disable_index_value", False),
                "sparse_index_dim": raw_sparse.get("sparse_index_dim", 128),
            }

        if sparse_config is None:
            configured_sparse_layers: list[int] = []
        else:
            sparse_frequency = sparse_config["sparse_attention_freq"]
            configured_sparse_layers = [
                layer_id
                for layer_id in range(config.num_hidden_layers)
                if layer_id < len(sparse_frequency)
                and bool(sparse_frequency[layer_id])
            ]
        configured_sparse_set = set(configured_sparse_layers)
        self._msa_configured_sparse_layers = configured_sparse_layers
        self._msa_configured_dense_layers = [
            layer_id
            for layer_id in range(config.num_hidden_layers)
            if layer_id not in configured_sparse_set
        ]

        if self._msa_debug_log:
            force_dense = os.environ.get("MSA_FORCE_DENSE", "0") == "1"
            logger.info(
                "[MSA][startup:config] tp_rank=%s enabled_by_model=%s "
                "force_dense=%s configured_sparse_layers=%s "
                "configured_dense_layers=%s block_size=%s topk_blocks=%s "
                "index_heads=%s index_dim=%s score_type=%s "
                "qkv_with_indexer=%s cudagraph_mode=%s",
                self._msa_debug_tp_rank,
                sparse_config is not None,
                force_dense,
                _format_msa_layer_ids(configured_sparse_layers),
                _format_msa_layer_ids(self._msa_configured_dense_layers),
                sparse_config.get("sparse_block_size", "-")
                if sparse_config is not None
                else "-",
                sparse_config.get("sparse_topk_blocks", "-")
                if sparse_config is not None
                else "-",
                sparse_config.get("sparse_num_index_heads", "-")
                if sparse_config is not None
                else "-",
                sparse_config.get("sparse_index_dim", "-")
                if sparse_config is not None
                else "-",
                sparse_config.get("sparse_score_type", "-")
                if sparse_config is not None
                else "-",
                _qkv_with_inder_enabled(),
                vllm_config.compilation_config.cudagraph_mode,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MiniMaxM3DecoderLayer(
                config,
                prefix,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                sparse_config=sparse_config,
            ),
            prefix=f"{prefix}.layers",
        )

        if self._msa_debug_log:
            force_dense = os.environ.get("MSA_FORCE_DENSE", "0") == "1"
            effective_msa_layers = (
                [] if force_dense else self._msa_configured_sparse_layers
            )
            effective_dense_layers = sorted(
                set(self._msa_configured_dense_layers)
                | (
                    set(self._msa_configured_sparse_layers)
                    if force_dense
                    else set()
                )
            )
            logger.info(
                "[MSA][startup:layers] tp_rank=%s local_layer_range=%s-%s "
                "effective_msa_layers=%s effective_dense_layers=%s "
                "indexer=%s index_cache=%s",
                self._msa_debug_tp_rank,
                self.start_layer,
                self.end_layer - 1,
                _format_msa_layer_ids(effective_msa_layers),
                _format_msa_layer_ids(effective_dense_layers),
                "skipped"
                if force_dense or not self._msa_configured_sparse_layers
                else "built",
                "skipped"
                if force_dense or not self._msa_configured_sparse_layers
                else "registered",
            )
            logger.info(
                "[MSA][startup:debug] runtime_summary_every=%s "
                "block_detail_layers=%s max_logged_tokens_per_layer=%s "
                "note=logical_block_ids_and_device_sync_not_for_benchmark",
                self._msa_debug_every,
                "all"
                if self._msa_debug_all_layers
                else (
                    str(self._msa_configured_sparse_layers[0])
                    if self._msa_configured_sparse_layers
                    else "-"
                ),
                _msa_debug_int("MSA_DEBUG_LOG_MAX_TOKENS", 32, minimum=1),
            )

        if get_pp_group().is_last_rank:
            self.norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> (
        torch.Tensor
        | IntermediateTensors
        | tuple[torch.Tensor, list[torch.Tensor]]
    ):
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state(
            [], self.start_layer, hidden_states, residual
        )
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    @staticmethod
    def _format_msa_block_snapshot(snapshot: dict[str, Any]) -> str:
        row_start = snapshot["first_stored_row"]
        token_entries: list[str] = []
        for offset, (position, token_topk) in enumerate(
            zip(snapshot["positions"], snapshot["topk"])
        ):
            head_entries = [
                f"h{head_id}={blocks}"
                for head_id, blocks in enumerate(token_topk)
            ]
            if len(head_entries) == 1:
                selected = head_entries[0].split("=", 1)[1]
            else:
                selected = "{" + ",".join(head_entries) + "}"
            token_entries.append(
                f"row={row_start + offset}:pos={position}:blocks={selected}"
            )
        return " | ".join(token_entries) if token_entries else "-"

    def log_msa_debug_state(self) -> None:
        """Emit one host-side runtime summary after model graph completion.

        This method must stay outside ``forward``.  In FULL CUDA Graph mode it
        observes buffer writes made by the latest replay, which is what makes
        decode logs describe real inference rather than capture-time tensors.
        """
        if not self._msa_debug_log:
            return
        self._msa_debug_step += 1
        if self._msa_debug_every <= 0:
            return
        if (self._msa_debug_step - 1) % self._msa_debug_every != 0:
            return

        local_sparse_attentions: list[MiniMaxM3Attention] = []
        for layer in self.layers[self.start_layer : self.end_layer]:
            attention = getattr(layer, "self_attn", None)
            if (
                isinstance(attention, MiniMaxM3Attention)
                and attention.use_sparse
            ):
                local_sparse_attentions.append(attention)

        if not local_sparse_attentions:
            logger.info(
                "[MSA][runtime] step=%s tp_rank=%s stage=unknown cg=UNKNOWN "
                "attention=dense reason=no_sparse_layers",
                self._msa_debug_step,
                self._msa_debug_tp_rank,
            )
            return

        first_detail_layer = local_sparse_attentions[0].layer_idx
        snapshots: list[dict[str, Any]] = []
        stale_layers: list[int] = []
        for attention in local_sparse_attentions:
            include_blocks = (
                self._msa_debug_all_layers
                or attention.layer_idx == first_detail_layer
            )
            snapshot = attention._get_msa_debug_snapshot(
                include_blocks=include_blocks
            )
            if snapshot is None:
                continue
            previous_generation = self._msa_debug_last_generations.get(
                attention.layer_idx
            )
            if previous_generation == snapshot["generation"]:
                stale_layers.append(attention.layer_idx)
            self._msa_debug_last_generations[attention.layer_idx] = snapshot[
                "generation"
            ]
            snapshot["include_blocks"] = include_blocks
            snapshots.append(snapshot)

        if not snapshots:
            logger.info(
                "[MSA][runtime] step=%s tp_rank=%s stage=unknown cg=UNKNOWN "
                "attention=unknown reason=trace_buffers_not_written",
                self._msa_debug_step,
                self._msa_debug_tp_rank,
            )
            return

        msa_layers = [
            snapshot["layer"]
            for snapshot in snapshots
            if snapshot["path"] == "msa"
        ]
        runtime_dense_layers = [
            snapshot["layer"]
            for snapshot in snapshots
            if snapshot["path"] == "dense"
        ]
        unset_layers = [
            snapshot["layer"]
            for snapshot in snapshots
            if snapshot["path"] == "unset"
        ]

        fallback_by_reason: dict[str, list[int]] = {}
        for snapshot in snapshots:
            if snapshot["path"] != "dense":
                continue
            reason = snapshot["reason"] or "unspecified"
            fallback_by_reason.setdefault(reason, []).append(snapshot["layer"])
        fallback_summary = ";".join(
            f"{reason}:{_format_msa_layer_ids(layer_ids)}"
            for reason, layer_ids in sorted(fallback_by_reason.items())
        ) or "-"

        stages = sorted({snapshot["stage"] for snapshot in snapshots})
        cg_modes = sorted({snapshot["cg_mode"] for snapshot in snapshots})
        stage = stages[0] if len(stages) == 1 else f"mixed({','.join(stages)})"
        cg_mode = (
            cg_modes[0]
            if len(cg_modes) == 1
            else f"mixed({','.join(cg_modes)})"
        )
        first_snapshot = snapshots[0]
        reasons = {snapshot["reason"] for snapshot in snapshots}
        if reasons & {"main_kv_cache_empty", "index_cache_empty"}:
            phase = "profile_or_cache_init"
        elif stage == "unknown" or stage.startswith("mixed("):
            phase = "startup_or_fallback"
        elif cg_mode == "FULL":
            phase = "decode_graph"
        else:
            phase = "inference"
        all_dense_layers = sorted(
            set(self._msa_configured_dense_layers)
            | set(runtime_dense_layers)
        )
        if msa_layers and all_dense_layers:
            effective_attention = "mixed(dense+msa)"
        elif msa_layers:
            effective_attention = "msa"
        else:
            effective_attention = "dense"

        logger.info(
            "[MSA][runtime] step=%s tp_rank=%s phase=%s stage=%s cg=%s "
            "attention=%s tokens=%s batch=%s configured_dense_layers=%s "
            "msa_layers=%s runtime_dense_layers=%s fallback=%s "
            "unset_layers=%s stale_layers=%s",
            self._msa_debug_step,
            self._msa_debug_tp_rank,
            phase,
            stage,
            cg_mode,
            effective_attention,
            first_snapshot["total_tokens"],
            first_snapshot["batch_size"],
            _format_msa_layer_ids(self._msa_configured_dense_layers),
            _format_msa_layer_ids(msa_layers),
            _format_msa_layer_ids(runtime_dense_layers),
            fallback_summary,
            _format_msa_layer_ids(unset_layers),
            _format_msa_layer_ids(stale_layers),
        )

        for snapshot in snapshots:
            if (
                snapshot["path"] != "msa"
                or not snapshot["include_blocks"]
            ):
                continue
            logger.info(
                "[MSA][blocks][L%s] step=%s tp_rank=%s stage=%s "
                "block_size=%s topk=%s stored_tokens=%s total_tokens=%s "
                "truncated=%s logical_selections=%s",
                snapshot["layer"],
                self._msa_debug_step,
                self._msa_debug_tp_rank,
                snapshot["stage"],
                snapshot["block_size"],
                snapshot["actual_topk"],
                snapshot["stored_tokens"],
                snapshot["total_tokens"],
                snapshot["truncated"],
                self._format_msa_block_snapshot(snapshot),
            )

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return FusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".qkv_proj", ".index_q_proj", "index_q"),
            (".qkv_proj", ".index_k_proj", "index_k"),
            # Dense MLP: pack gate_proj + up_proj → gate_up_proj
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        expert_params_mapping = self.get_expert_mapping()

        params_dict = dict(self.named_parameters())
        modules_dict = dict(self.named_modules())
        qkv_with_inder = _qkv_with_inder_enabled()
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            # Remap checkpoint "mlp" -> model "block_sparse_moe" (dense + MoE layers)
            name = name.replace("mlp.", "block_sparse_moe.")
            # Remap per-head QK-norm: checkpoint q_norm.weight -> param q_norm_weight
            name = name.replace("self_attn.q_norm.weight", "self_attn.q_norm_weight")
            name = name.replace("self_attn.k_norm.weight", "self_attn.k_norm_weight")
            # Skip RoPE buffer
            if "rotary_emb.inv_freq" in name:
                continue
            # --- MSA index weights ---
            # Checkpoint names like:
            #   model.layers.N.self_attn.index_q_proj.weight
            #   model.layers.N.self_attn.index_k_proj.weight
            #   model.layers.N.self_attn.index_q_norm.weight
            #   model.layers.N.self_attn.index_k_norm.weight
            #
            # Actual norm params live under indexer submodule:
            #   layers.N.self_attn.indexer.index_q_norm        (bare nn.Parameter)
            # Projection params either fold into the 5-way self_attn.qkv_proj
            # (VLLM_KUNLUN_QKV_WITH_INDER=1) or load into
            # self_attn.indexer.index_qk_proj (env=0).
            if "self_attn.index_" in name:
                # In MSA_FORCE_DENSE mode the indexer submodule and 5-way qkv
                # projection were never built, so index checkpoint weights have
                # no destination param. Skip them silently instead of emitting a
                # per-layer warning.
                if os.environ.get("MSA_FORCE_DENSE", "0") == "1":
                    continue
                is_index_proj = (
                    "self_attn.index_q_proj." in name
                    or "self_attn.index_k_proj." in name
                )
                if is_index_proj:
                    if "self_attn.index_q_proj." in name:
                        shard_id = "index_q"
                        separate_name = name.replace(
                            "self_attn.index_q_proj",
                            "self_attn.indexer.index_qk_proj",
                        )
                        fused_name = name.replace(
                            "self_attn.index_q_proj",
                            "self_attn.qkv_proj",
                        )
                    else:
                        shard_id = "index_k"
                        separate_name = name.replace(
                            "self_attn.index_k_proj",
                            "self_attn.indexer.index_qk_proj",
                        )
                        fused_name = name.replace(
                            "self_attn.index_k_proj",
                            "self_attn.qkv_proj",
                        )

                    if not qkv_with_inder:
                        module_name = separate_name.rsplit(".", 1)[0]
                        module = modules_dict.get(module_name)
                        if (
                            separate_name in params_dict
                            and isinstance(module, MiniMaxM3IndexQKParallelLinear)
                        ):
                            param = params_dict[separate_name]
                            weight_loader = param.weight_loader
                            weight_loader(param, loaded_weight, shard_id)
                            loaded_params.add(name)
                        continue

                    module_name = fused_name.rsplit(".", 1)[0]
                    module = modules_dict.get(module_name)
                    if not isinstance(
                        module, MiniMaxM3QKVParallelLinearWithIndexer
                    ):
                        continue
                    # 5-way projection params fold into self_attn.qkv_proj
                    # through stacked_params_mapping below.

                if not is_index_proj:
                    # Checkpoint self_attn.index_* → param name
                    # self_attn.indexer.index_*
                    wname = name.replace("self_attn.index_",
                                         "self_attn.indexer.index_")
                    # Norm params are bare nn.Parameter (no .weight suffix)
                    if wname not in params_dict and wname.endswith(".weight"):
                        wname_stripped = wname[:-7]
                        if wname_stripped in params_dict:
                            wname = wname_stripped
                    if wname in params_dict:
                        param = params_dict[wname]
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        weight_loader(param, loaded_weight)
                        loaded_params.add(name)
                    else:
                        logger.warning_once(
                            "[MSA] index weight %s not in params_dict "
                            "(tried %s)",
                            name,
                            wname,
                        )
                    continue
            # Skip spec decode (MTP) layers
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Skip expert weights (handled by expert_params_mapping below)
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                # Skip shared_experts weights (unpacked gate/up/down)
                if "shared_experts." in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                # Skip auxiliary quantized keys not in params_dict
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if (
                    "block_sparse_moe.shared_experts." in name
                    and _kunlun_fused_shared_expert_enabled(self.config)
                ):
                    shard_id = None
                    if ".gate_proj." in name:
                        shard_id = "w1"
                        param_name = name.replace(
                            ".shared_experts.gate_proj.",
                            ".experts.w13_",
                        )
                    elif ".up_proj." in name:
                        shard_id = "w3"
                        param_name = name.replace(
                            ".shared_experts.up_proj.",
                            ".experts.w13_",
                        )
                    elif ".down_proj." in name:
                        shard_id = "w2"
                        param_name = name.replace(
                            ".shared_experts.down_proj.",
                            ".experts.w2_",
                        )

                    if shard_id is not None and param_name in params_dict:
                        param = params_dict[param_name]
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            param_name,
                            shard_id=shard_id,
                            expert_id=self.config.num_local_experts,
                        )
                        loaded_params.add(name)
                    continue

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name, self):
                        continue

                    if name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    # Skip auxiliary quantized keys not in params_dict
                    if name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

        # Kunlun's fused norm+quant kernel implements plain RMSNorm.  Cache the
        # effective Gemma weight (1 + checkpoint weight) after all raw norm
        # weights have been loaded so forward does not launch a pointwise add.
        for module in self.modules():
            refresh_fused_weight = getattr(module, "refresh_fused_weight", None)
            if callable(refresh_fused_weight):
                refresh_fused_weight()

        if self._msa_debug_log:
            loaded_index_weights = sum(
                "self_attn.index_" in name for name in loaded_params
            )
            logger.info(
                "[MSA][startup:weights] tp_rank=%s loaded_parameters=%s "
                "loaded_index_parameters=%s index_weights=%s",
                self._msa_debug_tp_rank,
                len(loaded_params),
                loaded_index_weights,
                "skipped_by_force_dense"
                if os.environ.get("MSA_FORCE_DENSE", "0") == "1"
                else "loaded",
            )

        return loaded_params


class MiniMaxM3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle3):
    """MiniMax-M3 causal language model with MSA sparse attention support.

    Checkpoint weight prefix: ``language_model.model.layers.*``,
    ``language_model.lm_head.weight``, etc.  The ``language_model.`` prefix
    is stripped in ``load_weights`` so the inner model uses the standard
    ``model.layers.*`` naming.
    """

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        # MiniMaxM3VLConfig is multimodal — forward LLM params from text_config
        config = _ensure_text_config_attrs(config)
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        if hasattr(vllm_config.model_config, "max_model_len"):
            self.config.max_model_len = vllm_config.model_config.max_model_len
        self.model = MiniMaxM3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size, quant_config=None
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> (
        torch.Tensor
        | IntermediateTensors
        | tuple[torch.Tensor, list[torch.Tensor]]
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.model._msa_debug_log:
            try:
                self.model.log_msa_debug_state()
            except Exception:
                # Debug observability must not turn a successful inference into
                # a failed request.  Keep the traceback in the service log.
                logger.exception("[MSA][runtime] failed to drain debug trace")
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Strip the ``language_model.`` prefix from checkpoint weight names.
        def _is_vision_weight(name: str) -> bool:
            skip_prefixes = [
                "multi_modal_projector.", "vision_tower.", "patch_merge_mlp.",
            ]
            return any(name.startswith(p) for p in skip_prefixes)

        def _strip_prefix(name: str, tensor: torch.Tensor):
            if name.startswith("language_model."):
                name = name[len("language_model."):]
            return name, tensor

        loader = AutoWeightsLoader(self)
        filtered = ((n, t) for n, t in weights if not _is_vision_weight(n))
        return loader.load_weights(
            (_strip_prefix(n, t) for n, t in filtered)
        )

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


def get_spec_layer_idx_from_weight_name(
    config: PretrainedConfig, weight_name: str
) -> int | None:
    if hasattr(config, "num_mtp_modules") and (config.num_mtp_modules > 0):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_mtp_modules):
            if weight_name.startswith(f"model.layers.{layer_idx + i}."):
                return layer_idx + i
    return None
