#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
# Author: Wang Hao
# Email: wanghao129@baidu.com
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
"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023).
Punica: Multi-Tenant LoRA Serving.
https://arxiv.org/abs/2310.18547
"""

# SPDX-License-Identifier: Apache-2.0
from typing import Callable, Optional, Tuple, Union

import torch
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase

from vllm_kunlun.lora.ops.kunlun_ops import (
    bgmv_expand,
    bgmv_expand_slice,
    bgmv_shrink,
    sgmv_expand,
    sgmv_expand_slice,
    sgmv_shrink,
)

# Disable torchdynamo for all functions in this file


# The platforms that are compatible with the PyTorch-native implementation can
# inherit this class
class PunicaWrapperKunlun(PunicaWrapperBase):
    """
    PunicaWrapperKunlun with moe_fc
    """

    def __init__(
        self,
        max_num_batched_tokens: int,
        max_batches: int,
        device: Union[torch.device, str],
        **kwargs,
    ):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches, device)
        self._zero_index_buffers()
        self._single_lora_slot: Optional[int] = None

    def update_metadata(
        self,
        mapping,
        lora_index_to_id: list,
        max_loras: int,
        vocab_size: int,
        **kwargs,
    ):
        self._single_lora_slot = self._compute_single_lora_slot(
            mapping, lora_index_to_id
        )
        super().update_metadata(
            mapping, lora_index_to_id, max_loras, vocab_size, **kwargs
        )

    @staticmethod
    def _compute_single_lora_slot(mapping, lora_index_to_id: list) -> Optional[int]:
        """The one slot this step uses, or ``None`` if that is not well defined."""
        index_mapping = getattr(mapping, "index_mapping", None)
        if not index_mapping:
            return None
        ids = {i for i in set(index_mapping) if i > 0}
        if len(ids) != 1:
            return None
        try:
            return lora_index_to_id.index(next(iter(ids)))
        except ValueError:
            # The id has not been mapped to a slot (should not happen; the
            # activation precedes the step).  bgmv is always correct.
            return None

    @property
    def single_lora_slot(self) -> Optional[int]:
        """Slot number when this step has exactly one active adapter, else None."""
        return self._single_lora_slot

    def _zero_index_buffers(self) -> None:
        """Zero the persistent index buffers ``PunicaWrapperBase`` leaves empty."""
        for name in (
            "_token_lora_indices",
            "_sampler_indices",
            "_sampler_indices_padded",
            "_embeddings_indices",
            "_seq_start_locs",
            "_seq_lengths",
            "_lora_indices_per_batch",
        ):
            buffer = getattr(self, name, None)
            if isinstance(buffer, torch.Tensor):
                buffer.zero_()

    @staticmethod
    def _moe_grouping_placeholders(x: torch.Tensor):
        """Build the MoE grouping arguments the Kunlun sgmv/bgmv ops ignore."""
        expert_num = 9
        block_statistic = torch.zeros(
            [12, expert_num], dtype=torch.int32, device=x.device
        )
        sorted_tokens_num_lod = torch.zeros(
            expert_num + 1, dtype=torch.int32, device=x.device
        )
        moe_index = torch.zeros(x.size(0), dtype=torch.int32, device=x.device)
        return block_statistic, sorted_tokens_num_lod, moe_index

    def _logits_indices(self, rows: int) -> torch.Tensor:
        """Return one adapter index per row of the logits the sampler will see."""
        indices = self.sampler_indices
        if indices.numel() == rows:
            return indices
        token_indices = self.token_lora_indices
        if token_indices.numel() >= rows:
            return token_indices[:rows]
        if indices.numel() == 0:
            return torch.full((rows,), -1, dtype=torch.long, device=self.device)
        return indices[:1].expand(rows)

    def _mask_no_lora_rows(
        self, buffer: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Zero the rows of a token-major buffer that carry no adapter."""
        rows = min(buffer.size(0), indices.size(0))
        if rows == 0:
            return buffer
        buffer[:rows].masked_fill_(indices[:rows].unsqueeze(1) < 0, 0)
        return buffer

    def _shrink_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        block_statistic: torch.Tensor,
        sorted_tokens_num_lod: torch.Tensor,
        moe_index: torch.Tensor,
        scale: float,
    ):

        expert_m = torch.zeros(9, dtype=torch.int32, device=x.device)

        sgmv_shrink(
            x,
            w_t_all,
            y,
            block_statistic,
            sorted_tokens_num_lod,
            moe_index,
            expert_m,
            *self.prefill_metadata,
            scale,
        )

    def _shrink_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        block_statistic: torch.Tensor,
        sorted_tokens_num_lod: torch.Tensor,
        moe_index: torch.Tensor,
        scale: float,
    ):

        expert_m = torch.zeros(9, dtype=torch.int32, device=x.device)
        bgmv_shrink(
            x,
            w_t_all,
            y,
            block_statistic,
            sorted_tokens_num_lod,
            moe_index,
            expert_m,
            self.token_lora_indices,
            scale,
        )

    def _expand_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        block_statistic: torch.Tensor,
        sorted_tokens_num_lod: torch.Tensor,
        moe_index: torch.Tensor,
        add_inputs: bool,
    ):

        sgmv_expand(
            x,
            w_t_all,
            y,
            block_statistic,
            sorted_tokens_num_lod,
            moe_index,
            *self.prefill_metadata,
            add_inputs,
        )

    def _expand_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        block_statistic: torch.Tensor,
        sorted_tokens_num_lod: torch.Tensor,
        moe_index: torch.Tensor,
        add_inputs: bool,
    ):
        bgmv_expand(
            x,
            w_t_all,
            y,
            block_statistic,
            sorted_tokens_num_lod,
            moe_index,
            self.token_lora_indices,
            add_inputs,
        )

    def _expand_slice_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        block_statistic,
        sorted_tokens_num_lod: torch.Tensor,
        moe_index: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        add_inputs: bool,
    ):

        normed_scale = torch.ones([y.size(0), 1], dtype=torch.float32, device=x.device)

        sgmv_expand_slice(
            x,
            w_t_all,
            y,
            block_statistic,
            sorted_tokens_num_lod,
            moe_index,
            normed_scale,
            *self.prefill_metadata,
            y_offset,
            y_slice_size,
            add_inputs,
        )

    def _expand_slice_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        block_statistic: torch.Tensor,
        sorted_tokens_num_lod: torch.Tensor,
        moe_index: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        add_inputs: bool,
    ):

        normed_scale = torch.ones([y.size(0), 1], dtype=torch.float32, device=x.device)

        bgmv_expand_slice(
            x,
            w_t_all,
            y,
            block_statistic,
            sorted_tokens_num_lod,
            moe_index,
            normed_scale,
            self.token_lora_indices,
            y_offset,
            y_slice_size,
            add_inputs,
        )

    def _apply_expand(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        block_statistic,
        sorted_tokens_num_lod: torch.Tensor,
        moe_index: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        add_inputs: bool = True,
    ):
        """
        Perform the ` y[:,y_offset:y_offset+y_slice_size]+=x@w_t_all`
        computation, which is suitable for the
        GEMM of lora'b.
        """

        expand_slice_fun: Callable = (
            self._expand_slice_prefill if self.is_prefill else self._expand_slice_decode
        )
        expand_slice_fun(
            y,
            x,
            w_t_all,
            block_statistic,
            sorted_tokens_num_lod,
            moe_index,
            y_offset,
            y_slice_size,
            add_inputs,
        )

    def _apply_shrink(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        block_statistic: torch.Tensor,
        sorted_tokens_num_lod: torch.Tensor,
        moe_index: torch.Tensor,
        scale: float,
    ):
        """
        Perform the ` y+=x@w_t_all` computation, which is suitable for the
        GEMM of lora'a.
        When `is_prefill is` true, it indicates that it is currently the
        prefill stage, and the `_shrink_prefill` function should be called.
        Otherwise, it is the decode stage, and the _shrink_decode function
        should be called.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])

        shrink_fun: Callable = (
            self._shrink_prefill if self.is_prefill else self._shrink_decode
        )

        shrink_fun(
            y, x, w_t_all, block_statistic, sorted_tokens_num_lod, moe_index, scale
        )

        y = y.view_as(y_org)

    def add_shrink(
        self,
        y: Union[Tuple[torch.Tensor, ...], torch.Tensor],
        x: torch.Tensor,
        lora_a_stacked: Tuple[torch.Tensor, ...],
        block_statistic: torch.Tensor,
        sorted_tokens_num_lod: torch.Tensor,
        moe_index: torch.Tensor,
        scale: float,
        **kwargs,
    ):
        """
        Performs GEMM  for multiple slices of lora_a.
        When `is_prefill is` true, it indicates that it is currently the
        prefill stage, and the `_shrink_prefill` function should be called.
        Otherwise, it is the decode stage, and the _shrink_decode function
        should be called.

        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale

        Args:
            y (Union[Tuple[torch.Tensor, ...], torch.Tensor]): Output tensors
            x (torch.Tensor): Input tensor
            lora_a_stacked (Tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation
        """

        x = x.view(-1, x.shape[-1])

        for slice_idx in range(len(lora_a_stacked)):  # Each slice represents a layer

            self._apply_shrink(
                y[slice_idx],
                x,
                lora_a_stacked[slice_idx],
                block_statistic,
                sorted_tokens_num_lod,
                moe_index,
                scale,
            )

    def add_expand(
        self,
        y: torch.Tensor,
        x: Union[Tuple[torch.Tensor, ...], torch.Tensor],
        lora_b_stacked: Tuple[torch.Tensor, ...],
        block_statistic: torch.Tensor,
        sorted_tokens_num_lod: torch.Tensor,
        moe_index: torch.Tensor,
        lora_bias_stacked: Optional[Tuple[torch.Tensor, ...]],
        output_slices: Tuple[int, ...],
        offset_start: int = 0,
        add_inputs=True,
        **kwargs,
    ) -> None:
        """
        Performs GEMM and bias addition for multiple slices of lora_b.

        Semantics:
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i] +
                    lora_bias_stacked[i]
                offset += slice

        Args:
            y (torch.Tensor): Output tensor.
            x (Union[Tuple[torch.Tensor, ...], torch.Tensor]): Input tensors
            lora_b_stacked (Tuple[torch.Tensor, ...]): lora_b's weight
            lora_bias_stacked (Optional[Tuple[torch.Tensor, ...]]):
                bias's weight
            output_slices (Tuple[int, ...]): Every slice's size
            add_inputs (bool):  Defaults to True.
        """

        y_org = y
        y = y.view(-1, y.shape[-1])
        offset_left = offset_start

        if lora_bias_stacked is not None:
            self._apply_bias(
                self.token_lora_indices, y, output_slices, lora_bias_stacked
            )

        for slice_idx in range(len(lora_b_stacked)):
            self._apply_expand(
                y,
                x[slice_idx],
                lora_b_stacked[slice_idx],
                block_statistic,
                sorted_tokens_num_lod,
                moe_index,
                offset_left,
                output_slices[slice_idx],
                add_inputs=add_inputs,
            )
            offset_left += output_slices[slice_idx]

        y = y.view_as(y_org)

    def add_lora_embedding(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        add_inputs: bool = True,
        **kwargs,
    ) -> None:
        """
        Applies lora  specifically for VocabParallelEmbeddingWithLoRA.

        Semantics:
            y += x @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            add_inputs (bool): Default to True.
        """

        if self.no_lora:
            return

        expand_fun: Callable = (
            self._expand_prefill if self.is_prefill else self._expand_decode
        )
        (
            block_statistic,
            sorted_tokens_num_lod,
            moe_index,
        ) = self._moe_grouping_placeholders(x)

        # lora_b_stacked arrives as (max_loras, 1, embedding_dim, rank); the
        # kernels expect the 3D layout that add_lora_linear also passes.
        if lora_b_stacked.dim() == 4:
            lora_b_stacked = lora_b_stacked.squeeze(1)

        # x holds the lora_a embedding lookup per token; masking it here keeps
        # base-model tokens in a mixed batch from receiving an adapter delta.
        x = self._mask_no_lora_rows(x.clone(), self.token_lora_indices)

        expand_fun(
            y,
            x,
            lora_b_stacked,
            block_statistic,
            sorted_tokens_num_lod,
            moe_index,
            add_inputs,
        )

    def add_lora_linear(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: Tuple[torch.Tensor, ...],
        lora_b_stacked: Tuple[torch.Tensor, ...],
        scale: float,
        output_slices: Tuple[int, ...],
        *,
        buffer: Optional[Tuple[torch.Tensor, ...]] = None,
        **kwargs,
    ) -> None:
        """
        Applicable to linear-related lora.

        Semantics:
            for i in range(len(lora_a_stacked)):
                y[i] += (
                    x[i].unsqueeze(0)
                    @ lora_a_stacked[indices[i], layer_idx, :, :]
                    @ lora_b_stacked[indices[i], layer_idx, :, :]
                    * scale
                    ).squeeze(0)

        Args:
            y (torch.Tensor): Output tensor. Will be changed in-place.
            x (torch.Tensor): Input tensor
            lora_a_stacked (Tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (Tuple[torch.Tensor, ...]): lora_b's weight.
            scale (float): Scaling factor.
            output_slices (Tuple[int, ...]): Every slice's size.
            buffer (Optional[Tuple[torch.Tensor, ...]]): Defaults to None.
        """
        # Get lora_bias_stacked from kwargs (if present)
        lora_bias_stacked: Optional[Tuple[torch.Tensor, ...]] = kwargs.get(
            "lora_bias_stacked", None
        )

        if self.no_lora:
            return

        (
            block_statistic,
            sorted_tokens_num_lod,
            moe_index,
        ) = self._moe_grouping_placeholders(x)

        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)
        if lora_bias_stacked is not None:
            assert len(lora_bias_stacked) == len(output_slices)
            y = self._apply_bias(
                self.token_lora_indices, y, output_slices, lora_bias_stacked
            )

        if buffer is None:
            r = lora_b_stacked[0].size(-1)
            buffer = tuple(
                torch.zeros((x.size(0), r), dtype=torch.float16, device=x.device)
                for _ in range(len(output_slices))
            )
        # [tensor.squeeze_(1) for tensor in lora_a_stacked]
        new_lora_a_stacked = tuple(lora_a.squeeze(1) for lora_a in lora_a_stacked)
        self.add_shrink(
            buffer,
            x,
            new_lora_a_stacked,
            block_statistic,
            sorted_tokens_num_lod,
            moe_index,
            scale,
            **kwargs,
        )
        for buf in buffer:
            self._mask_no_lora_rows(buf, self.token_lora_indices)
        # [tensor.unsqueeze_(1) for tensor in lora_a_stacked]

        # [tensor.squeeze_(1) for tensor in lora_b_stacked]
        new_lora_b_stacked = tuple(lora_b.squeeze(1) for lora_b in lora_b_stacked)
        self.add_expand(
            y,
            buffer,
            new_lora_b_stacked,
            block_statistic,
            sorted_tokens_num_lod,
            moe_index,
            None,
            output_slices,
            add_inputs=True,
            **kwargs,
        )
        # [tensor.unsqueeze_(1) for tensor in lora_b_stacked]

    def add_lora_logits(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        scale,
        *,
        buffer: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """
        Applies lora  specifically for LogitsProcessorWithLoRA.

        Semantics:
            buffer = (x @ lora_a_stacked) * scale
            y += buffer @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_a_stacked (torch.Tensor): lora_a's weights.
            lora_b_stacked (torch.Tensor):lora_b's weights.
            scale (float): Scaling factor.
            buffer (Optional[torch.Tensor]):Default to None.
        """
        if self.no_lora:
            return

        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])

        if lora_a_stacked.dim() == 2:
            lora_a_stacked = lora_a_stacked.unsqueeze(0)
        if lora_b_stacked.dim() == 2:
            lora_b_stacked = lora_b_stacked.unsqueeze(0)

        # lora_a_stacked is (max_loras, 1, rank, hidden_size) and lora_b_stacked
        # is (max_loras, 1, vocab_size, rank), which is exactly the layout the
        # bgmv kernels accept -- so the rank is lora_b's last dim, not lora_a's.
        r = lora_b_stacked.size(-1)

        if buffer is None:
            buffer = torch.zeros((x.size(0), r), dtype=torch.float32, device=x.device)

        indices = self._logits_indices(x.size(0))
        # ``lora_ops`` clamps negative slots itself, so only an out-of-range
        # positive slot needs fixing here; keep ``indices`` unclamped so the
        # masking below can still recognize the rows that carry no adapter.
        kernel_indices = indices
        if indices.numel() and int(indices.max()) >= lora_a_stacked.size(0):
            kernel_indices = torch.clamp(indices, 0, lora_a_stacked.size(0) - 1)

        lora_a_reshaped = lora_a_stacked
        lora_b_reshaped = lora_b_stacked

        (
            block_statistic,
            sorted_tokens_num_lod,
            moe_index,
        ) = self._moe_grouping_placeholders(x)
        expert_m = torch.zeros(9, dtype=torch.int32, device=x.device)

        bgmv_shrink(
            x,
            lora_a_reshaped,
            buffer,
            block_statistic,
            sorted_tokens_num_lod,
            moe_index,
            expert_m,
            kernel_indices,
            scale,
        )
        # Rows whose index is -1 belong to requests served without an adapter;
        # zero them so the expand below leaves the base logits intact.
        self._mask_no_lora_rows(buffer, indices)
        bgmv_expand(
            buffer,
            lora_b_reshaped,
            y,
            block_statistic,
            sorted_tokens_num_lod,
            moe_index,
            kernel_indices,
            add_inputs=True,
        )

        y = y.view_as(y_org)
