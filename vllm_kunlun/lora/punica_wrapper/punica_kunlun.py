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

from typing import TYPE_CHECKING, Optional, Union, final

import torch
# Disable torchdynamo for all functions in this file
torch._dynamo.config.disable = True


# SPDX-License-Identifier: Apache-2.0
from typing import Callable, Optional, Tuple, Union


from vllm_kunlun.lora.ops.kunlun_ops import (
    bgmv_expand,
    bgmv_expand_slice,
    bgmv_shrink,
    sgmv_expand,
    sgmv_expand_slice,
    sgmv_shrink,
)

from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase


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

        expand_fun: Callable = (
            self._expand_prefill if self.is_prefill else self._expand_decode
        )
        expand_fun(y, x, lora_b_stacked, add_inputs)

    def add_lora_linear(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: Tuple[torch.Tensor, ...],
        lora_b_stacked: Tuple[torch.Tensor, ...],
        lora_bias_stacked: Optional[Tuple[torch.Tensor, ...]],
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
                    ).squeeze(0)+lora_bias_stacked[i]

        Args:
            y (torch.Tensor): Output tensor. Will be changed in-place.
            x (torch.Tensor): Input tensor
            lora_a_stacked (Tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (Tuple[torch.Tensor, ...]): lora_b's weight.
            lora_bias_stacked (Optional[Tuple[torch.Tensor, ...]]): lora's bias.
            scale (float): Scaling factor.
            output_slices (Tuple[int, ...]): Every slice's size.
            buffer (Optional[Tuple[torch.Tensor, ...]]): Defaults to None.
        """

        if self.no_lora:
            return

        expert_num = 9
        block_statistic = torch.zeros(
            [12, expert_num], dtype=torch.int32, device=x.device
        )
        sorted_tokens_num_lod = torch.zeros(
            expert_num + 1, dtype=torch.int32, device=x.device
        )
        token_nums = x.size(0)
        moe_index = torch.zeros(token_nums, dtype=torch.int32, device=x.device)

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
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])

        if lora_a_stacked.dim() == 2:
            lora_a_stacked = lora_a_stacked.unsqueeze(0)
        if lora_b_stacked.dim() == 2:
            lora_b_stacked = lora_b_stacked.unsqueeze(0)

        r = lora_a_stacked.size(-1)

        if buffer is None:
            buffer = torch.zeros((x.size(0), r), dtype=torch.float32, device=x.device)

        indices = self.sampler_indices
        if indices.max() >= lora_a_stacked.size(0):
            indices = torch.clamp(indices, 0, lora_a_stacked.size(0) - 1)

        lora_a_reshaped = lora_a_stacked.transpose(1, 2)
        lora_b_reshaped = lora_b_stacked.transpose(1, 2)

        bgmv_shrink(x, lora_a_reshaped, buffer, indices, scale)
        bgmv_expand(buffer, lora_b_reshaped, y, indices, add_inputs=True)

        y = y.view_as(y_org)