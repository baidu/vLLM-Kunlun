#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
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
# This file is a part of the vllm-kunlun project.
#
"""
Kunlun-optimized VocabParallelEmbedding using vLLM's PluggableLayer.register_oot mechanism.

Design:
- Uses @PluggableLayer.register_oot to register Kunlun-optimized VocabParallelEmbedding
- This class automatically replaces the default implementation when instantiated

OOT Mechanism:
- When code calls VocabParallelEmbedding(...), vLLM's PluggableLayer.__new__ checks
  op_registry_oot
- If "VocabParallelEmbedding" is found in OOT registry, it returns KunlunVocabParallelEmbedding instance
- This is the official vLLM way to replace operators without modifying source code

NOTE:
- VocabParallelEmbedding inherits from PluggableLayer, not CustomOp. PluggableLayer
  has no forward_native/forward_oot dispatch (that mechanism only exists on CustomOp),
  so the Kunlun-optimized logic must override forward() directly here -- defining
  forward_oot() (the CustomOp convention) is never called and silently falls back to
  the base class's forward(), which triggers vLLM's torch.compile'd
  get_masked_input_and_mask and can fail on Kunlun XPU.
"""

import logging

import torch
import xspeedgate_ops  # noqa
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

logger = logging.getLogger("vllm_kunlun.ops.vocab_parallel_embedding")

# Track if OOT class has logged (for logging once)
_oot_vocab_embedding_init_logged = False


# =============================================================================
# Helper function for masked input computation
# =============================================================================


def _masked_input_and_mask_torch(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """与 vLLM 参考实现逐位一致的 torch 兜底。

    老部署容器的算子提供方是 cocopod，没有 get_masked_input_and_mask。
    """
    org_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
    added_mask = (input_ >= added_vocab_start_index) & (input_ < added_vocab_end_index)
    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    valid_offset = org_vocab_start_index * org_mask + added_offset * added_mask
    vocab_mask = org_mask | added_mask
    return vocab_mask * (input_ - valid_offset), ~vocab_mask


# 只在 import 时探一次：_OpNamespace 的属性查找失败不会被缓存，放在热路径里
# 每步都要付一次查找开销。
_HAS_XSG_MASKED_OP = hasattr(torch.ops.xspeedgate_ops, "get_masked_input_and_mask")


def get_masked_input_and_mask(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # xspeedgate_ops.get_masked_input_and_mask 只接受 1D int32 输入：2D 会报
    # "input_ must be 1D tensor"。qwen4_exp 的 PLE n-gram 查表输入是 2D
    # [tokens, ngram_heads]，而该算子是逐元素 mask，展平后调用再还原形状与逐行
    # 调用等价。
    if input_.dtype != torch.int32:
        input_ = input_.to(torch.int32)
    if not _HAS_XSG_MASKED_OP:
        return _masked_input_and_mask_torch(
            input_,
            org_vocab_start_index,
            org_vocab_end_index,
            num_org_vocab_padding,
            added_vocab_start_index,
            added_vocab_end_index,
        )
    original_shape = tuple(input_.shape)
    if input_.dim() != 1:
        input_ = input_.reshape(-1)
    input_, vocab_mask = torch.ops.xspeedgate_ops.get_masked_input_and_mask(
        input_,
        org_vocab_start_index,
        org_vocab_end_index,
        num_org_vocab_padding,
        added_vocab_start_index,
        added_vocab_end_index,
    )
    if len(original_shape) != 1:
        input_ = input_.view(original_shape)
        vocab_mask = vocab_mask.view(original_shape)
    return input_, vocab_mask


# =============================================================================
# OOT-registered Kunlun VocabParallelEmbedding class
# =============================================================================


@PluggableLayer.register_oot(name="VocabParallelEmbedding")
class KunlunVocabParallelEmbedding(VocabParallelEmbedding):
    """
    Kunlun-optimized VocabParallelEmbedding registered via OOT mechanism.

    This class replaces the default VocabParallelEmbedding when instantiated through
    vLLM's PluggableLayer registry. When code calls VocabParallelEmbedding(...), vLLM's
    PluggableLayer.__new__ checks op_registry_oot and returns KunlunVocabParallelEmbedding
    instance.
    """

    def __init__(self, *args, **kwargs):
        global _oot_vocab_embedding_init_logged
        super().__init__(*args, **kwargs)
        if not _oot_vocab_embedding_init_logged:
            logger.info(
                "[KunlunOOT] KunlunVocabParallelEmbedding.__init__ called (OOT instantiation)"
            )
            _oot_vocab_embedding_init_logged = True

    def forward(self, input_):
        """Kunlun-optimized forward implementation.

        VocabParallelEmbedding derives from PluggableLayer, which has no
        forward_native/forward_oot dispatch (that mechanism only exists on
        CustomOp). forward() must be overridden directly, or this
        implementation is never invoked.
        """
        if self.tp_size > 1:
            # Build the mask using compiled function
            masked_input, input_mask = get_masked_input_and_mask(
                input_,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
        else:
            masked_input = input_

        # Get the embeddings
        output_parallel = self.quant_method.embedding(self, masked_input.long())

        # Mask the output embedding
        if self.tp_size > 1:
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)

        # Reduce across all the model parallel GPUs
        output = tensor_model_parallel_all_reduce(output_parallel)
        return output


# Log that OOT registration is complete
logger.info(
    "[KunlunOOT] Registered KunlunVocabParallelEmbedding via PluggableLayer.register_oot"
)
