# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)

@torch.compile(dynamic=True, backend="aot_eager")
def get_masked_input_and_mask(
        input_: torch.Tensor, org_vocab_start_index: int,
        org_vocab_end_index: int, num_org_vocab_padding: int,
        added_vocab_start_index: int,
        added_vocab_end_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    # torch.compile will fuse all of the pointwise ops below
    # into a single kernel, making it very fast
    org_vocab_mask = (input_ >= org_vocab_start_index) & (
        input_ < org_vocab_end_index)
    added_vocab_mask = (input_ >= added_vocab_start_index) & (
        input_ < added_vocab_end_index)
    added_offset = added_vocab_start_index - (
        org_vocab_end_index - org_vocab_start_index) - num_org_vocab_padding
    valid_offset = (org_vocab_start_index *
                    org_vocab_mask) + (added_offset * added_vocab_mask)
    vocab_mask = org_vocab_mask | added_vocab_mask
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask

def forward_native_kunlun(self, input_):
    if self.tp_size > 1:
        # Build the mask.
        masked_input, input_mask = get_masked_input_and_mask(
            input_, self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
            self.shard_indices.num_org_vocab_padding,
            self.shard_indices.added_vocab_start_index,
            self.shard_indices.added_vocab_end_index)
    else:
        masked_input = input_
    # Get the embeddings.
    output_parallel = self.quant_method.embedding(self,
                                                  masked_input.long())
    # Mask the output embedding.
    if self.tp_size > 1:
        output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
    # Reduce across all the model parallel GPUs.
    output = tensor_model_parallel_all_reduce(output_parallel)
    return output

VocabParallelEmbedding.forward_native = forward_native_kunlun