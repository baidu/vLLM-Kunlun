# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import vllm.v1.worker.gpu.spec_decode.rejection_sampler as _up
from vllm_kunlun.v1.worker.gpu._kernels import TorchKernel


def _flatten_sampled(
    flat, sampled, sampled_stride, num_sampled, cu_num_logits, **kwargs
):
    torch.ops.xspeedgate_ops.flatten_sampled_kernel(
        flat, sampled, num_sampled, cu_num_logits, sampled.shape[0], sampled_stride
    )


_up._flatten_sampled_kernel = TorchKernel(_flatten_sampled)
