#
# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
# Author: Tang Shiwen, Li Wei
# Email: tangshiwen@baidu.com, liwei157@baidu.com
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

import torch


def dequant_int4(
    qweight: torch.Tensor,
    scale: torch.Tensor,
    zp: torch.Tensor,
    int4_signed: bool = False,
    use_mode_fast: bool = False,
) -> torch.Tensor:

    fpweight = torch.empty(
        (
            qweight.shape[0],
            qweight.shape[2],
            scale.shape[1],
        ),
        dtype=scale.dtype,
        device=qweight.device,
    )

    qweight_t = qweight.transpose(1, 2).contiguous()
    qscale_t = scale.transpose(1, 2).contiguous() * 15.0

    zp_t = zp.transpose(1, 2).contiguous()
    zp_unpack = torch.stack((zp_t & 0xF, (zp_t >> 4) & 0xF), dim=-1)
    zp_fp = (
        zp_unpack.reshape(
            zp_unpack.shape[0],
            zp_unpack.shape[1],
            zp_unpack.shape[2] * zp_unpack.shape[3],
        )
        .contiguous()
        .to(scale.dtype)
        - 8.0
    )

    group_m = qweight_t.shape[-2] // qscale_t.shape[-2]

    torch.ops._C.dequant_int4(
        x=qweight_t,
        scale=qscale_t,
        zero=zp_fp,
        y=fpweight,
        group_m=group_m,
        int4_signed=int4_signed,
        use_mode_fast=use_mode_fast,
    )

    return fpweight.transpose(1, 2).contiguous()


def dequant_int4_native(weight_packed_uint8: torch.Tensor, scale: torch.Tensor):
    """Unpack uint4 weight from packed uint8 weight and dequant it to float16."""
    weight_upacked_fp16 = (
        torch.stack(
            (weight_packed_uint8 & 0xF, (weight_packed_uint8 >> 4) & 0xF),
            dim=-1,
        )
        .reshape(*weight_packed_uint8.shape[:-1], -1)
        .contiguous()
        .to(torch.float16)
        - 8.0
    )
    weight_upacked_fp16 *= scale.repeat(
        1, 1, weight_upacked_fp16.shape[-1] // scale.shape[-1]
    )
    return weight_upacked_fp16
