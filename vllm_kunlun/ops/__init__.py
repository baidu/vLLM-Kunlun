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
# This file is a part of the vllm-ascend project.
#

import vllm_kunlun.ops.fused_moe.layer

# base layers
import vllm_kunlun.ops.layernorm
import vllm_kunlun.ops.linear

# quantization
import vllm_kunlun.ops.quantization.awq
import vllm_kunlun.ops.quantization.compressed_tensors.compressed_tensors
import vllm_kunlun.ops.quantization.compressed_tensors.compressed_tensors_moe
import vllm_kunlun.ops.quantization.gptq
import vllm_kunlun.ops.quantization.kernels.kunlun_exllama_linear
import vllm_kunlun.ops.quantization.kernels.kunlun_scale_mm
import vllm_kunlun.ops.quantization.moe_wna16

# embedding
import vllm_kunlun.ops.rotary_embedding
import vllm_kunlun.ops.vocab_parallel_embedding
import vllm_kunlun.v1.sample.spec_decode.eagle
