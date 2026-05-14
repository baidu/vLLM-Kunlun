#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
#
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

from types import SimpleNamespace

from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

from vllm_kunlun.platforms.kunlun import KunlunPlatform


def _make_vllm_config(splitting_ops, cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE):
    compilation_config = SimpleNamespace(
        backend=None,
        cudagraph_mode=cudagraph_mode,
        custom_ops=[],
        mode=CompilationMode.VLLM_COMPILE,
        pass_config=SimpleNamespace(enable_fusion=True),
        splitting_ops=splitting_ops,
    )
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        compilation_config=compilation_config,
        model_config=SimpleNamespace(enforce_eager=False, use_mla=False),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            worker_cls="vllm.v1.worker.gpu_worker.Worker",
        ),
        speculative_config=None,
    )


def test_check_and_update_config_completes_legacy_kunlun_splitting_ops():
    vllm_config = _make_vllm_config(["vllm.unified_attention_with_output_kunlun"])

    KunlunPlatform.check_and_update_config(vllm_config)

    splitting_ops = vllm_config.compilation_config.splitting_ops
    assert "vllm::unified_attention_with_output_kunlun" in splitting_ops
    assert "vllm.unified_attention_with_output_kunlun" not in splitting_ops
    for op_name in CompilationConfig._attention_ops:
        assert op_name in splitting_ops


def test_check_and_update_config_preserves_custom_splitting_ops():
    custom_op = "custom_namespace::custom_op"
    vllm_config = _make_vllm_config(
        [
            "vllm::unified_attention",
            "vllm::unified_attention_with_output_kunlun",
            custom_op,
        ]
    )

    KunlunPlatform.check_and_update_config(vllm_config)

    splitting_ops = vllm_config.compilation_config.splitting_ops
    assert custom_op in splitting_ops
    assert splitting_ops.count("vllm::unified_attention") == 1
    assert splitting_ops.count("vllm::unified_attention_with_output_kunlun") == 1


def test_check_and_update_config_skips_splitting_ops_without_piecewise_graph():
    vllm_config = _make_vllm_config(
        ["vllm.unified_attention_with_output_kunlun"],
        cudagraph_mode=CUDAGraphMode.NONE,
    )

    KunlunPlatform.check_and_update_config(vllm_config)

    assert vllm_config.compilation_config.splitting_ops == [
        "vllm.unified_attention_with_output_kunlun"
    ]
