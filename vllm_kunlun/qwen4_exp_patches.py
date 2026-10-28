# SPDX-License-Identifier: Apache-2.0
"""qwen4_exp（Qwen3.8-Flash-Next）适配所需的上游补丁。

模型本体在 ``vllm_kunlun.models.qwen4_exp``；本模块只放"必须改上游 vLLM 才能
跑起来"的两处，都以运行期打补丁的方式做，vllm 包本身不动：

``vllm.v1.core.single_type_kv_cache_manager``
    ``MambaManager._allocated_block_reqs`` 只在 ``mamba_cache_mode == "align"``
    分支里初始化，但非 align 分支也会用它。qwen4_exp 的 MambaSpec 用
    ``mamba_cache_mode="none"``，一进 ``allocate_new_blocks`` 就 AttributeError。

``vllm.v1.worker.gpu_model_runner``
    qwen4_exp 的 PLE 层要 runner 额外喂 ``query_start_loc`` 与 ``ngram_context``
    两个 kwargs，否则模型 forward 直接 raise "PLE inputs were not prepared"。
    这段逻辑在上游那个带 qwen4_exp 的 PR 里，随包 vLLM 还没有。

config 类的注册不在这里：它要等 vllm 整个导入完才能做，放在
``vllm_kunlun.models.register_model()`` 里（见那里的注释）。

两处都是"随包 vLLM 版本落后于模型"造成的临时垫片：等 vLLM 升级到带 qwen4_exp
的版本后，本模块整体删除。
"""

import functools
import logging
from types import ModuleType


# --- vllm.v1.core.single_type_kv_cache_manager: 补 _allocated_block_reqs ----


def mamba_manager_applied(module: ModuleType) -> bool:
    """返回 MambaManager 是否已经无条件初始化 ``_allocated_block_reqs``。"""
    manager = getattr(module, "MambaManager", None)
    if manager is None:
        return True
    return getattr(manager, "_kunlun_allocated_block_reqs_patched", False)


def apply_mamba_manager(module: ModuleType) -> None:
    """让 MambaManager 在任何 mamba_cache_mode 下都有 ``_allocated_block_reqs``。"""
    manager = getattr(module, "MambaManager", None)
    if manager is None:
        return
    original_init = manager.__init__

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, "_allocated_block_reqs"):
            self._allocated_block_reqs: set[str] = set()

    manager.__init__ = __init__
    manager._kunlun_allocated_block_reqs_patched = True


# --- vllm.v1.worker.gpu_model_runner: PLE n-gram 输入 -----------------------


def ngram_runner_applied(module: ModuleType) -> bool:
    """返回 runner 是否已经会喂 PLE 的 n-gram 输入。"""
    runner = getattr(module, "GPUModelRunner", None)
    if runner is None:
        return True
    return getattr(runner, "_kunlun_ngram_patched", False)


def apply_ngram_runner(module: ModuleType) -> None:
    """给 GPUModelRunner 接上 qwen4_exp PLE 需要的 n-gram 输入准备逻辑。"""
    runner = getattr(module, "GPUModelRunner", None)
    if runner is None:
        return
    from vllm_kunlun.v1.worker.ngram_runner_mixin import install_ngram_support

    install_ngram_support(runner)
    runner._kunlun_ngram_patched = True


# --- vllm.config.speculative: teach MTP about qwen4_exp ---------------------


def speculative_config_applied(module: ModuleType) -> bool:
    """返回 SpeculativeConfig 是否已认识 qwen4_exp 的 MTP 草稿模型。"""
    cls = getattr(module, "SpeculativeConfig", None)
    if cls is None:
        return True
    return getattr(cls, "_kunlun_qwen4_exp_mtp_patched", False)


def apply_speculative_config(module: ModuleType) -> None:
    """让 ``--speculative-config method=mtp`` 在 qwen4_exp 上可用。

    上游 ``SpeculativeConfig.hf_config_override`` 按 model_type 把 draft config
    的 architectures 改写成对应的 MTP 模型（qwen3_next → Qwen3NextMTP 等），
    没有 qwen4_exp 的条目——不改写的话草稿模型会按整个主模型加载。仿照
    qwen3_5 的写法：n_predict 取 text_config.mtp_num_hidden_layers（本模型为
    1），架构改成 ``Qwen4ExpMTP``（MTP 权重就在主 checkpoint 里，故用户只需传
    method/num_speculative_tokens，无需单独的草稿模型路径）。
    """
    cls = module.SpeculativeConfig

    # method 校验要求 draft 的 model_type 在 MTPModelTypes 里（qwen3_5 同款
    # 做法是换成 <family>_mtp）；该 Literal 只在本模块内被 get_args 读取，
    # 直接扩展模块属性即可。
    from typing import Literal, get_args

    existing = get_args(module.MTPModelTypes)
    if "qwen4_exp_mtp" not in existing:
        module.MTPModelTypes = Literal[existing + ("qwen4_exp_mtp",)]
    original = cls.__dict__["hf_config_override"]
    if isinstance(original, staticmethod):
        original = original.__func__

    @functools.wraps(original)
    def hf_config_override(hf_config):
        if getattr(hf_config, "model_type", None) == "qwen4_exp":
            text_config = getattr(hf_config, "text_config", None)
            n_predict = getattr(text_config, "mtp_num_hidden_layers", None) or 1
            hf_config.model_type = "qwen4_exp_mtp"
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["Qwen4ExpMTP"]}
            )
            logging.getLogger("vllm_kunlun").info(
                "[KunlunPlugin] qwen4_exp MTP draft: architectures -> "
                "['Qwen4ExpMTP'], n_predict=%s",
                n_predict,
            )
            return hf_config
        return original(hf_config)

    cls.hf_config_override = staticmethod(hf_config_override)
    cls._kunlun_qwen4_exp_mtp_patched = True


# --- vllm.v1.spec_decode.llm_base_proposer: HC 多流 hidden 尺寸 --------------


def hc_proposer_applied(module: ModuleType) -> bool:
    """返回 proposer 是否已按 qwen4_exp 的 HC 流数扩展 hidden 缓冲。"""
    cls = getattr(module, "SpecDecodeBaseProposer", None)
    if cls is None:
        return True
    return getattr(cls, "_kunlun_qwen4_exp_hc_patched", False)


def apply_hc_proposer(module: ModuleType) -> None:
    """qwen4_exp MTP 草稿吃 [T, hc_count*H] 的多流 hidden。

    主模型经 ``get_mtp_target_hidden_states()`` 提供 pre-final-mixer 多流
    hidden（runner 通用机制），但 ``SpecDecodeBaseProposer.__init__`` 只认
    DeepSeek-V4 的 ``compress_ratios``+``hc_mult`` 字段来放大 hidden 缓冲，
    qwen4_exp 的流数字段叫 ``text_config.hc_count``。init 后按 hc_count 修正
    hidden_size 并重分配缓冲。
    """
    cls = module.SpecDecodeBaseProposer
    original_init = cls.__init__

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        draft_hf = self.draft_model_config.hf_config
        text_config = getattr(draft_hf, "text_config", None)
        hc_count = getattr(text_config, "hc_count", 0) or 0
        if getattr(draft_hf, "model_type", None) == "qwen4_exp_mtp" and hc_count > 1:
            expected = self.draft_model_config.get_hidden_size() * hc_count
            if self.hidden_size != expected:
                import torch

                self.hidden_size = expected
                self.hidden_states = torch.zeros(
                    (self.max_num_tokens, self.hidden_size),
                    dtype=self.dtype,
                    device=self.hidden_states.device,
                )
                logging.getLogger("vllm_kunlun").info(
                    "[KunlunPlugin] qwen4_exp MTP: proposer hidden buffer "
                    "expanded to hc_count(%d) * hidden(%d)",
                    hc_count,
                    self.draft_model_config.get_hidden_size(),
                )

    cls.__init__ = __init__

    # Qwen4ExpMTP 的 forward 和 DeepSeekMTP 一样返回 (sample_hidden,
    # multi_hidden) 二元组，但 model_returns_tuple() 只认 DeepSeekMTPModel。
    original_returns_tuple = cls.model_returns_tuple

    @functools.wraps(original_returns_tuple)
    def model_returns_tuple(self) -> bool:
        if self.method == "mtp" and "Qwen4ExpMTP" in (
            self.draft_model_config.hf_config.architectures or []
        ):
            return True
        return original_returns_tuple(self)

    cls.model_returns_tuple = model_returns_tuple
    cls._kunlun_qwen4_exp_hc_patched = True


# (target module, is_applied, apply_patch)，由 registration.compat_patches 汇总。

# --- vllm.v1.worker.gpu_model_runner: spec 元数据 builder 白名单 -------------


def spec_meta_builders_applied(module) -> bool:
    return isinstance(getattr(module, "GDNAttentionMetadataBuilder", None), tuple)


def apply_spec_meta_builders(module) -> None:
    """让 runner 把 spec-decode 的 ``num_accepted_tokens`` /
    ``num_decode_draft_tokens_cpu`` 也传给 kunlun 的 GDN / PLE metadata builder。

    upstream 的 ``_prepare_inputs`` 只对
    ``(Mamba2 | GDN | BailingLinear)AttentionMetadataBuilder`` 做 isinstance
    判断后才附带 spec 参数；kunlun 的 GDN / PLE builder 不继承这些类，导致
    真实 verify 步上 builder 的 spec 分支从不执行，FULL cudagraph 的 spec
    持久 buffer 永远停留在捕获时刻的 dummy 值（全 0），图内状态读写全部
    落到 NULL 页 0：verify logits 全 NaN、输出乱码。eager / PIECEWISE 之所以
    正常，是因为 forward 跟着元数据把 verify token 当 2-token prefill 处理，
    数学上等价；FULL 图则固定回放捕获时的 spec 路径，必须依赖持久 buffer
    每步刷新。

    ``isinstance`` 的 classinfo 支持嵌套 tuple，这里把 runner 模块命名空间里
    的 ``GDNAttentionMetadataBuilder`` 换绑成 tuple 即可，模块内该名字只在
    isinstance 判断中使用。
    """
    # The post-import dispatcher can observe a partially initialized runner.
    # Do not read its builder symbol until the class body has completed; a
    # later import event will retry this hook.
    base_builder = getattr(module, "GDNAttentionMetadataBuilder", None)
    if base_builder is None:
        logger = logging.getLogger("vllm_kunlun")
        logger.debug("[KunlunPlugin] delaying spec metadata hook: runner is partial")
        return

    from vllm_kunlun.v1.attention.backends.gdn_attn import (
        GDNAttentionMetadataBuilder as _kunlun_gdn_builder,
    )
    from vllm_kunlun.v1.attention.backends.ple_short_conv_attn import (
        PleShortConvAttentionMetadataBuilder as _kunlun_ple_builder,
    )

    module.GDNAttentionMetadataBuilder = (
        base_builder,
        _kunlun_gdn_builder,
        _kunlun_ple_builder,
    )


QWEN4_EXP_HOOKS = (
    (
        "vllm.v1.spec_decode.llm_base_proposer",
        hc_proposer_applied,
        apply_hc_proposer,
    ),
    (
        "vllm.config.speculative",
        speculative_config_applied,
        apply_speculative_config,
    ),
    (
        "vllm.v1.core.single_type_kv_cache_manager",
        mamba_manager_applied,
        apply_mamba_manager,
    ),
    ("vllm.v1.worker.gpu_model_runner", ngram_runner_applied, apply_ngram_runner),
    (
        "vllm.v1.worker.gpu_model_runner",
        spec_meta_builders_applied,
        apply_spec_meta_builders,
    ),
)
