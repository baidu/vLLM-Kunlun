# SPDX-License-Identifier: Apache-2.0
"""给 GPUModelRunner 补上 qwen4_exp PLE 需要的 n-gram 输入准备。

qwen4_exp 的 PLE 层要求 runner 在 ``model_kwargs`` 里额外喂两个入参：

``query_start_loc``
    每请求的 token 起始偏移，PLE 用它把打平的 token 序列还原成按请求分行。
``ngram_context``
    每请求前 ``ngram_size - 1`` 个"已算过"的 token，用来补齐本批第一个 token
    的 n-gram 上下文；没有历史时填 EOS。

这段逻辑属于上游那个带 qwen4_exp 的 PR，随包 vLLM 里还没有，所以这里以运行期
打补丁的形式加回去，vllm 包本身不动。改动落在三个方法上：

* ``__init__``：算出是否需要 n-gram、上下文长度、EOS，并分配上下文缓冲；
* ``_preprocess``：真实批次，按 input_batch 里的历史 token 填上下文；
* ``_init_model_kwargs``：dummy 批次（profile run / cudagraph capture），
  上下文整体填 EOS。

``_init_model_kwargs`` 这一处需要 ``_dummy_run`` 的三个局部变量
（``num_reqs`` / ``num_reqs_padded`` / ``num_scheduled_tokens``），而 dummy 批次的
形状必须和 capture 时一致——``num_reqs_padded`` 决定 PLE 内部 ``(num_reqs, num_tokens)``
工作区的行数，用 ``max_num_reqs`` 顶替会让捕获出来的图多做十几倍无用功。
上游是直接在 ``_dummy_run`` 函数体内插桩拿到这些值的；在不改 vllm 的前提下，
要么把 250 行的 ``_dummy_run`` 整个抄一份（跟着 vLLM 版本漂），要么按调用栈读
这三个局部变量。这里选后者：只读不写，且仅在调用方确实是 ``_dummy_run`` 时生效。

随包 vLLM 升级到带 qwen4_exp 的版本后，本模块应整体删除。
"""

import functools
import sys
from typing import Any

import numpy as np
import torch
from vllm.distributed import get_pp_group


def _setup(runner) -> None:
    """按 config 判断是否需要 n-gram 输入，并准备好上下文缓冲。"""
    text_config = runner.model_config.hf_text_config
    runner.uses_ngram_embedding = bool(getattr(text_config, "ple_layer_ids", ()))
    if not runner.uses_ngram_embedding:
        runner.ngram_context_len = 0
        runner.ngram_eos_token_id = 0
        return

    runner.ngram_context_len = int(text_config.ngram_size) - 1
    eos = text_config.eos_token_id
    if isinstance(eos, (list, tuple)):
        eos = eos[0]
    runner.ngram_eos_token_id = int(eos)
    if runner.ngram_context_len <= 0:
        raise ValueError("N-gram embedding requires context length >= 1.")
    if len(get_pp_group().ranks) > 1:
        raise RuntimeError("N-gram PLE embedding requires PP=1.")

    runner.ngram_context = runner._make_buffer(
        runner.max_num_reqs,
        runner.ngram_context_len,
        dtype=torch.int32,
    )
    # Q38 ASYNC: async scheduling 下 token_ids_cpu 里的采样 token 全是 -1 占位
    # （bookkeeping 只写占位，真值只在 GPU），CPU 构建的上下文对 decode 行必然
    # 带洞。这里维护一个 GPU 滚动尾窗：每步"上一窗右移一位 + 滚入当前输入
    # token"（当前输入已被上游 _prepare_input_ids 从 prev_sampled_token_ids
    # 回填进 input_ids.gpu，天然是正确值）。行首次进入带洞状态前必然经过一次
    # CPU 全知步（首个 decode 步的上下文全部来自 prompt），该步完成播种。
    runner.ngram_tail_gpu = torch.zeros(
        runner.max_num_reqs,
        runner.ngram_context_len,
        dtype=torch.int32,
        device=runner.device,
    )


def _prepare_ngram_context(runner, num_reqs: int, num_reqs_padded: int) -> torch.Tensor:
    """逐请求填 n-gram 上下文：取该请求已算过的最后几个 token，不足补 EOS。"""
    eos_token_id = int(runner.ngram_eos_token_id)
    if num_reqs_padded == 0 or runner.ngram_context_len == 0:
        return runner.ngram_context.gpu[:num_reqs_padded]

    context_cpu = runner.ngram_context.np[:num_reqs_padded]
    context_cpu.fill(eos_token_id)

    num_computed = runner.input_batch.num_computed_tokens_cpu
    token_ids = runner.input_batch.token_ids_cpu
    is_token_ids = runner.input_batch.is_token_ids

    for req_idx in range(num_reqs):
        end = int(num_computed[req_idx])
        if end <= 0:
            continue
        start = max(0, end - runner.ngram_context_len)
        context_tokens = token_ids[req_idx, start:end]
        if context_tokens.size == 0:
            continue
        if runner.enable_prompt_embeds and not is_token_ids[req_idx, start:end].all():
            safe_tokens = context_tokens.copy()
            safe_tokens[~is_token_ids[req_idx, start:end]] = eos_token_id
            context_tokens = safe_tokens
        context_cpu[req_idx, -context_tokens.size :] = context_tokens

    runner.ngram_context.copy_to_gpu(num_reqs_padded)

    # Q38 ASYNC: 见 _setup 注释。仅非 spec、async 开启时启用 GPU 修补。
    if (
        getattr(runner, "use_async_scheduling", False)
        and getattr(runner, "num_spec_tokens", 0) == 0
        and num_reqs > 0
    ):
        ctx = runner.ngram_context.gpu
        tail = runner.ngram_tail_gpu
        n = num_reqs
        prev_pos = runner.prev_positions.np[:n]
        cont_np = prev_pos >= 0
        prev_rows_np = np.where(cont_np, prev_pos, 0).astype(np.int64)
        dev = ctx.device
        prev_rows = torch.from_numpy(prev_rows_np).to(dev, non_blocking=True)
        cont = torch.from_numpy(cont_np).to(dev, non_blocking=True)
        tail_prev = tail.index_select(0, prev_rows)
        holes = (ctx[:n] < 0).any(dim=1) & cont
        ctx_fixed = torch.where(holes.unsqueeze(1), tail_prev, ctx[:n])
        ctx[:n].copy_(ctx_fixed)
        # 滚动尾窗：下一步的上下文 = 本步上下文左移一位 + 本步输入 token。
        qsl = runner.query_start_loc.gpu[1 : n + 1].to(torch.int64)
        cur_in = runner.input_ids.gpu.index_select(0, qsl - 1).to(ctx_fixed.dtype)
        new_tail = torch.cat([ctx_fixed[:, 1:], cur_in.unsqueeze(1)], dim=1)
        tail[:n].copy_(new_tail)

    return runner.ngram_context.gpu[:num_reqs_padded]


def _add_ngram_kwargs(
    runner,
    model_kwargs: dict[str, Any],
    *,
    num_reqs: int,
    num_reqs_padded: int,
    is_first_rank: bool,
    is_encoder_decoder: bool,
    use_dummy_context: bool,
    query_start_loc: torch.Tensor | None = None,
    num_scheduled_tokens: Any = None,
) -> None:
    """把 query_start_loc / ngram_context 塞进 model_kwargs。"""
    if not getattr(runner, "uses_ngram_embedding", False):
        return
    if not is_first_rank or is_encoder_decoder:
        return

    eos_token_id = int(runner.ngram_eos_token_id)
    if query_start_loc is None:
        if num_scheduled_tokens is None:
            raise RuntimeError("query_start_loc is required for N-gram input.")
        cu_num_tokens = np.cumsum(num_scheduled_tokens, dtype=np.int32)
        last = int(cu_num_tokens[-1]) if num_reqs > 0 else 0
        runner.query_start_loc.np[0] = 0
        if num_reqs > 0:
            runner.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        runner.query_start_loc.np[num_reqs + 1 :].fill(last)
        runner.query_start_loc.copy_to_gpu()
        query_start_loc = runner.query_start_loc.gpu[: num_reqs_padded + 1]
    model_kwargs["query_start_loc"] = query_start_loc

    if runner.ngram_context_len <= 0:
        return
    if use_dummy_context:
        runner.ngram_context.np[:num_reqs_padded].fill(eos_token_id)
        runner.ngram_context.copy_to_gpu(num_reqs_padded)
        model_kwargs["ngram_context"] = runner.ngram_context.gpu[:num_reqs_padded]
    else:
        model_kwargs["ngram_context"] = _prepare_ngram_context(
            runner, num_reqs, num_reqs_padded
        )


def _dummy_run_locals(dummy_run_code) -> dict[str, Any] | None:
    """沿调用栈找 ``_dummy_run`` 的栈帧，取回它的局部变量（只读）。"""
    frame = sys._getframe(1)
    for _ in range(8):
        if frame is None:
            return None
        if frame.f_code is dummy_run_code:
            return frame.f_locals
        frame = frame.f_back
    return None


def _unwrapped_code(func):
    """拿到真正的函数体 code 对象。

    ``_dummy_run`` 被 ``torch.inference_mode()`` 包过一层，类属性上挂的是
    ``decorate_context``；直接用它的 code 对象去比栈帧只会命中装饰器那一层，
    里面没有我们要的局部变量。
    """
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    return func.__code__


def install_ngram_support(runner_cls: type) -> None:
    """在 GPUModelRunner 上装好 n-gram 输入准备（幂等由调用方保证）。"""
    original_init = runner_cls.__init__
    original_preprocess = runner_cls._preprocess
    original_init_model_kwargs = runner_cls._init_model_kwargs
    dummy_run_code = _unwrapped_code(runner_cls._dummy_run)

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _setup(self)

    @functools.wraps(original_preprocess)
    def _preprocess(self, *args, **kwargs):
        result = original_preprocess(self, *args, **kwargs)
        if not getattr(self, "uses_ngram_embedding", False):
            return result

        input_ids = result[0]
        model_kwargs = next(item for item in result if isinstance(item, dict))
        is_first_rank = get_pp_group().is_first_rank
        is_encoder_decoder = self.model_config.is_encoder_decoder
        if is_first_rank and not is_encoder_decoder and input_ids is None:
            raise RuntimeError(
                "N-gram PLE embedding requires token ids on the first rank; "
                "inputs_embeds-only path is unsupported for this model."
            )
        num_reqs = self.input_batch.num_reqs
        _add_ngram_kwargs(
            self,
            model_kwargs,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs,
            is_first_rank=is_first_rank,
            is_encoder_decoder=is_encoder_decoder,
            use_dummy_context=False,
            query_start_loc=self.query_start_loc.gpu[: num_reqs + 1],
        )
        return result

    @functools.wraps(original_init_model_kwargs)
    def _init_model_kwargs(self):
        model_kwargs = original_init_model_kwargs(self)
        if not getattr(self, "uses_ngram_embedding", False):
            return model_kwargs
        dummy_locals = _dummy_run_locals(dummy_run_code)
        if dummy_locals is None:
            # 真实批次，由 _preprocess 的包装层负责，这里不插手。
            return model_kwargs
        missing = [
            k
            for k in ("num_reqs", "num_reqs_padded", "num_scheduled_tokens")
            if k not in dummy_locals
        ]
        if missing:
            raise RuntimeError(
                "_dummy_run 的局部变量里缺少 "
                f"{missing}，随包 vLLM 的 _dummy_run 实现变了，"
                "vllm_kunlun.v1.worker.ngram_runner_mixin 需要同步更新"
            )
        _add_ngram_kwargs(
            self,
            model_kwargs,
            num_reqs=dummy_locals["num_reqs"],
            num_reqs_padded=dummy_locals["num_reqs_padded"],
            is_first_rank=get_pp_group().is_first_rank,
            is_encoder_decoder=self.model_config.is_encoder_decoder,
            use_dummy_context=True,
            num_scheduled_tokens=dummy_locals["num_scheduled_tokens"],
        )
        return model_kwargs

    runner_cls.__init__ = __init__
    runner_cls._preprocess = _preprocess
    runner_cls._init_model_kwargs = _init_model_kwargs
