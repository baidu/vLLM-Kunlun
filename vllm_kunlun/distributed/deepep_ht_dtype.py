"""Let DeepEP high-throughput finalize accept a float16 expert output.

``prepare_finalize/deepep_ht.py`` guards its combine call with

    assert fused_expert_output.dtype == torch.bfloat16

because the reference DeepEP combine kernel is bfloat16-only. Kunlun's
``BufferV2.combine`` accepts either 2-byte float type, so for a float16 model
that assertion is the only thing in the way, and satisfying it would cost an
extra read+write of an ``[num_tokens, hidden]`` tensor per MoE layer per step.

Rebuild ``_finalize`` from its own source with that one assert statement
removed. The statement is located with ``ast`` instead of by text matching, and
the rebuilt function keeps the original module globals, so it stays in step with
upstream changes to the rest of the method. If the assert can no longer be
identified unambiguously, fall back to casting, which is correct but slower.

``_finalize`` is the private method that holds the assertion; the public
``finalize`` is a thin wrapper, so patching that one has no effect.
"""

import ast
import inspect
import logging
import sys
import textwrap

import torch

logger = logging.getLogger("vllm_kunlun")

_CLASSES = ("DeepEPHTPrepareAndFinalize",)


def applied(mod) -> bool:
    for name in _CLASSES:
        cls = getattr(mod, name, None)
        if cls is not None and not getattr(cls._finalize, "_kunlun_fp16_ok", False):
            return False
    return True


def _strip_bf16_assert(cls) -> bool:
    """Rebuild cls._finalize without the bfloat16 assertion. True on success."""
    try:
        src = textwrap.dedent(inspect.getsource(cls._finalize))
        tree = ast.parse(src)
        func = tree.body[0]
        if not isinstance(func, ast.FunctionDef) or func.decorator_list:
            return False
        spans = [
            (node.lineno, node.end_lineno)
            for node in ast.walk(func)
            if isinstance(node, ast.Assert) and "bfloat16" in ast.unparse(node.test)
        ]
        if len(spans) != 1:
            return False
        lo, hi = spans[0]
        lines = src.splitlines()
        kept = [
            line for i, line in enumerate(lines, 1) if not lo <= i <= (hi or lo)
        ]
        new_src = "\n".join(kept) + "\n"
        if "bfloat16" in new_src:
            return False
        module = sys.modules[cls.__module__]
        code = compile(new_src, "<vllm_kunlun:%s._finalize>" % cls.__name__, "exec")
        # Execute in the module's own namespace so the rebuilt function keeps
        # the exact globals the original used, then drop the stray binding.
        exec(code, module.__dict__)
        rebuilt = module.__dict__.pop(func.name)
        rebuilt.__qualname__ = cls.__name__ + "." + func.name
        rebuilt._kunlun_fp16_ok = True
        setattr(cls, func.name, rebuilt)
        return True
    except Exception:
        logger.warning(
            "[KunlunPlugin] could not rebuild %s._finalize without the bfloat16 "
            "assert; falling back to casting",
            cls.__name__,
            exc_info=True,
        )
        return False


def _install_cast_fallback(cls) -> None:
    orig = cls._finalize

    def _finalize(self, *args, _orig=orig, **kwargs):
        key = "fused_expert_output"
        if key in kwargs and torch.is_tensor(kwargs[key]):
            if kwargs[key].dtype != torch.bfloat16:
                kwargs[key] = kwargs[key].to(torch.bfloat16)
        elif len(args) >= 2 and torch.is_tensor(args[1]):
            if args[1].dtype != torch.bfloat16:
                args = args[:1] + (args[1].to(torch.bfloat16),) + args[2:]
        return _orig(self, *args, **kwargs)

    _finalize._kunlun_fp16_ok = True
    cls._finalize = _finalize


def apply(mod) -> None:
    for name in _CLASSES:
        cls = getattr(mod, name, None)
        if cls is None or getattr(cls._finalize, "_kunlun_fp16_ok", False):
            continue
        if _strip_bf16_assert(cls):
            logger.info(
                "[KunlunPlugin] %s._finalize rebuilt without the bfloat16 assert "
                "(float16 expert output goes to combine as is)",
                name,
            )
        else:
            _install_cast_fallback(cls)
            logger.info(
                "[KunlunPlugin] %s._finalize casts the expert output to bfloat16",
                name,
            )
