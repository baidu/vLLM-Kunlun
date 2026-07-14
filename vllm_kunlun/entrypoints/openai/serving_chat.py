"""Request-scoped reasoning parser kwargs patch for OpenAI chat serving."""

import copy
import sys
from functools import wraps

_PATCH_MARKER = "_vllm_kunlun_chat_template_kwargs_patch"
_NATIVE_MODULE = "vllm.entrypoints.openai.serving_chat"


def apply() -> None:
    """Patch the loaded native OpenAI chat serving module."""
    module = sys.modules[_NATIVE_MODULE]
    serving_cls = module.OpenAIServingChat
    original = serving_cls.create_chat_completion
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    async def patched(self, request, raw_request=None):
        parser_factory = self.reasoning_parser
        if parser_factory is None or not getattr(
            parser_factory,
            "supports_chat_template_kwargs",
            False,
        ):
            return await original(self, request, raw_request)

        request_serving = copy.copy(self)

        def create_parser(tokenizer):
            return parser_factory(
                tokenizer,
                chat_template_kwargs=request.chat_template_kwargs,
            )

        request_serving.reasoning_parser = create_parser
        return await original(request_serving, request, raw_request)

    setattr(patched, _PATCH_MARKER, True)
    serving_cls.create_chat_completion = patched
