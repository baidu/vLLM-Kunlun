# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
)
from vllm.logger import init_logger
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

logger = init_logger(__name__)


class MiniMaxM2ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for MiniMax M2 models.

    The parser supports the request-scoped ``thinking_mode`` value passed in
    ``chat_template_kwargs``:

    - ``enabled``: the chat template prefills ``<think>``. Generated text
      before ``</think>`` is reasoning and text after it is final content.
    - ``disabled``: the template prefills an empty thinking block. All
      generated text is final content.
    - ``adaptive``: the template leaves the assistant response open. The model
      may either answer directly or emit an explicit
      ``<think>...</think>`` block.

    The default is ``enabled`` to preserve the original MiniMax M2 behavior.
    """

    _VALID_THINKING_MODES = frozenset({"enabled", "disabled", "adaptive"})
    _GENERATION_PROMPT = "]~b]ai\n"

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        """Initialize the parser with request-scoped chat template settings."""
        super().__init__(tokenizer, *args, **kwargs)

        chat_template_kwargs = kwargs.get("chat_template_kwargs") or {}
        thinking_mode = chat_template_kwargs.get("thinking_mode", "enabled")
        if thinking_mode not in self._VALID_THINKING_MODES:
            raise ValueError(
                "thinking_mode must be one of: enabled, disabled, adaptive"
            )
        self._thinking_mode = thinking_mode

        # In adaptive mode, the prompt contains no thinking marker. Cache the
        # assistant header token IDs so is_reasoning_end() can distinguish the
        # initial prompt from the first generated token.
        try:
            generation_prompt_ids = self.model_tokenizer.encode(
                self._GENERATION_PROMPT, add_special_tokens=False
            )
        except TypeError:
            generation_prompt_ids = self.model_tokenizer.encode(
                self._GENERATION_PROMPT
            )
        self._generation_prompt_ids = tuple(generation_prompt_ids)

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    def _is_generation_prompt(self, input_ids: Sequence[int]) -> bool:
        """Return whether token IDs end at the assistant generation header."""
        prompt_length = len(self._generation_prompt_ids)
        return (
            prompt_length > 0
            and len(input_ids) >= prompt_length
            and tuple(input_ids[-prompt_length:]) == self._generation_prompt_ids
        )

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Return whether output should be handled as final content.

        Adaptive mode starts with the reasoning parser active. An explicit
        ``<think>`` keeps it in the reasoning phase; any other first generated
        token selects the direct-answer path.
        """
        if self._thinking_mode == "disabled":
            return True
        if self._thinking_mode == "adaptive":
            # Historical messages and the adaptive system instruction may
            # contain thinking markers. The assistant header identifies the
            # untouched prompt and must not end the current reasoning phase.
            if self._is_generation_prompt(input_ids):
                return False
            if self.start_token_id in input_ids:
                return super().is_reasoning_end(input_ids)
            # A non-empty marker-free delta means the model chose to answer
            # directly, so content/tool parsing can begin immediately.
            return bool(input_ids)
        return super().is_reasoning_end(input_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Extract token IDs that belong to the final answer."""
        if self._thinking_mode == "disabled":
            return input_ids
        if (
            self._thinking_mode == "adaptive"
            and self.start_token_id not in input_ids
            and self.end_token_id not in input_ids
        ):
            # Marker-free adaptive output is a direct answer.
            return input_ids
        return super().extract_content_ids(input_ids)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Extract reasoning and final content from a streaming delta.

        Enabled mode preserves the original MiniMax M2 convention: ``<think>``
        is in the prompt, so generated text is reasoning until ``</think>``.
        Disabled mode treats every delta as final content. Adaptive mode uses
        explicit markers when present and otherwise treats output as content.
        """
        if self._thinking_mode == "disabled":
            return DeltaMessage(content=delta_text) if delta_text else None

        if self._thinking_mode == "adaptive":
            delta_message = super().extract_reasoning_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
            )
            # Avoid emitting an empty reasoning field for a delta containing
            # only the closing marker followed by final content.
            if delta_message is not None and not delta_message.reasoning:
                delta_message.reasoning = None
            return delta_message

        # Skip single end token
        if len(delta_token_ids) == 1 and delta_token_ids[0] == self.end_token_id:
            return None

        # Check if end token has already appeared in previous tokens
        # meaning we're past the reasoning phase
        if self.end_token_id in previous_token_ids:
            # We're past the reasoning phase, this is content
            return DeltaMessage(content=delta_text)

        # Check if end token is in delta tokens
        if self.end_token_id in delta_token_ids:
            # End token in delta, split reasoning and content
            end_index = delta_text.find(self.end_token)
            reasoning = delta_text[:end_index]
            content = delta_text[end_index + len(self.end_token) :]
            return DeltaMessage(
                reasoning=reasoning if reasoning else None,
                content=content if content else None,
            )

        # No end token yet, all content is reasoning
        return DeltaMessage(reasoning=delta_text)

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        """Extract reasoning and final content from a complete model output."""
        if self._thinking_mode == "disabled":
            return None, model_output
        if (
            self._thinking_mode == "adaptive"
            and self.start_token not in model_output
            and self.end_token not in model_output
        ):
            # Marker-free adaptive output is a direct answer.
            return None, model_output
        return super().extract_reasoning(model_output, request)


class MiniMaxM2AppendThinkReasoningParser(ReasoningParser):
    """
    Reasoning parser for MiniMax M2 model.
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.end_token_id = self.vocab.get("</think>")
        self.start_token_id = self.vocab.get("<think>")

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        end_token_id = self.end_token_id
        start_token_id = self.start_token_id
        for input_id in reversed(input_ids):
            if input_id in (end_token_id, start_token_id):
                return input_id == end_token_id
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return input_ids

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if len(previous_token_ids) == 0:
            delta_text = "<think>" + delta_text
        return DeltaMessage(content=delta_text)

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        return None, "<think>" + model_output
