# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from transformers import PreTrainedTokenizerBase
from vllm.entrypoints.openai.protocol import DeltaMessage
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ResponsesRequest


class KimiK2ReasoningParser(ReasoningParser):
    """
    Reasoning parser for Kimi K2 model.

    The Kimi K2 model uses <think>...</think> tokens to denote reasoning text,
    and may implicitly end reasoning by starting a tool call section using
    <|tool_calls_section_begin|>.
    Thinking may also begin without a </think> token.

    To disable thinking mode, pass chat_template_kwargs={"thinking": False}
    in the API request. The chat template checks the `thinking` variable and
    pre-fills <think></think> so the model skips reasoning and outputs plain
    content directly.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction."
            )

        # Token definitions
        self._start_token = "<think>"
        self._end_token = "</think>"
        self._tool_section_start_token = "<|tool_calls_section_begin|>"

        # Get token IDs
        self._start_token_id = self.vocab.get(self._start_token)
        self._end_token_id = self.vocab.get(self._end_token)
        self._tool_section_start_token_id = self.vocab.get(
            self._tool_section_start_token
        )

        if self._start_token_id is None or self._end_token_id is None:
            raise RuntimeError(
                "KimiK2ReasoningParser could not locate think start/end "
                "tokens in the tokenizer!"
            )

        # Tracks whether the model is in pure-content mode (thinking disabled
        # in the prompt via chat_template_kwargs={"thinking": False}).
        # Set to True on the first streaming token when no <think> is seen.
        self._content_mode: bool = False

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """
        Check if the reasoning content ends in the input_ids.

        Reasoning ends when we see either:
        1. The end token (</think>)
        2. The tool section start token (<|tool_calls_section_begin|>)
        """
        start_token_id = self._start_token_id
        end_token_id = self._end_token_id
        tool_section_start_token_id = self._tool_section_start_token_id

        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == start_token_id:
                return False
            if input_ids[i] == end_token_id:
                return True
            # Implicit reasoning end via tool call section
            if (
                tool_section_start_token_id is not None
                and input_ids[i] == tool_section_start_token_id
            ):
                return True
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        """
        Check if the reasoning content ends in the input_ids on a decode step.
        """
        # Materialize iterable for membership checks
        delta_ids_set = set(delta_ids)

        # Check for explicit end token or implicit tool section start in delta
        if self._end_token_id in delta_ids_set:
            return True
        return (
            self._tool_section_start_token_id is not None
            and self._tool_section_start_token_id in delta_ids_set
        )

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract content token ids from the input_ids.
        """
        if self._end_token_id in input_ids:
            end_token_index = (
                len(input_ids) - 1 - input_ids[::-1].index(self._end_token_id)
            )

            if end_token_index != -1:
                return input_ids[end_token_index + 1 :]

        if (
            self._tool_section_start_token_id is not None
            and self._tool_section_start_token_id in input_ids
        ):
            tool_section_index = (
                len(input_ids)
                - 1
                - input_ids[::-1].index(self._tool_section_start_token_id)
            )

            if tool_section_index != -1:
                return input_ids[tool_section_index:]

        # still reasoning (no content)
        return []

    def extract_reasoning_content(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.
        """
        # thinking does not require a think start token but consume it if present
        raw_start = model_output.find(self._start_token)

        # If neither <think> nor </think> appears in the output, the model was
        # in non-thinking mode (enable_thinking=False caused the prompt to be
        # pre-filled with <think></think>).
        # Treat the entire output as content.
        if raw_start == -1 and model_output.find(self._end_token) == -1:
            tool_section_index = model_output.find(self._tool_section_start_token)
            if tool_section_index != -1:
                return (None, model_output[tool_section_index:] or None)
            return (None, model_output or None)

        start_token_index = 0 if raw_start != 0 else len(self._start_token)
        end_token_index = model_output.find(self._end_token)

        if end_token_index != -1:
            return (
                model_output[start_token_index:end_token_index],
                model_output[end_token_index + len(self._end_token) :] or None,
            )

        tool_section_index = model_output.find(self._tool_section_start_token)
        if tool_section_index != -1:
            return (
                model_output[start_token_index:tool_section_index],
                model_output[tool_section_index:] or None,
            )

        # still reasoning (no content)
        return (
            model_output[start_token_index:],
            None,
        )

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        Extract reasoning content from a delta message during streaming.
        """
        # If content mode detected (thinking disabled in prompt), treat all output as content
        if self._content_mode:
            return DeltaMessage(content=delta_text)

        # If reasoning has already ended in previous tokens, this is content
        if self.is_reasoning_end(previous_token_ids):
            return DeltaMessage(content=delta_text)

        # Skip single special tokens
        if len(delta_token_ids) == 1 and delta_token_ids[0] in [
            self._start_token_id,
            self._end_token_id,
        ]:
            return None

        # Detect non-thinking mode: if the first generated tokens don't start
        # with <think>, the prompt was pre-filled with <think></think> and the
        # model is outputting pure content. Switch to content mode.
        if (
            not previous_token_ids
            and delta_token_ids
            and self._start_token_id not in delta_token_ids
        ):
            self._content_mode = True
            return DeltaMessage(content=delta_text)

        if self._end_token_id in delta_token_ids:
            end_index = delta_text.find(self._end_token)
            reasoning = delta_text[:end_index]
            content = delta_text[end_index + len(self._end_token) :]
            return DeltaMessage(
                reasoning_content=reasoning, content=content if content else None
            )

        if self._tool_section_start_token_id in delta_token_ids:
            tool_index = delta_text.find(self._tool_section_start_token)
            reasoning = delta_text[:tool_index]
            content = delta_text[tool_index:]
            return DeltaMessage(reasoning_content=reasoning, content=content)

        # still reasoning (no end token)
        return DeltaMessage(reasoning_content=delta_text)

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Alias for extract_reasoning_streaming to match vllm base class API."""
        return self.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
