# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import uuid
from collections.abc import Sequence
from typing import Any

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser

logger = init_logger(__name__)


class MinimaxM3ToolParser(ToolParser):
    """Tool parser for MiniMax M3 namespace-delimited XML-style calls."""

    supports_required_and_named = False

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self.namespace = "]<]minimax[>["
        self.tool_call_start_token = f"{self.namespace}<tool_call>"
        self.tool_call_end_token = f"{self.namespace}</tool_call>"
        self.invoke_start_prefix = f"{self.namespace}<invoke"
        self.invoke_end_token = f"{self.namespace}</invoke>"
        self.current_tool_index = 0
        self.is_tool_call_started = False

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        # The M3 tool-call markers (namespace + ``<tool_call>``) are always a
        # multi-token sequence in the tokenizer, never a single merged vocab
        # entry, so this parser matches them as text markers. This mirrors the
        # upstream RustToolParser, which uses ``tool_call_start_token`` only for
        # fast text-based rejection and never looks the markers up in the vocab.

    def _generate_tool_call_id(self) -> str:
        return f"call_{uuid.uuid4().hex[:24]}"

    def _tool_schema_properties(self, function_name: str) -> dict[str, Any]:
        for tool in self.tools:
            function = getattr(tool, "function", None)
            if function is None or function.name != function_name:
                continue
            parameters = getattr(function, "parameters", None)
            if isinstance(parameters, dict):
                properties = parameters.get("properties")
                if isinstance(properties, dict):
                    return properties
        return {}

    def _schema_types(self, schema: Any) -> list[str]:
        if not isinstance(schema, dict):
            return ["string"]
        types: set[str] = set()
        raw_type = schema.get("type")
        if isinstance(raw_type, str):
            types.add(raw_type)
        elif isinstance(raw_type, list):
            types.update(t for t in raw_type if isinstance(t, str))
        for key in ("anyOf", "oneOf", "allOf"):
            for choice in schema.get(key, []) if isinstance(schema.get(key), list) else []:
                types.update(self._schema_types(choice))
        return list(types) or ["string"]

    def _convert_value(self, value: str, types: list[str]) -> Any:
        value = value.strip()
        if value.lower() in ("null", "none", "nil"):
            return None
        normalized = [t.lower() for t in types]
        for target in ("integer", "int", "number", "float", "boolean", "bool"):
            if target not in normalized:
                continue
            try:
                if target in ("integer", "int"):
                    return int(value)
                if target in ("number", "float"):
                    parsed = float(value)
                    return int(parsed) if parsed.is_integer() else parsed
                lowered = value.lower()
                if lowered in ("true", "1", "yes", "on"):
                    return True
                if lowered in ("false", "0", "no", "off"):
                    return False
            except (TypeError, ValueError):
                pass
        if "object" in normalized or "array" in normalized:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def _parse_element_body(self, body: str, properties: dict[str, Any]) -> Any:
        children: dict[str, Any] = {}
        text_parts: list[str] = []
        pos = 0
        start_pattern = re.compile(rf"{re.escape(self.namespace)}<([A-Za-z_][\w.-]*)>")
        while True:
            start = start_pattern.search(body, pos)
            if start is None:
                text_parts.append(body[pos:])
                break
            text_parts.append(body[pos:start.start()])
            name = start.group(1)
            close_token = f"{self.namespace}</{name}>"
            close_idx = body.find(close_token, start.end())
            if close_idx < 0:
                text_parts.append(body[start.start():])
                break
            child_body = body[start.end():close_idx]
            children[name] = self._parse_element_body(
                child_body, properties.get(name, {}) if isinstance(properties, dict) else {}
            )
            pos = close_idx + len(close_token)

        text = "".join(text_parts).strip()
        if children:
            if text:
                children["text"] = text
            return children
        return self._convert_value(text, self._schema_types(properties))

    def _parse_invokes(self, text: str) -> list[ToolCall]:
        calls: list[ToolCall] = []
        invoke_re = re.compile(
            rf"{re.escape(self.namespace)}<invoke\s+name=[\"']([^\"']+)[\"']>"
            rf"(.*?){re.escape(self.invoke_end_token)}",
            re.DOTALL,
        )
        for match in invoke_re.finditer(text):
            function_name = match.group(1)
            body = match.group(2)
            properties = self._tool_schema_properties(function_name)
            args = self._parse_element_body(body, properties)
            if not isinstance(args, dict):
                args = {}
            calls.append(
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=function_name,
                        arguments=json.dumps(args, ensure_ascii=False),
                    ),
                )
            )
        return calls

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        tool_calls = self._parse_invokes(model_output)
        if not tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        first_idx = model_output.find(self.tool_call_start_token)
        content = model_output[:first_idx] or None
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=content,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        start_in_delta = self.tool_call_start_token in delta_text
        if not previous_text or start_in_delta:
            self.current_tool_index = 0
            self.is_tool_call_started = start_in_delta

        if not self.is_tool_call_started and self.tool_call_start_token in current_text:
            self.is_tool_call_started = True

        if not self.is_tool_call_started:
            return DeltaMessage(content=delta_text) if delta_text else None

        content_before = None
        if start_in_delta:
            content_before = delta_text.split(self.tool_call_start_token, 1)[0] or None

        complete_calls = self._parse_invokes(current_text)
        deltas: list[DeltaToolCall] = []
        while self.current_tool_index < len(complete_calls):
            call = complete_calls[self.current_tool_index]
            deltas.append(
                DeltaToolCall(
                    index=self.current_tool_index,
                    id=self._generate_tool_call_id(),
                    function=DeltaFunctionCall(
                        name=call.function.name,
                        arguments=call.function.arguments,
                    ),
                    type="function",
                )
            )
            self.current_tool_index += 1

        if deltas or content_before:
            return DeltaMessage(content=content_before, tool_calls=deltas)
        return None
