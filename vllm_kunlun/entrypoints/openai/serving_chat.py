# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence as GenericSequence
from typing import Callable, Final, Optional, Union

import jinja2
import partial_json_parser
import regex as re
from fastapi import Request
from openai_harmony import Message as OpenAIMessage
from pydantic import TypeAdapter

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (ChatTemplateContentFormatOption,
                                         ConversationMessage,
                                         random_tool_call_id)
from vllm.entrypoints.harmony_utils import (
    get_developer_message, get_stop_tokens_for_assistant_actions,
    get_streamable_parser_for_assistant, get_system_message, parse_chat_input,
    parse_chat_output, render_for_completion)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    ChatCompletionLogProb, ChatCompletionLogProbs,
    ChatCompletionLogProbsContent, ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaFunctionCall, DeltaMessage,
    DeltaToolCall, ErrorResponse, FunctionCall, FunctionDefinition,
    PromptTokenUsageInfo, RequestResponseMetadata, ToolCall, UsageInfo)
from vllm.entrypoints.openai.serving_engine import (OpenAIServing,
                                                    clamp_prompt_logprobs)
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager
from vllm.entrypoints.openai.tool_parsers.mistral_tool_parser import (
    MistralToolCall)
from vllm.entrypoints.utils import get_max_tokens
from vllm.inputs.data import TokensPrompt as EngineTokensPrompt
from vllm.logger import init_logger
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.sequence import Logprob
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.transformers_utils.tokenizers import (maybe_serialize_tool_calls,
                                                truncate_tool_call_ids,
                                                validate_request_params)
from vllm.utils import as_list

logger = init_logger(__name__)


class OpenAIServingChat(OpenAIServing):

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
        enable_force_include_usage: bool,
    ) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        chunk_object_type: Final = "chat.completion.chunk"
        first_iteration = True

        # Send response for each token for each request.n (index)
        num_choices = 1 if request.n is None else request.n
        previous_num_tokens = [0] * num_choices
        finish_reason_sent = [False] * num_choices
        num_prompt_tokens = 0
        num_cached_tokens = None
        if self.use_harmony:
            harmony_parsers = [
                get_streamable_parser_for_assistant()
                for _ in range(num_choices)
            ]

        if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
            tool_choice_function_name = request.tool_choice.function.name
        else:
            tool_choice_function_name = None

        # Determine whether tools are in use with "auto" tool choice
        tool_choice_auto = (
            not tool_choice_function_name
            and self._should_stream_with_auto_tool_parsing(request))

        all_previous_token_ids: Optional[list[list[int]]]
        function_name_returned = [False] * num_choices

        # Always track previous_texts for comprehensive output logging
        previous_texts = [""] * num_choices

        # Only one of these will be used, thus previous_texts and
        # all_previous_token_ids will not be used twice in the same iteration.
        if tool_choice_auto or self.reasoning_parser:
            # These are only required in "auto" tool choice case
            all_previous_token_ids = [[]] * num_choices
            # For reasoning parser and tool call all enabled
            added_content_delta_arr = [False] * num_choices
            reasoning_end_arr = [False] * num_choices
        elif request.tool_choice == "required":
            all_previous_token_ids = None
        else:
            all_previous_token_ids = None

        enable_thinking: bool = request.chat_template_kwargs.get("enable_thinking", True) if request.chat_template_kwargs else True

        try:
            if self.reasoning_parser:
                reasoning_parser = self.reasoning_parser(tokenizer)
        except RuntimeError as e:
            logger.exception("Error in reasoning parser creation.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return
        # Prepare the tool parser if it's needed
        try:
            if tool_choice_auto and self.tool_parser:
                tool_parsers: list[Optional[ToolParser]] = [
                    self.tool_parser(tokenizer)
                ] * num_choices
            else:
                tool_parsers = [None] * num_choices
        except Exception as e:
            logger.exception("Error in tool parser creation.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return

        stream_options = request.stream_options
        if stream_options:
            include_usage = stream_options.include_usage \
                            or enable_force_include_usage
            include_continuous_usage = include_usage and \
                                       stream_options.continuous_usage_stats
        else:
            include_usage, include_continuous_usage = False, False

        try:
            async for res in result_generator:
                if res.prompt_token_ids is not None:
                    num_prompt_tokens = len(res.prompt_token_ids)
                    if res.encoder_prompt_token_ids is not None:
                        num_prompt_tokens += len(res.encoder_prompt_token_ids)

                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    num_cached_tokens = res.num_cached_tokens
                    # Send first response for each request.n (index) with
                    # the role
                    role = self.get_chat_request_role(request)

                    # NOTE num_choices defaults to 1 so this usually executes
                    # once per request
                    for i in range(num_choices):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(
                                role=role,
                                content="",
                            ),
                            logprobs=None,
                            finish_reason=None)
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name)

                        # if continuous usage stats are requested, add it
                        if include_continuous_usage:
                            chunk.usage = UsageInfo(
                                prompt_tokens=num_prompt_tokens,
                                completion_tokens=0,
                                total_tokens=num_prompt_tokens)

                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content: Union[str, list[dict[str, str]]] = ""
                        if conversation and "content" in conversation[
                                -1] and conversation[-1].get("role") == role:
                            last_msg_content = conversation[-1]["content"] or ""

                        if last_msg_content:
                            for i in range(num_choices):
                                choice_data = (
                                    ChatCompletionResponseStreamChoice(
                                        index=i,
                                        delta=DeltaMessage(
                                            content=last_msg_content),
                                        logprobs=None,
                                        finish_reason=None))
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    model=model_name)
                                if include_continuous_usage:
                                    chunk.usage = UsageInfo(
                                        prompt_tokens=num_prompt_tokens,
                                        completion_tokens=0,
                                        total_tokens=num_prompt_tokens)

                                data = chunk.model_dump_json(
                                    exclude_unset=True)
                                yield f"data: {data}\n\n"
                    first_iteration = False

                for output in res.outputs:
                    i = output.index
                    tool_parser = tool_parsers[i]

                    if finish_reason_sent[i]:
                        continue

                    if request.logprobs and request.top_logprobs is not None:
                        assert output.logprobs is not None, (
                            "Did not output logprobs")
                        logprobs = self._create_chat_logprobs(
                            token_ids=output.token_ids,
                            top_logprobs=output.logprobs,
                            tokenizer=tokenizer,
                            num_output_top_logprobs=request.top_logprobs,
                            return_as_token_id=request.
                            return_tokens_as_token_ids,
                        )
                    else:
                        logprobs = None

                    if self.use_harmony:
                        harmony_parser = harmony_parsers[i]
                        for token_id in output.token_ids:
                            harmony_parser.process(token_id)
                        # FIXME(woosuk): Support function calling
                        is_final = harmony_parser.current_channel == "final"
                        if not (request.include_reasoning or is_final):
                            # Skip the reasoning content.
                            continue
                        delta_text = harmony_parser.last_content_delta or ""
                    else:
                        delta_text = output.text

                    if not delta_text and not output.token_ids and \
                        not previous_num_tokens[i]:
                        # Chunked prefill case, don't return empty chunks
                        continue

                    delta_message: Optional[DeltaMessage]

                    # just update previous_texts and previous_token_ids
                    if ((tool_choice_auto or self.reasoning_parser)
                            and not self.use_harmony):
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_text = previous_texts[i]
                        previous_token_ids = all_previous_token_ids[i]
                        current_text = previous_text + delta_text

                        # avoid the None + list error.
                        if previous_token_ids:
                            current_token_ids = previous_token_ids + as_list(
                                output.token_ids)
                        else:
                            current_token_ids = as_list(output.token_ids)

                    if self.use_harmony:
                        if is_final:
                            delta_message = DeltaMessage(content=delta_text)
                        else:
                            delta_message = DeltaMessage(
                                reasoning_content=delta_text)
                    # handle streaming deltas for tools with named tool_choice
                    elif tool_choice_function_name:
                        if (self.reasoning_parser and not reasoning_end_arr[i]
                                and not reasoning_parser.is_reasoning_end(
                                    previous_token_ids)):
                            assert reasoning_parser is not None
                            delta_message = (
                                reasoning_parser.
                                extract_reasoning_content_streaming(
                                    previous_text,
                                    current_text,
                                    delta_text,
                                    previous_token_ids,
                                    current_token_ids,
                                    output.token_ids,
                                ))
                            # When encountering think end id in delta_token_ids
                            # or think end id in prompt_token_ids
                            # i.e {"enable_thinking": False},
                            # set reasoning status to end.
                            # Only keep 'content', remove 'reasoning_content'.
                            if reasoning_parser.is_reasoning_end(
                                    as_list(output.token_ids)) or (
                                        res.prompt_token_ids
                                        and reasoning_parser.is_reasoning_end(
                                            res.prompt_token_ids)):
                                reasoning_end_arr[i] = True
                                if delta_message and delta_message.content:
                                    # This need to be added to next `delta_text`
                                    current_text = delta_message.content
                                    delta_message.content = None
                                else:
                                    current_text = ""
                        else:
                            # Just to add remaining `content`
                            if self.reasoning_parser:
                                delta_text = previous_text + delta_text
                                current_text = ""

                            if function_name_returned[i]:
                                delta_tool_call = DeltaToolCall(
                                    function=DeltaFunctionCall(
                                        arguments=delta_text),
                                    index=i)
                            else:
                                delta_tool_call = DeltaToolCall(
                                    id=random_tool_call_id(),
                                    type="function",
                                    function=DeltaFunctionCall(
                                        name=tool_choice_function_name,
                                        arguments=delta_text),
                                    index=i)
                                function_name_returned[i] = True

                            delta_message = DeltaMessage(tool_calls=[
                                delta_tool_call,
                            ])

                    elif request.tool_choice == "required":
                        assert previous_texts is not None
                        previous_text = previous_texts[i]
                        current_text = previous_text + delta_text
                        fn_name_returned = function_name_returned[i]

                        if self.reasoning_parser:
                            _, content = \
                                reasoning_parser.extract_reasoning_content(
                                    current_text,
                                    request
                                )
                        else:
                            content = current_text
                        delta_message, function_name_returned[i] = (
                            self.extract_tool_call_required_streaming(
                                previous_text=previous_text,
                                current_text=content,
                                delta_text=delta_text,
                                function_name_returned=fn_name_returned))

                        # update the previous values for the next iteration
                        previous_texts[i] = current_text

                    # handle streaming deltas for tools with "auto" tool choice
                    # and reasoning parser
                    elif tool_choice_auto and self.reasoning_parser:
                        assert tool_parser is not None
                        assert reasoning_parser is not None
                        assert added_content_delta_arr is not None
                        assert reasoning_end_arr is not None
                        output_token_ids = as_list(output.token_ids)
                        if not reasoning_end_arr[i]:
                            delta_message = (
                                reasoning_parser.
                                extract_reasoning_content_streaming(
                                    previous_text,
                                    current_text,
                                    delta_text,
                                    previous_token_ids,
                                    current_token_ids,
                                    output_token_ids,
                                ))
                            # When encountering think end id in prompt_token_ids
                            # i.e {"enable_thinking": False},
                            # set reasoning status to end.
                            # Remove the text and token ids related
                            # to 'reasoning_content'.
                            if not enable_thinking:
                                reasoning_end_arr[i] = True
                                current_token_ids = output_token_ids
                                if delta_message and delta_message.reasoning_content:
                                    current_text = delta_message.reasoning_content
                                    delta_message.content = None
                                    delta_message.reasoning_content = None
                                else:
                                    current_text = delta_message.content
                            # When encountering think end id in delta_token_ids,
                            # set reasoning status to end.
                            # Remove the text and token ids related
                            # to 'reasoning_content'.
                            if reasoning_parser.is_reasoning_end(
                                    output_token_ids):
                                reasoning_end_arr[i] = True
                                current_token_ids =  \
                                    reasoning_parser.extract_content_ids(
                                        output_token_ids)
                                if delta_message and delta_message.content:
                                    current_text = delta_message.content
                                    delta_message.content = None
                                else:
                                    current_text = ""

                        # handle tool calls only after reasoning is done,
                        else:
                            delta_token_ids = output_token_ids
                            # First time to tool call,
                            # add the remaining text and token ids
                            # to delta from previous
                            if not added_content_delta_arr[i]:
                                added_content_delta_arr[i] = True
                                previous_text = ""
                                previous_token_ids = []
                                delta_text = current_text
                                delta_token_ids = current_token_ids

                            delta_message = (
                                tool_parser.extract_tool_calls_streaming(
                                    previous_text=previous_text,
                                    current_text=current_text,
                                    delta_text=delta_text,
                                    previous_token_ids=previous_token_ids,
                                    current_token_ids=current_token_ids,
                                    delta_token_ids=delta_token_ids,
                                    request=request))
                    # when only tool calls
                    elif tool_choice_auto:
                        assert tool_parser is not None
                        delta_message = (
                            tool_parser.extract_tool_calls_streaming(
                                previous_text=previous_text,
                                current_text=current_text,
                                delta_text=delta_text,
                                previous_token_ids=previous_token_ids,
                                current_token_ids=current_token_ids,
                                delta_token_ids=output.token_ids,
                                request=request))

                    # when only reasoning
                    elif self.reasoning_parser and enable_thinking:
                        delta_message = (reasoning_parser.
                                         extract_reasoning_content_streaming(
                                             previous_text,
                                             current_text,
                                             delta_text,
                                             previous_token_ids,
                                             current_token_ids,
                                             output.token_ids,
                                         ))
                    # handle streaming just a content delta
                    else:
                        delta_message = DeltaMessage(content=delta_text)

                    # update the previous values for the next iteration
                    if tool_choice_auto or self.reasoning_parser:
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_texts[i] = current_text
                        all_previous_token_ids[i] = current_token_ids
                    else:
                        # Update for comprehensive logging even in simple case
                        assert previous_texts is not None
                        previous_texts[i] += delta_text

                    # set the previous values for the next iteration
                    previous_num_tokens[i] += len(output.token_ids)

                    # if the message delta is None (e.g. because it was a
                    # "control token" for tool calls or the parser otherwise
                    # wasn't ready to send a token, then
                    #   get the next token without streaming a chunk
                    if delta_message is None:
                        continue

                    # Log streaming delta if output logging is enabled
                    if self.enable_log_outputs and self.request_logger:
                        delta_content = ""
                        if delta_message.content:
                            delta_content = delta_message.content
                        elif delta_message.tool_calls:
                            delta_content = "".join(
                                tc.function.arguments
                                for tc in delta_message.tool_calls
                                if tc.function and tc.function.arguments)

                        if delta_content:
                            self.request_logger.log_outputs(
                                request_id=request_id,
                                outputs=delta_content,
                                output_token_ids=as_list(output.token_ids),
                                finish_reason=output.finish_reason,
                                is_streaming=True,
                                delta=True,
                            )

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=None)

                    # if the model is finished generating
                    else:
                        # check to make sure we haven't "forgotten" to stream
                        #   any tokens that were generated but previously
                        #   matched by partial json parsing
                        # only happens if we are NOT using guided decoding
                        auto_tools_called = False
                        if tool_parser:
                            auto_tools_called = len(
                                tool_parser.prev_tool_call_arr) > 0
                            index = len(tool_parser.prev_tool_call_arr
                                        ) - 1 if auto_tools_called else 0
                        else:
                            index = 0

                        if self._should_check_for_unstreamed_tool_arg_tokens(
                                delta_message, output) and tool_parser:
                            latest_delta_len = 0
                            if ((isinstance(
                                    delta_message.tool_calls[0].function,
                                    DeltaFunctionCall)) and isinstance(
                                        delta_message.tool_calls[0].function.
                                        arguments, str)):
                                latest_delta_len = len(
                                    delta_message.tool_calls[0].function.
                                    arguments)

                            # get the expected call based on partial JSON
                            # parsing which "autocompletes" the JSON
                            expected_call = json.dumps(
                                tool_parser.prev_tool_call_arr[index].get(
                                    "arguments", {}),
                                ensure_ascii=False)

                            # get what we've streamed so far for arguments
                            # for the current tool
                            actual_call = tool_parser.streamed_args_for_tool[
                                index]
                            if (latest_delta_len > 0):
                                actual_call = actual_call[:-latest_delta_len]

                            # check to see if there's anything left to stream
                            remaining_call = expected_call.replace(
                                actual_call, "", 1)
                            # set that as a delta message
                            delta_message = DeltaMessage(tool_calls=[
                                DeltaToolCall(index=index,
                                              function=DeltaFunctionCall(
                                                  arguments=remaining_call).
                                              model_dump(exclude_none=True))
                            ])

                        # Send the finish response for each request.n only once
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=output.finish_reason
                            if not auto_tools_called else "tool_calls",
                            stop_reason=output.stop_reason)

                        finish_reason_sent[i] = True

                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name)

                    # handle usage stats if requested & if continuous
                    if include_continuous_usage:
                        completion_tokens = previous_num_tokens[i]
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=num_prompt_tokens + completion_tokens,
                        )

                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            # once the final token is handled, if stream_options.include_usage
            # is sent, send the usage
            if include_usage:
                completion_tokens = sum(previous_num_tokens)
                final_usage = UsageInfo(prompt_tokens=num_prompt_tokens,
                                        completion_tokens=completion_tokens,
                                        total_tokens=num_prompt_tokens +
                                        completion_tokens)
                if self.enable_prompt_tokens_details and num_cached_tokens:
                    final_usage.prompt_tokens_details = PromptTokenUsageInfo(
                        cached_tokens=num_cached_tokens)

                final_usage_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage)
                final_usage_data = (final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True))
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            num_completion_tokens = sum(previous_num_tokens)
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens,
            )

            # Log complete streaming response if output logging is enabled
            if self.enable_log_outputs and self.request_logger:
                # Log the complete response for each choice
                for i in range(num_choices):
                    full_text = (
                        previous_texts[i]
                        if previous_texts and i < len(previous_texts) else
                        f"<streaming_complete: {previous_num_tokens[i]} tokens>"
                    )
                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=full_text,
                        output_token_ids=
                        None,  # Consider also logging all token IDs
                        finish_reason="streaming_complete",
                        is_streaming=True,
                        delta=False,
                    )

        except Exception as e:
            # TODO: Use a vllm-specific Validation Error
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
    ) -> Union[ErrorResponse, ChatCompletionResponse]:

        created_time = int(time.time())
        final_res: Optional[RequestOutput] = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        assert final_res is not None

        choices: list[ChatCompletionResponseChoice] = []

        role = self.get_chat_request_role(request)
        for output in final_res.outputs:
            token_ids = output.token_ids
            out_logprobs = output.logprobs

            if request.logprobs and request.top_logprobs is not None:
                assert out_logprobs is not None, "Did not output logprobs"
                logprobs = self._create_chat_logprobs(
                    token_ids=token_ids,
                    top_logprobs=out_logprobs,
                    num_output_top_logprobs=request.top_logprobs,
                    tokenizer=tokenizer,
                    return_as_token_id=request.return_tokens_as_token_ids,
                )
            else:
                logprobs = None

            if self.use_harmony:
                reasoning_content, final_content, is_tool_call = (
                    parse_chat_output(token_ids))
                if not request.include_reasoning:
                    reasoning_content = None

                if is_tool_call:
                    # TODO(woosuk): Implement tool call for gpt-oss.
                    # For now, only Responses API supports tool call for
                    # gpt-oss.
                    raise NotImplementedError(
                        "Tool call in Chat Completion API is not supported "
                        "for gpt-oss yet. Please use Responses API instead.")
                else:
                    # Normal message
                    message = ChatMessage(
                        role=role,
                        reasoning_content=reasoning_content,
                        content=final_content,
                    )

                choice_data = ChatCompletionResponseChoice(
                    index=output.index,
                    message=message,
                    logprobs=logprobs,
                    finish_reason="tool_calls" if is_tool_call else
                    output.finish_reason if output.finish_reason else "stop",
                    stop_reason=output.stop_reason,
                )
                choices.append(choice_data)
                continue

            enable_thinking: bool = request.chat_template_kwargs.get("enable_thinking", True) if request.chat_template_kwargs else True
            if self.reasoning_parser and enable_thinking:
                try:
                    reasoning_parser = self.reasoning_parser(tokenizer)
                except RuntimeError as e:
                    logger.exception("Error in reasoning parser creation.")
                    return self.create_error_response(str(e))
                # If the reasoning parser is enabled,
                # tool calls are extracted exclusively from the content.
                reasoning_content, content = (
                    reasoning_parser.extract_reasoning_content(
                        output.text, request=request))
                if not request.include_reasoning:
                    reasoning_content = None
            else:
                reasoning_content = None
                content = output.text

            auto_tools_called = False
            # if auto tools are not enabled, and a named tool choice using
            #   outlines is not being used
            if (not self.enable_auto_tools or not self.tool_parser) and \
                (not isinstance(request.tool_choice,
                                ChatCompletionNamedToolChoiceParam
                                ) and request.tool_choice != "required"):
                message = ChatMessage(role=role,
                                      reasoning_content=reasoning_content,
                                      content=content)

            # if the request uses tools and specified a tool choice
            elif request.tool_choice and type(
                    request.tool_choice) is ChatCompletionNamedToolChoiceParam:

                tool_call_class = MistralToolCall if isinstance(
                    tokenizer, MistralTokenizer) else ToolCall
                message = ChatMessage(
                    role=role,
                    reasoning_content=reasoning_content,
                    content="",
                    tool_calls=[
                        tool_call_class(function=FunctionCall(
                            name=request.tool_choice.function.name,
                            arguments=content,
                        ))
                    ],
                )

            elif request.tool_choice and request.tool_choice == "required":
                tool_call_class = MistralToolCall if isinstance(
                    tokenizer, MistralTokenizer) else ToolCall

                # the fields of FunctionDefinition are a superset of the
                # tool call outputs and can be used for parsing
                assert content is not None
                tool_calls = TypeAdapter(
                    list[FunctionDefinition]).validate_json(content)
                message = ChatMessage(
                    role=role,
                    content="",
                    reasoning_content=reasoning_content,
                    tool_calls=[
                        tool_call_class(function=FunctionCall(
                            name=tool_call.name,
                            arguments=json.dumps(tool_call.parameters,
                                                 ensure_ascii=False)))
                        for tool_call in tool_calls
                    ])

            # if the request doesn't use tool choice
            # OR specifies to not use a tool
            elif not request.tool_choice or request.tool_choice == "none":

                message = ChatMessage(role=role,
                                      reasoning_content=reasoning_content,
                                      content=content)

            # handle when there are tools and tool choice is auto
            elif request.tools and (
                    request.tool_choice == "auto"
                    or request.tool_choice is None) and self.enable_auto_tools \
                    and self.tool_parser:

                try:
                    tool_parser = self.tool_parser(tokenizer)
                except RuntimeError as e:
                    logger.exception("Error in tool parser creation.")
                    return self.create_error_response(str(e))

                tool_call_info = tool_parser.extract_tool_calls(
                    content if content is not None else "", request=request)
                # In the OpenAI API the finish_reason is "tools_called"
                # if the tool choice is auto and the model produced a tool
                # call. The same is not true for named function calls
                auto_tools_called = tool_call_info.tools_called
                if tool_call_info.tools_called:
                    message = ChatMessage(role=role,
                                          reasoning_content=reasoning_content,
                                          content=tool_call_info.content,
                                          tool_calls=tool_call_info.tool_calls)

                else:
                    # FOR NOW make it a chat message; we will have to detect
                    # the type to make it later.
                    ret_content = content

                    # try to use content return from tool parser first,
                    # tool parser may do some modify for the content.
                    if (tool_call_info.content
                            and len(tool_call_info.content) > 0):
                        ret_content = tool_call_info.content

                    message = ChatMessage(role=role,
                                          reasoning_content=reasoning_content,
                                          content=ret_content)

            # undetermined case that is still important to handle
            else:
                logger.error(
                    "Error in chat_completion_full_generator - cannot determine"
                    " if tools should be extracted. Returning a standard chat "
                    "completion.")
                message = ChatMessage(role=role,
                                      reasoning_content=reasoning_content,
                                      content=content)

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason="tool_calls" if auto_tools_called else
                output.finish_reason if output.finish_reason else "stop",
                stop_reason=output.stop_reason)

            choices.append(choice_data)

        if request.echo:
            last_msg_content: Union[str, list[dict[str, str]]] = ""
            if (conversation and "content" in conversation[-1]
                    and conversation[-1].get("role") == role):
                last_msg_content = conversation[-1]["content"] or ""
            if isinstance(last_msg_content, list):
                last_msg_content = "\n".join(msg['text']
                                             for msg in last_msg_content)

            for choice in choices:
                full_message = last_msg_content + (choice.message.content
                                                   or "")
                choice.message.content = full_message

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        if final_res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs)
        usage = UsageInfo(prompt_tokens=num_prompt_tokens,
                          completion_tokens=num_generated_tokens,
                          total_tokens=num_prompt_tokens +
                          num_generated_tokens)
        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=final_res.num_cached_tokens)

        request_metadata.final_usage_info = usage

        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            prompt_logprobs=clamp_prompt_logprobs(final_res.prompt_logprobs),
            kv_transfer_params=final_res.kv_transfer_params,
        )

        # Log complete response if output logging is enabled
        if self.enable_log_outputs and self.request_logger:
            for choice in choices:
                output_text = ""
                if choice.message.content:
                    output_text = choice.message.content
                elif choice.message.tool_calls:
                    # For tool calls, log the function name and arguments
                    tool_call_descriptions = []
                    for tool_call in choice.message.tool_calls:
                        if hasattr(tool_call.function, "name") and hasattr(
                                tool_call.function, "arguments"):
                            tool_call_descriptions.append(
                                f"{tool_call.function.name}({tool_call.function.arguments})"
                            )
                    tool_calls_str = ", ".join(tool_call_descriptions)
                    output_text = f"[tool_calls: {tool_calls_str}]"

                if output_text:
                    # Get the corresponding output token IDs
                    output_token_ids = None
                    if choice.index < len(final_res.outputs):
                        output_token_ids = final_res.outputs[
                            choice.index].token_ids

                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=output_text,
                        output_token_ids=output_token_ids,
                        finish_reason=choice.finish_reason,
                        is_streaming=False,
                        delta=False,
                    )

        return response