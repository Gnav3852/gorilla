import json
import os
import time
from datetime import datetime
from typing import Any, List, Dict, Optional

import requests
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.utils import (
    convert_to_function_call,
    default_decode_ast_prompting,
)
from overrides import override

try:
    from openai_harmony import (
        SystemContent,
        Message,
        Conversation,
        Role,
        Author,
        load_harmony_encoding,
        HarmonyEncodingName,
    )

    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False
    print("Warning: openai-harmony not available. GPT-OSS support will be limited.")


class GPTOSSHandler(OSSHandler):
    """
    Handler for GPT-OSS models using Harmony format.

    GPT-OSS models require special Harmony format encoding/decoding
    and use remote inference via vLLM or similar serving infrastructure.
    """

    def __init__(self, model_name, temperature) -> None:
        
        if not model_name.startswith("openai/"):
            model_name = f"openai/{model_name}"

        super().__init__(model_name, temperature)
        self.is_fc_model = True
        self.model_style = ModelStyle.OSSMODEL
        self.harmony_available = HARMONY_AVAILABLE

        # Harmony encoding for GPT-OSS
        if not HARMONY_AVAILABLE:
            raise ImportError("openai-harmony is required for GPT-OSS models")
        try:
            self.harmony_encoding = load_harmony_encoding(
                HarmonyEncodingName.HARMONY_GPT_OSS
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Harmony encoding: {e}")

    @override
    def inference(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
    ):
        """Use the generic BaseHandler inference pipeline instead of OSSHandler's."""
        return BaseHandler.inference(
            self, test_entry, include_input_log, exclude_state_log
        )

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        return default_decode_ast_prompting(result, language, has_tool_call_tag)

    @override
    def decode_execute(self, result, has_tool_call_tag):
        return convert_to_function_call(result)

    @override
    def _format_prompt(self, messages, function):
        """Format prompt using Harmony format for GPT-OSS models."""
        if not HARMONY_AVAILABLE or not self.harmony_encoding:
            raise RuntimeError("Harmony encoding required for GPT-OSS models")
        return self._format_prompt_harmony(messages, function)

    def _format_prompt_harmony(self, messages, function):
        """Format prompt using Harmony format."""
        # Convert function docs to Harmony format
        harmony_tools = self._convert_functions_to_harmony(function) if function else {}

        # Create system content with tools
        system_content = SystemContent.new()

        # Add tools if available
        if harmony_tools:
            system_content = system_content.with_tools(harmony_tools)

        # Add conversation start date
        system_content = system_content.with_conversation_start_date(
            datetime.now().strftime("%Y-%m-%d")
        )

        # Create conversation
        conversation = Conversation()

        # Add system message
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
        conversation.messages.append(system_message)

        # Add remaining messages
        for msg in messages:
            if msg["role"] == "system":
                continue  # Skip system messages as we already added one

            role = Role.USER if msg["role"] == "user" else Role.ASSISTANT
            content = msg["content"]

            # Handle tool calls in assistant messages
            if msg["role"] == "assistant" and "tool_calls" in msg:
                # Add assistant message with tool calls
                assistant_msg = Message.from_role_and_content(Role.ASSISTANT, content)
                conversation.messages.append(assistant_msg)

                # Add tool responses
                for tool_call in msg["tool_calls"]:
                    tool_name = tool_call.get("function", {}).get("name", "")
                    tool_args = tool_call.get("function", {}).get("arguments", "{}")
                    tool_msg = Message.from_role_and_content(
                        Role.TOOL, f"Tool: {tool_name}, Args: {tool_args}"
                    )
                    conversation.messages.append(tool_msg)
            else:
                message = Message.from_role_and_content(role, content)
                conversation.messages.append(message)

        # Encode using Harmony and return token ids directly
        return self.harmony_encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )

    def _convert_functions_to_harmony(self, functions):
        """Convert BFCL function specs to Harmony namespace mapping.

        The latest ``openai_harmony`` API expects tools to be organised into a
        :class:`NamespaceConfig` which itself contains ``Tool`` objects.  Older
        versions of the library exposed ``ToolNamespaceConfig`` and
        ``ToolDescription`` instead.  To maintain compatibility we try the new
        imports first and gracefully fall back to the old ones if necessary.

        Parameters
        ----------
        functions: list
            List of BFCL function dictionaries.

        Returns
        -------
        dict
            Mapping of ``{namespace.name: namespace}`` suitable to be passed to
            :func:`SystemContent.with_tools`.
        """

        if not functions:
            return {}

        try:  # Newer ``openai_harmony`` versions
            from openai_harmony import NamespaceConfig, Tool, Function

            namespace = NamespaceConfig(name="functions", tools={})
            for func in functions:
                namespace.tools[func["name"]] = Tool(
                    function=Function(
                        name=func["name"],
                        description=func.get("description", ""),
                        parameters=func.get("parameters", {}),
                    )
                )
        except Exception:
            # Fallback for older versions where ToolNamespaceConfig/ToolDescription
            # are used instead of NamespaceConfig/Tool/Function
            try:
                from openai_harmony import ToolNamespaceConfig as NamespaceConfig
                from openai_harmony import ToolDescription as Function

                namespace = NamespaceConfig(name="functions", tools={})
                for func in functions:
                    namespace.tools[func["name"]] = Function(
                        name=func["name"],
                        description=func.get("description", ""),
                        parameters=func.get("parameters", {}),
                    )
            except Exception:
                # If harmony isn't available or another unexpected failure occurs
                # simply return empty tools to avoid crashing the caller.
                return {}

        return {namespace.name: namespace}

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        """Parse response from GPT-OSS model using Harmony."""
        if not HARMONY_AVAILABLE or not self.harmony_encoding:
            raise RuntimeError("Harmony encoding required for GPT-OSS models")
        return self._parse_harmony_response(api_response)

    def _parse_harmony_response(self, api_response: Any) -> dict:
        """Parse response using Harmony format, returning tool calls and text."""
        if not self.harmony_encoding:
            raise RuntimeError("Harmony encoding required for GPT-OSS models")

        try:
            data = api_response if isinstance(api_response, dict) else api_response

            # ----- Token extraction -----
            token_ids: Optional[List[int]] = None
            if hasattr(data, "output") and getattr(data, "output"):
                first = data.output[0]
                token_ids = getattr(first, "token_ids", None) or getattr(
                    first, "tokens", None
                )

            if (
                token_ids is None
                and hasattr(data, "choices")
                and getattr(data, "choices")
            ):
                choice = data.choices[0]
                token_ids = getattr(choice, "token_ids", None)
                if token_ids is None and getattr(choice, "logprobs", None) is not None:
                    token_ids = getattr(choice.logprobs, "token_ids", None) or getattr(
                        choice.logprobs, "tokens", None
                    )

            output_section = None
            if token_ids is None:
                if isinstance(data, dict):
                    output_section = data.get("output", [])
                    if output_section and isinstance(output_section[0], int):
                        token_ids = output_section
                else:
                    output_section = getattr(data, "output", [])

            assistant_messages: List[Dict[str, Any]] = []
            tool_call_ids: List[str] = []
            tool_names: List[str] = []
            tool_calls_exec: List[Dict[str, Any]] = []
            final_messages: List[str] = []
            reasoning_messages: List[str] = []

            # ----- Parse token based responses -----
            if token_ids:
                parsed_messages = self.harmony_encoding.parse_messages_from_completion_tokens(
                    token_ids, Role.ASSISTANT
                )

                for msg in parsed_messages:
                    if msg.role != Role.ASSISTANT:
                        continue

                    recipient = getattr(msg, "recipient", None)
                    channel = getattr(msg, "channel", None)

                    if recipient:
                        try:
                            args = json.loads(msg.content)
                        except Exception:
                            args = msg.content

                        name = recipient.split("functions.")[-1]
                        call_id = name
                        tool_calls_exec.append({name: args})
                        tool_names.append(name)
                        tool_call_ids.append(call_id)

                        assistant_messages.append(
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": call_id,
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": (
                                                json.dumps(args)
                                                if isinstance(args, (dict, list))
                                                else str(args)
                                            ),
                                        },
                                    }
                                ],
                                "content": "",
                            }
                        )
                    else:
                        if channel == "analysis":
                            reasoning_messages.append(msg.content)
                        else:
                            final_messages.append(msg.content)

                        assistant_messages.append(
                            {"role": "assistant", "content": msg.content}
                        )
            # ----- Parse structured JSON responses -----
            else:
                output_list = output_section or []
                assistant_message = {"role": "assistant", "content": ""}
                reasoning_content = ""
                for item in output_list:
                    if item.get("type") == "function_call":
                        name = item.get("name", "")
                        arguments = item.get("arguments", "")
                        call_id = item.get("call_id", name)
                        tool_calls_exec.append({name: arguments})
                        tool_names.append(name)
                        tool_call_ids.append(call_id)
                        assistant_message.setdefault("tool_calls", []).append(
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": arguments},
                            }
                        )
                    elif item.get("type") == "message" and item.get("role") == "assistant":
                        content = item.get("content", "")
                        if isinstance(content, list):
                            text = "".join(ci.get("text", "") for ci in content)
                        else:
                            text = content
                        assistant_message["content"] += text
                        final_messages.append(text)
                    elif item.get("type") == "reasoning":
                        reasoning_content = "".join(
                            ci.get("text", "") for ci in item.get("content", [])
                        )
                if reasoning_content:
                    reasoning_messages.append(reasoning_content)
                assistant_messages.append(assistant_message)

            final_text = final_messages[-1] if final_messages else ""
            tool_calls_chat = [
                {"id": tc_id, "name": name}
                for tc_id, name in zip(tool_call_ids, tool_names)
            ]

            return {
                "model_responses": tool_calls_exec if tool_calls_exec else final_text,
                "final_response": final_text,
                "model_responses_message_for_chat_history": assistant_messages,
                "model_responses_decoded": tool_names,
                "tool_calls": tool_calls_chat,
                "tool_call_ids": tool_call_ids,
                "reasoning_content": "\n".join(reasoning_messages),
                "input_token": (
                    data.get("usage", {}).get("input_tokens", 0)
                    if isinstance(data, dict)
                    else getattr(getattr(data, "usage", None), "prompt_tokens", 0)
                ),
                "output_token": (
                    data.get("usage", {}).get("output_tokens", 0)
                    if isinstance(data, dict)
                    else getattr(getattr(data, "usage", None), "completion_tokens", 0)
                ),
            }

        except Exception as e:
            raise RuntimeError(f"Harmony response parsing failed: {e}")

    def _add_assistant_message(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        """Common helper to append assistant messages to the chat history."""
        messages = model_response_data.get("model_responses_message_for_chat_history")
        if messages:
            inference_data["message"].extend(messages)
        else:
            inference_data["message"].append(
                {
                    "role": "assistant",
                    "content": model_response_data["model_responses"],
                }
            )
        return inference_data

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        return self._add_assistant_message(inference_data, model_response_data)

    @override
    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        return self._add_assistant_message(inference_data, model_response_data)


    #### FC methods ####

    @override
    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["message"] = []
        return inference_data

    @override
    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry.get("function", [])
        if not self.harmony_available or not self.harmony_encoding:
            raise RuntimeError("Harmony encoding required for GPT-OSS models")
        inference_data["tools"] = self._convert_functions_to_harmony(functions)
        return inference_data

    def _build_conversation(self, messages: List[Dict], tools: Dict) -> Conversation:
        system_content = SystemContent.new().with_conversation_start_date(
            datetime.now().strftime("%Y-%m-%d")
        )
        if tools:
            system_content = system_content.with_tools(tools)

        harmony_messages: List[Message] = [
            Message.from_role_and_content(Role.SYSTEM, system_content)
        ]

        for msg in messages:
            role = msg.get("role")
            if role == "user":
                harmony_messages.append(
                    Message.from_role_and_content(Role.USER, msg.get("content", ""))
                )
            elif role == "assistant":
                assistant_msg = Message.from_role_and_content(
                    Role.ASSISTANT, msg.get("content", "")
                ).with_channel("final")
                harmony_messages.append(assistant_msg)
                for tool_call in msg.get("tool_calls", []):
                    fn = tool_call.get("function", {})
                    tool_msg = (
                        Message.from_role_and_content(
                            Role.ASSISTANT, fn.get("arguments", "")
                        )
                        .with_recipient(f"functions.{fn.get('name', '')}")
                        .with_channel("commentary")
                    )
                    harmony_messages.append(tool_msg)
            elif role == "tool":
                tool_msg = (
                    Message.from_author_and_content(
                        Author.new(Role.TOOL, f"functions.{msg.get('name', '')}"),
                        msg.get("content", ""),
                    )
                    .with_recipient("assistant")
                    .with_channel("commentary")
                )
                harmony_messages.append(tool_msg)

        return Conversation.from_messages(harmony_messages)

    @override
    def _query_FC(self, inference_data: dict):
        if not self.harmony_available or not self.harmony_encoding:
            raise RuntimeError("Harmony encoding not available for GPT-OSS FC")

        message: list[dict] = inference_data["message"]
        tools: Dict = inference_data.get("tools", {})
        conversation = self._build_conversation(message, tools)

        conversation_tokens = self.harmony_encoding.render_conversation_for_completion(
            conversation=conversation,
            next_turn_role=Role.ASSISTANT,
        )
        prompt = self.harmony_encoding.decode_utf8(conversation_tokens)
        inference_data["inference_input_log"] = {"prompt": prompt}

        input_token_count = len(conversation_tokens)
        if self.max_context_length < input_token_count + 2:
            leftover_tokens_count = 1000
        else:
            leftover_tokens_count = min(
                4096, self.max_context_length - input_token_count - 2
            )

        payload = {
           
            "model": self.model_name,
            "input": prompt,
            "temperature": self.temperature,
            "max_output_tokens": leftover_tokens_count,
        }

        start_time = time.time()
        response = requests.post(
            f"{self.base_url}/responses", json=payload, timeout=72000
        )
        end_time = time.time()
        response.raise_for_status()
        return response.json(), end_time - start_time

    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        return self._parse_harmony_response(api_response)

    @override
    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    @override
    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    @override
    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        for execution_result, tool in zip(
            execution_results, model_response_data.get("tool_calls", [])
        ):
            inference_data["message"].append(
                {
                    "role": "tool",
                    "tool_call_id": tool.get("id", ""),
                    "name": tool.get("name", ""),
                    "content": execution_result,
                }
            )
        return inference_data
