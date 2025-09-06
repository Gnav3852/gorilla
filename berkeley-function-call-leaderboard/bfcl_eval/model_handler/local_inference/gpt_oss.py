import json
import os
import time
from datetime import datetime
from typing import Any, List, Dict

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
        HarmonyEncodingName
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
        super().__init__(model_name, temperature)
        self.is_fc_model = True
        self.model_style = ModelStyle.OSSMODEL
        self.harmony_available = HARMONY_AVAILABLE
        
        # Harmony encoding for GPT-OSS
        self.harmony_encoding = None
        if HARMONY_AVAILABLE:
            try:
                self.harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            except Exception as e:
                print(f"Warning: Failed to load Harmony encoding: {e}")
                self.harmony_encoding = None

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
        """
        Format prompt using Harmony format for GPT-OSS models.
        Falls back to standard format if Harmony is not available.
        """
        if not HARMONY_AVAILABLE or not self.harmony_encoding:
            return self._format_prompt_fallback(messages, function)
        
        try:
            return self._format_prompt_harmony(messages, function)
        except Exception as e:
            print(f"Warning: Harmony formatting failed, falling back: {e}")
            return self._format_prompt_fallback(messages, function)

    def _format_prompt_harmony(self, messages, function):
        """Format prompt using Harmony format."""
        # Convert function docs to Harmony format
        harmony_tools = (
            self._convert_functions_to_harmony(function) if function else {}
        )

        # Create system content with tools
        system_content = SystemContent.new()

        # Add tools if available
        for ns in harmony_tools.values():
            system_content = system_content.with_tools(ns)
        
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
                        Role.TOOL,
                        f"Tool: {tool_name}, Args: {tool_args}"
                    )
                    conversation.messages.append(tool_msg)
            else:
                message = Message.from_role_and_content(role, content)
                conversation.messages.append(message)
        
        # Encode using Harmony
        token_ids = self.harmony_encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )

        # Return the raw token ids so that `_query_prompting` can send them
        # directly in the request body (under the `input` field) without
        # decoding. This preserves Harmony markers and avoids a round-trip
        # through text.
        return token_ids

    def _format_prompt_fallback(self, messages, function):
        """Fallback formatting when Harmony is not available."""
        formatted_prompt = ""
        
        # Add system message
        if messages and messages[0]["role"] == "system":
            formatted_prompt += f"System: {messages[0]['content']}\n\n"
            messages = messages[1:]
        
        # Add function definitions
        if function:
            formatted_prompt += "Available functions:\n"
            for func in function:
                formatted_prompt += f"- {func['name']}: {func['description']}\n"
                if 'parameters' in func:
                    formatted_prompt += f"  Parameters: {json.dumps(func['parameters'], indent=2)}\n"
            formatted_prompt += "\n"
        
        # Add conversation
        for msg in messages:
            role = msg["role"].title()
            content = msg["content"]
            formatted_prompt += f"{role}: {content}\n"
        
        formatted_prompt += "Assistant: "
        return formatted_prompt

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
        """Parse response from GPT-OSS model."""
        try:
            # Try to parse as Harmony format first
            if HARMONY_AVAILABLE and self.harmony_encoding:
                return self._parse_harmony_response(api_response)
            else:
                return self._parse_fallback_response(api_response)
        except Exception as e:
            print(f"Warning: Response parsing failed, using fallback: {e}")
            return self._parse_fallback_response(api_response)

    def _parse_harmony_response(self, api_response):
        """Parse response using Harmony format."""
        if not self.harmony_encoding:
            return self._parse_fallback_response(api_response)

        try:
            token_ids = None

            # Try different locations for token ids depending on the response schema
            if hasattr(api_response, "output") and api_response.output:
                first = api_response.output[0]
                token_ids = getattr(first, "token_ids", None) or getattr(first, "tokens", None)

            if token_ids is None and hasattr(api_response, "choices") and api_response.choices:
                choice = api_response.choices[0]
                token_ids = getattr(choice, "token_ids", None)
                if token_ids is None and getattr(choice, "logprobs", None) is not None:
                    token_ids = getattr(choice.logprobs, "token_ids", None) or getattr(choice.logprobs, "tokens", None)

            if not token_ids:
                return self._parse_fallback_response(api_response)

            parsed_messages = self.harmony_encoding.parse_messages_from_completion_tokens(
                token_ids, Role.ASSISTANT
            )

            assistant_messages: List[Dict[str, Any]] = []
            tool_call_ids: List[str] = []
            tool_names: List[str] = []
            tool_calls: List[Dict[str, Any]] = []
            final_messages: List[str] = []
            reasoning_messages: List[str] = []

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

                    tool_calls.append({recipient: args})
                    tool_names.append(recipient)
                    tool_call_ids.append(recipient)

                    assistant_messages.append(
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": recipient,
                                    "type": "function",
                                    "function": {
                                        "name": recipient,
                                        "arguments": json.dumps(args)
                                        if isinstance(args, (dict, list))
                                        else str(args),
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

            model_responses: Any
            if tool_calls:
                model_responses = tool_calls
            else:
                # Use the last non-analysis message as final response
                model_responses = final_messages[-1] if final_messages else ""

            return {
                "model_responses": model_responses,
                "model_responses_message_for_chat_history": assistant_messages,
                "model_responses_decoded": tool_names,
                "tool_call_ids": tool_call_ids,
                "reasoning_content": "\n".join(reasoning_messages),
                "input_token": getattr(api_response.usage, "prompt_tokens", 0)
                if hasattr(api_response, "usage")
                else 0,
                "output_token": getattr(api_response.usage, "completion_tokens", 0)
                if hasattr(api_response, "usage")
                else 0,
            }

        except Exception as e:
            print(f"Warning: Harmony response parsing failed: {e}")
            return self._parse_fallback_response(api_response)

    def _parse_fallback_response(self, api_response):
        """Fallback response parsing."""
        if hasattr(api_response, 'choices') and api_response.choices:
            model_response = api_response.choices[0].text
        else:
            model_response = str(api_response)
        
        return {
            "model_responses": model_response,
            "input_token": getattr(api_response.usage, 'prompt_tokens', 0) if hasattr(api_response, 'usage') else 0,
            "output_token": getattr(api_response.usage, 'completion_tokens', 0) if hasattr(api_response, 'usage') else 0,
        }

    @override
    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        """Add assistant message to conversation."""
        messages = model_response_data.get(
            "model_responses_message_for_chat_history"
        )
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

    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        messages = model_response_data.get(
            "model_responses_message_for_chat_history"
        )
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

    def _get_model_path(self):
        """Get the model path for GPT-OSS models."""
        if "20b" in self.model_name:
            return os.getenv("GPT_OSS_20B_PATH", "./models/gpt-oss-20b")
        elif "120b" in self.model_name:
            return os.getenv("GPT_OSS_120B_PATH", "./models/gpt-oss-120b")
        else:
            return os.getenv("GPT_OSS_MODEL_PATH", "./models/gpt-oss")

    def _get_vllm_endpoint(self):
        """Get the vLLM endpoint for remote inference."""
        endpoint = os.getenv("VLLM_ENDPOINT", "localhost")
        port = os.getenv("VLLM_PORT", "8000")
        return f"http://{endpoint}:{port}"

    #### FC methods ####

    @override
    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["message"] = []
        return inference_data

    @override
    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry.get("function", [])
        if self.harmony_available and self.harmony_encoding:
            inference_data["tools"] = self._convert_functions_to_harmony(functions)
        else:
            inference_data["tools"] = {}
        return inference_data

    def _build_conversation(self, messages: List[Dict], tools: Dict) -> Conversation:
        system_content = SystemContent.new().with_conversation_start_date(
            datetime.now().strftime("%Y-%m-%d")
        )
        for ns in tools.values():
            system_content = system_content.with_tools(ns)
        conversation = Conversation()
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
        conversation.messages.append(system_message)

        for msg in messages:
            role = msg.get("role")
            if role == "user":
                user_msg = Message.from_role_and_content(
                    Role.USER, msg.get("content", "")
                )
                conversation.messages.append(user_msg)
            elif role == "assistant":
                content = msg.get("content", "")
                assistant_msg = Message.from_role_and_content(Role.ASSISTANT, content)
                assistant_msg = assistant_msg.with_channel("final")
                conversation.messages.append(assistant_msg)
                for tool_call in msg.get("tool_calls", []):
                    fn = tool_call.get("function", {})
                    tool_msg = Message.from_role_and_content(
                        Role.ASSISTANT, fn.get("arguments", "")
                    )
                    tool_msg = tool_msg.with_recipient(
                        f"functions.{fn.get('name', '')}"
                    )
                    tool_msg = tool_msg.with_channel("commentary")
                    conversation.messages.append(tool_msg)
            elif role == "tool":
                tool_message = Message.from_author_and_content(
                    Author.new(Role.TOOL, f"functions.{msg.get('name', '')}"),
                    msg.get("content", ""),
                )
                tool_message = tool_message.with_recipient("assistant")
                tool_message = tool_message.with_channel("commentary")
                conversation.messages.append(tool_message)
        return conversation

    @override
    def _query_FC(self, inference_data: dict):
        if not self.harmony_available or not self.harmony_encoding:
            raise RuntimeError("Harmony encoding not available for GPT-OSS FC")

        message: list[dict] = inference_data["message"]
        tools: Dict = inference_data.get("tools", {})
        conversation = self._build_conversation(message, tools)

        token_ids = self.harmony_encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )
        inference_data["inference_input_log"] = {"token_ids": token_ids}

        input_token_count = len(token_ids)
        if self.max_context_length < input_token_count + 2:
            leftover_tokens_count = 1000
        else:
            leftover_tokens_count = min(
                4096, self.max_context_length - input_token_count - 2
            )

        decoder_config = {
            "temperature": self.temperature,
            "max_new_tokens": leftover_tokens_count,
            "stop_token_ids": self.harmony_encoding.stop_tokens_for_assistant_actions(),
        }

        payload = {
            "model": self.model_path_or_id,
            "input": token_ids,
            "decoder_config": decoder_config,
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
        data = api_response if isinstance(api_response, dict) else api_response
        model_responses = []
        assistant_message = {"role": "assistant", "content": ""}
        tool_calls = []
        reasoning_content = ""

        output = data.get("output", [])
        if output and isinstance(output[0], int):
            entries = self.harmony_encoding.parse_messages_from_completion_tokens(
                output, Role.ASSISTANT
            )
            for entry in entries:
                entry_dict = entry.to_dict()
                recipient = entry_dict.get("recipient", "")
                if recipient.startswith("functions."):
                    name = recipient.split("functions.")[-1]
                    arguments = entry_dict.get("content", [{}])[0].get("text", "")
                    model_responses.append({name: arguments})
                    call_id = f"call_{len(tool_calls)}"
                    tool_calls.append({"id": call_id, "name": name})
                    assistant_message.setdefault("tool_calls", []).append(
                        {
                            "id": call_id,
                            "function": {"name": name, "arguments": arguments},
                        }
                    )
                else:
                    text = "".join(
                        c.get("text", "") for c in entry_dict.get("content", [])
                    )
                    assistant_message["content"] += text
                    model_responses.append(text)
        else:
            for item in output:
                if item.get("type") == "function_call":
                    name = item.get("name", "")
                    arguments = item.get("arguments", "")
                    model_responses.append({name: arguments})
                    tool_calls.append({"id": item.get("call_id", ""), "name": name})
                    assistant_message.setdefault("tool_calls", []).append(
                        {
                            "id": item.get("call_id", ""),
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
                    model_responses.append(text)
                elif item.get("type") == "reasoning":
                    reasoning_content = "".join(
                        ci.get("text", "") for ci in item.get("content", [])
                    )
        assistant_message["reasoning_content"] = reasoning_content

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": assistant_message,
            "tool_calls": tool_calls,
            "reasoning_content": reasoning_content,
            "input_token": data.get("usage", {}).get("input_tokens", 0),
            "output_token": data.get("usage", {}).get("output_tokens", 0),
        }

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
    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
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
