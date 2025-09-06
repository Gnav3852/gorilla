import json
import os
import time
from datetime import datetime
from typing import Any, List, Dict

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.constants.enums import ModelStyle
from overrides import override

try:
    from openai_harmony import (
        SystemContent, 
        Message, 
        Conversation, 
        Role, 
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
        conversation.add_message(Message.from_role_and_content(Role.SYSTEM, system_content))
        
        # Add remaining messages
        for msg in messages:
            if msg["role"] == "system":
                continue  # Skip system messages as we already added one
            
            role = Role.USER if msg["role"] == "user" else Role.ASSISTANT
            content = msg["content"]
            
            # Handle tool calls in assistant messages
            if msg["role"] == "assistant" and "tool_calls" in msg:
                # Add assistant message with tool calls
                conversation.add_message(Message(role=Role.ASSISTANT, content=content))
                
                # Add tool responses
                for tool_call in msg["tool_calls"]:
                    tool_name = tool_call.get("function", {}).get("name", "")
                    tool_args = tool_call.get("function", {}).get("arguments", "{}")
                    conversation.add_message(Message(
                        role=Role.TOOL, 
                        content=f"Tool: {tool_name}, Args: {tool_args}"
                    ))
            else:
                conversation.add_message(Message(role=role, content=content))
        
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
        """Convert BFCL function format to Harmony tool format."""
        tools = {}
        
        for func in functions:
            tool_name = func["name"]
            tool_def = {
                "type": "function",
                "function": {
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {})
                }
            }
            tools[tool_name] = tool_def
        
        return tools

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
        # This would need to be implemented based on the actual Harmony response format
        # For now, fall back to standard parsing
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
        inference_data["message"].append({
            "role": "assistant",
            "content": model_response_data["model_responses"]
        })
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
