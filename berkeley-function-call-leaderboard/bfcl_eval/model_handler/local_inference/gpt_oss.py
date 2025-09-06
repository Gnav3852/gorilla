import json
import os
import time
from datetime import datetime
from typing import Any, List, Dict

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.model_style import ModelStyle
from bfcl_eval.model_handler.utils import func_doc_language_specific_pre_processing
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
    and use recipient-based tool calling instead of standard function calling.
    """
    
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OSSMODEL
        self.is_fc_model = True
        
        if HARMONY_AVAILABLE:
            try:
                self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            except Exception as e:
                print(f"Warning: Failed to load Harmony encoding: {e}")
                self.encoding = None
        else:
            self.encoding = None
    
    @override
    def _format_prompt(self, messages, function):
        """
        Format prompt using Harmony encoding for GPT-OSS.
        
        Args:
            messages: List of conversation messages
            function: List of function definitions
            
        Returns:
            Formatted prompt string (fallback) or token IDs (preferred)
        """
        if not HARMONY_AVAILABLE or self.encoding is None:
            # Fallback to basic formatting if Harmony is not available
            return self._format_prompt_fallback(messages, function)
        
        try:
            # Convert BFCL function format to Harmony tool format
            tools = self._convert_functions_to_harmony_tools(function)
            
            # Create system content with tools
            system_content = SystemContent.new().with_tools(tools)
            
            # Add conversation start date
            system_content = system_content.with_conversation_start_date(
                datetime.now().strftime("%Y-%m-%d")
            )
            
            # Convert messages to Harmony format
            harmony_messages = []
            
            # Handle system message
            if messages and messages[0]["role"] == "system":
                harmony_messages.append(
                    Message.from_role_and_content(Role.SYSTEM, system_content)
                )
                remaining_messages = messages[1:]
            else:
                # Add system message if not present
                harmony_messages.append(
                    Message.from_role_and_content(Role.SYSTEM, system_content)
                )
                remaining_messages = messages
            
            # Convert remaining messages
            for msg in remaining_messages:
                role = Role(msg["role"]) if msg["role"] in ["user", "assistant"] else Role.USER
                harmony_messages.append(
                    Message.from_role_and_content(role, msg["content"])
                )
            
            # Create conversation and render to tokens
            conversation = Conversation.from_messages(harmony_messages)
            token_ids = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
            
            # Convert token IDs back to string for compatibility with existing pipeline
            return self.tokenizer.decode(token_ids)
            
        except Exception as e:
            print(f"Warning: Harmony formatting failed, using fallback: {e}")
            return self._format_prompt_fallback(messages, function)
    
    def _format_prompt_fallback(self, messages, function):
        """
        Fallback prompt formatting when Harmony is not available.
        Uses a simplified format similar to other OSS models.
        """
        formatted_prompt = ""
        
        # Add system message with tools
        formatted_prompt += "<|im_start|>system\n"
        formatted_prompt += "You are a helpful assistant with access to tools.\n\n"
        
        if function:
            formatted_prompt += "Available tools:\n"
            for tool in function:
                formatted_prompt += f"- {tool['name']}: {tool['description']}\n"
                formatted_prompt += f"  Parameters: {json.dumps(tool['parameters'], indent=2)}\n\n"
            
            formatted_prompt += "When you need to use a tool, respond with a JSON object in this format:\n"
            formatted_prompt += '{"name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}\n'
        
        formatted_prompt += "<|im_end|>\n"
        
        # Add conversation messages
        for msg in messages:
            if msg["role"] == "system":
                continue  # Already handled above
            formatted_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        
        formatted_prompt += "<|im_start|>assistant\n"
        
        return formatted_prompt
    
    def _convert_functions_to_harmony_tools(self, functions):
        """
        Convert BFCL function format to Harmony tool format.
        
        Args:
            functions: List of BFCL function definitions
            
        Returns:
            List of Harmony-compatible tool definitions
        """
        tools = []
        
        for func in functions:
            tool = {
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            
            # Convert parameters
            if "parameters" in func and "properties" in func["parameters"]:
                for param_name, param_info in func["parameters"]["properties"].items():
                    tool["parameters"]["properties"][param_name] = {
                        "type": param_info.get("type", "string"),
                        "description": param_info.get("description", "")
                    }
                    
                    # Handle default values
                    if "default" in param_info:
                        tool["parameters"]["properties"][param_name]["default"] = param_info["default"]
                    
                    # Add to required list if needed
                    if param_info.get("required", False):
                        tool["parameters"]["required"].append(param_name)
            
            # Handle required parameters from the main function definition
            if "parameters" in func and "required" in func["parameters"]:
                tool["parameters"]["required"] = func["parameters"]["required"]
            
            tools.append(tool)
        
        return tools
    
    @override
    def _query_FC(self, inference_data: dict):
        """
        Query GPT-OSS using Harmony format.
        
        Args:
            inference_data: Dictionary containing message and tools
            
        Returns:
            Tuple of (api_response, latency)
        """
        message = inference_data["message"]
        tools = inference_data["tools"]
        
        # Format using Harmony
        formatted_prompt = self._format_prompt(message, tools)
        
        # Log the formatted prompt
        inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}
        
        # Tokenize the formatted prompt to get token count
        input_token_count = len(self.tokenizer.tokenize(formatted_prompt))
        
        # Determine the number of tokens to request
        if self.max_context_length < input_token_count + 2:
            leftover_tokens_count = 1000
        else:
            leftover_tokens_count = min(
                4096,
                self.max_context_length - input_token_count - 2,
            )
        
        # Make inference call
        start_time = time.time()
        api_response = self.client.completions.create(
            model=self.model_path_or_id,
            temperature=self.temperature,
            prompt=formatted_prompt,
            max_tokens=leftover_tokens_count,
            timeout=72000,
        )
        end_time = time.time()
        
        return api_response, end_time - start_time
    
    @override
    def decode_ast(self, result, language="Python"):
        """
        Decode GPT-OSS Harmony format response.
        
        Args:
            result: Model response text
            language: Programming language (not used for GPT-OSS)
            
        Returns:
            List of function calls in BFCL format
        """
        if not HARMONY_AVAILABLE or self.encoding is None:
            return self._decode_ast_fallback(result)
        
        try:
            # Parse Harmony response
            messages = self.encoding.parse_messages_from_completion_tokens(
                result, Role.ASSISTANT
            )
            
            # Extract tool calls from Harmony format
            tool_calls = []
            for msg in messages:
                if msg.recipient.startswith("tool"):
                    tool_call = self._parse_harmony_tool_call(msg)
                    if tool_call:
                        tool_calls.append(tool_call)
            
            return tool_calls
            
        except Exception as e:
            print(f"Warning: Harmony parsing failed, using fallback: {e}")
            return self._decode_ast_fallback(result)
    
    def _decode_ast_fallback(self, result):
        """
        Fallback AST decoding when Harmony is not available.
        """
        try:
            # Try to parse as JSON first
            if result.strip().startswith('{'):
                tool_call = json.loads(result.strip())
                if "name" in tool_call and "arguments" in tool_call:
                    return [{tool_call["name"]: tool_call["arguments"]}]
            
            # Try to extract JSON from text
            import re
            json_match = re.search(r'\{[^}]*"name"[^}]*\}', result)
            if json_match:
                tool_call = json.loads(json_match.group())
                if "name" in tool_call and "arguments" in tool_call:
                    return [{tool_call["name"]: tool_call["arguments"]}]
            
            return []
            
        except Exception as e:
            print(f"Warning: Fallback parsing failed: {e}")
            return []
    
    def _parse_harmony_tool_call(self, message):
        """
        Parse tool call from Harmony message format.
        
        Args:
            message: Harmony message object
            
        Returns:
            Dictionary in BFCL format or None if parsing fails
        """
        try:
            # Extract tool name from recipient
            tool_name = message.recipient.replace("tool:", "").strip()
            
            # Parse arguments from content
            if hasattr(message, 'content') and message.content:
                try:
                    arguments = json.loads(message.content)
                except json.JSONDecodeError:
                    # Try to extract JSON from text
                    import re
                    json_match = re.search(r'\{.*\}', message.content)
                    if json_match:
                        arguments = json.loads(json_match.group())
                    else:
                        arguments = {}
            else:
                arguments = {}
            
            return {tool_name: arguments}
            
        except Exception as e:
            print(f"Warning: Failed to parse Harmony tool call: {e}")
            return None
    
    @override
    def decode_execute(self, result):
        """
        Decode GPT-OSS response for execution.
        
        Args:
            result: Model response text
            
        Returns:
            List of executable function calls
        """
        ast_result = self.decode_ast(result)
        
        # Convert to execution format
        execution_list = []
        for function_call in ast_result:
            for key, value in function_call.items():
                execution_list.append(
                    f"{key}({','.join([f'{k}={repr(v)}' for k, v in value.items()])})"
                )
        
        return execution_list
    
    @override
    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        """
        Pre-process query for function calling mode.
        
        Args:
            inference_data: Dictionary to populate with processed data
            test_entry: Test entry containing function definitions
            
        Returns:
            Updated inference_data dictionary
        """
        inference_data["message"] = []
        return inference_data
    
    @override
    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        """
        Compile tools for function calling.
        
        Args:
            inference_data: Dictionary to populate with tools
            test_entry: Test entry containing function definitions
            
        Returns:
            Updated inference_data dictionary
        """
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]
        
        functions = func_doc_language_specific_pre_processing(functions, test_category)
        inference_data["tools"] = functions
        
        return inference_data
    
    @override
    def _parse_query_response_FC(self, api_response: any) -> dict:
        """
        Parse GPT-OSS function calling response.
        
        Args:
            api_response: API response object
            
        Returns:
            Dictionary with parsed response data
        """
        try:
            model_responses = api_response.choices[0].text
            tool_call_ids = []  # GPT-OSS doesn't use tool call IDs
            
            return {
                "model_responses": model_responses,
                "model_responses_decoded": self.decode_ast(model_responses),
                "tool_call_ids": tool_call_ids,
                "input_token": api_response.usage.prompt_tokens,
                "output_token": api_response.usage.completion_tokens,
            }
            
        except Exception as e:
            print(f"Warning: Failed to parse GPT-OSS response: {e}")
            return {
                "model_responses": "",
                "model_responses_decoded": [],
                "tool_call_ids": [],
                "input_token": 0,
                "output_token": 0,
            }
