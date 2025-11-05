import json
from typing import Any, Dict, List, Optional

import boto3


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Tool Usage:
- **search_course_content**: Use for questions about specific course content or detailed educational materials
- **get_course_outline**: Use for questions about course structure, syllabus, lesson list, or course overview
- **Sequential tool calling**: You can use tools up to 2 times in separate rounds to refine searches or gather additional information
  * Round 1: Make initial tool call(s) to gather information
  * Round 2: After seeing results, you may make additional tool call(s) if needed for comparisons, refinements, or multi-part questions
  * Examples: (1) Get course outline to find lesson title → search for that topic, (2) Search one course → search another for comparison
- Synthesize tool results into accurate, fact-based responses
- If tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline/structure questions**: Use get_course_outline tool to retrieve course title, course link, and complete lesson information (lesson numbers and titles)
- **Course content questions**: Use search_course_content tool to find specific information
- **Multi-step queries**: Use sequential tool calls when you need information from multiple sources or need to refine based on initial results
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the tool results", "in round 1", or "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(
        self,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_session_token: str,
        aws_region: str,
        model_id: str,
    ):
        # Build client config
        client_config = {
            "service_name": "bedrock-runtime",
            "region_name": aws_region,
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
        }

        # Add session token if provided (for temporary credentials)
        if aws_session_token:
            client_config["aws_session_token"] = aws_session_token

        self.client = boto3.client(**client_config)
        self.model_id = model_id

        # Pre-build base API parameters
        self.base_params = {"temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters for Bedrock
        request_body = {
            **self.base_params,
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            request_body["tools"] = tools
            request_body["tool_choice"] = {"type": "auto"}

        # Get response from Claude via Bedrock
        bedrock_response = self.client.invoke_model(
            modelId=self.model_id, body=json.dumps(request_body)
        )

        # Parse response
        response_body = json.loads(bedrock_response["body"].read())

        # Handle tool execution if needed
        if response_body["stop_reason"] == "tool_use" and tool_manager:
            return self._handle_tool_execution(response_body, request_body, tool_manager)

        # Return direct response
        return response_body["content"][0]["text"]

    def _handle_tool_execution(
        self, initial_response: Dict[str, Any], base_params: Dict[str, Any], tool_manager
    ):
        """
        Handle execution of tool calls with support for sequential rounds (up to 2).

        Round flow:
        - Round 1: Execute tools from initial_response
        - Round 2: If Claude returns tool_use again, execute and get final response
        - Terminate: After 2 rounds OR when no tool_use blocks found

        Args:
            initial_response: The response body containing tool use requests
            base_params: Base API parameters (includes messages, system, and tools)
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        MAX_TOOL_ROUNDS = 2

        # Start with existing messages
        messages = base_params["messages"].copy()

        # Add AI's initial tool use response
        messages.append({"role": "assistant", "content": initial_response["content"]})

        current_round = 1
        current_response = initial_response
        tools = base_params.get("tools")

        while current_round <= MAX_TOOL_ROUNDS:
            # Execute tools from current response
            tool_results = self._execute_tools_from_response(current_response, tool_manager)

            # Add tool results to messages
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Decide whether to include tools in next call
            is_last_round = current_round == MAX_TOOL_ROUNDS
            include_tools = not is_last_round

            # Make next API call
            next_response = self._make_followup_call(
                messages, base_params["system"], include_tools, tools
            )

            # Check termination conditions
            if next_response["stop_reason"] != "tool_use":
                # No more tool calls - return final text
                return self._extract_text_from_response(next_response)

            if is_last_round:
                # Hit max rounds - extract text from response (ignore tool_use blocks)
                return self._extract_text_from_response(next_response)

            # Prepare for next round
            messages.append({"role": "assistant", "content": next_response["content"]})
            current_response = next_response
            current_round += 1

        # Fallback (should never reach here)
        return "Unable to generate response after maximum tool rounds."

    def _execute_tools_from_response(
        self, response: Dict[str, Any], tool_manager
    ) -> List[Dict[str, Any]]:
        """
        Extract and execute all tool_use blocks from a response.

        Args:
            response: API response containing tool_use blocks
            tool_manager: Manager to execute tools

        Returns:
            List of tool_result dicts
        """
        tool_results = []
        for content_block in response["content"]:
            if content_block.get("type") == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block["name"], **content_block["input"]
                    )
                except Exception as e:
                    # Return error as tool result
                    tool_result = f"Error executing tool '{content_block['name']}': {str(e)}"

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block["id"],
                        "content": tool_result,
                    }
                )
        return tool_results

    def _make_followup_call(
        self, messages: List[Dict], system_content: str, include_tools: bool, tools: Optional[List]
    ) -> Dict[str, Any]:
        """
        Make a followup API call with current message history.

        Args:
            messages: Current message history
            system_content: System prompt content
            include_tools: Whether to include tools in this call
            tools: Tool definitions (if include_tools is True)

        Returns:
            API response body
        """
        request_body = {
            **self.base_params,
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages,
            "system": system_content,
        }

        # Only include tools if not on final round
        if include_tools and tools:
            request_body["tools"] = tools
            request_body["tool_choice"] = {"type": "auto"}

        response = self.client.invoke_model(modelId=self.model_id, body=json.dumps(request_body))

        return json.loads(response["body"].read())

    def _extract_text_from_response(self, response: Dict[str, Any]) -> str:
        """
        Extract text content from response, ignoring tool_use blocks.

        Used for normal responses and when max rounds reached but Claude still
        wants to use tools.

        Args:
            response: API response body

        Returns:
            Extracted text or fallback message
        """
        text_blocks = [
            block["text"] for block in response["content"] if block.get("type") == "text"
        ]

        if not text_blocks:
            return "Unable to provide a complete response."

        return " ".join(text_blocks)
