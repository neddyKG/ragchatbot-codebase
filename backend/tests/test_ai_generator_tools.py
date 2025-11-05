"""
Tests for AIGenerator tool calling functionality

These tests verify that AIGenerator correctly:
1. Handles tool_use stop_reason and triggers tool execution
2. Constructs message history correctly for tool results
3. Makes second API call after tool execution
4. Extracts final response text
5. Handles errors gracefully
"""

import json
import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator
from tests.fixtures.mock_responses import (
    get_mock_bedrock_response_bytes,
    get_mock_direct_response,
    get_mock_final_response,
    get_mock_tool_use_response,
    get_mock_tool_use_response_outline,
    get_mock_tool_use_response_round2,
    get_mock_tool_use_response_round3,
)


class TestAIGeneratorToolCalling:
    """Test AIGenerator's tool calling orchestration"""

    def create_mock_ai_generator(self):
        """Helper to create AIGenerator with mocked Bedrock client"""
        with patch("boto3.client"):
            ai_gen = AIGenerator(
                aws_access_key_id="test_key",
                aws_secret_access_key="test_secret",
                aws_session_token="test_token",
                aws_region="us-east-1",
                model_id="test-model",
            )
            ai_gen.client = Mock()
            return ai_gen

    def test_direct_response_without_tools(self):
        """Test response when Claude doesn't use tools"""
        ai_gen = self.create_mock_ai_generator()

        # Mock Bedrock response
        mock_response = get_mock_direct_response()
        ai_gen.client.invoke_model.return_value = get_mock_bedrock_response_bytes(mock_response)

        # Call without tools
        result = ai_gen.generate_response(query="What is 2+2?", tools=None, tool_manager=None)

        # Verify single API call
        assert ai_gen.client.invoke_model.call_count == 1
        assert (
            result
            == "This is a general knowledge answer that doesn't require searching course content."
        )

    def test_tool_use_triggers_execution(self):
        """Test that tool_use stop_reason triggers tool execution flow"""
        ai_gen = self.create_mock_ai_generator()

        # Mock two responses: tool_use then final
        tool_use_response = get_mock_tool_use_response()
        final_response = get_mock_final_response()

        ai_gen.client.invoke_model.side_effect = [
            get_mock_bedrock_response_bytes(tool_use_response),
            get_mock_bedrock_response_bytes(final_response),
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = (
            "[ML Course] Supervised learning uses labeled data."
        )

        # Mock tools list
        tools = [{"name": "search_course_content", "description": "Search courses"}]

        # Execute
        result = ai_gen.generate_response(
            query="What is supervised learning?", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="supervised learning", course_name="Machine Learning"
        )

        # Verify two API calls
        assert ai_gen.client.invoke_model.call_count == 2

        # Verify final response
        assert "Supervised learning is a type of machine learning" in result

    def test_tool_results_added_to_messages(self):
        """Test tool results are correctly added to message history"""
        ai_gen = self.create_mock_ai_generator()

        tool_use_response = get_mock_tool_use_response()
        final_response = get_mock_final_response()

        captured_requests = []

        def capture_invoke(modelId, body):
            captured_requests.append(json.loads(body))
            if len(captured_requests) == 1:
                return get_mock_bedrock_response_bytes(tool_use_response)
            return get_mock_bedrock_response_bytes(final_response)

        ai_gen.client.invoke_model.side_effect = capture_invoke

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result content"

        tools = [{"name": "search_course_content"}]

        ai_gen.generate_response(query="Test query", tools=tools, tool_manager=mock_tool_manager)

        # Verify second request has correct message structure
        assert len(captured_requests) == 2
        second_request = captured_requests[1]

        # Should have 3 messages: user, assistant (with tool_use), user (with tool_result)
        assert len(second_request["messages"]) == 3
        assert second_request["messages"][0]["role"] == "user"
        assert second_request["messages"][1]["role"] == "assistant"
        assert second_request["messages"][2]["role"] == "user"

        # Check tool result structure
        tool_result_content = second_request["messages"][2]["content"]
        assert isinstance(tool_result_content, list)
        assert tool_result_content[0]["type"] == "tool_result"
        assert tool_result_content[0]["content"] == "Tool result content"

    def test_no_tool_execution_without_tool_manager(self):
        """Test that tool_use without tool_manager returns gracefully"""
        ai_gen = self.create_mock_ai_generator()

        tool_use_response = get_mock_tool_use_response()
        ai_gen.client.invoke_model.return_value = get_mock_bedrock_response_bytes(tool_use_response)

        # Call with tools but no tool_manager
        result = ai_gen.generate_response(
            query="Test", tools=[{"name": "search_course_content"}], tool_manager=None
        )

        # Should return text from first response (won't execute tool)
        assert result == "I need to search the course content to answer this question."

    def test_conversation_history_included(self):
        """Test conversation history is included in system prompt"""
        ai_gen = self.create_mock_ai_generator()

        direct_response = get_mock_direct_response()

        captured_requests = []

        def capture_invoke(modelId, body):
            captured_requests.append(json.loads(body))
            return get_mock_bedrock_response_bytes(direct_response)

        ai_gen.client.invoke_model.side_effect = capture_invoke

        ai_gen.generate_response(
            query="New question",
            conversation_history="User: Previous question\nAssistant: Previous answer",
        )

        # Verify history in system prompt
        assert len(captured_requests) == 1
        system_content = captured_requests[0]["system"]
        assert "Previous conversation:" in system_content
        assert "Previous question" in system_content
        assert "Previous answer" in system_content

    def test_tools_added_to_request(self):
        """Test tools are properly added to API request"""
        ai_gen = self.create_mock_ai_generator()

        direct_response = get_mock_direct_response()

        captured_requests = []

        def capture_invoke(modelId, body):
            captured_requests.append(json.loads(body))
            return get_mock_bedrock_response_bytes(direct_response)

        ai_gen.client.invoke_model.side_effect = capture_invoke

        tools = [
            {"name": "search_course_content", "description": "Search"},
            {"name": "get_course_outline", "description": "Get outline"},
        ]

        ai_gen.generate_response(query="Test", tools=tools)

        # Verify tools in request
        assert len(captured_requests) == 1
        request = captured_requests[0]
        assert "tools" in request
        assert request["tools"] == tools
        assert request["tool_choice"] == {"type": "auto"}

    def test_second_call_includes_tools_for_potential_round2(self):
        """Test second API call after tool execution still includes tools (for potential round 2)"""
        ai_gen = self.create_mock_ai_generator()

        tool_use_response = get_mock_tool_use_response()
        final_response = get_mock_final_response()

        captured_requests = []

        def capture_invoke(modelId, body):
            captured_requests.append(json.loads(body))
            if len(captured_requests) == 1:
                return get_mock_bedrock_response_bytes(tool_use_response)
            return get_mock_bedrock_response_bytes(final_response)

        ai_gen.client.invoke_model.side_effect = capture_invoke

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Result"

        tools = [{"name": "search_course_content"}]

        ai_gen.generate_response(query="Test", tools=tools, tool_manager=mock_tool_manager)

        # First call should have tools
        assert "tools" in captured_requests[0]

        # Second call SHOULD have tools (round 1, can still do round 2)
        assert "tools" in captured_requests[1]

    def test_single_round_still_works(self):
        """Test that single-round tool calling still works (backward compatibility)"""
        ai_gen = self.create_mock_ai_generator()

        # Mock two responses: tool_use then final
        tool_use_response = get_mock_tool_use_response()
        final_response = get_mock_final_response()

        ai_gen.client.invoke_model.side_effect = [
            get_mock_bedrock_response_bytes(tool_use_response),
            get_mock_bedrock_response_bytes(final_response),
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "ML search results"

        tools = [{"name": "search_course_content"}]

        result = ai_gen.generate_response(
            query="What is supervised learning?", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify single round: 2 API calls, 1 tool execution
        assert ai_gen.client.invoke_model.call_count == 2
        assert mock_tool_manager.execute_tool.call_count == 1
        assert "Supervised learning" in result

    def test_two_round_sequential_calling(self):
        """Test Claude can make 2 sequential tool calls"""
        ai_gen = self.create_mock_ai_generator()

        # Mock 3 responses: tool_use → tool_use → final
        round1_response = get_mock_tool_use_response()
        round2_response = get_mock_tool_use_response_round2()
        final_response = get_mock_final_response()

        ai_gen.client.invoke_model.side_effect = [
            get_mock_bedrock_response_bytes(round1_response),
            get_mock_bedrock_response_bytes(round2_response),
            get_mock_bedrock_response_bytes(final_response),
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "ML result about supervised learning",
            "DL result about neural networks",
        ]

        tools = [{"name": "search_course_content"}]

        result = ai_gen.generate_response(
            query="Compare ML and DL approaches", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify 2 rounds: 3 API calls, 2 tool executions
        assert ai_gen.client.invoke_model.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2
        assert result is not None

    def test_early_termination_after_round_1(self):
        """Test Claude can terminate after round 1 if satisfied"""
        ai_gen = self.create_mock_ai_generator()

        # Mock 2 responses: tool_use → final (no more tool_use)
        round1_response = get_mock_tool_use_response()
        final_response = get_mock_final_response()

        ai_gen.client.invoke_model.side_effect = [
            get_mock_bedrock_response_bytes(round1_response),
            get_mock_bedrock_response_bytes(final_response),
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Sufficient results"

        tools = [{"name": "search_course_content"}]

        result = ai_gen.generate_response(
            query="Simple question", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify early termination: only 2 API calls, 1 tool execution
        assert ai_gen.client.invoke_model.call_count == 2
        assert mock_tool_manager.execute_tool.call_count == 1
        assert "Supervised learning" in result

    def test_max_rounds_enforced(self):
        """Test that tool calls are ignored after 2 rounds"""
        ai_gen = self.create_mock_ai_generator()

        # Mock 3 responses all with tool_use (Claude doesn't stop)
        round1_response = get_mock_tool_use_response()
        round2_response = get_mock_tool_use_response_round2()
        round3_response = get_mock_tool_use_response_round3()  # Would be 3rd round

        ai_gen.client.invoke_model.side_effect = [
            get_mock_bedrock_response_bytes(round1_response),
            get_mock_bedrock_response_bytes(round2_response),
            get_mock_bedrock_response_bytes(round3_response),
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        tools = [{"name": "search_course_content"}]

        result = ai_gen.generate_response(
            query="Complex question", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify hard limit: 3 API calls, but only 2 tool executions
        assert ai_gen.client.invoke_model.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2
        # Should extract text from round3_response despite tool_use
        assert result is not None
        assert "would like to search again" in result.lower()

    def test_context_preserved_across_rounds(self):
        """Test that message history is preserved across rounds"""
        ai_gen = self.create_mock_ai_generator()

        round1_response = get_mock_tool_use_response()
        round2_response = get_mock_tool_use_response_round2()
        final_response = get_mock_final_response()

        captured_requests = []

        def capture_invoke(modelId, body):
            captured_requests.append(json.loads(body))
            idx = len(captured_requests) - 1
            if idx == 0:
                return get_mock_bedrock_response_bytes(round1_response)
            elif idx == 1:
                return get_mock_bedrock_response_bytes(round2_response)
            return get_mock_bedrock_response_bytes(final_response)

        ai_gen.client.invoke_model.side_effect = capture_invoke

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        tools = [{"name": "search_course_content"}]

        ai_gen.generate_response(query="Test", tools=tools, tool_manager=mock_tool_manager)

        # Verify message accumulation
        assert len(captured_requests) == 3

        # Initial call: 1 message (user query)
        assert len(captured_requests[0]["messages"]) == 1
        assert captured_requests[0]["messages"][0]["role"] == "user"

        # After round 1: 3 messages (user + assistant + tool_results)
        assert len(captured_requests[1]["messages"]) == 3
        assert captured_requests[1]["messages"][0]["role"] == "user"
        assert captured_requests[1]["messages"][1]["role"] == "assistant"
        assert captured_requests[1]["messages"][2]["role"] == "user"

        # After round 2: 5 messages
        assert len(captured_requests[2]["messages"]) == 5
        assert captured_requests[2]["messages"][3]["role"] == "assistant"
        assert captured_requests[2]["messages"][4]["role"] == "user"

    def test_tools_included_in_both_rounds(self):
        """Test tools are included in calls for rounds 1 and 2, but not final call"""
        ai_gen = self.create_mock_ai_generator()

        round1_response = get_mock_tool_use_response()
        round2_response = get_mock_tool_use_response_round2()
        round3_response = get_mock_tool_use_response_round3()

        captured_requests = []

        def capture_invoke(modelId, body):
            captured_requests.append(json.loads(body))
            idx = len(captured_requests) - 1
            if idx == 0:
                return get_mock_bedrock_response_bytes(round1_response)
            elif idx == 1:
                return get_mock_bedrock_response_bytes(round2_response)
            return get_mock_bedrock_response_bytes(round3_response)

        ai_gen.client.invoke_model.side_effect = capture_invoke

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        tools = [{"name": "search_course_content"}]

        ai_gen.generate_response(query="Test", tools=tools, tool_manager=mock_tool_manager)

        # Call 1 (initial): has tools
        assert "tools" in captured_requests[0]
        assert captured_requests[0]["tools"] == tools

        # Call 2 (after round 1): has tools
        assert "tools" in captured_requests[1]
        assert captured_requests[1]["tools"] == tools

        # Call 3 (after round 2, final): NO tools
        assert "tools" not in captured_requests[2]

    def test_tool_execution_error_handling(self):
        """Test graceful handling when tool execution fails"""
        ai_gen = self.create_mock_ai_generator()

        tool_use_response = get_mock_tool_use_response()
        final_response = get_mock_final_response()

        ai_gen.client.invoke_model.side_effect = [
            get_mock_bedrock_response_bytes(tool_use_response),
            get_mock_bedrock_response_bytes(final_response),
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")

        tools = [{"name": "search_course_content"}]

        # Should not raise exception
        result = ai_gen.generate_response(query="Test", tools=tools, tool_manager=mock_tool_manager)

        # Should get final response despite tool error
        assert result is not None
        assert ai_gen.client.invoke_model.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
