"""
Tests for AIGenerator tool calling functionality

These tests verify that AIGenerator correctly:
1. Handles tool_use stop_reason and triggers tool execution
2. Constructs message history correctly for tool results
3. Makes second API call after tool execution
4. Extracts final response text
5. Handles errors gracefully
"""

import pytest
import json
import sys
import os
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator
from tests.fixtures.mock_responses import (
    get_mock_tool_use_response,
    get_mock_final_response,
    get_mock_direct_response,
    get_mock_bedrock_response_bytes
)


class TestAIGeneratorToolCalling:
    """Test AIGenerator's tool calling orchestration"""

    def create_mock_ai_generator(self):
        """Helper to create AIGenerator with mocked Bedrock client"""
        with patch('boto3.client'):
            ai_gen = AIGenerator(
                aws_access_key_id="test_key",
                aws_secret_access_key="test_secret",
                aws_session_token="test_token",
                aws_region="us-east-1",
                model_id="test-model"
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
        result = ai_gen.generate_response(
            query="What is 2+2?",
            tools=None,
            tool_manager=None
        )

        # Verify single API call
        assert ai_gen.client.invoke_model.call_count == 1
        assert result == "This is a general knowledge answer that doesn't require searching course content."

    def test_tool_use_triggers_execution(self):
        """Test that tool_use stop_reason triggers tool execution flow"""
        ai_gen = self.create_mock_ai_generator()

        # Mock two responses: tool_use then final
        tool_use_response = get_mock_tool_use_response()
        final_response = get_mock_final_response()

        ai_gen.client.invoke_model.side_effect = [
            get_mock_bedrock_response_bytes(tool_use_response),
            get_mock_bedrock_response_bytes(final_response)
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "[ML Course] Supervised learning uses labeled data."

        # Mock tools list
        tools = [{'name': 'search_course_content', 'description': 'Search courses'}]

        # Execute
        result = ai_gen.generate_response(
            query="What is supervised learning?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            'search_course_content',
            query='supervised learning',
            course_name='Machine Learning'
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

        tools = [{'name': 'search_course_content'}]

        ai_gen.generate_response(
            query="Test query",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Verify second request has correct message structure
        assert len(captured_requests) == 2
        second_request = captured_requests[1]

        # Should have 3 messages: user, assistant (with tool_use), user (with tool_result)
        assert len(second_request['messages']) == 3
        assert second_request['messages'][0]['role'] == 'user'
        assert second_request['messages'][1]['role'] == 'assistant'
        assert second_request['messages'][2]['role'] == 'user'

        # Check tool result structure
        tool_result_content = second_request['messages'][2]['content']
        assert isinstance(tool_result_content, list)
        assert tool_result_content[0]['type'] == 'tool_result'
        assert tool_result_content[0]['content'] == "Tool result content"

    def test_no_tool_execution_without_tool_manager(self):
        """Test that tool_use without tool_manager returns gracefully"""
        ai_gen = self.create_mock_ai_generator()

        tool_use_response = get_mock_tool_use_response()
        ai_gen.client.invoke_model.return_value = get_mock_bedrock_response_bytes(tool_use_response)

        # Call with tools but no tool_manager
        result = ai_gen.generate_response(
            query="Test",
            tools=[{'name': 'search_course_content'}],
            tool_manager=None
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
            conversation_history="User: Previous question\nAssistant: Previous answer"
        )

        # Verify history in system prompt
        assert len(captured_requests) == 1
        system_content = captured_requests[0]['system']
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
            {'name': 'search_course_content', 'description': 'Search'},
            {'name': 'get_course_outline', 'description': 'Get outline'}
        ]

        ai_gen.generate_response(
            query="Test",
            tools=tools
        )

        # Verify tools in request
        assert len(captured_requests) == 1
        request = captured_requests[0]
        assert 'tools' in request
        assert request['tools'] == tools
        assert request['tool_choice'] == {'type': 'auto'}

    def test_second_call_removes_tools(self):
        """Test second API call after tool execution doesn't include tools"""
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

        tools = [{'name': 'search_course_content'}]

        ai_gen.generate_response(
            query="Test",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # First call should have tools
        assert 'tools' in captured_requests[0]

        # Second call should NOT have tools
        assert 'tools' not in captured_requests[1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
