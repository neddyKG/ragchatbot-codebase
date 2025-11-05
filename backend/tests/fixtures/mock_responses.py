"""Mock Bedrock API responses for testing"""


def get_mock_tool_use_response():
    """Returns a mock Bedrock response where Claude decides to use a tool"""
    return {
        "stop_reason": "tool_use",
        "content": [
            {
                "type": "text",
                "text": "I need to search the course content to answer this question.",
            },
            {
                "type": "tool_use",
                "id": "toolu_123456",
                "name": "search_course_content",
                "input": {"query": "supervised learning", "course_name": "Machine Learning"},
            },
        ],
    }


def get_mock_tool_use_response_round2():
    """Returns a mock Bedrock response for second round tool use"""
    return {
        "stop_reason": "tool_use",
        "content": [
            {
                "type": "text",
                "text": "Based on the first search, I need to search another course for comparison.",
            },
            {
                "type": "tool_use",
                "id": "toolu_789012",
                "name": "search_course_content",
                "input": {"query": "neural networks", "course_name": "Deep Learning"},
            },
        ],
    }


def get_mock_tool_use_response_outline():
    """Returns a mock Bedrock response using get_course_outline tool"""
    return {
        "stop_reason": "tool_use",
        "content": [
            {"type": "text", "text": "I need to get the course outline first."},
            {
                "type": "tool_use",
                "id": "toolu_345678",
                "name": "get_course_outline",
                "input": {"course_title": "Machine Learning"},
            },
        ],
    }


def get_mock_tool_use_response_round3():
    """Returns a mock Bedrock response for what would be a third round (should not execute)"""
    return {
        "stop_reason": "tool_use",
        "content": [
            {"type": "text", "text": "I would like to search again but should be at max rounds."},
            {
                "type": "tool_use",
                "id": "toolu_999999",
                "name": "search_course_content",
                "input": {"query": "third search attempt", "course_name": "Test"},
            },
        ],
    }


def get_mock_final_response():
    """Returns a mock Bedrock response with the final answer after tool use"""
    return {
        "stop_reason": "end_turn",
        "content": [
            {
                "type": "text",
                "text": "Supervised learning is a type of machine learning where algorithms learn from labeled training data with input features and output labels.",
            }
        ],
    }


def get_mock_direct_response():
    """Returns a mock Bedrock response for questions that don't need tools"""
    return {
        "stop_reason": "end_turn",
        "content": [
            {
                "type": "text",
                "text": "This is a general knowledge answer that doesn't require searching course content.",
            }
        ],
    }


def get_mock_bedrock_response_bytes(response_dict):
    """Convert response dict to Bedrock-style response with body.read()"""
    import io
    import json

    class MockBody:
        def __init__(self, data):
            self.data = data

        def read(self):
            import json

            return json.dumps(self.data).encode("utf-8")

    return {"body": MockBody(response_dict)}
