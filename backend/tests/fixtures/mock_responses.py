"""Mock Bedrock API responses for testing"""

def get_mock_tool_use_response():
    """Returns a mock Bedrock response where Claude decides to use a tool"""
    return {
        'stop_reason': 'tool_use',
        'content': [
            {
                'type': 'text',
                'text': 'I need to search the course content to answer this question.'
            },
            {
                'type': 'tool_use',
                'id': 'toolu_123456',
                'name': 'search_course_content',
                'input': {
                    'query': 'supervised learning',
                    'course_name': 'Machine Learning'
                }
            }
        ]
    }

def get_mock_final_response():
    """Returns a mock Bedrock response with the final answer after tool use"""
    return {
        'stop_reason': 'end_turn',
        'content': [
            {
                'type': 'text',
                'text': 'Supervised learning is a type of machine learning where algorithms learn from labeled training data with input features and output labels.'
            }
        ]
    }

def get_mock_direct_response():
    """Returns a mock Bedrock response for questions that don't need tools"""
    return {
        'stop_reason': 'end_turn',
        'content': [
            {
                'type': 'text',
                'text': 'This is a general knowledge answer that doesn\'t require searching course content.'
            }
        ]
    }

def get_mock_bedrock_response_bytes(response_dict):
    """Convert response dict to Bedrock-style response with body.read()"""
    import json
    import io

    class MockBody:
        def __init__(self, data):
            self.data = data

        def read(self):
            import json
            return json.dumps(self.data).encode('utf-8')

    return {
        'body': MockBody(response_dict)
    }
