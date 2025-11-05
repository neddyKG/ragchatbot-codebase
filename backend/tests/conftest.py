"""
Shared pytest fixtures for RAG system tests

This module provides reusable fixtures for:
- Test configuration setup
- Temporary database directories
- Mock RAG system components
- FastAPI test client
- Sample test data
"""

import pytest
import os
import sys
import tempfile
import shutil
from typing import Generator
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from vector_store import VectorStore
from document_processor import DocumentProcessor
from rag_system import RAGSystem


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Create a temporary database directory that's cleaned up after test"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config(temp_db_path: str) -> Config:
    """Create test configuration with temporary database"""
    config = Config()
    config.CHROMA_PATH = temp_db_path
    return config


@pytest.fixture
def empty_vector_store(test_config: Config) -> VectorStore:
    """Create an empty vector store for testing"""
    return VectorStore(
        test_config.CHROMA_PATH,
        test_config.EMBEDDING_MODEL,
        test_config.MAX_RESULTS
    )


@pytest.fixture
def document_processor(test_config: Config) -> DocumentProcessor:
    """Create a document processor with test configuration"""
    return DocumentProcessor(
        test_config.CHUNK_SIZE,
        test_config.CHUNK_OVERLAP
    )


@pytest.fixture
def sample_course_path() -> str:
    """Return path to sample course fixture"""
    return os.path.join(
        os.path.dirname(__file__),
        'fixtures',
        'sample_course.txt'
    )


@pytest.fixture
def vector_store_with_data(test_config: Config, sample_course_path: str) -> VectorStore:
    """Create vector store pre-loaded with sample course data"""
    # Initialize components
    processor = DocumentProcessor(test_config.CHUNK_SIZE, test_config.CHUNK_OVERLAP)
    store = VectorStore(test_config.CHROMA_PATH, test_config.EMBEDDING_MODEL, test_config.MAX_RESULTS)

    # Load sample course
    course, chunks = processor.process_course_document(sample_course_path)
    store.add_course_metadata(course)
    store.add_course_content(chunks)

    return store


@pytest.fixture
def mock_rag_system(test_config: Config) -> Mock:
    """Create a mock RAG system for API testing"""
    mock_system = Mock(spec=RAGSystem)

    # Mock session manager
    mock_system.session_manager.create_session.return_value = "test-session-id"
    mock_system.session_manager.clear_session.return_value = None

    # Mock query method - returns answer and sources
    mock_system.query.return_value = (
        "This is a test answer about machine learning.",
        [
            {
                "text": "Introduction to Machine Learning - Lesson 0: Overview",
                "link": "https://example.com/course/lesson0"
            },
            {
                "text": "Introduction to Machine Learning - Lesson 1: Supervised Learning",
                "link": "https://example.com/course/lesson1"
            }
        ]
    )

    # Mock course analytics
    mock_system.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": [
            "Introduction to Machine Learning",
            "Deep Learning Fundamentals"
        ]
    }

    return mock_system


@pytest.fixture
def mock_empty_rag_system(test_config: Config) -> Mock:
    """Create a mock RAG system with no courses loaded"""
    mock_system = Mock(spec=RAGSystem)

    mock_system.session_manager.create_session.return_value = "test-session-id"
    mock_system.query.return_value = ("No relevant content found.", [])
    mock_system.get_course_analytics.return_value = {
        "total_courses": 0,
        "course_titles": []
    }

    return mock_system


@pytest.fixture
def sample_query_request() -> dict:
    """Sample query request payload for API testing"""
    return {
        "query": "What is supervised learning?",
        "session_id": "test-session-123"
    }


@pytest.fixture
def sample_query_request_no_session() -> dict:
    """Sample query request without session ID"""
    return {
        "query": "Explain neural networks"
    }


@pytest.fixture
def mock_ai_generator():
    """Create a mock AI generator for testing"""
    mock = Mock()
    mock.generate_response.return_value = "This is a test response from the AI."
    return mock


@pytest.fixture
def mock_tool_manager():
    """Create a mock tool manager for testing"""
    mock = Mock()
    mock.execute_tool.return_value = "Search results: Neural networks are..."
    mock.get_last_sources.return_value = [
        {
            "text": "Introduction to Machine Learning - Lesson 2",
            "link": "https://example.com/course/lesson2"
        }
    ]
    mock.reset_sources.return_value = None
    return mock


# Pytest markers for categorizing tests
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
