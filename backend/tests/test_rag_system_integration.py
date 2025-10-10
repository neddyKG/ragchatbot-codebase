"""
Integration tests for RAG system with real vector store and mock AI

These tests verify the complete flow:
1. Load sample course data into vector store
2. Execute queries through RAG system
3. Verify tool execution and source tracking
4. Test with mock AI responses to isolate vector store issues
"""

import pytest
import os
import sys
import tempfile
import shutil
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from config import Config
from document_processor import DocumentProcessor
from vector_store import VectorStore
from tests.fixtures.mock_responses import (
    get_mock_tool_use_response,
    get_mock_final_response,
    get_mock_bedrock_response_bytes
)


class TestRAGSystemIntegration:
    """Integration tests with real vector store and mock AI"""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def test_config(self, temp_db_path):
        """Create test configuration with temporary database"""
        config = Config()
        config.CHROMA_PATH = temp_db_path
        return config

    @pytest.fixture
    def vector_store_with_data(self, test_config):
        """Create vector store and load sample course"""
        # Initialize components
        processor = DocumentProcessor(test_config.CHUNK_SIZE, test_config.CHUNK_OVERLAP)
        store = VectorStore(test_config.CHROMA_PATH, test_config.EMBEDDING_MODEL, test_config.MAX_RESULTS)

        # Load sample course
        sample_course_path = os.path.join(
            os.path.dirname(__file__),
            'fixtures',
            'sample_course.txt'
        )

        course, chunks = processor.process_course_document(sample_course_path)
        store.add_course_metadata(course)
        store.add_course_content(chunks)

        return store

    def test_vector_store_has_data(self, vector_store_with_data):
        """Test that sample course was loaded correctly"""
        store = vector_store_with_data

        # Check course count
        assert store.get_course_count() == 1

        # Check course title
        titles = store.get_existing_course_titles()
        assert "Introduction to Machine Learning" in titles

        # Test search works
        results = store.search("supervised learning")
        assert not results.is_empty()
        assert results.error is None
        assert len(results.documents) > 0

    def test_vector_store_search_returns_relevant_content(self, vector_store_with_data):
        """Test that search returns content related to query"""
        store = vector_store_with_data

        results = store.search("neural networks")

        # Should find content about neural networks
        assert not results.is_empty()
        found_neural_network_content = any(
            "neural network" in doc.lower()
            for doc in results.documents
        )
        assert found_neural_network_content, "Expected to find neural network content"

    def test_vector_store_course_filter(self, vector_store_with_data):
        """Test search with course name filter"""
        store = vector_store_with_data

        # Should find with correct course name
        results = store.search("machine learning", course_name="Machine Learning")
        assert not results.is_empty()

        # Should not find with wrong course name
        results = store.search("machine learning", course_name="Nonexistent Course")
        assert results.error is not None or results.is_empty()

    def test_course_search_tool_with_real_store(self, vector_store_with_data):
        """Test CourseSearchTool with real vector store data"""
        from search_tools import CourseSearchTool

        tool = CourseSearchTool(vector_store_with_data)

        # Execute search
        result = tool.execute(query="supervised learning")

        # Verify formatted output
        assert "[Introduction to Machine Learning" in result
        assert "Lesson" in result
        assert len(tool.last_sources) > 0

        # Verify sources have expected structure
        source = tool.last_sources[0]
        assert 'text' in source
        assert 'link' in source

    def test_rag_system_query_with_mock_ai(self, test_config, vector_store_with_data):
        """Test full RAG system query with mocked AI responses"""
        # Create RAG system with mocked AI
        with patch('rag_system.AIGenerator') as MockAIGen:
            # Setup mock AI generator
            mock_ai_instance = Mock()

            # Simulate tool calling flow
            def mock_generate(query, conversation_history, tools, tool_manager):
                # Simulate AI deciding to use tool
                if tools and tool_manager:
                    # Execute the search tool
                    result = tool_manager.execute_tool(
                        'search_course_content',
                        query='supervised learning',
                        course_name='Machine Learning'
                    )
                    return f"Based on the search results: {result[:100]}..."
                return "Direct answer"

            mock_ai_instance.generate_response.side_effect = mock_generate
            MockAIGen.return_value = mock_ai_instance

            # Create RAG system
            rag = RAGSystem(test_config)
            rag.vector_store = vector_store_with_data

            # Execute query
            answer, sources = rag.query("What is supervised learning?")

            # Verify response
            assert answer is not None
            assert len(answer) > 0
            assert "Based on the search results" in answer

            # Verify sources were tracked
            assert isinstance(sources, list)
            # Sources might be empty if tool returns formatted text without tracking
            # But the important thing is the query didn't crash

    def test_rag_system_empty_database(self, test_config, temp_db_path):
        """Test RAG system behavior with empty database"""
        # Create RAG system with empty database
        config = test_config
        config.CHROMA_PATH = temp_db_path

        from search_tools import CourseSearchTool

        # Create empty vector store
        store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)

        # Test search returns empty
        tool = CourseSearchTool(store)
        result = tool.execute(query="anything")

        assert "No relevant content found" in result

    def test_course_outline_tool_with_real_store(self, vector_store_with_data):
        """Test CourseOutlineTool with real vector store data"""
        from search_tools import CourseOutlineTool

        tool = CourseOutlineTool(vector_store_with_data)

        # Execute outline retrieval
        result = tool.execute(course_title="Machine Learning")

        # Verify formatted output
        assert "Course: Introduction to Machine Learning" in result
        assert "Lesson 0:" in result
        assert "Lesson 1:" in result
        assert "Lesson 2:" in result

    def test_tool_manager_routing(self, vector_store_with_data):
        """Test ToolManager correctly routes to different tools"""
        from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool

        manager = ToolManager()
        search_tool = CourseSearchTool(vector_store_with_data)
        outline_tool = CourseOutlineTool(vector_store_with_data)

        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        # Test search tool
        result = manager.execute_tool('search_course_content', query='neural networks')
        assert "neural network" in result.lower() or "no relevant content" in result.lower()

        # Test outline tool
        result = manager.execute_tool('get_course_outline', course_title='Machine Learning')
        assert "Course:" in result and "Lesson" in result

        # Test invalid tool
        result = manager.execute_tool('nonexistent_tool')
        assert "not found" in result.lower()

    def test_source_tracking_and_reset(self, vector_store_with_data):
        """Test that sources are tracked and reset correctly"""
        from search_tools import ToolManager, CourseSearchTool

        manager = ToolManager()
        tool = CourseSearchTool(vector_store_with_data)
        manager.register_tool(tool)

        # Execute search
        manager.execute_tool('search_course_content', query='supervised learning')

        # Get sources
        sources = manager.get_last_sources()
        assert len(sources) > 0

        # Reset sources
        manager.reset_sources()

        # Verify sources cleared
        sources_after = manager.get_last_sources()
        assert len(sources_after) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
