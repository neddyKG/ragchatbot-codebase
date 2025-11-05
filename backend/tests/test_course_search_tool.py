"""
Tests for CourseSearchTool.execute() method

These tests verify that the CourseSearchTool correctly:
1. Executes searches through the vector store
2. Formats results with proper headers and context
3. Tracks sources with lesson links
4. Handles edge cases (empty results, errors, filters)
"""

import os
import sys
from unittest.mock import MagicMock, Mock

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Test CourseSearchTool.execute() with various scenarios"""

    def test_execute_with_valid_results(self):
        """Test execute returns formatted results when search succeeds"""
        # Setup mock vector store
        mock_store = Mock()
        mock_results = SearchResults(
            documents=[
                "Machine learning is a subset of AI that enables systems to learn.",
                "Supervised learning uses labeled training data.",
            ],
            metadata=[
                {"course_title": "Introduction to Machine Learning", "lesson_number": 0},
                {"course_title": "Introduction to Machine Learning", "lesson_number": 1},
            ],
            distances=[0.1, 0.2],
            error=None,
        )
        mock_store.search.return_value = mock_results
        mock_store.get_lesson_link.return_value = "https://example.com/ml-course/lesson-0"

        # Create tool and execute
        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="machine learning", course_name="ML")

        # Verify search was called correctly
        mock_store.search.assert_called_once_with(
            query="machine learning", course_name="ML", lesson_number=None
        )

        # Verify result format
        assert "[Introduction to Machine Learning - Lesson 0]" in result
        assert "[Introduction to Machine Learning - Lesson 1]" in result
        assert "Machine learning is a subset of AI" in result
        assert "Supervised learning uses labeled training data" in result

    def test_execute_tracks_sources(self):
        """Test execute tracks sources with links for UI"""
        mock_store = Mock()
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )
        mock_store.search.return_value = mock_results
        mock_store.get_lesson_link.return_value = "https://example.com/test/lesson-1"

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="test")

        # Verify sources were tracked
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Test Course - Lesson 1"
        assert tool.last_sources[0]["link"] == "https://example.com/test/lesson-1"

    def test_execute_with_empty_results(self):
        """Test execute returns appropriate message when no results found"""
        mock_store = Mock()
        mock_results = SearchResults(documents=[], metadata=[], distances=[], error=None)
        mock_store.search.return_value = mock_results

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_execute_with_course_filter(self):
        """Test execute includes course name in 'not found' message"""
        mock_store = Mock()
        mock_results = SearchResults(documents=[], metadata=[], distances=[], error=None)
        mock_store.search.return_value = mock_results

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="test", course_name="Nonexistent Course")

        assert "No relevant content found in course 'Nonexistent Course'" in result

    def test_execute_with_lesson_filter(self):
        """Test execute includes lesson number in 'not found' message"""
        mock_store = Mock()
        mock_results = SearchResults(documents=[], metadata=[], distances=[], error=None)
        mock_store.search.return_value = mock_results

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="test", lesson_number=5)

        assert "No relevant content found in lesson 5" in result

    def test_execute_with_search_error(self):
        """Test execute returns error message when search fails"""
        mock_store = Mock()
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error="Database connection failed"
        )
        mock_store.search.return_value = mock_results

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="test")

        assert "Database connection failed" in result

    def test_execute_without_lesson_numbers(self):
        """Test execute handles results without lesson numbers"""
        mock_store = Mock()
        mock_results = SearchResults(
            documents=["General course information"],
            metadata=[{"course_title": "Test Course", "lesson_number": None}],
            distances=[0.1],
            error=None,
        )
        mock_store.search.return_value = mock_results

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="test")

        # Should only show course title, not lesson number
        assert "[Test Course]" in result
        assert "Lesson" not in result or "General course information" in result

    def test_execute_with_lesson_link_none(self):
        """Test execute handles missing lesson links gracefully"""
        mock_store = Mock()
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )
        mock_store.search.return_value = mock_results
        mock_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="test")

        # Verify source tracked with None link
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["link"] is None

    def test_get_tool_definition(self):
        """Test tool definition is correctly formatted for Anthropic API"""
        mock_store = Mock()
        tool = CourseSearchTool(mock_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
