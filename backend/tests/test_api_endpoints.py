"""
API endpoint tests for the RAG system FastAPI application

Tests cover:
- POST /api/query - Query processing with and without session
- GET /api/courses - Course statistics retrieval
- DELETE /api/session/{session_id} - Session management
- Error handling and edge cases

Note: These tests use a test app that doesn't mount static files
to avoid dependency on frontend directory during testing.
"""

import pytest
import sys
import os
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mark all tests in this module as unit tests (don't require heavy fixtures)
pytestmark = pytest.mark.unit


def create_test_app():
    """
    Create a test version of the FastAPI app without static file mounting

    This avoids issues with missing frontend directory during tests
    while maintaining all API endpoint functionality.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional

    # Create test app
    app = FastAPI(title="Course Materials RAG System - Test")

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Pydantic models for request/response
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class Source(BaseModel):
        text: str
        link: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Source]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources"""
        try:
            rag_system = app.state.rag_system
            # Create session if not provided
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()

            # Process query using RAG system
            answer, sources = rag_system.query(request.query, session_id)

            # Convert sources to Source model objects
            source_objects = [Source(text=s["text"], link=s.get("link")) for s in sources]

            return QueryResponse(
                answer=answer,
                sources=source_objects,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics"""
        try:
            rag_system = app.state.rag_system
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def clear_session(session_id: str):
        """Clear a conversation session"""
        try:
            rag_system = app.state.rag_system
            rag_system.session_manager.clear_session(session_id)
            return {"status": "success", "message": f"Session {session_id} cleared"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        """Root endpoint for health check"""
        return {"status": "healthy", "service": "RAG System API"}

    return app


@pytest.fixture
def local_mock_rag_system():
    """Create a local mock RAG system for API testing (avoids conftest imports)"""
    mock_system = Mock()

    # Mock session manager
    mock_system.session_manager = Mock()
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
def test_app(local_mock_rag_system):
    """Create test FastAPI app with mocked RAG system"""
    app = create_test_app()
    # Inject mock RAG system into app.state
    app.state.rag_system = local_mock_rag_system
    return app


@pytest.fixture
def client(test_app):
    """Create test client with mocked dependencies"""
    return TestClient(test_app)


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


class TestQueryEndpoint:
    """Tests for POST /api/query endpoint"""

    def test_query_with_session_id(self, client, sample_query_request):
        """Test query with existing session ID"""
        response = client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) > 0

    def test_query_without_session_id(self, client, sample_query_request_no_session, local_mock_rag_system):
        """Test query creates new session when not provided"""
        response = client.post("/api/query", json=sample_query_request_no_session)

        assert response.status_code == 200
        data = response.json()

        assert "session_id" in data
        assert data["session_id"] == "test-session-id"
        # Verify create_session was called
        local_mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_response_format(self, client, sample_query_request):
        """Test query response has correct format"""
        response = client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0
        assert isinstance(data["sources"], list)

        # Check source structure
        if len(data["sources"]) > 0:
            source = data["sources"][0]
            assert "text" in source
            assert "link" in source

    def test_query_with_empty_query(self, client):
        """Test query with empty query string"""
        response = client.post("/api/query", json={"query": ""})

        # Should still process but may return minimal response
        assert response.status_code in [200, 422]  # 422 for validation error

    def test_query_with_missing_query_field(self, client):
        """Test query without required query field"""
        response = client.post("/api/query", json={"session_id": "test"})

        assert response.status_code == 422  # Validation error

    def test_query_calls_rag_system(self, client, sample_query_request, local_mock_rag_system):
        """Test that query endpoint calls RAG system correctly"""
        response = client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        # Verify RAG system query was called
        local_mock_rag_system.query.assert_called_once()
        call_args = local_mock_rag_system.query.call_args
        assert call_args[0][0] == "What is supervised learning?"
        assert call_args[0][1] == "test-session-123"

    def test_query_error_handling(self, sample_query_request):
        """Test query endpoint handles RAG system errors"""
        # Create a client with a failing RAG system
        app = create_test_app()
        mock_failing_system = Mock()
        mock_failing_system.query.side_effect = Exception("Database error")
        app.state.rag_system = mock_failing_system

        failing_client = TestClient(app)
        response = failing_client.post("/api/query", json=sample_query_request)

        assert response.status_code == 500
        assert "detail" in response.json()


class TestCoursesEndpoint:
    """Tests for GET /api/courses endpoint"""

    def test_get_courses_success(self, client):
        """Test successful retrieval of course statistics"""
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

    def test_get_courses_with_data(self, client):
        """Test courses endpoint returns expected data"""
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert data["total_courses"] == 2
        assert len(data["course_titles"]) == 2
        assert "Introduction to Machine Learning" in data["course_titles"]
        assert "Deep Learning Fundamentals" in data["course_titles"]

    def test_get_courses_empty_database(self):
        """Test courses endpoint with no courses loaded"""
        mock_empty_rag_system = Mock()
        mock_empty_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        app = create_test_app()
        app.state.rag_system = mock_empty_rag_system

        empty_client = TestClient(app)
        response = empty_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert data["total_courses"] == 0
        assert len(data["course_titles"]) == 0

    def test_get_courses_calls_analytics(self, client, local_mock_rag_system):
        """Test that courses endpoint calls get_course_analytics"""
        response = client.get("/api/courses")

        assert response.status_code == 200
        local_mock_rag_system.get_course_analytics.assert_called_once()

    def test_get_courses_error_handling(self):
        """Test courses endpoint handles errors gracefully"""
        app = create_test_app()
        mock_failing_system = Mock()
        mock_failing_system.get_course_analytics.side_effect = Exception("Vector store error")
        app.state.rag_system = mock_failing_system

        failing_client = TestClient(app)
        response = failing_client.get("/api/courses")

        assert response.status_code == 500


class TestSessionEndpoint:
    """Tests for DELETE /api/session/{session_id} endpoint"""

    def test_clear_session_success(self, client):
        """Test successful session clearing"""
        response = client.delete("/api/session/test-session-123")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert "test-session-123" in data["message"]

    def test_clear_session_calls_manager(self, client, local_mock_rag_system):
        """Test that clear session calls session manager"""
        response = client.delete("/api/session/test-session-456")

        assert response.status_code == 200
        local_mock_rag_system.session_manager.clear_session.assert_called_once_with("test-session-456")

    def test_clear_nonexistent_session(self, client):
        """Test clearing a session that doesn't exist"""
        response = client.delete("/api/session/nonexistent-session")

        # Should still succeed (idempotent operation)
        assert response.status_code == 200

    def test_clear_session_error_handling(self):
        """Test session endpoint handles errors"""
        app = create_test_app()
        mock_failing_system = Mock()
        mock_failing_system.session_manager.clear_session.side_effect = Exception("Session error")
        app.state.rag_system = mock_failing_system

        failing_client = TestClient(app)
        response = failing_client.delete("/api/session/test-session")

        assert response.status_code == 500


class TestRootEndpoint:
    """Tests for GET / root endpoint"""

    def test_root_endpoint(self, client):
        """Test root endpoint returns health status"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert data["status"] == "healthy"


class TestIntegrationFlow:
    """Integration tests for complete API workflows"""

    def test_complete_query_flow(self, client, local_mock_rag_system):
        """Test complete flow: query -> get courses"""
        # Step 1: Submit query without session
        query_response = client.post("/api/query", json={
            "query": "What is machine learning?"
        })
        assert query_response.status_code == 200
        session_id = query_response.json()["session_id"]

        # Step 2: Get course list
        courses_response = client.get("/api/courses")
        assert courses_response.status_code == 200
        assert courses_response.json()["total_courses"] == 2

        # Step 3: Query with same session
        query2_response = client.post("/api/query", json={
            "query": "Tell me more about neural networks",
            "session_id": session_id
        })
        assert query2_response.status_code == 200

        # Step 4: Clear session
        clear_response = client.delete(f"/api/session/{session_id}")
        assert clear_response.status_code == 200

    def test_multiple_concurrent_sessions(self, client):
        """Test handling multiple sessions simultaneously"""
        # Create multiple sessions
        session_ids = []
        for i in range(3):
            response = client.post("/api/query", json={
                "query": f"Question {i}"
            })
            assert response.status_code == 200
            session_ids.append(response.json()["session_id"])

        # All should have same session ID (from mock)
        # In real system, these would be different
        assert all(sid == "test-session-id" for sid in session_ids)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
