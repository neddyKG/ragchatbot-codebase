# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Prerequisites

- Python 3.13 or higher
- uv (Python package manager)
- Anthropic API key

## Running the Application

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Create .env file with your API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# Run the application
./run.sh

# Or manually
cd backend && uv run uvicorn app:app --reload --port 8000

# Run any Python file with uv (ensures correct environment)
uv run python path/to/file.py
```

Application URLs:
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Architecture Overview

### RAG Pipeline Flow

This is a **tool-based RAG system** where Claude autonomously decides when to search the knowledge base:

1. User query → FastAPI (`app.py`)
2. `RAGSystem` orchestrator routes to `AIGenerator`
3. Claude receives query with `search_course_content` tool definition
4. Claude decides whether to invoke tool (general knowledge vs. course-specific)
5. If invoked: `ToolManager` → `CourseSearchTool` → `VectorStore` → ChromaDB
6. Search results → Claude synthesizes answer
7. Response + sources returned to user

### Vector Store Architecture (`vector_store.py`)

ChromaDB with **two separate collections**:

1. **`course_catalog`**: Stores course metadata for semantic course name matching
   - IDs: course title (unique identifier)
   - Metadata: title, instructor, course_link, lessons_json, lesson_count

2. **`course_content`**: Stores chunked course content
   - IDs: `{course_title}_{chunk_index}`
   - Metadata: course_title, lesson_number, chunk_index

The `search()` method performs two-step resolution:
1. If course_name provided → semantic search in catalog to resolve fuzzy name
2. Query content collection with filters (course_title and/or lesson_number)

Embeddings: `all-MiniLM-L6-v2` via sentence-transformers

### Document Processing (`document_processor.py`)

Expected document format:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: [title]
Lesson Link: [url]
[content]
```

Chunking strategy:
- Sentence-based splitting (handles abbreviations)
- 800 character chunks with 100 character overlap
- First chunk of each lesson prefixed with lesson context
- Last lesson chunks prefixed with both course and lesson context

### Tool-Based Search Pattern (`search_tools.py`)

`CourseSearchTool` implements the tool interface for Anthropic's tool calling:
- Accepts: `query` (required), `course_name` (optional), `lesson_number` (optional)
- Returns: Formatted results with course/lesson context headers
- Tracks `last_sources` for UI display

`ToolManager` registers tools and routes execution by name.

### AI Generation (`ai_generator.py`)

System prompt strategy:
- Instructs Claude to use search **only** for course-specific questions
- Maximum one search per query
- Answer general questions from existing knowledge
- No meta-commentary in responses

Two-turn flow for tool calls:
1. Initial API call with tools → Claude returns tool_use
2. Execute tool, add result to messages → Second API call without tools

### Session Management (`session_manager.py`)

Maintains conversation history per session:
- Max 2 exchanges (4 messages total) by default
- Formatted as "User: [message]\nAssistant: [message]"
- Passed as context in system prompt

### Configuration (`config.py`)

Key settings:
- Model: `claude-sonnet-4-20250514`
- Embeddings: `all-MiniLM-L6-v2`
- Chunk size: 800, overlap: 100
- Max search results: 5
- Max conversation history: 2 exchanges
- ChromaDB path: `./chroma_db`

### Startup Document Loading

On FastAPI startup, `app.py:startup_event`:
- Loads all `.pdf`, `.docx`, `.txt` files from `../docs` directory
- Checks existing course titles to avoid re-processing
- Processes each document → chunks → adds to both collections
- Continues on error (doesn't fail startup)

### API Endpoints

`POST /api/query`:
- Request: `{query: str, session_id?: str}`
- Response: `{answer: str, sources: List[str], session_id: str}`
- Creates session if not provided

`GET /api/courses`:
- Response: `{total_courses: int, course_titles: List[str]}`

## Important Implementation Details

**Course Title as Unique ID**: Course title serves as the unique identifier throughout the system (ChromaDB IDs, filters, deduplication).

**Dual Collection Strategy**: Separating catalog from content enables fuzzy course name matching while keeping content search efficient with metadata filters.

**Tool-Based vs Traditional RAG**: This architecture gives Claude autonomy to decide when retrieval is needed, enabling it to handle both general and domain-specific queries naturally.

**Lesson Context Injection**: Chunks are prefixed with lesson/course context to improve search relevance and Claude's ability to attribute information correctly.

**Source Tracking**: `CourseSearchTool.last_sources` populated during search, retrieved by `ToolManager`, returned in API response, then reset for next query.
- always use uv to run the server do not use pip directly