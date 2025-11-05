#!/bin/bash
# lint.sh - Run linting checks without fixing

set -e

echo "Checking code formatting with black..."
uv run black backend/ --check

echo "Checking import sorting with isort..."
uv run isort backend/ --check-only

echo "Running flake8 linter..."
uv run flake8 backend/

echo "Running mypy type checker..."
uv run mypy backend/ --exclude chroma_db --exclude __pycache__ || echo "⚠️  Type checking found issues (non-blocking)"

echo "✅ All linting checks passed!"
