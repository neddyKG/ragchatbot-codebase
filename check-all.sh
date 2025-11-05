#!/bin/bash
# check-all.sh - Run all quality checks and tests

set -e

echo "================================================"
echo "Running Code Quality Checks"
echo "================================================"

echo ""
echo "1. Checking code formatting..."
echo "------------------------------------------------"
uv run black backend/ --check || { echo "❌ Black formatting check failed. Run ./format.sh to fix."; exit 1; }

echo ""
echo "2. Checking import sorting..."
echo "------------------------------------------------"
uv run isort backend/ --check-only || { echo "❌ Import sorting check failed. Run ./format.sh to fix."; exit 1; }

echo ""
echo "3. Running flake8 linter..."
echo "------------------------------------------------"
uv run flake8 backend/

echo ""
echo "4. Running type checker..."
echo "------------------------------------------------"
uv run mypy backend/ --exclude chroma_db --exclude __pycache__ || echo "⚠️  Type checking found issues (non-blocking)"

echo ""
echo "5. Running tests..."
echo "------------------------------------------------"
cd backend && uv run pytest -v || { echo "❌ Tests failed"; exit 1; }

echo ""
echo "================================================"
echo "✅ All quality checks and tests passed!"
echo "================================================"
