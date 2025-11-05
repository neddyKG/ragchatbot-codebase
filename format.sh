#!/bin/bash
# format.sh - Automatically format code with black and isort

set -e

echo "Formatting imports with isort..."
uv run isort backend/

echo "Formatting code with black..."
uv run black backend/

echo "✨ Code formatting complete!"
