# Development Guide

This guide covers the code quality tools and workflows for this project.

## Code Quality Tools

This project uses several code quality tools to maintain consistency and catch issues early:

- **Black**: Automatic code formatting (100 character line length)
- **isort**: Automatic import sorting
- **flake8**: Linting for code style issues
- **mypy**: Static type checking (non-blocking)
- **pytest**: Unit and integration testing

## Installation

All development dependencies are installed automatically with:

```bash
uv sync --extra dev
```

## Available Scripts

### `./format.sh`

Automatically formats all code with black and isort. Run this before committing:

```bash
./format.sh
```

This will:
- Sort imports with isort
- Format code with black (100 char lines)

### `./lint.sh`

Runs all linting checks without making changes:

```bash
./lint.sh
```

This will check:
- Code formatting (black --check)
- Import sorting (isort --check-only)
- Code style (flake8)
- Type hints (mypy, non-blocking)

### `./check-all.sh`

Comprehensive quality check that runs all checks and tests:

```bash
./check-all.sh
```

This will:
1. Check code formatting
2. Check import sorting
3. Run flake8 linter
4. Run type checker
5. Run all tests with pytest

Use this before creating pull requests to ensure everything passes.

## Configuration

### pyproject.toml

Main configuration for all tools:

- **Black**: 100 char lines, Python 3.13, excludes chroma_db
- **isort**: Black-compatible profile, 100 char lines
- **mypy**: Lenient settings, ignores missing imports for third-party libs

### .flake8

Flake8 configuration:

- Max line length: 100
- Ignores: E203, W503, E501, F401, F811, E402, F841
- Per-file ignores for tests and fixtures

## Recommended Workflow

1. **Make your changes**
   ```bash
   # Edit code...
   ```

2. **Format code**
   ```bash
   ./format.sh
   ```

3. **Run quality checks**
   ```bash
   ./lint.sh
   ```

4. **Run tests**
   ```bash
   cd backend && uv run pytest -v
   ```

5. **Or run everything at once**
   ```bash
   ./check-all.sh
   ```

## Pre-commit Workflow

Before committing, always run:

```bash
./format.sh && ./lint.sh
```

Or for a complete check:

```bash
./check-all.sh
```

## Individual Tool Usage

You can also run tools individually:

```bash
# Format code
uv run black backend/
uv run isort backend/

# Check formatting
uv run black backend/ --check
uv run isort backend/ --check-only

# Lint
uv run flake8 backend/

# Type check
uv run mypy backend/

# Run tests
cd backend && uv run pytest -v
```

## CI/CD Integration

Add these scripts to your CI pipeline:

```yaml
# Example GitHub Actions workflow
- name: Run quality checks
  run: ./check-all.sh
```

## Editor Integration

### VS Code

Install these extensions:
- Python (ms-python.python)
- Black Formatter (ms-python.black-formatter)
- isort (ms-python.isort)
- Flake8 (ms-python.flake8)

Add to `.vscode/settings.json`:

```json
{
  "python.formatting.provider": "black",
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "editor.formatOnSave": true,
  "[python]": {
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

### PyCharm

1. Enable Black: Settings → Tools → Black
2. Enable isort: Settings → Tools → External Tools → Add isort
3. Enable on save: Settings → Tools → Actions on Save

## Troubleshooting

### Flake8 errors

If flake8 reports errors that black doesn't fix, check `.flake8` configuration.

### Mypy errors

Mypy is configured to be non-blocking. Warnings are informational only and won't fail the build.

### Test failures

If tests fail, run them locally:

```bash
cd backend && uv run pytest -v
```

Check the test output for specific failures.

## Best Practices

1. **Always format before committing**: Run `./format.sh`
2. **Run checks locally**: Don't rely on CI to catch formatting issues
3. **Write tests**: Add tests for new features and bug fixes
4. **Type hints**: Add type hints where possible (not strictly enforced)
5. **Clean commits**: One logical change per commit

## Questions?

See `CLAUDE.md` for architecture details and project setup instructions.
