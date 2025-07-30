# Testing and Linting Setup

This document describes the comprehensive testing and linting infrastructure that has been added to the Portfolio Tracker project.

## Overview

The project now includes:
- **Unit Testing** with pytest and comprehensive test coverage
- **Code Formatting** with black and isort
- **Linting** with flake8 and mypy
- **Pre-commit Hooks** for automated code quality
- **Coverage Reporting** with pytest-cov
- **Convenient Scripts** and Makefile for easy execution

## Quick Start

### Setup Development Environment
```bash
# Install all dependencies including dev tools
make setup-dev

# Or manually:
uv sync
uv run pre-commit install
```

### Run Tests
```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run tests without coverage
make test-no-cov

# Run tests quickly (skip slow tests)
make test-fast
```

### Run Linting
```bash
# Run all linting checks
make lint

# Format code automatically
make format

# Run type checking
make typecheck
```

### Development Workflow
```bash
# Complete development check (format, lint, test)
make dev
```

## Project Structure

```
portfolio-tracker/
├── tests/                          # Test directory
│   ├── conftest.py                 # Pytest configuration and fixtures
│   ├── unit/                       # Unit tests
│   │   ├── test_portfolio_models.py
│   │   ├── test_portfolio_storage.py
│   │   └── test_financial_metrics.py
│   ├── integration/                # Integration tests (placeholder)
│   └── fixtures/                   # Test data fixtures
├── scripts/                        # Development scripts
│   ├── lint.sh                     # Linting script
│   └── test.sh                     # Testing script
├── pyproject.toml                  # Project configuration
├── .flake8                         # Flake8 configuration
├── .pre-commit-config.yaml         # Pre-commit hooks configuration
└── Makefile                        # Development commands
```

## Testing Framework

### Pytest Configuration

**File**: `pyproject.toml`

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=../src",
    "--cov-report=term-missing",
    "--cov-report=html:htmlcov",
    "--cov-report=xml",
    "--cov-fail-under=80",
    "-v"
]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "external: marks tests that require external services",
]
```

### Test Fixtures

**File**: `tests/conftest.py`

Provides shared fixtures for:
- Sample financial instruments (stocks, crypto)
- Sample portfolios with transactions
- Test data directories
- Mock data managers
- Sample price and returns data

### Test Categories

#### Unit Tests (`tests/unit/`)
- **Models**: Test Pydantic models, validation, calculations
- **Storage**: Test file-based storage, serialization, data persistence
- **Metrics**: Test financial calculations, risk metrics, performance analysis

#### Integration Tests (`tests/integration/`)
- End-to-end workflows
- API integrations
- Full system tests

### Running Tests

```bash
# All tests with coverage
uv run pytest

# Unit tests only
uv run pytest tests/unit/ -m unit

# Integration tests only
uv run pytest tests/integration/ -m integration

# Skip slow tests
uv run pytest -m "not slow"

# Specific test file
uv run pytest tests/unit/test_portfolio_models.py

# Specific test
uv run pytest tests/unit/test_portfolio_models.py::TestPortfolio::test_create_empty_portfolio

# With coverage report
uv run pytest --cov=../src --cov-report=html
```

## Linting and Formatting

### Black - Code Formatting

**Configuration**: `pyproject.toml`

```toml
[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
exclude = '''
/(
    \.eggs
  | \.git
  | \.mypy_cache
  | \.venv
  | build
  | dist
  | data
)/
'''
```

**Usage**:
```bash
# Check formatting
uv run black --check src/ tests/

# Format files
uv run black src/ tests/
```

### isort - Import Sorting

**Configuration**: `pyproject.toml`

```toml
[tool.isort]
profile = "black"
multi_line_output = 3
line-length = 88
known_first_party = ["src"]
sections = ["FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"]
```

**Usage**:
```bash
# Check import sorting
uv run isort --check-only --diff src/ tests/

# Sort imports
uv run isort src/ tests/
```

### Flake8 - Style and Error Checking

**Configuration**: `.flake8`

```ini
[flake8]
max-line-length = 88
max-complexity = 10
ignore = 
    E203,  # whitespace before ':'
    E501,  # line too long (handled by black)
    W503,  # line break before binary operator
select = E,W,F,C
```

**Usage**:
```bash
# Run flake8
uv run flake8 src/ tests/
```

### MyPy - Type Checking

**Configuration**: `pyproject.toml`

```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
check_untyped_defs = true
strict_equality = true
```

**Usage**:
```bash
# Type check
uv run mypy src/
```

## Pre-commit Hooks

**File**: `.pre-commit-config.yaml`

Automatically runs on every commit:
- Black formatting
- isort import sorting
- flake8 linting
- mypy type checking
- General file checks (YAML, JSON, trailing whitespace)
- Security checks (detect secrets)

**Setup**:
```bash
# Install hooks
uv run pre-commit install

# Run on all files
uv run pre-commit run --all-files

# Run specific hook
uv run pre-commit run black
```

## Coverage Reporting

### Configuration

**File**: `pyproject.toml`

```toml
[tool.coverage.run]
source = ["../src"]
omit = [
    "*/tests/*",
    "*/test_*",
    "*/__pycache__/*",
    "src/ui/*",  # UI testing is complex
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
```

### Coverage Reports

- **Terminal**: Shows missing line numbers
- **HTML**: Detailed report in `htmlcov/index.html`
- **XML**: For CI/CD integration

### Target Coverage

- **Minimum**: 80% overall coverage
- **Current**: Varies by module (portfolio models: ~93%)

## Development Scripts

### `scripts/test.sh`

Feature-rich test runner with options:
- `--fast`: Skip slow tests
- `--unit`: Unit tests only
- `--integration`: Integration tests only
- `--no-cov`: Skip coverage
- `--help`: Show help

### `scripts/lint.sh`

Comprehensive linting script that runs:
1. Black formatting check
2. isort import check
3. flake8 style check
4. mypy type checking

Provides colored output and clear error messages.

## Makefile Commands

Convenient shortcuts for common tasks:

```bash
make help           # Show all available commands
make setup-dev      # Set up development environment
make test           # Run all tests
make test-unit      # Run unit tests only
make lint           # Run all linting
make format         # Format code
make clean          # Clean generated files
make dev            # Complete dev workflow
make info           # Show project info
```

## CI/CD Integration

### GitHub Actions (Placeholder)

The configuration supports CI/CD with:
- `make ci-test`: Tests for CI (XML coverage)
- `make ci-lint`: Linting for CI
- Pre-commit hooks can run in CI

### Coverage Integration

- Generates XML coverage reports
- Compatible with coverage services
- Fail-under threshold configurable

## Best Practices

### Writing Tests

1. **Use fixtures**: Leverage `conftest.py` fixtures
2. **Test edge cases**: Empty data, errors, boundary conditions
3. **Descriptive names**: `test_calculate_returns_with_insufficient_data`
4. **Arrange-Act-Assert**: Clear test structure
5. **Mock external dependencies**: Use `pytest-mock`

### Code Quality

1. **Run before commit**: Use pre-commit hooks
2. **Fix formatting**: Let black handle style
3. **Type annotations**: Add types for mypy
4. **Documentation**: Docstrings for complex functions
5. **Coverage**: Aim for 80%+ coverage

### Development Workflow

1. **Make changes**: Write code
2. **Run tests**: `make test-fast`
3. **Format code**: `make format`
4. **Check linting**: `make lint`
5. **Commit**: Pre-commit hooks run automatically

## Troubleshooting

### Common Issues

1. **Import errors**: Check PYTHONPATH in tests
2. **Coverage too low**: Add more unit tests
3. **Mypy errors**: Add type annotations
4. **Pre-commit fails**: Run `make format` first

### Debug Mode

```bash
# Verbose test output
uv run pytest -v -s

# Debug specific test
uv run pytest --pdb tests/unit/test_portfolio_models.py::TestPortfolio::test_create_empty_portfolio

# Skip coverage for faster runs
uv run pytest --no-cov
```

## Future Enhancements

### Planned Additions

1. **Integration tests**: Full API testing
2. **Performance tests**: Load testing with pytest-benchmark
3. **Mutation testing**: Test quality with mutmut
4. **Documentation tests**: Doctest integration
5. **Property-based testing**: Hypothesis integration

### Metrics Tracking

1. **Code complexity**: Track cyclomatic complexity
2. **Test performance**: Monitor test execution time
3. **Coverage trends**: Historical coverage tracking
4. **Quality gates**: Automated quality thresholds

## Configuration Files Summary

| File | Purpose |
|------|---------|
| `pyproject.toml` | Main project configuration (pytest, black, isort, mypy, coverage) |
| `.flake8` | Flake8 linting configuration |
| `.pre-commit-config.yaml` | Pre-commit hooks setup |
| `tests/conftest.py` | Pytest fixtures and configuration |
| `scripts/test.sh` | Test execution script |
| `scripts/lint.sh` | Linting script |
| `Makefile` | Development commands |

This comprehensive testing and linting setup ensures code quality, reliability, and maintainability for the Portfolio Tracker project.