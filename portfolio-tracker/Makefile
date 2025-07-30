# Portfolio Tracker Makefile
# Convenient commands for development, testing, and linting

.PHONY: help install test test-unit test-integration lint format clean run setup-dev

# Default target
help:  ## Show this help message
	@echo "Portfolio Tracker Development Commands"
	@echo "======================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

# Installation and setup
install:  ## Install project dependencies
	uv sync

setup-dev:  ## Set up development environment (install deps + pre-commit)
	uv sync
	uv run pre-commit install
	@echo "✅ Development environment set up!"
	@echo "📝 Don't forget to copy .env.example to .env and add your API keys"

# Testing
test:  ## Run all tests with coverage
	./scripts/test.sh

test-unit:  ## Run unit tests only
	./scripts/test.sh --unit

test-integration:  ## Run integration tests only
	./scripts/test.sh --integration

test-fast:  ## Run tests quickly (skip slow tests)
	./scripts/test.sh --fast

test-no-cov:  ## Run tests without coverage report
	./scripts/test.sh --no-cov

# Linting and formatting
lint:  ## Run all linting checks
	./scripts/lint.sh

format:  ## Format code with black and isort
	uv run black src/ tests/
	uv run isort src/ tests/
	@echo "✅ Code formatted!"

lint-fix:  ## Run linting and fix auto-fixable issues
	uv run black src/ tests/
	uv run isort src/ tests/
	uv run flake8 src/ tests/
	@echo "✅ Linting fixes applied!"

# Type checking
typecheck:  ## Run mypy type checking
	uv run mypy src/

# Pre-commit
pre-commit:  ## Run pre-commit on all files
	uv run pre-commit run --all-files

pre-commit-install:  ## Install pre-commit hooks
	uv run pre-commit install

# Application
run:  ## Run the main application
	python main.py

run-ui:  ## Run the Streamlit UI directly
	python run_app.py

run-sample:  ## Create a sample portfolio and run UI
	python main.py --mode sample
	python main.py

# Portfolio management
init:  ## Initialize the portfolio system
	python main.py --mode init

status:  ## Check system status
	python main.py --mode status

sample:  ## Create a sample portfolio
	python main.py --mode sample

# Development utilities
clean:  ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	@echo "✅ Cleaned up generated files!"

clean-data:  ## Clean up portfolio data (WARNING: deletes all portfolios!)
	@echo "⚠️  WARNING: This will delete all portfolio data!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/; \
		echo "🗑️  Portfolio data deleted!"; \
	else \
		echo "❌ Cancelled."; \
	fi

# Documentation
docs:  ## Generate documentation (placeholder)
	@echo "📚 Documentation generation not implemented yet"

# CI/CD helpers
ci-test:  ## Run tests for CI (no coverage HTML)
	uv run pytest tests/ --cov=src --cov-report=term --cov-report=xml

ci-lint:  ## Run linting for CI
	uv run black --check --diff src/ tests/
	uv run isort --check-only --diff src/ tests/
	uv run flake8 src/ tests/

# Development workflow
dev:  ## Complete development check (format, lint, test)
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test-fast
	@echo "🎉 Development checks completed!"

# Release helpers
version:  ## Show current version
	@grep "version" pyproject.toml | head -1 | cut -d'"' -f2

check:  ## Run all checks (lint + test)
	$(MAKE) lint
	$(MAKE) test

# Docker (placeholder for future)
docker-build:  ## Build Docker image (placeholder)
	@echo "🐳 Docker build not implemented yet"

docker-run:  ## Run Docker container (placeholder)
	@echo "🐳 Docker run not implemented yet"

# Backup and restore
backup:  ## Backup portfolio data
	@if [ -d "data" ]; then \
		backup_name="portfolio-backup-$$(date +%Y%m%d_%H%M%S).tar.gz"; \
		tar -czf "$$backup_name" data/; \
		echo "💾 Portfolio data backed up to $$backup_name"; \
	else \
		echo "❌ No data directory found"; \
	fi

# Show project info
info:  ## Show project information
	@echo "🤖 Portfolio Tracker"
	@echo "==================="
	@echo "Version: $$(grep 'version' pyproject.toml | head -1 | cut -d'\"' -f2)"
	@echo "Python: $$(python --version)"
	@echo "Dependencies: $$(uv pip list | wc -l) packages"
	@echo "Tests: $$(find tests -name '*.py' -not -name '__*' | wc -l) test files"
	@echo "Source files: $$(find src -name '*.py' | wc -l) Python files"
	@echo "Lines of code: $$(find src -name '*.py' -exec wc -l {} + | tail -1 | awk '{print $$1}')"
	@echo ""
	@echo "📁 Project structure:"
	@tree -I '__pycache__|*.pyc|.venv|htmlcov|.pytest_cache|.mypy_cache' -L 2 || ls -la