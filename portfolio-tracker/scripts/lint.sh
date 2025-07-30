#!/bin/bash
# Lint script for Portfolio Tracker

set -e  # Exit on any error

echo "🧹 Running Portfolio Tracker linting..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}📂 Current directory: $(pwd)${NC}"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}❌ pyproject.toml not found. Please run from project root.${NC}"
    exit 1
fi

# Run black (code formatting)
echo -e "${YELLOW}🔧 Running black (code formatting)...${NC}"
if uv run black --check --diff src/ tests/; then
    echo -e "${GREEN}✅ Black formatting check passed${NC}"
else
    echo -e "${RED}❌ Black formatting issues found. Run 'uv run black src/ tests/' to fix.${NC}"
    exit 1
fi

# Run isort (import sorting)
echo -e "${YELLOW}📦 Running isort (import sorting)...${NC}"
if uv run isort --check-only --diff src/ tests/; then
    echo -e "${GREEN}✅ isort check passed${NC}"
else
    echo -e "${RED}❌ isort issues found. Run 'uv run isort src/ tests/' to fix.${NC}"
    exit 1
fi

# Run flake8 (style and errors)
echo -e "${YELLOW}🔍 Running flake8 (style and errors)...${NC}"
if uv run flake8 src/ tests/; then
    echo -e "${GREEN}✅ flake8 check passed${NC}"
else
    echo -e "${RED}❌ flake8 issues found${NC}"
    exit 1
fi

# Run mypy (type checking)
echo -e "${YELLOW}🔬 Running mypy (type checking)...${NC}"
if uv run mypy src/; then
    echo -e "${GREEN}✅ mypy type checking passed${NC}"
else
    echo -e "${RED}❌ mypy type checking issues found${NC}"
    exit 1
fi

echo -e "${GREEN}🎉 All linting checks passed!${NC}"