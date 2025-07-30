#!/bin/bash
# Test script for Portfolio Tracker

set -e  # Exit on any error

echo "🧪 Running Portfolio Tracker tests..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}📂 Current directory: $(pwd)${NC}"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}❌ pyproject.toml not found. Please run from project root.${NC}"
    exit 1
fi

# Parse command line arguments
FAST_MODE=false
UNIT_ONLY=false
INTEGRATION_ONLY=false
COVERAGE_REPORT=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --fast)
            FAST_MODE=true
            echo -e "${BLUE}🚀 Fast mode enabled (skipping slow tests)${NC}"
            shift
            ;;
        --unit)
            UNIT_ONLY=true
            echo -e "${BLUE}🔬 Running unit tests only${NC}"
            shift
            ;;
        --integration)
            INTEGRATION_ONLY=true
            echo -e "${BLUE}🔗 Running integration tests only${NC}"
            shift
            ;;
        --no-cov)
            COVERAGE_REPORT=false
            echo -e "${BLUE}📊 Skipping coverage report${NC}"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --fast         Skip slow tests"
            echo "  --unit         Run unit tests only"
            echo "  --integration  Run integration tests only"
            echo "  --no-cov       Skip coverage report"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_ARGS="tests/"

# Add marker filters
if [ "$UNIT_ONLY" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS -m unit"
elif [ "$INTEGRATION_ONLY" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS -m integration"
fi

if [ "$FAST_MODE" = true ]; then
    if [ "$UNIT_ONLY" = true ] || [ "$INTEGRATION_ONLY" = true ]; then
        PYTEST_ARGS="$PYTEST_ARGS and not slow"
    else
        PYTEST_ARGS="$PYTEST_ARGS -m 'not slow'"
    fi
fi

# Add coverage options
if [ "$COVERAGE_REPORT" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS --cov=src --cov-report=term-missing --cov-report=html"
fi

# Run tests
echo -e "${YELLOW}🧪 Running tests with: uv run pytest $PYTEST_ARGS${NC}"
if uv run pytest $PYTEST_ARGS; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    
    # Show coverage report location if generated
    if [ "$COVERAGE_REPORT" = true ]; then
        echo -e "${BLUE}📊 Coverage report generated in htmlcov/index.html${NC}"
    fi
else
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi

echo -e "${GREEN}🎉 Test run completed successfully!${NC}"