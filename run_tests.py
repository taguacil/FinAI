#!/usr/bin/env python3
"""
Test runner script for FinAI project.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and print the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print("=" * 60)

    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    """Main test runner function."""
    print("🚀 FinAI Test Runner")
    print("=" * 60)

    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)

    # Check if we're in the right directory
    if not Path("src").exists() or not Path("tests").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)

    # Install test dependencies if needed
    print("\n📦 Checking test dependencies...")
    try:
        import coverage
        import pytest

        print("✅ Test dependencies already installed")
    except ImportError:
        print("📥 Installing test dependencies...")
        if not run_command(
            "pip install pytest pytest-cov coverage", "Installing test dependencies"
        ):
            print("❌ Failed to install test dependencies")
            sys.exit(1)

    # Run unit tests
    print("\n🧪 Running unit tests...")
    if not run_command("python -m pytest tests/unit/ -v --tb=short", "Unit tests"):
        print("❌ Unit tests failed")
        sys.exit(1)

    # Run data provider tests
    print("\n🔌 Running data provider tests...")
    if not run_command(
        "python -m pytest tests/unit/test_data_providers.py -v --tb=short",
        "Data provider tests",
    ):
        print("❌ Data provider tests failed")
        sys.exit(1)

    # Run tools tests
    print("\n🛠️ Running tools tests...")
    if not run_command(
        "python -m pytest tests/unit/test_tools.py -v --tb=short", "Tools tests"
    ):
        print("❌ Tools tests failed")
        sys.exit(1)

    # Run all tests with coverage
    print("\n📊 Running all tests with coverage...")
    if not run_command(
        "python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html",
        "All tests with coverage",
    ):
        print("❌ Tests with coverage failed")
        sys.exit(1)

    # Generate coverage report
    print("\n📈 Generating coverage report...")
    if not run_command("coverage report", "Coverage report"):
        print("⚠️ Coverage report generation failed")

    print("\n🎉 All tests completed successfully!")
    print("\n📁 Coverage reports generated:")
    print("   - HTML: htmlcov/index.html")
    print("   - Terminal: See above output")

    # Open coverage report in browser if possible
    try:
        import webbrowser

        coverage_file = Path("htmlcov/index.html")
        if coverage_file.exists():
            print(f"\n🌐 Opening coverage report in browser...")
            webbrowser.open(f"file://{coverage_file.absolute()}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
