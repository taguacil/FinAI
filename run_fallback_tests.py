#!/usr/bin/env python3
"""Test runner for transaction-based fallback functionality."""

import sys
import os
import subprocess
from pathlib import Path

def run_tests():
    """Run the fallback functionality tests."""
    print("🧪 Running Transaction-Based Fallback Tests")
    print("=" * 60)

    # Get the project root directory
    project_root = Path(__file__).parent
    tests_dir = project_root / "tests"

    if not tests_dir.exists():
        print("❌ Tests directory not found!")
        return False

    # Check if pytest is available
    try:
        import pytest
        print("✅ Pytest is available")
    except ImportError:
        print("❌ Pytest not found. Installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pytest"], check=True)
            print("✅ Pytest installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install pytest")
            return False

    # Run specific test files
    test_files = [
        "tests/unit/test_transaction_fallbacks.py",
        "tests/unit/test_position_fallbacks.py",
        "tests/unit/test_portfolio_fallbacks.py"
    ]

    # Check which test files exist
    existing_tests = []
    for test_file in test_files:
        if Path(test_file).exists():
            existing_tests.append(test_file)
            print(f"📁 Found test file: {test_file}")
        else:
            print(f"⚠️  Test file not found: {test_file}")

    if not existing_tests:
        print("❌ No test files found!")
        return False

    print(f"\n🚀 Running {len(existing_tests)} test files...")
    print("-" * 60)

    # Run tests with verbose output
    try:
        cmd = [
            sys.executable, "-m", "pytest",
            "-v",  # Verbose output
            "--tb=short",  # Short traceback format
            "--color=yes",  # Colored output
            "--durations=10",  # Show top 10 slowest tests
            "--markers",  # Show available markers
        ] + existing_tests

        result = subprocess.run(cmd, cwd=project_root, check=False)

        print("\n" + "=" * 60)
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print(f"❌ Some tests failed (exit code: {result.returncode})")
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ Error running tests: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def run_specific_test_category(category):
    """Run tests for a specific category."""
    print(f"🧪 Running {category} Tests")
    print("=" * 60)

    project_root = Path(__file__).parent

    try:
        cmd = [
            sys.executable, "-m", "pytest",
            "-v",
            "--tb=short",
            "--color=yes",
            "-m", category,  # Run only tests with this marker
            "tests/unit/",
        ]

        result = subprocess.run(cmd, cwd=project_root, check=False)

        print("\n" + "=" * 60)
        if result.returncode == 0:
            print(f"✅ All {category} tests passed!")
            return True
        else:
            print(f"❌ Some {category} tests failed (exit code: {result.returncode})")
            return False

    except Exception as e:
        print(f"❌ Error running {category} tests: {e}")
        return False

def show_test_markers():
    """Show available test markers."""
    print("🏷️  Available Test Markers")
    print("=" * 60)

    project_root = Path(__file__).parent

    try:
        cmd = [
            sys.executable, "-m", "pytest",
            "--markers",
            "tests/unit/",
        ]

        subprocess.run(cmd, cwd=project_root, check=False)

    except Exception as e:
        print(f"❌ Error showing markers: {e}")

def main():
    """Main function to run tests."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "all":
            success = run_tests()
        elif command == "transaction":
            success = run_specific_test_category("transaction_fallbacks")
        elif command == "position":
            success = run_specific_test_category("position_fallbacks")
        elif command == "portfolio":
            success = run_specific_test_category("portfolio_fallbacks")
        elif command == "snapshot":
            success = run_specific_test_category("snapshot_creation")
        elif command == "exchange":
            success = run_specific_test_category("exchange_rates")
        elif command == "markers":
            show_test_markers()
            return
        elif command == "help":
            print("""
🧪 Transaction-Based Fallback Test Runner

Usage:
  python run_fallback_tests.py [command]

Commands:
  all         - Run all fallback tests
  transaction - Run transaction fallback tests only
  position    - Run position fallback tests only
  portfolio   - Run portfolio fallback tests only
  snapshot    - Run snapshot creation tests only
  exchange    - Run exchange rate tests only
  markers     - Show available test markers
  help        - Show this help message

Examples:
  python run_fallback_tests.py all
  python run_fallback_tests.py transaction
  python run_fallback_tests.py position
            """)
            return
        else:
            print(f"❌ Unknown command: {command}")
            print("Use 'python run_fallback_tests.py help' for usage information")
            return

        sys.exit(0 if success else 1)
    else:
        # Default: run all tests
        success = run_tests()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
