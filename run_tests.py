#!/usr/bin/env python3
"""Test runner script for StockSense."""
import subprocess
import sys
import os
from pathlib import Path


def run_command(command: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {command[0]}")
        return False


def main():
    """Main test runner function."""
    print("🚀 StockSense Test Runner")
    print("=" * 60)
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not running in a virtual environment")
        print("   Consider activating your virtual environment first")
    
    # Install dependencies
    if not run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      "Installing dependencies"):
        print("❌ Failed to install dependencies")
        return 1
    
    # Run linting
    print("\n🔍 Running code quality checks...")
    lint_success = True
    
    # Black formatting check
    if not run_command([sys.executable, "-m", "black", "--check", "pipelines/", "tests/"], 
                      "Black formatting check"):
        print("💡 Run 'black pipelines/ tests/' to fix formatting issues")
        lint_success = False
    
    # Flake8 linting
    if not run_command([sys.executable, "-m", "flake8", "pipelines/", "tests/"], 
                      "Flake8 linting"):
        lint_success = False
    
    # MyPy type checking
    if not run_command([sys.executable, "-m", "mypy", "pipelines/"], 
                      "MyPy type checking"):
        lint_success = False
    
    # Run tests
    print("\n🧪 Running tests...")
    test_commands = [
        # Unit tests
        ([sys.executable, "-m", "pytest", "tests/unit/", "-v", "--tb=short"], 
         "Unit tests"),
        
        # Integration tests
        ([sys.executable, "-m", "pytest", "tests/integration/", "-v", "--tb=short"], 
         "Integration tests"),
        
        # All tests with coverage
        ([sys.executable, "-m", "pytest", "tests/", "--cov=pipelines", 
          "--cov-report=term-missing", "--cov-report=html:htmlcov"], 
         "All tests with coverage"),
    ]
    
    test_success = True
    for command, description in test_commands:
        if not run_command(command, description):
            test_success = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    if lint_success:
        print("✅ Code quality checks passed")
    else:
        print("❌ Code quality checks failed")
    
    if test_success:
        print("✅ All tests passed")
        print("\n🎉 StockSense is ready for deployment!")
        return 0
    else:
        print("❌ Some tests failed")
        print("\n🔧 Please fix the failing tests before deployment")
        return 1


if __name__ == "__main__":
    sys.exit(main())
