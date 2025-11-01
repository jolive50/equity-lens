#!/usr/bin/env python3
"""Convenience test runner for StockSense.

The script keeps local quality checks consistent by running linters and the test
suite in a known order.  Each command stops on failure but the script continues
so the summary reports everything that needs attention.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


def run(command: Iterable[str], description: str) -> bool:
    """Execute ``command`` and report whether it succeeded."""

    command_list = list(command)
    print("=" * 72)
    print(f"{description}")
    print("Command:", " ".join(command_list))
    print("=" * 72)

    try:
        subprocess.run(command_list, check=True)
    except FileNotFoundError:
        print(f"Missing executable: {command_list[0]}")
        return False
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}")
        return False
    else:
        print("Completed successfully.")
        return True


def preferred_python() -> str:
    """Return the interpreter path we should use for subprocess calls."""

    return sys.executable or "python"


def quality_commands(python: str) -> List[Tuple[List[str], str]]:
    """Return linting and static-analysis commands."""

    return [
        ([python, "-m", "black", "--check", "pipelines", "tests"], "Black formatting check"),
        ([python, "-m", "flake8", "pipelines", "tests"], "Flake8 linting"),
        ([python, "-m", "mypy", "pipelines"], "MyPy type checking"),
    ]


def test_commands(python: str) -> List[Tuple[List[str], str]]:
    """Return the pytest commands we usually run."""

    return [
        ([python, "-m", "pytest", "tests/unit", "-v", "--tb=short"], "Unit tests"),
        ([python, "-m", "pytest", "tests/integration", "-v", "--tb=short"], "Integration tests"),
        (
            [
                python,
                "-m",
                "pytest",
                "tests",
                "--cov=pipelines",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
            ],
            "Full suite with coverage",
        ),
    ]


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    print("StockSense quality gate")
    print("Repository root:", project_root)

    python = preferred_python()
    success = True

    for command, description in quality_commands(python):
        success &= run(command, description)

    for command, description in test_commands(python):
        success &= run(command, description)

    print("=" * 72)
    if success:
        print("All checks passed. Ready to ship!")
        return 0

    print("Some checks failed. Review the logs above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
