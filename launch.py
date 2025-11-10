#!/usr/bin/env python
"""Convenience launcher for backend (FastAPI) and frontend (Next.js).

Usage:
    python launch.py          # start both services
    python launch.py backend  # backend only
    python launch.py frontend # frontend only
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import signal
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = REPO_ROOT / "frontend"


def which_or_exit(binary: str) -> str:
    path = shutil.which(binary)
    if path:
        return path
    print(f"[launch] ERROR: '{binary}' not found on PATH.", file=sys.stderr)
    sys.exit(1)


def build_commands(targets: Iterable[str]) -> List[Tuple[str, List[str], Path]]:
    python_exec = sys.executable or which_or_exit("python")
    npm_exec = which_or_exit("npm")

    commands: List[Tuple[str, List[str], Path]] = []

    if "backend" in targets:
        commands.append(
            (
                "backend",
                [
                    python_exec,
                    "-m",
                    "uvicorn",
                    "api.main:app",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8000",
                    "--reload",
                    "--log-level",
                    "info",
                ],
                REPO_ROOT,
            )
        )

    if "frontend" in targets:
        commands.append(("frontend", [npm_exec, "run", "dev"], FRONTEND_DIR))

    return commands


async def stream_output(prefix: str, stream: asyncio.StreamReader) -> None:
    while True:
        line = await stream.readline()
        if not line:
            break
        text = line.decode(errors="replace").rstrip()
        print(f"[{prefix}] {text}")


async def launch_process(name: str, cmd: List[str], cwd: Path) -> asyncio.subprocess.Process:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    print(f"[launch] Started {name} (pid={proc.pid})")
    return proc


async def run_services(targets: Iterable[str]) -> int:
    commands = build_commands(targets)

    if not commands:
        print("[launch] Nothing to run.")
        return 0

    processes: List[asyncio.subprocess.Process] = []
    tasks: List[asyncio.Task] = []

    try:
        for name, cmd, cwd in commands:
            proc = await launch_process(name, cmd, cwd)
            processes.append(proc)
            tasks.append(asyncio.create_task(stream_output(name, proc.stdout)))  # type: ignore[arg-type]

        print("[launch] All services running. Press Ctrl+C to stop.")
        await asyncio.gather(*(proc.wait() for proc in processes))
        await asyncio.gather(*tasks)
        return 0

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[launch] Stopping services...")
        for proc in processes:
            if proc.returncode is None:
                proc.send_signal(signal.SIGINT)
        await asyncio.gather(*(proc.wait() for proc in processes), return_exceptions=True)
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch backend and/or frontend services.")
    parser.add_argument(
        "target",
        nargs="?",
        choices=("all", "backend", "frontend"),
        default="all",
        help="Which services to start (default: all).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = {"backend", "frontend"} if args.target == "all" else {args.target}

    if not FRONTEND_DIR.exists():
        print(f"[launch] ERROR: Frontend directory not found at {FRONTEND_DIR}", file=sys.stderr)
        return 1

    return asyncio.run(run_services(targets))


if __name__ == "__main__":
    sys.exit(main())
