"""Bounded subprocess execution for repository tools."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from harness.config import MAX_PROCESS_OUTPUT_LINES, MAX_TOOL_OUTPUT_CHARS, TEST_TIMEOUT_SECONDS
from harness.workspace import current_workspace


def truncate_output(
    output: str,
    *,
    max_lines: int = MAX_PROCESS_OUTPUT_LINES,
    max_chars: int = MAX_TOOL_OUTPUT_CHARS,
    keep_tail: bool = False,
) -> tuple[str, bool]:
    """Bound process output by lines and characters while preserving useful context."""
    lines = output.splitlines()
    truncated = len(lines) > max_lines
    if truncated:
        lines = lines[-max_lines:] if keep_tail else lines[:max_lines]
    bounded = "\n".join(lines)
    if len(bounded) > max_chars:
        truncated = True
        bounded = bounded[-max_chars:] if keep_tail else bounded[:max_chars]
    return bounded, truncated


def run_bounded_process(
    command: list[str],
    *,
    keep_tail: bool,
    timeout_seconds: float = TEST_TIMEOUT_SECONDS,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd or current_workspace().root,
            env=env or {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", errors="replace")
        output, truncated = truncate_output(partial, keep_tail=True)
        return {
            "exit_code": None,
            "output": output,
            "truncated": truncated,
            "timed_out": True,
        }

    output, truncated = truncate_output(completed.stdout, keep_tail=keep_tail)
    return {
        "exit_code": completed.returncode,
        "output": output,
        "truncated": truncated,
        "timed_out": False,
    }
