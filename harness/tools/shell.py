"""Allowlisted shell execution in a disposable sandbox."""

from __future__ import annotations

import os
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Any

from harness.config import (
    ALLOWED_SHELL_COMMANDS,
    MAX_SHELL_COMMAND_CHARS,
    SHELL_TIMEOUT_SECONDS,
    TEST_TIMEOUT_SECONDS,
)
from harness.exceptions import ToolArgumentError
from harness.tools.process import run_bounded_process
from harness.tools.sandbox import sandbox_environment, sandbox_ignore
from harness.tools.validation import require_exact_arguments
from harness.workspace import current_workspace

_run_bounded_process = run_bounded_process


def prepare_shell_command(command_text: str) -> tuple[list[str], str]:
    if not command_text.strip():
        raise ToolArgumentError("argument 'command' must not be empty")
    if len(command_text) > MAX_SHELL_COMMAND_CHARS:
        raise ToolArgumentError(
            f"argument 'command' must not exceed {MAX_SHELL_COMMAND_CHARS} characters"
        )
    try:
        tokens = shlex.split(command_text)
    except ValueError as exc:
        raise ToolArgumentError("command has invalid quoting") from exc
    if not tokens:
        raise ToolArgumentError("argument 'command' must not be empty")

    git_prefix = [
        "git",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
    ]
    workspace = current_workspace()
    if tokens == ["git", "status", "--short"]:
        return [*git_prefix, "status", "--short"], command_text
    if tokens in (
        ["git", "diff", "--check"],
        ["git", "diff", "--cached", "--check"],
    ):
        return [
            *git_prefix,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            *tokens[2:],
        ], command_text
    if tokens[:2] == ["git", "diff"] and (
        len(tokens) == 2 or (len(tokens) == 4 and tokens[2] == "--")
    ):
        prepared = [
            *git_prefix,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
        ]
        if len(tokens) == 4:
            _, display_path = workspace.resolve_path(tokens[3])
            prepared.extend(["--", display_path])
        return prepared, command_text
    if len(tokens) == 3 and tokens[0] == "rg":
        query = tokens[1]
        if not query:
            raise ToolArgumentError("rg query must not be empty")
        if len(query) > 200:
            raise ToolArgumentError("rg query must not exceed 200 characters")
        _, display_path = workspace.resolve_path(tokens[2])
        return [
            "rg",
            "--line-number",
            "--fixed-strings",
            "--no-heading",
            "--color",
            "never",
            "--",
            query,
            display_path,
        ], command_text
    if tokens == [
        ".venv/bin/python",
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-v",
    ]:
        configured_python = os.environ.get("HARNESS_PYTHON")
        python = (
            Path(configured_python)
            if configured_python
            else workspace.root / ".venv" / "bin" / "python"
        )
        if configured_python and not python.is_absolute():
            raise ToolArgumentError("HARNESS_PYTHON must be an absolute path")
        if not python.is_file():
            raise ToolArgumentError(f"test interpreter is missing: {python}")
        return [str(python), *tokens[1:]], command_text
    allowed = "; ".join(ALLOWED_SHELL_COMMANDS)
    raise ToolArgumentError(f"command is not allowlisted; allowed forms: {allowed}")


def run_shell(
    arguments: dict[str, Any], timeout_seconds: float = SHELL_TIMEOUT_SECONDS
) -> dict[str, Any]:
    validated = require_exact_arguments(arguments, required={"command": str})
    command, display_command = prepare_shell_command(validated["command"])
    with tempfile.TemporaryDirectory(prefix="harness-sandbox-") as temporary:
        sandbox_root = Path(temporary) / "repository"
        try:
            shutil.copytree(
                current_workspace().root,
                sandbox_root,
                symlinks=True,
                ignore=sandbox_ignore,
            )
        except OSError as exc:
            raise ToolArgumentError("could not create disposable sandbox") from exc
        result = _run_bounded_process(
            command,
            keep_tail=True,
            timeout_seconds=min(timeout_seconds, SHELL_TIMEOUT_SECONDS),
            cwd=sandbox_root,
            env=sandbox_environment(),
        )
    result["command"] = display_command
    result["sandbox"] = "disposable repository copy"
    return result


def run_tests(
    arguments: dict[str, Any], timeout_seconds: float = TEST_TIMEOUT_SECONDS
) -> dict[str, Any]:
    require_exact_arguments(arguments, required={})
    return run_shell(
        {"command": ".venv/bin/python -m unittest discover -s tests -v"},
        timeout_seconds=min(timeout_seconds, TEST_TIMEOUT_SECONDS),
    )
