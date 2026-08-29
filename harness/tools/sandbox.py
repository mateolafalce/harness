"""Disposable repository copies for side-effecting shell commands."""

from __future__ import annotations

import os
from pathlib import Path

from harness.config import EXCLUDED_DIRECTORY_NAMES
from harness.workspace import current_workspace


def sandbox_ignore(directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    workspace = current_workspace()
    relative_directory = Path(directory).resolve().relative_to(workspace.root)
    for name in names:
        relative = relative_directory / name
        if name in EXCLUDED_DIRECTORY_NAMES - {".git"}:
            ignored.add(name)
        elif workspace.is_sensitive(relative) and ".git" not in relative.parts:
            ignored.add(name)
        elif name == "events.jsonl":
            ignored.add(name)
    return ignored


def sandbox_environment() -> dict[str, str]:
    """Keep basic process settings while withholding credentials and proxies."""
    allowed_names = {
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PATH",
        "SYSTEMROOT",
        "TEMP",
        "TERM",
        "TMP",
        "TMPDIR",
        "WINDIR",
    }
    environment = {
        name: value for name, value in os.environ.items() if name in allowed_names
    }
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment
