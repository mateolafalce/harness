"""Single source of truth for agent tool schemas and handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from harness.config import (
    ALLOWED_SHELL_COMMANDS,
    MAX_LISTED_FILES,
    MAX_READ_LINES,
    MAX_SEARCH_RESULTS,
)
from harness.tools.basic import calculator, echo, get_current_time
from harness.tools.fs import git_diff, list_files, read_file, search_text
from harness.tools.patch import apply_patch
from harness.tools.shell import run_shell, run_tests


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    required: list[str]
    handler: Callable[..., dict[str, Any]]
    requires_approval: bool = False
    supports_timeout: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                    "additionalProperties": False,
                },
            },
        }


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="calculator",
        description=(
            "Evaluate an arithmetic expression using numbers, parentheses, "
            "and the operators +, -, *, /, //, %, and **."
        ),
        parameters={
            "expression": {
                "type": "string",
                "description": "Arithmetic expression to evaluate.",
            }
        },
        required=["expression"],
        handler=calculator,
    ),
    ToolSpec(
        name="get_current_time",
        description="Return the current time in an IANA time zone.",
        parameters={
            "timezone": {
                "type": "string",
                "description": (
                    "IANA time zone such as UTC or America/Argentina/Mendoza."
                ),
            }
        },
        required=["timezone"],
        handler=get_current_time,
    ),
    ToolSpec(
        name="echo",
        description="Return the supplied text unchanged.",
        parameters={
            "text": {
                "type": "string",
                "description": "Text to return.",
            }
        },
        required=["text"],
        handler=echo,
    ),
    ToolSpec(
        name="list_files",
        description=(
            "Recursively list files below a repository-relative directory. "
            f"Returns at most {MAX_LISTED_FILES} paths."
        ),
        parameters={
            "path": {
                "type": "string",
                "description": "Repository-relative directory, such as . or tests.",
            }
        },
        required=["path"],
        handler=list_files,
    ),
    ToolSpec(
        name="read_file",
        description=(
            "Read a bounded line range from a UTF-8 repository file. "
            f"At most {MAX_READ_LINES} lines are accepted per call. Reports "
            "applicable AGENTS.md paths; read nested instructions before "
            "modifying files in their scope."
        ),
        parameters={
            "path": {
                "type": "string",
                "description": "Repository-relative file path.",
            },
            "start_line": {
                "type": "integer",
                "minimum": 1,
                "description": "First one-based line to return.",
            },
            "max_lines": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_READ_LINES,
                "description": "Maximum number of lines to return.",
            },
        },
        required=["path", "start_line", "max_lines"],
        handler=read_file,
    ),
    ToolSpec(
        name="search_text",
        description=(
            "Search UTF-8 repository files for a literal, case-sensitive text "
            f"fragment. Returns at most {MAX_SEARCH_RESULTS} matches."
        ),
        parameters={
            "query": {
                "type": "string",
                "description": (
                    "Literal text to find; regular expressions are not used."
                ),
            },
            "path": {
                "type": "string",
                "description": "Repository-relative file or directory to search.",
            },
        },
        required=["query", "path"],
        handler=search_text,
    ),
    ToolSpec(
        name="git_diff",
        description=(
            "Show the staged and unstaged Git diff for a repository-relative path. "
            "Output is bounded and external diff programs are disabled."
        ),
        parameters={
            "path": {
                "type": "string",
                "description": (
                    "Repository-relative path, or . for the whole repository."
                ),
            }
        },
        required=["path"],
        handler=git_diff,
        supports_timeout=True,
    ),
    ToolSpec(
        name="run_tests",
        description=(
            "Run the repository's fixed unittest suite in a disposable copy. "
            "The call requires approval; output and runtime are bounded."
        ),
        parameters={},
        required=[],
        handler=run_tests,
        requires_approval=True,
        supports_timeout=True,
    ),
    ToolSpec(
        name="apply_patch",
        description=(
            "Apply an approved text patch inside the repository. Accepts the "
            "*** Begin Patch format or a standard unified Git diff."
        ),
        parameters={
            "patch": {
                "type": "string",
                "description": (
                    "Patch text, limited to 100,000 characters and 50 files."
                ),
            }
        },
        required=["patch"],
        handler=apply_patch,
        requires_approval=True,
        supports_timeout=True,
    ),
    ToolSpec(
        name="run_shell",
        description=(
            "Run one allowlisted command in a disposable repository copy. "
            "No shell interpreter is used; output and runtime are bounded. "
            f"Allowed forms: {'; '.join(ALLOWED_SHELL_COMMANDS)}."
        ),
        parameters={
            "command": {
                "type": "string",
                "description": (
                    "A single allowlisted command, up to 1,000 characters."
                ),
            }
        },
        required=["command"],
        handler=run_shell,
        requires_approval=True,
        supports_timeout=True,
    ),
)

TOOLS = [spec.schema() for spec in TOOL_SPECS]
TOOL_HANDLERS = {spec.name: spec.handler for spec in TOOL_SPECS}
TOOL_SPECS_BY_NAME = {spec.name: spec for spec in TOOL_SPECS}
APPROVAL_REQUIRED_TOOLS = {
    spec.name for spec in TOOL_SPECS if spec.requires_approval
}
