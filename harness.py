#!/usr/bin/env python3
"""Stage 3: a small, read-only coding agent backed by Cerebras."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown


DEFAULT_CONTEXT_WINDOW = 131_072
DEFAULT_MAX_TURNS = 8
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_LISTED_FILES = 500
MAX_READ_LINES = 200
MAX_TOOL_OUTPUT_CHARS = 16_000
MAX_SEARCH_RESULTS = 50
MAX_SEARCH_LINE_CHARS = 240
MAX_SEARCH_FILE_BYTES = 1_000_000
MAX_PROCESS_OUTPUT_LINES = 400
TEST_TIMEOUT_SECONDS = 60.0
EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}
DEFAULT_SYSTEM_PROMPT = (
    "You are a read-only coding agent. Inspect the repository with the available "
    "tools, select only the context needed to answer, and do not claim to have "
    "modified files. Tool results may be truncated; request a narrower path or "
    "range when needed."
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate an arithmetic expression using numbers, parentheses, "
                "and the operators +, -, *, /, //, %, and **."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression to evaluate.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Return the current time in an IANA time zone.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": (
                            "IANA time zone such as UTC or America/Argentina/Mendoza."
                        ),
                    }
                },
                "required": ["timezone"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Return the supplied text unchanged.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to return.",
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "Recursively list files below a repository-relative directory. "
                f"Returns at most {MAX_LISTED_FILES} paths."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repository-relative directory, such as . or tests.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a bounded line range from a UTF-8 repository file. "
                f"At most {MAX_READ_LINES} lines are accepted per call."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
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
                "required": ["path", "start_line", "max_lines"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": (
                "Search UTF-8 repository files for a literal, case-sensitive text "
                f"fragment. Returns at most {MAX_SEARCH_RESULTS} matches."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
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
                "required": ["query", "path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": (
                "Show the staged and unstaged Git diff for a repository-relative path. "
                "Output is bounded and external diff programs are disabled."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Repository-relative path, or . for the whole repository."
                        ),
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "Run the repository's fixed unittest suite without accepting a shell "
                "command. Output and runtime are bounded."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]


class AgentLoopError(RuntimeError):
    """Base class for controlled agent-loop failures."""


class AgentTimeoutError(AgentLoopError):
    """Raised when one user turn exceeds its wall-clock deadline."""


class MaxTurnsExceededError(AgentLoopError):
    """Raised when the model does not finish within the configured turn limit."""


class ToolProtocolError(AgentLoopError):
    """Raised when a model tool call cannot be correlated safely."""


class ToolArgumentError(ValueError):
    """Raised when tool arguments do not satisfy the declared input schema."""


def positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def positive_float(value: str) -> float:
    """Parse a finite, strictly positive command-line number."""
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return number


class JsonlEventLogger:
    """Append structured session events to a JSON Lines file."""

    def __init__(self, path: Path, session_id: str | None = None) -> None:
        self.path = path
        self.session_id = session_id or str(uuid.uuid4())
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **data: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "event": event,
            **data,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, default=str)
            stream.write("\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conversational Cerebras client with a small agent loop."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Question for the model. Omit it to start interactive mode.",
    )
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default="medium",
    )
    parser.add_argument("--max-completion-tokens", type=positive_int, default=8_192)
    parser.add_argument(
        "--max-turns",
        type=positive_int,
        default=DEFAULT_MAX_TURNS,
        help=f"Maximum model calls per user message (default: {DEFAULT_MAX_TURNS}).",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            "Wall-clock timeout in seconds for each complete agent loop "
            f"(default: {DEFAULT_TIMEOUT_SECONDS:g})."
        ),
    )
    parser.add_argument(
        "--context-window",
        type=positive_int,
        default=DEFAULT_CONTEXT_WINDOW,
        help=(
            "Model context window in tokens, used to calculate utilization "
            f"(default: {DEFAULT_CONTEXT_WINDOW})."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("events.jsonl"),
        help="JSON Lines event log (default: events.jsonl).",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Initial system instruction included in every conversation.",
    )
    return parser.parse_args(argv)


def _cached_token_count(usage: Any) -> int | None:
    prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
    return getattr(prompt_tokens_details, "cached_tokens", None)


def response_metrics(response: Any, latency_seconds: float, context_window: int) -> dict[str, Any]:
    """Extract token counts and derive context utilization."""
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    cached_tokens = _cached_token_count(usage)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    context_percent = (
        round(total_tokens / context_window * 100, 4)
        if total_tokens is not None
        else None
    )
    return {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_ms": round(latency_seconds * 1_000, 2),
        "context_window_tokens": context_window,
        "context_used_percent": context_percent,
    }


def format_metric(value: Any, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value}{suffix}"


def print_response(content: str, metrics: dict[str, Any]) -> None:
    """Render the Markdown answer and its request metrics."""
    console = Console()
    console.print()
    console.print("assistant> ", style="bold cyan", end="")
    console.print(Markdown(content))
    console.print(
        "metrics> "
        f"prompt={format_metric(metrics['prompt_tokens'])} tokens | "
        f"cached={format_metric(metrics['cached_tokens'])} tokens | "
        f"completion={format_metric(metrics['completion_tokens'])} tokens | "
        f"total={format_metric(metrics['total_tokens'])} tokens | "
        f"latency={format_metric(metrics['latency_ms'], ' ms')} | "
        f"context={format_metric(metrics['context_used_percent'], '%')}",
        style="dim",
    )


def _require_exact_arguments(
    arguments: Any,
    *,
    required: dict[str, type],
) -> dict[str, Any]:
    """Validate a small object schema without adding a framework dependency."""
    if not isinstance(arguments, dict):
        raise ToolArgumentError("arguments must be a JSON object")

    expected = set(required)
    received = set(arguments)
    missing = expected - received
    unexpected = received - expected
    if missing:
        raise ToolArgumentError(f"missing required argument(s): {', '.join(sorted(missing))}")
    if unexpected:
        raise ToolArgumentError(f"unexpected argument(s): {', '.join(sorted(unexpected))}")

    for name, expected_type in required.items():
        if not isinstance(arguments[name], expected_type):
            raise ToolArgumentError(
                f"argument '{name}' must be {expected_type.__name__}"
            )
    return arguments


_BINARY_OPERATORS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.FloorDiv: lambda left, right: left // right,
    ast.Mod: lambda left, right: left % right,
    ast.Pow: lambda left, right: left**right,
}
_UNARY_OPERATORS = {
    ast.UAdd: lambda value: value,
    ast.USub: lambda value: -value,
}
MAX_EXPRESSION_LENGTH = 200
MAX_EXPRESSION_NODES = 100
MAX_ABSOLUTE_RESULT = 1e100
MAX_ABSOLUTE_EXPONENT = 1_000


def _evaluate_arithmetic(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate_arithmetic(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolArgumentError("expression may contain only numbers and operators")
        if not math.isfinite(node.value):
            raise ToolArgumentError("numbers must be finite")
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        result = _UNARY_OPERATORS[type(node.op)](_evaluate_arithmetic(node.operand))
    elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_arithmetic(node.left)
        right = _evaluate_arithmetic(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_ABSOLUTE_EXPONENT:
            raise ToolArgumentError("absolute exponent must not exceed 1000")
        result = _BINARY_OPERATORS[type(node.op)](left, right)
    else:
        raise ToolArgumentError("expression contains an unsupported operation")

    if isinstance(result, complex) or not math.isfinite(result):
        raise ToolArgumentError("result must be a finite real number")
    if abs(result) > MAX_ABSOLUTE_RESULT:
        raise ToolArgumentError("absolute result is too large")
    return result


def calculator(arguments: dict[str, Any]) -> dict[str, int | float]:
    validated = _require_exact_arguments(arguments, required={"expression": str})
    expression = validated["expression"].strip()
    if not expression:
        raise ToolArgumentError("argument 'expression' must not be empty")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ToolArgumentError(
            f"argument 'expression' must not exceed {MAX_EXPRESSION_LENGTH} characters"
        )
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolArgumentError("expression is not valid arithmetic") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_EXPRESSION_NODES:
        raise ToolArgumentError("expression is too complex")
    try:
        return {"value": _evaluate_arithmetic(tree)}
    except (ArithmeticError, OverflowError) as exc:
        raise ToolArgumentError(str(exc) or type(exc).__name__) from exc


def get_current_time(arguments: dict[str, Any]) -> dict[str, str]:
    validated = _require_exact_arguments(arguments, required={"timezone": str})
    timezone_name = validated["timezone"].strip()
    if not timezone_name:
        raise ToolArgumentError("argument 'timezone' must not be empty")
    try:
        requested_timezone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ToolArgumentError(f"unknown IANA time zone: {timezone_name}") from exc
    current = datetime.now(requested_timezone)
    return {
        "timezone": timezone_name,
        "iso8601": current.isoformat(timespec="seconds"),
    }


def echo(arguments: dict[str, Any]) -> dict[str, str]:
    validated = _require_exact_arguments(arguments, required={"text": str})
    return {"text": validated["text"]}


def _repository_root() -> Path:
    """Return the repository boundary used by all filesystem tools."""
    return Path.cwd().resolve()


def _is_sensitive_path(path: Path) -> bool:
    if any(part in EXCLUDED_DIRECTORY_NAMES for part in path.parts):
        return True
    return any(
        part == ".env" or (part.startswith(".env.") and part != ".env.example")
        for part in path.parts
    )


def _resolve_repository_path(raw_path: str) -> tuple[Path, str]:
    path_text = raw_path.strip()
    if not path_text:
        raise ToolArgumentError("argument 'path' must not be empty")

    supplied = Path(path_text)
    if supplied.is_absolute():
        raise ToolArgumentError("argument 'path' must be repository-relative")

    root = _repository_root()
    candidate = (root / supplied).resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ToolArgumentError("path must stay inside the repository") from exc
    if _is_sensitive_path(relative):
        raise ToolArgumentError("path is excluded from repository tools")
    display_path = relative.as_posix() or "."
    return candidate, display_path


def _visible_files(path: Path) -> list[Path]:
    """Return deterministic files without descending into generated directories."""
    root = _repository_root()
    if path.is_file():
        return [path]

    files: list[Path] = []
    for directory, directory_names, file_names in os.walk(path, followlinks=False):
        relative_directory = Path(directory).relative_to(root)
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in EXCLUDED_DIRECTORY_NAMES
            and not _is_sensitive_path(relative_directory / name)
        )
        for name in sorted(file_names):
            candidate = Path(directory) / name
            relative = candidate.relative_to(root)
            if _is_sensitive_path(relative):
                continue
            try:
                candidate.resolve().relative_to(root)
            except (OSError, ValueError):
                continue
            if not candidate.is_file():
                continue
            files.append(candidate)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def _validate_positive_integer(name: str, value: Any, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ToolArgumentError(f"argument '{name}' must be a positive integer")
    if maximum is not None and value > maximum:
        raise ToolArgumentError(f"argument '{name}' must not exceed {maximum}")
    return value


def _truncate_output(
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


def list_files(arguments: dict[str, Any]) -> dict[str, Any]:
    validated = _require_exact_arguments(arguments, required={"path": str})
    path, display_path = _resolve_repository_path(validated["path"])
    if not path.exists():
        raise ToolArgumentError(f"path does not exist: {display_path}")
    if not path.is_dir():
        raise ToolArgumentError(f"path is not a directory: {display_path}")

    root = _repository_root()
    files = [item.relative_to(root).as_posix() for item in _visible_files(path)]
    returned_files: list[str] = []
    returned_characters = 0
    for file_name in files[:MAX_LISTED_FILES]:
        serialized_size = len(json.dumps(file_name, ensure_ascii=False)) + 1
        if returned_characters + serialized_size > MAX_TOOL_OUTPUT_CHARS:
            break
        returned_files.append(file_name)
        returned_characters += serialized_size
    truncated = len(returned_files) < len(files)
    return {
        "path": display_path,
        "files": returned_files,
        "truncated": truncated,
        "returned_count": len(returned_files),
        "total_count": len(files),
    }


def read_file(arguments: dict[str, Any]) -> dict[str, Any]:
    validated = _require_exact_arguments(
        arguments,
        required={"path": str, "start_line": int, "max_lines": int},
    )
    start_line = _validate_positive_integer("start_line", validated["start_line"])
    max_lines = _validate_positive_integer(
        "max_lines", validated["max_lines"], MAX_READ_LINES
    )
    path, display_path = _resolve_repository_path(validated["path"])
    if not path.is_file():
        raise ToolArgumentError(f"path is not a file: {display_path}")

    selected: list[str] = []
    has_more = False
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if line_number < start_line:
                    continue
                if len(selected) == max_lines:
                    has_more = True
                    break
                selected.append(line.rstrip("\n\r"))
    except (OSError, UnicodeError) as exc:
        raise ToolArgumentError(f"file is not readable UTF-8: {display_path}") from exc

    content, character_truncated = _truncate_output(
        "\n".join(selected), max_lines=max_lines
    )
    returned_lines = len(content.splitlines()) if content else 0
    return {
        "path": display_path,
        "start_line": start_line,
        "end_line": start_line + returned_lines - 1 if returned_lines else None,
        "content": content,
        "truncated": has_more or character_truncated,
        "next_start_line": (
            start_line + returned_lines if has_more and not character_truncated else None
        ),
    }


def search_text(arguments: dict[str, Any]) -> dict[str, Any]:
    validated = _require_exact_arguments(
        arguments, required={"query": str, "path": str}
    )
    query = validated["query"]
    if not query:
        raise ToolArgumentError("argument 'query' must not be empty")
    if len(query) > 200:
        raise ToolArgumentError("argument 'query' must not exceed 200 characters")
    path, display_path = _resolve_repository_path(validated["path"])
    if not path.exists():
        raise ToolArgumentError(f"path does not exist: {display_path}")

    root = _repository_root()
    matches: list[dict[str, Any]] = []
    match_characters = 0
    skipped_files = 0
    truncated = False
    for candidate in _visible_files(path):
        try:
            if candidate.stat().st_size > MAX_SEARCH_FILE_BYTES:
                skipped_files += 1
                continue
            file_matches: list[dict[str, Any]] = []
            with candidate.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if query not in line:
                        continue
                    text = line.rstrip("\n\r")
                    match_index = text.find(query)
                    excerpt_start = max(0, match_index - MAX_SEARCH_LINE_CHARS // 3)
                    excerpt = text[
                        excerpt_start : excerpt_start + MAX_SEARCH_LINE_CHARS
                    ]
                    file_matches.append(
                        {
                            "path": candidate.relative_to(root).as_posix(),
                            "line": line_number,
                            "column": match_index + 1,
                            "text": excerpt,
                            "line_truncated": len(text) > MAX_SEARCH_LINE_CHARS,
                            "excerpt_start_column": excerpt_start + 1,
                        }
                    )
                    if len(matches) + len(file_matches) > MAX_SEARCH_RESULTS:
                        truncated = True
                        break
        except (OSError, UnicodeError):
            skipped_files += 1
            continue
        for match in file_matches:
            serialized_size = len(json.dumps(match, ensure_ascii=False)) + 1
            if (
                len(matches) == MAX_SEARCH_RESULTS
                or match_characters + serialized_size > MAX_TOOL_OUTPUT_CHARS
            ):
                truncated = True
                break
            matches.append(match)
            match_characters += serialized_size
        if truncated:
            break

    return {
        "query": query,
        "path": display_path,
        "matches": matches,
        "truncated": truncated,
        "returned_count": len(matches),
        "skipped_files": skipped_files,
    }


def _run_bounded_process(
    command: list[str],
    *,
    keep_tail: bool,
    timeout_seconds: float = TEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=_repository_root(),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
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
        output, truncated = _truncate_output(partial, keep_tail=True)
        return {
            "exit_code": None,
            "output": output,
            "truncated": truncated,
            "timed_out": True,
        }

    output, truncated = _truncate_output(completed.stdout, keep_tail=keep_tail)
    return {
        "exit_code": completed.returncode,
        "output": output,
        "truncated": truncated,
        "timed_out": False,
    }


def git_diff(
    arguments: dict[str, Any], timeout_seconds: float = TEST_TIMEOUT_SECONDS
) -> dict[str, Any]:
    validated = _require_exact_arguments(arguments, required={"path": str})
    _, display_path = _resolve_repository_path(validated["path"])
    result = _run_bounded_process(
        ["git", "diff", "--no-ext-diff", "--no-color", "HEAD", "--", display_path],
        keep_tail=False,
        timeout_seconds=min(timeout_seconds, TEST_TIMEOUT_SECONDS),
    )
    result["path"] = display_path
    return result


def run_tests(
    arguments: dict[str, Any], timeout_seconds: float = TEST_TIMEOUT_SECONDS
) -> dict[str, Any]:
    _require_exact_arguments(arguments, required={})
    root = _repository_root()
    python = root / ".venv" / "bin" / "python"
    if not python.is_file():
        raise ToolArgumentError("repository virtual environment is missing: .venv")
    result = _run_bounded_process(
        [str(python), "-m", "unittest", "discover", "-s", "tests", "-v"],
        keep_tail=True,
        timeout_seconds=min(timeout_seconds, TEST_TIMEOUT_SECONDS),
    )
    result["command"] = ".venv/bin/python -m unittest discover -s tests -v"
    return result


TOOL_HANDLERS = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "echo": echo,
    "list_files": list_files,
    "read_file": read_file,
    "search_text": search_text,
    "git_diff": git_diff,
    "run_tests": run_tests,
}


def execute_tool(
    name: str,
    raw_arguments: str | None,
    *,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Parse, validate, and execute one tool call as a serializable envelope."""
    try:
        if raw_arguments is None:
            raise ToolArgumentError("tool arguments are missing")
        try:
            arguments = json.loads(raw_arguments)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ToolArgumentError("arguments are not valid JSON") from exc

        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            raise ToolArgumentError(f"unknown tool: {name}")
        if name in {"git_diff", "run_tests"} and timeout_seconds is not None:
            return {
                "ok": True,
                "result": handler(arguments, timeout_seconds=timeout_seconds),
            }
        return {"ok": True, "result": handler(arguments)}
    except Exception as exc:
        return {
            "ok": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }


def _tool_call_as_message(call: Any) -> dict[str, Any]:
    function = getattr(call, "function", None)
    return {
        "id": getattr(call, "id", None),
        "type": getattr(call, "type", "function"),
        "function": {
            "name": getattr(function, "name", None),
            "arguments": getattr(function, "arguments", None),
        },
    }


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AgentTimeoutError("agent loop timed out")
    return remaining


def _combined_metrics(
    responses: list[Any], latency_seconds: float, context_window: int
) -> dict[str, Any]:
    def usage_total(attribute: str) -> int | None:
        values = [
            getattr(getattr(response, "usage", None), attribute, None)
            for response in responses
        ]
        present = [value for value in values if value is not None]
        return sum(present) if present else None

    prompt_tokens = usage_total("prompt_tokens")
    cached_token_counts = [
        _cached_token_count(getattr(response, "usage", None))
        for response in responses
    ]
    present_cached_token_counts = [
        count for count in cached_token_counts if count is not None
    ]
    cached_tokens = (
        sum(present_cached_token_counts) if present_cached_token_counts else None
    )
    completion_tokens = usage_total("completion_tokens")
    total_tokens = usage_total("total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_ms": round(latency_seconds * 1_000, 2),
        "context_window_tokens": context_window,
        "context_used_percent": (
            round(total_tokens / context_window * 100, 4)
            if total_tokens is not None
            else None
        ),
        "model_calls": len(responses),
    }


def run_turn(
    client: Cerebras,
    messages: list[dict[str, Any]],
    prompt: str,
    args: argparse.Namespace,
    event_logger: JsonlEventLogger,
) -> int:
    """Run model/tool turns until the model answers or a guardrail stops it."""
    history_start = len(messages)
    user_message = {"role": "user", "content": prompt}
    messages.append(user_message)
    event_logger.log(
        "user_message",
        content=prompt,
        history_message_count=len(messages),
    )
    max_turns = getattr(args, "max_turns", DEFAULT_MAX_TURNS)
    timeout_seconds = getattr(args, "timeout", DEFAULT_TIMEOUT_SECONDS)
    started_at = time.monotonic()
    deadline = started_at + timeout_seconds
    responses: list[Any] = []
    event_logger.log(
        "agent_loop_started",
        max_turns=max_turns,
        timeout_seconds=timeout_seconds,
        available_tools=sorted(TOOL_HANDLERS),
    )

    try:
        for turn_number in range(1, max_turns + 1):
            remaining = _remaining_seconds(deadline)
            event_logger.log(
                "agent_turn_started",
                turn=turn_number,
                remaining_ms=round(remaining * 1_000, 2),
                history_message_count=len(messages),
            )
            event_logger.log(
                "api_request",
                turn=turn_number,
                model=args.model,
                history_message_count=len(messages),
                max_completion_tokens=args.max_completion_tokens,
                tool_choice="auto",
                timeout_seconds=remaining,
            )

            request_started_at = time.monotonic()
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    reasoning_effort=args.reasoning_effort,
                    max_completion_tokens=args.max_completion_tokens,
                    tools=TOOLS,
                    tool_choice="auto",
                    timeout=remaining,
                )
            except Exception as exc:
                event_logger.log(
                    "api_error",
                    turn=turn_number,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    latency_ms=round(
                        (time.monotonic() - request_started_at) * 1_000, 2
                    ),
                )
                raise

            _remaining_seconds(deadline)
            if not response.choices:
                event_logger.log(
                    "api_error",
                    turn=turn_number,
                    error_type="EmptyResponse",
                    message="The API response did not contain any choices.",
                    latency_ms=round(
                        (time.monotonic() - request_started_at) * 1_000, 2
                    ),
                )
                raise RuntimeError("The API response did not contain any choices.")

            responses.append(response)
            choice = response.choices[0]
            content = choice.message.content or ""
            reasoning = getattr(choice.message, "reasoning", None)
            tool_calls = list(getattr(choice.message, "tool_calls", None) or [])
            request_metrics = response_metrics(
                response,
                time.monotonic() - request_started_at,
                args.context_window,
            )
            event_logger.log(
                "model_response",
                turn=turn_number,
                content=content,
                reasoning=reasoning,
                response_id=getattr(response, "id", None),
                model=getattr(response, "model", args.model),
                finish_reason=getattr(choice, "finish_reason", None),
                metrics=request_metrics,
                tool_call_count=len(tool_calls),
            )

            if not tool_calls:
                messages.append({"role": "assistant", "content": content})
                event_logger.log(
                    "agent_decision",
                    turn=turn_number,
                    decision="final_response",
                    history_message_count=len(messages),
                )
                metrics = _combined_metrics(
                    responses,
                    time.monotonic() - started_at,
                    args.context_window,
                )
                print_response(content, metrics)
                event_logger.log(
                    "agent_loop_completed",
                    turns=turn_number,
                    metrics=metrics,
                    history_message_count=len(messages),
                )
                return 0

            serialized_calls = [_tool_call_as_message(call) for call in tool_calls]
            call_ids = [call["id"] for call in serialized_calls]
            protocol_error = None
            if any(not isinstance(call_id, str) or not call_id for call_id in call_ids):
                protocol_error = "model returned a tool call without a valid call_id"
            elif len(set(call_ids)) != len(call_ids):
                protocol_error = "model returned duplicate tool call_ids"
            elif any(call["type"] != "function" for call in serialized_calls):
                protocol_error = "model returned an unsupported tool call type"
            elif any(
                not isinstance(call["function"]["name"], str)
                or not call["function"]["name"]
                or not isinstance(call["function"]["arguments"], str)
                for call in serialized_calls
            ):
                protocol_error = "model returned a malformed function call"
            if protocol_error:
                event_logger.log(
                    "agent_decision",
                    turn=turn_number,
                    decision="tool_protocol_error",
                    message=protocol_error,
                )
                raise ToolProtocolError(protocol_error)

            assistant_message = {
                "role": "assistant",
                "content": content or None,
                "tool_calls": serialized_calls,
            }
            if reasoning is not None:
                assistant_message["reasoning"] = reasoning
            messages.append(assistant_message)
            event_logger.log(
                "agent_decision",
                turn=turn_number,
                decision="execute_tools",
                call_ids=call_ids,
                tool_names=[call["function"]["name"] for call in serialized_calls],
            )

            for call in serialized_calls:
                tool_timeout = _remaining_seconds(deadline)
                call_id = call["id"]
                name = call["function"]["name"]
                raw_arguments = call["function"]["arguments"]
                event_logger.log(
                    "tool_call_started",
                    turn=turn_number,
                    call_id=call_id,
                    tool=name,
                    raw_arguments=raw_arguments,
                )
                tool_started_at = time.monotonic()
                result = execute_tool(
                    name,
                    raw_arguments,
                    timeout_seconds=tool_timeout,
                )
                elapsed_ms = round(
                    (time.monotonic() - tool_started_at) * 1_000, 2
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                event_logger.log(
                    "tool_call_completed",
                    turn=turn_number,
                    call_id=call_id,
                    tool=name,
                    success=result["ok"],
                    result=result,
                    latency_ms=elapsed_ms,
                    history_message_count=len(messages),
                )
                _remaining_seconds(deadline)

        _remaining_seconds(deadline)
        event_logger.log(
            "agent_decision",
            turn=max_turns,
            decision="max_turns_exceeded",
            max_turns=max_turns,
        )
        raise MaxTurnsExceededError(
            f"agent did not produce a final response within {max_turns} model turns"
        )
    except (Exception, KeyboardInterrupt) as exc:
        del messages[history_start:]
        event_logger.log(
            "agent_loop_failed",
            error_type=type(exc).__name__,
            message=str(exc),
            latency_ms=round((time.monotonic() - started_at) * 1_000, 2),
            history_rolled_back=True,
        )
        raise


def clear_conversation_context(
    messages: list[dict[str, Any]],
    event_logger: JsonlEventLogger,
) -> int:
    """Remove conversation turns while preserving the initial system prompt."""
    retained_messages = (
        messages[:1] if messages and messages[0].get("role") == "system" else []
    )
    removed_count = len(messages) - len(retained_messages)
    messages[:] = retained_messages
    event_logger.log(
        "conversation_context_cleared",
        removed_message_count=removed_count,
        history_message_count=len(messages),
    )
    return removed_count


def interactive_cli(
    client: Cerebras,
    messages: list[dict[str, Any]],
    args: argparse.Namespace,
    event_logger: JsonlEventLogger,
) -> int:
    """Read questions until the user exits, retaining successful turns."""
    if not sys.stdin.isatty():
        print(
            "Error: provide a prompt when standard input is not interactive.",
            file=sys.stderr,
        )
        return 2

    print("Harness Stage 3. Type /help for help or /exit to quit.")
    while True:
        try:
            prompt = input("\nyou> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("\nUse /exit or Ctrl-D to quit.")
            continue

        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            return 0
        if prompt == "/help":
            print("Ask a question. Commands: /clear, /help, /exit, /quit")
            continue
        if prompt == "/clear":
            removed_count = clear_conversation_context(messages, event_logger)
            print(
                "Conversation context cleared "
                f"({removed_count} message{'s' if removed_count != 1 else ''} removed)."
            )
            continue

        try:
            run_turn(client, messages, prompt, args, event_logger)
        except KeyboardInterrupt:
            print("\nRequest interrupted.", file=sys.stderr)
        except Exception as exc:
            print(f"Error communicating with Cerebras: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)
    event_logger = JsonlEventLogger(args.log_file)
    event_logger.log(
        "session_started",
        model=args.model,
        context_window_tokens=args.context_window,
        max_turns=args.max_turns,
        timeout_seconds=args.timeout,
        available_tools=sorted(TOOL_HANDLERS),
        interactive=args.prompt is None,
    )

    exit_code = 1
    try:
        if not os.environ.get("CEREBRAS_API_KEY"):
            message = "CEREBRAS_API_KEY is missing from the environment or .env file."
            event_logger.log("configuration_error", message=message)
            print(f"Error: {message}", file=sys.stderr)
            exit_code = 2
        else:
            client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])
            messages = [{"role": "system", "content": args.system_prompt}]
            if args.prompt is None:
                exit_code = interactive_cli(client, messages, args, event_logger)
            else:
                try:
                    exit_code = run_turn(
                        client, messages, args.prompt, args, event_logger
                    )
                except KeyboardInterrupt:
                    print("\nRequest interrupted.", file=sys.stderr)
                    exit_code = 130
                except Exception as exc:
                    print(f"Error communicating with Cerebras: {exc}", file=sys.stderr)
                    exit_code = 1
        return exit_code
    finally:
        event_logger.log("session_ended", exit_code=exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
