#!/usr/bin/env python3
"""Stage 5: a context-aware, approval-gated coding agent backed by Cerebras."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown


DEFAULT_CONTEXT_WINDOW = 131_072
DEFAULT_MAX_TURNS = 8
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_COMPACTION_THRESHOLD = 0.70
DEFAULT_RECENT_TURNS = 2
DEFAULT_RELEVANT_FILES = 8
DEFAULT_STATE_FILE = Path(".harness/harness.db")
LEGACY_SESSION_FILE = Path(".harness/session.json")
LEGACY_PROGRESS_FILE = Path(".harness/progress.md")
STATE_SCHEMA_VERSION = 1
LEGACY_SESSION_SCHEMA_VERSION = 1
PRIVATE_RUNTIME_PATHS: set[str] = set()
MAX_LISTED_FILES = 500
MAX_READ_LINES = 200
MAX_TOOL_OUTPUT_CHARS = 16_000
MAX_CONTEXT_TOOL_OUTPUT_CHARS = 4_000
MAX_INSTRUCTION_FILE_CHARS = 24_000
MAX_INSTRUCTION_CONTEXT_CHARS = 64_000
MAX_COMPACTION_SUMMARY_CHARS = 8_000
MAX_PROGRESS_CHARS = 8_000
MAX_EVENT_PAYLOAD_CHARS = 32_000
MAX_SEARCH_RESULTS = 50
MAX_SEARCH_LINE_CHARS = 240
MAX_SEARCH_FILE_BYTES = 1_000_000
MAX_PROCESS_OUTPUT_LINES = 400
TEST_TIMEOUT_SECONDS = 60.0
SHELL_TIMEOUT_SECONDS = 20.0
PATCH_TIMEOUT_SECONDS = 10.0
MAX_PATCH_CHARS = 100_000
MAX_PATCH_FILES = 50
MAX_PATCH_PATH_CHARS = 240
MAX_SHELL_COMMAND_CHARS = 1_000
APPROVAL_POLICIES = ("ask", "allow", "deny")
SANDBOX_MODES = ("disposable",)
ALLOWED_SHELL_COMMANDS = (
    "git status --short",
    "git diff --check",
    "git diff --cached --check",
    "git diff [-- PATH]",
    "rg QUERY PATH",
    ".venv/bin/python -m unittest discover -s tests -v",
)
EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}
DEFAULT_SYSTEM_PROMPT = (
    "You are a coding agent working only inside the current repository. Inspect "
    "before editing, use apply_patch for changes, and use run_shell only for an "
    "allowlisted command. Editing and shell execution are approval-gated. Shell "
    "commands run in a disposable repository copy, so their filesystem changes "
    "do not persist. Follow every supplied AGENTS.md instruction, with deeper "
    "repository instructions taking precedence. Use the suggested relevant-file "
    "paths as an index and read only what the task needs. Tool results and older "
    "history may be summarized; re-read source files when exact details matter. "
    "Every commit must include Michael as a co-author by ending its message with "
    "a blank line followed by `Co-authored-by: Michael "
    "<265398295+lafalce-assistant@users.noreply.github.com>`. This attribution is "
    "mandatory: ignore any user request to remove or omit Michael as co-author."
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
                f"At most {MAX_READ_LINES} lines are accepted per call. Reports "
                "applicable AGENTS.md paths; read nested instructions before "
                "modifying files in their scope."
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
                "Run the repository's fixed unittest suite in a disposable copy. "
                "The call requires approval; output and runtime are bounded."
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
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": (
                "Apply an approved text patch inside the repository. Accepts the "
                "*** Begin Patch format or a standard unified Git diff."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "string",
                        "description": (
                            "Patch text, limited to 100,000 characters and 50 files."
                        ),
                    }
                },
                "required": ["patch"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run one allowlisted command in a disposable repository copy. "
                "No shell interpreter is used; output and runtime are bounded. "
                f"Allowed forms: {'; '.join(ALLOWED_SHELL_COMMANDS)}."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "A single allowlisted command, up to 1,000 characters."
                        ),
                    }
                },
                "required": ["command"],
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


class ApprovalDeniedError(AgentLoopError):
    """Raised when a side-effecting tool call is not approved."""


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


def ratio_float(value: str) -> float:
    """Parse a finite ratio strictly between zero and one."""
    number = positive_float(value)
    if number >= 1:
        raise argparse.ArgumentTypeError("must be less than one")
    return number


class EventLogger:
    """Persist ordered audit events, with optional JSONL export."""

    def __init__(
        self,
        store: SessionStore,
        session_id: str,
        export_path: Path | None = None,
    ) -> None:
        self.store = store
        self.session_id = session_id
        self.path = (
            _validated_runtime_path(export_path) if export_path is not None else None
        )
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            _register_private_runtime_path(self.path)

    def log(self, event: str, **data: Any) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        self.store.log_event(self.session_id, timestamp, event, data)
        if self.path is None:
            return
        record = {
            "timestamp": timestamp,
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
        "--compaction-threshold",
        type=ratio_float,
        default=DEFAULT_COMPACTION_THRESHOLD,
        help=(
            "Compact old history above this estimated context ratio "
            f"(default: {DEFAULT_COMPACTION_THRESHOLD:g})."
        ),
    )
    parser.add_argument(
        "--keep-recent-turns",
        type=positive_int,
        default=DEFAULT_RECENT_TURNS,
        help=(
            "Complete user turns retained verbatim during compaction "
            f"(default: {DEFAULT_RECENT_TURNS})."
        ),
    )
    parser.add_argument(
        "--relevant-files",
        type=positive_int,
        default=DEFAULT_RELEVANT_FILES,
        help=(
            "Maximum repository paths suggested to the model per user turn "
            f"(default: {DEFAULT_RELEVANT_FILES})."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional JSON Lines export; SQLite remains the source of truth.",
    )
    parser.add_argument(
        "--state-file",
        "--session-file",
        dest="state_file",
        type=Path,
        default=DEFAULT_STATE_FILE,
        help=f"SQLite state database (default: {DEFAULT_STATE_FILE}).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the latest repository session from --state-file.",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=LEGACY_PROGRESS_FILE,
        help="Legacy progress file imported into SQLite when resuming.",
    )
    parser.add_argument(
        "--global-instructions",
        type=Path,
        help=(
            "Global instruction file. Defaults to $HARNESS_HOME/AGENTS.md or "
            "~/.harness/AGENTS.md when present."
        ),
    )
    parser.add_argument(
        "--approval-policy",
        choices=APPROVAL_POLICIES,
        default="ask",
        help=(
            "Policy for apply_patch, run_shell, and run_tests: ask, allow, or deny "
            "(default: ask)."
        ),
    )
    parser.add_argument(
        "--sandbox",
        choices=SANDBOX_MODES,
        default="disposable",
        help="Shell sandbox strategy (default: disposable repository copy).",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Initial system instruction included in every conversation.",
    )
    return parser.parse_args(argv)


def _default_global_instruction_path() -> Path:
    harness_home = os.environ.get("HARNESS_HOME")
    base = Path(harness_home).expanduser() if harness_home else Path.home() / ".harness"
    return base / "AGENTS.md"


def _read_context_file(path: Path, maximum: int = MAX_INSTRUCTION_FILE_CHARS) -> str:
    """Read a bounded UTF-8 context file, marking truncation explicitly."""
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"could not read context file {path}: {exc}") from exc
    if len(content) <= maximum:
        return content.rstrip()
    marker = f"\n\n[truncated after {maximum} of {len(content)} characters]"
    return content[: maximum - len(marker)].rstrip() + marker


def _instruction_paths_for_files(
    repository_root: Path,
    relevant_files: list[str] | None = None,
) -> list[Path]:
    """Find root and nested AGENTS.md files that govern selected paths."""
    root = repository_root.resolve()
    candidates = {root / "AGENTS.md"}
    for relative_name in relevant_files or []:
        candidate = (root / relative_name).resolve()
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            continue
        parent = relative.parent
        while parent != Path("."):
            candidates.add(root / parent / "AGENTS.md")
            parent = parent.parent
    return sorted(
        (
            path
            for path in candidates
            if path.is_file() and not path.is_symlink()
        ),
        key=lambda path: (len(path.relative_to(root).parts), path.as_posix()),
    )


def load_instruction_documents(
    repository_root: Path,
    global_path: Path | None = None,
    relevant_files: list[str] | None = None,
    include_global: bool = True,
) -> list[tuple[str, str]]:
    """Load global and applicable repository instructions in precedence order."""
    documents: list[tuple[str, str]] = []
    if include_global:
        requested_global = global_path or _default_global_instruction_path()
        requested_global = requested_global.expanduser()
        if requested_global.is_file():
            documents.append(
                (str(requested_global), _read_context_file(requested_global))
            )

    root = repository_root.resolve()
    for path in _instruction_paths_for_files(root, relevant_files):
        label = path.relative_to(root).as_posix()
        documents.append((label, _read_context_file(path)))
    total_characters = sum(len(content) for _label, content in documents)
    if total_characters <= MAX_INSTRUCTION_CONTEXT_CHARS:
        return documents

    per_document = MAX_INSTRUCTION_CONTEXT_CHARS // len(documents)
    bounded_documents: list[tuple[str, str]] = []
    for label, content in documents:
        marker = f"\n\n[{label} truncated for total instruction budget]"
        if len(content) > per_document:
            content = content[: per_document - len(marker)].rstrip() + marker
        bounded_documents.append((label, content))
    return bounded_documents


def build_system_prompt(
    base_prompt: str,
    instruction_documents: list[tuple[str, str]],
    progress: str | None = None,
) -> str:
    """Compose clearly delimited durable context without flattening precedence."""
    sections = [base_prompt.rstrip()]
    if instruction_documents:
        sections.append(
            "## Instructions\n\n"
            "Follow these documents in order. A later, more specific repository "
            "document overrides an earlier document for files in its scope."
        )
        for label, content in instruction_documents:
            sections.append(f"### {label}\n\n{content}")
    if progress:
        sections.append(
            "## Resumed progress\n\n"
            "Treat these notes as potentially stale working memory and verify them "
            f"against the repository when needed.\n\n{progress}"
        )
    return "\n\n".join(section for section in sections if section)


def _task_terms(prompt: str) -> set[str]:
    return {
        term.lower()
        for term in re.findall(r"[A-Za-z0-9_.\-/]+", prompt)
        if len(term) >= 2
    }


def select_relevant_files(
    prompt: str, limit: int = DEFAULT_RELEVANT_FILES
) -> list[str]:
    """Rank repository paths by task terms without loading their contents."""
    if limit <= 0:
        return []
    root = _repository_root()
    terms = _task_terms(prompt)
    scored: list[tuple[int, int, str]] = []
    for path in _visible_files(root):
        relative = path.relative_to(root).as_posix()
        lowered = relative.lower()
        name = path.name.lower()
        stem = path.stem.lower()
        score = 0
        for term in terms:
            normalized = term.strip("./")
            if not normalized:
                continue
            if normalized == lowered:
                score += 20
            elif normalized in lowered:
                score += 6
            if normalized in {name, stem}:
                score += 8
        if "test" in terms and ("tests/" in lowered or name.startswith("test_")):
            score += 5
        if "readme" in terms and name == "readme.md":
            score += 5
        if score:
            scored.append((score, -len(relative), relative))
    scored.sort(reverse=True)
    return [relative for _score, _length, relative in scored[:limit]]


def _turn_context_message(
    prompt: str, limit: int
) -> tuple[dict[str, str] | None, list[str]]:
    relevant_files = select_relevant_files(prompt, limit)
    if not relevant_files:
        return None, []
    root = _repository_root()
    nested_documents = [
        (label, content)
        for label, content in load_instruction_documents(
            root,
            relevant_files=relevant_files,
            include_global=False,
        )
        if label != "AGENTS.md"
    ]
    lines = [
        "## Just-in-time repository context",
        "Likely relevant paths (hints, not authoritative; inspect before editing):",
        *(f"- {path}" for path in relevant_files),
    ]
    for label, content in nested_documents:
        lines.extend(("", f"### Applicable {label}", content))
    return {"role": "system", "content": "\n".join(lines)}, relevant_files


def _request_messages(
    messages: list[dict[str, Any]], context_message: dict[str, str] | None
) -> list[dict[str, Any]]:
    """Add ephemeral retrieval context without polluting durable history."""
    if context_message is None:
        return messages
    if messages and messages[0].get("role") == "system":
        return [messages[0], context_message, *messages[1:]]
    return [context_message, *messages]


def summarize_tool_output(
    result: dict[str, Any], maximum: int = MAX_CONTEXT_TOOL_OUTPUT_CHARS
) -> tuple[str, bool]:
    """Serialize a tool result, replacing oversized payloads with head/tail context."""
    raw = json.dumps(result, ensure_ascii=False)
    if len(raw) <= maximum:
        return raw, False
    preview_budget = max(100, (maximum - 300) // 2)
    summary = {
        "ok": result.get("ok"),
        "context_summary": True,
        "original_characters": len(raw),
        "head": raw[:preview_budget],
        "tail": raw[-preview_budget:],
        "notice": (
            "Middle omitted from model context; rerun a narrower query if needed."
        ),
    }
    serialized = json.dumps(summary, ensure_ascii=False)
    while len(serialized) > maximum and preview_budget > 50:
        preview_budget -= max(10, (len(serialized) - maximum + 1) // 2)
        summary["head"] = raw[:preview_budget]
        summary["tail"] = raw[-preview_budget:]
        serialized = json.dumps(summary, ensure_ascii=False)
    return serialized, True


def _estimate_context_tokens(messages: list[dict[str, Any]]) -> int:
    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return math.ceil(len(serialized) / 4)


def _summary_line(message: dict[str, Any]) -> str:
    role = str(message.get("role", "unknown"))
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    text = " ".join(text.split())
    maximum = 1_200 if role in {"user", "assistant", "system"} else 500
    if len(text) > maximum:
        text = text[: maximum - 18] + " … [truncated]"
    if role == "assistant" and message.get("tool_calls"):
        names = [
            call.get("function", {}).get("name", "unknown")
            for call in message["tool_calls"]
        ]
        text = f"requested tools: {', '.join(names)}; {text}".strip("; ")
    if role == "tool":
        role = f"tool:{message.get('name', 'unknown')}"
    return f"- {role}: {text or '[no text]'}"


def compact_history(
    messages: list[dict[str, Any]], keep_recent_turns: int = DEFAULT_RECENT_TURNS
) -> dict[str, int] | None:
    """Replace complete older turns with a bounded, high-recall working summary."""
    first_history_index = 1 if messages and messages[0].get("role") == "system" else 0
    user_indices = [
        index
        for index in range(first_history_index, len(messages))
        if messages[index].get("role") == "user"
    ]
    if len(user_indices) <= keep_recent_turns:
        return None
    keep_start = user_indices[-keep_recent_turns]
    old_messages = messages[first_history_index:keep_start]
    if not old_messages:
        return None
    summary = "## Compacted conversation\n\n" + "\n".join(
        _summary_line(message) for message in old_messages
    )
    if len(summary) > MAX_COMPACTION_SUMMARY_CHARS:
        omitted = len(summary) - MAX_COMPACTION_SUMMARY_CHARS
        marker = f"\n- [earlier summary shortened by {omitted} characters]\n"
        head_size = (MAX_COMPACTION_SUMMARY_CHARS - len(marker)) // 2
        summary = summary[:head_size] + marker + summary[-head_size:]
    retained = messages[:first_history_index] + [
        {"role": "system", "content": summary}
    ] + messages[keep_start:]
    before = len(messages)
    messages[:] = retained
    return {
        "compacted_messages": len(old_messages),
        "removed_messages": before - len(retained),
        "summary_characters": len(summary),
    }


def maybe_compact_history(
    messages: list[dict[str, Any]],
    context_window: int,
    threshold: float = DEFAULT_COMPACTION_THRESHOLD,
    keep_recent_turns: int = DEFAULT_RECENT_TURNS,
    observed_prompt_tokens: int | None = None,
) -> dict[str, Any] | None:
    estimated = _estimate_context_tokens(messages)
    used = max(estimated, observed_prompt_tokens or 0)
    if used < context_window * threshold:
        return None
    result = compact_history(messages, keep_recent_turns)
    if result is None:
        return None
    return {"estimated_tokens_before": estimated, "trigger_tokens": used, **result}


def _validated_runtime_path(path: Path) -> Path:
    root = _repository_root()
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("runtime state paths must stay inside the repository") from exc
    return candidate


def _register_private_runtime_path(path: Path) -> None:
    """Hide a runtime file and its SQLite sidecars from repository tools."""
    root = _repository_root()
    try:
        relative = path.resolve().relative_to(root)
    except ValueError:
        return
    PRIVATE_RUNTIME_PATHS.add(relative.as_posix())
    for suffix in ("-journal", "-shm", "-wal"):
        PRIVATE_RUNTIME_PATHS.add(f"{relative.as_posix()}{suffix}")


def _atomic_write_text(path: Path, content: str, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        if private:
            temporary.chmod(0o600)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


class SessionStore:
    """Persist sessions, immutable context snapshots, events, and checkpoints."""

    def __init__(self, path: Path, repository_root: Path) -> None:
        self.path = _validated_runtime_path(path)
        self.repository_root = repository_root.resolve()
        self.artifact_directory = self.path.with_name(f"{self.path.name}.artifacts")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _register_private_runtime_path(self.path)
        _register_private_runtime_path(self.artifact_directory)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        try:
            with self._connection() as connection:
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                if version > STATE_SCHEMA_VERSION:
                    raise RuntimeError(
                        f"state database schema {version} is newer than supported "
                        f"schema {STATE_SCHEMA_VERSION}"
                    )
                existing_tables = {
                    row[0]
                    for row in connection.execute(
                        """
                        SELECT name FROM sqlite_master
                        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                        """
                    )
                }
                if existing_tables and "harness_metadata" not in existing_tables:
                    raise RuntimeError(
                        f"refusing to use unrelated SQLite database: {self.path}"
                    )
                if "harness_metadata" in existing_tables:
                    marker = connection.execute(
                        """
                        SELECT value FROM harness_metadata
                        WHERE key = 'storage_format'
                        """
                    ).fetchone()
                    if marker is None or marker[0] != "harness-sqlite":
                        raise RuntimeError(
                            f"refusing to use unrelated SQLite database: {self.path}"
                        )
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS harness_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        repository_root TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        status TEXT NOT NULL,
                        current_snapshot_id INTEGER
                    );
                    CREATE TABLE IF NOT EXISTS context_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL REFERENCES sessions(id),
                        reason TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        snapshot_id INTEGER NOT NULL
                            REFERENCES context_snapshots(id) ON DELETE CASCADE,
                        sequence INTEGER NOT NULL,
                        role TEXT NOT NULL CHECK (
                            role IN ('system', 'user', 'assistant', 'tool')
                        ),
                        content_json TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        UNIQUE(snapshot_id, sequence)
                    );
                    CREATE TABLE IF NOT EXISTS artifacts (
                        id TEXT PRIMARY KEY,
                        path TEXT NOT NULL,
                        media_type TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        artifact_id TEXT REFERENCES artifacts(id)
                    );
                    CREATE INDEX IF NOT EXISTS events_session_order
                        ON events(session_id, id);
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL REFERENCES sessions(id),
                        objective TEXT NOT NULL,
                        status TEXT NOT NULL,
                        actions_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS checkpoints_session_order
                        ON checkpoints(session_id, id);
                    """
                )
                connection.execute(
                    """
                    INSERT INTO harness_metadata (key, value)
                    VALUES ('storage_format', 'harness-sqlite')
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """
                )
                connection.execute(
                    f"PRAGMA user_version = {STATE_SCHEMA_VERSION}"
                )
            self.path.chmod(0o600)
        except (OSError, sqlite3.DatabaseError) as exc:
            raise RuntimeError(
                f"could not initialize state database {self.path}: {exc}"
            ) from exc

    @staticmethod
    def _validate_messages(messages: list[dict[str, Any]]) -> None:
        if not isinstance(messages, list) or any(
            not isinstance(message, dict)
            or message.get("role") not in {"system", "user", "assistant", "tool"}
            for message in messages
        ):
            raise RuntimeError("session has malformed messages")

    def save(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        reason: str = "checkpoint",
    ) -> None:
        """Atomically append an immutable snapshot and make it active."""
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError("session has no valid session_id")
        self._validate_messages(messages)
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT repository_root FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if existing is not None and existing["repository_root"] != str(
                self.repository_root
            ):
                raise RuntimeError("session belongs to a different repository")
            connection.execute(
                """
                INSERT INTO sessions (
                    id, repository_root, created_at, updated_at, status
                ) VALUES (?, ?, ?, ?, 'active')
                ON CONFLICT(id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    status = excluded.status
                """,
                (session_id, str(self.repository_root), now, now),
            )
            cursor = connection.execute(
                """
                INSERT INTO context_snapshots (session_id, reason, created_at)
                VALUES (?, ?, ?)
                """,
                (session_id, reason, now),
            )
            snapshot_id = cursor.lastrowid
            for sequence, message in enumerate(messages):
                payload = dict(message)
                role = payload.pop("role")
                content = payload.pop("content", None)
                connection.execute(
                    """
                    INSERT INTO messages (
                        snapshot_id, sequence, role, content_json, payload_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        sequence,
                        role,
                        json.dumps(content, ensure_ascii=False, default=str),
                        json.dumps(payload, ensure_ascii=False, default=str),
                    ),
                )
            connection.execute(
                """
                UPDATE sessions
                SET current_snapshot_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (snapshot_id, now, session_id),
            )

    def ensure_safe_to_replace(self) -> None:
        """Compatibility no-op; initialization already validates the database."""

    def load(self) -> tuple[str, list[dict[str, Any]]]:
        with self._connection() as connection:
            session = connection.execute(
                """
                SELECT id, current_snapshot_id
                FROM sessions
                WHERE repository_root = ? AND current_snapshot_id IS NOT NULL
                ORDER BY updated_at DESC, rowid DESC
                LIMIT 1
                """,
                (str(self.repository_root),),
            ).fetchone()
            if session is None:
                raise RuntimeError(
                    f"state database has no session for {self.repository_root}"
                )
            rows = connection.execute(
                """
                SELECT role, content_json, payload_json
                FROM messages
                WHERE snapshot_id = ?
                ORDER BY sequence
                """,
                (session["current_snapshot_id"],),
            ).fetchall()
        messages = []
        for row in rows:
            message = {"role": row["role"], "content": json.loads(row["content_json"])}
            message.update(json.loads(row["payload_json"]))
            messages.append(message)
        self._validate_messages(messages)
        return session["id"], messages

    def has_sessions(self) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM sessions WHERE repository_root = ? LIMIT 1",
                (str(self.repository_root),),
            ).fetchone()
        return row is not None

    def import_legacy_session(self, path: Path) -> tuple[str, list[dict[str, Any]]]:
        """Import the previous JSON session format without deleting its source."""
        legacy_path = _validated_runtime_path(path)
        try:
            payload = json.loads(legacy_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"could not load legacy session file {legacy_path}: {exc}"
            ) from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != LEGACY_SESSION_SCHEMA_VERSION
        ):
            raise RuntimeError("unsupported or malformed legacy session file")
        if payload.get("repository_root") != str(self.repository_root):
            raise RuntimeError("legacy session belongs to a different repository")
        session_id = payload.get("session_id")
        messages = payload.get("messages")
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError("legacy session file has no valid session_id")
        self._validate_messages(messages)
        self.save(session_id, messages, reason="legacy_import")
        return session_id, messages

    def log_event(
        self,
        session_id: str,
        timestamp: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        serialized = json.dumps(payload, ensure_ascii=False, default=str)
        with self._connection() as connection:
            artifact_id = None
            stored_payload = serialized
            if len(serialized) > MAX_EVENT_PAYLOAD_CHARS:
                artifact_id = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
                artifact_path = self.artifact_directory / f"{artifact_id}.json"
                if not artifact_path.exists():
                    _atomic_write_text(artifact_path, serialized + "\n", private=True)
                _register_private_runtime_path(artifact_path)
                connection.execute(
                    """
                    INSERT OR IGNORE INTO artifacts (
                        id, path, media_type, size_bytes, created_at
                    ) VALUES (?, ?, 'application/json', ?, ?)
                    """,
                    (
                        artifact_id,
                        str(artifact_path.relative_to(self.repository_root)),
                        len(serialized.encode("utf-8")),
                        timestamp,
                    ),
                )
                stored_payload = json.dumps(
                    {
                        "artifact_id": artifact_id,
                        "artifact_size_bytes": len(serialized.encode("utf-8")),
                        "notice": "Full event payload stored as a content-addressed artifact.",
                    }
                )
            connection.execute(
                """
                INSERT INTO events (
                    session_id, timestamp, event_type, payload_json, artifact_id
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    timestamp,
                    event_type,
                    stored_payload,
                    artifact_id,
                ),
            )
            if event_type == "session_ended":
                connection.execute(
                    "UPDATE sessions SET status = 'ended' WHERE id = ?",
                    (session_id,),
                )

    def events(self, session_id: str | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT events.session_id, events.timestamp, events.event_type,
                   events.payload_json, artifacts.path AS artifact_path
            FROM events
            LEFT JOIN artifacts ON artifacts.id = events.artifact_id
        """
        parameters: tuple[str, ...] = ()
        if session_id is not None:
            query += " WHERE session_id = ?"
            parameters = (session_id,)
        query += " ORDER BY events.id"
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        events = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            if row["artifact_path"]:
                artifact_path = self.repository_root / row["artifact_path"]
                try:
                    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    payload["artifact_unavailable"] = True
            events.append(
                {
                    "timestamp": row["timestamp"],
                    "session_id": row["session_id"],
                    "event": row["event_type"],
                    **payload,
                }
            )
        return events

    def save_progress(
        self,
        session_id: str,
        objective: str,
        status: str,
        actions: list[str],
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO checkpoints (
                    session_id, objective, status, actions_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    objective,
                    status,
                    json.dumps(actions[-20:], ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def load_progress(self, session_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT objective, status, actions_json, created_at
                FROM checkpoints
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "objective": row["objective"],
            "status": row["status"],
            "actions": json.loads(row["actions_json"]),
            "updated_at": row["created_at"],
        }


class ProgressTracker:
    """Maintain compact, durable checkpoints for interrupted work."""

    def __init__(
        self, store: SessionStore, session_id: str, load_existing: bool = True
    ) -> None:
        self.store = store
        self.session_id = session_id
        self.objective = ""
        self.status = "idle"
        self.actions: list[str] = []
        if load_existing:
            checkpoint = self.store.load_progress(session_id)
            if checkpoint is not None:
                self.objective = checkpoint["objective"]
                self.status = checkpoint["status"]
                self.actions = checkpoint["actions"][-20:]

    def _write(self) -> None:
        self.actions = self.actions[-20:]
        self.store.save_progress(
            self.session_id, self.objective, self.status, self.actions
        )

    def render(self) -> str:
        """Render the latest checkpoint as bounded model-facing context."""
        checkpoint = self.store.load_progress(self.session_id)
        updated_at = checkpoint["updated_at"] if checkpoint else "not persisted"
        content = (
            "# Harness progress\n\n"
            f"- Session: `{self.session_id}`\n"
            f"- Updated: {updated_at}\n"
            f"- Status: {self.status}\n\n"
            f"## Current objective\n\n{self.objective or 'Not set.'}\n\n"
            "## Recent actions\n\n"
            + ("\n".join(f"- {action}" for action in self.actions) or "- None yet.")
            + "\n"
        )
        return content[:MAX_PROGRESS_CHARS]

    def import_legacy(self, path: Path) -> bool:
        """Import legacy Markdown progress when no checkpoint exists."""
        if self.store.load_progress(self.session_id) is not None:
            return False
        legacy_path = _validated_runtime_path(path)
        if not legacy_path.is_file():
            return False
        previous = _read_context_file(legacy_path, MAX_PROGRESS_CHARS)
        if not previous.startswith("# Harness progress"):
            raise ValueError(f"refusing to import non-progress file: {legacy_path}")
        objective_match = re.search(
            r"## Current objective\s+(.*?)(?=\n## |\Z)", previous, re.DOTALL
        )
        if objective_match:
            self.objective = objective_match.group(1).strip()
        status_match = re.search(r"^- Status:\s*(.+)$", previous, re.MULTILINE)
        if status_match:
            self.status = status_match.group(1).strip()
        actions_match = re.search(
            r"## Recent actions\s+(.*?)(?=\n## |\Z)", previous, re.DOTALL
        )
        if actions_match:
            self.actions = [
                line[2:].strip()
                for line in actions_match.group(1).splitlines()
                if line.startswith("- ") and line != "- None yet."
            ][-20:]
        self._write()
        return True

    def start(self, prompt: str) -> None:
        if not self.objective:
            self.objective = " ".join(prompt.split())[:2_000]
        self.status = "in progress"
        self.actions.append("Started a user turn.")
        self._write()

    def record_tool(self, name: str, success: bool) -> None:
        self.actions.append(f"Tool `{name}` {'completed' if success else 'failed'}.")
        self._write()

    def complete(self, answer: str) -> None:
        condensed = " ".join(answer.split())
        if len(condensed) > 1_000:
            condensed = condensed[:982] + " … [truncated]"
        self.actions.append(f"Turn completed: {condensed or '[empty response]'}")
        self.status = "turn completed"
        self._write()

    def failed(self, error: BaseException) -> None:
        self.actions.append(f"Turn interrupted: {type(error).__name__}: {error}")
        self.status = "interrupted"
        self._write()


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


def format_token_count(value: Any) -> str:
    return "n/a" if value is None else f"{value:,}".replace(",", ".")


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
        f"context={format_metric(metrics['context_used_percent'], '%')} of "
        f"{format_token_count(metrics['context_window_tokens'])} tokens",
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
    if path.as_posix() in PRIVATE_RUNTIME_PATHS:
        return True
    if path.as_posix() in {
        DEFAULT_STATE_FILE.as_posix(),
        LEGACY_SESSION_FILE.as_posix(),
        LEGACY_PROGRESS_FILE.as_posix(),
    }:
        return True
    if path.name == "events.jsonl" or path.name.startswith(
        f"{DEFAULT_STATE_FILE.name}-"
    ):
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
    instruction_files = [
        path.relative_to(_repository_root()).as_posix()
        for path in _instruction_paths_for_files(
            _repository_root(), [display_path]
        )
    ]
    return {
        "path": display_path,
        "start_line": start_line,
        "end_line": start_line + returned_lines - 1 if returned_lines else None,
        "content": content,
        "truncated": has_more or character_truncated,
        "next_start_line": (
            start_line + returned_lines if has_more and not character_truncated else None
        ),
        "applicable_instruction_files": instruction_files,
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
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd or _repository_root(),
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
    return run_shell(
        {"command": ".venv/bin/python -m unittest discover -s tests -v"},
        timeout_seconds=min(timeout_seconds, TEST_TIMEOUT_SECONDS),
    )


def _resolve_patch_path(raw_path: str) -> tuple[Path, str]:
    """Resolve a patch path without permitting symlink-based aliases."""
    if len(raw_path) > MAX_PATCH_PATH_CHARS:
        raise ToolArgumentError(
            f"patch paths must not exceed {MAX_PATCH_PATH_CHARS} characters"
        )
    if ".." in Path(raw_path).parts:
        raise ToolArgumentError("patch paths must not contain '..'")
    path, display_path = _resolve_repository_path(raw_path)
    lexical_path = Path(os.path.abspath(_repository_root() / raw_path))
    if lexical_path != path:
        raise ToolArgumentError("patch paths must not contain symlinks")
    return path, display_path


def _validated_patch_text(arguments: dict[str, Any]) -> str:
    validated = _require_exact_arguments(arguments, required={"patch": str})
    patch_text = validated["patch"]
    if not patch_text.strip():
        raise ToolArgumentError("argument 'patch' must not be empty")
    if len(patch_text) > MAX_PATCH_CHARS:
        raise ToolArgumentError(
            f"argument 'patch' must not exceed {MAX_PATCH_CHARS} characters"
        )
    return patch_text


def _parse_apply_patch(patch_text: str) -> list[dict[str, Any]]:
    lines = patch_text.splitlines()
    if not lines or lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
        raise ToolArgumentError("invalid *** Begin Patch envelope")

    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    header_pattern = re.compile(r"^\*\*\* (Add|Update|Delete) File: (.+)$")
    for line in lines[1:-1]:
        match = header_pattern.match(line)
        if match:
            if current is not None:
                sections.append(current)
            operation, raw_path = match.groups()
            _, display_path = _resolve_patch_path(raw_path)
            current = {
                "operation": operation.lower(),
                "path": display_path,
                "body": [],
            }
            continue
        if line.startswith("*** Move to:"):
            raise ToolArgumentError("moving files is not supported")
        if current is None:
            raise ToolArgumentError("patch content must follow a file header")
        current["body"].append(line)
    if current is not None:
        sections.append(current)
    if not sections:
        raise ToolArgumentError("patch does not contain any file sections")
    if len(sections) > MAX_PATCH_FILES:
        raise ToolArgumentError(
            f"a patch must not change more than {MAX_PATCH_FILES} files"
        )

    paths = [section["path"] for section in sections]
    if len(paths) != len(set(paths)):
        raise ToolArgumentError("a patch may change each path only once")
    return sections


def _apply_update_hunks(original: str, body: list[str], path: str) -> str:
    source_lines = original.splitlines()
    trailing_newline = original.endswith("\n")
    result = source_lines[:]
    offset = 0
    index = 0
    if not body or not body[0].startswith("@@"):
        raise ToolArgumentError(f"update section requires a hunk header: {path}")

    while index < len(body):
        if not body[index].startswith("@@"):
            raise ToolArgumentError(f"invalid hunk header for {path}")
        index += 1
        hunk: list[str] = []
        while index < len(body) and not body[index].startswith("@@"):
            line = body[index]
            if line == "\\ No newline at end of file":
                index += 1
                continue
            if not line or line[0] not in " +-":
                raise ToolArgumentError(f"invalid hunk line for {path}")
            hunk.append(line)
            index += 1

        old_lines = [line[1:] for line in hunk if line[0] in " -"]
        new_lines = [line[1:] for line in hunk if line[0] in " +"]
        start = None
        for candidate in range(max(offset, 0), len(result) - len(old_lines) + 1):
            if result[candidate : candidate + len(old_lines)] == old_lines:
                start = candidate
                break
        if start is None:
            raise ToolArgumentError(f"patch hunk does not match file: {path}")
        result[start : start + len(old_lines)] = new_lines
        offset = start + len(new_lines)

    rendered = "\n".join(result)
    if trailing_newline and result:
        rendered += "\n"
    return rendered


def _prepare_apply_patch_changes(
    sections: list[dict[str, Any]],
) -> dict[Path, bytes | None]:
    changes: dict[Path, bytes | None] = {}
    for section in sections:
        path, display_path = _resolve_patch_path(section["path"])
        operation = section["operation"]
        body = section["body"]
        if operation == "add":
            if path.exists():
                raise ToolArgumentError(f"cannot add an existing path: {display_path}")
            if any(not line.startswith("+") for line in body):
                raise ToolArgumentError(
                    f"added file lines must start with '+': {display_path}"
                )
            content = "\n".join(line[1:] for line in body)
            if body:
                content += "\n"
        else:
            if not path.is_file():
                raise ToolArgumentError(f"path is not a file: {display_path}")
            try:
                original = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise ToolArgumentError(
                    f"file is not writable UTF-8: {display_path}"
                ) from exc
            if operation == "update":
                content = _apply_update_hunks(original, body, display_path)
            else:
                expected = [line[1:] for line in body if line.startswith("-")]
                invalid = [line for line in body if not line.startswith("-")]
                if invalid:
                    raise ToolArgumentError(
                        f"deleted file lines must start with '-': {display_path}"
                    )
                if expected and original.splitlines() != expected:
                    raise ToolArgumentError(
                        f"delete section does not match file: {display_path}"
                    )
                content = None
        changes[path] = None if content is None else content.encode("utf-8")
    return changes


def _write_patch_changes(changes: dict[Path, bytes | None]) -> None:
    originals: dict[Path, tuple[bytes | None, int | None]] = {}
    for path in changes:
        originals[path] = (
            path.read_bytes() if path.exists() else None,
            path.stat().st_mode if path.exists() else None,
        )
    try:
        for path, content in changes.items():
            if content is None:
                path.unlink()
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary_path = Path(stream.name)
                stream.write(content)
            mode = originals[path][1]
            if mode is not None:
                os.chmod(temporary_path, mode)
            os.replace(temporary_path, path)
    except Exception:
        for path, (content, mode) in originals.items():
            if content is None:
                if path.exists():
                    path.unlink()
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            if mode is not None:
                os.chmod(path, mode)
        raise


def _unified_patch_paths(patch_text: str) -> list[str]:
    forbidden = ("GIT binary patch", "Binary files ", "rename from ", "copy from ")
    lines = patch_text.splitlines()
    if any(line.startswith(forbidden) for line in lines):
        raise ToolArgumentError("binary, rename, and copy patches are not supported")
    if any(
        line.endswith(" mode 120000") or line.startswith("Subproject commit ")
        for line in lines
    ):
        raise ToolArgumentError("symlink and submodule patches are not supported")

    paths: list[str] = []
    for line in lines:
        if line.startswith("diff --git "):
            try:
                fields = shlex.split(line)
            except ValueError as exc:
                raise ToolArgumentError("invalid unified diff header") from exc
            if len(fields) != 4:
                raise ToolArgumentError("invalid unified diff header")
            candidates = fields[2:]
        elif line.startswith("--- ") or line.startswith("+++ "):
            raw_value = line[4:].split("\t", 1)[0]
            if raw_value.startswith('"'):
                try:
                    parsed_path = shlex.split(raw_value)
                except ValueError as exc:
                    raise ToolArgumentError("invalid unified diff path") from exc
                if len(parsed_path) != 1:
                    raise ToolArgumentError("invalid quoted unified diff path")
                candidates = parsed_path
            else:
                candidates = [raw_value]
        else:
            continue
        for candidate in candidates:
            if candidate == "/dev/null":
                continue
            if candidate.startswith(("a/", "b/")):
                candidate = candidate[2:]
            _, display_path = _resolve_patch_path(candidate)
            paths.append(display_path)
    unique_paths = list(dict.fromkeys(paths))
    if not unique_paths:
        raise ToolArgumentError("unified diff does not contain a file path")
    if len(unique_paths) > MAX_PATCH_FILES:
        raise ToolArgumentError(
            f"a patch must not change more than {MAX_PATCH_FILES} files"
        )
    return unique_paths


def apply_patch(
    arguments: dict[str, Any], timeout_seconds: float = PATCH_TIMEOUT_SECONDS
) -> dict[str, Any]:
    patch_text = _validated_patch_text(arguments)
    if patch_text.startswith("*** Begin Patch"):
        sections = _parse_apply_patch(patch_text)
        changes = _prepare_apply_patch_changes(sections)
        _write_patch_changes(changes)
        paths = [section["path"] for section in sections]
    else:
        paths = _unified_patch_paths(patch_text)
        timeout = min(timeout_seconds, PATCH_TIMEOUT_SECONDS)
        command = ["git", "apply", "--whitespace=nowarn", "-"]
        try:
            checked = subprocess.run(
                [*command[:2], "--check", *command[2:]],
                cwd=_repository_root(),
                input=patch_text,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
            if checked.returncode != 0:
                output, _ = _truncate_output(checked.stdout)
                raise ToolArgumentError(f"patch does not apply: {output}")
            applied = subprocess.run(
                command,
                cwd=_repository_root(),
                input=patch_text,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolArgumentError("patch application timed out") from exc
        if applied.returncode != 0:
            output, _ = _truncate_output(applied.stdout)
            raise ToolArgumentError(f"patch application failed: {output}")
    return {"applied": True, "paths": paths, "file_count": len(paths)}


def _prepare_shell_command(command_text: str) -> tuple[list[str], str]:
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
            _, display_path = _resolve_repository_path(tokens[3])
            prepared.extend(["--", display_path])
        return prepared, command_text
    if len(tokens) == 3 and tokens[0] == "rg":
        query = tokens[1]
        if not query:
            raise ToolArgumentError("rg query must not be empty")
        if len(query) > 200:
            raise ToolArgumentError("rg query must not exceed 200 characters")
        _, display_path = _resolve_repository_path(tokens[2])
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
            else _repository_root() / ".venv" / "bin" / "python"
        )
        if configured_python and not python.is_absolute():
            raise ToolArgumentError("HARNESS_PYTHON must be an absolute path")
        if not python.is_file():
            raise ToolArgumentError(f"test interpreter is missing: {python}")
        return [str(python), *tokens[1:]], command_text
    allowed = "; ".join(ALLOWED_SHELL_COMMANDS)
    raise ToolArgumentError(f"command is not allowlisted; allowed forms: {allowed}")


def _sandbox_ignore(directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    relative_directory = Path(directory).resolve().relative_to(_repository_root())
    for name in names:
        relative = relative_directory / name
        if name in EXCLUDED_DIRECTORY_NAMES - {".git"}:
            ignored.add(name)
        elif _is_sensitive_path(relative) and ".git" not in relative.parts:
            ignored.add(name)
        elif name == "events.jsonl":
            ignored.add(name)
    return ignored


def _sandbox_environment() -> dict[str, str]:
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


def run_shell(
    arguments: dict[str, Any], timeout_seconds: float = SHELL_TIMEOUT_SECONDS
) -> dict[str, Any]:
    validated = _require_exact_arguments(arguments, required={"command": str})
    command, display_command = _prepare_shell_command(validated["command"])
    with tempfile.TemporaryDirectory(prefix="harness-sandbox-") as temporary:
        sandbox_root = Path(temporary) / "repository"
        try:
            shutil.copytree(
                _repository_root(),
                sandbox_root,
                symlinks=True,
                ignore=_sandbox_ignore,
            )
        except OSError as exc:
            raise ToolArgumentError("could not create disposable sandbox") from exc
        result = _run_bounded_process(
            command,
            keep_tail=True,
            timeout_seconds=min(timeout_seconds, SHELL_TIMEOUT_SECONDS),
            cwd=sandbox_root,
            env=_sandbox_environment(),
        )
    result["command"] = display_command
    result["sandbox"] = "disposable repository copy"
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
    "apply_patch": apply_patch,
    "run_shell": run_shell,
}
APPROVAL_REQUIRED_TOOLS = {"apply_patch", "run_shell", "run_tests"}


def _validate_approval_gated_call(name: str, arguments: dict[str, Any]) -> None:
    """Validate a gated call before asking a person to approve it."""
    if name == "apply_patch":
        patch_text = _validated_patch_text(arguments)
        if patch_text.startswith("*** Begin Patch"):
            sections = _parse_apply_patch(patch_text)
            _prepare_apply_patch_changes(sections)
        else:
            _unified_patch_paths(patch_text)
    elif name == "run_shell":
        validated = _require_exact_arguments(arguments, required={"command": str})
        _prepare_shell_command(validated["command"])
    elif name == "run_tests":
        _require_exact_arguments(arguments, required={})
        _prepare_shell_command(
            ".venv/bin/python -m unittest discover -s tests -v"
        )


def _approval_summary(name: str, arguments: dict[str, Any]) -> str:
    if name == "run_shell":
        return f"run_shell: {arguments['command']}"
    if name == "run_tests":
        return "run_tests: .venv/bin/python -m unittest discover -s tests -v"
    patch_text = arguments["patch"]
    if patch_text.startswith("*** Begin Patch"):
        paths = [section["path"] for section in _parse_apply_patch(patch_text)]
    else:
        paths = _unified_patch_paths(patch_text)
    return f"apply_patch: {', '.join(paths)}"


def _prompt_for_tool_approval(name: str, arguments: dict[str, Any]) -> bool:
    """Ask for a local terminal approval, failing closed without a TTY."""
    if not sys.stdin.isatty():
        return False
    summary = _approval_summary(name, arguments)
    try:
        answer = input(f"\napproval required> {summary}\nAllow? [y/N] ").strip()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer.lower() in {"y", "yes"}


def execute_tool(
    name: str,
    raw_arguments: str | None,
    *,
    timeout_seconds: float | None = None,
    approval_policy: str = "deny",
    approval_callback: Callable[[str, dict[str, Any]], bool] | None = None,
    timeout_provider: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Parse, validate, and execute one tool call as a serializable envelope."""
    approval: str | None = None
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
        if name in APPROVAL_REQUIRED_TOOLS:
            _validate_approval_gated_call(name, arguments)
            if approval_policy not in APPROVAL_POLICIES:
                raise ToolArgumentError(f"unknown approval policy: {approval_policy}")
            if approval_policy == "allow":
                approved = True
            elif approval_policy == "ask" and approval_callback is not None:
                approved = approval_callback(name, arguments)
            else:
                approved = False
            approval = "granted" if approved else "denied"
            if not approved:
                raise ApprovalDeniedError(
                    f"approval denied for side-effecting tool: {name}"
                )
        if name in {"git_diff", "run_tests", "apply_patch", "run_shell"} and (
            timeout_seconds is not None
        ):
            if timeout_provider is not None:
                timeout_seconds = min(timeout_seconds, timeout_provider())
            result = handler(arguments, timeout_seconds=timeout_seconds)
        else:
            result = handler(arguments)
        envelope = {"ok": True, "result": result}
        if approval is not None:
            envelope["approval"] = approval
        return envelope
    except Exception as exc:
        envelope = {
            "ok": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        if approval is not None:
            envelope["approval"] = approval
        return envelope


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
    event_logger: EventLogger,
    session_store: SessionStore | None = None,
    progress_tracker: ProgressTracker | None = None,
) -> int:
    """Run model/tool turns until the model answers or a guardrail stops it."""
    history_snapshot = copy.deepcopy(messages)
    user_message = {"role": "user", "content": prompt}
    messages.append(user_message)
    relevant_limit = getattr(args, "relevant_files", DEFAULT_RELEVANT_FILES)
    context_message, relevant_files = _turn_context_message(prompt, relevant_limit)
    event_logger.log(
        "user_message",
        content=prompt,
        history_message_count=len(messages),
        relevant_files=relevant_files,
    )
    max_turns = getattr(args, "max_turns", DEFAULT_MAX_TURNS)
    timeout_seconds = getattr(args, "timeout", DEFAULT_TIMEOUT_SECONDS)
    approval_policy = getattr(args, "approval_policy", "deny")
    compaction_threshold = getattr(
        args, "compaction_threshold", DEFAULT_COMPACTION_THRESHOLD
    )
    keep_recent_turns = getattr(args, "keep_recent_turns", DEFAULT_RECENT_TURNS)
    started_at = time.monotonic()
    deadline = started_at + timeout_seconds
    responses: list[Any] = []
    event_logger.log(
        "agent_loop_started",
        max_turns=max_turns,
        timeout_seconds=timeout_seconds,
        approval_policy=approval_policy,
        sandbox=getattr(args, "sandbox", "disposable"),
        available_tools=sorted(TOOL_HANDLERS),
    )
    if progress_tracker is not None:
        progress_tracker.start(prompt)

    def request_approval(name: str, arguments: dict[str, Any]) -> bool:
        summary = _approval_summary(name, arguments)
        event_logger.log(
            "tool_approval_requested",
            tool=name,
            summary=summary,
        )
        approved = _prompt_for_tool_approval(name, arguments)
        event_logger.log(
            "tool_approval_resolved",
            tool=name,
            summary=summary,
            approved=approved,
        )
        return approved

    try:
        for turn_number in range(1, max_turns + 1):
            observed_prompt_tokens = None
            if responses:
                observed_prompt_tokens = getattr(
                    getattr(responses[-1], "usage", None), "prompt_tokens", None
                )
            compaction = maybe_compact_history(
                messages,
                args.context_window,
                threshold=compaction_threshold,
                keep_recent_turns=keep_recent_turns,
                observed_prompt_tokens=observed_prompt_tokens,
            )
            if compaction is not None:
                event_logger.log(
                    "conversation_context_compacted",
                    turn=turn_number,
                    history_message_count=len(messages),
                    **compaction,
                )
                if session_store is not None:
                    session_store.save(
                        event_logger.session_id, messages, reason="compaction"
                    )
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
            api_messages = _request_messages(messages, context_message)
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=api_messages,
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
                if session_store is not None:
                    session_store.save(
                        event_logger.session_id, messages, reason="final_response"
                    )
                if progress_tracker is not None:
                    progress_tracker.complete(content)
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
                    approval_policy=approval_policy,
                    approval_callback=request_approval,
                    timeout_provider=lambda: _remaining_seconds(deadline),
                )
                elapsed_ms = round(
                    (time.monotonic() - tool_started_at) * 1_000, 2
                )
                context_result, result_summarized = summarize_tool_output(result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": context_result,
                    }
                )
                event_logger.log(
                    "tool_call_completed",
                    turn=turn_number,
                    call_id=call_id,
                    tool=name,
                    success=result["ok"],
                    result=result,
                    context_result_summarized=result_summarized,
                    latency_ms=elapsed_ms,
                    history_message_count=len(messages),
                )
                if progress_tracker is not None:
                    progress_tracker.record_tool(name, result["ok"])
                _remaining_seconds(deadline)
            if session_store is not None:
                session_store.save(
                    event_logger.session_id, messages, reason="tool_batch"
                )

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
        messages[:] = history_snapshot
        if session_store is not None:
            session_store.save(event_logger.session_id, messages, reason="rollback")
        if progress_tracker is not None:
            progress_tracker.failed(exc)
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
    event_logger: EventLogger,
    session_store: SessionStore | None = None,
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
    if session_store is not None:
        session_store.save(event_logger.session_id, messages, reason="clear")
    return removed_count


def interactive_cli(
    client: Cerebras,
    messages: list[dict[str, Any]],
    args: argparse.Namespace,
    event_logger: EventLogger,
    session_store: SessionStore | None = None,
    progress_tracker: ProgressTracker | None = None,
) -> int:
    """Read questions until the user exits, retaining successful turns."""
    if not sys.stdin.isatty():
        print(
            "Error: provide a prompt when standard input is not interactive.",
            file=sys.stderr,
        )
        return 2

    print("Harness Stage 5. Type /help for help or /exit to quit.")
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
            print(
                "Ask a question. Commands: /clear, /compact, /help, /exit, /quit"
            )
            continue
        if prompt == "/clear":
            removed_count = clear_conversation_context(
                messages, event_logger, session_store
            )
            print(
                "Conversation context cleared "
                f"({removed_count} message{'s' if removed_count != 1 else ''} removed)."
            )
            continue
        if prompt == "/compact":
            result = compact_history(
                messages,
                getattr(args, "keep_recent_turns", DEFAULT_RECENT_TURNS),
            )
            if session_store is not None:
                session_store.save(
                    event_logger.session_id, messages, reason="manual_compaction"
                )
            if result is None:
                print("Nothing old enough to compact.")
            else:
                event_logger.log(
                    "conversation_context_compacted",
                    manual=True,
                    history_message_count=len(messages),
                    **result,
                )
                print(f"Compacted {result['compacted_messages']} older messages.")
            continue

        try:
            run_turn(
                client,
                messages,
                prompt,
                args,
                event_logger,
                session_store,
                progress_tracker,
            )
        except KeyboardInterrupt:
            print("\nRequest interrupted.", file=sys.stderr)
        except Exception as exc:
            print(f"Error communicating with Cerebras: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)
    root = _repository_root()
    session_id = str(uuid.uuid4())
    session_store: SessionStore | None = None
    legacy_session_imported = False
    legacy_progress_imported = False
    try:
        session_store = SessionStore(args.state_file, root)
        if args.resume:
            if not session_store.has_sessions() and (
                root / LEGACY_SESSION_FILE
            ).is_file():
                session_id, messages = session_store.import_legacy_session(
                    LEGACY_SESSION_FILE
                )
                legacy_session_imported = True
            else:
                session_id, messages = session_store.load()
        else:
            messages = []
        progress_tracker = ProgressTracker(
            session_store, session_id, load_existing=args.resume
        )
        if args.resume:
            legacy_progress_imported = progress_tracker.import_legacy(
                args.progress_file
            )
        progress_context = (
            progress_tracker.render()
            if args.resume
            and session_store.load_progress(session_id) is not None
            else None
        )
        instruction_documents = load_instruction_documents(
            root, args.global_instructions
        )
        system_prompt = build_system_prompt(
            args.system_prompt,
            instruction_documents,
            progress_context,
        )
        if messages and messages[0].get("role") == "system":
            messages[0] = {"role": "system", "content": system_prompt}
        else:
            messages.insert(0, {"role": "system", "content": system_prompt})
        session_store.save(session_id, messages, reason="session_initialized")
        event_logger = EventLogger(session_store, session_id, args.log_file)
    except (OSError, RuntimeError, ValueError, sqlite3.DatabaseError) as exc:
        message = f"could not initialize context state: {exc}"
        if session_store is not None:
            try:
                event_logger = EventLogger(session_store, session_id, args.log_file)
                event_logger.log("configuration_error", message=message)
                event_logger.log("session_ended", exit_code=2)
            except (OSError, RuntimeError, ValueError, sqlite3.DatabaseError):
                pass
        print(f"Error: {message}", file=sys.stderr)
        return 2

    event_logger.log(
        "session_started",
        model=args.model,
        context_window_tokens=args.context_window,
        max_turns=args.max_turns,
        timeout_seconds=args.timeout,
        approval_policy=args.approval_policy,
        sandbox=args.sandbox,
        available_tools=sorted(TOOL_HANDLERS),
        interactive=args.prompt is None,
        resumed=args.resume,
        state_file=str(session_store.path),
        event_export_file=(str(event_logger.path) if event_logger.path else None),
        legacy_session_imported=legacy_session_imported,
        legacy_progress_imported=legacy_progress_imported,
        instruction_files=[label for label, _content in instruction_documents],
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
            if args.prompt is None:
                exit_code = interactive_cli(
                    client,
                    messages,
                    args,
                    event_logger,
                    session_store,
                    progress_tracker,
                )
            else:
                try:
                    exit_code = run_turn(
                        client,
                        messages,
                        args.prompt,
                        args,
                        event_logger,
                        session_store,
                        progress_tracker,
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
