"""Read-only repository inspection tools."""

from __future__ import annotations

import json
from typing import Any

from harness.agent.context import _instruction_paths_for_files
import harness.config as config
from harness.config import (
    MAX_READ_LINES,
    MAX_SEARCH_FILE_BYTES,
    MAX_SEARCH_LINE_CHARS,
    MAX_SEARCH_RESULTS,
    MAX_TOOL_OUTPUT_CHARS,
    TEST_TIMEOUT_SECONDS,
)
from harness.exceptions import ToolArgumentError
from harness.tools import process
from harness.tools.validation import require_exact_arguments, validate_positive_integer
from harness.workspace import _visible_files, current_workspace


def list_files(arguments: dict[str, Any]) -> dict[str, Any]:
    validated = require_exact_arguments(arguments, required={"path": str})
    path, display_path = current_workspace().resolve_path(validated["path"])
    if not path.exists():
        raise ToolArgumentError(f"path does not exist: {display_path}")
    if not path.is_dir():
        raise ToolArgumentError(f"path is not a directory: {display_path}")

    root = current_workspace().root
    files = [item.relative_to(root).as_posix() for item in _visible_files(path)]
    returned_files: list[str] = []
    returned_characters = 0
    for file_name in files[:config.MAX_LISTED_FILES]:
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
    validated = require_exact_arguments(
        arguments,
        required={"path": str, "start_line": int, "max_lines": int},
    )
    start_line = validate_positive_integer("start_line", validated["start_line"])
    max_lines = validate_positive_integer(
        "max_lines", validated["max_lines"], MAX_READ_LINES
    )
    workspace = current_workspace()
    path, display_path = workspace.resolve_path(validated["path"])
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

    content, character_truncated = process.truncate_output(
        "\n".join(selected), max_lines=max_lines
    )
    returned_lines = len(content.splitlines()) if content else 0
    instruction_files = [
        item.relative_to(workspace.root).as_posix()
        for item in _instruction_paths_for_files(workspace.root, [display_path])
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
    validated = require_exact_arguments(
        arguments, required={"query": str, "path": str}
    )
    query = validated["query"]
    if not query:
        raise ToolArgumentError("argument 'query' must not be empty")
    if len(query) > 200:
        raise ToolArgumentError("argument 'query' must not exceed 200 characters")
    workspace = current_workspace()
    path, display_path = workspace.resolve_path(validated["path"])
    if not path.exists():
        raise ToolArgumentError(f"path does not exist: {display_path}")

    root = workspace.root
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


def git_diff(
    arguments: dict[str, Any], timeout_seconds: float = TEST_TIMEOUT_SECONDS
) -> dict[str, Any]:
    validated = require_exact_arguments(arguments, required={"path": str})
    _, display_path = current_workspace().resolve_path(validated["path"])
    result = process.run_bounded_process(
        ["git", "diff", "--no-ext-diff", "--no-color", "HEAD", "--", display_path],
        keep_tail=False,
        timeout_seconds=min(timeout_seconds, TEST_TIMEOUT_SECONDS),
    )
    result["path"] = display_path
    return result
