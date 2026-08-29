"""Approved text-patch application inside the repository."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from harness.config import (
    MAX_PATCH_CHARS,
    MAX_PATCH_FILES,
    MAX_PATCH_PATH_CHARS,
    PATCH_TIMEOUT_SECONDS,
)
from harness.exceptions import ToolArgumentError
from harness.tools.process import truncate_output
from harness.tools.validation import require_exact_arguments
from harness.workspace import current_workspace


def _resolve_patch_path(raw_path: str) -> tuple[Path, str]:
    """Resolve a patch path without permitting symlink-based aliases."""
    if len(raw_path) > MAX_PATCH_PATH_CHARS:
        raise ToolArgumentError(
            f"patch paths must not exceed {MAX_PATCH_PATH_CHARS} characters"
        )
    if ".." in Path(raw_path).parts:
        raise ToolArgumentError("patch paths must not contain '..'")
    workspace = current_workspace()
    path, display_path = workspace.resolve_path(raw_path)
    lexical_path = Path(os.path.abspath(workspace.root / raw_path))
    if lexical_path != path:
        raise ToolArgumentError("patch paths must not contain symlinks")
    return path, display_path


def _validated_patch_text(arguments: dict[str, Any]) -> str:
    validated = require_exact_arguments(arguments, required={"patch": str})
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
        root = current_workspace().root
        try:
            checked = subprocess.run(
                [*command[:2], "--check", *command[2:]],
                cwd=root,
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
                output, _ = truncate_output(checked.stdout)
                raise ToolArgumentError(f"patch does not apply: {output}")
            applied = subprocess.run(
                command,
                cwd=root,
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
            output, _ = truncate_output(applied.stdout)
            raise ToolArgumentError(f"patch application failed: {output}")
    return {"applied": True, "paths": paths, "file_count": len(paths)}
