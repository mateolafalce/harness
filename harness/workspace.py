"""Repository boundary used by filesystem tools and runtime state."""

from __future__ import annotations

import os
import uuid
from contextvars import ContextVar, Token
from pathlib import Path

from harness.config import (
    DEFAULT_STATE_FILE,
    EXCLUDED_DIRECTORY_NAMES,
    LEGACY_PROGRESS_FILE,
    LEGACY_SESSION_FILE,
    MAX_INSTRUCTION_FILE_CHARS,
)
from harness.exceptions import ToolArgumentError

_active: ContextVar[Workspace | None] = ContextVar("harness_workspace", default=None)


class Workspace:
    """Explicit working copy of the repository the agent is allowed to touch."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or Path.cwd()).resolve()
        self.private_paths: set[str] = set()
        self._token: Token[Workspace | None] | None = None

    def activate(self) -> Workspace:
        self._token = _active.set(self)
        return self

    def deactivate(self) -> None:
        if self._token is not None:
            _active.reset(self._token)
            self._token = None

    def register_private_path(self, path: Path) -> None:
        """Hide a runtime file and its SQLite sidecars from repository tools."""
        try:
            relative = path.resolve().relative_to(self.root)
        except ValueError:
            return
        posix = relative.as_posix()
        self.private_paths.add(posix)
        for suffix in ("-journal", "-shm", "-wal"):
            self.private_paths.add(f"{posix}{suffix}")

    def is_sensitive(self, path: Path) -> bool:
        if any(part in EXCLUDED_DIRECTORY_NAMES for part in path.parts):
            return True
        if path.as_posix() in self.private_paths:
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

    def resolve_path(self, raw_path: str) -> tuple[Path, str]:
        path_text = raw_path.strip()
        if not path_text:
            raise ToolArgumentError("argument 'path' must not be empty")

        supplied = Path(path_text)
        if supplied.is_absolute():
            raise ToolArgumentError("argument 'path' must be repository-relative")

        candidate = (self.root / supplied).resolve()
        try:
            relative = candidate.relative_to(self.root)
        except ValueError as exc:
            raise ToolArgumentError("path must stay inside the repository") from exc
        if self.is_sensitive(relative):
            raise ToolArgumentError("path is excluded from repository tools")
        display_path = relative.as_posix() or "."
        return candidate, display_path

    def validated_runtime_path(self, path: Path) -> Path:
        candidate = path.expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(
                "runtime state paths must stay inside the repository"
            ) from exc
        return candidate


def current_workspace() -> Workspace:
    workspace = _active.get()
    if workspace is None:
        workspace = Workspace()
        _active.set(workspace)
    return workspace


def repository_root() -> Path:
    """Return the repository boundary used by all filesystem tools."""
    return current_workspace().root


def _repository_root() -> Path:
    """Compatibility alias for the active workspace root."""
    return repository_root()


def _is_sensitive_path(path: Path) -> bool:
    return current_workspace().is_sensitive(path)


def _resolve_repository_path(raw_path: str) -> tuple[Path, str]:
    return current_workspace().resolve_path(raw_path)


def _register_private_runtime_path(path: Path) -> None:
    current_workspace().register_private_path(path)


def _validated_runtime_path(path: Path) -> Path:
    return current_workspace().validated_runtime_path(path)


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


def _visible_files(path: Path) -> list[Path]:
    """Return deterministic files without descending into generated directories."""
    workspace = current_workspace()
    root = workspace.root
    if path.is_file():
        return [path]

    files: list[Path] = []
    for directory, directory_names, file_names in os.walk(path, followlinks=False):
        relative_directory = Path(directory).relative_to(root)
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in EXCLUDED_DIRECTORY_NAMES
            and not workspace.is_sensitive(relative_directory / name)
        )
        for name in sorted(file_names):
            candidate = Path(directory) / name
            relative = candidate.relative_to(root)
            if workspace.is_sensitive(relative):
                continue
            try:
                candidate.resolve().relative_to(root)
            except (OSError, ValueError):
                continue
            if not candidate.is_file():
                continue
            files.append(candidate)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())
