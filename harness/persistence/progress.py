"""Compact, durable checkpoints for interrupted work."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from harness.config import MAX_PROGRESS_CHARS
from harness.persistence.store import SessionStore
from harness.workspace import _read_context_file, current_workspace


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
        legacy_path = current_workspace().validated_runtime_path(path)
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
