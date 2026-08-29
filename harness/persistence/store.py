"""SQLite session store with immutable context snapshots."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from harness.config import (
    LEGACY_SESSION_SCHEMA_VERSION,
    MAX_EVENT_PAYLOAD_CHARS,
    STATE_SCHEMA_VERSION,
)
from harness.workspace import _atomic_write_text, current_workspace


class SessionStore:
    """Persist sessions, immutable context snapshots, events, and checkpoints."""

    def __init__(self, path: Path, repository_root: Path) -> None:
        workspace = current_workspace()
        self.path = workspace.validated_runtime_path(path)
        self.repository_root = repository_root.resolve()
        self.artifact_directory = self.path.with_name(f"{self.path.name}.artifacts")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        workspace.register_private_path(self.path)
        workspace.register_private_path(self.artifact_directory)
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
                connection.execute(f"PRAGMA user_version = {STATE_SCHEMA_VERSION}")
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
        legacy_path = current_workspace().validated_runtime_path(path)
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
                current_workspace().register_private_path(artifact_path)
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
                        "notice": (
                            "Full event payload stored as a content-addressed artifact."
                        ),
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
            query += " WHERE events.session_id = ?"
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
