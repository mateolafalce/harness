"""Ordered audit events, with optional JSONL export."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.persistence.store import SessionStore
from harness.workspace import current_workspace


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
        workspace = current_workspace()
        self.path = (
            workspace.validated_runtime_path(export_path)
            if export_path is not None
            else None
        )
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            workspace.register_private_path(self.path)

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
