"""Working-memory compaction for long conversations."""

from __future__ import annotations

import json
import math
from typing import Any

from harness.config import (
    DEFAULT_COMPACTION_THRESHOLD,
    DEFAULT_RECENT_TURNS,
    MAX_COMPACTION_SUMMARY_CHARS,
)


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
