"""Instruction loading, retrieval hints, and tool-result summarization."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from harness.config import (
    DEFAULT_RELEVANT_FILES,
    MAX_CONTEXT_TOOL_OUTPUT_CHARS,
    MAX_INSTRUCTION_CONTEXT_CHARS,
)
from harness.workspace import _read_context_file, _visible_files, current_workspace


def _default_global_instruction_path() -> Path:
    harness_home = os.environ.get("HARNESS_HOME")
    base = Path(harness_home).expanduser() if harness_home else Path.home() / ".harness"
    return base / "AGENTS.md"


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
    root = current_workspace().root
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
    root = current_workspace().root
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
