"""Cerebras coding agent package.

The public names below keep `import harness` stable for tests and callers.
Implementation lives in focused submodules.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

from cerebras.cloud.sdk import Cerebras

from harness.agent.compaction import compact_history, maybe_compact_history
from harness.agent.context import (
    build_system_prompt,
    load_instruction_documents,
    select_relevant_files,
    summarize_tool_output,
)
from harness.agent.loop import (
    _prompt_for_tool_approval,
    _remaining_seconds,
    clear_conversation_context,
    execute_tool,
    run_turn,
)
from harness.cli import interactive_cli, main, parse_args
from harness.config import (
    ALLOWED_SHELL_COMMANDS,
    APPROVAL_POLICIES,
    DEFAULT_COMPACTION_THRESHOLD,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_TURNS,
    DEFAULT_RECENT_TURNS,
    DEFAULT_RELEVANT_FILES,
    DEFAULT_STATE_FILE,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TIMEOUT_SECONDS,
    LEGACY_PROGRESS_FILE,
    LEGACY_SESSION_FILE,
    MAX_CONTEXT_TOOL_OUTPUT_CHARS,
    MAX_EVENT_PAYLOAD_CHARS,
    MAX_LISTED_FILES,
    MAX_PROCESS_OUTPUT_LINES,
    MAX_READ_LINES,
    MAX_SEARCH_FILE_BYTES,
    MAX_SEARCH_RESULTS,
    MAX_TOOL_OUTPUT_CHARS,
    SHELL_TIMEOUT_SECONDS,
    TEST_TIMEOUT_SECONDS,
)
from harness.display import (
    _combined_metrics,
    clear_transcript,
    empty_metrics,
    format_metrics_line,
    fullscreen_session,
    last_response_metrics,
    model_label,
    print_response,
    print_user_input,
    prompt_status_lines,
    prompt_status_session,
    read_prompt,
    response_metrics,
)
from harness.exceptions import (
    AgentLoopError,
    AgentTimeoutError,
    ApprovalDeniedError,
    MaxTurnsExceededError,
    ToolArgumentError,
    ToolProtocolError,
)
from harness.persistence.events import EventLogger
from harness.persistence.progress import ProgressTracker
from harness.persistence.store import SessionStore
from harness.tools.basic import calculator, echo, get_current_time
from harness.tools.fs import git_diff, list_files, read_file, search_text
from harness.tools.patch import apply_patch
from harness.tools.process import run_bounded_process
from harness.tools.registry import APPROVAL_REQUIRED_TOOLS, TOOL_HANDLERS, TOOLS
from harness.tools.shell import run_shell, run_tests
from harness.workspace import (
    Workspace,
    _register_private_runtime_path,
    _repository_root,
    _resolve_repository_path,
    current_workspace,
)

_run_bounded_process = run_bounded_process

__all__ = [
    "ALLOWED_SHELL_COMMANDS",
    "APPROVAL_POLICIES",
    "APPROVAL_REQUIRED_TOOLS",
    "AgentLoopError",
    "AgentTimeoutError",
    "ApprovalDeniedError",
    "Cerebras",
    "DEFAULT_COMPACTION_THRESHOLD",
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_RECENT_TURNS",
    "DEFAULT_RELEVANT_FILES",
    "DEFAULT_STATE_FILE",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_TIMEOUT_SECONDS",
    "EventLogger",
    "LEGACY_PROGRESS_FILE",
    "LEGACY_SESSION_FILE",
    "MAX_CONTEXT_TOOL_OUTPUT_CHARS",
    "MAX_EVENT_PAYLOAD_CHARS",
    "MAX_LISTED_FILES",
    "MAX_PROCESS_OUTPUT_LINES",
    "MAX_READ_LINES",
    "MAX_SEARCH_FILE_BYTES",
    "MAX_SEARCH_RESULTS",
    "MAX_TOOL_OUTPUT_CHARS",
    "MaxTurnsExceededError",
    "ProgressTracker",
    "SHELL_TIMEOUT_SECONDS",
    "SessionStore",
    "TEST_TIMEOUT_SECONDS",
    "TOOLS",
    "TOOL_HANDLERS",
    "ToolArgumentError",
    "ToolProtocolError",
    "Workspace",
    "apply_patch",
    "build_system_prompt",
    "calculator",
    "clear_conversation_context",
    "clear_transcript",
    "compact_history",
    "current_workspace",
    "echo",
    "empty_metrics",
    "execute_tool",
    "format_metrics_line",
    "fullscreen_session",
    "get_current_time",
    "git_diff",
    "interactive_cli",
    "last_response_metrics",
    "list_files",
    "load_instruction_documents",
    "main",
    "maybe_compact_history",
    "model_label",
    "os",
    "parse_args",
    "print_response",
    "print_user_input",
    "prompt_status_lines",
    "prompt_status_session",
    "read_prompt",
    "read_file",
    "response_metrics",
    "run_shell",
    "run_tests",
    "run_turn",
    "search_text",
    "select_relevant_files",
    "shutil",
    "subprocess",
    "summarize_tool_output",
    "sys",
]
