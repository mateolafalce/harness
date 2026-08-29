"""Command-line entry point for the Cerebras coding agent."""

from __future__ import annotations

import argparse
import math
import os
import sqlite3
import sys
import uuid
from pathlib import Path

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

from harness.agent.compaction import compact_history
from harness.agent.context import build_system_prompt, load_instruction_documents
from harness.agent.loop import clear_conversation_context, report_turn_error, run_turn
from harness.display import (
    clear_transcript,
    empty_metrics,
    fullscreen_session,
    last_response_metrics,
    print_user_input,
    prompt_status_session,
    read_prompt,
)
from harness.config import (
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
    SANDBOX_MODES,
)
from harness.persistence.events import EventLogger
from harness.persistence.progress import ProgressTracker
from harness.persistence.store import SessionStore
from harness.tools.registry import TOOL_HANDLERS
from harness.workspace import current_workspace


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


def interactive_cli(
    client: Cerebras,
    messages: list[dict],
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

    with fullscreen_session(), prompt_status_session():
        metrics = empty_metrics(getattr(args, "context_window", None))
        while True:
            try:
                prompt = read_prompt(
                    args.model,
                    getattr(args, "reasoning_effort", None),
                    metrics,
                ).strip()
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
            if prompt == "/clear":
                removed_count = clear_conversation_context(
                    messages, event_logger, session_store
                )
                clear_transcript()
                print(
                    "Conversation context cleared "
                    f"({removed_count} message{'s' if removed_count != 1 else ''} removed)."
                )
                continue
            print_user_input(prompt)
            if prompt == "/help":
                print(
                    "Ask a question. Commands: /clear, /compact, /help, /exit, /quit"
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
                    print(
                        f"Compacted {result['compacted_messages']} older messages."
                    )
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
                metrics = last_response_metrics() or metrics
            except KeyboardInterrupt:
                print("\nRequest interrupted.", file=sys.stderr)
            except Exception as exc:
                report_turn_error(exc)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = current_workspace()
    load_dotenv(workspace.root / ".env", override=False)
    root = workspace.root
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
                    report_turn_error(exc)
                    exit_code = 1
        return exit_code
    finally:
        event_logger.log("session_ended", exit_code=exit_code)
