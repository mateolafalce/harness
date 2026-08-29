"""Model/tool turn orchestration."""

from __future__ import annotations

import copy
import json
import sys
import time
from typing import Any, Callable

from cerebras.cloud.sdk import Cerebras

from harness.agent.compaction import compact_history, maybe_compact_history
from harness.agent.context import (
    _request_messages,
    _turn_context_message,
    summarize_tool_output,
)
from harness.config import APPROVAL_POLICIES, AgentSettings
from harness.display import _combined_metrics, print_response, response_metrics
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
from harness.tools.patch import (
    _parse_apply_patch,
    _prepare_apply_patch_changes,
    _unified_patch_paths,
    _validated_patch_text,
)
from harness.tools.registry import TOOL_HANDLERS, TOOL_SPECS_BY_NAME, TOOLS
from harness.tools.shell import prepare_shell_command
from harness.tools.validation import require_exact_arguments


def _validate_approval_gated_call(name: str, arguments: dict[str, Any]) -> None:
    """Validate a gated call before asking a person to approve it."""
    if name == "apply_patch":
        patch_text = _validated_patch_text(arguments)
        if patch_text.startswith("*** Begin Patch"):
            sections = _parse_apply_patch(patch_text)
            _prepare_apply_patch_changes(sections)
        else:
            _unified_patch_paths(patch_text)
    elif name == "run_shell":
        validated = require_exact_arguments(arguments, required={"command": str})
        prepare_shell_command(validated["command"])
    elif name == "run_tests":
        require_exact_arguments(arguments, required={})
        prepare_shell_command(".venv/bin/python -m unittest discover -s tests -v")


def _approval_summary(name: str, arguments: dict[str, Any]) -> str:
    if name == "run_shell":
        return f"run_shell: {arguments['command']}"
    if name == "run_tests":
        return "run_tests: .venv/bin/python -m unittest discover -s tests -v"
    patch_text = arguments["patch"]
    if patch_text.startswith("*** Begin Patch"):
        paths = [section["path"] for section in _parse_apply_patch(patch_text)]
    else:
        paths = _unified_patch_paths(patch_text)
    return f"apply_patch: {', '.join(paths)}"


def _prompt_for_tool_approval(name: str, arguments: dict[str, Any]) -> bool:
    """Ask for a local terminal approval, failing closed without a TTY."""
    if not sys.stdin.isatty():
        return False
    summary = _approval_summary(name, arguments)
    try:
        answer = input(f"\napproval required> {summary}\nAllow? [y/N] ").strip()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer.lower() in {"y", "yes"}


def execute_tool(
    name: str,
    raw_arguments: str | None,
    *,
    timeout_seconds: float | None = None,
    approval_policy: str = "deny",
    approval_callback: Callable[[str, dict[str, Any]], bool] | None = None,
    timeout_provider: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Parse, validate, and execute one tool call as a serializable envelope."""
    approval: str | None = None
    try:
        if raw_arguments is None:
            raise ToolArgumentError("tool arguments are missing")
        try:
            arguments = json.loads(raw_arguments)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ToolArgumentError("arguments are not valid JSON") from exc

        spec = TOOL_SPECS_BY_NAME.get(name)
        if spec is None:
            raise ToolArgumentError(f"unknown tool: {name}")
        if spec.requires_approval:
            _validate_approval_gated_call(name, arguments)
            if approval_policy not in APPROVAL_POLICIES:
                raise ToolArgumentError(f"unknown approval policy: {approval_policy}")
            if approval_policy == "allow":
                approved = True
            elif approval_policy == "ask" and approval_callback is not None:
                approved = approval_callback(name, arguments)
            else:
                approved = False
            approval = "granted" if approved else "denied"
            if not approved:
                raise ApprovalDeniedError(
                    f"approval denied for side-effecting tool: {name}"
                )
        if spec.supports_timeout and timeout_seconds is not None:
            if timeout_provider is not None:
                timeout_seconds = min(timeout_seconds, timeout_provider())
            result = spec.handler(arguments, timeout_seconds=timeout_seconds)
        else:
            result = spec.handler(arguments)
        envelope = {"ok": True, "result": result}
        if approval is not None:
            envelope["approval"] = approval
        return envelope
    except ApprovalDeniedError as exc:
        envelope = {
            "ok": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "approval": approval,
        }
        return envelope
    except AgentLoopError:
        raise
    except Exception as exc:
        envelope = {
            "ok": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        if approval is not None:
            envelope["approval"] = approval
        return envelope


def _tool_call_as_message(call: Any) -> dict[str, Any]:
    function = getattr(call, "function", None)
    return {
        "id": getattr(call, "id", None),
        "type": getattr(call, "type", "function"),
        "function": {
            "name": getattr(function, "name", None),
            "arguments": getattr(function, "arguments", None),
        },
    }


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AgentTimeoutError("agent loop timed out")
    return remaining


def run_turn(
    client: Cerebras,
    messages: list[dict[str, Any]],
    prompt: str,
    args: Any,
    event_logger: EventLogger,
    session_store: SessionStore | None = None,
    progress_tracker: ProgressTracker | None = None,
) -> int:
    """Run model/tool turns until the model answers or a guardrail stops it."""
    settings = (
        args if isinstance(args, AgentSettings) else AgentSettings.from_namespace(args)
    )
    history_snapshot = copy.deepcopy(messages)
    user_message = {"role": "user", "content": prompt}
    messages.append(user_message)
    context_message, relevant_files = _turn_context_message(
        prompt, settings.relevant_files
    )
    event_logger.log(
        "user_message",
        content=prompt,
        history_message_count=len(messages),
        relevant_files=relevant_files,
    )
    max_turns = settings.max_turns
    timeout_seconds = settings.timeout
    approval_policy = settings.approval_policy
    started_at = time.monotonic()
    deadline = started_at + timeout_seconds
    responses: list[Any] = []
    event_logger.log(
        "agent_loop_started",
        max_turns=max_turns,
        timeout_seconds=timeout_seconds,
        approval_policy=approval_policy,
        sandbox=settings.sandbox,
        available_tools=sorted(TOOL_HANDLERS),
    )
    if progress_tracker is not None:
        progress_tracker.start(prompt)

    def request_approval(name: str, arguments: dict[str, Any]) -> bool:
        summary = _approval_summary(name, arguments)
        event_logger.log(
            "tool_approval_requested",
            tool=name,
            summary=summary,
        )
        approved = _prompt_for_tool_approval(name, arguments)
        event_logger.log(
            "tool_approval_resolved",
            tool=name,
            summary=summary,
            approved=approved,
        )
        return approved

    try:
        for turn_number in range(1, max_turns + 1):
            observed_prompt_tokens = None
            if responses:
                observed_prompt_tokens = getattr(
                    getattr(responses[-1], "usage", None), "prompt_tokens", None
                )
            compaction = maybe_compact_history(
                messages,
                settings.context_window,
                threshold=settings.compaction_threshold,
                keep_recent_turns=settings.keep_recent_turns,
                observed_prompt_tokens=observed_prompt_tokens,
            )
            if compaction is not None:
                event_logger.log(
                    "conversation_context_compacted",
                    turn=turn_number,
                    history_message_count=len(messages),
                    **compaction,
                )
                if session_store is not None:
                    session_store.save(
                        event_logger.session_id, messages, reason="compaction"
                    )
            remaining = _remaining_seconds(deadline)
            event_logger.log(
                "agent_turn_started",
                turn=turn_number,
                remaining_ms=round(remaining * 1_000, 2),
                history_message_count=len(messages),
            )
            event_logger.log(
                "api_request",
                turn=turn_number,
                model=settings.model,
                history_message_count=len(messages),
                max_completion_tokens=settings.max_completion_tokens,
                tool_choice="auto",
                timeout_seconds=remaining,
            )

            request_started_at = time.monotonic()
            api_messages = _request_messages(messages, context_message)
            try:
                response = client.chat.completions.create(
                    model=settings.model,
                    messages=api_messages,
                    reasoning_effort=settings.reasoning_effort,
                    max_completion_tokens=settings.max_completion_tokens,
                    tools=TOOLS,
                    tool_choice="auto",
                    timeout=remaining,
                )
            except Exception as exc:
                event_logger.log(
                    "api_error",
                    turn=turn_number,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    latency_ms=round(
                        (time.monotonic() - request_started_at) * 1_000, 2
                    ),
                )
                raise

            _remaining_seconds(deadline)
            if not response.choices:
                event_logger.log(
                    "api_error",
                    turn=turn_number,
                    error_type="EmptyResponse",
                    message="The API response did not contain any choices.",
                    latency_ms=round(
                        (time.monotonic() - request_started_at) * 1_000, 2
                    ),
                )
                raise RuntimeError("The API response did not contain any choices.")

            responses.append(response)
            choice = response.choices[0]
            content = choice.message.content or ""
            reasoning = getattr(choice.message, "reasoning", None)
            tool_calls = list(getattr(choice.message, "tool_calls", None) or [])
            request_metrics = response_metrics(
                response,
                time.monotonic() - request_started_at,
                settings.context_window,
            )
            event_logger.log(
                "model_response",
                turn=turn_number,
                content=content,
                reasoning=reasoning,
                response_id=getattr(response, "id", None),
                model=getattr(response, "model", settings.model),
                finish_reason=getattr(choice, "finish_reason", None),
                metrics=request_metrics,
                tool_call_count=len(tool_calls),
            )

            if not tool_calls:
                messages.append({"role": "assistant", "content": content})
                if session_store is not None:
                    session_store.save(
                        event_logger.session_id, messages, reason="final_response"
                    )
                if progress_tracker is not None:
                    progress_tracker.complete(content)
                event_logger.log(
                    "agent_decision",
                    turn=turn_number,
                    decision="final_response",
                    history_message_count=len(messages),
                )
                metrics = _combined_metrics(
                    responses,
                    time.monotonic() - started_at,
                    settings.context_window,
                )
                print_response(content, metrics)
                event_logger.log(
                    "agent_loop_completed",
                    turns=turn_number,
                    metrics=metrics,
                    history_message_count=len(messages),
                )
                return 0

            serialized_calls = [_tool_call_as_message(call) for call in tool_calls]
            call_ids = [call["id"] for call in serialized_calls]
            protocol_error = None
            if any(not isinstance(call_id, str) or not call_id for call_id in call_ids):
                protocol_error = "model returned a tool call without a valid call_id"
            elif len(set(call_ids)) != len(call_ids):
                protocol_error = "model returned duplicate tool call_ids"
            elif any(call["type"] != "function" for call in serialized_calls):
                protocol_error = "model returned an unsupported tool call type"
            elif any(
                not isinstance(call["function"]["name"], str)
                or not call["function"]["name"]
                or not isinstance(call["function"]["arguments"], str)
                for call in serialized_calls
            ):
                protocol_error = "model returned a malformed function call"
            if protocol_error:
                event_logger.log(
                    "agent_decision",
                    turn=turn_number,
                    decision="tool_protocol_error",
                    message=protocol_error,
                )
                raise ToolProtocolError(protocol_error)

            assistant_message = {
                "role": "assistant",
                "content": content or None,
                "tool_calls": serialized_calls,
            }
            if reasoning is not None:
                assistant_message["reasoning"] = reasoning
            messages.append(assistant_message)
            event_logger.log(
                "agent_decision",
                turn=turn_number,
                decision="execute_tools",
                call_ids=call_ids,
                tool_names=[call["function"]["name"] for call in serialized_calls],
            )

            for call in serialized_calls:
                tool_timeout = _remaining_seconds(deadline)
                call_id = call["id"]
                name = call["function"]["name"]
                raw_arguments = call["function"]["arguments"]
                event_logger.log(
                    "tool_call_started",
                    turn=turn_number,
                    call_id=call_id,
                    tool=name,
                    raw_arguments=raw_arguments,
                )
                tool_started_at = time.monotonic()
                result = execute_tool(
                    name,
                    raw_arguments,
                    timeout_seconds=tool_timeout,
                    approval_policy=approval_policy,
                    approval_callback=request_approval,
                    timeout_provider=lambda: _remaining_seconds(deadline),
                )
                elapsed_ms = round((time.monotonic() - tool_started_at) * 1_000, 2)
                context_result, result_summarized = summarize_tool_output(result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": context_result,
                    }
                )
                event_logger.log(
                    "tool_call_completed",
                    turn=turn_number,
                    call_id=call_id,
                    tool=name,
                    success=result["ok"],
                    result=result,
                    context_result_summarized=result_summarized,
                    latency_ms=elapsed_ms,
                    history_message_count=len(messages),
                )
                if progress_tracker is not None:
                    progress_tracker.record_tool(name, result["ok"])
                _remaining_seconds(deadline)
            if session_store is not None:
                session_store.save(
                    event_logger.session_id, messages, reason="tool_batch"
                )

        _remaining_seconds(deadline)
        event_logger.log(
            "agent_decision",
            turn=max_turns,
            decision="max_turns_exceeded",
            max_turns=max_turns,
        )
        raise MaxTurnsExceededError(
            f"agent did not produce a final response within {max_turns} model turns"
        )
    except (Exception, KeyboardInterrupt) as exc:
        messages[:] = history_snapshot
        if session_store is not None:
            session_store.save(event_logger.session_id, messages, reason="rollback")
        if progress_tracker is not None:
            progress_tracker.failed(exc)
        event_logger.log(
            "agent_loop_failed",
            error_type=type(exc).__name__,
            message=str(exc),
            latency_ms=round((time.monotonic() - started_at) * 1_000, 2),
            history_rolled_back=True,
        )
        raise


def clear_conversation_context(
    messages: list[dict[str, Any]],
    event_logger: EventLogger,
    session_store: SessionStore | None = None,
) -> int:
    """Remove conversation turns while preserving the initial system prompt."""
    retained_messages = (
        messages[:1] if messages and messages[0].get("role") == "system" else []
    )
    removed_count = len(messages) - len(retained_messages)
    messages[:] = retained_messages
    event_logger.log(
        "conversation_context_cleared",
        removed_message_count=removed_count,
        history_message_count=len(messages),
    )
    if session_store is not None:
        session_store.save(event_logger.session_id, messages, reason="clear")
    return removed_count


def report_turn_error(exc: BaseException) -> None:
    """Print a user-facing error that distinguishes agent failures from API issues."""
    if isinstance(exc, AgentLoopError):
        print(f"Error: {exc}", file=sys.stderr)
    else:
        print(f"Error communicating with Cerebras: {exc}", file=sys.stderr)
