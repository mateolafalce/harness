"""Shared argument validation for model-supplied tool payloads."""

from __future__ import annotations

from typing import Any

from harness.exceptions import ToolArgumentError


def require_exact_arguments(
    arguments: Any,
    *,
    required: dict[str, type],
) -> dict[str, Any]:
    """Validate a small object schema without adding a framework dependency."""
    if not isinstance(arguments, dict):
        raise ToolArgumentError("arguments must be a JSON object")

    expected = set(required)
    received = set(arguments)
    missing = expected - received
    unexpected = received - expected
    if missing:
        raise ToolArgumentError(
            f"missing required argument(s): {', '.join(sorted(missing))}"
        )
    if unexpected:
        raise ToolArgumentError(
            f"unexpected argument(s): {', '.join(sorted(unexpected))}"
        )

    for name, expected_type in required.items():
        if not isinstance(arguments[name], expected_type):
            raise ToolArgumentError(
                f"argument '{name}' must be {expected_type.__name__}"
            )
    return arguments


def validate_positive_integer(
    name: str, value: Any, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ToolArgumentError(f"argument '{name}' must be a positive integer")
    if maximum is not None and value > maximum:
        raise ToolArgumentError(f"argument '{name}' must not exceed {maximum}")
    return value
