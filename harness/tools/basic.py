"""Small, side-effect-free tools used by the agent and tests."""

from __future__ import annotations

import ast
import math
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from harness.config import (
    MAX_ABSOLUTE_EXPONENT,
    MAX_ABSOLUTE_RESULT,
    MAX_EXPRESSION_LENGTH,
    MAX_EXPRESSION_NODES,
)
from harness.exceptions import ToolArgumentError
from harness.tools.validation import require_exact_arguments

_BINARY_OPERATORS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.FloorDiv: lambda left, right: left // right,
    ast.Mod: lambda left, right: left % right,
    ast.Pow: lambda left, right: left**right,
}
_UNARY_OPERATORS = {
    ast.UAdd: lambda value: value,
    ast.USub: lambda value: -value,
}


def _evaluate_arithmetic(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate_arithmetic(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolArgumentError("expression may contain only numbers and operators")
        if not math.isfinite(node.value):
            raise ToolArgumentError("numbers must be finite")
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        result = _UNARY_OPERATORS[type(node.op)](_evaluate_arithmetic(node.operand))
    elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_arithmetic(node.left)
        right = _evaluate_arithmetic(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_ABSOLUTE_EXPONENT:
            raise ToolArgumentError("absolute exponent must not exceed 1000")
        result = _BINARY_OPERATORS[type(node.op)](left, right)
    else:
        raise ToolArgumentError("expression contains an unsupported operation")

    if isinstance(result, complex) or not math.isfinite(result):
        raise ToolArgumentError("result must be a finite real number")
    if abs(result) > MAX_ABSOLUTE_RESULT:
        raise ToolArgumentError("absolute result is too large")
    return result


def calculator(arguments: dict[str, Any]) -> dict[str, int | float]:
    validated = require_exact_arguments(arguments, required={"expression": str})
    expression = validated["expression"].strip()
    if not expression:
        raise ToolArgumentError("argument 'expression' must not be empty")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ToolArgumentError(
            f"argument 'expression' must not exceed {MAX_EXPRESSION_LENGTH} characters"
        )
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolArgumentError("expression is not valid arithmetic") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_EXPRESSION_NODES:
        raise ToolArgumentError("expression is too complex")
    try:
        return {"value": _evaluate_arithmetic(tree)}
    except (ArithmeticError, OverflowError) as exc:
        raise ToolArgumentError(str(exc) or type(exc).__name__) from exc


def get_current_time(arguments: dict[str, Any]) -> dict[str, str]:
    validated = require_exact_arguments(arguments, required={"timezone": str})
    timezone_name = validated["timezone"].strip()
    if not timezone_name:
        raise ToolArgumentError("argument 'timezone' must not be empty")
    try:
        requested_timezone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ToolArgumentError(f"unknown IANA time zone: {timezone_name}") from exc
    current = datetime.now(requested_timezone)
    return {
        "timezone": timezone_name,
        "iso8601": current.isoformat(timespec="seconds"),
    }


def echo(arguments: dict[str, Any]) -> dict[str, str]:
    validated = require_exact_arguments(arguments, required={"text": str})
    return {"text": validated["text"]}
