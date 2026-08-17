#!/usr/bin/env python3
"""Minimal harness for gpt-oss-120b on Cerebras with shell access."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "Runs a Bash command in the current directory and returns "
                "stdout, stderr, and the exit code."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Bash command to run.",
                    }
                },
                "required": ["cmd"],
                "additionalProperties": False,
            },
        },
    }
]


def approve_command(cmd: str, auto_approve: bool) -> bool:
    if auto_approve:
        return True
    if not sys.stdin.isatty():
        return False

    answer = input(f"\nRun this command?\n  $ {cmd}\n[y/N] ").strip().lower()
    return answer in {"y", "yes"}


def run_bash(cmd: str, *, auto_approve: bool, timeout: int) -> str:
    if not approve_command(cmd, auto_approve):
        return "Execution rejected by the user."

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else exc.stdout
        )
        stderr = (
            exc.stderr.decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else exc.stderr
        )
        partial = (stdout or "") + (stderr or "")
        return f"Timed out after {timeout}s.\n…{partial[-7_900:]}"
    except OSError as exc:
        return f"Could not run the command: {exc}"

    header = f"exit_code: {result.returncode}\n"
    output = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    if len(header) + len(output) > 8_000:
        output = "output truncated (ending preserved):\n…" + output[-7_900:]
    return header + output


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    auto_approve: bool,
    timeout: int,
) -> str:
    if name == "run_bash":
        return run_bash(arguments["cmd"], auto_approve=auto_approve, timeout=timeout)
    return f"Unknown tool: {name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cerebras gpt-oss-120b agent with a Bash tool."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="How many files are in the current directory?",
        help="Task for the model.",
    )
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default="medium",
    )
    parser.add_argument("--max-completion-tokens", type=int, default=8_192)
    parser.add_argument("--max-tool-rounds", type=int, default=20)
    parser.add_argument("--command-timeout", type=int, default=60)
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Run commands without asking for confirmation (dangerous).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)
    if not os.environ.get("CEREBRAS_API_KEY"):
        print(
            "Error: CEREBRAS_API_KEY is missing from the environment or .env file.",
            file=sys.stderr,
        )
        return 2

    client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])
    messages: list[Any] = [
        {
            "role": "system",
            "content": (
                "You are an assistant with access to a Bash terminal. "
                "Use run_bash when you need to inspect or modify the environment. "
                "Explain the final result clearly and concisely."
            ),
        },
        {"role": "user", "content": args.prompt},
    ]

    for tool_round in range(args.max_tool_rounds + 1):
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            parallel_tool_calls=True,
            reasoning_effort=args.reasoning_effort,
            max_completion_tokens=args.max_completion_tokens,
        )
        message = response.choices[0].message

        if message.content:
            print(message.content)

        if not message.tool_calls:
            return 0

        if tool_round == args.max_tool_rounds:
            break

        # Preserve the complete message, including reasoning and tool_calls.
        messages.append(message.model_dump(exclude_none=True))

        for call in message.tool_calls:
            name = call.function.name
            raw_arguments = call.function.arguments
            print(f"→ {name}({raw_arguments})")

            try:
                arguments = json.loads(raw_arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("arguments are not a JSON object")
                result = execute_tool(
                    name,
                    arguments,
                    auto_approve=args.auto_approve,
                    timeout=args.command_timeout,
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                result = f"Invalid arguments for {name}: {exc}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                }
            )

    print(
        f"Error: reached the limit of {args.max_tool_rounds} tool rounds.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
