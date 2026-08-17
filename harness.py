#!/usr/bin/env python3
"""Command-line harness for gpt-oss-120b on Cerebras with shell access."""

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cerebras gpt-oss-120b agent with a Bash tool."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Task for the model. Omit it to start interactive mode.",
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
    return parser.parse_args(argv)


def run_turn(
    client: Cerebras,
    messages: list[Any],
    prompt: str,
    args: argparse.Namespace,
) -> int:
    """Send one user turn, including any tool calls requested by the model."""
    messages.append({"role": "user", "content": prompt})

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

        # Keep the complete assistant message so interactive conversations retain
        # context and tool calls remain paired with their results.
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return 0

        if tool_round == args.max_tool_rounds:
            break

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


def interactive_cli(
    client: Cerebras,
    messages: list[Any],
    args: argparse.Namespace,
) -> int:
    """Run a small read-eval-print loop while preserving conversation history."""
    if not sys.stdin.isatty():
        print(
            "Error: provide a prompt when standard input is not interactive.",
            file=sys.stderr,
        )
        return 2

    print("Harness interactive mode. Type /help for help or /exit to quit.")
    while True:
        try:
            prompt = input("\nyou> ").strip()
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
        if prompt == "/help":
            print("Enter a task for the agent. Commands: /help, /exit, /quit")
            continue

        try:
            exit_code = run_turn(client, messages, prompt, args)
        except KeyboardInterrupt:
            print("\nRequest interrupted.", file=sys.stderr)
            continue
        except Exception as exc:
            print(f"Error communicating with Cerebras: {exc}", file=sys.stderr)
            continue

        if exit_code:
            return exit_code


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
    ]
    if args.prompt is None:
        return interactive_cli(client, messages, args)

    try:
        return run_turn(client, messages, args.prompt, args)
    except KeyboardInterrupt:
        print("\nRequest interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error communicating with Cerebras: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
