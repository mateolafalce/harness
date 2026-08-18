import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import harness


def make_args(**overrides):
    values = {
        "model": "test-model",
        "reasoning_effort": "medium",
        "max_completion_tokens": 100,
        "context_window": 1_000,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def make_tool_call(call_id="call-1", name="calculator", arguments='{"expression":"2+2"}'):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def make_response(
    content="Hello",
    prompt_tokens=100,
    completion_tokens=25,
    cached_tokens=40,
    tool_calls=None,
):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(
        message=message,
        finish_reason="tool_calls" if tool_calls else "stop",
    )
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(
        id="response-1",
        model="test-model",
        choices=[choice],
        usage=usage,
    )


class ParseArgsTests(unittest.TestCase):
    def test_prompt_is_optional_for_interactive_mode(self):
        args = harness.parse_args([])

        self.assertIsNone(args.prompt)
        self.assertEqual(args.context_window, 131_072)
        self.assertEqual(args.log_file, Path("events.jsonl"))
        self.assertEqual(args.max_turns, 8)
        self.assertEqual(args.timeout, 30.0)

    def test_prompt_and_flags_are_parsed(self):
        args = harness.parse_args(
            [
                "--context-window",
                "4096",
                "--log-file",
                "run.jsonl",
                "--max-turns",
                "3",
                "--timeout",
                "2.5",
                "question",
            ]
        )

        self.assertEqual(args.prompt, "question")
        self.assertEqual(args.context_window, 4096)
        self.assertEqual(args.log_file, Path("run.jsonl"))
        self.assertEqual(args.max_turns, 3)
        self.assertEqual(args.timeout, 2.5)

    def test_context_window_must_be_positive(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            harness.parse_args(["--context-window", "0"])

    def test_limits_must_be_positive(self):
        for flag, value in (("--max-turns", "0"), ("--timeout", "nan")):
            with self.subTest(flag=flag), redirect_stderr(io.StringIO()), self.assertRaises(
                SystemExit
            ):
                harness.parse_args([flag, value])


class ToolTests(unittest.TestCase):
    def test_declares_three_strict_input_schemas(self):
        names = {tool["function"]["name"] for tool in harness.TOOLS}

        self.assertEqual(names, {"calculator", "get_current_time", "echo"})
        for tool in harness.TOOLS:
            function = tool["function"]
            self.assertTrue(function["strict"])
            self.assertFalse(function["parameters"]["additionalProperties"])

    def test_calculator_evaluates_arithmetic_without_eval(self):
        self.assertEqual(
            harness.calculator({"expression": "(2 + 3) * 4 ** 2"}),
            {"value": 80},
        )

    def test_calculator_rejects_code_and_extra_arguments(self):
        invalid = [
            {"expression": "__import__('os').getcwd()"},
            {"expression": "2 + 2", "precision": 2},
        ]

        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(
                harness.ToolArgumentError
            ):
                harness.calculator(arguments)

    def test_current_time_validates_iana_timezone(self):
        result = harness.get_current_time({"timezone": "UTC"})

        self.assertEqual(result["timezone"], "UTC")
        self.assertIn("+00:00", result["iso8601"])
        with self.assertRaisesRegex(harness.ToolArgumentError, "unknown IANA"):
            harness.get_current_time({"timezone": "Not/A_Zone"})

    def test_tool_errors_are_returned_as_results(self):
        malformed = harness.execute_tool("calculator", "not-json")
        unknown = harness.execute_tool("delete_everything", "{}")

        self.assertFalse(malformed["ok"])
        self.assertEqual(malformed["error"]["type"], "ToolArgumentError")
        self.assertFalse(unknown["ok"])
        self.assertIn("unknown tool", unknown["error"]["message"])


class MetricsTests(unittest.TestCase):
    def test_metrics_include_context_utilization(self):
        metrics = harness.response_metrics(make_response(), 0.25, 1_000)

        self.assertEqual(metrics["prompt_tokens"], 100)
        self.assertEqual(metrics["cached_tokens"], 40)
        self.assertEqual(metrics["completion_tokens"], 25)
        self.assertEqual(metrics["total_tokens"], 125)
        self.assertEqual(metrics["latency_ms"], 250.0)
        self.assertEqual(metrics["context_used_percent"], 12.5)

    def test_combined_metrics_sum_cached_tokens_from_all_model_calls(self):
        metrics = harness._combined_metrics(
            [make_response(cached_tokens=40), make_response(cached_tokens=35)],
            0.5,
            1_000,
        )

        self.assertEqual(metrics["cached_tokens"], 75)
        self.assertEqual(metrics["model_calls"], 2)


class PrintResponseTests(unittest.TestCase):
    def test_renders_markdown_instead_of_printing_source_syntax(self):
        stdout = io.StringIO()
        metrics = {
            "prompt_tokens": 10,
            "cached_tokens": 4,
            "completion_tokens": 5,
            "total_tokens": 15,
            "latency_ms": 25.0,
            "context_used_percent": 1.5,
        }

        with redirect_stdout(stdout):
            harness.print_response("# Title\n\n- **bold** item", metrics)

        output = stdout.getvalue()
        self.assertIn("assistant>", output)
        self.assertIn("Title", output)
        self.assertIn("bold item", output)
        self.assertNotIn("# Title", output)
        self.assertNotIn("**bold**", output)
        self.assertIn("metrics>", output)
        self.assertIn("cached=4 tokens", output)


class RunTurnTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.log_path = Path(self.temp_dir.name) / "events.jsonl"
        self.logger = harness.JsonlEventLogger(self.log_path, "session-1")
        self.args = make_args()

    def read_events(self):
        return [json.loads(line) for line in self.log_path.read_text().splitlines()]

    def test_calls_api_with_tools_and_keeps_history(self):
        client = Mock()
        client.chat.completions.create.return_value = make_response()
        messages = [{"role": "system", "content": "test"}]

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = harness.run_turn(
                client, messages, "Hi", self.args, self.logger
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "test"},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
        )
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["tools"], harness.TOOLS)
        self.assertEqual(kwargs["tool_choice"], "auto")
        self.assertGreater(kwargs["timeout"], 0)
        self.assertIn("assistant> Hello", stdout.getvalue())
        self.assertIn("context=12.5%", stdout.getvalue())

        events = self.read_events()
        self.assertEqual(events[-1]["event"], "agent_loop_completed")
        self.assertEqual(events[-1]["metrics"]["total_tokens"], 125)
        self.assertEqual(events[-1]["metrics"]["cached_tokens"], 40)
        decisions = [event for event in events if event["event"] == "agent_decision"]
        self.assertEqual(decisions[-1]["decision"], "final_response")

    def test_api_error_is_logged_and_failed_turn_is_removed_from_history(self):
        client = Mock()
        client.chat.completions.create.side_effect = RuntimeError("offline")
        messages = [{"role": "system", "content": "test"}]

        with self.assertRaisesRegex(RuntimeError, "offline"):
            harness.run_turn(client, messages, "Hi", self.args, self.logger)

        self.assertEqual(messages, [{"role": "system", "content": "test"}])
        events = self.read_events()
        self.assertEqual(events[-2]["event"], "api_error")
        self.assertEqual(events[-2]["error_type"], "RuntimeError")
        self.assertEqual(events[-1]["event"], "agent_loop_failed")
        self.assertTrue(events[-1]["history_rolled_back"])

    def test_executes_tool_and_correlates_result_by_call_id(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            make_response(content=None, tool_calls=[make_tool_call(call_id="calc-7")]),
            make_response(content="The answer is 4."),
        ]
        messages = [{"role": "system", "content": "test"}]

        with redirect_stdout(io.StringIO()):
            exit_code = harness.run_turn(
                client, messages, "What is 2+2?", self.args, self.logger
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        self.assertEqual(messages[2]["tool_calls"][0]["id"], "calc-7")
        self.assertEqual(messages[3]["role"], "tool")
        self.assertEqual(messages[3]["tool_call_id"], "calc-7")
        self.assertEqual(json.loads(messages[3]["content"])["result"], {"value": 4})
        self.assertEqual(messages[4]["content"], "The answer is 4.")

        tool_events = [
            event
            for event in self.read_events()
            if event["event"] == "tool_call_completed"
        ]
        self.assertEqual(tool_events[0]["call_id"], "calc-7")
        self.assertTrue(tool_events[0]["success"])

    def test_tool_failure_is_sent_to_model_and_loop_continues(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            make_response(
                content=None,
                tool_calls=[make_tool_call(arguments='{"expression":"1/0"}')],
            ),
            make_response(content="I could not calculate that."),
        ]
        messages = [{"role": "system", "content": "test"}]

        with redirect_stdout(io.StringIO()):
            harness.run_turn(client, messages, "Divide by zero", self.args, self.logger)

        result = json.loads(messages[3]["content"])
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["type"], "ToolArgumentError")
        self.assertEqual(messages[-1]["content"], "I could not calculate that.")

    def test_max_turns_stops_loop_and_rolls_back_history(self):
        self.args.max_turns = 1
        client = Mock()
        client.chat.completions.create.return_value = make_response(
            content=None, tool_calls=[make_tool_call()]
        )
        messages = [{"role": "system", "content": "test"}]

        with self.assertRaises(harness.MaxTurnsExceededError):
            harness.run_turn(client, messages, "Keep going", self.args, self.logger)

        self.assertEqual(client.chat.completions.create.call_count, 1)
        self.assertEqual(messages, [{"role": "system", "content": "test"}])
        decisions = [
            event
            for event in self.read_events()
            if event["event"] == "agent_decision"
        ]
        self.assertEqual(decisions[-1]["decision"], "max_turns_exceeded")

    @patch(
        "harness._remaining_seconds",
        side_effect=[1.0, harness.AgentTimeoutError("agent loop timed out")],
    )
    def test_timeout_stops_loop_and_rolls_back_history(self, _remaining):
        client = Mock()
        client.chat.completions.create.return_value = make_response()
        messages = [{"role": "system", "content": "test"}]

        with self.assertRaises(harness.AgentTimeoutError):
            harness.run_turn(client, messages, "Hi", self.args, self.logger)

        self.assertEqual(messages, [{"role": "system", "content": "test"}])
        self.assertEqual(self.read_events()[-1]["error_type"], "AgentTimeoutError")


class InteractiveCliTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        path = Path(self.temp_dir.name) / "events.jsonl"
        self.logger = harness.JsonlEventLogger(path, "session-1")
        self.args = make_args()
        self.messages = [{"role": "system", "content": "test"}]

    @patch("harness.sys.stdin.isatty", return_value=False)
    def test_non_interactive_input_requires_a_prompt(self, _isatty):
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            exit_code = harness.interactive_cli(
                object(), self.messages, self.args, self.logger
            )

        self.assertEqual(exit_code, 2)
        self.assertIn("provide a prompt", stderr.getvalue())

    @patch("harness.sys.stdin.isatty", return_value=True)
    @patch("builtins.input", side_effect=["/help", "/exit"])
    def test_help_then_exit_does_not_call_the_api(self, _input, _isatty):
        client = Mock()
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = harness.interactive_cli(
                client, self.messages, self.args, self.logger
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("/clear", stdout.getvalue())
        client.chat.completions.create.assert_not_called()

    @patch("harness.sys.stdin.isatty", return_value=True)
    @patch("builtins.input", side_effect=["/clear", "/exit"])
    def test_clear_removes_conversation_but_keeps_system_prompt(
        self, _input, _isatty
    ):
        client = Mock()
        self.messages.extend(
            [
                {"role": "user", "content": "Remember this"},
                {"role": "assistant", "content": "I will"},
            ]
        )
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = harness.interactive_cli(
                client, self.messages, self.args, self.logger
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(self.messages, [{"role": "system", "content": "test"}])
        self.assertIn("2 messages removed", stdout.getvalue())
        client.chat.completions.create.assert_not_called()

        events = [
            json.loads(line)
            for line in self.logger.path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(events[-1]["event"], "conversation_context_cleared")
        self.assertEqual(events[-1]["removed_message_count"], 2)
        self.assertEqual(events[-1]["history_message_count"], 1)

    def test_clear_context_without_system_prompt_removes_every_message(self):
        messages = [{"role": "user", "content": "Hello"}]

        removed_count = harness.clear_conversation_context(messages, self.logger)

        self.assertEqual(removed_count, 1)
        self.assertEqual(messages, [])


if __name__ == "__main__":
    unittest.main()
