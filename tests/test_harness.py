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
        "approval_policy": "deny",
        "sandbox": "disposable",
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
        self.assertEqual(args.approval_policy, "ask")
        self.assertEqual(args.sandbox, "disposable")
        self.assertEqual(args.compaction_threshold, 0.70)
        self.assertEqual(args.keep_recent_turns, 2)
        self.assertEqual(args.relevant_files, 8)
        self.assertEqual(args.session_file, Path(".harness/session.json"))
        self.assertEqual(args.progress_file, Path(".harness/progress.md"))
        self.assertFalse(args.resume)

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

    def test_compaction_threshold_must_be_a_ratio(self):
        for value in ("0", "1", "nan"):
            with self.subTest(value=value), redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                harness.parse_args(["--compaction-threshold", value])


class ToolTests(unittest.TestCase):
    def test_declares_strict_input_schemas_for_agent_tools(self):
        names = {tool["function"]["name"] for tool in harness.TOOLS}

        self.assertEqual(
            names,
            {
                "calculator",
                "get_current_time",
                "echo",
                "list_files",
                "read_file",
                "search_text",
                "git_diff",
                "run_tests",
                "apply_patch",
                "run_shell",
            },
        )
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


class RepositoryToolTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name).resolve()
        self.root_patch = patch("harness._repository_root", return_value=self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def write_text(self, relative_path, content):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_list_files_is_sorted_bounded_and_excludes_sensitive_paths(self):
        self.write_text("src/z.py", "z")
        self.write_text("src/a.py", "a")
        self.write_text(".env", "SECRET=value")
        self.write_text(".env.example", "SECRET=example")
        self.write_text(".git/config", "git")
        self.write_text(".venv/ignored.py", "ignored")

        result = harness.list_files({"path": "."})

        self.assertEqual(
            result["files"], [".env.example", "src/a.py", "src/z.py"]
        )
        self.assertFalse(result["truncated"])
        self.assertEqual(result["total_count"], 3)

    def test_list_files_reports_truncation(self):
        with patch("harness.MAX_LISTED_FILES", 2):
            for name in ("c.py", "a.py", "b.py"):
                self.write_text(name, name)

            result = harness.list_files({"path": "."})

        self.assertEqual(result["files"], ["a.py", "b.py"])
        self.assertTrue(result["truncated"])
        self.assertEqual(result["total_count"], 3)

    def test_read_file_selects_a_bounded_line_range(self):
        self.write_text("module.py", "one\ntwo\nthree\nfour\n")

        result = harness.read_file(
            {"path": "module.py", "start_line": 2, "max_lines": 2}
        )

        self.assertEqual(result["content"], "two\nthree")
        self.assertEqual(result["end_line"], 3)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["next_start_line"], 4)

    def test_read_file_rejects_traversal_sensitive_files_and_oversized_ranges(self):
        self.write_text(".env", "SECRET=value")
        outside_dir = tempfile.TemporaryDirectory()
        self.addCleanup(outside_dir.cleanup)
        outside_path = Path(outside_dir.name) / "outside.py"
        outside_path.write_text("outside", encoding="utf-8")
        (self.root / "escape.py").symlink_to(outside_path)

        invalid_arguments = [
            {"path": "../outside.py", "start_line": 1, "max_lines": 20},
            {"path": "escape.py", "start_line": 1, "max_lines": 20},
            {"path": ".env", "start_line": 1, "max_lines": 20},
            {
                "path": "missing.py",
                "start_line": 1,
                "max_lines": harness.MAX_READ_LINES + 1,
            },
        ]
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments), self.assertRaises(
                harness.ToolArgumentError
            ):
                harness.read_file(arguments)

    def test_read_file_truncates_an_extremely_long_line(self):
        self.write_text("large.txt", "x" * (harness.MAX_TOOL_OUTPUT_CHARS + 10))

        result = harness.read_file(
            {"path": "large.txt", "start_line": 1, "max_lines": 1}
        )

        self.assertEqual(len(result["content"]), harness.MAX_TOOL_OUTPUT_CHARS)
        self.assertTrue(result["truncated"])
        self.assertIsNone(result["next_start_line"])

    def test_search_text_returns_locations_and_stops_at_its_limit(self):
        content = "".join(f"needle {index}\n" for index in range(55))
        self.write_text("many.txt", content)

        result = harness.search_text({"query": "needle", "path": "."})

        self.assertEqual(result["returned_count"], harness.MAX_SEARCH_RESULTS)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["matches"][0]["line"], 1)
        self.assertEqual(result["matches"][-1]["line"], 50)

    def test_search_text_skips_binary_and_large_files(self):
        (self.root / "binary.dat").write_bytes(b"needle\xff")
        (self.root / "large.txt").write_bytes(
            b"needle" + b"x" * harness.MAX_SEARCH_FILE_BYTES
        )
        self.write_text("source.py", "value = 'needle'\n")

        result = harness.search_text({"query": "needle", "path": "."})

        self.assertEqual(result["returned_count"], 1)
        self.assertEqual(result["matches"][0]["path"], "source.py")
        self.assertEqual(result["skipped_files"], 2)

    @patch("harness.subprocess.run")
    def test_git_diff_uses_a_non_shell_command_and_truncates_output(self, run):
        run.return_value = SimpleNamespace(
            returncode=0,
            stdout="\n".join(f"diff line {index}" for index in range(500)),
        )

        result = harness.git_diff({"path": "deleted.py"})

        command = run.call_args.args[0]
        self.assertEqual(
            command[:5],
            ["git", "diff", "--no-ext-diff", "--no-color", "HEAD"],
        )
        self.assertEqual(command[-2:], ["--", "deleted.py"])
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertTrue(result["truncated"])
        self.assertLessEqual(
            len(result["output"].splitlines()), harness.MAX_PROCESS_OUTPUT_LINES
        )

    @patch("harness._run_bounded_process")
    def test_run_tests_uses_the_repository_virtual_environment(self, run):
        python = self.root / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.touch()
        run.return_value = {
            "exit_code": 1,
            "output": "failure details",
            "truncated": False,
            "timed_out": False,
        }

        result = harness.run_tests({})

        command = run.call_args.args[0]
        self.assertEqual(
            command,
            [
                str(python),
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-v",
            ],
        )
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(result["output"], "failure details")
        self.assertEqual(
            run.call_args.kwargs["timeout_seconds"], harness.SHELL_TIMEOUT_SECONDS
        )
        self.assertNotEqual(run.call_args.kwargs["cwd"], self.root)
        self.assertEqual(result["sandbox"], "disposable repository copy")

    @patch("harness._run_bounded_process")
    def test_run_tests_reports_timeouts_as_bounded_results(self, run):
        python = self.root / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.touch()
        run.return_value = {
            "exit_code": None,
            "output": "partial",
            "truncated": False,
            "timed_out": True,
        }

        result = harness.run_tests({})

        self.assertTrue(result["timed_out"])
        self.assertIsNone(result["exit_code"])
        self.assertEqual(result["output"], "partial")


class EditingAndShellToolTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name).resolve()
        self.root_patch = patch("harness._repository_root", return_value=self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def write_text(self, relative_path, content):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_apply_patch_updates_adds_and_deletes_text_files(self):
        self.write_text("update.txt", "old\nkeep\n")
        self.write_text("delete.txt", "gone\n")
        patch_text = """*** Begin Patch
*** Update File: update.txt
@@
-old
+new
 keep
*** Add File: nested/added.txt
+created
*** Delete File: delete.txt
-gone
*** End Patch"""

        result = harness.apply_patch({"patch": patch_text})

        self.assertTrue(result["applied"])
        self.assertEqual(result["file_count"], 3)
        self.assertEqual((self.root / "update.txt").read_text(), "new\nkeep\n")
        self.assertEqual((self.root / "nested/added.txt").read_text(), "created\n")
        self.assertFalse((self.root / "delete.txt").exists())

    def test_apply_patch_validates_every_hunk_before_writing(self):
        self.write_text("one.txt", "one\n")
        self.write_text("two.txt", "two\n")
        patch_text = """*** Begin Patch
*** Update File: one.txt
@@
-one
+changed
*** Update File: two.txt
@@
-missing
+changed
*** End Patch"""

        with self.assertRaisesRegex(harness.ToolArgumentError, "does not match"):
            harness.apply_patch({"patch": patch_text})

        self.assertEqual((self.root / "one.txt").read_text(), "one\n")
        self.assertEqual((self.root / "two.txt").read_text(), "two\n")

    def test_apply_patch_rejects_paths_outside_the_repository(self):
        patch_text = """*** Begin Patch
*** Add File: ../outside.txt
+unsafe
*** End Patch"""

        with self.assertRaisesRegex(harness.ToolArgumentError, "must not contain"):
            harness.apply_patch({"patch": patch_text})

    def test_apply_patch_rejects_symlink_aliases(self):
        self.write_text("target.txt", "safe\n")
        (self.root / "alias.txt").symlink_to("target.txt")
        patch_text = """*** Begin Patch
*** Update File: alias.txt
@@
-safe
+changed
*** End Patch"""

        with self.assertRaisesRegex(harness.ToolArgumentError, "symlinks"):
            harness.apply_patch({"patch": patch_text})

        self.assertEqual((self.root / "target.txt").read_text(), "safe\n")

    def test_apply_patch_accepts_a_unified_diff(self):
        self.write_text("example.txt", "before\n")
        patch_text = """diff --git a/example.txt b/example.txt
--- a/example.txt
+++ b/example.txt
@@ -1 +1 @@
-before
+after
"""

        result = harness.apply_patch({"patch": patch_text})

        self.assertTrue(result["applied"])
        self.assertEqual((self.root / "example.txt").read_text(), "after\n")

    @patch("harness._run_bounded_process")
    def test_run_shell_uses_a_disposable_repository_copy(self, run_process):
        self.write_text("tracked.txt", "content\n")
        captured = {}

        def run_in_sandbox(command, **kwargs):
            captured["command"] = command
            captured["cwd"] = kwargs["cwd"]
            (kwargs["cwd"] / "side-effect.txt").write_text("temporary")
            return {
                "exit_code": 0,
                "output": "ok",
                "truncated": False,
                "timed_out": False,
            }

        run_process.side_effect = run_in_sandbox

        with patch.dict("harness.os.environ", {"CEREBRAS_API_KEY": "secret"}):
            result = harness.run_shell({"command": "git status --short"})

        self.assertEqual(result["sandbox"], "disposable repository copy")
        self.assertEqual(result["output"], "ok")
        self.assertFalse((self.root / "side-effect.txt").exists())
        self.assertNotEqual(captured["cwd"], self.root)
        self.assertEqual(captured["command"][0], "git")
        self.assertNotIn("CEREBRAS_API_KEY", run_process.call_args.kwargs["env"])

    @patch("harness.shutil.copytree")
    def test_run_shell_rejects_non_allowlisted_commands_before_copying(self, copytree):
        invalid = (
            "rm -rf .",
            "git status --short; rm file",
            "python -c 'print(1)'",
            "rg needle ../outside",
        )

        for command in invalid:
            with self.subTest(command=command), self.assertRaises(
                harness.ToolArgumentError
            ):
                harness.run_shell({"command": command})

        copytree.assert_not_called()

    def test_execute_tool_denies_or_grants_side_effecting_calls_by_policy(self):
        denied_patch = """*** Begin Patch
*** Add File: denied.txt
+no
*** End Patch"""
        denied = harness.execute_tool(
            "apply_patch",
            json.dumps({"patch": denied_patch}),
        )

        self.assertFalse(denied["ok"])
        self.assertEqual(denied["approval"], "denied")
        self.assertEqual(denied["error"]["type"], "ApprovalDeniedError")
        self.assertFalse((self.root / "denied.txt").exists())

        allowed_patch = denied_patch.replace("denied.txt", "allowed.txt")
        allowed = harness.execute_tool(
            "apply_patch",
            json.dumps({"patch": allowed_patch}),
            approval_policy="ask",
            approval_callback=lambda _name, _arguments: True,
        )

        self.assertTrue(allowed["ok"])
        self.assertEqual(allowed["approval"], "granted")
        self.assertEqual((self.root / "allowed.txt").read_text(), "no\n")


class ContextEngineeringTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name).resolve()
        self.root_patch = patch("harness._repository_root", return_value=self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def write_text(self, relative_path, content):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_loads_global_root_and_nested_agents_in_precedence_order(self):
        global_path = self.write_text("global.md", "global rules")
        self.write_text("AGENTS.md", "root rules")
        self.write_text("src/AGENTS.md", "nested rules")
        self.write_text("src/module.py", "value = 1")

        documents = harness.load_instruction_documents(
            self.root,
            global_path,
            ["src/module.py"],
        )

        self.assertEqual(
            [label for label, _content in documents],
            [str(global_path), "AGENTS.md", "src/AGENTS.md"],
        )
        prompt = harness.build_system_prompt("base", documents)
        self.assertLess(prompt.index("global rules"), prompt.index("root rules"))
        self.assertLess(prompt.index("root rules"), prompt.index("nested rules"))
        read_result = harness.read_file(
            {"path": "src/module.py", "start_line": 1, "max_lines": 10}
        )
        self.assertEqual(
            read_result["applicable_instruction_files"],
            ["AGENTS.md", "src/AGENTS.md"],
        )

    def test_selects_relevant_paths_from_names_without_reading_contents(self):
        self.write_text("src/session_store.py", "unrelated")
        self.write_text("tests/test_session_store.py", "unrelated")
        self.write_text("docs/security.md", "session persistence")

        selected = harness.select_relevant_files(
            "Fix session_store.py and its tests", limit=2
        )

        self.assertEqual(
            set(selected),
            {"src/session_store.py", "tests/test_session_store.py"},
        )

    def test_summarizes_oversized_tool_output_for_model_context(self):
        result = {"ok": True, "result": {"output": "a" * 10_000}}

        content, summarized = harness.summarize_tool_output(result, maximum=1_000)
        payload = json.loads(content)

        self.assertTrue(summarized)
        self.assertLessEqual(len(content), 1_000)
        self.assertTrue(payload["context_summary"])
        self.assertEqual(payload["original_characters"], len(json.dumps(result)))
        self.assertIn("Middle omitted", payload["notice"])

    def test_compacts_old_turns_and_preserves_recent_turn_verbatim(self):
        messages = [{"role": "system", "content": "rules"}]
        for index in range(3):
            messages.extend(
                [
                    {"role": "user", "content": f"question {index}"},
                    {"role": "assistant", "content": f"answer {index}"},
                ]
            )

        result = harness.compact_history(messages, keep_recent_turns=1)

        self.assertIsNotNone(result)
        self.assertEqual(messages[0], {"role": "system", "content": "rules"})
        self.assertIn("Compacted conversation", messages[1]["content"])
        self.assertIn("question 0", messages[1]["content"])
        self.assertEqual(
            messages[-2:],
            [
                {"role": "user", "content": "question 2"},
                {"role": "assistant", "content": "answer 2"},
            ],
        )

    def test_compaction_waits_until_threshold_is_reached(self):
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "new"},
        ]

        untouched = harness.maybe_compact_history(
            messages, context_window=10_000, keep_recent_turns=1
        )
        compacted = harness.maybe_compact_history(
            messages,
            context_window=100,
            threshold=0.5,
            keep_recent_turns=1,
            observed_prompt_tokens=80,
        )

        self.assertIsNone(untouched)
        self.assertIsNotNone(compacted)
        self.assertIn("old", messages[1]["content"])

    def test_session_store_round_trips_and_rejects_other_repositories(self):
        store = harness.SessionStore(Path("state/session.json"), self.root)
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "continue"},
        ]
        store.save("session-7", messages)

        session_id, loaded = store.load()

        self.assertEqual(session_id, "session-7")
        self.assertEqual(loaded, messages)
        self.assertEqual(store.path.stat().st_mode & 0o777, 0o600)
        payload = json.loads(store.path.read_text(encoding="utf-8"))
        payload["repository_root"] = str(self.root / "other")
        store.path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "different repository"):
            store.load()

    def test_runtime_state_refuses_to_overwrite_unrelated_files(self):
        self.write_text("source.json", '{"application": true}')
        self.write_text("notes.md", "# Project notes\n")

        store = harness.SessionStore(Path("source.json"), self.root)

        with self.assertRaisesRegex(RuntimeError, "non-session"):
            store.ensure_safe_to_replace()
        with self.assertRaisesRegex(ValueError, "non-progress"):
            harness.ProgressTracker(Path("notes.md"), "session-1")
        self.assertEqual(
            (self.root / "source.json").read_text(), '{"application": true}'
        )
        self.assertEqual(
            (self.root / "notes.md").read_text(), "# Project notes\n"
        )

    def test_progress_tracker_preserves_objective_across_instances(self):
        tracker = harness.ProgressTracker(Path("state/progress.md"), "session-1")
        tracker.start("Migrate the session persistence layer")
        tracker.record_tool("read_file", True)

        resumed = harness.ProgressTracker(Path("state/progress.md"), "session-1")
        resumed.complete("Migration still needs verification")
        content = resumed.path.read_text(encoding="utf-8")

        self.assertIn("Migrate the session persistence layer", content)
        self.assertIn("Tool `read_file` completed", content)
        self.assertIn("Migration still needs verification", content)

    def test_run_turn_persists_summarized_context_and_progress(self):
        logger = harness.JsonlEventLogger(self.root / "events.jsonl", "session-9")
        store = harness.SessionStore(Path("state/session.json"), self.root)
        progress = harness.ProgressTracker(
            Path("state/progress.md"), "session-9", load_existing=False
        )
        large_text = "x" * 10_000
        client = Mock()
        client.chat.completions.create.side_effect = [
            make_response(
                content=None,
                tool_calls=[
                    make_tool_call(
                        name="echo",
                        arguments=json.dumps({"text": large_text}),
                    )
                ],
            ),
            make_response(content="Done"),
        ]
        messages = [{"role": "system", "content": "rules"}]

        with redirect_stdout(io.StringIO()):
            harness.run_turn(
                client,
                messages,
                "Echo a large value",
                make_args(context_window=100_000),
                logger,
                store,
                progress,
            )

        tool_payload = json.loads(messages[3]["content"])
        self.assertTrue(tool_payload["context_summary"])
        self.assertLessEqual(
            len(messages[3]["content"]), harness.MAX_CONTEXT_TOOL_OUTPUT_CHARS
        )
        self.assertEqual(store.load(), ("session-9", messages))
        self.assertIn("Turn completed: Done", progress.path.read_text())

    def test_main_resumes_session_identity_and_conversation(self):
        session_path = self.root / "session.json"
        progress_path = self.root / "progress.md"
        log_path = self.root / "events.jsonl"
        client = Mock()
        client.chat.completions.create.side_effect = [
            make_response(content="first answer"),
            make_response(content="second answer"),
        ]
        common = [
            "--session-file",
            str(session_path),
            "--progress-file",
            str(progress_path),
            "--log-file",
            str(log_path),
            "--approval-policy",
            "deny",
        ]

        with patch.dict(harness.os.environ, {"CEREBRAS_API_KEY": "key"}), patch(
            "harness.Cerebras", return_value=client
        ), redirect_stdout(io.StringIO()):
            first_exit = harness.main([*common, "first question"])
            second_exit = harness.main([*common, "--resume", "second question"])

        self.assertEqual((first_exit, second_exit), (0, 0))
        session_id, messages = harness.SessionStore(
            session_path, self.root
        ).load()
        self.assertEqual(
            [message["content"] for message in messages if message["role"] == "user"],
            ["first question", "second question"],
        )
        events = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
        ]
        starts = [event for event in events if event["event"] == "session_started"]
        self.assertEqual([event["resumed"] for event in starts], [False, True])
        self.assertTrue(all(event["session_id"] == session_id for event in starts))


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

    @patch("harness._prompt_for_tool_approval", return_value=False)
    def test_denied_approval_is_logged_and_returned_to_the_model(self, _approval):
        self.args.approval_policy = "ask"
        client = Mock()
        client.chat.completions.create.side_effect = [
            make_response(
                content=None,
                tool_calls=[
                    make_tool_call(
                        name="run_shell",
                        arguments='{"command":"git status --short"}',
                    )
                ],
            ),
            make_response(content="The command was not approved."),
        ]
        messages = [{"role": "system", "content": "test"}]

        with redirect_stdout(io.StringIO()):
            harness.run_turn(client, messages, "Check status", self.args, self.logger)

        tool_result = json.loads(messages[3]["content"])
        self.assertFalse(tool_result["ok"])
        self.assertEqual(tool_result["approval"], "denied")
        self.assertEqual(tool_result["error"]["type"], "ApprovalDeniedError")
        approval_events = [
            event
            for event in self.read_events()
            if event["event"].startswith("tool_approval_")
        ]
        self.assertEqual(
            [event["event"] for event in approval_events],
            ["tool_approval_requested", "tool_approval_resolved"],
        )
        self.assertFalse(approval_events[-1]["approved"])

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
