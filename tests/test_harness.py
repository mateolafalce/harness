import argparse
import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import harness


class ParseArgsTests(unittest.TestCase):
    def test_prompt_is_optional_for_interactive_mode(self):
        args = harness.parse_args([])

        self.assertIsNone(args.prompt)

    def test_prompt_and_flags_are_parsed(self):
        args = harness.parse_args(["--auto-approve", "inspect the project"])

        self.assertEqual(args.prompt, "inspect the project")
        self.assertTrue(args.auto_approve)


class InteractiveCliTests(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace()
        self.messages = [{"role": "system", "content": "test"}]

    @patch("harness.sys.stdin.isatty", return_value=False)
    def test_non_interactive_input_requires_a_prompt(self, _isatty):
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            exit_code = harness.interactive_cli(object(), self.messages, self.args)

        self.assertEqual(exit_code, 2)
        self.assertIn("provide a prompt", stderr.getvalue())

    @patch("harness.sys.stdin.isatty", return_value=True)
    @patch("builtins.input", side_effect=["/help", "/exit"])
    def test_help_then_exit_does_not_call_the_api(self, _input, _isatty):
        client = unittest.mock.Mock()

        exit_code = harness.interactive_cli(client, self.messages, self.args)

        self.assertEqual(exit_code, 0)
        client.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
