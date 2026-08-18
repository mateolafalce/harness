# Repository Guidelines

## Project Structure & Module Organization

`harness.py` contains the complete command-line application: argument parsing, the Cerebras agent loop, tool implementations, JSONL logging, and interactive-mode handling. Tests live in `tests/test_harness.py` and mirror those responsibilities with `unittest` classes. Runtime dependencies are pinned by compatible ranges in `requirements.txt`. Use `.env.example` as the configuration template; local `.env` files, generated `events.jsonl` logs, caches, and `.venv/` are development artifacts and must not be committed.

## Build, Test, and Development Commands

Always use the repository virtual environment:

```bash
python -m venv .venv                    # Create it if absent
.venv/bin/pip install -r requirements.txt
.venv/bin/python harness.py --help      # Inspect CLI options
.venv/bin/python harness.py             # Start interactive mode
.venv/bin/python -m unittest discover -s tests -v
```

Copy `.env.example` to `.env` and set `CEREBRAS_API_KEY` before making live model requests. The unit suite mocks API interactions and should not require network access.

## Coding Style & Naming Conventions

Follow standard PEP 8 with four-space indentation, type annotations, and short docstrings for public or non-obvious behavior. Use `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Keep tool schemas strict and validate all model-supplied arguments before execution. No formatter or linter is currently configured; preserve the existing import grouping and line length (approximately 88 characters).

## Testing Guidelines

Use Python's built-in `unittest` framework and `unittest.mock` for external services. Name test files `test_*.py`, classes `*Tests`, and methods `test_<behavior>`. Add regression coverage for parsing, tool validation, loop state, logging, and failure rollback when changing those areas. Run the full discovery command before submitting changes; no numeric coverage threshold is defined.

## Commit & Pull Request Guidelines

Write all commit messages—including subjects and bodies—in English. Use concise, imperative subjects (for example, `Validate duplicate tool call IDs`). End every commit message with a blank line followed by `Co-authored-by: Michael <265398295+lafalce-assistant@users.noreply.github.com>`. Pull request titles and bodies must also be in English. PR bodies should explain the behavioral change, list verification commands, link related issues, and include terminal output or screenshots when CLI rendering changes. Never commit API keys or logs containing prompts, responses, or tool arguments.
