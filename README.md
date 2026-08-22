# Harness — Stage 3: Read-only Mini Coding Agent

Conversational terminal client for `gpt-oss-120b` through the Cerebras API. The
model can inspect the repository in the current working directory, run its fixed
test suite, or request the Stage 2 utility tools. It cannot edit files or execute
model-supplied shell commands.

The loop is deliberately straightforward and does not use an agent framework:

1. add the user's message to the conversation history;
2. call the model with the tool schemas and `tool_choice="auto"`;
3. if the response contains `tool_calls`, validate and execute each one;
4. add each result with its corresponding `tool_call_id`;
5. return to step 2 until a final response is received.

## Read-only coding tools

Every filesystem path must be relative to the repository and remains confined to
that directory after symlinks are resolved. Generated directories such as
`.git`, `.venv`, and `__pycache__` are excluded, as are local `.env` files.

- `list_files`: recursively returns sorted paths below a directory, capped at
  500 files.
- `read_file`: reads a one-based range of up to 200 UTF-8 lines and caps content
  at 16,000 characters. Its result indicates whether more content exists.
- `search_text`: searches for a literal, case-sensitive fragment and returns up
  to 50 locations with bounded line excerpts. Binary files and files larger
  than 1 MB are skipped.
- `git_diff`: returns the staged and unstaged tracked-file diff against `HEAD`
  for a selected path. External diff programs are disabled.
- `run_tests`: invokes only
  `.venv/bin/python -m unittest discover -s tests -v`, with a 60-second timeout.

`git_diff` and `run_tests` return at most 400 lines and 16,000 characters.
Truncated results carry `truncated: true`, allowing the model to narrow the next
request instead of filling its context with an unbounded result.

The Stage 2 utility tools remain available and do not access the network or
modify files:

- `calculator`: evaluates numeric expressions containing `+`, `-`, `*`, `/`,
  `//`, `%`, and `**`. It uses a restricted interpreter based on Python's AST,
  not `eval`.
- `get_current_time`: returns the current time in an IANA time zone, such as
  `UTC` or `America/Argentina/Mendoza`.
- `echo`: returns text unchanged.

Each tool declares a strict JSON schema. Before execution, the harness verifies
that the arguments are valid JSON, all required fields are present, their types
are correct, and no additional properties exist. An unknown name, malformed
JSON, invalid arguments, or an execution error produces a structured result
with `ok: false`; this result is sent to the model so it can recover.

## Limits and Observability

Each user message has two configurable safeguards:

- `--max-turns`: maximum number of model calls, 8 by default;
- `--timeout`: total time limit for the loop, 30 seconds by default. The
  remaining time is also passed as the timeout for each API call.

If a limit is exceeded or the API fails, the incomplete turn is removed from
the history so the conversation retains a valid sequence.

The JSON Lines log includes session start and end events, user messages,
iteration starts, API requests and responses, agent decisions, tool starts and
results, `call_id` values, arguments, errors, latency, metrics, and termination
reasons. Prompts, responses, and arguments are stored in full, so the file may
contain sensitive information. Tool results stored in the log have the same
bounds as the results sent to the model.

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# Edit .env and add the API key.
```

Create an API key in the [Cerebras console](https://cloud.cerebras.ai/). The
program loads `CEREBRAS_API_KEY` from `.env`; an already exported environment
variable takes precedence.

## Usage

Interactive conversation:

```bash
.venv/bin/python harness.py
```

During an interactive conversation, `/clear` removes accumulated user,
assistant, and tool turns while preserving the system instruction. The command
does not send a request to the model. `/help` displays all available commands.

Single prompt:

```bash
.venv/bin/python harness.py "Find the tests for tool argument validation."
```

Configure limits and logging:

```bash
.venv/bin/python harness.py \
  --max-turns 5 \
  --timeout 20 \
  --log-file logs/session.jsonl \
  "Summarize the current diff and run the tests."
```

The utilization shown at the end sums the tokens from every call in the loop
and calculates `total_tokens / context_window * 100`. The default context
window is 131,072 tokens and can be changed with `--context-window`. The metrics
also show how many prompt tokens were retrieved from the provider's cache.

To see every option:

```bash
.venv/bin/python harness.py --help
```

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
