# Harness — Stage 2: First Agent Loop

Conversational terminal client for `gpt-oss-120b` through the Cerebras API. The
model can respond directly or request tools; the program executes each call,
returns its result, and repeats until it receives a final response.

The loop is deliberately straightforward and does not use an agent framework:

1. add the user's message to the conversation history;
2. call the model with the tool schemas and `tool_choice="auto"`;
3. if the response contains `tool_calls`, validate and execute each one;
4. add each result with its corresponding `tool_call_id`;
5. return to step 2 until a final response is received.

## Tools

The three tools do not access the network or modify files:

- `calculator`: evaluates numeric expressions containing `+`, `-`, `*`, `/`,
  `//`, `%`, and `**`. It uses a restricted interpreter based on Python's AST,
  not `eval`.
- `get_current_time`: returns the current time in an IANA time zone, such as
  `UTC` or `America/Argentina/Mendoza`.
- `echo`: returns text unchanged. It is included as a harmless third tool
  because the assignment requests three tools while listing only the two above.

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
contain sensitive information.

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
.venv/bin/python harness.py "What is (27 + 5) * 3?"
```

Configure limits and logging:

```bash
.venv/bin/python harness.py \
  --max-turns 5 \
  --timeout 20 \
  --log-file logs/session.jsonl \
  "What time is it in America/Argentina/Mendoza?"
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
