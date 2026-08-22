# Harness — Stage 4: Editing and Security

Conversational terminal client for `gpt-oss-120b` through the Cerebras API. The
agent can inspect and edit the current repository, but persistent edits and code
execution are separated behind explicit tools, approval policy, path checks,
bounded execution, and a disposable working copy.

The loop remains deliberately small and does not use an agent framework:

1. add the user's message to the conversation history;
2. call the model with strict tool schemas and `tool_choice="auto"`;
3. validate each requested tool and its arguments;
4. request approval when the tool edits files or executes project code;
5. return each bounded result with its corresponding `tool_call_id`;
6. continue until the model produces a final response or a loop limit is hit.

## Editing and approvals

`apply_patch` is the only model-facing tool that persists source changes. It
accepts either the `*** Begin Patch` marker format or a standard unified Git
diff, with a maximum input size of 100,000 characters. It supports adding,
updating, and deleting up to 50 UTF-8 text files per call. All paths must be
repository-relative;
absolute paths, traversal, excluded paths, and symlink aliases are rejected.
Marker patches validate every file and hunk before writing, so an invalid later
hunk does not leave earlier changes behind.

`run_tests` and `run_shell` execute project code or processes. These tools and
`apply_patch` use `--approval-policy`:

- `ask` (default): request confirmation in the local terminal and deny when no
  interactive TTY is available;
- `allow`: execute valid gated calls without prompting;
- `deny`: reject every gated call.

Approval decisions are returned to the model and recorded as structured JSONL
events. Argument and path validation happens before a prompt is shown. The
`allow` mode is intended only for an already trusted workflow.

## Limited shell and sandbox

`run_shell` parses one command with `shlex` and passes an argument vector directly
to `subprocess.run`; it never invokes `sh`, `bash`, or `shell=True`. Only these
forms are accepted, with a maximum command length of 1,000 characters:

```text
git status --short
git diff --check
git diff --cached --check
git diff [-- PATH]
rg QUERY PATH
.venv/bin/python -m unittest discover -s tests -v
```

Extra arguments, control operators, arbitrary Python, absolute paths, and paths
outside the project are rejected. Git hooks, filesystem monitors, external diff
programs, and text conversion are disabled for the allowed Git commands.

Every allowed command runs with the disposable sandbox: the repository is copied
to a temporary directory, generated caches, `.venv`, local `.env` files, and the
default event log are omitted, and the temporary tree is removed after the
process exits. `run_tests` uses the same path and invokes the project's original
virtual-environment interpreter with the disposable copy as its working
directory. The child environment keeps only basic locale, terminal, temporary
directory, and executable-path settings; API keys and proxy variables are not
forwarded. Shell filesystem changes therefore do not persist in the project.

The disposable copy limits damage to repository state; it is not an OS security
boundary and does not block host reads or network access from already approved
project code. Run this harness itself in a container or VM when processing an
untrusted repository.

## Repository tools and limits

The Stage 3 inspection tools remain available:

- `list_files`: sorted recursive paths, capped at 500 files;
- `read_file`: up to 200 UTF-8 lines and 16,000 characters;
- `search_text`: up to 50 literal matches; skips binary files and files over 1 MB;
- `git_diff`: bounded tracked-file diff with external diff programs disabled;
- `run_tests`: the fixed unittest command, now approval-gated and sandboxed.

Generated directories such as `.git`, `.venv`, and `__pycache__` are excluded
from filesystem tools, as are local `.env` files. Process output is capped at
400 lines and 16,000 characters. Generic shell execution has a 20-second maximum;
patch subprocesses have a 10-second maximum. Both also receive no more than the
time remaining in the complete agent loop.

The Stage 2 utilities remain available: `calculator`, `get_current_time`, and
`echo`. Every tool declares a strict JSON schema and rejects missing, mistyped,
or additional arguments. Tool failures are structured results, allowing the
model to recover without corrupting conversation history.

Each user message additionally has `--max-turns` (8 by default) and `--timeout`
(30 seconds by default). If the loop fails or exceeds a limit, its incomplete
messages are removed from conversation history.

## Observability

The JSONL log includes session configuration, messages, API calls, tool calls,
approval requests and outcomes, errors, latency, token metrics, and termination
reasons. Prompts, responses, and tool arguments are stored in full, so logs can
contain sensitive data and must not be committed.

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# Edit .env and add CEREBRAS_API_KEY.
```

Create an API key in the [Cerebras console](https://cloud.cerebras.ai/). An
already exported `CEREBRAS_API_KEY` takes precedence over `.env`.

## Usage

Interactive mode, with approval prompts enabled:

```bash
.venv/bin/python harness.py
```

Single read-only request with all gated actions denied:

```bash
.venv/bin/python harness.py --approval-policy deny \
  "Review the current implementation without changing it."
```

Trusted editing session with explicit limits:

```bash
.venv/bin/python harness.py \
  --approval-policy ask \
  --max-turns 5 \
  --timeout 20 \
  --log-file logs/session.jsonl \
  "Fix the failing test and verify it."
```

During interactive mode, `/clear` resets conversation turns while preserving the
system instruction. `/help` lists interactive commands.

This stage follows the separation between sandbox capabilities and approval
policy described by the [official Codex documentation](https://learn.chatgpt.com/docs/agent-approvals-security.md),
and the `allow` / `ask` / `deny` tool policy model documented by
[OpenCode tools](https://opencode.ai/docs/tools/) and
[OpenCode agents](https://opencode.ai/docs/agents/).

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

To inspect all CLI options:

```bash
.venv/bin/python harness.py --help
```
