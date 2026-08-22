# Harness — Stage 5: Context Engineering

Conversational terminal client for `gpt-oss-120b` through the Cerebras API. The
agent can inspect and edit the current repository while curating its limited
context window. Persistent edits and code execution remain separated behind
explicit tools, approval policy, path checks, bounded execution, and a disposable
working copy.

The loop remains deliberately small and does not use an agent framework:

1. load global and repository instructions;
2. rank likely relevant file paths for the current request;
3. add the user's message to durable conversation history;
4. call the model with strict tool schemas and `tool_choice="auto"`;
5. validate each requested tool and request approval for side effects;
6. summarize oversized results before adding them to model context;
7. compact complete older turns when the configured threshold is reached;
8. persist the session and progress notes after safe checkpoints;
9. continue until the model produces a final response or a loop limit is hit.

## Context engineering

The system prompt is assembled from the CLI `--system-prompt`, an optional global
instruction file, and repository `AGENTS.md` files. The default global location is
`$HARNESS_HOME/AGENTS.md`, or `~/.harness/AGENTS.md` when `HARNESS_HOME` is unset.
The root `AGENTS.md` is loaded at startup. When relevant files are selected,
nested `AGENTS.md` files from their parent directories are added just in time;
deeper documents take precedence inside their scope. Each instruction file is
bounded to 24,000 characters and the combined instruction context to 64,000.

Relevant-file selection ranks paths from task words, exact filenames, and test
signals. It sends at most eight path hints by default, without eagerly reading
file bodies. The existing `list_files`, `search_text`, and `read_file` tools then
provide progressive disclosure. Set the cap with `--relevant-files`.

Tool results remain complete in the JSONL event log, but payloads over 4,000
characters are represented in model history by a valid JSON head/tail summary.
This prevents a long test run or diff from dominating subsequent attention while
letting the model request a narrower query when exact middle content matters.

Before each model call, history size is estimated conservatively. Above
`--compaction-threshold` (0.70 by default), complete older user turns are replaced
with a bounded working summary; the newest two turns remain verbatim by default.
Use `--keep-recent-turns` to tune retention or `/compact` to request compaction in
interactive mode. Architectural decisions, user requests, tool names, failures,
and answer excerpts are retained, while raw old tool payloads are discarded.

## Sessions and progress

Conversation state is atomically persisted to `.harness/session.json` with mode
`0600`. The state contains a schema version, repository identity, session ID, and
model messages. It is rejected if malformed or opened from another repository.
Resume it with:

```bash
.venv/bin/python harness.py --resume
```

Use `--session-file PATH` for another repository-relative state file. On resume,
the current `AGENTS.md` instructions replace the stored system prompt so policy
changes take effect, while compacted history and recent turns remain available.
Session files and JSONL logs are excluded from model-facing repository tools.

`.harness/progress.md` is a compact, human-readable checkpoint for long work. It
records the durable objective, recent tool outcomes, completion, or interruption
after each safe step. On resume it is injected as explicitly fallible working
memory: the model is told to verify it against current source. Configure its path
with `--progress-file`.

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

The inspection tools remain available:

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

The utility tools remain available: `calculator`, `get_current_time`, and
`echo`. Every tool declares a strict JSON schema and rejects missing, mistyped,
or additional arguments. Tool failures are structured results, allowing the
model to recover without corrupting conversation history.

Each user message additionally has `--max-turns` (8 by default) and `--timeout`
(30 seconds by default). If the loop fails or exceeds a limit, its incomplete
messages are removed from conversation history.

## Observability

The JSONL log includes session configuration, selected paths, compactions, API
calls, tool calls,
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
system instruction. `/compact` summarizes eligible older turns. `/help` lists
interactive commands.

The context design follows Anthropic's
[Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents):
keep the smallest high-signal context, retrieve details just in time, compact old
history, and persist structured notes outside the context window. The security
model continues to separate sandbox capabilities from explicit approval policy.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

To inspect all CLI options:

```bash
.venv/bin/python harness.py --help
```
