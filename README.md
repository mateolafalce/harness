# Harness

Small Cerebras coding agent for `gpt-oss-120b`. Inspects and edits current
repository. Manages limited model context. No agent framework.

Core loop:

1. Load global and repository instructions.
2. Rank likely relevant files.
3. Add user message to durable history.
4. Call model with strict tools and `tool_choice="auto"`.
5. Validate tools; request approval for side effects.
6. Summarize large outputs.
7. Compact old complete turns near context limit.
8. Save session and progress checkpoints.
9. Continue until final response or guardrail stop.

## Context

Instruction order:

1. `--system-prompt`
2. `$HARNESS_HOME/AGENTS.md`, or `~/.harness/AGENTS.md`
3. Root `AGENTS.md`
4. Nested `AGENTS.md` files governing selected paths

Deeper repository instructions win inside their scope. Limit: 24,000 characters
per file, 64,000 combined. `read_file` reports applicable instruction paths.

Relevant-file selection scores task words, exact filenames, and test signals.
Default: eight path hints, no eager file-body loading. Tune with
`--relevant-files`. Model retrieves exact context through `list_files`,
`search_text`, and `read_file`.

Tool payloads over 4,000 characters become valid JSON head/tail summaries in
model history. Full results remain in JSONL logs. Model can rerun narrower query
when omitted middle matters.

History compacts above `--compaction-threshold` (`0.70` default). Complete old
turns become bounded working summary. Latest two turns stay verbatim; tune with
`--keep-recent-turns`. Interactive `/compact` triggers manual compaction.

## Sessions and progress

Session state saves atomically to `.harness/session.json`, mode `0600`. Contains
schema version, repository identity, session ID, and model messages.

Resume:

```bash
.venv/bin/python harness.py --resume
```

Use `--session-file PATH` for another repository-relative file. Resume reloads
current `AGENTS.md`; stored system instructions never override new policy.
Malformed state, cross-repository state, and unrelated existing files are
rejected. Session files and JSONL logs stay hidden from model-facing repository
tools.

Long tasks write `.harness/progress.md`: objective, status, recent tool outcomes,
completion, interruption. Resume injects notes as fallible memory; model must
verify against source. Configure with `--progress-file`.

## Editing and approvals

`apply_patch` provides only persistent model-facing edit path. Accepts
`*** Begin Patch` format or unified Git diff. Limits: 100,000 characters, 50
UTF-8 files. Rejects absolute paths, traversal, excluded paths, and symlink
aliases. Validates all files and hunks before writing; failed patch leaves no
partial edits.

`apply_patch`, `run_tests`, and `run_shell` use `--approval-policy`:

- `ask`: prompt local terminal; deny without interactive TTY.
- `allow`: execute valid gated calls without prompt. Trusted workflows only.
- `deny`: reject all gated calls.

Validation runs before approval. Decision returns to model and JSONL log.

## Shell sandbox

`run_shell` uses `shlex` plus direct `subprocess.run`; never `shell=True`. Command
limit: 1,000 characters. Allowed forms only:

```text
git status --short
git diff --check
git diff --cached --check
git diff [-- PATH]
rg QUERY PATH
.venv/bin/python -m unittest discover -s tests -v
```

Rejects extra arguments, control operators, arbitrary Python, absolute paths,
and outside-project paths. Disables Git hooks, filesystem monitors, external
diff programs, and text conversion.

Commands run in disposable repository copy. Copy omits generated caches,
`.venv`, local `.env` files, and default event log. Child environment keeps only
basic locale, terminal, temporary-directory, and executable-path values. API
keys and proxy variables stay out. Shell filesystem changes never persist.

Disposable copy limits repository damage. It is not OS isolation: approved code
can still read host data or use network. Use container or VM for untrusted
repositories.

## Tools and limits

- `list_files`: sorted recursive paths; 500-file cap.
- `read_file`: 200 UTF-8 lines; 16,000-character cap.
- `search_text`: 50 literal matches; skips binary and files over 1 MB.
- `git_diff`: bounded tracked-file diff; external diff disabled.
- `run_tests`: fixed unittest command; approval-gated and sandboxed.
- `calculator`, `get_current_time`, `echo`: strict utility tools.

All schemas reject missing, mistyped, or extra arguments. Tool errors return
structured results so model can recover.

Excluded: `.git`, `.venv`, `__pycache__`, caches, local `.env` files. Process
output cap: 400 lines and 16,000 characters. Shell timeout: 20 seconds. Patch
timeout: 10 seconds. User-turn defaults: 8 model calls, 30 seconds. Failed or
expired loops roll incomplete messages back.

## Observability

JSONL events cover session config, selected paths, compactions, API calls, tool
calls, approvals, errors, latency, tokens, and termination. Logs store prompts,
responses, and tool arguments in full. Sensitive. Never commit them.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# Edit .env and add CEREBRAS_API_KEY.
```

Create key in [Cerebras console](https://cloud.cerebras.ai/). Exported
`CEREBRAS_API_KEY` wins over `.env`.

## Use

Interactive:

```bash
.venv/bin/python harness.py
```

Read-only:

```bash
.venv/bin/python harness.py --approval-policy deny \
  "Review the current implementation without changing it."
```

Trusted edit session:

```bash
.venv/bin/python harness.py \
  --approval-policy ask \
  --max-turns 5 \
  --timeout 20 \
  --log-file logs/session.jsonl \
  "Fix the failing test and verify it."
```

Interactive commands: `/clear`, `/compact`, `/help`, `/exit`, `/quit`.

All options:

```bash
.venv/bin/python harness.py --help
```

## Test

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Design follows Anthropic's
[Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents):
small high-signal context, just-in-time retrieval, compaction, durable notes.
