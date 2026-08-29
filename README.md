# Harness

Small Cerebras coding agent for `gpt-oss-120b`. Inspects and edits the current
repository. Manages limited model context. No agent framework.

On a TTY, interactive mode occupies the whole terminal: black background,
boxed `>` prompt with the model on the right, and token metrics under the
box. Typed messages appear as a `>` bar with the time on the right; replies
as `assistant>`. One-shot prompts skip the fullscreen UI.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Set `CEREBRAS_API_KEY` in `.env`. Create a key in the
[Cerebras console](https://cloud.cerebras.ai/). An exported
`CEREBRAS_API_KEY` overrides the file.

## Usage

```bash
.venv/bin/python harness.py
# or: .venv/bin/python -m harness
```

One-shot:

```bash
.venv/bin/python harness.py "Review this repository without changing it."
```

Resume the latest session from `.harness/harness.db`:

```bash
.venv/bin/python harness.py --resume
```

Interactive commands: `/help`, `/clear`, `/compact`, `/exit`, `/quit`.
Ctrl-D also quits. Scroll the transcript with the mouse wheel or Page Up
and Page Down. All flags: `.venv/bin/python harness.py --help`.

## Docker

```bash
docker compose run --rm harness
docker compose run --rm harness --resume
```

## Test

```bash
.venv/bin/python -m unittest discover -s tests -v
```
