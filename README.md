# Harness (in progress)

A terminal agent that uses `gpt-oss-120b` through Cerebras and exposes a
`run_bash` tool to the model.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your API key.
```

Create an API key in the [Cerebras console](https://cloud.cerebras.ai/).
The harness automatically loads `CEREBRAS_API_KEY` from `.env`; an environment
variable already exported in the shell takes precedence and is not overwritten.

## Usage

```bash
.venv/bin/python harness.py "How many files are in this directory?"
```

The program asks for confirmation before running each command. In an isolated
environment where you accept the risk of arbitrary command execution, you can
skip confirmation:

```bash
.venv/bin/python harness.py --auto-approve "Review the tests and explain the failures"
```

Useful options:

```bash
.venv/bin/python harness.py --help
```

> `--auto-approve` lets the model run any command with the same permissions as
> your user. Use it only inside a disposable container or workspace.
