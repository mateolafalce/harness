# Harness turn protocol

TLC checks one turn of `run_turn` around a single `apply_patch` whose approval is already granted. The spec follows the current code. It is not a proposal for different behavior.

Run the commands from the repository root. `tla2tools.jar` is the TLA+ tools jar. The traces in this directory were produced with TLC2 Version 2026.10.23.135453. TLC exits non-zero for each command below.

`-workers 1` keeps the trace order stable. `-noGenerateSpecTE` stops TLC from writing a trace module next to the spec. `-metadir` keeps the state directory out of the repository.

## Crash trace

`specs/tla/HarnessTurn.cfg` checks `Inv` on the full instance. The crash trace is shorter than the rollback trace, so TLC reports it and stops.

```bash
java -XX:+UseParallelGC -jar tla2tools.jar \
  -workers 1 \
  -nowarning \
  -noGenerateSpecTE \
  -cleanup \
  -metadir /tmp/tlc-harness-crash \
  -config specs/tla/HarnessTurn.cfg \
  specs/tla/HarnessTurn.tla
```

The counterexample is `specs/tla/trace_crash.txt`.

## Rollback trace

`NoCrash` is defined in `HarnessTurn.tla` and is not part of the committed config. Adding it as a state constraint hides the crash branch. TLC then reports the rollback violation of the same `Inv`.

```bash
cfg=$(mktemp --suffix=.cfg)
cat specs/tla/HarnessTurn.cfg > "$cfg"
printf '\nCONSTRAINT NoCrash\n' >> "$cfg"
java -XX:+UseParallelGC -jar tla2tools.jar \
  -workers 1 \
  -nowarning \
  -noGenerateSpecTE \
  -cleanup \
  -metadir /tmp/tlc-harness-rollback \
  -config "$cfg" \
  specs/tla/HarnessTurn.tla
```

The counterexample is `specs/tla/trace_rollback.txt`.

## Inv

`Agreement` is the desired property. The repository is patched if and only if the current SQLite snapshot contains this turn's tool result:

```
Agreement == (repo = "patched") <=> (session = "tool_result")
```

`session` is the current snapshot, the one `SessionStore.load` and `--resume` return. Older snapshots remain in SQLite and are not the current session. `"snapshot"` means the messages from before this turn. `"tool_result"` means the current snapshot contains the tool result.

`Inv` requires `Agreement` on the states the code publishes:

```
Published == phase \in {"start", "batch_saved", "final", "resumed", "rolled_back"}
Inv == Published => Agreement
```

Those phases are: before the turn, after `save(reason="tool_batch")`, after a final response, after `--resume`, and after the `except` handler has finished (`messages[:] = history_snapshot`, `save(reason="rollback")`, and `progress.failed`).

`Agreement` is already false in the open interval that starts when `apply_patch` writes the file and ends at `save(reason="tool_batch")`. That interval is the prefix of both traces, not a third behavior. Checking the unguarded biconditional as a state invariant would stop TLC at `ApplyPatch`, before `Crash` / `Resume` and before the rollback handler. The guard does not make `Inv` true. Both traces end in a published state where `Agreement` is false.

## Trace A, crash

`apply_patch` has written the file. The process dies after `record_tool` and before `save(reason="tool_batch")`. `SystemExit` is not caught by `except (Exception, KeyboardInterrupt)`, so the handler does not run. `--resume` loads the previous snapshot. The file stays patched.

1. `TakeHistorySnapshot`. `history_snapshot = copy.deepcopy(messages)` before the user message is appended.
2. `AppendUserMessage`. The user message is only in process memory.
3. `StartProgress`. `progress.start` persists status `in progress` before the `try`.
4. `ModelReturnsApplyPatch`. The assistant tool call is appended.
5. `ApplyPatch`. The write happens inside `execute_tool`. The session is not saved.
6. `RememberToolResult`. The tool result is appended to memory.
7. `LogToolCompleted`. `tool_call_completed` is logged. This is not a session save.
8. `RecordTool`. `progress.record_tool` stores `Tool apply_patch completed`. The model calls this `tool_anotada`. The code status string is still `in progress`.
9. `Crash`. The process is dead. No rollback save. The file stays patched. Memory still holds the tool result, but that memory dies with the process.
10. `Resume`. `--resume` loads the current SQLite snapshot and writes `session_initialized` from those messages, with the progress checkpoint copied into the system prompt as a note that may be stale. The tool message does not reappear.

Final state: `repo = "patched"`, `session = "snapshot"`, `session_reason = "session_initialized"`, `memory = "snapshot"`, `progress = "tool_anotada"`, `tool_noted = TRUE`, `stale_note = TRUE`, `phase = "resumed"`. `Inv` is false.

## Trace B, rollback

The patch was applied and `record_tool` has run. `save(reason="tool_batch")` then stores the tool result. The next model call raises `Exception`. The `except` handler saves the history snapshot and does not revert the file.

1. `TakeHistorySnapshot`
2. `AppendUserMessage`
3. `StartProgress`
4. `ModelReturnsApplyPatch`
5. `ApplyPatch`
6. `RememberToolResult`
7. `LogToolCompleted`
8. `RecordTool`
9. `SaveToolBatch`. The current snapshot contains the tool result. `Agreement` holds.
10. `ModelRaises`. The API error is logged and re-raised into the `except`.
11. `RestoreHistorySnapshot`. `messages[:] = history_snapshot`, taken before the user message.
12. `SaveRollback`. The current snapshot is the prior messages again. The `tool_batch` row remains as history and is not what `load` returns. The file stays patched.
13. `MarkTurnInterrupted`. `progress.failed` sets status `interrupted` and keeps the tool completion line (`tool_noted`).

Final state: `repo = "patched"`, `session = "snapshot"`, `session_reason = "rollback"`, `memory = "snapshot"`, `progress = "interrupted"`, `tool_noted = TRUE`, `api_failed = TRUE`, `phase = "rolled_back"`. `Inv` is false.

## Instance

One turn, one patch, at most one crash, at most one later API failure. No compaction, shell, UI, tokens, or model. The first model step returns `apply_patch`. The next model step returns a final answer or raises `Exception`. On the final-answer path, `SaveToolBatch` has already run, so `Agreement` holds and `Inv` holds.
