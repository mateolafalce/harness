------------------------------ MODULE HarnessTurn ------------------------------
(***************************************************************************)
(* One turn of run_turn around a single apply_patch. Approval is already   *)
(* granted. The actions follow the current code, not a desired protocol.   *)
(*                                                                          *)
(* Sources:                                                                 *)
(*   harness/agent/loop.py          run_turn, execute_tool                  *)
(*   harness/tools/patch.py         apply_patch, _write_patch_changes       *)
(*   harness/persistence/progress.py                                        *)
(*   harness/persistence/store.py   current snapshot is what load returns   *)
(*   harness/cli.py                 --resume and build_system_prompt        *)
(*                                                                          *)
(* Finite instance: one turn, one patch, at most one crash, at most one    *)
(* later API failure. No compaction, shell, UI, tokens, or model internals. *)
(* The model is an action that returns apply_patch. The next model step    *)
(* either returns a final answer or raises Exception.                       *)
(*                                                                          *)
(* Order taken from the code:                                               *)
(*   1. history_snapshot = deepcopy(messages), before the user message.     *)
(*   2. apply_patch writes the repo inside execute_tool, before any save    *)
(*      of the session messages.                                            *)
(*   3. The tool result is then appended to memory, tool_call_completed is  *)
(*      logged, and record_tool persists the checkpoint.                    *)
(*   4. save(reason = "tool_batch") runs only after that batch.             *)
(*   5. A later Exception is caught by except (Exception, KeyboardInterrupt).*)
(*      That handler restores history_snapshot, save(reason = "rollback"),  *)
(*      then progress.failed. It does not revert files.                     *)
(*   6. SystemExit is not that except. A dead process does not roll back.   *)
(*   7. --resume loads the current SQLite snapshot. The progress checkpoint *)
(*      is copied into the system prompt only on that startup, as a note    *)
(*      that may be stale. The tool message is absent when the save never   *)
(*      stored it.                                                          *)
(*                                                                          *)
(* The one crash in this instance is the action Crash. It is enabled only  *)
(* after RecordTool and before SaveToolBatch: the file is already patched, *)
(* the tool result is in process memory, and the session save has not run. *)
(*                                                                          *)
(* Progress names:                                                          *)
(*   idle           code status "idle"                                      *)
(*   in_progress    code status "in progress" (progress.start)              *)
(*   tool_anotada   record_tool has stored "Tool `apply_patch` completed."  *)
(*                  The code status string is still "in progress".          *)
(*   interrupted    code status "interrupted" (progress.failed). The tool   *)
(*                  line stays in the checkpoint (tool_noted).              *)
(*   completed      code status "turn completed"                            *)
(*                                                                          *)
(* session is the current SQLite snapshot, the one load and --resume read. *)
(* Older snapshots stay in the database and are not the current session.   *)
(*   "snapshot"     prior messages, no tool result from this turn           *)
(*   "tool_result"  current snapshot contains that tool result              *)
(* memory is the live process:                                              *)
(*   "snapshot"     messages from before this turn (the history snapshot)   *)
(*   "user"         user message appended                                   *)
(*   "assistant"    assistant tool call appended                            *)
(*   "tool"         tool result appended                                    *)
(*   "final"        final assistant answer appended                         *)
(***************************************************************************)

VARIABLES
  repo,
  session,
  session_reason,
  memory,
  progress,
  tool_noted,
  phase,
  process,
  crashed,
  api_failed,
  resumed,
  stale_note

vars == <<repo, session, session_reason, memory, progress, tool_noted,
          phase, process, crashed, api_failed, resumed, stale_note>>

TypeOK ==
  /\ repo \in {"clean", "patched"}
  /\ session \in {"snapshot", "tool_result"}
  /\ session_reason \in {
       "prior",
       "tool_batch",
       "rollback",
       "final_response",
       "session_initialized"
     }
  /\ memory \in {"snapshot", "user", "assistant", "tool", "final"}
  /\ progress \in {
       "idle",
       "in_progress",
       "tool_anotada",
       "interrupted",
       "completed"
     }
  /\ tool_noted \in BOOLEAN
  /\ phase \in {
       "start",
       "snapshotted",
       "user_added",
       "progress_started",
       "patch_requested",
       "wrote",
       "result_in_memory",
       "logged",
       "tool_noted",
       "batch_saved",
       "api_error",
       "restored",
       "rollback_saved",
       "rolled_back",
       "dead",
       "resumed",
       "final"
     }
  /\ process \in {"alive", "dead"}
  /\ crashed \in BOOLEAN
  /\ api_failed \in BOOLEAN
  /\ resumed \in BOOLEAN
  /\ stale_note \in BOOLEAN

(***************************************************************************)
(* Agreement is the desired property: the repo is patched iff the current  *)
(* SQLite snapshot contains this turn's tool result.                       *)
(*                                                                          *)
(* Inv asks for Agreement on published states: before the turn, after      *)
(* save(reason = "tool_batch"), after a final response, after --resume,    *)
(* and after the except handler has finished. The open interval from the   *)
(* patch write until that save, the dead process, and the interior of the  *)
(* except are steps inside the two traces. TLC violates Inv at the end of  *)
(* those traces. It does not treat the open interval as a third behavior.  *)
(***************************************************************************)
Agreement ==
  (repo = "patched") <=> (session = "tool_result")

Published ==
  phase \in {"start", "batch_saved", "final", "resumed", "rolled_back"}

Inv ==
  Published => Agreement

\* Excludes the crash branch so TLC can report the rollback counterexample.
\* The committed config does not use this constraint.
NoCrash == crashed = FALSE

Init ==
  /\ repo = "clean"
  /\ session = "snapshot"
  /\ session_reason = "prior"
  /\ memory = "snapshot"
  /\ progress = "idle"
  /\ tool_noted = FALSE
  /\ phase = "start"
  /\ process = "alive"
  /\ crashed = FALSE
  /\ api_failed = FALSE
  /\ resumed = FALSE
  /\ stale_note = FALSE

TakeHistorySnapshot ==
  /\ phase = "start"
  /\ process = "alive"
  /\ phase' = "snapshotted"
  /\ UNCHANGED <<repo, session, session_reason, memory, progress,
                 tool_noted, process, crashed, api_failed, resumed,
                 stale_note>>

AppendUserMessage ==
  /\ phase = "snapshotted"
  /\ process = "alive"
  /\ memory' = "user"
  /\ phase' = "user_added"
  /\ UNCHANGED <<repo, session, session_reason, progress, tool_noted,
                 process, crashed, api_failed, resumed, stale_note>>

StartProgress ==
  /\ phase = "user_added"
  /\ process = "alive"
  /\ progress' = "in_progress"
  /\ phase' = "progress_started"
  /\ UNCHANGED <<repo, session, session_reason, memory, tool_noted,
                 process, crashed, api_failed, resumed, stale_note>>

ModelReturnsApplyPatch ==
  /\ phase = "progress_started"
  /\ process = "alive"
  /\ memory' = "assistant"
  /\ phase' = "patch_requested"
  /\ UNCHANGED <<repo, session, session_reason, progress, tool_noted,
                 process, crashed, api_failed, resumed, stale_note>>

ApplyPatch ==
  /\ phase = "patch_requested"
  /\ process = "alive"
  /\ repo' = "patched"
  /\ phase' = "wrote"
  /\ UNCHANGED <<session, session_reason, memory, progress, tool_noted,
                 process, crashed, api_failed, resumed, stale_note>>

RememberToolResult ==
  /\ phase = "wrote"
  /\ process = "alive"
  /\ memory' = "tool"
  /\ phase' = "result_in_memory"
  /\ UNCHANGED <<repo, session, session_reason, progress, tool_noted,
                 process, crashed, api_failed, resumed, stale_note>>

LogToolCompleted ==
  /\ phase = "result_in_memory"
  /\ process = "alive"
  /\ phase' = "logged"
  /\ UNCHANGED <<repo, session, session_reason, memory, progress,
                 tool_noted, process, crashed, api_failed, resumed,
                 stale_note>>

RecordTool ==
  /\ phase = "logged"
  /\ process = "alive"
  /\ progress' = "tool_anotada"
  /\ tool_noted' = TRUE
  /\ phase' = "tool_noted"
  /\ UNCHANGED <<repo, session, session_reason, memory, process, crashed,
                 api_failed, resumed, stale_note>>

\* SystemExit, or a dead process, after the write and before tool_batch.
\* The except handler does not run: no rollback save, files stay patched.
Crash ==
  /\ phase = "tool_noted"
  /\ process = "alive"
  /\ ~crashed
  /\ process' = "dead"
  /\ crashed' = TRUE
  /\ phase' = "dead"
  /\ UNCHANGED <<repo, session, session_reason, memory, progress,
                 tool_noted, api_failed, resumed, stale_note>>

SaveToolBatch ==
  /\ phase = "tool_noted"
  /\ process = "alive"
  /\ session' = "tool_result"
  /\ session_reason' = "tool_batch"
  /\ phase' = "batch_saved"
  /\ UNCHANGED <<repo, memory, progress, tool_noted, process, crashed,
                 api_failed, resumed, stale_note>>

ModelReturnsFinal ==
  /\ phase = "batch_saved"
  /\ process = "alive"
  /\ ~api_failed
  /\ memory' = "final"
  /\ session_reason' = "final_response"
  /\ progress' = "completed"
  /\ phase' = "final"
  /\ UNCHANGED <<repo, session, tool_noted, process, crashed, api_failed,
                 resumed, stale_note>>

\* Second model call. Exception is logged and re-raised into the except.
ModelRaises ==
  /\ phase = "batch_saved"
  /\ process = "alive"
  /\ ~api_failed
  /\ api_failed' = TRUE
  /\ phase' = "api_error"
  /\ UNCHANGED <<repo, session, session_reason, memory, progress,
                 tool_noted, process, crashed, resumed, stale_note>>

\* messages[:] = history_snapshot, which was taken before the user message.
RestoreHistorySnapshot ==
  /\ phase = "api_error"
  /\ process = "alive"
  /\ memory' = "snapshot"
  /\ phase' = "restored"
  /\ UNCHANGED <<repo, session, session_reason, progress, tool_noted,
                 process, crashed, api_failed, resumed, stale_note>>

SaveRollback ==
  /\ phase = "restored"
  /\ process = "alive"
  /\ session' = "snapshot"
  /\ session_reason' = "rollback"
  /\ phase' = "rollback_saved"
  /\ UNCHANGED <<repo, memory, progress, tool_noted, process, crashed,
                 api_failed, resumed, stale_note>>

\* progress.failed. The tool line already stored by record_tool remains.
\* The repo is not reverted.
MarkTurnInterrupted ==
  /\ phase = "rollback_saved"
  /\ process = "alive"
  /\ progress' = "interrupted"
  /\ phase' = "rolled_back"
  /\ UNCHANGED <<repo, session, session_reason, memory, tool_noted,
                 process, crashed, api_failed, resumed, stale_note>>

\* --resume. Messages come from SQLite, not from the dead process.
\* A new session_initialized snapshot is the loaded history plus a system
\* prompt that includes the progress note. The tool message is not in that
\* snapshot, so it does not reappear. The note can be stale.
Resume ==
  /\ phase = "dead"
  /\ process = "dead"
  /\ ~resumed
  /\ memory' = IF session = "tool_result" THEN "tool" ELSE "snapshot"
  /\ stale_note' = TRUE
  /\ resumed' = TRUE
  /\ session_reason' = "session_initialized"
  /\ process' = "alive"
  /\ phase' = "resumed"
  /\ UNCHANGED <<repo, session, progress, tool_noted, crashed, api_failed>>

Next ==
  \/ TakeHistorySnapshot
  \/ AppendUserMessage
  \/ StartProgress
  \/ ModelReturnsApplyPatch
  \/ ApplyPatch
  \/ RememberToolResult
  \/ LogToolCompleted
  \/ RecordTool
  \/ Crash
  \/ SaveToolBatch
  \/ ModelReturnsFinal
  \/ ModelRaises
  \/ RestoreHistorySnapshot
  \/ SaveRollback
  \/ MarkTurnInterrupted
  \/ Resume

Spec == Init /\ [][Next]_vars

=============================================================================
