# Phase 1: consistent observational DAG status fields

This completes Phase 1 of the user-adopted chat design. Scope is the existing
`dag-status.v1.json` projection and public readback. Immutable node requests,
attempts, terminal records, native execution gates and EOD completion formats
remain unchanged. No status field grants permission to execute.

## Exact additive contract

Preserve existing node keys. Every projected/readback node additionally exposes:

- `trigger_reason`: a code derived from observed state/action, never a caller flag
  or guessed historical cause. Finite non-null values: `NOT_STARTED`,
  `WAITING_UPSTREAM`, `BLOCKING_FAILURE`, `ATTEMPT_RUNNING`, `NATIVE_EXECUTED`,
  `NATIVE_REPLAY`, `NATIVE_RECOVERY`, `RECORDED_TERMINAL`, `EVIDENCE_INVALID`.
  Use null when no supported trigger can be established from the supplied evidence.
- `blocking_reason`: exact typed failure code when available, otherwise null;
  stale/invalid recorded evidence uses `VALIDATION_FAILED` without inventing a
  more specific explanation.
- `retryable`: boolean from the fixed failure taxonomy; false when there is no
  retryable failure. This is informational and cannot bypass native reconciliation.
- `upstream_refs`: only exact `upstream.*` refs from a validated recorded request;
  empty when no request was recorded. Never use a mutable current head.
- `output_refs`: exact verified terminal refs; empty if none can be trusted.
- `started_at`, `finished_at`: original validated journal timestamps or null.
  Never fill absent timestamps from now, cutoff, mtime or an adjacent node.
- `attempt`: original validated attempt number; 0 only when absence of an attempt
  is established; null when recorded evidence is invalid and the count is unknown.

`state` retains the existing nine values. Lifecycle success remains separate
from native EOD completion; all16 node successes alone cannot mark a day complete.

## Implementation path and limits

Use one pure node-projection helper shared by `daily_runner.py` and
`daily_status.py`. The runner retains validated request documents it already uses;
the reader passes the exact request it already reads. Do not introduce additional
full native replays, filesystem scans or repeated source hashing to populate these
observational fields. Only recorded requests/attempts justify consumed refs/time.

Map existing `command_status` (`EXECUTED`, `NO_ACTION`, `ADOPTED`) where available;
otherwise use the exact recorded lifecycle/typed failure. Generic invalid-evidence
rows must retain their original diagnostic reason while exposing conservative
common fields. Pending nodes cannot acquire fabricated attempts or timestamps.

The projection writer remains the existing lock-owned atomic `dag-status.v1.json`
writer. Readback remains read-only. Existing additional keys such as terminal,
start and request-key details remain available for compatibility. If a current
strict consumer rejects additive fields, identify it before implementation and
choose the smallest versioned adjustment; do not silently broaden validators.

## Acceptance and stop conditions

Architect then Critic before edits. Focused tests must cover exact original time
and attempt preservation, initial/pending/missing-input/blocked/running/succeeded/
recovered/stale rows, typed retryability, recorded upstream/output refs, no current
head substitution, no reader writes/native calls and unchanged immutable journal
bytes. Exercise both real runner projection and public status readback, including
a completed DAG whose completion still requires native validation.

Stop after the phase contract and focused checks are satisfied. Do not alter pool
formats, Decision semantics, freshness policy, automation or native computation
as part of Phase 1. The already-running 3c46 acceptance remains frozen evidence;
subsequent source changes require an explicit coverage/source-delta statement.

## Accepted Architect semantics

Apply this trigger precedence, with only already-validated caller data:

1. Invalid recorded journal/output evidence -> `EVIDENCE_INVALID`.
2. Invocation-local `EXECUTED`, `NO_ACTION`, or `ADOPTED` -> `NATIVE_EXECUTED`,
   `NATIVE_REPLAY`, or `NATIVE_RECOVERY`, respectively. This records the actual
   current invocation; a typed blocking failure is still separately reported.
3. `SKIPPED` plus typed `UPSTREAM_INCOMPLETE` -> `WAITING_UPSTREAM`.
4. Any other typed blocking failure -> `BLOCKING_FAILURE`.
5. Validated start with no terminal -> `ATTEMPT_RUNNING`.
6. Validated terminal without invocation-local action -> `RECORDED_TERMINAL`.
7. Exact request-key journal inspection with `attempt=0` -> `NOT_STARTED`.
8. Otherwise -> null, including no selected request key. Existing lifecycle state
   can remain NOT_STARTED; do not pretend its attempt history was inspected.

Blocked/skipped rows created before an exact request key exists have `attempt=null`.
A validated existing start preserves its original attempt/time even when the current
invocation reports POST_WRITE_IN_DOUBT. Invalid evidence has unknown attempt/time
and no trusted refs. Absence of blocking reason implies `retryable=false`; typed
failure uses the fixed taxonomy. Preserve the existing diagnostic `reason`.

The runner may show invocation action in its existing mutable status projection.
Neither `command_status` nor trigger metadata is added to immutable request/start/
terminal records. Readback ignores the stored projection, obtains original journal
bytes, and never reconstructs invocation-local action from persisted status.

Pass recorded request bytes only where the caller has established the request is
recorded. A merely constructed candidate request does not populate `upstream_refs`.
The helper performs no I/O, replay, hashing, clock reads or authority decisions.
No INPUTS_READY or NO_SESSION claim is inferred from request existence or weekdays;
non-trading-day behavior remains at day/result level and creates no fake DAG.

## Implementation receipt

Implemented in `operations/status_projection.py`, `daily_runner.py` and
`daily_status.py`. The runner publishes its validated start to the existing
projection before entering the potentially long native writer. It retains
recorded requests already used by the runner, without new native reads.
83 focused regression checks passed in 28.35s; three-file mypy, four-file
flake8 and Black passed. New consumer code also replayed the 16 real native
Aug27 journal nodes from frozen producer3c46, preserving original refs/times/attempts
and all 285 inspected source byte hashes/mtimes. No native completion was inferred
from the observational read. Receipt: `.agent/acceptance/phase1-native-journal-readback.json`.
