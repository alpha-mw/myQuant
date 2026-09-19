# Phase 2: exact dependency rejection before production

Two current-path failures establish the scope:

1. A native-valid LOW observation with signal date 20260819 placed at the expected
   20260820 path is rejected, but the runner emits only VALIDATION_FAILED and drops
   the reason. Top100 is skipped without the exact upstream cause. Reproduction:
   `.agent/acceptance/phase2-before.json` (other node adapters are controlled).
2. Installed historical catch-up accepted a minimal readback-only loop context,
   ran maintenance, then Factor rollover failed with KeyError `release_commit`.
   `calendar_capture_parent` is also required by that same producer. Maintenance
   core and auxiliaries completed, but no Factor rollover/Top100/handoff did.
   This invalid producer context must be rejected by EXECUTE preflight.

## Bounded implementation

Keep topology, immutable journal schemas, business writers, and existing native
validation unchanged. Add an internal `DependencyInputError(ContractError)` with
an existing taxonomy `failure_code` and a fixed code-owned `reason_code`. No
classification by arbitrary exception text; unknown exceptions remain conservative.

Core observation validation first runs the existing native artifact validator.
Then reject exact field groups with these diagnostics:

| Field mismatch | Failure code | Reason code |
|---|---|---|
| native observation schema/semantics | SCHEMA_MISMATCH | CORE_OBSERVATION_SCHEMA_MISMATCH |
| signal_date | DATE_MISMATCH | CORE_OBSERVATION_DATE_MISMATCH |
| Factor pointer/generation identity | POINTER_MISMATCH | CORE_OBSERVATION_FACTOR_BINDING_MISMATCH |
| Market pointer/manifest | POINTER_MISMATCH | CORE_OBSERVATION_MARKET_BINDING_MISMATCH |
| PIT pointer/manifest/membership | POINTER_MISMATCH | CORE_OBSERVATION_PIT_BINDING_MISMATCH |
| Calendar compilation/custody refs | CALENDAR_MISMATCH | CORE_OBSERVATION_CALENDAR_BINDING_MISMATCH |
| alias or sealed signal identity | POINTER_MISMATCH | CORE_OBSERVATION_SIGNAL_MISMATCH |
| explicit expected source byte SHA | SHA_MISMATCH | CORE_SOURCE_SHA_MISMATCH |
| owning reader reports missing file | INPUT_MISSING | CORE_SOURCE_MISSING |

Native artifact validation is not weakened or replaced. Keep old message suffixes
where compatible with the more specific reason. The exception carries diagnostics,
not proof that canonical state is unchanged and not a retry permit.

Catch the typed exception only at read-only dependency/probe boundaries. A fresh
exactly inspected no-attempt request becomes BLOCKED with the precise code/reason.
If an existing attempt is RUNNING, preserve state=RUNNING and its start/attempt,
keep finished_at=null, and report POST_WRITE_IN_DOUBT; do not downgrade it to a
safe input failure. A previously
SUCCEEDED node whose native inputs no longer validate is STALE. For an existing non-success terminal, preserve its recorded lifecycle state,
start/finish timestamps, attempt and immutable terminal failure. Add the current
dependency diagnosis separately and suppress currently unverified output refs in
the observational common fields. Exceptions during or after execute stay
under the existing POST_WRITE_IN_DOUBT handler, even if they are this new type.

Preserve the existing SKIPPED lifecycle for downstream nodes that cannot run, but
add deterministic `upstream_blockers` diagnostics to the current invocation's
status projection. Flatten already-known direct/transitive causes into exact
`node_id`, `failure_code`, `reason_code` rows, deduplicated and sorted. Top100 must
therefore explicitly identify LOW/W80 date mismatch while remaining non-ready and
never invoking its publisher. No dependency is satisfied by diagnostic metadata.
These are observed invocation diagnostics, not a new immutable attempt/receipt
surface; independent readback still does not trust the mutable projection.

The pure `dependencies` helper also rejects non-NodeState values. The runner
already converts journal states to this enum; no valid current caller changes.

## EXECUTE-only producer context admission

In `verify_execution_install_and_research_policies`, before any loop construction,
maintenance, claim or provider, require:

- `release_commit`: the exact full commit from the already-verified
  release-install input payload;
- `calendar_capture_parent`: a nonempty normalized canonical absolute path,
  without `..`, symlink aliases or a non-directory existing target.

The path may be the existing separately governed release-authority capture root;
do not restrict it to the workspace or create it during preflight. Actual native
filesystem checks remain authoritative at write time. Do not infer write permission
from a successful path check. Keep passive/archived `read_factor_loop_context`
compatible with existing minimal contexts that do not invoke the producer.

Missing/wrong producer fields produce explicit typed preflight reasons and zero
business writes. Correct the synthetic fixture input factory to declare these
required fields from its real verified installation, never by editing old refs.

## Verification and preserved failures

Architect then Critic before implementation. Test each native-valid observation
mismatch group, malformed native observation, false state types, source SHA drift,
precise transitive Top100 cause/no publisher call, existing RUNNING protection,
post-write typed-error protection, and valid native core publication/replay.
Test incomplete/incorrect producer contexts before claims/providers/directories and
preserve legacy passive context readback. Run Phase 1 compatibility and relevant
core/execution-control tests, then stop this phase when its contract is met.

The failed installed historical request and its immutable bindings/core remain
untouched. Its receipt is retained under
`/private/tmp/myquant-public-history-native-20260909T031046Z`.
Do not retry that incomplete context or hand-edit its hashes. Any later corrected
fixture context is a new immutable input revision. A finalized maintenance replay
still requires an already validated Calendar capture; it may not acquire a new
provider response as a side effect of recovery.

## Accepted Architect precision

Add nullable `reason_code` to the observational common fields. Add an exact
`dependency_error` pair (`failure_code`, `reason_code`) only on rows that currently
failed a code-owned dependency check. Keep any existing free-text `reason` separate.
The exception constructor accepts only a code-owned reason mapping to one existing
failure code; arbitrary messages cannot classify an error.

Explicit expected source-byte SHA is checked by the existing owning reader BEFORE
JSON/native observation parsing. This native custody check must not be moved later.
Within the resulting parsed observation, the first failing group wins in this
order: native schema/semantics, signal date, Factor binding, Market binding, PIT
binding, Calendar binding, then alias/sealed-signal identity. Thus byte custody and
in-observation precedence are separate deterministic layers.

Catch only immediately around `adapter.probe` after exact journal inspection:

| Existing recorded state | Current observational handling |
|---|---|
| NOT_STARTED / exact attempt0 | BLOCKED; typed failure and reason; no times/output refs |
| RUNNING | Keep RUNNING/start/attempt; POST_WRITE_IN_DOUBT; null finished_at; current dependency pair retained |
| SUCCEEDED | STALE with typed failure/reason; no trusted standard output refs |
| FAILED/BLOCKED/PARTIAL terminal | Keep lifecycle, original start/finish/attempt and original terminal failure; attach current dependency pair; standard output refs empty |

For non-success terminals, `blocking_reason`/`retryable` continue to reflect the
original recorded failure; current dependency failure/code are available in the
separate pair and `reason_code`. This prevents a new retryable missing-input
finding from overriding a prior non-retryable writer failure. Raw immutable
terminal content is unchanged and remains historical evidence, not a current
native-output validation claim. Public readback obtains its own current byte
validation and does not read this mutable invocation diagnosis.

`upstream_blockers` contains sorted, deduplicated root rows with exactly `node_id`,
`failure_code`, `reason_code`. Propagate a parent's existing root rows; otherwise
include its current dependency pair and, if different, its original/indeterminate
failure (with unknown reason_code=null). Never replace specific leaf causes with
intermediate UPSTREAM_INCOMPLETE rows. Without any specific cause use that parent
and UPSTREAM_INCOMPLETE/null. All cause propagation is pure data; no new scans,
state changes, native validations, retry permits or source substitutions.

Producer capture-parent checks are lexical canonical absolute path, no `..`, no
symlink in existing components, and directory type for an existing target. Missing
tail directories are permitted; native write-time ownership/permission checks
remain authoritative. No directory creation, lock, claim, provider or loop occurs
as part of the check. Minimal passive/archived contexts remain readable.

## Implementation acceptance

COMPLETED_LOCAL_VALIDATION. All 135 focused checks passed; after the final test
import cleanup, 16 execution-control checks passed again. Six source files passed
mypy, eight source/test files passed flake8, and nine files passed Black. Exact
current source hashes and the legacy helper lint limitation are retained in
`.agent/acceptance/phase2-validation.json`. This does not certify later phases,
production deployment, or source changes after this receipt.

Missing-source classification recognizes FileNotFoundError directly, or the exact
SystemStorageError type with FileNotFoundError as its cause. It does not downgrade
SystemSecurityError subclasses and does not inspect exception text.
