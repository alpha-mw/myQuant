# Phase13 research input failure mapping

Status: IMPLEMENTED_LOCAL_VALIDATION. Architect amendments accepted; Critic APPROVE.
Validation: `.agent/acceptance/phase13-research-diagnostics-validation.json`.
This is a bounded extension of the existing
dependency diagnostic contract, not a new retry or write authority.

## Problem and intended behavior

Before this change, research wrapper, cutoff, source and source-bundle validators raised
generic ContractError/CommandError. DayRunner preserves only DependencyInputError
from probes, so known missing files, wrong SHA, wrong dates, incomplete upstream
products and source-shape failures lose their category and reason.

Extend the existing immutable REASONS registry and raise DependencyInputError at
the owning check. Keep existing reason strings where the cause is unambiguous.
Split combined missing-versus-SHA checks into exact reasons. Do not infer a
category from arbitrary exception text, exception chains, or caller flags.

## Scope and compatibility

- Own operations/dependency_diagnostics.py, research_request.py,
  research_sources.py, research_projection.py, research_cutoff.py,
  research_cutoff_contract.py, research_source_bundle.py and
  research_file_readback.py; daily_runner.py owns only the narrow rethrow below.
  Add focused tests and update audit/register/ledger.
- For the native CLI file reader only, add an internal CommandError subclass
  carrying a fixed missing/SHA reason. Preserve public blocker_code, status,
  fields, exit codes, and successful bytes/ref behavior. Establish workspace
  containment before classifying a missing target; unsafe paths, permissions,
  unstable reads and unknown native failures remain generic. ResearchFileReadback
  alone translates this typed native failure into a DependencyInputError.
- No immutable document fields or stored historical receipts are rewritten.
  Existing callers catching ContractError or CommandError still work.
- Exact categories: absent bytes -> INPUT_MISSING; differing expected byte hash
  -> SHA_MISMATCH; day -> DATE_MISMATCH; ref/lineage/mode mismatch ->
  POINTER_MISMATCH; incomplete required native upstream -> UPSTREAM_INCOMPLETE;
  malformed shape -> SCHEMA_MISMATCH; immutable write conflicts ->
  IDEMPOTENCY_CONFLICT. Clock/deadline corruption, ambiguous combined checks,
  security/permission and unknown failures remain their existing generic errors
  unless an owning check supplies one exact cause.
- SecureSystemStorage missing conversion follows existing CoreContext rules:
  direct FileNotFoundError, SystemNotFound, or exact SystemStorageError with
  immediate FileNotFoundError cause only; subclasses/security errors never
  downgrade based on their nested causes.
- Preserve DayRunner exception recovery semantics. Add only an outer
  `except DependencyInputError: raise` before its generic catch so constructor
  failures during registry resolution retain their type without any attempt claim.
  Read-only probe failures
  use rejected_probe: NOT_STARTED blocked, SUCCEEDED stale, RUNNING retains
  POST_WRITE_IN_DOUBT, historical failed terminals retain their original failure.
  Typed failures during execute remain POST_WRITE_IN_DOUBT without fake terminal.
  Constructor/materialization failures outside a node probe propagate as typed
  errors; do not claim a node attempt or safe retry without journal proof.

## Acceptance and stop conditions

1. Real malformed/missing/wrong-SHA research files and wrong-day cutoff/source
   bundle produce the exact category/reason; malformed JSON is SCHEMA_MISMATCH
   only from its owning JSON decoder.
2. Native probe -> runner -> downstream blockers preserves the leaf reason and
   retry/owner/next-node fields, with no failed-node writer invocation.
3. Existing RUNNING/FAILED/SUCCEEDED and post-write tests keep custody unchanged;
   no text-based classification or automatic retry is introduced.
4. Native CLI error JSON remains byte-equivalent for supplied public blocker
   codes. Symlink escape, permission/security failures and untrusted codes are
   not classified as recoverable missing input.
5. Run focused diagnostics, runner, source/wrapper/cutoff/native source tests and
   scoped static checks. Fix only regressions caused by this scope. Record source
   hashes and limitations. No live APIs, state promotion, scheduler or book edits.

Phase14 ledger-bound evaluator remains a separate reviewed change. Phase13 full
producer coverage will be assessed from the explicit mapped boundaries rather
than inferred from the taxonomy constants or this bounded research change alone.

## Accepted Architect amendments

Architect APPROVE_WITH_CHANGES: all eight amendments accepted. The table below
is exhaustive for this change. Unknown projector/schema/security/clock/race
failures remain generic. Combined wrapper reread drift remains generic; only the
separate payload SHA branch is typed. Fixed-path conflicts alone receive the
idempotency category. Existing reason strings stay unchanged except explicitly
split missing/SHA/day branches in the table.

The native successful read path remains unchanged, including readable 0644
Parquet. On a direct FileNotFoundError only, a second descriptor-safe native
SecureSystemStorage read must independently prove safe absence before assigning
SAFE_SOURCE_MISSING. On a native SHA failure only, the descriptor-safe read must
return exactly the same bytes before assigning SAFE_SOURCE_SHA_MISMATCH. If the
second read succeeds in the absence case, sees different bytes, or rejects
security/mode/link/ownership/size/containment, the original generic error remains.
This bounded confirmation is only on errors, with no repair, permission change or
retry loop. A valid 0644 file still succeeds normally; a SHA failure on a source
that the stricter confirmation cannot prove stays generic. That limitation is
explicit. No broad file-reader/security rewrite is in scope.

The internal CommandError subclass accepts only those two fixed categories and
does not serialize them. ResearchFileReadback translates only code-owned labels
CUTOFF_SOURCE_REF_INVALID, CUTOFF_FOCUS_PIT_REF_INVALID,
CUTOFF_EVENT_POINTER_INVALID, CUTOFF_EVENT_SOURCE_INVALID,
CUTOFF_ANNOUNCEMENT_REF_INVALID, and PORTFOLIO_SOURCE_REF_INVALID. Other labels,
including arbitrary caller text, preserve the original CommandError. The source
document/secure wrapper readers get their own fixed typed decode/check branches;
unrelated native projector errors remain untouched.

Acceptance includes public CLI stdout golden bytes/exit 2; safe missing leaf and
parent; stable SHA versus changing bytes; absolute/traversal/case alias/symlink/
hard-link/unsafe mode or owner/oversize/permission failures; security errors with
nested missing causes; malformed stable JSON; immutable conflict; constructor
rethrow without journal mutation; and all existing probe/post-write custody states.

## Exhaustive new reason ownership

| Reason | Category | Owning check | Execution stage |
| --- | --- | --- | --- |
| `RESEARCH_REQUEST_DOCUMENT_INVALID` | SCHEMA_MISMATCH | research_request.load_research_request document type | constructor |
| `RESEARCH_REQUEST_WRAPPER_INVALID` | SCHEMA_MISMATCH | research_request.load_research_request exact wrapper shape | constructor |
| `RESEARCH_SOURCE_JSON_INVALID` | SCHEMA_MISMATCH | research_file_readback.parse_research_json exact decoder failure | constructor/probe/recovery |
| `RESEARCH_SOURCE_SCHEMA_INVALID` | SCHEMA_MISMATCH | research_sources.ResearchSources.__init__ request fields | constructor |
| `RESEARCH_COMPANY_SOURCE_SCHEMA_INVALID` | SCHEMA_MISMATCH | research_sources.ResearchSources.__init__ company field type | constructor |
| `CUTOFF_RECEIPT_FIELDS_INVALID` | SCHEMA_MISMATCH | research_cutoff_contract.validate_cutoff_contract exact receipt fields | constructor/recovery |
| `CUTOFF_SOURCE_TIME_SHAPE_INVALID` | SCHEMA_MISMATCH | research_cutoff_contract.ordered_source_times exact row fields | constructor/recovery |
| `CUTOFF_SOURCE_TIME_ROLE_INVALID` | SCHEMA_MISMATCH | research_cutoff_contract.ordered_source_times fixed role grammar | constructor/recovery |
| `CUTOFF_SOURCE_BUNDLE_SHAPE_INVALID` | SCHEMA_MISMATCH | research_source_bundle.SourceBundle.__init__ exact fields | constructor/recovery |
| `CUTOFF_NATIVE_FIELDS_INVALID` | SCHEMA_MISMATCH | research_source_bundle.SourceBundle.__init__ native fields | constructor/recovery |
| `CUTOFF_COMPANY_FIELDS_INVALID` | SCHEMA_MISMATCH | research_source_bundle.SourceBundle.__init__ company fields | constructor/recovery |
| `CUTOFF_AUXILIARY_REFS_INVALID` | SCHEMA_MISMATCH | research_source_bundle.SourceBundle._verify_origins stage keys | constructor/recovery |
| `CUTOFF_FUNDAMENTAL_DESCRIPTOR_INVALID` | SCHEMA_MISMATCH | research_source_bundle.SourceBundle._verify_origins descriptor fields | constructor/recovery |
| `RESEARCH_SOURCE_MISSING` | INPUT_MISSING | research_file_readback.read_research_bytes secure missing read | constructor/probe |
| `RESEARCH_NATIVE_SOURCE_MISSING` | INPUT_MISSING | research_file_readback.ResearchFileReadback.source_file typed native safe absence | constructor/probe/recovery |
| `CUTOFF_RETAINED_REF_MISSING` | INPUT_MISSING | research_cutoff._read JournalStorage returns None | constructor/recovery |
| `CUTOFF_COMMITTED_OBJECT_MISSING` | INPUT_MISSING | research_cutoff._complete missing committed object without repair | recovery |
| `CUTOFF_FOCUS_SOURCE_DECLARATION_REQUIRED` | INPUT_MISSING | research_source_bundle.SourceBundle._verify_origins absent focus declaration | constructor/recovery |
| `CUTOFF_MACRO_ADMISSION_MISSING` | INPUT_MISSING | research_source_bundle.SourceBundle._macro absent native admission | constructor/recovery |
| `RESEARCH_REQUEST_SHA_MISMATCH` | SHA_MISMATCH | research_request.load_research_request exact wrapper SHA | constructor |
| `RESEARCH_REQUEST_PAYLOAD_SHA_MISMATCH` | SHA_MISMATCH | research_request.load_research_request native payload SHA | constructor |
| `RESEARCH_SOURCE_SHA_MISMATCH` | SHA_MISMATCH | research_sources.ResearchSources._read secure source SHA | constructor/probe |
| `RESEARCH_PREVIEW_REQUEST_SHA_MISMATCH` | SHA_MISMATCH | research_sources.ResearchSources.__init__ supplied preview bytes SHA | constructor |
| `RESEARCH_NATIVE_SOURCE_SHA_MISMATCH` | SHA_MISMATCH | research_file_readback.ResearchFileReadback.source_file typed native secure SHA | constructor/probe/recovery |
| `CUTOFF_RETAINED_REF_SHA_MISMATCH` | SHA_MISMATCH | research_cutoff._read retained source SHA | constructor/recovery |
| `CUTOFF_COMMITTED_BYTES_SHA_MISMATCH` | SHA_MISMATCH | research_cutoff._complete reconstructed committed bytes SHA | recovery |
| `RESEARCH_SOURCE_DATE_MISMATCH` | DATE_MISMATCH | research_sources.ResearchSources.__init__ native pool/request day split | constructor |
| `CUTOFF_RECEIPT_DAY_INVALID` | DATE_MISMATCH | research_cutoff_contract.validate_cutoff_contract UTC and Shanghai day | constructor/recovery |
| `CUTOFF_SOURCE_BUNDLE_DAY_INVALID` | DATE_MISMATCH | research_source_bundle.SourceBundle.__init__ declared trade day | constructor/recovery |
| `CUTOFF_CLOCK_TARGET_DAY_CHANGED` | DATE_MISMATCH | research_cutoff.prepare_cutoff_inputs actual cutoff trade day | materialization |
| `RESEARCH_REQUEST_CUTOFF_BINDING_INVALID` | POINTER_MISMATCH | research_request.load_research_request exact wrapper/native refs | constructor |
| `RESEARCH_SOURCE_POOL_BINDING_INVALID` | POINTER_MISMATCH | research_sources.ResearchSources.__init__ pool origin refs/policy after day split | constructor |
| `RESEARCH_SOURCE_REQUEST_MISMATCH` | POINTER_MISMATCH | research_sources.ResearchSources.project request vs fixed template | probe/execute |
| `FOCUS_RECIPE_CONTEXT_MISMATCH` | POINTER_MISMATCH | research_sources.ResearchSources.project exact focus context | probe/execute |
| `CUTOFF_COMMITMENT_PATH_INVALID` | POINTER_MISMATCH | research_cutoff.read_cutoff_inputs fixed path and bundle ref | constructor/recovery |
| `CUTOFF_TIMING_POLICY_MODE_MISMATCH` | POINTER_MISMATCH | research_source_bundle.SourceBundle.__init__ timing mode binding | constructor/recovery |
| `CUTOFF_HANDOFF_MODE_MISMATCH` | POINTER_MISMATCH | research_source_bundle.SourceBundle.__init__ handoff mode profile | constructor/recovery |
| `CUTOFF_RETAINED_HANDOFF_PATH_INVALID` | POINTER_MISMATCH | research_source_bundle.SourceBundle.__init__ fixed retained paths | constructor/recovery |
| `CUTOFF_CORE_PATH_INVALID` | POINTER_MISMATCH | research_source_bundle.SourceBundle._verify_origins fixed Core path | constructor/recovery |
| `CUTOFF_PINNED_THEME_MISMATCH` | POINTER_MISMATCH | research_source_bundle.SourceBundle._verify_origins pinned Theme origin | constructor/recovery |
| `CUTOFF_ACQUIRED_THEME_MISMATCH` | POINTER_MISMATCH | research_source_bundle.SourceBundle._verify_origins acquired Theme binding | constructor/recovery |
| `CUTOFF_STAGE_ATTEMPT_MISMATCH` | POINTER_MISMATCH | research_source_bundle.SourceBundle._verify_origins stage attempt path | constructor/recovery |
| `CUTOFF_AUXILIARY_SOURCE_MISMATCH` | POINTER_MISMATCH | research_source_bundle.SourceBundle._verify_origins stage source binding | constructor/recovery |
| `EXPOSURE_THEME_UPSTREAM_INCOMPLETE` | UPSTREAM_INCOMPLETE | research_sources.ResearchSources.project / research_projection.project_research_source required Theme | probe/execute |
| `FOCUS_MEMBERSHIP_UPSTREAM_MISSING` | UPSTREAM_INCOMPLETE | research_sources.ResearchSources.project / research_projection.project_research_source required membership | probe/execute |
| `CUTOFF_CORE_TERMINAL_INCOMPLETE` | UPSTREAM_INCOMPLETE | research_source_bundle.SourceBundle._verify_origins unfinished Core terminal | constructor/recovery |
| `CUTOFF_AUXILIARY_INCOMPLETE` | UPSTREAM_INCOMPLETE | research_source_bundle.SourceBundle._verify_origins unfinished native auxiliary stage | constructor/recovery |
| `CUTOFF_CRITICAL_SOURCE_INCOMPLETE:industry` | UPSTREAM_INCOMPLETE | research_source_bundle.SourceBundle.project/_macro required industry projection | constructor/recovery |
| `CUTOFF_CRITICAL_SOURCE_INCOMPLETE:theme` | UPSTREAM_INCOMPLETE | research_source_bundle.SourceBundle.project/_macro required theme projection | constructor/recovery |
| `CUTOFF_CRITICAL_SOURCE_INCOMPLETE:exposure` | UPSTREAM_INCOMPLETE | research_source_bundle.SourceBundle.project/_macro required exposure projection | constructor/recovery |
| `CUTOFF_CRITICAL_SOURCE_INCOMPLETE:fundamental` | UPSTREAM_INCOMPLETE | research_source_bundle.SourceBundle.project/_macro required fundamental projection | constructor/recovery |
| `CUTOFF_CRITICAL_SOURCE_INCOMPLETE:macro` | UPSTREAM_INCOMPLETE | research_source_bundle.SourceBundle.project/_macro required macro projection | constructor/recovery |
| `CUTOFF_SOURCE_BUNDLE_CONFLICT` | IDEMPOTENCY_CONFLICT | research_source_bundle.SourceBundle.__init__ existing fixed bundle bytes differ | constructor/recovery |
| `CUTOFF_COMMITTED_OBJECT_CONFLICT` | IDEMPOTENCY_CONFLICT | research_cutoff._complete existing fixed object differs | recovery |
