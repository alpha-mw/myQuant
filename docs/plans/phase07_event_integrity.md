# Phase 7 prerequisite: native event closure integrity and frozen readback

Status: APPROVED by Architect amendment adoption and final Critic APPROVE; bounded repair locally verified on 2026-09-12.

The full objective remains Phase0–15. This change repairs a necessary financial
input contract; it does not claim all Phase7/8 requirements are already fulfilled.
The source design explicitly forbids this run from modifying holdings, owner
stops, trading strategy, Paper authority or broker state. No live provider or
actual portfolio operation is part of local verification.

## Current evidence and defect

The current native daily close consumes seven CLOSED_EMPTY event dimensions:
executions, orders, fills, funding, cost_basis_changes, corporate_actions and
manual_changes. Unknown/nonempty dimensions must not become a no-action close.
The existing derived accounting module supports fills but does not authorize
Store mutation. Nonempty financial-event integration therefore remains a separate
explicitly governed contract, not an implied extension of the no-action policy.

`phase7-event-validation-before.json` reproduces four accepted corrupt closures
with valid recomputed content hashes: seal before cutoff, malformed seal, null
cutoff and an extra override field. `build_empty_closure` checks chronology but
`validate_closure` does not replay that check. Native generation/pointer readers
also lack equal validation coverage and the corporate reader ignores its retained
pointer bytes, traversing current-head ancestry again during historical replay.

## Repair the owning v1 contract

Add code-owned exact field sets for the existing closure, generation and pointer
shapes, using the fields produced by the existing builders. Reject unknown fields,
invalid content hashes, non-object nested values, malformed refs, wrong schema,
forbidden authority values, invalid/duplicate/unsorted dates, generation identity
or pointer/generation date-set mismatch. Strict bool false remains mandatory.

Use a shared native timestamp validator. Preserve valid explicit timezone-aware
ISO timestamps including existing offsets and subsecond precision; compare parsed
instants. Require sealed_at >= cutoff_at for a closure, and generated_at >= every
closure sealed_at for a generation. Trade dates must be canonical ISO dates.
Require cutoff's Shanghai session date to match trade_date, consistent with the
native daily owner-cutoff producers. Missing/naive/malformed timestamps reject.
The builder and readback must use the same validation; a rehashed invalid object
is still invalid. No wall clock, file time or target-date backdating is used.

Policy and owner refs are exact workspace-relative path/SHA pairs. Retain the
existing native source_receipt_ref alternative `catalog:<generation>#receipt:<id>`
explicitly, with bounded identifier grammar; do not silently interpret it as a
physical file. Do not newly require historical closures to share a generation's
current policy ref: the original API can carry historical policies through a
successor, and they remain individually bound.

Shared generation decoding enforces the exact safe
`generations/<generation_id>.v1.json` path, native canonical JSON/content hashes,
expected table-byte SHA, date lists, generation IDs and false authority. Preserve
all existing successor/CAS/history/immutable-restatement behavior. Validate all
inputs before publication; an invalid timestamp must not leave a generation or
pointer written. Apply the same decoder to current and registered-ancestor readers
so they cannot disagree. No new writer, transaction or compatibility path.

## Frozen corporate readback

Add an owning read-only `load_frozen_generation(root, pointer_bytes,
expected_pointer_sha256)` API. It validates bounded exact pointer bytes and resolves
only their declared immutable generation. It does not read current.v1.json, walk
current ancestry, acquire a lock or write. Recheck native generation bytes before
return. Its caller establishes provenance through the original Store-plan event
preimage and retained corporate recipe.

Corporate adapter preparation retains its existing registered-ancestor validation
before copying pointer bytes into the day journal. Runtime probes and completed
corporate replay thereafter use that exact retained pointer. Completed replay still
checks the recipe SHA against the original native Store plan. Do not infer an
unretained historical preimage, scan for a latest generation, or alter old receipts.
Existing valid recipe/output shapes remain unchanged.

Catalog pseudo-ref resolution is included below because the canonical workspace
has four actual closures using it. Named action observations/adjustments remain
Phase8 work. No unresolved moving threshold becomes executable.

## Acceptance

- Rehashed early/malformed/null-time/extra-field closures reject in builder,
  publisher, current loader and frozen/ancestor loader.
- Nonempty events in every dimension, unknown/missing dimensions, wrong authority,
  malformed/unsafe refs, malformed generation clocks and wrong date/ID/path/SHA
  bindings reject before any native write.
- Valid offsets/subseconds and cutoff equality work; cutoff session mismatch
  rejects. Generation equality at latest closure seal works.
- Frozen readback succeeds with current pointer absent/corrupt/advanced, while
  generation tampering and out-of-root/symlink paths reject. No locks/writes.
- Native corporate adapter and completed replay retain exact original output bytes
  after current-head change; valid legacy v1 values remain readable.
- Existing native Store five-day/no-position-change/backlog/CAS recovery checks
  pass without financial-state mutation in real data.
- Record focused checks/source hashes and remaining full-scope gaps. Full installed
  current-source DAG/CI remains Phase15; no whole-phase/whole-goal completion claim
  based only on this contract repair.


## Accepted Architect amendments: native catalog receipt resolution

The canonical read-only audit `.agent/acceptance/phase7-event-current-shapes.json`
shows nine current dates, no time-order violations, exact builder keysets, and
four real catalog pseudo-refs. Those four point to generation
g20260828T142512-4ff10dbe and date-specific native no-action receipts. Their hashed
owner declaration binds the receipt IDs and seven explicit empty event dimensions.

The exact closure field set is: schema_id, trade_date, sealed_at, cutoff_at,
status, dimensions, policy_ref, owner_declaration_ref, source_receipt_ref,
late_event_behavior, actual_holdings_mutation_authority, cash_mutation_authority,
broker_order_trade_authority, content_sha256.
Generation: schema_id, generation_id, generated_at, policy_ref, trade_dates,
closures, late_event_behavior, broker_order_trade_authority, content_sha256.
Pointer: schema_id, generation_id, generation, trade_dates,
previous_pointer_sha256, broker_order_trade_authority, content_sha256.
Nested generation ref is exact path/SHA; previous_pointer_sha256 is null or a
valid SHA. Dates are sorted unique nonempty and equal in every layer.

Physical refs require canonical relative POSIX paths, nonempty ASCII text, no
absolute/traversal/backslash/repeated separator/control component, plus lowercase
64-hex SHA. Only source_receipt_ref can instead use exactly
catalog:<generation-id>#receipt:<receipt-id>. Both identifiers use bounded ASCII
letters/digits/underscore/hyphen/dot, must start alphanumeric, forbid dot/dotdot
components and additional colon/hash/slash delimiters. Generation length <=128;
receipt length <=256. Unknown suffixes reject.

Add the read-only resolver in the native strategy_records package. It considers
only catalog.v1.json, catalog.v2.json and catalog.v3.json within the exact named
_store/catalogs/<generation-id> directory (actual prefix `_record_store/catalogs`).
Require exactly one existing candidate and validate it against its matching schema;
multiple candidates are ambiguous even if only one parses. No recursive scan,
mtime ordering or current-pointer lookup. Apply native canonical/content-hash,
generation/schema/catalog and external-binding validators. Retain/recheck exact
catalog bytes. Return its physical path/SHA as diagnostic provenance; do not change
existing corporate projection field sets or persisted pseudo-ref values.

The supported symbolic receipt schema is exactly
myquant.strategy_record_no_action_receipt.v1. Its exact fields are schema_id,
receipt_id, created_at, status, reason, active_record_id, active_checkpoint,
payload_copied, v17_mainline_authority, broker_order_trade_authority, content_sha256.
Require unique receipt ID, exact semantic SHA, NO_ACTION, nonempty reason,
payload_copied=false and both authority flags strictly false. Its checkpoint must
match native `_active_closure` for the receipt's referenced registered record in
that catalog, and its created_at Shanghai date equals closure trade_date.
Unknown receipt schemas fail. Do not invent absent holdings/cash authority flags
for this schema; its fixed no-action/status/checkpoint contract is the applicable
native non-mutation assertion.

Extract/reuse a schema-specific intrinsic no-action receipt validator from the
existing manager helper `_find_and_validate_continuity_receipt`. Existing mutation
callers must still separately require the current pointer's exact active checkpoint;
the historical resolver merely proves the explicitly named receipt and its record,
never manufactures a committed pointer or grants mutation authority.

For a catalog pseudo-ref, additionally require the SHA-bound retrospective owner
declaration's exact trade-date row to identify this source_receipt_id and have all
seven event dimensions empty, plus its existing owner/policy/authority/time contract.
The no-action receipt is a cross-check, not a substitute for the explicit owner
empty-event declaration. Physical maintenance-attempt refs retain the existing
physical SHA-read path; this repair does not infer their event content.

Corporate preparation alone may verify registered event ancestry. All subsequent
probes/execute/completed replay read the exact retained pointer. A syntactically
valid but unresolved pseudo-ref is a typed provenance failure and cannot satisfy
the Store dependency. Add real canonical four-receipt read-only verification,
unknown receipt/date/checkpoint/authority negatives, ambiguous catalog versions,
symlink/path escape and current-Store-pointer-forbidden tests.

Implementation readback found native repository policy/owner files use 0644. Physical refs use the owning bounded stable file reader, permit owner-owned 0400/0600/0644 only, reject symlink aliases/group writes, and retain SHA verification. No source permissions were changed.
