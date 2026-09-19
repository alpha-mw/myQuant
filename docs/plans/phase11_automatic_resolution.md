# Phase 11 Part B — automatic resolution through daily-close

Status: COMPLETED_LOCAL_VALIDATION after all six Architect amendments and Critic
APPROVE. Final combined Phase11 suite:382 PASS33.54s, nine Part B source mypy,
sixteen-file Black, scoped flake8 and public CLI complexity10 PASS. Receipt:
`.agent/acceptance/phase11-validation.json`. Component tests explicitly control
full EOD/installed provenance and unrelated native producer admission; they do
not certify an installed full DAG or live/unattended run. Part A is locally validated; this
document covers the remaining automatic anchor/target and invocation recovery.
No deployment, scheduler, live-provider call, trading or financial-policy change.

## Interface and source authority

Add `cn-daily-automatic-request.v1` to the same public `production daily-close`
request reader/installed dispatcher. Exact fields:
`schema_version, market, strategy_id, graph_sha256, action, release_install_ref,
calendar_ref, raw_calendar_ref, seed_completion_ref, recipe_ref, day_input_refs`.
Scope remains CN/aggressive_tech_manufacturing/current graph; action is PLAN or
CATCH_UP. All refs are exact; seed may be null, recipe_ref is required and points
to collection v2. Recipes and native inputs are disjoint source declarations.
There is no caller-supplied target or previous date.

Expected close comes only from native replay of the supplied Calendar/raw capture.
First resolution requires its observed Shanghai date to equal the actual local
date, observation <= actual clock, and the target to be a closed open session in
that Calendar. The standard native Calendar acquisition supplies these refs;
this resolver does not introduce another provider client or authorize live calls
during local verification. PLAN is a zero-write operation over those exact refs.
An existing immutable resolution resumes using its original Calendar and modes;
it never substitutes a newer Calendar or retimes an old current request.

Use Phase9 completed-head JSON as a locator only. Validate its exact contract and
its completion through full native EOD replay, matching provenance and scope.
Reject a malformed present head, unsafe alias, future head or Calendar coverage
gap. Do not fall back from a corrupt head. If the JSON is missing but Phase9's
head mirror exists, report an interrupted/conflicting head, not a missing seed.
If both are absent, require the explicit initial completed seed and fully replay
it. Ordinary bootstrap EXECUTE remains the first-day setup path.

Inspect only fixed completion.v1.json paths for native Calendar open dates after
that locator through the authorized close. Native-validate every present EOD and
its own input/Calendar/predecessor edge. Exact predecessor refs are checked where
the native handoff supplies them, using Part A's helper. Advance the anchor only
over a contiguous completed prefix; a later completion beyond a missing date is
a conflict before any writer. This is a bounded Calendar-derived enumeration,
not a latest-result directory scan or another completion index.

Select the remaining dates automatically. The supplied collection may cover
completed dates as well as missing ones within this locator-to-target range;
derive a new filtered collection and exact explicit CATCH_UP request for the
remaining dates. Preserve source bytes. Missing per-date ownership reports the
exact dates and prevents all new business work. Validate template shape/source
refs and supplied-input routing before committing a runnable resolution. When
target already equals the resolved anchor, retain a one-row replay of the target
so Part A still enforces current-serving scope if appropriate.

## Immutable resolution and bounded overlap

Confine orchestration metadata below
`results/operations/daily_production/CN/automatic/aggressive_tech_manufacturing/`.
Use one descriptor-safe nonblocking `.run.lock`. Lock order is automatic run,
then one day, then native publication where already required; no reverse call.
A busy lock raises expected `AUTO_RUN_BUSY` immediately (exit 2), with no wrapper
or wait. Unknown/corrupt pending state raises an expected conflict and is never replaced.
The ordinary JournalStorage writer's allowed paths do not expand; a narrowly
scoped storage wrapper owns only this lock, run files and one pending-run record.

For K = SHA256(canonical exact `{path, sha256}` auto-request ref), use immutable
`runs/K/resolution.v1.json`, `collection.json`
and `request.json`. Resolution is one self-contained canonical document, written
before the other two and before business callbacks. Exact fields:
`schema_version, auto_request_ref, market, strategy_id, graph_sha256,
release_install_ref, calendar_ref, raw_calendar_ref, observed_head, locator_ref,
anchor_ref, adopted_completion_refs, target_trade_date, ordered_trade_dates,
day_scopes, derived_collection, derived_collection_ref, derived_request,
derived_request_ref, resolved_at, authority`.
Schema is `cn-daily-catchup-resolution.v1`; authority is the existing all-false
mapping. observed_head is null or the original head document plus exact observed
JSON and mirror byte SHAs. Derived refs use the fixed paths above and canonical
bytes of their embedded documents. day_scopes uses Part A's exact routing enums.
adopted_completion_refs records only the contiguous prefix after the locator.
All date order, path, ref, source, embedded-byte and scope fields are reconstructed
and validated. Readback uses the retained head document, not the mutable head.
Missing derived files after a crash can be written from this exact resolution;
conflicting existing bytes fail. No timestamp or head is reselected on resume.

After resolution and both derived files read back, atomically record the one
`pending-run.v1.json` before dispatch. Exact fields: schema_version, state,
auto_request_ref, resolution_ref; state ACTIVE or IDLE. This is a recovery lease,
not EOD authority. Its only mutable path is code-owned, mode0600, descriptor
validated and replaced under the automatic lock with readback/fsync. No unlink.
The same request resumes the saved resolution. A different request while an
ACTIVE run still has missing EOD work returns AUTO_PENDING_REQUEST_CONFLICT with
the exact pending request ref; it cannot replace an old claim or silently adopt
new policies/inputs. EOD and fallback use the same prepared immutable request.

When every resolved EOD is natively complete, the lease can become IDLE even if
the last current presentation is incomplete/expired. Record no fabricated serving
success: return the existing Part A result, whose current gate remains incomplete.
A later new target can consume that EOD as historical evidence. If a crash leaves
ACTIVE after all EODs complete, a later invocation proves this same full lineage
before marking it IDLE. A blocked current request that lacks native completion
remains explicit pending work except the narrowly proven EXPIRED_UNSTARTED closure
below; no historical mode rewrite or replacement of an existing native claim is allowed.
An immutable saved resolution alone, before lease activation, has started no
business work and does not prohibit another request from resolving.

## Result and integration

Add an exact `cn-daily-automatic-result.v1` wrapper with action, auto_request_ref,
target_trade_date, anchor_ref, adopted_completion_refs, ordered_trade_dates,
day_scopes, missing_input_dates, resolution_ref, status, result and authority.
PLAN returns no resolution_ref/result and status PLANNED or BLOCKED, and writes
nothing (including no lock/directory). CATCH_UP returns the existing validated
production-result.v1 in result, with status copied from its execution_state.
Early input/conflict failures remain typed exceptions; do not fabricate rows.

The installed dispatcher validates and executes the resolution, then delegates
only to Part A's existing CATCH_UP path. The public boundary validates the wrapper,
exact originating request, native Calendar target and stored resolution, plus
every returned completion through the installed native completion replay. Add
one read-only installed automatic-resolution reader if required for that boundary;
do not trust declarations from a returned wrapper or use host-side fallbacks.
Legacy v1/v2 requests/results and public commands keep their existing behavior.
The shared production exit-code helper must dispatch by the known result schema:
automatic PLAN PLANNED exits 0, BLOCKED exits 2; automatic CATCH_UP uses its
validated nested production result's existing exit rule. Unknown schema or
inconsistent outer/nested status remains an error, never an implicit success.
Add the read-only resolution operation explicitly to native_bridge's fixed
installed operation mapping and verify its origin like the existing daily-close
and completion-replay entries; no dynamic-import or missing-operation fallback.

## Acceptance and stop conditions

- Missing/corrupt/head-mirror-only/future head, invalid seed, Calendar gap,
  nonadjacent or wrong-ref completed prefix and completion beyond a gap all stop
  before business writes; no latest scan or mutable-head substitution.
- Native Calendar selects target and gaps; weekend/holiday latest close remains
  historical; same-day current policy follows Part A. Caller cannot override dates.
- PLAN is byte/mtime and directory-inventory unchanged, including a busy lease.
- Each resolution/fileset/lease crash boundary recovers exact bytes or conflicts;
  source/embedded-byte tampering and changed modes/refs fail even when rehashed.
- Overlapping invocations return promptly; same request resumes, conflicting new
  request cannot steal incomplete work; fully complete EOD with failed serving can
  release the lease without claiming publication success.
- Repeated completed execution writes nothing and performs no producer callbacks.
  Explicitly test a moving/missing current head after resolution and provenance.
- Public installed-dispatch contract tests and native Calendar/EOD lineage tests,
  plus focused legacy Part A and storage regressions. Full installed current-source
  full DAG remains Phase15; tests must disclose controlled native seams.

No source implementation before both reviews. Resolve only concrete review gaps;
do not add a scheduler, general task database, new completion authority or loose
input recovery. Preserve old failed requests and all unrelated worktree changes.

## Accepted Architect amendments

All six Architect items are accepted. These exact rules supersede any shorter
description above:

1. Head/seed exclusivity: either head form present requires a null seed. A valid
   JSON must have its exact deterministic JS mirror; mirror-only is interrupted.
   Both absent requires a fixed-path seed within Calendar coverage and no later
   than target, fully replayed. Read/recheck both files as one pair at initial
   selection. Change/inconsistency returns a conflict before resolution creation.
   `observed_head` is exactly null or `{document, json_sha256, mirror_sha256}`;
   derive mirror bytes through native `head_js` from canonical head bytes.

2. `ordered_trade_dates` is the full Calendar-open range strictly after locator
   through target. `adopted_completion_refs` is its exact completed prefix;
   anchor is locator or prefix tail. Derived collection/request contain only dates
   strictly after anchor. Reject extra recipes/inputs outside the full range and
   overlapping ownership. `day_scopes` is a date-keyed map for the full range,
   plus the target replay row when target equals anchor. Thus zero-gap replay is
   explicit even when the full range is empty. Readback reconstructs these exact
   meanings, not merely sorted-date/hash consistency.

3. `resolved_at` must be >= native Calendar observation, observed head registration
   and every admitted EOD's native validation time, and <= actual clock. Resume
   preserves it. Run identity hashes the exact request ref, separately validating
   request content SHA. Busy/pending conflicts are expected errors, exit 2.

4. Add a narrow `EXPIRED_UNSTARTED` terminal closure for a pending resolution whose
   original CURRENT day is now before the actual Shanghai date. Keep the old
   resolution/modes/requests intact. Fully replay any completed prefix first.
   For every unresolved day, prove absence of native logical claims/attempts,
   handoff/materialization/completion and DAG producer-start/callback evidence.
   Use the fixed native maintenance root `data/private/cn_daily_maintenance`,
   retained native journal claim/attempt semantics, and the known per-day DAG
   paths. Do not classify absence from a missing completion alone. Missing/unsafe
   roots, unbound attempt evidence, unknown day files or uncertain ownership mean
   absence is unproven and the run remains ACTIVE. A filesystem inventory used
   for start-evidence exclusion is never a source for EOD/anchor selection.

   Hold the automatic lock, existing native maintenance lock, then per-day locks
   in increasing Calendar order while checking and sealing closure. All are
   nonblocking; release and report conflict on contention. Normal Part A releases
   its short day binding lock before native maintenance. Native callbacks acquire
   day locks under maintenance, so this closure order must match that order.
   The absence check is non-creating for native maintenance roots/locks. Existing
   logical claim directories or any start evidence prohibit this closure even if
   their recorded business write count is zero. This intentionally narrow path
   handles a crash after lease activation but before actual native work.

   Persist immutable `runs/K/closure.v1.json` before making the lease IDLE, schema
   `cn-daily-catchup-closure.v1`, exact fields `schema_version, resolution_ref,
   state, completed_prefix_refs, unresolved_trade_dates, checked_at, authority`.
   state is EXPIRED_UNSTARTED; authority is all-false. `checked_at` is the actual
   under-lock clock. A closure repeat validates exact recorded inputs/proofs and
   preserves bytes/time; it does not rerun the old request. Before IDLE recovery
   after a closure-write crash, recheck absence under the same locks. New native
   evidence after closure is a conflict, not permission to activate that request.
   A new automatic request can then derive historical work from a fresh Calendar.

5. Automatic-derived CATCH_UP dispatch requires an in-process, nonserializable
   capability valid only while that exact workspace/request/resolution holds the
   automatic lock. Reject direct public execution of automatic-root request or
   collection paths without it, including a copied request that still references
   the owned collection. Public direct EXECUTE of a generated per-day catch-up
   request/recipe is also rejected by its fixed generated path; those entries are
   private to the existing catch-up controller. Private native calls remain
   reached only through the validated controller. This prevents an old owned
   request from racing the EXPIRED_UNSTARTED absence proof through public dispatch.
   A closed resolution never receives a capability again. The capability carries
   no financial authority, is revoked in finally, and cannot bypass Part A checks.

6. Branch on the exact automatic request schema before the legacy validator in
   both public and installed dispatch. Add fixed installed resolution readback.
   Public exit handling validates known schemas: automatic PLAN PLANNED=0,
   BLOCKED=2; CATCH_UP uses the validated nested result. Unknown/inconsistent
   wrappers remain exit 3. Test origin and missing-operation rejection, no fallback.

Extend acceptance with exact-ref same-bytes/different-path identities, mismatched
head mirrors, stale-unstarted closure at every lease/closure boundary, no release
when any start/unknown evidence exists, direct owned-request rejection, revoked
capability refusal, and safe progress of a newer historical request after closure.

Implementation details consistent with the approved boundaries: execution primes
coordinator day lock files before ACTIVE, so a crash immediately after lease
activation has existing locks for the non-creating absence proof. Native
maintenance roots/locks are never created by closure. A matching IDLE lease may
serve a completed repeat only after full native completion proof while holding
the automatic lock; it is not rewritten ACTIVE/IDLE merely for replay. This keeps
all completed-repeat bytes/mtimes unchanged. The installed resolution reader also
accepts an exact PLAN request ref with resolution_ref=null for independent
read-only public-result validation. Expected automatic errors validate their
bounded fields across installed/host Python class identity; no fallback reader.
