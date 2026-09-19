# Phase 7 Part C: registered-source batch and daily integration

Status: C1 Architect amendments accepted; Critic APPROVE;
IMPLEMENTED_LOCAL_VALIDATION. C2 remains unimplemented.

C1 evidence: 192 combined checks passed in27.27s across native registered close,
coverage, performance, declaration, old Store/adoption/portfolio/Decision and
closed native imports. A subsequent Calendar-target precision fix and explicit
morning-calendar case passed22 focused checks in11.67s. Counts overlap. The
implementation review found one retrospective-input adoption bypass; it was
fixed before both branches and three negative cases passed. Critic confirmed
that blocker resolved. Original readonly reproduction is
`.agent/acceptance/phase7-registered-close-before.json`; acceptance is
`.agent/acceptance/phase7-c1-validation.json`.

The v2 native transaction, Store adapter, baseline portfolio and frozen readback
are exercised on synthetic owner BUY data. Source planning, recipe/preparation,
cutoff/corporate/Dashboard wiring and actual installed daily operation are not
certified by these tests. No actual account, provider or scheduler write occurred.
Full Phase7 and the original Phase0–15 goal remain open.

## Required completed path

Owner facts -> exact registered intraday source -> versioned daily inputs ->
corporate/Decision gates -> existing native official-close writer -> Store/EOD
publication -> frozen readback and Dashboard. Decision uses the previous official
book; the writer uses the already registered intraday book. No fill is applied a
second time. Parts A/B already provide the performance transformation and BUY
declaration, but do not yet make this path reachable.

Implement C1 first (native transaction, adapter and portfolio), then C2 (source,
preparation, recipe/cutoff, full DAG and display). Each part needs its concrete
changes verified. A native C1 success does not certify C2 or the broader SELL,
funding and corporate profiles.

## C1: existing native batch, version 2

Add optional `registered_event_declaration_ref=None` to the existing
`close_through_latest` Python owner and Store adapter argument contract. Existing
v1 calls and public CLI flags remain unchanged. A non-null ref selects exactly
the Part B registered BUY profile; no arbitrary callable or implicit directory
selection. Without a declaration, an already non-official active state cannot be
reported as an already completed official close.
Missing and explicitly None adapter fields normalize to the unchanged v1 argument
set. Non-null values must be exact refs; unknown extra arguments still reject.

Fresh C1 finalization requires the declaration's writer SHA/id to equal current
Store, its native writer date T to equal the exact Market/Calendar target T, and
its baseline to be the immediately preceding OPEN date. Read Part B native proof,
both catalogs/performance, actual pointers, policy, Market, benchmark and Calendar
before planning. Require no T Event v1 empty closure. All normal native source,
policy and price/benchmark completeness checks remain. The declaration satisfies
only the registered T event-evidence slot; never manufacture an empty closure.

The first C1 profile handles exactly `[T]`. A registered intraday state older than
the requested target is an explicit unresolved historical-finalization case,
not a no-action success or silently skipped date. C2's catch-up integration must
address such ordered historical work with exact per-day evidence.

Create `myquant.cn_official_close_batch_plan.v2` at the same transaction family's
`plan.v2.json`. Keep every normal plan field, including writer preimages and
`source_active_record_id`, and add:

```
source_profile = OWNER_DECLARED_BUYS_V1
registered_event_declaration_ref
decision_baseline_pointer_ref
decision_baseline_catalog_ref
decision_baseline_record_id
source_valuation_date = T
```

The baseline refs are the declaration's original immutable refs, not a derived
replacement. `last_official_date` is baseline date; `missing_dates=[T]` means the
official T close is missing even though the performance tail contains intraday T.
Use a distinct implementation discriminator in the existing input fingerprint;
include all profile/declaration/baseline identities. Existing v1 fingerprints and
schemas remain unchanged. Reject unknown/extra v2 fields and mismatched bindings.

Under the same native operation lock, preserve original writer bytes at
`source-pointer.v1.json` and original baseline bytes at
`decision-source-pointer.v1.json` before any financial staging. Both must match
the declaration's immutable sources. Keep original `committed-pointer.v1.json`
format for final native pointer custody. No old pointer reconstruction.

Continue using the existing `build_record` and catalog CAS path. The new record
is valuation-only relative to the intraday source: shares, average cost, cost
basis and cash remain identical. Its registration time is truthful. Use the
declaration id/SHA/time as the bound registered continuity evidence; do not call
it an Event v1 empty closure. The existing strict-close record publication class
and metadata remain in force. Wire Part A `official_close_source` for the single
performance replacement, preserving prior rows, units and cumulative flows.
Native lineage retains baseline -> intraday -> official records, with no
correction/supersession relabeling.

Use `myquant.strategy_daily_close_receipt.v2`, carrying the declaration ref and
source_profile instead of `event_closure_sha256`. All existing false financial/
broker mutation authority flags remain false. `myquant.cn_official_close_batch_completion.v2`
and `completion.v2.json` additionally bind the registered declaration and baseline
pointer ref. No v2 payload is written under a v1 plan/completion filename.

## C1 readback, recovery and callers

Extend native inspect/frozen-inspect/recover with an explicit optional version
argument defaulting to v1. Derive only the selected fixed plan/completion paths;
never try v1 then v2 or scan for a plan. Retain old v1 behavior. New proofs must
validate all v2 fields, original declaration, both source pointers, final pointer
and native catalogs/lineage/performance/strict record. Recompute expected official
performance from Part A and compare to the committed series. Continue checking
writer-to-final holdings/cash equality. Frozen replay reads no current head.

Metadata recovery remains after exact native CAS proof and never repeats CAS.
Crash after CAS/before completion preserves original identity and clocks; unknown
or missing custody cannot downgrade into legacy recovery. Completed v2 adoption
uses the exact active receipt/transaction/plan and declaration; no new plan or
substitution of current SHA as the original writer preimage.

Store adapter selects v1/v2 from the bound plan ref/schema, uses the corresponding
native inspector, and returns existing output roles with the version-correct
completion ref. Do not add a second Store node/writer. Portfolio binding selects
the v2 baseline SHA/catalog/id and fixed `decision-source-pointer.v1.json`, while
v1 retains its existing source selection. Caller-supplied retained refs cannot
override that v2 source. Validate the original declaration and dual-source
binding during portfolio replay; no intraday book can become Decision T-1.

Update existing Store adoption/materialization/completed-Store readers only where
required for explicit v2 dispatch and source identity. The general daily recipe/
cutoff/native-input v2 integration is C2, not implied by these native components.

## C1 acceptance

- Real synthetic native baseline close, owner BUY registration and declaration,
  then Store adapter preparation -> execution -> readback. Exactly one additional
  valuation CAS, zero new fills/orders, unchanged writer cash/quantity/cost.
- Decision portfolio retains the original baseline while final Store contains
  the registered BUY and current strict-close valuation.
- Same-date performance row is replaced, previous rows unchanged, original
  intraday lineage retained, no duplicate fees/flow/units and no correction label.
- Read-only planning writes nothing; prepare/execute ambiguity rejects; repeats
  and already-committed adoption preserve bytes and timestamps.
- Crash before CAS, after CAS, before pointer custody and before completion
  recover only through the existing native path; later-head frozen replay uses
  no current pointer or business writer.
- Missing/mismatched declaration, old baseline, foreign pointer, empty-event
  conflict, bad source/date/price/SHA, unknown version and corrupt retained
  pointer/receipt/performance fail before any new business write.
- Existing no-trade batch, five-session/backlog, v1 adoption, portfolio and
  completed-Store tests remain passing. No real financial/provider/deployment run.

## C2 obligations retained

Source planning must select the exact registered declaration before attempting
empty publication, including adoption of an already finalized v2 close. New
source/preparation/recipe/native-input/cutoff versions carry the explicit profile
and declaration. Corporate evidence must distinguish owner-declared changes from
empty evidence; Decision remains baseline T-1. Preserve source registration and
capture times and prospective/OOS exclusion. Dashboard must distinguish the
day's registered changes from the close writer's zero new transactions. Verify
the whole DAG through the configured launcher, forward/late recovery, per-day
catch-up, EOD and frozen readback before calling this integration complete.

## Accepted Architect precision

The four build_record continuity values are fixed:

```
continuity_receipt_id = daily-close/<T>/<declaration_ref.sha256[:16]>
continuity_receipt_sha256 = declaration_ref.sha256
continuity_receipt_created_at = declaration.registered_at
continuity_checkpoint_digest = content_sha256(writer_pointer.active_closure)
```

Receipt v2 has the same receipt id and explicitly contains
registered_event_declaration_ref, source_profile,
writer_active_checkpoint_digest and decision_baseline_pointer_ref. The inspector
compares these with both final manifest/manual continuity fields and plan v2.
The declaration SHA never replaces the writer active-closure checkpoint digest.

Path/schema versions must match in both directions: plan.v1/schema1,
plan.v2/schema2, completion.v1/schema1 and completion.v2/schema2. Reject unknown
versions, cross-version paths and conflicting files from both versions in one
transaction. The adapter derives version from both the exact ref path and schema;
the output completion role uses only the validated version's fixed path.

Completed-v2 adoption occurs before ordinary source-preimage rejection. Read the
exact declaration to identify its original writer. If current is that writer,
prepare/execute normally. Otherwise require the current active record's unique
v2 receipt and fixed transaction plan, then prove the complete frozen v2 commit.
Caller expected Store SHA must be either the original writer SHA (same-request
repeat) or that proved current final SHA (fresh adoption); any third SHA rejects.
No other preimage may differ. Only then return PLAN_ADOPTED, keeping the plan's
writer preimage unchanged. Incomplete proof cannot fall into v1, execution or
recovery. Historical replay uses a recorded plan ref, not current adoption.

V2 proof explicitly verifies baseline -> writer -> final pointer AND record
chains, both transaction source-pointer copies against the original declaration,
the full native declaration proof, exact final position/cash equality, final
strict-close/publication/no-funding profile, and Part A-recomputed performance
against the committed series. A final CAS edge alone is insufficient.

The single-T restriction applies uniformly to plan, prepare, adapter probe,
execute and adoption. Use one explicit unsupported error when declaration date
is older than the exact target; do not close the old intraday record and append
later empty days in C1, or report the old intraday row as official NO_ACTION.

Coverage must also preserve evidence types. Extend the pure coverage owner with
an optional registered-event-date input supplied only after native declaration
verification. For that profile, `event_closed` remains false, a separate
`registered_event_proven` field is true, and completeness uses the exclusive
evidence choice. Both evidence kinds on T, duplicate registered evidence or absent
evidence block. Existing calls without this input retain their exact output shape.
Passing an event date to the pure coverage helper grants no authority by itself.
Do not fabricate a closure or edit a returned blocker list to pass coverage.
