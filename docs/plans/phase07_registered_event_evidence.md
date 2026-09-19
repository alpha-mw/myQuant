# Phase 7 Part B: registered financial event evidence

Status: Architect amendments accepted; Critic APPROVE;
INITIAL_BUY_PROFILE_IMPLEMENTED_LOCAL_VALIDATION.

The existing manager now exposes the declaration command and guards empty Event
publication against registered same-day changes, including before NO_ACTION.
The sibling retains original writer-pointer bytes and has frozen native readback.
Only the initial owner-declared BUY profile is admitted. Part C wiring and the
broader SELL/funding/corporate profiles are still required.

Validation: 221 combined tests passed in34.20s across registered declarations,
Event/source producers, configured source/launcher/CLI, closed native imports,
performance finalization and native Store/adoption. The native BUY fixture uses
the real Store manager for financial registration. Tests prove zero second
financial CAS, original timestamp/byte/mtime preservation, frozen replay without
heads, exact fact/fee/cash/position checks, both empty/declaration conflict orders,
companion-only crash handling and real Record operation-lock serialization. The
later same-day no-trade-head guard test explicitly controls catalog admission;
its actual financial publication remains Part C's responsibility.

Seven-file Black/scoped flake8, helper/package/bridge mypy and existing record
access governance checks pass. Only the two owned manager functions were formatted;
unrelated manager changes were preserved. Receipt:
`.agent/acceptance/phase7-registered-event-part-b-validation.json`.

Part A's calculation is locally implemented. This part must supply the missing
source facts; Part C will wire them into native batch v2 and the daily DAG.

## Proposed evidence location and ownership

Preserve Event v1 and its current pointer as empty-closure history. A registered
financial transition uses a mutually exclusive sibling declaration; it is never
placed inside an Event v1 `CLOSED_EMPTY` row. Avoid a second mutable pointer or a
latest-file scan.

Proposed exact immutable location under the existing Event owner:
`<record_root>/_event_store/registered/<writer_pointer_sha256>/declaration.v1.json`.
The same directory retains the original `writer-pointer.v1.json` bytes before
the declaration is published. Declaration binds that exact companion ref.
The filename is selected from the exact observed registered Store pointer SHA,
not directory order, time or an inferred historical record. Absence is a blocker.
The source plan retains the declaration's complete path/SHA and later readers
use that ref exclusively.

The existing record manager would own declaration publication, using an explicit
owner-supplied fact document and the expected current Store SHA. It copies no
trades into financial state and performs no Store CAS. Publication needs exact
native source readback, owner-safe immutable file publication and current-pointer
recheck under the existing Record operation lock. Conflicting bytes for the same
source SHA reject; no implicit owner-fact creation or retiming. The declaration
records actual local registration time separately from the owner's declared time.

## Required declaration content

The closed schema binds strategy, T date, owner, original owner declaration ref,
owner-declared time, actual registration time, T-1 baseline pointer SHA/record id,
registered writer pointer SHA/record id and seven named domains: executions,
orders, fills, funding, cost-basis changes, corporate actions and manual changes.
Every domain is explicit, either OWNER_DECLARED_NONE or OWNER_DECLARED_FACTS with
a closed set of exact fact refs. These states are not Event v1 CLOSED_EMPTY.
Missing fields cannot imply emptiness. All broker/order/trade/holdings execution
authorities are false. This document supplies owner facts; it does not grant a
daily writer permission to create a transaction.

The baseline pointer's original bytes must already exist in verified prior-EOD
custody; reconstructing an old pointer from its catalog is forbidden. The writer
pointer must be its direct successor. The declaration alone does not certify a
previous EOD; Part C must independently prove and compare that EOD's Store output.

## Financial evidence validator

The native reader must validate both exact catalogs/active closures, direct
lineage, performance, record inventory, manifest/manual/Parquet ledger and the
registered source's date and non-official state. It retains all read refs and
rechecks bytes. No ambient current pointer is used during later replay.

Modern owner trades require `cn_aggressive_manual_execution.v3` with exact
`owner_declared_manual_execution_applied` status, complete typed trade rows,
finite positive integral shares/prices, trade value and explicit fee closure,
and attributed pre/post quantity/cost/cash differences. The actual modern record
uses `OWNER_FEE_POLICY_APPLIED_BROKER_STATEMENT_READBACK_PENDING`; preserve that
owner-declared evidence level and do not relabel it broker-verified. Do not guess
fees from a missing field or introduce a new fee schedule.

Funding requires the existing exact supplement and
`myquant.strategy_performance_cash_flow.v1` proof plus original unitization.
Corporate posting requires the existing isolated application and Phase8 native
`OBSERVED_NATIVE_POSTING` proof. Unsupported legacy, generic manual, mixed or
unattributed changes block rather than being called no-change. Those boundaries
must be explicit in the subsequent closed schema and tests.

## Integration obligations

- Source selection must identify a registered transition before it can create a
  current-day empty closure. Same-day empty and nonempty evidence together block.
- New source/preparation/recipe/native-input versions carry this exact declaration
  and intrinsic transition proof, while preserving the historical Event pointer.
- Decision reads the T-1 pointer; the existing close writer starts from the
  registered intraday pointer. Part A finalization retains units/cumulative flow.
- Every source-plan/cutoff/Store/receipt/completion replay must dispatch explicitly
  by version/profile. Never put a transition SHA in `event_closure_sha256`.
- The source-plan and batch transaction retain original timestamps and both
  pointer byte sequences before financial publication. Late owner reports remain
  retrospective and cannot acquire true OOS admission.

## Questions for architecture review

Confirm that the exact immutable sibling location plus source-plan custody is
adequate without a new Event generation/pointer schema; otherwise identify the
smallest change required in the existing Event owner. Identify the required
locking/readback conditions and any existing source-path restrictions this would
violate. This proposal is not permission to implement an incomplete source gate
or to declare unreported domains empty.

## Accepted protocol and ownership

Use the existing manager command family. Add
`publish-registered-event-declaration` with required `--project-root`,
`--record-root`, `--owner-fact`, `--owner-fact-sha256` and
`--expected-pointer-sha`. It owns the existing Record operation lock for both
CLI and programmatic calls; the main dispatch skips a second outer lock, as it
already does for daily Event publication. No caller-supplied registration clock.

Pure schemas/arithmetic live in `quant_investor.strategy_records`; a fixed native
script helper calls the existing catalog, performance, inventory, validate_record
and bounded file readers. Add that one helper to the closed native bridge module
map; no dynamic module or callable selection from JSON. The normal Event writer
uses the same native read-only nonempty guard.

The owner fact document has exactly:

```
schema_version = cn-registered-owner-facts.v1
owner_fact_id
strategy_id = aggressive_tech_manufacturing
trade_date
owner
owner_declared_at
baseline_store_pointer_ref
writer_pointer_sha256
writer_record_id
domains
evidence_level = OWNER_DECLARED
broker_statement_verified = false
authority
content_sha256
```

The published declaration has exactly:

```
schema_version = cn-registered-financial-declaration.v1
declaration_id
strategy_id
trade_date
owner
owner_fact_ref
owner_declared_at
registered_at
evidence_level = OWNER_DECLARED
broker_statement_verified = false
baseline_store_pointer_ref
baseline_catalog_ref
baseline_record_id
writer_store_pointer_ref
writer_catalog_ref
writer_record_id
writer_record_refs
domains
late_event_behavior = OFFICIAL_CLOSE_RESTATEMENT_REQUIRED
authority
content_sha256
```

Identifiers are bounded nonempty native identifiers; trade_date is canonical
YYYY-MM-DD. All refs are exact workspace-relative path/SHA pairs. Domains have
exactly the seven existing Event names, each with exact `state`/`fact_refs` keys.
NONE requires an empty list; FACTS requires sorted unique nonempty refs. The
current trade profile binds those refs to the already registered native record's
manifest/manual/ledger/P&L, rather than accepting arbitrary workspace files.
Executions, fills and cost-basis changes must declare FACTS including the manual
ref; orders explicitly declares NONE or matching owner facts, without inferring
broker order identifiers. Funding, corporate actions and other manual adjustments
must explicitly declare NONE for this first trade-only profile.

`writer_record_refs` has exactly manifest, manual_manifest, ledger, pnl,
performance_manifest, performance_series and performance_owner_declaration.
Authority has exactly store_mutation, actual_holdings_mutation, cash_mutation,
broker, order, execution, trade and daily_writer, all boolean false. The owner
must match the native manual owner declaration's approved_by; evidence remains
owner-declared. Preserve the original fee-evidence status verbatim.

The baseline ref must already point to an immutable native batch committed
pointer, never current.v1.json. Prove catalog v3/performance and an official
baseline state of an earlier date. Writer previous-pointer SHA, previous record
id, native source_record and direct lineage must all match that baseline.
Writer is the exact current non-official T state at publication. Part C still
must bind this baseline to a verified previous EOD and the native Calendar.

## Initial financial shape and full-goal limits

The first supported shape is the actual modern owner-declared BUY row listed in
the Architect review. Its exact fields are trade_id, symbol, name, side, shares,
execution_price, trade_value, commission_cny, stamp_duty_cny, transfer_fee_cny,
final_total_fee_cny, cost_basis_cny, avg_cost_cny_per_share, commission_rate,
commission_minimum_cny, commission_includes_regulatory_and_handling,
transfer_fee_rate, fill_cost_status, reported_by, source_channel and trade_date.
Require unique trade ids, positive integral shares, finite positive prices,
nonnegative stated fees/rates, same T date and owner, and exact current
`OWNER_FEE_POLICY_APPLIED_BROKER_STATEMENT_READBACK_PENDING` evidence status.
Use the native one-cent money tolerance for stated price/value/fee/cost
reconciliation; do not infer broker verification or invent a fee schedule.

Validate value=shares*price, total fee=sum of explicit fee components,
BUY cost=value+fee, and average cost*shares=cost. Aggregate the exact supplied
BUY rows against baseline quantities/cost basis and writer quantities/cost basis;
cash must decrease by the supplied total costs. Untouched symbols remain
identical. Local/rejected/pending trades, funding/corrections or corporate/manual
payloads are unsupported in this profile. SELL, exact funding and isolated
corporate profiles remain required subsequent work for the full Phase7 goal;
passing this initial profile cannot close that goal.

## Atomic custody, conflicts and replay

1. Under Record operation lock, validate owner fact bytes/SHA, baseline immutable
   pointer, Store current, both native catalogs/records/inventories/performance,
   declared facts, and absence of target-day Event v1 closure.
2. Verify all relevant source bytes again. Sample the actual manager UTC time
   after validation; it cannot precede owner-declared or native publication times.
3. Preserve original writer-pointer bytes using the existing atomic immutable
   Store writer, then publish the sealed declaration last. New directories/files
   use0700/0600; existing safe parent modes are preserved. Reject links, aliases,
   unsafe ownership/modes and mismatched bytes. Fsync files and containing dirs.
4. Re-read exact bytes and source Store/Event identities before returning.
   Repeated publication of an identical fact ref with unchanged current source
   returns the original declaration/timestamp with zero rewrites. A conflicting
   declaration is never overwritten. A retained pointer without a declaration
   can be used only while its exact source is still current; no late old-pointer
   reconstruction or clock backdating.
5. Historical reader accepts only explicit declaration ref, validates its fixed
   companion writer pointer and baseline refs, and replays native financial proof.
   It never reads Store/Event current, scans sibling folders, or invokes a writer.

The daily empty Event writer must re-read Store inside its existing operation
lock, including before its NO_ACTION branch. A registered T-day applied/nonempty
transition in the selected lineage rejects empty publication even when this
declaration is absent. Unknown/unreadable T records also block. A later no-trade
record cannot hide an applied record earlier on the same T lineage. Recheck
before Event publication and after readback. Conversely the declaration writer
rejects an existing T empty closure and requires explicit restatement handling.
Do not alter or relabel the old closure. Lock order remains Record operation ->
Event current only for the actual Event pointer writer.

## Verification for Part B

- Build a synthetic modern BUY through existing native record registration,
  then publish/read the declaration with real pointer/catalog/performance/
  inventory/ledger readers. Assert no extra Store CAS or financial byte changes.
- Valid repeated publication preserves all bytes/mtimes and original clocks.
- Frozen replay works after heads advance/remove, with current readers and all
  writers forbidden; missing companion pointer or any corrupted source rejects.
- Missing/ambiguous domains, mismatched owner/date/source, wrong fees/cash/cost,
  unsupported SELL/funding/corporate/mixed rows, unsafe refs and clocks reject.
- Both publication orders between empty Event and registered declaration fail
  safely on conflict. A registered trade before declaration must never permit an
  empty closure; test the real lock boundary and current recheck.
- All existing native Event producer/configured source/Store tests remain passing.
- Part B does not claim batch v2, source-plan dispatch, daily financial close,
  Dashboard event display, full DAG, deployment or unattended completion.
