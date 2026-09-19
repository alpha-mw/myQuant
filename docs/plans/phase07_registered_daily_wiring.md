# Phase 7 C2: registered events through the configured daily path

Status: Architect revisions accepted; Critic APPROVE. Implementation starts with
the native registered transition report and corporate v3 node, then source and
transport wiring. A locally verified component does not complete C2.

Local checkpoint: corporate recipe/projection v3 and the separate registered
transition report are implemented, as are source-plan v2, preparation
construction/commitment v2 and execute-recipe v6. The configured source path
generates the exact registered request and repairs interrupted request publication.
240 combined tests pass; `.agent/acceptance/phase7-c2-source-corporate-validation.json`
records the source hashes and explicit full-EOD/install/transport fixture seams.
The next local checkpoint implements acquired bundle v2, cutoff v2, research
wrapper v3, materialization v6 and native input v7. Two native scenarios now reach
Corporate v3, C1 Store close and Morning risk-source validation;176 regressions,
3 native legacy-cutoff cases and34 final custody/grammar/preflight checks pass in
separate scopes. `.agent/acceptance/phase7-c2-native7-validation.json` records the
exact boundaries. Dashboard/serving integration now has separate local validation:
84 Python checks,28 browser contract/tampering checks and an HTTP desktop layout
review. `.agent/acceptance/phase7-c2-dashboard-validation.json` records source hashes
and controlled five-domain/full-EOD admission seams. Finalized original-request
recovery and full configured/installed EOD acceptance remain open.

The existing native C1 transaction works. This step must make it reachable through
configured source provisioning, request preparation, maintenance handoff, research
cutoff, corporate reconciliation, Decision, Store, EOD and Dashboard readback.
Do not call a native adapter-only fixture proof of this complete path.

## Source selection and request construction

Add a shared native read-only selector for one exact observed Store pointer and
target date. It returns ordinary empty-source mode, a current registered intraday
declaration, or an already finalized v2 transaction/declaration proved through
the active receipt and frozen native verifier. Never scan for a declaration.
An intraday source without its exact declaration blocks before empty publication.
An older intraday source remains explicitly unsupported until the ordered
historical registered-source path is implemented; no date skipping.

For a registered source, use source-plan v2 carrying non-null
registered_event_declaration_ref and event operation USE_REGISTERED_DECLARATION.
Retain the original Event pointer for history, but never publish a T empty
closure. Readbacks compare the exact declaration, native source identity and
original Store/Event preimages. Ordinary source-plan v1 is unchanged.

Preparation must use the same selector and evidence. Extend its immutable
construction/commitment version for a registered ref and emit execute recipe v6.
The static config format and launcher argument selection remain unchanged.
All Calendar/source locator/preparation crash recovery continues from exact
retained refs and original times. Neither a changed config nor a different
declaration can reuse the same commitment.

## Versioned daily transport

Recipe v6 adds registered_event_declaration_ref to recipe v5, requiring a non-null
exact ref. Carry it through materialization v6 and native-input v7. These versions
require a matching native Store plan v2; old versions retain plan v1 semantics.
Store argument construction passes the exact ref into the C1 owner. Bootstrap
and previous-EOD checks compare the correct writer preimage and separate Decision
baseline; the declaration baseline must equal the verified previous EOD Store
output, not merely an arbitrary historical official record.

Acquired-source bundle v2 and research-cutoff v2 retain the declaration and all
native proof source refs. Add explicit owner-declared, registered, and source
publication timestamps to the cutoff's typed time registry. Rebuild and compare
them on replay. Late declarations/custody remain retrospective and never receive
true OOS admission. The existing research request wrapper can continue pointing
to the exact cutoff ref if its validation explicitly supports both cutoff
versions; otherwise version that wrapper without changing legacy meanings.

## Corporate and financial event semantics

Corporate recipe/projection v3 must accept the registered declaration as an
exclusive alternative to a T empty closure. It must expose OWNER_DECLARED evidence
and preserve the broker-unverified fee status. The existing Event history remains
available for earlier-day continuity. Current T empty plus registered facts blocks.

The existing corporate report remains exactly T-1: its portfolio_source_ref and
company_rows retain Decision-baseline semantics. Add the separate responsibility
kind `registered_financial_transition_reconciliation`, covering the sorted union
of baseline and writer positions. Do not move writer positions into the old kind.
Changed/new positions require policy revalidation and remain non-executable; the
new report does not independently evaluate stop/trailing policies absent from its
inputs. Morning later uses its exact policy refs against final Store positions.

Store still performs only C1 valuation and exactly one CAS. Corporate unresolved
thresholds remain distinct from an actual financial conflict; do not block ordinary
valuation merely because a new owner anchor is unavailable. Use existing risk
vetoes and no automatic anchor/stop reset.

## Completion and display

All native-input, materialization, cutoff, corporate, completed-Store, Morning,
Dashboard and prospective-ledger readers must explicitly dispatch the new
versions and validate the same declaration and dual source bindings. No old
recorded ref or timestamp is replaced with current state.

Dashboard must show the day's registered owner changes separately from the close
writer's zero new transactions. Preserve the accounting renderer and source-backed
tables, display owner-declared/broker-unverified evidence accurately, and retain
unconfirmed risk states. New output metadata must be rebuilt from frozen evidence
on readback; it cannot be an unverified display override.

## Acceptance

- Configured source inspection with a registered BUY selects the exact declaration
  and reaches request construction without creating an empty Event or calling a
  financial writer. Missing/changed/conflicting declarations fail before producers.
- The same explicit input reaches native-input v7 and the existing DAG, with
  previous-EOD Decision book, registered writer book, official Store close and
  accurate event/risk display. Test both existing-position add and new symbol.
- Real native financial/Calendar/Market/Event/benchmark modules are exercised;
  controlled Factor/provider/full-EOD seams are disclosed until a fresh installed
  whole-DAG proof exists. No actual account or live provider operation in tests.
- Completed-v2 adoption, cross-day retained request recovery and frozen replay
  survive later heads without another CAS, source recapture or clock reset.
- Every added version rejects cross-version fields/paths, wrong declaration,
  baseline/previous-EOD mismatch, missing sources and tampered rendered metadata.
- Existing ordinary no-trade, cutoff/preparation, bootstrap/automatic, CLI/shell,
  Morning, Dashboard and ledger regressions pass.

Full SELL/funding/corporate profiles, ordered historical registered-source work,
final current-source installation, real scheduler cutover and consecutive
unattended trading-day evidence remain required by the original goal.

## Fixed registered-profile version matrix

| Contract | New profile |
|---|---|
| Source plan | v2 |
| Preparation construction/commitment | v2 |
| Execute recipe | v6 |
| Acquired source bundle | v2 |
| Research cutoff | v2 |
| Research request wrapper | v3 |
| Materialization | v6 |
| Native input | v7 |
| Corporate recipe/projection | v3 |
| Baseline corporate report | Existing kind unchanged |
| Registered transition report | New kind v1 |

All older versions retain exact meanings and bytes. Source locator/result/static
config fields can stay unchanged as reference envelopes, but their validators
must explicitly validate the referenced plan/commitment version and fixed path.
No cross-version path/schema combinations or fallback probing. Same-day conflicting
v1/v2 plans/commitments reject; an existing request always recovers its original
version before current-source selection. New source artifacts retain Calendar
capture/budget/lock behavior unchanged.

Source-plan v2 adds exact fields registered_event_declaration_ref,
registered_source_state, registered_store_plan_ref, decision_baseline_pointer_ref
and writer_pointer_ref. Event fields are USE_REGISTERED_DECLARATION,
event_generation_id=null, event_source_mode=REGISTERED_FINANCIAL_TRANSITION.
REGISTERED_INTRADAY has registered_store_plan_ref=null and observed Store=writer.
FINALIZED_V2_RECOVERY binds the original exact plan.v2 and proved final pointer.

Cutoff v2 adds registered_event_declaration_ref, registered_writer_pointer_ref,
registered_source_state and registered_store_plan_ref. Its time registry adds
REGISTERED_OWNER_FACT/SOURCE_DECLARED, REGISTERED_DECLARATION/LOCAL_REGISTRATION,
REGISTERED_STORE_PUBLICATION/LOCAL_PUBLICATION. Each appears in source_refs and is
rebuilt from native proof. Ordinary wrapper v2 keeps cutoff v1; wrapper v3 requires
cutoff v2. No timestamp normalization or retrospective promotion.

## Registered transition report contract

Register the new artifact kind with identity `registered_transition_id`, existing
research common fields and these exact payload fields:

```
as_of trade_date strategy_id source_profile registered_event_declaration_ref
decision_baseline_pointer_ref decision_baseline_catalog_ref decision_baseline_record_id
writer_pointer_ref writer_catalog_ref writer_record_id owner_fact_ref
owner_declared_at registered_at evidence_level broker_statement_verified fee_evidence_level
domain_states position_rows cash_before_cny cash_after_cny cash_delta_cny
financial_admission_state risk_readiness_state blocker_codes source_refs
custody_at timing_status prospective
```

Each position row has exactly:

```
symbol baseline_position_state writer_position_state shares_before shares_after
shares_delta avg_cost_before avg_cost_after cost_basis_before cost_basis_after
cost_basis_delta change_kind fact_refs policy_revalidation_required
risk_execution_state blocker_codes
```

Amounts use finite decimal strings; absent positions use explicit ABSENT state,
zero shares/cost basis and null average cost. Present states use PRESENT and their
native values. Rows equal the sorted symbol union; change_kind is UNCHANGED,
EXISTING_POSITION_ADD or NEW_POSITION. Unchanged rows claim no policy evaluation
(risk_execution_state=NOT_EVALUATED); changed rows are NON_EXECUTABLE with
policy_revalidation_required=true and OWNER_POLICY_REVALIDATION_REQUIRED.
The overall risk_readiness_state is OWNER_POLICY_REVALIDATION_REQUIRED for this
BUY profile. No exact stop, anchor or automatic action is computed here.

financial_admission_state=VALIDATED_REGISTERED_TRANSITION means native Part B
proof and BUY accounting bridge pass. Evidence remains OWNER_DECLARED,
broker_statement_verified=false, fee status verbatim and prospective=false.
All authority flags stay the existing research-only false scope. Source refs bind
both books, owner fact/declaration and all native readback inputs. Identity and
bytes are rebuilt from those refs at recorded custody time on replay.

Corporate recipe v3 adds the declaration ref to v2; report kind is fixed in code.
Projection v3 has the existing v2 information plus registered_transition_ref,
registered_transition_state, registered_evidence_level, broker_statement_verified,
financial_admission_state and risk_readiness_state. Its financial_event_state
explicitly describes registered owner facts; event_closure is null. The node
outputs registered_transition alongside financial_events/event_generation/
reconciliation. Existing baseline report is preserved. Financial conflicts block
Store; policy revalidation alone does not block valuation.

Morning compares the new report's writer quantities/cost and position union with
the final Store, then applies exact current stop/trailing policies. Missing
baseline corporate rows for new positions cannot hide them. Dashboard rebuilds
separate owner BUY, zero close-writer new transactions, official valuation,
pending broker verification and required policy-revalidation displays from frozen
reports. No renderer-supplied financial/risk override.

## Finalized source: recovery only

A standalone C1 finalization is not enough to construct a new daily request.
Plan v2 stores a Calendar SHA but not its original path/raw Calendar ref. A fresh
same-T capture has a new observation time and bytes, and is not substitutable.

The selector distinguishes REGISTERED_INTRADAY, FINALIZED_V2_RECOVERY and
BLOCKED_FINALIZED_V2_INPUT_CUSTODY_MISSING. FINALIZED recovery requires an original
execution request/recipe, handoff2/4, cutoff2, materialization6/native7 and exact
native Store plan2. Configured-source recovery additionally requires the original
REQUEST_AVAILABLE locator and preparation commitment matching config/date/
declaration. Source Calendar belongs to preparation; the distinct Core Calendar
in the handoff supplies the native plan Calendar SHA. Replay both original raw
pairs to the same target without comparing their hashes. Require prepared_at <=
transaction_planned_at <= cutoff custody <= cas_observed_at, original non-Store
preimages and previous-EOD baseline equality. Direct EXECUTE with the complete
original execution package does not require a configured-source locator. Exact
automatic provenance and metadata-only recovery rules are specified in
`phase07_registered_recovery.md`.
If an EOD already exists, use completed replay directly. No new Calendar/source
selection/cutoff is sampled during recovery.

For configured origin, the original locator -> commitment -> request -> recipe
chain establishes preparation custody. PREPARING_SOURCES alone is insufficient. Missing custody
blocks fresh admission even if the native Store is valid. Never search by SHA,
infer a Calendar path, normalize hashes, backfill a locator, or rewrite plan v2.
Fresh enrollment of standalone completed financial state would require a future
version with pre-CAS full input custody; it is not silently added here.

## Dashboard registered profile: Architect amendments

Keep the existing five-domain `daily_dashboard_evidence` artifact and accounting
v1/v2 renderer unchanged. Native input v7 selects Dashboard evidence recipe v2:
the v1 fields plus exact `corporate_terminal_ref`, `store_plan_ref` and
`registered_event_declaration_ref`. The five authority terminal refs stay the
same; the Dashboard request adds `registered.corporate_terminal` as an ordinary
source ref. Native5/6 require recipe v1 and forbid registered fields.

The v7 adapter rebuilds the exact corporate v3 projection, baseline report and
registered transition from its SUCCEEDED same-day/release terminal. It replays
the C1 frozen plan-v2 commit and compares all Store terminal outputs, declaration,
baseline pointer/catalog/id and writer identity. Final official record must be
the writer's direct child, preserve exact holdings/cash, and record zero new
close-writer trades/orders/fills. Every source terminal finishes before Dashboard
recipe custody. The Dashboard terminal forwards the same already-verified
`registered_transition` ref as its fifth output, without copying the artifact.

Completed Dashboard replay requires exactly five outputs for native7 (capture,
v1, v2, daily_evidence, registered_transition), and exactly the original four for
native5/6. Replay invokes completed Corporate/Store readers and the same cross
binding, using frozen sources only. Unknown/missing/extra/version-mixed outputs
block. No broader authority set or new Dashboard artifact kind.

Serving descriptor `cn-daily-dashboard-serving.v2` retains all v1 fields plus
`registered_transition_ref`, the complete `registered_transition` artifact and
`registered_close_summary`. The summary has exactly:

```
source_profile store_plan_ref decision_baseline_pointer_ref writer_pointer_ref
final_pointer_ref writer_record_id final_record_id official_valuation
close_writer_trade_count close_writer_order_count close_writer_fill_count
```

Summary fields are rebuilt from frozen C1/Store proof; profile is
OWNER_DECLARED_BUYS_V1, official valuation is true and all three close-writer
counts are zero. The existing JSON/JS filenames and selector SHA cover the full
descriptor bytes, including embedded report for HTTP and file views. Serving
intent/receipt keep their field sets but validate exact four/five output sets by
native profile. UI displays owner changes and close-writer valuation separately,
with broker pending and policy-revalidation status derived from the report.

Architect APPROVE_WITH_CHANGES accepted this exact boundary, all amendments were
incorporated, and Critic APPROVE is recorded. Dashboard-specific implementation
now has local validation, including native financial rendering, exact five-output
publication/readback and frozen completed Dashboard replay. Five-domain/full EOD
admission is explicitly controlled in these tests; full configured/installed
acceptance remains separate.
