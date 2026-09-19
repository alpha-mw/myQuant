# Phase 8: source-bound corporate-action reconciliation

Status: Architect amendments accepted; Critic APPROVE; COMPLETED_LOCAL_VALIDATION.

Current-source acceptance: 174 focused checks passed (11.41 seconds), 16 source
files passed mypy, 25 files passed Black, six new implementation modules passed
flake8 including complexity <=10, and the changed integration/test files passed
scoped lint. The exact source hashes and scopes are retained in
`.agent/acceptance/phase8-validation.json`.

Native isolated dividend and split records both produce RECONCILED_RESEARCH_ONLY
from registered pre/post accounting and an explicit synthetic owner declaration;
all thresholds remain NON_EXECUTABLE and repeat preparation/probing works with
current Store/event heads removed and no file changes. New v3 materialization,
v4 native loading, corporate execution and completed replay pass with controlled
outer handoff/EOD-admission seams; this is not a current-source installed full DAG
receipt. All five named source kinds, source/time/ref faults, record/policy ancestry,
mixed/zero attribution and current-day Store blocking are covered. Native source
declarations are not independently authenticated issuer statements.

The actual retained Zijin window now identifies its Aug21 transition despite an
equal final daily pair. That proof is explicitly retrospective and does not supply
the missing issuer/account/owner evidence or remove the canonical risk veto.
Whole-path installed acceptance remains Phase15; actual scheduled v3/v4 adoption
and source provisioning remain part of rollout rather than this local proof.

## Intended result and authority

Complete the existing CORPORATE_ACTION_RECON node's missing evidence path. Each
held symbol gets a full owner-tracking-window adjustment check, named corporate
acts when supported by exact source evidence, and independent cost-basis, share,
cash and threshold-anchor reconciliation. The report has no financial writer,
trade action or executable threshold authority. No ratios are used to guess an
event kind, dividend entitlement, rights subscription, tax treatment or account
posting. Existing unresolved risk vetoes remain. This is code capability and
isolated verification, not invented reconciliation for canonical Zijin.

The baseline audit proves why the last daily pair is insufficient: Zijin's Aug21
factor change remains in its Aug19 tracking interval, while Sep02/03 are equal.
See phase08_reconciliation_audit.md and its exact native retrospective receipt.

## Input route and compatibility

Add exact execution recipe v3 = v2 + required corporate_action_context_ref. It
retains the existing theme-source mode checks. A context is explicitly supplied
and SHA-bound; never select a latest file, enumerate source directories, or
silently synthesize an empty context. Legacy recipe v1/v2 remain readable and keep
their current behavior. New v3 materialization emits native inputs v4 = v3 +
corporate_action_context_ref. All shape dispatch, loaded-context checks, handoff,
catchup-template handling, completed replay and late-portfolio readers must carry
or validate that ref. The existing fixed registry selects the corporate v2
projection from native v4; it is not a caller-defined adapter/mode/command.

The context has exact fields: schema_version=cn-corporate-action-context.v1,
strategy_id, as_of, tracking_policy_ref, named_events_ref and anchor_reviews_ref.
The last two refs may be null, which means missing evidence, never no events.
strategy/as_of must equal the frozen Decision recipe. That recipe replays the
native Phase6 pre-close portfolio, original plan and original catalog; do not
select today's Store head. The tracking policy ref is explicit and must be the
existing owner-trailing-anchor-policy.v1 contract, effective by as_of, with false
Store/actual-holdings/broker/order/execution/trade authority. Its Store baseline
record+ledger must be on the portfolio's registered frozen ancestry. Preserve its
80%/65% research calculations, anchor semantics and corporate-action invalidation.
This work does not create or modify owner policies.

Capture the context's actual custody time in the new corporate recipe. Named
announcement times and any owner-review effective time must be <= as_of for
on-time use; future knowledge fails. Late local custody remains marked retrospective
and cannot qualify as prospective evidence. Recheck all source bytes at use and
replay. Replayed v1 corporate recipes retain their exact old output bytes.

## Named evidence and intrinsic validation

An optional exact cn-corporate-action-events.v1 document has strategy_id, as_of and
an ordered unique list of event_refs. Each referenced canonical
cn-corporate-action-event.v1 object has exact fields schema_version, event_id,
symbol, kind, effective_trade_date, announced_at, announcement_ref and
accounting_records. kind is one of SPLIT, DIVIDEND, RIGHTS, BONUS_ISSUE or
SHARE_CONVERSION. announced_at is an aware timestamp; dates/symbols/IDs are strict;
announcement_ref is a physical immutable evidence ref, read and SHA checked. These
are normalized source declarations; the raw announcement is retained as evidence,
not independently authenticated or automatically parsed by this local workflow.
An event is reported as SOURCE_DECLARED, not as an executed account action.
accounting_records is null or an exact before_record_id/after_record_id pair.
Duplicate identity, contradictory same-ID content, malformed refs, unknown kinds
or future announcements reject. Multiple legitimate events on a symbol/date are
allowed with different IDs; never collapse dividend plus bonus into one inferred
kind. Absence from a supplied list does not establish complete issuer coverage.

The exact optional cn-corporate-anchor-reviews.v1 list contains event_ref,
policy_ref, source_record_id, reviewed_at, disposition and tracking_start_date.
Only disposition=OWNER_DECLARED_RESET is supported, and it remains a source
owner declaration, not authorization to update a policy or create an executable
threshold. Dates must be within event-to-as_of and its event/policy/source record
must match the report's bound evidence. Unknown/missing/ambiguous review blocks
anchor reconciliation. The input is provided explicitly; no acknowledgement is
fabricated from a ratio or an analyst's interpretation.

## Full tracking window

Use the existing strict frozen Market reader and exact per-held-symbol refs.
Use the bound Calendar's complete ordered open dates from each policy tracking
start through T, inclusive. Require one finite positive close and adj_factor for
each required date; duplicate/gap/nonfinite rows produce explicit per-symbol
missing/conflict states and never a no-change conclusion. Missing policy anchor,
policy baseline not ancestral, or a changed position lifecycle remains explicit.
Enumerate every factor transition and keep its before/after dates and values.
Match declared events by symbol/effective date; unmatched transitions stay
ADJUSTMENT_FACTOR_CHANGE with NAMED_EVENT_EVIDENCE_MISSING. Named events remain
visible even if the factor did not change. The last daily comparison is retained
only as evidence, not as a substitute for the full interval.

## Native financial evidence, no posting calculations

When an event declares accounting record IDs, resolve both only inside the frozen
portfolio catalog's active ancestry. After must directly descend from before, and
the action effective date must fall in that native transition's date interval.
Use native catalog/external-record validation and load_holdings_identity for both
records. No current pointer or unregistered ledger is accepted.

The registered after-manual's corporate_actions array must contain exactly one
matching application with exact fields schema_version=
cn-corporate-action-application.v1, event_ref and symbol. This application only
links a source event to already registered financial records; it is not a new
writer or a new authority. Its event ref/symbol must match. Unsupported existing
manual shapes are UNCONFIRMED, never guessed.

Report before/after/delta for native total cost_basis, shares and cash, with the
native ledger/manual refs. Values are observed changes, not inferred dividend
or subscription calculations. If multiple financial acts are present in the same
record transition, report unattributed transition deltas and keep event-specific
adjustments UNCONFIRMED; do not assign the aggregate to every event. Only a single
matching corporate application with no trades/fills/funding/manual changes and
no other symbol cost/share change may support event-specific deltas. Native
no-action records with all zero deltas do not prove an asserted financial posting.
Different event kinds do not impose guessed cash/tax/cost formulas; evidence is
reconciled to what the native account actually records.

Anchor adjustment is separately reported from the explicit owner review with its
original policy and exact record binding. A complete evidence package may yield
RECONCILED_RESEARCH_ONLY; executable=false and all business-authority flags remain
false. Without the independent owner review, anchor remains OWNER_REVIEW_REQUIRED.
No new threshold price is computed or substituted into the native risk calculator.

## DAG output and financial conflict

Corporate recipe v2 binds context, Decision recipe, Store plan, cutoff and actual
custody alongside existing retained event/Calendar/Market refs. Produce a registered
versioned reconciliation report as an additional named output, preserving the
legacy financial_events projection for its old recipe profile. New profile's
financial projection explicitly points to the reconciliation report and includes
its summary state. Completed corporate replay reconstructs exact named outputs
from original refs and verifies native-input/recipe binding and timing.

Per-symbol unreconciled historical actions keep thresholds NON_EXECUTABLE but do
not invent an unresolved current-day financial posting. A declared action with
effective date T conflicts with the standing CLOSED_EMPTY event declaration:
report the contradiction, set this node BLOCKED with existing INPUT_INCOMPATIBLE
(or exact existing equivalent taxonomy), and prevent no-action Store execution.
Do not replace the existing v1 empty-event schema with a permissive financial
writer as part of this phase. The report cannot silently label a real nonempty
financial application NO_POSITION_CHANGE. Broader Phase7 posting/manual adoption
remains governed by its own native authority.

## Verification and completion evidence

- Real native portfolio + policy + frozen Market/Calendar: historical transition
  within anchor window is detected even when the last pair is equal.
- Named source cases for all five kinds, missing source/owner review, future
  announcement, duplicate/conflict, unknown fields, bad SHA/path/symlink and
  multiple simultaneous events. Explicit source declaration status is retained.
- Native pre/post account evidence positive case, missing/unregistered/wrong
  ancestry/application, mixed financial events and zero-delta false posting.
  No public provider, financial mutation, policy alteration or current-head read
  during frozen replay.
- New execute recipe -> real materialization -> native v4 -> corporate node ->
  retained output -> completed replay, plus legacy exact replay and prospective
  exclusion. New actual custody cannot be backdated.
- Same-day source/CLOSED_EMPTY conflict blocks Store before its writer; historical
  unresolved threshold preserves the veto without falsely rewriting daily events.
- Native canonical Zijin replay remains explicitly retrospective/UNCONFIRMED until
  actual source/account/owner evidence exists. Save source hashes, receipt scopes,
  tests and uncovered production facts. Phase8 implementation acceptance requires
  this integrated path, not only pure-helper tests. Whole goal remains Phase0–15.

## Accepted Architect amendments: exact contracts and versions

All JSON objects below reject unknown/missing fields, bool-as-number, malformed
physical refs, unsafe paths, nonfinite decimals and unknown enum values. Source
objects use canonical JSON, except the existing native owner policy retains its
native JSON encoding. Source objects contain no prospective/executable claim.

- Context exact fields are those already listed above. strategy_id must be
  aggressive_tech_manufacturing. Nullable refs never mean empty evidence.
- Events wrapper exact fields: schema_version, strategy_id, as_of, event_refs.
  Refs are sorted uniquely by (path, sha256); duplicate event_id across loaded
  objects rejects even if the bytes match. Event exact fields are those above;
  accounting_records is null or exactly before_record_id/after_record_id. Event
  kind is the five-value enum above. Announcement time <= as_of. SOURCE_DECLARED
  never authenticates issuer identity or confirms an account entitlement.
- Anchor reviews wrapper exact fields: schema_version, strategy_id, owner,
  declaration_id, declared_at, authority, reviews. authority is exactly
  {research_interpretation:true, store_mutation:false, actual_holdings_mutation:false,
  broker:false, order:false, execution:false, trade:false, policy_mutation:false}.
  owner must match the bound policy.owner, declaration_id is a strict identifier,
  declared_at is aware and <= as_of. Each review has exactly event_ref, policy_ref,
  source_record_id, reviewed_at, disposition, tracking_start_date; disposition is
  OWNER_DECLARED_RESET. Sorted unique event refs, no duplicate event identities;
  effective_date <= tracking_start_date <= as_of-date and reviewed_at <= declared_at.
  The source record must be in the frozen active ancestry and match the event's
  reconciled after-record. This is solely interpretation of a supplied declaration.
- Use only native Phase6 frozen portfolio/catalog. Policy baseline ledger SHA and
  record ancestry must match. Invalid ancestry is an explicit UNCONFIRMED state.
  Positive financial proof applies only to historical records already present.
- The policy tracking window cannot be shortened: Calendar first/last coverage
  must contain the start/T and enumerate every ordered open session. Use finite
  positive Decimal-normalized close/adj_factor values for every required session.
  Policy-only/unheld symbols are not current reconciled holdings.
- Native financial interval is before valuation_date < event effective_date <=
  after valuation_date, with exact direct-parent lineage. Only one matching native
  application, no other event/trade/fill/funding/manual evidence, no other symbol
  share/cost changes, and at least one nonzero account delta permit attribution.
  Missing explicit empty dimensions in a manual transition do not prove emptiness:
  use the matching native no-event declarations/known native manual event fields,
  otherwise return UNCONFIRMED. Aggregate deltas may remain as unattributed evidence.

Registered corporate_action_reconciliation payload exact fields (in addition to
the standard research envelope/common false authority and reconciliation_id):
as_of, trade_date, strategy_id, context_ref, decision_recipe_ref, store_plan_ref,
portfolio_source_ref, tracking_policy_ref, source_refs, company_rows, summary_state,
blocker_codes, custody_at, timing_status, prospective.

Each company row has exactly symbol, tracking_start_date, window_state,
required_dates, market_ref, transitions, events, threshold_state, blocker_codes.
window_state enum: VERIFIED, ANCHOR_MISSING, POLICY_BASELINE_UNCONFIRMED,
CALENDAR_GAP, MARKET_GAP, MARKET_CONFLICT. threshold_state is NON_EXECUTABLE.
Each transition: kind=ADJUSTMENT_FACTOR_CHANGE, previous_trade_date, trade_date,
before_factor, after_factor, event_ids. Decimal strings preserve normalized numeric
values; no binary-float equality. Every transition remains visible.

Each event report: event_ref, event_id, kind, effective_trade_date,
source_state=SOURCE_DECLARED, financial, anchor, reconciliation_state, blocker_codes.
financial exact fields: state, before_record_id, after_record_id,
cost_basis_adjustment, shares_adjustment, cash_adjustment,
unattributed_transition, source_refs, blocker_codes. state is
OBSERVED_NATIVE_POSTING or UNCONFIRMED. Each adjustment is null or exactly
{before, after, delta, unit}; decimal strings and unit CNY or SHARES. Aggregate
unattributed_transition is null or the same three named adjustment objects.
anchor exact fields: state, review_ref, policy_ref, source_record_id,
old_tracking_start_date, tracking_start_date, blocker_codes. state is
OWNER_DECLARED_RESEARCH_RESET or OWNER_REVIEW_REQUIRED; unknown entries are null.
Event reconciliation_state: RECONCILED_RESEARCH_ONLY, OWNER_REVIEW_REQUIRED,
UNCONFIRMED. Summary: RECONCILED_RESEARCH_ONLY, NO_ADJUSTMENT_OBSERVED,
UNCONFIRMED, CURRENT_FINANCIAL_CONFLICT. None grants threshold execution.

Finite blocker codes: ANCHOR_MISSING, POLICY_BASELINE_UNCONFIRMED,
POSITION_LIFECYCLE_UNCONFIRMED, CALENDAR_WINDOW_INCOMPLETE, MARKET_SESSION_GAP,
MARKET_SESSION_CONFLICT, NAMED_EVENT_EVIDENCE_MISSING, ACCOUNTING_EVIDENCE_MISSING,
ACCOUNTING_ANCESTRY_UNCONFIRMED, ACCOUNTING_EVENT_LINK_MISSING,
ACCOUNTING_MIXED_TRANSITION, ACCOUNTING_ZERO_DELTA, OWNER_REVIEW_REQUIRED,
OWNER_REVIEW_RECORD_UNCONFIRMED, CURRENT_EVENT_EMPTY_CONFLICT,
SOURCE_CUSTODY_AFTER_DECISION. Structural/SHA/unsupported-schema errors reject
before a report, using typed ContractError rather than inserting unknown codes.

Corporate recipe v2 exact fields: schema_version, existing event_pointer_ref,
previous_trade_date, market_refs, calendar_ref, market_snapshot_ref,
corporate_action_context_ref, decision_recipe_ref, research_request_ref,
store_plan_ref, custody_at. The registry supplies its exact native-v4 bindings;
legacy recipes have only the original five fields. Actual code-owned custody time
is persisted once; report created_at equals that retained custody time, and native
publication/terminal time cannot precede it. Late custody is LATE_RECORDED,
prospective=false; announced_at<=as_of alone never proves on-time historical custody.

New financial projection = all old projection fields plus schema_version=
cn-corporate-financial-projection.v2, reconciliation_ref, reconciliation_state.
New terminal output names are exactly financial_events, event_generation,
reconciliation. Legacy names/output bytes remain unchanged. A same-day declared
action conflicts with CLOSED_EMPTY: write the evidence report, terminal BLOCKED,
existing failure code CORPORATE_ACTION_UNRECONCILED, and no Store writer.

Version matrix is mandatory: execute v3; materialization v3 with all v2 fields plus
corporate_action_context_ref at materialization.v3.json; native inputs v4;
corporate recipe v2. All three materialization paths participate in conflict
checks. Legacy nullable dataclass defaults do not relax required loaded refs.
Catchup carries the explicitly supplied context ref. Completed replay, Decision
report verification and late-portfolio detection must recognize native v4.

## Accepted Critic precision

Timing is exactly ON_TIME or LATE_RECORDED. Compare aware instants; equality at
as_of is ON_TIME. Source custody is sampled internally after all context/event/
announcement/owner-policy/owner-review/portfolio/Calendar/Market bytes have been
read and validated, then persisted once in a fixed identity-bound internal
corporate recipe. There is no caller clock/custody argument in the public input
contracts; extra custody fields reject. Missing/malformed/future retained custody
rejects during replay. custody_at > as_of adds SOURCE_CUSTODY_AFTER_DECISION;
prospective=false unconditionally for this standalone artifact. Whole-DAG ledger
eligibility remains the owning native ledger's decision. Repeated preparation
selects its one exact fixed intent path from the bound refs, never a latest scan,
so a retry preserves the original actual custody time and report identity.

Owner declaration chronology is event.announced_at <= reviewed_at <= declared_at
<= as_of, using aware instants. effective_trade_date <= tracking_start_date <=
as_of session date. A violation rejects; it never becomes a successful review.

Deterministic precedence:
- Event: invalid/missing financial attribution -> UNCONFIRMED; otherwise missing
  exact valid review -> OWNER_REVIEW_REQUIRED; both valid -> RECONCILED_RESEARCH_ONLY.
- Company threshold_state is always NON_EXECUTABLE. Its sorted unique blockers
  union window/lifecycle/transition-missing-source/event/owner/conflict codes.
- Summary: any same-day event/CLOSED_EMPTY conflict -> CURRENT_FINANCIAL_CONFLICT;
  otherwise any incomplete window/lifecycle/unmatched transition or any event not
  RECONCILED_RESEARCH_ONLY -> UNCONFIRMED; otherwise any declared event ->
  RECONCILED_RESEARCH_ONLY; otherwise a verified complete window with zero factor
  transitions for all positive holdings -> NO_ADJUSTMENT_OBSERVED. A late-custody
  reason changes timing only and does not erase otherwise valid financial evidence.
- Companies sort by symbol; transitions by (trade_date,previous_trade_date); events
  by (effective_trade_date,event_id); refs by (path,sha256); IDs and blocker codes
  sort uniquely lexicographically. Pure report construction is invariant to
  internal mapping/row ordering. Serialized source lists require their declared
  canonical ordering and reject reordered input rather than changing its SHA.

Add boundary tests for cutoff equality, late custody, future/extra caller custody,
review-before-announcement, every precedence branch and pure-construction order
invariance, while retaining strict source-list ordering checks.
