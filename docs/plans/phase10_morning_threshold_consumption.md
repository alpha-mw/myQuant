# Phase 10 — frozen owner-threshold Morning review

Status: COMPLETED_LOCAL_VALIDATION. Architect amendments accepted; Critic APPROVE.
192 focused tests and static checks passed. Full installed/live acceptance remains
a separate final gate; see `.agent/acceptance/phase10-validation.json`.

## Objective and boundaries

Close the audited Morning gap: exact prior completed EOD + same-day captured
quotes + existing owner thresholds -> deterministic, reviewable Morning report.
No maintenance, upstream producer, current-head selection, provider call, broker,
order, holdings/cash mutation, policy edit, changed factor/weight or new threshold
rule. Morning may seal only its own report/receipt through the existing path.
Existing live prospective provenance and exact-next-open-session gates remain.

## Versioned intended path

Extend the same installed Morning CLI/native operations with an explicit
`morning-strategy-request.v3`. Preserve all v2 fields/bytes/behavior and add one
required `threshold_policy_refs` object with exactly `trailing` and
`initial_stop` physical refs. `owner_policy_ref` keeps its existing quote-scope
meaning; do not silently reinterpret v1 quote policies as risk policies.

V3 accepts REPLAY/PREFLIGHT/SEAL as today. V3 SEAL output is fixed
`results/operations/morning_strategy/CN/<day>/0945-strategy.v3.md`; receipt is
`0945-run.v3.json` / `morning-strategy-run.v3`. The v3 replay result adds
`threshold_policy_refs`, `threshold_review`, and deterministic `report_markdown`
to v2's evidence fields, with `morning-strategy-replay.v3`. V2 remains exact.
Normal report preparation returns text without an upstream or report writer.
The existing caller can retain those exact report bytes for SEAL, whose normal
receipt/history path verifies them. No new report-writing executable or command.

V3 receipt additionally binds the two policy refs and exact threshold-review
content SHA. Historical receipt replay rebuilds the same view from original
refs/time rather than accepting declared report content. Existing cutover's
receipt reader may recognize exact v3 receipts through the same native history
operation; no cutover policy/state/automation will be changed in this phase.
Fixed receipt paths must be version-selected, never discovered by scanning.

## Frozen evidence and owner rules

After the existing full native EOD replay, load only its explicit Store terminal,
committed pointer, immutable catalog/ledgers/manual manifests, native inputs,
Calendar close-session receipt, frozen Market manifest and exact per-symbol
frames, and corporate node recipe/frozen Event generation. Reuse native snapshot
readers, event custody and policy rules; no mutable current/latest files. Require
new-profile EOD v4/v5 inputs and Phase 8 reconciliation evidence for v3 Morning.
All read bytes are hash-bound and rechecked before returning and sealing.

Validate the supplied trailing/initial-stop policy's exact supported native
schema/authority, strategy/owner identity, baseline ledger binding and rules.
Reuse the existing policy checks where separable; factor shared pure validators
only when needed rather than duplicate their meaning. Verify the owner policy
was recorded/effective by the captured quote request time. Do not make a later
policy appear available at 09:45. Full source refs and original policy times
remain in the review. No new owner policy or real threshold values are created.

The code-owned `calculate_position_risk` remains the threshold formula owner.
Reconstruct its inputs from frozen EOD data: exact closed-session window from the
policy anchor, exact entry reference where declared, baseline/current quantity
and cost continuity, native manual-event lifecycle, complete Event closure and
full-window adjustment-factor evidence. Preserve corporate-action and lifecycle
blockers and owner review requirements. A reconciliation report alone never
permits a reset or removes a NON_EXECUTABLE condition. Do not use retired ledger
threshold columns as active owner policy.

Calculate trailing peak/threshold/confirmed triggers using strict EOD closes
only, at the previous trade date. An initial stop retains its native strict-close
trigger and owner-confirmation rule. Same-day quote comparison is a separate
observation: at/below an otherwise usable threshold is WARNING_NOT_BREACH, never
an EOD breach or automatic trade; no intraday peak update. A blocked/unconfirmed
threshold has no usable quote comparison. Unconfigured stops remain explicitly
NOT_CONFIGURED. Cover every positive frozen holding, with extra quote-only
symbols identified separately; do not invent anchors for watch symbols.

## Review and report contract

A pure sealed `morning-owner-threshold-review.v1` contains run_date,
previous_trade_date, previous_completion_ref, quote capture/raw refs and quote
time, threshold policy refs, exact source refs, held-symbol rows, quote-only
symbols, summary state, and false authority. Rows bind symbol, prior-close risk
calculation result, original policy applicability/blockers, quote observation,
and non-executable/owner-review status. Decimal text preserves precision.
Summary is COMPLETE_RESEARCH_REVIEW or PARTIAL_RESEARCH_REVIEW. A valid report
with unavailable symbol-level thresholds remains an honest partial research
review; unavailable required source bytes/schema/policy custody reject input.

Render one deterministic Markdown report from that sealed review and the existing
Decision/quote evidence. It must include the fixed authority/timing declarations,
previous EOD/date and evidence refs, all held-symbol risk rows, quote-only scope,
explicit blockers and an exact canonical review payload/hash. Do not include
current validation time in deterministic text; preserve separate validation time
in the native result/receipt. SEAL and historical readback require exact native
rendered bytes, not merely a few declarations or a matching caller-provided hash.
V2 report declaration behavior remains for exact legacy receipts only.

For v3 an unavailable/invalid prior EOD maps to
MORNING_UPSTREAM_DAG_INCOMPLETE, with the requested prior date and bounded failing
node ids from the existing read-only status projection when available. If no
node can be identified, report completion admission failure without guessing a
node. Preserve native failure internally; never run maintenance or resume from
Morning. Public expected failure retains exit2 and the existing CommandError
fields mechanism; unexpected implementation errors remain exit3.

## Implementation and verification

Own only Morning request/result/report/receipt/history/CLI integration and small
frozen-risk-source/pure policy helpers plus relevant tests/docs. Preserve all
other dirty worktree edits. No implementation-agent delegation; required review
agents are read-only.

Acceptance covers native isolated Store/Market/Event/Calendar source reads and
real quote parser with explicit outer full-EOD fixture admission where needed;
separate full installed EOD/Morning proof remains Phase15. Test:

1. Valid frozen owner rules, unchanged formulas/quantities/cash and no write/lock.
2. Current Store/Market/Event heads advanced or corrupted without changing replay.
3. Missing/wrong prior EOD and wrong next-open/quote day -> exact prerequisite error.
4. Policy SHA, shape/authority, effective time and baseline mismatches reject.
5. Exact close history gap, changed quantity/cost, new entry, or corporate action
   stays NON_EXECUTABLE; missing stop/anchor is explicit per symbol.
6. Intraday below stop produces only WARNING_NOT_BREACH; equal boundary, no peak
   update, and prior EOD confirmed breach retains owner-confirmation semantics.
7. Complete/partial review contains all holdings and quote-only symbols exactly.
8. Report text/row/ref/hash tampering rejected; no fake time backdating.
9. V3 SEAL and historical readback bind original refs and exact deterministic report;
   repeat preserves bytes/mtime. V2 fixtures and cutover/history behavior remain.
10. Focused Morning unit/CLI/receipt tests, relevant risk/corporate tests, Black,
    flake8 and mypy. No full-repo rerun until final broad gate or new evidence.

Stop on an undisclosed owner-rule interpretation, missing indispensable original
source, or observed authority expansion. Report exact gap and keep other work
moving; do not repair it by changing a policy, data SHA, date, or source pointer.

## Accepted Architect amendments and exact contracts

Architect APPROVE_WITH_CHANGES; all seven amendments below are accepted before
Critic review. Formula implementation remains exclusively calculate_position_risk;
Phase8 is evidence, never threshold/reset authority.

### Policy identity, time and admission matrix

A Morning trailing ref different from the EOD corporate-context tracking ref
cannot apply new anchors/resets or clear blockers. Preserve the supplied policy
as context, mark affected trailing rows
TRAILING_POLICY_NOT_RECONCILED_AT_PRIOR_EOD, omit usable trailing quote comparison,
and return a partial review. Initial-stop evaluation remains independent where
its own frozen position/lifecycle proof succeeds. A new usable trailing policy
requires a completed EOD reconciliation under that exact ref.

Initial-stop effective_from AND owner_confirmation_recorded_at must be <= the
original quote request time. Trailing v1 has no invented recorded-at field: retain
its effective_from, exact policy SHA/request binding and existing EOD recipe
custody. Do not infer availability from mtime. Preserve EOD validation time and
synthetic/replay provenance in the review; late/synthetic EOD evidence remains
REPLAY_ONLY and cannot acquire live/prospective admission. Historical readback
uses original quote/request/receipt times. No wall-clock live PREFLIGHT on history.

Accepted native-input schemas are exactly cn-daily-native-inputs.v4 and
cn-daily-native-inputs.v5. Both must satisfy their existing full native replay and
provide store_plan_ref, calendar_ref, market_snapshot_ref,
adjustment_market_refs, corporate_action_context_ref, decision_recipe_ref and
research_request_ref. The selected Store terminal must provide pointer, ledger
and its exact existing native close output set; load the frozen catalog through
that committed pointer. The selected corporate terminal must provide the native
cn-corporate-recipe.v2 request/recipe, frozen event_pointer_ref, financial_events,
event_generation and reconciliation refs. The recipe's Calendar/Market/frame/
context/Decision/research/Store refs must match the native-input profile and its
Event SHA must match the native Store plan preimage. The selected Decision
terminal supplies result and its version-appropriate decision.v2.json/capture;
full native replay remains the semantic admission owner. V5 additionally retains
its existing Dashboard policy gate; Morning never depends on serving publication.
Reject mixed profiles and missing required refs rather than infer defaults.

Both policy baseline record ids and ledger path/SHA must identify exact ancestors
of the selected frozen catalog, with native quantities/costs and lifecycle proof.
Their embedded historical pointer_path fields must match the fixed native logical
path but NEVER trigger a current-pointer read. Initial-stop market_evidence and
calibration fields are retained owner-declared policy context: validate/preserve
policy bytes and supported shape, do not re-resolve their historical mutable
_latest pointer, recalibrate the stop, or use those metrics as new authority.

### Exact review and row shapes

Review top-level fields are exactly:

schema_version, run_date, previous_trade_date, previous_completion_ref,
eod_validated_at, synthetic, evidence_mode, quote_capture_ref, quote_raw_ref,
quote_requested_at, decision_result_ref, threshold_policy_refs, policy_times,
source_refs, rows, quote_only_symbols, summary_state, authority, content_sha256.

schema_version = morning-owner-threshold-review.v1. evidence_mode is
PRIOR_EOD_BEFORE_QUOTE only for non-synthetic contemporaneous native provenance
with EOD validation <= quote request; otherwise REPLAY_ONLY. This is descriptive,
not prospective admission. Native Morning live provenance remains independently
required. All authority fields are exactly the existing six FALSE_AUTHORITY
booleans. Refs use the existing exact path/SHA contract; source_refs are unique
and lexically ordered by path/SHA and contain the consumed physical sources.
threshold_policy_refs has exactly trailing and initial_stop. policy_times has
exactly trailing and initial_stop, each with effective_from and
owner_confirmation_recorded_at; the latter is null only for trailing v1.

Each held row has exactly:
symbol, name, position_ref, policy_binding, eod_risk, quote_observation,
executable, investment_authority, actions.

executable/investment_authority=false, actions=[] always. position_ref equals the
frozen EOD ledger ref. policy_binding has exactly trailing and initial_stop; each
has state, policy_ref, blocker_codes; initial_stop additionally has configured_price
(positive finite Decimal text or null), retained even when it is unusable. State
is BOUND, NOT_CONFIGURED, UNCONFIRMED,
or (trailing only) NOT_RECONCILED_AT_PRIOR_EOD. Rows are sorted by symbol and match
all positive frozen holdings exactly. Quote-only symbols are the sorted captured
quote set minus the positive holdings; no invented positions or anchors.

eod_risk is the exact native calculate_position_risk result, with these fields:
symbol, name, as_of, tracking_start_date, calculation_state, threshold_state,
moving_take_profit_review_price, moving_take_profit_reduce_price,
moving_stop_price, peak_price, peak_date, strict_close, profit_giveback_ratio,
trailing_trigger, owner_stop_price, owner_stop_trigger, owner_review_state,
actions, executable, investment_authority, blockers, trailing_blockers,
owner_stop_blockers, content_sha256.

Preserve native allowed states: calculation_state UNCONFIRMED / CALCULATED /
NOT_APPLICABLE_UNTIL_POSITIVE_PROFIT_PEAK; threshold_state NON_EXECUTABLE /
RESEARCH_ONLY / NON_EXECUTABLE_HOLDINGS_STALE; trailing_trigger NOT_CONFIGURED /
CLEAR / REVIEW / REDUCTION_REVIEW; owner_stop_trigger NOT_CONFIGURED / UNCONFIRMED /
CLEAR / BREACH; owner_review_state NOT_APPLICABLE /
OWNER_CONFIRMATION_REQUIRED_EXIT_REVIEW. Date/price fields may be null only as
native calculation permits. Values remain native finite Decimal text (optional
sign, digits, optional fraction and exponent; no surrounding whitespace, booleans,
NaN or Infinity). Preserve exact native rounding/formulas; no JSON floats for
new threshold/quote comparison values.

quote_observation has exactly price, state, trailing_review_comparison,
trailing_reduce_comparison, initial_stop_comparison. price is positive Decimal
text or null. state is AVAILABLE or UNAVAILABLE. Comparison enums are CLEAR,
WARNING_NOT_BREACH, NOT_APPLICABLE, NOT_CONFIGURED, UNCONFIRMED, QUOTE_UNAVAILABLE.
For a usable threshold, positive quote <= threshold -> WARNING_NOT_BREACH,
otherwise CLEAR. Quote <=0/missing is UNAVAILABLE and usable-threshold comparisons
become QUOTE_UNAVAILABLE. A blocked/unconfigured threshold keeps its corresponding
UNCONFIRMED/NOT_CONFIGURED comparison, regardless of quote. A correctly evaluated
no-positive-profit trailing window is NOT_APPLICABLE. Do not replace the prior
EOD owner_stop_trigger or owner_review_state when the quote recovers.

Blocker vocabulary is the existing native risk/corporate codes plus:
TRAILING_POLICY_NOT_RECONCILED_AT_PRIOR_EOD, POLICY_BASELINE_NOT_ANCESTOR,
POSITION_LIFECYCLE_CHANGED, POSITION_COST_OR_QUANTITY_CHANGED,
NEW_EVENT_REQUIRES_ANCHOR_REVIEW, EXACT_ENTRY_REF_MISMATCH,
ENTRY_REFERENCE_UNCONFIRMED, OWNER_STOP_HISTORY_GAP,
OWNER_STOP_CORPORATE_ACTION_REVIEW, OWNER_STOP_POSITION_CHANGED,
STRICT_CLOSE_UNAVAILABLE, OWNER_STOP_NOT_EFFECTIVE_AT_PRIOR_EOD, and
LIFECYCLE_UNCONFIRMED:<YYYYMMDD>.
Use the actual existing finite native code sets; do not put arbitrary exception
strings into the report. Required source/schema/custody failures reject input.
Lists are sorted unique. Per-symbol uncertainty is retained in each independent
stop/trailing blocker collection rather than erased or applied indiscriminately.

COMPLETE_RESEARCH_REVIEW requires every positive holding to have both policy
bindings BOUND, a usable/evaluated native trailing state (including the explicit
no-positive-peak state), a usable initial stop, no lifecycle/corporate/source
blockers, and an available quote. NOT_CONFIGURED in either branch, cross-policy
mismatch, missing/invalid evidence or unavailable quote -> PARTIAL_RESEARCH_REVIEW.
An honest partial review remains renderable/sealable as completed report work;
v3 receipt includes review_summary_state so operation completion cannot conceal
partial business evidence. Extra quote-only symbols do not invent requirements
for unheld positions. Empty positive holdings remains unsupported by the current
native quote-scope contract.

Canonical content_sha256 is SHA256 of canonical_json_bytes of the review with
only its own content_sha256 removed. The eod_risk native content hash is independently
validated against its existing native seal. Exact field sets/enums/types, symbol
sets, hashes and deterministic rendering are checked by producer, CLI result
validation, SEAL and historical replay as appropriate.

### Independent threshold windows and receipt timing

Trailing uses its policy tracking_start_date through the prior EOD. Initial-stop
uses its own policy effective date through prior EOD, and native owner-confirmed
quantity/cost identity, complete session history and constant adjustment factor.
When the stop effective local date is later than the prior EOD date, preserve
configured_price as disclosed policy context, mark initial_stop UNCONFIRMED with
OWNER_STOP_NOT_EFFECTIVE_AT_PRIOR_EOD, pass no owner stop to the EOD calculator,
and provide no usable quote comparison. This missing closed-session policy window
produces a partial review; never backdate the policy or pretend a prior-EOD breach.
Otherwise use the native independent stop-window checks without changing its
strict-close trigger. Any finer same-day policy/close-time ambiguity also remains
UNCONFIRMED rather than inferring earlier policy applicability.

An invalid/missing trailing anchor never erases a separately proved initial stop.
A confirmed prior-EOD stop breach remains OWNER_CONFIRMATION_REQUIRED_EXIT_REVIEW
even if the next quote is above the stop. Both branches retain no executable or
trade authority.

V3 deterministic Markdown omits wall-clock validation time. V3 receipt retains
actual first validated_at and review_summary_state. Repeated SEAL validates and
adopts exact existing report/receipt bytes. Historical replay reconstructs at the
original receipt time, preserving policy/quote/EOD custody; no current clock is
substituted for original availability and no successful receipt is backdated.

Implementation precision: both owner effective and confirmation clocks are used
for prior-close applicability. If they fall on the prior EOD date, use the exact
native Calendar timezone/session_close_local evidence; missing evidence remains
unconfirmed and a clock after that close cannot create a prior-close breach.
Both native-input v4 and v5 now use complete frozen Store commit replay, preserving
the approved no-current-head consumption matrix. Older native profiles retain
their previous replay behavior.
