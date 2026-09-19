# Phase 6: source-bound Decision v2 report and pre-Decision portfolio state

Status: APPROVED for scoped local implementation. Architect amendments accepted; final Critic APPROVE on 2026-09-12. No Phase6 source
implementation or production deployment is authorized by this plan alone; the
user's standing request authorizes scoped local implementation after this gate.

## Intended behavior and native baseline

The chat's Phase6 requires a research Decision bound to Factor, Top100, Theme,
Industry, Exposure, Fundamental, Macro and portfolio state, with a dated
`results/intelligence/decision/aggressive_tech_manufacturing/YYYYMMDD/decision.v2.json`
and per-symbol decision, evidence refs, blockers and confidence.

The canonical `investment_decision` currently has THESIS_INVALIDATED,
INSUFFICIENT_EVIDENCE, WATCHLIST, RESEARCH_APPROVED and PAPER_CANDIDATE. The chat's
six action-like names are introduced as examples. Use the canonical five states
verbatim in `decision`; do not invent ADD/HOLD/REDUCE/EXIT policies or change any
scores, weights, thresholds, vetoes or portfolio construction. Add HELD/NOT_HELD
as factual portfolio membership, separately from research state. Confidence means
source completeness, not probability of profit or a new selection score.

The existing native compiler/capture and their stable CLI/API/output shapes stay
intact. The new report is a mandatory versioned output of the new DAG Decision
recipe, with exact native compiled decisions and independently verified portfolio
provenance. Existing legacy recorded recipes/captures replay without a report.

Store is ordered after Decision, so Decision must bind the exact pre-close book
already used by the prepared native Store plan. It cannot depend on that day's
new Store terminal or add a graph cycle. The native plan already binds the source
active record, last official date and pointer/catalog hashes. `_holdings_identity`
validates native ledger/manual cash; `load_catalog_snapshot` validates retained
pointer/catalog/external bindings without selecting current state.

## Portfolio source and custody

At new materialization, after the native Store plan is prepared and before any
DAG consumer/Store execution, verify the current Store pointer equals that plan's
exact preimage. Native load and a final byte/identity read must agree. Retain those
exact pointer bytes under the day journal's immutable inputs, never in a new Store
writer root. Verify catalog SHA, source_active_record_id and source ledger/manual
refs against the plan. Do not rewrite a source pointer, ledger, cash, costs, shares
or book metadata.

Extract a bounded native `load_catalog_snapshot_bytes` decoder from the existing
snapshot loader so an exact SHA-checked pointer retained outside the Store root
can resolve its original native catalog. Both readers share the existing content
hash/schema/active-closure/catalog/external-binding checks; file reader retains
its stable double read. No current fallback, latest scan, provider, lock or writer
in historical replay. Use the native ledger/manual reader or extract its exact
read-only implementation into its owning package if dependency direction requires;
do not implement a weaker parallel accounting parser.

Portfolio rows retain sorted unique normalized symbols and exact finite Decimal
shares/avg_cost/cost_basis and cash. Reject missing/duplicate/negative quantities
and inconsistent native identity according to existing owning constraints. Zero
share rows do not become held. Native source record/valuation date must be before
the decision session and agree with the prepared plan's last official date; do
not require calendar adjacency because governed Store backlog may span sessions.
A future effective record or a timestamp after actual custody rejects. Preserve
native record sealed_at, pointer published_at and actual portfolio/report creation
and capture times separately. A valid native seal or pointer publication after
the historical cutoff is retained only as LATE_RECORDED factual context with
prospective=false and PORTFOLIO_SOURCE_NOT_AVAILABLE_AT_DECISION. It cannot be
recast as PIT/OOS or on-time portfolio evidence; it does not change native Decision
bytes. Missing, corrupt or unbound portfolio data still blocks the report.
The archived bootstrap baseline illustrates this distinction: Aug21 source record
sealed Aug21, while the synthetic Aug27 close was prepared/published Sep9.

Materialization freezes the source before downstream execution, so later Store
advancement cannot prevent resume. On a crash before the materialization receipt,
a retained source is reusable only if the original plan/preimage identity and
complete native closure still validate. Never fill an absent historical preimage
from today's book. Missing preimage evidence blocks the new Decision binding.

## Versioned schemas and report

Add exact `cn-daily-native-inputs.v3` = existing v2 fields plus required
`decision_recipe_ref`. Preserve v1/v2 shapes and reject unknown versions/fields.
The existing outer materialization receipt already binds exact native-input bytes;
verify/replay readers must validate the nested v3 recipe closure as well. New
materialization emits v3; existing immutable materializations keep their version.
`NativeDailyInputs` may have a default-null recipe ref solely for legacy callers.

The immutable recipe `cn-daily-decision-recipe.v2` contains exactly schema_version,
trade_date, as_of, strategy_id, research_request_ref, store_plan_ref,
portfolio_source_ref and code-owned report_policy. It accepts no arbitrary policy,
threshold, callable or output path. The source ref identifies the registered
`research_portfolio_state` artifact. Its exact payload consists of identity plus
standard inactive/no-authority fields and: as_of, trade_date, strategy_id,
store_plan_ref, frozen_pointer_ref, catalog_ref, source_record_id,
source_effective_trade_date, source_sealed_at, pointer_published_at, source_refs,
positions, cash, timing_status, prospective, reason_codes. Positions are symbol/shares/avg_cost/cost_basis.
All times derive from retained native evidence except actual custody time.

The registered `daily_research_decision_report` payload contains identity plus
standard inactive/no-authority fields and: as_of, trade_date, strategy_id,
report_policy, native_result_ref, portfolio_state_ref, source_bindings,
company_rows, blocker_codes, timing_status, prospective. Exact company rows contain symbol,
decision (unchanged canonical state), native_decision_ref, portfolio_membership,
evidence_refs, blockers, reason_codes, confidence and confidence_reason_codes.
The company set/order remains exact native Top100; held out-of-pool symbols stay
visible in the portfolio binding but are not assigned fabricated decisions.

`source_bindings` has exactly these lowercase keys: factor, top100, theme,
industry, exposure, fundamental, macro, portfolio.
Each key contains exactly artifact_refs, physical_refs, status and reason_codes.
Ref lists are sorted exact artifact refs or path/SHA refs; status is COMPLETE,
PARTIAL or MISSING, and reason codes are sorted. Do not mix ref schemas or infer filenames. Bind the actual selected upstream terminal/capture refs and
original native result's company/context/assessment/source identities. Factor
also resolves through the exact rank generation/observation closure already
verified by Core; Top100 uses the exact manifest. Verify company/date/SHA closure
and require original artifacts to match the compiled native result; a caller
cannot replace an upstream ref with another artifact of the same kind.

Confidence is categorical source completeness independent of the native Decision
label: COMPLETE_SOURCE_BOUND when every required domain/company binding is valid,
on-time and warning-free; PARTIAL_SOURCE_BOUND when all critical bindings exist
with advisory warnings, admissible partial coverage or late portfolio custody;
MISSING when a critical required binding is absent. Native INSUFFICIENT_EVIDENCE
is not itself a confidence mapping. Persist exact source-based reason codes. These labels
and source binding do not alter canonical Decision state or admission. Native
blockers, hard-veto reasons and low-frequency advisory warnings remain distinct.

## Execution, immutability and replay

New Decision execution publishes/adopts the unchanged native ResearchCapture,
then validates the frozen portfolio and eight-domain closure before publishing
the v2 report, journal-side report capture and terminal in that order. Missing or
invalid portfolio cannot publish a canonical Decision report. A valid report may
contain native INSUFFICIENT_EVIDENCE rows; lifecycle completeness is separate from
investment admission, as in the existing compiler.

Publish report bytes once at the required dated path using existing secure
immutable write semantics. Existing identical bytes are idempotent; different
bytes are a conflict, never overwritten. Retain an actual-time side capture under
the existing day journal with exact recipe/result/report refs, request key,
research cutoff and false authority. New terminal outputs retain `capture` and
`result` and append named `decision.v2.json` and `decision_report_capture` refs.
Portfolio/report artifact created_at values are actual code-owned creation times.
Replay/adoption reads the retained artifact timestamp before rebuilding and never
samples a replacement clock. Capture-after-write failure is adopted only after
deterministic revalidation of all original refs/bytes and timestamps; no second
canonical report is selected by recency. Identical same-input bytes are NO_ACTION;
changed evidence at the same dated path is an immutable conflict.

Recorded native-input schema selects replay: v1/v2 have no decision recipe and
retain two-output Decision terminals; v3 requires a non-null v2 decision recipe
and all four named outputs. The node request identity includes the exact recipe
ref and must agree with the native-input profile. Null/unknown v3 profiles reject. Update native registry, materialization/native-input
readback, Decision completed replay and EOD/ledger consumers to reconstruct the
exact selected profile. Do not put the new report or portfolio artifact into the
canonical compiler's evidence bundle or widen research authority. New side
captures/report validation remain in their owning adapter/readback. Status may
expose verified report refs without inferring completion. Prospective eligibility
continues to require original on-time source/capture evidence; late/synthetic or
recomputed portfolio bindings remain excluded under existing ledger rules.

## Acceptance and stop conditions

- Real native preimage/ledger/manual readback, zero/nonzero holdings, out-of-pool
  holdings, missing/duplicate/nonfinite/future portfolio sources and all SHA/record
  identity failures. No current pointer read during frozen replay.
- Exact eight-domain and company/date source closure; same-kind swapped artifact,
  swapped company, wrong Top100, wrong generation and omitted refs reject.
- Every canonical Decision state preserved; confidence/reason display cannot
  change thresholds, hard vetoes or native result bytes.
- New native input/recipe/report profiles and old immutable shapes both replay;
  unknown profiles, forged states/confidence/refs and canonical-path conflicts
  reject. Missing portfolio blocks only dependent Decision claims/actions.
- Crash adoption and repeat preserve bytes/mtimes; later Store pointer advance
  leaves original portfolio binding reproducible. No graph change or cycle.
- Focused materialization/Decision/capture/completion/ledger regressions and
  applicable static checks. Record exact source/test/native proof scope. Full
  installed/current-source CI and consecutive-day acceptance remain Phase15.
- Stop to revise this plan if an implementation would require new strategy rules,
  prospective backdating, current-head fallback, weaker native financial checks,
  a Store authority change, or external deployment.


## Accepted Architect precision

Architect review on 2026-09-12 returned REVISE; all amendments above are adopted.
These exact temporal constraints are mandatory:

- source_effective_trade_date < decision session date, and equal to the plan's last_official_date;
- source_sealed_at <= pointer_published_at;
- source_sealed_at <= portfolio artifact created_at;
- pointer_published_at <= portfolio artifact created_at;
- portfolio artifact created_at <= Decision report capture time;
- report artifact created_at <= its side capture time.

Late source seals/publications/custody are preserved with LATE_RECORDED and
prospective=false. A source seal/publication later than the original cutoff adds
PORTFOLIO_SOURCE_NOT_AVAILABLE_AT_DECISION; custody alone after cutoff has an
explicit custody-late reason and likewise caps confidence at PARTIAL_SOURCE_BOUND.
Neither state grants a source-availability or prospective permit. Future economic
state or timestamp beyond actual custody is an error, not a retrospective option.

Capture the native Store preimage immediately after prepare_materialized_store_plan:
read and match the exact preimage SHA, retain exact bytes in the journal, decode
through shared native validation, bind the plan/catalog/active record/ledger/manual,
then perform one final pointer byte/identity comparison. No new long-held Store
lock; original Store CAS still detects any competing advancement. Subsequent
report/replay never reads the current Store pointer.

Reuse owning accounting identity validation and reject duplicate symbols,
nonfinite/negative shares or cost values, inconsistent cost basis, and malformed
manual cash. Zero-share source rows remain visible and classify NOT_HELD.
Company rows resolve every native_decision_ref and evidence ref by full identity,
company/date and exact native-result/domain binding; same kind is insufficient.

For v3, non-null decision_recipe_ref is mandatory. Default-null on the dataclass
is only a legacy construction facility. Historical Decision dispatch derives
profile from recorded native-input schema, not an inferred filename or current
producer default. Preserve actual creation/capture times through partial-write
adoption; source/result/portfolio/report/side-capture closure is fully rebuilt.

Read-only baseline evidence: `.agent/acceptance/phase6-portfolio-baseline.json`
proves the exact Aug28 preimage is the committed Aug27 pointer, source record
20260909_112312-b04, effective date 20260827, but source seal and publication
2026-09-09T03:23:12Z. Native retained catalog and holdings identity readers passed
without current selection and source SHA/mtime changes. This is retrospective
context, not on-time PIT evidence. No Phase6 implementation is claimed by it.


## Accepted Critic precision

The Critic's first review returned REVISE for exact schema/timing definitions.
The following fixed contracts replace any broader wording above.

### Domain status derivation

All eight lowercase keys are mandatory and case-sensitive. Invalid/tampered refs
reject rather than downgrade. A genuinely absent required domain/company binding
is MISSING, with DOMAIN_SOURCE_MISSING:<domain> or COMPANY_SOURCE_MISSING:<domain>:<symbol>.
Within an otherwise valid domain, derive status per company first, then aggregate
MISSING > PARTIAL > COMPLETE for the domain. Global Factor/Top100/portfolio checks
apply to every row. Exact nonmissing rules are:

| Domain | COMPLETE | PARTIAL |
|---|---|---|
| factor | Native validated rank READY, exact generation and LOW/W80 refs | None; a required absent binding is MISSING |
| top100 | Native validated exact 100-row manifest/rank closure | None; a required absent binding is MISSING |
| theme | Native membership projection company status MEMBERSHIP_ONLY or NO_MEMBERSHIP, with complete registered provider capture | UNMAPPED is MISSING; no additional PARTIAL class is invented |
| industry | Native source projection company status AVAILABLE and exact company binding | UNMAPPED/AMBIGUOUS is MISSING; no additional PARTIAL class is invented |
| exposure | Native company economic_exposure_state HIGH, MEDIUM or LOW with bound source fact | UNVERIFIED or absent company fact is MISSING; low exposure itself is complete evidence |
| fundamental | Native assessment COMPLETE, no blocker_codes, freshness no warnings/critical missing | Native assessment PARTIAL with no blocker_codes, or any Phase5 warning with no critical missing; MISSING/BLOCKED or native blockers is MISSING |
| macro | Native CANONICAL_MACRO_READY and exact closure, Phase5 no warnings/critical missing | Same ready closure with Phase5 warning_codes and no critical missing; absent closure/critical missing is MISSING; PIPELINE_DATA_VETO remains existing upstream failure and is never a success waiver |
| portfolio | Entire exact native portfolio closure valid and timing_status ON_TIME | Entire closure valid and timing_status LATE_RECORDED; corrupt/absent/unbound is an error or MISSING, never PARTIAL |

For Fundamental PARTIAL, persist FUNDAMENTAL_NATIVE_PARTIAL_COVERAGE; carry exact
Phase5 source warning reasons separately. Macro PARTIAL carries exact Phase5
warning reasons. Confidence uses the eight per-company derived statuses:
MISSING if any is MISSING, PARTIAL_SOURCE_BOUND if none is missing and any is
PARTIAL, otherwise COMPLETE_SOURCE_BOUND. Native Decision label is never an input
to this classification. Status COMPLETE means complete verified evidence, not a
positive investment conclusion; verified NO_MEMBERSHIP/low exposure stays visible.

### Fixed temporal vocabulary

Both portfolio and report timing_status use exactly ON_TIME or LATE_RECORDED.
This field describes the portfolio-source/custody boundary at the native Decision
cutoff; it is not a claim about report delivery time. The report copies the
independently replayed portfolio timing conclusion. Report artifact creation and
side-capture publication times remain separate, and the whole-DAG prospective
ledger must include the latter as its delivery bound.

Parse explicit UTC instants, preserving original values; compare parsed instants,
not strings. Equality at cutoff is ON_TIME. Missing/malformed native timestamps
reject. A seal or pointer publication after actual portfolio custody rejects;
portfolio custody/report creation after actual report capture rejects. A future
effective trade date rejects. Otherwise, if seal, pointer publication or portfolio
custody is after cutoff, classify LATE_RECORDED. Add these exact sorted codes to
portfolio payload reason_codes and source_bindings.portfolio.reason_codes:

- PORTFOLIO_SOURCE_NOT_AVAILABLE_AT_DECISION: seal or pointer publication after cutoff;
- PORTFOLIO_CUSTODY_AFTER_DECISION: portfolio artifact created_at after cutoff.

The report's affected company confidence_reason_codes include the same codes;
confidence is capped at PARTIAL_SOURCE_BOUND for every late portfolio case.
Both artifacts have prospective=false unconditionally: even ON_TIME is a factual
source-time comparison, never standalone OOS/admission authority. Only existing
full-ledger validation can establish whole-DAG prospective eligibility, and it
must exclude a LATE_RECORDED portfolio and all late report delivery. This avoids
changing an immutable report if a later side capture crosses the cutoff.

### Fixed policy and side capture

report_policy is exactly the string `native-five-state-source-completeness.v1`
in recipe and report. Require equality to the compiled constant; null, objects,
unknown strings and extra policy knobs reject. The constant refers only to the
identity Decision-state projection and exact source-completeness rules above.

Side capture schema is exactly `cn-daily-decision-report-capture.v2`, with fields:

schema_version, request_key, decision_recipe_ref, native_capture_ref,
native_result_ref, portfolio_state_ref, report_ref, research_cutoff, captured_at,
authority.

Authority is the existing exact false-authority map. Ref objects are exact
path/SHA pairs. Path is exactly
`results/operations/daily_production/CN/YYYYMMDD/research/decision-reports/<request_key>/capture.v2.json`
using the exact existing DailyJournal ROOT and fixed suffix. No caller may
supply a path. Request_key must be the exact native Decision node request identity.
The native capture/result refs must equal the unchanged ResearchCapture outputs.
The recipe/portfolio/report refs must equal the validated v3 closure and required
fixed dated report path. research_cutoff equals original as_of; captured_at is
actual code-owned UTC time sampled after report write, never as_of or file mtime.
Capture time must be no earlier than all dependent artifact/custody times and no
later than the terminal finish. Readback never samples a replacement timestamp.

Before writing an absent side capture, validate every retained artifact/ref and
regenerate report bytes using its saved created_at. A report left without a side
capture may be adopted only after this full validation. An existing side capture
is validated and adopted with original captured_at; mismatch is an immutable
conflict. Crash after side capture but before terminal also preserves exact bytes
and times. Same input produces NO_ACTION; changed input at the dated canonical
report path fails closed.

Additional focused tests: cutoff equality; seal-late, publication-late,
custody-only late and combinations; malformed/future source/custody timestamps;
status/confidence caps for each; missing/extra/case-variant domain keys; unknown
policy; crash between report/capture and between capture/terminal with unchanged
original timestamps. Economic-date, source-time and delivery-time assertions
must be checked separately.
