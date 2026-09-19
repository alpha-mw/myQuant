# Phase12 input preparation — baseline and integration design

Status: INPUT_PROFILE_IMPLEMENTED_VALIDATION_IN_PROGRESS; exact contract review is complete.
No live run or scheduler change. Existing-profile compatibility repair is separate
from the not-yet-enabled new timing profile below.

## Current evidence

- The main workspace has no native daily DAG/head. A native Factor `verify_active`
  readback in this turn reports ACTIVE, as_of20260911, no blockers, pointer SHA
  `956239b3fda03fa29835e02e3aac61112a3390da656068b9236d3277544ab821`.
  Existing baseline is usable for first-day planning; it is not a fabricated EOD.
- `bootstrap.py` requires an existing active Factor baseline and immediate open
  succession, then verifies the Store close prefix. First setup must use the
  existing explicit bootstrap EXECUTE, not manufacture a completed seed.
- Request recipe v4 requires a fixed research_sources.as_of before maintenance.
  `theme_capture_stage.py` requires capture before that cutoff; research source
  and capture writers require actual clock at or after it. A future fixed cutoff
  therefore creates an avoidable timing barrier in a single immediate invocation;
  an already elapsed cutoff prevents new Theme acquisition. Retries cannot rewrite
  the immutable request to extend it. Daily preparation must resolve this mismatch.
- Source design Phase4 explicitly accepts SUCCEEDED or
  PARTIAL_WITH_EXPLICIT_MISSING, while the implemented Phase4 profile maps missing
  optional focus/exposure to a PARTIAL lifecycle and blocks full EOD. Investigate
  that alignment before requiring fabricated economic-exposure facts to get a
  routine daily EOD. The original request defines the intended outcome; tests and
  an earlier implementation plan do not override it.

## Proposed direction

Use one reviewed new execution/materialization/native-input profile (v5/v5/v6)
for scheduled preparation, leaving immutable old profiles and replay unchanged.
Repair existing producer/materializer responsibilities; no separate scheduler,
new completion authority, loose latest scan or alternative Factor path.

For current-day work, separate the predeclared acquisition deadline from the
actual research cutoff. Keep a code-owned deadline within the target Shanghai
date; after native maintenance/source acquisition and exact source reads, seal
one actual cutoff timestamp and its exact source refs. It must be at or after
all source availability/capture times and no later than the deadline/actual clock.
Every downstream research/Decision/corporate-context binding uses that immutable
resolved cutoff. Repeat/resume reads the same timing receipt; no timestamp reset,
backdating, wait-until-cutoff loop or adoption of future source data.

Theme capture remains DC primary/registered TDX fallback with its existing claim,
custody and budget. Its new profile separates acquisition deadline from projection
cutoff, selected after capture, rather than projecting at a predeclared future
time. Pinned sources receive the same source-time verification before sealing.
Corporate context/list wrappers may be derived for this cutoff from exact supplied
evidence refs, but no underlying event/review/owner-policy fact, timestamp, quantity
or approval changes. Unknown events remain unknown, not CLOSED_EMPTY.

For historical catch-up, use exact retained per-date PIT-valid captures when
available. Missing historical Theme/exposure sources must be represented as
explicit missing evidence; do not fetch now and pretend it was observed then.
Native Market/PIT/Calendar, critical Fundamental/Macro, corporate/accounting and
financial continuity guards stay authoritative. Determine precisely which
optional Theme/exposure missing states can finish execution with an explicit
partial business report and conservative existing Decision admission. Never mark
an unreadable/corrupt source, failed parser, wrong SHA, or critical missing input
successful. Missing reports retain both requested focus companies.

The daily input preparer takes an exact installed policy/source configuration,
acquires native Calendar only in the authorized production slot, selects the
expected date and existing baseline/complete prefix through their owning readers,
and creates immutable request/recipe/source bindings. It reuses an exact pending
or same-day EOD request for fallback, without rereading credentials/providers for
completed work. Named source configuration, available capture indexes and input
policy references must be inspected before defining final schema; no invented
live source data. First-day bootstrap is explicit and then transitions to the
existing automatic CATCH_UP path after native completion.

Extend the same launcher only as needed to dispatch that explicit bootstrap
profile safely. No-producers must still refuse an unsealed first EOD. First seed,
input registration and schedule cutover are separate truthful states.

## Required review and next implementation

Architect: confirm the timing defect and optional-partial interpretation against
the source design, assess the minimum profile/receipt changes, and identify exact
remaining contract choices for a final execution-ready plan. Do not reopen
unrelated phases or invent source providers. Critic will review the finalized
schemas, invariants and acceptance before source edits.

Acceptance must include current-day capture followed immediately by compile,
interrupted timing seal and exact replay, no future/backdated knowledge, explicit
missing optional data versus critical blockers, native initial bootstrap and
subsequent automatic daily/fallback reuse. Source/clock/provider seams must remain
disclosed. Actual current-source install/scheduler and full-DAG receipts are still
required; preparing only a blueprint or mock input is not Phase12 completion.

## Architecture decisions after source review

The Architect reviewed the timing proposal, the execution/admission distinction,
and native portfolio custody separately. The following supersedes tentative text
above and records the concrete constraints; it is not an implementation approval
for unfinished contracts.

### Preserve existing portfolio and Decision contracts

The native portfolio artifact accepts whole-second UTC timestamps only. Its
`created_at` is physical source custody, and `payload.as_of` is the Decision
cutoff. Freezing it after a previously sealed dynamic cutoff would make every new
run `LATE_RECORDED`. A new portfolio kind is unnecessary if the acquisition path
loads and rechecks the complete book and all admitted research sources first,
then uses one internally selected actual cutoff for both fields.

Split native source loading from state construction; do not supply a provisional
cutoff to `PortfolioSource`. Preserve the old fixed-cutoff function and old bytes.
The current-day path must:

1. Prepare the native Store plan, retain its exact preimage pointer, load and
   validate the native catalog/ledger/manual book, and load/recheck all exact
   admitted research-source bytes and original source-time fields.
2. Record physical read completion using an aware UTC clock. The existing
   whole-second artifact contract requires at most one bounded sleep to the next
   second (less than or equal to one second), followed by one clock sample. The
   serialized cutoff must be at or after physical read completion and every
   original source time. A regressed clock, failed bound, or expired deadline
   blocks; there is no resampling loop or waiting toward a predeclared research
   cutoff. Never change global legacy timestamp grammar or ceil the cutoff into
   the future. Original fractional source times remain in the timing proof.
3. Build exact native portfolio bytes in memory using that cutoff for `as_of`
   and `created_at`. Its ordinary plan-owned `state.v1.json` path remains fixed.
4. Construct a cutoff commitment that binds the expected portfolio byte SHA,
   exact input-source set and times, request/core/maintenance/Store ancestry,
   acquisition deadline, cutoff, and actual receipt seal time. Recheck source
   bytes before persisting this immutable commitment **before** the state bytes.
5. Write only the committed state bytes, then read back commitment, state and
   sources. A commitment without its state is incomplete. Recovery can recreate
   only its exact committed SHA; it cannot select another source set or time.

The ordering for a fresh current-day commitment is:

`native plan/source times/physical custody <= cutoff == portfolio as_of ==
portfolio created_at <= receipt sealed_at <= actual clock <= acquisition deadline`.

Cutoff and deadline must belong to the target Shanghai date. Exact existing
state/commitment replay may occur after the deadline without resampling. If the
commitment is absent after the deadline, stop. A preexisting legacy state or a
state with `created_at != as_of` cannot be adopted into dynamic mode. Current
Store-head advancement does not change retained portfolio replay; the owning
Store CAS is still authoritative. `prospective=false` remains unchanged.

Decision recipe v2 can remain unchanged because it already binds research
request, native Store plan, cutoff and portfolio state. Its reader must validate
the new request's transitive timing/state closure before accepting the recipe.

### Execution completion does not grant Exposure admission

New-profile Exposure execution may finish with native `BLOCKED`/`UNVERIFIED`
projection bytes only after a shared code-owned classifier proves all of:

- The ordinary Theme node succeeded and there is exactly one native Exposure row
  for every expected Top100 company, without missing or duplicate companies.
- The only tolerated blockers are sorted unique
  `ECONOMIC_EXPOSURE_UNVERIFIED:<expected-company>`. Reconstruct the set from
  rows whose gate is `PASS`, state is `UNVERIFIED`, qualified evidence refs are
  empty, and native reason is exactly `ECONOMIC_EXPOSURE_SOURCE_REQUIRED`.
  Reconstructed and declared blocker sets must be equal.
- Every declared source still passes the native parser, physical ref/SHA,
  company/Theme binding and cutoff checks. Unreadable/missing declared refs,
  malformed or duplicate evidence, wrong company, future information, unknown
  reasons and all other blockers remain hard failures. A provided zero revenue
  share with `THEME_REVENUE_SHARE_NOT_POSITIVE` is not missing-source tolerance.
- Original projection status/rows/blockers/source refs are unchanged. Both focus
  companies remain in their independent `PARTIAL_WITH_EXPLICIT_MISSING` reports.
  A focus partial cannot clear an ordinary projection blocker.
- Native Decision criteria remain unchanged. Removing qualified revenue evidence
  can only preserve or reduce admission; affected companies cannot gain
  `ADD_CANDIDATE`. Replay must use the same classifier as production.

Old-profile lifecycle semantics remain unchanged. Ordinary Theme structural
failure, critical Fundamental/Macro missingness, corporate reconciliation,
Store/financial continuity and PIT/Market/Calendar integrity still block EOD.

### Required version and ancestry closure

The new integration must define and review exact fields before implementation:

| Surface | New profile responsibility | Compatibility |
| --- | --- | --- |
| Execution recipe v5 | Explicit timing policy/deadline and corporate input template | v1-v4 retain exact validators and semantics |
| Theme acquisition policy/handoff v3 | Raw acquisition before cutoff; exact native capture/claim/budget/PIT scope; projection after commitment | Never reinterpret handoff v2 |
| Research request v2 | Resolved cutoff, cutoff-commitment ref, exact source descriptors and missing-only completion policy | Legacy native request shapes unchanged |
| Materialization v5 / native inputs v6 | Exact request/timing/portfolio/corporate/template ancestry and existing serving policy | Old profiles remain exact |
| Corporate input template v1 | Strategy, tracking policy, exact named event refs and owner review refs, without a speculative cutoff | Derive native context/event-list wrappers at resolved cutoff; underlying facts and unknowns unchanged |
| Collection/binding v3 / automatic request v2 | Recipe-profile selection, exact saved request identity and pending lease | Collection v2 remains recipe-v4 only |
| Decision recipe v2 / portfolio v1 | Validate new request closure transitively | No added fields or new timestamp grammar |
| Completion/ledger/Morning | Replay original version-selected timing/source/classifier closure | No latest-version or current-head fallback |

The cutoff commitment must have one fixed execution-owned path and exact schema,
including finite named source roles and original availability/capture-time refs.
Source-role readers must use owning native validators; source effective dates and
file mtimes are not availability. Policies without native availability need
conservative actual custody for current-day use, never invented historical PIT.
The full physical input set must be committed before any dynamic state write so
an interrupted run cannot attach newly acquired facts to an older cutoff.

Historical catch-up requires exact per-date retained Theme and other source
captures with original availability at or before its fixed cutoff. A receipt
sealed now is explicitly retrospective/LATE_RECORDED and not PIT/OOS evidence.
Missing ordinary Theme blocks; finite optional focus/Exposure gaps can use only
the reviewed missing-only classifier. No present-day capture substitutes for a
historical capture. Existing native portfolio late classification remains.

### Initial provisioning and actual blockers

Fresh native Factor readback is ACTIVE as_of20260911 with no blockers. The exact
native Store catalog is valid, but its current generation is
`g-daily-close-20260903-a4b3d496a4f12be2`, active record
`20260904_110322-b02`. Current Event generation
`event-close-20260903-daily-policy-v1` contains closures only through20260903;
it has no September4-11 closures. This is an actual input/financial-prefix gap,
not permission to invent empty days. First bootstrap must satisfy immediate
native Calendar succession and the native Store prefix before first EOD.

Existing exact September11 maintenance attempt is retained at
`data/private/cn_daily_maintenance/attempts/20260911T135111Z-2020-c3fc0db656d4f550`.
Exact attempt/ended readback reports PARTIAL, core blockers empty, Factor input
READY, and `MACRO_WRITE_VETO_ACTIVE`. Fundamental was HEALTH_ONLY and its stage
does not supply `research_source_ref`; the blocked Macro stage supplies none
either. The current veto SHA is
`0730ed36967ce75453486aed0967d1c238c344b1073c369843184f65fcf26c4b`, created
2026-09-06T02:54:56Z from the Sep4-target Macro failure
`MACRO_RELEASE_CONTRACT_BLOCKED` / `MacroMaintenanceError` / `UNAVAILABLE`.
No veto or source was changed. Exact read-only audit:
`.agent/acceptance/phase12-real-input-audit.json`.
Required owner/event evidence still needs source-backed audit. Standing
empty-close policy alone is not proof that an unobserved day was empty.

The input preparer and launcher must reuse the exact initial or automatic request
for the same day/fallback. Completed work remains credential/provider free.
Bootstrap input generation, bootstrap execution, first native EOD, installed
release, scheduler cutover/readback and independent unattended run stay distinct
acceptance states. The existing maintenance and scheduled authorization boundaries
remain; no new provider/broker or actual-holdings authority is introduced.

### Required acceptance and stop conditions

- Current-day native capture, source custody, cutoff commitment, portfolio freeze,
  research compile and native readback in one invocation; no future-cutoff loop.
- Crash before commitment leaves no dynamic state. Crash after commitment before
  state recovers only committed bytes. Source-set change, source drift, future
  timestamps, clock regression, wrong state SHA or missing post-deadline commitment
  blocks with no new state or substituted source.
- Repeat after deadline reads existing complete receipts without resampling or
  providers. Legacy fixed-cutoff/historical bytes and classifications stay intact.
- Missing-only Exposure/focus can finish execution while native admission remains
  conservative; every critical input/source-corruption negative still blocks EOD.
- Explicit native bootstrap, subsequent automatic daily/fallback identity reuse,
  installed current-source launch, scheduler readback and independent real receipt.

The exact contract is now `phase12_cutoff_commitment.md`; Architect amendments
and final Critic APPROVE preceded implementation. Native integration validation
is still in progress. Do not implement a loose generic timestamp declaration
or a caller-supplied cutoff as a shortcut. Do not call this design Phase12 complete.

### Owning source-time readers resolved by the Architect

The bounded follow-up architecture review resolved source-time extraction without
adding a parallel availability authority. Use the existing exact physical reader
and extend/delegate `prospective_sources.read_research_source_times`; do not scan
arbitrary JSON `timestamp` fields or infer availability from filesystem metadata.

| Role | Owning native validation | Original time and physical closure |
| --- | --- | --- |
| Industry taxonomy | `_daily_industry_projection` -> `project_tushare_industry_source` -> taxonomy plan/capture validators | Final taxonomy capture `timestamp` is the provider envelope; plan `created_at`/`document_observed_at` are provenance. Bind exact plan/capture refs. |
| Industry membership | Same projection with membership plan, every partition and final capture validators | Final membership capture `timestamp` bounds plan and all partition timestamps. Bind exact plan/capture/ordered partition refs. |
| Theme pool/focus DC and required TDX | Owning Theme handoff reader, native capture/partition validators, `_daily_theme_projection` | Separate final DC/TDX capture timestamps; native fallback-derived scopes only. Acquired input additionally binds claim `claimed_at` and actual handoff `sealed_at`. Null focus source and unused fallback produce no invented time row. |
| Exposure | `_daily_exposure_evidence` -> `build_company_source_evidence` plus exact source-file reader | Preserve each declaration's `available_at`, company, source type/page and physical ref. Shared-file envelope uses latest declared time. Explicit absent declaration produces no time row and retains native UNVERIFIED output. |
| Fundamental | `_daily_fundamental_source(include_native_time=True, decision_as_of=cutoff)` -> frozen pointer inspection -> `bind_native_availability` | Original descriptor `available_at` and returned maximum native `derivation_timestamp`. Preserve fractions; compare conservative availability ceiling. Bind retained pointer, generation manifest and native table closure. |
| Macro | First seal: native current closure verification. Replay: intrinsic `validate_macro_readiness_closure` and `macro_freshness_from_closure` | Closure `available_at`; observation times remain inside native frozen replay. Bind closure, its journal/prepared/input/frozen-pointer and observation manifest/table/evidence closure. Current heads are first-seal admission only. |
| Event Store | `load_frozen_generation` and existing corporate evidence replay | Generation `generated_at` bounds target closure `sealed_at`. Bind retained pointer/generation and exact policy/owner/source/catalog receipt refs, including symbolic resolver outputs. |
| Corporate policy/events/reviews | Native trailing-policy validator and `corporate_contracts.event`, `owner_reviews` through `ReconciliationSources` | Policy `effective_from` is only effective time; physical custody supplies current knowledge. Events use original declared `announced_at`; reviews use `declared_at` bounding `reviewed_at`. Bind exact policy/events/raw announcements/reviews and frozen record ancestry. Null refs remain unknown. |
| Portfolio | Split native source loading from construction; use existing native source and state replay | Plan `transaction_planned_at`, native record `sealed_at`, pointer `published_at`, and actual physical-read completion. Bind plan/retained pointer/catalog/ledger/manual and committed state SHA. |

Descriptive time rows must distinguish provider capture, source-declared time,
validated local handoff/closure availability, owner effective time and actual
physical custody. For each domain, exact original refs plus canonical expected
native projection/context/state SHA at the resolved cutoff are authoritative.
Replaying that same native output is mandatory; a time row alone cannot admit
data. Fundamental/Macro nested leaves stay under their owning manifest/closure
validators rather than a second parser of table or observation semantics.

The exact commitment fields and version wiring are now implemented under the
reviewed contract. Continue native integration validation using these readers;
do not restart source-reader discovery or the completed architecture reviews.

## Existing-profile repair found during implementation audit

`theme_core_binding.bind_theme_acquisition` called the v2-only recipe validator,
so supported execution recipes v3/v4 failed before native acquisition with
`EXECUTE_RECIPE_V2_FIELDS_INVALID`. The binder now dispatches through the owning
version-aware validator and explicitly requires acquisition mode. No recipe
schema or native gate changed. A regression first reproduced the v3 failure;
coverage now includes v2/v3/v4, both Theme scopes, invalid profiles, native
capture/handoff and provider-free replay. Validation receipt is recorded
separately from the unfinished new timing profile:
`.agent/acceptance/phase12-theme-profile-validation.json` (59 PASS21.63s;
scoped mypy/Black/flake8 PASS). The native capture/handoff cases use real recipe
binding and native pool/capture/readback, with core/PIT context and external
transport explicitly controlled. They are not installed whole-DAG evidence.

## Bounded implementation: missing-only Exposure completion

Status: COMPONENT_COMPLETED_LOCAL_VALIDATION. The final
Critic approved the exact selector/API and finite focus allowlist after its two
requested amendments. This component is required by the
new Phase12 input profile; it does not complete timing/provisioning integration.

The exact recipe field is `source_completion_policy`. Absence selects unchanged
legacy behavior. Its only accepted present value is the string
`native-optional-exposure-missing.v1`; null, booleans, other strings and values
reject with a typed contract error. Dispatch is in
`research_projection.project_research_source`, at the start of the Exposure
branch, before evidence processing. No native/compiler public request shape is
expanded by this component. New wrapper-v2 request/recipe construction and completed replay now propagate
and validate the selector together. Legacy requests retain their absent-selector
behavior; existing automatic task configurations have not been migrated.

Operations-owned pure functions have these exact keyword-only contracts:

- `optional_exposure_completion(*, recipe)` returns bool: false only when field
  absent, true only for the exact selector, otherwise raises ContractError.
- `validate_optional_exposure_completion(*, companies, theme, projection,
  daily_policy, evidence)` returns a tuple of sorted unique tolerated ordinary
  blocker strings, including empty tuple when ready. It validates exact expected
  company sets/sha and native artifacts, rejects extra/duplicate evidence
  companies, independently rebuilds the projection through the owning native
  builder, and requires identical canonical bytes before applying the missing-only
  predicate specified above. It never edits or returns replacement artifacts.
- `validate_optional_focus_completion(*, report, membership, pit, focus_theme,
  industry, evidence, industry_source_refs, daily_policy)` returns the tuple of
  original sorted unique focus missing codes including company suffixes. It
  independently invokes native `build_focus_evidence` with the exact supplied
  native inputs and requires byte equality, both required companies, exact
  top-level versus per-row missing-code reconstruction, and exact completion
  state (`SUCCEEDED` iff no missing codes, otherwise
  `PARTIAL_WITH_EXPLICIT_MISSING`). It is not an independent physical-ref reader;
  the caller's original source/PIT/native reconstruction remains mandatory.

The finite focus missing-only allowlist is:

- `FOCUS_SOURCE_MISSING`
- `FOCUS_PIT_NOT_ELIGIBLE` (native eligibility result, not invalid PIT evidence)
- `FOCUS_MEMBERSHIP_UNAVAILABLE`
- `FOCUS_INDUSTRY_SOURCE_MISSING`
- `FOCUS_INDUSTRY_UNAVAILABLE`
- `FOCUS_EXPOSURE_SOURCE_MISSING`
- `FOCUS_EXPOSURE_NOT_QUALIFIED` only when native economic exposure is absent,
  or is UNVERIFIED with no qualified evidence refs and native reason exactly
  `ECONOMIC_EXPOSURE_SOURCE_REQUIRED` or
  `TECHNOLOGY_MEMBERSHIP_NOT_ADMITTED`. Evidenced zero-share
  `THEME_REVENUE_SHARE_NOT_POSITIVE` is not tolerated.

`FOCUS_POOL_MEMBERSHIP_CONFLICT`, unknown codes/reasons, malformed/changed refs,
wrong company/PIT/source binding and every other focus state hard-fail. Ordinary
Theme must be READY with no blockers or unmapped rows. Required/critical domains
are untouched.

Only the exact new selector and successful ordinary plus applicable focus
validators may turn an otherwise PARTIAL Exposure node into SUCCEEDED. Original
projection bytes/status=BLOCKED/rows/blockers, focus report bytes, company source
refs, native Decision inputs, criteria and outputs stay unchanged. Tests must
prove the positive-with-evidence native Decision can actually reach an existing
admitted state, then removing that evidence reduces it to INSUFFICIENT_EVIDENCE;
comparing two already-blocked decisions alone is insufficient. Existing canonical
five-state names remain; do not introduce ADD_CANDIDATE as a native state.

Implementation ownership: operations/exposure_completion.py,
operations/research_projection.py and focused tests. All selectors absent in
existing fixtures keep their former lifecycle. Native source corruption and
unknown profile negatives must fail before successful lifecycle classification.

Component validation: `.agent/acceptance/phase12-exposure-completion-validation.json`;
59PASS,2 legacy skips,2 expensive focus cases deselected,19.05s; 2-source mypy
and3-file Black/flake8 PASS. Two native100-company legacy source tests also
completed before the wider selection was stopped. Legacy requests keep their absent selector. New wrapper-v2 propagation is
implemented; complete native cutoff/input integration verification is still
required. No installed/real-EOD claim.
