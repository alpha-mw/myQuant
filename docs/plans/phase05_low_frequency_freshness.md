# Phase 5: point-in-time low-frequency evidence and advisory freshness

The chat requires FRESH / ACCEPTABLE_LAG / STALE_WARNING / MISSING, point-in-time
selection, known_at <= decision_time, correct supersession, and blocking only
for missing critical Fundamental evidence. Ordinary lag must remain nonblocking.

An exact reproduction in `.agent/acceptance/phase5-before.json` shows the current
Fundamental assessment builder choosing a 20260830 row for a 20260828 cutoff.
It sorts each company's entire frame and takes the final row without filtering.
The current approved Fundamental policy is ADVISORY_NO_FIXED_MAXIMUM. Keep its
bytes, native score weights, minimum coverage and research states unchanged.

## Native Fundamental selection

Before percentiles or per-company scoring, normalize supported session-date
representations without converting numeric YYYYMMDD into Unix nanoseconds.
Reject invalid dates. Restrict the entire scoring cohort to rows whose session
date is no later than the decision's session date. Future rows must not affect
either the chosen company row or cross-sectional normalization.

Select the greatest eligible date per company. Identical duplicate rows may
collapse, but conflicting rows for the same company/date must reject unless an
existing native revision contract can order them. Do not choose by input order,
mtime, or an invented revision field. The canonical Fundamental mart's existing
announcement/revision logic remains authoritative and unchanged.

All-future or absent company rows remain missing critical evidence. Noncritical
missing metrics remain governed by existing native assessment coverage; do not
turn every null metric into a new blocker or relax the existing minimum coverage.

For registered Fundamental pointers, keep exact pointer/generation/table replay
and bind declared available_at to native derivation timestamps from the verified
manifest/pointer. It cannot precede native availability or exceed the decision cutoff.
Accept UTC instants without fabricating an earlier time: if the Intelligence
envelope needs whole seconds, use a conservative upper bound and retain the
original source time in the freshness evidence. Do not use read time or mtime.
Legacy source fixtures/readback remain explicitly legacy; they cannot claim
registered native availability that they do not carry.

## Four-state observation contract

Add one registered research-only `low_frequency_source_freshness` artifact.
Its fields are freshness_id, domain, as_of, trade_date, source_refs, policy,
entries, freshness_state, warning_codes and critical_missing_codes, plus the
standard inactive/no-authority fields. It describes source admissibility and
freshness; it never changes Factor weights, Macro score weights or trading gates.

Each entry has exactly subject_id, snapshot_date, known_at, age_days,
freshness_state, warning_codes and critical_missing_codes. Unknown dates/times
are null and correspond to missing evidence, not fabricated freshness.
Entries and code lists are sorted deterministically. source_refs bind actual
physical input bytes. Domain is FUNDAMENTAL or MACRO.

Fundamental entries use the selected native company snapshot and verified source
availability. Classification is:

- MISSING when no eligible snapshot exists or native coverage identifies missing
  critical evidence;
- FRESH when an eligible snapshot's session date equals the decision date;
- ACCEPTABLE_LAG for earlier eligible snapshots within the existing registered
  quarterly period-lag band (`macro.registry.PERIOD_MAX_LAG_DAYS[quarterly]`,
  currently 180 calendar days);
- STALE_WARNING beyond that band, retaining the evidence and its native scores.

The reused quarterly band is a descriptive warning threshold only. It is not a
maximum permitted age and does not modify ADVISORY_NO_FIXED_MAXIMUM. Record the
exact code-owned band and its source in the artifact policy field; no caller can
inject a different threshold. Missing noncritical metrics add warnings while
the native coverage decision remains authoritative.

Aggregate state is MISSING if critical_missing_codes is nonempty; otherwise the
worst available freshness state (STALE_WARNING, then ACCEPTABLE_LAG, then FRESH).
Noncritical missing entries remain explicit warnings and do not become blockers.

## Macro uses its owning temporal rules

Reuse native observation validation, vintage selection, frequency-specific
availability/period-lag bands and the existing Macro snapshot's freshness data.
Do not add indicators or change score/overlay calculations. Ordinary monthly or
quarterly observations can therefore be ACCEPTABLE_LAG without daily equality.

Read observations through the exact pointer/generation retained by the validated
Macro readiness closure. `load_observations(generation_id=...)` currently still
reads the current pointer before switching to the requested generation; do not
use that as an archived-only reader. Extract a bounded native pointer/row decoder
if needed, preserving generation/table SHA, row count, content-set hash, schema
and evidence-record validation. No latest/head lookup, copies or provider calls
are permitted on the archived lane.

The existing CANONICAL_MACRO_READY and PIPELINE_DATA_VETO authorities remain
unchanged. Never clear a live veto or waive readiness/Calendar/PIT constraints.
No PIT-admissible observations is missing critical data. Existing observer-only
coverage/insufficient-history diagnostics remain warnings; do not promote them
or theoretical Macro scores into new control authority. Stale native indicators
produce STALE_WARNING while their original signal/quality status remains visible.

## Production and immutable replay

New Fundamental/Macro source-node recipes explicitly declare
`freshness_contract=low-frequency-source-freshness.v1`. This is a fixed internal
recipe contract selector, not a caller-defined policy or callable. Existing
recorded recipes without the field retain their old output shapes. Readers
accept only the absent legacy selector or this exact value and reconstruct the
corresponding original recipe. New production always emits the selector.

For new recipes, append the freshness artifact under the named output ref
`low_frequency_source_freshness`; retain original source/risk artifacts unchanged
when their inputs are valid. A missing source still produces a MISSING report
with all required subjects, then an existing BLOCKED/PARTIAL INPUT_MISSING node
outcome. Warnings alone cannot prevent source-node success or Decision execution.
Native malformed/PIT/future/ambiguous evidence remains a typed input failure,
never a warning-shaped permit to use invalid data.

Update shared recipe/output-name reconstruction, source capture, status readback,
completed research/Macro replay and research-only revision checks as needed.
Decision consumes original native company/risk artifacts only, not the freshness
report. Old immutable requests, source artifacts and terminal receipts are not
rewritten. Old evidence that actually violates PIT must be rejected rather than
preserved as valid. Full current-source release/CI acceptance remains separate.

## Verification

Architect then Critic review precedes implementation. Required cases:

1. Future row excluded before both selection and percentile normalization;
   all-future company missing; later eligible snapshot wins; conflicting same-day
   rows reject independently of input order.
2. Known-at after cutoff and backdated registered availability reject; timestamp
   precision cannot make evidence appear earlier than its native source.
3. All four states through the real source adapter. Old but PIT-valid complete
   Fundamental data yields warnings and permits Decision; native critical missing
   coverage blocks, while noncritical nulls do not gain new blocking authority.
4. Macro native vintage/frequency behavior, noncritical stale diagnostics,
   no-PIT-data missingness and unchanged authoritative veto rejection.
5. Archived Macro reads work with exact frozen pointers, without current-head,
   provider, lock or writer calls; tampered generation/row/evidence refs reject.
6. New report/capture/ref replay is exact and read-only; old recipe/output shapes
   still replay for valid original inputs. Unknown selectors and forged report
   states/refs reject.
7. Relevant source/Decision/Macro/ledger regressions and changed-file checks.
   No Fundamental promotion, source-pointer edit, scheduler change or live run.

## Accepted Architect precision

Fundamental session dates accept canonical YYYYMMDD strings, exact ISO
YYYY-MM-DD strings (an existing repository input form), eight-digit integral
values excluding booleans, date objects, and datetime/Timestamp values. Naive
datetime values use the Shanghai session date; timezone-aware values convert to
Shanghai before extracting that date. Reject floats, nulls, malformed/ambiguous
text and unsupported representations. Filter the entire cohort before selection
or percentile calculations. Conflicting eligible duplicate company/date rows are
an input error; identity includes all normalized original row columns, including
all scoring metrics and native provenance. Exact duplicate rows may collapse.
Unsupported provenance types must reject instead of being discarded to choose a
winner. A separate reproduction, `phase5-ambiguous-before.json`, shows current
selection changing between 1.0 and 2.0 when identical row sets are reordered.

The Fundamental report policy is exact:

```text
threshold_source = macro.registry.PERIOD_MAX_LAG_DAYS.quarterly
warning_after_days = 180 (read from the compiled registry)
blocking_after_days = null
effect = WARNING_ONLY
fundamental_policy = ADVISORY_NO_FIXED_MAXIMUM
```

Critical missingness is exactly a required company without an eligible snapshot
(including all-future rows), or an assessment failing the existing native minimum
coverage contract. No new critical-metric list, score weights or coverage limit.
STALE_WARNING alone retains the native assessment and permits Decision execution.

The common entry schema additionally contains nullable frequency,
availability_age_days, period_lag_days, availability_limit_days and
period_lag_limit_days. These are null for Fundamental; its age_days refers to the
selected snapshot date. For Macro, snapshot_date is selected period_end and
known_at is the selected vintage's available_at. Both ages and their registered
frequency-specific limits are explicit, and either exceeded limit produces
STALE_WARNING. Preserve native stale-availability, stale-period, history, quality
and source diagnostics separately. The expected Macro subject set is derived
from the compiled indicator registry; no admissible registered observations for
the required Macro closure is critical missing, while missing individual
noncritical indicators continue through existing coverage/readiness semantics.
No observer score or coverage warning becomes a new trading/control authority.

The archived Macro reader accepts exact frozen pointer bytes plus the expected
SHA and generation ID from the validated readiness closure. It resolves only
that generation's manifest/table and preserves schema, observer flags, row count,
content-set hash, table/manifest SHA, evidence-record drift validation and native
MacroObservation.from_mapping. It never reads a current pointer or source path
selected by recency. Native vintage conflicts, priority, cutoff and frequency
rules remain unchanged.

Sub-second availability uses a ceiling only where a whole-second Intelligence
timestamp is required; the original native timestamp remains in the report.
Recipe selector absence means legacy; only the exact new selector is accepted
otherwise. Warning-only reports cannot change SUCCEEDED to PARTIAL/BLOCKED.
Critical missing reports are emitted before an existing INPUT_MISSING outcome.
Legacy compatibility does not validate future-dated or corrupt original evidence.
