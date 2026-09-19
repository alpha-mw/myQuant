# Daily Investment Intelligence Compilation

`quant-investor research compile-daily` is the stable, offline, research-only
composition path from one exact Factor production head into source-bound
Industry/Theme gates and five-state Investment Decisions. It does not restore
the retired version-named Intelligence package or any versioned CLI.

## Authority boundary

The command returns one canonical machine-JSON line and writes nothing. It
cannot call a provider or model, publish a System or Mainline generation,
construct a portfolio, write a Paper ledger, mutate Strategy Record Store or
holdings, connect a broker, create an order, execute, or trade.

Every emitted artifact remains:

```text
research_only = true
production = false
run_state = INACTIVE
authority.* = false
```

## Required sequence

```text
verified Factor production head + exact LOW/W80 OPEN observations
  -> effective-dated daily_research_policy
  -> deterministic factor_research_rank
  -> exact SW2021 Industry projection
  -> exact DC/registered-TDX-fallback Theme membership projection
  -> decision_context
  -> five-state investment_decision
  -> research_evaluation
  -> evidence_bundle
```

The command consumes a canonical exact request below `--workspace-root` and an
expected request SHA. LOW/W80 observations are separate workspace-relative
path/SHA inputs. The Factor head is copied under the Factor lock from the exact
expected pointer; the pointer is rechecked under the lock after compilation.
Any pointer, generation, signal, symbol-set, observation, PIT, Market, date, or
company-keyset drift blocks the whole command.

## Policy is mandatory

There are no implicit policy defaults. `daily_research_policy` must seal:

- explicit research strategy identity and effective interval;
- exact LOW/W80 IDs, `HIGHER_IS_BETTER` directions and weights;
- average-tie zero-to-one cross-sectional percentile algorithm;
- missing/nonfinite blocker, minimum cohort, exact pool size, pool-boundary
  tie-break and final ordering; the fixed safety ceiling is 200 companies;
- provider-qualified technology Theme IDs and provider precedence;
- `RESEARCH_APPROVED` and `PAPER_CANDIDATE` thresholds;
- Fundamental freshness policy `ADVISORY_NO_FIXED_MAXIMUM`.

The owner-approved prospective Phase A policy is code-owned and published only
through `research policy-publish`. It uses:

```text
strategy_id = aggressive_tech_manufacturing
created_at = 2026-08-21T16:00:00Z
effective_signal_date = 20260822
pool_size = 100
minimum_cohort = 3000
LOW/W80 = 0.5 / 0.5
RESEARCH_APPROVED = 0.80
PAPER_CANDIDATE = 0.90
technology_policy_state = UNCONFIGURED
technology_theme_ids = []
```

`UNCONFIGURED` is not equivalent to an allowlist that rejects every company.
It may be used only to publish the Factor Top100 research pool. Theme projection
and `compile-daily` reject it. A later ACTIVE allowlist must be a distinct,
later-effective immutable policy revision; v1 is never rewritten.

Maxwell approved the ACTIVE Theme seed after the 2026-08-26 close. The immutable
v2 policy is therefore prospective from Factor signal date `20260827`; it never
relabels the 20260824 pool:

```text
primary provider = TUSHARE_DC
fallback provider = TUSHARE_TDX
fallback rule = ONLY_REGISTERED_DC_FALLBACK_COMPANY_KEYSET
technology policy state = ACTIVE
primary DC IDs = 11 owner-approved seed themes
TDX aliases = six exact-name mappings plus two explicit owner-approved aliases
```

The corresponding `theme_governance_policy` also fixes seven top-level domains
and the economic-exposure states `HIGH / MEDIUM / LOW / UNVERIFIED`.
Membership is explicitly not economic exposure. Until annual/interim reports,
company announcements, IR records, revenue/product/customer, order/capacity or
capex evidence is replayed, a Theme-pass company remains `UNVERIFIED` with
`ECONOMIC_EXPOSURE_SOURCE_REQUIRED` and cannot become a formal Decision.

The compiler does not infer the System strategy identity from a historical
Strategy Record directory. A test or implementation-smoke policy is not an
owner policy and cannot be used for an investment conclusion.

### Source-bound exposure and Fundamental MVP

Maxwell approved the first research-only producer policy on 2026-08-28. Theme
membership never supplies an exposure level. A company can receive one only
from an exact annual report, interim report, company announcement, IR record or
revenue-structure file that quantifies Theme-related revenue share:

```text
HIGH       revenue share >= 30%
MEDIUM     10% <= revenue share < 30%
LOW        0% < revenue share < 10%
UNVERIFIED no positive quantitative revenue share
```

Product, customer, order, capacity and capex evidence may support the research
narrative but cannot alone upgrade the level. `LOW` is a hard research veto and
cannot become `PAPER_CANDIDATE`. Every evidence artifact binds a workspace-
relative source path, source SHA, availability time and exact page when
applicable.

The Fundamental MVP reads one exact registered, binding-aware and Gate-2-passed
`fundamental_daily.parquet`. It computes full-cohort average-tie zero-to-one
percentiles for ROE, ROA, inverse debt/assets, OCF/profit, FCF/profit, net-profit
growth, forecast revision and FCF/price. The five components are equal-weighted
at 20%; minimum coverage is 60%; `industry_cycle` remains explicitly missing.
The registered snapshot cutoff is disclosed but is not mechanically required
to equal the Factor signal date. Missing metrics are never imputed.

## Source projections

`project_tushare_industry_source` replays the exact stable SW2021 taxonomy and
membership plans, captures, and all partition documents. Exactly one effective
L3 membership is `AVAILABLE`; none is `UNMAPPED`; conflicting memberships are
`AMBIGUOUS`. `stock_basic.industry`, company names, and inferred mappings are
forbidden.

`project_tushare_theme_source` replays the exact DC plan/capture/partitions and
uses TDX only for the exact fallback keyset returned by the registered DC
fallback rule. Provider membership is `MEMBERSHIP_ONLY`: it can pass or reject
the technology hard gate but cannot fabricate revenue or economic exposure.
Complete empty membership is `NO_MEMBERSHIP`. A company needs a separate exact
`theme_assessment` before Theme becomes `AVAILABLE` in Decision.

## Fundamental, hypothesis, and risk

This first milestone deliberately does not accept caller-supplied Theme
exposure, Fundamental scores, hypothesis status, risk status, or generic
supporting artifacts. Provider Theme membership can pass or reject the
technology hard gate, but it is not economic exposure. Decision therefore
remains `INSUFFICIENT_EVIDENCE` and the evidence bundle remains `BLOCKED` until
separately reviewed deterministic producers with deep source replay exist.

Fundamental freshness remains disclosed under
`ADVISORY_NO_FIXED_MAXIMUM`; the compiler does not invent a same-day rule or a
Fundamental score.

## CLI

```bash
quant-investor research compile-daily \
  --workspace-root /Users/maxwell/mySpace/myQuant \
  --request <workspace-relative-canonical-request.json> \
  --expected-request-sha256 <exact-sha256>
```

Publish the code-owned immutable Phase A policy:

```bash
quant-investor research policy-publish \
  --workspace-root /Users/maxwell/mySpace/myQuant
```

Publish the later-effective owner Theme v2 bundle:

```bash
quant-investor research theme-policy-publish \
  --workspace-root /Users/maxwell/mySpace/myQuant
```

This publishes `theme-governance.v1.json` first and `v2.json` second. An
interruption after the governance leaf is inert; exact replay completes or
returns `NO_ACTION`. The v1 bytes are never modified.

Publish one eligible immutable Factor Top100 pool:

```bash
quant-investor research pool-publish \
  --workspace-root /Users/maxwell/mySpace/myQuant \
  --request <workspace-relative-canonical-request.json> \
  --expected-request-sha256 <exact-sha256>
```

`pool-publish` derives the rank itself from the expected Factor pointer, exact
LOW/W80 observation path/SHAs and the exact v1 policy path/SHA. It does not
accept a rank, selected symbols, output path, manifest or receipt. A signal
before `20260822` fails before any pool-store directory is created.

For `signal_date >= 20260827`, the request must instead bind the exact v2
policy path/SHA. The writer accepts only the code-owned v1 or v2 bytes. The v2
pool remains Factor-only until same-date DC/registered-TDX-fallback source
replay completes; its manifest says `PENDING_SOURCE_REPLAY`, never a fabricated
technology shortlist.

The immutable pool root is strategy-scoped:

```text
results/intelligence/research_pool/aggressive_tech_manufacturing/YYYY-MM-DD/
  factor_research_rank.json
  manifest.json
  publish_receipt.json
  selected_symbols.json
  top100.parquet
```

Publication uses an owner-only same-filesystem sibling staging directory and a
native atomic no-replace directory rename. There is no active/current/latest
pointer or mtime resolver. Exact replay returns `NO_ACTION` with
`publication_state=ALREADY_SUCCEEDED`; any different or unsafe existing closure
raises `RESEARCH_POOL_CONFLICT` without replacement. New publications use
`daily_research_tabular_pool_manifest`, including the exact table SHA and source
bindings. `generated_at` is actual first-publication time, retained on repeats;
the rank retains its separate source-derived timestamp. Table readback checks
the original SHA, exact 100-row order and decimal schema. Original four-leaf
legacy publications remain readable, but cannot satisfy a new tabular publication
or be upgraded in place.

Provider capture, scheduled routing, System assembly/activation, Mainline
candidate publication, I6 portfolio construction, and Paper execution remain
separate later phases. The policy and pool writers own only their exact
research-only roots.

### PROJECT_ENV Theme replay

The daily producer also supports `cn-daily-theme-acquisition.v2` for the required
PCB / AI hardware focus evidence. Its exact fields are:

```json
{
  "schema_version": "cn-daily-theme-acquisition.v2",
  "provider_priority": ["TUSHARE_DC", "TUSHARE_TDX"],
  "fallback_mode": "NATIVE_DC_PARTITION_FALLBACK",
  "maximum_companies": 100,
  "special_company_keyset": ["002384.SZ", "002463.SZ"]
}
```

Bind this policy through the existing execution recipe's `theme_acquisition_ref`.
The same daily claim covers the ordinary pool and a separate two-company focus
capture, with DC primary and an independently derived registered TDX fallback
for each scope. The v2 handoff binds both scopes and the exact retained PIT
selection, manifest and membership bytes. It is written as `handoff.v2.json`;
the policy's bytes select this filename, without discovery or fallback.

Pinned source input may instead use `cn-daily-theme-evidence-source.v2` with
exactly `schema_version`, `pool` and `pcb_ai_hardware`. Both non-null scopes use
the existing six DC/TDX source-reference fields. A null `pcb_ai_hardware` records
explicit missing focus data; it does not silently disable the requirement.

Theme publishes the ordinary artifact and the named output ref
`pcb_ai_hardware_membership`. Exposure publishes ordinary evidence and
`pcb_ai_hardware_evidence`, retaining both companies even when sources are
missing. Membership, industry, economic exposure, source refs and completeness
confidence remain separate. Topic labels do not establish company membership or
revenue exposure. Missing focus evidence leaves Exposure PARTIAL and prevents a
complete EOD claim while retaining the report. Ordinary Top100 rank, selection,
Theme/Exposure facts and Decision inputs remain independent of out-of-pool focus
facts. Overlapping companies must have matching native membership; conflicts are
reported explicitly.

V1 source, capture and materialization receipts remain readable. V1 does not
prove the two-company focus requirement. New partial research input revisions
still use existing journal gates and require exactly false authority fields;
successful terminals and old day claims cannot be rebound.

After one v2 Top100 is immutably published, the release-owned Theme replay
entrypoint builds the exact ASCII-sorted DC plan from that pool, captures DC as
primary, derives the registered fallback company keyset, and captures TDX only
for that keyset:

```bash
<installed-python> -I <clean-release-repository>/scripts/operations/run_cn_theme_replay.py \
  --workspace-root /Users/maxwell/mySpace/myQuant \
  --expected-import-root <installed-root> \
  --selected-symbols results/intelligence/research_pool/aggressive_tech_manufacturing/<YYYY-MM-DD>/selected_symbols.json \
  --expected-selected-symbols-sha256 <exact-sha256> \
  --policy results/policies/research/aggressive_tech_manufacturing/v2.json \
  --expected-policy-sha256 <exact-sha256> \
  --allow-live
```

The entrypoint reuses installed `read_project_env_token` and reads only the
owner-controlled workspace `.env` key `TUSHARE_TOKEN`. It never sources the
file, reads Keychain, logs/hashes/persists the token, retries a provider call,
or creates a second membership path. Existing exact capture roots replay with
zero network calls. A DC registry failure blocks before TDX; incomplete DC
company partitions alone form the registered TDX fallback keyset. Capture
completion remains membership-only and cannot upgrade economic exposure. The
TDX member endpoint may return industry or broad-index identities in addition
to concepts; they remain sealed in raw partitions, while projection admits
only IDs present in the exact same-date captured TDX concept registry.

## Legacy v1 09:45 daily snapshot strategy

`research morning-strategy` extends the same stable research lane; it does not
create a second data or investment system. The automation first captures one
credential-free Sina response with `scripts/capture_sina_cn_quotes.py`, then
runs `PREFLIGHT`, produces the Codex research narrative, and runs `SEAL`.

Required core closure:

```text
previous-trade-date Factor VERIFIED / ACTIVE / READY
exact LOW/W80 OPEN observations
Store-v3 registered pointer and active_closure
same-trading-date Sina capture at or after 09:30 Asia/Shanghai with exact raw SHA
actual capture time, provider time, market session and timing status preserved
installed/scheduler origin verified by the surrounding automation
```

An absent same-date Top100, Macro blocker, Fundamental partial state, Theme
economic-exposure gap, or benchmark-relative tail is auxiliary and may produce
`PARTIAL`; it cannot be silently filled. Stale Factor, invalid Store,
unavailable quote, pre-open or wrong-date capture, provider-date mismatch,
unsafe output, SHA drift or authority drift is a core blocker. `09:47` is only
the boundary between `ON_TIME_0945` and `MORNING_INTRADAY`; it is not a failure
boundary. Midday, afternoon and post-close same-date snapshots remain eligible
as `MIDDAY_SNAPSHOT`, `AFTERNOON_INTRADAY` and `POST_CLOSE_SNAPSHOT`.

`compile-daily` may remove the Macro pipeline-data veto only from one exact
`cn-macro-readiness-closure.v1` source. The closure is content-addressed,
research-only, and binds the complete successful dual-pointer journal, prepared
transaction, frozen Market/PIT/Release/Observations pointer bytes, installed
generation trees, original inputs, and any archived Macro-veto clear receipt.
Intrinsic validation remains replayable after later normal pointer advances;
current readiness additionally requires the four current pointers to equal the
closure and no live Macro veto. `available_at` must be no later than the
decision cutoff. A repair completed after a historical cutoff cannot
retroactively rewrite that decision's `PIPELINE_DATA_VETO` state.

`CANONICAL_MACRO_READY` means only that the Macro data pipeline closure is
healthy. It does not classify the economic regime as low risk, create company
alpha, loosen owner limits, activate System, or authorize Paper/trading.

The deterministic output and machine receipt are:

```text
results/operations/morning_strategy/CN/<YYYYMMDD>/0945-strategy.md
results/operations/morning_strategy/CN/<YYYYMMDD>/0945-run.v1.json
```

The receipt binds previous trade date, Factor pointer, observations, Store
pointer, quote request/response/raw SHA, scheduled reference time, exact actual
capture time, market session, timing status, capture delay, optional Top100
manifest, output SHA, core/auxiliary blockers and false broker/order/execution/
holdings-mutation flags. The `0945-*` names identify the scheduled canonical
slot, not an asserted quote timestamp. Exact replay is `NO_ACTION`; different
bytes conflict.

The 20:20 `research morning-cutover` receipt separates
`core_production_status`, `holdings_status`, and `auxiliary_status`. Its state
machine is:

```text
EVENING_PRIMARY
  -> eligible core -> DUAL_RUN
     (primary automation 09:45 + temporary 21:00 fallback automation;
      Dashboard 21:30 retained)

DUAL_RUN
  -> two successful real 09:45 receipts -> MORNING_PRIMARY

MORNING_PRIMARY
  -> missing/invalid current 09:45 receipt -> DUAL_RUN fallback resumed
```

The retired one-time 20:40 task is not part of this path. Scheduler updates
must preserve the pre-update automation configs, use Codex `automation_update`,
read back Shanghai slots, and roll back on any partial update. The source code
only seals the deterministic eligibility/action receipt.

A single RRULE cannot safely encode both 09:45 and 21:00 without creating a
cross-product of hours and minutes. During `DUAL_RUN`, the existing `automation`
ID is the 09:45 primary and a distinct temporary `cn-evening-review-fallback`
retains the former 21:00 prompt. Promotion pauses that fallback; rollback
resumes it. No duplicate data DAG is created.

### Low-frequency evidence freshness (Phase 5)

New Fundamental and Macro source recipes declare
`freshness_contract: low-frequency-source-freshness.v1` and publish the named
`low_frequency_source_freshness` output alongside their original artifacts.
Recipes recorded without this selector retain their legacy output shapes during
immutable replay; unknown selector values reject.

The report classifies each required company or registered Macro indicator as
`FRESH`, `ACCEPTABLE_LAG`, `STALE_WARNING`, or `MISSING`. It records selected
snapshot/period dates, original availability, physical source refs, exact policy
limits, warnings and critical missing codes. Unknown source dates remain null.
Status readback exposes the recorded summary after output SHA validation; it does
not rebuild evidence, change node state or certify EOD completion.

Fundamental selection filters the entire cohort at the decision date before
choosing snapshots or computing percentiles. Conflicting same-company/date rows
reject. Registered source availability cannot predate verified native derivation
or exceed the decision cutoff. Whole-second source envelopes round availability
up; the report retains the original timestamp. The existing quarterly registry
band (currently 180 days) produces a warning only. No maximum-age veto, score
weight or minimum-coverage threshold is changed. Missing eligible company rows
or existing native minimum-coverage failures produce critical missing reports.

Macro reports use native vintage/source priority and both registered availability
age and period-lag limits. Individual indicator/history/coverage diagnostics
remain warnings; no admissible required Macro observations is critical missing.
Historical replay reads the exact frozen observation pointer and generation
bound by the validated readiness closure, preserving native schema, observer
flags, table/manifest hashes, row count, content-set and retained evidence checks.
It never substitutes a current pointer. Fresh production admission still checks
canonical readiness and the existing live veto.

A critical missing report is captured before the source node returns
`PARTIAL / INPUT_MISSING`. Warning-only reports preserve source success and
native Decision compilation. Decision receives the original company/risk
artifacts; the freshness report grants no portfolio, Paper, broker or execution
authority. Completed-source replay regenerates the selected recipe's exact
report and named refs, and rejects forged states, bytes or output mappings.

### Decision v2 and pre-close portfolio context (Phase 6)

New materializations emit `cn-daily-native-inputs.v3`, which adds the required
non-null `decision_recipe_ref` to v2. Its exact
`cn-daily-decision-recipe.v2` profile binds the original research request, native
Store plan and `research_portfolio_state`. The report policy is the compiled
`native-five-state-source-completeness.v1` string; caller policy knobs reject.
Existing immutable native inputs v1/v2 keep their legacy two-output Decision
shape and do not gain a report during replay.

Immediately after Store plan preparation, materialization retains the exact
current Store preimage under the existing day journal and verifies its native
catalog, active record, ledger/manual and plan hashes, then double-checks the
selected pointer. It does not change holdings, cash, cost basis or Store state.
Subsequent replay reads retained bytes, even after the Store head advances.
Economic valuation date, native record seal, pointer publication and actual
custody remain separate. The source effective date must precede the decision
session; invalid temporal ordering rejects. A valid late seal/publication or
custody is `LATE_RECORDED`, with explicit source/custody reason codes and
`prospective=false`; it is factual retrospective context.

The new Decision adapter retains the unchanged native compiler and research
capture, then publishes exactly:

```text
results/intelligence/decision/aggressive_tech_manufacturing/YYYYMMDD/decision.v2.json
```

The report has the eight fixed source bindings `factor`, `top100`, `theme`,
`industry`, `exposure`, `fundamental`, `macro` and `portfolio`. Each binding has
artifact refs, physical path/SHA refs, `COMPLETE/PARTIAL/MISSING` status and reasons.
Each Top100 row has its original native Decision state/ref, `HELD/NOT_HELD`
membership, evidence refs, native blockers/reasons and source-completeness
confidence. The five native states remain unchanged; the report introduces no
ADD/HOLD/REDUCE/EXIT policy or trading instruction. Out-of-pool holdings remain in
the portfolio context and receive no fabricated Decision.

Confidence is `COMPLETE_SOURCE_BOUND`, `PARTIAL_SOURCE_BOUND` or `MISSING`, based
on required evidence. It is independent of the Decision label and is not a return
probability. Verified NO_MEMBERSHIP and LOW exposure are complete negative
evidence. Native adequate-but-partial Fundamental coverage, advisory freshness
warnings and late portfolio context cap completeness without changing investment
thresholds or source-node admission. All report and portfolio authorities remain
false; the native compiler evidence bundle is unchanged.

A v3 Decision terminal has exactly `capture`, `result`, `decision.v2.json` and
`decision_report_capture`. The side capture is
`cn-daily-decision-report-capture.v2` at
`results/operations/daily_production/CN/YYYYMMDD/research/decision-reports/<request_key>/capture.v2.json`.
It binds the recipe, original native capture/result, portfolio, report, research
cutoff and actual capture time. Identical writes are idempotent; differing bytes
at the dated report path conflict. Recovery after report publication or after side
capture revalidates original sources and preserves saved creation/capture times.

The report's timing label describes portfolio source/custody. Its actual creation
and side-capture delivery times remain visible separately. Standalone artifacts
always have `prospective=false`. Whole-DAG ledger validation remains authoritative:
a late portfolio is retrospective even if delivery later meets another deadline,
and actual Decision terminal/capture timing still bounds report delivery.

### Event closure integrity and historical provenance (Phase 7 prerequisite)

Native event v1 closure, generation and pointer objects now share exact validation
across builders, publication, current readback, ancestor readback and frozen replay.
A valid recomputed hash does not excuse extra fields, invalid timestamps, nonempty
financial dimensions or authority flags. Closure seal cannot precede owner cutoff;
the cutoff's Shanghai date must match the event date, and generation time cannot
precede a closure seal. Valid timezone offsets/subseconds retain original text.

Corporate adapter preparation verifies registered event ancestry once and retains
exact pointer bytes in its day recipe. Subsequent probes and completed replay use
those bytes through the native frozen-generation decoder. Missing, corrupt or
advanced current event heads do not change an already retained source selection.
Generation/path/hash/clock corruption still rejects.

Physical policy and owner-declaration refs use bounded native file reads with exact
SHA, owner checks and no symlink aliases; native repository policy mode 0644 remains
readable without changing file permissions. A source receipt beginning with
`catalog:<generation>#receipt:<id>` is resolved as native catalog metadata, not as
a filename. Only the three registered catalog filenames in the exact named
generation are considered; ambiguity rejects. The resolver checks the native
catalog, one exact no-action receipt, its semantic SHA/date/checkpoint/authority,
and the independently SHA-bound owner's explicit empty-event row. It does not
select a current Store pointer or grant new financial authority.

All seven event dimensions must still be explicitly closed empty for the standing
no-position-change close. Nonempty fills, cash movement, corporate actions or
manual changes cannot be relabeled empty. Their governed application/reconciliation
and Phase8 named-action evidence remain separate unfinished requirements; this
repair does not modify positions, cash or owner thresholds.

### Adopting an already completed native close

New native batch preparation retains the exact pre-close Store pointer in its
transaction's `source-pointer.v1.json` before staging or CAS. The original plan
schema, fingerprint and financial receipt hashes remain unchanged. A repeated
planner call for an already closed day returns internal `PLAN_ADOPTED` only after
the exact original plan, registered commit, all non-Store preimages and complete
source/committed/completion custody pass native validation. It performs no second
financial close or metadata recovery.

Materialization keeps the original plan and pre-close portfolio; actual new
custody time is not backdated. Later replay validates those immutable files and
does not select today's Store pointer. Missing or corrupt retained source blocks
adoption even when the current holdings look plausible. A legacy completed close
without source custody retains its existing registered read-only proof, but cannot
become a new adopted materialization. Partial legacy commit custody cannot trigger
financial execution. Other manual records and nonempty financial event application
remain outside this verified adoption path.

### Full-window corporate reconciliation

Execution recipe v3 requires an explicit `corporate_action_context_ref` and selects
`materialization.v3.json`, native inputs v4 and corporate recipe v2. Legacy inputs
retain their prior corporate projection. The context binds the exact existing
owner tracking policy and optional normalized named-event/owner-review documents;
missing documents mean missing evidence. Exact shapes are in
`docs/plans/phase08_corporate_reconciliation.md`.

The corporate node reads the frozen pre-close book and every Calendar/Market
session from the tracking start through the target date. It reports every factor
transition and keeps source-declared split, dividend, rights, bonus and conversion
events separate from observed market adjustments. It never infers an entitlement
or posting from the factor ratio. Registered before/after records and an exact
native application link establish observed account deltas; mixed or zero-delta
transitions cannot be attributed to a single event. A supplied owner declaration
is assessed independently and does not revise a policy or create executable
thresholds.

Named outputs are `financial_events`, `event_generation` and `reconciliation`.
The report retains exact sources and actual first custody time; late custody is
excluded from prospective ledger eligibility. Repeated preparation and completed
replay use retained evidence. A declared current-day action contradicting the
standing empty-event closure produces `CORPORATE_ACTION_UNRECONCILED`, keeps the
corporate node blocked and prevents the no-action Store writer.

Local validation includes synthetic native dividend and split postings. It is
not evidence that the canonical Zijin action has been reconciled, nor that new
scheduled inputs have been deployed. Its actual source/financial/owner evidence
must be supplied and verified before a reconciliation claim; the existing risk
veto remains in place.

## Completed-EOD Dashboard publication

Execution recipe v4 / materialization v4 / native inputs v5 select
`native-eod-first.v1`. The Dashboard node captures the financial pair and five
exact DAG source bindings. Current serving publication occurs only after native
EOD replay succeeds, under the private Dashboard publication lock. The completed
head advances monotonically; legacy exporters cannot overwrite the new mode.

Serving intent binds six data files. A post-readback selector-commit binds the
actual selector clock and both selector byte forms. The final serving receipt is
separate from the EOD hash. A partial publish returns an incomplete public result
without a public completion ref; the exact locally sealed EOD remains available
for diagnosis. Resume recovers serving evidence without rerunning producers.

After native valid-through, the UI retains a dated historical snapshot marked
STALE. It does not represent current holdings. Unproven leftover selector files
cannot establish pre-expiry success. An existing valid pre-expiry selector-commit
may support a missing historical receipt without changing its original clocks.
Read-only status validates the same complete record/byte closure as publication
replay and distinguishes expired recorded publications. EOD replay itself does
not depend on serving files or the mutable completed head.

## DAG-bound Morning v2 and v3

The installed `research morning-strategy` route also accepts the exact v2 and v3
request schemas. V2 preserves its existing quote-scope and report behavior. V3 is
the owner-threshold consumer: its request adds `threshold_policy_refs` with
exactly `trailing` and `initial_stop`, each a physical path/SHA reference to an
existing owner policy. `owner_policy_ref` still names the quote-scope policy.
This route does not create or modify owner policies.

V3 requires an exact completed prior EOD using native-input v4 or v5, its native
next-open-session proof, and matching same-day quote capture/raw bytes. It replays
Store/Market/Event/Calendar/Corporate/Decision evidence from frozen refs. Both v4
and v5 Store replay use complete frozen commit proof. Morning never runs
maintenance, repairs producers, selects a replacement EOD, or reads serving
Dashboard state. Missing upstream admission returns
`MORNING_UPSTREAM_DAG_INCOMPLETE` and bounded failing node ids when available.

REPLAY and PREFLIGHT are read-only and return deterministic `report_markdown`
plus an exact threshold review. Retain those report bytes at
`results/operations/morning_strategy/CN/<day>/0945-strategy.v3.md`, with private
Morning directories, then invoke SEAL with that exact output ref. A successful
v3 receipt is `0945-run.v3.json`. SEAL and historical readback rebuild the report
and verify its complete content, policy refs and threshold-review hash. Repeated
SEAL keeps original receipt bytes and time; historical readback does not enter
live PREFLIGHT. V3 receipts are recognized by the existing cutover recommendation
path; that path still does not apply a scheduler change.

Trailing prices/peaks come only from strict closes through the prior EOD, using
the existing owner formula. A different Morning trailing-policy ref remains
unusable until an EOD reconciliation binds it. Initial stops have independent
position, lifecycle, adjustment and time checks. Both owner effective time and
confirmation must precede the quote. A newly effective stop cannot invent a
prior-close breach; same-day activation needs the original Calendar close-clock
evidence. The configured price remains disclosed when the calculation is blocked.

Intraday touching is `WARNING_NOT_BREACH`, never an EOD breach or a new peak.
A prior-close confirmed stop breach keeps its owner-confirmation requirement even
when the next quote recovers. Missing or unconfigured thresholds remain explicit;
extra quote-only symbols never gain a position or an anchor. No threshold carries
execution authority. V3 receipt `review_summary_state` distinguishes a complete
research review from a partial one; receipt operation completion does not imply
that every stock has usable thresholds. Synthetic and replay reports identify
their provenance and do not count as live/unattended success evidence.
