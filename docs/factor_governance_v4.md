# FactorGovernanceProtocol v4 research scaffold

FactorGovernanceProtocol v4 is a research-only shadow contract. It does not
replace the current v15 FactorGovernance v3 runtime, change an activation
pointer, write `mined_factors.json`, append a production WAL, or authorize new
risk. v2 and v3 evidence remain historical/current evidence in their own
contracts and are never auto-upgraded to v4.

## Readiness states

The target production set has exactly ten healthy factors.

- `ready_target_10`: exactly ten healthy factors and every set-level contract
  passes.
- `underfilled_accelerated_mining`: five through nine healthy factors pass the
  minimum baseline. Weekly mining continues in accelerated report-only mode.
- `no_new_risk`: fewer than five factors, fewer than three families, more than
  ten factors, an invalid weight/slot/evidence/runtime contract, stale or
  blocked health, or a missing/invalid activation receipt.

`new_risk_eligible` is a readiness result, not authority. This scaffold always
returns `new_risk_authorized=false` and
`production_apply_enabled=false`. Any production activation must fail closed
unless a separate production workflow verifies the same-day hash-bound receipt
and every independent production gate. The v15 default remains authoritative;
v17 consumes this protocol only as a nonauthorizing, version-neutral input.

For exactly five normalized factors, the 20% per-factor cap means every factor
must have exactly 20% absolute weight. The 35% family cap therefore forbids two
of those factors from sharing a family: the five-factor baseline mathematically
requires five distinct families. For larger sets the minimum family count is
three, subject to the same 35% cap.

## Independent quality and shadow observation

Factor quality is now evaluated in a separate
`factor-quality-readiness.v1` domain. It answers whether an explicit research
set has sufficient technical evidence for report-only shadow observation. It
does not reinterpret `healthy`, `factor_governance_ready`,
`new_risk_eligible`, or any activation field.

The quality evaluator keeps Gates 1-8, strict-calendar maturity, family BH
`q<=0.10`, verified v4 runtime and replay identity, fresh health, unique names,
and unique slots. It deliberately excludes production state, positive weight,
the 20%/35% allocation caps, and the same-day activation receipt. Therefore a
zero-weight `production_candidate` may qualify for shadow observation while the
same record remains ineligible for production. In particular, five qualified
records need at least three families in the quality domain; the unchanged
production five-factor gate still requires five families because of its weight
caps.

Quality statuses are deterministic:

- `invalid`: malformed input, duplicate identity/slot, runtime-set identity
  drift, or expected content-hash drift;
- `blocked`: no qualified records, or at least three records with fewer than
  three qualified families;
- `partially_qualified`: some, but not all, records qualify;
- `insufficient_for_shadow`: one or two qualified records;
- `shadow_observation`: three or four qualified records in at least three
  families;
- `ready_underfilled`: five through nine qualified records in at least three
  families;
- `ready_target_10`: ten qualified records in at least three families;
- `shadow_observation_above_target`: more than ten qualified records; the set
  may be observed but is not quality-ready as a target set.

The cycle-free `quality_set_identity_sha256` binds sorted
`name/family/slot` identity and must be embedded in every quality record's v4
runtime contract. The separate `quality_set_sha256` also binds calendar,
runtime-contract, and replay hashes. Persisted quality assessments retain exact
normalized source evidence; validation recomputes every row, count, blocker,
status, eligibility flag, and hash. Neither hash is part of the production v4
protocol hash or activation receipt.

`assess_factor_governance_readiness_v4` accepts quality records only through an
explicit optional sidecar. Calls that omit them retain the historical output
shape exactly. When present, `quality_assessment` is additive and report-only.
Any research summary may normalize it for display but must never read it when
calculating Factor readiness, activation gates, blockers, or authorization.
Historical artifacts without the nested quality summary remain historical only.

## Healthy factor contract

A factor is healthy only when all of the following pass together:

1. State is exactly `production_factor` and weight has a finite, non-zero
   magnitude.
2. Governance gates 1 through 8 are present and pass.
3. Maturity is either 12 distinct actual month-end RankIC sessions or 8
   non-overlapping cohorts of 30 consecutive open sessions from the exact
   strict-Parquet-bound calendar. A standalone 90-day diagnostic, weekend
   arithmetic, or a caller-declared horizon is not authoritative.
4. Benjamini-Hochberg is applied by family and the factor q-value is at most
   0.10.
5. The factor has one non-empty slot and no other production factor occupies
   that slot.
6. Its runtime contract is verified and its canonical semantic SHA matches.
7. Its latest health observation is fresh and `healthy`.
8. The set has a valid same-day v4 activation receipt bound to the protocol,
   transaction plan, registry bytes, production factor set, and aggregate
   runtime contracts.

The normalized absolute-weight caps are 20% per factor and 35% per family.
`data_blocked` is not an alpha failure and must not increment failure streaks,
but it blocks factor health and set readiness until fresh data closes it.

Candidate admission is also machine-checkable before any registry proposal.
The candidate must carry a healthy strict-Parquet source receipt bound to the
full-A universe, exact owner-only `0600` ontology/catalog/screening readback,
one recomputed family-BH row with q-value at most 0.10, exact dedup evidence
against the replay comparison factor set, `initial_weight=0`, an authoritative
maturity route, all eight gates, and verified v4 replay evidence. Caller-supplied
BH values and dedup booleans are diagnostic only. The canonical replay context
binds the screening and dedup semantic hashes, so an artifact cannot be omitted
or substituted after replay. Failure returns an explicit blocked assessment;
the validator raises and no candidate registry proposal is allowed.

## Cadence and proposals

Weekly mining and factor-health runs are report-only. They cannot create an
add/replace/watch/reduce/deprecate transaction or change weights.

Month-end may emit at most two proposals. `watch_proposal`,
`reduce_proposal`, and `deprecate_proposal` remain proposals; none is an apply
command. While the set has fewer than ten factors, a reviewed add proposal may
fill a missing slot. Once the set reaches ten, additions are forbidden and each
replacement proposal must be one-in-one-out. Its paired incremental-edge 95%
confidence lower bound must be strictly greater than zero.

Distinct matured alpha failures map to month-end proposals as follows: one is
`watch_proposal`, two is `reduce_proposal` with a suggested 50% weight, and
three or more is `deprecate_proposal`. Duplicate failure-window IDs count once.
Data-blocked windows never enter that count, but they identify the blocking
factor and block factor-set new risk. Every result remains `apply=false`.

## Canonical replay v4

Each A/B/C/D replay arm contains exactly this chain:

`Eligibility -> Quant -> Funnel -> CodexS1 -> Bayesian -> RiskAdvisor -> CodexIC -> PortfolioConstructor`

Every stage binds the exact v4 context, predecessor byte SHA, predecessor
semantic SHA, and its own output semantic SHA. Eligibility binds the admitted
symbol universe. Quant may score only that universe; Funnel may only narrow
it. Bayesian binds CodexS1, RiskAdvisor binds Bayesian, CodexIC binds semantic
hashes for every upstream stage, and PortfolioConstructor consumes exact
CodexIC decision hashes.

RiskAdvisor is advisory-only. Its output remains hash-bound in the replay, but
a positive PortfolioConstructor weight does not require RiskAdvisor approval.
A positive weight does require a hash-bound CodexIC `buy` decision.

The replay declares exactly one transition mode:

- `add`: the production set has fewer than ten factors, B remains the same as
  A, and C/D add one challenger into a previously empty slot;
- `replace`: the production set has exactly ten factors, B removes one named
  incumbent, and C/D insert one challenger into the incumbent's exact
  family/slot.

Every one of at least eight non-overlapping 30-open-session cohorts carries
gross return, turnover, cost rate, recomputed cost, and recomputed net return
for all four arms. The paired D-minus-A net-return series is recomputed from
those cohorts. At target ten, the replay rejects a challenger unless the
Student-t two-sided 95% lower confidence bound is strictly positive; an
underfilled `add` records the bound but does not use it to invent production
authority.

Replay context binds the exact eligibility set, open-session calendar, PIT
membership, byte-inventoried market Parquet input, runtime contract, frozen
candidate catalog, screening evidence, dedup evidence, quantitative evidence,
strict latest pointer, and snapshot manifest. Replay and evidence schemas are
v4-only. Local readback accepts one explicit absolute regular file, owned by
the current user, mode `0600`, with canonical JSON bytes and no
scan/latest/fallback behavior.

## WAL, CAS, rollback, and receipt

`build_factor_v4_transaction_plan` produces an inert plan with four independent
bindings:

- an append-only WAL blueprint containing before/after registry byte hashes;
- a CAS blueprint comparing the exact expected registry SHA before any swap;
- an inverse rollback blueprint that compares the proposed SHA and restores
  the exact before-registry and factor-set hashes;
- a separately authorized activation-receipt requirement.

The plan carries `plan_only=true`, `registry_mutation_performed=false`, and
`production_apply_enabled=false`. The WAL is `planned_not_written`, CAS is
`planned_not_attempted`, and inverse rollback is `planned_not_applied`.
Building or validating the plan performs none of those operations.

`FactorV4ShadowTransactionStore` is the separately testable research store.
It has no default path and owns only fixed files beneath a caller-supplied
`shadow_root`; it cannot discover or open the current Factor registry or
activation pointer. Under an independent file lock it can initialize a v4
shadow registry, append and fsync a 0600 WAL intent, write a 0600 inverse
manifest containing the exact before bytes, CAS the exact expected shadow
registry SHA through an atomic replace, read back the result, append a commit
record, and write a 0600 shadow receipt. Rollback requires a separate scope,
CAS-compares the post-transaction SHA, restores the exact before bytes, appends
rollback WAL records, and replaces the shadow receipt with a revocation.

The store's receipt schema is
`factor-governance-shadow-activation-receipt.v4`; it explicitly carries
`production_activation_performed=false`. It cannot satisfy the production
activation-receipt validator used by readiness and therefore cannot unlock
production risk.

An activation request is not a receipt. A receipt validates only with the exact
`factor-governance-activation-receipt.v4` schema, an explicit
`factor_v4_production_activation` authorization scope, same-day timezone-aware
activation time, and canonical hashes for its complete activation context and
receipt. Receipt validation still does not perform activation.

## Offline scripts

The pre-admission runner freezes the exact 230-candidate catalog before loading
market data, hashes the resolved table/serving Parquet inventories before and
after recomputation, then recomputes every raw p-value and the full
ontology-family BH denominator from strict Parquet inputs:

```bash
RUN_ID=factor_v4_pre_admission_<timestamp>
PRIVATE_ROOT=reports/factor_governance/private/v4_pre_admission
CATALOG_PATH="$PRIVATE_ROOT/$RUN_ID/candidate_catalog.v4.json"

./.venv/bin/python scripts/build_factor_v4_pre_admission_report.py \
  freeze-catalog \
  --run-id "$RUN_ID" \
  --private-root "$PRIVATE_ROOT" \
  --catalog-path "$CATALOG_PATH"

./.venv/bin/python scripts/build_factor_v4_pre_admission_report.py \
  screen \
  --run-id "$RUN_ID" \
  --private-root "$PRIVATE_ROOT" \
  --catalog-path "$CATALOG_PATH" \
  --expected-catalog-sha256 <catalog-file-sha256>
```

The output is private report-only evidence. `pending_codex` means screening
finished but Codex S1, Codex IC, or canonical replay evidence is absent. It is
not a healthy-factor count, registry proposal, activation receipt, or authority
to take new risk.

The repository currently validates externally supplied v4 Codex-stage, dedup,
canonical-replay, transaction, and production-receipt artifacts, but it does
not yet contain the complete producers or production receipt issuer for that
chain. The replay-validation and transaction-plan scripts below therefore
cannot convert screening output into an activation receipt.

The strict local run
`factor_v4_pre_admission_20260718_083224` evaluated 230/230 candidates with no
compute failures. It produced 174 BH-passing rows across 13 passing ontology
families and bound 90 table plus 5,728 serving Parquet files through
`market_data_input_sha256=130402f1c9d42a811229719690a90c5828aa3a52e5f4342041cf081f6785128f`.
It correctly remained `pending_codex`, with an empty proposal list and every
production/apply flag false. The earlier `factor_v4_pre_admission_20260718_065904`
run is retained as superseded diagnostic history because it predates the exact
market-Parquet byte inventory binding.

### Private A_quant formal-catalog classification

“Formal catalog” here means only that the ontology and candidate catalog are
readable by the exact v4 schema validators; it is not formal factor admission.
The private v4.1 classifier partitions exactly 100 pinned A_quant ideas into 37
new classification definitions, 6 structural aliases, and 57 incompatible
ideas. It preserves all 230 base candidates and adds only the 37 new
definitions (`230 + 37 = 267`), while preserving the 13 base primitives and
adding five classification identities (`13 + 5 = 18`).

The adapter is validate-only. Every descriptor remains non-executable, not
screening-eligible, and not proposal-eligible. The materializer recomputes
source authenticity, while the adapter explicitly does not claim independent
source-authenticity or runtime-equivalence proof. This lane does not load
market data, execute signals, screen candidates, compute statistics, establish
qualification, write the registry, create a proposal, apply a change, create a
research head, replay evidence, or build a transaction plan.

`scripts/build_factor_v4_1_formal_catalog.py` requires explicit absolute
Discovery/base paths, byte and semantic hashes, the exact code-file bindings,
and five registry/CN protected-control hashes. It publishes an owner-only,
no-clobber seven-file research bundle. Protected-control hashes are persisted
in the readback report but prove only build and locked precommit identity;
external maintenance is not locked and postcommit stability is outside bundle
acceptance. The accepted reference bundle is
`factor_v4_1_formal_catalog_20260718T191045Z`, with readiness
`EXPLORATORY_FORMAL_CATALOG_CLASSIFICATION_ONLY` and no admission, activation,
or new-risk authority.

### Private pinned-operator runtime equivalence

The separate v4.1 operator proof has exactly one claim:
`pinned_aquant_operators_under_hash_bound_myquant_pit_envelope`. It loads the
exact `FactorExpression` and operator source blobs from pinned A_quant commit
`4424dcecc384f614b0e9fd5e36cf094e9244bad5` through `git cat-file`; it never
imports the A_quant worktree. The proof binds both source-byte hashes, the exact
37 ordered definitions, all eight predecessor artifacts, all four proof-code
files, the Python executable, and NumPy/Pandas/SciPy module versions and bytes.

The deterministic differential fixture covers changing PIT membership, NaN,
positive and negative infinity, zero denominators, cross-sectional ties, and
the maximum 200-session `ts_mean` window. The raw pinned operator result must
diverge from the explicit PIT envelope for at least one exact candidate, while
the enveloped pinned runtime and local evaluator must match for every candidate
and every primitive probe. Fixture and runtime descriptors are recomputed on
readback rather than trusted from the artifact.

`scripts/build_factor_v4_1_operator_runtime_equivalence.py` requires explicit
absolute paths and expected SHA-256 values for these eight inputs:

- `aquant_source_receipt.v4_1.json`
- `source_idea_audit.v4_1.json`
- `primitive_mapping_proof.v4_1.json`
- `candidate_catalog.v4.json`
- `formal_catalog_materialization_readback.v4_1.json`
- `no_label_operator_profile.v4_1.json`
- `no_label_signal_diagnostic.v4_1.json`
- `no_label_diagnostic_readback.v4_1.json`

It also requires expected byte hashes for the builder, evaluator, equivalence
module, and private-bundle I/O module. It publishes only
`operator_runtime_equivalence_proof.v4_1.json` plus its independently rebuilt
readback under the owner-private, no-clobber
`reports/factor_governance/private/v4_1_operator_runtime_equivalence/` root.

The accepted reference bundle is
`factor_v4_1_operator_runtime_equivalence_20260718T215545Z`. Its proof byte SHA
is `aae6aaa6ece964c59e0703d0bd8e1069e7b080f70f9e365bde118ff62c2a8cf7`
and its independent readback byte SHA is
`07b3d166ac870890f22828fb11e23c2b2d7de13f5af4fc96c04a8fe901f4cba1`.
The proof records 37 exact candidate matches, 34 raw-runtime/PIT-envelope
divergences, seven exact primitive probes, and 105 infinite probe outputs.

This artifact deliberately sets only
`operator_runtime_equivalence_verified=true`. Signal computability, screening,
BH, maturity, qualification, admission, proposal, registry, apply, production,
and new-risk fields remain false. In particular, it does not establish the
pinned semantics or PIT availability of the raw fundamental fields; it cannot
replace the separate 37-signal computability proof or a Factor v4 activation
receipt.

### Private exact-37 source-semantic computability

The separate computability lane proves only that all 37 formal-catalog
definitions produce finite observations when the exact A_quant Git objects at
commit `4424dcecc384f614b0e9fd5e36cf094e9244bad5` are interpreted through the
pinned transformations and aligned to the accepted myQuant PIT envelope. It
does not read the A_quant worktree. The 27 previously computed signal
descriptors must remain byte-for-byte equivalent; the ten previously blocked
turnover/fundamental signals are computed independently in two fresh passes.

`scripts/build_factor_v4_1_signal_computability.py` requires explicit absolute
paths and expected SHA-256 values for the execution/worktree baselines, source
receipt and audit, primitive mapping, formal catalog, no-label bundle,
operator-equivalence bundle, PIT membership and manifest, components file,
strict snapshot table, builder, and contract. It reads A_quant bars,
financials, and transformation code only from the pinned Git commit, enforces
the exact bounded calendar deficiencies (50 off-calendar dates, 262 missing
through the maximum observed date, and a 15-session tail), and aborts before
publication on input drift, pass mismatch, resource breach, or worktree drift.

The accepted reference bundle is
`factor_v4_1_signal_computability_20260719T003554Z` below the owner-private,
no-clobber `reports/factor_governance/private/v4_1_signal_computability/` root.
Its receipt, proof, and independent readback byte SHA-256 values are,
respectively:

- `1d595826febeb98078a4c7b28dbecaac641f301e998fd88afe0b7b753430870e`
- `1bdb61112f70f4cf31b9435d8f027ba384e4575ee81168782d64d052749e73ab`
- `3f379493ce5804db4d4708319d389343e650afee83abe9453d0a936a3f375c3b`

Both passes produced result manifest
`6f28bc59305ed2cded79aec7ca9eb363e48682f64909a62e60befcb9e1202c9c`.
The proof preserves 27 predecessor descriptors, computes ten formerly blocked
signals, and records zero observations outside the PIT mask. It deliberately
keeps completeness, same-snapshot, screening, data-quality, producer-lineage,
BH, maturity, admission, registry, apply, production, portfolio, and new-risk
claims false. Its readiness is
`EXPLORATORY_PINNED_SOURCE_SEMANTIC_COMPUTABILITY_ONLY`; it is not a Factor v4
activation receipt and cannot authorize production.

### Future strict-full-A candidate preregistration

`scripts/build_factor_v4_2_candidate_preregistration.py` is an offline,
research-only producer for a genuinely future strict-Parquet `full_a` cutoff.
It rejects `cutoff <= 2026-07-17` and therefore cannot retroactively upgrade
the current 2026-07-17 evidence. The `v4_2` cycle-name segment identifies the
evidence contract; every governance artifact still has
`protocol_version="v4"`.

The `publish` subcommand has no private-root, run-id, or cycle-id override. It
uses the fixed owner-private root
`reports/factor_governance/private/v4_2_candidate_preregistration/` and derives
one directory name from the exact CN/full-A cutoff and snapshot:
`cn_full_a_v4_2_<cutoff YYYYMMDD>_<snapshot_id>`. The root must already exist,
belong to the current user, and have mode `0700`. Publication requires explicit
absolute paths plus expected hashes for the strict snapshot/PIT/table inputs,
the pinned A_quant Git object, myQuant `alpha158.py`, the comparison
ontology/catalog, and every member of the fixed ten-file code-binding set.
Input, code, and the five protected controls are revalidated under the private
bundle publication lock before the Darwin-only
`renameatx_np(RENAME_EXCL)` commit. That lock serializes only publishers using
this private-bundle lane; it does not lock or serialize external CN maintenance.

The atomic directory contains exactly fourteen input artifacts plus one
generated readback report. The report is deliberately
`PRECOMMIT_INTENT_ONLY`; it cannot claim commit, no-clobber, fsync, or durability
success from inside its own staged bytes. PRECOMMITTED and DISCOVERY states are
both persisted, but PRECOMMITTED is lineage-only and DISCOVERY is the sole
final/current state. No external state pointer is mutated.

Successful publication proves the exclusive bundle-directory commit and
canonical readback within the private-bundle filesystem boundary. Precommit
revalidation narrows, but cannot eliminate, the interval before the exclusive
rename: neither it nor the postcommit protected-control/immutable-source
diagnostics proves that external CN source bytes were unchanged at the exact
rename instant. Those postcommit fields are diagnostic and do not claim
cross-system maintenance serialization.

Use the explicit historical readback command with the expected report byte and
semantic hashes:

```bash
./.venv/bin/python scripts/build_factor_v4_2_candidate_preregistration.py \
  readback \
  --bundle-path <absolute-owner-private-bundle-path> \
  --expected-readback-report-byte-sha256 <sha256> \
  --expected-readback-report-semantic-sha256 <sha256>
```

Historical readback verifies the raw pointer and component bytes embedded in
the bundle without consulting their current live files. It reopens only the
recorded immutable snapshot manifest, PIT generation manifest/membership, and
table inventory. Serving remains manifest-only with `was_scanned=false`.
Neither subcommand has registry, proposal, WAL, budget, apply, replay,
transaction, provider, portfolio, broker, order, or trade authority.

### Pinned A_quant definition-only v4.3 preregistration

`scripts/build_factor_v4_3_candidate_preregistration.py` records an independent
six-candidate discovery slice sourced from the pinned A_quant Git commit
`4424dcecc384f614b0e9fd5e36cf094e9244bad5`. A_quant contributes only frozen
source definitions and economic intuition: myQuant does not inherit A_quant
measurements, maturity, FDR results, weights, review decisions, admission state,
or production authority. The six identities are
`event_guidance_revision_90d`, `event_earnings_drift_60d`,
`fund_roe_delta_annual`, `pv_small_float_cap`, `value_book_to_price`, and
`industry_relative_momentum_20d`.

This slice is separate from, and cannot amend, the exact-four v4.2 bundle. It
uses the fixed owner-private root
`reports/factor_governance/private/v4_3_candidate_preregistration/` and the
fixed cycle directory
`cn_full_a_v4_3_20260717_20260717T172132Z`. The root must be pre-created by the
current owner with mode `0700`; generated regular files are mode `0600`, are
published exactly once, and are never overwritten. The bundle binds the exact
A_quant Git objects, the strict 2026-07-17 full-A snapshot/PIT/table/serving
sources, comparison catalogs, all six frozen v4.2 contract files, code files,
and protected controls by canonical byte and semantic hashes. The v4.2
six-file runtime lock is observed before publication and revalidated during
both the locked precommit check and the final race-window check; historical
readback uses its embedded cycle-root evidence instead of scanning live files.

Because the source consultation and candidate selection occurred after the
fixed market cutoff, the artifact records
`selection_independence=UNPROVEN`,
`publication_time_authority=LOCAL_UNVERIFIED`, and
`measurement_authorized=false`. The publication day is excluded; the first 30
strictly later CN open sessions are embargoed, and measurement may begin only
on the 31st later open session after independent publication-time evidence is
available. The 240-open-session and 12-month-end maturity thresholds are policy
requirements only and are not claimed by preregistration.

All six candidates have `initial_weight=0` and a readiness state of
`PROSPECTIVE_PREREGISTRATION_ONLY / measurement_not_run`. Fixed fail-closed
blockers include missing original guidance `p_change` bounds, unproved
book-to-price PIT equivalence for the `1/pb` proxy, unproved ROE report-type
semantics, unproved earnings availability-date equivalence, and insufficient
historical PIT industry generations. No Gate 1-8, maturity, family-BH,
walk-forward, cost, neutralization, stability, redundancy, replay, Codex review,
admission, or production claim is created.

Historical verification requires the explicit absolute bundle path plus the
expected readback report byte and semantic hashes:

```bash
./.venv/bin/python scripts/build_factor_v4_3_candidate_preregistration.py \
  readback \
  --bundle-path <absolute-owner-private-bundle-path> \
  --expected-readback-report-byte-sha256 <sha256> \
  --expected-readback-report-semantic-sha256 <sha256>
```

The v4.3 preregistration lane is research/report-only and has no registry,
proposal, WAL, receipt, pointer, budget, apply, replay, transaction, provider,
portfolio, broker, order, or trade authority.

### Private v4.3 prior-diagnostic fifth-factor nomination

The separate owner-private run
`cn_full_a_v4_3_prior_nomination_20260717_20260717T172132Z` records the
historical selection of `pv_low_vol_of_vol_20d` from the pinned
`VOL_OF_VOL_20D` implementation. Its canonical definition identity is
`eb401bc44af71069b87eee44a3c4bb5ba73abe5337dc38a9ab1ac9e6b4bb261a`,
in family `volatility_of_volatility` and slot
`primitive:volatility_of_volatility`.

This is explicitly an outcome-informed nomination, not prospective formal
evidence. The historical IC, p-value, Bonferroni result, coverage, and winner
ranking may explain why the definition was nominated, but they cannot be
inherited as screening, family-BH, maturity, qualification, admission, health,
or activation evidence. The three accepted file byte hashes are:

- runtime binding:
  `3f1ea68884f49c42b9641070e1acf23c2501e50ecb3f9a0c2434c75ca64d2471`;
- nomination:
  `5a567fb7b462259196c4c920bc78dd7c31d02b2b63bd4c14a05d29f92e25db29`;
- independent readback:
  `ddff8247913af13d186acb53ff3e72b1197d5c76fc95b4564a409f8dd1575f98`.

Historical readback accepts only those canonical owner-private bytes and does
not consult current mutable market, registry, pointer, or diagnostic inputs.
Every authority and side-effect field remains false.

### Future five-candidate v4.4 preregistration

Evidence contract v4.4 preserves `protocol_version="v4"` and leaves both
existing v4.3 lanes unchanged. It expands the exact-four v4.2 prospective set
with the outcome-informed fifth definition above. The five candidates have
zero initial weights and five distinct names, identities, families, and slots:

- `alpha_range_position_momentum_20d`;
- `pv_low_overnight_gap_20d`;
- `pv_low_vol_ratio_10_60`;
- `pv_price_volume_consistency_20d`;
- `pv_low_vol_of_vol_20d`.

`scripts/build_factor_v4_4_candidate_preregistration.py` rejects any cutoff on
or before 2026-07-19 before platform probing, root inspection, or source
collection. A valid future cycle is named
`cn_full_a_v4_4_<cutoff YYYYMMDD>_<snapshot_id>`. The real fixed private root
`reports/factor_governance/private/v4_4_candidate_preregistration/` must remain
absent until a genuinely later strict-full-A cutoff is available and a separate
operator publication action is intended; implementation tests publish only to
isolated temporary roots.

The bundle inventory is exactly 26 input files plus one generated readback:
fourteen byte-preserved v4.2 predecessor artifacts, all three byte-preserved
v4.3 prior-diagnostic artifacts, and nine current v4.4/cycle-state artifacts.
The embedded v4.2 graph is sealed evidence, not a previously published bundle
or a cross-cycle state predecessor. v4.4 creates its own local genesis,
PRECOMMITTED, and DISCOVERY chain.

The expanded selection records
`outcome_informed_selection=true`,
`external_label_independence=false`, and
`prior_statistics_inherited_as_formal_evidence=false`. Runtime equivalence,
computability, measurement, IC statistics, family BH, maturity, walk-forward,
cost, neutralization, stability, structural/formal/correlation dedup, replay,
and transaction planning all remain `not_run`. Its status is only
`PROSPECTIVE_PREREGISTRATION_ONLY / measurement_not_run`; it does not report
five healthy factors.

Publication requires exact expected hashes for fifteen code files and five
protected controls. Git object reads discard all inherited `GIT_*` variables
and use deterministic disabled config paths. Inputs are collected twice around
the private-bundle lock and must match byte-for-byte before the Darwin-only
exclusive rename. The lock serializes only publishers using this private lane;
precommit revalidation and postcommit diagnostics do not prove external CN
maintenance bytes were unchanged at the exact rename instant. Historical
readback does not consult the live pointer, protected controls, or mutable
diagnostic sources.

All v4.4 health, qualification, admission, registry, proposal, apply,
activation, and new-risk authority fields remain false. The lane has no
registry, WAL, receipt, pointer, provider, network, portfolio, broker, order,
or trade side effects.

The remaining validation/planning scripts require explicit input and output
paths and write only the requested research artifact as mode `0600`:

```bash
./.venv/bin/python scripts/build_factor_v4_readiness_plan.py \
  --input-json <readiness-input.json> \
  --quality-records-json <quality-input.json> \
  --output-json <research-readiness.json>

./.venv/bin/python scripts/build_factor_v4_replay_validation.py \
  --kind replay \
  --input-json <canonical-replay.v4.json> \
  --output-json <replay-validation.json>

./.venv/bin/python scripts/build_factor_v4_transaction_plan.py \
  --input-json <transaction-input.json> \
  --output-json <research-transaction-plan.json>
```

These commands never accept a registry path or activation pointer and do not
have an apply flag.

`--quality-records-json` is optional. Its file is an exact JSON object with
`schema_version="factor-quality-input.v1"`, `protocol_version="v4"`, a
`quality_records` array, and `expected_quality_set_sha256` as a lowercase hash
or `null`. A missing or malformed quality file produces an informational
`invalid` quality assessment and a deterministic stderr message. The command
still exits according to production `factor_governance_ready`, never according
to quality status.
