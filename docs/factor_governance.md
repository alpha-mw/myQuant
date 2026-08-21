# Stable Factor Governance

Factor governance is one stable, prospective research lifecycle. It keeps
bootstrap definitions, preregistration, source-bound signal capture, matured
observations, evaluation, admission, contextual validation, readiness, and
downstream stock selection as different states with different authority.

The canonical Python package is:

```python
quant_investor.factors.governance
```

The canonical operator read is:

```bash
quant-investor factor status --help
```

`factor status` is the prospective validation-status builder: it consumes one
exact canonical request plus its SHA-256 and grants no production authority.
It is intentionally different from the isolated production readers:

```bash
quant-investor factor production-status --workspace-root <root>
quant-investor factor production-verify --workspace-root <root>
quant-investor factor production-signal --workspace-root <root> --factor-id <LOW-or-W80>
```

Production reads resolve only the Factor pointer, permanent marker, immutable
Factor generation, and sealed signals. They report
`authority_domain=FACTOR_PRODUCTION_ONLY`, never evaluate System runtime state,
and grant neither System nor trading authority. The sole first-activation
operator is documented in `docs/runbooks/factor_production.md`.

After daily rollover, `factor production-observe` may append exact-once OPEN
`factor.production_observation` records for the sealed LOW/W80 head. These
records bind the Factor generation/pointer, signal/symbol set, Market/PIT and
Calendar closure and are non-authorizing. They are deliberately distinct from
the mature `factor.prospective_observation` used by the 390-session admission
lifecycle. OPEN production observations prove preregistration only; future
outcomes, IC, RankIC, maturity and admission remain unavailable until their
separate exact gates close.

## Lifecycle

```text
installed release, components, and Factor validator manifest
  -> bootstrap exception closure OR prospective preregistration
  -> exact source-decode attestations and signal captures
  -> first-session configuration selection
  -> matured, predecessor-linked observations
  -> non-authorizing composite-state and custody chain
  -> replayable execution and turnover evidence
  -> prospective evaluation and admitted factor set
  -> intrinsic, non-authorizing Factor validation receipt
  -> System-run contextual replay of the complete raw-source closure
  -> protected System validation attestation and completion
  -> read-only Factor status
  -> independent Intelligence readiness and candidate generation
  -> separate System activation authority
```

The stable package exposes these responsibility-specific APIs:

| Stage | Stable API | Authority boundary |
|---|---|---|
| Bootstrap definitions | `bootstrap_factor_definitions` | Creates deterministic research definitions, not admission |
| Bootstrap evidence | `build_bootstrap_exception_evidence` / `validate_bootstrap_exception_evidence` | Binds the approved bootstrap exception source and implementation closure without claiming readiness or selectability |
| Bootstrap set | `build_bootstrap_factor_set` / `validate_bootstrap_factor_set` | Establishes the starting research set, not prospective proof |
| Bootstrap signals | `compute_bootstrap_signals` | Recomputes the exact bootstrap formulas from strict Parquet inputs |
| Trusted Bootstrap closure | `FactorValidationStore.initialize_bootstrap` | Resolves and stores the exact release plus seven-source exception closure; it does not create READY status or activate anything |
| Installed validator manifest | `FactorValidationStore.build_validator_manifest` / `validate_validator_manifest` | Binds the finite installed implementations, components, release, and compiled Factor contracts |
| Preregistration | `FactorValidationStore.mine` / `validate_preregistration` | Derives candidates and 390 planned sessions from exact stored sources and the store clock; callers supply neither values nor time |
| Selection and signal capture | `FactorValidationStore.observe_signal` / `validate_configuration_selection` / `validate_signal_capture` | Atomically selects at ordinal zero, then captures the immutable selected configuration from strict source objects |
| Matured observation | `FactorValidationStore.observe_label` / `validate_observation` | Reopens the exact capture sources and t+1/t+30 label object to append one predecessor-linked observation |
| Candidate custody | `validate_custody_record` / `validate_composite_state` | Intrinsically validates the non-authorizing composite candidate state; mutation is only through the trusted store and System candidate-state CAS |
| Audit observation append | `append_observation_cas` / `read_observation_head` / `validate_observation_head` | Maintains a separate non-authorizing audit head; it is not the production prospective-state authority |
| Final prospective stages | `FactorValidationStore.evaluate(action=...)` plus the execution, evaluation, admission, and receipt validators | Executes only `FINALIZE_EXECUTION`, `EVALUATE_PREREGISTRATION`, `BUILD_ADMITTED_SET`, or `BUILD_INTRINSIC_RECEIPT` in DAG order |
| Contextual validation | `SystemStore.build_validation_run_request` / `SystemStore.run_validation` / `validate_contextual_result` | The fixed System profile reopens and replays every bound source before protected custody is committed; caller callbacks and result mappings are forbidden |
| Status | `FactorValidationStore.build_status` / `validate_factor_status` | Builds an inert projection only after exact receipt, context, and protected System attestation readback |

Statistics used by the lifecycle live in
`quant_investor.factors.governance.statistics`. Callers must not inject claimed
p-values, maturity counts, deduplication booleans, or health states where the
stable runtime can recompute them.

Stable artifact kinds are `factor.bootstrap_exception_evidence`,
`factor.bootstrap_set`, `factor.validator_manifest`,
`factor.source_decode_attestation`, `factor.preregistration`,
`factor.configuration_selection`, `factor.signal_capture`,
`factor.prospective_observation`, `factor.custody_record`,
`factor.composite_state`, `factor.execution_turnover_evidence`,
`factor.prospective_evaluation`, `factor.admitted_set`,
`factor.validation_receipt`, `factor.contextual_validation_result`, and
`factor.status`. `factor.observation_head` remains an audit-only projection.
The protected `system.validation_attestation` and its exact-once custody record
are System artifacts. None of these payloads carries a runtime or protocol
selector, and none authorizes activation by itself.

The bootstrap set is exactly the equal-weight combination of
`pv_low_dollar_volume_5d` and
`pv_blend_volstab19x2_mom90_amihud5_w80`. The nearby W75 configuration is
control-only, nonselectable, and zero weight. Bootstrap uses strict `amount`,
`adj_close`, and `vol` Parquet inputs; aliases and reconstructed substitutes are
rejected.

## Required separation

- A factor idea is not a bootstrap definition.
- A bootstrap definition is not a preregistered prospective factor.
- A completed historical backtest is not forward evidence.
- A caller-supplied coverage or turnover scalar is not evidence.
- A mature prospective factor is not admitted until every required contract
  and set-level gate passes.
- An admitted factor is not a stock candidate, portfolio position, order, or
  trade.
- A read-only health result is not publication or activation authority.

Unknown or incompatible evidence fails closed. `data_blocked` identifies a data
problem rather than an alpha failure, but it still prevents a healthy/readiness
claim until exact fresh evidence closes the gap. Missing observations cannot be
filled with historical, synthetic, cached, or latest-by-mtime substitutes.

Initial configuration completeness and daily signal coverage are separate
gates. The strict PIT-universe source object itself is the exact full-A
denominator; there is no eligibility or complete-case filter. The first signal
session selects a primary or its one preregistered alternate using finite signal
and required raw inputs plus PIT industry, positive total market value, and
tradability. Daily coverage is only finite selected signal divided by that same
full PIT row count. Missing industry or market value may remove a row from
neutralization and RankIC, but never from the daily coverage denominator. Once
selected, a configuration cannot switch to an alternate or splice another
configuration into the chain.

## Status contract

The top-level status payload contains exactly `status_id`, `active`, `observed`,
`readiness`, `blockers`, and `activation_mutation_authorized`. Readiness is
`READY` or `BLOCKED`; it is never inferred from a display label.

The active and observed lanes remain independent:

- `active.state` is `ACTIVE` or `ABSENT`;
- an active lane is `BOOTSTRAP`, `PROSPECTIVE`, or `NONE`;
- an active route is `BOOTSTRAP_EXCEPTION`, `PROSPECTIVE_ADMISSION`, or `NONE`;
- `active` contains exactly `state`, `lane`, `admission_route`,
  `producer_identity`, `factor_set_ref`, `factor_ids`,
  `validation_receipt_ref`, `contextual_result_ref`, and
  `validation_attestation_ref`;
- `observed` contains exactly `composite_state_ref`, `cycle_state`, `terminal`,
  and `blockers`; a null composite is the exact `NOT_STARTED` projection.

Observed artifact references use the exact five-field reference contract. A
prospective admission does not rewrite the active lane, and an active bootstrap
set does not claim that prospective observation is mature. Status remains
read-only, and `activation_mutation_authorized` is always false.

An intrinsic receipt binds the sealed policy, evidence, and active-set closure,
but its intrinsic validator cannot reopen raw source bytes. The fixed
contextual callback performs that full replay, and System then binds the result
to an exact validation request, installed code/components, source snapshot,
protected attestation, and completion record. `FactorValidationStore.build_status`
requires exact receipt, contextual result, active set, and protected System
attestation agreement before projecting `READY`; intrinsic-only or mismatched
input is blocked. `validate_factor_status` validates only the sealed projection.
The System generation compare-and-swap remains the only **System** active-pointer
authority. The separate `results/factors/_active.json` pointer grants only
`FACTOR_PRODUCTION_ONLY` authority and can be created only by
`quant-investor factor production-activate --expected-empty`; it never grants
System, Mainline, Investment, portfolio, Strategy Record, or trading authority.
After that immutable genesis cutover, `quant-investor factor
production-rollover` is the sole supported successor writer. It requires an
exact current-pointer preimage and an execute-mode PIT/Market/History closure,
preserves the permanent marker, and archives every predecessor pointer.

Invalid stable factor input raises `FactorGovernanceError` with exit code 2,
default public code `FACTOR_VALIDATION_FAILED`, and no path-bearing public
fields. The error never authorizes fallback or mutation.

## Admission evidence and non-gates

The stable Factor admission policy is deliberately narrow and deterministic.
Maturity is the conjunction of at least 300 valid daily RankIC sessions, 12
closed calendar month-end observations, and 8 disjoint cohorts defined by the
canonical 30-open-session ordinals. Admission then requires a t-statistic
strictly greater than 3, DSR at least 0.95, complete 10-block PBO with all 252
splits and PBO at most 0.50, within-family BH q-value at most 0.10, all 45 CPCV
paths with positive-path ratio at least 0.55, and replayed turnover at most 12
after annualization to 252 canonical open sessions. The execution evidence
binds a complete round-trip cost of 1bp as 0.5bp per unit of absolute weight
change, including initial entry and terminal exit; sparse rows mean every
unlisted universe weight is exactly zero.

Trial ICIR dispersion is the sample standard deviation (`ddof=1`) across every
executed configuration. A missing or non-finite ICIR, an incomplete 10-block
column, or fewer than two selected configurations fails closed before
admission. Redundancy clusters use transitive union-find components. Their
representative ordering is DSR descending, mean purged-OOS CPCV RankIC
descending, then ASCII configuration ID. The weight score is
`max(0, mean_path_ic) * path_count / (path_count + 10)` over all 45 paths. At
most ten eligible representatives survive the same deterministic ordering;
largest-remainder allocation then emits 12-decimal weights summing exactly to
one.

Capacity, drawdown, runtime compatibility, crowding, and current production
health are not Factor admission gates in this policy. They may be evaluated by
downstream System, portfolio, execution, or health controls, but Factor does
not invent or imply admission thresholds for them.

## Operator workflow

1. Run `quant-investor system verify` against the intended checkout.
2. Verify strict CN Parquet/PIT inputs needed by the requested factor judgment.
3. Prepare a canonical request containing exactly
   `active_factor_set_ref`, `active_validation_receipt_ref`,
   `active_contextual_result_ref`, `active_validation_attestation_ref`, and
   `observed_composite_state_ref`, then bind its SHA-256.
4. Run `quant-investor factor status --request <path>
   --expected-request-sha256 <sha256>`; there is no implicit active-read
   fallback.
5. Report the exact status payload, active lane and route, observed state,
   readiness, blockers, factor evidence, and protected-state hashes.
6. Route data gaps to the maintenance lane. Route new research to a separate
   preregistration; do not alter the current set from a health review.

Retired factor-state surfaces must be absent from the stable runtime. Their
presence is a cutover blocker, not state to protect or reuse. Required legacy
summaries belong only in the cutover migration archive, never in active runtime
documentation.

See the [unified cutover guide](migrations/unified-cutover/README.md) for
removed-to-stable mappings and replacement tests.
