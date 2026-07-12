# myQuant v13.1 Freeze-Exception Architecture Review

## Goal

Deliver a trustworthy local Dashboard 2.0, a PIT-safe technology-oriented Theme
2.0 observer/formal gate, and FactorGovernanceProtocol v2 without changing the
four canonical v13 branches or introducing a broker/execution path.

## Scope

- Local static dashboard export, validation, loading, and presentation.
- Theme taxonomy, multi-membership, evidence, lifecycle, mandate, PE/VC prior,
  observer/formal lanes, and explicit rollback.
- Factor evidence, slot governance, transition planning, atomic targeted
  registry mutation, and inverse rollback.
- Offline chronological replay, activation gates, and freeze-exception audit.

## Non-goals

- Web workspace, SSE, Web API, authentication, or a new frontend framework.
- Live data-provider, LLM, Notion API, broker, or order execution.
- Blindly restoring the historical 14-factor registry or accepting the current
  one-factor state as methodology evidence.
- Adding Theme as a fifth canonical branch or weakening RiskGuard and
  PortfolioConstructor authority.

## Architect pass

### Authority boundaries

The governed path remains:

`strict PIT data -> deterministic funnel -> four branches -> Bayesian -> RiskGuard -> PortfolioConstructor`

Theme is a structured eligibility and constraint overlay. PE/VC knowledge may
only alter rank inside an already eligible set. Factor governance may only
mutate its targeted slot through a versioned plan whose protocol, evidence,
data, code, and registry hashes are bound before apply.

Dashboard data is a downstream review projection. It never becomes a strategy,
registry, ledger, or execution source of truth. The latest valid
`manual_execution_manifest.json` selects a contained `next_ledger_path` in
`ledger_after_manual_switch.{csv,parquet}`; that selected ledger remains the
account/execution source.

### State transitions

- Theme defaults to observer. Formal eligibility has no forced-admission path
  and can be killed back to observer-only independently.
- Factor mining and health default to report-only. Apply is a separate command
  path and monthly mutation budget is enforced by persisted evidence, not by
  scheduler convention.
- Dashboard replaces one complete private snapshot at a time. Unknown values
  remain null and downstream UI cannot silently manufacture zeros.
- Joint replay uses one dataset SHA across all five scenarios. Thresholds are
  sealed at validation end and cannot change after holdout opens.

### Failure semantics

- Theme evidence or PIT failure: formal pool is empty; observer remains usable.
- Factor evidence/protocol failure or zero production factors: Quant confidence
  is `governance_blocked`; no legacy proxy is selected.
- Dashboard semantic/reconciliation failure: the affected metric is blocked
  and a visible blocker is emitted.
- A failed Theme, Factor, Dashboard, or DAG acceptance gate cannot enable its
  production switch. Other independently validated read-only surfaces may
  remain available.

## Critic pass

### Risks found and required controls

1. A display-only metadata cap can become a machine eligibility cap. Machine
   membership maps must be complete; display sampling happens only after formal
   decisions and must not be read by the candidate pool.
2. Forced Theme admission hides model disagreement. Natural zero must remain a
   valid empty formal result; observation/watch lanes provide visibility
   without relaxing eligibility.
3. A positive PE/VC thesis can become an authority bypass. The prior is capped
   at ten percentile ranks after eligibility and cannot offset crowding,
   valuation, stale evidence, edge, RiskGuard, or portfolio constraints.
4. A mining run can accidentally redefine the entire production registry. v2
   transitions must target exactly one slot, leave unrelated records unchanged,
   and enforce at most one applied month-end swap.
5. Repeated executions can fake persistence or factor failure streaks. Both
   protocols count distinct, chronologically valid evidence windows; same-day
   repeats are idempotent and data-blocked windows do not advance or reset
   failure streaks.
6. Sparse dashboard records can produce plausible but false annualized metrics.
   Annualization is unavailable below 95% open-day coverage or 60 valid daily
   returns; attribution must reconcile to NAV within one basis point plus
   explicitly named cash/fee residuals.
7. A static app can still leak private data or execute stored markup. Production
   snapshots are ignored, uploads are session-only, raw data never enters URL or
   localStorage, and all labels/reasons are rendered as text.
8. A holdout can be tuned after inspection. The threshold seal and dataset SHA
   are checked whenever holdout is marked open; a caller-supplied expected hash
   is mandatory.
9. Factor automatic transition and Theme formal activation have different
   readiness clocks. Their switches are independent: Factor may activate only
   after its protocol and offline DAG gates pass, while Theme additionally
   requires 20 distinct live-shadow trading days. The joint path waits for all.
10. The requested change exceeds the frozen strategy's ordinary bug-fix lane.
    Work therefore stays isolated on a `freeze-exception` branch; tests and
    Codex review can approve the PR, but merge still requires Maxwell's explicit
    confirmation.
11. A draft can be approved today with a historical `available_at` and leak PE/VC
    knowledge into an old replay. Canonical availability is therefore the later
    of source availability and approval time, and every read also checks
    `approved_at <= as_of`.
12. Industry labels are not PIT membership evidence. Theme v2 prequalification
    requires an active `theme_membership.v2` detail; absent trusted symbol-level
    mappings keep formal coverage blocked rather than inventing a seed map.
13. Prequalification happens before Bayesian, RiskGuard, and construction, so it
    cannot truthfully emit a formal pool. A hash-bound post-control
    reconciliation is the only formal producer and records every symbol gate.
14. Unknown crowding or valuation risk is not low risk. Missing/freshness status
    for either independent axis blocks formal prequalification while the
    observation lane remains available.
15. A caller-provided `passed=true`, paired delta, trading-day list, or
    acceptance boolean is not evidence. Scenario and factor-arm artifacts are
    re-read by expected SHA, derived values are recomputed, and metric pass is
    evaluated against the pre-holdout threshold seal.
16. Registry metadata cannot be the monthly mutation budget because inverse
    rollback would refund it. The budget is an independent append-only ledger;
    rollback preserves the reservation and a second same-month swap is blocked.
17. A static dashboard can still compute plausible metrics from sparse calendar
    rows. Formal trading-day masks originate in strict Parquet, every rolling
    window enforces coverage, and attribution covers every valid NAV-return day
    or exposes an explicit residual/blocker.
18. Passing a tactical-cap unit test does not prove the real DAG carries the
    contract. The decision path must forward and enforce the prospective
    prequalified lane before PortfolioConstructor, then recheck count and NAV in
    post-control reconciliation.
19. A content-addressed JSON file is not proof that a canonical replay ran. The
    current joint and Factor replay helpers are report-only normalizers; runtime
    formal Theme activation and forward Factor mutation remain code-level
    blocked until readback-bound canonical producers exist. Caller-controlled
    hashes, permissions, or self-hashes cannot lift those blockers.
20. An approved PE/VC row cannot be accepted solely because its JSON says
    `status=approved`. Every canonical row must match a valid, mode-0600 approval
    ledger event binding the reviewed draft hash, canonical content hash,
    version, theme, and approval date; missing or tampered provenance fails
    closed.
21. The Theme v2 membership main database is separate from legacy concept and
    industry labels. Observer loading may use only the approved PIT v2 store;
    formal eligibility additionally requires a verified store hash, required
    coverage status, and `available_at <= as_of`.
22. Dashboard privacy applies to the complete export bundle, not only the main
    snapshot. Every real JSON, JavaScript, CSV, and export summary is atomically
    written mode 0600 and ignored by Git; JSON/JavaScript payloads must read back
    identically. Public samples remain synthetic.
23. Trading-calendar masking must precede return construction. Filtering a
    return after computing it against a weekend or other non-open observation
    can silently drop or shift P&L, so performance, excess, monthly, drawdown,
    and rolling metrics operate on the compressed open-day series.
24. A run-directory name is not account authority. Dashboard holdings must be
    selected by the latest valid manual-execution manifest, including status,
    recorded/execution time, contained `next_ledger_path`, declared ledger hash,
    and stable readback; a later invalid or no-action directory cannot replace
    that baseline.
25. Legacy eight-gate booleans do not prove Factor Protocol v2 readiness.
    Production Quant runtime must bind the local protocol hash, production-set
    metadata, slot/family identity, risk budgets, and canonical producer/evidence
    state; otherwise it is `governance_blocked`. Explicit shadow scoring may
    remain observable but has zero production confidence.
26. Historical-factor comparison is not reconstructed from the live selectable
    set. The old-14 arm uses a separate self-hashed shadow manifest binding each
    named registry record and weight; it is read-only and cannot change registry
    lifecycle or selection state.
27. A holdout dataset hash cannot define a new upgrade cycle. The fixed v13.1
    freeze-exception cycle has one permanent mode-0600 seal lock and one ledger
    entry, so a second dataset or threshold set is rejected before persistence.
28. Finite-looking metadata cannot make invalid Factor weights safe. Runtime
    rejects NaN, infinity, negligible weight/direction, and an overflowing total
    before normalization; invalid numerics can never bypass the 20%/35% budgets.
29. Cross-sectional return, volatility, and breadth are diagnostics, not a
    fallback Quant branch. The global Quant verdict inherits only a
    governance-aware, production-eligible `quant_result`; blocked or report-only
    results remain score/confidence zero throughout RiskGuard and evidence packs.

## Acceptance and stop condition

The safety mechanisms may be handed off once their focused/public tests,
Dashboard semantic/visual checks, Theme and Factor sweeps, and staged-upgrade
quality gate pass. Production activation is complete only after the rollback
drills and canonical strict-Parquet five-path replay also pass. Until then each
affected surface remains disabled and preserves its blocker in the joint replay
manifest. No merge action is part of this implementation task.
