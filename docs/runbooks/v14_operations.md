# myQuant v14 Operations

This is the current local operating contract for the v14 DAG on `main`.
The repository runs offline by default and keeps deterministic data, risk, and
production-authorization gates authoritative.

## Active architecture

- Formal branches: `quant`, `fundamental`, and `macro`.
- Likelihood inputs: `quant` and `fundamental`.
- Macro v2 remains an observer and control-context input; it is not a Bayesian
  stock-selection likelihood.
- Intelligence is retired from the active DAG and current catalog. Historical
  Intelligence marts, tags, reports, and replay evidence remain immutable.
- The v13 incubation/freeze and freeze-exception merge protocol are retired.
  They do not constrain current `main` or schedules.

## Independent safety boundaries

Retiring the v13 freeze does not relax these controls:

- Canonical CN data must pass `quant-investor market storage-validate --market CN`.
- Branch readiness and data governance fail closed when v14 provenance or
  generation evidence is missing.
- Current full-A Quant readiness is bound to the complete canonical coverage
  scope and its component hash. Evidence-backed non-trading absences are
  excluded; historical serving files are never added to the current scope.
  Coverage, classification, evidence, or PIT hash drift returns a structured
  blocked report.
- RiskGuard and deterministic control-chain vetoes remain authoritative.
- Quant FactorGovernanceProtocol v2 stays report-only until its independent
  production-apply authorization is implemented; PR4 stops at
  `forward_factor_apply_not_authorized_pr4`.
- No schedule may auto-checkout, cherry-pick, merge, call a broker, or create
  orders/trades.
- Legacy Macro catalog entries are diagnostic only and cannot claim v14
  production readiness.
- A missing, stale, or unverifiable canonical Macro generation blocks branch
  fusion before MacroAgent, Markov, RiskGuard, and the investable funnel. The
  blocked path emits a neutral VETO diagnostic, keeps the 0.55/0.50 baseline,
  preserves existing holdings, and authorizes no new decision or order.

## Offline preflight

Run from the repository root with the repository virtual environment:

```bash
./.venv/bin/quant-investor market storage-validate --market CN
./.venv/bin/quant-investor market data-governance --market CN
PYTHON=./.venv/bin/python scripts/staged_upgrade_quality_gate.sh
```

For operator-facing commands, verify the exact CLI help before changing a
schedule. Missing source data, provenance, receipts, or approvals is a blocker;
do not substitute CSV, stale snapshots, inferred values, or hand-written pass
flags.

## Fundamental authoritative rebuild

An authoritative full-A Fundamental refresh is an explicit two-stage operation:

1. `market fundamental-maintain --allow-live --authoritative-full-rebuild`
   fetches into an isolated data root and a separate v3 checkpoint root. It
   binds the exact full-A scope, market pointer, PIT membership, request
   outcomes, financial-period coverage, canonical market-bar file set and
   per-symbol first/last trade bounds, and Parquet readback fingerprints.
2. `market fundamental-promote --expected-pointer-sha256 <sha>` independently
   revalidates the staged generation and advances the canonical Fundamental
   pointer with compare-and-swap semantics.

Financial completeness is measured only against matured post-listing quarters;
an observed pre-listing period in one endpoint does not expand another
endpoint's denominator, and a quarter whose reporting lag ends after the bound
eligibility end is not mature. Daily history is intersected with the exact
canonical bar bounds; the authoritative boundary tolerance is 62 days while
monthly coverage still requires at least 90% and no run of more than two
missing months. The bar file-set SHA, bounds SHA, and policy are replayed during
promotion.

The live rebuild must use a new run/checkpoint root after any v2 checkpoint,
scope drift, audit-policy or evidence-binding change, malformed response, or
failed request. Do not edit or weaken an older checkpoint binding to make it
resume. Promotion rejects legacy primary provenance, checkpoint or canonical
bar drift, incomplete endpoint coverage, or a derived mart that cannot be
reproduced from the accepted raw checkpoint and the exact bound PIT membership.
Weekly readiness schedules remain local and read-only; they must never add
`--allow-live` or perform promotion.

## Macro authoritative refresh

Macro primary refresh is a separate, explicit live operation. First record the
exact SHA-256 of `data/parquet/cn/_catalog.json` and `_latest.json`, verify that
the requested `as_of` is the latest complete session, then run:

```bash
./.venv/bin/quant-investor market macro-refresh \
  --market CN \
  --as-of <latest_complete_trade_date> \
  --data-root data/parquet/cn/macro_daily \
  --run-id <new_generation_id> \
  --expected-catalog-sha256 <catalog_sha256> \
  --expected-market-pointer-sha256 <market_pointer_sha256> \
  --allow-live
```

The publisher binds five raw Tushare endpoints, the exact canonical bar
pointer and input partitions, the deterministic transform, and the generated
Parquet. It publishes through the shared catalog-writer lock, two-object CAS,
and a recoverable transaction journal. Endpoint I/O completion must remain in
the 72-hour capture window, and every selected month must meet its declared
`month_end + max_release_lag_days` lower bound. The formula cross-section uses
only symbols whose terminal bar is the requested session. Before switch and
during recovery, the publisher verifies the complete strict-catalog required
table closure, the exact market-pointer SHA, and every generation/input hash;
orphaned partial transactions may be retried only after deterministic cleanup.
Never hand-edit a generation or replay it historically;
`historical_replay_eligible=false` is permanent. A schedule may report this
command as a manual remediation step but must never run it.

The one-time post-cutover event-score schema migration is
`scripts/retire_event_score_catalog_residual.py`. It requires the exact current
catalog SHA, market-pointer SHA, legacy event-score Parquet SHA, a new run id,
`--apply`, and `RETIRE_EVENT_SCORE_SCHEMA_V14`. It preserves the legacy source
bytes, publishes an immutable filtered generation, and commits only after a
fresh strict reader proves that `intelligence_score` is no longer visible.
Do not remove only the catalog column declaration or rewrite the legacy source
file in place.

## Schedule routing

Current schedules must name the three active branches, omit Intelligence from
active DAG work, and point to this runbook. Weekly Quant governance may report
proposals but must not mutate the production registry or weights. Fundamental
and Macro jobs must preserve their own provenance and activation gates.

Schedule edits are complete only after TOML parse, invariant readback, and
entrypoint/help validation. Unrelated schedule metadata must remain unchanged.
