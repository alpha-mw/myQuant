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
   outcomes, financial-period coverage, and Parquet readback fingerprints.
2. `market fundamental-promote --expected-pointer-sha256 <sha>` independently
   revalidates the staged generation and advances the canonical Fundamental
   pointer with compare-and-swap semantics.

The live rebuild must use a new run/checkpoint root after any v2 checkpoint,
scope drift, malformed response, or failed request. Promotion rejects legacy
primary provenance, checkpoint drift, incomplete financial coverage, or a
derived mart that cannot be reproduced from the accepted raw checkpoint and
the exact bound PIT membership. Weekly readiness schedules remain local and
read-only; they must never add `--allow-live` or perform promotion.

## Schedule routing

Current schedules must name the three active branches, omit Intelligence from
active DAG work, and point to this runbook. Weekly Quant governance may report
proposals but must not mutate the production registry or weights. Fundamental
and Macro jobs must preserve their own provenance and activation gates.

Schedule edits are complete only after TOML parse, invariant readback, and
entrypoint/help validation. Unrelated schedule metadata must remain unchanged.
