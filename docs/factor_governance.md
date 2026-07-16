# FactorGovernanceProtocol v3

v3 governs evidence, readiness, and mutation safety. It does not authorize a
factor identity or production write by itself.

## Canonical replay

The chain is fixed and exact:

`quant -> deterministic_funnel -> bayesian -> risk_guard -> ic_coordinator -> portfolio_constructor`

For every A/B/C/D arm, the replay binds the Quant-scored universe,
Funnel-eligible subset, stage output symbols, predecessor byte SHA,
predecessor semantic SHA, and current semantic SHA. The Quant stage also binds
the `quant/fundamental/macro` readiness and object/semantic hashes. The IC stage
readbacks exact `branch_verdicts/risk_decision/ic_hints` inputs and exact
decision outputs by per-symbol semantic SHA; PortfolioConstructor must bind
those IC output hashes. Positive weights require both RiskGuard approval and an
IC `buy` action.

The evidence envelope names one explicit absolute replay path and its exact
canonical-file SHA. Readback rejects symlinks, unsafe modes/link counts,
noncanonical JSON bytes, identity drift, context drift, and any IC input/output
hash mismatch. A successful local readback does not by itself authenticate the
producer or authorize an apply.

Runtime accepts only v3 metadata and evidence. Historical v2 payloads are
rejected and are never automatically upgraded.

## Readiness

`governance_runtime_status()` requires:

- protocol v3 and the exact local protocol hash;
- a strictly loaded registry and selectable production factors;
- exact `registry_file_sha256` and `production_factor_set_sha256` identities;
- unique family/slot ownership and valid runtime contracts;
- normalized absolute weight at most 20% per factor and 35% per family;
- PIT/calendar/evidence hashes and canonical replay readback;
- a valid production activation receipt and released kill switch.

Failure is `governance_blocked`, confidence zero, with no historical or legacy
proxy fallback.

## Baseline bootstrap

`factor-baseline-bootstrap-plan.v1` is staging-only. The plan requires at least
five factors across at least three families, unique slots, normalized weights,
runtime/PIT/calendar contracts, evidence hashes, expected registry CAS, WAL,
and an inverse rollback manifest. Its API and CLI have no apply option.

Bootstrap apply, registry mutation, production activation receipt, and
kill-switch release each require a separate explicit Maxwell authorization.
Until then the tracked registry remains byte-identical. Normal one-for-one
month-end transition resumes only after the baseline exists.

## Maturity and diagnostics

Maturity requires either 12 distinct month-end RankIC periods or 8
non-overlapping 30-day forward cohorts. Family multiple testing uses
Benjamini-Hochberg at `q <= 0.10`. A 90-day check is diagnostic only and cannot
authorize production alone.

Weekly mining, health, historical comparison, and shadow selection remain
report-only. They cannot directly promote, deprecate, reconcile, or mutate the
registry.

## Verification

```bash
pytest tests/unit/test_factor_governance_v3.py -v
pytest tests/unit/test_factor_runtime_contract_v2.py -v
python scripts/build_factor_baseline_bootstrap_plan.py --help
```

The second command retains historical coverage and must demonstrate that v3
runtime rejects v2 fixtures.
