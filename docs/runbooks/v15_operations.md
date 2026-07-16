# myQuant v15 Operations

This is the current operating contract for the no-Theme v15 mainline.  The
system remains offline by default and fail closed for new risk.

## Active architecture

- Canonical branches: `quant`, `fundamental`, `macro`.
- Bayesian likelihoods: `quant`, `fundamental`; Macro is control context.
- Deterministic path: `DeterministicFunnel -> branch context -> Bayesian ->
  RiskGuard -> ICCoordinator -> PortfolioConstructor`.
- Theme packages, agents, environment variables, overlays, reports and current
  artifacts are retired.  Historical mixed strategy records remain immutable.
- Current analysis artifacts are written under `results/v15/` and must carry
  the exact v15 schema family.  A v14 artifact cannot be relabelled as v15.

## Authorization boundary

`v15_run_readiness` is the only current authorization summary.  New risk is
authorized only when strict market data, all three branches, Factor governance,
PortfolioConstructor and an unexpired hash-bound human receipt pass together.
Scheduled runs do not supply that receipt, so `new_risk_authorized=false` is
the default.

Risk reduction is a separate sell-only advisory/manual lane.  It requires an
existing holding, a same-day fresh quote and a same-day human action.  It never
authorizes a BUY, broker call or real order.

## Data commands

- Market maintenance remains `market maintain`; `market run` never refreshes
  data.
- Fundamental authoritative rebuild remains the existing two-stage
  `fundamental-maintain --allow-live --authoritative-full-rebuild` followed by
  `fundamental-promote --expected-pointer-sha256` workflow.
- Macro authoritative publication remains `market macro-refresh` with
  `--allow-live`, issuer-bound NBS URL, exact catalog SHA and exact market
  pointer SHA.  `macro-maintain` is observer/staging only; there is no
  `macro-promote` command.
- Factor bootstrap generation is plan-only.  It cannot write the registry,
  release a kill switch or create an activation receipt.

Live Fundamental rebuild/promotion, live Macro refresh, Factor bootstrap
apply, Macro activation and private Theme evidence purge each require a
separate explicit authorization.

## Local gates

```bash
./.venv/bin/quant-investor market storage-validate --market CN
./.venv/bin/quant-investor market data-governance --market CN
PYTHON=./.venv/bin/python scripts/staged_upgrade_quality_gate.sh
./.venv/bin/python -m pytest tests/unit/test_v15_no_theme_contract.py -q
./.venv/bin/python scripts/check_cn_dashboard_export.py
```

The first provider-hard-off v15 smoke is expected to prove the v15 contract,
absence of Theme output and `new_risk_authorized=false`; long-duration Factor
or Macro blockers are expected and must remain visible.
