# Theme Rotation Testing Matrix

Theme Rotation coverage is organized around default-off safety, deterministic
metadata, fail-safe parsing, optional behavior gates, and production wiring
audits.

## Coverage Areas

| Area | Tests | Contract |
| --- | --- | --- |
| Scanner | `tests/unit/test_theme_scanner.py` | Empty inputs are safe; ranking is deterministic; small themes are filtered; malformed local data does not raise. |
| ThemeAgent | `tests/unit/test_theme_agent.py` | Standalone non-canonical verdicts; neutral without metadata; reads dict and dataclass payloads. |
| Metadata/context | `tests/unit/test_theme_metadata_in_context.py` | Disabled, success, symbol-limit, empty-map, and scanner-error metadata payloads. |
| Bayesian metadata | `tests/unit/test_bayesian_theme_metadata.py` | Per-symbol theme metadata can be extracted and passed through without changing likelihood schema. |
| Reporting | `tests/unit/test_theme_reporting.py` | Theme radar markdown is presentation-only and safe for disabled/error/malformed payloads. |
| Funnel boost | `tests/unit/test_deterministic_funnel_theme_boost.py` | Default disabled no-op; enabled boost is capped and deterministic. |
| Boost diagnostics | `tests/unit/test_theme_boost_diagnostics.py` | Offline baseline-vs-boosted comparison reports entered/dropped candidates and deltas. |
| RiskGuard overlay | `tests/unit/test_theme_risk_constraints.py`, `tests/unit/test_risk_guard_theme.py` | Disabled no-op; enabled constraints tighten action, gross, or position limits without adding hard veto behavior. |
| Portfolio caps | `tests/unit/test_theme_portfolio_caps.py`, `tests/unit/test_portfolio_constructor_theme_caps.py` | Disabled no-op; enabled caps reduce over-cap theme exposure deterministically. |
| Snapshot storage | `tests/unit/test_theme_storage.py`, `tests/unit/test_theme_snapshot_persistence.py` | Disabled writes no files; enabled writes local JSON; bad JSON is skipped safely. |
| Replay/calibration | `tests/unit/test_theme_replay.py`, `tests/unit/test_theme_calibration_dataset.py`, `tests/unit/test_theme_calibration_report.py`, `tests/unit/test_theme_threshold_diagnostics.py` | Offline-only dataset and report generation from local snapshots and frames. |
| Default-off contract | `tests/unit/test_theme_default_off_contract.py` | Theme behavior toggles default to `0`; no canonical theme branch; no `theme_likelihood`; disabled snapshots write no files. |
| Config matrix | `tests/unit/test_theme_config_matrix.py` | Each explicit switch affects only its intended module. |
| Fail-safe contract | `tests/unit/test_theme_fail_safe_contract.py` | Malformed inputs, metadata, snapshots, replay, and empty calibration datasets do not break. |
| Production wiring audit | `tests/unit/test_theme_no_production_wiring.py` | No theme likelihood, no canonical theme branch, no offline replay/calibration imports in production DAG, no external network/LLM imports in theme modules. |

## Focused Contract Tests

```bash
./.venv/bin/python -m pytest \
  tests/unit/test_theme_default_off_contract.py \
  tests/unit/test_theme_config_matrix.py \
  tests/unit/test_theme_fail_safe_contract.py \
  tests/unit/test_theme_no_production_wiring.py \
  -v
```

## Full Theme Regression

```bash
./.venv/bin/python -m pytest \
  tests/unit/test_theme_scanner.py \
  tests/unit/test_theme_agent.py \
  tests/unit/test_theme_metadata_in_context.py \
  tests/unit/test_bayesian_theme_metadata.py \
  tests/unit/test_theme_reporting.py \
  tests/unit/test_deterministic_funnel_theme_boost.py \
  tests/unit/test_theme_boost_diagnostics.py \
  tests/unit/test_theme_risk_constraints.py \
  tests/unit/test_risk_guard_theme.py \
  tests/unit/test_theme_portfolio_caps.py \
  tests/unit/test_portfolio_constructor_theme_caps.py \
  tests/unit/test_theme_storage.py \
  tests/unit/test_theme_snapshot_persistence.py \
  tests/unit/test_theme_replay.py \
  tests/unit/test_theme_calibration_dataset.py \
  tests/unit/test_theme_calibration_report.py \
  tests/unit/test_theme_threshold_diagnostics.py \
  tests/unit/test_theme_default_off_contract.py \
  tests/unit/test_theme_config_matrix.py \
  tests/unit/test_theme_fail_safe_contract.py \
  tests/unit/test_theme_no_production_wiring.py \
  -v
```

## Quick No-Coverage Sweep

Coverage runs can be heavier in this repository. The no-coverage sweep is useful
for fast local validation while iterating on theme tests.

```bash
./.venv/bin/python -m pytest tests/unit/test_theme_*.py -v --no-cov
```

## Environment Inventory

```bash
./.venv/bin/python -m pytest tests/unit/test_llm_env_inventory.py -v
```

## Optional Broader Unit Run

Run the full unit suite only when the worktree and known unrelated failures are
understood:

```bash
./.venv/bin/python -m pytest tests/unit -v
```

## Safety Gate

Phase 6A contract tests are the safety gate before any feature is enabled:

- default-off config must pass.
- config matrix must prove switch scoping.
- fail-safe contracts must pass.
- production wiring audit must pass.

Do not enable funnel boost, RiskGuard overlay, or portfolio theme caps unless
the focused contract tests and relevant feature-specific tests pass in the same
worktree.
