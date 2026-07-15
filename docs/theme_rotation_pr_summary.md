# Theme Rotation PR Summary

> Legacy Phase 6 change summary. Its default-off list records that historical
> rollout only; v13.1 operations follow `docs/runbooks/v13_1_freeze_exception.md`
> and keep the portfolio cap enabled while Theme v2 formal remains off.

## What Changed

- Added a deterministic Theme Rotation scanner for local A-share OHLCV frames
  grouped by `industry_map`.
- Added serializable theme contracts for phases, theme scores, and scan results.
- Added metadata adapter output under
  `GlobalContext.metadata["theme_rotation"]`.
- Added a standalone non-canonical `ThemeAgent`.
- Added theme radar markdown rendering.
- Added optional default-off funnel boost, RiskGuard overlay, portfolio caps,
  snapshot persistence, and offline replay/calibration diagnostics.
- Added Phase 6A QC tests for default-off behavior, config scoping, fail-safe
  handling, and no-production-wiring audits.

## What Did Not Change

- No `theme` canonical branch was added.
- `CANONICAL_BRANCH_ORDER` remains
  `quant/fundamental/macro`.
- No `theme_likelihood` was added.
- Bayesian likelihood/types/posterior formulas are unchanged.
- DAG canonical branch logic is unchanged.
- Funnel, RiskGuard, and PortfolioConstructor behavior remains unchanged unless
  explicit theme toggles are enabled.
- Narrator, `daily_runner.py`, and web/API behavior are unchanged.
- No external API, LLM, broker, or execution service is called by Theme
  Rotation.

## Why It Is Safe

- Behavior-affecting toggles are default-off.
- Scanner metadata alone does not change ranking, posterior math, risk limits,
  or final weights.
- Optional consumers are scoped:
  - funnel boost affects only funnel scoring when enabled.
  - RiskGuard overlay affects only risk limits when enabled.
  - portfolio caps affect only final weights when enabled.
  - snapshots write only local JSON files when enabled.
- Replay, calibration, and boost diagnostics are explicit offline tools.
- Fail-safe tests cover empty data, malformed metadata, bad snapshots, bad
  replay input, and empty calibration datasets.
- Production wiring tests guard against a fifth canonical branch, theme
  likelihood, offline diagnostics in the production DAG, and network/LLM imports
  in theme modules.

## Default-Off Config List

- `THEME_SCANNER_ENABLED=0`
- `THEME_FUNNEL_BOOST_ENABLED=0`
- `THEME_RISK_GUARD_ENABLED=0`
- `THEME_PORTFOLIO_CAP_ENABLED=0`
- `THEME_SNAPSHOT_ENABLED=0`
- `THEME_SNAPSHOT_SAVE_DISABLED=0`

Other theme knobs are inert unless their owning toggle is enabled:

- `THEME_MIN_MEMBER_COUNT=5`
- `THEME_TOP_N=20`
- `THEME_METADATA_SYMBOL_LIMIT=300`
- `THEME_SYMBOL_BOOST_CAP=0.10`
- `THEME_RISK_OVEREXTENDED_GROSS_CAP=0.60`
- `THEME_RISK_OVEREXTENDED_MAX_WEIGHT=0.10`
- `THEME_RISK_DISTRIBUTION_GROSS_CAP=0.45`
- `THEME_RISK_DISTRIBUTION_MAX_WEIGHT=0.08`
- `THEME_RISK_FAKE_BREAKOUT_MAX_WEIGHT=0.10`
- `THEME_PORTFOLIO_MAX_THEME_EXPOSURE=0.35`
- `THEME_PORTFOLIO_OVEREXTENDED_MAX_THEME_EXPOSURE=0.25`
- `THEME_PORTFOLIO_DISTRIBUTION_MAX_THEME_EXPOSURE=0.15`
- `THEME_SNAPSHOT_DIR=results/theme_snapshots`

## Test Evidence

Recommended evidence to attach before merge:

```bash
./.venv/bin/python -m pytest \
  tests/unit/test_theme_default_off_contract.py \
  tests/unit/test_theme_config_matrix.py \
  tests/unit/test_theme_fail_safe_contract.py \
  tests/unit/test_theme_no_production_wiring.py \
  -v
```

```bash
./.venv/bin/python -m pytest tests/unit/test_theme_*.py -v --no-cov
```

```bash
./.venv/bin/python -m pytest tests/unit/test_llm_env_inventory.py -v
```

For the full theme regression command, see
`docs/theme_rotation_testing_matrix.md`.

## Rollout Plan

1. Merge with all toggles off.
2. Enable scanner metadata only.
3. Enable local snapshot collection.
4. Run offline replay/calibration.
5. Run paper A/B funnel boost diagnostics.
6. Test conservative RiskGuard overlay in paper/offline.
7. Test portfolio theme caps in paper/offline.
8. Consider limited production rollout with low boost cap, conservative caps,
   daily monitoring, and immediate rollback toggles.

## Known Limitations

- Theme definitions currently start from `industry_map`, not a full
  concept-board ontology.
- No live news, policy, capital-flow, or LLM scoring is used.
- There is no `theme_likelihood` and no canonical fifth branch.
- Boost, risk overlay, and portfolio caps require calibration before production
  enablement.
- Worktree had pre-existing dirty files during this documentation pass;
  reviewers should inspect the final diff carefully.

## Follow-Up Work

- Build a richer theme ontology from reviewed concept-board and industry data.
- Add policy/news/capital-flow inputs only if they can be made deterministic,
  local, testable, and default-off.
- Accumulate enough snapshots for statistically useful replay/calibration.
- Define production enablement thresholds for funnel boost, risk overlay, and
  portfolio caps.
- Add reviewer-facing dashboard artifacts after the offline calibration process
  stabilizes.
