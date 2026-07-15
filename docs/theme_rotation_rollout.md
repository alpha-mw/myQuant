# Theme Rotation Rollout Plan

> Legacy Phase 6 rollout history, not the current activation runbook. For v14
> use `docs/runbooks/v14_operations.md`; the portfolio cap is now a
> default-on safety constraint, while Theme v2 formal is independently
> observer-only and kill-switched.

This rollout keeps Theme Rotation default-off and advances from metadata
observability to paper-only behavior changes before any limited production use.

## Stage 0: Code Merged, All Toggles Off

Configuration:

```text
THEME_SCANNER_ENABLED=0
THEME_FUNNEL_BOOST_ENABLED=0
THEME_RISK_GUARD_ENABLED=0
THEME_PORTFOLIO_CAP_ENABLED=0
THEME_SNAPSHOT_ENABLED=0
```

Expected behavior:

- Trading behavior is unchanged.
- No canonical theme branch exists.
- No `theme_likelihood` exists.
- Snapshot persistence writes no files.

Validation:

- Run default-off and no-production-wiring contract tests.
- Confirm `CANONICAL_BRANCH_ORDER` remains
  `quant/fundamental/macro`.

## Stage 1: Metadata Only

Configuration:

```text
THEME_SCANNER_ENABLED=1
THEME_FUNNEL_BOOST_ENABLED=0
THEME_RISK_GUARD_ENABLED=0
THEME_PORTFOLIO_CAP_ENABLED=0
THEME_SNAPSHOT_ENABLED=0
```

Expected behavior:

- `GlobalContext.metadata["theme_rotation"]` is populated.
- Reports and Bayesian decision metadata can pass through theme metadata.
- Rankings, risk limits, portfolio weights, and posterior math are unchanged.

Watch:

- `theme_count`
- `top_themes`
- `symbols_with_theme`
- `truncated_symbol_count`
- empty or incomplete `industry_map`
- malformed local frames
- report readability and metadata size

## Stage 2: Snapshot Collection

Configuration:

```text
THEME_SCANNER_ENABLED=1
THEME_SNAPSHOT_ENABLED=1
```

Keep disabled:

```text
THEME_FUNNEL_BOOST_ENABLED=0
THEME_RISK_GUARD_ENABLED=0
THEME_PORTFOLIO_CAP_ENABLED=0
```

Expected behavior:

- Local JSON snapshots are written under `THEME_SNAPSHOT_DIR`.
- Trading behavior remains unchanged.

Collection rule:

- Collect at least N trading days before using results for threshold decisions.
  Choose N based on the market regime coverage required by the reviewer.

Watch:

- snapshot `status`
- snapshot `path`
- skipped disabled payloads
- storage growth
- malformed JSON handling

## Stage 3: Offline Calibration

Use:

- `ThemeCalibrationDataset`
- `ThemeCalibrationReport`

Evaluate:

- phase summaries
- symbol theme score buckets
- theme score buckets
- risk flag summaries
- threshold diagnostics
- forward alpha and hit rates
- max drawdown and runup
- missing frame counts and malformed snapshot counts

Decision point:

- Confirm which phases, buckets, and risk flags have enough evidence.
- Decide thresholds and caps before any paper behavior change.
- Leave all production behavior toggles off until calibration has been reviewed.

## Stage 4: Paper A/B Funnel Boost

Configuration in paper/offline only:

```text
THEME_FUNNEL_BOOST_ENABLED=1
```

Use:

- `ThemeBoostDiagnostics`
- baseline funnel output
- boosted funnel output
- local forward outcome review

Compare:

- entered symbols
- dropped symbols
- rank deltas
- score deltas
- phase summary
- risk flag counts
- forward alpha after entry

Decision point:

- Keep `THEME_SYMBOL_BOOST_CAP` low.
- Do not move to live use if entered candidates lack forward evidence.

## Stage 5: Conservative Risk Overlay

Configuration in paper/offline only:

```text
THEME_RISK_GUARD_ENABLED=1
```

Verify:

- overextended themes reduce gross exposure and max weight.
- distribution-risk themes use stricter caps.
- fake-breakout flags cap position size without creating a hard veto.
- existing RiskGuard hard veto behavior remains authoritative.

Decision point:

- Review all symbols changed from buy/add behavior to hold or lower caps.

## Stage 6: Portfolio Theme Caps

Configuration in paper/offline only:

```text
THEME_PORTFOLIO_CAP_ENABLED=1
```

Verify:

- same-theme exposure is capped.
- overextended and distribution themes use stricter exposure caps.
- capped themes are reduced deterministically.
- no unrelated theme is increased solely because caps are enabled.

Decision point:

- Review final weights against existing concentration and sector caps.

## Stage 7: Production Limited Rollout

Prerequisites:

- Stage 1 metadata is stable.
- Stage 2 snapshots cover enough trading days.
- Stage 3 calibration has reviewer acceptance.
- Stage 4 paper A/B does not degrade candidate quality.
- Stage 5 and Stage 6 paper runs reduce intended risks.
- Phase 6A contract tests pass.

Initial production posture:

- Keep boost cap low.
- Keep portfolio caps conservative.
- Enable at most one behavior-changing toggle at a time.
- Monitor daily and keep rollback ready.

## Rollback

Full rollback:

```text
THEME_FUNNEL_BOOST_ENABLED=0
THEME_RISK_GUARD_ENABLED=0
THEME_PORTFOLIO_CAP_ENABLED=0
THEME_SNAPSHOT_ENABLED=0
```

Metadata-only rollback:

```text
THEME_SCANNER_ENABLED=1
THEME_FUNNEL_BOOST_ENABLED=0
THEME_RISK_GUARD_ENABLED=0
THEME_PORTFOLIO_CAP_ENABLED=0
THEME_SNAPSHOT_ENABLED=0
```

Complete theme-off rollback:

```text
THEME_SCANNER_ENABLED=0
THEME_FUNNEL_BOOST_ENABLED=0
THEME_RISK_GUARD_ENABLED=0
THEME_PORTFOLIO_CAP_ENABLED=0
THEME_SNAPSHOT_ENABLED=0
```

Snapshot rollback:

- Set `THEME_SNAPSHOT_ENABLED=0`.
- Leave existing local snapshot files untouched unless retention policy removes
  them.

## Monitoring Checklist

- `theme_count`
- `top_themes`
- `symbols_with_theme`
- `truncated_symbol_count`
- scanner `status` and `diagnostic_notes`
- funnel entered/dropped symbols
- funnel score deltas and rank deltas
- RiskGuard theme flags
- RiskGuard action, gross, and position caps
- portfolio theme exposures before and after caps
- snapshot path and status
- malformed snapshot count
- missing frame count
- calibration hit rates
- calibration forward alpha by phase and bucket
- drawdown by phase and risk flag
