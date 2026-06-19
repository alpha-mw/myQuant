# Theme Rotation Configuration Matrix

All behavior-affecting Theme Rotation toggles are default-off. The scanner can
produce metadata when explicitly enabled, but ranking, risk limits, final
weights, and file writes each require their own opt-in switch.

## Environment Variables

| Variable | Default | Type | Module affected | Production impact | Safe rollout recommendation |
| --- | --- | --- | --- | --- | --- |
| `THEME_SCANNER_ENABLED` | `0` | bool | `market/dag/context.py`, `themes/scanner.py` | Produces `GlobalContext.metadata["theme_rotation"]` only. No ranking, risk, or portfolio behavior changes by itself. | Enable first in metadata-only paper runs; keep all downstream toggles off. |
| `THEME_MIN_MEMBER_COUNT` | `5` | int | `ThemeScanner` | Filters themes with fewer members than the threshold. | Keep default until theme definitions are reviewed; lower only for explicit small-theme diagnostics. |
| `THEME_TOP_N` | `20` | int | `ThemeScanner`, metadata adapter | Limits ranked themes emitted in metadata/reporting. | Keep default for daily reports; increase only if metadata size is monitored. |
| `THEME_METADATA_SYMBOL_LIMIT` | `300` | int | `theme_context.build_theme_rotation_metadata` | Limits per-symbol theme maps in metadata. | Keep bounded; monitor `truncated_symbol_count` before increasing. |
| `THEME_FUNNEL_BOOST_ENABLED` | `0` | bool | `DeterministicFunnel` | Can change momentum-leader funnel scores and candidate ranking. | Use only in offline or paper A/B first; compare entered/dropped symbols before live rollout. |
| `THEME_SYMBOL_BOOST_CAP` | `0.10` | float | `DeterministicFunnel` | Caps positive per-symbol theme boost when funnel boost is enabled. | Start below default for limited rollout; never raise without calibration evidence. |
| `THEME_RISK_GUARD_ENABLED` | `0` | bool | `theme_context`, `RiskGuard` | Can tighten action cap, gross exposure cap, and per-symbol position limits. | Enable only after paper validation of overextended/distribution risk behavior. |
| `THEME_RISK_OVEREXTENDED_GROSS_CAP` | `0.60` | float | `theme_context.build_theme_risk_constraints` | Gross exposure cap used for overextended themes when overlay is enabled. | Keep conservative; verify it does not mask existing hard veto behavior. |
| `THEME_RISK_OVEREXTENDED_MAX_WEIGHT` | `0.10` | float | `theme_context.build_theme_risk_constraints` | Per-symbol max weight for overextended themes when overlay is enabled. | Test against current portfolio weights before enabling. |
| `THEME_RISK_DISTRIBUTION_GROSS_CAP` | `0.45` | float | `theme_context.build_theme_risk_constraints` | Gross exposure cap for distribution-risk themes when overlay is enabled. | Treat as stricter than overextended; validate in paper/offline first. |
| `THEME_RISK_DISTRIBUTION_MAX_WEIGHT` | `0.08` | float | `theme_context.build_theme_risk_constraints` | Per-symbol max weight for distribution-risk themes when overlay is enabled. | Keep strict; monitor symbols moved from buy/add to hold. |
| `THEME_RISK_FAKE_BREAKOUT_MAX_WEIGHT` | `0.10` | float | `theme_context.build_theme_risk_constraints` | Per-symbol cap for fake-breakout risk when overlay is enabled. | Enable only after fake-breakout calibration has acceptable precision. |
| `THEME_PORTFOLIO_CAP_ENABLED` | `0` | bool | `theme_context`, `PortfolioConstructor` | Can reduce final target weights to respect theme exposure caps. | Paper test with existing holdings and target weights before any live use. |
| `THEME_PORTFOLIO_MAX_THEME_EXPOSURE` | `0.35` | float | `theme_context.build_theme_portfolio_constraints` | Base max exposure for one primary theme when caps are enabled. | Keep conservative and review exposure redistribution manually. |
| `THEME_PORTFOLIO_OVEREXTENDED_MAX_THEME_EXPOSURE` | `0.25` | float | `theme_context.build_theme_portfolio_constraints` | Stricter cap for overextended themes when caps are enabled. | Validate that capped themes are reduced without increasing unrelated risks. |
| `THEME_PORTFOLIO_DISTRIBUTION_MAX_THEME_EXPOSURE` | `0.15` | float | `theme_context.build_theme_portfolio_constraints` | Strictest cap for distribution-risk themes when caps are enabled. | Use only after distribution flags have enough replay evidence. |
| `THEME_SNAPSHOT_ENABLED` | `0` | bool | `theme_context`, `ThemeSnapshotStore` | Writes local JSON theme snapshots only. No ranking, risk, or portfolio behavior changes. | Enable after metadata looks stable; choose a local artifact directory. |
| `THEME_SNAPSHOT_DIR` | `results/theme_snapshots` | string | `ThemeSnapshotStore` | Root directory for local snapshot JSON files. | Use a run-artifact path that is included in retention/cleanup policy. |
| `THEME_SNAPSHOT_SAVE_DISABLED` | `0` | bool | `theme_context.persist_theme_rotation_snapshot` | When true, saves disabled scanner payloads if snapshot writes are enabled. | Leave off for normal runs; enable only to audit disabled-state pipelines. |

## Rollout Rules

- `THEME_SCANNER_ENABLED=1` is metadata-only.
- `THEME_FUNNEL_BOOST_ENABLED=1` can affect candidate ranking.
- `THEME_RISK_GUARD_ENABLED=1` can affect risk limits.
- `THEME_PORTFOLIO_CAP_ENABLED=1` can affect final weights.
- `THEME_SNAPSHOT_ENABLED=1` writes local files only.
- Replay, calibration, and boost diagnostics are explicit offline tools, not
  production DAG imports.
