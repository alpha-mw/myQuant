# Theme Rotation System

## Purpose

The Theme Rotation system is a deterministic A-share observability layer for
sector heat, rotation phase, and early acceleration detection. It converts local
OHLCV frames plus an `industry_map` into compact metadata that can be rendered,
audited, replayed, or optionally consumed by explicitly enabled downstream
overlays.

The system is default-off for behavior-changing paths. It does not add a
canonical research branch, does not add a Bayesian likelihood, and does not call
external data, LLM, broker, or execution services.

## Architecture

Core components:

- `quant_investor/themes/scanner.py`: `ThemeScanner` groups symbols by
  `industry_map`, computes deterministic breadth, momentum, acceleration,
  volume confirmation, overextension, and fake-breakout diagnostics, then emits
  `ThemeScanResult`.
- `quant_investor/themes/types.py`: serializable `ThemePhase`, `ThemeScore`,
  and `ThemeScanResult` contracts.
- `quant_investor/market/dag/theme_context.py`: metadata adapter that builds
  `global_context.metadata["theme_rotation"]`, extracts per-symbol theme
  metadata, builds optional RiskGuard and portfolio constraints, and persists
  optional snapshots.
- `quant_investor/agents/theme_agent.py`: standalone non-canonical
  `ThemeAgent` that reads theme metadata and returns a normal `BranchVerdict`
  with `metadata["branch_name"] == "theme"`. It is not wired into the canonical
  research DAG.
- `quant_investor/reporting/theme_renderer.py`: markdown renderer for the
  theme radar section.
- `quant_investor/funnel/deterministic_funnel.py`: optional
  `theme_boost_enabled` score adjustment for the momentum-leader funnel profile.
- `quant_investor/funnel/theme_boost_diagnostics.py`: explicit offline A/B
  diagnostics comparing baseline and boosted funnel outputs.
- `quant_investor/agents/risk_guard.py`: optional theme risk overlay when
  theme constraints are provided and `theme_risk_guard_enabled` is true.
- `quant_investor/agents/portfolio_constructor.py`: optional theme exposure
  caps when `theme_portfolio_cap_enabled` is true.
- `quant_investor/themes/storage.py`: local JSON snapshot store.
- `quant_investor/themes/replay.py` and
  `quant_investor/themes/calibration.py`: explicit offline replay and
  calibration diagnostics.

## Data Flow

```text
local OHLCV frames + industry_map
  -> ThemeScanner
  -> ThemeScanResult
  -> build_theme_rotation_metadata
  -> GlobalContext.metadata["theme_rotation"]
  -> report renderer
  -> BayesianDecisionRecord metadata passthrough
  -> optional funnel boost
  -> optional RiskGuard overlay
  -> optional PortfolioConstructor caps
  -> optional local snapshot
  -> explicit offline replay/calibration
```

## Non-Canonical Design

Theme Rotation is deliberately not a canonical branch.

- `CANONICAL_BRANCH_ORDER` remains
  `("quant", "fundamental", "intelligence", "macro")`.
- `ThemeAgent` is a standalone helper with `branch_name == "theme"` metadata,
  not an input to canonical branch weighting.
- The research DAG does not import `ThemeAgent` as a fifth branch.
- Theme metadata can be rendered or passed through records without changing the
  canonical research contract.

There is also no `theme_likelihood`.

- `LikelihoodSet` remains the quant, fundamental, and intelligence likelihood
  contract.
- No posterior formula includes a theme term.
- Theme signals can be calibrated offline, but they do not alter Bayesian math
  unless a future explicit phase changes that contract.

## Determinism And Side Effects

The scanner and adapters are deterministic for the same local inputs:

- Symbols and themes are sorted for stable tie-breaking.
- Theme IDs are derived from `industry_map` values.
- Metrics use local frames and provided symbol market state only.
- Metadata includes deterministic markers such as `no_llm` and `no_network`.

The theme modules do not fetch market data, call LLM providers, or reach out to
network services. Snapshot persistence writes only local JSON files when
explicitly enabled.

## Metadata Shape

`global_context.metadata["theme_rotation"]` uses schema
`theme_rotation.v1`.

Important fields:

- `enabled`: whether the scanner path was enabled.
- `status`: `success`, `disabled`, or `error`.
- `market`, `universe_key`, `as_of`: scan scope.
- `theme_scores`: map of theme ID to score payload. Each payload includes
  `theme_id`, `theme_name`, `phase`, `score`, `confidence`, `member_count`,
  `breadth`, `momentum`, `acceleration`, `volume_confirmation`,
  `overextension_risk`, `fake_breakout_risk`, `top_symbols`, `risk_flags`,
  `evidence`, and `metadata`.
- `symbol_scores`: map of symbol to normalized theme score in `[0, 1]`.
- `symbol_primary_theme`: map of symbol to primary theme ID.
- `symbol_phase`: map of symbol to phase.
- `symbol_risk_flags`: map of symbol to theme risk flags.
- `top_themes`: compact ranked theme list for report rendering.
- `diagnostic_notes`: safe status notes.
- `metadata`: deterministic integration metadata, scanned counts, symbol limit,
  and truncation count.

The DAG also exposes compatibility aliases in `GlobalContext.metadata`:

- `theme_scores`
- `symbol_theme_score`
- `symbol_primary_theme`
- `symbol_theme_phase`
- `theme_alerts`
- `theme_snapshot`

## Optional Consumers

All behavior-changing consumers require explicit configuration:

- Funnel boost: `THEME_FUNNEL_BOOST_ENABLED=1` can change momentum-leader
  candidate scores, capped by `THEME_SYMBOL_BOOST_CAP`.
- RiskGuard overlay: `THEME_RISK_GUARD_ENABLED=1` can tighten action, gross, or
  position limits for overextended, distribution, or fake-breakout theme risk.
- Portfolio caps: `THEME_PORTFOLIO_CAP_ENABLED=1` can reduce final weights to
  respect single-theme exposure caps.
- Snapshot persistence: `THEME_SNAPSHOT_ENABLED=1` writes local JSON snapshots
  under `THEME_SNAPSHOT_DIR`.

Offline-only consumers are explicit:

- `ThemeBoostDiagnostics` compares baseline and boosted funnel outputs.
- `ThemeCalibrationDataset` replays snapshots against local forward frames.
- `ThemeCalibrationReport` summarizes phase, bucket, theme, risk-flag, and
  threshold diagnostics.

## Fail-Safe Behavior

The system is designed to fail closed:

- Empty data or empty `industry_map` produces empty theme metadata rather than
  an exception.
- Malformed frames fall back to neutral symbol metrics.
- Disabled scanner config writes a `status == "disabled"` metadata payload.
- Malformed `theme_rotation` metadata returns unavailable per-symbol theme
  metadata.
- Optional RiskGuard and portfolio constraint builders return empty constraints
  when metadata is missing or malformed.
- Disabled snapshot persistence writes no files and returns `status ==
  "disabled"`.
- Snapshot write failures return `status == "error"` with diagnostic notes
  instead of changing rankings, risk limits, or weights.
- Snapshot loading skips malformed JSON and returns the latest valid snapshot or
  `None`.
- Replay/calibration handle malformed snapshots and empty datasets with
  metadata counters and warnings.

## Limitations

- MVP theme definitions start from `industry_map`; a full concept-board ontology
  is not yet integrated.
- Live news, policy signals, capital-flow data, and concept-board membership are
  not part of the scanner contract yet.
- No live LLM, provider, or network scoring is used.
- Funnel boosts, risk overlays, and portfolio caps require offline replay and
  calibration before production enablement.
- Theme Rotation is observability-first; behavior changes must remain gated by
  explicit environment toggles and rollout evidence.
