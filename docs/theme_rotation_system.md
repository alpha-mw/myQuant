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
- `quant_investor/themes/smoothing.py`: pure SMA10 smoothing helpers for theme
  heat, 5-day heat delta, persistence count, and trend state.
- `quant_investor/themes/policy.py`: local JSONL policy event parser and
  deterministic policy catalyst scorer. It reads only
  `THEME_POLICY_EVENT_PATH` when `THEME_POLICY_CATALYST_ENABLED=1`.
- `quant_investor/themes/types.py`: serializable `ThemePhase`, `ThemeScore`,
  and `ThemeScanResult` contracts.
- `quant_investor/market/dag/theme_context.py`: metadata adapter that builds
  `global_context.metadata["theme_rotation"]`, extracts per-symbol theme
  metadata, builds optional RiskGuard and portfolio constraints, evaluates the
  optional governance sidecar, and persists optional snapshots/artifacts.
- `quant_investor/themes/governance.py`: pure sidecar evaluator that converts
  existing `theme_rotation.v1` metadata into `theme_governance.v1` labels:
  `admitted_shadow`, `watchlist_strong`, `watchlist_rebuild`, `rejected`,
  `umbrella_only`, and `unavailable`.
- `quant_investor/reporting/theme_governance_renderer.py`: compact markdown
  renderer that states governance is shadow-only and the executable decision
  remains baseline.
- `quant_investor/market/full_report.py`: full-market summary/trade reports
  append theme radar and governance sections when those payloads are present.
- `scripts/run_theme_governance_diagnostics.py`: offline local CLI that loads a
  saved theme snapshot or explicit JSON payload and writes governance JSON/MD
  artifacts.
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
- `quant_investor/monitoring/theme_holding_guard.py`: pure holding-side guard
  evaluator that maps already-held symbols to theme phase/risk signals for
  daily review display.
- `quant_investor/themes/replay.py` and
  `quant_investor/themes/calibration.py`: explicit offline replay and
  calibration diagnostics.

## Data Flow

```text
local OHLCV frames + industry_map
  -> ThemeScanner
  -> optional local policy catalyst component
  -> ThemeScanResult
  -> build_theme_rotation_metadata
  -> GlobalContext.metadata["theme_rotation"]
  -> report renderer
  -> optional governance sidecar/report/artifact
  -> BayesianDecisionRecord metadata passthrough
  -> optional funnel boost
  -> optional RiskGuard overlay
  -> optional PortfolioConstructor caps
  -> optional local snapshot
  -> optional holding-side daily-review guard
  -> explicit offline replay/calibration
```

Governance output is not fed into Bayesian likelihoods, RiskGuard,
ICCoordinator, PortfolioConstructor, candidate ranking, branch weights, or final
allocation. It is a sidecar answer to: which themes are observable mainlines,
which are overheated or fragile, and which have insufficient samples or need
rebuild.

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
  `evidence`, and `metadata`. Scanner output also preserves `raw_score` and can
  include `smoothed_score`, `heat_10d`, `heat_delta_5d`, `persistence_count`,
  `trend_state`, and `smoothing_status`.
- Policy catalyst fields are present with defaults and become active only when
  `THEME_POLICY_CATALYST_ENABLED=1`: `policy_catalyst_score`,
  `policy_confidence`, `policy_stage`, `policy_evidence`, and
  `policy_risk_flags`.
- Crowding diagnostic fields are present with neutral defaults and become
  active only when `THEME_CROWDING_ENABLED=1`: `theme_turnover_share`,
  `turnover_share_sma10`, `turnover_share_stretch`,
  `turnover_share_delta_5d`, `turnover_share_trend`,
  `theme_limitup_ratio`, `limitup_norm`,
  `member_turnover_concentration`, `crowding_risk`, `crowding_status`, and
  `crowding_diagnostic_notes`.
- `symbol_scores`: map of symbol to normalized theme score in `[0, 1]`.
- `symbol_smoothed_scores`: separate normalized SMA10 theme score map when
  enough local history exists; `symbol_scores` remains raw/current for
  compatibility.
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
- `symbol_theme_smoothed_score`
- `symbol_primary_theme`
- `symbol_theme_phase`
- `theme_alerts`
- `theme_snapshot`

`global_context.metadata["theme_governance"]` uses schema
`theme_governance.v1` when `THEME_GOVERNANCE_ENABLED=1`.

Important fields:

- `enabled`: whether governance evaluation was requested and available.
- `status`: `success`, `disabled`, `error`, or `unavailable`.
- `market`, `universe_key`, `as_of`: inherited scan scope.
- `decisions` / `top_themes`: per-theme governance rows with gate label,
  current/raw score, SMA10 heat, 5-day heat delta, persistence count, trend
  state, confidence, breadth, member count, phase, risk flags, registry fields,
  reasons, and diagnostics.
- `summary_counts`: counts for all gate labels.
- `diagnostic_notes`: registry/scanner/evaluator notes.
- `metadata`: deterministic markers plus `shadow_only=true`.

Optional registry support is JSON-only via
`THEME_GOVERNANCE_REGISTRY_PATH`. The schema is:

```json
{
  "themes": [
    {
      "theme_id": "industry::semiconductor",
      "theme_type": "tradable",
      "style_tag": "growth_cycle",
      "parent_theme": "technology",
      "theme_name": "Semiconductor",
      "notes": "Optional local governance note."
    }
  ]
}
```

`theme_type=umbrella` forces `umbrella_only`. If the registry is unset,
missing, or malformed, the evaluator falls back to inferred `industry::...`
themes as tradable and emits diagnostic notes instead of fabricating a healthier
state.

## Smoothing And Trend Confirmation

Theme heat uses a deterministic 10-observation simple moving average:

- Default window: `THEME_SMOOTHING_WINDOW=10`.
- Minimum observations: `THEME_SMOOTHING_MIN_OBSERVATIONS=5`.
- Trend states: `warming`, `cooling`, `stable`, `spike_unconfirmed`, and
  `insufficient_history`.
- Scanner output does not overwrite `score` or `symbol_scores`; it adds
  smoothed fields beside the raw fields.
- Governance admission uses confirmed smoothing when available. A raw one-day
  spike without enough history becomes `watchlist_strong` with
  `raw_spike_not_confirmed`, not `admitted_shadow`.
- Governance can read recent local snapshots from `THEME_SNAPSHOT_DIR`; missing
  history is diagnostic-only and never treated as healthy confirmation.

## Crowding Diagnostics

Crowding Diagnostics are default-off and use only the same local market frames
already provided to `ThemeScanner`. When `THEME_CROWDING_ENABLED=0`, all new
fields remain neutral, no new risk flags are emitted, and no overextension or
funnel score is changed.

When enabled, the scanner computes three local metrics per theme:

- `theme_turnover_share`: latest theme member turnover divided by latest
  scanned-universe turnover. If `amount` is missing, the scanner approximates
  it with `close * vol` and records `amount_approximated`.
- `theme_limitup_ratio`: latest limit-up member count divided by theme member
  count. Limit-up is approximated from local `pct_chg`, `close`, and `high`;
  `pct_chg` values with absolute max above `1` are treated as percentage
  points, otherwise as decimal returns. If `pct_chg` is missing, it is derived
  from local close history.
- `member_turnover_concentration`: top-three member turnover divided by theme
  turnover. Themes with fewer than four members use `1.0` and record a
  diagnostic note.

Recent snapshot history from `THEME_SNAPSHOT_DIR` supplies the
`theme_turnover_share` SMA10 baseline through `smooth_numeric_series`; missing
or old snapshots keep `turnover_share_stretch=0` and record
`turnover_share_smoothing_status=insufficient_history`.

The combined score is:

```text
crowding_risk =
  0.45 * turnover_share_stretch
  + 0.35 * limitup_norm
  + 0.20 * member_turnover_concentration
```

The score is clamped to `[0, 1]`. If the scanned universe is smaller than
`THEME_CROWDING_MIN_UNIVERSE` (default `30`), the scanner sets
`crowding_status="insufficient_universe"`, records diagnostics, and keeps risk
fields neutral. When enabled and eligible, crowding contributes
`0.30 * crowding_risk` to `overextension_risk`. `crowding_risk >= 0.70` adds
`theme_crowded`; `theme_limitup_ratio >= 0.20` with breadth below `0.40` adds
`theme_narrow_leadership`.

The deterministic funnel recognizes those two flags only inside the already
enabled `THEME_FUNNEL_BOOST_ENABLED` path: `theme_crowded` applies `-0.03` and
`theme_narrow_leadership` applies `-0.02`.

## Policy Catalyst Layer

The Policy Catalyst Layer is a deterministic ThemeScanner sidecar. It is
default-off and reads only local JSONL policy-event fixtures or manually
maintained caches from `THEME_POLICY_EVENT_PATH`.

Input JSONL fields are:

- `event_id`, `title`, `issuer`, `publish_date`, `effective_date`
- `policy_level`, `policy_type`
- `theme_tags`, `industry_tags`, `symbol_tags`
- `evidence_text`, `source_url`

The scorer matches events to already-scanned themes by normalized theme tags,
industry tags, or symbol tags intersecting existing theme members. Symbol tags
only improve beneficiary clarity; they do not create symbols, candidates, or a
candidate-pool source.

The policy component writes per-theme metadata and can add a capped score
component of at most `THEME_POLICY_CATALYST_WEIGHT * 100`. Missing or malformed
files set `policy_catalyst_status=unavailable` and leave theme scores
unchanged. Policy evidence remains metadata; it is not a buy/sell signal, not a
Bayesian likelihood, and not a canonical branch.

## Optional Consumers

All behavior-changing consumers require explicit configuration:

- Funnel boost: `THEME_FUNNEL_BOOST_ENABLED=1` can change momentum-leader
  candidate scores, capped by `THEME_SYMBOL_BOOST_CAP`. It reads raw theme
  scores by default. `THEME_FUNNEL_BOOST_SCORE_SOURCE=smoothed` must be set
  explicitly to use `symbol_smoothed_scores`; if smoothed scores are missing,
  no theme boost is applied for that symbol.
- RiskGuard overlay: `THEME_RISK_GUARD_ENABLED=1` can tighten action, gross, or
  position limits for overextended, distribution, or fake-breakout theme risk.
- Portfolio caps: `THEME_PORTFOLIO_CAP_ENABLED=1` can reduce final weights to
  respect single-theme exposure caps.
- Snapshot persistence: `THEME_SNAPSHOT_ENABLED=1` writes local JSON snapshots
  under `THEME_SNAPSHOT_DIR`.
- Holding-side guard: `THEME_HOLDING_GUARD_ENABLED=1` lets the CN aggressive
  daily review read the latest persisted theme snapshot and show watch/tighten
  diagnostics for existing holdings whose primary theme has moved to
  `overextended` or `distribution`. `THEME_HOLDING_GUARD_TIGHTEN_RATIO`
  defaults to `0.5` and only changes the report display stop for tighten rows;
  it does not rewrite source ledgers, target weights, or executable orders.
- Governance sidecar: `THEME_GOVERNANCE_ENABLED=1` attaches
  `theme_governance.v1` metadata. `THEME_GOVERNANCE_ARTIFACT_ENABLED=1` writes
  a local governance JSON artifact under `THEME_GOVERNANCE_OUTPUT_DIR`.
- Policy catalyst: `THEME_POLICY_CATALYST_ENABLED=1` reads local JSONL policy
  events and adds only capped theme-score metadata. It does not bypass the v13
  DAG or create candidate-pool entries.
- Crowding diagnostics: `THEME_CROWDING_ENABLED=1` lets the scanner add local
  crowding metrics and gated risk flags. `THEME_CROWDING_MIN_UNIVERSE` defaults
  to `30`; smaller scanned universes produce diagnostics only.

Offline-only consumers are explicit:

- `ThemeBoostDiagnostics` compares baseline and boosted funnel outputs.
- `ThemeCalibrationDataset` replays snapshots against local forward frames.
- `ThemeCalibrationReport` summarizes phase, bucket, theme, risk-flag, and
  threshold diagnostics.
- `run_theme_governance_diagnostics.py` renders governance JSON/MD from local
  snapshots without changing executable decisions.

Replay and calibration outputs explicitly mark `pit_industry_labels=false`:
industry labels are as-of run date, not point-in-time; replay carries mild
reclassification look-ahead. This is a disclosed limitation, not a data repair.

## Holding-side Guard

Holding-side Guard closes the gap between theme entry metadata and existing
portfolio review. It is default-off and runs only inside the CN aggressive daily
review after tracker artifacts have been written.

Inputs:

- existing `holdings_review.csv` rows for held symbols and display prices;
- latest local `theme_snapshot.v1` loaded through `ThemeSnapshotStore`;
- `theme_rotation.v1` maps: `symbol_primary_theme`, `symbol_phase`,
  `symbol_risk_flags`, and `theme_scores`.

Rules are deterministic and advisory:

- `distribution` phase or `theme_distribution_risk` => `tighten`;
- `overextended` phase or `theme_overextended`,
  `theme_overextended_no_chase`, `theme_fake_breakout_risk` => `watch`;
- missing snapshot, missing mapping, blank phase, or `unclassified` =>
  `none` with diagnostics.

When enabled, the report gains a `主题状态守卫` section. Tighten rows show the
original stage stop and a display-only tightened stop computed by shrinking the
current-price-to-stage-stop buffer by `THEME_HOLDING_GUARD_TIGHTEN_RATIO`.
This is intentionally read-only: it affects review text and diagnostics, not
the persisted holding ledger, broker state, PortfolioConstructor target
weights, or RiskGuard gates.

## A_quant Inspiration Boundary

The sidecar borrows A_quant-style operating discipline, not A_quant production
inputs:

- Used concepts: a lightweight ontology, separation of heat vs fragility,
  four-way governance gating, and local artifact discipline.
- Not used: A_quant current weights, DuckDB/Tushare storage paths, tactical
  portfolio blends, or any promotion of theme labels into investment signals.

The label `admitted_shadow` is intentional. It means observable governance
admission only; it is not a buy/sell signal and does not change
`final_decision_source`, which remains `baseline` in the shadow path.

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
- Disabled governance writes no governance artifact and returns a disabled
  `theme_governance.v1` payload.
- Missing or malformed policy event JSONL returns `policy_catalyst_status ==
  "unavailable"` and leaves theme scores unchanged.
- Disabled crowding diagnostics keep neutral fields and emit no crowding risk
  flags. Missing `amount`, missing `pct_chg`, insufficient snapshot history, or
  insufficient universe are diagnostic-only and do not interrupt the scan.
- Missing or malformed registry JSON falls back to inferred local industry
  themes and records diagnostics.
- Missing or malformed per-theme numeric fields become `unavailable`, never
  healthy labels.
- Snapshot write failures return `status == "error"` with diagnostic notes
  instead of changing rankings, risk limits, or weights.
- Snapshot loading skips malformed JSON and returns the latest valid snapshot or
  `None`.
- Holding-side guard fails open: missing snapshots or unmapped holdings only add
  diagnostics and do not block daily review or synthesize sell orders.
- Replay/calibration handle malformed snapshots and empty datasets with
  metadata counters and warnings.

## Limitations

- MVP theme definitions start from `industry_map`; a full concept-board ontology
  is not yet integrated.
- `industry_map` labels are as-of the run date, not point-in-time. Historical
  replay/calibration therefore carries mild reclassification look-ahead until a
  PIT membership source is introduced.
- Live news, live policy feeds, capital-flow data, and concept-board membership
  are not part of the scanner contract yet. Policy catalyst v1 accepts only
  local fixtures or manually maintained JSONL caches.
- Crowding limit-up detection is an approximation from local bars. Code-prefix
  thresholds cover STAR/ChiNext, Beijing, and regular boards; ST 5% limits are
  not reliably identifiable from code alone and are not modeled in this phase.
- No live LLM, provider, or network scoring is used.
- Funnel boosts, risk overlays, and portfolio caps require offline replay and
  calibration before production enablement.
- Governance thresholds are defaults for observation. Replay/calibration is
  required before any future behavior toggle can be discussed.
- Theme Rotation is observability-first; behavior changes must remain gated by
  explicit environment toggles and rollout evidence.
