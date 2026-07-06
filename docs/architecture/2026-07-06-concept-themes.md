# Concept-Level Theme Design Proposal

Date: 2026-07-06

Status: Phase 5a design only. Do not implement until Phase 5b is explicitly
approved.

## Goal

Extend Theme Rotation beyond static `industry_map` labels so cross-industry
themes such as low-altitude economy, humanoid robotics, advanced packaging, or
domestic AI compute can be represented without introducing look-ahead bias.

Hard redline: current Tushare concept constituents must not be used to replay
history. They are usually restated after the fact and would be the highest-risk
source of theme backtest look-ahead.

## Route A: Local Return-Correlation Statistical Themes

Route A derives statistical themes from local price returns only.

Proposed flow:

1. Read local OHLCV frames already available to `ThemeScanner`.
2. Build deterministic return vectors over fixed windows such as 20/60/120
   trading days.
3. Filter symbols with insufficient local bars or stale prices.
4. Compute pairwise return correlation from local returns.
5. Build a correlation graph with deterministic thresholds and minimum degree.
6. Form clusters with a dependency-free greedy connected-component or
   deterministic community routine.
7. Emit `stat::cluster::<date>::<rank>` theme IDs with members, window,
   average intra-cluster correlation, breadth, and diagnostics.

Advantages:

- Zero external data.
- Naturally point-in-time if only bars available as of the replay date are used.
- Fully deterministic and offline.
- Useful for discovering emerging cross-industry co-movement before labels are
  manually curated.

Weaknesses:

- Cluster names are not semantic without human review.
- Clusters can be unstable across windows and market regimes.
- Correlation can group liquidity, beta, or index effects rather than a real
  concept.
- It can accidentally encode short-term crowded trades as a "theme".
- Primary-theme attribution is hard to explain in reports.

Best use:

- Research/discovery sidecar.
- Candidate suggestion source for human-maintained PIT concept membership.
- Not recommended as the canonical concept source for production Theme Rotation.

## Route B: PIT JSONL Concept Membership

Route B uses a local point-in-time JSONL membership table. It follows the same
operating pattern as `themes/policy.py`: local file, one JSON object per line,
strict parser, date parsing, diagnostic notes, malformed-file fail-open when
the concept path is optional.

Each membership record has an explicit effective interval:

- `effective_from`: first date the membership can be used.
- `effective_to`: exclusive end date; blank means active until replaced.

The scanner evaluates membership with `effective_from <= as_of` and
`effective_to > as_of` when `effective_to` is present. Future-dated records are
ignored during replay. No current vendor concept file can be backfilled into
old dates unless the effective interval is sourced and reviewed as point in
time.

Advantages:

- Semantic concept names are explicit and reportable.
- Point-in-time rules are auditable.
- Manual or semi-automated maintenance can cite source evidence.
- Concepts can coexist with industry themes without replacing `industry_map`.
- Replay/calibration can use the same dated membership logic.

Weaknesses:

- Requires maintenance discipline.
- Coverage starts sparse unless historical evidence is curated.
- Bad effective dates still create bias, so source review is mandatory.
- Membership confidence and concept scope need governance.

Best use:

- Canonical implementation route for named cross-industry concepts.
- Production-ready only after parser, contract tests, and replay diagnostics are
  approved.

## `theme_membership.v1` Schema Proposal

One JSONL object per symbol-theme membership:

```json
{
  "schema_version": "theme_membership.v1",
  "membership_id": "concept-low-altitude-000001-2026-01-15",
  "theme_id": "concept::low-altitude-economy",
  "theme_name": "Low-altitude Economy",
  "theme_type": "concept",
  "symbol": "000001.SZ",
  "symbol_name": "Example Co",
  "effective_from": "2026-01-15",
  "effective_to": "",
  "membership_status": "active",
  "confidence": 0.8,
  "source_type": "manual_review",
  "source_ref": "policy_event:2026-low-altitude-plan",
  "evidence_text": "Short source-backed reason for inclusion.",
  "maintainer": "local",
  "created_at": "2026-01-15T00:00:00Z",
  "updated_at": "2026-01-15T00:00:00Z",
  "tags": ["policy", "equipment"]
}
```

Validation rules:

- `schema_version` must equal `theme_membership.v1`.
- `theme_id` must use `concept::` or another explicitly registered prefix.
- `theme_type` must be `concept` for Phase 5b.
- `symbol`, `effective_from`, and `theme_id` are required.
- `effective_from` and `effective_to` accept `YYYY-MM-DD` or `YYYYMMDD`.
- `effective_to` is exclusive.
- `confidence` is clamped to `[0, 1]`; malformed confidence is diagnostic.
- Duplicate active records for the same symbol/theme/as-of are deduped
  deterministically by latest `updated_at`, then `membership_id`.
- Future memberships relative to `as_of` are ignored.

## Merge Semantics With `industry_map`

Industry themes remain the baseline. Concept themes are additive.

Proposed scanner input:

- `industry_map`: existing symbol to industry label map.
- `theme_memberships`: optional active PIT memberships as of `as_of`.

A symbol may have:

- one industry theme from `industry_map`;
- zero or more concept themes from PIT membership.

The scan should score industry and concept themes through the same local
`ThemeScore` contract, with additional metadata:

- `theme_type`: `industry`, `concept`, or `statistical`.
- `membership_source`: `industry_map`, `theme_membership.v1`, or
  `stat_cluster`.
- `pit_membership`: true for concept membership records with effective dates.

Primary-theme adjudication:

1. Keep all memberships in `symbol_theme_memberships` for audit.
2. Compute one `ThemeScore` per eligible industry/concept theme.
3. Default primary theme remains the existing industry theme.
4. A concept may become `symbol_primary_theme` only when all are true:
   - concept membership is active as of `as_of`;
   - concept member count passes `THEME_MIN_MEMBER_COUNT`;
   - concept score exceeds the industry theme score by
     `THEME_CONCEPT_PRIMARY_MARGIN` (proposed default `0.05` on normalized
     `[0, 1]` score);
   - concept is not in distribution phase.
5. Ties or missing concept metadata fall back to industry primary while keeping
   concept membership visible in metadata.

This keeps old consumers stable while making cross-industry concepts visible
for reports, calibration, and optional downstream consumers.

## Replay, Calibration, And Smoothing Compatibility

Historical compatibility:

- Old snapshots without concept fields load as industry-only.
- `schema_version` can remain additive under `theme_rotation.v1` if only fields
  are added; a new snapshot schema is required only if storage layout changes.
- `symbol_primary_theme`, `symbol_phase`, and `symbol_risk_flags` stay present.
- New fields should be optional:
  - `symbol_theme_memberships`
  - `theme_membership_source`
  - `theme_type`
  - `pit_membership`

Replay:

- Replays must evaluate active membership as of the snapshot `as_of`.
- Concept membership rows with future `effective_from` are excluded.
- Industry labels continue to carry the Phase 3 non-PIT note.
- Concept rows can set `pit_membership=true` only when sourced from
  `theme_membership.v1`.

Calibration:

- Existing phase, score-bucket, risk-flag, and threshold diagnostics can group
  by `theme_type`.
- Concept-vs-industry evidence should be reported separately until sample sizes
  are adequate.
- Threshold sweep can add concept-only rows after Phase 5b.

Smoothing:

- Smoothing keys remain `theme_id` based.
- `concept::...` histories are independent from `industry::...` histories.
- Missing concept history is diagnostic only; it must not be treated as
  confirmed theme persistence.

## Switches And Contract Tests

Proposed default-off switches:

- `THEME_CONCEPT_MEMBERSHIP_ENABLED=0`
- `THEME_CONCEPT_MEMBERSHIP_PATH=data/theme_membership.jsonl`
- `THEME_CONCEPT_MEMBERSHIP_REQUIRED=0`
- `THEME_CONCEPT_PRIMARY_MARGIN=0.05`
- `THEME_STAT_CLUSTER_ENABLED=0`

Contract-test impact:

- Add new default-off keys to `test_theme_default_off_contract.py`.
- Extend fail-safe tests for missing, malformed, future-dated, and duplicate
  JSONL records.
- Extend no-production-wiring tests to ensure concept membership is not a new
  canonical branch or Bayesian likelihood.
- Add replay tests proving `effective_from/effective_to` are respected.
- Add default-off scanner tests proving output is byte-for-byte compatible on
  existing industry-only inputs except additive neutral fields if approved.
- Add calibration tests separating `theme_type=concept` from industry rows.

## Decision Matrix

| Dimension | Route A: Statistical Clusters | Route B: PIT JSONL Membership |
| --- | --- | --- |
| Maintenance Cost | Low runtime cost, medium review cost for naming clusters | Medium to high ongoing curation cost |
| Look-Ahead Risk | Low if returns are strictly as-of; semantic labels can be post-hoc if not controlled | Low when effective dates are enforced; high if backfilled from current vendor concept lists |
| Coverage | Broad but noisy; every liquid symbol can be clustered | Sparse at first, improves with curated records |
| Determinism | High if clustering is dependency-free and seeded by sorted inputs | High with strict parser and deterministic dedupe |
| Explainability | Weak without manual labels | Strong; source evidence can be shown |
| Production Suitability | Discovery sidecar only at first | Recommended canonical route after approval |

## Recommendation

Use Route B as the canonical implementation route for named concept-level
themes, with Route A reserved as an offline discovery aid.

Rationale:

- The main problem is semantic cross-industry concept membership, not merely
  co-movement.
- Reports and daily review need explainable theme names and source evidence.
- PIT JSONL effective dates directly address the stated look-ahead redline.
- Route A can later suggest candidate memberships, but it should not create
  production concept labels without review.

Phase 5b should therefore implement only the approved subset of Route B first:
local parser, PIT active-membership resolver, default-off scanner integration,
metadata/reporting, and replay/calibration tests. Statistical clustering should
remain out of runtime scope unless separately approved.
