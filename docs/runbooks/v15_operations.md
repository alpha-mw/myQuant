# myQuant v15 Operations

This is the current operating contract for the no-Theme v15 mainline.  The
system remains offline by default and fail closed for new risk.

The additive `quant-investor-v17-v3` research entrypoint is isolated from this
runbook. It does not change any V15 command, artifact, Dashboard consumer,
schedule or production/default authority.

## Active architecture

- Canonical branches: `quant`, `fundamental`, `macro`.
- Bayesian likelihoods: `quant`, `fundamental`; Macro is control context.
- Factor-attested runtime path: `Quant -> DeterministicFunnel -> Bayesian ->
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

`branch_data_ready` green means only that the three canonical data branches
`quant`, `fundamental` and `macro` passed. Factor governance remains an
independent gate. A valid human receipt is also only one necessary gate: it
does not override data, Factor, candidate or portfolio blockers and never
automatically changes `new_risk_authorized` to true.

Risk reduction is a separate sell-only advisory/manual lane.  It requires an
existing holding, a same-day fresh quote and a same-day human action.  It never
authorizes a BUY, broker call or real order.

## Data commands

- Market maintenance remains `market maintain`; `market run` never refreshes
  data.
- CN `parquet-direct` keeps Tushare as the primary daily-bar source. An
  explicitly supplied `--secondary-daily-source eastmoney` may probe only
  PIT-active unresolved symbols on the exact target date. The probe writes
  `cn-secondary-daily-evidence.v1` with raw-response bytes/SHA, exact URL and
  parameters, normalized-row SHA, query-run ID, and PIT binding. It can add
  only validated exact-date bars; an empty or malformed response leaves
  `true_missing` blocking, and it never classifies suspension, writes a
  synthetic bar, or enables staged maintenance. The default is `none`.
- Fundamental authoritative rebuild remains the existing two-stage
  `fundamental-maintain --allow-live --authoritative-full-rebuild` followed by
  `fundamental-promote --expected-pointer-sha256` workflow.
- Macro authoritative publication has independently CAS-bound observation,
  release-calendar and mart layers. First compile and atomically publish a
  reviewed canonical `macro_observations` generation with
  `macro-official-compile` and `macro-observation-bootstrap`; later exact-scope
  chains use `macro-observation-official-refresh` and
  `macro-local-observation-roll`. Publish the immutable official release
  calendar with `macro-release-calendar-publish`. Then run `market
  macro-maintain --authoritative-refresh` with the exact catalog, market,
  observation and release-calendar pointer identities. The stage writes an
  immutable MacroSnapshot and v15 control projection; `market macro-promote
  --expected-catalog-sha256` revalidates the stage and performs the four-way
  cutover under ordered locks. Local `--input-json` and
  `--input-observations` modes remain non-production and unchanged.
- Factor bootstrap generation is plan-only.  It cannot write the registry,
  release a kill switch or create an activation receipt.

Fundamental readiness separates structural field coverage from observed-value
availability.  The eight v15 fields must exist in every canonical record, but
their numeric values remain nullable: cash-flow/profit ratios are undefined
without a positive profit denominator, and a verified provider may return no
qualifying forecast.  Only a verified primary v3 generation with Gate2,
checkpoint, derivation and endpoint-audit closure may accept those nulls.
Readiness reports `full_field_coverage_ratio` for the structural contract and
`full_value_observation_ratio` plus per-field counts as diagnostics.  Missing
records, missing columns, records with no usable required value, or unverified
generations still fail closed; never synthesize zeroes or ratios to improve a
coverage number.

Live Fundamental rebuild/promotion, live Macro staging/promotion, Factor bootstrap
apply, Macro activation and private Theme evidence purge each require a
separate explicit authorization.

## Macro v15 cutover

The observation generation is an upstream evidence source, not a control by
itself. It must come from a replayable official NBS/PBC web bundle plus an
explicit cutoff-safe strict-Parquet market observation, or from another
reviewed hash/CAS-bound publisher. Standalone observation files and sanitized
observer staging are ineligible.

A production schema label is never sufficient. Bootstrap and local-update
metadata, immutable parent lineage, exact official/local row scope, evidence
roles, and every observation-to-evidence mapping are recomputed before the
generation may feed v15. Logical `as_of` and real `decision_cutoff_at` are
independent: selected rows must satisfy both `period_end <= as_of` and
`available_at <= decision_cutoff_at`, and an update cannot regress its parent
target date or cutoff.

The release calendar is a separate immutable, prefix-extending generation
store. Its plan, capture manifest, raw issuer bytes, pinned
`market-open-days.v1`, registry and code-owned critical policy are hash-bound;
its `_latest.json` advances only by an explicit expected-pointer CAS. Macro may
lag the target by zero, one or two sessions in that pinned CN open-day list.
For any positive lag, no critical official event may fall in the gap from the
Macro logical date's Shanghai close through the decision cutoff: an event in
that gap blocks even if already resolved. Lag zero still requires pinned
evidence and complete resolution; a lag above two also blocks.

The current bootstrap compiles 36 observations for 12 official indicators
(three periods each) directly from fixed National Bureau of Statistics and
People's Bank of China response entities. It adds three independently bound
`market.breadth` observations for three ascending trade dates, giving 39 rows
and 13/16 national indicators (`81.25%`). Rounded cumulative social-financing
values are never differenced into a monthly flow; fiscal expenditure and
market volatility observations remain absent rather than inferred. Raw HTML,
each exact market-snapshot manifest, the matching closing-coverage manifest,
the hash-bound full-A scope artifact and the exact Parquet part bytes are
copied into the immutable observation generation and mapped to every
observation.

The local binding plan is strict JSON using
`cn-local-breadth-bootstrap-plan.v1`. Its root contains exactly
`schema_version`, `market` (`CN`) and `targets`. `targets` contains exactly
three entries in ascending `target_trade_date` order; every entry contains
exactly these seven fields:

```json
{
  "target_trade_date": "YYYYMMDD",
  "snapshot_manifest_path": "<immutable_snapshot_manifest>",
  "expected_snapshot_manifest_sha256": "<sha256>",
  "coverage_manifest_path": "<closing_coverage_manifest>",
  "expected_coverage_manifest_sha256": "<sha256>",
  "scope_artifact_path": "<full_a_scope_artifact>",
  "expected_scope_artifact_sha256": "<sha256>"
}
```

Each target has its own exact closing-coverage proof. A later snapshot or one
coverage manifest reused as evidence for an earlier sparse date is rejected.
The final target date must equal the Asia/Shanghai calendar date represented
by bootstrap `--as-of`; the daily target date must likewise equal daily
`--as-of`.

Compile the reviewed official capture without a canonical write, then publish
the complete bootstrap with one expected-pointer CAS:

```bash
./.venv/bin/quant-investor market macro-official-compile \
  --plan <official_plan.json> \
  --capture-manifest <capture_manifest.json> \
  --raw-root <immutable_raw_capture_root> \
  --output-dir results/v15/macro_official_web \
  --run-id <official_bundle_id>

./.venv/bin/quant-investor market macro-observation-bootstrap \
  --official-manifest <official_bundle>/normalization_manifest.json \
  --expected-official-manifest-sha256 <official_manifest_sha256> \
  --expected-official-plan-sha256 <official_plan_sha256> \
  --local-binding-plan <cn_local_breadth_bootstrap_plan.json> \
  --expected-local-binding-plan-sha256 <binding_plan_sha256> \
  --as-of <latest_target_trade_date> \
  --observations-root data/parquet/cn/macro_observations \
  --run-id <observation_generation_id> \
  --expected-pointer-sha256 EMPTY
```

Later daily maintenance may append one fully bound local breadth observation
without republishing the official response entities:

```bash
./.venv/bin/quant-investor market macro-local-observation-publish \
  --snapshot-manifest <pre_refresh_snapshot_manifest> \
  --expected-snapshot-manifest-sha256 <snapshot_manifest_sha256> \
  --coverage-manifest <target_date_closing_coverage_manifest> \
  --expected-coverage-manifest-sha256 <coverage_manifest_sha256> \
  --target-trade-date <maintenance_trade_date> \
  --scope-artifact <target_date_full_a_scope_artifact> \
  --expected-scope-artifact-sha256 <scope_artifact_sha256> \
  --as-of <maintenance_trade_date> \
  --observations-root data/parquet/cn/macro_observations \
  --run-id <local_observation_generation_id> \
  --expected-pointer-sha256 <current_observation_pointer_sha256>
```

After cutover to the exact 39-row projection chain, refresh all 36 official
rows while retaining the latest three consecutive local rows with:

```bash
./.venv/bin/quant-investor market macro-observation-official-refresh \
  --official-manifest <official_bundle>/normalization_manifest.json \
  --expected-official-manifest-sha256 <official_manifest_sha256> \
  --expected-official-plan-sha256 <official_plan_sha256> \
  --target-as-of <logical_target_date> \
  --decision-cutoff-at <timezone_aware_cutoff> \
  --market-open-days <market_open_days.json> \
  --expected-market-open-days-sha256 <market_open_days_sha256> \
  --observations-root data/parquet/cn/macro_observations \
  --run-id <official_refresh_generation_id> \
  --expected-pointer-sha256 <current_observation_pointer_sha256>
```

Catch up or roll the three-row local window without growing the generation:

```bash
./.venv/bin/quant-investor market macro-local-observation-roll \
  --snapshot-manifest <pre_refresh_snapshot_manifest> \
  --expected-snapshot-manifest-sha256 <snapshot_manifest_sha256> \
  --coverage-manifest <target_date_closing_coverage_manifest> \
  --expected-coverage-manifest-sha256 <coverage_manifest_sha256> \
  --target-trade-date <local_target_trade_date> \
  --scope-artifact <target_date_full_a_scope_artifact> \
  --expected-scope-artifact-sha256 <scope_artifact_sha256> \
  --target-as-of <logical_target_date> \
  --decision-cutoff-at <timezone_aware_cutoff> \
  --market-open-days <market_open_days.json> \
  --expected-market-open-days-sha256 <market_open_days_sha256> \
  --observations-root data/parquet/cn/macro_observations \
  --run-id <local_roll_generation_id> \
  --expected-pointer-sha256 <current_observation_pointer_sha256>
```

Both commands preserve an exact 39-row, 13-indicator, three-history scope and
read back the declared parent lineage. `macro-local-observation-publish`
remains the legacy append command for its existing chain; it is not a
substitute for the exact-scope roll.

Local `available_at` is the maximum of the market and coverage snapshot times,
both stable manifest mtimes, the full-A scope artifact mtime and the selected
Parquet part mtime, rounded upward to microseconds. It must remain at or before
the Shanghai 15:00 cutoff. The bootstrap receipt includes
`local_bootstrap_plan_sha256`, `local_target_trade_dates`,
`local_snapshot_manifest_sha256s`, `local_coverage_manifest_sha256s`,
`local_scope_artifact_sha256s`, `local_coverage_contract_sha256s` and
`local_effective_available_at_values`. The daily receipt exposes the same
binding as scalar `local_target_trade_date`,
`local_snapshot_manifest_sha256`, `local_coverage_manifest_sha256`,
`local_scope_artifact_sha256`, `local_coverage_contract_sha256` and
`local_effective_available_at`. Capture these four hashes before Macro
staging:

```bash
catalog_sha256=<sha256 data/parquet/cn/_catalog.json>
market_pointer_sha256=<sha256 data/parquet/cn/_latest.json>
macro_observations_pointer_sha256=<sha256 data/parquet/cn/macro_observations/_latest.json>
macro_release_calendar_pointer_sha256=<sha256 data/parquet/cn/macro_release_calendar/_latest.json>
```

Then stage and promote with separate commands:

```bash
./.venv/bin/quant-investor market macro-maintain \
  --market CN \
  --as-of <latest_complete_trade_date> \
  --authoritative-refresh \
  --canonical-root data/parquet/cn/macro_daily \
  --observations-root data/parquet/cn/macro_observations \
  --staging-root results/v15/macro_observation_staging \
  --run-id <new_generation_id> \
  --expected-catalog-sha256 "$catalog_sha256" \
  --expected-market-pointer-sha256 "$market_pointer_sha256" \
  --expected-macro-observations-pointer-sha256 "$macro_observations_pointer_sha256" \
  --macro-release-calendar-root data/parquet/cn/macro_release_calendar \
  --expected-macro-release-calendar-pointer-sha256 "$macro_release_calendar_pointer_sha256" \
  --nbs-cn-pmi-url <issuer_bound_nbs_release_https_url> \
  --allow-live

./.venv/bin/quant-investor market macro-promote \
  --staging-root results/v15/macro_observation_staging/<new_generation_id> \
  --canonical-root data/parquet/cn/macro_daily \
  --expected-catalog-sha256 "$catalog_sha256"
```

`--as-of` is the market target date. The production observation generation
may carry a logical `as_of` behind that target by zero, one, or two pinned CN
open sessions, but only when the staged release-readiness evidence proves the
same lag and an empty critical-event gap. The observation logical date is not
silently advanced to the market date, and a lagged generation without that
four-way hash-bound readiness evidence remains ineligible.

The projection takes `macro_score`, `liquidity_score` and `policy_signal` from
the ready, at-least-80%-covered MacroSnapshot domains. Only market volatility
percentile remains derived from the exact bound bars generation. The stage
persists and hash-binds `macro_snapshot.json` and `v15_controls.json` through
the generation manifest, primary provenance and strict catalog. Production
readers and the active DAG recompute this projection and reject any row or hash
mismatch. A v14 mart is historical evidence only and remains fail closed; it
must never be relabelled or copied into a v15 manifest.

The staging receipt binds all four identities: catalog bytes, market pointer,
Macro-observations pointer and release-calendar pointer/generation. Promotion
reacquires locks in the fixed order market writer -> catalog writer -> Macro
observation writer -> release-calendar writer, then re-reads all four before
the catalog switch; any drift leaves the prior catalog authoritative. The
retired one-step live writer always fails with
`macro_authoritative_stage_promotion_required`; production must use the
separate stage/promote commands.

Bars may advance before Macro for the same session. The bars writer performs a
bars/PIT-only postcommit readback so a stale Macro generation cannot deadlock
the cutover. Full `storage-validate` remains blocked until the matching v15
Macro generation is promoted, and must pass after promotion before analysis.

At the start of each DAG run, runtime captures one release-calendar pointer,
loads that immutable generation once and constructs one hash-bound
`MacroReadinessEvidence`. The same object and cutoff are reused for Macro
assessment, three-branch readiness and `v15_run_readiness`; the run never
reloads a newer calendar or infers a second freshness decision mid-flight.

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
