# Macro v2 Observer

Macro v2 is a measurement-only, point-in-time observer introduced during the
v13.1 incubation freeze. It does not replace the production `MacroAgent`, alter
Bayesian inputs, change candidate scores, loosen risk limits, or write portfolio
weights.

## Safety status

- `MACRO_V2_OBSERVER_ENABLED=0` by default.
- `MACRO_V2_OBSERVER_KILL_SWITCH=1` by default and takes precedence.
- `MACRO_V2_PRODUCTION_ENABLED=0` and
  `MACRO_V2_PRODUCTION_KILL_SWITCH=1` are disclosure-only in this phase; no
  production mutation path exists.
- Hypothetical industry adjustments are bounded to `[-5,+5]`, persisted with
  `applied=false`, and are not attached to any production score surface.
- Provider transports are dependency-injected and opt-in. The CLI does not
  discover credentials, infer publication timestamps, or perform an implicit
  network fallback. Schedules and production activation remain deferred.

## Immutable observation store

Validated local observations can be appended to a cumulative immutable
generation:

```bash
quant-investor market macro-maintain \
  --market CN \
  --as-of 2024-05-10 \
  --input-observations private/macro/observations.jsonl \
  --observations-root data/parquet/cn/macro_observations \
  --run-id local_20240510
```

Each observation must carry timezone-aware `release_at`, `available_at`, and
`fetched_at`, plus indicator, period, vintage, unit, frequency, source, and
source-record provenance. The semantic content hash is recomputed. Exact
duplicates are idempotent; conflicting natural identities fail closed.

The store writes complete cumulative generations under `_generations/` and
advances `_latest.json` only after Parquet and manifest readback/hash checks.
The pointer binds both files and records the parent generation. Corruption,
schema drift, a compare-and-swap race, malformed input, or an unsafe symlink
leaves the canonical pointer unchanged.

## Offline workflow

Build a report from a local JSON, JSONL, or Parquet observation file:

```bash
quant-investor market macro-analyze \
  --market CN \
  --as-of 2024-05-10 \
  --observations private/macro/observations.parquet
```

`--observations` may also point to the immutable observation-store root. The
observer pins and discloses one validated generation; the DAG never calls a
provider or advances the store.

Date-only `--as-of` values use the Asia/Shanghai 15:00 close as the published
cutoff. An observation is usable only when its timezone-aware `available_at`
is at or before that cutoff. `period_end` and `release_at` are not substitutes
for availability.

Generated, ignored artifacts are written atomically with mode `0600` under:

```text
results/macro_observer/CN/<as_of>/<snapshot_hash>/<generation_id>/
```

For legacy standalone files without generation provenance, the final
`<generation_id>` component is omitted.

The directory contains `macro_snapshot.json`, `macro_readiness.json`,
`macro_report.md`, and `macro_observations_manifest.json`. The snapshot hash
excludes generated time and paths and includes the cutoff, registry/model
versions, and every selected PIT observation hash.

## Compatibility mart maintenance

The legacy four-field macro mart remains available for current consumers:

```bash
quant-investor market macro-maintain \
  --market CN \
  --as-of 2024-05-10 \
  --input-json private/macro/compatibility_row.json
```

The input must contain `macro_score`, `liquidity_score`,
`volatility_percentile`, and `policy_signal`. Maintenance writes an immutable
generation and advances a single atomic pointer only after schema, readback,
range, and hash validation. Empty input, unimplemented live access, older
snapshots, invalid schema, and interrupted writes leave the last-good pointer
unchanged.

For observation maintenance, `--allow-live` remains blocked unless a reviewed
provider transport is injected by the caller. `--allow-tushare-fallback` is a
second explicit permission and applies only to indicators missing from the
official result. Official and fallback values are never averaged; official
provenance wins for the same indicator and period.

## PIT replay

Replay requires a local strict-Parquet trading calendar containing `cal_date`
and `is_open`:

```bash
quant-investor market macro-replay \
  --market CN \
  --start-date 2024-01-01 \
  --end-date 2024-05-10 \
  --observations-root data/parquet/cn/macro_observations \
  --calendar private/calendars/sse_trade_cal.parquet
```

The run reads and validates the latest observation generation once, then uses
that pinned in-memory set for every open date. Outputs are private mode `0600`
files under `results/macro_replay/CN/<run_id>/`: `daily_snapshots.parquet`,
`replay_manifest.json`, and `replay_report.md`. The manifest binds the
observation generation/content set, pointer hash, calendar hash, registry/model
versions, date range, and output hash. Every row is `observer_only=true`,
`production_eligible=false`, and `applied=false`.

## Offline Tushare raw normalization

Phase 3 adds an offline evidence compiler for documented Tushare macro tables.
It does not import or call the Tushare client. The initial whitelist is:

| Endpoint | Raw field | Indicator | Unit |
| --- | --- | --- | --- |
| `cn_gdp` | `gdp_yoy` | `cn.gdp_yoy` | `%` |
| `cn_cpi` | `nt_yoy` | `cn.cpi_yoy` | `%` |
| `cn_ppi` | `ppi_yoy` | `cn.ppi_yoy` | `%` |
| `cn_m` | `m1_yoy` | `cn.m1_yoy` | `%` |
| `cn_m` | `m2_yoy` | `cn.m2_yoy` | `%` |
| `sf_month` | `inc_month` | `cn.social_financing_flow` | `CNY_100M` |
| `cn_pmi` | `pmi010000` / `PMI010000` | `cn.pmi_manufacturing` | `index` |

Normalization requires three separate local files: a raw Tushare response using
`macro-tushare-raw-bundle.v1`, an operator-approved scope using
`macro-backfill-plan.v1`, and an official-page capture using
`macro-release-evidence-capture.v1`. The plan is not accepted from the raw
bundle. All three source files are copied into the immutable normalization
bundle and byte-hash bound.

Availability evidence requires `time_precision=timestamp`, timezone-aware
`release_at` and `available_at`, an allowlisted official source, record ID, and
an issuer-bound HTTPS URL. GDP/CPI/PPI/PMI evidence is restricted to NBS;
money-supply and social-financing evidence is restricted to PBOC. Tushare
`cn_schedule` contains a publication date but no time. It is retained as
date-level evidence only and cannot make a row promotable. Missing, conflicting,
date-only, future, pre-period, malformed, unexpected, or incomplete scope is
quarantined and blocks publication.

Prepare a private evidence bundle without advancing canonical storage:

```bash
quant-investor market macro-normalize-tushare \
  --market CN \
  --input-json private/macro/tushare_raw_bundle.json \
  --plan-json private/macro/backfill_plan.json \
  --evidence-json private/macro/release_evidence_capture.json \
  --run-id backfill_2024q1
```

Outputs are staged atomically with directories mode `0700` and files mode
`0600` under `results/macro_normalization/CN/<run_id>/`. The raw response,
backfill plan, evidence capture, observations, quarantine, receipts, and
manifest are byte-hash bound.
Inputs containing credential-like keys, URL userinfo, or credential query
parameters are rejected before any source bytes are persisted.

Only a zero-quarantine, exact-scope bundle can be explicitly published. The
caller must bind the current observation pointer with compare-and-swap:

```bash
quant-investor market macro-backfill-publish \
  --market CN \
  --manifest results/macro_normalization/CN/backfill_2024q1/normalization_manifest.json \
  --observations-root data/parquet/cn/macro_observations \
  --run-id macro_backfill_2024q1 \
  --expected-pointer-sha256 EMPTY \
  --expected-manifest-sha256 <normalization-manifest-sha256> \
  --expected-plan-sha256 <backfill-plan-sha256>
```

Use `EMPTY` only for the first generation; otherwise pass the exact SHA-256 of
the current `_latest.json`. Artifact tampering, missing periods, receipt drift,
quarantine rows, or a pointer race leaves the canonical pointer unchanged. The
publisher recompiles the saved raw/plan/evidence inputs and recomputes scope,
counts, receipts, quarantine, and observations instead of trusting manifest
claims.

The remaining nine national registry indicators, official-site raw parsers,
and all twelve industry-chain raw mappings remain explicitly unsupported until
real redacted fixtures and timestamp-level publication evidence exist.

## Forward observation ledger

Phase 4 adds a forward-only evidence clock. It records the latest trading
session whose 15:00 Asia/Shanghai close has actually passed according to the
process clock and a strict local Parquet calendar:

```bash
quant-investor market macro-observe-forward \
  --market CN \
  --calendar private/calendars/sse_trade_cal.parquet \
  --observations-root data/parquet/cn/macro_observations \
  --state-root results/macro_forward_observation \
  --expected-pointer-sha256 EMPTY
```

Use `EMPTY` only for enrollment. Every later run must provide the current
forward-ledger `_latest.json` SHA-256. A same-session run is idempotent only
when both the snapshot and observation generation are unchanged. A changed
same-session result, pointer race, stale calendar, skipped open session,
artifact corruption, or symlink blocks the append. There is no CLI parameter
for pretending an earlier capture time and no historical backfill path.

Each immutable generation contains a cumulative hash-chained `ledger.jsonl`,
`summary.json`, and `manifest.json`, with private mode `0600` files behind an
atomic pointer. Events bind the snapshot, selected observations, observation
generation/pointer, strict calendar, readiness, coverage, blockers, and actual
recording time.

Reaching 90 sequential entries sets only `forward_duration_reached=true`.
`measurement_maturity_reached`, `production_eligible`,
`activation_authorized`, and `applied` remain false because outcome/stability
evidence and an authoritative production score are not implemented. Degraded
sessions remain visible as readiness gaps rather than disappearing from the
clock.

## Coverage and acquisition audit

Phase 5 exposes the gap between locally cached values and genuinely usable PIT
evidence. It audits all 16 national indicators and all 96 industry components
(12 chains × 8 components):

```bash
quant-investor market macro-coverage-audit \
  --market CN \
  --as-of 2026-07-14 \
  --observations data/parquet/cn/macro_observations \
  --raw-root data/parquet/cn/dag_core_raw \
  --output-dir results/macro_coverage_audit
```

The raw inventory currently recognizes only the reviewed Phase 3 endpoint
whitelist. A local raw table, value, source label, snapshot ID, or `fetched_at`
does not prove when the statistic became available. Such rows are reported as
`raw_present_pit_evidence_missing` and never counted as PIT coverage.

National indicators are classified as `pit_signal_ready`,
`observation_present_not_signal_ready`, `raw_present_pit_evidence_missing`,
`mapped_raw_not_usable_as_of`, `mapped_raw_missing`, or
`mapping_not_implemented`. Every industry chain is
expanded into output, orders, inventory, price/margin, profits, capacity
utilization, capex, and exports, so missing components cannot disappear inside
an aggregate chain score. Unconfirmed industry authorities are written as
`UNCONFIRMED` rather than inferred.

Private mode-`0600` outputs contain `coverage_audit.json`, detailed national
and industry CSVs, a Markdown report, and a hash-bound manifest under
`results/macro_coverage_audit/CN/<as_of>/<audit_hash>/`. Identical reruns are
idempotent only after artifact hash readback. The audit is always observer-only
and cannot publish observations or activate production scoring.

## Offline acquisition plan

Phase 6 converts one complete, hash-valid coverage audit into a deterministic
official-data acquisition contract. It does not fetch data and never emits an
observation:

```bash
quant-investor market macro-acquisition-plan \
  --market CN \
  --coverage-audit results/macro_coverage_audit/CN/<as_of>/<audit_hash>/coverage_audit.json \
  --output-dir results/macro_acquisition_plan
```

The input must contain exactly all 16 national registry indicators and all
12x8 industry components. A partial but internally re-hashed audit is rejected.
The planner maps each coverage status to explicit work:

- existing raw without PIT evidence -> `bind_timestamp_release_evidence`;
- mapped but absent raw -> `acquire_raw_and_release_evidence`;
- missing official mapping -> `implement_official_mapping`;
- local market confirmation -> `build_local_strict_parquet_observation`;
- industry authority not established -> `confirm_authority_and_mapping`;
- PIT-ready observations -> `none`, retained as `satisfied`.

NBS, PBOC, Customs, MOF, and NDRC routes have explicit issuer domains. Industry
authority remains `UNCONFIRMED` until reviewed source ownership is established.
Every open official task requires an immutable raw-capture SHA-256, an
issuer-bound HTTPS URL, a source record ID, timezone-aware `release_at` and
`available_at`, capture no earlier than availability, exact
period/value/unit/frequency, and a zero-quarantine recompile.

Private mode-`0600` JSON, CSV, Markdown, and hash-manifest outputs are written
under `results/macro_acquisition_plan/CN/<as_of>/<plan_hash>/`. Reruns are
idempotent and artifact drift fails closed. All outputs retain
`production_eligible=false`, `activation_authorized=false`, `applied=false`,
and `observation_count=0`.

## Deferred production gates

Production scoring remains blocked until a separate reviewed change establishes
an authoritative 0–100 score surface, 90 trading days of
forward observation, no-leakage and stability evidence, Architect/Critic
approval, and Maxwell's explicit merge/activation confirmation.
