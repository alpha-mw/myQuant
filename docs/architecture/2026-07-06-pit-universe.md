# Point-In-Time CN Universe Design For Delisting Survivorship

Status: Phase 9a design approved. Phase 9b implementation was authorized on
2026-07-06 with a one-time online Tushare membership refresh window.

Historical note: this document was originally a STOP-gate design. Phase 9b was
not permitted until both approvals were explicit:

1. Design approval for this document.
2. One-time online backfill-window approval, because Phase 9b must call
   Tushare and therefore breaks the repository's offline default.

## Problem

The current CN universe is survivorship-biased in two important places:

- `quant_investor/market/download_cn.py` loads listing dates from
  `stock_basic(list_status="L")`, so only currently listed names are visible.
- `quant_investor/stock_database.py` initializes full-market metadata from
  `stock_basic(list_status="L")`, then filters ST / delisting-like names out.

`MarketDataReader.list_symbols("full_a")` also returns symbols present in the
current serving snapshot. That is correct for today's operational research, but
it is not a point-in-time historical universe. Factor backtests, theme replay,
calibration, and historical audits can therefore overstate results by excluding
dead names that would have been investable at the historical signal date.

Existing useful hooks already exist:

- `quant_investor/market/download.py` has a tactical `_load_inactive_symbols`
  helper that queries `L`, `D`, and `P`, but it is not canonical storage.
- `quant_investor/market/tushare_data_cleaning.py` already recognizes
  `stock_basic` fields including `list_date`, `delist_date`, and `list_status`.
- `quant_investor/factors/backtest.py` already honors `bundle.universe_mask`.
- `quant_investor/factors/tradability.py` already honors a `delisted` field.

## Goals

- Store a local PIT listing-membership table for CN symbols.
- Use the PIT table to answer `is_listed(symbol, date)` deterministically.
- Feed PIT membership into market maintenance, DAG universe construction,
  deterministic funnel gates, theme replay/calibration, and factor backtests.
- Keep all behavior default-off until tests and real backfill evidence exist.
- Fail closed for production-grade historical conclusions when PIT coverage is
  required but unavailable.

## Non-Goals

- No Phase 9a runtime code changes.
- No Phase 9a online Tushare calls.
- No deletion of existing current-serving bars or dashboard artifacts.
- No attempt to infer delist prices, liquidation prices, or synthetic bars from
  current data. Missing historical prices remain blockers or sensitivity inputs.

## Phase 9b Implementation Status

Phase 9b implements the PIT membership layer and default-off integration points:

- `quant_investor/market/pit_universe.py` stores and evaluates `L`, `D`, and
  `P` listing membership, with pure `listing_status`, `is_listed`,
  `build_pit_universe_mask`, and `build_pit_delisted_field` helpers.
- `scripts/refresh_pit_universe.py` is the explicit online refresh entrypoint.
  It is dry-run by default, requires `--allow-online`, and writes only when
  `--execute` is combined with `PIT_UNIVERSE_BACKFILL_ENABLED=1`.
- Since the 2026-07-16 lifecycle hardening, an executed refresh writes an
  immutable `_generations/<generation-id>/manifest.json` plus Parquet pair and
  advances only the discovery manifest. The legacy fixed Parquet bytes remain
  frozen for snapshots that already bind them.
- Production authority is the PIT generation recorded in active market
  coverage (`cn-full-a-coverage.v4`). Runtime, history audit, and repair never
  select production PIT from the discovery manifest.
- `MarketDataReader.list_symbols(..., as_of=date)` can filter by PIT membership
  when `PIT_UNIVERSE_ENABLED=1`; flags off preserve existing serving-inventory
  behavior.
- `market/dag_executor.py`, `funnel/candidate_filter.py`, and
  `market/download_cn.py` consume PIT metadata when enabled and keep current
  behavior when disabled.
- `stock_database.py` remains unchanged in Phase 9b because its `stock_list`
  table has no `delist_date` / `list_status` columns. The independent PIT
  reference store is the canonical historical membership source until a
  separate schema migration is approved.
- Full historical bar backfill for delisted symbols is not attempted in one
  step. Phase 9b records a deterministic cost estimate and lands the membership
  artifact needed to stage later date-scoped repairs.

Approved local refresh evidence from the 2026-07-06 online window:

- `source_run_id`: `pit-universe-20260706-ebe1b3e02745`
- `stock_basic` provider calls: 3 (`L`, `D`, `P`)
- Row counts: `L=5534`, `D=328`, `P=0`, total `5862`
- Local artifacts:
  `data/parquet/cn/reference/stock_basic_membership.parquet`,
  `data/parquet/cn/reference/stock_basic_membership_latest.json`,
  `data/cn_universe/stock_basic_membership_latest.json`,
  `data/cn_universe/raw/stock_basic_pit_snapshot_20260706_ebe1b3e02745.jsonl`,
  `results/pit_universe/refresh_execute_20260706.json`
- The artifact paths are under ignored `data/` / `results/` roots and are local
  evidence, not forced into the code commit.

## Source Pull And Landing Plan

Phase 9b should add a dedicated offline-capable module, tentatively
`quant_investor/market/pit_universe.py`, plus a small CLI/script entrypoint for
refresh and dry-run cost estimation.

Tushare source calls:

```text
stock_basic(exchange="", list_status="L",
            fields="ts_code,name,area,industry,market,list_date,delist_date,list_status")
stock_basic(exchange="", list_status="D",
            fields="ts_code,name,area,industry,market,list_date,delist_date,list_status")
stock_basic(exchange="", list_status="P",
            fields="ts_code,name,area,industry,market,list_date,delist_date,list_status")
```

Landing artifacts:

- Raw immutable snapshot:
  `data/cn_universe/raw/stock_basic_pit_snapshot_<observed_at>.jsonl`
- Frozen legacy derived table (read compatibility only):
  `data/parquet/cn/reference/stock_basic_membership.parquet`
- Immutable generation pair:
  `data/parquet/cn/reference/_generations/<generation-id>/manifest.json` and
  `stock_basic_membership.parquet`
- Latest discovery pointer / manifest (not production authority):
  `data/parquet/cn/reference/stock_basic_membership_latest.json`
- Compatibility export for inspection:
  `data/cn_universe/stock_basic_membership_latest.json`

Canonical schema:

```text
schema_version: "cn_pit_universe.v1"
symbol: str               # normalized ts_code, e.g. 000001.SZ
name: str
area: str | null
industry: str | null
board_market: str | null  # source "market" field, e.g. main board / ChiNext
source_list_status: str   # L, D, P
list_date: YYYYMMDD | ""
delist_date: YYYYMMDD | ""
effective_from: YYYYMMDD  # list_date when available
effective_to: YYYYMMDD | "" # delist_date when available
observed_at: ISO8601 UTC
source: "tushare.stock_basic"
source_run_id: str
raw_payload_hash: str
```

Normalization rules:

- Normalize `symbol` to uppercase `ts_code`.
- Keep `L`, `D`, and `P`; do not filter out ST or names containing delisting
  markers at storage time. Those belong in tradability/risk gates, not source
  membership.
- Deduplicate by `(symbol, source_list_status, observed_at)` in the raw layer
  and by `symbol` in the latest derived table, preferring the most informative
  row: non-empty `delist_date`, then `D`, then `P`, then `L`.
- If the same symbol appears in multiple statuses with contradictory dates,
  retain all raw rows and mark the derived row
  `membership_quality="conflicting_status_rows"`.

## PIT API

The public helper should expose both a boolean and a richer status:

```python
is_listed(symbol: str, date: str | date) -> bool
listing_status(symbol: str, date: str | date) -> PITListingStatus
listed_symbols(date: str | date, universe_key: str = "full_a") -> list[str]
```

Suggested `PITListingStatus` fields:

```text
symbol
date
in_universe: bool
research_eligible: bool
tradable: bool
reason: pre_listing | listed | pending | delisted | missing_pit_record | conflicting_status_rows
list_date
delist_date
source_list_status
observed_at
```

Boolean policy:

- `date < list_date` means not listed.
- Empty `list_date` means unknown; if PIT universe is required, fail closed.
- Non-empty `delist_date` and `date >= delist_date` means not listed for new
  research or buy decisions.
- `source_list_status == "P"` means in the historical universe after
  `list_date`, but not buyable/tradable unless a later implementation can prove
  the exact pause interval from exchange status data.
- Missing PIT record means fail closed when `PIT_UNIVERSE_REQUIRED=1`; otherwise
  record diagnostics and fall back to the current serving universe.

## Integration Points

Default-off environment switches:

```text
PIT_UNIVERSE_ENABLED=0
PIT_UNIVERSE_REQUIRED=0
PIT_UNIVERSE_SOURCE_ROOT=data/parquet/cn/reference
PIT_UNIVERSE_BACKFILL_ENABLED=0
```

`market maintain` / `download_cn.py`:

- Replace current active-only listing-date helper with PIT membership when
  enabled.
- Completeness checks should not count post-delist symbols as missing for a
  target date.
- Historical repair queues should include symbols that were listed on the
  target date, including currently delisted names, when historical bars are
  being backfilled.
- If PIT is enabled but missing for a target date, add `DataQualityIssue` and
  fail closed when required.

`MarketDataReader`:

- Add optional `as_of`/`trade_date` to `list_symbols`.
- With PIT enabled, `list_symbols("full_a", as_of=date)` should return
  `listed_symbols(date)` intersected with available bar coverage where the
  caller requests readable bars.
- Keep the current serving-inventory behavior when PIT is disabled.

DAG / GlobalContext:

- Build `global_context.universe_tiers["total"]` from PIT-listed symbols for
  the analysis date when enabled.
- Add metadata:
  `pit_universe_enabled`, `pit_universe_required`,
  `pit_universe_coverage_ratio`, `pit_universe_snapshot_id`,
  `pit_universe_missing_count`.
- Add missing PIT records to `data_quality_quarantine` when required.

Funnel gates:

- `DataQualityGate`: exclude symbols with `missing_pit_record` or
  `conflicting_status_rows` when required.
- `TradabilityGate`: exclude symbols whose `listing_status(...).tradable` is
  false, using reason `delisted`, `pre_listing`, or `pending`.
- `LiquidityGate`: unchanged; it should receive only PIT-valid candidates.

Theme replay / calibration:

- Use `listed_symbols(replay_date)` as the replay universe, then apply existing
  theme membership, smoothing, and calibration logic.
- Old replay artifacts without PIT metadata remain readable but must carry
  `pit_universe=false` and a survivorship warning.

`quant_investor/factors/backtest.py`:

- Populate `MatrixDataBundle.universe_mask[row][date]` from
  `is_listed(symbol, signal_date)`.
- Populate bundle field `delisted` and/or `tradability_mask` so the existing
  tradability audit path can block delisted execution dates.
- Record metadata on `SingleFactorBacktestRun` and aggregate results:
  `pit_universe_policy_version`, `pit_universe_snapshot_id`,
  `pit_universe_required`, and PIT coverage counts.
- If `PIT_UNIVERSE_REQUIRED=1` and coverage is incomplete, do not produce a
  production-grade backtest result.

## Historical Bar Backfill Strategy

Backfill should be staged and resumable. Phase 9b should not immediately try a
full-history rebuild.

1. Refresh and store PIT membership (`L`, `D`, `P`) only.
2. Produce an offline audit:
   - symbols currently absent from serving but PIT-listed on historical dates
   - dates where delisted or pending names create missing bar blockers
   - estimated API call budget before any provider call
3. Backfill by date-scoped all-market calls first:
   - `daily(trade_date=YYYYMMDD)`
   - `adj_factor(trade_date=YYYYMMDD)`
   - optional `daily_basic(trade_date=YYYYMMDD)` when existing canonical
     requirements need it
4. Use symbol-scoped calls only for tail repair where all-market date calls
   cannot recover a delisted name.
5. Promote snapshots only when strict storage validation and PIT coverage gates
   both pass.

Estimated provider call budget:

```text
stock_basic refresh calls = 3                         # L, D, P
date_scoped_bar_calls ~= missing_trade_dates * 3      # daily, adj_factor, daily_basic
symbol_tail_calls ~= unresolved_symbol_dates * 2      # daily, adj_factor fallback
total_calls ~= 3 + date_scoped_bar_calls + symbol_tail_calls
```

For the recent maintenance pattern with roughly 20 blocked historical trade
dates, the preferred date-scoped path would be approximately:

```text
3 stock_basic calls + 20 * 3 date-scoped calls = about 63 provider calls
```

Worst-case symbol-scoped repair is much larger:

```text
20 dates * 5500 symbols * 2 calls = about 220000 calls
```

That worst-case path should be rejected by dry-run cost gates. Actual Tushare
point cost and QPS limits are account-dependent and must be verified during the
approved online window. Phase 9b should use the existing configured Tushare rate
limit (`TUSHARE_RATE_LIMIT_PER_MIN`) and emit a `--dry-run-cost` report before
any write or backfill.

## Deferred-Backfill Sensitivity Alternative

If PIT membership is implemented but historical delisted bars are not yet
backfilled, existing historical results must remain non-production-grade. They
can still be reported with survivorship sensitivity bands:

1. For each signal date, compute:

```text
listed_count = count(PIT symbols listed on date)
served_count = count(symbols with readable bars on date)
delist_gap_count = listed_count - served_count
delist_gap_ratio = delist_gap_count / listed_count
```

2. Estimate the portfolio exposure that could have been assigned to missing
   names using the strategy's selection fraction and book size.
3. Report three adjusted return/alpha bands:
   - Base: missing names earn the same period's 5th-percentile forward return
     among served names.
   - Severe: missing names take a -30% one-period delisting shock.
   - Extreme: unresolved delisted holdings take a -100% terminal loss.
4. Apply the penalty only to historical evidence and calibration summaries; do
   not mutate original backtest ledgers.
5. Display both gross reported metrics and sensitivity-adjusted metrics, with
   metadata `survivorship_sensitivity_only=true`.

This is a temporary disclosure method, not a substitute for PIT bars.

## Migration Plan

1. Implement PIT schema/types and pure `is_listed` tests from synthetic rows.
2. Implement local store read/write and `_latest` manifest tests.
3. Implement `stock_basic` online refresh behind
   `PIT_UNIVERSE_BACKFILL_ENABLED=1`; dry-run by default.
4. Add dry-run cost estimator and fail closed when estimated calls exceed the
   approved window.
5. Wire `market maintain` completeness logic in default-off mode.
6. Wire `MarketDataReader`, DAG metadata, funnel gates, theme replay, and factor
   backtests in default-off mode.
7. Run one approved online stock-basic refresh only; review artifacts.
8. Run approved staged historical bar backfill only after cost report approval.
9. Flip `PIT_UNIVERSE_ENABLED=1` in shadow/reporting mode.
10. Flip `PIT_UNIVERSE_REQUIRED=1` only after replay/backtest contracts and
    storage validation are green.

## Rollback Plan

- Set `PIT_UNIVERSE_ENABLED=0` and `PIT_UNIVERSE_REQUIRED=0`.
- Keep the PIT artifacts on disk for auditability; do not delete raw snapshots.
- Restore `MarketDataReader.list_symbols` behavior to current serving inventory.
- Keep backtest/replay artifacts readable; mark any PIT-enabled run metadata so
  mixed-result comparisons are explicit.
- If a bad PIT snapshot is promoted, roll back only the
  `stock_basic_membership_latest.json` pointer to the prior manifest.

## Acceptance Criteria For Phase 9b

- No behavior change with all PIT flags off.
- Synthetic PIT fixture covers active, pre-listing, delisted, pending, missing,
  and conflicting rows.
- `is_listed` and `listing_status` are pure and deterministic.
- PIT mask helpers plus `MatrixDataBundle.universe_mask` and tradability
  `delisted` tests prove that factor backtests can exclude pre-listing and
  post-delist cells when the PIT layer is explicitly injected.
- DAG/funnel tests prove PIT-missing records are quarantined when required.
- Old snapshots and old backtest artifacts remain readable.
- Online tests are not part of normal CI; provider calls require the explicit
  approved backfill window.
