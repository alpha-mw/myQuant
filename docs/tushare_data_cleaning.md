# Tushare Daily Data Cleaning and Factor Readiness

This upgrade inserts an offline cleaning layer into the CN Tushare daily
download path. It is data hygiene only: it does not change stock selection,
posterior scoring, RiskGuard, PortfolioConstructor, target weights, orders,
LLM behavior, broker behavior, frontend behavior, or factor admission.

## Runtime Behavior

`CNFullMarketDownloader.download_stock` assembles the merged daily DataFrame,
then runs the cleaner before writing Parquet canonical bars. The raw merged frame
is written first under `data/raw_backups/tushare`. If cleaning fails, the
existing Parquet canonical snapshot remains unchanged and a failure report is
written.

Parquet is the default storage contract. The cleaned DataFrame reaches the
canonical store only after structural validation, row quarantine,
de-duplication, sorting, and report writes succeed. The cleaner never
forward-fills OHLCV, never synthesizes missing trading rows, and never
winsorizes, neutralizes, ranks, or otherwise transforms factor values.

## Artifacts

- Raw backups: `data/raw_backups/tushare/<table>/`
- Cleaning reports and row/cell flags: `data/cleaning_reports/tushare/<table>/`
- Quarantined rows: `data/quarantine/tushare/<table>/`
- Factor-readiness reports, masks, and coverage summaries:
  `data/factor_readiness/tushare/<table>/`
- Storage audit and Parquet migration reports:
  `data/cleaning_reports/tushare/<table>/storage/`
- Parquet canonical bars: `data/parquet/CN/bars/` and
  `data/parquet_serving/CN/bars/`

Row flags mark row-level invalid dates, invalid symbols, invalid prices,
invalid OHLC relations, negative volume/amount, duplicate keys, quarantine, and
drop status. Cell flags preserve cell-level invalid values without fabricating
repairs.

Factor-ready masks are symbol x date matrices for `has_row`, `valid_ohlc`,
`valid_volume`, `valid_amount`, `tradable`, `factor_eligible`,
`benchmark_member`, `index_weight_available`, and `adjusted_price_ready`.
Missing tradability or benchmark sources remain explicit false/warning states;
the cleaner does not fabricate tradability.

Matrix coverage summaries report expected, observed, missing, and quarantined
symbol-date cells plus field coverage ratios. A cleaning pass is not a
factor-readiness pass: missing trade calendar, adjusted factors, limit data,
suspend data, or benchmark membership can keep readiness at `not_ready`.

## Storage Audit and Parquet

The storage audit records backend support and whether matrix-style tables such
as `daily`, `adj_factor`, `daily_basic`, and `index_weight` are ready for
Parquet factor research. Production reads fail closed when Parquet snapshots or
manifests are missing.

Parquet writes are handled by `MarketDataStore`; no CSV fallback is allowed for
runtime reads. CSV outputs from this layer are limited to audit/report artifacts
or explicit migration work.

## Environment Flags

- `MYQUANT_TUSHARE_AUTO_CLEAN=0` disables the automatic post-download cleaner.
- `MYQUANT_TUSHARE_FACTOR_READINESS=0` disables readiness sidecars.
- `MYQUANT_TUSHARE_STORAGE_AUDIT=0` disables storage audit reports.
- `MYQUANT_TUSHARE_PARQUET_SHADOW_WRITE=1` is retained for old audit reports;
  canonical runtime writes use `MarketDataStore`.
- `MYQUANT_TUSHARE_PARQUET_CANONICAL=1` is a legacy compatibility flag; runtime
  reads are Parquet-only.
- `MYQUANT_TUSHARE_DELETE_REDUNDANT_CSV=0` keeps legacy CSV artifact deletion
  disabled.

Directory flags:

- `MYQUANT_TUSHARE_CLEANING_REPORT_DIR`
- `MYQUANT_TUSHARE_RAW_BACKUP_DIR`
- `MYQUANT_TUSHARE_QUARANTINE_DIR`
- `MYQUANT_TUSHARE_FACTOR_READINESS_DIR`
- `MYQUANT_TUSHARE_PARQUET_DIR`
- `MYQUANT_TUSHARE_PARQUET_COMPRESSION`

## Offline Utility

Run a local Parquet cleanup pass without live provider calls:

```bash
./.venv/bin/python scripts/clean_tushare_downloads.py \
  --root-dir data/cn_market_full \
  --table daily
```

Use `--no-promote` to generate reports without writing canonical storage. Tests
for this layer use local temporary fixtures only and do not call Tushare,
yfinance, LLM, broker, or provider APIs.
