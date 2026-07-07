# Web Data Service Split Notes

Scope for a future split of `web/services/data_service.py`.

## Current Responsibility Clusters

- Local storage bootstrap: SQLite connection setup, schema creation,
  stock-list indexing, and missing-data download coordination.
- Metadata enrichment: CN/Tushare metadata, US/yfinance metadata, local metadata
  cache, and stock-list upserts.
- Research cache: profile, snapshot, time-series, peer relationships, provider
  fetches, cache freshness, and research bundle assembly.
- Web payload builders: market pulse, stock search/detail/dossier/overview,
  factor signals, key metrics, completeness, and display formatting.
- Public service API: `get_statistics`, `get_market_overview`, `get_stocks`,
  `get_stock_detail`, `get_stock_dossier`, `get_stock_overview`, `get_ohlcv`,
  `import_csv_data`, and `get_competitors`.

## Proposed Package Shape

- `web/services/data_store.py` for connection, schema, local indexing, and
  low-level JSON helpers.
- `web/services/metadata_service.py` for symbol normalization and metadata
  population.
- `web/services/research_cache.py` for profile/snapshot/series/peer persistence
  and provider fetch orchestration.
- `web/services/payload_builders.py` for metrics, tags, factor signals, market
  pulse, completeness, and formatting.
- Keep `web/services/data_service.py` as the public facade exporting the current
  function names until API routes and tests are migrated.

## Non-Goals

- Do not change workspace API response schemas, auth behavior, cache freshness
  windows, database table definitions, or provider fallback policy during the
  first split.
- Do not introduce network calls in tests; provider fetches must remain
  monkeypatchable and offline in unit coverage.

## Validation

- Add facade import tests before moving route call sites.
- Preserve existing workspace API and data-service unit tests.
- Run the global contract subset and workspace API contract tests after each
  extraction step.
