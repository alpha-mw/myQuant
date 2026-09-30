# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Read `AGENTS.md` first: it defines the stable CLI/API surface, the offline-by-default
rule (no live Tushare/yfinance/LLM/broker calls during verification), the
`system activate`-only writer for `results/system/_active.json`, and the rule that
removed commands must fail explicitly rather than be restored.

## Commands

Python 3.13+, managed with `uv` (`uv sync`, `.venv/`). The CLI entry point is
`quant-investor` (`quant_investor.cli.main:main`).

```bash
uv run pytest tests/unit -q                                   # full suite (~50 min, ~5.3k tests)
uv run pytest tests/unit/test_market_data_reader_parquet.py -q  # one file
uv run pytest "tests/unit/test_x.py::test_name" -q            # one test
uv run pytest tests/unit -q -k unified                        # stable-contract subset (CI step)
```

CI (`.github/workflows/ci-cd.yml`) additionally runs, and these must stay clean:

```bash
uv run python scripts/check_strategy_record_access.py
uv run flake8 quant_investor --count --select=E9,F63,F7,F82 --show-source --statistics
uv run flake8 quant_investor/contracts quant_investor/system quant_investor/factors/governance \
  quant_investor/intelligence quant_investor/mainline quant_investor/cli --max-complexity=10 --max-line-length=100
uv run black --check <same six packages>          # line length 100
uv run mypy <same six packages> --ignore-missing-imports
node portfolio_dashboard/tests/cn_aggressive_dashboard_contract_v1.test.js
```

Many modules outside those six packages are not black-clean; do not reformat whole
files you did not otherwise touch.

`tests/unit/test_unified_release_install.py` builds a real offline release with
`uv --offline`; it fails unless `UV_CACHE_DIR` points at a populated cache
(e.g. `UV_CACHE_DIR=$HOME/.cache/uv`).

## Architecture

### Everything is content-addressed and fail-closed
Most artifacts are immutable files referenced by `{path, sha256}` refs
(`quant_investor/operations/daily_contract.py:validate_ref`). Readers re-hash what
they read and raise a named blocker instead of falling back to CSV, cached,
"latest by mtime" or inferred data. When changing a writer, find every verifier
of that artifact (often in `scripts/` and `quant_investor/operations/`) — hashes of
one file are frequently pinned inside other sealed files.

### CN market data (`quant_investor/market/`)
- Pointer `data/parquet/cn/_latest.json` → immutable snapshot
  `data/parquet/cn/_snapshots/<id>.json` + `<id>/table/bars/year=YYYY/month=MM/part.parquet`.
  Daily upserts rewrite only touched month partitions; unchanged partitions are
  hardlinked across snapshots.
- `MarketDataReader` (`market_data_reader.py`) reads v4 snapshots from `table/`
  only; symbol inventory and latest dates come from a cached per-snapshot index.
  `frozen_snapshot_ref={"path","sha256"}` binds a reader to one historical
  snapshot regardless of the current pointer.
- `serving/bars/symbol=X/bars.parquet` is a retired per-symbol projection. It is
  still published during a transition (`config.CN_MARKET_SERVING_PROJECTION`,
  pruned to the newest 3 snapshots by `MarketDataStore.prune_snapshot_serving_layers`)
  only because runtimes pinned to older releases still read it. New code must not
  read or pin serving files. Until production moves to a release containing that
  change, launchd job `com.myquant.prune-market-serving` (22:00,
  `scripts/operations/prune_market_serving.sh`, logs in `logs/`) prunes serving to
  the newest 3 snapshots under the market writer lock.
- `MarketDataStore` (`market_data_store.py`) owns publication (`upsert_bars`),
  validation, and snapshot reactivation; `market_daily_capture.py` builds private
  shadow candidates before production publish.

### Daily production DAG
Orchestrated by `scripts/daily_*.py` with contracts in `quant_investor/operations/`
(see `docs/architecture/daily_evidence_dag.md`, `docs/cn-daily-official-close.md`).
Each trade date has an immutable journal under
`results/operations/daily_production/CN/<YYYYMMDD>/` written through
`DailyJournal.storage` (write-once; conflicting rewrites raise). Stages seal inputs
by ref and later stages/morning jobs replay them.

### Strategy records and dashboard
- `results/strategy_records/CN/<strategy>/<record>/` are sealed; the store catalog
  (`quant_investor/strategy_records/store.py`, `scripts/manage_cn_strategy_records.py
  verify`) hashes every file in each record dir, so existing records can never be
  edited or have files added.
- Dashboard export: `scripts/export_cn_aggressive_dashboard_data.py` →
  `scripts/cn_dashboard_common.py:validate_record` (v1) and `scripts/cn_dashboard_v2.py`;
  outputs only to `portfolio_dashboard/private/generated/`. Historical replay
  (`quant_investor/operations/dashboard_replay_sources.py`) serves only retained
  bytes — any new file the builder reads must also be retained/replayable.
- Official-close valuation evidence (`strict_market_close_evidence.json`) is
  written by `scripts/cn_official_close_batch.py` and validated against the
  snapshot's canonical table partition (`operations/strict_close_table_source.py`).

### Factor, intelligence, mainline
Stable packages `quant_investor/{contracts,system,factors,intelligence,mainline,cli}`
back the public `QuantInvestor` API and CLI; public readers resolve exactly one
`results/system/_active.json` generation and never scan for newer results. Factor
governance/production lives in `quant_investor/factors/` (see
`docs/factor_governance.md`, `docs/runbooks/factor_production.md`).

## Production runtime and working-tree etiquette

- Scheduled jobs (Hermes automations in `~/.hermes/profiles/myquant/automations/*.md`
  and launchd `com.myquant.daily-factor-loop`) run from **frozen release installs**
  under `~/mySpace/myQuant-release-authority/<commit>-*/installs/…`, built by
  `quant-investor system release-prepare` from a clean detached checkout. Editing
  this repo does not change those jobs until a new release is built and the
  automation contracts are repointed. Exception: the dashboard job runs repo
  scripts directly.
- Different jobs are pinned to different commits; a change to a shared on-disk
  format must stay readable by every pinned release still in use.
- The working tree is routinely dirty with uncommitted work from other agents
  (Codex, Hermes). Never checkout/reset/stash/clean it; commit only your own hunks,
  preferably from a separate `git worktree`.
- `data/`, `results/`, `reports/` are git-ignored and hold production state; do not
  delete snapshot, record, or journal directories without tracing who references them.
