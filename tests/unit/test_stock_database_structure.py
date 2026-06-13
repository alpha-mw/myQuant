from __future__ import annotations

import importlib
import importlib.util

from quant_investor import stock_database


def test_stock_database_types_are_split_and_reexported() -> None:
    spec = importlib.util.find_spec("quant_investor.stock_database_types")
    assert spec is not None
    types_module = importlib.import_module("quant_investor.stock_database_types")

    assert stock_database.DownloadTask is types_module.DownloadTask
    assert stock_database.BackfillPlan is types_module.BackfillPlan
    assert stock_database.DownloadProgress is types_module.DownloadProgress
    assert stock_database.PROJECT_ROOT == types_module.PROJECT_ROOT
    assert stock_database.CONSISTENCY_FIELDS is types_module.CONSISTENCY_FIELDS
    assert stock_database.SUPPORTED_MARKETS is types_module.SUPPORTED_MARKETS
    assert stock_database._default_db_path is types_module.default_db_path
    assert stock_database._default_cache_dir is types_module.default_cache_dir

    task = stock_database.DownloadTask(
        ts_code="000001.SZ",
        start_date="20250101",
        end_date="20250131",
        reason="fixture",
    )
    plan = stock_database.BackfillPlan(
        years=1,
        anchor_start="20250101",
        anchor_end="20250131",
        target_start="20240101",
        tasks=[task],
    )
    assert plan.stock_count == 1
    assert stock_database.DownloadProgress(0, 0, [], types_module.datetime.now()).progress_pct == 0.0


def test_stock_database_download_runtime_is_split_into_mixin() -> None:
    spec = importlib.util.find_spec("quant_investor.stock_database_download")
    assert spec is not None
    download_module = importlib.import_module("quant_investor.stock_database_download")

    assert issubclass(stock_database.StockDatabase, download_module.StockDatabaseDownloadMixin)
    assert (
        stock_database.StockDatabase._prepare_daily_frame
        is download_module.StockDatabaseDownloadMixin._prepare_daily_frame
    )
    assert (
        stock_database.StockDatabase._execute_tasks
        is download_module.StockDatabaseDownloadMixin._execute_tasks
    )
    assert (
        stock_database.StockDatabase.download_task
        is download_module.StockDatabaseDownloadMixin.download_task
    )
