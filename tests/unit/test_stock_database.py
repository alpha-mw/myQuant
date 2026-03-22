"""
StockDatabase 单元测试
"""

import sqlite3
from pathlib import Path

import pandas as pd

from quant_investor.stock_database import DownloadProgress, DownloadTask, StockDatabase


def _build_db(tmp_path: Path) -> StockDatabase:
    db_path = tmp_path / "test_stock.db"
    cache_dir = tmp_path / "cache"
    return StockDatabase(
        db_path=str(db_path),
        cache_dir=str(cache_dir),
        verbose=False,
        init_universe=False,
    )


def _seed_stock_list(db: StockDatabase, rows):
    conn = sqlite3.connect(db.db_path)
    conn.executemany(
        """
        INSERT INTO stock_list
        (ts_code, name, industry, market, list_date, is_hs300, is_zz500, is_zz1000, last_update)
        VALUES (?, ?, ?, ?, ?, 0, 0, 0, ?)
        """,
        [
            (
                ts_code,
                ts_code,
                "行业",
                market if len(row) == 3 else "CN",
                list_date,
                "2026-03-14T00:00:00",
            )
            for row in rows
            for ts_code, list_date, market in [row if len(row) == 3 else (*row, "CN")]
        ],
    )
    conn.commit()
    conn.close()


def _seed_daily_rows(db: StockDatabase, rows: list[tuple[str, str, float]]):
    conn = sqlite3.connect(db.db_path)
    conn.executemany(
        """
        INSERT INTO daily_data
        (ts_code, trade_date, open, high, low, close, volume, amount)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (ts_code, trade_date, close_price, close_price + 1, close_price - 1, close_price, 1000.0, 10000.0)
            for ts_code, trade_date, close_price in rows
        ],
    )
    conn.commit()
    conn.close()


def test_plan_historical_backfill_respects_listing_dates(tmp_path):
    db = _build_db(tmp_path)
    _seed_stock_list(
        db,
        [
            ("000001.SZ", "20100101"),
            ("000002.SZ", "20100101"),
            ("000003.SZ", "20100101"),
        ],
    )
    _seed_daily_rows(
        db,
        [
            ("000001.SZ", "20230213", 10.0),
            ("000002.SZ", "20240105", 20.0),
            ("000003.SZ", "20150101", 30.0),
        ],
    )

    plan = db.plan_historical_backfill(years=7, anchor_start="20230213")
    tasks_by_code = {task.ts_code: task for task in plan.tasks}

    assert plan.target_start == "20160213"
    assert plan.stock_count == 2
    assert tasks_by_code["000001.SZ"].start_date == "20160213"
    assert tasks_by_code["000001.SZ"].end_date == "20230213"
    assert tasks_by_code["000002.SZ"].start_date == "20160213"
    assert tasks_by_code["000002.SZ"].end_date == "20240105"
    assert "000003.SZ" not in tasks_by_code


def test_plan_download_tasks_fills_prefix_and_suffix(tmp_path):
    db = _build_db(tmp_path)
    _seed_stock_list(db, [("000001.SZ", "20100101")])
    _seed_daily_rows(
        db,
        [
            ("000001.SZ", "20240110", 10.0),
            ("000001.SZ", "20240120", 11.0),
        ],
    )

    tasks = db.plan_download_tasks("20240101", "20240131")

    assert len(tasks) == 2
    assert tasks[0].reason == "prefix_fill"
    assert tasks[0].start_date == "20240101"
    assert tasks[0].end_date == "20240110"
    assert tasks[1].reason == "suffix_fill"
    assert tasks[1].start_date == "20240120"
    assert tasks[1].end_date == "20240131"


def test_download_missing_stocks_only_runs_empty_range_tasks(tmp_path):
    db = _build_db(tmp_path)
    _seed_stock_list(
        db,
        [
            ("000001.SZ", "20100101", "CN"),
            ("000002.SZ", "20100101", "CN"),
        ],
    )
    _seed_daily_rows(
        db,
        [
            ("000001.SZ", "20240110", 10.0),
            ("000001.SZ", "20240120", 11.0),
        ],
    )

    executed_tasks = []

    def fake_execute(tasks, max_workers=1, log_label=""):
        executed_tasks.extend(tasks)
        return DownloadProgress(total_stocks=len(tasks), completed_stocks=len(tasks), failed_stocks=[], last_update=pd.Timestamp("2026-03-16"))

    db._execute_tasks = fake_execute

    progress = db.download_missing_stocks(
        start_date="20240101",
        end_date="20240131",
        max_workers=1,
    )

    assert progress.total_stocks == 1
    assert [task.ts_code for task in executed_tasks] == ["000002.SZ"]
    assert executed_tasks[0].reason == "empty_range"
    assert executed_tasks[0].start_date == "20240101"
    assert executed_tasks[0].end_date == "20240131"


def test_download_task_replaces_overlap_rows_for_consistency(tmp_path):
    db = _build_db(tmp_path)
    _seed_stock_list(db, [("000001.SZ", "20100101")])
    _seed_daily_rows(db, [("000001.SZ", "20230213", 10.0)])

    class FakePro:
        def daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
            assert ts_code == "000001.SZ"
            assert start_date == "20230210"
            assert end_date == "20230213"
            return pd.DataFrame(
                {
                    "trade_date": ["20230213", "20230210"],
                    "open": [11.0, 9.0],
                    "high": [12.0, 10.0],
                    "low": [10.0, 8.0],
                    "close": [11.0, 9.0],
                    "vol": [1500.0, 1200.0],
                    "amount": [11000.0, 9000.0],
                }
            )

        def adj_factor(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "trade_date": ["20230213", "20230210"],
                    "adj_factor": [1.0, 1.0],
                }
            )

    db._get_tushare_client = lambda: FakePro()

    task = DownloadTask(
        ts_code="000001.SZ",
        start_date="20230210",
        end_date="20230213",
        reason="prefix_fill",
        existing_start="20230213",
    )

    assert db.download_task(task) is True

    conn = sqlite3.connect(db.db_path)
    price_rows = conn.execute(
        """
        SELECT trade_date, close
        FROM daily_data
        WHERE ts_code = '000001.SZ'
        ORDER BY trade_date
        """
    ).fetchall()
    message = conn.execute(
        """
        SELECT message
        FROM download_log
        WHERE ts_code = '000001.SZ'
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()[0]
    conn.close()

    assert price_rows == [("20230210", 9.0), ("20230213", 11.0)]
    assert "action=replace" in message


def test_plan_historical_backfill_includes_us_and_skips_short_gap(tmp_path):
    db = _build_db(tmp_path)
    _seed_stock_list(
        db,
        [
            ("000001.SZ", "20100101", "CN"),
            ("000002.SZ", "20100101", "CN"),
            ("AAPL", "19801212", "US"),
        ],
    )
    _seed_daily_rows(
        db,
        [
            ("000001.SZ", "20160215", 10.0),
            ("000002.SZ", "20160325", 20.0),
            ("AAPL", "20230213", 30.0),
        ],
    )

    plan = db.plan_historical_backfill(years=7, anchor_start="20230213")
    tasks_by_code = {task.ts_code: task for task in plan.tasks}

    assert "000001.SZ" not in tasks_by_code
    assert tasks_by_code["000002.SZ"].start_date == "20160213"
    assert tasks_by_code["000002.SZ"].end_date == "20160325"
    assert tasks_by_code["AAPL"].market == "US"
    assert tasks_by_code["AAPL"].start_date == "20160213"
    assert tasks_by_code["AAPL"].end_date == "20230213"


def test_plan_historical_backfill_empty_stock_downloads_to_anchor_end(tmp_path):
    db = _build_db(tmp_path)
    _seed_stock_list(
        db,
        [
            ("AAPL", "19801212", "US"),
            ("SNOW", "20240101", "US"),
        ],
    )
    _seed_daily_rows(
        db,
        [
            ("AAPL", "20230213", 30.0),
            ("AAPL", "20260312", 35.0),
        ],
    )

    plan = db.plan_historical_backfill(years=7, anchor_start="20230213")
    tasks_by_code = {task.ts_code: task for task in plan.tasks}

    assert tasks_by_code["SNOW"].start_date == "20240101"
    assert tasks_by_code["SNOW"].end_date == plan.anchor_end
    assert tasks_by_code["SNOW"].market == "US"


def test_download_task_uses_us_source_and_replaces_overlap_rows(tmp_path):
    db = _build_db(tmp_path)
    _seed_stock_list(db, [("AAPL", "19801212", "US")])
    _seed_daily_rows(db, [("AAPL", "20230213", 10.0)])

    def fake_download_us_raw_data(task: DownloadTask) -> pd.DataFrame:
        assert task.ts_code == "AAPL"
        assert task.start_date == "20230210"
        assert task.end_date == "20230213"
        return pd.DataFrame(
            {
                "Date": ["2023-02-13", "2023-02-10"],
                "Open": [11.0, 9.0],
                "High": [12.0, 10.0],
                "Low": [10.0, 8.0],
                "Close": [11.0, 9.0],
                "Volume": [1500.0, 1200.0],
            }
        )

    db._download_us_raw_data = fake_download_us_raw_data

    task = DownloadTask(
        ts_code="AAPL",
        start_date="20230210",
        end_date="20230213",
        reason="prefix_fill",
        market="US",
        existing_start="20230213",
    )

    assert db.download_task(task) is True

    conn = sqlite3.connect(db.db_path)
    price_rows = conn.execute(
        """
        SELECT trade_date, close
        FROM daily_data
        WHERE ts_code = 'AAPL'
        ORDER BY trade_date
        """
    ).fetchall()
    message = conn.execute(
        """
        SELECT message
        FROM download_log
        WHERE ts_code = 'AAPL'
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()[0]
    conn.close()

    assert price_rows == [("20230210", 9.0), ("20230213", 11.0)]
    assert "action=replace" in message


def test_download_task_cn_converts_to_qfq_prices(tmp_path):
    db = _build_db(tmp_path)
    _seed_stock_list(db, [("000001.SZ", "20100101", "CN")])

    class FakePro:
        def daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
            assert ts_code == "000001.SZ"
            return pd.DataFrame(
                {
                    "trade_date": ["20230213", "20230210"],
                    "open": [18.0, 9.0],
                    "high": [21.0, 11.0],
                    "low": [17.0, 8.0],
                    "close": [20.0, 10.0],
                    "vol": [2000.0, 1000.0],
                    "amount": [20000.0, 10000.0],
                }
            )

        def adj_factor(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
            assert ts_code == "000001.SZ"
            assert start_date == "20230210"
            assert end_date == "20230213"
            return pd.DataFrame(
                {
                    "trade_date": ["20230213", "20230210"],
                    "adj_factor": [2.0, 1.0],
                }
            )

    db._get_tushare_client = lambda: FakePro()

    task = DownloadTask(
        ts_code="000001.SZ",
        start_date="20230210",
        end_date="20230213",
        reason="manual",
        market="CN",
        existing_end="20230213",
    )

    assert db.download_task(task) is True

    conn = sqlite3.connect(db.db_path)
    rows = conn.execute(
        """
        SELECT trade_date, open, close, adj_factor, price_mode, data_source
        FROM daily_data
        WHERE ts_code = '000001.SZ'
        ORDER BY trade_date
        """
    ).fetchall()
    conn.close()

    assert rows == [
        ("20230210", 4.5, 5.0, 1.0, "qfq", "tushare"),
        ("20230213", 18.0, 20.0, 2.0, "qfq", "tushare"),
    ]


def test_plan_historical_backfill_skips_exhausted_task_from_log(tmp_path):
    db = _build_db(tmp_path)
    _seed_stock_list(db, [("000001.SZ", "20100101", "CN")])
    _seed_daily_rows(db, [("000001.SZ", "20160325", 10.0)])

    conn = sqlite3.connect(db.db_path)
    conn.execute(
        """
        INSERT INTO download_log
        (ts_code, start_date, end_date, records_count, status, message, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "000001.SZ",
            "20160213",
            "20160325",
            1,
            "success",
            "historical_backfill; overlap=1; boundary=verified; actual=20160325-20160325",
            "2026-03-14T00:00:00",
        ),
    )
    conn.commit()
    conn.close()

    plan = db.plan_historical_backfill(years=7, anchor_start="20230213")
    assert plan.tasks == []


def test_plan_historical_backfill_skips_empty_task_after_empty_log(tmp_path):
    db = _build_db(tmp_path)
    _seed_stock_list(db, [("AAPL", "19801212", "US")])
    _seed_daily_rows(db, [("000001.SZ", "20230213", 10.0)])

    conn = sqlite3.connect(db.db_path)
    conn.execute(
        """
        INSERT INTO download_log
        (ts_code, start_date, end_date, records_count, status, message, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "AAPL",
            "20160213",
            "20230213",
            0,
            "failed",
            "historical_backfill; market=US; source=yfinance; confirmed_empty=1; empty",
            "2026-03-15T00:00:00",
        ),
    )
    conn.commit()
    conn.close()

    plan = db.plan_historical_backfill(years=7, anchor_start="20230213")
    assert "AAPL" not in {task.ts_code for task in plan.tasks}


def test_plan_historical_backfill_ignores_stale_empty_log_without_source(tmp_path):
    db = _build_db(tmp_path)
    _seed_stock_list(db, [("AAPL", "19801212", "US")])
    _seed_daily_rows(db, [("000001.SZ", "20230213", 10.0)])

    conn = sqlite3.connect(db.db_path)
    conn.execute(
        """
        INSERT INTO download_log
        (ts_code, start_date, end_date, records_count, status, message, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "AAPL",
            "20160213",
            "20230213",
            0,
            "failed",
            "historical_backfill; empty",
            "2026-03-15T00:00:00",
        ),
    )
    conn.commit()
    conn.close()

    plan = db.plan_historical_backfill(years=7, anchor_start="20230213")
    assert "AAPL" in {task.ts_code for task in plan.tasks}
