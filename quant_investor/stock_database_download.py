"""Download/runtime mixin for StockDatabase."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False

from quant_investor.stock_database_types import (
    CONSISTENCY_FIELDS,
    NO_DATA_ERROR_PATTERNS,
    PRICE_MODE_QFQ,
    STANDARDIZATION_NOTE,
    SUPPORTED_MARKETS,
    US_UNIVERSE_FILE,
    VOLUME_MODE_RAW,
    DownloadProgress,
    DownloadTask,
)


class StockDatabaseDownloadMixin:
    def _load_overlap_frame(self, task: DownloadTask) -> pd.DataFrame:
        conn = self._connect()
        overlap_df = pd.read_sql_query(
            """
            SELECT trade_date, open, high, low, close, volume, amount
            FROM daily_data
            WHERE ts_code = ? AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date
            """,
            conn,
            params=[task.ts_code, task.start_date, task.end_date],
        )
        conn.close()
        return overlap_df

    def _prepare_daily_frame(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "vol": "volume",
            "Date": "trade_date",
            "Datetime": "trade_date",
            "date": "trade_date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Amount": "amount",
        }
        df = raw_df.rename(columns=rename_map).copy()
        if "amount" not in df.columns:
            df["amount"] = 0.0
        expected_columns = {"trade_date", "open", "high", "low", "close", "volume", "amount"}
        missing_columns = expected_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"下载结果缺少字段: {sorted(missing_columns)}")

        if "adj_factor" not in df.columns:
            df["adj_factor"] = np.nan
        if "price_mode" not in df.columns:
            df["price_mode"] = PRICE_MODE_QFQ
        if "data_source" not in df.columns:
            df["data_source"] = None

        ordered_columns = [
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "adj_factor",
            "price_mode",
            "data_source",
        ]
        df = df[ordered_columns]
        df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "", regex=False)
        df["trade_date"] = df["trade_date"].str.slice(0, 8)

        for column in CONSISTENCY_FIELDS:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce")

        df["amount"] = df["amount"].fillna(0.0)
        df = df.dropna(subset=["trade_date", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates(subset=["trade_date"], keep="last")
        df = df.sort_values("trade_date").reset_index(drop=True)
        return df

    def _build_consistency_message(
        self,
        task: DownloadTask,
        overlap_df: pd.DataFrame,
        downloaded_df: pd.DataFrame,
    ) -> str:
        if overlap_df.empty:
            return task.reason

        merged = overlap_df.merge(downloaded_df, on="trade_date", suffixes=("_old", "_new"))
        if merged.empty:
            return f"{task.reason}; overlap=0"

        mismatch_mask = np.zeros(len(merged), dtype=bool)
        for field in CONSISTENCY_FIELDS:
            old_values = merged[f"{field}_old"].to_numpy(dtype=float)
            new_values = merged[f"{field}_new"].to_numpy(dtype=float)
            mismatch_mask |= ~np.isclose(old_values, new_values, equal_nan=True)

        mismatch_count = int(mismatch_mask.sum())
        if mismatch_count > 0:
            return f"{task.reason}; overlap={len(merged)}; mismatched={mismatch_count}; action=replace"
        return f"{task.reason}; overlap={len(merged)}; boundary=verified"

    def _load_us_metadata(self) -> Dict[str, Dict[str, Optional[str]]]:
        """从缓存的美股 universe 中加载基础元数据。"""
        if not US_UNIVERSE_FILE.exists():
            self._log(f"未找到美股股票池缓存: {US_UNIVERSE_FILE}")
            return {}

        try:
            universe = json.loads(US_UNIVERSE_FILE.read_text())
        except Exception as exc:
            self._log(f"读取美股股票池缓存失败: {exc}")
            return {}

        if isinstance(universe, dict) and isinstance(universe.get("all"), list):
            symbols = universe["all"]
        elif isinstance(universe, list):
            symbols = universe
        else:
            self._log("美股股票池缓存格式异常，已跳过 US 补充")
            return {}

        metadata: Dict[str, Dict[str, Optional[str]]] = {}
        for symbol in symbols:
            normalized_symbol = str(symbol).strip().upper()
            if not normalized_symbol:
                continue
            metadata[normalized_symbol] = {
                "name": normalized_symbol,
                "industry": "US",
                "market": "US",
                "list_date": None,
            }
        return metadata

    def _download_cn_raw_data(self, task: DownloadTask) -> pd.DataFrame:
        pro = self._get_tushare_client()
        raw_df = pro.daily(ts_code=task.ts_code, start_date=task.start_date, end_date=task.end_date)
        if raw_df is None or raw_df.empty:
            return pd.DataFrame()

        adjustment_end = max(task.end_date, task.existing_end or task.end_date)
        adj_df = pro.adj_factor(
            ts_code=task.ts_code,
            start_date=task.start_date,
            end_date=adjustment_end,
        )
        if adj_df is None or adj_df.empty:
            raw_df = raw_df.copy()
            raw_df["adj_factor"] = 1.0
            raw_df["price_mode"] = PRICE_MODE_QFQ
            raw_df["data_source"] = "tushare"
            return raw_df

        raw_df = raw_df.copy()
        raw_df["trade_date"] = raw_df["trade_date"].astype(str)

        adj_df = adj_df[["trade_date", "adj_factor"]].copy()
        adj_df["trade_date"] = adj_df["trade_date"].astype(str)
        adj_df = adj_df.sort_values("trade_date").drop_duplicates(subset=["trade_date"], keep="last")
        anchor_factor = float(adj_df["adj_factor"].iloc[-1])
        if np.isclose(anchor_factor, 0.0):
            raise RuntimeError(f"{task.ts_code} 复权锚点因子异常: {anchor_factor}")

        merged = raw_df.merge(adj_df, on="trade_date", how="left")
        merged["adj_factor"] = merged["adj_factor"].ffill().bfill()
        if merged["adj_factor"].isna().any():
            raise RuntimeError(f"{task.ts_code} 存在缺失复权因子，无法生成前复权价格")

        scale = merged["adj_factor"] / anchor_factor
        for column in ("open", "high", "low", "close"):
            merged[column] = pd.to_numeric(merged[column], errors="coerce") * scale

        merged["price_mode"] = PRICE_MODE_QFQ
        merged["data_source"] = "tushare"
        return merged

    def _download_us_raw_data(self, task: DownloadTask) -> pd.DataFrame:
        if not YFINANCE_AVAILABLE:
            raise RuntimeError("yfinance 未安装，无法下载美股数据")

        start_text = datetime.strptime(task.start_date, "%Y%m%d").strftime("%Y-%m-%d")
        end_text = datetime.strptime(self._next_day(task.end_date), "%Y%m%d").strftime("%Y-%m-%d")

        for attempt in range(3):
            try:
                raw_df = yf.Ticker(task.ts_code).history(
                    start=start_text,
                    end=end_text,
                    interval="1d",
                    auto_adjust=True,
                    actions=False,
                )
                if raw_df is None or raw_df.empty:
                    if self._probe_us_recent_data(task.ts_code):
                        if attempt == 2:
                            raise RuntimeError("美股历史区间返回空结果，但近期仍有可用行情")
                        time.sleep(1 + attempt)
                        continue
                    empty_df = pd.DataFrame()
                    empty_df.attrs["confirmed_empty"] = True
                    return empty_df
                raw_df = raw_df.reset_index()
                raw_df["adj_factor"] = np.nan
                raw_df["price_mode"] = PRICE_MODE_QFQ
                raw_df["data_source"] = "yfinance"
                return raw_df
            except Exception as exc:
                message = str(exc).lower()
                if any(pattern in message for pattern in NO_DATA_ERROR_PATTERNS):
                    if self._probe_us_recent_data(task.ts_code):
                        if attempt == 2:
                            raise RuntimeError(f"美股下载异常，但近期探测存在行情: {exc}") from exc
                        time.sleep(1 + attempt)
                        continue
                    empty_df = pd.DataFrame()
                    empty_df.attrs["confirmed_empty"] = True
                    return empty_df
                if attempt == 2:
                    raise RuntimeError(f"美股下载失败: {exc}") from exc
                time.sleep(1 + attempt)

        return pd.DataFrame()

    def _probe_us_recent_data(self, symbol: str) -> bool:
        """用近期行情探针区分“真没数据”和“历史请求偶发空结果”."""
        probe_df = yf.Ticker(symbol).history(
            period="1mo",
            interval="1d",
            auto_adjust=True,
            actions=False,
        )
        return probe_df is not None and not probe_df.empty

    @staticmethod
    def _source_for_market(market: str) -> str:
        normalized_market = (market or "CN").upper()
        if normalized_market == "US":
            return "yfinance"
        return "tushare"

    def _upsert_market_data_config(self, market: str, source: str) -> None:
        """记录当前市场的统一价格口径，方便回测前校验。"""
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO market_data_config
            (market, price_mode, volume_mode, data_source, note, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(market) DO UPDATE SET
                price_mode=excluded.price_mode,
                volume_mode=excluded.volume_mode,
                data_source=excluded.data_source,
                note=excluded.note,
                updated_at=excluded.updated_at
            """,
            (
                market.upper(),
                PRICE_MODE_QFQ,
                VOLUME_MODE_RAW,
                source,
                STANDARDIZATION_NOTE,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        conn.close()

    def download_task(self, task: DownloadTask) -> bool:
        """
        执行单个下载任务。

        任务区间与现有数据允许 1 天重叠；若重叠日不一致，将以本次下载结果覆盖。
        """
        try:
            market = (task.market or "CN").upper()
            source = self._source_for_market(market)
            if market == "CN":
                raw_df = self._download_cn_raw_data(task)
            elif market == "US":
                raw_df = self._download_us_raw_data(task)
            else:
                raise ValueError(f"暂不支持市场: {market}")

            confirmed_empty = bool(getattr(raw_df, "attrs", {}).get("confirmed_empty"))
            if raw_df is None or raw_df.empty:
                empty_marker = "; confirmed_empty=1" if confirmed_empty else ""
                self._log_download(
                    task.ts_code,
                    task.start_date,
                    task.end_date,
                    0,
                    "failed",
                    f"{task.reason}; market={market}; source={source}{empty_marker}; empty",
                )
                return False

            prepared_df = self._prepare_daily_frame(raw_df)
            if prepared_df.empty:
                self._log_download(
                    task.ts_code,
                    task.start_date,
                    task.end_date,
                    0,
                    "failed",
                    f"{task.reason}; market={market}; source={source}; cleaned_empty",
                )
                return False

            overlap_df = self._load_overlap_frame(task)
            message = self._build_consistency_message(task, overlap_df, prepared_df)
            actual_start = prepared_df["trade_date"].iloc[0]
            actual_end = prepared_df["trade_date"].iloc[-1]
            message = f"{message}; market={market}; source={source}; actual={actual_start}-{actual_end}"

            records = [
                (
                    task.ts_code,
                    row.trade_date,
                    float(row.open),
                    float(row.high),
                    float(row.low),
                    float(row.close),
                    float(row.volume),
                    float(row.amount),
                    None if pd.isna(row.adj_factor) else float(row.adj_factor),
                    row.price_mode,
                    row.data_source,
                )
                for row in prepared_df.itertuples(index=False)
            ]

            conn = self._connect()
            conn.executemany(
                """
                INSERT OR REPLACE INTO daily_data
                (ts_code, trade_date, open, high, low, close, volume, amount, adj_factor, price_mode, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            conn.commit()
            conn.close()
            self._upsert_market_data_config(market, source)

            self._log_download(
                task.ts_code,
                task.start_date,
                task.end_date,
                len(prepared_df),
                "success",
                message,
            )
            return True

        except Exception as exc:
            self._log_download(
                task.ts_code,
                task.start_date,
                task.end_date,
                0,
                "failed",
                f"{task.reason}; {exc}",
            )
            return False

    def download_stock_data(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        reason: str = "manual",
    ) -> bool:
        """兼容旧接口，执行单只股票单区间下载。"""
        task = DownloadTask(
            ts_code=ts_code,
            start_date=self._normalize_date(start_date) or start_date,
            end_date=self._normalize_date(end_date) or end_date,
            reason=reason,
        )
        return self.download_task(task)

    def _log_download(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        records_count: int,
        status: str,
        message: str = "",
    ) -> None:
        """记录下载日志。"""
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO download_log
            (ts_code, start_date, end_date, records_count, status, message, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (ts_code, start_date, end_date, records_count, status, message, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()

    def _is_backfill_exhausted(self, task: DownloadTask) -> bool:
        """判断同一回填任务是否已证明无法拿到更早数据。"""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT status, message
            FROM download_log
            WHERE ts_code = ? AND start_date = ? AND end_date = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (task.ts_code, task.start_date, task.end_date),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return False

        status, message = row
        expected_source = self._source_for_market(task.market)
        if status == "failed" and message and (
            f"market={task.market.upper()}" in message
            and f"source={expected_source}" in message
            and "confirmed_empty=1" in message
            and (
                message.endswith("; empty")
                or message.endswith("; cleaned_empty")
                or "; no_data" in message
            )
        ):
            return True

        if status != "success" or not message:
            return False

        match = re.search(r"actual=(\d{8})-(\d{8})", message)
        if not match:
            return False

        actual_start = match.group(1)
        return actual_start > task.start_date

    def plan_price_standardization(
        self,
        market: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> List[DownloadTask]:
        """为已有数据重建统一回测价格序列。"""
        market_filter = self._normalize_market_filter(market)
        tasks: List[DownloadTask] = []
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                s.ts_code,
                s.market,
                MIN(d.trade_date) AS min_trade_date,
                MAX(d.trade_date) AS max_trade_date,
                COUNT(d.trade_date) AS row_count,
                SUM(
                    CASE
                        WHEN d.price_mode = ? AND d.data_source = ?
                        THEN 1 ELSE 0
                    END
                ) AS standardized_rows
            FROM stock_list s
            JOIN daily_data d ON d.ts_code = s.ts_code
            GROUP BY s.ts_code, s.market
            ORDER BY s.ts_code
            """,
            (PRICE_MODE_QFQ, "tushare"),
        )
        rows = cursor.fetchall()
        conn.close()

        for ts_code, stock_market, min_trade_date, max_trade_date, row_count, standardized_rows in rows:
            normalized_market = (stock_market or "CN").upper()
            if normalized_market not in SUPPORTED_MARKETS:
                continue
            if market_filter is not None and normalized_market != market_filter:
                continue
            if row_count == 0 or min_trade_date is None or max_trade_date is None:
                continue
            if normalized_market == "US":
                continue
            if int(standardized_rows or 0) >= int(row_count):
                continue

            tasks.append(
                DownloadTask(
                    ts_code=ts_code,
                    start_date=min_trade_date,
                    end_date=max_trade_date,
                    reason="price_standardization",
                    market=normalized_market,
                    existing_start=min_trade_date,
                    existing_end=max_trade_date,
                )
            )

        if batch_size is not None:
            tasks = tasks[:batch_size]
        return tasks

    def standardize_price_series(
        self,
        market: Optional[str] = None,
        max_workers: int = 1,
        batch_size: Optional[int] = None,
    ) -> DownloadProgress:
        """
        把已有价格统一成可回测口径。

        当前策略:
        - CN: 重建为前复权 OHLC
        - US: 现有序列已是 adjusted OHLC，仅补充元数据
        """
        market_filter = self._normalize_market_filter(market)
        if market_filter in (None, "US"):
            conn = self._connect()
            conn.execute(
                """
                UPDATE daily_data
                SET price_mode = ?, data_source = COALESCE(data_source, ?)
                WHERE ts_code IN (SELECT ts_code FROM stock_list WHERE market = 'US')
                """,
                (PRICE_MODE_QFQ, "yfinance"),
            )
            conn.commit()
            conn.close()
            self._upsert_market_data_config("US", "yfinance")
            if market_filter == "US":
                return DownloadProgress(0, 0, [], datetime.now())

        tasks = self.plan_price_standardization(market="CN" if market_filter is None else market_filter, batch_size=batch_size)
        if not tasks:
            return DownloadProgress(0, 0, [], datetime.now())

        return self._execute_tasks(
            tasks,
            max_workers=max_workers,
            log_label="价格标准化",
        )

    def _execute_tasks(
        self,
        tasks: List[DownloadTask],
        max_workers: int = 5,
        log_label: str = "批量下载",
    ) -> DownloadProgress:
        if not tasks:
            self._log(f"{log_label}: 没有需要处理的任务")
            return DownloadProgress(0, 0, [], datetime.now())

        self.progress = DownloadProgress(
            total_stocks=len(tasks),
            completed_stocks=0,
            failed_stocks=[],
            last_update=datetime.now(),
        )

        self._log(f"{log_label}: 任务数 {len(tasks)}，并行线程 {max_workers}")

        def handle_result(task: DownloadTask, success: bool) -> None:
            if success:
                self.progress.completed_stocks += 1
            else:
                self.progress.failed_stocks.append(task.ts_code)

            processed = self.progress.completed_stocks + len(self.progress.failed_stocks)
            if processed % 20 == 0 or processed == self.progress.total_stocks:
                self._log(
                    f"{log_label} 进度: {self.progress.progress_pct:.1f}% "
                    f"({processed}/{self.progress.total_stocks})"
                )

        if max_workers <= 1:
            for task in tasks:
                handle_result(task, self.download_task(task))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(self.download_task, task): task for task in tasks}
                for future in as_completed(future_map):
                    task = future_map[future]
                    try:
                        handle_result(task, future.result())
                    except Exception as exc:
                        self._log(f"下载异常 {task.ts_code}: {exc}")
                        handle_result(task, False)

        self.progress.last_update = datetime.now()
        self._log(
            f"{log_label} 完成: 成功 {self.progress.completed_stocks}, "
            f"失败 {len(self.progress.failed_stocks)}"
        )
        return self.progress
