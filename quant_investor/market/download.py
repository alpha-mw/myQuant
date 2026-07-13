"""
统一市场下载入口。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_investor.config import config
from quant_investor.market.config import get_market_settings, normalize_categories
from quant_investor.market.download_cn import CNFullMarketDownloader
from quant_investor.market.download_us import FullMarketDownloader as USFullMarketDownloader
from quant_investor.market.market_data_store import MarketDataStore
from quant_investor.market.tushare_data_cleaning import CLEANING_STATUS_FAIL, clean_tushare_dataframe


class MarketDownloader:
    """按市场选择具体下载 provider 的统一外壳。"""

    def __init__(self, market: str, **kwargs: Any) -> None:
        settings = get_market_settings(market)
        data_dir = kwargs.pop("data_dir", settings.data_dir)
        self.market = settings.market
        if self.market == "CN":
            self._impl = CNFullMarketDownloader(data_dir=data_dir, **kwargs)
        else:
            self._impl = USFullMarketDownloader(data_dir=data_dir, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._impl, name)


def create_downloader(market: str, **kwargs: Any) -> MarketDownloader:
    return MarketDownloader(market, **kwargs)


def _compact_trade_date(value: Any) -> str:
    text = str(value or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _coverage_payload(*, covered_count: int, expected_count: int, status: str, error: str = "") -> dict[str, Any]:
    return {
        "status": status,
        "covered_count": int(covered_count),
        "expected_count": int(expected_count),
        "coverage_ratio": covered_count / expected_count if expected_count else 1.0,
        "error": error,
    }


def _blocking_categories_from_completeness(completeness: dict[str, Any]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    categories = completeness.get("categories") if isinstance(completeness, dict) else {}
    if not isinstance(categories, dict):
        return blockers
    for category, payload in sorted(categories.items()):
        if not isinstance(payload, dict):
            continue
        blocking_count = int(payload.get("blocking_incomplete_count") or 0)
        if blocking_count <= 0:
            continue
        blockers.append(
            {
                "category": str(category),
                "blocking_incomplete_count": blocking_count,
                "blocking_missing_symbols": list(payload.get("blocking_missing_symbols") or []),
                "blocking_stale_symbols": list(payload.get("blocking_stale_symbols") or []),
                "blocking_unreadable_symbols": list(payload.get("blocking_unreadable_symbols") or []),
                "coverage_ratio": payload.get("coverage_ratio"),
                "expected": payload.get("expected"),
                "latest_trade_date": payload.get("latest_trade_date"),
            }
        )
    return blockers


def _failed_download_batches(download_results: dict[str, Any]) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    for round_payload in download_results.get("rounds") or []:
        if not isinstance(round_payload, dict):
            continue
        for category, results in (round_payload.get("categories") or {}).items():
            failed_symbols = [
                {
                    "symbol": str(item.get("symbol") or ""),
                    "status": str(item.get("status") or ""),
                    "error": str(item.get("error") or ""),
                    "latest_local_date": str(item.get("latest_local_date") or ""),
                    "latest_trade_date": str(item.get("latest_trade_date") or ""),
                    "local_status": str(item.get("local_status") or ""),
                }
                for item in (results or [])
                if isinstance(item, dict) and str(item.get("status") or "").lower() not in {"updated", "cached", "stale_cached"}
            ]
            if failed_symbols:
                failed.append(
                    {
                        "round": round_payload.get("round"),
                        "category": str(category),
                        "failed_count": len(failed_symbols),
                        "symbols": failed_symbols,
                    }
                )
    return failed


def _write_cn_maintenance_progress_artifacts(
    *,
    downloader: Any,
    selected_categories: list[str],
    download_results: dict[str, Any],
    completeness: dict[str, Any],
    parquet_commit: dict[str, Any],
) -> dict[str, Any]:
    data_dir = Path(getattr(downloader, "data_dir"))
    progress_path = data_dir / "progress_summary.json"
    failed_batches_path = data_dir / "failed_batches.json"
    failed_batches = _failed_download_batches(download_results)
    blocking_categories = _blocking_categories_from_completeness(completeness)
    commit_status = str(parquet_commit.get("status") or "").strip().upper()
    status = "OK" if completeness.get("complete") and commit_status not in {"BLOCKED", "FAILED"} else "BLOCKED"
    generated_at = _utc_now_iso()
    latest_available = (
        _compact_trade_date(parquet_commit.get("latest_available_trade_date"))
        or _compact_trade_date(completeness.get("effective_target_trade_date"))
        or _compact_trade_date(completeness.get("latest_trade_date"))
    )
    latest_complete = _compact_trade_date(parquet_commit.get("latest_complete_trade_date"))
    failed_payload = {
        "generated_at": generated_at,
        "market": "CN",
        "categories": list(selected_categories),
        "failed_batch_count": len(failed_batches),
        "blocking_category_count": len(blocking_categories),
        "failed_downloads": failed_batches,
        "blocking_categories": blocking_categories,
    }
    progress_payload = {
        "generated_at": generated_at,
        "market": "CN",
        "status": status,
        "storage_mode": str((download_results.get("config") or {}).get("storage_mode") or parquet_commit.get("storage_mode") or "legacy"),
        "categories": list(selected_categories),
        "latest_available_trade_date": latest_available,
        "latest_complete_trade_date": latest_complete,
        "quarantined_tail_dates": list(parquet_commit.get("quarantined_tail_dates") or []),
        "coverage": parquet_commit.get("coverage") or {},
        "blockers": list(parquet_commit.get("blockers") or []),
        "same_day_close_probe": (download_results.get("config") or {}).get("same_day_close_probe", {}),
        "daily_basic_coverage": parquet_commit.get("daily_basic_coverage") or (download_results.get("config") or {}).get("daily_basic_coverage") or {},
        "adj_factor_coverage": parquet_commit.get("adj_factor_coverage") or (download_results.get("config") or {}).get("adj_factor_coverage") or {},
        "round_count": len(download_results.get("rounds") or []),
        "failed_batch_count": len(failed_batches),
        "blocking_category_count": len(blocking_categories),
        "stats": dict(getattr(downloader, "stats", {}) or {}),
        "clean_snapshot": {
            "status": commit_status,
            "snapshot_id": str(parquet_commit.get("snapshot_id") or ""),
            "manifest_path": str(parquet_commit.get("manifest_path") or ""),
            "table_root": str(parquet_commit.get("table_root") or ""),
        },
        "progress_summary_path": str(progress_path),
        "failed_batches_path": str(failed_batches_path),
    }
    _atomic_json_write(failed_batches_path, failed_payload)
    _atomic_json_write(progress_path, progress_payload)
    return {
        "status": status,
        "progress_summary_path": str(progress_path),
        "failed_batches_path": str(failed_batches_path),
        "failed_batch_count": len(failed_batches),
        "blocking_category_count": len(blocking_categories),
    }


class CNParquetBatchMaintainer:
    """Date-scoped CN maintainer that writes directly to Parquet canonical."""

    def __init__(
        self,
        *,
        data_dir: str | None = None,
        data_root: str | Path | None = None,
        years: int = 3,
        max_workers: int = 4,
        batch_size: int = 50,
    ) -> None:
        self.downloader = CNFullMarketDownloader(
            data_dir=data_dir,
            years=years,
            max_workers=max_workers,
            batch_size=batch_size,
        )
        self.data_dir = Path(self.downloader.data_dir)
        self.store = MarketDataStore(market="CN", data_root=data_root or getattr(config, "MARKET_DATA_BASE_DIR", "data"))

    def maintain(
        self,
        *,
        categories: list[str] | None = None,
        fail_on_incomplete: bool = False,
        allowed_stale_symbols: list[str] | None = None,
        target_date: str = "auto",
    ) -> dict[str, Any]:
        components = self.downloader.load_components()
        target_categories = self.downloader._resolve_target_categories(components, categories)
        explicit_target_date = _compact_trade_date(target_date)
        if explicit_target_date and str(target_date).strip().lower() != "auto":
            same_day_probe = {
                "applicable": False,
                "available": True,
                "reason": "explicit_target_date",
                "trade_date": explicit_target_date,
            }
            target_trade_date = explicit_target_date
        else:
            same_day_probe = self.downloader._probe_strict_same_day_close_availability(
                components=components,
                target_categories=target_categories,
            )
            target_trade_date = self.downloader.latest_trade_date
        early_stop_reason = ""
        if explicit_target_date and str(target_date).strip().lower() != "auto":
            early_stop_reason = "explicit_target_date"
        elif same_day_probe.get("applicable") and same_day_probe.get("available") is False:
            target_trade_date = self.downloader.stable_trade_date
            early_stop_reason = "strict_same_day_unavailable"

        daily_df, daily_error = self._fetch_endpoint(
            "daily",
            target_trade_date,
            "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
        )
        adj_df, adj_error = self._fetch_endpoint(
            "adj_factor",
            target_trade_date,
            "ts_code,trade_date,adj_factor",
        )
        daily_basic_df, daily_basic_error = self._fetch_endpoint(
            "daily_basic",
            target_trade_date,
            "ts_code,trade_date,turnover_rate,volume_ratio,pe,pb,total_mv,circ_mv",
        )
        target_symbols = {
            str(symbol or "").strip().upper()
            for category in target_categories
            for symbol in components.get(category, []) or []
            if str(symbol or "").strip()
        }
        expected_scope_sha256 = hashlib.sha256(
            "\n".join(sorted(target_symbols)).encode("utf-8")
        ).hexdigest()
        daily_symbols = set(daily_df["ts_code"].astype(str)) if not daily_df.empty else set()
        allowed = self.downloader._normalize_allowed_symbols(allowed_stale_symbols)
        suspended_symbols = self.downloader._load_latest_suspended_symbols(target_trade_date)
        inactive_symbols = self._load_inactive_symbols(target_trade_date, target_symbols)
        scope_suspended_symbols = suspended_symbols & target_symbols
        scope_inactive_symbols = inactive_symbols & target_symbols
        scope_allowed_symbols = allowed & target_symbols
        observed_target_symbols = daily_symbols & target_symbols
        inactive_absent = scope_inactive_symbols - daily_symbols
        suspended_absent = (
            scope_suspended_symbols - daily_symbols - inactive_absent
        )
        allowed_absent = (
            scope_allowed_symbols
            - daily_symbols
            - inactive_absent
            - suspended_absent
        )
        non_blocking_absent = suspended_absent | inactive_absent
        true_missing_symbols = target_symbols - observed_target_symbols - non_blocking_absent
        missing_daily = sorted(true_missing_symbols)
        bars_frame = self._build_bars_frame(daily_df, adj_df, daily_basic_df)
        valid_adj_symbols: set[str] = set()
        if not bars_frame.empty and "adj_factor" in bars_frame.columns:
            adj_values = pd.to_numeric(bars_frame["adj_factor"], errors="coerce")
            valid_adj_symbols = set(bars_frame.loc[adj_values > 0, "ts_code"].astype(str))
        missing_adj = sorted(daily_symbols - valid_adj_symbols)
        daily_basic_symbols = set(daily_basic_df["ts_code"].astype(str)) & daily_symbols if not daily_basic_df.empty else set()
        daily_basic_coverage = _coverage_payload(
            covered_count=len(daily_basic_symbols),
            expected_count=len(daily_symbols),
            status="warning" if len(daily_basic_symbols) < len(daily_symbols) else "OK",
            error=daily_basic_error,
        )
        adj_factor_coverage = _coverage_payload(
            covered_count=len(valid_adj_symbols),
            expected_count=len(daily_symbols),
            status="OK" if not missing_adj and not adj_error else "BLOCKED",
            error=adj_error,
        )
        blockers: list[str] = []
        if daily_error:
            blockers.append(f"daily_endpoint_error:{daily_error}")
        if missing_daily:
            blockers.append(f"daily_missing:{len(missing_daily)}")
        if missing_adj:
            blockers.append("adj_factor_missing")
        if adj_error:
            blockers.append(f"adj_factor_endpoint_error:{adj_error}")
        if scope_allowed_symbols:
            blockers.append("unverified_allowed_stale_symbols_not_permitted")
        expected_count = len(target_symbols)
        coverage_complete_count = len(observed_target_symbols | non_blocking_absent)
        coverage_ratio = coverage_complete_count / expected_count if expected_count else 1.0
        complete = not blockers
        existing_latest_available = ""
        existing_latest_complete = ""
        if complete:
            existing_validation = self.store.validate_latest()
            existing_coverage = existing_validation.get("coverage") if isinstance(existing_validation, dict) else {}
            if not isinstance(existing_coverage, dict):
                existing_coverage = {}
            existing_latest_available = (
                _compact_trade_date(existing_coverage.get("latest_available_trade_date"))
                or _compact_trade_date(existing_validation.get("latest_trade_date") if isinstance(existing_validation, dict) else "")
            )
            existing_latest_complete = _compact_trade_date(
                existing_validation.get("latest_complete_trade_date") if isinstance(existing_validation, dict) else ""
            )
        commit_latest_available = max(
            [date for date in [target_trade_date, existing_latest_available] if date],
            default=target_trade_date,
        )
        commit_latest_complete = max(
            [date for date in [target_trade_date, existing_latest_complete] if date],
            default=target_trade_date,
        )
        completeness = self._completeness_payload(
            target_categories=target_categories,
            target_trade_date=target_trade_date,
            expected_count=expected_count,
            coverage_complete_count=coverage_complete_count,
            coverage_ratio=coverage_ratio,
            missing_daily=missing_daily,
            missing_adj=missing_adj,
            suspended_symbols=sorted(suspended_absent),
            inactive_symbols=sorted(inactive_absent),
            requested_allowed_stale_symbols=sorted(scope_allowed_symbols),
            requested_allowed_absent_symbols=sorted(allowed_absent),
            non_blocking_absent_symbols=sorted(non_blocking_absent),
            blockers=blockers,
            early_stop_reason=early_stop_reason,
        )
        download_results = self._download_results_payload(
            target_categories=target_categories,
            target_trade_date=target_trade_date,
            same_day_probe=same_day_probe,
            daily_basic_coverage=daily_basic_coverage,
            adj_factor_coverage=adj_factor_coverage,
            missing_daily=missing_daily,
            missing_adj=missing_adj,
            bars_frame=bars_frame,
            completeness=completeness,
            early_stop_reason=early_stop_reason,
        )
        if complete:
            parquet_commit = self.store.upsert_bars(
                bars_frame,
                target_trade_date=target_trade_date,
                source="market_maintenance_parquet_direct",
                metadata={
                    "status": "OK",
                    "storage_mode": "parquet-direct",
                    "latest_available_trade_date": commit_latest_available,
                    "latest_complete_trade_date": commit_latest_complete,
                    "coverage": {
                        "coverage_schema_version": "cn-full-a-coverage.v2",
                        "complete": True,
                        "coverage_ratio": coverage_ratio,
                        "coverage_complete_count": coverage_complete_count,
                        "expected_scope_count": expected_count,
                        "observed_bar_count": len(observed_target_symbols),
                        "blocking_incomplete_count": 0,
                        "categories_checked": list(target_categories),
                        "latest_available_trade_date": commit_latest_available,
                        "latest_complete_trade_date": commit_latest_complete,
                        "upsert_target_trade_date": target_trade_date,
                        "coverage_trade_date": target_trade_date,
                        "expected_scope_sha256": expected_scope_sha256,
                        "suspended_symbols": sorted(suspended_absent),
                        "inactive_symbols": sorted(inactive_absent),
                        "allowed_stale_symbols": [],
                        "requested_allowed_stale_symbols": sorted(scope_allowed_symbols),
                        "requested_allowed_absent_symbols": sorted(allowed_absent),
                        "non_blocking_absent_symbols": sorted(non_blocking_absent),
                        "true_missing_symbols": [],
                        "classification_sets_disjoint": True,
                        "suspended_evidence_symbols": sorted(scope_suspended_symbols),
                        "inactive_evidence_symbols": sorted(scope_inactive_symbols),
                        "daily_basic_coverage": daily_basic_coverage,
                        "adj_factor_coverage": adj_factor_coverage,
                    },
                    "blockers": [],
                    "daily_basic_coverage": daily_basic_coverage,
                    "adj_factor_coverage": adj_factor_coverage,
                },
            )
        else:
            validation = self.store.validate_latest()
            latest_complete = _compact_trade_date(validation.get("latest_complete_trade_date"))
            parquet_commit = {
                "status": "BLOCKED",
                "storage_mode": "parquet-direct",
                "latest_available_trade_date": target_trade_date,
                "latest_complete_trade_date": latest_complete,
                "latest_trade_date": latest_complete,
                "quarantined_tail_dates": [target_trade_date] if target_trade_date and target_trade_date != latest_complete else [],
                "coverage": {
                    "coverage_schema_version": "cn-full-a-coverage.v2",
                    "complete": False,
                    "coverage_ratio": coverage_ratio,
                    "coverage_complete_count": coverage_complete_count,
                    "expected_scope_count": expected_count,
                    "observed_bar_count": len(observed_target_symbols),
                    "blocking_incomplete_count": len(missing_daily) + len(missing_adj),
                    "categories_checked": list(target_categories),
                    "latest_available_trade_date": target_trade_date,
                    "latest_complete_trade_date": latest_complete,
                    "coverage_trade_date": target_trade_date,
                    "expected_scope_sha256": expected_scope_sha256,
                    "suspended_symbols": sorted(suspended_absent),
                    "inactive_symbols": sorted(inactive_absent),
                    "allowed_stale_symbols": [],
                    "requested_allowed_stale_symbols": sorted(scope_allowed_symbols),
                    "requested_allowed_absent_symbols": sorted(allowed_absent),
                    "non_blocking_absent_symbols": sorted(non_blocking_absent),
                    "true_missing_symbols": sorted(true_missing_symbols),
                    "classification_sets_disjoint": True,
                    "suspended_evidence_symbols": sorted(scope_suspended_symbols),
                    "inactive_evidence_symbols": sorted(scope_inactive_symbols),
                    "daily_basic_coverage": daily_basic_coverage,
                    "adj_factor_coverage": adj_factor_coverage,
                },
                "blockers": blockers,
                "daily_basic_coverage": daily_basic_coverage,
                "adj_factor_coverage": adj_factor_coverage,
            }
        self.store.append_health_event("market_maintenance_parquet_direct", parquet_commit)
        download_results["parquet_commit"] = parquet_commit
        download_results["completeness"] = completeness
        self.downloader._save_report(download_results)
        artifacts = _write_cn_maintenance_progress_artifacts(
            downloader=self.downloader,
            selected_categories=target_categories,
            download_results=download_results,
            completeness=completeness,
            parquet_commit=parquet_commit,
        )
        if fail_on_incomplete and not complete:
            raise RuntimeError("A股 Parquet direct 数据未完整更新到目标交易日，已按要求终止")
        return {
            "status": "maintained",
            "storage_mode": "parquet-direct",
            "download_results": download_results,
            "completeness": completeness,
            "categories": target_categories,
            "parquet_commit": parquet_commit,
            "maintenance_artifacts": artifacts,
        }

    def _fetch_endpoint(self, endpoint: str, trade_date: str, fields: str) -> tuple[pd.DataFrame, str]:
        func = getattr(self.downloader.pro, endpoint, None)
        if func is None:
            return pd.DataFrame(), "provider_unavailable"
        try:
            frame = func(trade_date=trade_date, fields=fields)
        except TypeError:
            try:
                frame = func(trade_date=trade_date)
            except Exception as exc:
                return pd.DataFrame(), str(exc)
        except Exception as exc:
            return pd.DataFrame(), str(exc)
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            return pd.DataFrame(), "empty"
        cleaned, _quarantined, _row_flags, _cell_flags, report = clean_tushare_dataframe(
            frame,
            table_name=endpoint,
            metadata={"storage_mode": "parquet-direct", "target_trade_date": trade_date},
        )
        if report.status == CLEANING_STATUS_FAIL:
            return pd.DataFrame(), f"cleaning_failed:{endpoint}:{report.blocker_count}"
        work = cleaned.copy()
        if "trade_date" not in work.columns:
            work["trade_date"] = trade_date
        if "ts_code" not in work.columns:
            return pd.DataFrame(), "missing_ts_code"
        work["ts_code"] = work["ts_code"].astype(str).str.strip().str.upper()
        work["trade_date"] = work["trade_date"].map(_compact_trade_date)
        work = work.loc[work["ts_code"].ne("") & work["trade_date"].eq(trade_date)].copy()
        return work.drop_duplicates(subset=["ts_code", "trade_date"], keep="last").reset_index(drop=True), ""

    def _load_inactive_symbols(self, target_trade_date: str, target_symbols: set[str]) -> set[str]:
        target_date = _compact_trade_date(target_trade_date)
        if not target_date or not target_symbols:
            return set()
        func = getattr(self.downloader.pro, "stock_basic", None)
        if func is None:
            return set()
        inactive: set[str] = set()
        for list_status in ("D", "L", "P"):
            try:
                frame = func(
                    exchange="",
                    list_status=list_status,
                    fields="ts_code,name,list_status,list_date,delist_date",
                )
            except TypeError:
                try:
                    frame = func(list_status=list_status)
                except Exception:
                    continue
            except Exception:
                continue
            if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty or "ts_code" not in frame.columns:
                continue
            work = frame.copy()
            work["ts_code"] = work["ts_code"].astype(str).str.strip().str.upper()
            work = work.loc[work["ts_code"].isin(target_symbols)].copy()
            if work.empty:
                continue
            if "list_date" in work.columns:
                list_dates = work["list_date"].map(_compact_trade_date)
                inactive.update(work.loc[list_dates.ne("") & list_dates.gt(target_date), "ts_code"].astype(str))
            if "delist_date" in work.columns:
                delist_dates = work["delist_date"].map(_compact_trade_date)
                delisted_by_target = delist_dates.ne("") & delist_dates.le(target_date)
                inactive.update(work.loc[delisted_by_target, "ts_code"].astype(str))
        return inactive & target_symbols

    @staticmethod
    def _build_bars_frame(daily_df: pd.DataFrame, adj_df: pd.DataFrame, daily_basic_df: pd.DataFrame) -> pd.DataFrame:
        if daily_df.empty:
            return pd.DataFrame()
        bars = daily_df.copy()
        if not adj_df.empty:
            bars = bars.merge(adj_df[["ts_code", "trade_date", "adj_factor"]], on=["ts_code", "trade_date"], how="left")
        else:
            bars["adj_factor"] = pd.NA
        basic_columns = [
            column
            for column in ["ts_code", "trade_date", "turnover_rate", "volume_ratio", "pe", "pb", "total_mv", "circ_mv"]
            if column in daily_basic_df.columns
        ]
        if len(basic_columns) > 2:
            bars = bars.merge(daily_basic_df[basic_columns], on=["ts_code", "trade_date"], how="left")
        for column in ["open", "high", "low", "close"]:
            if column in bars.columns:
                bars[f"adj_{column}"] = pd.to_numeric(bars[column], errors="coerce") * pd.to_numeric(
                    bars["adj_factor"],
                    errors="coerce",
                )
        return bars.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    def _completeness_payload(
        self,
        *,
        target_categories: list[str],
        target_trade_date: str,
        expected_count: int,
        coverage_complete_count: int,
        coverage_ratio: float,
        missing_daily: list[str],
        missing_adj: list[str],
        suspended_symbols: list[str],
        inactive_symbols: list[str],
        requested_allowed_stale_symbols: list[str],
        requested_allowed_absent_symbols: list[str],
        non_blocking_absent_symbols: list[str],
        blockers: list[str],
        early_stop_reason: str,
    ) -> dict[str, Any]:
        blocking_symbols = sorted(set(missing_daily) | set(missing_adj))
        observed_count = max(
            coverage_complete_count - len(non_blocking_absent_symbols),
            0,
        )
        categories = {
            category: {
                "expected": expected_count,
                "latest_trade_date": target_trade_date,
                "date_counts": {target_trade_date: coverage_complete_count},
                "status_counts": {
                    "up_to_date": observed_count,
                    "non_trading_or_allowed": len(non_blocking_absent_symbols),
                    "missing": len(missing_daily),
                    "adj_factor_missing": len(missing_adj),
                },
                "missing_symbols": list(missing_daily),
                "stale_symbols": [],
                "suspended_stale_symbols": [{"symbol": symbol, "latest_local_date": ""} for symbol in suspended_symbols],
                "inactive_symbols": list(inactive_symbols),
                "allowed_stale_symbols": [],
                "requested_allowed_stale_symbols": list(
                    requested_allowed_stale_symbols
                ),
                "requested_allowed_absent_symbols": list(
                    requested_allowed_absent_symbols
                ),
                "non_blocking_absent_symbols": list(non_blocking_absent_symbols),
                "unreadable_symbols": [],
                "blocking_missing_symbols": blocking_symbols,
                "blocking_stale_symbols": [],
                "blocking_unreadable_symbols": [],
                "blocking_incomplete_count": len(blocking_symbols),
                "coverage_complete_count": coverage_complete_count,
                "coverage_ratio": coverage_ratio,
            }
            for category in target_categories
        }
        return {
            "complete": not blockers,
            "blocking_incomplete_count": len(blocking_symbols),
            "categories_checked": list(target_categories),
            "categories": categories,
            "data_quality_issues": [],
            "data_quality_issue_count": 0,
            "latest_trade_date": target_trade_date,
            "strict_trade_date": self.downloader.strict_trade_date,
            "stable_trade_date": self.downloader.stable_trade_date,
            "effective_target_trade_date": target_trade_date,
            "freshness_mode": self.downloader.freshness_mode,
            "coverage_ratio": coverage_ratio,
            "coverage_complete_count": coverage_complete_count,
            "expected_scope_count": expected_count,
            "observed_bar_count": observed_count,
            "suspended_absent_symbols": list(suspended_symbols),
            "inactive_absent_symbols": list(inactive_symbols),
            "allowed_stale_symbols": [],
            "requested_allowed_stale_symbols": list(
                requested_allowed_stale_symbols
            ),
            "requested_allowed_absent_symbols": list(
                requested_allowed_absent_symbols
            ),
            "non_blocking_absent_symbols": list(non_blocking_absent_symbols),
            "true_missing_symbols": list(missing_daily),
            "coverage_threshold": self.downloader.coverage_threshold,
            "early_stop_reason": early_stop_reason,
            "blockers": blockers,
            "resolver": self.downloader.resolver.snapshot(),
        }

    def _download_results_payload(
        self,
        *,
        target_categories: list[str],
        target_trade_date: str,
        same_day_probe: dict[str, Any],
        daily_basic_coverage: dict[str, Any],
        adj_factor_coverage: dict[str, Any],
        missing_daily: list[str],
        missing_adj: list[str],
        bars_frame: pd.DataFrame,
        completeness: dict[str, Any],
        early_stop_reason: str,
    ) -> dict[str, Any]:
        ok_symbols = sorted(set(bars_frame.get("ts_code", pd.Series(dtype=str)).astype(str)) - set(missing_adj)) if not bars_frame.empty else []
        failed_symbols = [
            {"symbol": symbol, "status": "failed", "error": "daily_missing", "latest_trade_date": target_trade_date}
            for symbol in missing_daily
        ]
        failed_symbols.extend(
            {"symbol": symbol, "status": "failed", "error": "adj_factor_missing", "latest_trade_date": target_trade_date}
            for symbol in missing_adj
        )
        updated_symbols = [
            {
                "symbol": symbol,
                "status": "updated",
                "local_status": "up_to_date",
                "records": 1,
                "mode": "parquet_direct",
                "latest_local_date": target_trade_date,
                "latest_trade_date": target_trade_date,
                "resolved_path": "",
                "api_calls": 0,
                "error": None,
            }
            for symbol in ok_symbols
        ]
        return {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "storage_mode": "parquet-direct",
            "config": {
                "storage_mode": "parquet-direct",
                "latest_trade_date": target_trade_date,
                "strict_trade_date": self.downloader.strict_trade_date,
                "stable_trade_date": self.downloader.stable_trade_date,
                "effective_target_trade_date": target_trade_date,
                "freshness_mode": self.downloader.freshness_mode,
                "coverage_ratio": completeness.get("coverage_ratio", 0.0),
                "coverage_threshold": self.downloader.coverage_threshold,
                "early_stop_reason": early_stop_reason,
                "same_day_close_probe": same_day_probe,
                "categories": list(target_categories),
                "daily_basic_coverage": daily_basic_coverage,
                "adj_factor_coverage": adj_factor_coverage,
            },
            "categories": {category: [*updated_symbols, *failed_symbols] for category in target_categories},
            "rounds": [
                {
                    "round": 1,
                    "storage_mode": "parquet-direct",
                    "effective_target_trade_date": target_trade_date,
                    "early_stop_reason": early_stop_reason,
                    "categories": {category: [*updated_symbols, *failed_symbols] for category in target_categories},
                    "completeness": completeness,
                }
            ],
            "preflight_completeness": completeness,
            "completeness": completeness,
            "daily_basic_coverage": daily_basic_coverage,
            "adj_factor_coverage": adj_factor_coverage,
        }


def run_market_maintenance(
    market: str,
    categories: list[str] | None = None,
    max_rounds: int = 1,
    fail_on_incomplete: bool = False,
    allowed_stale_symbols: list[str] | None = None,
    deprecated_alias: bool = False,
    storage_mode: str = "auto",
    staged: bool = False,
    resume: bool = False,
    max_batches_per_run: int = 1,
    min_symbol_success_rate: float = 0.95,
    target_date: str = "auto",
    daily_window: bool = False,
    **kwargs: Any,
) -> Any:
    settings = get_market_settings(market)
    selected_categories = normalize_categories(settings.market, categories)
    requested_storage_mode = str(storage_mode or "auto").strip().lower()
    if requested_storage_mode == "auto":
        resolved_storage_mode = "parquet-direct" if settings.market == "CN" and not staged else "legacy"
    else:
        resolved_storage_mode = requested_storage_mode
    if deprecated_alias:
        print("⚠️ `quant-investor market download` 已兼容保留；请改用 `quant-investor market maintain`。")

    if requested_storage_mode not in {"auto", "legacy", "parquet-direct"}:
        raise ValueError(f"unsupported storage_mode: {storage_mode}")
    if staged:
        if settings.market != "CN":
            raise ValueError("staged maintenance is only supported for CN")
        if requested_storage_mode == "parquet-direct":
            raise ValueError("staged maintenance cannot be combined with parquet-direct storage_mode")
        from quant_investor.market.staged_maintenance import run_staged_maintenance

        return run_staged_maintenance(
            market=settings.market,
            categories=selected_categories,
            batch_size=int(kwargs.pop("batch_size", 200)),
            max_batches_per_run=int(max_batches_per_run),
            min_symbol_success_rate=float(min_symbol_success_rate),
            target_date=target_date,
            daily_window=bool(daily_window),
            resume=bool(resume),
            fail_on_incomplete=fail_on_incomplete,
            allowed_stale_symbols=allowed_stale_symbols,
            years=int(kwargs.pop("years", 3)),
            max_workers=int(kwargs.pop("max_workers", 4)),
            data_dir=kwargs.pop("data_dir", settings.data_dir),
        )
    if settings.market == "CN" and resolved_storage_mode == "legacy":
        raise ValueError("CN legacy CSV maintenance is disabled outside staged maintenance; use parquet-direct or --staged")
    if resolved_storage_mode == "parquet-direct":
        if settings.market != "CN":
            raise ValueError("parquet-direct storage mode is only supported for CN")
        maintainer = CNParquetBatchMaintainer(
            data_dir=kwargs.pop("data_dir", settings.data_dir),
            data_root=getattr(config, "MARKET_DATA_BASE_DIR", "data"),
            years=int(kwargs.pop("years", 3)),
            max_workers=int(kwargs.pop("max_workers", 4)),
            batch_size=int(kwargs.pop("batch_size", 50)),
        )
        return maintainer.maintain(
            categories=selected_categories,
            fail_on_incomplete=fail_on_incomplete,
            allowed_stale_symbols=allowed_stale_symbols,
            target_date=target_date,
        )

    if settings.market == "CN":
        if not str(getattr(config, "TUSHARE_TOKEN", "") or "").strip():
            raise RuntimeError("CN maintenance requires TUSHARE_TOKEN；请先配置主 Tushare Pro Token。")
        if not str(getattr(config, "TUSHARE_URL", "") or "").strip():
            raise RuntimeError("CN maintenance requires TUSHARE_URL；请先配置高积分 Tushare Pro URL。")

    downloader = create_downloader(settings.market, **kwargs)

    if settings.market == "CN":
        components = downloader.load_components()
        if getattr(downloader, "pro", None) is None:
            raise RuntimeError("CN maintenance 无法初始化 Tushare Pro；请检查主 Token 和高积分 URL。")
        download_results = downloader.download_all(
            components=components,
            max_rounds=max_rounds,
            fail_on_incomplete=fail_on_incomplete,
            allowed_stale_symbols=allowed_stale_symbols,
            categories=selected_categories,
        )
        completeness = downloader.build_completeness_report(
            components=components,
            allowed_stale_symbols=allowed_stale_symbols,
            categories=selected_categories,
        )
        downloader._print_completeness_summary(completeness)
        return {
            "status": "maintained",
            "download_results": download_results,
            "completeness": completeness,
            "categories": selected_categories,
        }

    universe = downloader.load_universe()
    scoped_universe = {
        key: value
        for key, value in universe.items()
        if key in selected_categories or key in {"stats", "metadata"}
    }
    return {
        "status": "maintained",
        "download_results": downloader.download_all(scoped_universe),
        "categories": selected_categories,
    }


def run_download(
    market: str,
    categories: list[str] | None = None,
    check_complete: bool = False,
    max_rounds: int = 1,
    fail_on_incomplete: bool = False,
    allowed_stale_symbols: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    settings = get_market_settings(market)
    selected_categories = normalize_categories(settings.market, categories)

    if check_complete and settings.market == "CN":
        from quant_investor.market.market_data_store import run_storage_validate

        validation = run_storage_validate(market=settings.market)
        return {
            "status": validation.get("status"),
            "market": settings.market,
            "categories": selected_categories,
            "storage_validate": validation,
            "message": "CN legacy CSV completeness checks are disabled; use Parquet storage validation.",
        }

    return run_market_maintenance(
        market=market,
        categories=categories,
        max_rounds=max_rounds,
        fail_on_incomplete=fail_on_incomplete,
        allowed_stale_symbols=allowed_stale_symbols,
        deprecated_alias=True,
        **kwargs,
    )
