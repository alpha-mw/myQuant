#!/usr/bin/env python3
"""
Enhanced Data Layer - 增强版数据层 (向后兼容 shim)

所有逻辑已迁移至 data/ 包。本文件保留向后兼容的导入接口。

新代码请使用:
    from quant_investor.data import DataHub, get_data_hub
    from quant_investor.data import DataSourceBase, TushareDataSource, YahooDataSource
    from quant_investor.data import OHLCVData, FundamentalData, MacroData
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

# ==================== 数据结构 ====================
from quant_investor.data.models import OHLCVData, TickData, FundamentalData, MacroData
from quant_investor.branch_contracts import (
    CorporateDocumentSnapshot,
    ForecastSnapshot,
    FundamentalSnapshot,
    ManagementSnapshot,
    OwnershipSnapshot,
)
from quant_investor.corporate_doc_store import CorporateDocumentStore

# ==================== 数据获取基类 ====================
from quant_investor.data.sources.base import (
    DataSourceBase,
    _parse_any_date,
    _normalize_ohlcv_frame,
    _filter_ohlcv_by_date,
)

# ==================== 数据源 ====================
from quant_investor.data.sources.tushare_cn import TushareDataSource
from quant_investor.data.sources.tushare_us import USTushareDataSource
from quant_investor.data.sources.yahoo import YahooDataSource

# USLocalCSVDataSource & USCompositeDataSource — inline for backward compat
import pandas as pd


def normalize_kline_frame_for_model(
    df: pd.DataFrame,
    target_freq: str = "B",
    fill_limit: int = 5,
) -> pd.DataFrame:
    """规范 K 线索引，确保时序模型可稳定识别频率。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=getattr(df, "columns", []))

    normalized = df.copy()
    if "date" in normalized.columns:
        index = pd.to_datetime(normalized["date"], errors="coerce")
    else:
        index = pd.to_datetime(normalized.index, errors="coerce")

    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)

    normalized = normalized.loc[~pd.isna(index)].copy()
    index = index[~pd.isna(index)]
    if normalized.empty:
        return normalized

    normalized.index = pd.DatetimeIndex(index)
    if normalized.index.tz is not None:
        normalized.index = normalized.index.tz_localize(None)
    normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()

    inferred = pd.infer_freq(normalized.index)
    effective_freq = inferred if inferred in {"B", "C", "D"} else target_freq
    normalized = normalized.asfreq(effective_freq)
    normalized = normalized.ffill(limit=max(int(fill_limit), 0))
    normalized.index.name = "date"
    normalized["date"] = normalized.index
    return normalized


class USLocalCSVDataSource(DataSourceBase):
    """本地美股 CSV 数据源，读取 data/us_market_full。"""

    def __init__(self, data_dir=None):
        from quant_investor.data.storage.csv_store import CSVStore
        default_dir = Path(__file__).resolve().parents[1] / "data" / "us_market_full"
        self._store = CSVStore(data_dir or default_dir)

    def get_ohlcv(self, symbol, start_date, end_date, freq="1d"):
        return self._store.read(symbol, start_date, end_date)

    def get_fundamental(self, symbol):
        fundamental_path = self._store._base_dir / f"{symbol}_fundamental.json"
        if fundamental_path.exists():
            import json
            try:
                data = json.loads(fundamental_path.read_text())
                fields = {k: v for k, v in data.items() if hasattr(FundamentalData, k)}
                return FundamentalData(symbol=symbol, **fields)
            except Exception:
                pass
        return FundamentalData(symbol=symbol)


def _fundamental_has_data(fd: FundamentalData) -> bool:
    """检查 FundamentalData 是否包含有效数据（至少3个非 None 字段）。"""
    fields = ("roe", "roa", "gross_margin", "net_margin", "pe", "pb", "debt_ratio", "current_ratio")
    return sum(1 for f in fields if getattr(fd, f, None) is not None) >= 3


class USCompositeDataSource(DataSourceBase):
    """美股复合数据源：Tushare（高积分）→ Yahoo Finance → SEC EDGAR。"""

    def __init__(self, local_data_dir=None, max_staleness_days=7):
        self._local = USLocalCSVDataSource(local_data_dir)
        self._tushare = USTushareDataSource()
        self._yahoo = YahooDataSource()
        self._max_staleness_days = max_staleness_days
        self.last_ohlcv_source = "unknown"
        self.last_fundamental_source = "unknown"

    def get_ohlcv(self, symbol, start_date, end_date):
        requested_end = _parse_any_date(end_date)
        local_df = self._local.get_ohlcv(symbol, start_date, end_date)
        if not local_df.empty:
            latest_date = pd.to_datetime(local_df["date"], errors="coerce").max()
            if pd.isna(requested_end) or (
                pd.notna(latest_date)
                and latest_date >= requested_end - pd.Timedelta(days=self._max_staleness_days)
            ):
                self.last_ohlcv_source = "local_csv"
                return local_df

        tushare_df = self._tushare.get_ohlcv(symbol, start_date, end_date)
        if not tushare_df.empty:
            self.last_ohlcv_source = "tushare_us_daily"
            return tushare_df

        yahoo_df = self._yahoo.get_ohlcv(symbol, start_date, end_date)
        if not yahoo_df.empty:
            self.last_ohlcv_source = "yahoo"
            return yahoo_df

        if not local_df.empty:
            self.last_ohlcv_source = "local_csv_stale"
            return local_df

        self.last_ohlcv_source = "unavailable"
        return pd.DataFrame()

    def get_fundamental(self, symbol):
        """
        三级回退获取基本面数据：
        1. Tushare（us_fina_indicator + 三张报表，高积分接口）
        2. Yahoo Finance（yfinance.Ticker.info）
        3. SEC EDGAR（XBRL company facts，公开免费接口）
        """
        # --- 1. Tushare ---
        try:
            result = self._tushare.get_fundamental(symbol)
            if _fundamental_has_data(result):
                self.last_fundamental_source = "tushare"
                return result
        except Exception:
            pass

        # --- 2. Yahoo Finance ---
        try:
            result = self._yahoo.get_fundamental(symbol)
            if _fundamental_has_data(result):
                self.last_fundamental_source = "yahoo"
                return result
        except Exception:
            pass

        # --- 3. SEC EDGAR（公开渠道回退）---
        try:
            from quant_investor.sec_fundamental import fetch_sec_fundamental
            sec_data = fetch_sec_fundamental(symbol)
            if sec_data:
                result = FundamentalData(symbol=symbol, **{
                    k: v for k, v in sec_data.items()
                    if hasattr(FundamentalData, k) and v is not None
                })
                if _fundamental_has_data(result):
                    self.last_fundamental_source = "sec_edgar"
                    return result
        except Exception:
            pass

        self.last_fundamental_source = "unavailable"
        return FundamentalData(symbol=symbol)

    def get_daily_basic(self, symbol, trade_date=None):
        """获取 PE/PB 等每日基础指标：Tushare 优先，回退 Yahoo。"""
        # Tushare us_fina_indicator 包含 PE/PB
        try:
            result = self._tushare.get_fundamental(symbol)
            if any(getattr(result, f, None) is not None for f in ("pe", "pb")):
                return {
                    "pe": getattr(result, "pe", None),
                    "pb": getattr(result, "pb", None),
                    "ps": getattr(result, "ps", None),
                    "dividend_yield": getattr(result, "dividend_yield", None),
                }
        except Exception:
            pass
        # Yahoo fallback
        try:
            result = self._yahoo.get_fundamental(symbol)
            return {
                "pe": getattr(result, "pe", None),
                "pb": getattr(result, "pb", None),
                "ps": getattr(result, "ps", None),
                "dividend_yield": getattr(result, "dividend_yield", None),
            }
        except Exception:
            return {}


# ==================== 数据处理 ====================
from quant_investor.data.processing.cleaner import DataCleaner
from quant_investor.data.processing.features import FeatureEngineer
from quant_investor.data.processing.labels import LabelGenerator


# ==================== 增强版数据层主类 ====================
from quant_investor.data.hub import DataHub


class EnhancedDataLayer(DataHub):
    """向后兼容别名，并补充 V9 点时快照接口。"""

    def __init__(self, market: str = "CN", verbose: bool = True):
        super().__init__(market=market, verbose=verbose)
        default_doc_dir = Path(__file__).resolve().parents[1] / "data" / "corporate_docs"
        self._doc_store = CorporateDocumentStore(base_dir=default_doc_dir)

    @staticmethod
    def _normalize_as_of(as_of: Any) -> tuple[str, str]:
        ts = pd.to_datetime(as_of or datetime.now(), errors="coerce")
        if pd.isna(ts):
            ts = pd.Timestamp(datetime.now())
        return ts.strftime("%Y-%m-%d"), ts.strftime("%Y%m%d")

    @staticmethod
    def _time_fields(as_of: Any) -> dict[str, str]:
        ts = pd.to_datetime(as_of or datetime.now(), errors="coerce")
        if pd.isna(ts):
            ts = pd.Timestamp(datetime.now())
        now_ts = pd.Timestamp(datetime.now())
        as_of_text = ts.strftime("%Y-%m-%d")
        effective_time = ts.strftime("%Y-%m-%dT%H:%M:%S")
        ingest_time = now_ts.strftime("%Y-%m-%dT%H:%M:%S")
        return {
            "as_of": as_of_text,
            "publish_time": effective_time,
            "effective_time": effective_time,
            "ingest_time": ingest_time,
        }

    @classmethod
    def _neutral_snapshot_state(
        cls,
        snapshot_type: str,
        as_of: Any,
        source: str = "neutral",
        reason: str = "provider_missing",
        provider_name: str = "unknown",
        is_estimated: bool = True,
    ) -> dict[str, Any]:
        time_fields = cls._time_fields(as_of)
        return {
            **time_fields,
            "source": source,
            "revision_id": f"{snapshot_type}:{provider_name}:{time_fields['as_of']}",
            "is_estimated": is_estimated,
            "data_quality": {
                "status": "neutral_snapshot",
                "reason": reason,
                "provider_missing": reason == "provider_missing",
                "provider_name": provider_name,
            },
            "provenance": {
                "snapshot_type": snapshot_type,
                "provider_name": provider_name,
                "reason": reason,
                "provider_missing": reason == "provider_missing",
            },
        }

    @staticmethod
    def _apply_missing_semantics(
        snapshot: Any,
        *,
        reason: str,
        missing_scope: str,
        provider_name: str,
        provider_missing: bool = False,
        snapshot_missing: bool = False,
        note: str = "",
        source: str | None = None,
    ) -> Any:
        snapshot.available = False
        if source is not None:
            snapshot.source = source
        snapshot.data_quality.update(
            {
                "status": "neutral_snapshot",
                "reason": reason,
                "missing_scope": missing_scope,
                "provider_missing": provider_missing,
                "snapshot_missing": snapshot_missing,
                "provider_name": provider_name,
            }
        )
        snapshot.provenance.update(
            {
                "reason": reason,
                "missing_scope": missing_scope,
                "provider_missing": provider_missing,
                "snapshot_missing": snapshot_missing,
                "provider_name": provider_name,
            }
        )
        if note and note not in snapshot.notes:
            snapshot.notes.append(note)
        return snapshot

    @staticmethod
    def _mark_available(snapshot: Any, provider_name: str) -> Any:
        snapshot.data_quality.update(
            {
                "status": "provider_snapshot",
                "reason": "provider_payload",
                "missing_scope": "",
                "provider_missing": False,
                "snapshot_missing": False,
                "provider_name": provider_name,
            }
        )
        snapshot.provenance.update(
            {
                "reason": "provider_payload",
                "missing_scope": "",
                "provider_missing": False,
                "snapshot_missing": False,
                "provider_name": provider_name,
            }
        )
        snapshot.is_estimated = False
        return snapshot

    @staticmethod
    def _normalize_snapshot_missing(
        snapshot: Any,
        *,
        default_scope: str,
        default_provider: str,
        default_reason: str,
        default_note: str,
        default_source: str | None = None,
    ) -> Any:
        if snapshot.available:
            return snapshot
        data_quality = dict(getattr(snapshot, "data_quality", {}) or {})
        missing_scope = str(data_quality.get("missing_scope", default_scope) or default_scope)
        provider_missing = bool(data_quality.get("provider_missing", False))
        snapshot_missing = bool(
            data_quality.get("snapshot_missing", not provider_missing)
        )
        reason = str(data_quality.get("reason", default_reason) or default_reason)
        provider_name = str(data_quality.get("provider_name", default_provider) or default_provider)
        return EnhancedDataLayer._apply_missing_semantics(
            snapshot,
            reason=reason,
            missing_scope=missing_scope,
            provider_name=provider_name,
            provider_missing=provider_missing,
            snapshot_missing=snapshot_missing,
            note=default_note,
            source=default_source,
        )

    def get_point_in_time_fundamental_snapshot(self, symbol: str, as_of: Any) -> FundamentalSnapshot:
        """获取点时财务/估值快照，缺 provider 时返回 neutral snapshot。"""
        as_of_text, trade_date = self._normalize_as_of(as_of)
        snapshot = FundamentalSnapshot(
            symbol=symbol,
            available=False,
            **self._neutral_snapshot_state(
                snapshot_type="fundamental",
                as_of=as_of_text,
                provider_name=str(getattr(self, "last_fundamental_source", "direct")),
                reason="snapshot_unavailable",
            ),
        )
        try:
            fundamental = self.get_fundamental(symbol)
            daily_basic = self.get_daily_basic(symbol, trade_date)
        except Exception as exc:
            return self._apply_missing_semantics(
                snapshot,
                reason="provider_error",
                missing_scope="global",
                provider_name=str(getattr(self, "last_fundamental_source", "direct")),
                note=f"fundamental_provider_error: {type(exc).__name__}",
            )

        raw_values = {
            "roe": getattr(fundamental, "roe", None),
            "roa": getattr(fundamental, "roa", None),
            "gross_margin": getattr(fundamental, "gross_margin", None),
            "net_margin": getattr(fundamental, "net_margin", None),
            "revenue_growth": getattr(fundamental, "revenue_growth", None),
            "profit_growth": getattr(fundamental, "profit_growth", None),
            "debt_ratio": getattr(fundamental, "debt_ratio", None),
            "current_ratio": getattr(fundamental, "current_ratio", None),
            "cash_flow": getattr(fundamental, "cash_flow", None),
            "pe": daily_basic.get("pe", getattr(fundamental, "pe", None)),
            "pb": daily_basic.get("pb", getattr(fundamental, "pb", None)),
            "ps": daily_basic.get("ps", getattr(fundamental, "ps", None)),
            "dividend_yield": daily_basic.get(
                "dividend_yield",
                getattr(fundamental, "dividend_yield", None),
            ),
        }
        available_fields = 0
        for field, value in raw_values.items():
            if value is None or pd.isna(value):
                continue
            setattr(snapshot, field, float(value))
            available_fields += 1

        snapshot.available = available_fields > 0
        snapshot.source = (
            str(getattr(self, "last_fundamental_source", "direct")) if snapshot.available else "neutral"
        )
        snapshot.provenance["provider_name"] = snapshot.source
        snapshot.data_quality["provider_name"] = snapshot.source
        if not snapshot.available:
            return self._apply_missing_semantics(
                snapshot,
                reason="snapshot_missing",
                missing_scope="symbol",
                provider_name=snapshot.source,
                snapshot_missing=True,
                note="fundamental_snapshot_unavailable",
            )
        else:
            self._mark_available(snapshot, snapshot.source)
        return snapshot

    def get_earnings_forecast_snapshot(self, symbol: str, as_of: Any) -> ForecastSnapshot:
        """获取盈利预测快照；provider 缺失时返回 neutral snapshot。"""
        as_of_text, _ = self._normalize_as_of(as_of)
        snapshot = ForecastSnapshot(
            symbol=symbol,
            available=False,
            provider="none",
            **self._neutral_snapshot_state(
                snapshot_type="forecast",
                as_of=as_of_text,
                provider_name="earnings_forecast_provider",
            ),
        )
        provider = getattr(self._source, "get_earnings_forecast_snapshot", None)
        if not callable(provider):
            return self._apply_missing_semantics(
                snapshot,
                reason="provider_missing",
                missing_scope="global",
                provider_name="earnings_forecast_provider",
                provider_missing=True,
                note="forecast_provider_missing",
            )
        try:
            payload = provider(symbol=symbol, as_of=as_of_text)
        except Exception as exc:
            return self._apply_missing_semantics(
                snapshot,
                reason="provider_error",
                missing_scope="global",
                provider_name="earnings_forecast_provider",
                note=f"forecast_provider_error: {type(exc).__name__}",
            )
        if isinstance(payload, ForecastSnapshot):
            if payload.available:
                payload.data_quality.setdefault("provider_missing", False)
                payload.provenance.setdefault("provider_missing", False)
                payload.data_quality.setdefault("missing_scope", "")
                payload.provenance.setdefault("missing_scope", "")
                return self._mark_available(payload, str(payload.provider or payload.source or "provider"))
            return self._normalize_snapshot_missing(
                payload,
                default_scope="symbol",
                default_provider=str(payload.provider or payload.source or "earnings_forecast_provider"),
                default_reason="snapshot_missing",
                default_note="forecast_snapshot_missing",
            )
        if isinstance(payload, dict):
            for field, value in payload.items():
                if hasattr(snapshot, field):
                    setattr(snapshot, field, value)
            snapshot.available = bool(snapshot.available or snapshot.coverage_count > 0)
            snapshot.source = str(payload.get("source", "provider"))
            snapshot.provider = str(payload.get("provider", snapshot.provider))
            if snapshot.available:
                return self._mark_available(snapshot, snapshot.provider)
            return self._apply_missing_semantics(
                snapshot,
                reason="snapshot_missing",
                missing_scope=str(payload.get("missing_scope", "symbol") or "symbol"),
                provider_name=snapshot.provider,
                snapshot_missing=True,
                note="forecast_snapshot_missing",
            )
        return self._apply_missing_semantics(
            snapshot,
            reason="snapshot_missing",
            missing_scope="symbol",
            provider_name="earnings_forecast_provider",
            snapshot_missing=True,
            note="forecast_provider_empty",
        )

    def get_management_snapshot(self, symbol: str, as_of: Any) -> ManagementSnapshot:
        """获取管理层快照；provider 缺失时返回 neutral snapshot。"""
        as_of_text, _ = self._normalize_as_of(as_of)
        snapshot = ManagementSnapshot(
            symbol=symbol,
            available=False,
            **self._neutral_snapshot_state(
                snapshot_type="management",
                as_of=as_of_text,
                provider_name="management_provider",
            ),
        )
        provider = getattr(self._source, "get_management_snapshot", None)
        if not callable(provider):
            return self._apply_missing_semantics(
                snapshot,
                reason="provider_missing",
                missing_scope="global",
                provider_name="management_provider",
                provider_missing=True,
                note="management_provider_missing",
            )
        try:
            payload = provider(symbol=symbol, as_of=as_of_text)
        except Exception as exc:
            return self._apply_missing_semantics(
                snapshot,
                reason="provider_error",
                missing_scope="global",
                provider_name="management_provider",
                note=f"management_provider_error: {type(exc).__name__}",
            )
        if isinstance(payload, ManagementSnapshot):
            if payload.available:
                payload.data_quality.setdefault("provider_missing", False)
                payload.provenance.setdefault("provider_missing", False)
                payload.data_quality.setdefault("missing_scope", "")
                payload.provenance.setdefault("missing_scope", "")
                return self._mark_available(payload, str(payload.source or "management_provider"))
            return self._normalize_snapshot_missing(
                payload,
                default_scope="symbol",
                default_provider=str(payload.source or "management_provider"),
                default_reason="snapshot_missing",
                default_note="management_snapshot_missing",
            )
        if isinstance(payload, dict):
            for field, value in payload.items():
                if hasattr(snapshot, field):
                    setattr(snapshot, field, value)
            snapshot.available = bool(payload.get("available", False))
            snapshot.source = str(payload.get("source", "provider"))
            if snapshot.available:
                return self._mark_available(snapshot, snapshot.source)
            return self._apply_missing_semantics(
                snapshot,
                reason="snapshot_missing",
                missing_scope=str(payload.get("missing_scope", "symbol") or "symbol"),
                provider_name=snapshot.source,
                snapshot_missing=True,
                note="management_snapshot_missing",
            )
        return self._apply_missing_semantics(
            snapshot,
            reason="snapshot_missing",
            missing_scope="symbol",
            provider_name="management_provider",
            snapshot_missing=True,
            note="management_provider_empty",
        )

    def get_ownership_snapshot(self, symbol: str, as_of: Any) -> OwnershipSnapshot:
        """获取股东结构快照；provider 缺失时返回 neutral snapshot。"""
        as_of_text, _ = self._normalize_as_of(as_of)
        snapshot = OwnershipSnapshot(
            symbol=symbol,
            available=False,
            **self._neutral_snapshot_state(
                snapshot_type="ownership",
                as_of=as_of_text,
                provider_name="ownership_provider",
            ),
        )
        provider = getattr(self._source, "get_ownership_snapshot", None)
        if not callable(provider):
            return self._apply_missing_semantics(
                snapshot,
                reason="provider_missing",
                missing_scope="global",
                provider_name="ownership_provider",
                provider_missing=True,
                note="ownership_provider_missing",
            )
        try:
            payload = provider(symbol=symbol, as_of=as_of_text)
        except Exception as exc:
            return self._apply_missing_semantics(
                snapshot,
                reason="provider_error",
                missing_scope="global",
                provider_name="ownership_provider",
                note=f"ownership_provider_error: {type(exc).__name__}",
            )
        if isinstance(payload, OwnershipSnapshot):
            if payload.available:
                payload.data_quality.setdefault("provider_missing", False)
                payload.provenance.setdefault("provider_missing", False)
                payload.data_quality.setdefault("missing_scope", "")
                payload.provenance.setdefault("missing_scope", "")
                return self._mark_available(payload, str(payload.source or "ownership_provider"))
            return self._normalize_snapshot_missing(
                payload,
                default_scope="symbol",
                default_provider=str(payload.source or "ownership_provider"),
                default_reason="snapshot_missing",
                default_note="ownership_snapshot_missing",
            )
        if isinstance(payload, dict):
            for field, value in payload.items():
                if hasattr(snapshot, field):
                    setattr(snapshot, field, value)
            snapshot.available = bool(payload.get("available", False))
            snapshot.source = str(payload.get("source", "provider"))
            if snapshot.available:
                return self._mark_available(snapshot, snapshot.source)
            return self._apply_missing_semantics(
                snapshot,
                reason="snapshot_missing",
                missing_scope=str(payload.get("missing_scope", "symbol") or "symbol"),
                provider_name=snapshot.source,
                snapshot_missing=True,
                note="ownership_snapshot_missing",
            )
        return self._apply_missing_semantics(
            snapshot,
            reason="snapshot_missing",
            missing_scope="symbol",
            provider_name="ownership_provider",
            snapshot_missing=True,
            note="ownership_provider_empty",
        )

    def get_document_semantic_snapshot(self, symbol: str, as_of: Any) -> CorporateDocumentSnapshot:
        """从离线 semantic snapshot 读取公司文档语义。"""
        as_of_text, _ = self._normalize_as_of(as_of)
        if not self._doc_store.base_dir.exists():
            snapshot = CorporateDocumentSnapshot(
                symbol=symbol,
                available=False,
                **self._neutral_snapshot_state(
                    snapshot_type="corporate_document",
                    as_of=as_of_text,
                    source="offline_snapshot",
                    provider_name="offline_doc_store",
                    reason="provider_missing",
                ),
            )
            return self._apply_missing_semantics(
                snapshot,
                reason="provider_missing",
                missing_scope="global",
                provider_name="offline_doc_store",
                provider_missing=True,
                note="document_store_missing",
                source="offline_snapshot",
            )

        document_path = self._doc_store._symbol_path(symbol)
        if not document_path.exists():
            snapshot = CorporateDocumentSnapshot(
                symbol=symbol,
                available=False,
                **self._neutral_snapshot_state(
                    snapshot_type="corporate_document",
                    as_of=as_of_text,
                    source="offline_snapshot",
                    provider_name="offline_doc_store",
                    reason="snapshot_missing",
                ),
            )
            return self._apply_missing_semantics(
                snapshot,
                reason="snapshot_missing",
                missing_scope="symbol",
                provider_name="offline_doc_store",
                snapshot_missing=True,
                note="document_snapshot_missing",
                source="offline_snapshot",
            )
        try:
            snapshot = self._doc_store.get_semantic_snapshot(symbol=symbol, as_of=as_of_text)
        except Exception as exc:
            snapshot = CorporateDocumentSnapshot(
                symbol=symbol,
                available=False,
                **self._neutral_snapshot_state(
                    snapshot_type="corporate_document",
                    as_of=as_of_text,
                    provider_name="offline_doc_store",
                    reason="provider_error",
                ),
            )
            return self._apply_missing_semantics(
                snapshot,
                reason="provider_error",
                missing_scope="global",
                provider_name="offline_doc_store",
                note=f"document_snapshot_error: {type(exc).__name__}",
                source="offline_snapshot",
            )

        if snapshot.available:
            snapshot.data_quality.setdefault("provider_missing", False)
            snapshot.data_quality.setdefault("snapshot_missing", False)
            snapshot.data_quality.setdefault("missing_scope", "")
            snapshot.provenance.setdefault("provider_missing", False)
            snapshot.provenance.setdefault("snapshot_missing", False)
            snapshot.provenance.setdefault("missing_scope", "")
            return self._mark_available(snapshot, "offline_doc_store")

        return self._normalize_snapshot_missing(
            snapshot,
            default_scope="symbol",
            default_provider="offline_doc_store",
            default_reason="snapshot_missing",
            default_note="document_snapshot_missing",
            default_source="offline_snapshot",
        )


# ==================== 测试 ====================

if __name__ == '__main__':
    print("=" * 80)
    print("Enhanced Data Layer - 测试 (via data/ package)")
    print("=" * 80)

    # 测试A股
    print("\n【测试A股】")
    data_layer = EnhancedDataLayer(market="CN", verbose=True)

    df = data_layer.fetch_and_process(
        symbol="000001.SZ",
        start_date="20240101",
        end_date="20240225",
        label_periods=5
    )

    if not df.empty:
        print(f"\n数据预览:")
        print(df.head())
        print(f"\n因子列表:")
        factor_cols = [c for c in df.columns if c.startswith(('return_', 'volatility_', 'rsi_', 'macd_', 'ma_bias_', 'label_'))]
        print(factor_cols)
