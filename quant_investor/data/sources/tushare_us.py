#!/usr/bin/env python3
"""美股 Tushare 数据源"""

import pandas as pd
import logging
from typing import Any


def _safe_float(value: Any) -> float | None:
    """将任意值安全转换为 float，None/NaN/空字符串返回 None。"""
    if value is None:
        return None
    try:
        v = float(value)
        import math
        return None if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return None


from quant_investor.data.models import FundamentalData
from quant_investor.data.sources.base import DataSourceBase, _normalize_ohlcv_frame, _filter_ohlcv_by_date
from quant_investor.data._tushare_client import TushareClientPool


class USTushareDataSource(DataSourceBase):
    """Tushare 美股日线数据源"""

    def __init__(self):
        self._client = TushareClientPool()
        self._quota_exhausted = False
        self._logger = logging.getLogger("data.sources.TushareUS")

    def get_ohlcv(self, symbol: str, start_date: str, end_date: str, freq: str = "D") -> pd.DataFrame:
        if not self._client.available or self._quota_exhausted:
            return pd.DataFrame()
        try:
            df = self._client.query("us_daily", ts_code=symbol, start_date=start_date, end_date=end_date)
            if df is None or df.empty:
                return pd.DataFrame()
            df = _normalize_ohlcv_frame(df)
            return _filter_ohlcv_by_date(df, start_date, end_date)
        except Exception as e:
            if "每天最多访问该接口" in str(e):
                self._quota_exhausted = True
            self._logger.warning(f"获取OHLCV失败 {symbol}: {e}")
            return pd.DataFrame()

    def get_fundamental(self, symbol: str) -> FundamentalData:
        """从 Tushare us_fina_indicator 拉取财务指标，失败则从三张报表计算。"""
        if not self._client.available or self._quota_exhausted:
            return FundamentalData(symbol=symbol)

        # --- 优先尝试 us_fina_indicator（预计算财务比率）---
        try:
            df = self._client.query("us_fina_indicator", ts_code=symbol)
            if df is not None and not df.empty:
                if "ann_date" in df.columns:
                    df = df.sort_values("ann_date", ascending=False)
                row = df.iloc[0]
                result = FundamentalData(
                    symbol=symbol,
                    report_date=str(row.get("ann_date", "")),
                    pe=_safe_float(row.get("pe")),
                    pb=_safe_float(row.get("pb")),
                    ps=_safe_float(row.get("ps")),
                    roe=_safe_float(row.get("roe")),
                    roa=_safe_float(row.get("roa")),
                    gross_margin=_safe_float(row.get("grossprofit_margin")),
                    net_margin=_safe_float(row.get("netprofit_margin")),
                    debt_ratio=_safe_float(row.get("debt_to_assets")),
                    current_ratio=_safe_float(row.get("current_ratio")),
                )
                if any(v is not None for v in [result.roe, result.pe, result.net_margin]):
                    return result
        except Exception as e:
            if "每天最多访问该接口" in str(e):
                self._quota_exhausted = True
            self._logger.warning(f"us_fina_indicator 查询失败 {symbol}: {e}")

        # --- 回退：从三张报表计算 ---
        return self._compute_from_statements(symbol)

    def get_daily_basic(self, symbol: str, trade_date: str | None = None) -> dict[str, Any]:
        """返回美股日常估值字段，供点时快照使用。"""
        if not self._client.available or self._quota_exhausted:
            return {}
        try:
            df = self._client.query("us_fina_indicator", ts_code=symbol)
            if df is None or df.empty:
                return {}
            if "ann_date" in df.columns:
                df = df.sort_values("ann_date", ascending=False)
            row = df.iloc[0]
            result = {
                "pe": _safe_float(row.get("pe")),
                "pb": _safe_float(row.get("pb")),
                "ps": _safe_float(row.get("ps")),
            }
            return {key: value for key, value in result.items() if value is not None}
        except Exception as e:
            if "每天最多访问该接口" in str(e):
                self._quota_exhausted = True
            self._logger.warning(f"us_fina_indicator daily_basic 查询失败 {symbol}: {e}")
            return {}

    def _compute_from_statements(self, symbol: str) -> FundamentalData:
        """从利润表 + 资产负债表 + 现金流量表计算基本面指标。"""
        result = FundamentalData(symbol=symbol)
        try:
            income_df = self._client.query("us_income", ts_code=symbol)
            if income_df is not None and not income_df.empty:
                if "ann_date" in income_df.columns:
                    income_df = income_df.sort_values("ann_date", ascending=False)
                row0 = income_df.iloc[0]
                revenue = _safe_float(row0.get("revenue") or row0.get("total_revenue"))
                gross_profit = _safe_float(row0.get("gross_profit"))
                net_profit = _safe_float(row0.get("net_profit") or row0.get("net_income"))
                result.report_date = str(row0.get("ann_date", ""))
                if revenue and revenue != 0:
                    if gross_profit is not None:
                        result.gross_margin = round(gross_profit / revenue * 100, 4)
                    if net_profit is not None:
                        result.net_margin = round(net_profit / revenue * 100, 4)
                # YoY revenue growth
                if len(income_df) >= 2:
                    prev_revenue = _safe_float(
                        income_df.iloc[1].get("revenue") or income_df.iloc[1].get("total_revenue")
                    )
                    prev_net = _safe_float(
                        income_df.iloc[1].get("net_profit") or income_df.iloc[1].get("net_income")
                    )
                    if revenue and prev_revenue and prev_revenue != 0:
                        result.revenue_growth = round((revenue - prev_revenue) / abs(prev_revenue) * 100, 4)
                    if net_profit and prev_net and prev_net != 0:
                        result.profit_growth = round((net_profit - prev_net) / abs(prev_net) * 100, 4)
        except Exception as e:
            self._logger.warning(f"us_income 查询失败 {symbol}: {e}")

        try:
            bs_df = self._client.query("us_balancesheet", ts_code=symbol)
            if bs_df is not None and not bs_df.empty:
                if "ann_date" in bs_df.columns:
                    bs_df = bs_df.sort_values("ann_date", ascending=False)
                row0 = bs_df.iloc[0]
                total_assets = _safe_float(row0.get("total_assets"))
                total_liab = _safe_float(row0.get("total_liab") or row0.get("total_liabilities"))
                equity = _safe_float(row0.get("total_equity") or row0.get("stockholders_equity"))
                current_assets = _safe_float(row0.get("total_cur_assets") or row0.get("current_assets"))
                current_liab = _safe_float(row0.get("total_cur_liab") or row0.get("current_liabilities"))
                if total_assets and total_liab and total_assets != 0:
                    result.debt_ratio = round(total_liab / total_assets * 100, 4)
                if current_assets and current_liab and current_liab != 0:
                    result.current_ratio = round(current_assets / current_liab, 4)
                # ROE from net profit / equity
                if equity and equity != 0 and result.net_margin is not None:
                    # approximate: net_profit derived from margin + revenue
                    pass
        except Exception as e:
            self._logger.warning(f"us_balancesheet 查询失败 {symbol}: {e}")

        try:
            cf_df = self._client.query("us_cashflow", ts_code=symbol)
            if cf_df is not None and not cf_df.empty:
                if "ann_date" in cf_df.columns:
                    cf_df = cf_df.sort_values("ann_date", ascending=False)
                row0 = cf_df.iloc[0]
                result.cash_flow = _safe_float(
                    row0.get("net_cash_flows_oper") or row0.get("operating_cash_flow")
                )
        except Exception as e:
            self._logger.warning(f"us_cashflow 查询失败 {symbol}: {e}")

        return result
