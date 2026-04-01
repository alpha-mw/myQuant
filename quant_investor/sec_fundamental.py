"""
SEC EDGAR 公开数据源 — 美股基本面数据回退层。

当 Tushare 和 Yahoo Finance 均无法提供足够数据时，
从 SEC EDGAR 免费 API 获取 XBRL 财务数据并计算基本面指标。

数据来源：
  - 公司 CIK 索引：https://www.sec.gov/files/company_tickers.json
  - XBRL 公司财务事实：https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json
  - 公司提交记录：https://data.sec.gov/submissions/CIK{cik10}.json

频率限制：SEC EDGAR 要求 User-Agent 头，并建议不超过 10 req/s。
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any

_logger = logging.getLogger("sec_fundamental")

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_SEC_BASE = "https://data.sec.gov"
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_USER_AGENT = "quant-investor-research/1.0 (research@example.com)"
_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "sec_cache"
_TICKERS_CACHE = _CACHE_DIR / "company_tickers.json"
_TICKERS_TTL_DAYS = 7  # 每隔7天刷新 ticker→CIK 映射

# 最近一次请求时间，实现简单限速
_last_request_lock = threading.Lock()
_last_request_time: float = 0.0
_MIN_REQUEST_INTERVAL = 0.12  # ~8 req/s，低于 SEC 10 req/s 上限


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: int = 15) -> dict | None:
    """带限速和 User-Agent 的 HTTP GET，返回解析后的 JSON 或 None。"""
    global _last_request_time
    try:
        import urllib.request
        with _last_request_lock:
            now = time.monotonic()
            wait = _MIN_REQUEST_INTERVAL - (now - _last_request_time)
            if wait > 0:
                time.sleep(wait)
            _last_request_time = time.monotonic()

        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        _logger.debug(f"SEC HTTP GET failed [{url}]: {exc}")
        return None


# ---------------------------------------------------------------------------
# CIK lookup
# ---------------------------------------------------------------------------

def _ensure_cache_dir() -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_tickers_map() -> dict[str, str]:
    """返回 ticker(大写) → CIK(10位补零字符串) 的映射。带本地缓存，TTL 7天。"""
    _ensure_cache_dir()
    if _TICKERS_CACHE.exists():
        age_days = (time.time() - _TICKERS_CACHE.stat().st_mtime) / 86400
        if age_days < _TICKERS_TTL_DAYS:
            try:
                data = json.loads(_TICKERS_CACHE.read_text(encoding="utf-8"))
                return data
            except Exception:
                pass

    raw = _http_get(_TICKERS_URL)
    if not raw:
        return {}

    # SEC 格式: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    mapping: dict[str, str] = {}
    for entry in raw.values():
        ticker = str(entry.get("ticker", "")).upper().strip()
        cik = str(entry.get("cik_str", "")).strip()
        if ticker and cik:
            mapping[ticker] = cik.zfill(10)

    try:
        _TICKERS_CACHE.write_text(json.dumps(mapping), encoding="utf-8")
    except Exception:
        pass

    return mapping


def get_cik(ticker: str) -> str | None:
    """将股票代码转换为 SEC CIK（10位补零字符串），找不到返回 None。"""
    mapping = _load_tickers_map()
    return mapping.get(ticker.upper().strip())


# ---------------------------------------------------------------------------
# XBRL facts parser
# ---------------------------------------------------------------------------

def _get_company_facts(cik10: str) -> dict | None:
    """从 SEC EDGAR 获取公司所有 XBRL 财务事实。"""
    url = f"{_SEC_BASE}/api/xbrl/companyfacts/CIK{cik10}.json"
    return _http_get(url)


def _latest_annual_value(facts: dict, concept: str, namespace: str = "us-gaap") -> float | None:
    """
    从 XBRL facts dict 中取出指定 concept 的最新年报值（10-K）。
    优先取 USD 单位，其次取无单位（pure）。
    """
    try:
        ns_data = facts.get("facts", {}).get(namespace, {})
        concept_data = ns_data.get(concept, {})
        units = concept_data.get("units", {})

        # 优先 USD，其次 pure（比率类指标）
        for unit_key in ("USD", "pure", "shares"):
            entries = units.get(unit_key, [])
            if not entries:
                continue
            # 只保留年报（form=10-K）
            annual = [e for e in entries if e.get("form") in ("10-K", "10-K/A") and e.get("val") is not None]
            if not annual:
                # 放宽：允许 20-F (外国私人发行人)
                annual = [e for e in entries if e.get("form") in ("20-F", "20-F/A") and e.get("val") is not None]
            if not annual:
                continue
            # 按 end 日期排序，取最新
            annual.sort(key=lambda e: e.get("end", ""), reverse=True)
            val = annual[0].get("val")
            if val is not None:
                v = float(val)
                if not (math.isnan(v) or math.isinf(v)):
                    return v
    except Exception:
        pass
    return None


def _latest_two_annual_values(facts: dict, concept: str, namespace: str = "us-gaap") -> tuple[float | None, float | None]:
    """返回最新和上一年的年报值，用于计算 YoY 增长率。"""
    try:
        ns_data = facts.get("facts", {}).get(namespace, {})
        entries = ns_data.get(concept, {}).get("units", {}).get("USD", [])
        annual = [e for e in entries if e.get("form") in ("10-K", "10-K/A", "20-F") and e.get("val") is not None]
        annual.sort(key=lambda e: e.get("end", ""), reverse=True)
        if len(annual) >= 2:
            return float(annual[0]["val"]), float(annual[1]["val"])
        if len(annual) == 1:
            return float(annual[0]["val"]), None
    except Exception:
        pass
    return None, None


# ---------------------------------------------------------------------------
# 主入口：计算基本面指标
# ---------------------------------------------------------------------------

def _safe_div(num: float | None, denom: float | None, scale: float = 1.0) -> float | None:
    if num is None or denom is None or denom == 0:
        return None
    return round(num / denom * scale, 4)


def _yoy_growth(latest: float | None, prev: float | None) -> float | None:
    if latest is None or prev is None or prev == 0:
        return None
    return round((latest - prev) / abs(prev) * 100, 4)


def fetch_sec_fundamental(ticker: str) -> dict[str, Any]:
    """
    从 SEC EDGAR 获取美股基本面数据，返回可填充 FundamentalData 的字段字典。
    成功时返回 dict（部分字段可能为 None），失败时返回空 dict。
    """
    cik = get_cik(ticker)
    if not cik:
        _logger.debug(f"SEC: CIK not found for {ticker}")
        return {}

    facts = _get_company_facts(cik)
    if not facts:
        _logger.debug(f"SEC: No XBRL facts for {ticker} (CIK={cik})")
        return {}

    # --- 营收 ---
    revenues, revenues_prev = _latest_two_annual_values(facts, "Revenues")
    if revenues is None:
        revenues, revenues_prev = _latest_two_annual_values(facts, "RevenueFromContractWithCustomerExcludingAssessedTax")
    if revenues is None:
        revenues, revenues_prev = _latest_two_annual_values(facts, "SalesRevenueNet")

    # --- 净利润 ---
    net_income, net_income_prev = _latest_two_annual_values(facts, "NetIncomeLoss")

    # --- 毛利润 ---
    gross_profit, _ = _latest_two_annual_values(facts, "GrossProfit")

    # --- 资产 ---
    total_assets = _latest_annual_value(facts, "Assets")
    total_liab = _latest_annual_value(facts, "Liabilities")
    current_assets = _latest_annual_value(facts, "AssetsCurrent")
    current_liab = _latest_annual_value(facts, "LiabilitiesCurrent")
    equity = _latest_annual_value(facts, "StockholdersEquity")
    if equity is None:
        equity = _latest_annual_value(facts, "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")

    # --- 现金流 ---
    operating_cf = _latest_annual_value(facts, "NetCashProvidedByUsedInOperatingActivities")

    # --- 计算比率 ---
    gross_margin = _safe_div(gross_profit, revenues, 100.0)
    net_margin = _safe_div(net_income, revenues, 100.0)
    roe = _safe_div(net_income, equity, 100.0)
    roa = _safe_div(net_income, total_assets, 100.0)
    debt_ratio = _safe_div(total_liab, total_assets, 100.0)
    current_ratio = _safe_div(current_assets, current_liab)
    revenue_growth = _yoy_growth(revenues, revenues_prev)
    profit_growth = _yoy_growth(net_income, net_income_prev)

    result = {
        "gross_margin": gross_margin,
        "net_margin": net_margin,
        "roe": roe,
        "roa": roa,
        "debt_ratio": debt_ratio,
        "current_ratio": current_ratio,
        "cash_flow": operating_cf,
        "revenue_growth": revenue_growth,
        "profit_growth": profit_growth,
    }

    # 过滤掉全 None 的结果
    non_null = sum(1 for v in result.values() if v is not None)
    if non_null == 0:
        return {}

    _logger.debug(f"SEC: {ticker} (CIK={cik}) — {non_null} fields populated")
    return result
