"""
A股激进科技制造策略正式复盘跟踪器。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quant_investor.factors.report import (
    load_factor_library_shadow_status,
    render_factor_library_shadow_markdown,
)
from quant_investor.market.analyze import load_cn_stock_names
from quant_investor.market.config import get_market_settings
from quant_investor.market.download_cn import CNFullMarketDownloader
from quant_investor.llm_provider_priority import coerce_review_model_priority
from quant_investor.pipeline import QuantInvestor
from quant_investor.reporting.formal_diagnostics import (
    HoldingDecisionDiagnostic,
    apply_report_decision_guardrail,
    build_holding_decision_diagnostics,
    collect_formal_report_warnings,
    render_holding_diagnostic_markdown_table,
)
from quant_investor.research_run_config import ResolvedReviewModels


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_DIR = (
    PROJECT_ROOT / "results" / "strategy_records" / "CN" / "aggressive_tech_manufacturing"
)
DEFAULT_NOTES_PATH = DEFAULT_BASE_DIR / "latest_notes_payload.md"
DEFAULT_INITIAL_CAPITAL = 1_000_000.0
QUOTE_TIMEOUT = 20
INDEX_QUOTES = {
    "sh000001": "上证指数",
    "sz399001": "深证成指",
    "sz399006": "创业板指",
    "sh000300": "沪深300",
    "sz399905": "中证500",
    "sz399852": "中证1000",
    "sh000688": "科创50",
    "sz399673": "创业板50",
}
THEME_BASKETS = {
    "先进材料": ["688519.SH", "688295.SH"],
    "AI存储": ["688525.SH"],
    "电子制造": ["002384.SZ", "002008.SZ"],
    "电力设备": ["601179.SH", "600487.SH"],
    "光通信": ["601869.SH"],
}
CANDIDATE_POOL_PATH = PROJECT_ROOT / "results" / "cn_analysis_full" / "all_candidates.json"
CATEGORY_THEME_LABELS = {
    "hs300": "大盘核心资产",
    "zz500": "中盘制造主线",
    "zz1000": "小盘成长弹性",
}


@dataclass
class ProposedOrder:
    symbol: str
    action: str
    shares: int
    price: float
    trade_value: float
    realized_pnl: float
    reason: str


def _load_daily_config_llm_settings() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "daily_config.py"
    if not config_path.exists():
        return {
            "review_model_priority": coerce_review_model_priority([]),
            "agent_model": "",
            "agent_fallback_model": "",
            "master_model": "",
            "master_fallback_model": "",
            "master_reasoning_effort": "",
        }

    spec = importlib.util.spec_from_file_location("_daily_cfg_for_tracker", config_path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    cfg: dict[str, Any] = dict(getattr(module, "DAILY_CONFIG", {}) or {})
    resolved = ResolvedReviewModels.from_mapping(cfg)
    return resolved.to_runtime_kwargs()


def _plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        payload = value.to_dict()
        if isinstance(payload, dict):
            return dict(_jsonable(payload))
    if isinstance(value, dict):
        return dict(_jsonable(value))
    data = getattr(value, "__dict__", None)
    if isinstance(data, dict):
        return dict(_jsonable(data))
    return {}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if hasattr(value, "to_dict") and not isinstance(value, pd.DataFrame):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _serialize_reviewed_branch_verdicts(result: Any, symbol: str) -> dict[str, Any]:
    payload = dict(getattr(result, "reviewed_research_by_symbol", {}) or {}).get(symbol, {})
    if not isinstance(payload, dict):
        return {}
    return {str(name): _jsonable(verdict) for name, verdict in payload.items()}


def _serialize_symbol_review_bundle(result: Any, symbol: str) -> dict[str, Any]:
    review_bundle = getattr(result, "review_bundle", None)
    if review_bundle is None:
        return {}
    branch_overlays = dict(getattr(review_bundle, "branch_overlay_verdicts_by_symbol", {}) or {}).get(symbol, {})
    master_hints = dict(getattr(review_bundle, "master_hints_by_symbol", {}) or {})
    return {
        "branch_overlays": {str(name): _jsonable(verdict) for name, verdict in dict(branch_overlays).items()},
        "master_hint": _jsonable(master_hints.get(symbol)),
    }


def _llm_usage_summary_to_dict(summary: Any) -> dict[str, Any]:
    return {
        "call_count": int(getattr(summary, "call_count", 0) or 0),
        "success_count": int(getattr(summary, "success_count", 0) or 0),
        "fallback_count": int(getattr(summary, "fallback_count", 0) or 0),
        "failed_count": int(getattr(summary, "failed_count", 0) or 0),
        "total_tokens": int(getattr(summary, "total_tokens", 0) or 0),
        "estimated_cost_usd": round(float(getattr(summary, "estimated_cost_usd", 0.0) or 0.0), 8),
    }


def _trade_recommendation_to_dict(recommendation: Any) -> dict[str, Any]:
    if recommendation is None:
        return {}
    payload = _plain_dict(recommendation)
    if payload:
        return payload
    return {
        "symbol": str(getattr(recommendation, "symbol", "")),
        "action": str(getattr(recommendation, "action", "")),
        "weight": float(getattr(recommendation, "weight", 0.0) or 0.0),
        "confidence": float(getattr(recommendation, "confidence", 0.0) or 0.0),
        "one_line_conclusion": str(getattr(recommendation, "one_line_conclusion", "")),
        "risk_flags": list(getattr(recommendation, "risk_flags", []) or []),
        "metadata": dict(getattr(recommendation, "metadata", {}) or {}),
    }


def _run_unified_review_mainline_for_holdings(
    *,
    source_ledger: pd.DataFrame,
    latest_trade_date: str,
    source_record: str,
) -> dict[str, Any]:
    review_by_symbol: dict[str, dict[str, Any]] = {}
    degraded_symbols: dict[str, str] = {}
    llm_settings = _load_daily_config_llm_settings()
    review_models = ResolvedReviewModels.from_mapping(llm_settings)
    aggregate_attempt_usage = {
        "call_count": 0,
        "success_count": 0,
        "fallback_count": 0,
        "failed_count": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
    aggregate_effective_usage = {
        "call_count": 0,
        "success_count": 0,
        "fallback_count": 0,
        "failed_count": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
    model_role_metadata: dict[str, Any] = {}
    fallback_reasons: list[str] = []
    session_ids: dict[str, str] = {}

    for row in source_ledger.itertuples():
        symbol = str(getattr(row, "symbol", "")).strip().upper()
        if not symbol:
            continue
        current_value = _safe_float(getattr(row, "current_value", 0.0), 0.0)
        cost_basis = _safe_float(getattr(row, "cost_basis", 0.0), 0.0)
        review_capital = max(current_value, cost_basis, 10_000.0)
        investor = QuantInvestor(
            stock_pool=[symbol],
            market="CN",
            lookback_years=1.0,
            total_capital=review_capital,
            risk_level="积极",
            verbose=False,
            enable_agent_layer=True,
            universe_key="full_a",
            enable_document_semantics=True,
            **review_models.to_runtime_kwargs(),
            recall_context={
                "strategy": "aggressive_tech_manufacturing",
                "source_record": source_record,
                "latest_trade_date": latest_trade_date,
                "holding_symbol": symbol,
            },
        )
        result = investor.run()
        attempt_usage = _llm_usage_summary_to_dict(getattr(result, "llm_usage_summary", None))
        effective_usage = _llm_usage_summary_to_dict(getattr(result, "llm_effective_summary", None))
        for key in ("call_count", "success_count", "fallback_count", "failed_count", "total_tokens"):
            aggregate_attempt_usage[key] += int(attempt_usage[key])
            aggregate_effective_usage[key] += int(effective_usage[key])
        aggregate_attempt_usage["estimated_cost_usd"] = round(
            float(aggregate_attempt_usage["estimated_cost_usd"]) + float(attempt_usage["estimated_cost_usd"]),
            8,
        )
        aggregate_effective_usage["estimated_cost_usd"] = round(
            float(aggregate_effective_usage["estimated_cost_usd"]) + float(effective_usage["estimated_cost_usd"]),
            8,
        )
        role_payload = _plain_dict(getattr(result, "model_role_metadata", None))
        if role_payload and not model_role_metadata:
            model_role_metadata = role_payload
        review_bundle = getattr(result, "review_bundle", None)
        fallback_reasons.extend(list(getattr(review_bundle, "fallback_reasons", []) or []))
        degraded_reason = ""
        if attempt_usage["call_count"] <= 0:
            degraded_reason = (
                f"{symbol} review-layer 未产生有效 LLM 调用，已按降级模式继续；"
                "请核对模型配额、provider 可用性或 free-tier 限制。"
            )
            degraded_symbols[symbol] = degraded_reason
            fallback_reasons.append(degraded_reason)
        recommendations = list(getattr(getattr(result, "final_strategy", None), "recommendations", []) or [])
        if not recommendations:
            recommendations = list(getattr(getattr(result, "final_strategy", None), "trade_recommendations", []) or [])
        recommendation = next((item for item in recommendations if str(getattr(item, "symbol", "")).upper() == symbol), None)
        ic_hint = dict((getattr(result, "ic_hints_by_symbol", {}) or {}).get(symbol, {}))
        session_ids[symbol] = str(getattr(result, "llm_usage_session_id", "") or "")
        reviewed_branch_verdicts = _serialize_reviewed_branch_verdicts(result, symbol)
        symbol_review_bundle = _serialize_symbol_review_bundle(result, symbol)
        review_by_symbol[symbol] = {
            "llm_usage": attempt_usage,
            "llm_attempt_summary": attempt_usage,
            "llm_effective_summary": effective_usage,
            "llm_session_id": session_ids[symbol],
            "ic_hint": ic_hint,
            "recommendation": _trade_recommendation_to_dict(recommendation),
            "report_excerpt": str(getattr(result, "final_report", "") or "")[:2000],
            "llm_degraded": bool(degraded_reason),
            "llm_degraded_reason": degraded_reason,
            "reviewed_branch_verdicts": reviewed_branch_verdicts,
            "branch_overlays": dict(symbol_review_bundle.get("branch_overlays", {}) or {}),
            "master_hint": dict(symbol_review_bundle.get("master_hint", {}) or {}),
        }
        time.sleep(0.8)

    return {
        "reviewed_symbols": list(review_by_symbol.keys()),
        "by_symbol": review_by_symbol,
        "degraded_symbols": degraded_symbols,
        "llm_usage_summary": aggregate_attempt_usage,
        "llm_attempt_summary": aggregate_attempt_usage,
        "llm_effective_summary": aggregate_effective_usage,
        "model_role_metadata": model_role_metadata,
        "fallback_reasons": sorted(dict.fromkeys(item for item in fallback_reasons if str(item).strip())),
        "session_ids": session_ids,
    }


def _now_local() -> datetime:
    return datetime.now().astimezone()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _safe_pct(value: float, base: float) -> float:
    if abs(base) < 1e-12:
        return 0.0
    return value / base


def _map_symbol_to_quote_code(symbol: str) -> str:
    text = str(symbol).strip().upper()
    if text.startswith(("SH", "SZ")) and "." not in text:
        return text.lower()
    code, market = text.split(".")
    prefix = "sh" if market == "SH" else "sz"
    return f"{prefix}{code}"


def _decode_quote_payload(content: bytes) -> str:
    for encoding in ("gbk", "gb18030", "utf-8"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def _parse_quote_payload(line: str) -> dict[str, Any] | None:
    text = str(line).strip()
    if not text or "=" not in text or "~" not in text:
        return None

    prefix, payload = text.split("=", 1)
    quote_code = prefix.replace("v_", "").strip()
    payload = payload.strip().strip(";").strip('"')
    parts = payload.split("~")
    if len(parts) < 6:
        return None

    current = _safe_float(parts[3])
    prev_close = _safe_float(parts[4])
    change = _safe_float(parts[31], current - prev_close) if len(parts) > 31 else current - prev_close
    change_pct = (
        _safe_float(parts[32], _safe_pct(change, prev_close) * 100.0)
        if len(parts) > 32
        else _safe_pct(change, prev_close) * 100.0
    )
    return {
        "quote_code": quote_code,
        "name": parts[1].strip() or quote_code,
        "current": current,
        "prev_close": prev_close,
        "open": _safe_float(parts[5]),
        "high": _safe_float(parts[33]) if len(parts) > 33 else 0.0,
        "low": _safe_float(parts[34]) if len(parts) > 34 else 0.0,
        "time": parts[30].strip() if len(parts) > 30 else "",
        "change": change,
        "change_pct": change_pct,
    }


def _fetch_tencent_quotes(quote_codes: list[str]) -> dict[str, dict[str, Any]]:
    if not quote_codes:
        return {}

    chunks = [quote_codes[idx : idx + 60] for idx in range(0, len(quote_codes), 60)]
    result: dict[str, dict[str, Any]] = {}
    session = requests.Session()

    for chunk in chunks:
        url = "https://qt.gtimg.cn/q=" + ",".join(chunk)
        response = session.get(url, timeout=QUOTE_TIMEOUT)
        response.raise_for_status()
        payload = _decode_quote_payload(response.content)
        for raw_line in payload.split(";"):
            parsed = _parse_quote_payload(raw_line)
            if parsed is None:
                continue
            result[parsed["quote_code"]] = parsed
    return result


def _price_series(frame: pd.DataFrame) -> pd.Series:
    return frame["close"].astype(float)


def _load_history_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    frame = frame.sort_values("trade_date").reset_index(drop=True)
    return frame


def _metric_return(close: pd.Series, periods: int) -> float:
    if close.empty:
        return 0.0
    if len(close) <= periods:
        base = float(close.iloc[0])
    else:
        base = float(close.iloc[-(periods + 1)])
    current = float(close.iloc[-1])
    return _safe_pct(current - base, base)


def _derive_stage_levels(frame: pd.DataFrame, current_price: float) -> tuple[float, float]:
    if frame.empty or current_price <= 0:
        return round(current_price * 1.08, 2), round(current_price * 0.94, 2)

    recent = frame.tail(60).copy()
    high = recent["high"].astype(float) if "high" in recent.columns else recent["close"].astype(float)
    low = recent["low"].astype(float) if "low" in recent.columns else recent["close"].astype(float)
    close = recent["close"].astype(float)
    prev_close = close.shift(1).fillna(close)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = float(true_range.tail(14).mean()) if len(true_range) >= 2 else current_price * 0.02
    atr = max(atr, current_price * 0.005)

    ma20 = float(close.tail(20).mean()) if len(close) >= 20 else float(close.mean())
    low20 = float(low.tail(20).min()) if len(low) >= 5 else current_price * 0.96
    high20 = float(high.tail(20).max()) if len(high) >= 5 else current_price * 1.06

    support = min(current_price, max(low20, ma20 - 0.75 * atr))
    resistance = max(high20, current_price + 1.5 * atr)

    stop_price = max(current_price * 0.75, min(support * 0.99, current_price * 0.985))
    target_price = max(current_price * 1.05, resistance)
    return round(target_price, 2), round(stop_price, 2)


def _score_full_market_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics

    scored = metrics.copy()
    rank_weights = {
        "ret1": 0.08,
        "ret5": 0.14,
        "ret20": 0.24,
        "ret60": 0.22,
        "close_vs_ma20": 0.12,
        "ma20_vs_ma60": 0.10,
        "ma60_vs_ma120": 0.06,
        "dd20": 0.04,
    }
    for column in rank_weights:
        scored[f"{column}_pct"] = scored[column].rank(method="average", pct=True)

    scored["score_full_market"] = 0.0
    for column, weight in rank_weights.items():
        scored["score_full_market"] += scored[f"{column}_pct"] * weight

    scored["score_full_market"] = scored["score_full_market"].round(6)
    scored = scored.sort_values(
        by=["score_full_market", "ret20", "ret60", "symbol"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    scored["rank_full_market"] = range(1, len(scored) + 1)
    return scored


def _compute_category_breadth(
    category: str,
    symbols: list[str],
    data_root: Path,
    latest_trade_date: str,
    completeness_report: dict[str, Any],
) -> dict[str, Any]:
    covered = 0
    adv_1d = 0
    adv_20d = 0
    ma20_gt_ma60 = 0
    ret_1d_values: list[float] = []
    ret_20d_values: list[float] = []
    ret_60d_values: list[float] = []

    for symbol in symbols:
        path = data_root / category / f"{symbol}.csv"
        if not path.exists():
            continue
        frame = _load_history_frame(path)
        if frame.empty:
            continue
        latest_local_date = str(frame["trade_date"].iloc[-1]).replace("-", "")
        if latest_local_date != latest_trade_date:
            continue

        close = _price_series(frame).dropna().astype(float)
        if len(close) < 2:
            continue

        ret1 = _metric_return(close, 1)
        ret20 = _metric_return(close, 20)
        ret60 = _metric_return(close, 60)
        ma20 = float(close.tail(20).mean()) if len(close) >= 20 else float(close.mean())
        ma60 = float(close.tail(60).mean()) if len(close) >= 60 else float(close.mean())

        covered += 1
        adv_1d += int(ret1 > 0)
        adv_20d += int(ret20 > 0)
        ma20_gt_ma60 += int(ma20 > ma60)
        ret_1d_values.append(ret1)
        ret_20d_values.append(ret20)
        ret_60d_values.append(ret60)

    payload = completeness_report["categories"][category]
    return {
        "ret1_positive_ratio": adv_1d / covered if covered else 0.0,
        "ret20_positive_ratio": adv_20d / covered if covered else 0.0,
        "ma20_gt_ma60_ratio": ma20_gt_ma60 / covered if covered else 0.0,
        "avg_ret1": sum(ret_1d_values) / len(ret_1d_values) if ret_1d_values else 0.0,
        "avg_ret20": sum(ret_20d_values) / len(ret_20d_values) if ret_20d_values else 0.0,
        "avg_ret60": sum(ret_60d_values) / len(ret_60d_values) if ret_60d_values else 0.0,
        "latest_count": covered,
        "expected": int(payload.get("expected", len(symbols))),
        "suspended_stale_count": len(payload.get("suspended_stale_symbols", [])),
    }


def _compute_full_market_metrics(
    components: dict[str, Any],
    data_root: Path,
    latest_trade_date: str,
) -> pd.DataFrame:
    stock_names = load_cn_stock_names()
    rows: list[dict[str, Any]] = []
    for category in ("hs300", "zz500", "zz1000"):
        for symbol in components.get(category, []):
            path = data_root / category / f"{symbol}.csv"
            if not path.exists():
                continue
            frame = _load_history_frame(path)
            if frame.empty or "trade_date" not in frame.columns or "close" not in frame.columns:
                continue

            latest_local_date = str(frame["trade_date"].iloc[-1]).replace("-", "")
            if latest_local_date != latest_trade_date:
                continue

            close = _price_series(frame).dropna().astype(float)
            if len(close) < 20:
                continue

            ma20 = float(close.tail(20).mean())
            ma60 = float(close.tail(60).mean()) if len(close) >= 60 else float(close.mean())
            ma120 = float(close.tail(120).mean()) if len(close) >= 120 else ma60
            latest_close = float(close.iloc[-1])
            target_price, stop_price = _derive_stage_levels(frame, latest_close)

            rows.append(
                {
                    "symbol": symbol,
                    "name": stock_names.get(symbol, symbol),
                    "category": category,
                    "ret1": _metric_return(close, 1),
                    "ret5": _metric_return(close, 5),
                    "ret20": _metric_return(close, 20),
                    "ret60": _metric_return(close, 60),
                    "close_vs_ma20": _safe_pct(latest_close - ma20, ma20),
                    "ma20_vs_ma60": _safe_pct(ma20 - ma60, ma60),
                    "ma60_vs_ma120": _safe_pct(ma60 - ma120, ma120),
                    "dd20": _safe_pct(latest_close - float(close.tail(20).max()), float(close.tail(20).max())),
                    "latest_close": latest_close,
                    "stage_target_price": target_price,
                    "stage_stop_price": stop_price,
                }
            )

    metrics = pd.DataFrame(rows)
    return _score_full_market_metrics(metrics)


def _summarize_theme_strength(review: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for theme, symbols in THEME_BASKETS.items():
        subset = review[review["symbol"].isin(symbols)].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "theme": theme,
                "symbols": subset["symbol"].tolist(),
                "avg_today_change_pct": float(subset["today_change_pct"].mean()),
                "avg_score": float(subset["score_full_market"].mean()),
                "avg_rank": float(subset["rank_full_market"].mean()),
                "avg_ret20": float(subset["ret20"].mean()),
            }
        )
    rows.sort(
        key=lambda item: (
            item["avg_score"],
            item["avg_today_change_pct"],
            item["avg_ret20"],
            -item["avg_rank"],
            item["theme"],
        ),
        reverse=True,
    )
    return rows


def _market_style_conclusion(indices: dict[str, dict[str, Any]], breadth: dict[str, dict[str, Any]]) -> str:
    hs300 = breadth.get("hs300", {})
    zz500 = breadth.get("zz500", {})
    zz1000 = breadth.get("zz1000", {})
    kc50 = indices.get("sh000688", {})
    hs300_idx = indices.get("sh000300", {})
    cyb = indices.get("sz399006", {})

    if (
        kc50.get("change_pct", 0.0) > hs300_idx.get("change_pct", 0.0)
        and zz1000.get("ret20_positive_ratio", 0.0) >= hs300.get("ret20_positive_ratio", 0.0)
    ):
        return "成长修复偏强，科技高弹性重新获得承接。"
    if (
        hs300.get("ret20_positive_ratio", 0.0) >= zz1000.get("ret20_positive_ratio", 0.0)
        and hs300_idx.get("change_pct", 0.0) >= cyb.get("change_pct", 0.0)
    ):
        return "风格仍偏大盘稳健，成长方向只是局部修复。"
    if kc50.get("change_pct", 0.0) > 0 and cyb.get("change_pct", 0.0) < 0:
        return "市场处于结构性修复，科创强于泛成长，资金更偏向有景气验证的硬科技。"
    return "市场仍是结构性分化，未回到全面进攻。"


def _tech_mainline_conclusion(theme_strength: list[dict[str, Any]]) -> tuple[str, str]:
    if not theme_strength:
        return "暂无足够主题样本。", "暂无足够主题样本。"

    strongest = theme_strength[0]
    weakest = theme_strength[-1]
    strong_text = (
        f"{strongest['theme']} 的完整日线强度最强，平均全市场评分 {strongest['avg_score']:.3f}，"
        f"盘中涨跌 {strongest['avg_today_change_pct']:+.2f}%。"
    )
    weak_text = (
        f"{weakest['theme']} 的完整日线强度最弱，平均全市场评分 {weakest['avg_score']:.3f}，"
        f"盘中涨跌 {weakest['avg_today_change_pct']:+.2f}%。"
    )
    return strong_text, weak_text


def _position_role(row: pd.Series) -> str:
    rank = int(row["rank_full_market"])
    price = float(row["current_price"])
    stop_price = float(row["stage_stop_price"])
    if rank <= 30 and price >= stop_price:
        return "核心持有"
    if rank <= 120 and price >= stop_price:
        return "稳定核心"
    if rank <= 260 and price >= stop_price * 0.995:
        return "观察持有"
    return "降级观察"


def _position_action(row: pd.Series) -> str:
    price = float(row["current_price"])
    stop_price = float(row["stage_stop_price"])
    score = float(row["score_full_market"])
    if price < stop_price and score < 0.72:
        return "减仓待确认"
    if price < stop_price:
        return "继续观察"
    return "继续持有"


def _position_reason(row: pd.Series) -> str:
    rank = int(row["rank_full_market"])
    score = float(row["score_full_market"])
    today = float(row["today_change_pct"])
    price = float(row["current_price"])
    stop_price = float(row["stage_stop_price"])

    if price < stop_price and score < 0.72:
        return (
            f"盘中 {today:+.2f}% 且仍低于阶段止损位 {stop_price:.2f}，"
            f"完整日线强度只在全市场第 {rank} 位，需把减仓判断留在下一次确认。"
        )
    if rank <= 20:
        return (
            f"完整日线仍处第一梯队，全市场排名第 {rank} 位，"
            f"盘中 {today:+.2f}% 说明主线承接仍在。"
        )
    if price < stop_price:
        return (
            f"盘中 {today:+.2f}% ，尚未重新站回阶段止损位 {stop_price:.2f}，"
            f"但全市场评分 {score:.3f} 仍未完全失真，先观察修复延续性。"
        )
    return (
        f"完整日线评分 {score:.3f}，全市场排名第 {rank} 位，"
        f"盘中 {today:+.2f}% ，继续按主线内部分化而非失效处理。"
    )


def _build_rebalance_plan(review: pd.DataFrame) -> list[ProposedOrder]:
    weak = review[
        (review["current_price"] < review["stage_stop_price"])
        & (review["score_full_market"] < 0.72)
        & (review["today_change_pct"] <= 0.5)
    ].sort_values(["market_weight", "score_full_market"], ascending=[False, True])

    if weak.empty:
        return []

    orders: list[ProposedOrder] = []
    for row in weak.head(1).itertuples():
        lot_size = 100
        shares = int(int(row.shares_before) * 0.2 // lot_size) * lot_size
        if shares <= 0:
            continue
        trade_value = round(shares * float(row.current_price), 2)
        realized = round(shares * (float(row.current_price) - float(row.buy_price)), 2)
        orders.append(
            ProposedOrder(
                symbol=row.symbol,
                action="sell",
                shares=shares,
                price=round(float(row.current_price), 2),
                trade_value=trade_value,
                realized_pnl=realized,
                reason=(
                    f"{row.symbol} 仍低于阶段止损位 {float(row.stage_stop_price):.2f}，"
                    "且完整日线已明显落后于组合主线，执行温和减仓。"
                ),
            )
        )
    return orders


def _load_local_candidate_symbols() -> list[str]:
    if not CANDIDATE_POOL_PATH.exists():
        return []
    try:
        payload = json.loads(CANDIDATE_POOL_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    values = payload.get("full_a")
    if not isinstance(values, list):
        return []
    symbols: list[str] = []
    for item in values:
        symbol = str(item or "").strip().upper()
        if symbol:
            symbols.append(symbol)
    return symbols


def _theme_label_for_symbol(symbol: str, category: str) -> str:
    for theme, symbols in THEME_BASKETS.items():
        if symbol in symbols:
            return theme
    return CATEGORY_THEME_LABELS.get(category, "正式筛选候选")


def _evidence_quality_label(
    *,
    completeness_passed: bool,
    analysis_trade_date: str,
    strict_trade_date: str,
    source: str,
) -> str:
    if completeness_passed:
        return "高"
    if source == "latest_formal_screening" and analysis_trade_date and analysis_trade_date == strict_trade_date:
        return "中"
    return "中等偏弱"


def _build_candidate_pool(
    *,
    full_metrics: pd.DataFrame,
    held_symbols: list[str],
    completeness_passed: bool,
    analysis_trade_date: str,
    strict_trade_date: str,
) -> pd.DataFrame:
    if full_metrics.empty:
        return pd.DataFrame()

    candidate_symbols = _load_local_candidate_symbols()
    candidate_priority = {symbol: idx for idx, symbol in enumerate(candidate_symbols)}
    held_set = {str(symbol).strip().upper() for symbol in held_symbols}
    metrics = full_metrics.copy()
    metrics["symbol"] = metrics["symbol"].astype(str).str.upper()
    metrics = metrics[~metrics["symbol"].isin(held_set)].copy()
    if metrics.empty:
        return metrics

    metrics["candidate_source"] = metrics["symbol"].map(
        lambda symbol: "latest_formal_screening" if symbol in candidate_priority else "full_market_strength"
    )
    metrics["candidate_priority"] = metrics["symbol"].map(lambda symbol: candidate_priority.get(symbol, 999999))
    metrics["theme_label"] = metrics.apply(
        lambda row: _theme_label_for_symbol(str(row["symbol"]), str(row.get("category", ""))),
        axis=1,
    )
    metrics["evidence_quality"] = metrics["candidate_source"].map(
        lambda source: _evidence_quality_label(
            completeness_passed=completeness_passed,
            analysis_trade_date=analysis_trade_date,
            strict_trade_date=strict_trade_date,
            source=str(source),
        )
    )
    metrics = metrics.sort_values(
        by=["candidate_priority", "score_full_market", "ret20", "ret60", "symbol"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)
    metrics["candidate_rank"] = range(1, len(metrics) + 1)
    return metrics.head(12).reset_index(drop=True)


def _build_switch_plan(
    *,
    holdings_review: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    completeness_passed: bool,
) -> pd.DataFrame:
    if holdings_review.empty or candidate_pool.empty:
        return pd.DataFrame()

    weak_holdings = holdings_review.sort_values(
        by=["rank_full_market", "score_full_market", "today_change_pct"],
        ascending=[False, True, True],
    ).head(3)
    best_candidates = candidate_pool.sort_values(
        by=["candidate_priority", "score_full_market", "ret20", "symbol"],
        ascending=[True, False, False, True],
    ).head(3)

    rows: list[dict[str, Any]] = []
    for weak_row, candidate_row in zip(weak_holdings.itertuples(), best_candidates.itertuples()):
        score_gap = round(float(candidate_row.score_full_market) - float(weak_row.score_full_market), 6)
        ret20_gap = round(float(candidate_row.ret20) - float(weak_row.ret20), 6)
        superior = (
            score_gap >= 0.12
            and int(candidate_row.rank_full_market) + 120 <= int(weak_row.rank_full_market)
            and ret20_gap >= 0.08
        )
        actionable = superior and completeness_passed
        rows.append(
            {
                "sell_symbol": weak_row.symbol,
                "sell_name": weak_row.name,
                "sell_role": weak_row.position_role,
                "sell_rank_full_market": int(weak_row.rank_full_market),
                "sell_score_full_market": round(float(weak_row.score_full_market), 6),
                "buy_symbol": candidate_row.symbol,
                "buy_name": candidate_row.name,
                "buy_theme": candidate_row.theme_label,
                "buy_rank_full_market": int(candidate_row.rank_full_market),
                "buy_score_full_market": round(float(candidate_row.score_full_market), 6),
                "score_gap": score_gap,
                "ret20_gap": ret20_gap,
                "candidate_source": candidate_row.candidate_source,
                "evidence_quality": candidate_row.evidence_quality,
                "priority": "high" if superior else "watch",
                "action": "switch_now" if actionable else ("prepare_switch" if superior else "watch_only"),
                "switch_ratio_hint": "20%" if actionable else "先观察，不执行",
                "trigger_threshold": (
                    "候选继续留在本地强度前120，且现持仓未收复阶段止损位"
                    if not actionable
                    else "按20%试探性换仓，若候选继续强于卖出对象再递增"
                ),
                "no_switch_condition": (
                    "当前本地快照未完成 strict 完整性，先不把结构优势直接转成实单"
                    if superior and not completeness_passed
                    else "若现持仓重新回到前250名且候选优势收敛，则继续持有原仓"
                ),
            }
        )
    return pd.DataFrame(rows)


def _apply_orders(
    source_ledger: pd.DataFrame,
    orders: list[ProposedOrder],
    cash_before: float,
    quote_prices: dict[str, float],
) -> tuple[pd.DataFrame, float, float]:
    ledger = source_ledger.copy()
    ledger = ledger.drop(columns=[column for column in ["market_weight"] if column in ledger.columns])
    ledger["shares"] = ledger["shares"].astype(int)
    cash_after = round(cash_before, 2)
    realized_total = 0.0
    order_map = {order.symbol: order for order in orders}

    updated_rows: list[dict[str, Any]] = []
    for row in ledger.itertuples():
        shares = int(row.shares)
        avg_cost = float(row.avg_cost)
        order = order_map.get(row.symbol)
        if order and order.action == "sell":
            shares = max(0, shares - order.shares)
            cash_after = round(cash_after + order.trade_value, 2)
            realized_total += order.realized_pnl

        price = float(quote_prices.get(row.symbol, getattr(row, "current_price", 0.0)))
        cost_basis = round(shares * avg_cost, 2)
        current_value = round(shares * price, 2)
        unrealized = round(current_value - cost_basis, 2)
        updated_rows.append(
            {
                "symbol": row.symbol,
                "name": row.name,
                "shares": shares,
                "avg_cost": round(avg_cost, 6),
                "cost_basis": cost_basis,
                "current_price": round(price, 2),
                "current_value": current_value,
                "unrealized_pnl": unrealized,
                "unrealized_pnl_pct": round(_safe_pct(unrealized, cost_basis), 6),
            }
        )

    updated = pd.DataFrame(updated_rows)
    invested = float(updated["current_value"].sum()) if not updated.empty else 0.0
    if invested > 0:
        updated["market_weight"] = (updated["current_value"] / invested).round(6)
    else:
        updated["market_weight"] = 0.0
    return updated, cash_after, round(realized_total, 2)


def _load_previous_record(
    base_dir: Path,
    source_record: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    run_dirs = [path for path in base_dir.iterdir() if path.is_dir()]
    if not run_dirs:
        raise RuntimeError("策略目录下不存在上一条正式记录，无法做连续复盘。")

    if source_record:
        latest_dir = base_dir / source_record
        if not latest_dir.exists():
            raise RuntimeError(f"指定的 source_record 不存在: {source_record}")
    else:
        latest_dir = sorted(run_dirs, key=lambda path: path.name)[-1]
    manifest_path = latest_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ledger = pd.read_csv(latest_dir / "ledger.csv", encoding="utf-8-sig")
    pnl_summary = pd.read_csv(latest_dir / "pnl_summary.csv", encoding="utf-8-sig")
    return ledger, manifest, pnl_summary


def _allocate_run_timestamp(base_dir: Path, now: datetime) -> str:
    base = now.strftime("%Y%m%d_%H%M")
    candidate = base
    counter = 1
    while (base_dir / candidate).exists():
        candidate = f"{base}_{counter:02d}"
        counter += 1
    return candidate


def _format_symbol_set(symbols: list[str]) -> str:
    return " / ".join(symbols)


def _format_top_holdings_by_unrealized_pnl(frame: pd.DataFrame, *, positive: bool) -> str:
    if frame.empty or "unrealized_pnl" not in frame.columns:
        return "无"

    filtered = frame[frame["unrealized_pnl"] > 0] if positive else frame[frame["unrealized_pnl"] < 0]
    if filtered.empty:
        return "无"

    ordered = filtered.sort_values("unrealized_pnl", ascending=not positive)
    return "；".join(
        f"{row.symbol}({row.name}) {row.unrealized_pnl:,.2f} 元"
        for row in ordered.head(3).itertuples()
    )


def _format_top_delta_vs_source_record(frame: pd.DataFrame, *, positive: bool) -> str:
    if frame.empty or "delta_vs_source_record" not in frame.columns:
        return "无"

    filtered = (
        frame[frame["delta_vs_source_record"] > 0]
        if positive
        else frame[frame["delta_vs_source_record"] < 0]
    )
    if filtered.empty:
        return "无"

    ordered = filtered.sort_values("delta_vs_source_record", ascending=not positive)
    return "；".join(
        f"{row.symbol} {row.delta_vs_source_record:,.2f} 元"
        for row in ordered.head(3).itertuples()
    )


def _format_signed_money(value: float) -> str:
    return f"{value:+,.2f} 元"


def _format_holding_advice_line(row: Any) -> str:
    price = _safe_float(getattr(row, "current_price", 0.0), 0.0)
    stop_price = _safe_float(getattr(row, "stage_stop_price", 0.0), 0.0)
    target_price = _safe_float(getattr(row, "stage_target_price", 0.0), 0.0)
    stop_buffer = _safe_pct(price - stop_price, stop_price)
    hard_signal = "未触发阶段止损" if price >= stop_price else "低于阶段止损，需跟踪减仓确认"
    return (
        f"- `{row.symbol}`（{row.name}）：建议 `{row.recommended_action}`，"
        f"持仓角色 `{row.position_role}`；当前价 `{price:.2f}`，"
        f"阶段止损 `{stop_price:.2f}`（缓冲 {stop_buffer:+.2%}），"
        f"阶段目标 `{target_price:.2f}`；浮动盈亏 `{_format_signed_money(float(row.unrealized_pnl))}`，"
        f"较上一条记录 `{_format_signed_money(float(row.delta_vs_source_record))}`；"
        f"全市场强度排名 `{int(row.rank_full_market)}`，今日涨跌 `{float(row.today_change_pct):+.2f}%`；"
        f"{hard_signal}。"
    )


def _format_candidate_advice_line(row: Any, switch_row: dict[str, Any] | None) -> str:
    source_label = "最新正式筛选结果" if str(row.candidate_source) == "latest_formal_screening" else "本地全市场强度"
    relative_advantage = (
        f"相对 `{switch_row['sell_symbol']}` 更优，强度排名前移 `{int(switch_row['sell_rank_full_market']) - int(switch_row['buy_rank_full_market'])}` 位"
        f"，20日动量高出 `{float(switch_row['ret20_gap']):+.2%}`；"
        if switch_row
        else f"当前在本地候选中位列前 `{int(row.candidate_rank)}`，20日收益 `{float(row.ret20):+.2%}`；"
    )
    major_risk = (
        "本地 strict 快照仍有缺口，结论依赖主导本地快照延续性；"
        if str(row.evidence_quality) != "高"
        else "若回撤跌破阶段止损位，短线强度可能失真；"
    )
    trigger = (
        str(switch_row["trigger_threshold"])
        if switch_row
        else "若连续两次正式复盘仍在候选前列，可升级为优先观察对象"
    )
    return (
        f"- `{row.symbol}`（{row.name}）：主线 `{row.theme_label}`，来源 `{source_label}`；"
        f"{relative_advantage}主要风险：{major_risk}"
        f"触发条件：{trigger}；证据质量：`{row.evidence_quality}`。"
    )


def _format_warning_count_summary(warnings: list[Any]) -> str:
    if not warnings:
        return "无"
    counts = Counter(str(warning.code) for warning in warnings)
    return "，".join(f"{code}={count}" for code, count in sorted(counts.items()))


def _format_warning_messages(warnings: list[Any], *, severity: str | None = None, limit: int = 6) -> str:
    selected = [
        warning
        for warning in warnings
        if severity is None or str(getattr(warning, "severity", "")) == severity
    ]
    if not selected:
        return "无"
    messages = [str(warning.human_message).strip() for warning in selected if str(warning.human_message).strip()]
    return "；".join(messages[:limit]) + (" ..." if len(messages) > limit else "")


def _format_trade_date(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text or "未知"


def _dominant_trade_date(date_counts: dict[str, Any]) -> str:
    normalized = {
        str(date).strip(): int(count or 0)
        for date, count in (date_counts or {}).items()
        if str(date).strip()
    }
    if not normalized:
        return ""
    return max(normalized.items(), key=lambda item: (item[1], item[0]))[0]


def _resolve_analysis_trade_date(completeness_report: dict[str, Any]) -> str:
    categories = dict(completeness_report.get("categories", {}) or {})
    full_a = dict(categories.get("full_a", {}) or {})
    dominant = _dominant_trade_date(full_a.get("date_counts", {}) or {})
    if dominant:
        return dominant

    aggregate_counts: dict[str, int] = {}
    for payload in categories.values():
        for date, count in (payload.get("date_counts", {}) or {}).items():
            key = str(date).strip()
            if not key:
                continue
            aggregate_counts[key] = aggregate_counts.get(key, 0) + int(count or 0)
    dominant = _dominant_trade_date(aggregate_counts)
    if dominant:
        return dominant
    return str(
        completeness_report.get("effective_target_trade_date")
        or completeness_report.get("latest_trade_date")
        or ""
    )


def _build_data_status_summary(
    completeness_report: dict[str, Any],
    analysis_trade_date: str,
) -> str:
    target_trade_date = _format_trade_date(completeness_report.get("latest_trade_date"))
    effective_target_trade_date = _format_trade_date(
        completeness_report.get("effective_target_trade_date")
        or completeness_report.get("latest_trade_date")
    )
    strict_trade_date = _format_trade_date(
        completeness_report.get("strict_trade_date")
        or completeness_report.get("latest_trade_date")
    )
    freshness_mode = str(completeness_report.get("freshness_mode") or "strict")
    blocking_count = int(completeness_report.get("blocking_incomplete_count", 0) or 0)
    coverage_ratio = float(completeness_report.get("coverage_ratio", 0.0) or 0.0)
    pre_listing_count = len(completeness_report.get("pre_listing_symbols", []) or [])
    full_a = (
        completeness_report.get("categories", {})
        .get("full_a", {})
        or {}
    )
    dominant_count = int((full_a.get("date_counts", {}) or {}).get(analysis_trade_date, 0) or 0)
    dominant_expected = int(full_a.get("expected", 0) or 0)
    suspended_stale_count = len(
        (
            completeness_report.get("categories", {})
            .get("full_a", {})
            .get("suspended_stale_symbols", [])
            or []
        )
    )

    base = f"本地主导A股日线快照位于 `{_format_trade_date(analysis_trade_date)}`"
    if _format_trade_date(analysis_trade_date) != effective_target_trade_date:
        base += (
            f"，而当前完整性目标交易日为 `{effective_target_trade_date}`"
            f"（strict 最新 `{strict_trade_date}` / 报告 latest `{target_trade_date}`）"
        )
    else:
        base += f"，当前完整性目标交易日同样为 `{effective_target_trade_date}`"
    extras = [
        f"主导快照覆盖 `{dominant_count}/{dominant_expected}`",
        f"覆盖率 `{coverage_ratio:.1%}`",
        f"停牌/长期停牌例外 `{suspended_stale_count}` 个",
        f"预上市样本 `{pre_listing_count}` 个",
        f"完整性模式 `{freshness_mode}`",
    ]
    if blocking_count > 0:
        extras.insert(0, f"阻塞缺口 `{blocking_count}` 个")
    else:
        extras.insert(0, "阻塞缺口 `0` 个")
    return base + "（" + "；".join(extras) + "）"


def _build_data_snapshot_lines(
    completeness_report: dict[str, Any],
    quote_snapshot: str,
    analysis_trade_date: str,
) -> list[str]:
    categories = dict(completeness_report.get("categories", {}) or {})
    full_a = dict(categories.get("full_a", {}) or {})
    analysis_count = int((full_a.get("date_counts", {}) or {}).get(analysis_trade_date, 0) or 0)
    lines = [
        (
            f"- 分析采用的主导本地快照：`{_format_trade_date(analysis_trade_date)}`，"
            f"`full_a` 覆盖 `{analysis_count}/{int(full_a.get('expected', 0) or 0)}`"
        ),
        f"- 本地最新交易日：`{_format_trade_date(completeness_report.get('latest_trade_date'))}`",
        f"- strict 最新交易日：`{_format_trade_date(completeness_report.get('strict_trade_date'))}`",
        f"- stable 最新交易日：`{_format_trade_date(completeness_report.get('stable_trade_date'))}`",
        f"- 当前采用目标交易日：`{_format_trade_date(completeness_report.get('effective_target_trade_date'))}`",
        f"- 新鲜度模式：`{completeness_report.get('freshness_mode') or 'strict'}`",
        (
            f"- 覆盖完成度：`{int(completeness_report.get('coverage_complete_count', 0) or 0)}/"
            f"{int(completeness_report.get('expected_scope_count', 0) or 0)}` "
            f"（{float(completeness_report.get('coverage_ratio', 0.0) or 0.0):.1%}）"
        ),
        (
            f"- 完整性状态：`{'通过' if completeness_report.get('complete') else '未通过'}`，"
            f"阻塞缺口 `{int(completeness_report.get('blocking_incomplete_count', 0) or 0)}` 个"
        ),
        (
            f"- 指数/持仓盘中快照：`{quote_snapshot}`；指数与持仓现价使用盘中行情，"
            "广度、主题与全市场强弱仍以本地最新日线快照为准。"
            if quote_snapshot
            else "- 指数/持仓盘中快照：`N/A`，本轮全部结论基于本地日线快照。"
        ),
    ]

    pre_listing = list(completeness_report.get("pre_listing_symbols", []) or [])
    if pre_listing:
        lines.append(
            "- 预上市样本：`"
            + "，".join(
                f"{item.get('symbol')}@{_format_trade_date(item.get('list_date'))}"
                for item in pre_listing[:8]
            )
            + (" ...`" if len(pre_listing) > 8 else "`")
        )

    for category in completeness_report.get("categories_checked", []):
        payload = categories.get(category, {})
        if not payload:
            continue
        category_dominant_trade_date = _dominant_trade_date(payload.get("date_counts", {}) or {})
        category_dominant_count = int((payload.get("date_counts", {}) or {}).get(category_dominant_trade_date, 0) or 0)
        lines.append(
            f"- {category}：目标 `{_format_trade_date(payload.get('latest_trade_date'))}`，"
            f"主导本地快照 `{_format_trade_date(category_dominant_trade_date)}` "
            f"（`{category_dominant_count}/{int(payload.get('expected', 0) or 0)}`），"
            f"目标覆盖 `{int(payload.get('coverage_complete_count', 0) or 0)}/"
            f"{int(payload.get('expected', 0) or 0)}`，"
            f"阻塞缺口 `{int(payload.get('blocking_incomplete_count', 0) or 0)}`，"
            f"停牌例外 `{len(payload.get('suspended_stale_symbols', []) or [])}`"
        )
        blocking_missing = list(payload.get("blocking_missing_symbols", []) or [])
        if blocking_missing:
            lines.append(
                "- "
                + category
                + " 阻塞缺失：`"
                + "，".join(blocking_missing[:10])
                + (" ...`" if len(blocking_missing) > 10 else "`")
            )
        blocking_stale = list(payload.get("blocking_stale_symbols", []) or [])
        if blocking_stale:
            lines.append(
                "- "
                + category
                + " 阻塞旧档：`"
                + "，".join(
                    f"{item.get('symbol')}@{_format_trade_date(item.get('latest_local_date'))}"
                    for item in blocking_stale[:10]
                )
                + (" ...`" if len(blocking_stale) > 10 else "`")
            )
        suspended = list(payload.get("suspended_stale_symbols", []) or [])
        if suspended:
            lines.append(
                "- "
                + category
                + " 停牌/长期停牌例外：`"
                + "，".join(
                    f"{item.get('symbol')}@{_format_trade_date(item.get('latest_local_date'))}"
                    for item in suspended[:10]
                )
                + (" ...`" if len(suspended) > 10 else "`")
            )
    return lines


def _write_outputs(
    base_dir: Path,
    run_dir: Path,
    report_text: str,
    holdings_review: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    switch_plan_df: pd.DataFrame,
    ledger: pd.DataFrame,
    orders_df: pd.DataFrame,
    pnl_summary_df: pd.DataFrame,
    manifest: dict[str, Any],
    market_snapshot: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = run_dir / "raw_exports"
    raw_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "analysis_report.md"
    holdings_path = run_dir / "holdings_review.csv"
    candidate_path = run_dir / "candidate_pool.csv"
    switch_path = run_dir / "switch_plan.csv"
    ledger_path = run_dir / "ledger.csv"
    orders_path = run_dir / "orders.csv"
    pnl_path = run_dir / "pnl_summary.csv"
    snapshot_path = run_dir / "market_snapshot.json"
    manifest_path = run_dir / "manifest.json"

    report_path.write_text(report_text, encoding="utf-8")
    holdings_review.to_csv(holdings_path, index=False, encoding="utf-8-sig")
    candidate_pool.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    switch_plan_df.to_csv(switch_path, index=False, encoding="utf-8-sig")
    ledger.to_csv(ledger_path, index=False, encoding="utf-8-sig")
    orders_df.to_csv(orders_path, index=False, encoding="utf-8-sig")
    pnl_summary_df.to_csv(pnl_path, index=False, encoding="utf-8-sig")
    snapshot_path.write_text(json.dumps(market_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    prefix = f"aggressive_portfolio_{manifest['timestamp']}_formal"
    shutil.copy2(report_path, raw_dir / f"{prefix}_report.md")
    shutil.copy2(holdings_path, raw_dir / f"{prefix}_holdings_review.csv")
    shutil.copy2(candidate_path, raw_dir / f"{prefix}_candidate_pool.csv")
    shutil.copy2(switch_path, raw_dir / f"{prefix}_switch_plan.csv")
    shutil.copy2(ledger_path, raw_dir / f"{prefix}_ledger.csv")
    shutil.copy2(orders_path, raw_dir / f"{prefix}_orders.csv")
    shutil.copy2(pnl_path, raw_dir / f"{prefix}_pnl_summary.csv")


def _build_notes_payload(
    trade_date: str,
    data_status: str,
    market_core_view: str,
    pnl_summary: dict[str, Any],
    orders: list[ProposedOrder],
    switch_plan_df: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    tomorrow_focus: list[str],
) -> str:
    if orders:
        order_text = "；".join(
            f"{order.symbol} {order.action} {order.shares}股 @ {order.price:.2f}" for order in orders
        )
    else:
        order_text = "无，本日维持现有结构。"

    if switch_plan_df.empty:
        switch_text = "否，暂无明显优于现持仓的可执行换仓对象。"
        switch_detail_text = "维持原组合，继续跟踪候选观察池。"
    else:
        top_switch = switch_plan_df.iloc[0]
        action = str(top_switch["action"])
        if action == "switch_now":
            switch_text = "是，已形成明确换仓优先级。"
        elif action == "prepare_switch":
            switch_text = "否，但已形成预备换仓对象。"
        else:
            switch_text = "否，先保留观察池。"
        switch_detail_text = (
            f"{top_switch['sell_symbol']} -> {top_switch['buy_symbol']}，优先级 `{top_switch['priority']}`，"
            f"触发条件：{top_switch['trigger_threshold']}"
        )

    if candidate_pool.empty:
        candidate_text = "暂无可用备选。"
    else:
        top_rows = candidate_pool.head(3)
        candidate_text = "；".join(
            f"{row.symbol}({row.name})/{row.theme_label}/证据{row.evidence_quality}"
            for row in top_rows.itertuples()
        )

    return "\n".join(
        [
            f"# A股日度复盘 {trade_date}",
            "",
            f"- 数据完整性：{data_status}",
            f"- 市场核心判断：{market_core_view}",
            (
                f"- 组合盈亏：截至 `{pnl_summary['quote_snapshot']}`，总资产 "
                f"`{pnl_summary['total_value_after']:,.2f} 元`，较初始资金 "
                f"`{pnl_summary['portfolio_pnl_after']:,.2f} 元` "
                f"（{pnl_summary['portfolio_pnl_pct_after']:.2%}），较上一条正式记录变动 "
                f"`{pnl_summary['delta_vs_source_record']:,.2f} 元`。"
            ),
            f"- 备选投资建议：{candidate_text}",
            f"- 是否调仓：{'是' if orders else '否'}",
            f"- 调仓内容：{order_text}",
            f"- 是否换仓：{switch_text}",
            f"- 换仓内容：{switch_detail_text}",
            f"- 明日观察重点：{'；'.join(tomorrow_focus)}",
        ]
    )


def run_tracker(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    now = _now_local()
    timestamp = _allocate_run_timestamp(Path(args.base_dir), now)
    timestamp_long = now.strftime("%Y-%m-%d %H:%M:%S %Z")
    base_dir = Path(args.base_dir)
    run_dir = base_dir / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)

    source_ledger, source_manifest, source_pnl = _load_previous_record(
        base_dir,
        source_record=args.source_record,
    )
    source_record = str(source_manifest.get("timestamp") or source_ledger.iloc[0].get("source_record", "unknown"))
    cash_before = _safe_float(source_pnl["cash_after"].iloc[-1] if "cash_after" in source_pnl.columns else 0.0)
    initial_capital = _safe_float(source_manifest.get("capital_cny"), DEFAULT_INITIAL_CAPITAL)
    source_total_value = _safe_float(
        source_pnl["total_value_after"].iloc[-1] if "total_value_after" in source_pnl.columns else initial_capital
    )

    cn_data_root = Path(get_market_settings('CN').data_dir)
    downloader = CNFullMarketDownloader(
        data_dir=str(cn_data_root),
        years=args.years,
        max_workers=4,
    )
    components = downloader.load_components()
    completeness_before = downloader.build_completeness_report(
        components=components,
        allowed_stale_symbols=args.allowed_stale_symbols,
    )
    attempted_backfill = False
    download_report_path = None
    completeness_after = completeness_before

    latest_trade_date = str(completeness_after.get("latest_trade_date") or "")
    analysis_trade_date = _resolve_analysis_trade_date(completeness_after)
    completeness_passed = bool(completeness_after["complete"])

    review_layer = _run_unified_review_mainline_for_holdings(
        source_ledger=source_ledger,
        latest_trade_date=analysis_trade_date,
        source_record=source_record,
    )
    review_by_symbol = dict(review_layer.get("by_symbol", {}) or {})
    degraded_symbols = dict(review_layer.get("degraded_symbols", {}) or {})
    review_attempt_summary = dict(review_layer.get("llm_attempt_summary", review_layer.get("llm_usage_summary", {})) or {})
    review_effective_summary = dict(review_layer.get("llm_effective_summary", {}) or {})
    review_model_role_metadata = dict(review_layer.get("model_role_metadata", {}) or {})

    full_metrics = _compute_full_market_metrics(
        components=components,
        data_root=cn_data_root,
        latest_trade_date=analysis_trade_date,
    )
    metrics_map = {
        row.symbol: row._asdict()
        for row in full_metrics.itertuples(index=False)
    }

    holding_quote_codes = [_map_symbol_to_quote_code(symbol) for symbol in source_ledger["symbol"]]
    index_quote_codes = list(INDEX_QUOTES.keys())
    quote_error = ""
    try:
        quote_payload = _fetch_tencent_quotes(index_quote_codes + holding_quote_codes)
    except Exception as exc:
        quote_payload = {}
        quote_error = str(exc)
    quote_snapshot = max(
        [quote_payload.get(code, {}).get("time", "") for code in index_quote_codes + holding_quote_codes],
        default="",
    )

    indices = {
        code: {
            **quote_payload[code],
            "name": INDEX_QUOTES[code],
        }
        for code in index_quote_codes
        if code in quote_payload
    }

    breadth = {
        category: _compute_category_breadth(
            category=category,
            symbols=components.get(category, []),
            data_root=cn_data_root,
            latest_trade_date=analysis_trade_date,
            completeness_report=completeness_after,
        )
        for category in ("hs300", "zz500", "zz1000")
    }

    current_rows: list[dict[str, Any]] = []
    previous_value_map = {
        row.symbol: float(getattr(row, "current_value", 0.0))
        for row in source_ledger.itertuples()
    }
    for row in source_ledger.itertuples():
        symbol = row.symbol
        metric = metrics_map.get(symbol, {})
        quote = quote_payload.get(_map_symbol_to_quote_code(symbol), {})
        review_payload = review_by_symbol.get(symbol, {})
        recommendation = dict(review_payload.get("recommendation", {}) or {})
        ic_hint = dict(review_payload.get("ic_hint", {}) or {})
        master_hint = dict(review_payload.get("master_hint", {}) or {})
        llm_attempt_summary = dict(review_payload.get("llm_attempt_summary", {}) or {})
        llm_effective_summary = dict(review_payload.get("llm_effective_summary", {}) or {})
        llm_action = str(recommendation.get("action") or ic_hint.get("action") or "hold")
        llm_confidence_source = ""
        llm_confidence_value: float | None = None
        if "confidence" in recommendation and recommendation.get("confidence") is not None:
            llm_confidence_value = _safe_float(recommendation.get("confidence"))
            llm_confidence_source = "recommendation.confidence"
        elif "confidence_hint" in ic_hint and ic_hint.get("confidence_hint") is not None:
            llm_confidence_value = _safe_float(ic_hint.get("confidence_hint"))
            llm_confidence_source = "ic_hint.confidence_hint"
        elif "confidence_hint" in master_hint and master_hint.get("confidence_hint") is not None:
            llm_confidence_value = _safe_float(master_hint.get("confidence_hint"))
            llm_confidence_source = "master_hint.confidence_hint"
        llm_confidence = float(llm_confidence_value or 0.0)
        llm_conclusion = str(
            recommendation.get("one_line_conclusion")
            or ic_hint.get("thesis")
            or ""
        ).strip()
        llm_risk_flags = list(recommendation.get("risk_flags") or ic_hint.get("risk_flags") or [])
        llm_session_id = str(review_payload.get("llm_session_id", "") or "")
        current_price = _safe_float(quote.get("current"), _safe_float(metric.get("latest_close"), getattr(row, "current_price", 0.0)))
        current_value = round(int(row.shares) * current_price, 2)
        buy_value = round(float(row.cost_basis), 2)
        unrealized = round(current_value - buy_value, 2)
        today_change_pct = round(_safe_float(quote.get("change_pct")), 2)
        staged_target = round(_safe_float(metric.get("stage_target_price"), getattr(row, "stage_target_price", current_price * 1.1)), 2)
        staged_stop = round(_safe_float(metric.get("stage_stop_price"), getattr(row, "stage_stop_price", current_price * 0.94)), 2)
        current_rows.append(
            {
                "symbol": symbol,
                "name": row.name,
                "category": metric.get("category", ""),
                "shares_before": int(row.shares),
                "buy_price": round(float(row.avg_cost), 6),
                "buy_value": buy_value,
                "current_price": round(current_price, 2),
                "current_value": current_value,
                "unrealized_pnl": unrealized,
                "unrealized_pnl_pct": round(_safe_pct(unrealized, buy_value), 6),
                "today_change_pct": today_change_pct,
                "ret5": round(_safe_float(metric.get("ret5")), 6),
                "ret20": round(_safe_float(metric.get("ret20")), 6),
                "ret60": round(_safe_float(metric.get("ret60")), 6),
                "close_vs_ma20": round(_safe_float(metric.get("close_vs_ma20")), 6),
                "ma20_vs_ma60": round(_safe_float(metric.get("ma20_vs_ma60")), 6),
                "ma60_vs_ma120": round(_safe_float(metric.get("ma60_vs_ma120")), 6),
                "dd20": round(_safe_float(metric.get("dd20")), 6),
                "rank_full_market": int(metric["rank_full_market"]) if metric and "rank_full_market" in metric else 9999,
                "score_full_market": round(_safe_float(metric.get("score_full_market")), 6),
                "stage_target_price": staged_target,
                "stage_stop_price": staged_stop,
                "delta_vs_source_record": round(current_value - previous_value_map.get(symbol, 0.0), 2),
                "llm_action": llm_action,
                "llm_confidence": round(llm_confidence, 6),
                "llm_conclusion": llm_conclusion,
                "llm_risk_flags": "；".join(str(item).strip() for item in llm_risk_flags if str(item).strip()),
                "llm_session_id": llm_session_id,
                "llm_attempt_calls": int(llm_attempt_summary.get("call_count", 0) or 0),
                "llm_failed_calls": int(llm_attempt_summary.get("failed_count", 0) or 0),
                "llm_effective_calls": int(llm_effective_summary.get("call_count", 0) or 0),
                "llm_confidence_source": llm_confidence_source,
                "llm_degraded": bool(review_payload.get("llm_degraded", False)),
            }
        )

    holdings_review = pd.DataFrame(current_rows)
    total_market_value_before = round(float(holdings_review["current_value"].sum()), 2)
    total_value_before = round(total_market_value_before + cash_before, 2)
    if total_market_value_before > 0:
        holdings_review["market_weight"] = (
            holdings_review["current_value"] / total_market_value_before
        ).round(6)
    else:
        holdings_review["market_weight"] = 0.0
    holdings_review["position_role"] = holdings_review.apply(_position_role, axis=1)
    holdings_review["recommended_action"] = holdings_review.apply(_position_action, axis=1)
    holdings_review["reason"] = holdings_review.apply(_position_reason, axis=1)
    holdings_review = holdings_review.sort_values(
        by=["score_full_market", "today_change_pct", "symbol"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    candidate_pool = _build_candidate_pool(
        full_metrics=full_metrics,
        held_symbols=holdings_review["symbol"].tolist(),
        completeness_passed=completeness_passed,
        analysis_trade_date=analysis_trade_date,
        strict_trade_date=str(completeness_after.get("strict_trade_date") or ""),
    )
    switch_plan_df = _build_switch_plan(
        holdings_review=holdings_review,
        candidate_pool=candidate_pool,
        completeness_passed=completeness_passed,
    )

    theme_strength = _summarize_theme_strength(holdings_review)
    style_text = _market_style_conclusion(indices=indices, breadth=breadth)
    strongest_theme_text, weakest_theme_text = _tech_mainline_conclusion(theme_strength)
    orders = _build_rebalance_plan(holdings_review)
    # 当前策略仍以主线内部修复为主，只在出现明确弱化共振时执行减仓。
    if len(orders) == 1 and theme_strength and theme_strength[0]["avg_score"] >= 0.8:
        orders = []

    order_rows = []
    for order in orders:
        name = source_ledger[source_ledger["symbol"] == order.symbol]["name"].iloc[0]
        order_rows.append(
            {
                "timestamp": timestamp_long,
                "action": order.action,
                "symbol": order.symbol,
                "name": name,
                "shares": order.shares,
                "price": order.price,
                "trade_value": order.trade_value,
                "realized_pnl": order.realized_pnl,
                "reason": order.reason,
            }
        )
    orders_df = pd.DataFrame(
        order_rows,
        columns=["timestamp", "action", "symbol", "name", "shares", "price", "trade_value", "realized_pnl", "reason"],
    )

    quote_prices = {row["symbol"]: float(row["current_price"]) for row in current_rows}
    updated_ledger, cash_after, realized_pnl = _apply_orders(
        source_ledger=source_ledger,
        orders=orders,
        cash_before=cash_before,
        quote_prices=quote_prices,
    )
    ledger_meta = holdings_review[
        ["symbol", "stage_target_price", "stage_stop_price", "position_role"]
    ].rename(columns={"position_role": "thesis_status"})
    updated_ledger = updated_ledger.merge(ledger_meta, on="symbol", how="left")

    total_market_value_after = round(float(updated_ledger["current_value"].sum()), 2)
    total_value_after = round(total_market_value_after + cash_after, 2)
    portfolio_pnl_after = round(total_value_after - initial_capital, 2)
    portfolio_pnl_before = round(total_value_before - initial_capital, 2)

    float_winners = holdings_review.sort_values("unrealized_pnl", ascending=False)
    float_losers = holdings_review.sort_values("unrealized_pnl", ascending=True)

    rebalance_reason = (
        "执行温和调仓，削减已明显落后于主线的弱支线仓位。"
        if orders
        else "今天不执行调仓，继续把强修复与弱滞涨的分化再观察一个交易日。"
    )
    data_status = _build_data_status_summary(completeness_after, analysis_trade_date=analysis_trade_date)
    tomorrow_focus = [
        "确认先进材料与光通信的修复能否延续，不让单日反弹误判为全面重启",
        "继续观察 `大族激光 / 中国西电` 是否能重新站回阶段止损位",
        "跟踪科创50 相对沪深300 的强弱差，判断资金是否继续偏向硬科技",
    ]
    if not completeness_passed:
        tomorrow_focus.insert(
            0,
            (
                f"核对本地A股快照阻塞缺口 `{int(completeness_after.get('blocking_incomplete_count', 0) or 0)}` 个"
                "是否属于真实缺数还是停牌/预上市扰动"
            ),
        )
    if (
        str(completeness_after.get("strict_trade_date") or "") 
        and completeness_after.get("strict_trade_date") != completeness_after.get("effective_target_trade_date")
    ):
        tomorrow_focus.insert(
            0,
            (
                f"关注 strict 交易日 `{_format_trade_date(completeness_after.get('strict_trade_date'))}`"
                " 是否在明日变成必须补齐的新主导快照"
            ),
        )
    data_snapshot_lines = _build_data_snapshot_lines(
        completeness_report=completeness_after,
        quote_snapshot=quote_snapshot,
        analysis_trade_date=analysis_trade_date,
    )
    branch_signals_by_symbol = {
        symbol: {
            "reviewed_branch_verdicts": dict(payload.get("reviewed_branch_verdicts", {}) or {}),
            "branch_overlays": dict(payload.get("branch_overlays", {}) or {}),
            "master_hint": dict(payload.get("master_hint", {}) or {}),
            "ic_hint": dict(payload.get("ic_hint", {}) or {}),
            "recommendation": dict(payload.get("recommendation", {}) or {}),
            "report_excerpt": str(payload.get("report_excerpt", "") or ""),
        }
        for symbol, payload in review_by_symbol.items()
    }
    fundamental_coverage_by_symbol: dict[str, dict[str, Any]] = {}
    enhanced_data_flags_by_symbol: dict[str, dict[str, Any]] = {}
    kline_diagnostics: dict[str, dict[str, Any]] = {}
    intelligence_diagnostics: dict[str, dict[str, Any]] = {}
    for symbol, payload in branch_signals_by_symbol.items():
        reviewed_branchs = dict(payload.get("reviewed_branch_verdicts", {}) or {})
        fundamental_payload = dict(reviewed_branchs.get("fundamental", {}) or {})
        fundamental_meta = dict(fundamental_payload.get("metadata", {}) or {})
        fundamental_quality = dict(fundamental_meta.get("data_quality", {}) or {})
        snapshot_quality_by_symbol = dict(fundamental_quality.get("snapshot_quality_by_symbol", {}) or {})
        fundamental_coverage_by_symbol[symbol] = {
            "coverage_ratio": fundamental_quality.get("coverage_ratio"),
            "missing_modules": (
                dict(fundamental_quality.get("missing_modules", {}) or {}).get(symbol, [])
                if isinstance(fundamental_quality.get("missing_modules"), dict)
                else fundamental_quality.get("missing_modules", [])
            ),
            "snapshot_quality": dict(snapshot_quality_by_symbol.get(symbol, {}) or {}),
            "module_coverage": dict(fundamental_meta.get("module_coverage", {}) or {}),
        }
        enhanced_data_flags_by_symbol[symbol] = dict(snapshot_quality_by_symbol.get(symbol, {}) or {})
        if reviewed_branchs.get("kline"):
            kline_diagnostics[symbol] = dict(reviewed_branchs.get("kline", {}) or {})
        if reviewed_branchs.get("intelligence"):
            intelligence_diagnostics[symbol] = dict(reviewed_branchs.get("intelligence", {}) or {})

    formal_warnings = collect_formal_report_warnings(
        target_date=str(completeness_after.get("effective_target_trade_date") or latest_trade_date or ""),
        dominant_local_snapshot_date=analysis_trade_date,
        completeness_state=completeness_after,
        holdings_review=holdings_review.to_dict(orient="records"),
        branch_diagnostics=branch_signals_by_symbol,
        fundamental_coverage_by_symbol=fundamental_coverage_by_symbol,
        enhanced_data_flags_by_symbol=enhanced_data_flags_by_symbol,
        kline_diagnostics=kline_diagnostics,
        intelligence_diagnostics=intelligence_diagnostics,
        review_layer_diagnostics={
            "effective_call_count": int(review_effective_summary.get("call_count", 0) or 0),
            "attempt_call_count": int(review_attempt_summary.get("call_count", 0) or 0),
            "degraded_symbols": list(degraded_symbols.keys()),
        },
    )
    holding_diagnostics = build_holding_decision_diagnostics(
        holdings_review=holdings_review.to_dict(orient="records"),
        warnings=formal_warnings,
        provisional_label_by_symbol={
            str(row.symbol): str(getattr(row, "recommended_action", ""))
            for row in holdings_review.itertuples()
        },
        data_date_by_symbol={str(row.symbol): analysis_trade_date for row in holdings_review.itertuples()},
        branch_signals_by_symbol=branch_signals_by_symbol,
    )
    decision_guardrail = apply_report_decision_guardrail(
        provisional_label="rebalance" if orders else "no_action",
        warnings=formal_warnings,
        holding_diagnostics=holding_diagnostics,
        llm_confidences=list(holdings_review["llm_confidence"]) if "llm_confidence" in holdings_review.columns else [],
    )
    if decision_guardrail.display_label in {"no_action_evidence_impaired", "hold_arbitrated"}:
        adjusted_diagnostics: list[HoldingDecisionDiagnostic] = []
        for item in holding_diagnostics:
            adjusted_diagnostics.append(
                HoldingDecisionDiagnostic(
                    symbol=item.symbol,
                    name=item.name,
                    data_date=item.data_date,
                    final_label=decision_guardrail.display_label,
                    branch_vs_final=(
                        "conflict_downgraded"
                        if item.branch_vs_final == "conflict_requires_arbitration"
                        else item.branch_vs_final
                    ),
                    llm_confidence=item.llm_confidence,
                    warning_codes=list(item.warning_codes),
                    decision_impact=(
                        item.decision_impact
                        if item.decision_impact != "none"
                        else "downgraded_final_label"
                    ),
                    arbitration_note=item.arbitration_note or decision_guardrail.arbitration_note,
                )
            )
        holding_diagnostics = adjusted_diagnostics

    diagnostic_by_symbol = {item.symbol: item for item in holding_diagnostics}

    def _diagnostic_attr(symbol: Any, attr: str, default: Any = "") -> Any:
        item = diagnostic_by_symbol.get(str(symbol).strip().upper())
        return getattr(item, attr, default) if item is not None else default

    holdings_review["report_data_date"] = holdings_review["symbol"].map(
        lambda value: _diagnostic_attr(value, "data_date", "")
    )
    holdings_review["report_guardrail_label"] = holdings_review["symbol"].map(
        lambda value: _diagnostic_attr(value, "final_label", "unknown")
    )
    holdings_review["report_branch_vs_final"] = holdings_review["symbol"].map(
        lambda value: _diagnostic_attr(value, "branch_vs_final", "unknown")
    )
    holdings_review["report_warning_codes"] = holdings_review["symbol"].map(
        lambda value: ",".join(_diagnostic_attr(value, "warning_codes", []))
    )
    holdings_review["report_decision_impact"] = holdings_review["symbol"].map(
        lambda value: _diagnostic_attr(value, "decision_impact", "none")
    )
    holdings_review["report_arbitration_note"] = holdings_review["symbol"].map(
        lambda value: _diagnostic_attr(value, "arbitration_note", "")
    )
    diagnostic_table = render_holding_diagnostic_markdown_table(holding_diagnostics)
    typed_warning_codes = sorted({warning.code for warning in formal_warnings})
    switch_rows_by_buy_symbol = {
        str(row.buy_symbol): row._asdict()
        for row in switch_plan_df.itertuples(index=False)
    }
    candidate_lines = [
        _format_candidate_advice_line(row, switch_rows_by_buy_symbol.get(str(row.symbol)))
        for row in candidate_pool.head(5).itertuples()
    ]
    factor_shadow_status = load_factor_library_shadow_status(
        root_dir=PROJECT_ROOT / "data" / "factor_library",
        as_of=now.strftime("%Y-%m-%d"),
    )
    factor_shadow_lines = render_factor_library_shadow_markdown(factor_shadow_status).splitlines()
    if factor_shadow_lines[:2] == ["## Factor Library Status (Read-only Shadow)", ""]:
        factor_shadow_lines = factor_shadow_lines[2:]
    top_switch_action = str(switch_plan_df.iloc[0]["action"]) if not switch_plan_df.empty else ""
    switch_now = top_switch_action == "switch_now"
    switch_prepare = top_switch_action == "prepare_switch"

    report_lines = [
        "# A股激进科技制造策略正式复盘报告",
        "",
        "## 1. 记录信息",
        "",
        "- 市场：A股（CN）",
        "- 策略：`aggressive_tech_manufacturing`",
        f"- 上一条正式记录：`{source_record}`",
        f"- 本次正式记录时间：{timestamp_long}",
        f"- 盘中快照：{quote_snapshot or 'N/A'}",
        f"- 完整性校验：**{'已通过' if completeness_passed else '未通过'}**",
        "- 分析口径：**直接基于本地已有数据，不自动补数，不把完整性校验作为正式结论前置阻断。**",
        "- 分析链路：**统一 DAG / review-layer（逐持仓主线复核）**",
        "",
        "## 0. 正式结果速览",
        "",
        f"- 正式结论：**{rebalance_reason}**",
        f"- 报告展示标签：**`{decision_guardrail.display_label}`**",
        f"- 数据完整性状态：**{data_status}**",
        f"- 今日是否执行调仓：**{'是' if orders else '否'}**",
        (
            f"- 今日是否执行换仓：**{'是' if switch_now else '否'}**"
            if switch_plan_df is not None
            else "- 今日是否执行换仓：**否**"
        ),
        (
            f"- Typed diagnostics：`{', '.join(typed_warning_codes)}`"
            if typed_warning_codes
            else "- Typed diagnostics：无"
        ),
        (
            f"- 决策护栏说明：{decision_guardrail.arbitration_note}"
            if str(decision_guardrail.arbitration_note).strip()
            else "- 决策护栏说明：无"
        ),
        (
            f"- LLM 复核状态：**已执行**，原始尝试 `{review_attempt_summary.get('call_count', 0)}` 次"
            f"（成功 `{review_attempt_summary.get('success_count', 0)}` / 失败 `{review_attempt_summary.get('failed_count', 0)}` / "
            f"fallback `{review_attempt_summary.get('fallback_count', 0)}`），"
            f"有效输出 `{review_effective_summary.get('call_count', 0)}` 次，"
            f"`{review_attempt_summary.get('total_tokens', 0)}` tokens，"
            f"估算成本 `${float(review_attempt_summary.get('estimated_cost_usd', 0.0)):.6f}`"
        ),
        (
            "- Review-layer 降级：`"
            + "，".join(sorted(degraded_symbols.keys()))
            + "`"
            if degraded_symbols
            else "- Review-layer 降级：无"
        ),
        (
            "- 备选投资建议："
            + "；".join(
                f"{row.symbol}({row.name})/{row.theme_label}"
                for row in candidate_pool.head(3).itertuples()
            )
            if not candidate_pool.empty
            else "- 备选投资建议：暂无"
        ),
        f"- 明日准备事项：{'；'.join(tomorrow_focus)}",
        "",
        "## 2. 数据状态与本地快照",
        "",
        f"- 当前运行时间：`{timestamp_long}`",
        f"- 首轮完整性：`{'通过' if completeness_before['complete'] else '未通过'}`，阻塞缺口 `{int(completeness_before['blocking_incomplete_count'])}` 个",
        f"- 是否执行补数：`{'是' if attempted_backfill else '否'}`",
        "- 本轮策略：即使存在数据滞后或局部缺口，也继续给出正式分析与正式建议，但会明确披露数据限制。",
        *data_snapshot_lines,
        "",
        "## 3. 当前组合盈亏判断",
        "",
        f"- 截至 `quote_snapshot={quote_snapshot or 'N/A'}`，组合总资产 **{total_value_after:,.2f} 元**，较初始资金 **{portfolio_pnl_after:,.2f} 元**，收益率 **{portfolio_pnl_after / initial_capital:.2%}**。",
        f"- 相对上一条正式记录 `{source_record}` 的 **{source_total_value:,.2f} 元**，当前盘中净值变动 **{total_value_after - source_total_value:,.2f} 元**。",
        "- 当前浮盈仓位：" + _format_top_holdings_by_unrealized_pnl(float_winners, positive=True),
        "- 当前浮亏前三：" + _format_top_holdings_by_unrealized_pnl(float_losers, positive=False),
        "- 相对上一条正式记录的正向收益贡献："
        + _format_top_delta_vs_source_record(holdings_review, positive=True),
        "- 相对上一条正式记录的拖累来源前三："
        + _format_top_delta_vs_source_record(holdings_review, positive=False),
        "",
        "## 4. A股整体市场风格与指数结构",
        "",
        "### 4.1 指数结构（盘中快照）",
        "",
    ]

    for code in index_quote_codes:
        payload = indices.get(code)
        if not payload:
            continue
        report_lines.append(f"- {payload['name']}：{payload['change_pct']:+.2f}%")

    report_lines.extend(
        [
            "",
            f"结论：{style_text}",
            "",
            "### 4.2 广度与市场内部状态（基于最新完整日线）",
            "",
            (
                f"- HS300：1日上涨占比 {breadth['hs300']['ret1_positive_ratio']:.1%}，"
                f"20日上涨占比 {breadth['hs300']['ret20_positive_ratio']:.1%}，"
                f"`MA20 > MA60` 占比 {breadth['hs300']['ma20_gt_ma60_ratio']:.1%}"
            ),
            (
                f"- ZZ500：1日上涨占比 {breadth['zz500']['ret1_positive_ratio']:.1%}，"
                f"20日上涨占比 {breadth['zz500']['ret20_positive_ratio']:.1%}，"
                f"`MA20 > MA60` 占比 {breadth['zz500']['ma20_gt_ma60_ratio']:.1%}"
            ),
            (
                f"- ZZ1000：1日上涨占比 {breadth['zz1000']['ret1_positive_ratio']:.1%}，"
                f"20日上涨占比 {breadth['zz1000']['ret20_positive_ratio']:.1%}，"
                f"`MA20 > MA60` 占比 {breadth['zz1000']['ma20_gt_ma60_ratio']:.1%}"
            ),
            "",
            "### 4.3 科技 / 高端制造主线强弱",
            "",
            f"- {strongest_theme_text}",
            f"- {weakest_theme_text}",
            (
                f"- 主题强弱排序：{', '.join(item['theme'] for item in theme_strength)}"
                if theme_strength
                else "- 主题强弱排序：暂无"
            ),
            "",
            "## 5. 当前策略持仓复盘",
            "",
            "### 5.1 持仓相对强弱",
            "",
            "- 今日相对最强："
            + _format_symbol_set(holdings_review.sort_values(
                ["today_change_pct", "score_full_market"], ascending=[False, False]
            ).head(3)["symbol"].tolist()),
            "- 今日相对最弱："
            + _format_symbol_set(holdings_review.sort_values(
                ["today_change_pct", "score_full_market"], ascending=[True, True]
            ).head(3)["symbol"].tolist()),
            "",
            "### 5.2 当前弱点与结构预警",
            "",
        ]
    )

    weak_rows = holdings_review[holdings_review["current_price"] < holdings_review["stage_stop_price"]]
    if weak_rows.empty:
        report_lines.append("- 目前持仓都在阶段止损位上方，尚未触发新的硬性减仓信号。")
    else:
        for row in weak_rows.itertuples():
            report_lines.append(
                f"- `{row.symbol}` 当前价 {row.current_price:.2f} 仍低于阶段止损位 {row.stage_stop_price:.2f}，"
                f"状态 `{row.position_role}`。"
            )

    report_lines.extend(
        [
            "",
            "### 5.3 统一 DAG / Review Layer 复核",
            "",
            (
                f"- 分支模型：`{review_model_role_metadata.get('resolved_branch_model') or review_model_role_metadata.get('branch_model') or 'N/A'}`；"
                f"主模型：`{review_model_role_metadata.get('resolved_master_model') or review_model_role_metadata.get('master_model') or 'N/A'}`"
            ),
            (
                f"- 原始尝试汇总：`{review_attempt_summary.get('call_count', 0)}` 次，"
                f"成功 `{review_attempt_summary.get('success_count', 0)}` 次，"
                f"失败 `{review_attempt_summary.get('failed_count', 0)}` 次，"
                f"fallback `{review_attempt_summary.get('fallback_count', 0)}` 次，"
                f"tokens `{review_attempt_summary.get('total_tokens', 0)}`，"
                f"成本 `${float(review_attempt_summary.get('estimated_cost_usd', 0.0)):.6f}`"
            ),
            (
                f"- 有效输出汇总：`{review_effective_summary.get('call_count', 0)}` 次，"
                f"成功 `{review_effective_summary.get('success_count', 0)}` 次，"
                f"tokens `{review_effective_summary.get('total_tokens', 0)}`，"
                f"成本 `${float(review_effective_summary.get('estimated_cost_usd', 0.0)):.6f}`"
            ),
            (
                "- review-layer fallback："
                + "；".join(review_layer.get("fallback_reasons", [])[:6])
                if review_layer.get("fallback_reasons")
                else "- review-layer fallback：无"
            ),
            (
                "- review-layer 降级标的："
                + "；".join(f"{symbol}: {reason}" for symbol, reason in list(degraded_symbols.items())[:6])
                if degraded_symbols
                else "- review-layer 降级标的：无"
            ),
            "",
            "### 5.4 是否需要调仓",
            "",
            f"- 报告展示标签：`{decision_guardrail.display_label}`",
            (
                f"- 护栏仲裁说明：{decision_guardrail.arbitration_note}"
                if str(decision_guardrail.arbitration_note).strip()
                else "- 护栏仲裁说明：无"
            ),
            (
                f"- 触发 warning codes：`{', '.join(typed_warning_codes)}`"
                if typed_warning_codes
                else "- 触发 warning codes：无"
            ),
            "",
            "#### 5.4.1 决策诊断",
            "",
            diagnostic_table,
            "",
            "#### 5.4.2 正式建议",
            "",
            f"- 正式建议：**{'执行温和调仓' if orders else '本次不执行调仓'}**",
            (
                "- 原因：主线最强分支重新走强，今天更像内部修复而不是主线失效；弱支线虽然没有完全修复，但还不足以在单日里推翻长期主线。"
                if not orders
                else "- 原因：弱支线在完整日线与盘中都继续落后，已满足温和减仓条件。"
            ),
        ]
    )
    for row in holdings_review.itertuples():
        report_lines.append(_format_holding_advice_line(row))
    report_lines.extend(
        [
            "",
            "#### 5.4.3 证据质量与工程诊断",
            "",
            f"- 诊断汇总：{_format_warning_count_summary(formal_warnings)}",
            f"- Material 级证据缺口：{_format_warning_messages(formal_warnings, severity='material', limit=3)}",
            (
                f"- LLM / review-layer 状态：原始尝试 `{review_attempt_summary.get('call_count', 0)}` 次，"
                f"有效输出 `{review_effective_summary.get('call_count', 0)}` 次；"
                + (
                    "降级标的 `"
                    + "，".join(sorted(degraded_symbols.keys()))
                    + "`。"
                    if degraded_symbols
                    else "无逐标的 review-layer 降级。"
                )
            ),
            "- 工程诊断说明：provider、snapshot、K 线 evaluator、旧 intelligence batch 等诊断仅用于说明证据等级，不直接作为逐票投资建议正文。",
            "",
            "##### 持仓诊断明细",
            "",
            diagnostic_table,
        ]
    )
    report_lines.extend(
        [
            "",
            "### 5.5 备选投资建议",
            "",
            (
                "- 本轮备选池来自本地 `results/cn_analysis_full/all_candidates.json` 与最新主导本地快照强度交叉筛选。"
                if not candidate_pool.empty
                else "- 本轮未提取到有效备选池。"
            ),
            *candidate_lines,
            "",
            "### 5.6 现持仓与备选标的换仓比较",
            "",
            (
                "- 正式换仓结论：**执行换仓**，优先按 20% 试探性替换最弱持仓。"
                if switch_now
                else (
                    "- 正式换仓结论：**暂不换仓，但已形成预备换仓对象**。"
                    if switch_prepare
                    else "- 正式换仓结论：**暂不换仓**，现阶段以观察池跟踪为主。"
                )
            ),
            (
                "- 不换仓条件：当前本地 strict 快照仍不完整，先不把结构优势直接转成实单。"
                if not completeness_passed
                else "- 不换仓条件：若弱持仓重新收复阶段止损位且候选优势收敛，则维持原组合。"
            ),
            "",
        ]
    )
    if switch_plan_df.empty:
        report_lines.append("- 当前没有形成明确的一对一换仓比较。")
    else:
        for row in switch_plan_df.itertuples(index=False):
            report_lines.append(
                f"- `{row.sell_symbol}`（{row.sell_name}） vs `{row.buy_symbol}`（{row.buy_name}）："
                f"候选主线 `{row.buy_theme}`，强度分差 `{float(row.score_gap):+.3f}`，"
                f"20日动量差 `{float(row.ret20_gap):+.2%}`；建议 `{row.action}`，"
                f"优先级 `{row.priority}`，比例提示 `{row.switch_ratio_hint}`；"
                f"触发阈值：{row.trigger_threshold}；不换仓条件：{row.no_switch_condition}"
            )

    report_lines.extend(
        [
            "",
            "### 5.7 因子库状态（只读影子观察）",
            "",
            *factor_shadow_lines,
            "",
            "## 6. 面向明日的观察重点和准备事项",
            "",
        ]
    )
    for idx, item in enumerate(tomorrow_focus, start=1):
        report_lines.append(f"{idx}. {item}")

    report_text = "\n".join(report_lines)

    pnl_summary = {
        "record_time": timestamp_long,
        "quote_snapshot": quote_snapshot,
        "initial_capital": initial_capital,
        "cash_before": round(cash_before, 2),
        "market_value_before": total_market_value_before,
        "total_value_before": total_value_before,
        "portfolio_pnl_before": portfolio_pnl_before,
        "portfolio_pnl_pct_before": round(_safe_pct(portfolio_pnl_before, initial_capital), 6),
        "realized_pnl_from_rebalance": realized_pnl,
        "cash_after": round(cash_after, 2),
        "market_value_after": total_market_value_after,
        "total_value_after": total_value_after,
        "portfolio_pnl_after": portfolio_pnl_after,
        "portfolio_pnl_pct_after": round(_safe_pct(portfolio_pnl_after, initial_capital), 6),
        "delta_vs_source_record": round(total_value_after - source_total_value, 2),
    }
    pnl_summary_df = pd.DataFrame([pnl_summary])

    notes_text = _build_notes_payload(
        trade_date=now.strftime("%Y-%m-%d"),
        data_status=data_status,
        market_core_view=f"{style_text}{strongest_theme_text}{weakest_theme_text}",
        pnl_summary=pnl_summary,
        orders=orders,
        switch_plan_df=switch_plan_df,
        candidate_pool=candidate_pool,
        tomorrow_focus=tomorrow_focus,
    )
    DEFAULT_NOTES_PATH.write_text(notes_text, encoding="utf-8")
    formal_diagnostics_payload = {
        "warnings": [_jsonable(item.to_dict()) for item in formal_warnings],
        "holding_diagnostics": [_jsonable(item.to_dict()) for item in holding_diagnostics],
        "decision_guardrail": _jsonable(decision_guardrail.to_dict()),
        "typed_warning_codes": typed_warning_codes,
        "branch_diagnostics_by_symbol": _jsonable(branch_signals_by_symbol),
    }

    manifest = {
        "market": "CN",
        "strategy": "aggressive_tech_manufacturing",
        "timestamp": timestamp,
        "recorded_at": timestamp_long,
        "source_record": source_record,
        "formal_record": True,
        "completeness_passed": completeness_passed,
        "capital_cny": initial_capital,
        "quote_snapshot": quote_snapshot,
        "action_taken_today": bool(orders),
        "analysis_chain": "unified_dag_review_layer_per_holding",
        "analysis_input_policy": "local_snapshot_no_backfill_no_gate",
        "files": {
            "analysis_report": "analysis_report.md",
            "holdings_review": "holdings_review.csv",
            "candidate_pool": "candidate_pool.csv",
            "switch_plan": "switch_plan.csv",
            "orders": "orders.csv",
            "ledger": "ledger.csv",
            "pnl_summary": "pnl_summary.csv",
            "market_snapshot": "market_snapshot.json",
        },
        "raw_exports": {
            "report": f"raw_exports/aggressive_portfolio_{timestamp}_formal_report.md",
            "orders": f"raw_exports/aggressive_portfolio_{timestamp}_formal_orders.csv",
            "ledger": f"raw_exports/aggressive_portfolio_{timestamp}_formal_ledger.csv",
            "pnl_summary": f"raw_exports/aggressive_portfolio_{timestamp}_formal_pnl_summary.csv",
            "holdings_review": f"raw_exports/aggressive_portfolio_{timestamp}_formal_holdings_review.csv",
            "candidate_pool": f"raw_exports/aggressive_portfolio_{timestamp}_formal_candidate_pool.csv",
            "switch_plan": f"raw_exports/aggressive_portfolio_{timestamp}_formal_switch_plan.csv",
        },
        "data_snapshot": {
            "latest_trade_date": latest_trade_date,
            "analysis_trade_date": analysis_trade_date,
            "strict_trade_date": completeness_after.get("strict_trade_date"),
            "stable_trade_date": completeness_after.get("stable_trade_date"),
            "effective_target_trade_date": completeness_after.get("effective_target_trade_date"),
            "freshness_mode": completeness_after.get("freshness_mode"),
            "coverage_ratio": completeness_after.get("coverage_ratio"),
            "blocking_incomplete_count": completeness_after.get("blocking_incomplete_count"),
            "completeness": completeness_after,
            "download_report": str(download_report_path) if download_report_path else None,
        },
        "review_layer": {
            "reviewed_symbols": list(review_layer.get("reviewed_symbols", []) or []),
            "llm_usage_summary": review_attempt_summary,
            "llm_attempt_summary": review_attempt_summary,
            "llm_effective_summary": review_effective_summary,
            "model_role_metadata": review_model_role_metadata,
            "fallback_reasons": list(review_layer.get("fallback_reasons", []) or []),
            "degraded_symbols": degraded_symbols,
            "session_ids": dict(review_layer.get("session_ids", {}) or {}),
            "symbol_diagnostics": _jsonable(branch_signals_by_symbol),
        },
        "candidate_pool": candidate_pool.to_dict(orient="records"),
        "switch_plan": switch_plan_df.to_dict(orient="records"),
        "formal_diagnostics": formal_diagnostics_payload,
    }
    market_snapshot = {
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "quote_snapshot": quote_snapshot,
        "analysis_trade_date": analysis_trade_date,
        "indices": indices,
        "breadth": breadth,
        "data_status": data_status,
        "completeness": completeness_after,
        "portfolio": {
            "total_value": total_value_after,
            "portfolio_pnl": portfolio_pnl_after,
            "portfolio_pnl_pct": round(_safe_pct(portfolio_pnl_after, initial_capital), 6),
            "delta_vs_source_record": round(total_value_after - source_total_value, 2),
        },
        "theme_strength": theme_strength,
        "candidate_pool": candidate_pool.to_dict(orient="records"),
        "switch_plan": switch_plan_df.to_dict(orient="records"),
        "review_layer": {
            "reviewed_symbols": list(review_layer.get("reviewed_symbols", []) or []),
            "llm_usage_summary": review_attempt_summary,
            "llm_attempt_summary": review_attempt_summary,
            "llm_effective_summary": review_effective_summary,
            "model_role_metadata": review_model_role_metadata,
            "fallback_reasons": list(review_layer.get("fallback_reasons", []) or []),
            "degraded_symbols": degraded_symbols,
            "session_ids": dict(review_layer.get("session_ids", {}) or {}),
            "symbol_diagnostics": _jsonable(branch_signals_by_symbol),
        },
        "formal_diagnostics": formal_diagnostics_payload,
        "download_report": str(download_report_path) if download_report_path else None,
        "quote_fetch_error": quote_error or None,
    }

    _write_outputs(
        base_dir=base_dir,
        run_dir=run_dir,
        report_text=report_text,
        holdings_review=holdings_review,
        candidate_pool=candidate_pool,
        switch_plan_df=switch_plan_df,
        ledger=updated_ledger,
        orders_df=orders_df,
        pnl_summary_df=pnl_summary_df,
        manifest=manifest,
        market_snapshot=market_snapshot,
    )

    return {
        "timestamp": timestamp,
        "timestamp_long": timestamp_long,
        "run_dir": str(run_dir),
        "latest_trade_date": latest_trade_date,
        "analysis_trade_date": analysis_trade_date,
        "completeness_passed": completeness_passed,
        "action_taken_today": bool(orders),
        "switch_action_today": top_switch_action or "none",
        "report_guardrail_label": decision_guardrail.display_label,
        "typed_warning_codes": typed_warning_codes,
        "data_status": data_status,
        "style_view": style_text,
        "review_layer_degraded_symbols": degraded_symbols,
        "candidate_pool_top": candidate_pool.head(3).to_dict(orient="records"),
        "switch_plan": switch_plan_df.to_dict(orient="records"),
        "tech_mainline": {
            "strongest": strongest_theme_text,
            "weakest": weakest_theme_text,
        },
        "formal_diagnostics": formal_diagnostics_payload,
        "pnl_summary": pnl_summary,
        "elapsed_sec": round(time.time() - started, 2),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A股激进科技制造策略正式复盘跟踪器")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--years", type=int, default=7)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--source-record", default=None)
    parser.add_argument("--allowed-stale-symbols", nargs="*", default=[])
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_tracker(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
