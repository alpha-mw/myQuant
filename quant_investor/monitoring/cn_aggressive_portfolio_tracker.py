"""
A股激进科技制造策略正式复盘跟踪器。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from quant_investor.factors.report import (
    load_factor_library_shadow_status,
    render_factor_library_shadow_markdown,
)
from quant_investor.market.config import get_market_settings
from quant_investor.market.download_cn import CNFullMarketDownloader
from quant_investor.market.market_data_reader import MarketDataReader
from quant_investor.monitoring.cn_aggressive_rebalance import (
    CATEGORY_THEME_LABELS,
    CANDIDATE_POOL_PATH,
    INDEX_QUOTES,
    QUOTE_TIMEOUT,
    THEME_BASKETS,
    ProposedOrder,
    _allocate_run_timestamp,
    _apply_orders,
    _build_candidate_pool,
    _build_rebalance_plan,
    _build_switch_plan,
    _fetch_tencent_quotes,
    _load_previous_record,
    _map_symbol_to_quote_code,
    _market_style_conclusion,
    _position_action,
    _position_reason,
    _position_role,
    _summarize_theme_strength,
    _tech_mainline_conclusion,
)
from quant_investor.monitoring.cn_aggressive_reporting import (
    _build_data_snapshot_lines,
    _build_data_status_summary,
    _build_notes_payload,
    _format_candidate_advice_line,
    _format_holding_advice_line,
    _format_holding_snapshot_set,
    _format_top_delta_vs_source_record,
    _format_top_holdings_by_unrealized_pnl,
    _format_trade_date,
    _format_warning_count_summary,
    _format_warning_messages,
    _resolve_analysis_trade_date,
    _write_outputs,
)
from quant_investor.monitoring.cn_aggressive_review_layer import (
    REQUIRED_DAG_BRANCHES,
    _build_dag_four_branch_compliance,
    _codex_handoff_review_layer,
    _empty_llm_usage_summary,
    _llm_usage_summary_to_dict,
    _load_daily_config_llm_settings,
    _render_dag_compliance_markdown,
    _serialize_reviewed_branch_verdicts,
    _serialize_symbol_review_bundle,
    _trade_recommendation_to_dict,
)
from quant_investor.monitoring.cn_aggressive_utils import (
    _jsonable,
    _now_local,
    _plain_dict,
    _safe_float,
    _safe_pct,
)
from quant_investor.monitoring import cn_aggressive_market_metrics as _market_metrics
from quant_investor.monitoring.cn_aggressive_report_renderer import build_formal_report_text as _build_formal_report_text
from quant_investor.monitoring.cn_aggressive_review_runtime import run_unified_review_mainline_for_holdings as _run_unified_review_mainline_impl
from quant_investor.llm_policy import llm_handoff_reason
from quant_investor.pipeline import QuantInvestor
from quant_investor.reporting.formal_diagnostics import (
    HoldingDecisionDiagnostic,
    apply_report_decision_guardrail,
    build_holding_decision_diagnostics,
    collect_formal_report_warnings,
    is_previous_day_realtime_decision_sufficient,
    render_holding_diagnostic_markdown_table,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_DIR = (
    PROJECT_ROOT / "results" / "strategy_records" / "CN" / "aggressive_tech_manufacturing"
)
DEFAULT_NOTES_PATH = DEFAULT_BASE_DIR / "latest_notes_payload.md"
DEFAULT_INITIAL_CAPITAL = 1_000_000.0
MARKET_METRICS_CACHE_SCHEMA_VERSION = _market_metrics.MARKET_METRICS_CACHE_SCHEMA_VERSION
MARKET_METRICS_COMPONENT_KEYS = _market_metrics.MARKET_METRICS_COMPONENT_KEYS
MARKET_METRICS_CATEGORIES = _market_metrics.MARKET_METRICS_CATEGORIES
MARKET_METRICS_OUTPUT_COLUMNS = _market_metrics.MARKET_METRICS_OUTPUT_COLUMNS
MARKET_METRICS_REQUIRED_COLUMNS = _market_metrics.MARKET_METRICS_REQUIRED_COLUMNS
MarketMetricsBundle = _market_metrics.MarketMetricsBundle
_compute_category_breadth = _market_metrics._compute_category_breadth
_compute_full_market_metrics = _market_metrics._compute_full_market_metrics
_compute_market_metrics_and_breadth = _market_metrics._compute_market_metrics_and_breadth
_components_fingerprint = _market_metrics._components_fingerprint
_derive_stage_levels = _market_metrics._derive_stage_levels
_load_cached_market_metrics_bundle = _market_metrics._load_cached_market_metrics_bundle
_load_history_frame = _market_metrics._load_history_frame
_load_or_compute_market_metrics_bundle = (
    _market_metrics.load_or_compute_market_metrics_bundle
)
_market_metrics_cache_dir = _market_metrics._market_metrics_cache_dir
_metric_return = _market_metrics._metric_return
_normalize_market_metrics_frame = _market_metrics._normalize_market_metrics_frame
_price_series = _market_metrics._price_series
_read_frame_from_result = _market_metrics._read_frame_from_result
_reader_snapshot_payload = _market_metrics._reader_snapshot_payload
_score_full_market_metrics = _market_metrics._score_full_market_metrics
_validate_market_metrics_frame = _market_metrics._validate_market_metrics_frame
_write_market_metrics_cache = _market_metrics._write_market_metrics_cache

__all__ = [
    "CATEGORY_THEME_LABELS",
    "CANDIDATE_POOL_PATH",
    "DEFAULT_BASE_DIR",
    "DEFAULT_INITIAL_CAPITAL",
    "DEFAULT_NOTES_PATH",
    "INDEX_QUOTES",
    "MARKET_METRICS_CACHE_SCHEMA_VERSION",
    "MARKET_METRICS_CATEGORIES",
    "MARKET_METRICS_COMPONENT_KEYS",
    "MARKET_METRICS_OUTPUT_COLUMNS",
    "MARKET_METRICS_REQUIRED_COLUMNS",
    "PROJECT_ROOT",
    "ProposedOrder",
    "QUOTE_TIMEOUT",
    "REQUIRED_DAG_BRANCHES",
    "THEME_BASKETS",
    "build_parser",
    "main",
    "run_tracker",
    "_build_formal_report_text",
    "_run_unified_review_mainline_impl",
    "_codex_handoff_review_layer",
    "_empty_llm_usage_summary",
    "_format_holding_advice_line",
    "_format_holding_snapshot_set",
    "_format_top_delta_vs_source_record",
    "_format_top_holdings_by_unrealized_pnl",
    "_format_warning_count_summary",
    "_format_warning_messages",
    "_llm_usage_summary_to_dict",
    "_plain_dict",
    "_render_dag_compliance_markdown",
    "_serialize_reviewed_branch_verdicts",
    "_serialize_symbol_review_bundle",
    "_trade_recommendation_to_dict",
]



def _run_unified_review_mainline_for_holdings(
    *,
    source_ledger: pd.DataFrame,
    latest_trade_date: str,
    source_record: str,
) -> dict[str, Any]:
    return _run_unified_review_mainline_impl(
        source_ledger=source_ledger,
        latest_trade_date=latest_trade_date,
        source_record=source_record,
        quant_investor_cls=QuantInvestor,
        load_daily_config_llm_settings=_load_daily_config_llm_settings,
        llm_handoff_reason_fn=llm_handoff_reason,
        sleep_fn=time.sleep,
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
    market_data_reader = MarketDataReader(market="CN")
    skip_market_metrics_prewarm = bool(getattr(args, "skip_market_metrics_prewarm", False))

    market_metrics_bundle = _load_or_compute_market_metrics_bundle(
        base_dir=base_dir,
        components=components,
        reader=market_data_reader,
        latest_trade_date=analysis_trade_date,
        completeness_report=completeness_after,
        skip_prewarm=skip_market_metrics_prewarm,
    )
    full_metrics = market_metrics_bundle.full_metrics
    breadth = market_metrics_bundle.breadth
    market_metrics_cache_meta = market_metrics_bundle.cache_meta

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
    codex_handoff_active = bool(review_layer.get("codex_handoff", False))

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
    diagnostic_completeness_after = {
        **completeness_after,
        "quote_snapshot": quote_snapshot,
    }
    decision_data_sufficient = is_previous_day_realtime_decision_sufficient(
        target_date=str(completeness_after.get("effective_target_trade_date") or latest_trade_date or ""),
        dominant_local_snapshot_date=analysis_trade_date,
        completeness_state=diagnostic_completeness_after,
        quote_snapshot=quote_snapshot,
    )

    indices = {
        code: {
            **quote_payload[code],
            "name": INDEX_QUOTES[code],
        }
        for code in index_quote_codes
        if code in quote_payload
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
        if codex_handoff_active and llm_confidence_value is None:
            llm_confidence: float | None = None
            llm_confidence_source = "codex_handoff"
        else:
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
                "llm_confidence": round(llm_confidence, 6) if llm_confidence is not None else None,
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
        decision_data_sufficient=decision_data_sufficient,
        analysis_trade_date=analysis_trade_date,
        strict_trade_date=str(completeness_after.get("strict_trade_date") or ""),
    )
    switch_plan_df = _build_switch_plan(
        holdings_review=holdings_review,
        candidate_pool=candidate_pool,
        completeness_passed=completeness_passed,
        decision_data_sufficient=decision_data_sufficient,
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
    data_status = _build_data_status_summary(
        completeness_after,
        analysis_trade_date=analysis_trade_date,
        decision_data_sufficient=decision_data_sufficient,
    )
    tomorrow_focus = [
        "确认先进材料与光通信的修复能否延续，不让单日反弹误判为全面重启",
        "继续观察 `大族激光 / 中国西电` 是否能重新站回阶段止损位",
        "跟踪科创50 相对沪深300 的强弱差，判断资金是否继续偏向硬科技",
    ]
    if not completeness_passed and decision_data_sufficient:
        tomorrow_focus.insert(
            0,
            (
                f"收盘后确认 strict 交易日 `{_format_trade_date(completeness_after.get('strict_trade_date'))}` "
                "日线落库，并复核盘中前日线+实时行情结论"
            ),
        )
    elif not completeness_passed:
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
        decision_data_sufficient=decision_data_sufficient,
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
    dag_four_branch_compliance = _build_dag_four_branch_compliance(
        review_symbols=list(branch_signals_by_symbol.keys()),
        effective_local_holding_symbols=[
            str(symbol).strip().upper()
            for symbol in list(source_ledger["symbol"])
            if str(symbol).strip()
        ],
        branch_signals_by_symbol=branch_signals_by_symbol,
    )
    fundamental_coverage_by_symbol: dict[str, dict[str, Any]] = {}
    enhanced_data_flags_by_symbol: dict[str, dict[str, Any]] = {}
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
        if reviewed_branchs.get("intelligence"):
            intelligence_diagnostics[symbol] = dict(reviewed_branchs.get("intelligence", {}) or {})

    formal_warnings = collect_formal_report_warnings(
        target_date=str(completeness_after.get("effective_target_trade_date") or latest_trade_date or ""),
        dominant_local_snapshot_date=analysis_trade_date,
        completeness_state=diagnostic_completeness_after,
        holdings_review=holdings_review.to_dict(orient="records"),
        branch_diagnostics=branch_signals_by_symbol,
        fundamental_coverage_by_symbol=fundamental_coverage_by_symbol,
        enhanced_data_flags_by_symbol=enhanced_data_flags_by_symbol,
        intelligence_diagnostics=intelligence_diagnostics,
        review_layer_diagnostics={
            "effective_call_count": int(review_effective_summary.get("call_count", 0) or 0),
            "attempt_call_count": int(review_attempt_summary.get("call_count", 0) or 0),
            "degraded_symbols": list(degraded_symbols.keys()),
            "fallback_reasons": list(review_layer.get("fallback_reasons", []) or []),
            "codex_handoff": codex_handoff_active,
            "local_llm_disabled": bool(review_layer.get("local_llm_disabled", False)),
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

    report_text = _build_formal_report_text(
        source_record=source_record,
        timestamp_long=timestamp_long,
        quote_snapshot=quote_snapshot,
        completeness_passed=completeness_passed,
        decision_data_sufficient=decision_data_sufficient,
        dag_four_branch_compliance=dag_four_branch_compliance,
        rebalance_reason=rebalance_reason,
        decision_guardrail=decision_guardrail,
        data_status=data_status,
        orders=orders,
        switch_plan_df=switch_plan_df,
        typed_warning_codes=typed_warning_codes,
        codex_handoff_active=codex_handoff_active,
        review_attempt_summary=review_attempt_summary,
        review_effective_summary=review_effective_summary,
        degraded_symbols=degraded_symbols,
        candidate_pool=candidate_pool,
        tomorrow_focus=tomorrow_focus,
        completeness_before=completeness_before,
        attempted_backfill=attempted_backfill,
        data_snapshot_lines=data_snapshot_lines,
        total_value_after=total_value_after,
        portfolio_pnl_after=portfolio_pnl_after,
        initial_capital=initial_capital,
        source_total_value=source_total_value,
        float_winners=float_winners,
        float_losers=float_losers,
        holdings_review=holdings_review,
        index_quote_codes=index_quote_codes,
        indices=indices,
        style_text=style_text,
        breadth=breadth,
        strongest_theme_text=strongest_theme_text,
        weakest_theme_text=weakest_theme_text,
        theme_strength=theme_strength,
        review_model_role_metadata=review_model_role_metadata,
        review_layer=review_layer,
        diagnostic_table=diagnostic_table,
        formal_warnings=formal_warnings,
        candidate_lines=candidate_lines,
        switch_now=switch_now,
        switch_prepare=switch_prepare,
        factor_shadow_lines=factor_shadow_lines,
    )

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
        "dag_four_branch_compliance": _jsonable(dag_four_branch_compliance),
    }
    runtime_profile = {
        "schema_version": "cn_aggressive_runtime_profile.v1",
        "market": "CN",
        "strategy": "aggressive_tech_manufacturing",
        "record_id": timestamp,
        "generated_at": _now_local().isoformat(),
        "snapshot_id": str(market_metrics_cache_meta.get("snapshot_id") or ""),
        "analysis_trade_date": analysis_trade_date,
        "total_elapsed_sec": round(time.time() - started, 3),
        "stages": [
            {
                "name": "market_metrics_prewarm",
                "status": str(market_metrics_cache_meta.get("status") or "unknown"),
                "elapsed_sec": round(float(market_metrics_cache_meta.get("compute_elapsed_sec", 0.0) or 0.0), 3),
                "metadata": _jsonable(market_metrics_cache_meta),
            }
        ],
    }

    manifest = {
        "market": "CN",
        "strategy": "aggressive_tech_manufacturing",
        "timestamp": timestamp,
        "recorded_at": timestamp_long,
        "source_record": source_record,
        "formal_record": True,
        "completeness_passed": completeness_passed,
        "decision_data_sufficient": decision_data_sufficient,
        "decision_data_mode": (
            "previous_day_daily_plus_realtime"
            if decision_data_sufficient and not completeness_passed
            else "strict_daily"
        ),
        "capital_cny": initial_capital,
        "quote_snapshot": quote_snapshot,
        "action_taken_today": bool(orders),
        "analysis_chain": "unified_dag_review_layer_per_holding",
        "analysis_input_policy": (
            "previous_day_daily_plus_realtime_no_backfill_no_gate"
            if decision_data_sufficient and not completeness_passed
            else "local_snapshot_no_backfill_no_gate"
        ),
        "files": {
            "analysis_report": "analysis_report.md",
            "holdings_review": "holdings_review.csv",
            "candidate_pool": "candidate_pool.csv",
            "switch_plan": "switch_plan.csv",
            "orders": "orders.csv",
            "ledger": "ledger.csv",
            "pnl_summary": "pnl_summary.csv",
            "market_snapshot": "market_snapshot.json",
            "runtime_profile": "runtime_profile.json",
        },
        "raw_exports": {
            "report": f"raw_exports/aggressive_portfolio_{timestamp}_formal_report.md",
            "orders": f"raw_exports/aggressive_portfolio_{timestamp}_formal_orders.csv",
            "ledger": f"raw_exports/aggressive_portfolio_{timestamp}_formal_ledger.csv",
            "pnl_summary": f"raw_exports/aggressive_portfolio_{timestamp}_formal_pnl_summary.csv",
            "holdings_review": f"raw_exports/aggressive_portfolio_{timestamp}_formal_holdings_review.csv",
            "candidate_pool": f"raw_exports/aggressive_portfolio_{timestamp}_formal_candidate_pool.csv",
            "switch_plan": f"raw_exports/aggressive_portfolio_{timestamp}_formal_switch_plan.csv",
            "runtime_profile": "raw_exports/runtime_profile.json",
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
            "decision_data_sufficient": decision_data_sufficient,
            "decision_data_mode": (
                "previous_day_daily_plus_realtime"
                if decision_data_sufficient and not completeness_passed
                else "strict_daily"
            ),
            "completeness": completeness_after,
            "download_report": str(download_report_path) if download_report_path else None,
            "market_metrics_cache": _jsonable(market_metrics_cache_meta),
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
            "codex_handoff": codex_handoff_active,
            "local_llm_disabled": bool(review_layer.get("local_llm_disabled", False)),
        },
        "dag_four_branch_compliance": _jsonable(dag_four_branch_compliance),
        "candidate_pool": candidate_pool.to_dict(orient="records"),
        "switch_plan": switch_plan_df.to_dict(orient="records"),
        "formal_diagnostics": formal_diagnostics_payload,
        "market_metrics_prewarm": _jsonable(market_metrics_cache_meta),
        "runtime_profile": runtime_profile,
    }
    market_snapshot = {
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "quote_snapshot": quote_snapshot,
        "analysis_trade_date": analysis_trade_date,
        "indices": indices,
        "breadth": breadth,
        "market_metrics_prewarm": _jsonable(market_metrics_cache_meta),
        "runtime_profile": runtime_profile,
        "data_status": data_status,
        "completeness": completeness_after,
        "decision_data_sufficient": decision_data_sufficient,
        "decision_data_mode": (
            "previous_day_daily_plus_realtime"
            if decision_data_sufficient and not completeness_passed
            else "strict_daily"
        ),
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
            "codex_handoff": codex_handoff_active,
            "local_llm_disabled": bool(review_layer.get("local_llm_disabled", False)),
        },
        "dag_four_branch_compliance": _jsonable(dag_four_branch_compliance),
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
        "decision_data_sufficient": decision_data_sufficient,
        "action_taken_today": bool(orders),
        "switch_action_today": top_switch_action or "none",
        "report_guardrail_label": decision_guardrail.display_label,
        "typed_warning_codes": typed_warning_codes,
        "data_status": data_status,
        "style_view": style_text,
        "review_layer_degraded_symbols": degraded_symbols,
        "codex_handoff": codex_handoff_active,
        "local_llm_disabled": bool(review_layer.get("local_llm_disabled", False)),
        "candidate_pool_top": candidate_pool.head(3).to_dict(orient="records"),
        "switch_plan": switch_plan_df.to_dict(orient="records"),
        "tech_mainline": {
            "strongest": strongest_theme_text,
            "weakest": weakest_theme_text,
        },
        "dag_four_branch_compliance": _jsonable(dag_four_branch_compliance),
        "formal_diagnostics": formal_diagnostics_payload,
        "market_metrics_prewarm": _jsonable(market_metrics_cache_meta),
        "full_market_metrics_cache": _jsonable(market_metrics_cache_meta),
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
    parser.add_argument(
        "--skip-market-metrics-prewarm",
        action="store_true",
        help="工程排障用：跳过启动前 full-market metrics 缓存预热",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_tracker(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
