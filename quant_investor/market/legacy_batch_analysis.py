"""Legacy QuantInvestor batch-analysis compatibility helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from quant_investor.market.config import get_market_settings
from quant_investor.market.full_report import (
    _derive_stock_conclusion,
    _derive_stock_drag_drivers,
    _derive_stock_support_drivers,
    _to_mapping,
    category_name,
)
from quant_investor.market.market_data_reader import MarketDataReader
from quant_investor.market.name_map import get_stock_name
from quant_investor.market.us_market_cap_filter import USMarketCapFilter
from quant_investor.pipeline import QuantInvestor


def _analysis_meta_from_result(
    result: Any,
    *,
    market: str,
    universe: str,
    category: str,
    batch_id: int,
    symbols: list[str],
    analysis_kwargs: dict[str, Any],
) -> dict[str, Any]:
    report_bundle = getattr(result, "agent_report_bundle", None)
    orchestration = dict(getattr(result, "agent_orchestration", {}) or {})
    model_role_metadata = getattr(result, "model_role_metadata", None)
    execution_trace = getattr(result, "execution_trace", None)
    what_if_plan = getattr(result, "what_if_plan", None)
    if model_role_metadata is None and report_bundle is not None:
        model_role_metadata = getattr(
            report_bundle,
            "model_role_metadata",
            None,
        )
    if execution_trace is None and report_bundle is not None:
        execution_trace = getattr(report_bundle, "execution_trace", None)
    if what_if_plan is None and report_bundle is not None:
        what_if_plan = getattr(report_bundle, "what_if_plan", None)

    model_role_payload = _to_mapping(model_role_metadata)
    execution_trace_payload = _to_mapping(execution_trace)
    what_if_plan_payload = _to_mapping(what_if_plan)
    global_context = getattr(result, "global_context", None)
    global_context_payload = (
        global_context.to_dict()
        if hasattr(global_context, "to_dict")
        else {}
    )
    global_metadata = (
        dict(global_context_payload.get("metadata", {}))
        if isinstance(global_context_payload, dict)
        else {}
    )
    data_snapshot_payload = dict(
        orchestration.get("data_snapshot", {})
        or global_metadata.get("data_snapshot", {})
        or {}
    )
    symbol_packets = dict(
        orchestration.get("symbol_research_packets", {}) or {}
    )
    symbol_research_packets = {
        symbol: (
            packet.to_dict() if hasattr(packet, "to_dict") else dict(packet)
        )
        for symbol, packet in symbol_packets.items()
    }
    shortlist_payload = [
        item.to_dict() if hasattr(item, "to_dict") else dict(item)
        for item in list(orchestration.get("shortlist", []) or [])
    ]
    portfolio_decision = orchestration.get("portfolio_decision")
    portfolio_decision_payload = (
        portfolio_decision.to_dict()
        if hasattr(portfolio_decision, "to_dict")
        else {}
    )
    bayesian_records = list(getattr(result, "bayesian_records", []) or [])
    funnel_output = getattr(result, "funnel_output", None)
    return {
        "market": market,
        "universe": universe,
        "category": category,
        "batch_id": batch_id,
        "symbols": list(symbols),
        "batch_count": 1,
        "total_stocks": len(symbols),
        "category_count": 1,
        "agent_layer_enabled": bool(
            getattr(
                result,
                "agent_layer_enabled",
                analysis_kwargs.get("enable_agent_layer", False),
            )
        ),
        "branch_model": str(analysis_kwargs.get("agent_model", "")),
        "master_model": str(analysis_kwargs.get("master_model", "")),
        "master_reasoning_effort": str(
            analysis_kwargs.get("master_reasoning_effort", "")
        ),
        "agent_timeout": float(
            analysis_kwargs.get("agent_timeout", 0.0) or 0.0
        ),
        "master_timeout": float(
            analysis_kwargs.get("master_timeout", 0.0) or 0.0
        ),
        "model_role_metadata": model_role_payload,
        "execution_trace": execution_trace_payload,
        "what_if_plan": what_if_plan_payload,
        "review_bundle": _to_mapping(report_bundle),
        "ic_hints_by_symbol": dict(
            getattr(result, "ic_hints_by_symbol", {}) or {}
        ),
        "report_protocol_version": str(
            getattr(report_bundle, "report_protocol_version", "")
        ),
        "ic_protocol_version": str(
            getattr(report_bundle, "ic_protocol_version", "")
        ),
        "branch_schema_version": str(
            getattr(result, "branch_schema_version", "")
        ),
        "global_context": global_context_payload,
        "data_snapshot": data_snapshot_payload,
        "symbol_research_packets": symbol_research_packets,
        "shortlist": shortlist_payload,
        "portfolio_decision": portfolio_decision_payload,
        "bayesian_records": [
            record.to_dict() if hasattr(record, "to_dict") else dict(record)
            for record in bayesian_records
        ],
        "funnel_summary": dict(orchestration.get("funnel_summary", {}) or {}),
        "pipeline_mode": str(getattr(result, "pipeline_mode", "legacy")),
        "bayesian_shortlist_symbols": list(
            getattr(result, "bayesian_shortlist_symbols", []) or []
        ),
        "bayesian_record_count": len(bayesian_records),
        "funnel_candidates_count": (
            len(getattr(funnel_output, "candidates", []) or [])
            if funnel_output is not None
            else 0
        ),
        "funnel_excluded_count": (
            len(getattr(funnel_output, "excluded_symbols", []) or [])
            if funnel_output is not None
            else 0
        ),
    }


def get_us_stock_name(symbol: str) -> str:
    return get_stock_name(symbol, market="US")


def _parquet_data_root_from_market_dir(
    base_dir: Path,
    *,
    market: str = "CN",
    allow_default: bool = True,
) -> Path | None:
    market_key = str(market or "").strip().lower()
    candidates = [
        base_dir,
        base_dir.parent,
        base_dir.parent.parent,
        base_dir.parent.parent.parent,
    ]
    for candidate in candidates:
        if (candidate / "parquet" / market_key).exists():
            return candidate
        if (candidate / "parquet_serving" / market_key).exists():
            return candidate
    return Path("data") if allow_default else None


def _list_parquet_symbols(category: str, *, market: str, data_root: Path | None) -> list[str]:
    if data_root is None:
        return []
    try:
        symbols = MarketDataReader(
            market=market,
            data_root=data_root,
        ).list_symbols(category)
        if str(market or "").strip().upper() == "US":
            symbols, _metadata = USMarketCapFilter().filter_symbols(symbols, fetch_missing=False)
        return symbols
    except Exception:
        return []


def get_all_local_symbols(
    category: str,
    market: str = "CN",
    data_dir: str | None = None,
) -> list[str]:
    settings = get_market_settings(market)
    base_dir = Path(data_dir or settings.data_dir)
    if settings.market == "CN":
        return _list_parquet_symbols(
            category,
            market="CN",
            data_root=_parquet_data_root_from_market_dir(
                base_dir,
                market="CN",
                allow_default=data_dir is None,
            ),
        )
    if settings.market == "US":
        return _list_parquet_symbols(
            category,
            market="US",
            data_root=_parquet_data_root_from_market_dir(
                base_dir,
                market="US",
                allow_default=data_dir is None,
            ),
        )
    return []


def analyze_batch(
    symbols: list[str],
    category: str,
    batch_id: int,
    market: str = "CN",
    universe: str = "full_a",
    total_capital: float = 1_000_000,
    risk_level: str = "中等",
    verbose: bool = True,
    analysis_kwargs: dict[str, Any] | None = None,
) -> Optional[dict[str, Any]]:
    settings = get_market_settings(market)
    scoped_category_name = category_name(category, settings.market)
    analyzer_kwargs = dict(analysis_kwargs or {})
    analyzer_kwargs.setdefault("enable_kline", False)
    analyzer_kwargs.setdefault("enable_kronos", False)
    analyzer_kwargs.setdefault("enable_macro", True)
    analyzer_kwargs.setdefault("enable_quant", True)
    analyzer_kwargs.setdefault("enable_fundamental", True)
    analyzer_kwargs.setdefault("enable_intelligence", True)
    analyzer_kwargs.setdefault("verbose", verbose)
    analyzer_kwargs.setdefault("master_reasoning_effort", "high")
    analyzer_kwargs.setdefault("universe_key", universe)

    print(f"\n{'=' * 80}")
    print(f"📊 分析 {scoped_category_name} - 批次 {batch_id}")
    print(f"{'=' * 80}")
    print(f"本批股票数: {len(symbols)}")
    print(f"前10只: {symbols[:10]}")

    try:
        analyzer = QuantInvestor(
            stock_pool=symbols,
            market=settings.market,
            total_capital=total_capital,
            risk_level=risk_level,
            **analyzer_kwargs,
        )
        result = analyzer.run()

        recommendations = []
        for recommendation in result.final_strategy.trade_recommendations:
            payload = asdict(recommendation)
            payload["category"] = category
            payload["category_name"] = scoped_category_name
            payload["one_line_conclusion"] = (
                payload.get("one_line_conclusion")
                or _derive_stock_conclusion(payload)
            )
            payload["support_drivers"] = (
                payload.get("support_drivers")
                or _derive_stock_support_drivers(payload)
            )
            payload["drag_drivers"] = (
                payload.get("drag_drivers")
                or _derive_stock_drag_drivers(payload)
            )
            payload["weight_cap_reasons"] = (
                payload.get("weight_cap_reasons")
                or [
                    "组合目标总仓位 "
                    f"{result.final_strategy.target_exposure:.0%}，"
                    "单票按风险上限约束。"
                ]
            )
            macro_branch = result.branch_results.get("macro")
            payload["macro_score"] = (
                float(macro_branch.score) if macro_branch else 0.0
            )
            recommendations.append(payload)

        analysis: dict[str, Any] = {
            "market": settings.market,
            "universe": universe,
            "category": category,
            "category_name": scoped_category_name,
            "batch_id": batch_id,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "stocks": symbols,
            "stock_count": len(symbols),
            "branches": {},
            "strategy": {
                "target_exposure": result.final_strategy.target_exposure,
                "style_bias": result.final_strategy.style_bias,
                "candidate_symbols": result.final_strategy.candidate_symbols,
                "position_limits": result.final_strategy.position_limits,
                "branch_consensus": result.final_strategy.branch_consensus,
                "risk_summary": result.final_strategy.risk_summary,
                "execution_notes": result.final_strategy.execution_notes,
                "research_mode": result.final_strategy.research_mode,
            },
            "recommendations": recommendations,
            "execution_log": list(getattr(result, "execution_log", [])),
        }
        analysis["analysis_meta"] = _analysis_meta_from_result(
            result,
            market=settings.market,
            universe=universe,
            category=category,
            batch_id=batch_id,
            symbols=symbols,
            analysis_kwargs=analyzer_kwargs,
        )

        for name, branch in result.branch_results.items():
            analysis["branches"][name] = {
                "score": branch.score,
                "confidence": branch.confidence,
                "conclusion": branch.conclusion,
                "support_drivers": list(branch.support_drivers),
                "drag_drivers": list(branch.drag_drivers),
                "investment_risks": list(branch.investment_risks),
                "coverage_notes": list(branch.coverage_notes),
                "diagnostic_notes": list(branch.diagnostic_notes),
                "module_coverage": dict(branch.module_coverage),
                "debate_status": str(
                    branch.metadata.get("debate_status", "skipped")
                ),
                "top_symbols": [
                    {"symbol": symbol, "score": score}
                    for symbol, score in sorted(
                        branch.symbol_scores.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:5]
                ],
            }

        print(f"✅ 批次 {batch_id} 分析完成")
        print(f"   目标仓位: {analysis['strategy']['target_exposure']:.0%}")
        print(f"   候选标的: {len(analysis['strategy']['candidate_symbols'])} 只")
        return analysis
    except Exception as exc:
        print(f"❌ 批次 {batch_id} 分析失败: {exc}")
        import traceback

        traceback.print_exc()
        return None


def save_batch_result(
    result: dict[str, Any],
    market: str = "CN",
    output_dir: str | None = None,
) -> str:
    settings = get_market_settings(market)
    target_dir = Path(output_dir or settings.analysis_output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / (
        f"batch_{result['category']}_"
        f"{int(result['batch_id']):03d}_{result['timestamp']}.json"
    )
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(f"💾 批次结果已保存: {output_path}")
    return str(output_path)


def analyze_category_full(
    category: str,
    market: str = "CN",
    universe: str | None = None,
    batch_size: Optional[int] = None,
    data_dir: str | None = None,
    output_dir: str | None = None,
    total_capital: float = 1_000_000,
    risk_level: str = "中等",
    verbose: bool = True,
    analysis_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    settings = get_market_settings(market)
    scoped_batch_size = batch_size or settings.default_batch_size
    symbols = get_all_local_symbols(
        category,
        market=settings.market,
        data_dir=data_dir,
    )
    total = len(symbols)

    print(f"\n{'=' * 80}")
    print(f"🚀 开始全量分析 {category_name(category, settings.market)}")
    print(f"{'=' * 80}")

    if total == 0:
        print(f"❌ 没有找到 {category} 的数据")
        return []

    print(f"总计 {total} 只股票需要分析")
    print(f"批次大小: {scoped_batch_size} 只")
    print(f"预计批次: {(total + scoped_batch_size - 1) // scoped_batch_size} 批")
    print(f"预计时间: {total * 2 / 60:.1f} 分钟")
    print(f"{'=' * 80}")

    all_results: list[dict[str, Any]] = []
    num_batches = (total + scoped_batch_size - 1) // scoped_batch_size
    for index in range(num_batches):
        start_idx = index * scoped_batch_size
        end_idx = min(start_idx + scoped_batch_size, total)
        batch_symbols = symbols[start_idx:end_idx]
        print(
            f"\n⏳ 进度: 批次 {index + 1}/{num_batches} "
            f"({start_idx + 1}-{end_idx}/{total})"
        )
        result = analyze_batch(
            batch_symbols,
            category,
            index + 1,
            market=settings.market,
            universe=universe or category,
            total_capital=total_capital,
            risk_level=risk_level,
            verbose=verbose,
            analysis_kwargs=analysis_kwargs,
        )
        if result:
            all_results.append(result)
            save_batch_result(
                result,
                market=settings.market,
                output_dir=output_dir,
            )
    return all_results


__all__ = [
    "analyze_batch",
    "analyze_category_full",
    "get_all_local_symbols",
    "get_us_stock_name",
    "save_batch_result",
]
