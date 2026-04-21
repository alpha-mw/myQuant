from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import pandas as pd

from quant_investor.agent_protocol import BranchVerdict, DataQualityIssue, GlobalContext
from quant_investor.branch_contracts import BranchResult, UnifiedDataBundle
from quant_investor.config import config
from quant_investor.funnel.deterministic_funnel import FunnelConfig, FunnelOutput
from quant_investor.market.config import get_market_settings
from quant_investor.market.dag.packets import (
    _clamp,
    _build_cross_section_quant,
    _build_global_quant_verdict,
    _build_market_snapshot,
    _build_quant_branch_result,
    _build_symbol_tradability,
    _frame_summary,
)
from quant_investor.market.data_quality import build_data_quality_diagnostics
from quant_investor.market.shared_csv_reader import SharedCSVReadResult, SharedCSVReader
from quant_investor.llm_gateway import detect_provider
from quant_investor.model_roles import ModelRoleResolution
from quant_investor.reporting.run_artifacts import build_model_role_metadata


@dataclass
class MarketContextState:
    all_symbols: list[str]
    read_results: dict[str, SharedCSVReadResult]
    frames: dict[str, pd.DataFrame]
    tradability_snapshot: dict[str, dict[str, Any]]
    data_quality_issues: list[DataQualityIssue]
    quarantined_symbols: list[str]
    researchable_symbols: list[str]
    candidate_symbols: list[str]
    provider_health: dict[str, dict[str, Any]]
    market_snapshot: dict[str, Any]
    macro_verdict: BranchVerdict
    global_quant_verdict: BranchVerdict
    quant_result: BranchResult
    global_context: GlobalContext
    model_roles: Any
    full_market_kline_result: BranchResult
    funnel_output: FunnelOutput
    resolver_snapshot: dict[str, Any] = field(default_factory=dict)


def _is_quarantined_read_result(read_result: Any) -> bool:
    issues = list(getattr(read_result, "issues", []) or [])
    return bool(issues)


def _prepare_market_context(
    *,
    market: str,
    universe_key: str,
    selected_categories: list[str],
    symbols: list[str],
    company_profile_map: Mapping[str, Mapping[str, Any]],
    shared_reader: SharedCSVReader,
    scoped_data_snapshot: Mapping[str, Any],
    download_stage: Mapping[str, Any] | None,
    enable_agent_layer: bool,
    agent_timeout: float,
    master_timeout: float,
    master_reasoning_effort: str,
    branch_model_resolution: ModelRoleResolution,
    master_model_resolution: ModelRoleResolution,
    branch_candidate_models: list[str],
    master_candidate_models: list[str],
    company_name_map: Mapping[str, str],
    funnel_profile: str,
    max_candidates: int,
    trend_windows: tuple[int, ...],
    volume_spike_threshold: float,
    breakout_distance_pct: float,
    sector_bucket_limit: int,
    macro_agent: Any,
    kline_agent: Any,
    funnel_cls: Any,
    provider_health_detector: Callable[..., dict[str, dict[str, Any]]],
    ensure_branch_verdict: Callable[..., BranchVerdict],
    branch_verdict_to_result: Callable[[BranchVerdict, str], BranchResult],
) -> MarketContextState:
    settings = get_market_settings(market)
    all_symbols = list(symbols)
    resolver_snapshot = shared_reader.snapshot()

    read_results: dict[str, SharedCSVReadResult] = {}
    frames: dict[str, pd.DataFrame] = {}
    tradability_snapshot: dict[str, dict[str, Any]] = {}
    data_quality_issues: list[DataQualityIssue] = []
    quarantined_symbols: list[str] = []
    researchable_symbols: list[str] = []
    industry_map: dict[str, str] = {}
    symbol_market_state: dict[str, dict[str, Any]] = {}
    for symbol in all_symbols:
        profile = dict(company_profile_map.get(symbol, {}) or {})
        read_result = shared_reader.read_symbol_frame(symbol, universe_key=universe_key)
        read_results[symbol] = read_result
        frames[symbol] = read_result.frame
        tradability_snapshot[symbol] = _build_symbol_tradability(
            symbol,
            read_result,
            company_name=company_name_map.get(symbol, ""),
            sector=str(profile.get("sector", "") or profile.get("industry", "")),
            industry=str(profile.get("industry", "") or profile.get("sector", "")),
            trend_windows=trend_windows,
            volume_spike_threshold=volume_spike_threshold,
            breakout_distance_pct=breakout_distance_pct,
        )
        symbol_market_state[symbol] = dict(tradability_snapshot[symbol].get("market_state", {}) or {})
        industry_label = str(tradability_snapshot[symbol].get("industry") or tradability_snapshot[symbol].get("sector") or "").strip()
        if industry_label:
            industry_map[symbol] = industry_label
        data_quality_issues.extend(read_result.issues)
        if _is_quarantined_read_result(read_result):
            quarantined_symbols.append(symbol)
        else:
            researchable_symbols.append(symbol)

    symbols = list(researchable_symbols)

    cross_section_quant = _build_cross_section_quant(frames)
    macro_overview = {
        "regime": "neutral",
        "macro_score": cross_section_quant.get("average_return", 0.0),
        "liquidity_score": cross_section_quant.get("breadth", 0.0),
        "volatility_percentile": min(95.0, max(5.0, cross_section_quant.get("average_volatility", 0.0) * 100.0 + 50.0)),
        "policy_signal": "neutral",
    }
    snapshot_latest_trade_date = str(scoped_data_snapshot.get("local_latest_trade_date", ""))
    snapshot_freshness_mode = str(scoped_data_snapshot.get("freshness_mode", "stable"))
    market_snapshot = _build_market_snapshot(
        market=settings.market,
        universe_key=universe_key,
        frames=frames,
        global_summary={"candidate_count": len(symbols)},
        latest_trade_date=(
            download_stage.get("completeness_after", {}).get("latest_trade_date", "")
            if download_stage
            else snapshot_latest_trade_date
        ),
        macro_overview=macro_overview,
    )

    macro_verdict = macro_agent.run({"market_snapshot": market_snapshot})
    macro_overview["regime"] = str(macro_verdict.metadata.get("regime", "neutral"))
    macro_overview["macro_score"] = float(macro_verdict.final_score)
    macro_overview["liquidity_score"] = float(cross_section_quant.get("breadth", 0.0))
    market_snapshot.update(macro_overview)
    global_quant_verdict = _build_global_quant_verdict(
        cross_section_quant=cross_section_quant,
        symbol_count=len(symbols),
    )
    quant_result = _build_quant_branch_result(frames=frames)
    liquidity_scores = {
        symbol: float(
            max(
                min(1.0, max(0.0, _frame_summary(frame).get("rows", 0) / 250.0)),
                tradability_snapshot.get(symbol, {}).get("liquidity_score", 0.0),
            )
        )
        for symbol, frame in frames.items()
    }
    illiquid_symbols = [symbol for symbol, score in liquidity_scores.items() if score < 0.10]
    sector_strengths: dict[str, float] = {}
    sector_members: dict[str, list[float]] = {}
    for symbol, info in tradability_snapshot.items():
        sector = str(info.get("industry") or info.get("sector") or "").strip()
        if not sector or sector == "unknown":
            continue
        sector_members.setdefault(sector, []).append(float(info.get("momentum_strength", 0.0)))
    if sector_members:
        sector_avgs = {
            sector: sum(values) / max(len(values), 1)
            for sector, values in sector_members.items()
        }
        ordered = sorted(sector_avgs.items(), key=lambda item: (-item[1], item[0]))
        total = max(len(ordered) - 1, 1)
        for rank, (sector, score) in enumerate(ordered):
            percentile = 1.0 if len(ordered) == 1 else 1.0 - (rank / total)
            sector_strengths[sector] = _clamp(0.55 * percentile + 0.45 * float(score), 0.0, 1.0)

    style_exposures: dict[str, Any] = {
        "style_bias": macro_verdict.metadata.get("style_bias", "balanced"),
        "default": 0.50,
    }
    for symbol, info in tradability_snapshot.items():
        sector = str(info.get("industry") or info.get("sector") or "unknown")
        sector_strength = float(sector_strengths.get(sector, 0.50))
        momentum_strength = float(info.get("momentum_strength", 0.0))
        style_exposures[symbol] = {
            "prior": _clamp(0.35 + 0.35 * sector_strength + 0.30 * momentum_strength, 0.15, 0.90),
            "sector": str(info.get("sector") or sector),
            "industry": sector,
            "momentum_strength": momentum_strength,
        }

    completeness_payload = {}
    if download_stage:
        completeness_payload = dict(
            download_stage.get("completeness_after")
            or download_stage.get("completeness_before")
            or {}
        )
    effective_latest_trade_date = str(
        completeness_payload.get("latest_trade_date")
        or snapshot_latest_trade_date
    )
    effective_freshness_mode = str(
        completeness_payload.get("freshness_mode")
        or snapshot_freshness_mode
        or "stable"
    )
    target_exposure = float(macro_verdict.metadata.get("target_gross_exposure", 0.5))
    max_single_weight = 0.12
    if str(funnel_profile or "").strip().lower() == "momentum_leader":
        breadth = float(cross_section_quant.get("breadth", 0.0))
        weak_regime = str(macro_verdict.metadata.get("regime", "neutral")) in {"趋势下跌", "震荡高波"}
        if weak_regime or float(macro_verdict.final_score) < 0.0 or breadth < 0.48:
            target_exposure = min(target_exposure, 0.45) * 0.75
            max_single_weight = 0.10
        elif str(macro_verdict.metadata.get("regime", "neutral")) == "趋势上涨" and breadth > 0.55:
            target_exposure = min(target_exposure * 1.08, 0.72)
            max_single_weight = 0.14

    global_context = GlobalContext(
        market=settings.market,
        universe_key=universe_key,
        rebalance_date=effective_latest_trade_date,
        latest_trade_date=effective_latest_trade_date,
        universe_symbols=list(all_symbols),
        universe_hash="",
        industry_map=industry_map,
        liquidity_filter={
            "candidate_count": len(all_symbols),
            "researchable_count": len(symbols),
            "quarantined_count": len(quarantined_symbols),
            "category_count": len(selected_categories),
            "suspended": list(quarantined_symbols),
            "illiquid": list(illiquid_symbols),
            "liquidity_scores": liquidity_scores,
            "sector_bucket_limit": int(sector_bucket_limit),
        },
        macro_regime=str(macro_verdict.metadata.get("regime", "neutral")),
        cross_section_quant={**cross_section_quant, "macro_score": float(macro_verdict.final_score)},
        style_exposures=style_exposures,
        correlation_matrix={},
        risk_budget={
            "target_exposure": target_exposure,
            "max_single_weight": max_single_weight,
            "sector_bucket_limit": int(sector_bucket_limit),
        },
        data_quality_issues=data_quality_issues,
        data_quality_diagnostics=build_data_quality_diagnostics(
            total_symbols=all_symbols,
            researchable_symbols=researchable_symbols,
            shortlistable_symbols=[],
            final_selected_symbols=[],
            quarantined_symbols=quarantined_symbols,
            issues=data_quality_issues,
        ),
        model_capability_map=provider_health_detector(
            agent_model=branch_model_resolution.primary_model,
            master_model=master_model_resolution.primary_model,
        ),
        symbol_name_map=dict(company_name_map),
        data_quality_quarantine=list(quarantined_symbols),
        freshness_mode=effective_freshness_mode,
        effective_target_trade_date=str(
            completeness_payload.get("effective_target_trade_date")
            or effective_latest_trade_date
        ),
        universe_tiers={
            "total": list(all_symbols),
            "researchable": list(researchable_symbols),
            "shortlistable": [],
            "final_selected": [],
        },
        metadata={
            "resolver": resolver_snapshot,
            "resolver_directory_priority": list((resolver_snapshot or {}).get("directory_priority", [])),
            "physical_directories_used_for_full_a": list((resolver_snapshot or {}).get("physical_directories_used_for_full_a", [])),
            "data_quality_issue_count": len(data_quality_issues),
            "candidate_count": len(all_symbols),
            "researchable_count": len(symbols),
            "quarantined_count": len(quarantined_symbols),
            "quarantined_symbols": list(quarantined_symbols[:32]),
            "global_quant_verdict": global_quant_verdict.to_dict(),
            "provider_health": {},
            "data_snapshot": dict(scoped_data_snapshot),
            "symbol_market_state": symbol_market_state,
            "selection_profile": {
                "funnel_profile": str(funnel_profile or "classic").strip().lower() or "classic",
                "trend_windows": list(trend_windows),
                "volume_spike_threshold": float(volume_spike_threshold),
                "breakout_distance_pct": float(breakout_distance_pct),
                "max_candidates": int(max_candidates),
                "sector_bucket_limit": int(sector_bucket_limit),
            },
        },
    )
    provider_health = provider_health_detector(
        agent_model=branch_model_resolution.primary_model,
        master_model=master_model_resolution.primary_model,
    )
    global_context.model_capability_map = provider_health
    global_context.metadata["provider_health"] = provider_health
    if symbols:
        import hashlib

        global_context.universe_hash = hashlib.sha256(",".join(sorted(symbols)).encode("utf-8")).hexdigest()[:16]

    model_roles = build_model_role_metadata(
        branch_model=branch_model_resolution.primary_model,
        master_model=master_model_resolution.primary_model,
        agent_fallback_model=branch_model_resolution.fallback_model,
        master_fallback_model=master_model_resolution.fallback_model,
        resolved_branch_model=branch_model_resolution.resolved_model,
        resolved_master_model=master_model_resolution.resolved_model,
        master_reasoning_effort=master_reasoning_effort,
        branch_provider=detect_provider(branch_model_resolution.resolved_model),
        master_provider=detect_provider(master_model_resolution.resolved_model),
        branch_timeout=agent_timeout,
        master_timeout=master_timeout,
        agent_layer_enabled=bool(enable_agent_layer),
        branch_fallback_used=bool(branch_model_resolution.fallback_used),
        master_fallback_used=bool(master_model_resolution.fallback_used),
        branch_fallback_reason=str(branch_model_resolution.fallback_reason),
        master_fallback_reason=str(master_model_resolution.fallback_reason),
        universe_key=universe_key,
        universe_size=len(symbols),
        universe_hash=global_context.universe_hash,
        metadata={
            "resolver": resolver_snapshot,
            "data_quality_issue_count": len(data_quality_issues),
            "agent_layer_enabled": bool(enable_agent_layer),
            "provider_health": provider_health,
            "ordered_review_models": {
                "branch": list(branch_candidate_models),
                "master": list(master_candidate_models),
            },
        },
    )

    full_market_bundle = UnifiedDataBundle(
        market=settings.market,
        symbols=list(symbols),
        symbol_data={symbol: frames.get(symbol, pd.DataFrame()) for symbol in symbols},
        fundamentals={},
        event_data={},
        sentiment_data={},
        macro_data=dict(market_snapshot),
        metadata={
            "symbol_provenance": {
                symbol: {
                    "path": read_results[symbol].path,
                    "resolver_trace": read_results[symbol].resolver_trace,
                    "data_quality_issues": [issue.to_dict() for issue in read_results[symbol].issues],
                }
                for symbol in symbols
            }
        },
    )
    full_market_kline_verdict = ensure_branch_verdict(
        kline_agent.run(
            {
                "data_bundle": full_market_bundle,
                "stock_pool": list(symbols),
                "market": settings.market,
                "verbose": False,
                "mode": "full_market",
            }
        ),
        symbol="__market__",
        branch_name="kline",
    )
    full_market_kline_result = branch_verdict_to_result(full_market_kline_verdict, "kline")

    funnel = funnel_cls(
        FunnelConfig(
            max_candidates=int(max_candidates or getattr(config, "FUNNEL_MAX_CANDIDATES", 200) or 200),
            profile=str(funnel_profile or "classic").strip().lower() or "classic",
            trend_windows=tuple(int(item) for item in trend_windows if int(item) > 0) or tuple(getattr(config, "FUNNEL_TREND_WINDOWS", (20, 60, 120))),
            volume_spike_threshold=float(volume_spike_threshold),
            breakout_distance_pct=float(breakout_distance_pct),
            sector_bucket_limit=int(sector_bucket_limit if str(funnel_profile or "").strip().lower() == "momentum_leader" else 0),
        )
    )
    funnel_output = funnel.run(
        quant_result=quant_result,
        kline_result=full_market_kline_result,
        global_context=global_context,
    )
    candidate_symbols = [symbol for symbol in funnel_output.candidates if symbol in researchable_symbols]
    if not candidate_symbols:
        candidate_symbols = list(researchable_symbols)
    global_context.universe_tiers = {
        "total": list(all_symbols),
        "researchable": list(researchable_symbols),
        "shortlistable": list(candidate_symbols),
        "final_selected": [],
    }
    global_context.data_quality_diagnostics = build_data_quality_diagnostics(
        total_symbols=all_symbols,
        researchable_symbols=researchable_symbols,
        shortlistable_symbols=candidate_symbols,
        final_selected_symbols=[],
        quarantined_symbols=quarantined_symbols,
        issues=data_quality_issues,
    )
    global_context.metadata["candidate_count"] = len(candidate_symbols)
    global_context.metadata["shortlistable_count"] = len(candidate_symbols)
    candidate_sector_counts: dict[str, int] = {}
    for symbol in candidate_symbols:
        sector = str(industry_map.get(symbol) or tradability_snapshot.get(symbol, {}).get("industry") or tradability_snapshot.get(symbol, {}).get("sector") or "").strip()
        if not sector or sector == "unknown":
            continue
        candidate_sector_counts[sector] = candidate_sector_counts.get(sector, 0) + 1
    global_context.metadata["candidate_sector_counts"] = candidate_sector_counts

    return MarketContextState(
        all_symbols=all_symbols,
        read_results=read_results,
        frames=frames,
        tradability_snapshot=tradability_snapshot,
        data_quality_issues=data_quality_issues,
        quarantined_symbols=quarantined_symbols,
        researchable_symbols=researchable_symbols,
        candidate_symbols=candidate_symbols,
        provider_health=provider_health,
        market_snapshot=market_snapshot,
        macro_verdict=macro_verdict,
        global_quant_verdict=global_quant_verdict,
        quant_result=quant_result,
        global_context=global_context,
        model_roles=model_roles,
        full_market_kline_result=full_market_kline_result,
        funnel_output=funnel_output,
        resolver_snapshot=resolver_snapshot,
    )
