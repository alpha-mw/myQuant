from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from inspect import Parameter, signature
from typing import Any, Callable, Mapping

import pandas as pd

from quant_investor.agent_protocol import BranchVerdict, DataQualityIssue, GlobalContext
from quant_investor.branch_contracts import BranchResult
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
)
from quant_investor.market.dag.theme_context import (
    build_disabled_theme_rotation_metadata,
    build_theme_rotation_metadata,
    persist_theme_rotation_snapshot,
)
from quant_investor.market.data_quality import build_data_quality_diagnostics
from quant_investor.market.branch_readiness import (
    STATUS_BLOCK,
    assess_branch_data_readiness,
    write_branch_readiness_report,
)
from quant_investor.market.read_result import MarketDataReadResult
from quant_investor.market.runtime_profile import profile_stage
from quant_investor.llm_gateway import detect_provider
from quant_investor.model_roles import ModelRoleResolution
from quant_investor.reporting.run_artifacts import build_model_role_metadata


DAG_RUNTIME_PRICE_VOLUME_COLUMNS: tuple[str, ...] = (
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
    "adj_close",
)
DAG_RUNTIME_LOOKBACK_CALENDAR_DAYS = 420


@dataclass
class MarketContextState:
    all_symbols: list[str]
    read_results: dict[str, MarketDataReadResult]
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
    funnel_output: FunnelOutput
    resolver_snapshot: dict[str, Any] = field(default_factory=dict)
    branch_data_readiness: dict[str, Any] = field(default_factory=dict)
    branch_data_payload: dict[str, Any] = field(default_factory=dict)


def _is_quarantined_read_result(read_result: Any) -> bool:
    issues = list(getattr(read_result, "issues", []) or [])
    return bool(issues)


def _provider_label(resolution: ModelRoleResolution) -> str:
    metadata = dict(resolution.metadata or {})
    if (
        metadata.get("review_layer_mode") == "codex_handoff"
        or resolution.resolved_model == "codex-handoff"
    ):
        return "codex"
    try:
        return detect_provider(resolution.resolved_model)
    except Exception:
        return ""


def _compact_runtime_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return ""
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _runtime_lookback_start_date(
    latest_trade_date: Any,
    *,
    calendar_days: int = DAG_RUNTIME_LOOKBACK_CALENDAR_DAYS,
) -> str:
    compact = _compact_runtime_date(latest_trade_date)
    if not compact:
        return ""
    parsed = pd.to_datetime(compact, format="%Y%m%d", errors="coerce")
    if pd.isna(parsed):
        return ""
    return (pd.Timestamp(parsed) - timedelta(days=max(int(calendar_days), 1))).strftime("%Y%m%d")


def _call_accepts_keyword(callable_obj: Callable[..., Any], keyword: str) -> bool:
    try:
        parameters = signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(
        parameter.name == keyword or parameter.kind == Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _read_symbol_frames_with_projection(
    batch_reader: Callable[..., Any],
    symbols: list[str],
    *,
    universe_key: str,
    start_date: str = "",
) -> dict[str, MarketDataReadResult]:
    kwargs: dict[str, Any] = {"universe_key": universe_key}
    if _call_accepts_keyword(batch_reader, "columns"):
        kwargs["columns"] = DAG_RUNTIME_PRICE_VOLUME_COLUMNS
    if start_date and _call_accepts_keyword(batch_reader, "start_date"):
        kwargs["start_date"] = start_date
    return dict(batch_reader(symbols, **kwargs) or {})


def _prepare_market_context(
    *,
    market: str,
    universe_key: str,
    selected_categories: list[str],
    symbols: list[str],
    company_profile_map: Mapping[str, Mapping[str, Any]],
    shared_reader: Any,
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
    funnel_cls: Any,
    provider_health_detector: Callable[..., dict[str, dict[str, Any]]],
    runtime_profiler: Any | None = None,
) -> MarketContextState:
    settings = get_market_settings(market)
    all_symbols = list(symbols)
    resolver_snapshot = shared_reader.snapshot()

    read_results: dict[str, MarketDataReadResult] = {}
    frames: dict[str, pd.DataFrame] = {}
    tradability_snapshot: dict[str, dict[str, Any]] = {}
    data_quality_issues: list[DataQualityIssue] = []
    quarantined_symbols: list[str] = []
    researchable_symbols: list[str] = []
    industry_map: dict[str, str] = {}
    symbol_market_state: dict[str, dict[str, Any]] = {}
    batch_read_results: dict[str, MarketDataReadResult] = {}
    raw_read_results: dict[str, MarketDataReadResult] = {}
    frame_summaries: dict[str, dict[str, Any]] = {}
    runtime_lookback_start_date = _runtime_lookback_start_date(
        scoped_data_snapshot.get("local_latest_trade_date")
        or scoped_data_snapshot.get("latest_trade_date")
    )
    with profile_stage(
        runtime_profiler,
        "dag_batch_read",
        {"symbol_count": len(all_symbols), "universe_key": universe_key},
    ) as stage_metadata:
        stage_metadata["projected_columns"] = list(DAG_RUNTIME_PRICE_VOLUME_COLUMNS)
        stage_metadata["projected_column_count"] = len(DAG_RUNTIME_PRICE_VOLUME_COLUMNS)
        stage_metadata["runtime_lookback_calendar_days"] = DAG_RUNTIME_LOOKBACK_CALENDAR_DAYS
        if runtime_lookback_start_date:
            stage_metadata["runtime_lookback_start_date"] = runtime_lookback_start_date
        batch_reader = getattr(shared_reader, "read_symbol_frames", None)
        if callable(batch_reader):
            batch_read_results = _read_symbol_frames_with_projection(
                batch_reader,
                all_symbols,
                universe_key=universe_key,
                start_date=runtime_lookback_start_date,
            )
        per_symbol_fallback_count = 0
        for symbol in all_symbols:
            read_result = batch_read_results.get(symbol)
            if read_result is None:
                per_symbol_fallback_count += 1
                read_result = shared_reader.read_symbol_frame(symbol, universe_key=universe_key)
            raw_read_results[symbol] = read_result
        stage_metadata["batch_result_count"] = len(batch_read_results)
        stage_metadata["per_symbol_fallback_count"] = per_symbol_fallback_count
    with profile_stage(
        runtime_profiler,
        "dag_tradability_snapshot",
        {"symbol_count": len(all_symbols), "universe_key": universe_key},
    ) as stage_metadata:
        for symbol in all_symbols:
            profile = dict(company_profile_map.get(symbol, {}) or {})
            read_result = raw_read_results[symbol]
            read_results[symbol] = read_result
            frames[symbol] = read_result.frame
            tradability = _build_symbol_tradability(
                symbol,
                read_result,
                company_name=company_name_map.get(symbol, ""),
                sector=str(profile.get("sector", "") or profile.get("industry", "")),
                industry=str(profile.get("industry", "") or profile.get("sector", "")),
                trend_windows=trend_windows,
                volume_spike_threshold=volume_spike_threshold,
                breakout_distance_pct=breakout_distance_pct,
            )
            tradability_snapshot[symbol] = tradability
            market_state = dict(tradability.get("market_state", {}) or {})
            frame_summaries[symbol] = {
                "rows": int(market_state.get("rows", 0) or 0),
                "latest_close": float(market_state.get("latest_close", 0.0) or 0.0),
                "average_return": float(market_state.get("average_return", 0.0) or 0.0),
                "volatility": float(market_state.get("volatility", 0.0) or 0.0),
            }
            symbol_market_state[symbol] = dict(tradability_snapshot[symbol].get("market_state", {}) or {})
            industry_label = str(tradability_snapshot[symbol].get("industry") or tradability_snapshot[symbol].get("sector") or "").strip()
            if industry_label:
                industry_map[symbol] = industry_label
            data_quality_issues.extend(read_result.issues)
            if _is_quarantined_read_result(read_result):
                quarantined_symbols.append(symbol)
            else:
                researchable_symbols.append(symbol)
        stage_metadata["researchable_count"] = len(researchable_symbols)
        stage_metadata["quarantined_count"] = len(quarantined_symbols)
        stage_metadata["issue_count"] = len(data_quality_issues)

    symbols = list(researchable_symbols)

    with profile_stage(
        runtime_profiler,
        "dag_quant_context",
        {"researchable_count": len(symbols), "universe_key": universe_key},
    ) as stage_metadata:
        with profile_stage(
            runtime_profiler,
            "dag_cross_section_quant",
            {"researchable_count": len(symbols), "frame_count": len(frames)},
        ) as cross_section_metadata:
            cross_section_quant = _build_cross_section_quant(
                frames,
                frame_summaries=frame_summaries,
            )
            cross_section_metadata["breadth"] = float(cross_section_quant.get("breadth", 0.0))
            cross_section_metadata["average_return"] = float(
                cross_section_quant.get("average_return", 0.0)
            )
            cross_section_metadata["average_volatility"] = float(
                cross_section_quant.get("average_volatility", 0.0)
            )
        macro_overview = {
            "regime": "neutral",
            "macro_score": cross_section_quant.get("average_return", 0.0),
            "liquidity_score": cross_section_quant.get("breadth", 0.0),
            "volatility_percentile": min(95.0, max(5.0, cross_section_quant.get("average_volatility", 0.0) * 100.0 + 50.0)),
            "policy_signal": "neutral",
        }
        snapshot_latest_trade_date = str(scoped_data_snapshot.get("local_latest_trade_date", ""))
        snapshot_freshness_mode = str(scoped_data_snapshot.get("freshness_mode", "stable"))
        effective_snapshot_trade_date = (
            download_stage.get("completeness_after", {}).get("latest_trade_date", "")
            if download_stage
            else snapshot_latest_trade_date
        )
        with profile_stage(
            runtime_profiler,
            "dag_market_snapshot",
            {
                "researchable_count": len(symbols),
                "latest_trade_date": effective_snapshot_trade_date,
                "universe_key": universe_key,
            },
        ) as market_snapshot_metadata:
            market_snapshot = _build_market_snapshot(
                market=settings.market,
                universe_key=universe_key,
                frames=frames,
                global_summary={"candidate_count": len(symbols)},
                latest_trade_date=effective_snapshot_trade_date,
                macro_overview=macro_overview,
                frame_summaries=frame_summaries,
            )
            market_snapshot_metadata["snapshot_key_count"] = len(market_snapshot)

        with profile_stage(
            runtime_profiler,
            "dag_macro_verdict",
            {"market": settings.market, "universe_key": universe_key},
        ) as macro_metadata:
            macro_verdict = macro_agent.run({"market_snapshot": market_snapshot})
            macro_metadata["macro_regime"] = str(
                macro_verdict.metadata.get("regime", "neutral")
            )
            macro_metadata["macro_score"] = float(macro_verdict.final_score)
        macro_overview["regime"] = str(macro_verdict.metadata.get("regime", "neutral"))
        macro_overview["macro_score"] = float(macro_verdict.final_score)
        macro_overview["liquidity_score"] = float(cross_section_quant.get("breadth", 0.0))
        market_snapshot.update(macro_overview)
        with profile_stage(
            runtime_profiler,
            "dag_global_quant_verdict",
            {"researchable_count": len(symbols), "universe_key": universe_key},
        ) as global_quant_metadata:
            global_quant_verdict = _build_global_quant_verdict(
                cross_section_quant=cross_section_quant,
                symbol_count=len(symbols),
            )
            global_quant_metadata["global_quant_score"] = float(global_quant_verdict.final_score)
            global_quant_metadata["global_quant_confidence"] = float(
                global_quant_verdict.final_confidence
            )
        with profile_stage(
            runtime_profiler,
            "dag_quant_branch_result",
            {"researchable_count": len(symbols), "frame_count": len(frames)},
        ) as quant_branch_metadata:
            quant_result = _build_quant_branch_result(
                frames=frames,
                frame_summaries=frame_summaries,
            )
            quant_branch_metadata["scored_symbol_count"] = len(quant_result.symbol_scores)
        stage_metadata["macro_regime"] = str(macro_verdict.metadata.get("regime", "neutral"))
        stage_metadata["breadth"] = float(cross_section_quant.get("breadth", 0.0))
    liquidity_scores = {
        symbol: float(
            max(
                min(
                    1.0,
                    max(
                        0.0,
                        float(frame_summaries.get(symbol, {}).get("rows", 0) or 0)
                        / 250.0,
                    ),
                ),
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

    if bool(getattr(config, "THEME_SCANNER_ENABLED", False)):
        theme_rotation_metadata = build_theme_rotation_metadata(
            frames=frames,
            industry_map=industry_map,
            symbol_market_state=symbol_market_state,
            market=settings.market,
            universe_key=universe_key,
            as_of=effective_latest_trade_date,
            min_member_count=int(getattr(config, "THEME_MIN_MEMBER_COUNT", 5)),
            top_n=int(getattr(config, "THEME_TOP_N", 20)),
            symbol_limit=int(getattr(config, "THEME_METADATA_SYMBOL_LIMIT", 300)),
        )
    else:
        theme_rotation_metadata = build_disabled_theme_rotation_metadata(
            market=settings.market,
            universe_key=universe_key,
            as_of=effective_latest_trade_date,
        )

    provider_health = provider_health_detector(
        agent_model=branch_model_resolution.primary_model,
        master_model=master_model_resolution.primary_model,
    )
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
        model_capability_map=provider_health,
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
            "theme_rotation": theme_rotation_metadata,
            "theme_scores": dict(theme_rotation_metadata.get("theme_scores", {}) or {}),
            "symbol_theme_score": dict(theme_rotation_metadata.get("symbol_scores", {}) or {}),
            "symbol_primary_theme": dict(theme_rotation_metadata.get("symbol_primary_theme", {}) or {}),
            "symbol_theme_phase": dict(theme_rotation_metadata.get("symbol_phase", {}) or {}),
            "theme_alerts": list(theme_rotation_metadata.get("diagnostic_notes", []) or []),
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
    global_context.model_capability_map = provider_health
    global_context.metadata["provider_health"] = provider_health
    if symbols:
        import hashlib

        global_context.universe_hash = hashlib.sha256(",".join(sorted(symbols)).encode("utf-8")).hexdigest()[:16]
    theme_rotation_payload = global_context.metadata.get("theme_rotation", {})
    snapshot_status = persist_theme_rotation_snapshot(
        theme_rotation=theme_rotation_payload if isinstance(theme_rotation_payload, Mapping) else {},
        enabled=bool(getattr(config, "THEME_SNAPSHOT_ENABLED", False)),
        root_dir=str(getattr(config, "THEME_SNAPSHOT_DIR", "results/theme_snapshots")),
        market=settings.market,
        universe_key=universe_key,
        as_of=effective_latest_trade_date,
        run_id=global_context.universe_hash
        or str(scoped_data_snapshot.get("local_latest_trade_date") or ""),
        save_disabled=bool(getattr(config, "THEME_SNAPSHOT_SAVE_DISABLED", False)),
    )
    global_context.metadata["theme_snapshot"] = snapshot_status
    if isinstance(theme_rotation_payload, dict):
        theme_rotation_payload["snapshot_status"] = str(snapshot_status.get("status") or "")
        theme_rotation_payload["snapshot_path"] = (
            str(snapshot_status.get("path") or "")
            if str(snapshot_status.get("status") or "") == "success"
            else ""
        )

    role_metadata = {
        "resolver": resolver_snapshot,
        "data_quality_issue_count": len(data_quality_issues),
        "agent_layer_enabled": bool(enable_agent_layer),
        "provider_health": provider_health,
        "ordered_review_models": {
            "branch": list(branch_candidate_models),
            "master": list(master_candidate_models),
        },
    }
    for key, value in {
        **dict(branch_model_resolution.metadata or {}),
        **dict(master_model_resolution.metadata or {}),
    }.items():
        role_metadata.setdefault(str(key), value)
    role_metadata.setdefault(
        "review_layer_mode",
        "local_llm" if enable_agent_layer else "disabled",
    )

    model_roles = build_model_role_metadata(
        branch_model=branch_model_resolution.primary_model,
        master_model=master_model_resolution.primary_model,
        agent_fallback_model=branch_model_resolution.fallback_model,
        master_fallback_model=master_model_resolution.fallback_model,
        resolved_branch_model=branch_model_resolution.resolved_model,
        resolved_master_model=master_model_resolution.resolved_model,
        master_reasoning_effort=master_reasoning_effort,
        branch_provider=_provider_label(branch_model_resolution),
        master_provider=_provider_label(master_model_resolution),
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
        metadata=role_metadata,
    )

    funnel = funnel_cls(
        FunnelConfig(
            max_candidates=int(max_candidates or getattr(config, "FUNNEL_MAX_CANDIDATES", 500) or 500),
            profile=str(funnel_profile or "classic").strip().lower() or "classic",
            trend_windows=tuple(int(item) for item in trend_windows if int(item) > 0) or tuple(getattr(config, "FUNNEL_TREND_WINDOWS", (20, 60, 120))),
            volume_spike_threshold=float(volume_spike_threshold),
            breakout_distance_pct=float(breakout_distance_pct),
            sector_bucket_limit=int(sector_bucket_limit if str(funnel_profile or "").strip().lower() == "momentum_leader" else 0),
            theme_boost_enabled=bool(getattr(config, "THEME_FUNNEL_BOOST_ENABLED", False)),
            theme_boost_cap=float(getattr(config, "THEME_SYMBOL_BOOST_CAP", 0.10)),
        )
    )
    with profile_stage(
        runtime_profiler,
        "dag_funnel",
        {"researchable_count": len(researchable_symbols), "max_candidates": int(max_candidates)},
    ) as stage_metadata:
        funnel_output = funnel.run(
            quant_result=quant_result,
            global_context=global_context,
        )
        stage_metadata["candidate_count"] = len(getattr(funnel_output, "candidates", []) or [])
        stage_metadata["excluded_count"] = len(getattr(funnel_output, "excluded_symbols", {}) or {})
    candidate_symbols = [symbol for symbol in funnel_output.candidates if symbol in researchable_symbols]
    if not candidate_symbols:
        candidate_symbols = list(researchable_symbols)
    with profile_stage(
        runtime_profiler,
        "dag_branch_readiness",
        {"candidate_count": len(candidate_symbols), "universe_key": universe_key},
    ) as stage_metadata:
        branch_governance_report = assess_branch_data_readiness(
            frames=frames,
            read_results=read_results,
            candidate_symbols=candidate_symbols,
            market=settings.market,
            category=universe_key,
            as_of=effective_latest_trade_date,
        )
        branch_governance_artifacts = write_branch_readiness_report(branch_governance_report)
        stage_metadata["blocked_symbol_count"] = len(branch_governance_report.blocked_symbols)
        stage_metadata["quantifiable_universe_count"] = len(branch_governance_report.quantifiable_universe)
        stage_metadata["investable_universe_count"] = len(branch_governance_report.investable_universe)
    branch_data_readiness = branch_governance_report.to_dict(include_branch_data=False)
    branch_data_payload = dict(branch_governance_report.branch_data)
    macro_ready = branch_governance_report.readiness.get("macro")
    macro_blocked = bool(macro_ready and macro_ready.status == STATUS_BLOCK)
    blocked_symbols = set(branch_governance_report.blocked_symbols)
    for symbol in list(blocked_symbols):
        if symbol in candidate_symbols:
            funnel_output.excluded_symbols.setdefault(symbol, "branch_data_readiness_block")
    if macro_blocked:
        for symbol in candidate_symbols:
            funnel_output.excluded_symbols.setdefault(symbol, "macro_data_readiness_block")
        candidate_symbols = []
    else:
        candidate_symbols = [symbol for symbol in candidate_symbols if symbol not in blocked_symbols]
    funnel_output.candidates = list(candidate_symbols)
    funnel_output.candidate_scores = {
        symbol: score
        for symbol, score in dict(funnel_output.candidate_scores).items()
        if symbol in set(candidate_symbols)
    }
    funnel_output.funnel_metadata = dict(funnel_output.funnel_metadata or {})
    funnel_output.funnel_metadata.update(
        {
            "branch_data_governance_status": {
                branch: readiness.status
                for branch, readiness in branch_governance_report.readiness.items()
            },
            "branch_data_blocked_count": len(branch_governance_report.blocked_symbols),
            "macro_data_readiness_block": macro_blocked,
        }
    )
    if branch_data_payload.get("macro_data"):
        market_snapshot.update(dict(branch_data_payload.get("macro_data") or {}))
        global_context.macro_data.update(dict(branch_data_payload.get("macro_data") or {}))
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
    global_context.metadata["branch_data_readiness"] = branch_data_readiness
    global_context.metadata["branch_readiness_artifacts"] = branch_governance_artifacts
    global_context.metadata["four_branch_fusion_blocked"] = macro_blocked
    global_context.metadata["blocked_branch_symbols"] = list(branch_governance_report.blocked_symbols[:128])
    global_context.metadata["quantifiable_universe_count"] = len(branch_governance_report.quantifiable_universe)
    global_context.metadata["investable_universe_count"] = len(branch_governance_report.investable_universe)
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
        funnel_output=funnel_output,
        resolver_snapshot=resolver_snapshot,
        branch_data_readiness=branch_data_readiness,
        branch_data_payload=branch_data_payload,
    )
