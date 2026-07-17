"""
统一的 CN/US 市场样本分析、批量分析与组合级报告。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_investor.config import config
from quant_investor.fundamental_research.storage import canonical_json_bytes, sha256_bytes
from quant_investor.market.config import get_market_settings, normalize_categories, normalize_universe
from quant_investor.market.data_snapshot import build_market_data_snapshot
from quant_investor.llm_provider_priority import resolve_runtime_role_models
from quant_investor.market.dag_executor import execute_market_dag
from quant_investor.market.legacy_synthesis import (
    synthesize_legacy_analysis_results_from_dag
    as _synthesize_legacy_analysis_results_from_dag,
)
import quant_investor.market.full_report as _full_report
import quant_investor.market.legacy_batch_analysis as _legacy_batch
from quant_investor.market import name_map as _name_map
from quant_investor.market.report_persistence import (
    persist_market_analysis_outputs,
    write_analysis_run_manifest,
)
from quant_investor.market.runtime_profile import (
    MarketRuntimeProfiler,
)

_STOCK_NAME_CACHE = _name_map._STOCK_NAME_CACHE
get_stock_name = _name_map.get_stock_name
_is_unknown_stock_name = _name_map.is_unknown_stock_name
load_cn_stock_names = _name_map.load_cn_stock_names
load_stock_names = _name_map.load_stock_names
load_us_stock_names = _name_map.load_us_stock_names

ActionConsistencyGuard = _full_report.ActionConsistencyGuard
BRANCH_LABELS = _full_report.BRANCH_LABELS
BRANCH_SUPPORT_DENOMINATOR = _full_report.BRANCH_SUPPORT_DENOMINATOR
ConclusionRenderer = _full_report.ConclusionRenderer
DiagnosticsBucketizer = _full_report.DiagnosticsBucketizer
ExecutiveSummaryBuilder = _full_report.ExecutiveSummaryBuilder
_aggregate_branch_summary = _full_report._aggregate_branch_summary
_branch_label = _full_report._branch_label
_canonical_branch_map = _full_report._canonical_branch_map
_confidence_label = _full_report._confidence_label
_dedupe_text = _full_report._dedupe_text
_default_branch_conclusion = _full_report._default_branch_conclusion
_derive_stock_conclusion = _full_report._derive_stock_conclusion
_derive_stock_drag_drivers = _full_report._derive_stock_drag_drivers
_derive_stock_support_drivers = _full_report._derive_stock_support_drivers
_safe_average = _full_report._safe_average
_sanitize_text = _full_report._sanitize_text
_to_mapping = _full_report._to_mapping
build_full_market_trade_plan = _full_report.build_full_market_trade_plan
category_name = _full_report.category_name
save_candidate_index = _full_report.save_candidate_index
_DEFAULT_FULL_REPORT_BUNDLE_BUILDER = (
    _full_report._build_full_market_report_bundle
)
_DEFAULT_LEGACY_ANALYZE_BATCH = _legacy_batch.analyze_batch
_DEFAULT_LEGACY_ANALYZE_CATEGORY_FULL = _legacy_batch.analyze_category_full


def get_us_stock_name(symbol: str) -> str:
    return _legacy_batch.get_us_stock_name(symbol)


def get_all_local_symbols(
    category: str,
    market: str = "CN",
    data_dir: str | None = None,
) -> list[str]:
    return _legacy_batch.get_all_local_symbols(
        category,
        market=market,
        data_dir=data_dir,
    )


def _call_legacy_batch(
    target: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    overrides = {
        "analyze_batch": analyze_batch,
        "category_name": category_name,
        "get_all_local_symbols": get_all_local_symbols,
        "save_batch_result": save_batch_result,
    }
    originals = {name: getattr(_legacy_batch, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(_legacy_batch, name, value)
        return target(*args, **kwargs)
    finally:
        for name, value in originals.items():
            setattr(_legacy_batch, name, value)


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
) -> dict[str, Any] | None:
    return _call_legacy_batch(
        _DEFAULT_LEGACY_ANALYZE_BATCH,
        symbols,
        category,
        batch_id,
        market=market,
        universe=universe,
        total_capital=total_capital,
        risk_level=risk_level,
        verbose=verbose,
        analysis_kwargs=analysis_kwargs,
    )


def save_batch_result(
    result: dict[str, Any],
    market: str = "CN",
    output_dir: str | None = None,
) -> str:
    return _legacy_batch.save_batch_result(
        result,
        market=market,
        output_dir=output_dir,
    )


def analyze_category_full(
    category: str,
    market: str = "CN",
    universe: str | None = None,
    batch_size: int | None = None,
    data_dir: str | None = None,
    output_dir: str | None = None,
    total_capital: float = 1_000_000,
    risk_level: str = "中等",
    verbose: bool = True,
    analysis_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return _call_legacy_batch(
        _DEFAULT_LEGACY_ANALYZE_CATEGORY_FULL,
        category,
        market=market,
        universe=universe,
        batch_size=batch_size,
        data_dir=data_dir,
        output_dir=output_dir,
        total_capital=total_capital,
        risk_level=risk_level,
        verbose=verbose,
        analysis_kwargs=analysis_kwargs,
    )


def _build_full_market_report_bundle(*args: Any, **kwargs: Any):
    return _DEFAULT_FULL_REPORT_BUNDLE_BUILDER(*args, **kwargs)


_DEFAULT_ANALYZE_REPORT_BUNDLE_WRAPPER = _build_full_market_report_bundle


def generate_full_report(
    all_results: dict[str, list[dict[str, Any]]],
    market: str = "CN",
    output_dir: str | None = None,
    total_capital: float = 1_000_000,
    top_k: int = 12,
) -> dict[str, str]:
    report_bundle_builder = _build_full_market_report_bundle
    if report_bundle_builder is _DEFAULT_ANALYZE_REPORT_BUNDLE_WRAPPER:
        report_bundle_builder = _DEFAULT_FULL_REPORT_BUNDLE_BUILDER
    overrides = {
        "get_stock_name": get_stock_name,
        "load_stock_names": load_stock_names,
        "_is_unknown_stock_name": _is_unknown_stock_name,
        "category_name": category_name,
        "save_candidate_index": save_candidate_index,
        "_build_full_market_report_bundle": report_bundle_builder,
    }
    originals = {name: getattr(_full_report, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(_full_report, name, value)
        return _full_report.generate_full_report(
            all_results,
            market=market,
            output_dir=output_dir,
            total_capital=total_capital,
            top_k=top_k,
        )
    finally:
        for name, value in originals.items():
            setattr(_full_report, name, value)


def run_market_analysis(
    market: str,
    universe: str | None = None,
    mode: str = "batch",
    categories: list[str] | None = None,
    batch_size: int | None = None,
    total_capital: float = 1_000_000,
    top_k: int = 12,
    shortlist_size: int | None = None,
    verbose: bool = True,
    master_reasoning_effort: str = "high",
    agent_fallback_model: str = "",
    master_fallback_model: str = "",
    data_snapshot: dict[str, Any] | None = None,
    **analysis_kwargs: Any,
) -> dict[str, Any]:
    analysis_kwargs = dict(analysis_kwargs)
    decision_protocol = str(
        analysis_kwargs.get("decision_protocol", "v15") or "v15"
    ).strip().lower()
    if decision_protocol not in {"v15", "v16"}:
        raise ValueError("decision_protocol must be v15 or v16")
    branch_config, master_config = resolve_runtime_role_models(
        review_model_priority=analysis_kwargs.get("review_model_priority", []),
        agent_model=str(analysis_kwargs.get("agent_model", "") or ""),
        agent_fallback_model=str(agent_fallback_model or analysis_kwargs.get("agent_fallback_model", "") or ""),
        master_model=str(analysis_kwargs.get("master_model", "") or ""),
        master_fallback_model=str(master_fallback_model or analysis_kwargs.get("master_fallback_model", "") or ""),
    )
    analysis_kwargs["enable_agent_layer"] = bool(analysis_kwargs.get("enable_agent_layer", True))
    analysis_kwargs["review_model_priority"] = list(analysis_kwargs.get("review_model_priority", []) or [])
    analysis_kwargs["agent_model"] = branch_config.primary_model
    analysis_kwargs["master_model"] = master_config.primary_model
    analysis_kwargs.setdefault("master_reasoning_effort", master_reasoning_effort)
    analysis_kwargs["agent_fallback_model"] = branch_config.fallback_model
    analysis_kwargs["master_fallback_model"] = master_config.fallback_model
    settings = get_market_settings(market)
    selected_categories = (
        normalize_universe(settings.market, universe)
        if universe is not None
        else normalize_categories(settings.market, categories)
    )
    dag_universe = universe if universe is not None else (selected_categories[0] if len(selected_categories) == 1 else None)
    scoped_data_snapshot = dict(
        data_snapshot
        or build_market_data_snapshot(
            market=settings.market,
            universe=dag_universe,
            categories=selected_categories,
        )
    )
    runtime_profiler = MarketRuntimeProfiler(
        market=settings.market,
        universe=dag_universe or (selected_categories[0] if len(selected_categories) == 1 else "custom"),
        categories=list(selected_categories),
        metadata={
            "mode": mode,
            "top_k": int(top_k),
            "shortlist_size": max(1, int(shortlist_size if shortlist_size is not None else top_k)),
        },
    )
    dag_artifacts = execute_market_dag(
        market=settings.market,
        universe=dag_universe,
        categories=selected_categories,
        mode=mode,
        batch_size=batch_size,
        total_capital=total_capital,
        top_k=top_k,
        shortlist_size=max(
            1,
            int(shortlist_size if shortlist_size is not None else top_k),
        ),
        verbose=verbose,
        enable_agent_layer=bool(analysis_kwargs.get("enable_agent_layer", True)),
        review_model_priority=list(analysis_kwargs.get("review_model_priority", []) or []),
        agent_model=str(analysis_kwargs.get("agent_model", "")),
        agent_fallback_model=str(analysis_kwargs.get("agent_fallback_model", "")),
        master_model=str(analysis_kwargs.get("master_model", "")),
        master_fallback_model=str(analysis_kwargs.get("master_fallback_model", "")),
        master_reasoning_effort=str(analysis_kwargs.get("master_reasoning_effort", master_reasoning_effort)),
        agent_timeout=float(
            analysis_kwargs.get("agent_timeout", config.DEFAULT_AGENT_TIMEOUT_SECONDS)
        ),
        master_timeout=float(
            analysis_kwargs.get("master_timeout", config.DEFAULT_MASTER_TIMEOUT_SECONDS)
        ),
        funnel_profile=str(analysis_kwargs.get("funnel_profile", config.FUNNEL_PROFILE) or config.FUNNEL_PROFILE),
        max_candidates=int(analysis_kwargs.get("max_candidates", config.FUNNEL_MAX_CANDIDATES) or config.FUNNEL_MAX_CANDIDATES),
        trend_windows=list(analysis_kwargs.get("trend_windows", config.FUNNEL_TREND_WINDOWS) or config.FUNNEL_TREND_WINDOWS),
        volume_spike_threshold=float(
            analysis_kwargs.get("volume_spike_threshold", config.FUNNEL_VOLUME_SPIKE_THRESHOLD)
            or config.FUNNEL_VOLUME_SPIKE_THRESHOLD
        ),
        breakout_distance_pct=float(
            analysis_kwargs.get("breakout_distance_pct", config.FUNNEL_BREAKOUT_DISTANCE_PCT)
            or config.FUNNEL_BREAKOUT_DISTANCE_PCT
        ),
        recall_context=analysis_kwargs.get("recall_context"),
        data_snapshot=scoped_data_snapshot,
        runtime_profiler=runtime_profiler,
        decision_protocol=decision_protocol,
        v16_factor_readiness_path=str(
            analysis_kwargs.get(
                "v16_factor_readiness_path",
                "results/v16/factor_governance/readiness.json",
            )
        ),
        v16_review_root=str(
            analysis_kwargs.get("v16_review_root", "results/v16/codex_review")
        ),
        v16_config_path=str(
            analysis_kwargs.get(
                "v16_config_path",
                Path(__file__).resolve().parents[1]
                / "v16"
                / "four_branch_config.json",
            )
        ),
        v16_prompt_path=str(
            analysis_kwargs.get(
                "v16_prompt_path",
                Path(__file__).resolve().parents[1] / "v16" / "stage1_prompt.md",
            )
        ),
        v16_run_id=str(analysis_kwargs.get("v16_run_id", "") or ""),
        v16_expiry_hours=float(
            analysis_kwargs.get("v16_expiry_hours", 24.0) or 24.0
        ),
    )
    if decision_protocol == "v16":
        pending = dict(dag_artifacts.get("v16_stage1", {}) or {})
        return {
            "architecture_version": "16.0.0",
            "branch_schema_version": "v16.four-branch",
            "results": [],
            "reports": {
                "stage1_request": str(
                    dict(pending.get("review", {}) or {}).get(
                        "stage1_request_path", ""
                    )
                )
            },
            "analysis_meta": dict(dag_artifacts),
            "runtime_profile": {},
            "v16_stage1": pending,
        }
    all_results = _synthesize_legacy_analysis_results_from_dag(
        dag_artifacts=dag_artifacts,
        market=settings.market,
        universe=dag_universe or "full_a",
        categories=selected_categories,
        total_capital=total_capital,
    )
    schema_envelope = _full_report._require_current_market_schema_envelope(
        dag_artifacts.get("report_bundle"),
        label="market DAG ReportBundle",
    )
    canonical_branch_summaries = _full_report._canonical_branch_map(
        dict(dag_artifacts.get("branch_summaries", {}) or {}),
        label="market DAG branch_summaries",
        require_exact=True,
    )
    review_bundle = dag_artifacts.get("review_bundle")
    model_role_metadata = dag_artifacts["model_role_metadata"]
    execution_trace = dag_artifacts["execution_trace"]
    what_if_plan = dag_artifacts["what_if_plan"]
    global_context = dag_artifacts["global_context"]
    portfolio_decision = dag_artifacts["portfolio_decision"]
    shortlist = list(dag_artifacts.get("shortlist", []) or [])
    bayesian_records = list(dag_artifacts.get("bayesian_records", []) or [])
    funnel_output = dag_artifacts.get("funnel_output")
    symbol_packets = dag_artifacts["symbol_research_packets"]
    fundamental_deterministic_bases: dict[str, dict[str, Any]] = {}
    for symbol, packet in symbol_packets.items():
        record = dict(
            getattr(packet, "metadata", {}).get(
                "fundamental_deterministic_base", {}
            )
            or {}
        )
        if not record:
            continue
        base_score = float(record.get("base_score", 0.0))
        record["base_score"] = base_score
        record["base_score_sha256"] = sha256_bytes(
            canonical_json_bytes({"base_score": base_score})
        )
        fundamental_deterministic_bases[str(symbol)] = record
    analysis_meta: dict[str, Any] = {
        **schema_envelope,
        "market": settings.market,
        "universe": dag_universe or "full_a",
        "batch_count": len(selected_categories),
        "total_stocks": len(symbol_packets),
        "category_count": len(selected_categories),
        "symbols": list(symbol_packets.keys()),
        "analysis_kwargs": dict(analysis_kwargs),
        "shortlist_size": max(1, int(shortlist_size if shortlist_size is not None else top_k)),
        "review_model_priority": list(analysis_kwargs.get("review_model_priority", []) or []),
        "branch_model": str(analysis_kwargs.get("agent_model", "")),
        "master_model": str(analysis_kwargs.get("master_model", "")),
        "master_reasoning_effort": str(analysis_kwargs.get("master_reasoning_effort", master_reasoning_effort)),
        "agent_layer_enabled": bool(analysis_kwargs.get("enable_agent_layer", True)),
        "agent_timeout": float(
            analysis_kwargs.get("agent_timeout", config.DEFAULT_AGENT_TIMEOUT_SECONDS)
        ),
        "master_timeout": float(
            analysis_kwargs.get("master_timeout", config.DEFAULT_MASTER_TIMEOUT_SECONDS)
        ),
        "model_role_metadata": model_role_metadata.to_dict(),
        "execution_trace": execution_trace.to_dict(),
        "what_if_plan": what_if_plan.to_dict(),
        "global_context": global_context.to_dict(),
        "symbol_research_packets": {
            symbol: packet.to_dict()
            for symbol, packet in symbol_packets.items()
        },
        "fundamental_deterministic_bases": fundamental_deterministic_bases,
        "shortlist": [item.to_dict() for item in shortlist],
        "bayesian_shortlist_symbols": [item.symbol for item in shortlist],
        "bayesian_record_count": len(bayesian_records),
        "funnel_candidates_count": (
            len(getattr(funnel_output, "candidates", []) or [])
            if funnel_output is not None
            else 0
        ),
        "funnel_excluded_count": (
            len(getattr(funnel_output, "excluded_symbols", {}) or {})
            if funnel_output is not None
            else 0
        ),
        "portfolio_decision": portfolio_decision.to_dict(),
        "review_bundle": review_bundle.to_dict() if hasattr(review_bundle, "to_dict") else {},
        "ic_hints_by_symbol": dict(review_bundle.ic_hints_by_symbol if review_bundle else {}),
        "bayesian_records": [
            record.to_dict() if hasattr(record, "to_dict") else dict(record)
            for record in bayesian_records
        ],
        "funnel_summary": dict(dag_artifacts.get("funnel_summary", {})),
        "branch_summaries": {
            name: verdict.to_dict() if hasattr(verdict, "to_dict") else dict(verdict)
            for name, verdict in canonical_branch_summaries.items()
        },
        "data_quality_issues": list(dag_artifacts.get("data_quality_issues", [])),
        "resolver": dict(dag_artifacts.get("resolver", {})),
        "data_snapshot": dict(
            dag_artifacts.get("data_snapshot", {})
            or (
                global_context.to_dict().get("metadata", {})
                if hasattr(global_context, "to_dict")
                else {}
            ).get("data_snapshot", {})
            or scoped_data_snapshot
        ),
    }
    persistence = persist_market_analysis_outputs(
        all_results=all_results,
        market=settings.market,
        total_capital=total_capital,
        top_k=top_k,
        analysis_output_dir=settings.analysis_output_dir,
        category_count=len(selected_categories),
        runtime_profiler=runtime_profiler,
        report_bundle=dag_artifacts["report_bundle"],
        generate_full_report=generate_full_report,
    )
    analysis_meta["runtime_profile"] = persistence.runtime_profile
    analysis_run_manifest = write_analysis_run_manifest(
        market=settings.market,
        analysis_output_dir=settings.analysis_output_dir,
        report_paths=persistence.report_paths,
        analysis_meta=analysis_meta,
    )
    persistence.report_paths["analysis_run_manifest"] = analysis_run_manifest
    companion_manifests = sorted(
        path
        for path in Path(analysis_run_manifest).parent.glob(
            "analysis_run_manifest.*_dossier.v1.json"
        )
    )
    if companion_manifests:
        persistence.report_paths[
            "fundamental_research_counterfactual_analysis_manifest"
        ] = str(companion_manifests[0])
    return {
        **schema_envelope,
        "results": all_results,
        "reports": persistence.report_paths,
        "analysis_meta": analysis_meta,
        "runtime_profile": analysis_meta["runtime_profile"],
    }
