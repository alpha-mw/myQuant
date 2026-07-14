"""Report persistence helpers for market analysis outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from quant_investor.market.config import resolve_market_analysis_output_dir
from quant_investor.market.runtime_profile import profile_stage


GenerateFullReport = Callable[..., dict[str, str]]


@dataclass(frozen=True)
class MarketAnalysisPersistenceResult:
    """Persisted report paths plus the captured runtime profile payload."""

    report_paths: dict[str, Any]
    runtime_profile: dict[str, Any]


def _runtime_profile_dir(
    report_paths: dict[str, Any],
    *,
    analysis_output_dir: str | Path,
) -> Path:
    profile_anchor = report_paths.get("trade_report") or report_paths.get(
        "summary_report"
    )
    if profile_anchor:
        return Path(str(profile_anchor)).parent
    return Path(analysis_output_dir)


def write_runtime_profile_artifacts(
    *,
    market: str,
    analysis_output_dir: str | Path,
    report_paths: dict[str, Any],
    runtime_profiler: Any,
    runtime_profile_payload: dict[str, Any],
) -> dict[str, str]:
    """Write runtime profile JSON/Markdown next to the market reports."""

    resolved_output_dir = resolve_market_analysis_output_dir(
        market,
        analysis_output_dir,
    )
    profile_dir = _runtime_profile_dir(
        report_paths,
        analysis_output_dir=resolved_output_dir,
    )
    profile_dir = resolve_market_analysis_output_dir(market, profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runtime_profile_json = (
        profile_dir / f"{market}_Runtime_Profile_{profile_timestamp}.json"
    )
    runtime_profile_md = (
        profile_dir / f"{market}_Runtime_Profile_{profile_timestamp}.md"
    )
    runtime_profile_json.write_text(
        json.dumps(
            runtime_profile_payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    runtime_profile_md.write_text(
        runtime_profiler.to_markdown(),
        encoding="utf-8",
    )
    return {
        "runtime_profile_json": str(runtime_profile_json),
        "runtime_profile_md": str(runtime_profile_md),
    }


def persist_market_analysis_outputs(
    *,
    all_results: dict[str, list[dict[str, Any]]],
    market: str,
    total_capital: float,
    top_k: int,
    analysis_output_dir: str | Path,
    category_count: int,
    runtime_profiler: Any,
    report_bundle: Any,
    generate_full_report: GenerateFullReport,
) -> MarketAnalysisPersistenceResult:
    """Persist full-market reports and runtime profile artifacts."""

    resolved_output_dir = resolve_market_analysis_output_dir(
        market,
        analysis_output_dir,
    )
    with profile_stage(
        runtime_profiler,
        "analysis_report_persistence",
        {
            "category_count": int(category_count),
            "result_count": sum(len(items) for items in all_results.values()),
        },
    ) as stage_metadata:
        report_paths: dict[str, Any] = dict(
            generate_full_report(
                all_results,
                market=market,
                output_dir=str(resolved_output_dir),
                total_capital=total_capital,
                top_k=top_k,
            )
        )
        stage_metadata["report_path_count"] = len(report_paths)

    runtime_profile_payload = runtime_profiler.to_dict()
    runtime_paths = write_runtime_profile_artifacts(
        market=market,
        analysis_output_dir=resolved_output_dir,
        report_paths=report_paths,
        runtime_profiler=runtime_profiler,
        runtime_profile_payload=runtime_profile_payload,
    )
    report_paths["report_bundle"] = report_bundle
    report_paths.update(runtime_paths)
    return MarketAnalysisPersistenceResult(
        report_paths=report_paths,
        runtime_profile=runtime_profile_payload,
    )


__all__ = [
    "MarketAnalysisPersistenceResult",
    "persist_market_analysis_outputs",
    "write_runtime_profile_artifacts",
]
