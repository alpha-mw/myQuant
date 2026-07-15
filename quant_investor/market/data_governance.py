"""Read-only and explicit-fill data governance entrypoint for CN market data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from quant_investor.market.branch_readiness import (
    DEFAULT_READINESS_ROOT,
    DEFAULT_FUNDAMENTAL_ROOT,
    DEFAULT_MACRO_ROOT,
    assess_branch_data_readiness,
    make_run_id,
    write_branch_readiness_report,
)
from quant_investor.market.market_data_reader import MarketDataReader


def _normalize_categories(categories: Sequence[str] | str | None, category: str = "") -> list[str]:
    values: list[str] = []
    if isinstance(categories, str):
        values.extend(item.strip() for item in categories.split(",") if item.strip())
    elif categories:
        values.extend(str(item).strip() for item in categories if str(item).strip())
    if category and not values:
        values.extend(item.strip() for item in str(category).split(",") if item.strip())
    return list(dict.fromkeys(values or ["full_a"]))


def _read_local_frames(
    *,
    market: str,
    category: str,
    as_of: str,
    data_dir: str | Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], MarketDataReader]:
    reader = MarketDataReader(market=market, data_root=data_dir or "data", mode_policy="strict")
    symbols = reader.list_symbols(universe_key=category, category=None if category == "full_a" else category)
    frames: dict[str, pd.DataFrame] = {}
    read_results: dict[str, Any] = {}
    for symbol in symbols:
        result = reader.read_symbol_frame(
            symbol,
            universe_key=category,
            category=None if category == "full_a" else category,
            end_date=as_of,
        )
        frames[str(symbol)] = result.frame
        read_results[str(symbol)] = result
    return frames, read_results, reader


def run_data_governance(
    *,
    market: str = "CN",
    category: str = "full_a",
    categories: Sequence[str] | str | None = None,
    as_of: str = "",
    allow_live: bool = False,
    allow_public_fallback: bool = False,
    output_dir: str | Path = DEFAULT_READINESS_ROOT,
    data_dir: str | Path | None = None,
    fundamental_root: str | Path = DEFAULT_FUNDAMENTAL_ROOT,
    macro_root: str | Path = DEFAULT_MACRO_ROOT,
) -> dict[str, Any]:
    """Audit branch data readiness and optionally refresh local marts.

    Default behavior is local/read-only. Provider calls happen only when
    ``allow_live`` or ``allow_public_fallback`` is explicitly true.
    """

    if str(market).upper() != "CN":
        raise ValueError("data-governance currently supports CN only")
    selected_categories = _normalize_categories(categories, category)
    run_id = make_run_id(as_of)

    if allow_live or allow_public_fallback:
        from quant_investor.market.fundamental_mart import run_cn_fundamental_maintenance
        from quant_investor.market.macro_mart import run_cn_macro_maintenance

        run_cn_fundamental_maintenance(
            market="CN",
            universes="full_a" if "full_a" in selected_categories else ",".join(selected_categories),
            as_of=as_of,
            data_root=fundamental_root,
            allow_live=bool(allow_live),
        )
        run_cn_macro_maintenance(
            as_of=as_of,
            data_root=macro_root,
            allow_live=bool(allow_live),
            allow_public_fallback=bool(allow_public_fallback),
            run_id=run_id,
        )

    reports: list[dict[str, Any]] = []
    artifacts_by_category: dict[str, dict[str, str]] = {}
    for selected_category in selected_categories:
        frames, read_results, reader = _read_local_frames(
            market="CN",
            category=selected_category,
            as_of=as_of,
            data_dir=data_dir,
        )
        effective_as_of = as_of or str(reader.snapshot().get("latest_complete_trade_date") or "")
        report = assess_branch_data_readiness(
            frames=frames,
            read_results=read_results,
            candidate_symbols=list(frames.keys()),
            market="CN",
            category=selected_category,
            as_of=effective_as_of,
            fundamental_root=fundamental_root,
            macro_root=macro_root,
            run_id=run_id if len(selected_categories) == 1 else f"{run_id}_{selected_category}",
        )
        report.metadata.update(
            {
                "allow_live": bool(allow_live),
                "allow_public_fallback": bool(allow_public_fallback),
                "local_read_only": not bool(allow_live or allow_public_fallback),
            }
        )
        artifacts = write_branch_readiness_report(report, output_dir=output_dir)
        artifacts_by_category[selected_category] = artifacts
        payload = report.to_dict(include_branch_data=False)
        payload["artifacts"] = artifacts
        reports.append(payload)

    overall_status = (
        "blocked"
        if any(
            readiness.get("status") == "block"
            for report in reports
            for readiness in dict(report.get("readiness", {})).values()
            if isinstance(readiness, dict)
        )
        else "passed"
    )
    return {
        "run_id": run_id,
        "status": overall_status,
        "market": "CN",
        "categories": selected_categories,
        "as_of": as_of,
        "allow_live": bool(allow_live),
        "allow_public_fallback": bool(allow_public_fallback),
        "local_read_only": not bool(allow_live or allow_public_fallback),
        "reports": reports,
        "artifacts": artifacts_by_category,
    }


__all__ = ["run_data_governance"]
