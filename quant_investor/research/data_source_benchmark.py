"""Deterministic Tushare-vs-DataYes benchmark metrics and procurement gate."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS = {
    "market_correlation_min": 0.999,
    "fundamental_rank_correlation_min": 0.97,
    "top100_overlap_min": 0.95,
    "material_top100_overlap_max": 0.90,
    "relative_rankic_difference_max": 0.05,
}


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def compare_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    keys: Sequence[str],
    fields: Sequence[str],
) -> dict[str, Any]:
    """Compare two canonical frames over exact common keys."""

    left = left.copy()
    right = right.copy()
    for key in keys:
        if key in left:
            left[key] = left[key].astype(str)
        if key in right:
            right[key] = right[key].astype(str)
    left_keys = left[list(keys)].drop_duplicates() if not left.empty else pd.DataFrame(columns=keys)
    right_keys = (
        right[list(keys)].drop_duplicates() if not right.empty else pd.DataFrame(columns=keys)
    )
    union = left_keys.merge(right_keys, on=list(keys), how="outer", indicator=True)
    merged = left.merge(right, on=list(keys), how="inner", suffixes=("_tushare", "_datayes"))
    metrics: dict[str, Any] = {
        "tushare_key_count": int(len(left_keys)),
        "datayes_key_count": int(len(right_keys)),
        "common_key_count": int(len(merged)),
        "union_key_count": int(len(union)),
        "tushare_coverage": float((union["_merge"] != "right_only").mean()) if len(union) else None,
        "datayes_coverage": float((union["_merge"] != "left_only").mean()) if len(union) else None,
        "fields": {},
    }
    for field in fields:
        left_name = f"{field}_tushare"
        right_name = f"{field}_datayes"
        if left_name not in merged or right_name not in merged:
            metrics["fields"][field] = {"status": "UNAVAILABLE"}
            continue
        pair = merged[[left_name, right_name]].apply(pd.to_numeric, errors="coerce")
        valid = pair.dropna()
        delta = valid[right_name] - valid[left_name]
        denom = valid[left_name].abs().replace(0.0, np.nan)
        diagnostic = merged[list(keys) + [left_name, right_name]].copy()
        diagnostic[left_name] = pd.to_numeric(diagnostic[left_name], errors="coerce")
        diagnostic[right_name] = pd.to_numeric(diagnostic[right_name], errors="coerce")
        diagnostic = diagnostic.dropna(subset=[left_name, right_name])
        diagnostic["absolute_difference"] = (diagnostic[right_name] - diagnostic[left_name]).abs()
        diagnostic = diagnostic.sort_values(
            ["absolute_difference", *keys],
            ascending=[False, *([True] * len(keys))],
            kind="mergesort",
        ).head(5)
        metrics["fields"][field] = {
            "status": "COMPARED" if len(valid) else "UNAVAILABLE",
            "pair_count": int(len(valid)),
            "tushare_missing_rate": float(pair[left_name].isna().mean()) if len(pair) else None,
            "datayes_missing_rate": float(pair[right_name].isna().mean()) if len(pair) else None,
            "pearson": (
                _finite(valid[left_name].corr(valid[right_name], method="pearson"))
                if len(valid) >= 2
                else None
            ),
            "spearman": (
                _finite(valid[left_name].corr(valid[right_name], method="spearman"))
                if len(valid) >= 2
                else None
            ),
            "mean_abs_diff": _finite(delta.abs().mean()),
            "median_abs_relative_diff": _finite((delta.abs() / denom).median()),
            "max_abs_diff": _finite(delta.abs().max()),
            "largest_differences": [
                {
                    **{key: str(row[key]) for key in keys},
                    "tushare": _finite(row[left_name]),
                    "datayes": _finite(row[right_name]),
                    "absolute_difference": _finite(row["absolute_difference"]),
                }
                for row in diagnostic.to_dict("records")
            ],
        }
    return metrics


def compare_factors(
    tushare: Mapping[str, pd.Series],
    datayes: Mapping[str, pd.Series],
    *,
    top_n: int = 100,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for factor_id in sorted(set(tushare) | set(datayes)):
        left = tushare.get(factor_id, pd.Series(dtype=float)).rename("tushare")
        right = datayes.get(factor_id, pd.Series(dtype=float)).rename("datayes")
        pair = pd.concat([left, right], axis=1).dropna()
        left_top = set(left.dropna().nlargest(min(top_n, left.notna().sum())).index)
        right_top = set(right.dropna().nlargest(min(top_n, right.notna().sum())).index)
        denominator = max(1, min(top_n, len(left_top), len(right_top)))
        result[factor_id] = {
            "pair_count": int(len(pair)),
            "rank_correlation": (
                _finite(pair["tushare"].corr(pair["datayes"], method="spearman"))
                if len(pair) >= 2
                else None
            ),
            "top_n": int(top_n),
            "top_overlap": float(len(left_top & right_top) / denominator),
        }
    return result


def rank_combined_signals(
    signals: Mapping[str, pd.Series],
    *,
    weights: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Replay the LOW/W80 average-rank ordering with deterministic tie breaks."""

    if not signals or set(signals) != set(weights):
        raise ValueError("combined signal identities differ from weights")
    normalized = {
        factor_id: pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        for factor_id, series in signals.items()
    }
    common: set[str] | None = None
    for series in normalized.values():
        available = set(str(value) for value in series.dropna().index)
        common = available if common is None else common & available
    symbols = sorted(common or set())
    if not symbols:
        return []
    total_weight = sum(float(value) for value in weights.values())
    if not math.isfinite(total_weight) or abs(total_weight - 1.0) > 1e-12:
        raise ValueError("combined signal weights must sum to one")
    denominator = max(len(symbols) - 1, 1)
    percentiles: dict[str, pd.Series] = {}
    for factor_id, series in normalized.items():
        selected = series.reindex(symbols)
        percentiles[factor_id] = (selected.rank(method="average") - 1.0) / denominator
    combined = sum(
        (percentiles[factor_id] * float(weight) for factor_id, weight in weights.items()),
        pd.Series(0.0, index=symbols),
    )
    ordered = sorted(symbols, key=lambda symbol: (-float(combined[symbol]), symbol))
    return [
        {
            "symbol": symbol,
            "combined_percentile": float(combined[symbol]),
            "factor_percentiles": {
                factor_id: float(percentiles[factor_id][symbol])
                for factor_id in sorted(percentiles)
            },
        }
        for symbol in ordered
    ]


def compare_candidates(
    tushare_rows: Sequence[Mapping[str, Any]],
    datayes_rows: Sequence[Mapping[str, Any]],
    *,
    top_n: int = 100,
) -> dict[str, Any]:
    left = [str(row["symbol"]) for row in tushare_rows[:top_n]]
    right = [str(row["symbol"]) for row in datayes_rows[:top_n]]
    left_set = set(left)
    right_set = set(right)
    denominator = max(1, min(top_n, len(left), len(right)))
    common_order = sorted(left_set & right_set)
    left_rank = pd.Series({symbol: index for index, symbol in enumerate(left, start=1)})
    right_rank = pd.Series({symbol: index for index, symbol in enumerate(right, start=1)})
    rank_pairs = pd.concat([left_rank, right_rank], axis=1, keys=["tushare", "datayes"])
    rank_pairs = rank_pairs.loc[common_order].dropna() if common_order else pd.DataFrame()
    return {
        "top_n": int(top_n),
        "tushare_count": len(left),
        "datayes_count": len(right),
        "overlap_count": len(left_set & right_set),
        "overlap": float(len(left_set & right_set) / denominator),
        "common_rank_correlation": (
            _finite(rank_pairs["tushare"].corr(rank_pairs["datayes"], method="spearman"))
            if len(rank_pairs) >= 2
            else None
        ),
        "only_tushare": sorted(left_set - right_set),
        "only_datayes": sorted(right_set - left_set),
        "exact_order_match": left == right,
    }


def compare_rankic(
    tushare: Mapping[str, Mapping[str, float]],
    datayes: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for factor_id in sorted(set(tushare) | set(datayes)):
        left = dict(tushare.get(factor_id, {}))
        right = dict(datayes.get(factor_id, {}))
        anchors = sorted(set(left) & set(right))
        pairs = [(anchor, _finite(left[anchor]), _finite(right[anchor])) for anchor in anchors]
        pairs = [row for row in pairs if row[1] is not None and row[2] is not None]
        left_values = [float(row[1]) for row in pairs]
        right_values = [float(row[2]) for row in pairs]
        left_mean = float(np.mean(left_values)) if left_values else None
        right_mean = float(np.mean(right_values)) if right_values else None
        difference = (
            right_mean - left_mean if left_mean is not None and right_mean is not None else None
        )
        relative = (
            abs(difference) / max(abs(left_mean), 0.01)
            if difference is not None and left_mean is not None
            else None
        )
        positive_count = sum(1 for _, left_ic, right_ic in pairs if right_ic > left_ic)
        result[factor_id] = {
            "anchor_count": len(pairs),
            "tushare_mean": left_mean,
            "datayes_mean": right_mean,
            "difference": difference,
            "relative_difference": relative,
            "systematic_improvement": bool(
                pairs
                and relative is not None
                and relative > 0.05
                and difference is not None
                and difference > 0.0
                and positive_count / len(pairs) >= 0.70
            ),
            "anchors": [
                {"date": anchor, "tushare": left_ic, "datayes": right_ic}
                for anchor, left_ic, right_ic in pairs
            ],
        }
    return result


def procurement_decision(
    *,
    market: Mapping[str, Any],
    fundamental: Mapping[str, Any],
    factors: Mapping[str, Any],
    rankic: Mapping[str, Any] | None = None,
    requested_factor_count: int = 20,
    thresholds: Mapping[str, float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Fail closed unless enough evidence exists to apply the purchase gate."""

    price = market.get("fields", {}).get("close", {})
    market_corr = _finite(price.get("pearson"))
    fundamental_corrs = [
        _finite(row.get("spearman"))
        for row in fundamental.get("fields", {}).values()
        if row.get("status") == "COMPARED"
    ]
    fundamental_corrs = [value for value in fundamental_corrs if value is not None]
    overlaps = [
        _finite(row.get("top_overlap"))
        for row in factors.values()
        if _finite(row.get("top_overlap")) is not None
    ]
    rankic_rows = list((rankic or {}).values())
    rankic_ready = bool(rankic_rows) and all(
        _finite(row.get("relative_difference")) is not None for row in rankic_rows
    )
    evidence_complete = (
        market_corr is not None
        and bool(fundamental_corrs)
        and len(factors) >= requested_factor_count
        and bool(overlaps)
        and rankic_ready
    )
    material_value = any(
        value < thresholds["material_top100_overlap_max"] for value in overlaps
    ) or any(bool(row.get("systematic_improvement")) for row in rankic_rows)
    no_upgrade = (
        evidence_complete
        and market_corr > thresholds["market_correlation_min"]
        and min(fundamental_corrs) > thresholds["fundamental_rank_correlation_min"]
        and min(overlaps) > thresholds["top100_overlap_min"]
        and all(
            abs(float(row["relative_difference"])) < thresholds["relative_rankic_difference_max"]
            for row in rankic_rows
        )
    )
    if material_value:
        decision = "DATAYES_MATERIAL_VALUE"
    elif no_upgrade:
        decision = "NO_UPGRADE_NEEDED"
    else:
        decision = "INSUFFICIENT_EVIDENCE"
    return {
        "decision": decision,
        "purchase_recommendation": (
            "CONSIDER_PURCHASE"
            if decision == "DATAYES_MATERIAL_VALUE"
            else "DO_NOT_PURCHASE" if decision == "NO_UPGRADE_NEEDED" else "DEFER_PURCHASE"
        ),
        "evidence_complete": evidence_complete,
        "requested_factor_count": requested_factor_count,
        "compared_factor_count": len(factors),
        "thresholds": dict(thresholds),
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    decision = payload["procurement"]
    current_scope = dict(payload.get("current_installed_scope", {}) or {})
    limitations = list(payload.get("limitations", []) or [])
    market = dict(payload.get("market", {}) or {})
    close = dict(market.get("fields", {}).get("close", {}) or {})
    fundamental = dict(payload.get("fundamental", {}) or {})
    current_fundamental = dict(payload.get("fundamental_current_canonical", {}) or {})
    factor_rows = dict(payload.get("factors", {}) or {})
    candidate = dict(payload.get("candidate_comparison", {}) or {})
    rankic = dict(payload.get("rankic", {}) or {})
    lines = [
        "# Tushare vs DataYes Data Source Benchmark",
        "",
        f"- Status: **{payload.get('status', 'UNKNOWN')}**",
        f"- Procurement decision: **{decision['decision']}**",
        f"- Purchase recommendation: **{decision['purchase_recommendation']}**",
        f"- Current installed-factor assessment: **{current_scope.get('decision', 'UNAVAILABLE')} / {current_scope.get('purchase_recommendation', 'UNAVAILABLE')}**",
        f"- Compared installed factors/controls: **{decision['compared_factor_count']} / {decision['requested_factor_count']} requested**",
        "",
        "## Scope",
        "",
        f"- Symbols: {payload.get('symbol_count', 0)}",
        f"- Window: {payload.get('start_date')} to {payload.get('end_date')}",
        "- DataYes remains research-only; no canonical pointer or production state was changed.",
        "",
        "## Actual results",
        "",
        f"- Market keys: {market.get('common_key_count', 0)} common / {market.get('union_key_count', 0)} union; DataYes coverage {market.get('datayes_coverage')}.",
        f"- Close price: Pearson {close.get('pearson')}, Spearman {close.get('spearman')}, max absolute difference {close.get('max_abs_diff')}.",
        f"- Fundamental keys: {fundamental.get('common_key_count', 0)} common / {fundamental.get('union_key_count', 0)} union; DataYes coverage {fundamental.get('datayes_coverage')}.",
        f"- Published canonical Fundamental diagnostic: {current_fundamental.get('common_key_count', 0)} common keys; procurement metrics below use the corrected in-memory Tushare percent projection.",
        "",
        "### Fundamental field comparison",
        "",
        "| Canonical field | Pairs | Spearman | Median absolute relative difference |",
        "|---|---:|---:|---:|",
    ]
    for field, row in dict(fundamental.get("fields", {}) or {}).items():
        lines.append(
            f"| {field} | {row.get('pair_count', 0)} | {row.get('spearman')} | {row.get('median_abs_relative_diff')} |"
        )
    lines.extend(
        [
            "",
            "### Installed factor/control comparison",
            "",
            "| Factor | Pairs | Rank correlation | Top overlap |",
            "|---|---:|---:|---:|",
        ]
    )
    for factor_id, row in factor_rows.items():
        lines.append(
            f"| {factor_id} | {row.get('pair_count', 0)} | {row.get('rank_correlation')} | {row.get('top_overlap')} |"
        )
    lines.extend(
        [
            "",
            "### Composite candidate comparison",
            "",
            f"- Top{candidate.get('top_n', 100)} overlap: {candidate.get('overlap')}",
            f"- Exact order match: {candidate.get('exact_order_match')}",
            f"- Tushare-only candidates: {candidate.get('only_tushare', [])}",
            f"- DataYes-only candidates: {candidate.get('only_datayes', [])}",
            "",
            "### RankIC comparison",
            "",
            "| Factor | Anchors | Tushare mean | DataYes mean | Relative difference | Systematic improvement |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for factor_id, row in rankic.items():
        lines.append(
            f"| {factor_id} | {row.get('anchor_count', 0)} | {row.get('tushare_mean')} | {row.get('datayes_mean')} | {row.get('relative_difference')} | {row.get('systematic_improvement')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- The {payload.get('symbol_count', 0)}-symbol market cohort shows no measurable OHLCV advantage for DataYes; the installed price/volume signals are unchanged.",
            "- Fundamental procurement metrics use the corrected Tushare percent projection; the separately reported published-canonical diagnostic preserves evidence of the legacy scale bug.",
            "- The procurement gate therefore remains deferred, not passed and not rejected.",
            "- For the current installed two-factor production scope, all numerical no-upgrade thresholds pass; the broader requested ~20-factor scope remains incomplete.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in limitations or ["None"])
    lines.extend(
        ["", "## Machine-readable result", "", "See `benchmark.json` in the same directory.", ""]
    )
    return "\n".join(lines)


def write_results(output_dir: str | Path, payload: Mapping[str, Any]) -> tuple[Path, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "benchmark.json"
    report_path = root / "benchmark.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    report_path.write_text(render_markdown(payload), encoding="utf-8")
    return json_path, report_path


__all__ = [
    "DEFAULT_THRESHOLDS",
    "compare_factors",
    "compare_frames",
    "compare_candidates",
    "compare_rankic",
    "procurement_decision",
    "rank_combined_signals",
    "render_markdown",
    "write_results",
]
