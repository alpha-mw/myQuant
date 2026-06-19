"""Offline diagnostics for deterministic theme funnel boost comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from quant_investor.agent_protocol import GlobalContext
from quant_investor.branch_contracts import BranchResult
from quant_investor.funnel.deterministic_funnel import (
    DeterministicFunnel,
    FunnelConfig,
    FunnelOutput,
)
from quant_investor.market.dag.theme_context import extract_symbol_theme_metadata


@dataclass
class ThemeBoostSymbolDelta:
    symbol: str
    baseline_rank: int | None = None
    boosted_rank: int | None = None
    rank_delta: int | None = None
    baseline_score: float | None = None
    boosted_score: float | None = None
    score_delta: float = 0.0
    baseline_selected: bool = False
    boosted_selected: bool = False
    primary_theme_id: str = ""
    primary_theme_name: str = ""
    theme_phase: str = ""
    theme_symbol_score: float = 0.0
    theme_risk_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "baseline_rank": self.baseline_rank,
            "boosted_rank": self.boosted_rank,
            "rank_delta": self.rank_delta,
            "baseline_score": self.baseline_score,
            "boosted_score": self.boosted_score,
            "score_delta": self.score_delta,
            "baseline_selected": self.baseline_selected,
            "boosted_selected": self.boosted_selected,
            "primary_theme_id": self.primary_theme_id,
            "primary_theme_name": self.primary_theme_name,
            "theme_phase": self.theme_phase,
            "theme_symbol_score": self.theme_symbol_score,
            "theme_risk_flags": list(self.theme_risk_flags),
        }


@dataclass
class ThemeBoostDiagnostics:
    baseline_count: int = 0
    boosted_count: int = 0
    overlap_count: int = 0
    overlap_ratio: float = 0.0
    entered_symbols: list[str] = field(default_factory=list)
    dropped_symbols: list[str] = field(default_factory=list)
    improved_symbols: list[ThemeBoostSymbolDelta] = field(default_factory=list)
    deteriorated_symbols: list[ThemeBoostSymbolDelta] = field(default_factory=list)
    largest_score_increases: list[ThemeBoostSymbolDelta] = field(default_factory=list)
    largest_score_decreases: list[ThemeBoostSymbolDelta] = field(default_factory=list)
    deltas_by_symbol: dict[str, ThemeBoostSymbolDelta] = field(default_factory=dict)
    phase_summary: dict[str, int] = field(default_factory=dict)
    risk_flag_counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_count": self.baseline_count,
            "boosted_count": self.boosted_count,
            "overlap_count": self.overlap_count,
            "overlap_ratio": self.overlap_ratio,
            "entered_symbols": list(self.entered_symbols),
            "dropped_symbols": list(self.dropped_symbols),
            "improved_symbols": [
                delta.to_dict()
                for delta in self.improved_symbols
            ],
            "deteriorated_symbols": [
                delta.to_dict()
                for delta in self.deteriorated_symbols
            ],
            "largest_score_increases": [
                delta.to_dict()
                for delta in self.largest_score_increases
            ],
            "largest_score_decreases": [
                delta.to_dict()
                for delta in self.largest_score_decreases
            ],
            "deltas_by_symbol": {
                symbol: delta.to_dict()
                for symbol, delta in self.deltas_by_symbol.items()
            },
            "phase_summary": dict(self.phase_summary),
            "risk_flag_counts": dict(self.risk_flag_counts),
            "metadata": dict(self.metadata),
        }

    def to_markdown(self, *, max_rows: int = 20) -> str:
        row_limit = max(int(max_rows or 0), 0)
        lines = [
            "## Theme Boost A/B Diagnostics",
            "",
            f"- baseline_count: {self.baseline_count}",
            f"- boosted_count: {self.boosted_count}",
            f"- overlap_count: {self.overlap_count}",
            f"- overlap_ratio: {self.overlap_ratio:.4f}",
            f"- entered_symbols: {_format_symbol_list(self.entered_symbols, row_limit)}",
            f"- dropped_symbols: {_format_symbol_list(self.dropped_symbols, row_limit)}",
            "",
            "### Top Improved Symbols",
            _format_delta_table(self.improved_symbols, row_limit),
            "",
            "### Top Deteriorated Symbols",
            _format_delta_table(self.deteriorated_symbols, row_limit),
            "",
            "### Phase Summary",
            _format_count_map(self.phase_summary),
            "",
            "### Risk Flag Counts",
            _format_count_map(self.risk_flag_counts),
        ]
        return "\n".join(lines).rstrip() + "\n"


def build_theme_boost_diagnostics_from_outputs(
    *,
    baseline_output: FunnelOutput,
    boosted_output: FunnelOutput,
    global_context: Any,
    top_n: int = 20,
) -> ThemeBoostDiagnostics:
    limit = max(int(top_n or 0), 0)
    baseline_candidates = list(getattr(baseline_output, "candidates", []) or [])
    boosted_candidates = list(getattr(boosted_output, "candidates", []) or [])
    baseline_ranks = _rank_map(baseline_candidates)
    boosted_ranks = _rank_map(boosted_candidates)
    baseline_scores = dict(getattr(baseline_output, "candidate_scores", {}) or {})
    boosted_scores = dict(getattr(boosted_output, "candidate_scores", {}) or {})
    baseline_symbols = set(baseline_candidates)
    boosted_symbols = set(boosted_candidates)
    all_symbols = sorted(baseline_symbols | boosted_symbols)
    overlap_symbols = baseline_symbols & boosted_symbols

    deltas_by_symbol: dict[str, ThemeBoostSymbolDelta] = {}
    for symbol in all_symbols:
        baseline_rank = baseline_ranks.get(symbol)
        boosted_rank = boosted_ranks.get(symbol)
        baseline_score = _optional_float(baseline_scores.get(symbol))
        boosted_score = _optional_float(boosted_scores.get(symbol))
        rank_delta = (
            baseline_rank - boosted_rank
            if baseline_rank is not None and boosted_rank is not None
            else None
        )
        theme_metadata = _symbol_theme_metadata(global_context, symbol)
        deltas_by_symbol[symbol] = ThemeBoostSymbolDelta(
            symbol=symbol,
            baseline_rank=baseline_rank,
            boosted_rank=boosted_rank,
            rank_delta=rank_delta,
            baseline_score=baseline_score,
            boosted_score=boosted_score,
            score_delta=round(
                float(boosted_score or 0.0) - float(baseline_score or 0.0),
                12,
            ),
            baseline_selected=symbol in baseline_symbols,
            boosted_selected=symbol in boosted_symbols,
            primary_theme_id=str(theme_metadata.get("primary_theme_id") or ""),
            primary_theme_name=str(theme_metadata.get("primary_theme_name") or ""),
            theme_phase=str(theme_metadata.get("phase") or ""),
            theme_symbol_score=float(theme_metadata.get("symbol_score") or 0.0),
            theme_risk_flags=[
                str(flag)
                for flag in list(theme_metadata.get("risk_flags", []) or [])
                if str(flag)
            ],
        )

    entered_symbols = sorted(
        boosted_symbols - baseline_symbols,
        key=lambda symbol: (boosted_ranks.get(symbol, 10**9), symbol),
    )
    dropped_symbols = sorted(
        baseline_symbols - boosted_symbols,
        key=lambda symbol: (baseline_ranks.get(symbol, 10**9), symbol),
    )
    improved = [
        delta
        for delta in deltas_by_symbol.values()
        if delta.rank_delta is not None and delta.rank_delta > 0
    ]
    deteriorated = [
        delta
        for delta in deltas_by_symbol.values()
        if delta.rank_delta is not None and delta.rank_delta < 0
    ]
    score_ranked = list(deltas_by_symbol.values())
    phase_summary: dict[str, int] = {}
    risk_flag_counts: dict[str, int] = {}
    for symbol in boosted_candidates:
        delta = deltas_by_symbol.get(symbol)
        if delta is None:
            continue
        if delta.theme_phase:
            phase_summary[delta.theme_phase] = phase_summary.get(delta.theme_phase, 0) + 1
        for flag in delta.theme_risk_flags:
            risk_flag_counts[flag] = risk_flag_counts.get(flag, 0) + 1

    union_count = len(all_symbols)
    return ThemeBoostDiagnostics(
        baseline_count=len(baseline_candidates),
        boosted_count=len(boosted_candidates),
        overlap_count=len(overlap_symbols),
        overlap_ratio=(
            len(overlap_symbols) / max(union_count, 1)
            if union_count
            else 0.0
        ),
        entered_symbols=list(entered_symbols),
        dropped_symbols=list(dropped_symbols),
        improved_symbols=sorted(
            improved,
            key=lambda delta: (-(delta.rank_delta or 0), delta.symbol),
        )[:limit],
        deteriorated_symbols=sorted(
            deteriorated,
            key=lambda delta: ((delta.rank_delta or 0), delta.symbol),
        )[:limit],
        largest_score_increases=sorted(
            score_ranked,
            key=lambda delta: (-delta.score_delta, delta.symbol),
        )[:limit],
        largest_score_decreases=sorted(
            score_ranked,
            key=lambda delta: (delta.score_delta, delta.symbol),
        )[:limit],
        deltas_by_symbol=deltas_by_symbol,
        phase_summary=dict(sorted(phase_summary.items())),
        risk_flag_counts=dict(sorted(risk_flag_counts.items())),
        metadata={
            "deterministic": True,
            "no_llm": True,
            "no_network": True,
            "diagnostic_only": True,
            "top_n": limit,
            "baseline_funnel_metadata": dict(
                getattr(baseline_output, "funnel_metadata", {}) or {}
            ),
            "boosted_funnel_metadata": dict(
                getattr(boosted_output, "funnel_metadata", {}) or {}
            ),
        },
    )


def compare_theme_boost_candidates(
    *,
    quant_result: BranchResult,
    global_context: GlobalContext,
    base_config: FunnelConfig | None = None,
    boost_cap: float | None = None,
    top_n: int = 20,
) -> ThemeBoostDiagnostics:
    source_config = base_config or FunnelConfig()
    cap = (
        float(boost_cap)
        if boost_cap is not None
        else float(getattr(source_config, "theme_boost_cap", 0.10) or 0.10)
    )
    baseline_config = replace(source_config, theme_boost_enabled=False)
    boosted_config = replace(
        source_config,
        theme_boost_enabled=True,
        theme_boost_cap=cap,
    )
    baseline_output = DeterministicFunnel(baseline_config).run(
        quant_result=quant_result,
        global_context=global_context,
    )
    boosted_output = DeterministicFunnel(boosted_config).run(
        quant_result=quant_result,
        global_context=global_context,
    )
    return build_theme_boost_diagnostics_from_outputs(
        baseline_output=baseline_output,
        boosted_output=boosted_output,
        global_context=global_context,
        top_n=top_n,
    )


def _rank_map(candidates: list[str]) -> dict[str, int]:
    return {
        str(symbol): index + 1
        for index, symbol in enumerate(candidates)
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _symbol_theme_metadata(global_context: Any, symbol: str) -> dict[str, Any]:
    try:
        metadata = extract_symbol_theme_metadata(
            global_context=global_context,
            symbol=symbol,
        )
    except Exception:
        return {}
    return metadata if isinstance(metadata, dict) else {}


def _format_symbol_list(symbols: list[str], limit: int) -> str:
    if not symbols or limit <= 0:
        return "none"
    values = [str(symbol) for symbol in symbols[:limit]]
    suffix = f" (+{len(symbols) - limit} more)" if len(symbols) > limit else ""
    return ", ".join(values) + suffix


def _format_delta_table(deltas: list[ThemeBoostSymbolDelta], limit: int) -> str:
    rows = list(deltas[:limit]) if limit > 0 else []
    if not rows:
        return "_none_"
    lines = [
        "| symbol | baseline_rank | boosted_rank | rank_delta | score_delta | theme_phase | risk_flags |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for delta in rows:
        lines.append(
            "| {symbol} | {baseline_rank} | {boosted_rank} | {rank_delta} | {score_delta:.6f} | {phase} | {flags} |".format(
                symbol=delta.symbol,
                baseline_rank=_display_optional(delta.baseline_rank),
                boosted_rank=_display_optional(delta.boosted_rank),
                rank_delta=_display_optional(delta.rank_delta),
                score_delta=float(delta.score_delta),
                phase=delta.theme_phase or "",
                flags=", ".join(delta.theme_risk_flags),
            )
        )
    return "\n".join(lines)


def _format_count_map(values: dict[str, int]) -> str:
    if not values:
        return "_none_"
    return "\n".join(
        f"- {key}: {value}"
        for key, value in sorted(values.items())
    )


def _display_optional(value: Any) -> str:
    return "" if value is None else str(value)
