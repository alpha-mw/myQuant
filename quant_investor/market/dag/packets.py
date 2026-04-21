from __future__ import annotations

from statistics import fmean
from typing import Any, Mapping

import pandas as pd

from quant_investor.agent_protocol import BranchVerdict, SymbolResearchPacket
from quant_investor.branch_contracts import BranchResult, UnifiedDataBundle
from quant_investor.market.dag.common import _dedupe_texts
from quant_investor.market.shared_csv_reader import SharedCSVReadResult


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {
            "rows": 0,
            "latest_close": 0.0,
            "average_return": 0.0,
            "volatility": 0.0,
        }
    working = frame.copy()
    close_col = "close" if "close" in working.columns else "Close" if "Close" in working.columns else ""
    if not close_col:
        return {
            "rows": int(len(working)),
            "latest_close": 0.0,
            "average_return": 0.0,
            "volatility": 0.0,
        }
    close = pd.to_numeric(working[close_col], errors="coerce").dropna()
    average_return = 0.0
    volatility = 0.0
    if len(close) >= 2:
        returns = close.pct_change().dropna()
        average_return = float(returns.tail(20).mean()) if not returns.empty else 0.0
        volatility = float(returns.tail(60).std()) if len(returns) >= 3 else 0.0
    latest_close = float(close.iloc[-1]) if not close.empty else 0.0
    return {
        "rows": int(len(working)),
        "latest_close": latest_close,
        "average_return": average_return,
        "volatility": volatility,
    }


def _close_series(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    working = frame.copy()
    close_col = "close" if "close" in working.columns else "Close" if "Close" in working.columns else ""
    if not close_col:
        return pd.Series(dtype=float)
    return pd.to_numeric(working[close_col], errors="coerce").dropna()


def _volume_series(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    working = frame.copy()
    volume_col = "volume" if "volume" in working.columns else "vol" if "vol" in working.columns else ""
    if not volume_col:
        return pd.Series(dtype=float)
    return pd.to_numeric(working[volume_col], errors="coerce").dropna()


def _window_return(close: pd.Series, window: int) -> float:
    if window <= 0 or len(close) <= window:
        return 0.0
    base = float(close.iloc[-window - 1])
    latest = float(close.iloc[-1])
    if abs(base) <= 1e-8:
        return 0.0
    return (latest / base) - 1.0


def _trend_stability(close: pd.Series) -> float:
    if close.empty:
        return 0.0
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    latest = float(close.iloc[-1])
    score = 0.0
    if len(ma20.dropna()) >= 2:
        latest_ma20 = float(ma20.iloc[-1])
        prev_ma20 = float(ma20.iloc[-2])
        if latest > latest_ma20:
            score += 0.4
        if latest_ma20 >= prev_ma20:
            score += 0.3
    if len(ma60.dropna()) >= 2:
        latest_ma60 = float(ma60.iloc[-1])
        prev_ma60 = float(ma60.iloc[-2])
        if latest > latest_ma60:
            score += 0.2
        if latest_ma60 >= prev_ma60:
            score += 0.1
    return _clamp(score, 0.0, 1.0)


def _volume_confirmation(volume: pd.Series, *, spike_threshold: float) -> tuple[float, float]:
    if volume.empty:
        return 0.0, 0.0
    baseline = float(volume.tail(20).mean()) if len(volume) >= 20 else float(volume.mean())
    if baseline <= 0.0:
        return 0.0, 0.0
    ratio = float(volume.iloc[-1]) / baseline
    threshold = max(float(spike_threshold), 1.0)
    score = 0.0 if ratio <= 1.0 else _clamp((ratio - 1.0) / max(threshold - 1.0, 0.25), 0.0, 1.0)
    return ratio, score


def _breakout_metrics(
    close: pd.Series,
    *,
    breakout_distance_pct: float,
    breakout_window: int,
) -> tuple[float, float, float]:
    if close.empty:
        return 1.0, 0.0, 0.0
    window = max(int(breakout_window), 20)
    history = close.tail(window)
    if history.empty:
        return 1.0, 0.0, 0.0
    highest = float(history.max())
    latest = float(history.iloc[-1])
    if highest <= 1e-8:
        return 1.0, 0.0, 0.0
    distance = max(0.0, (highest - latest) / highest)
    threshold = max(float(breakout_distance_pct), 0.01)
    readiness = 1.0 - _clamp(distance / threshold, 0.0, 1.0)
    running_high = history.cummax().replace(0.0, pd.NA)
    drawdown_series = 1.0 - history.div(running_high).fillna(1.0)
    drawdown = float(drawdown_series.max()) if not drawdown_series.empty else 0.0
    return distance, readiness, _clamp(drawdown, 0.0, 1.0)


def _momentum_signal_strength(
    close: pd.Series,
    *,
    trend_windows: tuple[int, ...],
    trend_stability: float,
) -> tuple[float, dict[str, float]]:
    windows = tuple(sorted({max(int(item), 1) for item in trend_windows})) or (20, 60, 120)
    returns = {f"return_{window}d": _window_return(close, window) for window in windows}
    weighted_returns = 0.0
    total_weight = 0.0
    for index, window in enumerate(windows, start=1):
        weight = float(index)
        total_weight += weight
        weighted_returns += weight * _clamp((returns[f"return_{window}d"] + 0.20) / 0.60, 0.0, 1.0)
    normalized_return = weighted_returns / max(total_weight, 1.0)
    strength = _clamp(0.72 * normalized_return + 0.28 * trend_stability, 0.0, 1.0)
    return strength, returns


def _build_symbol_market_state(
    frame: pd.DataFrame,
    *,
    trend_windows: tuple[int, ...],
    volume_spike_threshold: float,
    breakout_distance_pct: float,
) -> dict[str, Any]:
    summary = _frame_summary(frame)
    close = _close_series(frame)
    volume = _volume_series(frame)
    stability = _trend_stability(close)
    momentum_strength, returns = _momentum_signal_strength(
        close,
        trend_windows=trend_windows,
        trend_stability=stability,
    )
    breakout_window = max(trend_windows[-1] if trend_windows else 120, 60)
    distance_from_high, breakout_readiness, max_drawdown = _breakout_metrics(
        close,
        breakout_distance_pct=breakout_distance_pct,
        breakout_window=breakout_window,
    )
    volume_ratio, volume_confirmation = _volume_confirmation(
        volume,
        spike_threshold=volume_spike_threshold,
    )
    latest_pullback = max(0.0, -returns.get("return_5d", _window_return(close, 5)))
    drawdown_penalty = _clamp(max_drawdown / 0.18, 0.0, 1.0)
    fake_breakout_risk = _clamp(
        breakout_readiness * (1.0 - volume_confirmation) * 0.55
        + _clamp(latest_pullback / 0.08, 0.0, 1.0) * 0.25
        + drawdown_penalty * 0.20,
        0.0,
        1.0,
    )
    liquidity_score = _clamp(
        0.65 * _clamp(float(summary.get("rows", 0)) / 250.0, 0.0, 1.0)
        + 0.35 * min(volume_ratio / max(volume_spike_threshold, 1.0), 1.0),
        0.0,
        1.0,
    )
    return {
        **summary,
        **returns,
        "trend_windows": list(trend_windows),
        "trend_stability": stability,
        "momentum_strength": momentum_strength,
        "volume_spike_ratio": volume_ratio,
        "volume_confirmation": volume_confirmation,
        "distance_from_high_pct": distance_from_high,
        "breakout_readiness": breakout_readiness,
        "max_drawdown_pct": max_drawdown,
        "fake_breakout_risk": fake_breakout_risk,
        "liquidity_score": liquidity_score,
        "is_breakout_ready": breakout_readiness >= 0.5 and volume_confirmation >= 0.35,
    }


def _build_market_snapshot(
    *,
    market: str,
    universe_key: str,
    frames: dict[str, pd.DataFrame],
    global_summary: dict[str, Any],
    latest_trade_date: str,
    macro_overview: dict[str, Any],
) -> dict[str, Any]:
    closes = [summary["latest_close"] for summary in (_frame_summary(frame) for frame in frames.values()) if summary["latest_close"] > 0]
    frame_summaries = [_frame_summary(frame) for frame in frames.values() if not frame.empty]
    avg_return = fmean([summary["average_return"] for summary in frame_summaries]) if frame_summaries else 0.0
    volatility = fmean([summary["volatility"] for summary in frame_summaries]) if frame_summaries else 0.0
    breadth = 0.0
    if frames:
        positive = sum(1 for summary in frame_summaries if summary["average_return"] > 0)
        breadth = positive / max(len(frames), 1)
    return {
        "market": market,
        "universe_key": universe_key,
        "regime": macro_overview.get("regime", "neutral"),
        "policy_signal": macro_overview.get("policy_signal", "neutral"),
        "macro_score": float(macro_overview.get("macro_score", 0.0)),
        "liquidity_score": float(macro_overview.get("liquidity_score", 0.0)),
        "volatility_percentile": float(macro_overview.get("volatility_percentile", 50.0)),
        "candidate_count": int(global_summary.get("candidate_count", len(frames))),
        "symbol_count": int(len(frames)),
        "average_return": float(avg_return),
        "average_volatility": float(volatility),
        "breadth": float(breadth),
        "latest_trade_date": latest_trade_date,
        "latest_price": max(closes) if closes else 0.0,
    }


def _build_global_quant_verdict(
    *,
    cross_section_quant: Mapping[str, Any],
    symbol_count: int,
) -> BranchVerdict:
    average_return = float(cross_section_quant.get("average_return", 0.0))
    average_volatility = float(cross_section_quant.get("average_volatility", 0.0))
    breadth = float(cross_section_quant.get("breadth", 0.0))
    candidate_count = int(cross_section_quant.get("candidate_count", symbol_count))
    sample_count = int(cross_section_quant.get("sample_count", candidate_count))
    score = _clamp(average_return * 8.0 + (breadth - 0.5) * 0.6 - average_volatility * 0.4, -1.0, 1.0)
    confidence = _clamp(0.35 + min(sample_count, max(symbol_count, 1)) / max(symbol_count, 1) * 0.12, 0.0, 1.0)
    thesis = (
        "横截面量化结果已在全局上下文中一次性计算并收敛。"
        if score >= 0
        else "横截面量化结果显示全局环境偏谨慎，需降低预期。"
    )
    return BranchVerdict(
        agent_name="quant",
        thesis=thesis,
        symbol=None,
        final_score=score,
        final_confidence=confidence,
        investment_risks=[
            f"candidate_count={candidate_count}",
            f"sample_count={sample_count}",
            f"breadth={breadth:.3f}",
        ],
        coverage_notes=[
            "cross-sectional quant computed once in GlobalContext",
            f"average_return={average_return:+.4f}",
            f"average_volatility={average_volatility:.4f}",
        ],
        diagnostic_notes=[
            "global quant summary derived from shared context",
        ],
        metadata={
            "branch_name": "quant",
            "global_context_only": True,
            "candidate_count": candidate_count,
            "sample_count": sample_count,
            "average_return": average_return,
            "average_volatility": average_volatility,
            "breadth": breadth,
        },
    )


def _build_quant_branch_result(
    *,
    frames: Mapping[str, pd.DataFrame],
) -> BranchResult:
    symbol_scores: dict[str, float] = {}
    for symbol, frame in frames.items():
        summary = _frame_summary(frame)
        score = summary["average_return"] * 8.0 - summary["volatility"] * 2.0
        symbol_scores[symbol] = _clamp(score, -1.0, 1.0)
    return BranchResult(
        branch_name="quant",
        final_score=float(fmean(symbol_scores.values()) if symbol_scores else 0.0),
        final_confidence=_clamp(0.35 + min(len(symbol_scores), 50) / 120.0, 0.0, 1.0),
        symbol_scores=symbol_scores,
        conclusion="横截面量化分支已基于 shared context 与价格代理完成全市场压缩评分。",
        signals={
            "branch_mode": "cross_section_funnel",
            "alpha_factors": ["short_term_return", "volatility_penalty"],
        },
        investment_risks=["量化压缩当前未引入更重因子库。"],
        coverage_notes=[f"symbols={len(symbol_scores)}", "full_market_deterministic_funnel"],
        diagnostic_notes=["global_quant_branch_result"],
        metadata={"reliability": 0.70},
    )


def _build_symbol_quant_verdict(
    *,
    symbol: str,
    quant_result: BranchResult,
) -> BranchVerdict:
    score = float(quant_result.symbol_scores.get(symbol, quant_result.final_score))
    return BranchVerdict(
        agent_name="quant",
        thesis="量化分支当前基于收益/波动率横截面代理给出 deterministic 结论。",
        symbol=symbol,
        final_score=score,
        final_confidence=float(quant_result.final_confidence),
        investment_risks=list(quant_result.investment_risks),
        coverage_notes=list(quant_result.coverage_notes),
        diagnostic_notes=list(quant_result.diagnostic_notes),
        metadata={"branch_name": "quant", **dict(quant_result.metadata or {})},
    )


def _build_cross_section_quant(frames: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    if not frames:
        return {
            "candidate_count": 0,
            "sample_count": 0,
            "average_return": 0.0,
            "average_volatility": 0.0,
            "breadth": 0.0,
        }
    summaries = [_frame_summary(frame) for frame in frames.values() if frame is not None and not frame.empty]
    if not summaries:
        return {
            "candidate_count": len(frames),
            "sample_count": 0,
            "average_return": 0.0,
            "average_volatility": 0.0,
            "breadth": 0.0,
        }
    positive = sum(1 for summary in summaries if summary["average_return"] > 0)
    return {
        "candidate_count": len(frames),
        "sample_count": len(summaries),
        "average_return": round(fmean(summary["average_return"] for summary in summaries), 6),
        "average_volatility": round(fmean(summary["volatility"] for summary in summaries), 6),
        "breadth": round(positive / max(len(summaries), 1), 6),
    }


def _build_symbol_tradability(
    symbol: str,
    read_result: SharedCSVReadResult,
    *,
    company_name: str = "",
    sector: str = "",
    industry: str = "",
    trend_windows: tuple[int, ...] = (20, 60, 120),
    volume_spike_threshold: float = 1.35,
    breakout_distance_pct: float = 0.06,
) -> dict[str, Any]:
    frame = read_result.frame
    market_state = _build_symbol_market_state(
        frame,
        trend_windows=trend_windows,
        volume_spike_threshold=volume_spike_threshold,
        breakout_distance_pct=breakout_distance_pct,
    )
    sector_label = str(sector or industry or "unknown")
    industry_label = str(industry or sector or "unknown")
    return {
        "symbol": symbol,
        "company_name": company_name,
        "tradable": bool(frame is not None and not frame.empty),
        "sector": sector_label,
        "industry": industry_label,
        "source_path": read_result.path,
        "resolver_strategy": read_result.resolver_trace.get("resolution_strategy", ""),
        "data_quality_issue_count": len(read_result.issues),
        "liquidity_score": float(market_state.get("liquidity_score", 0.0)),
        "momentum_strength": float(market_state.get("momentum_strength", 0.0)),
        "volume_confirmation": float(market_state.get("volume_confirmation", 0.0)),
        "fake_breakout_risk": float(market_state.get("fake_breakout_risk", 0.0)),
        "market_state": market_state,
    }


def _build_symbol_research_packet(
    *,
    symbol: str,
    company_name: str,
    market: str,
    universe_key: str,
    category: str,
    branch_verdicts: dict[str, BranchVerdict],
    read_result: SharedCSVReadResult,
    macro_verdict: BranchVerdict,
    global_quant_verdict: BranchVerdict,
    review_bundle: Any | None,
) -> SymbolResearchPacket:
    frame_summary = _frame_summary(read_result.frame)
    packet = SymbolResearchPacket(
        symbol=symbol,
        company_name=company_name,
        market=market,
        category=category,
        universe_key=universe_key,
        branch_verdicts=dict(branch_verdicts),
        branch_scores={name: float(verdict.final_score) for name, verdict in branch_verdicts.items()},
        branch_confidences={name: float(verdict.final_confidence) for name, verdict in branch_verdicts.items()},
        branch_theses={name: str(verdict.thesis) for name, verdict in branch_verdicts.items()},
        risk_flags=_dedupe_texts(
            [item for verdict in branch_verdicts.values() for item in verdict.investment_risks]
            + [issue.message for issue in read_result.issues]
        ),
        coverage_notes=_dedupe_texts(
            [item for verdict in branch_verdicts.values() for item in verdict.coverage_notes]
        ),
        diagnostic_notes=_dedupe_texts(
            [item for verdict in branch_verdicts.values() for item in verdict.diagnostic_notes]
        ),
        metadata={
            "company_name": company_name,
            "resolved_path": read_result.path,
            "resolver_trace": dict(read_result.resolver_trace),
            "macro_regime": macro_verdict.metadata.get("regime", "neutral"),
            "macro_score": float(macro_verdict.final_score),
            "global_quant_summary": global_quant_verdict.to_dict(),
            "latest_close": float(frame_summary.get("latest_close", 0.0)),
            "price_summary": frame_summary,
            "data_quality_issues": [issue.to_dict() for issue in read_result.issues],
            "review_fallback_reasons": list(review_bundle.fallback_reasons if review_bundle else []),
        },
    )
    return packet


def _build_symbol_bundle(
    *,
    symbol: str,
    frame: pd.DataFrame,
    read_result: SharedCSVReadResult,
    market: str,
    market_snapshot: Mapping[str, Any],
) -> UnifiedDataBundle:
    return UnifiedDataBundle(
        market=market,
        symbols=[symbol],
        symbol_data={symbol: frame},
        fundamentals={},
        event_data={},
        sentiment_data={},
        macro_data=dict(market_snapshot),
        metadata={
            "symbol_provenance": {
                symbol: {
                    "path": read_result.path,
                    "resolver_trace": read_result.resolver_trace,
                    "data_quality_issues": [issue.to_dict() for issue in read_result.issues],
                }
            },
        },
    )
