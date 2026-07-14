from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Mapping

import numpy as np
import pandas as pd

from quant_investor.agent_protocol import BranchVerdict, SymbolResearchPacket
from quant_investor.branch_contracts import BranchResult, UnifiedDataBundle
from quant_investor.factors.runtime import (
    ProductionEvaluationContext,
    production_runtime_input_sha256,
    production_runtime_metadata_is_ready,
    production_runtime_score_is_ready,
    score_with_mined_factors,
)
from quant_investor.market.dag.common import _dedupe_texts
from quant_investor.market.read_result import MarketDataReadResult


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(frozen=True)
class _PreparedMarketStateFrame:
    summary: dict[str, Any]
    close: pd.Series
    volume: pd.Series


_QUANT_BRANCH_VALIDATION_SEAL = object()


@dataclass(frozen=True, slots=True)
class _QuantBranchValidationToken:
    """Process-local proof that branch validation scanned the real frames."""

    production_input_sha256: str
    production_output_sha256: str
    evaluation_context_sha256: str
    symbol_count: int
    symbol_set_sha256: str
    symbol_scores_sha256: str
    registry_sha256: str
    factor_set_sha256: str
    contracts_sha256: str
    receipt_sha256: str
    final_score_hex: str
    final_confidence_hex: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _QUANT_BRANCH_VALIDATION_SEAL:
            raise TypeError("quant branch validation tokens are internal")

    def __reduce__(self) -> object:
        raise TypeError("quant branch validation tokens are not serializable")


def _empty_frame_summary(rows: int = 0) -> dict[str, Any]:
    return {
        "rows": int(rows),
        "latest_close": 0.0,
        "average_return": 0.0,
        "volatility": 0.0,
    }


def _pct_change_values(close: pd.Series) -> np.ndarray:
    values = close.to_numpy(dtype=float, copy=False)
    if values.size < 2:
        return np.array([], dtype=float)
    previous = values[:-1]
    current = values[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = current / previous - 1.0
    return returns[~np.isnan(returns)]


def _numeric_frame_series(
    frame: pd.DataFrame,
    column: str,
    empty: pd.Series,
) -> pd.Series:
    if not column:
        return empty
    values = frame[column]
    if pd.api.types.is_numeric_dtype(values.dtype):
        return values.dropna()
    return pd.to_numeric(values, errors="coerce").dropna()


def _prepare_market_state_frame(frame: pd.DataFrame) -> _PreparedMarketStateFrame:
    empty = pd.Series(dtype=float)
    if frame is None or frame.empty:
        return _PreparedMarketStateFrame(
            summary=_empty_frame_summary(),
            close=empty,
            volume=empty,
        )
    working = frame
    close_col = (
        "close" if "close" in working.columns else "Close" if "Close" in working.columns else ""
    )
    volume_col = (
        "volume" if "volume" in working.columns else "vol" if "vol" in working.columns else ""
    )
    close = _numeric_frame_series(working, close_col, empty)
    volume = _numeric_frame_series(working, volume_col, empty)
    if not close_col:
        return _PreparedMarketStateFrame(
            summary=_empty_frame_summary(len(working)),
            close=close,
            volume=volume,
        )
    average_return = 0.0
    volatility = 0.0
    if len(close) >= 2:
        returns = _pct_change_values(close)
        average_return = float(np.mean(returns[-20:])) if returns.size else 0.0
        volatility = (
            float(np.std(returns[-60:], ddof=1))
            if returns.size >= 3
            else 0.0
        )
    latest_close = float(close.iloc[-1]) if not close.empty else 0.0
    return _PreparedMarketStateFrame(
        summary={
            "rows": int(len(working)),
            "latest_close": latest_close,
            "average_return": average_return,
            "volatility": volatility,
        },
        close=close,
        volume=volume,
    )


def _frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return _prepare_market_state_frame(frame).summary


def _close_series(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    working = frame.copy()
    close_col = (
        "close" if "close" in working.columns else "Close" if "Close" in working.columns else ""
    )
    if not close_col:
        return pd.Series(dtype=float)
    return pd.to_numeric(working[close_col], errors="coerce").dropna()


def _volume_series(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    working = frame.copy()
    volume_col = (
        "volume" if "volume" in working.columns else "vol" if "vol" in working.columns else ""
    )
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


def _latest_moving_average_pair(close: pd.Series, window: int) -> tuple[float, float] | None:
    if window <= 0 or len(close) < window + 1:
        return None
    latest_ma = float(close.tail(window).mean())
    previous_ma = float(close.iloc[-window - 1 : -1].mean())
    return latest_ma, previous_ma


def _trend_stability(close: pd.Series) -> float:
    if close.empty:
        return 0.0
    latest = float(close.iloc[-1])
    score = 0.0
    ma20 = _latest_moving_average_pair(close, 20)
    if ma20 is not None:
        latest_ma20, prev_ma20 = ma20
        if latest > latest_ma20:
            score += 0.4
        if latest_ma20 >= prev_ma20:
            score += 0.3
    ma60 = _latest_moving_average_pair(close, 60)
    if ma60 is not None:
        latest_ma60, prev_ma60 = ma60
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
    history_values = history.to_numpy(dtype=float, copy=False)
    history_values = history_values[~np.isnan(history_values)]
    drawdown = 0.0
    if history_values.size:
        running_high = np.maximum.accumulate(history_values)
        valid = np.abs(running_high) > 1e-12
        if valid.any():
            drawdowns = np.zeros_like(history_values, dtype=float)
            drawdowns[valid] = 1.0 - history_values[valid] / running_high[valid]
            drawdown = float(np.nanmax(drawdowns))
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
    prepared = _prepare_market_state_frame(frame)
    summary = prepared.summary
    close = prepared.close
    volume = prepared.volume
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


def _summary_records(
    frames: Mapping[str, pd.DataFrame],
    frame_summaries: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for symbol, frame in frames.items():
        if frame is None or frame.empty:
            continue
        summary = dict((frame_summaries or {}).get(symbol, {}) or {})
        if not summary:
            summary = _frame_summary(frame)
        records.append(summary)
    return records


def _build_market_snapshot(
    *,
    market: str,
    universe_key: str,
    frames: dict[str, pd.DataFrame],
    global_summary: dict[str, Any],
    latest_trade_date: str,
    macro_overview: dict[str, Any],
    frame_summaries: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    summaries = _summary_records(frames, frame_summaries)
    closes = [
        summary["latest_close"]
        for summary in summaries
        if summary["latest_close"] > 0
    ]
    avg_return = (
        fmean([summary["average_return"] for summary in summaries])
        if summaries
        else 0.0
    )
    volatility = (
        fmean([summary["volatility"] for summary in summaries]) if summaries else 0.0
    )
    breadth = 0.0
    if frames:
        positive = sum(1 for summary in summaries if summary["average_return"] > 0)
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


def _quant_validation_identities(
    quant_result: BranchResult,
) -> dict[str, Any] | None:
    try:
        runtime_metadata = dict(
            quant_result.metadata.get("mined_factor_runtime", {}) or {}
        )
        registry = dict(runtime_metadata.get("registry", {}) or {})
        governance = dict(registry.get("governance_runtime", {}) or {})
        activation = dict(
            governance.get("quant_production_activation", {}) or {}
        )
        return {
            "production_input_sha256": str(
                runtime_metadata["production_input_sha256"]
            ),
            "production_output_sha256": str(
                runtime_metadata["production_output_attestation_sha256"]
            ),
            "evaluation_context_sha256": str(
                runtime_metadata["production_evaluation_context_sha256"]
            ),
            "symbol_count": int(runtime_metadata["symbol_count"]),
            "symbol_set_sha256": str(runtime_metadata["symbol_set_sha256"]),
            "symbol_scores_sha256": str(
                runtime_metadata["symbol_scores_sha256"]
            ),
            "registry_sha256": str(registry["registry_sha256"]),
            "factor_set_sha256": str(
                governance["production_factor_set_sha256"]
            ),
            "contracts_sha256": str(
                governance["factor_runtime_contracts_sha256"]
            ),
            "receipt_sha256": str(activation["receipt_file_sha256"]),
            "final_score_hex": float(quant_result.final_score).hex(),
            "final_confidence_hex": float(
                quant_result.final_confidence
            ).hex(),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _new_quant_validation_token(
    quant_result: BranchResult,
) -> _QuantBranchValidationToken | None:
    identities = _quant_validation_identities(quant_result)
    if identities is None:
        return None
    return _QuantBranchValidationToken(
        **identities,
        _seal=_QUANT_BRANCH_VALIDATION_SEAL,
    )


def _quant_validation_token_matches(
    token: _QuantBranchValidationToken | None,
    quant_result: BranchResult,
) -> bool:
    if (
        type(token) is not _QuantBranchValidationToken
        or token._seal is not _QUANT_BRANCH_VALIDATION_SEAL
    ):
        return False
    identities = _quant_validation_identities(quant_result)
    return bool(
        identities is not None
        and all(
            getattr(token, field_name) == value
            for field_name, value in identities.items()
        )
    )


def _build_global_quant_verdict(
    *,
    cross_section_quant: Mapping[str, Any],
    symbol_count: int,
    quant_result: BranchResult | None = None,
    expected_frames: Mapping[str, pd.DataFrame] | None = None,
    validation_token: _QuantBranchValidationToken | None = None,
) -> BranchVerdict:
    average_return = float(cross_section_quant.get("average_return", 0.0))
    average_volatility = float(cross_section_quant.get("average_volatility", 0.0))
    breadth = float(cross_section_quant.get("breadth", 0.0))
    candidate_count = int(cross_section_quant.get("candidate_count", symbol_count))
    sample_count = int(cross_section_quant.get("sample_count", candidate_count))
    quant_metadata = dict((quant_result.metadata if quant_result else {}) or {})
    runtime_metadata = dict(quant_metadata.get("mined_factor_runtime", {}) or {})
    governance_status = str(
        quant_metadata.get("governance_status")
        or runtime_metadata.get("governance_status")
        or "governance_blocked"
    )
    factor_mode = str(
        quant_metadata.get("factor_mode")
        or runtime_metadata.get("factor_mode")
        or "governance_blocked"
    )
    production_eligible = bool(
        quant_metadata.get(
            "production_eligible",
            runtime_metadata.get("production_eligible", False),
        )
    )
    production_quant_evidence = bool(
        quant_result is not None
        and governance_status == "ready"
        and factor_mode == "governed_mined_factors"
        and production_eligible
        and _quant_validation_token_matches(validation_token, quant_result)
        and production_runtime_metadata_is_ready(
            runtime_metadata,
            expected_symbols=list(quant_result.symbol_scores),
            expected_symbol_scores=quant_result.symbol_scores,
            expected_input_digest=validation_token.production_input_sha256,
            expected_evaluation_context_sha256=(
                validation_token.evaluation_context_sha256
            ),
        )
        and len(quant_result.symbol_scores) == symbol_count
        and np.isclose(
            float(quant_result.final_score),
            float(fmean(quant_result.symbol_scores.values())),
            rtol=0.0,
            atol=1e-12,
        )
        and np.isfinite(float(quant_result.final_score))
        and np.isfinite(float(quant_result.final_confidence))
    )
    score = float(quant_result.final_score) if production_quant_evidence else 0.0
    confidence = (
        _clamp(float(quant_result.final_confidence), 0.0, 1.0)
        if production_quant_evidence
        else 0.0
    )
    thesis = (
        "全局 Quant 汇总仅使用 FactorGovernanceProtocol v2 合资格生产因子结果。"
        if production_quant_evidence
        else (
            "FactorGovernanceProtocol v2 未授权生产 Quant 证据；"
            "横截面收益、波动率和广度仅作诊断。"
        )
    )
    runtime_blockers = [
        str(item)
        for item in runtime_metadata.get("runtime_blockers", []) or []
        if str(item).strip()
    ]
    return BranchVerdict(
        agent_name="quant",
        thesis=thesis,
        symbol=None,
        final_score=score,
        final_confidence=confidence,
        investment_risks=(
            [
                f"candidate_count={candidate_count}",
                f"sample_count={sample_count}",
            ]
            if production_quant_evidence
            else [
                f"production_quant_evidence_blocked:{governance_status}",
                *[f"factor_runtime_blocker:{item}" for item in runtime_blockers],
            ]
        ),
        coverage_notes=[
            "cross-section diagnostics computed once in GlobalContext",
            f"diagnostic_average_return={average_return:+.4f}",
            f"diagnostic_average_volatility={average_volatility:.4f}",
            f"diagnostic_breadth={breadth:.3f}",
        ],
        diagnostic_notes=[
            "global_quant_score_from_governance_aware_quant_result",
            "cross_section_return_volatility_breadth_diagnostic_only",
        ],
        metadata={
            "branch_name": "quant",
            "global_context_only": True,
            "source": "governance_aware_quant_result",
            "governance_status": governance_status,
            "factor_mode": factor_mode,
            "production_eligible": production_eligible,
            "production_quant_evidence": production_quant_evidence,
            "cross_section_diagnostic_only": True,
            "candidate_count": candidate_count,
            "sample_count": sample_count,
            "average_return": average_return,
            "average_volatility": average_volatility,
            "breadth": breadth,
        },
    )


def _build_quant_branch_result_with_validation(
    *,
    frames: Mapping[str, pd.DataFrame],
    frame_summaries: Mapping[str, Mapping[str, Any]] | None = None,
    evaluation_context: ProductionEvaluationContext | None = None,
    evaluation_context_blockers: list[str] | None = None,
) -> tuple[BranchResult, _QuantBranchValidationToken | None]:
    mined = score_with_mined_factors(
        frames,
        evaluation_context=evaluation_context,
    )
    mined_metadata = mined.to_metadata()
    branch_input_digest: str | None = None
    if mined.production_input_sha256:
        try:
            registry = dict(mined_metadata.get("registry", {}) or {})
            governance = dict(registry.get("governance_runtime", {}) or {})
            runtime_contracts = dict(
                governance.get("factor_runtime_contracts", {}) or {}
            )
            branch_input_digest = production_runtime_input_sha256(
                frames,
                runtime_contracts,
            )
        except (TypeError, ValueError):
            branch_input_digest = None
    runtime_ready = production_runtime_score_is_ready(
        mined,
        expected_symbols=list(frames),
        expected_input_digest=branch_input_digest,
        expected_evaluation_context=evaluation_context,
    )
    if runtime_ready:
        symbol_scores = dict(mined.symbol_scores)
        factors_used = list(mined.factors_used)
        factor_mode = "governed_mined_factors"
        conclusion = "横截面量化分支已接入通过 8 道门的 production mined factors。"
        investment_risks = [
            "量化分支只消费 production_factor；paper/research 因子权重为 0 且不进入选股。",
            f"mined_factor_coverage={mined.coverage_rate:.2%}",
        ]
        coverage_notes = [
            f"symbols={len(symbol_scores)}",
            f"production_factors={mined.factor_count}",
            f"factor_coverage={mined.coverage_rate:.2%}",
        ]
        diagnostic_notes = [
            "global_quant_branch_result",
            "mined_factor_registry_enforced",
        ]
        metadata = {
            "reliability": _clamp(0.72 + min(mined.factor_count, 5) * 0.03, 0.0, 0.90),
            "factor_mode": factor_mode,
            "governance_status": mined.governance_status,
            "confidence_multiplier": float(mined.confidence_multiplier),
            "production_eligible": bool(mined.production_eligible),
            "runtime_mode": mined.runtime_mode,
            "legacy_fallback_allowed": False,
            "mined_factor_runtime": mined_metadata,
        }
    else:
        symbol_scores = {str(symbol): 0.0 for symbol in frames}
        factors_used = []
        factor_mode = "governance_blocked"
        conclusion = (
            "横截面量化分支没有通过 FactorGovernanceProtocol v2 完整运行时契约的因子，"
            "按治理协议阻断；"
            "不会回退到收益/波动率代理。"
        )
        investment_risks = [
            "当前 selectable 记录未通过 v2 protocol/set/slot/budget/evidence 完整门禁。",
            "量化分支置信度为 0；legacy proxy fallback 被禁止。",
        ]
        coverage_notes = [
            f"symbols={len(symbol_scores)}",
            "governance_blocked_no_protocol_eligible_production_factor",
        ]
        diagnostic_notes = [
            "global_quant_branch_result",
            "mined_factor_runtime_contract_not_ready",
            "legacy_fallback_forbidden",
            *[f"factor_runtime_blocker:{item}" for item in mined.runtime_blockers],
            *[
                f"production_evaluation_context_blocker:{item}"
                for item in (evaluation_context_blockers or [])
            ],
        ]
        metadata = {
            "reliability": 0.0,
            "factor_mode": factor_mode,
            "governance_status": "governance_blocked",
            "confidence_multiplier": 0.0,
            "production_eligible": False,
            "runtime_mode": mined.runtime_mode,
            "legacy_fallback_allowed": False,
            "mined_factor_runtime": mined_metadata,
            "production_evaluation_context_blockers": list(
                evaluation_context_blockers or []
            ),
        }
    result = BranchResult(
        branch_name="quant",
        final_score=float(fmean(symbol_scores.values()) if symbol_scores else 0.0),
        final_confidence=(
            _clamp(
                0.38
                + min(len(symbol_scores), 50) / 120.0
                + min(mined.factor_count, 5) * 0.02,
                0.0,
                1.0,
            )
            if runtime_ready
            else 0.0
        ),
        symbol_scores=symbol_scores,
        conclusion=conclusion,
        signals={
            "branch_mode": "cross_section_funnel",
            "factor_mode": factor_mode,
            "alpha_factors": factors_used,
        },
        investment_risks=investment_risks,
        coverage_notes=coverage_notes,
        diagnostic_notes=diagnostic_notes,
        metadata=metadata,
    )
    token = _new_quant_validation_token(result) if runtime_ready else None
    return result, token


def _build_quant_branch_result(
    *,
    frames: Mapping[str, pd.DataFrame],
    frame_summaries: Mapping[str, Mapping[str, Any]] | None = None,
    evaluation_context: ProductionEvaluationContext | None = None,
    evaluation_context_blockers: list[str] | None = None,
) -> BranchResult:
    """Compatibility wrapper returning the historical BranchResult only."""

    result, _token = _build_quant_branch_result_with_validation(
        frames=frames,
        frame_summaries=frame_summaries,
        evaluation_context=evaluation_context,
        evaluation_context_blockers=evaluation_context_blockers,
    )
    return result


def _build_symbol_quant_verdict(
    *,
    symbol: str,
    quant_result: BranchResult,
) -> BranchVerdict:
    score = float(quant_result.symbol_scores.get(symbol, quant_result.final_score))
    factor_mode = str(
        quant_result.metadata.get("factor_mode", "governance_blocked")
    )
    if factor_mode == "governed_mined_factors":
        thesis = "量化分支基于已治理 production mined factors 给出 deterministic 结论。"
    else:
        thesis = "量化分支缺少合资格 production factor，证据被治理协议阻断。"
    return BranchVerdict(
        agent_name="quant",
        thesis=thesis,
        symbol=symbol,
        final_score=score,
        final_confidence=float(quant_result.final_confidence),
        investment_risks=list(quant_result.investment_risks),
        coverage_notes=list(quant_result.coverage_notes),
        diagnostic_notes=list(quant_result.diagnostic_notes),
        metadata={"branch_name": "quant", **dict(quant_result.metadata or {})},
    )


def _build_cross_section_quant(
    frames: Mapping[str, pd.DataFrame],
    *,
    frame_summaries: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not frames:
        return {
            "candidate_count": 0,
            "sample_count": 0,
            "average_return": 0.0,
            "average_volatility": 0.0,
            "breadth": 0.0,
        }
    summaries = _summary_records(frames, frame_summaries)
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
    read_result: MarketDataReadResult,
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
    read_result: MarketDataReadResult,
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
        branch_scores={
            name: float(verdict.final_score) for name, verdict in branch_verdicts.items()
        },
        branch_confidences={
            name: float(verdict.final_confidence) for name, verdict in branch_verdicts.items()
        },
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
            "review_fallback_reasons": list(
                review_bundle.fallback_reasons if review_bundle else []
            ),
        },
    )
    return packet


def _build_symbol_bundle(
    *,
    symbol: str,
    frame: pd.DataFrame,
    read_result: MarketDataReadResult,
    market: str,
    market_snapshot: Mapping[str, Any],
    branch_data_readiness: Mapping[str, Any] | None = None,
    branch_data_payload: Mapping[str, Any] | None = None,
) -> UnifiedDataBundle:
    branch_payload = dict(branch_data_payload or {})
    fundamentals_by_symbol = dict(branch_payload.get("fundamentals", {}) or {})
    event_data_by_symbol = dict(branch_payload.get("event_data", {}) or {})
    sentiment_data_by_symbol = dict(branch_payload.get("sentiment_data", {}) or {})
    macro_payload = dict(market_snapshot)
    macro_payload.update(dict(branch_payload.get("macro_data", {}) or {}))
    symbol_fundamentals = dict(fundamentals_by_symbol.get(symbol, {}) or {})
    symbol_events = list(event_data_by_symbol.get(symbol, []) or [])
    symbol_sentiment = dict(sentiment_data_by_symbol.get(symbol, {}) or {})
    return UnifiedDataBundle(
        market=market,
        symbols=[symbol],
        symbol_data={symbol: frame},
        fundamentals={symbol: symbol_fundamentals} if symbol_fundamentals else {},
        event_data={symbol: symbol_events} if symbol_events else {},
        sentiment_data={symbol: symbol_sentiment} if symbol_sentiment else {},
        macro_data=macro_payload,
        metadata={
            "symbol_provenance": {
                symbol: {
                    "path": read_result.path,
                    "resolver_trace": read_result.resolver_trace,
                    "data_quality_issues": [issue.to_dict() for issue in read_result.issues],
                }
            },
            "branch_data_readiness": dict(branch_data_readiness or {}),
            "branch_data_sources": {
                "fundamental": symbol_fundamentals.get("source", ""),
                "intelligence": symbol_sentiment.get("source", ""),
                "macro": macro_payload.get("source", ""),
            },
        },
    )
