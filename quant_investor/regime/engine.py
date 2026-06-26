from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from quant_investor.agent_protocol import BranchVerdict
from quant_investor.regime.features import build_regime_feature_snapshot
from quant_investor.regime.persistence import append_regime_signal, load_regime_history
from quant_investor.regime.transition import (
    bayesian_regime_update,
    default_transition_matrix,
    estimate_transition_matrix,
    normalize_probabilities,
)
from quant_investor.regime.types import (
    REGIME_RANGE_HIGH_VOL,
    REGIME_RANGE_LOW_VOL,
    REGIME_STATES,
    REGIME_TREND_DOWN,
    REGIME_TREND_UP,
    REGIME_UNKNOWN,
    RegimeSignal,
)


DEFAULT_REGIME_PRIOR = {
    REGIME_TREND_UP: 0.25,
    REGIME_RANGE_LOW_VOL: 0.25,
    REGIME_RANGE_HIGH_VOL: 0.25,
    REGIME_TREND_DOWN: 0.20,
    REGIME_UNKNOWN: 0.05,
}


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


class MarkovRegimeEngine:
    def __init__(
        self,
        history_path: str = "results/regime/markov_regime_history.jsonl",
        enabled: bool = True,
        execution_target: str = "production",
        smoothing: float = 1.0,
        persist_enabled: bool = True,
    ) -> None:
        self.history_path = str(history_path or "results/regime/markov_regime_history.jsonl")
        self.enabled = bool(enabled)
        target = str(execution_target or "").strip().lower()
        self._execution_target_diagnostics: list[str] = []
        if target == "disabled":
            self.execution_target = "disabled"
            self.enabled = False
            self._execution_target_diagnostics.append("markov_execution_target_disabled")
        elif target == "shadow":
            self.execution_target = "production"
            self._execution_target_diagnostics.append(
                "markov_shadow_deprecated_normalized_to_production"
            )
        elif target in {"", "production"}:
            self.execution_target = "production"
        else:
            self.execution_target = "production"
            self._execution_target_diagnostics.append(
                "invalid_execution_target_normalized_to_production"
            )
        self.smoothing = float(smoothing or 1.0)
        self.persist_enabled = bool(persist_enabled)

    def run(
        self,
        *,
        market: str,
        universe_key: str,
        as_of: str,
        frames: Mapping[str, pd.DataFrame],
        tradability_snapshot: Mapping[str, Mapping[str, Any]],
        cross_section_quant: Mapping[str, Any],
        macro_verdict: BranchVerdict | Mapping[str, Any] | None,
        market_snapshot: Mapping[str, Any] | None = None,
    ) -> RegimeSignal:
        feature_snapshot = build_regime_feature_snapshot(
            market=market,
            universe_key=universe_key,
            as_of=as_of,
            frames=frames,
            tradability_snapshot=tradability_snapshot,
            cross_section_quant=cross_section_quant,
            macro_verdict=macro_verdict,
        )
        feature_snapshot.metadata["execution_target"] = self.execution_target
        feature_snapshot.metadata["execution_mode"] = (
            "production" if self.enabled else "disabled"
        )
        feature_snapshot.metadata["enabled"] = self.enabled
        feature_snapshot.metadata["market_snapshot_key_count"] = len(market_snapshot or {})
        diagnostic_notes = list(feature_snapshot.diagnostics)
        diagnostic_notes.extend(self._execution_target_diagnostics)

        if not self.enabled:
            return RegimeSignal(
                as_of=feature_snapshot.as_of,
                market=feature_snapshot.market,
                universe_key=feature_snapshot.universe_key,
                dominant_regime=REGIME_UNKNOWN,
                probabilities={
                    state: (1.0 if state == REGIME_UNKNOWN else 0.0)
                    for state in REGIME_STATES
                },
                transition_matrix=default_transition_matrix(),
                confidence=1.0,
                transition_risk=0.0,
                risk_on_score=0.0,
                volatility_score=0.0,
                pressure_score=0.0,
                suggested_gross_exposure_cap=min(
                    feature_snapshot.macro_target_gross_exposure,
                    0.45,
                ),
                suggested_max_single_weight=0.12,
                turnover_cap=None,
                feature_snapshot=feature_snapshot.to_dict(),
                diagnostic_notes=diagnostic_notes + ["markov_regime_disabled"],
            )

        risk_on_score, volatility_score, pressure_score = self._scores(feature_snapshot)
        likelihood = self._likelihood(
            risk_on_score=risk_on_score,
            volatility_score=volatility_score,
            pressure_score=pressure_score,
            breadth=feature_snapshot.breadth,
            momentum_share=feature_snapshot.momentum_share,
            fake_breakout_share=feature_snapshot.fake_breakout_share,
            sample_count=feature_snapshot.sample_count,
            diagnostics=diagnostic_notes,
        )
        history = load_regime_history(
            self.history_path,
            market=feature_snapshot.market,
            universe_key=feature_snapshot.universe_key,
            before_or_equal_as_of=feature_snapshot.as_of,
            limit=252,
        )
        previous_posterior = self._previous_posterior(history)
        transition_matrix = self._transition_matrix(history)
        posterior = bayesian_regime_update(previous_posterior, transition_matrix, likelihood)
        dominant_regime = max(
            REGIME_STATES,
            key=lambda state: (posterior.get(state, 0.0), -REGIME_STATES.index(state)),
        )
        confidence = float(max(posterior.values())) if posterior else 0.0
        transition_risk = float(
            posterior.get(REGIME_RANGE_HIGH_VOL, 0.0)
            + posterior.get(REGIME_TREND_DOWN, 0.0)
        )
        suggested_gross_exposure_cap = min(
            feature_snapshot.macro_target_gross_exposure,
            posterior.get(REGIME_TREND_UP, 0.0) * 0.72
            + posterior.get(REGIME_RANGE_LOW_VOL, 0.0) * 0.58
            + posterior.get(REGIME_RANGE_HIGH_VOL, 0.0) * 0.42
            + posterior.get(REGIME_TREND_DOWN, 0.0) * 0.25
            + posterior.get(REGIME_UNKNOWN, 0.0) * 0.45,
        )
        if posterior.get(REGIME_TREND_DOWN, 0.0) >= 0.45:
            suggested_max_single_weight = 0.07
        elif (
            posterior.get(REGIME_RANGE_HIGH_VOL, 0.0)
            + posterior.get(REGIME_TREND_DOWN, 0.0)
            >= 0.55
        ):
            suggested_max_single_weight = 0.09
        else:
            suggested_max_single_weight = 0.12
        if transition_risk >= 0.60:
            turnover_cap: float | None = 0.30
        elif confidence < 0.45:
            turnover_cap = 0.40
        else:
            turnover_cap = None

        signal = RegimeSignal(
            as_of=feature_snapshot.as_of,
            market=feature_snapshot.market,
            universe_key=feature_snapshot.universe_key,
            dominant_regime=dominant_regime,
            probabilities=posterior,
            transition_matrix=transition_matrix,
            confidence=_clamp(confidence),
            transition_risk=_clamp(transition_risk),
            risk_on_score=_clamp(risk_on_score),
            volatility_score=_clamp(volatility_score),
            pressure_score=_clamp(pressure_score),
            suggested_gross_exposure_cap=_clamp(suggested_gross_exposure_cap),
            suggested_max_single_weight=_clamp(suggested_max_single_weight),
            turnover_cap=turnover_cap,
            feature_snapshot=feature_snapshot.to_dict(),
            diagnostic_notes=diagnostic_notes,
        )
        if self.persist_enabled and not self._latest_record_matches(
            market=feature_snapshot.market,
            universe_key=feature_snapshot.universe_key,
            as_of=feature_snapshot.as_of,
        ):
            persistence_notes = append_regime_signal(self.history_path, signal)
            signal.diagnostic_notes.extend(persistence_notes)
        return signal

    @staticmethod
    def _scores(feature_snapshot: Any) -> tuple[float, float, float]:
        normalized_return = _clamp((feature_snapshot.average_return + 0.015) / 0.030)
        breadth_score = _clamp(feature_snapshot.breadth)
        momentum_score = _clamp(feature_snapshot.momentum_share)
        liquidity_score = _clamp(feature_snapshot.average_liquidity)
        macro_score_norm = _clamp((feature_snapshot.macro_score + 1.0) / 2.0)
        volatility_score = _clamp(feature_snapshot.average_volatility / 0.035)
        pressure_score = _clamp(
            0.40 * feature_snapshot.median_drawdown / 0.20
            + 0.35 * feature_snapshot.fake_breakout_share
            + 0.25 * (1.0 - feature_snapshot.average_liquidity)
        )
        risk_on_score = _clamp(
            0.30 * normalized_return
            + 0.25 * breadth_score
            + 0.20 * momentum_score
            + 0.15 * liquidity_score
            + 0.10 * macro_score_norm
            - 0.25 * volatility_score
            - 0.20 * pressure_score
        )
        return risk_on_score, volatility_score, pressure_score

    @staticmethod
    def _likelihood(
        *,
        risk_on_score: float,
        volatility_score: float,
        pressure_score: float,
        breadth: float,
        momentum_share: float,
        fake_breakout_share: float,
        sample_count: int,
        diagnostics: list[str],
    ) -> dict[str, float]:
        neutral_score = 1.0 - min(abs(risk_on_score - 0.50) * 2.0, 1.0)
        low_coverage = 0.0
        if sample_count <= 0:
            low_coverage = 0.75
        elif sample_count < 5:
            low_coverage = 0.50
        elif sample_count < 10:
            low_coverage = 0.25
        poor_coverage = any("empty" in note or "missing" in note for note in diagnostics)
        unknown = 0.04 + low_coverage + (0.25 if poor_coverage else 0.0)
        raw = {
            REGIME_TREND_UP: (
                0.05
                + 0.55 * risk_on_score
                + 0.20 * _clamp(breadth)
                + 0.15 * (1.0 - volatility_score)
                + 0.05 * _clamp(momentum_share)
            ),
            REGIME_TREND_DOWN: (
                0.05
                + 0.45 * (1.0 - risk_on_score)
                + 0.20 * (1.0 - _clamp(breadth))
                + 0.20 * pressure_score
                + 0.10 * volatility_score
            ),
            REGIME_RANGE_LOW_VOL: (
                0.05
                + 0.35 * (1.0 - volatility_score)
                + 0.30 * neutral_score
                + 0.20 * (1.0 - pressure_score)
                + 0.10 * _clamp(breadth)
            ),
            REGIME_RANGE_HIGH_VOL: (
                0.05
                + 0.35 * volatility_score
                + 0.35 * pressure_score
                + 0.15 * neutral_score
                + 0.05 * _clamp(fake_breakout_share)
            ),
            REGIME_UNKNOWN: unknown,
        }
        return normalize_probabilities(raw)

    @staticmethod
    def _previous_posterior(history: list[dict[str, Any]]) -> dict[str, float]:
        if not history:
            return normalize_probabilities(DEFAULT_REGIME_PRIOR)
        latest = history[-1]
        probabilities = latest.get("probabilities")
        if isinstance(probabilities, Mapping):
            return normalize_probabilities(probabilities)
        dominant = str(latest.get("dominant_regime") or "")
        if dominant in REGIME_STATES:
            return normalize_probabilities({dominant: 1.0})
        return normalize_probabilities(DEFAULT_REGIME_PRIOR)

    def _transition_matrix(
        self,
        history: list[dict[str, Any]],
    ) -> dict[str, dict[str, float]]:
        default_matrix = default_transition_matrix()
        valid_records = [
            record
            for record in history
            if str(record.get("dominant_regime") or "") in REGIME_STATES
        ]
        if len(valid_records) < 2:
            return default_matrix
        estimated = estimate_transition_matrix(valid_records, smoothing=self.smoothing)
        blend_weight = min(1.0, len(valid_records) / 20.0)
        blended: dict[str, dict[str, float]] = {}
        for state in REGIME_STATES:
            blended[state] = normalize_probabilities(
                {
                    target: (
                        default_matrix[state][target] * (1.0 - blend_weight)
                        + estimated[state][target] * blend_weight
                    )
                    for target in REGIME_STATES
                }
            )
        return blended

    def _latest_record_matches(self, *, market: str, universe_key: str, as_of: str) -> bool:
        latest = load_regime_history(
            self.history_path,
            market=market,
            universe_key=universe_key,
            limit=1,
        )
        if not latest:
            return False
        record = latest[-1]
        return str(record.get("as_of") or "") == str(as_of or "")
