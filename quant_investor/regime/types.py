from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


REGIME_TREND_UP = "趋势上涨"
REGIME_RANGE_LOW_VOL = "震荡低波"
REGIME_RANGE_HIGH_VOL = "震荡高波"
REGIME_TREND_DOWN = "趋势下跌"
REGIME_UNKNOWN = "未知"

REGIME_STATES = (
    REGIME_TREND_UP,
    REGIME_RANGE_LOW_VOL,
    REGIME_RANGE_HIGH_VOL,
    REGIME_TREND_DOWN,
    REGIME_UNKNOWN,
)


@dataclass
class RegimeFeatureSnapshot:
    as_of: str
    market: str
    universe_key: str
    average_return: float
    average_volatility: float
    breadth: float
    momentum_share: float
    breakout_ready_share: float
    fake_breakout_share: float
    median_drawdown: float
    average_liquidity: float
    average_volume_confirmation: float
    macro_score: float
    macro_target_gross_exposure: float
    sample_count: int
    diagnostics: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RegimeSignal:
    as_of: str
    market: str
    universe_key: str
    dominant_regime: str
    probabilities: dict[str, float]
    transition_matrix: dict[str, dict[str, float]]
    confidence: float
    transition_risk: float
    risk_on_score: float
    volatility_score: float
    pressure_score: float
    suggested_gross_exposure_cap: float
    suggested_max_single_weight: float
    turnover_cap: float | None
    feature_snapshot: dict[str, Any]
    diagnostic_notes: list[str]
    schema_version: str = "2026-06-25.markov-regime.v1"
    execution_mode: str = "production"
    enabled: bool = True
    status: str = "active"
    regime_scope: str = "full_market"
    scope_key: str = ""
    base_universe_key: str = ""
    source_universe_key: str = ""
    requested_symbol_count: int = 0
    source_symbol_count: int = 0
    explicit_symbol_count: int = 0
    unsampled_symbol_count: int = 0
    sampled: bool = False
    production_eligible: bool = True
    source_description: str = ""
    history_record_count: int = 0
    transition_matrix_source: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
