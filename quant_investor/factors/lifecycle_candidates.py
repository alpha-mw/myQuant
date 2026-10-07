"""Definitions for the first prospective Factor candidate batch.

Price-volume candidates in this module are also registered in the prospective
half of ``quant_investor.factors.governance.implementations``. The Bootstrap
tree replay still reads only LOW/W80. The fundamental candidate stays
research-only until a governed ``FUNDAMENTAL`` source role exists. This module
remains the single definition source for hypotheses, trial accounting and
research replay; it grants no admission, selection or production weight.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any, Final

import numpy as np
import pandas as pd

from .governance.bootstrap import (
    BLEND_W75_CONTROL,
    BLEND_W80,
    CANONICAL_PARQUET as BOOTSTRAP_PARQUET_FORMAT,
    LOW_DOLLAR_VOLUME,
)

# Research replay historically used this label; production installed signals use
# the Bootstrap Parquet provenance string. Both prove the same strict source.
CANONICAL_PARQUET: Final = "canonical_parquet"
_ALLOWED_SOURCE_FORMATS: Final = frozenset({CANONICAL_PARQUET, BOOTSTRAP_PARQUET_FORMAT})
CANDIDATE_BATCH_ID: Final = "factor-candidates-2026q4-b1"
CANDIDATE_AUTHORITY: Final = "NON_AUTHORIZING_RESEARCH_CANDIDATE"
PIT_CLASSIFICATION: Final = "PIT"

VOLATILITY_PENALTY_5D: Final = "pv_volatility_penalty_5d"
VOLATILITY_PENALTY_10D: Final = "pv_volatility_penalty_10d"
DOWNSIDE_VOLATILITY_20D: Final = "pv_downside_volatility_20d"
SHORT_REVERSAL_5D: Final = "pv_short_reversal_5d"
MAX_RETURN_20D: Final = "pv_max_return_20d"
OCF_TO_PROFIT: Final = "fund_fin_ocf_to_profit"

PRICE_VOLUME_INPUTS: Final = ("trade_date", "adj_close")
MARKET_SOURCE_ROLES: Final = ("EXCHANGE_CALENDAR", "MARKET", "PIT_MEMBERSHIP")
FUNDAMENTAL_SOURCE_ROLES: Final = (
    "EXCHANGE_CALENDAR",
    "MARKET",
    "PIT_MEMBERSHIP",
    "FUNDAMENTAL",
)


class LifecycleCandidateError(ValueError):
    """Raised when a candidate input cannot be proven strict or point-in-time."""


@dataclass(frozen=True)
class CandidateDefinition:
    factor_id: str
    family: str
    direction: str
    hypothesis: str
    operator: str
    window_open_sessions: int | None
    input_fields: tuple[str, ...]
    required_source_roles: tuple[str, ...]
    references: tuple[str, ...]

    def normalized_expression(self) -> str:
        body: dict[str, Any] = {"input": list(self.input_fields), "operator": self.operator}
        if self.window_open_sessions is not None:
            body["window_open_sessions"] = self.window_open_sessions
        return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def as_row(self) -> dict[str, Any]:
        return {
            "batch_id": CANDIDATE_BATCH_ID,
            "factor_id": self.factor_id,
            "family": self.family,
            "direction": self.direction,
            "hypothesis": self.hypothesis,
            "normalized_expression": self.normalized_expression(),
            "input_fields": list(self.input_fields),
            "required_source_roles": list(self.required_source_roles),
            "references": list(self.references),
            "authority": CANDIDATE_AUTHORITY,
        }


CANDIDATES: Final[tuple[CandidateDefinition, ...]] = (
    CandidateDefinition(
        factor_id=VOLATILITY_PENALTY_5D,
        family="LOW_VOLATILITY",
        direction="HIGHER_IS_BETTER",
        hypothesis=(
            "Low-risk anomaly: leverage-constrained and lottery-seeking investors "
            "overpay for volatile names, so recent low realized volatility earns more."
        ),
        operator="NEGATIVE_RETURN_STD",
        window_open_sessions=5,
        input_fields=PRICE_VOLUME_INPUTS,
        required_source_roles=MARKET_SOURCE_ROLES,
        references=("Ang, Hodrick, Xing and Zhang 2006", "Frazzini and Pedersen 2014"),
    ),
    CandidateDefinition(
        factor_id=VOLATILITY_PENALTY_10D,
        family="LOW_VOLATILITY",
        direction="HIGHER_IS_BETTER",
        hypothesis=(
            "Same low-risk mechanism over a two-week window; preregistered as a "
            "window variant of the 5-session candidate, not an independent idea."
        ),
        operator="NEGATIVE_RETURN_STD",
        window_open_sessions=10,
        input_fields=PRICE_VOLUME_INPUTS,
        required_source_roles=MARKET_SOURCE_ROLES,
        references=("Ang, Hodrick, Xing and Zhang 2006",),
    ),
    CandidateDefinition(
        factor_id=DOWNSIDE_VOLATILITY_20D,
        family="LOW_VOLATILITY",
        direction="HIGHER_IS_BETTER",
        hypothesis=(
            "Downside risk is what retail-dominated A-share holders capitulate on; "
            "names with small recent downside dispersion carry less crash risk."
        ),
        operator="NEGATIVE_DOWNSIDE_RETURN_STD",
        window_open_sessions=20,
        input_fields=PRICE_VOLUME_INPUTS,
        required_source_roles=MARKET_SOURCE_ROLES,
        references=("Ang, Chen and Xing 2006",),
    ),
    CandidateDefinition(
        factor_id=SHORT_REVERSAL_5D,
        family="SHORT_TERM_REVERSAL",
        direction="HIGHER_IS_BETTER",
        hypothesis=(
            "Liquidity provision and retail overreaction: one-week losers rebound, "
            "a well documented and persistent effect in A-shares."
        ),
        operator="NEGATIVE_WINDOW_RETURN",
        window_open_sessions=5,
        input_fields=PRICE_VOLUME_INPUTS,
        required_source_roles=MARKET_SOURCE_ROLES,
        references=("Jegadeesh 1990", "Lehmann 1990"),
    ),
    CandidateDefinition(
        factor_id=MAX_RETURN_20D,
        family="LOTTERY",
        direction="HIGHER_IS_BETTER",
        hypothesis=(
            "Lottery preference: stocks with an extreme recent daily gain are "
            "overpriced and subsequently underperform (MAX effect)."
        ),
        operator="NEGATIVE_MAX_DAILY_RETURN",
        window_open_sessions=20,
        input_fields=PRICE_VOLUME_INPUTS,
        required_source_roles=MARKET_SOURCE_ROLES,
        references=("Bali, Cakici and Whitelaw 2011",),
    ),
    CandidateDefinition(
        factor_id=OCF_TO_PROFIT,
        family="EARNINGS_QUALITY",
        direction="HIGHER_IS_BETTER",
        hypothesis=(
            "Accrual anomaly: profit backed by operating cash flow persists, while "
            "accrual-heavy earnings mean-revert and are mispriced."
        ),
        operator="PIT_LATEST_AVAILABLE_VALUE",
        window_open_sessions=None,
        input_fields=("symbol", "value", "available_date", "source_classification"),
        required_source_roles=FUNDAMENTAL_SOURCE_ROLES,
        references=("Sloan 1996",),
    ),
)

_BY_ID: Final = {row.factor_id: row for row in CANDIDATES}
PRICE_VOLUME_CANDIDATES: Final = tuple(
    row.factor_id for row in CANDIDATES if row.operator != "PIT_LATEST_AVAILABLE_VALUE"
)


def candidate_definitions() -> list[dict[str, Any]]:
    """Return the batch rows in UTF-8 factor-ID order."""

    return [
        _BY_ID[factor_id].as_row()
        for factor_id in sorted(_BY_ID, key=lambda value: value.encode("utf-8"))
    ]


def batch_trial_accounting() -> dict[str, Any]:
    """Nominal and family-level trial counts that a DSR over this batch must charge."""

    families: dict[str, list[str]] = {}
    for row in CANDIDATES:
        families.setdefault(row.family, []).append(row.factor_id)
    return {
        "batch_id": CANDIDATE_BATCH_ID,
        "nominal_trial_count": len(CANDIDATES),
        "family_count": len(families),
        "families": {name: sorted(ids) for name, ids in sorted(families.items())},
        "bootstrap_ids_excluded": sorted([LOW_DOLLAR_VOLUME, BLEND_W80, BLEND_W75_CONTROL]),
    }


def require_candidate(factor_id: str) -> CandidateDefinition:
    """Return one candidate definition or fail closed."""

    definition = _BY_ID.get(factor_id)
    if definition is None:
        raise LifecycleCandidateError(f"unknown lifecycle candidate: {factor_id}")
    return definition


def _require_candidate(factor_id: str) -> CandidateDefinition:
    return require_candidate(factor_id)


def candidate_panel(factor_id: str, adj_close: pd.DataFrame) -> pd.DataFrame:
    """Compute one price-volume candidate on a ``date x symbol`` adjusted-close panel.

    A value exists only when every return in its window is finite; there is no
    shorter-window or forward-filled substitute.
    """

    definition = _require_candidate(factor_id)
    window = definition.window_open_sessions
    if window is None:
        raise LifecycleCandidateError(f"{factor_id} is not a price-volume candidate")
    if not isinstance(adj_close, pd.DataFrame) or adj_close.empty:
        raise LifecycleCandidateError("adjusted-close panel is empty")
    close = adj_close.astype(float).where(adj_close.astype(float) > 0.0)
    returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    if definition.operator == "NEGATIVE_RETURN_STD":
        return -returns.rolling(window, min_periods=window).std(ddof=1)
    if definition.operator == "NEGATIVE_DOWNSIDE_RETURN_STD":
        downside = returns.clip(upper=0.0).where(returns.notna())
        return -downside.rolling(window, min_periods=window).std(ddof=1)
    if definition.operator == "NEGATIVE_WINDOW_RETURN":
        return -(close / close.shift(window) - 1.0)
    if definition.operator == "NEGATIVE_MAX_DAILY_RETURN":
        return -returns.rolling(window, min_periods=window).max()
    raise LifecycleCandidateError(f"unsupported candidate operator {definition.operator}")


def _strict_close_panel(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    if not isinstance(frames, Mapping) or not frames:
        raise LifecycleCandidateError("candidate signals require in-memory symbol frames")
    columns: dict[str, pd.Series] = {}
    for symbol in sorted((str(value) for value in frames), key=lambda value: value.encode()):
        frame = frames[symbol]
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise LifecycleCandidateError(f"strict candidate frame is empty: {symbol}")
        missing = [field for field in PRICE_VOLUME_INPUTS if field not in frame.columns]
        if missing:
            raise LifecycleCandidateError(f"strict candidate input is missing {missing}")
        dates = pd.to_datetime(frame["trade_date"], errors="coerce")
        if dates.isna().any() or dates.duplicated().any():
            raise LifecycleCandidateError(f"strict trade_date is invalid or duplicated: {symbol}")
        values = pd.to_numeric(frame["adj_close"], errors="coerce").astype(float)
        columns[symbol] = pd.Series(values.to_numpy(), index=dates).sort_index()
    return pd.DataFrame(columns).sort_index()


def compute_price_volume_candidates(
    frames: Mapping[str, pd.DataFrame],
    *,
    source_format: str,
    factor_ids: Sequence[str] | None = None,
) -> dict[str, pd.Series]:
    """Latest-session cross-section for each requested price-volume candidate."""

    if source_format not in _ALLOWED_SOURCE_FORMATS:
        raise LifecycleCandidateError("candidate signals require canonical Parquet provenance")
    requested = list(PRICE_VOLUME_CANDIDATES if factor_ids is None else factor_ids)
    if not requested or len(requested) != len(set(requested)):
        raise LifecycleCandidateError("candidate request is empty or duplicated")
    for factor_id in requested:
        if factor_id not in PRICE_VOLUME_CANDIDATES:
            raise LifecycleCandidateError(f"{factor_id} is not a price-volume candidate")
    panel = _strict_close_panel(frames)
    return {
        factor_id: candidate_panel(factor_id, panel).iloc[-1].astype(float)
        for factor_id in requested
    }


def compute_pit_fundamental_candidate(
    rows: pd.DataFrame,
    *,
    as_of: str,
    factor_id: str = OCF_TO_PROFIT,
) -> pd.Series:
    """Latest value per symbol that was already available at ``as_of``.

    Every row must declare its ``available_date`` and a ``PIT`` source
    classification.  A row that became available after ``as_of`` is rejected
    outright rather than silently dropped, because its presence means the caller
    assembled the input with look-ahead.
    """

    definition = _require_candidate(factor_id)
    if definition.operator != "PIT_LATEST_AVAILABLE_VALUE":
        raise LifecycleCandidateError(f"{factor_id} is not a PIT fundamental candidate")
    if not isinstance(rows, pd.DataFrame) or rows.empty:
        raise LifecycleCandidateError("PIT fundamental input is empty")
    missing = [field for field in definition.input_fields if field not in rows.columns]
    if missing:
        raise LifecycleCandidateError(f"PIT fundamental input is missing {missing}")
    cutoff = pd.Timestamp(str(as_of))
    available = pd.to_datetime(rows["available_date"], errors="coerce")
    if available.isna().any():
        raise LifecycleCandidateError("PIT fundamental row lacks an available_date")
    if (rows["source_classification"].astype(str) != PIT_CLASSIFICATION).any():
        raise LifecycleCandidateError("fundamental input is not PIT-classified")
    if (available > cutoff).any():
        raise LifecycleCandidateError("PIT fundamental input contains rows after as_of")
    values = pd.to_numeric(rows["value"], errors="coerce").astype(float)
    frame = pd.DataFrame(
        {"symbol": rows["symbol"].astype(str), "available": available, "value": values}
    )
    frame = frame[np.isfinite(frame["value"])]
    if frame.empty:
        raise LifecycleCandidateError("PIT fundamental input has no finite value")
    latest = frame.sort_values(["symbol", "available"], kind="mergesort").groupby("symbol").tail(1)
    series = pd.Series(latest["value"].to_numpy(), index=latest["symbol"].to_numpy(), dtype=float)
    return series.sort_index(key=lambda index: [value.encode() for value in index])


__all__ = [
    "CANDIDATES",
    "CANDIDATE_AUTHORITY",
    "CANDIDATE_BATCH_ID",
    "CANONICAL_PARQUET",
    "CandidateDefinition",
    "LifecycleCandidateError",
    "OCF_TO_PROFIT",
    "PRICE_VOLUME_CANDIDATES",
    "batch_trial_accounting",
    "candidate_definitions",
    "candidate_panel",
    "compute_pit_fundamental_candidate",
    "compute_price_volume_candidates",
    "require_candidate",
]
