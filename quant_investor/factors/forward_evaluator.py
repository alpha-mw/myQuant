"""Deterministic, side-effect-free helpers for forward factor evidence.

This module computes diagnostic evidence only.  It does not read market data,
publish receipts, mutate a factor registry, or grant production authority.
Missing prerequisites are represented by :class:`Availability` with status
``UNAVAILABLE``; malformed caller input raises ``ValueError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
import math
import re
from statistics import fmean, stdev
from typing import Any, Generic, Mapping, Sequence, TypeVar
from zoneinfo import ZoneInfo

T = TypeVar("T")

SHANGHAI_TIMEZONE = ZoneInfo("Asia/Shanghai")
SHANGHAI_CLOSE = time(15, 0)
SUPPORTED_HORIZONS = (1, 5, 10, 20, 60)
CHALLENGER_HORIZON = 20
ROUND_TRIP_COST_BPS = 20.0
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class EvidenceStatus(str, Enum):
    """Typed result status for pure forward-evidence helpers."""

    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    DUPLICATE_ORIGIN_CONFLICT = "DUPLICATE_ORIGIN_CONFLICT"


class FactorTier(str, Enum):
    """Research/production allocation tiers."""

    CORE = "CORE"
    CHALLENGER = "CHALLENGER"
    EXPERIMENTAL = "EXPERIMENTAL"


@dataclass(frozen=True)
class Availability(Generic[T]):
    """A typed value that is either available or explicitly unavailable."""

    status: EvidenceStatus
    value: T | None = None
    blockers: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return self.status is EvidenceStatus.AVAILABLE


def _available(value: T) -> Availability[T]:
    return Availability(status=EvidenceStatus.AVAILABLE, value=value)


def _unavailable(*blockers: str) -> Availability[Any]:
    normalized = tuple(dict.fromkeys(blockers or ("missing_prerequisite",)))
    return Availability(status=EvidenceStatus.UNAVAILABLE, blockers=normalized)


@dataclass(frozen=True)
class ForwardDiagnosticReceipt:
    """Exact report-only inputs used to assess challenger eligibility."""

    factor_name: str
    definition_sha256: str | None = None
    factor_set_sha256: str | None = None
    quant_policy_sha256: str | None = None
    source_lineage_sha256: str | None = None
    horizon_sessions: int | None = None
    origin_count: int | None = None
    minimum_symbols_per_origin: int | None = None
    mean_rank_ic: float | None = None
    annualized_rank_ic_ir: float | None = None
    cost_adjusted_group_return: float | None = None
    stability: float | None = None
    max_abs_existing_factor_correlation: float | None = None
    freshness_open_sessions: int | None = None
    status: EvidenceStatus = EvidenceStatus.AVAILABLE


@dataclass(frozen=True)
class FactorTierInput:
    """One factor's explicit inputs to tier allocation."""

    factor_name: str
    diagnostic_receipt: ForwardDiagnosticReceipt | None = None
    production_active_set_member: bool = False
    activation_closure: bool = False
    health_closure: bool = False


@dataclass(frozen=True)
class FactorTierDecision:
    """Deterministic tier decision for one factor."""

    factor_name: str
    tier: FactorTier
    status: EvidenceStatus
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class TierAllocation:
    """ASCII-sorted factor names partitioned into exactly one tier."""

    core: tuple[str, ...]
    challenger: tuple[str, ...]
    experimental: tuple[str, ...]
    decisions: tuple[FactorTierDecision, ...]


def _is_sha256(value: str | None) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _receipt_missing_fields(receipt: ForwardDiagnosticReceipt) -> tuple[str, ...]:
    fields: tuple[tuple[str, object], ...] = (
        ("definition_sha256", receipt.definition_sha256),
        ("factor_set_sha256", receipt.factor_set_sha256),
        ("quant_policy_sha256", receipt.quant_policy_sha256),
        ("source_lineage_sha256", receipt.source_lineage_sha256),
        ("horizon_sessions", receipt.horizon_sessions),
        ("origin_count", receipt.origin_count),
        ("minimum_symbols_per_origin", receipt.minimum_symbols_per_origin),
        ("mean_rank_ic", receipt.mean_rank_ic),
        ("annualized_rank_ic_ir", receipt.annualized_rank_ic_ir),
        ("cost_adjusted_group_return", receipt.cost_adjusted_group_return),
        ("stability", receipt.stability),
        (
            "max_abs_existing_factor_correlation",
            receipt.max_abs_existing_factor_correlation,
        ),
        ("freshness_open_sessions", receipt.freshness_open_sessions),
    )
    return tuple(name for name, value in fields if value is None)


def _receipt_invalid_fields(receipt: ForwardDiagnosticReceipt) -> tuple[str, ...]:
    invalid: list[str] = []
    for sha_name, sha_value in (
        ("definition_sha256", receipt.definition_sha256),
        ("factor_set_sha256", receipt.factor_set_sha256),
        ("quant_policy_sha256", receipt.quant_policy_sha256),
        ("source_lineage_sha256", receipt.source_lineage_sha256),
    ):
        if sha_value is not None and not _is_sha256(sha_value):
            invalid.append(sha_name)
    for metric_name, metric_value in (
        ("mean_rank_ic", receipt.mean_rank_ic),
        ("annualized_rank_ic_ir", receipt.annualized_rank_ic_ir),
        ("cost_adjusted_group_return", receipt.cost_adjusted_group_return),
        ("stability", receipt.stability),
        (
            "max_abs_existing_factor_correlation",
            receipt.max_abs_existing_factor_correlation,
        ),
    ):
        if metric_value is not None and (
            isinstance(metric_value, bool)
            or not isinstance(metric_value, (int, float))
            or not math.isfinite(metric_value)
        ):
            invalid.append(metric_name)
    for count_name, count_value in (
        ("horizon_sessions", receipt.horizon_sessions),
        ("origin_count", receipt.origin_count),
        ("minimum_symbols_per_origin", receipt.minimum_symbols_per_origin),
        ("freshness_open_sessions", receipt.freshness_open_sessions),
    ):
        if count_value is not None and (
            isinstance(count_value, bool) or not isinstance(count_value, int) or count_value < 0
        ):
            invalid.append(count_name)
    if receipt.stability is not None and not 0.0 <= receipt.stability <= 1.0:
        invalid.append("stability")
    if (
        receipt.max_abs_existing_factor_correlation is not None
        and not 0.0 <= receipt.max_abs_existing_factor_correlation <= 1.0
    ):
        invalid.append("max_abs_existing_factor_correlation")
    return tuple(dict.fromkeys(invalid))


def evaluate_factor_tier(
    factor_name: str,
    diagnostic_receipt: ForwardDiagnosticReceipt | None = None,
    *,
    production_active_set_member: bool = False,
    activation_closure: bool = False,
    health_closure: bool = False,
) -> FactorTierDecision:
    """Evaluate one factor without inferring absent governance evidence.

    ``CORE`` requires all three explicit production closures.  A non-core
    factor is ``CHALLENGER`` only when its exact 20-session diagnostic receipt
    passes every threshold; it otherwise remains ``EXPERIMENTAL``.
    """

    if not isinstance(factor_name, str) or not factor_name:
        raise ValueError("factor_name must be a non-empty string")
    for label, value in (
        ("production_active_set_member", production_active_set_member),
        ("activation_closure", activation_closure),
        ("health_closure", health_closure),
    ):
        if type(value) is not bool:
            raise ValueError(f"{label} must be bool")

    if production_active_set_member and activation_closure and health_closure:
        return FactorTierDecision(
            factor_name=factor_name,
            tier=FactorTier.CORE,
            status=EvidenceStatus.AVAILABLE,
        )

    if diagnostic_receipt is None:
        return FactorTierDecision(
            factor_name=factor_name,
            tier=FactorTier.EXPERIMENTAL,
            status=EvidenceStatus.UNAVAILABLE,
            blockers=("diagnostic_receipt_missing",),
        )
    if diagnostic_receipt.factor_name != factor_name:
        raise ValueError("diagnostic receipt factor_name mismatch")
    if diagnostic_receipt.status is not EvidenceStatus.AVAILABLE:
        return FactorTierDecision(
            factor_name=factor_name,
            tier=FactorTier.EXPERIMENTAL,
            status=EvidenceStatus.UNAVAILABLE,
            blockers=("diagnostic_receipt_unavailable",),
        )

    missing = _receipt_missing_fields(diagnostic_receipt)
    invalid = _receipt_invalid_fields(diagnostic_receipt)
    if missing or invalid:
        blockers = tuple(f"missing:{name}" for name in missing) + tuple(
            f"invalid:{name}" for name in invalid
        )
        return FactorTierDecision(
            factor_name=factor_name,
            tier=FactorTier.EXPERIMENTAL,
            status=EvidenceStatus.UNAVAILABLE,
            blockers=blockers,
        )

    threshold_failures: list[str] = []
    if diagnostic_receipt.horizon_sessions != CHALLENGER_HORIZON:
        threshold_failures.append("horizon_sessions_not_20")
    if diagnostic_receipt.origin_count < 60:  # type: ignore[operator]
        threshold_failures.append("origin_count_below_60")
    if diagnostic_receipt.minimum_symbols_per_origin < 100:  # type: ignore[operator]
        threshold_failures.append("minimum_symbols_per_origin_below_100")
    if diagnostic_receipt.mean_rank_ic <= 0.02:  # type: ignore[operator]
        threshold_failures.append("mean_rank_ic_not_above_0_02")
    if diagnostic_receipt.annualized_rank_ic_ir < 0.5:  # type: ignore[operator]
        threshold_failures.append("annualized_rank_ic_ir_below_0_5")
    if diagnostic_receipt.cost_adjusted_group_return <= 0.0:  # type: ignore[operator]
        threshold_failures.append("cost_adjusted_group_return_not_positive")
    if diagnostic_receipt.stability < 0.60:  # type: ignore[operator]
        threshold_failures.append("stability_below_0_60")
    if diagnostic_receipt.max_abs_existing_factor_correlation >= 0.70:  # type: ignore[operator]
        threshold_failures.append("max_abs_existing_factor_correlation_not_below_0_70")
    if diagnostic_receipt.freshness_open_sessions > 5:  # type: ignore[operator]
        threshold_failures.append("freshness_above_5_open_sessions")

    return FactorTierDecision(
        factor_name=factor_name,
        tier=FactorTier.EXPERIMENTAL if threshold_failures else FactorTier.CHALLENGER,
        status=EvidenceStatus.AVAILABLE,
        blockers=tuple(threshold_failures),
    )


def allocate_factor_tiers(inputs: Sequence[FactorTierInput]) -> TierAllocation:
    """Partition unique factor names into deterministic ASCII-sorted tiers."""

    if isinstance(inputs, (str, bytes)):
        raise ValueError("inputs must be a sequence of FactorTierInput")
    decisions: list[FactorTierDecision] = []
    seen: set[str] = set()
    for item in inputs:
        if not isinstance(item, FactorTierInput):
            raise ValueError("inputs must contain only FactorTierInput")
        if item.factor_name in seen:
            raise ValueError(f"duplicate factor_name: {item.factor_name}")
        seen.add(item.factor_name)
        decisions.append(
            evaluate_factor_tier(
                item.factor_name,
                item.diagnostic_receipt,
                production_active_set_member=item.production_active_set_member,
                activation_closure=item.activation_closure,
                health_closure=item.health_closure,
            )
        )
    decisions.sort(key=lambda row: row.factor_name)
    return TierAllocation(
        core=tuple(row.factor_name for row in decisions if row.tier is FactorTier.CORE),
        challenger=tuple(row.factor_name for row in decisions if row.tier is FactorTier.CHALLENGER),
        experimental=tuple(
            row.factor_name for row in decisions if row.tier is FactorTier.EXPERIMENTAL
        ),
        decisions=tuple(decisions),
    )


def _normalize_session(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date string") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{label} must be canonical YYYY-MM-DD")
    return value


def shanghai_horizon_end_sessions(
    origin_session: str,
    open_sessions: Sequence[str],
    *,
    horizons: Sequence[int] = SUPPORTED_HORIZONS,
) -> Availability[dict[int, str]]:
    """Resolve exact future end sessions from a strict Shanghai open calendar."""

    origin = _normalize_session(origin_session, label="origin_session")
    if isinstance(open_sessions, (str, bytes)) or not open_sessions:
        return _unavailable("open_sessions_missing")
    calendar = tuple(_normalize_session(session, label="open_session") for session in open_sessions)
    if tuple(sorted(set(calendar))) != calendar:
        raise ValueError("open_sessions must be strictly increasing and unique")
    requested = tuple(horizons)
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(type(horizon) is not int or horizon <= 0 for horizon in requested)
    ):
        raise ValueError("horizons must be unique positive integers")
    if any(horizon not in SUPPORTED_HORIZONS for horizon in requested):
        raise ValueError(f"horizons must be drawn from {SUPPORTED_HORIZONS}")
    try:
        origin_index = calendar.index(origin)
    except ValueError:
        return _unavailable("origin_session_not_in_open_calendar")
    missing = tuple(horizon for horizon in requested if origin_index + horizon >= len(calendar))
    if missing:
        return _unavailable(*(f"horizon_{horizon}_end_session_missing" for horizon in missing))
    return _available({horizon: calendar[origin_index + horizon] for horizon in requested})


resolve_horizon_end_sessions = shanghai_horizon_end_sessions


def open_session_freshness(
    source_session: str | None,
    as_of_session: str | None,
    open_sessions: Sequence[str] | None,
) -> Availability[int]:
    """Count exact Shanghai open sessions from source to as-of (same day is zero)."""

    blockers = []
    if source_session is None:
        blockers.append("source_session_missing")
    if as_of_session is None:
        blockers.append("as_of_session_missing")
    if open_sessions is None or not open_sessions:
        blockers.append("open_sessions_missing")
    if blockers:
        return _unavailable(*blockers)
    assert source_session is not None and as_of_session is not None
    assert open_sessions is not None
    source = _normalize_session(source_session, label="source_session")
    as_of = _normalize_session(as_of_session, label="as_of_session")
    if isinstance(open_sessions, (str, bytes)):
        raise ValueError("open_sessions must be a date sequence")
    calendar = tuple(_normalize_session(session, label="open_session") for session in open_sessions)
    if tuple(sorted(set(calendar))) != calendar:
        raise ValueError("open_sessions must be strictly increasing and unique")
    if source not in calendar:
        return _unavailable("source_session_not_in_open_calendar")
    if as_of not in calendar:
        return _unavailable("as_of_session_not_in_open_calendar")
    source_index = calendar.index(source)
    as_of_index = calendar.index(as_of)
    if source_index > as_of_index:
        return _unavailable("source_session_after_as_of_session")
    return _available(as_of_index - source_index)


def _finite_number(value: object, *, label: str) -> Availability[float]:
    if value is None:
        return _unavailable(f"{label}_missing")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return _available(number)


def adjusted_close_simple_return(
    start_adjusted_close: float | None,
    end_adjusted_close: float | None,
) -> Availability[float]:
    """Compute ``end / start - 1`` from two positive adjusted closes."""

    start = _finite_number(start_adjusted_close, label="start_adjusted_close")
    end = _finite_number(end_adjusted_close, label="end_adjusted_close")
    blockers = start.blockers + end.blockers
    if blockers:
        return _unavailable(*blockers)
    assert start.value is not None and end.value is not None
    if start.value <= 0.0 or end.value <= 0.0:
        raise ValueError("adjusted closes must be positive")
    return _available(end.value / start.value - 1.0)


simple_adjusted_close_return = adjusted_close_simple_return


def adjusted_return(
    asset_return: float | None,
    reference_return: float | None,
    *,
    reference_label: str,
) -> Availability[float]:
    """Subtract a market or industry reference return from an asset return."""

    asset = _finite_number(asset_return, label="asset_return")
    reference = _finite_number(reference_return, label=reference_label)
    blockers = asset.blockers + reference.blockers
    if blockers:
        return _unavailable(*blockers)
    assert asset.value is not None and reference.value is not None
    return _available(asset.value - reference.value)


def market_adjusted_return(
    asset_return: float | None,
    market_return: float | None,
) -> Availability[float]:
    return adjusted_return(asset_return, market_return, reference_label="market_return")


def industry_adjusted_return(
    asset_return: float | None,
    industry_return: float | None,
) -> Availability[float]:
    return adjusted_return(asset_return, industry_return, reference_label="industry_return")


@dataclass(frozen=True)
class AdjustedReturns:
    simple_return: float
    market_adjusted_return: float
    industry_adjusted_return: float


def market_industry_adjusted_returns(
    asset_return: float | None,
    market_return: float | None,
    industry_return: float | None,
) -> Availability[AdjustedReturns]:
    market = market_adjusted_return(asset_return, market_return)
    industry = industry_adjusted_return(asset_return, industry_return)
    asset = _finite_number(asset_return, label="asset_return")
    blockers = asset.blockers + market.blockers + industry.blockers
    if blockers:
        return _unavailable(*blockers)
    assert asset.value is not None and market.value is not None and industry.value is not None
    return _available(
        AdjustedReturns(
            simple_return=asset.value,
            market_adjusted_return=market.value,
            industry_adjusted_return=industry.value,
        )
    )


def label_maturity(
    end_session: str | None,
    observed_at: datetime | None,
) -> Availability[bool]:
    """Return whether the exact end session has reached 15:00 Shanghai time."""

    if end_session is None or observed_at is None:
        blockers = []
        if end_session is None:
            blockers.append("end_session_missing")
        if observed_at is None:
            blockers.append("observed_at_missing")
        return _unavailable(*blockers)
    end = _normalize_session(end_session, label="end_session")
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware")
    close_at = datetime.combine(
        date.fromisoformat(end),
        SHANGHAI_CLOSE,
        tzinfo=SHANGHAI_TIMEZONE,
    )
    return _available(observed_at.astimezone(SHANGHAI_TIMEZONE) >= close_at)


is_label_matured = label_maturity


def _finite_vector(
    values: Sequence[float | None] | None,
    *,
    label: str,
    minimum: int,
) -> Availability[tuple[float, ...]]:
    if values is None:
        return _unavailable(f"{label}_missing")
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a numeric sequence")
    if len(values) < minimum:
        return _unavailable(f"{label}_requires_at_least_{minimum}")
    normalized: list[float] = []
    for value in values:
        if value is None:
            return _unavailable(f"{label}_contains_missing")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must contain only finite numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} must contain only finite numbers")
        normalized.append(number)
    return _available(tuple(normalized))


def pearson_ic(
    factor_values: Sequence[float | None] | None,
    forward_returns: Sequence[float | None] | None,
) -> Availability[float]:
    """Compute per-origin Pearson information coefficient."""

    factors = _finite_vector(factor_values, label="factor_values", minimum=2)
    returns = _finite_vector(forward_returns, label="forward_returns", minimum=2)
    blockers = factors.blockers + returns.blockers
    if blockers:
        return _unavailable(*blockers)
    assert factors.value is not None and returns.value is not None
    if len(factors.value) != len(returns.value):
        return _unavailable("factor_return_length_mismatch")
    mean_factor = fmean(factors.value)
    mean_return = fmean(returns.value)
    factor_deviations = tuple(value - mean_factor for value in factors.value)
    return_deviations = tuple(value - mean_return for value in returns.value)
    factor_ss = sum(value * value for value in factor_deviations)
    return_ss = sum(value * value for value in return_deviations)
    if factor_ss == 0.0 or return_ss == 0.0:
        return _unavailable("pearson_zero_variance")
    numerator = sum(
        factor * outcome for factor, outcome in zip(factor_deviations, return_deviations)
    )
    return _available(numerator / math.sqrt(factor_ss * return_ss))


per_origin_pearson_ic = pearson_ic


def average_tie_ranks(
    values: Sequence[float | None] | None,
) -> Availability[tuple[float, ...]]:
    """Return one-based ranks using the average rank for exact ties."""

    normalized = _finite_vector(values, label="rank_values", minimum=2)
    if not normalized.available:
        return Availability(
            status=normalized.status,
            blockers=normalized.blockers,
        )
    assert normalized.value is not None
    indexed = sorted(enumerate(normalized.value), key=lambda row: (row[1], row[0]))
    ranks = [0.0] * len(indexed)
    start = 0
    while start < len(indexed):
        stop = start + 1
        while stop < len(indexed) and indexed[stop][1] == indexed[start][1]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        for offset in range(start, stop):
            ranks[indexed[offset][0]] = average_rank
        start = stop
    return _available(tuple(ranks))


def spearman_rank_ic(
    factor_values: Sequence[float | None] | None,
    forward_returns: Sequence[float | None] | None,
) -> Availability[float]:
    """Compute per-origin Spearman RankIC with average ranks for ties."""

    factor_ranks = average_tie_ranks(factor_values)
    return_ranks = average_tie_ranks(forward_returns)
    blockers = factor_ranks.blockers + return_ranks.blockers
    if blockers:
        return _unavailable(*blockers)
    assert factor_ranks.value is not None and return_ranks.value is not None
    return pearson_ic(factor_ranks.value, return_ranks.value)


per_origin_spearman_rank_ic = spearman_rank_ic


@dataclass(frozen=True)
class OriginInformationCoefficients:
    pearson_ic: float
    spearman_rank_ic: float


def per_origin_information_coefficients(
    factor_values: Sequence[float | None] | None,
    forward_returns: Sequence[float | None] | None,
) -> Availability[OriginInformationCoefficients]:
    pearson = pearson_ic(factor_values, forward_returns)
    spearman = spearman_rank_ic(factor_values, forward_returns)
    blockers = pearson.blockers + spearman.blockers
    if blockers:
        return _unavailable(*blockers)
    assert pearson.value is not None and spearman.value is not None
    return _available(
        OriginInformationCoefficients(
            pearson_ic=pearson.value,
            spearman_rank_ic=spearman.value,
        )
    )


def annualized_rank_ic_ir(
    rank_ics: Sequence[float | None] | None,
) -> Availability[float]:
    """Annualize mean RankIC over sample std with ``sqrt(252 / 20)``."""

    normalized = _finite_vector(rank_ics, label="rank_ics", minimum=2)
    if not normalized.available:
        return Availability(
            status=normalized.status,
            blockers=normalized.blockers,
        )
    assert normalized.value is not None
    sample_std = stdev(normalized.value)
    if sample_std == 0.0:
        return _unavailable("rank_ic_sample_std_zero")
    return _available(fmean(normalized.value) / sample_std * math.sqrt(252.0 / 20.0))


def apply_flat_round_trip_cost(
    gross_return: float | None,
    *,
    cost_bps: float = ROUND_TRIP_COST_BPS,
) -> Availability[float]:
    """Subtract one flat round-trip cost from a gross return."""

    gross = _finite_number(gross_return, label="gross_return")
    cost = _finite_number(cost_bps, label="cost_bps")
    blockers = gross.blockers + cost.blockers
    if blockers:
        return _unavailable(*blockers)
    assert gross.value is not None and cost.value is not None
    if cost.value < 0.0:
        raise ValueError("cost_bps must be non-negative")
    return _available(gross.value - cost.value / 10_000.0)


def cost_adjusted_top_bottom_quintile_return(
    factor_values: Sequence[float | None] | None,
    forward_returns: Sequence[float | None] | None,
    *,
    cost_bps: float = ROUND_TRIP_COST_BPS,
) -> Availability[float]:
    """Compute exact-size top-minus-bottom quintile return net of 20 bps."""

    factors = _finite_vector(factor_values, label="factor_values", minimum=5)
    returns = _finite_vector(forward_returns, label="forward_returns", minimum=5)
    blockers = factors.blockers + returns.blockers
    if blockers:
        return _unavailable(*blockers)
    assert factors.value is not None and returns.value is not None
    factor_vector = factors.value
    return_vector = returns.value
    if len(factor_vector) != len(return_vector):
        return _unavailable("factor_return_length_mismatch")
    group_size = len(factor_vector) // 5
    ordered = sorted(
        range(len(factor_vector)),
        key=lambda index: (factor_vector[index], index),
    )
    bottom = ordered[:group_size]
    top = ordered[-group_size:]
    gross = fmean(return_vector[index] for index in top) - fmean(
        return_vector[index] for index in bottom
    )
    return apply_flat_round_trip_cost(gross, cost_bps=cost_bps)


cost_adjusted_group_return = cost_adjusted_top_bottom_quintile_return


def top_quintile_turnover(
    previous_top: Sequence[str] | None,
    current_top: Sequence[str] | None,
) -> Availability[float]:
    """Compute ``1 - overlap / top_size`` for equal-sized top groups."""

    if previous_top is None or current_top is None:
        blockers = []
        if previous_top is None:
            blockers.append("previous_top_missing")
        if current_top is None:
            blockers.append("current_top_missing")
        return _unavailable(*blockers)
    if isinstance(previous_top, (str, bytes)) or isinstance(current_top, (str, bytes)):
        raise ValueError("top groups must be symbol sequences")
    previous = tuple(previous_top)
    current = tuple(current_top)
    if not previous or not current:
        return _unavailable("top_group_empty")
    if any(not isinstance(symbol, str) or not symbol for symbol in previous + current):
        raise ValueError("top group symbols must be non-empty strings")
    if len(set(previous)) != len(previous) or len(set(current)) != len(current):
        raise ValueError("top group symbols must be unique")
    if len(previous) != len(current):
        return _unavailable("top_group_size_mismatch")
    overlap = len(set(previous).intersection(current))
    return _available(1.0 - overlap / len(current))


turnover_one_minus_overlap = top_quintile_turnover


def top_quintile_capacity(
    top_quintile_adv: Sequence[float | None] | None,
) -> Availability[float]:
    """Return mean top-quintile ADV multiplied by one percent."""

    adv = _finite_vector(top_quintile_adv, label="top_quintile_adv", minimum=1)
    if not adv.available:
        return Availability(status=adv.status, blockers=adv.blockers)
    assert adv.value is not None
    if any(value < 0.0 for value in adv.value):
        raise ValueError("ADV values must be non-negative")
    return _available(fmean(adv.value) * 0.01)


capacity_mean_top_quintile_adv = top_quintile_capacity


def rank_ic_sign_stability(
    rank_ics: Sequence[float | None] | None,
) -> Availability[float]:
    """Return the fraction of RankIC origins sharing the aggregate sign."""

    normalized = _finite_vector(rank_ics, label="rank_ics", minimum=1)
    if not normalized.available:
        return Availability(status=normalized.status, blockers=normalized.blockers)
    assert normalized.value is not None
    aggregate = fmean(normalized.value)
    if aggregate == 0.0:
        return _unavailable("mean_rank_ic_zero_sign")
    if aggregate > 0.0:
        same_sign = sum(value > 0.0 for value in normalized.value)
    else:
        same_sign = sum(value < 0.0 for value in normalized.value)
    return _available(same_sign / len(normalized.value))


stability_same_rank_ic_sign = rank_ic_sign_stability


def max_abs_existing_factor_pearson(
    candidate_values: Sequence[float | None] | None,
    existing_factor_values: Mapping[str, Sequence[float | None]] | None,
) -> Availability[float]:
    """Return the maximum absolute Pearson correlation to existing factors."""

    if existing_factor_values is None or not existing_factor_values:
        return _unavailable("existing_factor_values_missing")
    if not isinstance(existing_factor_values, Mapping):
        raise ValueError("existing_factor_values must be a mapping")
    correlations: list[float] = []
    for name in sorted(existing_factor_values):
        if not isinstance(name, str) or not name:
            raise ValueError("existing factor names must be non-empty strings")
        correlation = pearson_ic(candidate_values, existing_factor_values[name])
        if not correlation.available:
            return _unavailable(*(f"{name}:{blocker}" for blocker in correlation.blockers))
        assert correlation.value is not None
        correlations.append(abs(correlation.value))
    return _available(max(correlations))


max_abs_existing_correlation = max_abs_existing_factor_pearson


@dataclass(frozen=True, order=True)
class OriginDedupKey:
    factor_name: str
    definition_sha256: str
    factor_set_sha256: str
    quant_policy_sha256: str
    horizon_sessions: int
    source_lineage_sha256: str


@dataclass(frozen=True)
class OriginObservationRecord:
    """Minimal canonical record accepted by the origin de-duplicator."""

    factor_name: str
    definition_sha256: str
    factor_set_sha256: str
    quant_policy_sha256: str
    horizon_sessions: int
    source_lineage_sha256: str
    observation_ref: str
    observation_byte_sha256: str
    observation_semantic_sha256: str

    @property
    def key(self) -> OriginDedupKey:
        return OriginDedupKey(
            factor_name=self.factor_name,
            definition_sha256=self.definition_sha256,
            factor_set_sha256=self.factor_set_sha256,
            quant_policy_sha256=self.quant_policy_sha256,
            horizon_sessions=self.horizon_sessions,
            source_lineage_sha256=self.source_lineage_sha256,
        )


@dataclass(frozen=True)
class DuplicateOrigin:
    key: OriginDedupKey
    kept_ref: str
    duplicate_refs: tuple[str, ...]


@dataclass(frozen=True)
class DuplicateOriginConflict:
    key: OriginDedupKey
    observation_refs: tuple[str, ...]
    byte_sha256_values: tuple[str, ...]
    semantic_sha256_values: tuple[str, ...]


@dataclass(frozen=True)
class OriginDeduplicationResult:
    status: EvidenceStatus
    records: tuple[OriginObservationRecord, ...] = ()
    duplicates: tuple[DuplicateOrigin, ...] = ()
    conflicts: tuple[DuplicateOriginConflict, ...] = ()
    blockers: tuple[str, ...] = ()

    @property
    def blocked(self) -> bool:
        return self.status is not EvidenceStatus.AVAILABLE


def _extract_record_field(record: object, names: Sequence[str]) -> object:
    if isinstance(record, Mapping):
        for name in names:
            if name in record:
                return record[name]
    else:
        for name in names:
            if hasattr(record, name):
                return getattr(record, name)
    return None


def _normalize_origin_record(
    record: object,
) -> Availability[OriginObservationRecord]:
    aliases: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("factor_name", ("factor_name", "name")),
        ("definition_sha256", ("definition_sha256",)),
        ("factor_set_sha256", ("factor_set_sha256",)),
        ("quant_policy_sha256", ("quant_policy_sha256", "policy_sha256")),
        ("horizon_sessions", ("horizon_sessions", "horizon")),
        ("source_lineage_sha256", ("source_lineage_sha256",)),
        ("observation_ref", ("observation_ref", "ref", "relative_path")),
        (
            "observation_byte_sha256",
            ("observation_byte_sha256", "observation_sha256", "byte_sha256"),
        ),
        (
            "observation_semantic_sha256",
            ("observation_semantic_sha256", "semantic_sha256"),
        ),
    )
    values = {name: _extract_record_field(record, field_aliases) for name, field_aliases in aliases}
    missing = tuple(name for name, value in values.items() if value is None)
    if missing:
        return _unavailable(*(f"origin_record_missing:{name}" for name in missing))
    for name in (
        "factor_name",
        "definition_sha256",
        "factor_set_sha256",
        "quant_policy_sha256",
        "source_lineage_sha256",
        "observation_ref",
        "observation_byte_sha256",
        "observation_semantic_sha256",
    ):
        if not isinstance(values[name], str) or not values[name]:
            raise ValueError(f"{name} must be a non-empty string")
    for name in (
        "definition_sha256",
        "factor_set_sha256",
        "quant_policy_sha256",
        "source_lineage_sha256",
        "observation_byte_sha256",
        "observation_semantic_sha256",
    ):
        if not _is_sha256(values[name]):  # type: ignore[arg-type]
            raise ValueError(f"{name} must be a lowercase SHA-256")
    horizon = values["horizon_sessions"]
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon_sessions must be a positive integer")
    observation_ref = values["observation_ref"]
    assert isinstance(observation_ref, str)
    try:
        observation_ref.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("observation_ref must be ASCII") from exc
    return _available(
        OriginObservationRecord(
            factor_name=str(values["factor_name"]),
            definition_sha256=str(values["definition_sha256"]),
            factor_set_sha256=str(values["factor_set_sha256"]),
            quant_policy_sha256=str(values["quant_policy_sha256"]),
            horizon_sessions=horizon,
            source_lineage_sha256=str(values["source_lineage_sha256"]),
            observation_ref=observation_ref,
            observation_byte_sha256=str(values["observation_byte_sha256"]),
            observation_semantic_sha256=str(values["observation_semantic_sha256"]),
        )
    )


def deduplicate_origins(
    records: Sequence[object] | None,
) -> OriginDeduplicationResult:
    """De-duplicate exact origins or block conflicting byte/semantic identities."""

    if records is None or not records:
        return OriginDeduplicationResult(
            status=EvidenceStatus.UNAVAILABLE,
            blockers=("origin_records_missing",),
        )
    if isinstance(records, (str, bytes)):
        raise ValueError("records must be a sequence")
    normalized: list[OriginObservationRecord] = []
    for record in records:
        result = _normalize_origin_record(record)
        if not result.available:
            return OriginDeduplicationResult(
                status=EvidenceStatus.UNAVAILABLE,
                blockers=result.blockers,
            )
        assert result.value is not None
        normalized.append(result.value)

    grouped: dict[OriginDedupKey, list[OriginObservationRecord]] = {}
    for record in normalized:
        grouped.setdefault(record.key, []).append(record)

    canonical: list[OriginObservationRecord] = []
    duplicates: list[DuplicateOrigin] = []
    conflicts: list[DuplicateOriginConflict] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda row: row.observation_ref)
        byte_shas = tuple(sorted({row.observation_byte_sha256 for row in group}))
        semantic_shas = tuple(sorted({row.observation_semantic_sha256 for row in group}))
        if len(byte_shas) != 1 or len(semantic_shas) != 1:
            conflicts.append(
                DuplicateOriginConflict(
                    key=key,
                    observation_refs=tuple(row.observation_ref for row in group),
                    byte_sha256_values=byte_shas,
                    semantic_sha256_values=semantic_shas,
                )
            )
            continue
        canonical.append(group[0])
        if len(group) > 1:
            duplicates.append(
                DuplicateOrigin(
                    key=key,
                    kept_ref=group[0].observation_ref,
                    duplicate_refs=tuple(row.observation_ref for row in group[1:]),
                )
            )

    if conflicts:
        return OriginDeduplicationResult(
            status=EvidenceStatus.DUPLICATE_ORIGIN_CONFLICT,
            duplicates=tuple(duplicates),
            conflicts=tuple(conflicts),
            blockers=("DUPLICATE_ORIGIN_CONFLICT",),
        )
    return OriginDeduplicationResult(
        status=EvidenceStatus.AVAILABLE,
        records=tuple(canonical),
        duplicates=tuple(duplicates),
    )


deduplicate_origin_observations = deduplicate_origins


__all__ = [
    "AdjustedReturns",
    "Availability",
    "CHALLENGER_HORIZON",
    "DuplicateOrigin",
    "DuplicateOriginConflict",
    "EvidenceStatus",
    "FactorTier",
    "FactorTierDecision",
    "FactorTierInput",
    "ForwardDiagnosticReceipt",
    "OriginDedupKey",
    "OriginDeduplicationResult",
    "OriginInformationCoefficients",
    "OriginObservationRecord",
    "ROUND_TRIP_COST_BPS",
    "SHANGHAI_CLOSE",
    "SHANGHAI_TIMEZONE",
    "SUPPORTED_HORIZONS",
    "TierAllocation",
    "adjusted_close_simple_return",
    "adjusted_return",
    "allocate_factor_tiers",
    "annualized_rank_ic_ir",
    "apply_flat_round_trip_cost",
    "average_tie_ranks",
    "capacity_mean_top_quintile_adv",
    "cost_adjusted_group_return",
    "cost_adjusted_top_bottom_quintile_return",
    "deduplicate_origin_observations",
    "deduplicate_origins",
    "evaluate_factor_tier",
    "industry_adjusted_return",
    "is_label_matured",
    "label_maturity",
    "market_adjusted_return",
    "market_industry_adjusted_returns",
    "max_abs_existing_correlation",
    "max_abs_existing_factor_pearson",
    "open_session_freshness",
    "pearson_ic",
    "per_origin_information_coefficients",
    "per_origin_pearson_ic",
    "per_origin_spearman_rank_ic",
    "rank_ic_sign_stability",
    "resolve_horizon_end_sessions",
    "shanghai_horizon_end_sessions",
    "simple_adjusted_close_return",
    "spearman_rank_ic",
    "stability_same_rank_ic_sign",
    "top_quintile_capacity",
    "top_quintile_turnover",
    "turnover_one_minus_overlap",
]
