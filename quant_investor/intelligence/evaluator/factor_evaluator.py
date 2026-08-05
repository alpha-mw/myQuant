"""Deterministic, authority-free factor evaluation for forward research.

The evaluator consumes normalized observation/label joins supplied by its
caller.  It does not read artifacts, discover a latest run, call a provider,
change factor weights, or mutate governance state.  Every cross-sectional
statistic is calculated independently inside an explicit forward origin before
the origin-level values are aggregated.  This prevents a large universe at one
origin from silently dominating the result.

Numeric policy
--------------

* calculations use :class:`~decimal.Decimal` with precision 50;
* serialized values use 12 decimal places and ``ROUND_HALF_EVEN``;
* factor orientation is applied to the score before ranking;
* ties receive their average rank;
* quintiles use ``ceil(5 * average_rank / symbol_count)`` and never split a
  tie across buckets;
* ``icir_base`` is unannualized while the public ``icir`` metric uses the
  deliberately naive ``sqrt(252 / horizon_sessions)`` multiplier;
* the gross spread is Q5 minus Q1 and the diagnostic cost-adjusted spread is
  charged a single flat 20 basis points;
* the separately reported Q5 long-only cost-adjusted return consumes the
  supplied label directly and never deducts that flat charge again.

The returned dictionaries are JSON-compatible.  They are intentionally not
receipts: content addressing, source references, evaluation windows, and
authority closure belong to the receipt layer.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from datetime import date
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_EVEN, localcontext
from typing import Any, Final

from quant_investor.intelligence._core import (
    IntelligenceContractError,
    decimal_text,
    decimal_value,
    identifier,
)

FACTOR_METRIC_FORMULA_VERSION: Final = "myquant.v17.research-intelligence.factor-metrics.v1"

HIGHER_IS_BETTER: Final = "HIGHER_IS_BETTER"
LOWER_IS_BETTER: Final = "LOWER_IS_BETTER"
ORIENTATIONS: Final = (HIGHER_IS_BETTER, LOWER_IS_BETTER)

FLAT_SPREAD_COST: Final = Decimal("0.002")

INSUFFICIENT_SYMBOLS: Final = "INSUFFICIENT_SYMBOLS"
INSUFFICIENT_AVAILABLE_ORIGINS: Final = "INSUFFICIENT_AVAILABLE_ORIGINS"
INSUFFICIENT_JOINT_COVERAGE: Final = "INSUFFICIENT_JOINT_COVERAGE"
INSUFFICIENT_INDUSTRY_MAPPING_COVERAGE: Final = "INSUFFICIENT_INDUSTRY_MAPPING_COVERAGE"
ZERO_SCORE_VARIANCE: Final = "ZERO_SCORE_VARIANCE"
ZERO_RETURN_VARIANCE: Final = "ZERO_RETURN_VARIANCE"
EMPTY_QUANTILE: Final = "EMPTY_QUANTILE"
INSUFFICIENT_IC_ORIGINS: Final = "INSUFFICIENT_IC_ORIGINS"
ZERO_IC_VARIANCE: Final = "ZERO_IC_VARIANCE"
NO_CONSECUTIVE_ORIGIN_TRANSITIONS: Final = "NO_CONSECUTIVE_ORIGIN_TRANSITIONS"
OVERLAPPING_FORWARD_WINDOWS: Final = "OVERLAPPING_FORWARD_WINDOWS"
RETURN_NOT_COMPOUNDABLE: Final = "RETURN_NOT_COMPOUNDABLE"
COMPLETE_DRAWDOWN_PATH_UNAVAILABLE: Final = "COMPLETE_DRAWDOWN_PATH_UNAVAILABLE"

NAIVE_ANNUALIZATION_SERIAL_CORRELATION_UNADJUSTED: Final = (
    "NAIVE_ANNUALIZATION_SERIAL_CORRELATION_UNADJUSTED"
)
INDUSTRY_ADJUSTED_LABEL_DIAGNOSTIC_NOT_RESIDUAL_MODEL: Final = (
    "INDUSTRY_ADJUSTED_LABEL_DIAGNOSTIC_NOT_RESIDUAL_MODEL"
)
NON_TRADING_DIAGNOSTIC_PATH: Final = "NON_TRADING_DIAGNOSTIC_PATH"
ORIGIN_GAPS_NOT_BRIDGED: Final = "ORIGIN_GAPS_NOT_BRIDGED"
FLAT_20BP_SPREAD_DIAGNOSTIC: Final = "FLAT_20BP_SPREAD_DIAGNOSTIC"

BLOCKER_CODES: Final = (
    COMPLETE_DRAWDOWN_PATH_UNAVAILABLE,
    EMPTY_QUANTILE,
    INSUFFICIENT_AVAILABLE_ORIGINS,
    INSUFFICIENT_IC_ORIGINS,
    INSUFFICIENT_INDUSTRY_MAPPING_COVERAGE,
    INSUFFICIENT_JOINT_COVERAGE,
    INSUFFICIENT_SYMBOLS,
    NO_CONSECUTIVE_ORIGIN_TRANSITIONS,
    OVERLAPPING_FORWARD_WINDOWS,
    RETURN_NOT_COMPOUNDABLE,
    ZERO_IC_VARIANCE,
    ZERO_RETURN_VARIANCE,
    ZERO_SCORE_VARIANCE,
)
LIMITATION_CODES: Final = (
    FLAT_20BP_SPREAD_DIAGNOSTIC,
    INDUSTRY_ADJUSTED_LABEL_DIAGNOSTIC_NOT_RESIDUAL_MODEL,
    NAIVE_ANNUALIZATION_SERIAL_CORRELATION_UNADJUSTED,
    NON_TRADING_DIAGNOSTIC_PATH,
    ORIGIN_GAPS_NOT_BRIDGED,
)

QUANTILE_METRIC_IDS: Final = tuple(f"quantile_return_q{quantile}" for quantile in range(1, 6))
METRIC_IDS: Final = (
    "rank_ic",
    "icir_base",
    "icir",
    *QUANTILE_METRIC_IDS,
    "long_short_spread",
    "turnover",
    "score_coverage",
    "label_coverage",
    "joint_coverage",
    "industry_mapping_coverage",
    "origin_maturity_coverage",
    "neutralized_alpha",
    "cost_adjusted_return",
    "q5_long_only_cost_adjusted_return",
    "drawdown",
    "stability",
)

_ORIGIN_FIELDS: Final = {
    "label_session",
    "next_open_session",
    "origin_id",
    "origin_session",
    "symbol_rows",
}
_SYMBOL_FIELDS: Final = {
    "cost_adjusted_return",
    "industry_adjusted_return",
    "industry_id",
    "score",
    "score_status",
    "symbol",
    "total_return",
}
_SCORE_STATUSES: Final = {"AVAILABLE", "UNAVAILABLE"}


def _sorted_codes(values: Collection[str]) -> list[str]:
    """Return unique canonical codes in bytewise ASCII order."""

    return sorted(set(values), key=lambda value: value.encode("ascii"))


def _session(value: Any, *, label: str) -> str:
    """Validate one canonical ISO session date."""

    if type(value) is not str:
        raise IntelligenceContractError(f"{label} must be an ISO session date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise IntelligenceContractError(f"{label} must be an ISO session date") from exc
    if parsed.isoformat() != value:
        raise IntelligenceContractError(f"{label} must be canonical")
    return value


def _optional_decimal(value: Any, *, label: str) -> Decimal | None:
    if value is None:
        return None
    return decimal_value(value, label=label)


def _normalize_symbol_rows(value: Any, *, label: str, orientation: str) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise IntelligenceContractError(f"{label} must be a non-empty sequence")

    multiplier = Decimal("1") if orientation == HIGHER_IS_BETTER else Decimal("-1")
    rows: list[dict[str, Any]] = []
    symbols: set[str] = set()
    for index, raw_row in enumerate(value):
        row_label = f"{label}[{index}]"
        if type(raw_row) is not dict or set(raw_row) != _SYMBOL_FIELDS:
            raise IntelligenceContractError(
                f"{row_label} must contain the exact normalized symbol fields"
            )
        symbol = raw_row["symbol"]
        if type(symbol) is not str or not symbol or not symbol.isascii():
            raise IntelligenceContractError(f"{row_label}.symbol must be non-empty ASCII")
        if symbol in symbols:
            raise IntelligenceContractError(f"{label} contains duplicate symbols")
        symbols.add(symbol)

        score_status = raw_row["score_status"]
        if score_status not in _SCORE_STATUSES:
            raise IntelligenceContractError(f"{row_label}.score_status is invalid")
        if score_status == "AVAILABLE":
            if raw_row["score"] is None:
                raise IntelligenceContractError(f"{row_label}.score is required when available")
            score = decimal_value(raw_row["score"], label=f"{row_label}.score")
            oriented_score = score * multiplier
        else:
            if raw_row["score"] is not None:
                raise IntelligenceContractError(f"{row_label}.score must be null when unavailable")
            score = None
            oriented_score = None

        industry_id = raw_row["industry_id"]
        if industry_id is not None and (type(industry_id) is not str or not industry_id.strip()):
            raise IntelligenceContractError(
                f"{row_label}.industry_id must be null or a non-empty string"
            )
        rows.append(
            {
                "cost_adjusted_return": _optional_decimal(
                    raw_row["cost_adjusted_return"],
                    label=f"{row_label}.cost_adjusted_return",
                ),
                "industry_adjusted_return": _optional_decimal(
                    raw_row["industry_adjusted_return"],
                    label=f"{row_label}.industry_adjusted_return",
                ),
                "industry_id": industry_id,
                "oriented_score": oriented_score,
                "score": score,
                "score_status": score_status,
                "symbol": symbol,
                "total_return": _optional_decimal(
                    raw_row["total_return"], label=f"{row_label}.total_return"
                ),
            }
        )
    return sorted(rows, key=lambda row: str(row["symbol"]).encode("ascii"))


def _normalize_origins(
    origins: Sequence[Mapping[str, Any]], *, orientation: str
) -> list[dict[str, Any]]:
    if isinstance(origins, (str, bytes)) or not isinstance(origins, Sequence) or not origins:
        raise IntelligenceContractError("origins must be a non-empty sequence")

    normalized: list[dict[str, Any]] = []
    origin_ids: set[str] = set()
    for index, raw_origin in enumerate(origins):
        label = f"origins[{index}]"
        if type(raw_origin) is not dict or set(raw_origin) != _ORIGIN_FIELDS:
            raise IntelligenceContractError(
                f"{label} must contain the exact normalized origin fields"
            )
        origin_id = identifier(raw_origin["origin_id"], label=f"{label}.origin_id")
        if origin_id in origin_ids:
            raise IntelligenceContractError("origins contains duplicate origin IDs")
        origin_ids.add(origin_id)

        origin_session = _session(raw_origin["origin_session"], label=f"{label}.origin_session")
        next_open_session = _session(
            raw_origin["next_open_session"], label=f"{label}.next_open_session"
        )
        label_session = _session(raw_origin["label_session"], label=f"{label}.label_session")
        if not origin_session < next_open_session <= label_session:
            raise IntelligenceContractError(f"{label} session chronology is not future-only")
        normalized.append(
            {
                "label_session": label_session,
                "next_open_session": next_open_session,
                "origin_id": origin_id,
                "origin_session": origin_session,
                "symbol_rows": _normalize_symbol_rows(
                    raw_origin["symbol_rows"],
                    label=f"{label}.symbol_rows",
                    orientation=orientation,
                ),
            }
        )
    return sorted(
        normalized,
        key=lambda row: (
            str(row["origin_session"]).encode("ascii"),
            str(row["origin_id"]).encode("ascii"),
        ),
    )


def _mean(values: Sequence[Decimal]) -> Decimal:
    if not values:
        raise IntelligenceContractError("cannot compute the mean of no values")
    return sum(values, Decimal("0")) / Decimal(len(values))


def _sample_std(values: Sequence[Decimal]) -> Decimal:
    if len(values) < 2:
        raise IntelligenceContractError("sample standard deviation needs two values")
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / Decimal(len(values) - 1)
    return variance.sqrt()


def _average_ranks(values: Sequence[Decimal]) -> list[Decimal]:
    """Return one-based average ranks, preserving the caller's order."""

    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [Decimal("0")] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        first_rank = Decimal(cursor + 1)
        last_rank = Decimal(end)
        average_rank = (first_rank + last_rank) / Decimal("2")
        for position in range(cursor, end):
            ranks[ordered[position][0]] = average_rank
        cursor = end
    return ranks


def _rank_ic(scores: Sequence[Decimal], returns: Sequence[Decimal]) -> Decimal:
    if len(scores) != len(returns) or len(scores) < 2:
        raise IntelligenceContractError("RankIC inputs must have equal non-trivial size")
    score_ranks = _average_ranks(scores)
    return_ranks = _average_ranks(returns)
    score_mean = _mean(score_ranks)
    return_mean = _mean(return_ranks)
    numerator = sum(
        (score_rank - score_mean) * (return_rank - return_mean)
        for score_rank, return_rank in zip(score_ranks, return_ranks)
    )
    score_square = sum(((rank - score_mean) ** 2 for rank in score_ranks), Decimal("0"))
    return_square = sum(((rank - return_mean) ** 2 for rank in return_ranks), Decimal("0"))
    denominator = (score_square * return_square).sqrt()
    if denominator == 0:
        raise IntelligenceContractError("RankIC variance is zero")
    return numerator / denominator


def _quintile_rows(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[Mapping[str, Any]]]:
    """Assign oriented-score rows by average-rank percentile without tie splits."""

    ranks = _average_ranks([row["oriented_score"] for row in rows])
    count = Decimal(len(rows))
    buckets: dict[int, list[Mapping[str, Any]]] = {quantile: [] for quantile in range(1, 6)}
    for row, rank in zip(rows, ranks):
        bucket = int(((Decimal("5") * rank) / count).to_integral_value(rounding=ROUND_CEILING))
        bucket = min(5, max(1, bucket))
        buckets[bucket].append(row)
    return buckets


def _equal_weights(symbols: Sequence[str]) -> tuple[dict[str, str], dict[str, Decimal]]:
    """Return deterministic 12-place equal weights that sum exactly to one."""

    ordered = sorted(symbols, key=lambda symbol: symbol.encode("ascii"))
    if not ordered:
        return {}, {}
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        base = Decimal(decimal_text(Decimal("1") / Decimal(len(ordered))))
        exact: dict[str, Decimal] = {}
        for symbol in ordered[:-1]:
            exact[symbol] = base
        exact[ordered[-1]] = Decimal("1") - base * Decimal(len(ordered) - 1)
        return (
            {symbol: decimal_text(exact[symbol]) for symbol in ordered},
            exact,
        )


def _metric_row(
    *,
    metric_id: str,
    value: Decimal | None,
    unit: str,
    origin_ids: Sequence[str],
    sample_count: int,
    blockers: Collection[str] = (),
    limitations: Collection[str] = (),
) -> dict[str, Any]:
    """Build one normalized aggregate metric row."""

    normalized_origins = sorted(set(origin_ids), key=lambda origin_id: origin_id.encode("ascii"))
    return {
        "available_origin_count": len(normalized_origins),
        "blocker_codes": _sorted_codes(blockers),
        "formula_version": FACTOR_METRIC_FORMULA_VERSION,
        "input_origin_ids": normalized_origins,
        "limitations": _sorted_codes(limitations),
        "metric_id": metric_id,
        "sample_count": sample_count,
        "status": "AVAILABLE" if value is not None else "UNAVAILABLE",
        "unit": unit,
        "value": None if value is None else decimal_text(value),
    }


def _evaluate_origin(
    origin: Mapping[str, Any],
    *,
    min_symbols: int,
    min_joint_coverage: Decimal,
    min_industry_mapping_coverage: Decimal,
) -> dict[str, Any]:
    rows = list(origin["symbol_rows"])
    score_rows = [row for row in rows if row["oriented_score"] is not None]
    label_rows = [row for row in rows if row["total_return"] is not None]
    joint_rows = [
        row for row in rows if row["oriented_score"] is not None and row["total_return"] is not None
    ]
    industry_rows = [row for row in joint_rows if row["industry_id"] is not None]

    total_count = Decimal(len(rows))
    score_coverage = Decimal(len(score_rows)) / total_count
    label_coverage = Decimal(len(label_rows)) / total_count
    joint_coverage = Decimal(len(joint_rows)) / total_count
    industry_mapping_coverage = Decimal(len(industry_rows)) / total_count

    core_blockers: set[str] = set()
    if len(joint_rows) < min_symbols:
        core_blockers.add(INSUFFICIENT_SYMBOLS)
    if joint_coverage < min_joint_coverage:
        core_blockers.add(INSUFFICIENT_JOINT_COVERAGE)

    metrics: dict[str, Decimal | None] = {
        "score_coverage": score_coverage,
        "label_coverage": label_coverage,
        "joint_coverage": joint_coverage,
        "industry_mapping_coverage": industry_mapping_coverage,
    }
    metric_blockers: dict[str, set[str]] = {
        metric_id: set(core_blockers) for metric_id in METRIC_IDS
    }
    for metric_id in (
        "score_coverage",
        "label_coverage",
        "joint_coverage",
        "industry_mapping_coverage",
    ):
        metric_blockers[metric_id] = set()

    q5_weights_text: dict[str, str] = {}
    q5_weights_exact: dict[str, Decimal] = {}

    if core_blockers:
        for metric_id in METRIC_IDS:
            metrics.setdefault(metric_id, None)
    else:
        scores = [row["oriented_score"] for row in joint_rows]
        returns = [row["total_return"] for row in joint_rows]
        if len(set(scores)) == 1:
            metric_blockers["rank_ic"].add(ZERO_SCORE_VARIANCE)
            rank_ic = None
        elif len(set(returns)) == 1:
            metric_blockers["rank_ic"].add(ZERO_RETURN_VARIANCE)
            rank_ic = None
        else:
            rank_ic = _rank_ic(scores, returns)
        metrics["rank_ic"] = rank_ic

        buckets = _quintile_rows(joint_rows)
        if any(not buckets[quantile] for quantile in range(1, 6)):
            for metric_id in (
                *QUANTILE_METRIC_IDS,
                "long_short_spread",
                "neutralized_alpha",
                "cost_adjusted_return",
                "q5_long_only_cost_adjusted_return",
            ):
                metric_blockers[metric_id].add(EMPTY_QUANTILE)
                metrics[metric_id] = None
        else:
            quantile_returns: dict[int, Decimal] = {}
            for quantile in range(1, 6):
                quantile_return = _mean([row["total_return"] for row in buckets[quantile]])
                quantile_returns[quantile] = quantile_return
                metrics[f"quantile_return_q{quantile}"] = quantile_return

            spread = quantile_returns[5] - quantile_returns[1]
            metrics["long_short_spread"] = spread
            metrics["cost_adjusted_return"] = spread - FLAT_SPREAD_COST

            q5_weights_text, q5_weights_exact = _equal_weights(
                [str(row["symbol"]) for row in buckets[5]]
            )
            q5_cost_rows = [
                row["cost_adjusted_return"]
                for row in buckets[5]
                if row["cost_adjusted_return"] is not None
            ]
            if len(q5_cost_rows) == len(buckets[5]):
                metrics["q5_long_only_cost_adjusted_return"] = _mean(q5_cost_rows)
            else:
                metric_blockers["q5_long_only_cost_adjusted_return"].add(
                    INSUFFICIENT_JOINT_COVERAGE
                )
                metrics["q5_long_only_cost_adjusted_return"] = None

            adjusted_q1 = [
                row["industry_adjusted_return"]
                for row in buckets[1]
                if row["industry_id"] is not None and row["industry_adjusted_return"] is not None
            ]
            adjusted_q5 = [
                row["industry_adjusted_return"]
                for row in buckets[5]
                if row["industry_id"] is not None and row["industry_adjusted_return"] is not None
            ]
            if (
                industry_mapping_coverage < min_industry_mapping_coverage
                or len(adjusted_q1) != len(buckets[1])
                or len(adjusted_q5) != len(buckets[5])
            ):
                metric_blockers["neutralized_alpha"].add(INSUFFICIENT_INDUSTRY_MAPPING_COVERAGE)
                metrics["neutralized_alpha"] = None
            else:
                metrics["neutralized_alpha"] = _mean(adjusted_q5) - _mean(adjusted_q1)

    for metric_id in METRIC_IDS:
        metrics.setdefault(metric_id, None)

    return {
        "blockers": metric_blockers,
        "label_session": origin["label_session"],
        "metrics": metrics,
        "next_open_session": origin["next_open_session"],
        "origin_id": origin["origin_id"],
        "origin_session": origin["origin_session"],
        "q5_weights": q5_weights_text,
        "q5_weights_exact": q5_weights_exact,
        "sample_counts": {
            "industry": len(industry_rows),
            "joint": len(joint_rows),
            "labels": len(label_rows),
            "scores": len(score_rows),
            "subjects": len(rows),
        },
    }


def _aggregate_origin_metric(
    evaluations: Sequence[Mapping[str, Any]],
    *,
    metric_id: str,
    unit: str,
    minimum_origins: int,
    insufficient_code: str = INSUFFICIENT_AVAILABLE_ORIGINS,
    limitations: Sequence[str] = (),
) -> dict[str, Any]:
    available = [row for row in evaluations if row["metrics"].get(metric_id) is not None]
    blockers: set[str] = set()
    for row in evaluations:
        blockers.update(row["blockers"].get(metric_id, ()))
    if len(available) < minimum_origins:
        blockers.add(insufficient_code)
        value = None
    else:
        value = _mean([row["metrics"][metric_id] for row in available])
    return _metric_row(
        metric_id=metric_id,
        value=value,
        unit=unit,
        origin_ids=[str(row["origin_id"]) for row in available],
        sample_count=len(available),
        blockers=blockers,
        limitations=limitations,
    )


def _aggregate_coverage(
    evaluations: Sequence[Mapping[str, Any]], *, metric_id: str, numerator: str
) -> dict[str, Any]:
    denominator = sum(int(row["sample_counts"]["subjects"]) for row in evaluations)
    available = sum(int(row["sample_counts"][numerator]) for row in evaluations)
    value = Decimal(available) / Decimal(denominator)
    return _metric_row(
        metric_id=metric_id,
        value=value,
        unit="RATIO",
        origin_ids=[str(row["origin_id"]) for row in evaluations],
        sample_count=denominator,
    )


def _industry_coverage(evaluations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    subjects = sum(int(row["sample_counts"]["subjects"]) for row in evaluations)
    mapped = sum(int(row["sample_counts"]["industry"]) for row in evaluations)
    value = Decimal(mapped) / Decimal(subjects)
    return _metric_row(
        metric_id="industry_mapping_coverage",
        value=value,
        unit="RATIO",
        origin_ids=[str(row["origin_id"]) for row in evaluations],
        sample_count=subjects,
    )


def _origin_maturity_coverage(
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    matured = [row for row in evaluations if int(row["sample_counts"]["labels"]) > 0]
    return _metric_row(
        metric_id="origin_maturity_coverage",
        value=Decimal(len(matured)) / Decimal(len(evaluations)),
        unit="RATIO",
        origin_ids=[str(row["origin_id"]) for row in matured],
        sample_count=len(evaluations),
    )


def _icir_rows(
    evaluations: Sequence[Mapping[str, Any]],
    *,
    horizon_sessions: int,
    min_available_origins: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    available = [row for row in evaluations if row["metrics"].get("rank_ic") is not None]
    origin_ids = [str(row["origin_id"]) for row in available]
    values = [row["metrics"]["rank_ic"] for row in available]
    blockers: set[str] = set()
    for row in evaluations:
        blockers.update(row["blockers"].get("rank_ic", ()))
    if len(values) < max(2, min_available_origins):
        blockers.add(INSUFFICIENT_IC_ORIGINS)
        icir = None
    else:
        sample_std = _sample_std(values)
        if sample_std == 0:
            blockers.add(ZERO_IC_VARIANCE)
            icir = None
        else:
            icir = _mean(values) / sample_std
    annualized = (
        None if icir is None else icir * (Decimal("252") / Decimal(horizon_sessions)).sqrt()
    )
    base_row = _metric_row(
        metric_id="icir_base",
        value=icir,
        unit="RATIO",
        origin_ids=origin_ids,
        sample_count=len(values),
        blockers=blockers,
    )
    annualized_row = _metric_row(
        metric_id="icir",
        value=annualized,
        unit="RATIO",
        origin_ids=origin_ids,
        sample_count=len(values),
        blockers=blockers,
        limitations=(NAIVE_ANNUALIZATION_SERIAL_CORRELATION_UNADJUSTED,),
    )
    return base_row, annualized_row


def _turnover_row(
    evaluations: Sequence[Mapping[str, Any]], *, min_available_origins: int
) -> tuple[dict[str, Any], dict[str, Decimal | None], bool]:
    transition_values: list[Decimal] = []
    transition_origin_ids: list[str] = []
    per_origin: dict[str, Decimal | None] = {str(row["origin_id"]): None for row in evaluations}
    gap_detected = False
    for previous, current in zip(evaluations, evaluations[1:]):
        if previous["next_open_session"] != current["origin_session"]:
            gap_detected = True
            continue
        if not previous["q5_weights_exact"] or not current["q5_weights_exact"]:
            continue
        symbols = sorted(
            set(previous["q5_weights_exact"]) | set(current["q5_weights_exact"]),
            key=lambda symbol: symbol.encode("ascii"),
        )
        turnover = Decimal("0.5") * sum(
            abs(
                current["q5_weights_exact"].get(symbol, Decimal("0"))
                - previous["q5_weights_exact"].get(symbol, Decimal("0"))
            )
            for symbol in symbols
        )
        transition_values.append(turnover)
        transition_origin_ids.extend([str(previous["origin_id"]), str(current["origin_id"])])
        per_origin[str(current["origin_id"])] = turnover

    blockers: set[str] = set()
    if not transition_values or len(transition_values) < max(1, min_available_origins - 1):
        blockers.add(NO_CONSECUTIVE_ORIGIN_TRANSITIONS)
        value = None
    else:
        value = _mean(transition_values)
    limitations = (ORIGIN_GAPS_NOT_BRIDGED,) if gap_detected else ()
    return (
        _metric_row(
            metric_id="turnover",
            value=value,
            unit="RATIO",
            origin_ids=transition_origin_ids,
            sample_count=len(transition_values),
            blockers=blockers,
            limitations=limitations,
        ),
        per_origin,
        gap_detected,
    )


def _drawdown_row(
    evaluations: Sequence[Mapping[str, Any]], *, min_available_origins: int
) -> dict[str, Any]:
    available = [
        row
        for row in evaluations
        if row["metrics"].get("q5_long_only_cost_adjusted_return") is not None
    ]
    blockers: set[str] = set()
    if len(available) < min_available_origins:
        blockers.add(INSUFFICIENT_AVAILABLE_ORIGINS)
    if len(available) != len(evaluations):
        blockers.add(COMPLETE_DRAWDOWN_PATH_UNAVAILABLE)
    for previous, current in zip(available, available[1:]):
        if current["origin_session"] <= previous["label_session"]:
            blockers.add(OVERLAPPING_FORWARD_WINDOWS)
    if any(
        row["metrics"]["q5_long_only_cost_adjusted_return"] < Decimal("-1") for row in available
    ):
        blockers.add(RETURN_NOT_COMPOUNDABLE)

    if blockers:
        value = None
    else:
        wealth = Decimal("1")
        peak = Decimal("1")
        value = Decimal("0")
        for row in available:
            wealth *= Decimal("1") + row["metrics"]["q5_long_only_cost_adjusted_return"]
            if wealth > peak:
                peak = wealth
            drawdown = Decimal("1") - wealth / peak
            if drawdown > value:
                value = drawdown
    return _metric_row(
        metric_id="drawdown",
        value=value,
        unit="RATIO",
        origin_ids=[str(row["origin_id"]) for row in available],
        sample_count=len(available),
        blockers=blockers,
    )


def evaluate_factor(
    *,
    factor_id: str,
    origins: Sequence[Mapping[str, Any]],
    orientation: str,
    horizon_sessions: int,
    min_symbols: int,
    min_available_origins: int,
    min_joint_coverage: Any,
    min_industry_mapping_coverage: Any,
) -> dict[str, Any]:
    """Evaluate one factor over explicit forward origins.

    Parameters
    ----------
    factor_id:
        Canonical research factor identifier.  It is validated but deliberately
        not copied into the result because the receipt layer binds subject
        identity.
    origins:
        Sequence of exact dictionaries with ``origin_id``, ``origin_session``,
        ``label_session``, ``next_open_session``, and ``symbol_rows``.  Each
        symbol row must have exactly ``symbol``, ``score_status``, ``score``,
        ``total_return``, ``cost_adjusted_return``,
        ``industry_adjusted_return``, and ``industry_id``.
    orientation:
        ``HIGHER_IS_BETTER`` or ``LOWER_IS_BETTER``.  Lower-is-better factors
        are negated before all ranking and Q1/Q5 assignments.
    horizon_sessions:
        Positive forward-label horizon used only by the naive annualization
        multiplier.  No date or label is inferred from it.
    min_symbols, min_available_origins, min_joint_coverage,
    min_industry_mapping_coverage:
        Explicit fail-closed research policy thresholds.  Missing labels remain
        missing; no imputation, pooling, or fallback is performed.

    Returns
    -------
    dict
        Exactly ``status``, ``metrics``, ``origin_metrics``, and
        ``limitations``.  Aggregate ``metrics`` are sorted normalized rows;
        ``origin_metrics`` preserve the inputs needed to replay turnover and
        regime-conditioned analysis.
    """

    identifier(factor_id, label="factor_id")
    if orientation not in ORIENTATIONS:
        raise IntelligenceContractError("orientation is not allowlisted")
    if type(horizon_sessions) is not int or horizon_sessions <= 0:
        raise IntelligenceContractError("horizon_sessions must be a positive integer")
    if type(min_symbols) is not int or min_symbols <= 0:
        raise IntelligenceContractError("min_symbols must be a positive integer")
    if type(min_available_origins) is not int or min_available_origins <= 0:
        raise IntelligenceContractError("min_available_origins must be a positive integer")
    joint_threshold = decimal_value(
        min_joint_coverage,
        label="min_joint_coverage",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )
    industry_threshold = decimal_value(
        min_industry_mapping_coverage,
        label="min_industry_mapping_coverage",
        minimum=Decimal("0"),
        maximum=Decimal("1"),
    )

    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        normalized = _normalize_origins(origins, orientation=orientation)
        evaluated = [
            _evaluate_origin(
                origin,
                min_symbols=min_symbols,
                min_joint_coverage=joint_threshold,
                min_industry_mapping_coverage=industry_threshold,
            )
            for origin in normalized
        ]

        metrics_by_id: dict[str, dict[str, Any]] = {}
        metrics_by_id["score_coverage"] = _aggregate_coverage(
            evaluated, metric_id="score_coverage", numerator="scores"
        )
        metrics_by_id["label_coverage"] = _aggregate_coverage(
            evaluated, metric_id="label_coverage", numerator="labels"
        )
        metrics_by_id["joint_coverage"] = _aggregate_coverage(
            evaluated, metric_id="joint_coverage", numerator="joint"
        )
        metrics_by_id["industry_mapping_coverage"] = _industry_coverage(evaluated)
        metrics_by_id["origin_maturity_coverage"] = _origin_maturity_coverage(evaluated)

        aggregate_specs = {
            "rank_ic": ("CORRELATION", ()),
            **{metric_id: ("RETURN", ()) for metric_id in QUANTILE_METRIC_IDS},
            "long_short_spread": ("RETURN", ()),
            "neutralized_alpha": (
                "RETURN",
                (INDUSTRY_ADJUSTED_LABEL_DIAGNOSTIC_NOT_RESIDUAL_MODEL,),
            ),
            "cost_adjusted_return": (
                "RETURN",
                (FLAT_20BP_SPREAD_DIAGNOSTIC,),
            ),
            "q5_long_only_cost_adjusted_return": ("RETURN", ()),
        }
        for metric_id, (unit, metric_limitations) in aggregate_specs.items():
            metrics_by_id[metric_id] = _aggregate_origin_metric(
                evaluated,
                metric_id=metric_id,
                unit=unit,
                minimum_origins=min_available_origins,
                limitations=metric_limitations,
            )

        icir, annualized_icir = _icir_rows(
            evaluated,
            horizon_sessions=horizon_sessions,
            min_available_origins=min_available_origins,
        )
        metrics_by_id["icir_base"] = icir
        metrics_by_id["icir"] = annualized_icir

        turnover, turnover_by_origin, gap_detected = _turnover_row(
            evaluated, min_available_origins=min_available_origins
        )
        metrics_by_id["turnover"] = turnover
        metrics_by_id["drawdown"] = _drawdown_row(
            evaluated, min_available_origins=min_available_origins
        )

        available_ic = [row for row in evaluated if row["metrics"].get("rank_ic") is not None]
        stability_blockers: set[str] = set()
        if len(available_ic) < min_available_origins:
            stability_blockers.add(INSUFFICIENT_IC_ORIGINS)
            stability = None
        else:
            stability = Decimal(
                sum(row["metrics"]["rank_ic"] > Decimal("0") for row in available_ic)
            ) / Decimal(len(available_ic))
        metrics_by_id["stability"] = _metric_row(
            metric_id="stability",
            value=stability,
            unit="RATIO",
            origin_ids=[str(row["origin_id"]) for row in available_ic],
            sample_count=len(available_ic),
            blockers=stability_blockers,
        )

        core_available = [
            row for row in evaluated if row["metrics"].get("long_short_spread") is not None
        ]
        aggregate_joint = Decimal(metrics_by_id["joint_coverage"]["value"])
        if len(core_available) < min_available_origins or aggregate_joint < joint_threshold:
            status = "UNAVAILABLE"
        elif all(metrics_by_id[metric_id]["status"] == "AVAILABLE" for metric_id in METRIC_IDS):
            status = "COMPLETE"
        else:
            status = "PARTIAL"

        origin_metrics: list[dict[str, Any]] = []
        for row in evaluated:
            output_metrics = {
                metric_id: (
                    None
                    if row["metrics"].get(metric_id) is None
                    else decimal_text(row["metrics"][metric_id])
                )
                for metric_id in METRIC_IDS
                if metric_id
                not in {
                    "icir_base",
                    "icir",
                    "origin_maturity_coverage",
                    "drawdown",
                    "stability",
                }
            }
            output_metrics["turnover"] = (
                None
                if turnover_by_origin[str(row["origin_id"])] is None
                else decimal_text(turnover_by_origin[str(row["origin_id"])])
            )
            origin_metrics.append(
                {
                    "label_session": row["label_session"],
                    "metrics": output_metrics,
                    "next_open_session": row["next_open_session"],
                    "origin_id": row["origin_id"],
                    "origin_session": row["origin_session"],
                    "q5_weights": row["q5_weights"],
                }
            )

        limitations: set[str] = {
            NON_TRADING_DIAGNOSTIC_PATH,
            NAIVE_ANNUALIZATION_SERIAL_CORRELATION_UNADJUSTED,
            INDUSTRY_ADJUSTED_LABEL_DIAGNOSTIC_NOT_RESIDUAL_MODEL,
            FLAT_20BP_SPREAD_DIAGNOSTIC,
        }
        if gap_detected:
            limitations.add(ORIGIN_GAPS_NOT_BRIDGED)
        return {
            "status": status,
            "metrics": [metrics_by_id[metric_id] for metric_id in METRIC_IDS],
            "origin_metrics": origin_metrics,
            "limitations": _sorted_codes(limitations),
        }


__all__ = [
    "BLOCKER_CODES",
    "EMPTY_QUANTILE",
    "FACTOR_METRIC_FORMULA_VERSION",
    "FLAT_20BP_SPREAD_DIAGNOSTIC",
    "FLAT_SPREAD_COST",
    "HIGHER_IS_BETTER",
    "INDUSTRY_ADJUSTED_LABEL_DIAGNOSTIC_NOT_RESIDUAL_MODEL",
    "INSUFFICIENT_AVAILABLE_ORIGINS",
    "INSUFFICIENT_IC_ORIGINS",
    "INSUFFICIENT_INDUSTRY_MAPPING_COVERAGE",
    "INSUFFICIENT_JOINT_COVERAGE",
    "INSUFFICIENT_SYMBOLS",
    "LIMITATION_CODES",
    "LOWER_IS_BETTER",
    "METRIC_IDS",
    "NAIVE_ANNUALIZATION_SERIAL_CORRELATION_UNADJUSTED",
    "NON_TRADING_DIAGNOSTIC_PATH",
    "NO_CONSECUTIVE_ORIGIN_TRANSITIONS",
    "ORIENTATIONS",
    "ORIGIN_GAPS_NOT_BRIDGED",
    "OVERLAPPING_FORWARD_WINDOWS",
    "QUANTILE_METRIC_IDS",
    "RETURN_NOT_COMPOUNDABLE",
    "ZERO_IC_VARIANCE",
    "ZERO_RETURN_VARIANCE",
    "ZERO_SCORE_VARIANCE",
    "evaluate_factor",
]
