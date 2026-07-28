"""Pure walk-forward calibration for v17 v3 branch fusion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal, localcontext
import hashlib
import json
from zoneinfo import ZoneInfo

import numpy as np

from .branch_fusion import BranchOutput, fuse_branches
from .decimal_normalization import DECIMAL_PRECISION, DecimalInput, normalize_decimal

SHANGHAI = ZoneInfo("Asia/Shanghai")
CANDIDATE_QUANT_WEIGHTS = tuple(Decimal(f"{value / 100:.2f}") for value in range(25, 76, 5))
CALIBRATION_MONTHS = 60
OUTER_FOLDS = 5
OUTER_MONTHS_PER_FOLD = 12
OUTER_OOS_MONTHS = OUTER_FOLDS * OUTER_MONTHS_PER_FOLD
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_BLOCK_LENGTH = 12
BOOTSTRAP_BLOCKS = 5
BOOTSTRAP_SEED = 170_317
BOOTSTRAP_MATRIX_VERSION = "myquant.v17.v3.bootstrap-index-matrix.v1"
BOOTSTRAP_MATRIX_HEADER = {
    "version": BOOTSTRAP_MATRIX_VERSION,
    "shape": [BOOTSTRAP_REPLICATES, CALIBRATION_MONTHS],
    "dtype": "<i8",
    "order": "C",
}


class CalibrationError(ValueError):
    """Fail-closed calibration input or statistical validity error."""


@dataclass(frozen=True)
class MonthEndOrigin:
    month: str
    session: date
    origin_at: datetime


@dataclass(frozen=True)
class CalibrationMonth:
    origin: date | datetime | str
    label_252_end_session: date | datetime | str
    ordered_pool: tuple[str, ...]
    quant_branch: BranchOutput | Mapping[str, object]
    fundamental_branch: BranchOutput | Mapping[str, object]
    forward_return_60: Mapping[str, DecimalInput]
    forward_return_252: Mapping[str, DecimalInput]
    label_252_mature: bool = True


@dataclass(frozen=True)
class MonthlyFusionMetric:
    origin: date
    quant_weight: Decimal
    hit60: Decimal
    q25_252: Decimal


@dataclass(frozen=True)
class WeightAssessment:
    quant_weight: Decimal
    valid: bool
    mean_hit60: Decimal | None
    se60: Decimal | None
    z60: Decimal | None
    mean_q25_252: Decimal | None
    se252: Decimal | None
    z252: Decimal | None
    blocker: str | None = None


@dataclass(frozen=True)
class CalibrationFold:
    index: int
    training_origins: tuple[date, ...]
    oos_origins: tuple[date, ...]
    selected_weight: Decimal
    selected_assessment: WeightAssessment


@dataclass(frozen=True)
class CalibrationResult:
    status: str
    promoted: bool
    active_weight: Decimal
    active_assessment: WeightAssessment
    folds: tuple[CalibrationFold, ...]
    oos_mean_hit60: Decimal
    oos_mean_q25_252: Decimal
    oos_p5_hit60: Decimal
    oos_p5_q25_252: Decimal
    bootstrap_matrix_sha256: str
    evidence_bound: str = "research_screening_bound"
    effective_outer_blocks: int = OUTER_FOLDS
    blockers: tuple[str, ...] = ()


def _as_date(value: date | datetime | str, *, label: str) -> date:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise CalibrationError(f"{label} datetime must be timezone-aware")
        return value.astimezone(SHANGHAI).date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise CalibrationError(f"{label} must be an ISO date") from exc
        if parsed.isoformat() != value:
            raise CalibrationError(f"{label} must be a canonical ISO date")
        return parsed
    raise CalibrationError(f"{label} must be a date, aware datetime, or ISO date")


def _next_month(month: str) -> str:
    year, number = (int(part) for part in month.split("-"))
    if number == 12:
        return f"{year + 1:04d}-01"
    return f"{year:04d}-{number + 1:02d}"


def schedule_month_end_origins(
    canonical_sessions: Sequence[date | datetime | str],
) -> tuple[MonthEndOrigin, ...]:
    """Map each consecutive Shanghai calendar month to its last canonical session."""

    sessions = tuple(_as_date(value, label="canonical_session") for value in canonical_sessions)
    if not sessions:
        raise CalibrationError("canonical sessions must not be empty")
    if any(left >= right for left, right in zip(sessions, sessions[1:])):
        raise CalibrationError("canonical sessions must be unique and strictly increasing")
    by_month: dict[str, date] = {}
    for session in sessions:
        by_month[session.strftime("%Y-%m")] = session
    ordered_months = tuple(by_month)
    for previous, current in zip(ordered_months, ordered_months[1:]):
        if _next_month(previous) != current:
            raise CalibrationError(f"scheduled month skipped:{previous}:{current}")
    return tuple(
        MonthEndOrigin(
            month=month,
            session=session,
            origin_at=datetime.combine(session, time(15, 0), tzinfo=SHANGHAI),
        )
        for month, session in by_month.items()
    )


def circular_moving_block_bootstrap_matrix() -> np.ndarray:
    """Return the frozen replicate-major 10000x60 circular block matrix."""

    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    starts = generator.integers(
        0,
        CALIBRATION_MONTHS,
        size=(BOOTSTRAP_REPLICATES, BOOTSTRAP_BLOCKS),
        dtype=np.int64,
    )
    offsets = np.arange(BOOTSTRAP_BLOCK_LENGTH, dtype=np.int64)
    matrix = np.concatenate(
        [
            (starts[:, block_index : block_index + 1] + offsets) % CALIBRATION_MONTHS
            for block_index in range(BOOTSTRAP_BLOCKS)
        ],
        axis=1,
    )
    return np.ascontiguousarray(matrix, dtype="<i8")


def bootstrap_matrix_header_bytes() -> bytes:
    return json.dumps(
        BOOTSTRAP_MATRIX_HEADER,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def bootstrap_matrix_sha256(matrix: np.ndarray | None = None) -> str:
    payload = circular_moving_block_bootstrap_matrix() if matrix is None else matrix
    if (
        payload.shape != (BOOTSTRAP_REPLICATES, CALIBRATION_MONTHS)
        or payload.dtype.str != "<i8"
        or not payload.flags.c_contiguous
    ):
        raise CalibrationError("bootstrap matrix identity mismatch")
    digest = hashlib.sha256()
    digest.update(bootstrap_matrix_header_bytes())
    digest.update(payload.tobytes(order="C"))
    return digest.hexdigest()


def _linear_quantile(values: Sequence[Decimal], probability: Decimal) -> Decimal:
    if not values:
        raise CalibrationError("quantile values must not be empty")
    if not Decimal("0") <= probability <= Decimal("1"):
        raise CalibrationError("quantile probability must be within [0,1]")
    ordered = sorted(values)
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        position = Decimal(len(ordered) - 1) * probability
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - Decimal(lower)
        result = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
        return +result


def _mean(values: Sequence[Decimal]) -> Decimal:
    if not values:
        raise CalibrationError("mean values must not be empty")
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        result = sum(values, Decimal("0")) / Decimal(len(values))
        return +result


def _bootstrap_means(
    values: Sequence[Decimal],
    matrix: np.ndarray,
) -> tuple[Decimal, ...]:
    if len(values) != CALIBRATION_MONTHS:
        raise CalibrationError("bootstrap requires exactly 60 monthly values")
    block_sums: list[Decimal] = []
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        for start in range(CALIBRATION_MONTHS):
            block_sums.append(
                sum(
                    (
                        values[(start + offset) % CALIBRATION_MONTHS]
                        for offset in range(BOOTSTRAP_BLOCK_LENGTH)
                    ),
                    Decimal("0"),
                )
            )
        divisor = Decimal(CALIBRATION_MONTHS)
        starts = matrix[:, ::BOOTSTRAP_BLOCK_LENGTH]
        result = tuple(
            sum((block_sums[int(start)] for start in row), Decimal("0")) / divisor for row in starts
        )
    return result


def _bootstrap_se(values: Sequence[Decimal], matrix: np.ndarray) -> Decimal:
    replicates = _bootstrap_means(values, matrix)
    center = _mean(replicates)
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        variance = sum(
            ((value - center) * (value - center) for value in replicates),
            Decimal("0"),
        ) / Decimal(len(replicates) - 1)
        result = variance.sqrt()
        return +result


def _coerce_return_map(
    value: Mapping[str, DecimalInput],
    *,
    label: str,
) -> Mapping[str, Decimal]:
    if not isinstance(value, Mapping):
        raise CalibrationError(f"{label} must be a mapping")
    result: dict[str, Decimal] = {}
    for symbol, raw in value.items():
        if not isinstance(symbol, str) or not symbol or symbol.strip() != symbol:
            raise CalibrationError(f"{label} contains invalid symbol")
        result[symbol] = normalize_decimal(raw, label=f"{label}.{symbol}")
    return result


def _coerce_month(
    value: CalibrationMonth | Mapping[str, object],
) -> CalibrationMonth:
    if isinstance(value, CalibrationMonth):
        month = value
    elif isinstance(value, Mapping):
        pool = value.get("ordered_pool")
        if isinstance(pool, (str, bytes)) or not isinstance(pool, Sequence):
            raise CalibrationError("calibration month ordered_pool must be a sequence")
        month = CalibrationMonth(
            origin=value.get("origin"),  # type: ignore[arg-type]
            label_252_end_session=value.get("label_252_end_session"),  # type: ignore[arg-type]
            ordered_pool=tuple(str(item) for item in pool),
            quant_branch=value.get("quant_branch"),  # type: ignore[arg-type]
            fundamental_branch=value.get("fundamental_branch"),  # type: ignore[arg-type]
            forward_return_60=value.get("forward_return_60"),  # type: ignore[arg-type]
            forward_return_252=value.get("forward_return_252"),  # type: ignore[arg-type]
            label_252_mature=value.get("label_252_mature"),  # type: ignore[arg-type]
        )
    else:
        raise CalibrationError("calibration months must be dataclasses or mappings")
    if type(month.label_252_mature) is not bool:
        raise CalibrationError("label_252_mature must be boolean")
    return month


def _monthly_metrics(
    month: CalibrationMonth,
    origin: date,
) -> Mapping[Decimal, MonthlyFusionMetric]:
    returns60 = _coerce_return_map(
        month.forward_return_60,
        label=f"{origin}.forward_return_60",
    )
    returns252 = _coerce_return_map(
        month.forward_return_252,
        label=f"{origin}.forward_return_252",
    )
    output: dict[Decimal, MonthlyFusionMetric] = {}
    for weight in CANDIDATE_QUANT_WEIGHTS:
        fused = fuse_branches(
            month.quant_branch,
            month.fundamental_branch,
            ordered_pool=month.ordered_pool,
            quant_weight=weight,
            top_n=24,
        )
        if len(fused.common_ready_domain) < 24:
            raise CalibrationError(f"common_ready_below_24:{origin.isoformat()}")
        selected = fused.selected_symbols
        missing60 = [symbol for symbol in selected if symbol not in returns60]
        missing252 = [symbol for symbol in selected if symbol not in returns252]
        if missing60 or missing252:
            raise CalibrationError(f"forward_labels_incomplete:{origin.isoformat()}")
        with localcontext() as context:
            context.prec = DECIMAL_PRECISION
            hit60 = Decimal(sum(returns60[symbol] > 0 for symbol in selected)) / Decimal(
                len(selected)
            )
        q25 = _linear_quantile(
            [returns252[symbol] for symbol in selected],
            Decimal("0.25"),
        )
        output[weight] = MonthlyFusionMetric(origin, weight, hit60, q25)
    return output


def _assess_weight(
    weight: Decimal,
    metrics: Sequence[MonthlyFusionMetric],
    matrix: np.ndarray,
) -> WeightAssessment:
    if len(metrics) != CALIBRATION_MONTHS:
        raise CalibrationError("weight assessment requires exactly 60 months")
    hit_values = tuple(item.hit60 for item in metrics)
    q25_values = tuple(item.q25_252 for item in metrics)
    mean_hit = _mean(hit_values)
    mean_q25 = _mean(q25_values)
    se60 = _bootstrap_se(hit_values, matrix)
    se252 = _bootstrap_se(q25_values, matrix)
    if se60 == 0 or se252 == 0:
        return WeightAssessment(
            quant_weight=weight,
            valid=False,
            mean_hit60=mean_hit,
            se60=se60,
            z60=None,
            mean_q25_252=mean_q25,
            se252=se252,
            z252=None,
            blocker="zero_bootstrap_variance",
        )
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        z60 = +((mean_hit - Decimal("0.50")) / se60)
        z252 = +(mean_q25 / se252)
    if not z60.is_finite() or not z252.is_finite():
        return WeightAssessment(
            quant_weight=weight,
            valid=False,
            mean_hit60=mean_hit,
            se60=se60,
            z60=None,
            mean_q25_252=mean_q25,
            se252=se252,
            z252=None,
            blocker="nonfinite_z_statistic",
        )
    return WeightAssessment(
        quant_weight=weight,
        valid=True,
        mean_hit60=mean_hit,
        se60=se60,
        z60=z60,
        mean_q25_252=mean_q25,
        se252=se252,
        z252=z252,
    )


def _select_weight(
    metrics_by_month: Sequence[Mapping[Decimal, MonthlyFusionMetric]],
    matrix: np.ndarray,
) -> WeightAssessment:
    assessments = tuple(
        _assess_weight(
            weight,
            [monthly[weight] for monthly in metrics_by_month],
            matrix,
        )
        for weight in CANDIDATE_QUANT_WEIGHTS
    )
    valid = tuple(item for item in assessments if item.valid)
    if not valid:
        raise CalibrationError("all_fusion_weights_invalid")

    def key(item: WeightAssessment) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        assert item.z60 is not None and item.z252 is not None
        return (
            min(item.z60, item.z252),
            item.z60 + item.z252,
            -abs(item.quant_weight - Decimal("0.50")),
            -item.quant_weight,
        )

    return max(valid, key=key)


def select_fusion_weight(
    metrics_by_month: Sequence[Mapping[Decimal, MonthlyFusionMetric]],
    *,
    bootstrap_matrix: np.ndarray | None = None,
) -> WeightAssessment:
    """Select one weight from exactly 60 months with the approved z tie rules."""

    if len(metrics_by_month) != CALIBRATION_MONTHS:
        raise CalibrationError("fusion weight selection requires exactly 60 months")
    matrix = (
        circular_moving_block_bootstrap_matrix() if bootstrap_matrix is None else bootstrap_matrix
    )
    if (
        matrix.shape != (BOOTSTRAP_REPLICATES, CALIBRATION_MONTHS)
        or matrix.dtype.str != "<i8"
        or not matrix.flags.c_contiguous
    ):
        raise CalibrationError("bootstrap matrix identity mismatch")
    return _select_weight(metrics_by_month, matrix)


def calibrate_fusion(
    months: Sequence[CalibrationMonth | Mapping[str, object]],
    *,
    canonical_sessions: Sequence[date | datetime | str] | None = None,
    scheduled_origins: Sequence[date | datetime | str] | None = None,
    active_cutoff: date | datetime | str | None = None,
) -> CalibrationResult:
    """Run five rolling 60m->12m folds and independently refit active weight."""

    if canonical_sessions is None:
        raise CalibrationError("canonical_sessions are required to verify 252-session label ends")
    normalized_months = tuple(_coerce_month(item) for item in months)
    if not normalized_months:
        raise CalibrationError("calibration months must not be empty")
    observed_origins = tuple(
        _as_date(item.origin, label="calibration_month.origin") for item in normalized_months
    )
    full_schedule = schedule_month_end_origins(canonical_sessions)
    schedule_by_month = {item.month: item.session for item in full_schedule}
    periods = tuple(value.strftime("%Y-%m") for value in observed_origins)
    if len(periods) != len(set(periods)):
        raise CalibrationError("calibration months contain duplicate calendar month")
    for previous, current in zip(periods, periods[1:]):
        if _next_month(previous) != current:
            raise CalibrationError(f"scheduled month skipped:{previous}:{current}")
    canonical_expected = tuple(schedule_by_month.get(period) for period in periods)
    if any(value is None for value in canonical_expected):
        raise CalibrationError("calibration month absent from canonical sessions")
    expected_origins = tuple(value for value in canonical_expected if value is not None)
    if scheduled_origins is not None:
        declared_origins = tuple(
            _as_date(value, label="scheduled_origin") for value in scheduled_origins
        )
        if declared_origins != expected_origins:
            raise CalibrationError("declared scheduled origins do not match canonical month ends")
    if observed_origins != expected_origins:
        raise CalibrationError("calibration months do not exactly match scheduled origins")

    sessions = tuple(_as_date(value, label="canonical_session") for value in canonical_sessions)
    session_index = {session: index for index, session in enumerate(sessions)}
    label_ends = tuple(
        _as_date(item.label_252_end_session, label="label_252_end_session")
        for item in normalized_months
    )
    for origin, label_end in zip(observed_origins, label_ends, strict=True):
        if origin not in session_index or label_end not in session_index:
            raise CalibrationError("origin or 252 label end absent from canonical sessions")
        if session_index[label_end] - session_index[origin] != 252:
            raise CalibrationError(f"label_252_end_session_offset_invalid:{origin.isoformat()}")
    cutoff = (
        sessions[-1] if active_cutoff is None else _as_date(active_cutoff, label="active_cutoff")
    )
    if cutoff not in session_index:
        raise CalibrationError("active_cutoff must be a canonical session")
    mature_indices = [
        index
        for index, (month, label_end) in enumerate(zip(normalized_months, label_ends, strict=True))
        if month.label_252_mature and label_end <= cutoff
    ]
    if len(mature_indices) < OUTER_OOS_MONTHS:
        raise CalibrationError("fewer_than_60_mature_scheduled_months")
    oos_indices = mature_indices[-OUTER_OOS_MONTHS:]
    if oos_indices != list(range(oos_indices[0], oos_indices[0] + OUTER_OOS_MONTHS)):
        raise CalibrationError("outer_oos_months_not_consecutive")

    matrix = circular_moving_block_bootstrap_matrix()
    all_metrics = tuple(
        _monthly_metrics(month, origin)
        for month, origin in zip(normalized_months, expected_origins, strict=True)
    )
    folds: list[CalibrationFold] = []
    stitched_oos: list[MonthlyFusionMetric] = []
    for fold_index in range(OUTER_FOLDS):
        fold_oos_indices = oos_indices[
            fold_index * OUTER_MONTHS_PER_FOLD : (fold_index + 1) * OUTER_MONTHS_PER_FOLD
        ]
        fold_start_index = fold_oos_indices[0]
        fold_start_origin = expected_origins[fold_start_index]
        eligible_training = [
            index
            for index in range(fold_start_index)
            if normalized_months[index].label_252_mature and label_ends[index] < fold_start_origin
        ]
        if len(eligible_training) < CALIBRATION_MONTHS:
            raise CalibrationError(
                f"fold_{fold_index + 1}_fewer_than_60_leakage_free_training_months"
            )
        training_indices = eligible_training[-CALIBRATION_MONTHS:]
        if training_indices != list(
            range(training_indices[0], training_indices[0] + CALIBRATION_MONTHS)
        ):
            raise CalibrationError(f"fold_{fold_index + 1}_training_months_not_consecutive")
        if any(label_ends[index] >= fold_start_origin for index in training_indices):
            raise CalibrationError(f"fold_{fold_index + 1}_training_label_leakage")
        selected = _select_weight(
            [all_metrics[index] for index in training_indices],
            matrix,
        )
        oos_metrics = [all_metrics[index][selected.quant_weight] for index in fold_oos_indices]
        stitched_oos.extend(oos_metrics)
        folds.append(
            CalibrationFold(
                index=fold_index + 1,
                training_origins=tuple(expected_origins[index] for index in training_indices),
                oos_origins=tuple(expected_origins[index] for index in fold_oos_indices),
                selected_weight=selected.quant_weight,
                selected_assessment=selected,
            )
        )
    if len(stitched_oos) != CALIBRATION_MONTHS:
        raise AssertionError("outer stitching must produce exactly 60 months")
    oos_hit = tuple(item.hit60 for item in stitched_oos)
    oos_q25 = tuple(item.q25_252 for item in stitched_oos)
    oos_hit_bootstrap = _bootstrap_means(oos_hit, matrix)
    oos_q25_bootstrap = _bootstrap_means(oos_q25, matrix)
    p5_hit = _linear_quantile(oos_hit_bootstrap, Decimal("0.05"))
    p5_q25 = _linear_quantile(oos_q25_bootstrap, Decimal("0.05"))
    blockers: list[str] = []
    if p5_hit <= Decimal("0.50"):
        blockers.append("oos_p5_hit60_not_above_0.50")
    if p5_q25 <= Decimal("0"):
        blockers.append("oos_p5_q25_252_not_positive")

    active_indices = mature_indices[-CALIBRATION_MONTHS:]
    if active_indices != list(range(active_indices[0], active_indices[0] + CALIBRATION_MONTHS)):
        raise CalibrationError("active_window_months_not_consecutive")
    if any(label_ends[index] > cutoff for index in active_indices):
        raise CalibrationError("active_window_contains_immature_252_label")
    active_metrics = [all_metrics[index] for index in active_indices]
    active_assessment = _select_weight(active_metrics, matrix)
    return CalibrationResult(
        status="READY" if not blockers else "PROMOTION_BLOCKED",
        promoted=not blockers,
        active_weight=active_assessment.quant_weight,
        active_assessment=active_assessment,
        folds=tuple(folds),
        oos_mean_hit60=_mean(oos_hit),
        oos_mean_q25_252=_mean(oos_q25),
        oos_p5_hit60=p5_hit,
        oos_p5_q25_252=p5_q25,
        bootstrap_matrix_sha256=bootstrap_matrix_sha256(matrix),
        blockers=tuple(blockers),
    )


__all__ = [
    "BOOTSTRAP_MATRIX_HEADER",
    "BOOTSTRAP_MATRIX_VERSION",
    "CANDIDATE_QUANT_WEIGHTS",
    "CalibrationError",
    "CalibrationFold",
    "CalibrationMonth",
    "CalibrationResult",
    "MonthEndOrigin",
    "MonthlyFusionMetric",
    "WeightAssessment",
    "bootstrap_matrix_header_bytes",
    "bootstrap_matrix_sha256",
    "calibrate_fusion",
    "circular_moving_block_bootstrap_matrix",
    "schedule_month_end_origins",
    "select_fusion_weight",
]
