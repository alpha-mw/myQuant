"""Fail-closed V17 v4 walk-forward calibration closure.

The closure keeps the 120-month decision schedule distinct from antecedent
training origins.  Antecedent origins may precede the closure window, but every
origin is reconstructed with the same PIT, history-span, same-pool, label, and
historical Factor-control rules.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
from typing import Any, Final, NoReturn
from zoneinfo import ZoneInfo

import numpy as np

from quant_investor.factors.production_control_v1 import (
    ACTIVE_SET_SCHEMA_VERSION,
    ARTIFACT_REF_SCHEMA_VERSION as FACTOR_ARTIFACT_REF_SCHEMA_VERSION,
    validate_active_set_pointer,
)
from quant_investor.v17_v4_contract import (
    PROTOCOL_VERSION,
    canonical_bytes,
    seal_semantic,
    validate_artifact,
)
from quant_investor.v17_v4_contract.canonical import (
    CanonicalContractError,
    load_canonical_resource,
    validate_semantic_sha,
)
from quant_investor.v17_v4_contract.identities import (
    IdentityContractError,
    require_opaque_id,
    require_sha256,
    require_utc_timestamp,
)
from quant_investor.v17_v4_contract.validators import (
    validate_typed_artifact,
)

SHANGHAI: Final = ZoneInfo("Asia/Shanghai")
CLOSURE_MONTHS: Final = 120
CALIBRATION_MONTHS: Final = 60
OUTER_FOLDS: Final = 5
OUTER_MONTHS_PER_FOLD: Final = 12
QUANT_MIN_OPEN_SESSIONS: Final = 1260
FUNDAMENTAL_MIN_OPEN_SESSIONS: Final = 2520
COMMON_READY_MINIMUM: Final = 24
TOP_N: Final = 24
BOOTSTRAP_REPLICATES: Final = 10_000
BOOTSTRAP_BLOCK_LENGTH: Final = 12
BOOTSTRAP_BLOCKS: Final = 5
BOOTSTRAP_SEED: Final = 170_317
BOOTSTRAP_MATRIX_VERSION: Final = (
    "myquant.v17.v4.bootstrap-index-matrix.v1"
)
CANDIDATE_QUANT_WEIGHTS: Final = tuple(
    Decimal(value) / Decimal(100) for value in range(25, 76, 5)
)
ORIGIN_INVENTORY_VERSION: Final = (
    "myquant.v17.v4.calibration-origin-inventory.v1"
)
CALIBRATION_RECEIPT_VERSION: Final = (
    "myquant.v17.v4.calibration-receipt.v1"
)
FUSION_PROMOTION_RECEIPT_VERSION: Final = (
    "myquant.v17.v4.fusion-promotion-receipt.v1"
)
_REF_FIELDS: Final = frozenset(
    {
        "artifact_id",
        "artifact_version",
        "byte_sha256",
        "cutoff",
        "relative_path",
        "semantic_sha256",
        "strategy_id",
    }
)
_FACTOR_REF_FIELDS: Final = frozenset(
    {
        "artifact_schema",
        "byte_sha256",
        "relative_path",
        "schema_version",
        "semantic_sha256",
    }
)
_ORIGIN_SOURCE_ROLES: Final = (
    "benchmark_total_return",
    "corporate_actions",
    "fundamental_branch",
    "initial_pool",
    "official_delisting_cash",
    "pit_catalog",
    "preselect_locator",
    "quant_branch",
)
_ORIGIN_SOURCE_VERSIONS: Final = {
    "benchmark_total_return": (
        "myquant.v17.v4.dataset.benchmark_total_return.v1"
    ),
    "corporate_actions": "myquant.v17.v4.dataset.corporate_actions.v1",
    "fundamental_branch": "myquant.v17.v4.branch-output.v1",
    "initial_pool": "myquant.v17.v4.initial-pool-output.v1",
    "official_delisting_cash": (
        "myquant.v17.v4.dataset.official_delisting_cash.v1"
    ),
    "pit_catalog": "myquant.v17.v4.pit-generation-catalog.v1",
    "preselect_locator": "myquant.v17.v4.preselect-locator.v1",
    "quant_branch": "myquant.v17.v4.branch-output.v1",
}
_LABEL_ROLES: Final = ("label_60", "label_252")
_NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}


class CalibrationClosureError(ValueError):
    """Raised when any calibration source, schedule, or statistic drifts."""

    exit_code = 2


def _blocked(reason: str) -> NoReturn:
    raise CalibrationClosureError(f"CALIBRATION_CLOSURE_BLOCKED:{reason}")


def _as_date(value: Any, *, label: str) -> date:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            _blocked(f"{label}_timezone_missing")
        return value.astimezone(SHANGHAI).date()
    if isinstance(value, date):
        return value
    if type(value) is str:
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            _blocked(f"{label}_invalid")
        if parsed.isoformat() != value:
            _blocked(f"{label}_noncanonical")
        return parsed
    _blocked(f"{label}_invalid")


def _instant(value: Any, *, label: str) -> datetime:
    try:
        text = require_utc_timestamp(value, label=label)
    except IdentityContractError:
        _blocked(f"{label}_invalid")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        _blocked(f"{label}_invalid")


def _decimal(value: Any, *, label: str) -> Decimal:
    if type(value) is bool or type(value) not in {str, int, float, Decimal}:
        _blocked(f"{label}_invalid")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        _blocked(f"{label}_invalid")
    if not result.is_finite():
        _blocked(f"{label}_nonfinite")
    return result


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        _blocked("decimal_nonfinite")
    result = format(value, "f")
    if "." in result:
        result = result.rstrip("0").rstrip(".")
    return result or "0"


def _next_month(month: str) -> str:
    year, number = (int(part) for part in month.split("-"))
    if number == 12:
        return f"{year + 1:04d}-01"
    return f"{year:04d}-{number + 1:02d}"


def _origin_cutoff(origin: date) -> datetime:
    return datetime.combine(
        origin,
        time(15, 0),
        tzinfo=SHANGHAI,
    ).astimezone(timezone.utc)


def _validate_ref(
    value: Mapping[str, Any],
    *,
    strategy_id: str,
    cutoff_at: datetime,
    expected_version: str | None = None,
    version_prefix: str | None = None,
    label: str,
) -> dict[str, str]:
    if type(value) is not dict or set(value) != set(_REF_FIELDS):
        _blocked(f"{label}_shape")
    try:
        require_opaque_id(value["artifact_id"], label=f"{label}.artifact_id")
        version = require_opaque_id(
            value["artifact_version"],
            label=f"{label}.artifact_version",
        )
        require_sha256(value["byte_sha256"], label=f"{label}.byte_sha256")
        require_sha256(
            value["semantic_sha256"],
            label=f"{label}.semantic_sha256",
        )
    except IdentityContractError:
        _blocked(f"{label}_identity")
    if (
        value["strategy_id"] != strategy_id
        or (expected_version is not None and version != expected_version)
        or (
            version_prefix is not None
            and not version.startswith(version_prefix)
        )
        or version.startswith("myquant.v17.v3.")
    ):
        _blocked(f"{label}_binding")
    observed_cutoff = _instant(value["cutoff"], label=f"{label}.cutoff")
    if observed_cutoff > cutoff_at:
        _blocked(f"{label}_after_cutoff")
    path = value["relative_path"]
    if (
        type(path) is not str
        or not path
        or path.startswith("/")
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        _blocked(f"{label}_path")
    return {field: str(value[field]) for field in _REF_FIELDS}


def _read_exact_artifact(
    reference: Mapping[str, str],
    *,
    artifact_loader: Callable[[Mapping[str, str]], bytes],
    label: str,
) -> dict[str, Any]:
    try:
        raw = artifact_loader(reference)
    except Exception as exc:
        raise CalibrationClosureError(
            f"CALIBRATION_CLOSURE_BLOCKED:{label}_read_failed"
        ) from exc
    if (
        type(raw) is not bytes
        or hashlib.sha256(raw).hexdigest()
        != reference["byte_sha256"]
    ):
        _blocked(f"{label}_byte_sha")
    try:
        value = load_canonical_resource(raw, label=label)
        normalized = validate_semantic_sha(value)
    except CanonicalContractError:
        _blocked(f"{label}_canonical")
    if (
        normalized.get("semantic_sha256")
        != reference["semantic_sha256"]
        or normalized.get("version")
        != reference["artifact_version"]
        or normalized.get("strategy_id") != reference["strategy_id"]
        or normalized.get("cutoff") != reference["cutoff"]
    ):
        _blocked(f"{label}_document_binding")
    identity = next(
        (
            normalized.get(field)
            for field in (
                "artifact_id",
                "catalog_id",
                "dataset_id",
                "inventory_id",
                "label_id",
                "locator_id",
                "output_id",
                "pointer_id",
                "receipt_id",
            )
            if field in normalized
        ),
        None,
    )
    if identity != reference["artifact_id"]:
        _blocked(f"{label}_artifact_id")
    return normalized


def _read_exact_bytes(
    reference: Mapping[str, str],
    *,
    artifact_loader: Callable[[Mapping[str, str]], bytes],
    label: str,
) -> bytes:
    try:
        raw = artifact_loader(reference)
    except Exception as exc:
        raise CalibrationClosureError(
            f"CALIBRATION_CLOSURE_BLOCKED:{label}_read_failed"
        ) from exc
    if (
        type(raw) is not bytes
        or hashlib.sha256(raw).hexdigest()
        != reference["byte_sha256"]
    ):
        _blocked(f"{label}_byte_sha")
    return raw


def _validate_native_artifact(
    artifact: Mapping[str, Any],
    *,
    label: str,
) -> None:
    try:
        if artifact.get("version") == (
            "myquant.v17.v4.pit-generation-catalog.v1"
        ):
            validate_artifact(artifact)
        else:
            validate_typed_artifact(
                artifact,
                schema_checked=True,
            )
    except Exception:
        _blocked(f"{label}_native_schema")


def _validate_factor_ref(
    value: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, str]:
    if type(value) is not dict or set(value) != set(_FACTOR_REF_FIELDS):
        _blocked(f"{label}_shape")
    if (
        value.get("artifact_schema") != ACTIVE_SET_SCHEMA_VERSION
        or value.get("schema_version")
        != FACTOR_ARTIFACT_REF_SCHEMA_VERSION
    ):
        _blocked(f"{label}_schema")
    try:
        require_sha256(value["byte_sha256"], label=f"{label}.byte_sha256")
        require_sha256(
            value["semantic_sha256"],
            label=f"{label}.semantic_sha256",
        )
    except IdentityContractError:
        _blocked(f"{label}_identity")
    path = value["relative_path"]
    if (
        type(path) is not str
        or not path
        or path.startswith("/")
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        _blocked(f"{label}_path")
    return {field: str(value[field]) for field in _FACTOR_REF_FIELDS}


def _read_exact_factor_active_set(
    reference: Mapping[str, str],
    *,
    artifact_loader: Callable[[Mapping[str, str]], bytes],
    label: str,
) -> dict[str, Any]:
    try:
        raw = artifact_loader(reference)
    except Exception as exc:
        raise CalibrationClosureError(
            f"CALIBRATION_CLOSURE_BLOCKED:{label}_read_failed"
        ) from exc
    if (
        type(raw) is not bytes
        or hashlib.sha256(raw).hexdigest()
        != reference["byte_sha256"]
    ):
        _blocked(f"{label}_byte_sha")
    try:
        value = load_canonical_resource(raw, label=label)
        normalized = validate_active_set_pointer(value)
    except (CanonicalContractError, ValueError):
        _blocked(f"{label}_canonical")
    if (
        normalized["schema_version"] != reference["artifact_schema"]
        or normalized["semantic_sha256"]
        != reference["semantic_sha256"]
    ):
        _blocked(f"{label}_document_binding")
    return normalized


@dataclass(frozen=True)
class OriginCalibrationInput:
    origin: date | datetime | str
    quant_history_start_session: date | datetime | str
    fundamental_history_start_session: date | datetime | str
    label_60_end_session: date | datetime | str
    label_252_end_session: date | datetime | str
    ordered_pool: tuple[str, ...]
    quant_scores: Mapping[str, Any]
    fundamental_scores: Mapping[str, Any]
    forward_return_60: Mapping[str, Any]
    forward_return_252: Mapping[str, Any]
    delisted_60: Mapping[str, bool]
    delisted_252: Mapping[str, bool]
    official_terminal_cash_60: Mapping[str, bool]
    official_terminal_cash_252: Mapping[str, bool]
    source_refs: Mapping[str, Mapping[str, Any]]
    label_refs: Mapping[str, Mapping[str, Any]]
    factor_active_set_ref: Mapping[str, Any]
    factor_effective_from_session: date | datetime | str
    factor_effective_to_session: date | datetime | str | None
    factor_set_sha256: str


@dataclass(frozen=True)
class MonthlyMetric:
    origin: date
    quant_weight: Decimal
    hit60: Decimal
    q25_252: Decimal


@dataclass(frozen=True)
class WeightAssessment:
    quant_weight: Decimal
    mean_hit60: Decimal
    mean_q25_252: Decimal
    hit60_lower_95: Decimal
    q25_252_lower_95: Decimal


@dataclass(frozen=True)
class CalibrationFold:
    index: int
    training_origins: tuple[date, ...]
    oos_origins: tuple[date, ...]
    selected_weight: Decimal


@dataclass(frozen=True)
class CalibrationClosure:
    strategy_id: str
    run_id: str
    cutoff: str
    input_origin_count: int
    closure_origins: tuple[date, ...]
    active_refit_origins: tuple[date, ...]
    folds: tuple[CalibrationFold, ...]
    active_assessment: WeightAssessment
    oos_mean_hit60: Decimal
    oos_mean_q25_252: Decimal
    oos_hit60_lower_95: Decimal
    oos_q25_252_lower_95: Decimal
    bootstrap_matrix_sha256: str
    promoted: bool
    blockers: tuple[str, ...]
    origin_inventory: Mapping[str, Any]


def schedule_month_end_origins(
    canonical_sessions: Sequence[date | datetime | str],
) -> tuple[date, ...]:
    sessions = tuple(
        _as_date(value, label="canonical_session")
        for value in canonical_sessions
    )
    if not sessions or any(
        left >= right for left, right in zip(sessions, sessions[1:])
    ):
        _blocked("canonical_sessions_not_strict")
    by_month: dict[str, date] = {}
    for session in sessions:
        by_month[session.strftime("%Y-%m")] = session
    months = tuple(by_month)
    for previous, current in zip(months, months[1:]):
        if _next_month(previous) != current:
            _blocked(f"scheduled_month_skipped:{previous}:{current}")
    return tuple(by_month.values())


def circular_block_bootstrap_matrix() -> np.ndarray:
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
            (
                starts[:, block_index : block_index + 1]
                + offsets
            )
            % CALIBRATION_MONTHS
            for block_index in range(BOOTSTRAP_BLOCKS)
        ],
        axis=1,
    )
    return np.ascontiguousarray(matrix, dtype="<i8")


def bootstrap_matrix_sha256(
    matrix: np.ndarray | None = None,
) -> str:
    payload = (
        circular_block_bootstrap_matrix()
        if matrix is None
        else matrix
    )
    if (
        payload.shape != (BOOTSTRAP_REPLICATES, CALIBRATION_MONTHS)
        or payload.dtype.str != "<i8"
        or not payload.flags.c_contiguous
    ):
        _blocked("bootstrap_matrix_identity")
    header = {
        "dtype": "<i8",
        "order": "C",
        "shape": [BOOTSTRAP_REPLICATES, CALIBRATION_MONTHS],
        "version": BOOTSTRAP_MATRIX_VERSION,
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            header,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(payload.tobytes(order="C"))
    return digest.hexdigest()


def _mapping(
    value: Mapping[str, Any],
    *,
    symbols: tuple[str, ...],
    label: str,
    boolean: bool = False,
) -> dict[str, Decimal | bool]:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(symbols)
    ):
        _blocked(f"{label}_domain")
    result: dict[str, Decimal | bool] = {}
    for symbol in symbols:
        raw = value[symbol]
        if boolean:
            if type(raw) is not bool:
                _blocked(f"{label}_not_boolean:{symbol}")
            result[symbol] = raw
        else:
            result[symbol] = _decimal(raw, label=f"{label}.{symbol}")
    return result


def _percentiles(
    values: Mapping[str, Decimal],
) -> dict[str, Decimal]:
    count = Decimal(len(values))
    ordered = sorted(values.values())
    return {
        symbol: Decimal(
            sum(candidate <= value for candidate in ordered)
        )
        / count
        for symbol, value in values.items()
    }


def _linear_quantile(
    values: Sequence[Decimal],
    probability: Decimal,
) -> Decimal:
    if not values:
        _blocked("quantile_empty")
    ordered = sorted(values)
    with localcontext() as context:
        context.prec = 40
        position = Decimal(len(ordered) - 1) * probability
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - Decimal(lower)
        return +(
            ordered[lower]
            + (ordered[upper] - ordered[lower]) * fraction
        )


def _mean(values: Sequence[Decimal]) -> Decimal:
    if not values:
        _blocked("mean_empty")
    with localcontext() as context:
        context.prec = 40
        return +(sum(values, Decimal(0)) / Decimal(len(values)))


def _monthly_metrics(
    *,
    origin: date,
    pool: tuple[str, ...],
    quant_scores: Mapping[str, Decimal],
    fundamental_scores: Mapping[str, Decimal],
    forward60: Mapping[str, Decimal],
    forward252: Mapping[str, Decimal],
) -> dict[Decimal, MonthlyMetric]:
    quant_percentiles = _percentiles(quant_scores)
    fundamental_percentiles = _percentiles(fundamental_scores)
    metrics: dict[Decimal, MonthlyMetric] = {}
    for weight in CANDIDATE_QUANT_WEIGHTS:
        fused = {
            symbol: (
                weight * quant_percentiles[symbol]
                + (Decimal(1) - weight)
                * fundamental_percentiles[symbol]
            )
            for symbol in pool
        }
        selected = tuple(
            sorted(pool, key=lambda symbol: (-fused[symbol], symbol))[
                :TOP_N
            ]
        )
        if len(selected) != TOP_N:
            _blocked(f"common_ready_below_24:{origin}")
        hit60 = Decimal(
            sum(forward60[symbol] > 0 for symbol in selected)
        ) / Decimal(TOP_N)
        q25 = _linear_quantile(
            [forward252[symbol] for symbol in selected],
            Decimal("0.25"),
        )
        metrics[weight] = MonthlyMetric(
            origin=origin,
            quant_weight=weight,
            hit60=hit60,
            q25_252=q25,
        )
    return metrics


def _bootstrap_lower(
    values: Sequence[Decimal],
    matrix: np.ndarray,
) -> Decimal:
    if len(values) != CALIBRATION_MONTHS:
        _blocked("bootstrap_requires_60_months")
    array = np.asarray([float(value) for value in values], dtype="<f8")
    replicates = array[matrix].mean(axis=1)
    lower = float(np.quantile(replicates, 0.05, method="linear"))
    if not np.isfinite(lower):
        _blocked("bootstrap_nonfinite")
    return Decimal(str(lower))


def _assessment(
    weight: Decimal,
    metrics: Sequence[MonthlyMetric],
    matrix: np.ndarray,
) -> WeightAssessment:
    hit = tuple(row.hit60 for row in metrics)
    q25 = tuple(row.q25_252 for row in metrics)
    return WeightAssessment(
        quant_weight=weight,
        mean_hit60=_mean(hit),
        mean_q25_252=_mean(q25),
        hit60_lower_95=_bootstrap_lower(hit, matrix),
        q25_252_lower_95=_bootstrap_lower(q25, matrix),
    )


def _select_weight(
    metrics: Sequence[Mapping[Decimal, MonthlyMetric]],
    matrix: np.ndarray,
) -> WeightAssessment:
    if len(metrics) != CALIBRATION_MONTHS:
        _blocked("weight_selection_requires_60_months")
    assessments = tuple(
        _assessment(
            weight,
            [month[weight] for month in metrics],
            matrix,
        )
        for weight in CANDIDATE_QUANT_WEIGHTS
    )

    def key(value: WeightAssessment) -> tuple[Decimal, ...]:
        return (
            min(
                value.hit60_lower_95 - Decimal("0.50"),
                value.q25_252_lower_95,
            ),
            value.hit60_lower_95 + value.q25_252_lower_95,
            -abs(value.quant_weight - Decimal("0.50")),
            -value.quant_weight,
        )

    return max(assessments, key=key)


def _origin_semantic(
    *,
    origin: date,
    quant_history_start: date,
    fundamental_history_start: date,
    label_60_end: date,
    label_252_end: date,
    pool: tuple[str, ...],
    quant_scores: Mapping[str, Decimal],
    fundamental_scores: Mapping[str, Decimal],
    forward60: Mapping[str, Decimal],
    forward252: Mapping[str, Decimal],
    source_refs: Mapping[str, Mapping[str, str]],
    label_refs: Mapping[str, Mapping[str, str]],
    factor_ref: Mapping[str, str],
    factor_effective_from: date,
    factor_effective_to: date | None,
    factor_set_sha256: str,
) -> str:
    payload = {
        "factor_active_set_ref": factor_ref,
        "factor_effective_from_session": factor_effective_from.isoformat(),
        "factor_effective_to_session": (
            None
            if factor_effective_to is None
            else factor_effective_to.isoformat()
        ),
        "factor_set_sha256": factor_set_sha256,
        "forward_return_252": {
            symbol: _decimal_text(forward252[symbol])
            for symbol in sorted(forward252)
        },
        "forward_return_60": {
            symbol: _decimal_text(forward60[symbol])
            for symbol in sorted(forward60)
        },
        "fundamental_history_start_session": (
            fundamental_history_start.isoformat()
        ),
        "fundamental_scores": {
            symbol: _decimal_text(fundamental_scores[symbol])
            for symbol in sorted(fundamental_scores)
        },
        "label_252_end_session": label_252_end.isoformat(),
        "label_60_end_session": label_60_end.isoformat(),
        "label_refs": label_refs,
        "ordered_pool": list(pool),
        "origin": origin.isoformat(),
        "quant_history_start_session": quant_history_start.isoformat(),
        "quant_scores": {
            symbol: _decimal_text(quant_scores[symbol])
            for symbol in sorted(quant_scores)
        },
        "source_refs": source_refs,
    }
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def run_calibration_closure(
    origins: Sequence[OriginCalibrationInput],
    *,
    canonical_sessions: Sequence[date | datetime | str],
    active_cutoff: date | datetime | str,
    strategy_id: str,
    run_id: str,
    cutoff: str,
    created_at: str,
    artifact_loader: Callable[[Mapping[str, str]], bytes],
) -> CalibrationClosure:
    try:
        strategy = require_opaque_id(strategy_id, label="strategy_id")
        run = require_opaque_id(run_id, label="run_id")
        cutoff_text = require_utc_timestamp(cutoff, label="cutoff")
        created_text = require_utc_timestamp(
            created_at,
            label="created_at",
        )
    except IdentityContractError:
        _blocked("artifact_identity")
    cutoff_at = _instant(cutoff_text, label="cutoff")
    created = _instant(created_text, label="created_at")
    if created < cutoff_at:
        _blocked("created_at_before_cutoff")
    sessions = tuple(
        _as_date(value, label="canonical_session")
        for value in canonical_sessions
    )
    if len(sessions) < FUNDAMENTAL_MIN_OPEN_SESSIONS + 252:
        _blocked("canonical_session_span_insufficient")
    if any(left >= right for left, right in zip(sessions, sessions[1:])):
        _blocked("canonical_sessions_not_strict")
    session_index = {session: index for index, session in enumerate(sessions)}
    cutoff_session = _as_date(active_cutoff, label="active_cutoff")
    if cutoff_session not in session_index:
        _blocked("active_cutoff_not_open_session")
    normalized_origins = tuple(
        _as_date(item.origin, label="origin") for item in origins
    )
    if len(normalized_origins) < CLOSURE_MONTHS:
        _blocked("origin_count_below_120")
    months = tuple(origin.strftime("%Y-%m") for origin in normalized_origins)
    if len(months) != len(set(months)):
        _blocked("duplicate_origin_month")
    for previous, current in zip(months, months[1:]):
        if _next_month(previous) != current:
            _blocked(f"scheduled_month_skipped:{previous}:{current}")
    scheduled = {
        value.strftime("%Y-%m"): value
        for value in schedule_month_end_origins(sessions)
    }
    if any(
        scheduled.get(month) != origin
        for month, origin in zip(months, normalized_origins, strict=True)
    ):
        _blocked("origin_not_shanghai_month_end")
    closure_origins = normalized_origins[-CLOSURE_MONTHS:]
    oos_origins = closure_origins[-CALIBRATION_MONTHS:]

    metrics: list[dict[Decimal, MonthlyMetric]] = []
    label_252_ends: list[date] = []
    inventory_rows: list[dict[str, Any]] = []
    for item, origin in zip(origins, normalized_origins, strict=True):
        if origin not in session_index:
            _blocked(f"origin_not_open_session:{origin}")
        origin_at = _origin_cutoff(origin)
        if origin_at > cutoff_at:
            _blocked(f"origin_after_cutoff:{origin}")
        quant_start = _as_date(
            item.quant_history_start_session,
            label="quant_history_start_session",
        )
        fundamental_start = _as_date(
            item.fundamental_history_start_session,
            label="fundamental_history_start_session",
        )
        label_60_end = _as_date(
            item.label_60_end_session,
            label="label_60_end_session",
        )
        label_252_end = _as_date(
            item.label_252_end_session,
            label="label_252_end_session",
        )
        if any(
            value not in session_index
            for value in (
                quant_start,
                fundamental_start,
                label_60_end,
                label_252_end,
            )
        ):
            _blocked(f"origin_session_binding_missing:{origin}")
        origin_index = session_index[origin]
        if origin_index - session_index[quant_start] + 1 < (
            QUANT_MIN_OPEN_SESSIONS
        ):
            _blocked(f"quant_span_below_1260:{origin}")
        if origin_index - session_index[fundamental_start] + 1 < (
            FUNDAMENTAL_MIN_OPEN_SESSIONS
        ):
            _blocked(f"fundamental_span_below_2520:{origin}")
        if session_index[label_60_end] - origin_index != 60:
            _blocked(f"label_60_offset:{origin}")
        if session_index[label_252_end] - origin_index != 252:
            _blocked(f"label_252_offset:{origin}")
        if label_252_end > cutoff_session:
            _blocked(f"label_252_immature:{origin}")
        label_252_ends.append(label_252_end)

        pool = tuple(item.ordered_pool)
        if (
            len(pool) < COMMON_READY_MINIMUM
            or len(pool) > 500
            or len(pool) != len(set(pool))
            or any(
                type(symbol) is not str
                or not symbol
                or symbol != symbol.strip()
                for symbol in pool
            )
        ):
            _blocked(f"initial_pool_invalid:{origin}")
        quant_scores_raw = _mapping(
            item.quant_scores,
            symbols=pool,
            label=f"{origin}.quant_scores",
        )
        fundamental_scores_raw = _mapping(
            item.fundamental_scores,
            symbols=pool,
            label=f"{origin}.fundamental_scores",
        )
        forward60_raw = _mapping(
            item.forward_return_60,
            symbols=pool,
            label=f"{origin}.forward_return_60",
        )
        forward252_raw = _mapping(
            item.forward_return_252,
            symbols=pool,
            label=f"{origin}.forward_return_252",
        )
        delisted60 = _mapping(
            item.delisted_60,
            symbols=pool,
            label=f"{origin}.delisted_60",
            boolean=True,
        )
        delisted252 = _mapping(
            item.delisted_252,
            symbols=pool,
            label=f"{origin}.delisted_252",
            boolean=True,
        )
        terminal60 = _mapping(
            item.official_terminal_cash_60,
            symbols=pool,
            label=f"{origin}.official_terminal_cash_60",
            boolean=True,
        )
        terminal252 = _mapping(
            item.official_terminal_cash_252,
            symbols=pool,
            label=f"{origin}.official_terminal_cash_252",
            boolean=True,
        )
        for symbol in pool:
            if (
                delisted60[symbol] is True
                and terminal60[symbol] is not True
            ) or (
                delisted252[symbol] is True
                and terminal252[symbol] is not True
            ):
                _blocked(
                    f"delisting_without_official_terminal_cash:"
                    f"{origin}:{symbol}"
                )
        if (
            type(item.source_refs) is not dict
            or tuple(sorted(item.source_refs)) != _ORIGIN_SOURCE_ROLES
        ):
            _blocked(f"origin_source_role_inventory:{origin}")
        source_refs = {
            role: _validate_ref(
                item.source_refs[role],
                strategy_id=strategy,
                cutoff_at=origin_at,
                expected_version=_ORIGIN_SOURCE_VERSIONS[role],
                label=f"{origin}.source_refs.{role}",
            )
            for role in _ORIGIN_SOURCE_ROLES
        }
        canonical_source_roles = (
            "fundamental_branch",
            "initial_pool",
            "pit_catalog",
            "preselect_locator",
            "quant_branch",
        )
        source_artifacts = {
            role: _read_exact_artifact(
                source_refs[role],
                artifact_loader=artifact_loader,
                label=f"{origin}.source_artifacts.{role}",
            )
            for role in canonical_source_roles
        }
        for role in (
            "benchmark_total_return",
            "corporate_actions",
            "official_delisting_cash",
        ):
            _read_exact_bytes(
                source_refs[role],
                artifact_loader=artifact_loader,
                label=f"{origin}.source_artifacts.{role}",
            )
        for role in canonical_source_roles:
            _validate_native_artifact(
                source_artifacts[role],
                label=f"{origin}.{role}",
            )
        if (
            source_refs["quant_branch"]["semantic_sha256"]
            == source_refs["fundamental_branch"]["semantic_sha256"]
        ):
            _blocked(f"branch_identity_collision:{origin}")
        if (
            type(item.label_refs) is not dict
            or set(item.label_refs) != set(_LABEL_ROLES)
        ):
            _blocked(f"label_role_inventory:{origin}")
        label_refs = {
            role: _validate_ref(
                item.label_refs[role],
                strategy_id=strategy,
                cutoff_at=cutoff_at,
                expected_version=(
                    "myquant.v17.v4.total-return-labels.v1"
                ),
                label=f"{origin}.label_refs.{role}",
            )
            for role in _LABEL_ROLES
        }
        label_artifacts = {
            role: _read_exact_artifact(
                label_refs[role],
                artifact_loader=artifact_loader,
                label=f"{origin}.label_artifacts.{role}",
            )
            for role in _LABEL_ROLES
        }
        for role in _LABEL_ROLES:
            _validate_native_artifact(
                label_artifacts[role],
                label=f"{origin}.{role}",
            )
        for role, label_end in (
            ("label_60", label_60_end),
            ("label_252", label_252_end),
        ):
            available = _instant(
                label_refs[role]["cutoff"],
                label=f"{origin}.{role}.cutoff",
            )
            if available < _origin_cutoff(label_end):
                _blocked(f"{role}_available_before_end:{origin}")
        factor_ref = _validate_factor_ref(
            item.factor_active_set_ref,
            label=f"{origin}.factor_active_set_ref",
        )
        factor_artifact = _read_exact_factor_active_set(
            factor_ref,
            artifact_loader=artifact_loader,
            label=f"{origin}.factor_active_set_artifact",
        )
        factor_from = _as_date(
            item.factor_effective_from_session,
            label="factor_effective_from_session",
        )
        factor_to = (
            None
            if item.factor_effective_to_session is None
            else _as_date(
                item.factor_effective_to_session,
                label="factor_effective_to_session",
            )
        )
        if origin < factor_from or (
            factor_to is not None and origin > factor_to
        ):
            _blocked(f"historical_factor_set_not_effective:{origin}")
        try:
            factor_set_sha = require_sha256(
                item.factor_set_sha256,
                label="factor_set_sha256",
            )
        except IdentityContractError:
            _blocked(f"factor_set_sha_invalid:{origin}")

        quant_scores = {
            symbol: value
            for symbol, value in quant_scores_raw.items()
            if isinstance(value, Decimal)
        }
        fundamental_scores = {
            symbol: value
            for symbol, value in fundamental_scores_raw.items()
            if isinstance(value, Decimal)
        }
        forward60 = {
            symbol: value
            for symbol, value in forward60_raw.items()
            if isinstance(value, Decimal)
        }
        forward252 = {
            symbol: value
            for symbol, value in forward252_raw.items()
            if isinstance(value, Decimal)
        }
        pit_catalog = source_artifacts["pit_catalog"]
        if (
            pit_catalog["decision_session"] != origin.isoformat()
            or pit_catalog["cutoff"] != source_refs["pit_catalog"]["cutoff"]
            or pit_catalog["history_start"]
            > fundamental_start.isoformat()
            or any(
                pit_catalog["dataset_refs"][role]
                != source_refs[role]
                for role in (
                    "benchmark_total_return",
                    "corporate_actions",
                    "official_delisting_cash",
                )
            )
        ):
            _blocked(f"pit_catalog_native_binding:{origin}")
        preselect = source_artifacts["preselect_locator"]
        initial_pool = source_artifacts["initial_pool"]
        quant_branch = source_artifacts["quant_branch"]
        fundamental_branch = source_artifacts["fundamental_branch"]
        if (
            preselect["origin"] != origin.isoformat()
            or preselect["pit_catalog_ref"]
            != source_refs["pit_catalog"]
            or initial_pool["origin"] != origin.isoformat()
            or initial_pool["preselect_locator_ref"]
            != source_refs["preselect_locator"]
            or initial_pool["ordered_pool"] != list(pool)
            or quant_branch["origin"] != origin.isoformat()
            or quant_branch["branch_kind"] != "QUANT"
            or quant_branch["initial_pool_ref"]
            != source_refs["initial_pool"]
            or fundamental_branch["origin"] != origin.isoformat()
            or fundamental_branch["branch_kind"] != "FUNDAMENTAL"
            or fundamental_branch["initial_pool_ref"]
            != source_refs["initial_pool"]
        ):
            _blocked(f"origin_native_source_chain:{origin}")
        for role, artifact, expected_scores in (
            ("quant_branch", quant_branch, quant_scores),
            (
                "fundamental_branch",
                fundamental_branch,
                fundamental_scores,
            ),
        ):
            rows = artifact["score_rows"]
            if [row["symbol"] for row in rows] != list(pool):
                _blocked(f"{role}_pool_order:{origin}")
            observed_scores = {
                row["symbol"]: _decimal(
                    row["score"],
                    label=f"{origin}.{role}.{row['symbol']}",
                )
                for row in rows
            }
            if observed_scores != expected_scores:
                _blocked(f"{role}_native_scores:{origin}")
        for role, kind, label_end, returns, delisted, terminal in (
            (
                "label_60",
                "LABEL_60",
                label_60_end,
                forward60,
                delisted60,
                terminal60,
            ),
            (
                "label_252",
                "LABEL_252",
                label_252_end,
                forward252,
                delisted252,
                terminal252,
            ),
        ):
            artifact = label_artifacts[role]
            rows = artifact["rows"]
            if (
                artifact["origin"] != origin.isoformat()
                or artifact["label_kind"] != kind
                or artifact["label_end_session"]
                != label_end.isoformat()
                or [row["symbol"] for row in rows] != list(pool)
            ):
                _blocked(f"{role}_native_binding:{origin}")
            observed_returns = {
                row["symbol"]: _decimal(
                    row["forward_return"],
                    label=(
                        f"{origin}.{role}."
                        f"{row['symbol']}.forward_return"
                    ),
                )
                for row in rows
            }
            observed_delisted = {
                row["symbol"]: row["delisted"] for row in rows
            }
            observed_terminal = {
                row["symbol"]: row["official_terminal_cash"]
                for row in rows
            }
            if (
                observed_returns != returns
                or observed_delisted != delisted
                or observed_terminal != terminal
            ):
                _blocked(f"{role}_native_rows:{origin}")
        factor_activated_at = _instant(
            factor_artifact["activated_at"],
            label=f"{origin}.factor_active_set.activated_at",
        )
        if (
            factor_activated_at > origin_at
            or factor_artifact["as_of"] != factor_from.isoformat()
            or factor_artifact["production_factor_set_sha256"]
            != factor_set_sha
        ):
            _blocked(f"historical_factor_set_readback:{origin}")
        metrics.append(
            _monthly_metrics(
                origin=origin,
                pool=pool,
                quant_scores=quant_scores,
                fundamental_scores=fundamental_scores,
                forward60=forward60,
                forward252=forward252,
            )
        )
        origin_sha = _origin_semantic(
            origin=origin,
            quant_history_start=quant_start,
            fundamental_history_start=fundamental_start,
            label_60_end=label_60_end,
            label_252_end=label_252_end,
            pool=pool,
            quant_scores=quant_scores,
            fundamental_scores=fundamental_scores,
            forward60=forward60,
            forward252=forward252,
            source_refs=source_refs,
            label_refs=label_refs,
            factor_ref=factor_ref,
            factor_effective_from=factor_from,
            factor_effective_to=factor_to,
            factor_set_sha256=factor_set_sha,
        )
        inventory_rows.append(
            {
                "factor_active_set_ref": factor_ref,
                "factor_set_sha256": factor_set_sha,
                "origin": origin.isoformat(),
                "origin_semantic_sha256": origin_sha,
                "source_closure_sha256": hashlib.sha256(
                    canonical_bytes(
                        {
                            "label_refs": label_refs,
                            "source_refs": source_refs,
                        }
                    )
                ).hexdigest(),
            }
        )

    index_by_origin = {
        origin: index for index, origin in enumerate(normalized_origins)
    }
    oos_indices = [index_by_origin[origin] for origin in oos_origins]
    matrix = circular_block_bootstrap_matrix()
    folds: list[CalibrationFold] = []
    stitched: list[MonthlyMetric] = []
    for fold_index in range(OUTER_FOLDS):
        fold_oos = oos_indices[
            fold_index
            * OUTER_MONTHS_PER_FOLD : (fold_index + 1)
            * OUTER_MONTHS_PER_FOLD
        ]
        fold_start = normalized_origins[fold_oos[0]]
        eligible = [
            index
            for index in range(fold_oos[0])
            if label_252_ends[index] < fold_start
        ]
        if len(eligible) < CALIBRATION_MONTHS:
            _blocked(
                f"fold_{fold_index + 1}_training_below_60"
            )
        training = eligible[-CALIBRATION_MONTHS:]
        if training != list(
            range(training[0], training[0] + CALIBRATION_MONTHS)
        ):
            _blocked(
                f"fold_{fold_index + 1}_training_not_consecutive"
            )
        selected = _select_weight(
            [metrics[index] for index in training],
            matrix,
        )
        stitched.extend(
            metrics[index][selected.quant_weight] for index in fold_oos
        )
        folds.append(
            CalibrationFold(
                index=fold_index + 1,
                training_origins=tuple(
                    normalized_origins[index] for index in training
                ),
                oos_origins=tuple(
                    normalized_origins[index] for index in fold_oos
                ),
                selected_weight=selected.quant_weight,
            )
        )
    if len(stitched) != CALIBRATION_MONTHS:
        _blocked("outer_oos_stitching")
    oos_hit = tuple(row.hit60 for row in stitched)
    oos_q25 = tuple(row.q25_252 for row in stitched)
    hit_lower = _bootstrap_lower(oos_hit, matrix)
    q25_lower = _bootstrap_lower(oos_q25, matrix)
    blockers: list[str] = []
    if hit_lower <= Decimal("0.50"):
        blockers.append("oos_hit60_lower_95_not_above_0.50")
    if q25_lower <= 0:
        blockers.append("oos_q25_252_lower_95_not_above_zero")
    active_indices = [
        index_by_origin[origin] for origin in closure_origins[-60:]
    ]
    active = _select_weight(
        [metrics[index] for index in active_indices],
        matrix,
    )
    inventory = seal_semantic(
        {
            "authority": dict(_NO_AUTHORITY),
            "closure_origin_count": CLOSURE_MONTHS,
            "closure_origins": [
                origin.isoformat() for origin in closure_origins
            ],
            "created_at": created_text,
            "cutoff": cutoff_text,
            "input_origin_count": len(origins),
            "inventory_id": f"{run}-calibration-origins",
            "origins": inventory_rows,
            "protocol_version": PROTOCOL_VERSION,
            "run_id": run,
            "strategy_id": strategy,
            "version": ORIGIN_INVENTORY_VERSION,
        }
    )
    return CalibrationClosure(
        strategy_id=strategy,
        run_id=run,
        cutoff=cutoff_text,
        input_origin_count=len(origins),
        closure_origins=closure_origins,
        active_refit_origins=closure_origins[-60:],
        folds=tuple(folds),
        active_assessment=active,
        oos_mean_hit60=_mean(oos_hit),
        oos_mean_q25_252=_mean(oos_q25),
        oos_hit60_lower_95=hit_lower,
        oos_q25_252_lower_95=q25_lower,
        bootstrap_matrix_sha256=bootstrap_matrix_sha256(matrix),
        promoted=not blockers,
        blockers=tuple(blockers),
        origin_inventory=inventory,
    )


def artifact_ref(
    artifact: Mapping[str, Any],
    *,
    relative_path: str,
) -> dict[str, str]:
    if (
        type(artifact) is not dict
        or type(relative_path) is not str
        or relative_path.startswith("/")
        or any(
            part in {"", ".", ".."}
            for part in relative_path.split("/")
        )
    ):
        _blocked("artifact_ref_invalid")
    identity = next(
        (
            artifact.get(field)
            for field in (
                "artifact_id",
                "catalog_id",
                "inventory_id",
                "label_id",
                "locator_id",
                "output_id",
                "receipt_id",
            )
            if field in artifact
        ),
        None,
    )
    if type(identity) is not str:
        _blocked("artifact_ref_identity")
    return {
        "artifact_id": identity,
        "artifact_version": str(artifact.get("version")),
        "byte_sha256": hashlib.sha256(
            canonical_bytes(artifact) + b"\n"
        ).hexdigest(),
        "cutoff": str(artifact.get("cutoff")),
        "relative_path": relative_path,
        "semantic_sha256": str(artifact.get("semantic_sha256")),
        "strategy_id": str(artifact.get("strategy_id")),
    }


def build_calibration_receipt(
    closure: CalibrationClosure,
    *,
    calibration_kind: str,
    receipt_id: str,
    created_at: str,
    origin_inventory_ref: Mapping[str, Any],
    artifact_loader: Callable[[Mapping[str, str]], bytes],
) -> dict[str, Any]:
    if calibration_kind not in {
        "QUANT_TIMING",
        "FUNDAMENTAL_FORWARD",
    }:
        _blocked("calibration_kind_invalid")
    try:
        receipt = require_opaque_id(receipt_id, label="receipt_id")
        created = require_utc_timestamp(created_at, label="created_at")
    except IdentityContractError:
        _blocked("calibration_receipt_identity")
    inventory_ref = _validate_ref(
        origin_inventory_ref,
        strategy_id=closure.strategy_id,
        cutoff_at=_instant(closure.cutoff, label="cutoff"),
        expected_version=ORIGIN_INVENTORY_VERSION,
        label="origin_inventory_ref",
    )
    if (
        inventory_ref["semantic_sha256"]
        != closure.origin_inventory["semantic_sha256"]
    ):
        _blocked("origin_inventory_ref_mismatch")
    inventory_document = _read_exact_artifact(
        inventory_ref,
        artifact_loader=artifact_loader,
        label="origin_inventory",
    )
    try:
        validate_artifact(inventory_document)
    except ValueError:
        _blocked("origin_inventory_validation")
    if inventory_document != closure.origin_inventory:
        _blocked("origin_inventory_readback_mismatch")
    minimum_span = (
        QUANT_MIN_OPEN_SESSIONS
        if calibration_kind == "QUANT_TIMING"
        else FUNDAMENTAL_MIN_OPEN_SESSIONS
    )
    return seal_semantic(
        {
            "accepted": True,
            "authority": dict(_NO_AUTHORITY),
            "bootstrap": {
                "block_length_months": BOOTSTRAP_BLOCK_LENGTH,
                "generator": "PCG64",
                "matrix_sha256": closure.bootstrap_matrix_sha256,
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
            },
            "calibration_kind": calibration_kind,
            "closure_origin_count": CLOSURE_MONTHS,
            "created_at": created,
            "cutoff": closure.cutoff,
            "input_origin_count": closure.input_origin_count,
            "minimum_open_session_span": minimum_span,
            "origin_inventory_ref": inventory_ref,
            "protocol_version": PROTOCOL_VERSION,
            "receipt_id": receipt,
            "run_id": closure.run_id,
            "status": "ACCEPTED",
            "strategy_id": closure.strategy_id,
            "version": CALIBRATION_RECEIPT_VERSION,
        }
    )


def build_fusion_promotion_receipt(
    closure: CalibrationClosure,
    *,
    receipt_id: str,
    created_at: str,
    origin_inventory_ref: Mapping[str, Any],
    quant_calibration_receipt_ref: Mapping[str, Any],
    fundamental_calibration_receipt_ref: Mapping[str, Any],
    artifact_loader: Callable[[Mapping[str, str]], bytes],
) -> dict[str, Any]:
    try:
        receipt = require_opaque_id(receipt_id, label="receipt_id")
        created = require_utc_timestamp(created_at, label="created_at")
    except IdentityContractError:
        _blocked("fusion_receipt_identity")
    cutoff_at = _instant(closure.cutoff, label="cutoff")
    inventory_ref = _validate_ref(
        origin_inventory_ref,
        strategy_id=closure.strategy_id,
        cutoff_at=cutoff_at,
        expected_version=ORIGIN_INVENTORY_VERSION,
        label="origin_inventory_ref",
    )
    quant_ref = _validate_ref(
        quant_calibration_receipt_ref,
        strategy_id=closure.strategy_id,
        cutoff_at=cutoff_at,
        expected_version=CALIBRATION_RECEIPT_VERSION,
        label="quant_calibration_receipt_ref",
    )
    fundamental_ref = _validate_ref(
        fundamental_calibration_receipt_ref,
        strategy_id=closure.strategy_id,
        cutoff_at=cutoff_at,
        expected_version=CALIBRATION_RECEIPT_VERSION,
        label="fundamental_calibration_receipt_ref",
    )
    if quant_ref == fundamental_ref:
        _blocked("calibration_receipt_ref_collision")
    inventory_document = _read_exact_artifact(
        inventory_ref,
        artifact_loader=artifact_loader,
        label="origin_inventory",
    )
    quant_document = _read_exact_artifact(
        quant_ref,
        artifact_loader=artifact_loader,
        label="quant_calibration_receipt",
    )
    fundamental_document = _read_exact_artifact(
        fundamental_ref,
        artifact_loader=artifact_loader,
        label="fundamental_calibration_receipt",
    )
    try:
        validate_artifact(inventory_document)
        validate_artifact(
            quant_document,
            artifact_loader=artifact_loader,
        )
        validate_artifact(
            fundamental_document,
            artifact_loader=artifact_loader,
        )
    except ValueError:
        _blocked("calibration_receipt_validation")
    expected_bootstrap = {
        "block_length_months": BOOTSTRAP_BLOCK_LENGTH,
        "generator": "PCG64",
        "matrix_sha256": closure.bootstrap_matrix_sha256,
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
    }
    if (
        inventory_document != closure.origin_inventory
        or quant_document["calibration_kind"] != "QUANT_TIMING"
        or fundamental_document["calibration_kind"]
        != "FUNDAMENTAL_FORWARD"
        or quant_document["origin_inventory_ref"] != inventory_ref
        or fundamental_document["origin_inventory_ref"] != inventory_ref
        or quant_document["bootstrap"] != expected_bootstrap
        or fundamental_document["bootstrap"] != expected_bootstrap
        or quant_document["run_id"] != closure.run_id
        or fundamental_document["run_id"] != closure.run_id
        or quant_document["closure_origin_count"] != CLOSURE_MONTHS
        or fundamental_document["closure_origin_count"]
        != CLOSURE_MONTHS
        or quant_document["input_origin_count"]
        != closure.input_origin_count
        or fundamental_document["input_origin_count"]
        != closure.input_origin_count
    ):
        _blocked("calibration_receipt_readback_mismatch")
    return seal_semantic(
        {
            "accepted": closure.promoted,
            "active_quant_weight": _decimal_text(
                closure.active_assessment.quant_weight
            ),
            "active_refit_origins": [
                origin.isoformat()
                for origin in closure.active_refit_origins
            ],
            "authority": dict(_NO_AUTHORITY),
            "blockers": list(closure.blockers),
            "bootstrap": {
                "block_length_months": BOOTSTRAP_BLOCK_LENGTH,
                "generator": "PCG64",
                "matrix_sha256": closure.bootstrap_matrix_sha256,
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
            },
            "created_at": created,
            "cutoff": closure.cutoff,
            "folds": [
                {
                    "fold_index": fold.index,
                    "oos_origins": [
                        origin.isoformat()
                        for origin in fold.oos_origins
                    ],
                    "selected_quant_weight": _decimal_text(
                        fold.selected_weight
                    ),
                    "training_origins": [
                        origin.isoformat()
                        for origin in fold.training_origins
                    ],
                }
                for fold in closure.folds
            ],
            "fundamental_calibration_receipt_ref": fundamental_ref,
            "origin_inventory_ref": inventory_ref,
            "oos_hit60_lower_95": _decimal_text(
                closure.oos_hit60_lower_95
            ),
            "oos_mean_hit60": _decimal_text(
                closure.oos_mean_hit60
            ),
            "oos_mean_q25_252": _decimal_text(
                closure.oos_mean_q25_252
            ),
            "oos_q25_252_lower_95": _decimal_text(
                closure.oos_q25_252_lower_95
            ),
            "protocol_version": PROTOCOL_VERSION,
            "quant_calibration_receipt_ref": quant_ref,
            "receipt_id": receipt,
            "run_id": closure.run_id,
            "status": (
                "PROMOTED"
                if closure.promoted
                else "CALIBRATION_CLOSURE_BLOCKED"
            ),
            "strategy_id": closure.strategy_id,
            "version": FUSION_PROMOTION_RECEIPT_VERSION,
        }
    )


__all__ = [
    "BOOTSTRAP_MATRIX_VERSION",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "CALIBRATION_RECEIPT_VERSION",
    "CANDIDATE_QUANT_WEIGHTS",
    "CLOSURE_MONTHS",
    "CalibrationClosure",
    "CalibrationClosureError",
    "CalibrationFold",
    "FUNDAMENTAL_MIN_OPEN_SESSIONS",
    "FUSION_PROMOTION_RECEIPT_VERSION",
    "ORIGIN_INVENTORY_VERSION",
    "OriginCalibrationInput",
    "QUANT_MIN_OPEN_SESSIONS",
    "WeightAssessment",
    "artifact_ref",
    "bootstrap_matrix_sha256",
    "build_calibration_receipt",
    "build_fusion_promotion_receipt",
    "circular_block_bootstrap_matrix",
    "run_calibration_closure",
    "schedule_month_end_origins",
]
